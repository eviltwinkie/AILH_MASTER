#!/usr/bin/env python3
"""
Physics-Aware Correlation Engine
Joint (position, velocity) search and dispersion-aware modeling

Version: 3.0.0
Revision: 1
Date: 2025-11-25
Status: Production
"""

import numpy as np
from scipy.signal import butter, filtfilt, correlate
from scipy.optimize import minimize
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from correlator_v3_config import *


@dataclass
class JointSearchResult:
    """Result from joint (x, c) search"""
    optimal_position_m: float
    optimal_velocity_mps: float
    confidence_score: float
    search_heatmap: np.ndarray
    position_grid: np.ndarray
    velocity_grid: np.ndarray


@dataclass
class DispersionResult:
    """Result from dispersion-aware analysis"""
    base_velocity_mps: float
    dispersion_coefficient: float
    per_band_velocities: Dict[str, float]
    fit_quality: float


class PhysicsAwareCorrelator:
    """
    Physics-aware correlation with joint (x,c) estimation and dispersion modeling.
    """

    def __init__(self, sample_rate=4096, use_gpu=True, verbose=False):
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose

        if self.verbose:
            print(f"[i] PhysicsAwareCorrelator initialized (GPU: {self.use_gpu})")

    def bandpass_filter(self, signal: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        nyquist = self.sample_rate / 2
        low = low_hz / nyquist
        high = high_hz / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def compute_correlation(self, sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
        """Compute cross-correlation"""
        if self.use_gpu:
            sig_a_gpu = cp.asarray(sig_a, dtype=cp.float32)
            sig_b_gpu = cp.asarray(sig_b, dtype=cp.float32)

            # FFT-based correlation
            fft_a = cp.fft.fft(sig_a_gpu)
            fft_b = cp.fft.fft(sig_b_gpu)
            cross_spectrum = fft_a * cp.conj(fft_b)
            corr = cp.fft.ifft(cross_spectrum).real
            corr = cp.fft.fftshift(corr)

            return cp.asnumpy(corr)
        else:
            return correlate(sig_a, sig_b, mode='same', method='fft')

    def delay_to_position(self, delay_samples: int, velocity_mps: float,
                          sensor_separation_m: float) -> float:
        """Convert time delay to leak position"""
        time_delay_sec = delay_samples / self.sample_rate
        return (sensor_separation_m - velocity_mps * time_delay_sec) / 2

    def position_to_delay(self, position_m: float, velocity_mps: float,
                          sensor_separation_m: float) -> float:
        """Convert leak position to expected time delay"""
        if velocity_mps <= 0:
            raise ValueError(f"Wave speed must be positive, got {velocity_mps}")
        return (sensor_separation_m - 2 * position_m) / velocity_mps

    def joint_search(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sensor_separation_m: float,
        velocity_range: Tuple[float, float] = None,
        position_resolution_m: float = None,
        velocity_resolution_mps: float = None
    ) -> JointSearchResult:
        """
        Search over (x, c) space jointly to find optimal leak position and wave speed.

        Instead of assuming wave speed c, we search over both position x and velocity c.
        """
        if velocity_range is None:
            velocity_range = (VELOCITY_SEARCH_MIN_MPS, VELOCITY_SEARCH_MAX_MPS)
        if position_resolution_m is None:
            position_resolution_m = POSITION_SEARCH_RESOLUTION_M
        if velocity_resolution_mps is None:
            velocity_resolution_mps = VELOCITY_SEARCH_STEP_MPS

        # Validate inputs
        if sensor_separation_m <= 0:
            raise ValueError(f"Sensor separation must be positive, got {sensor_separation_m}")
        if velocity_range[0] <= 0 or velocity_range[1] <= 0:
            raise ValueError(f"Wave speed must be positive, got range {velocity_range}")
        if velocity_range[0] >= velocity_range[1]:
            raise ValueError(f"Invalid velocity range: min ({velocity_range[0]}) >= max ({velocity_range[1]})")

        if self.verbose:
            print(f"\n[i] Joint (x,c) search:")
            print(f"    Sensor separation: {sensor_separation_m:.1f}m")
            print(f"    Velocity range: {velocity_range[0]}-{velocity_range[1]} m/s")
            print(f"    Resolution: {position_resolution_m}m, {velocity_resolution_mps} m/s")

        # Create search grids
        positions = np.arange(0, sensor_separation_m + position_resolution_m, position_resolution_m)
        velocities = np.arange(velocity_range[0], velocity_range[1] + velocity_resolution_mps,
                               velocity_resolution_mps)

        # Compute correlation once
        correlation = self.compute_correlation(signal_a, signal_b)
        n_samples = len(correlation)
        center_idx = n_samples // 2

        # Search heatmap
        heatmap = np.zeros((len(positions), len(velocities)))

        # Grid search over (x, c)
        for i, x in enumerate(positions):
            for j, c in enumerate(velocities):
                # Expected delay for this (x, c)
                expected_delay_sec = self.position_to_delay(x, c, sensor_separation_m)
                expected_delay_samples = int(expected_delay_sec * self.sample_rate)

                # Correlation value at this delay
                delay_idx = center_idx + expected_delay_samples
                if 0 <= delay_idx < n_samples:
                    heatmap[i, j] = correlation[delay_idx]

        # Find maximum
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        optimal_position_m = positions[max_idx[0]]
        optimal_velocity_mps = velocities[max_idx[1]]
        confidence_score = heatmap[max_idx] / (np.max(np.abs(correlation)) + 1e-12)

        if self.verbose:
            print(f"    [✓] Optimal position: {optimal_position_m:.2f}m")
            print(f"    [✓] Optimal velocity: {optimal_velocity_mps:.1f} m/s")
            print(f"    [✓] Confidence: {confidence_score:.3f}")

        return JointSearchResult(
            optimal_position_m=optimal_position_m,
            optimal_velocity_mps=optimal_velocity_mps,
            confidence_score=float(confidence_score),
            search_heatmap=heatmap,
            position_grid=positions,
            velocity_grid=velocities
        )

    def dispersion_aware_search(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sensor_separation_m: float,
        frequency_bands: List[Tuple[float, float]] = None
    ) -> DispersionResult:
        """
        Fit dispersion model: c_k = c_0 + α·f_k

        Different frequencies propagate at different speeds (dispersion).
        We estimate c_0 (base velocity) and α (dispersion coefficient).
        """
        if frequency_bands is None:
            # Default bands
            frequency_bands = [
                (100, 400),   # Low
                (400, 800),   # Mid
                (800, 1200),  # High
                (1200, 1500)  # Very high
            ]

        if self.verbose:
            print(f"\n[i] Dispersion-aware analysis:")
            print(f"    Frequency bands: {len(frequency_bands)}")

        # Estimate velocity in each band
        band_velocities = {}
        center_freqs = []
        velocities = []

        for low_hz, high_hz in frequency_bands:
            # Filter signal to this band
            sig_a_band = self.bandpass_filter(signal_a, low_hz, high_hz)
            sig_b_band = self.bandpass_filter(signal_b, low_hz, high_hz)

            # Joint search for this band
            result = self.joint_search(sig_a_band, sig_b_band, sensor_separation_m,
                                      position_resolution_m=1.0,  # Coarser for speed
                                      velocity_resolution_mps=100)

            center_freq = (low_hz + high_hz) / 2
            center_freqs.append(center_freq)
            velocities.append(result.optimal_velocity_mps)
            band_velocities[f"{low_hz}-{high_hz}Hz"] = result.optimal_velocity_mps

            if self.verbose:
                print(f"    Band {low_hz}-{high_hz} Hz: c = {result.optimal_velocity_mps:.1f} m/s")

        # Fit linear model: c(f) = c_0 + α·f
        center_freqs = np.array(center_freqs)
        velocities = np.array(velocities)

        # Linear regression
        coeffs = np.polyfit(center_freqs, velocities, 1)
        alpha = coeffs[0]  # Dispersion coefficient
        c_0 = coeffs[1]    # Base velocity

        # Fit quality (R²)
        predicted = c_0 + alpha * center_freqs
        ss_res = np.sum((velocities - predicted) ** 2)
        ss_tot = np.sum((velocities - np.mean(velocities)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-12))

        if self.verbose:
            print(f"    [✓] Base velocity c_0: {c_0:.1f} m/s")
            print(f"    [✓] Dispersion α: {alpha:.4f} m/s/Hz")
            print(f"    [✓] Fit quality R²: {r_squared:.3f}")

        return DispersionResult(
            base_velocity_mps=float(c_0),
            dispersion_coefficient=float(alpha),
            per_band_velocities=band_velocities,
            fit_quality=float(r_squared)
        )


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PHYSICS-AWARE CORRELATOR TEST")
    print("=" * 80)

    # Generate synthetic leak signal
    sample_rate = 4096
    duration = 10.0
    n_samples = int(sample_rate * duration)

    # Leak parameters
    sensor_separation_m = 100.0
    true_leak_position_m = 35.0
    true_wave_speed_mps = 1400.0

    # Time delay
    true_delay_sec = (sensor_separation_m - 2 * true_leak_position_m) / true_wave_speed_mps
    true_delay_samples = int(true_delay_sec * sample_rate)

    # Generate signals
    t = np.arange(n_samples) / sample_rate
    leak_signal = np.sin(2 * np.pi * 800 * t) * np.exp(-t / 5.0)  # Leak @ 800 Hz

    # Sensor A receives leak immediately, B receives delayed
    signal_a = leak_signal + np.random.randn(n_samples) * 0.1
    signal_b = np.roll(leak_signal, true_delay_samples) + np.random.randn(n_samples) * 0.1

    # Test joint search
    correlator = PhysicsAwareCorrelator(sample_rate=sample_rate, verbose=True)

    print("\n[TEST 1] Joint (x, c) search")
    result = correlator.joint_search(signal_a, signal_b, sensor_separation_m)

    error_position = abs(result.optimal_position_m - true_leak_position_m)
    error_velocity = abs(result.optimal_velocity_mps - true_wave_speed_mps)

    print(f"\n[RESULTS]")
    print(f"  True position: {true_leak_position_m:.2f}m, Estimated: {result.optimal_position_m:.2f}m")
    print(f"  Position error: {error_position:.2f}m")
    print(f"  True velocity: {true_wave_speed_mps:.1f} m/s, Estimated: {result.optimal_velocity_mps:.1f} m/s")
    print(f"  Velocity error: {error_velocity:.1f} m/s")
    print(f"  Test: {'PASS' if error_position < 5.0 and error_velocity < 200 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] Physics-aware correlator test complete!")
