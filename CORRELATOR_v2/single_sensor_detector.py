#!/usr/bin/env python3
"""
CORRELATOR_v2 - Single Sensor Leak Detection Module

Detect leaks using only a single sensor (no correlation required).
Useful for:
- Quick screening when only one sensor available
- Pre-deployment validation
- Identifying leak presence before sensor pair deployment
- Low-cost monitoring applications

Detection methods:
1. Band power analysis (leak frequency concentration)
2. Statistical features (RMS, kurtosis, spectral entropy)
3. Spectral signature matching
4. Temporal pattern analysis

Based on CORRELATOR_v1 leak_signature_analysis.py, enhanced with v3.2 features.

Author: AILH Development Team
Date: 2025-11-19
Version: 3.2.0
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy.signal import welch, spectrogram
import warnings

from correlator_config import *
from statistical_features import StatisticalFeatureExtractor, SignalStatistics


@dataclass
class SingleSensorResult:
    """
    Result from single-sensor leak detection.

    Attributes:
        leak_detected: Boolean indicating if leak was detected
        confidence: Detection confidence (0-1)
        detection_reason: Explanation of why leak was detected/rejected
        statistics: Signal statistical features
        band_power_ratio: Ratio of band power to total power
        spectral_centroid: Center of mass of spectrum
        dominant_frequencies: Top N dominant frequencies (Hz)
        quality_score: Overall signal quality (0-1)
    """
    leak_detected: bool
    confidence: float
    detection_reason: str
    statistics: SignalStatistics
    band_power_ratio: float
    spectral_centroid: float
    dominant_frequencies: List[Tuple[float, float]]  # (freq_hz, power)
    quality_score: float


class SingleSensorDetector:
    """
    Detect leaks using single sensor analysis.

    Detection criteria:
    1. Band power ratio > threshold (leak energy concentration)
    2. High kurtosis (impulsive signal)
    3. Low spectral entropy (structured signal)
    4. High RMS (elevated signal energy)
    5. Dominant frequency in leak range (100-1500 Hz)
    """

    def __init__(
        self,
        sample_rate: int = 4096,
        freq_band: Tuple[float, float] = (100, 1500),
        verbose: bool = False
    ):
        """
        Initialize single sensor detector.

        Args:
            sample_rate: Audio sample rate (Hz)
            freq_band: Frequency band for leak detection (low_hz, high_hz)
            verbose: Print detection details
        """
        self.sample_rate = sample_rate
        self.freq_band = freq_band
        self.verbose = verbose

        # Initialize feature extractor
        self.feature_extractor = StatisticalFeatureExtractor(
            sample_rate=sample_rate,
            verbose=False
        )

        # Detection thresholds (tuned from v1 experience)
        self.band_power_threshold = 0.35  # Band power ratio
        self.rms_threshold_multiplier = 1.2  # RMS vs baseline
        self.kurtosis_threshold = 2.0  # High kurtosis indicator
        self.entropy_ratio_threshold = 0.9  # Low entropy indicator
        self.min_quality_score = 0.5  # Minimum signal quality

    def detect(
        self,
        signal: np.ndarray,
        baseline_signal: Optional[np.ndarray] = None
    ) -> SingleSensorResult:
        """
        Detect leak in single sensor signal.

        Args:
            signal: Audio signal from sensor
            baseline_signal: Optional known good (no-leak) baseline

        Returns:
            SingleSensorResult with detection outcome
        """
        if self.verbose:
            print(f"\n[i] Single sensor leak detection")
            print(f"    Signal length: {len(signal)} samples ({len(signal)/self.sample_rate:.1f}s)")
            print(f"    Frequency band: {self.freq_band[0]}-{self.freq_band[1]} Hz")

        # Extract statistical features
        stats = self.feature_extractor.extract_features(signal, freq_band=self.freq_band)

        # Calculate spectral metrics
        spectral_centroid = self._calculate_spectral_centroid(signal)
        dominant_freqs = self._find_dominant_frequencies(signal, n_peaks=5)

        # Quality assessment
        quality_score = self._assess_signal_quality(signal, stats)

        # Detection logic
        leak_detected = False
        confidence = 0.0
        reasons = []

        # Criterion 1: Band power analysis
        if stats.band_power_ratio > self.band_power_threshold:
            leak_detected = True
            confidence += 0.3
            reasons.append(f"High band power ratio ({stats.band_power_ratio:.3f} > {self.band_power_threshold})")

        # Criterion 2: High kurtosis (impulsive signal)
        if stats.kurtosis > self.kurtosis_threshold:
            leak_detected = True
            confidence += 0.2
            reasons.append(f"High kurtosis ({stats.kurtosis:.2f}, impulsive signal)")

        # Criterion 3: Dominant frequency in leak range
        if dominant_freqs and len(dominant_freqs) > 0:
            dom_freq = dominant_freqs[0][0]
            if self.freq_band[0] <= dom_freq <= self.freq_band[1]:
                confidence += 0.15
                reasons.append(f"Dominant frequency in leak range ({dom_freq:.1f} Hz)")

        # Criterion 4: Low spectral entropy (structured)
        if baseline_signal is not None:
            baseline_stats = self.feature_extractor.extract_features(baseline_signal, freq_band=self.freq_band)
            comparison = self.feature_extractor.compare_signals(stats, baseline_stats)

            # Compare to baseline
            if comparison['rms_ratio'] > self.rms_threshold_multiplier:
                leak_detected = True
                confidence += 0.2
                reasons.append(f"RMS elevated vs baseline ({comparison['rms_ratio']:.2f}x)")

            if comparison['spectral_entropy_ratio'] < self.entropy_ratio_threshold:
                leak_detected = True
                confidence += 0.15
                reasons.append(f"Spectral entropy lower than baseline (structured signal)")

        else:
            # No baseline - use absolute thresholds
            if stats.spectral_entropy < 8.0:  # Low entropy (structured)
                confidence += 0.1
                reasons.append(f"Low spectral entropy ({stats.spectral_entropy:.2f}, structured signal)")

            if stats.rms > 0.1:  # High RMS
                confidence += 0.1
                reasons.append(f"Elevated RMS ({stats.rms:.3f})")

        # Quality check
        if quality_score < self.min_quality_score:
            confidence *= 0.5  # Reduce confidence for poor quality signals
            reasons.append(f"WARNING: Low signal quality ({quality_score:.2f})")

        # Normalize confidence
        confidence = min(1.0, confidence)

        # Final decision
        if not leak_detected or confidence < 0.5:
            leak_detected = False
            detection_reason = "No strong leak signature detected. " + "; ".join(reasons) if reasons else "Signal appears normal"
        else:
            detection_reason = "LEAK DETECTED: " + "; ".join(reasons)

        if self.verbose:
            print(f"\n[{'✓' if leak_detected else '✗'}] Detection result: {'LEAK' if leak_detected else 'NO LEAK'}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Quality: {quality_score:.2f}")
            print(f"    Reason: {detection_reason}")

        return SingleSensorResult(
            leak_detected=leak_detected,
            confidence=confidence,
            detection_reason=detection_reason,
            statistics=stats,
            band_power_ratio=stats.band_power_ratio,
            spectral_centroid=spectral_centroid,
            dominant_frequencies=dominant_freqs,
            quality_score=quality_score
        )

    def _calculate_spectral_centroid(self, signal: np.ndarray) -> float:
        """
        Calculate spectral centroid (center of mass of spectrum).

        Leaks often have higher spectral centroid than background noise.

        Args:
            signal: Audio signal

        Returns:
            Spectral centroid in Hz
        """
        freqs, psd = welch(signal, fs=self.sample_rate, nperseg=min(2048, len(signal)))

        # Weighted average of frequencies
        centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)

        return float(centroid)

    def _find_dominant_frequencies(
        self,
        signal: np.ndarray,
        n_peaks: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Find dominant frequencies in signal.

        Args:
            signal: Audio signal
            n_peaks: Number of peaks to return

        Returns:
            List of (frequency_hz, power) tuples, sorted by power (descending)
        """
        freqs, psd = welch(signal, fs=self.sample_rate, nperseg=min(2048, len(signal)))

        # Find peaks
        from scipy.signal import find_peaks
        peak_indices, _ = find_peaks(psd, height=np.max(psd) * 0.1)

        # Sort by power
        peak_powers = psd[peak_indices]
        sorted_indices = np.argsort(peak_powers)[::-1][:n_peaks]

        # Extract top peaks
        dominant = [
            (float(freqs[peak_indices[i]]), float(psd[peak_indices[i]]))
            for i in sorted_indices
        ]

        return dominant

    def _assess_signal_quality(
        self,
        signal: np.ndarray,
        stats: SignalStatistics
    ) -> float:
        """
        Assess overall signal quality (0-1).

        Good quality indicators:
        - Sufficient RMS (not too quiet)
        - Low clipping (not saturated)
        - Reasonable dynamic range

        Args:
            signal: Audio signal
            stats: Statistical features

        Returns:
            Quality score (0-1)
        """
        quality = 1.0

        # Check for clipping (signal saturated)
        clipping_threshold = 0.95
        max_val = np.max(np.abs(signal))
        if max_val > clipping_threshold:
            quality *= 0.5
            if self.verbose:
                print(f"    [!] Signal clipping detected ({max_val:.2f})")

        # Check for very low RMS (too quiet)
        if stats.rms < 0.01:
            quality *= 0.6
            if self.verbose:
                print(f"    [!] Very low signal level (RMS={stats.rms:.4f})")

        # Check for DC offset
        dc_offset = np.abs(np.mean(signal))
        if dc_offset > 0.1:
            quality *= 0.8
            if self.verbose:
                print(f"    [!] DC offset detected ({dc_offset:.3f})")

        # Check for very high kurtosis (possible transient artifact)
        if stats.kurtosis > 20:
            quality *= 0.7
            if self.verbose:
                print(f"    [!] Very high kurtosis (possible artifact)")

        return quality

    def analyze_temporal_pattern(
        self,
        signal: np.ndarray,
        window_sec: float = 1.0
    ) -> Dict[str, float]:
        """
        Analyze temporal patterns in signal.

        Leaks often show consistent patterns over time, while transient
        noise shows varying patterns.

        Args:
            signal: Audio signal
            window_sec: Window size for temporal analysis (seconds)

        Returns:
            Dictionary of temporal metrics
        """
        window_samples = int(window_sec * self.sample_rate)
        n_windows = len(signal) // window_samples

        if n_windows < 2:
            return {
                'temporal_consistency': 0.0,
                'rms_variation': 0.0,
                'spectral_variation': 0.0
            }

        # Calculate RMS per window
        rms_values = []
        entropy_values = []

        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window = signal[start:end]

            # RMS
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)

            # Spectral entropy
            freqs, psd = welch(window, fs=self.sample_rate, nperseg=min(512, len(window)))
            psd_norm = psd / (np.sum(psd) + 1e-12)
            from scipy.stats import entropy
            ent = entropy(psd_norm + 1e-12)
            entropy_values.append(ent)

        # Calculate consistency
        rms_variation = np.std(rms_values) / (np.mean(rms_values) + 1e-12)
        spectral_variation = np.std(entropy_values) / (np.mean(entropy_values) + 1e-12)

        # High consistency = leak (steady signal)
        # Low consistency = transient noise (varying signal)
        temporal_consistency = 1.0 / (1.0 + rms_variation + spectral_variation)

        return {
            'temporal_consistency': float(temporal_consistency),
            'rms_variation': float(rms_variation),
            'spectral_variation': float(spectral_variation)
        }


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("SINGLE SENSOR LEAK DETECTOR TEST")
    print("=" * 80)

    # Generate test signals
    sample_rate = 4096
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Test 1: Leak signal (tonal @ 800 Hz with noise)
    print("\n[TEST 1] Leak signal (800 Hz tone + noise)")
    leak_signal = np.sin(2 * np.pi * 800 * t) * 0.5
    leak_signal += np.random.randn(len(t)) * 0.1

    detector = SingleSensorDetector(sample_rate=sample_rate, verbose=True)
    result = detector.detect(leak_signal)

    # Test 2: Normal background noise
    print("\n" + "=" * 80)
    print("\n[TEST 2] Background noise (no leak)")
    noise_signal = np.random.randn(len(t)) * 0.1

    result_noise = detector.detect(noise_signal, baseline_signal=noise_signal)

    # Test 3: Leak vs baseline comparison
    print("\n" + "=" * 80)
    print("\n[TEST 3] Leak detection with baseline comparison")
    result_baseline = detector.detect(leak_signal, baseline_signal=noise_signal)

    print("\n" + "=" * 80)
    print("\n[✓] Single sensor detector tests complete!")
    print(f"    Test 1 (leak): {'PASS' if result.leak_detected else 'FAIL'}")
    print(f"    Test 2 (noise): {'PASS' if not result_noise.leak_detected else 'FAIL'}")
    print(f"    Test 3 (baseline): {'PASS' if result_baseline.leak_detected else 'FAIL'}")
