#!/usr/bin/env python3
"""
Leak Detection Correlator - Correlation Engine Module

This module implements cross-correlation algorithms for estimating time delays
between acoustic signals from two hydrophone sensors. Includes GPU acceleration
and multiple correlation methods (basic, GCC-PHAT, frequency-domain).

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional, Union
import time

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from correlator_config import *


class CorrelationEngine:
    """
    High-performance cross-correlation engine for leak detection.

    Supports multiple correlation methods:
    - Basic time-domain correlation
    - GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
    - Frequency-domain correlation
    - Adaptive filtering

    Can use GPU acceleration when available.
    """

    def __init__(
        self,
        method: str = DEFAULT_CORRELATION_METHOD,
        use_gpu: bool = USE_GPU_CORRELATION,
        bandpass_filter: bool = True,
        verbose: bool = False
    ):
        """
        Initialize correlation engine.

        Args:
            method (str): Correlation method ('basic', 'gcc_phat', 'frequency_domain')
            use_gpu (bool): Use GPU acceleration if available
            bandpass_filter (bool): Apply bandpass filter before correlation
            verbose (bool): Print verbose output
        """
        self.method = method
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.bandpass_filter = bandpass_filter
        self.verbose = verbose

        # Validate method
        if method not in CORRELATION_METHODS:
            raise ValueError(
                f"Unknown correlation method '{method}'. "
                f"Available: {list(CORRELATION_METHODS.keys())}"
            )

        if self.verbose:
            print(f"[i] Correlation engine initialized")
            print(f"    Method: {method}")
            print(f"    GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
            print(f"    Bandpass: {'Enabled' if bandpass_filter else 'Disabled'}")

    def preprocess_signal(
        self,
        signal_data: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ) -> np.ndarray:
        """
        Preprocess signal before correlation.

        Steps:
        1. Normalize to zero mean
        2. Apply bandpass filter (if enabled)
        3. Optional: Pre-whitening

        Args:
            signal_data (np.ndarray): Input signal
            sample_rate (int): Sample rate in Hz

        Returns:
            np.ndarray: Preprocessed signal
        """
        # Copy to avoid modifying original
        processed = signal_data.copy()

        # Remove DC component (zero mean)
        processed = processed - np.mean(processed)

        # Apply bandpass filter
        if self.bandpass_filter:
            processed = self._apply_bandpass(processed, sample_rate)

        # Optional: Pre-whitening (spectral flattening)
        if PREWHITEN_ENABLE:
            processed = self._prewhiten(processed)

        # Normalize to unit variance
        std = np.std(processed)
        if std > 0:
            processed = processed / std

        return processed

    def _apply_bandpass(
        self,
        signal_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.

        Args:
            signal_data (np.ndarray): Input signal
            sample_rate (int): Sample rate in Hz

        Returns:
            np.ndarray: Filtered signal
        """
        # Design Butterworth bandpass filter
        nyquist = sample_rate / 2
        low = BANDPASS_LOW_HZ / nyquist
        high = BANDPASS_HIGH_HZ / nyquist

        # Ensure valid frequency range
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))

        if low >= high:
            if self.verbose:
                print(f"[!] Warning: Invalid bandpass range, skipping filter")
            return signal_data

        try:
            b, a = signal.butter(
                BANDPASS_ORDER,
                [low, high],
                btype='band'
            )
            filtered = signal.filtfilt(b, a, signal_data)
            return filtered
        except Exception as e:
            if self.verbose:
                print(f"[!] Warning: Bandpass filter failed: {e}")
            return signal_data

    def _prewhiten(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply pre-whitening (pre-emphasis) filter.

        Pre-whitening flattens the spectrum by applying a first-order
        high-pass filter: y[n] = x[n] - α*x[n-1]

        Args:
            signal_data (np.ndarray): Input signal

        Returns:
            np.ndarray: Pre-whitened signal
        """
        return np.append(
            signal_data[0],
            signal_data[1:] - PREWHITEN_ALPHA * signal_data[:-1]
        )

    def correlate(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sample_rate: int = SAMPLE_RATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-correlation between two signals.

        Args:
            signal_a (np.ndarray): First signal
            signal_b (np.ndarray): Second signal
            sample_rate (int): Sample rate in Hz

        Returns:
            Tuple of (correlation, lags):
                correlation (np.ndarray): Cross-correlation values
                lags (np.ndarray): Corresponding time lags (samples)
        """
        if self.verbose:
            t_start = time.time()

        # Preprocess signals
        proc_a = self.preprocess_signal(signal_a, sample_rate)
        proc_b = self.preprocess_signal(signal_b, sample_rate)

        # Compute correlation using selected method
        if self.method == 'basic':
            correlation, lags = self._correlate_basic(proc_a, proc_b)

        elif self.method == 'gcc_phat':
            correlation, lags = self._correlate_gcc_phat(proc_a, proc_b, sample_rate)

        elif self.method == 'frequency_domain':
            correlation, lags = self._correlate_frequency(proc_a, proc_b)

        elif self.method == 'adaptive':
            correlation, lags = self._correlate_adaptive(proc_a, proc_b)

        else:
            raise ValueError(f"Method '{self.method}' not implemented")

        if self.verbose:
            t_elapsed = time.time() - t_start
            print(f"[i] Correlation computed in {t_elapsed:.3f}s")

        return correlation, lags

    def _correlate_basic(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic time-domain cross-correlation.

        Uses scipy.signal.correlate with FFT method for efficiency.

        Args:
            signal_a (np.ndarray): First signal
            signal_b (np.ndarray): Second signal

        Returns:
            Tuple of (correlation, lags)
        """
        if self.use_gpu and GPU_AVAILABLE:
            # GPU-accelerated correlation
            signal_a_gpu = cp.asarray(signal_a)
            signal_b_gpu = cp.asarray(signal_b)

            # Use CuPy's correlate (uses cuFFT)
            correlation_gpu = cp.correlate(signal_a_gpu, signal_b_gpu, mode='full')
            correlation = cp.asnumpy(correlation_gpu)

        else:
            # CPU correlation using SciPy
            correlation = signal.correlate(signal_a, signal_b, mode='full', method='fft')

        # Generate lag array
        lags = signal.correlation_lags(len(signal_a), len(signal_b), mode='full')

        # Normalize correlation
        correlation = correlation / (np.std(signal_a) * np.std(signal_b) * len(signal_a))

        return correlation, lags

    def _correlate_gcc_phat(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

        GCC-PHAT is robust to reverberation and noise by normalizing
        the cross-spectrum phase while removing magnitude information.

        R_PHAT(τ) = IFFT[ (X₁(f) · X₂*(f)) / |X₁(f) · X₂*(f)| ]

        Args:
            signal_a (np.ndarray): First signal
            signal_b (np.ndarray): Second signal
            sample_rate (int): Sample rate in Hz

        Returns:
            Tuple of (correlation, lags)
        """
        # Zero-pad for better frequency resolution
        n_samples = len(signal_a)
        n_fft = int(ZERO_PAD_FACTOR * 2**np.ceil(np.log2(n_samples)))

        if self.use_gpu and GPU_AVAILABLE:
            # GPU implementation
            signal_a_gpu = cp.asarray(signal_a)
            signal_b_gpu = cp.asarray(signal_b)

            # Compute FFTs
            X1 = cp.fft.fft(signal_a_gpu, n=n_fft)
            X2 = cp.fft.fft(signal_b_gpu, n=n_fft)

            # Cross-spectrum
            cross_spectrum = X1 * cp.conj(X2)

            # PHAT weighting: normalize by magnitude
            magnitude = cp.abs(cross_spectrum) + PHAT_EPSILON
            gcc_phat = cross_spectrum / magnitude

            # IFFT to get correlation
            correlation_gpu = cp.fft.ifft(gcc_phat)
            correlation = cp.real(cp.fft.fftshift(cp.asnumpy(correlation_gpu)))

        else:
            # CPU implementation
            # Compute FFTs
            X1 = np.fft.fft(signal_a, n=n_fft)
            X2 = np.fft.fft(signal_b, n=n_fft)

            # Cross-spectrum
            cross_spectrum = X1 * np.conj(X2)

            # PHAT weighting
            magnitude = np.abs(cross_spectrum) + PHAT_EPSILON
            gcc_phat = cross_spectrum / magnitude

            # IFFT to get correlation
            correlation_complex = np.fft.ifft(gcc_phat)
            correlation = np.real(np.fft.fftshift(correlation_complex))

        # Generate lag array (centered around zero)
        lags = np.arange(-n_fft//2, n_fft//2)

        # Truncate to valid range based on max expected time delay
        max_lag_samples = seconds_to_samples(MAX_TIME_DELAY_SEC, sample_rate)
        valid_idx = np.abs(lags) <= max_lag_samples
        correlation = correlation[valid_idx]
        lags = lags[valid_idx]

        return correlation, lags

    def _correlate_frequency(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency-domain cross-correlation.

        Efficient FFT-based correlation:
        corr(x, y) = IFFT[ FFT(x) · conj(FFT(y)) ]

        Args:
            signal_a (np.ndarray): First signal
            signal_b (np.ndarray): Second signal

        Returns:
            Tuple of (correlation, lags)
        """
        n_samples = max(len(signal_a), len(signal_b))
        n_fft = int(ZERO_PAD_FACTOR * 2**np.ceil(np.log2(n_samples * 2 - 1)))

        if self.use_gpu and GPU_AVAILABLE:
            # GPU implementation
            signal_a_gpu = cp.asarray(signal_a)
            signal_b_gpu = cp.asarray(signal_b)

            X1 = cp.fft.fft(signal_a_gpu, n=n_fft)
            X2 = cp.fft.fft(signal_b_gpu, n=n_fft)

            correlation_gpu = cp.fft.ifft(X1 * cp.conj(X2))
            correlation = cp.real(cp.fft.fftshift(cp.asnumpy(correlation_gpu)))

        else:
            # CPU implementation
            X1 = np.fft.fft(signal_a, n=n_fft)
            X2 = np.fft.fft(signal_b, n=n_fft)

            correlation_complex = np.fft.ifft(X1 * np.conj(X2))
            correlation = np.real(np.fft.fftshift(correlation_complex))

        # Generate lags
        lags = np.arange(-n_fft//2, n_fft//2)

        # Normalize
        correlation = correlation / np.max(np.abs(correlation) + 1e-10)

        return correlation, lags

    def _correlate_adaptive(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive filtering correlation.

        Uses LMS adaptive filter to remove common noise before correlation.

        Args:
            signal_a (np.ndarray): First signal (reference)
            signal_b (np.ndarray): Second signal (target)

        Returns:
            Tuple of (correlation, lags)
        """
        # Apply LMS adaptive filter to remove common noise
        filtered_b = self._lms_filter(signal_a, signal_b)

        # Then compute basic correlation on filtered signals
        return self._correlate_basic(signal_a, filtered_b)

    def _lms_filter(
        self,
        reference: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Least Mean Squares (LMS) adaptive filter.

        Adapts filter to minimize difference between target and filtered reference.

        Args:
            reference (np.ndarray): Reference signal
            target (np.ndarray): Target signal

        Returns:
            np.ndarray: Filtered target signal
        """
        n_samples = len(target)
        n_taps = ADAPTIVE_FILTER_ORDER
        mu = ADAPTIVE_MU

        # Initialize filter weights and output
        weights = np.zeros(n_taps)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)

        # LMS adaptation loop
        for n in range(n_taps, n_samples):
            # Extract reference window
            x = reference[n - n_taps:n][::-1]  # Reverse for convolution

            # Filter output
            y = np.dot(weights, x)
            output[n] = y

            # Error signal
            e = target[n] - y
            error[n] = e

            # Update weights (LMS rule)
            weights = weights + mu * e * x

        return error  # Return error signal (noise-reduced target)

    def compute_snr(
        self,
        correlation: np.ndarray,
        peak_idx: int,
        noise_window: int = 100
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio of correlation peak.

        SNR = 20 * log10(peak_power / noise_power)

        Args:
            correlation (np.ndarray): Correlation function
            peak_idx (int): Index of correlation peak
            noise_window (int): Window size for noise estimation

        Returns:
            float: SNR in dB
        """
        # Peak power
        peak_power = correlation[peak_idx] ** 2

        # Noise power (estimate from regions far from peak)
        n_corr = len(correlation)
        noise_window = min(noise_window, n_corr // 4)

        # Take noise samples from both ends of correlation
        noise_left = correlation[:noise_window]
        noise_right = correlation[-noise_window:]
        noise = np.concatenate([noise_left, noise_right])

        noise_power = np.mean(noise ** 2) + 1e-10  # Avoid division by zero

        # SNR in dB
        snr_db = 10 * np.log10(peak_power / noise_power)

        return snr_db

    def __repr__(self) -> str:
        return (
            f"CorrelationEngine(method='{self.method}', "
            f"gpu={'enabled' if self.use_gpu else 'disabled'})"
        )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def plot_correlation(
    correlation: np.ndarray,
    lags: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    peak_idx: Optional[int] = None,
    output_file: Optional[str] = None,
    title: str = "Cross-Correlation"
):
    """
    Plot correlation function with optional peak marker.

    Args:
        correlation (np.ndarray): Correlation values
        lags (np.ndarray): Lag values (samples)
        sample_rate (int): Sample rate in Hz
        peak_idx (int, optional): Index of correlation peak to mark
        output_file (str, optional): Save plot to file
        title (str): Plot title
    """
    import matplotlib.pyplot as plt

    # Convert lags to time (seconds)
    time_lags = lags / sample_rate

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_lags, correlation, 'b-', linewidth=0.5, label='Correlation')

    if peak_idx is not None:
        ax.plot(
            time_lags[peak_idx],
            correlation[peak_idx],
            'ro',
            markersize=10,
            label=f'Peak (τ={time_lags[peak_idx]:.4f}s)'
        )

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Time Delay (seconds)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"[✓] Plot saved to {output_file}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# MAIN - Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Correlation Engine Testing')
    parser.add_argument('--method', default='gcc_phat',
                       choices=list(CORRELATION_METHODS.keys()),
                       help='Correlation method')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test with synthetic signals')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("CORRELATION ENGINE TEST")
    print("=" * 80)

    # Initialize engine
    engine = CorrelationEngine(
        method=args.method,
        use_gpu=args.gpu,
        verbose=args.verbose
    )

    print(f"\n[i] Engine: {engine}")
    print(f"[i] GPU Available: {GPU_AVAILABLE}")

    if args.test_synthetic:
        print("\n[i] Testing with synthetic signals...")

        # Generate synthetic leak signal
        t = np.linspace(0, SAMPLE_LENGTH_SEC, SAMPLE_RATE * SAMPLE_LENGTH_SEC)
        leak_freq = 800  # Hz (typical leak frequency)

        # Leak signal (burst of noise at leak frequency)
        leak_signal = np.random.randn(len(t)) * 0.1
        leak_signal += np.sin(2 * np.pi * leak_freq * t) * 0.5

        # Add bandlimited noise
        noise_a = np.random.randn(len(t)) * 0.2
        noise_b = np.random.randn(len(t)) * 0.2

        # Simulate time delay (e.g., 0.05 seconds = 50ms)
        true_delay_sec = 0.05
        delay_samples = int(true_delay_sec * SAMPLE_RATE)

        # Sensor A receives leak signal + noise
        signal_a = leak_signal + noise_a

        # Sensor B receives delayed leak signal + different noise
        signal_b = np.concatenate([
            np.zeros(delay_samples),
            leak_signal[:-delay_samples]
        ]) + noise_b

        print(f"[i] Synthetic signals generated")
        print(f"    Duration: {SAMPLE_LENGTH_SEC}s")
        print(f"    Sample rate: {SAMPLE_RATE} Hz")
        print(f"    True delay: {true_delay_sec}s ({delay_samples} samples)")

        # Compute correlation
        print(f"\n[i] Computing correlation...")
        t_start = time.time()
        correlation, lags = engine.correlate(signal_a, signal_b)
        t_elapsed = time.time() - t_start

        print(f"[✓] Correlation computed in {t_elapsed:.4f}s")
        print(f"    Correlation length: {len(correlation)}")
        print(f"    Lag range: [{lags[0]}, {lags[-1]}] samples")

        # Find peak
        peak_idx = np.argmax(correlation)
        estimated_delay_samples = lags[peak_idx]
        estimated_delay_sec = samples_to_seconds(estimated_delay_samples)

        print(f"\n[i] Results:")
        print(f"    Peak index: {peak_idx}")
        print(f"    Estimated delay: {estimated_delay_sec:.4f}s ({estimated_delay_samples} samples)")
        print(f"    True delay: {true_delay_sec:.4f}s ({delay_samples} samples)")
        print(f"    Error: {abs(estimated_delay_sec - true_delay_sec) * 1000:.2f}ms")

        # Compute SNR
        snr = engine.compute_snr(correlation, peak_idx)
        print(f"    SNR: {snr:.1f} dB")

        # Plot if requested
        if args.plot:
            print(f"\n[i] Generating plots...")

            import matplotlib.pyplot as plt

            # Plot signals
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            axes[0].plot(t, signal_a, 'b-', linewidth=0.5)
            axes[0].set_ylabel('Sensor A', fontsize=12)
            axes[0].set_title('Input Signals', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(t, signal_b, 'r-', linewidth=0.5)
            axes[1].set_ylabel('Sensor B', fontsize=12)
            axes[1].set_xlabel('Time (seconds)', fontsize=12)
            axes[1].grid(True, alpha=0.3)

            # Plot correlation
            time_lags = samples_to_seconds(lags)
            axes[2].plot(time_lags, correlation, 'g-', linewidth=1)
            axes[2].axvline(x=estimated_delay_sec, color='r', linestyle='--',
                           label=f'Estimated: {estimated_delay_sec:.4f}s')
            axes[2].axvline(x=true_delay_sec, color='b', linestyle='--',
                           label=f'True: {true_delay_sec:.4f}s')
            axes[2].set_ylabel('Correlation', fontsize=12)
            axes[2].set_xlabel('Time Delay (seconds)', fontsize=12)
            axes[2].set_title('Cross-Correlation', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()
            plt.savefig('correlation_test.png', dpi=150, bbox_inches='tight')
            print(f"[✓] Plot saved to correlation_test.png")
            plt.close()

    print("\n[✓] Test complete")
