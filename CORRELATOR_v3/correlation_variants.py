#!/usr/bin/env python3
"""
Multiple Correlation Variants
GCC-PHAT, GCC-Roth, GCC-SCOT, Classical, Wavelet
"""

import numpy as np
from scipy.signal import correlate, hilbert
import pywt
from typing import Dict, Optional, List

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from correlator_v3_config import *


def gcc_phat(signal_a: np.ndarray, signal_b: np.ndarray,
             sample_rate: int = 4096, use_gpu: bool = True) -> np.ndarray:
    """
    GCC with Phase Transform (PHAT) weighting.
    W(f) = 1 / |X(f)Y*(f)|
    Robust to reverberation.
    """
    if use_gpu and GPU_AVAILABLE:
        sig_a = cp.asarray(signal_a, dtype=cp.float32)
        sig_b = cp.asarray(signal_b, dtype=cp.float32)

        # FFT
        X = cp.fft.fft(sig_a)
        Y = cp.fft.fft(sig_b)

        # Cross-spectrum
        cross_spectrum = X * cp.conj(Y)

        # PHAT weighting
        magnitude = cp.abs(cross_spectrum) + 1e-10
        weighted = cross_spectrum / magnitude

        # IFFT
        correlation = cp.fft.ifft(weighted).real
        correlation = cp.fft.fftshift(correlation)

        return cp.asnumpy(correlation)
    else:
        X = np.fft.fft(signal_a)
        Y = np.fft.fft(signal_b)
        cross_spectrum = X * np.conj(Y)
        magnitude = np.abs(cross_spectrum) + 1e-10
        weighted = cross_spectrum / magnitude
        correlation = np.fft.ifft(weighted).real
        correlation = np.fft.fftshift(correlation)
        return correlation


def gcc_roth(signal_a: np.ndarray, signal_b: np.ndarray,
             sample_rate: int = 4096, use_gpu: bool = True) -> np.ndarray:
    """
    GCC with Roth weighting.
    W(f) = 1 / |X(f)|²
    Good for colored noise.
    """
    if use_gpu and GPU_AVAILABLE:
        sig_a = cp.asarray(signal_a, dtype=cp.float32)
        sig_b = cp.asarray(signal_b, dtype=cp.float32)

        X = cp.fft.fft(sig_a)
        Y = cp.fft.fft(sig_b)

        cross_spectrum = X * cp.conj(Y)

        # Roth weighting
        magnitude_x_sq = cp.abs(X) ** 2 + 1e-10
        weighted = cross_spectrum / magnitude_x_sq

        correlation = cp.fft.ifft(weighted).real
        correlation = cp.fft.fftshift(correlation)

        return cp.asnumpy(correlation)
    else:
        X = np.fft.fft(signal_a)
        Y = np.fft.fft(signal_b)
        cross_spectrum = X * np.conj(Y)
        magnitude_x_sq = np.abs(X) ** 2 + 1e-10
        weighted = cross_spectrum / magnitude_x_sq
        correlation = np.fft.ifft(weighted).real
        correlation = np.fft.fftshift(correlation)
        return correlation


def gcc_scot(signal_a: np.ndarray, signal_b: np.ndarray,
             sample_rate: int = 4096, use_gpu: bool = True) -> np.ndarray:
    """
    GCC with Smoothed Coherence Transform (SCOT).
    W(f) = 1 / sqrt(|X(f)|² · |Y(f)|²)
    Robust to reverberant environments.
    """
    if use_gpu and GPU_AVAILABLE:
        sig_a = cp.asarray(signal_a, dtype=cp.float32)
        sig_b = cp.asarray(signal_b, dtype=cp.float32)

        X = cp.fft.fft(sig_a)
        Y = cp.fft.fft(sig_b)

        cross_spectrum = X * cp.conj(Y)

        # SCOT weighting
        denominator = cp.sqrt(cp.abs(X) ** 2 * cp.abs(Y) ** 2) + 1e-10
        weighted = cross_spectrum / denominator

        correlation = cp.fft.ifft(weighted).real
        correlation = cp.fft.fftshift(correlation)

        return cp.asnumpy(correlation)
    else:
        X = np.fft.fft(signal_a)
        Y = np.fft.fft(signal_b)
        cross_spectrum = X * np.conj(Y)
        denominator = np.sqrt(np.abs(X) ** 2 * np.abs(Y) ** 2) + 1e-10
        weighted = cross_spectrum / denominator
        correlation = np.fft.ifft(weighted).real
        correlation = np.fft.fftshift(correlation)
        return correlation


def classical_correlation(signal_a: np.ndarray, signal_b: np.ndarray,
                          use_gpu: bool = True) -> np.ndarray:
    """
    Unweighted time-domain cross-correlation.
    """
    if use_gpu and GPU_AVAILABLE:
        sig_a = cp.asarray(signal_a, dtype=cp.float32)
        sig_b = cp.asarray(signal_b, dtype=cp.float32)

        X = cp.fft.fft(sig_a)
        Y = cp.fft.fft(sig_b)
        cross_spectrum = X * cp.conj(Y)
        correlation = cp.fft.ifft(cross_spectrum).real
        correlation = cp.fft.fftshift(correlation)

        return cp.asnumpy(correlation)
    else:
        return correlate(signal_a, signal_b, mode='same', method='fft')


def wavelet_correlation(signal_a: np.ndarray, signal_b: np.ndarray,
                       wavelet: str = 'db4', levels: int = 5) -> np.ndarray:
    """
    Multi-resolution wavelet-based correlation.
    Performs correlation at multiple wavelet decomposition levels.
    """
    # Wavelet decomposition
    coeffs_a = pywt.wavedec(signal_a, wavelet, level=levels)
    coeffs_b = pywt.wavedec(signal_b, wavelet, level=levels)

    # Correlate at each level
    corr_levels = []
    for i in range(len(coeffs_a)):
        # Pad to same length
        max_len = max(len(coeffs_a[i]), len(coeffs_b[i]))
        ca = np.pad(coeffs_a[i], (0, max_len - len(coeffs_a[i])))
        cb = np.pad(coeffs_b[i], (0, max_len - len(coeffs_b[i])))

        # Correlate
        corr = correlate(ca, cb, mode='same', method='fft')
        corr_levels.append(corr)

    # Reconstruct correlation from all levels
    # Weight by energy (coarser levels = more weight)
    weighted_corr = None
    for i, corr in enumerate(corr_levels):
        weight = 2 ** i  # Higher levels get more weight
        if weighted_corr is None:
            # Upsample to full length
            weighted_corr = np.interp(
                np.linspace(0, len(corr) - 1, len(signal_a)),
                np.arange(len(corr)),
                corr
            ) * weight
        else:
            corr_upsampled = np.interp(
                np.linspace(0, len(corr) - 1, len(signal_a)),
                np.arange(len(corr)),
                corr
            )
            weighted_corr += corr_upsampled * weight

    # Normalize
    weighted_corr /= np.sum([2 ** i for i in range(len(corr_levels))])

    return weighted_corr


def fuse_correlations(
    correlations: Dict[str, np.ndarray],
    method: str = 'weighted_average',
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Fuse multiple correlation functions.

    Methods:
    - 'weighted_average': Weighted average of all methods
    - 'best_snr': Select method with highest SNR (peak-to-noise ratio)
    - 'majority_vote': Vote on peak location
    """
    if method == 'weighted_average':
        if weights is None:
            # Equal weights
            weights = {k: 1.0 / len(correlations) for k in correlations.keys()}

        fused = None
        for name, corr in correlations.items():
            weight = weights.get(name, 1.0)
            if fused is None:
                fused = corr * weight
            else:
                fused += corr * weight

        return fused

    elif method == 'best_snr':
        # Compute SNR for each
        best_snr = -np.inf
        best_corr = None

        for name, corr in correlations.items():
            peak_val = np.max(np.abs(corr))
            noise_val = np.mean(np.abs(corr))
            snr = peak_val / (noise_val + 1e-12)

            if snr > best_snr:
                best_snr = snr
                best_corr = corr

        return best_corr

    elif method == 'majority_vote':
        # Find peak in each
        peaks = []
        for corr in correlations.values():
            peak_idx = np.argmax(np.abs(corr))
            peaks.append(peak_idx)

        # Use median peak location
        median_peak_idx = int(np.median(peaks))

        # Return correlation with peak closest to median
        best_corr = None
        min_dist = np.inf

        for corr in correlations.values():
            peak_idx = np.argmax(np.abs(corr))
            dist = abs(peak_idx - median_peak_idx)
            if dist < min_dist:
                min_dist = dist
                best_corr = corr

        return best_corr

    else:
        raise ValueError(f"Unknown fusion method: {method}")


class MultiMethodCorrelator:
    """
    Correlator that runs multiple methods and fuses results.
    """

    def __init__(self, methods: List[str] = None, fusion_method: str = 'weighted_average',
                 use_gpu: bool = True, verbose: bool = False):
        self.methods = methods or ENABLED_CORRELATION_METHODS
        self.fusion_method = fusion_method
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose

        if self.verbose:
            print(f"[i] MultiMethodCorrelator initialized")
            print(f"    Methods: {self.methods}")
            print(f"    Fusion: {fusion_method}")

    def correlate(self, signal_a: np.ndarray, signal_b: np.ndarray,
                  sample_rate: int = 4096) -> Dict[str, np.ndarray]:
        """
        Run all enabled correlation methods.
        """
        correlations = {}

        for method in self.methods:
            if self.verbose:
                print(f"    Computing {method}...")

            if method == 'gcc_phat':
                corr = gcc_phat(signal_a, signal_b, sample_rate, self.use_gpu)
            elif method == 'gcc_roth':
                corr = gcc_roth(signal_a, signal_b, sample_rate, self.use_gpu)
            elif method == 'gcc_scot':
                corr = gcc_scot(signal_a, signal_b, sample_rate, self.use_gpu)
            elif method == 'classical':
                corr = classical_correlation(signal_a, signal_b, self.use_gpu)
            elif method == 'wavelet':
                corr = wavelet_correlation(signal_a, signal_b)
            else:
                print(f"[!] Unknown method: {method}")
                continue

            correlations[method] = corr

        return correlations

    def correlate_and_fuse(self, signal_a: np.ndarray, signal_b: np.ndarray,
                           sample_rate: int = 4096) -> np.ndarray:
        """
        Correlate with all methods and fuse.
        """
        correlations = self.correlate(signal_a, signal_b, sample_rate)
        fused = fuse_correlations(correlations, method=self.fusion_method)

        if self.verbose:
            print(f"    [✓] Fused {len(correlations)} methods")

        return fused


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("CORRELATION VARIANTS TEST")
    print("=" * 80)

    # Generate test signals
    sample_rate = 4096
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    # Signal with known delay
    delay_samples = 100
    signal_a = np.sin(2 * np.pi * 500 * t) + np.random.randn(n_samples) * 0.1
    signal_b = np.roll(signal_a, delay_samples)

    # Test each method
    methods = ['gcc_phat', 'gcc_roth', 'gcc_scot', 'classical', 'wavelet']

    for method in methods:
        print(f"\n[TEST] {method.upper()}")

        if method == 'gcc_phat':
            corr = gcc_phat(signal_a, signal_b, sample_rate, use_gpu=False)
        elif method == 'gcc_roth':
            corr = gcc_roth(signal_a, signal_b, sample_rate, use_gpu=False)
        elif method == 'gcc_scot':
            corr = gcc_scot(signal_a, signal_b, sample_rate, use_gpu=False)
        elif method == 'classical':
            corr = classical_correlation(signal_a, signal_b, use_gpu=False)
        elif method == 'wavelet':
            corr = wavelet_correlation(signal_a, signal_b)

        # Find peak
        center = len(corr) // 2
        peak_idx = np.argmax(np.abs(corr))
        estimated_delay = peak_idx - center

        error = abs(estimated_delay - delay_samples)
        print(f"  True delay: {delay_samples}, Estimated: {estimated_delay}, Error: {error}")
        print(f"  Test: {'PASS' if error < 5 else 'FAIL'}")

    # Test multi-method fusion
    print("\n[TEST] Multi-Method Fusion")
    correlator = MultiMethodCorrelator(methods=['gcc_phat', 'gcc_scot'], verbose=True)
    fused_corr = correlator.correlate_and_fuse(signal_a, signal_b, sample_rate)

    center = len(fused_corr) // 2
    peak_idx = np.argmax(np.abs(fused_corr))
    estimated_delay = peak_idx - center
    error = abs(estimated_delay - delay_samples)

    print(f"  Fused delay: {estimated_delay}, Error: {error}")
    print(f"  Test: {'PASS' if error < 5 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] Correlation variants test complete!")
