#!/usr/bin/env python3
"""
Coherence Analysis and Band Selection
Compute magnitude-squared coherence and auto-detect leak-relevant frequency bands

Version: 3.0.0
Revision: 1
Date: 2025-11-25
Status: Production
"""

import numpy as np
from scipy.signal import coherence, welch, butter, filtfilt
from typing import List, Tuple
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from correlator_v3_config import *


@dataclass
class CoherenceResult:
    """Result from coherence analysis"""
    frequencies: np.ndarray
    coherence: np.ndarray
    high_coherence_bands: List[Tuple[float, float]]
    mean_coherence: float


class CoherenceAnalyzer:
    """
    Compute coherence and auto-select leak-relevant frequency bands.

    Magnitude-squared coherence: C_xy(f) = |E[X(f)Y*(f)]|² / (E[|X(f)|²]E[|Y(f)|²])
    """

    def __init__(self, sample_rate=4096, verbose=False):
        self.sample_rate = sample_rate
        self.verbose = verbose

        if self.verbose:
            print(f"[i] CoherenceAnalyzer initialized (sample_rate={sample_rate} Hz)")

    def compute_coherence(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        nperseg: int = 512
    ) -> CoherenceResult:
        """
        Compute magnitude-squared coherence between two signals.

        Returns coherence for each frequency bin (0-1).
        High coherence indicates signals are strongly correlated at that frequency.
        """
        # Compute coherence using Welch's method
        freqs, coh = coherence(signal_a, signal_b, fs=self.sample_rate,
                               nperseg=nperseg, noverlap=nperseg // 2)

        # Auto-detect high-coherence bands
        high_coherence_bands = self.auto_select_bands(
            coh, freqs,
            threshold=COHERENCE_THRESHOLD,
            min_width_hz=MIN_BAND_WIDTH_HZ
        )

        mean_coh = np.mean(coh)

        if self.verbose:
            print(f"\n[i] Coherence analysis:")
            print(f"    Mean coherence: {mean_coh:.3f}")
            print(f"    High-coherence bands: {len(high_coherence_bands)}")
            for low, high in high_coherence_bands:
                band_mask = (freqs >= low) & (freqs <= high)
                band_coh = np.mean(coh[band_mask])
                print(f"      {low:.0f}-{high:.0f} Hz (coherence: {band_coh:.3f})")

        return CoherenceResult(
            frequencies=freqs,
            coherence=coh,
            high_coherence_bands=high_coherence_bands,
            mean_coherence=float(mean_coh)
        )

    def auto_select_bands(
        self,
        coherence: np.ndarray,
        frequencies: np.ndarray,
        threshold: float = 0.7,
        min_width_hz: float = 50
    ) -> List[Tuple[float, float]]:
        """
        Auto-detect high-coherence frequency bands.

        Returns list of (low_hz, high_hz) tuples for each contiguous
        region where coherence > threshold.
        """
        # Find regions above threshold
        above_threshold = coherence > threshold

        # Find contiguous regions
        bands = []
        in_band = False
        band_start = None

        for i, (freq, above) in enumerate(zip(frequencies, above_threshold)):
            if above and not in_band:
                # Start of new band
                band_start = freq
                in_band = True
            elif not above and in_band:
                # End of band
                band_end = frequencies[i - 1]
                if band_end - band_start >= min_width_hz:
                    bands.append((band_start, band_end))
                in_band = False

        # Close last band if still open
        if in_band:
            band_end = frequencies[-1]
            if band_end - band_start >= min_width_hz:
                bands.append((band_start, band_end))

        # Limit to MAX_COHERENCE_BANDS
        if len(bands) > MAX_COHERENCE_BANDS:
            # Keep bands with highest mean coherence
            band_scores = []
            for low, high in bands:
                mask = (frequencies >= low) & (frequencies <= high)
                mean_coh = np.mean(coherence[mask])
                band_scores.append((mean_coh, (low, high)))

            band_scores.sort(reverse=True)
            bands = [band for _, band in band_scores[:MAX_COHERENCE_BANDS]]

        return bands

    def coherence_weighted_correlation(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        bands: List[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Perform correlation weighted by coherence in each band.

        High-coherence bands contribute more to final correlation.
        """
        if bands is None:
            # Compute coherence and auto-select bands
            coh_result = self.compute_coherence(signal_a, signal_b)
            bands = coh_result.high_coherence_bands

        if len(bands) == 0:
            # No high-coherence bands, use full-band correlation
            if self.verbose:
                print("[!] No high-coherence bands found, using full-band")
            return np.fft.fftshift(np.fft.ifft(
                np.fft.fft(signal_a) * np.conj(np.fft.fft(signal_b))
            ).real)

        # Bandpass filter and correlate each band
        weighted_corr = None
        total_weight = 0

        for low_hz, high_hz in bands:
            # Filter to this band
            sig_a_band = self.bandpass_filter(signal_a, low_hz, high_hz)
            sig_b_band = self.bandpass_filter(signal_b, low_hz, high_hz)

            # Compute coherence in this band (weight)
            freqs, coh = coherence(sig_a_band, sig_b_band, fs=self.sample_rate)
            band_coherence = np.mean(coh)
            weight = band_coherence ** COHERENCE_WEIGHT_EXPONENT

            # Correlate
            corr = np.fft.fftshift(np.fft.ifft(
                np.fft.fft(sig_a_band) * np.conj(np.fft.fft(sig_b_band))
            ).real)

            # Accumulate weighted
            if weighted_corr is None:
                weighted_corr = corr * weight
            else:
                weighted_corr += corr * weight

            total_weight += weight

            if self.verbose:
                print(f"    Band {low_hz:.0f}-{high_hz:.0f} Hz: coherence={band_coherence:.3f}, weight={weight:.3f}")

        # Normalize
        if total_weight > 0:
            weighted_corr /= total_weight

        if self.verbose:
            print(f"    [✓] Weighted correlation from {len(bands)} bands")

        return weighted_corr

    def bandpass_filter(self, signal: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Apply bandpass filter"""
        nyquist = self.sample_rate / 2
        low = low_hz / nyquist
        high = high_hz / nyquist

        # Clamp to valid range
        low = max(0.01, min(0.99, low))
        high = max(0.01, min(0.99, high))

        if high <= low:
            # Invalid range, return original
            return signal

        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("COHERENCE ANALYZER TEST")
    print("=" * 80)

    # Generate test signals with leak in specific frequency band
    sample_rate = 4096
    duration = 10.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    # Leak signal at 800 Hz (high coherence expected in 700-900 Hz band)
    leak_freq = 800
    leak_signal = np.sin(2 * np.pi * leak_freq * t)

    # Add random noise (low coherence)
    noise_a = np.random.randn(n_samples) * 0.3
    noise_b = np.random.randn(n_samples) * 0.3

    # Sensor signals
    signal_a = leak_signal + noise_a
    signal_b = np.roll(leak_signal, 50) + noise_b  # Delayed leak

    # Test coherence analysis
    analyzer = CoherenceAnalyzer(sample_rate=sample_rate, verbose=True)

    print("\n[TEST 1] Coherence computation")
    result = analyzer.compute_coherence(signal_a, signal_b)

    # Check if leak frequency band detected
    leak_band_detected = False
    for low, high in result.high_coherence_bands:
        if low <= leak_freq <= high:
            leak_band_detected = True
            print(f"  [✓] Leak band detected: {low:.0f}-{high:.0f} Hz (contains {leak_freq} Hz)")

    print(f"  Test: {'PASS' if leak_band_detected else 'FAIL'}")

    # Test coherence-weighted correlation
    print("\n[TEST 2] Coherence-weighted correlation")
    weighted_corr = analyzer.coherence_weighted_correlation(signal_a, signal_b)

    # Find peak
    center = len(weighted_corr) // 2
    peak_idx = np.argmax(np.abs(weighted_corr))
    estimated_delay = peak_idx - center

    print(f"  True delay: 50 samples, Estimated: {estimated_delay}")
    error = abs(estimated_delay - 50)
    print(f"  Error: {error} samples")
    print(f"  Test: {'PASS' if error < 10 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] Coherence analyzer test complete!")
