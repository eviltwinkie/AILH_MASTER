#!/usr/bin/env python3
"""
CORRELATOR_v2 - Statistical Features Extraction Module

Extract advanced statistical features from audio signals for enhanced
leak characterization and detection confidence.

Features extracted (v3.2 - ported from CORRELATOR_v1):
- RMS (Root Mean Square): Signal energy level
- Kurtosis: Signal impulsiveness (leaks often have high kurtosis)
- Spectral Entropy: Frequency distribution complexity
- Dominant Frequency: Peak frequency component in signal

These features complement existing SNR and peak sharpness metrics
to improve leak detection accuracy.

Author: AILH Development Team
Date: 2025-11-19
Version: 3.2.0
"""

import numpy as np
from scipy.stats import kurtosis, entropy
from scipy.signal import welch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignalStatistics:
    """
    Statistical features extracted from audio signal.

    Attributes:
        rms: Root Mean Square (signal energy level)
        kurtosis: Statistical kurtosis (impulsiveness measure)
        spectral_entropy: Shannon entropy of power spectrum (complexity)
        dominant_frequency_hz: Frequency with highest power
        dominant_frequency_power: Power at dominant frequency
        band_power_ratio: Ratio of band power to total power
    """
    rms: float
    kurtosis: float
    spectral_entropy: float
    dominant_frequency_hz: float
    dominant_frequency_power: float
    band_power_ratio: float


class StatisticalFeatureExtractor:
    """
    Extract statistical features from audio signals for leak characterization.

    Features:
    - RMS: Signal energy level (higher for leaks)
    - Kurtosis: Signal impulsiveness (leaks have high kurtosis due to turbulence)
    - Spectral Entropy: Frequency complexity (leaks have lower entropy)
    - Dominant Frequency: Peak frequency component (leak signature)
    """

    def __init__(self, sample_rate: int = 4096, verbose: bool = False):
        """
        Initialize statistical feature extractor.

        Args:
            sample_rate: Audio sample rate (Hz)
            verbose: Print extraction details
        """
        self.sample_rate = sample_rate
        self.verbose = verbose

    def extract_features(
        self,
        signal: np.ndarray,
        freq_band: Tuple[float, float] = (100, 1500)
    ) -> SignalStatistics:
        """
        Extract all statistical features from signal.

        Args:
            signal: Audio signal (1D numpy array)
            freq_band: Frequency band for band power calculation (Hz)

        Returns:
            SignalStatistics object with all extracted features
        """
        # RMS (signal energy)
        rms = self._calculate_rms(signal)

        # Kurtosis (impulsiveness)
        kurt = self._calculate_kurtosis(signal)

        # Spectral features
        spec_entropy, dom_freq, dom_power = self._calculate_spectral_features(signal)

        # Band power ratio
        band_ratio = self._calculate_band_power_ratio(signal, freq_band)

        return SignalStatistics(
            rms=rms,
            kurtosis=kurt,
            spectral_entropy=spec_entropy,
            dominant_frequency_hz=dom_freq,
            dominant_frequency_power=dom_power,
            band_power_ratio=band_ratio
        )

    def _calculate_rms(self, signal: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) of signal.

        RMS represents the signal energy level. Leaks typically have
        higher RMS due to turbulent flow noise.

        Args:
            signal: Audio signal

        Returns:
            RMS value
        """
        return float(np.sqrt(np.mean(signal ** 2)))

    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """
        Calculate statistical kurtosis of signal.

        Kurtosis measures the "tailedness" of the distribution.
        High kurtosis indicates impulsive signals (sharp peaks).
        Leaks often exhibit high kurtosis due to turbulent bursts.

        Args:
            signal: Audio signal

        Returns:
            Kurtosis value (excess kurtosis, normal distribution = 0)

        Notes:
            - Normal distribution: kurtosis ≈ 0
            - Leak signals: kurtosis > 3 (heavy tails, impulsive)
            - Background noise: kurtosis ≈ 0 (Gaussian-like)
        """
        return float(kurtosis(signal, fisher=True))

    def _calculate_spectral_features(
        self,
        signal: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate spectral entropy and dominant frequency.

        Spectral entropy measures frequency distribution complexity.
        Lower entropy indicates more structured/tonal signals (leaks).
        Higher entropy indicates more random/noise-like signals.

        Args:
            signal: Audio signal

        Returns:
            Tuple of (spectral_entropy, dominant_freq_hz, dominant_power)

        Notes:
            - Low spectral entropy → leak (tonal, structured)
            - High spectral entropy → noise (random, diffuse)
        """
        # Compute power spectral density
        freqs, psd = welch(signal, fs=self.sample_rate, nperseg=min(2048, len(signal)))

        # Normalize PSD for entropy calculation
        psd_norm = psd / (np.sum(psd) + 1e-12)

        # Calculate Shannon entropy
        spec_entropy = float(entropy(psd_norm + 1e-12))

        # Find dominant frequency
        peak_idx = np.argmax(psd)
        dominant_freq = float(freqs[peak_idx])
        dominant_power = float(psd[peak_idx])

        return spec_entropy, dominant_freq, dominant_power

    def _calculate_band_power_ratio(
        self,
        signal: np.ndarray,
        freq_band: Tuple[float, float]
    ) -> float:
        """
        Calculate ratio of power in frequency band to total power.

        Leaks concentrate energy in specific frequency bands (100-1500 Hz).
        High band power ratio indicates leak signature.

        Args:
            signal: Audio signal
            freq_band: (low_freq, high_freq) in Hz

        Returns:
            Band power ratio (0-1)

        Notes:
            - Leak signals: high band power ratio (>0.4 typical)
            - Background noise: low band power ratio (<0.3 typical)
        """
        freqs, psd = welch(signal, fs=self.sample_rate, nperseg=min(2048, len(signal)))

        # Band power (within specified frequency range)
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        band_power = np.sum(psd[band_mask])

        # Total power
        total_power = np.sum(psd)

        # Ratio
        ratio = float(band_power / (total_power + 1e-12))

        return ratio

    def extract_dominant_frequency_from_segment(
        self,
        signal: np.ndarray,
        segment_ms: int = 100
    ) -> Tuple[float, float]:
        """
        Extract dominant frequency from a specific segment of the signal.

        This is useful for characterizing individual leak signatures
        within a longer recording.

        Args:
            signal: Audio signal
            segment_ms: Segment length in milliseconds

        Returns:
            Tuple of (dominant_freq_hz, dominant_power)
        """
        segment_samples = int((segment_ms / 1000.0) * self.sample_rate)

        if len(signal) < segment_samples:
            segment_samples = len(signal)

        # Extract middle segment
        start_idx = max(0, (len(signal) - segment_samples) // 2)
        end_idx = start_idx + segment_samples
        segment = signal[start_idx:end_idx]

        # FFT analysis
        fft_vals = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / self.sample_rate)

        # Find peak
        peak_idx = np.argmax(fft_vals)
        dominant_freq = float(freqs[peak_idx])
        dominant_power = float(fft_vals[peak_idx])

        return dominant_freq, dominant_power

    def compare_signals(
        self,
        signal: SignalStatistics,
        baseline: SignalStatistics
    ) -> Dict[str, float]:
        """
        Compare signal statistics to baseline.

        Args:
            signal: Statistics from test signal
            baseline: Statistics from baseline (no-leak) signal

        Returns:
            Dictionary of comparison ratios

        Notes:
            - RMS ratio > 1.2: Elevated energy (leak indicator)
            - Kurtosis ratio > 1.2: More impulsive (leak indicator)
            - Spectral entropy ratio < 1.0: More structured (leak indicator)
        """
        return {
            'rms_ratio': signal.rms / (baseline.rms + 1e-12),
            'kurtosis_ratio': signal.kurtosis / (baseline.kurtosis + 1e-12),
            'spectral_entropy_ratio': signal.spectral_entropy / (baseline.spectral_entropy + 1e-12),
            'band_power_ratio_diff': signal.band_power_ratio - baseline.band_power_ratio
        }


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("STATISTICAL FEATURES EXTRACTOR TEST")
    print("=" * 80)

    # Generate test signals
    sample_rate = 4096
    duration = 10.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Test signal 1: Pure sine wave (leak-like)
    freq_leak = 800  # Hz
    signal_leak = np.sin(2 * np.pi * freq_leak * t)

    # Test signal 2: White noise (background)
    signal_noise = np.random.randn(len(t)) * 0.1

    # Test signal 3: Leak + noise
    signal_mixed = signal_leak + signal_noise

    # Extract features
    extractor = StatisticalFeatureExtractor(sample_rate=sample_rate, verbose=True)

    print("\n[i] Extracting features from leak signal (pure sine)...")
    stats_leak = extractor.extract_features(signal_leak)
    print(f"    RMS: {stats_leak.rms:.6f}")
    print(f"    Kurtosis: {stats_leak.kurtosis:.3f}")
    print(f"    Spectral Entropy: {stats_leak.spectral_entropy:.3f}")
    print(f"    Dominant Frequency: {stats_leak.dominant_frequency_hz:.1f} Hz")
    print(f"    Band Power Ratio: {stats_leak.band_power_ratio:.3f}")

    print("\n[i] Extracting features from noise signal...")
    stats_noise = extractor.extract_features(signal_noise)
    print(f"    RMS: {stats_noise.rms:.6f}")
    print(f"    Kurtosis: {stats_noise.kurtosis:.3f}")
    print(f"    Spectral Entropy: {stats_noise.spectral_entropy:.3f}")
    print(f"    Dominant Frequency: {stats_noise.dominant_frequency_hz:.1f} Hz")
    print(f"    Band Power Ratio: {stats_noise.band_power_ratio:.3f}")

    print("\n[i] Extracting features from mixed signal (leak + noise)...")
    stats_mixed = extractor.extract_features(signal_mixed)
    print(f"    RMS: {stats_mixed.rms:.6f}")
    print(f"    Kurtosis: {stats_mixed.kurtosis:.3f}")
    print(f"    Spectral Entropy: {stats_mixed.spectral_entropy:.3f}")
    print(f"    Dominant Frequency: {stats_mixed.dominant_frequency_hz:.1f} Hz")
    print(f"    Band Power Ratio: {stats_mixed.band_power_ratio:.3f}")

    print("\n[i] Comparing leak signal to noise baseline...")
    comparison = extractor.compare_signals(stats_leak, stats_noise)
    print(f"    RMS Ratio: {comparison['rms_ratio']:.3f}")
    print(f"    Kurtosis Ratio: {comparison['kurtosis_ratio']:.3f}")
    print(f"    Spectral Entropy Ratio: {comparison['spectral_entropy_ratio']:.3f}")
    print(f"    Band Power Ratio Diff: {comparison['band_power_ratio_diff']:.3f}")

    print("\n[✓] Statistical features extractor test complete!")
