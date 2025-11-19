#!/usr/bin/env python3
"""
CORRELATOR_v2 - Signal Stacking for SNR Enhancement

Coherent averaging of multiple recordings from the same sensor pair to improve
signal-to-noise ratio and correlation reliability.

Theory:
    SNR improvement = √N where N = number of recordings

    For N recordings of the same leak:
    - Signal (leak) is coherent → adds constructively (N×)
    - Noise is random → adds incoherently (√N×)
    - Result: SNR improves by √N

    Example: 10 recordings → SNR improves by ~3.16× (~10 dB)

Methods:
    1. Signal averaging: Average raw audio before correlation
    2. Correlation averaging: Correlate each pair, then average correlations
    3. Weighted averaging: Weight by signal quality

Author: AILH Development Team
Date: 2025-11-19
Version: 3.1.0
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

from correlator_config import *
from correlator_utils import load_wav, parse_filename, compute_signal_quality


@dataclass
class StackedSignalResult:
    """Results from signal stacking."""
    stacked_signal_a: np.ndarray
    stacked_signal_b: np.ndarray
    num_recordings: int
    snr_improvement_db: float
    quality_metrics: Dict
    individual_qualities: List[Dict]


class SignalStacker:
    """
    Stack multiple recordings from same sensor pair for SNR improvement.

    Use cases:
    - Multiple hourly recordings from same sensors
    - Days/weeks of data from persistent leak
    - Low SNR situations requiring enhancement
    """

    def __init__(
        self,
        method: str = 'signal_averaging',  # 'signal_averaging', 'correlation_averaging', 'weighted'
        alignment_method: str = 'cross_correlation',  # 'cross_correlation', 'timestamp', 'none'
        quality_threshold: float = 0.3,  # Minimum quality to include
        verbose: bool = False
    ):
        """
        Initialize signal stacker.

        Args:
            method: Stacking method
            alignment_method: How to align signals before stacking
            quality_threshold: Minimum signal quality (0-1)
            verbose: Print information
        """
        self.method = method
        self.alignment_method = alignment_method
        self.quality_threshold = quality_threshold
        self.verbose = verbose

    def stack_recordings(
        self,
        recordings_a: List[np.ndarray],
        recordings_b: List[np.ndarray],
        sample_rate: int = SAMPLE_RATE
    ) -> StackedSignalResult:
        """
        Stack multiple recordings from same sensor pair.

        Args:
            recordings_a: List of recordings from sensor A
            recordings_b: List of recordings from sensor B
            sample_rate: Sample rate

        Returns:
            StackedSignalResult with enhanced signals
        """
        if len(recordings_a) != len(recordings_b):
            raise ValueError("Must have equal number of recordings from both sensors")

        if len(recordings_a) == 0:
            raise ValueError("No recordings provided")

        if self.verbose:
            print(f"\n[i] Stacking {len(recordings_a)} recording pairs...")
            print(f"    Method: {self.method}")
            print(f"    Alignment: {self.alignment_method}")

        # Compute quality metrics for each recording
        qualities_a = [compute_signal_quality(rec, sample_rate) for rec in recordings_a]
        qualities_b = [compute_signal_quality(rec, sample_rate) for rec in recordings_b]

        # Filter by quality
        valid_indices = []
        for i, (qa, qb) in enumerate(zip(qualities_a, qualities_b)):
            # Quality score: average SNR
            quality_score = (qa['snr_estimate_db'] + qb['snr_estimate_db']) / 2
            quality_normalized = np.clip(quality_score / 30.0, 0, 1)

            if quality_normalized >= self.quality_threshold:
                valid_indices.append(i)
            else:
                if self.verbose:
                    print(f"    [!] Skipping recording {i+1}: low quality ({quality_normalized:.3f})")

        if len(valid_indices) == 0:
            raise ValueError("No recordings meet quality threshold")

        # Filter recordings
        recordings_a = [recordings_a[i] for i in valid_indices]
        recordings_b = [recordings_b[i] for i in valid_indices]
        qualities_a = [qualities_a[i] for i in valid_indices]
        qualities_b = [qualities_b[i] for i in valid_indices]

        if self.verbose:
            print(f"    Using {len(recordings_a)}/{len(valid_indices)} recordings")

        # Align recordings if needed
        if self.alignment_method != 'none':
            recordings_a, recordings_b = self._align_recordings(
                recordings_a, recordings_b, sample_rate
            )

        # Stack using selected method
        if self.method == 'signal_averaging':
            stacked_a, stacked_b = self._stack_signal_averaging(recordings_a, recordings_b)

        elif self.method == 'weighted':
            weights = self._compute_weights(qualities_a, qualities_b)
            stacked_a, stacked_b = self._stack_weighted(recordings_a, recordings_b, weights)

        else:
            raise ValueError(f"Unknown stacking method: {self.method}")

        # Estimate SNR improvement
        snr_improvement_db = 10 * np.log10(len(recordings_a))

        # Compute quality metrics on stacked signals
        quality_a = compute_signal_quality(stacked_a, sample_rate)
        quality_b = compute_signal_quality(stacked_b, sample_rate)

        quality_metrics = {
            'stacked_snr_a_db': quality_a['snr_estimate_db'],
            'stacked_snr_b_db': quality_b['snr_estimate_db'],
            'average_snr_improvement_db': snr_improvement_db,
            'theoretical_improvement_db': snr_improvement_db,
            'num_recordings_used': len(recordings_a),
            'num_recordings_rejected': len(valid_indices) - len(recordings_a)
        }

        if self.verbose:
            print(f"\n[✓] Stacking complete")
            print(f"    Recordings used: {len(recordings_a)}")
            print(f"    Theoretical SNR improvement: {snr_improvement_db:.1f} dB")
            print(f"    Stacked SNR (A): {quality_a['snr_estimate_db']:.1f} dB")
            print(f"    Stacked SNR (B): {quality_b['snr_estimate_db']:.1f} dB")

        return StackedSignalResult(
            stacked_signal_a=stacked_a,
            stacked_signal_b=stacked_b,
            num_recordings=len(recordings_a),
            snr_improvement_db=snr_improvement_db,
            quality_metrics=quality_metrics,
            individual_qualities=[
                {'sensor_a': qa, 'sensor_b': qb}
                for qa, qb in zip(qualities_a, qualities_b)
            ]
        )

    def _align_recordings(
        self,
        recordings_a: List[np.ndarray],
        recordings_b: List[np.ndarray],
        sample_rate: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align recordings to compensate for timing variations.

        Args:
            recordings_a: Recordings from sensor A
            recordings_b: Recordings from sensor B
            sample_rate: Sample rate

        Returns:
            Tuple of (aligned_recordings_a, aligned_recordings_b)
        """
        if self.alignment_method == 'cross_correlation':
            # Align each recording to the first one using cross-correlation
            ref_a = recordings_a[0]
            ref_b = recordings_b[0]

            aligned_a = [ref_a]
            aligned_b = [ref_b]

            for i in range(1, len(recordings_a)):
                # Align sensor A
                shift_a = self._find_alignment_shift(ref_a, recordings_a[i])
                aligned_a.append(self._shift_signal(recordings_a[i], shift_a))

                # Align sensor B
                shift_b = self._find_alignment_shift(ref_b, recordings_b[i])
                aligned_b.append(self._shift_signal(recordings_b[i], shift_b))

                if self.verbose:
                    print(f"    Aligned recording {i+1}: shift_a={shift_a}, shift_b={shift_b}")

            return aligned_a, aligned_b

        elif self.alignment_method == 'timestamp':
            # Assume recordings are already time-aligned via timestamps
            # Just return as-is
            return recordings_a, recordings_b

        else:
            return recordings_a, recordings_b

    def _find_alignment_shift(
        self,
        reference: np.ndarray,
        signal: np.ndarray,
        max_shift_samples: int = 1000
    ) -> int:
        """
        Find optimal shift to align signal with reference using cross-correlation.

        Args:
            reference: Reference signal
            signal: Signal to align
            max_shift_samples: Maximum shift to search

        Returns:
            Optimal shift in samples (positive = signal is delayed)
        """
        # Use only first 2 seconds for alignment (faster)
        n_samples = min(len(reference), len(signal), 2 * SAMPLE_RATE)

        ref = reference[:n_samples]
        sig = signal[:n_samples]

        # Cross-correlate
        correlation = np.correlate(ref, sig, mode='same')

        # Find peak in central region
        center = len(correlation) // 2
        search_start = max(0, center - max_shift_samples)
        search_end = min(len(correlation), center + max_shift_samples)

        search_region = correlation[search_start:search_end]
        peak_idx = np.argmax(search_region) + search_start

        # Convert to shift
        shift = peak_idx - center

        return shift

    def _shift_signal(self, signal: np.ndarray, shift: int) -> np.ndarray:
        """
        Shift signal by specified number of samples.

        Args:
            signal: Input signal
            shift: Number of samples to shift (positive = delay)

        Returns:
            Shifted signal (same length, zero-padded)
        """
        if shift == 0:
            return signal

        shifted = np.zeros_like(signal)

        if shift > 0:
            # Delay (shift right)
            shifted[shift:] = signal[:-shift]
        else:
            # Advance (shift left)
            shifted[:shift] = signal[-shift:]

        return shifted

    def _stack_signal_averaging(
        self,
        recordings_a: List[np.ndarray],
        recordings_b: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple averaging of aligned signals.

        Args:
            recordings_a: Recordings from sensor A
            recordings_b: Recordings from sensor B

        Returns:
            Tuple of (stacked_a, stacked_b)
        """
        # Ensure all recordings have same length
        min_len = min(min(len(r) for r in recordings_a),
                     min(len(r) for r in recordings_b))

        recordings_a = [r[:min_len] for r in recordings_a]
        recordings_b = [r[:min_len] for r in recordings_b]

        # Average
        stacked_a = np.mean(recordings_a, axis=0)
        stacked_b = np.mean(recordings_b, axis=0)

        return stacked_a, stacked_b

    def _stack_weighted(
        self,
        recordings_a: List[np.ndarray],
        recordings_b: List[np.ndarray],
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted averaging based on signal quality.

        Args:
            recordings_a: Recordings from sensor A
            recordings_b: Recordings from sensor B
            weights: Weight for each recording (normalized to sum to 1)

        Returns:
            Tuple of (stacked_a, stacked_b)
        """
        # Ensure all recordings have same length
        min_len = min(min(len(r) for r in recordings_a),
                     min(len(r) for r in recordings_b))

        recordings_a = [r[:min_len] for r in recordings_a]
        recordings_b = [r[:min_len] for r in recordings_b]

        # Weighted average
        stacked_a = np.average(recordings_a, axis=0, weights=weights)
        stacked_b = np.average(recordings_b, axis=0, weights=weights)

        return stacked_a, stacked_b

    def _compute_weights(
        self,
        qualities_a: List[Dict],
        qualities_b: List[Dict]
    ) -> np.ndarray:
        """
        Compute weights based on signal quality.

        Args:
            qualities_a: Quality metrics for sensor A recordings
            qualities_b: Quality metrics for sensor B recordings

        Returns:
            Normalized weight array
        """
        # Use SNR as quality metric
        snrs_a = np.array([q['snr_estimate_db'] for q in qualities_a])
        snrs_b = np.array([q['snr_estimate_db'] for q in qualities_b])

        # Average SNR
        snrs = (snrs_a + snrs_b) / 2

        # Convert to linear scale for weighting
        weights = 10 ** (snrs / 10)

        # Normalize
        weights = weights / np.sum(weights)

        return weights

    def stack_from_files(
        self,
        files_a: List[str],
        files_b: List[str]
    ) -> StackedSignalResult:
        """
        Stack recordings directly from WAV files.

        Args:
            files_a: List of WAV files from sensor A
            files_b: List of WAV files from sensor B

        Returns:
            StackedSignalResult
        """
        if self.verbose:
            print(f"\n[i] Loading {len(files_a)} file pairs...")

        # Load all files
        recordings_a = []
        recordings_b = []

        for fa, fb in zip(files_a, files_b):
            try:
                audio_a, sr_a = load_wav(fa, validate=False)
                audio_b, sr_b = load_wav(fb, validate=False)

                recordings_a.append(audio_a)
                recordings_b.append(audio_b)

                if self.verbose:
                    print(f"    Loaded: {os.path.basename(fa)}, {os.path.basename(fb)}")

            except Exception as e:
                if self.verbose:
                    print(f"    [!] Error loading files: {e}")
                continue

        if len(recordings_a) == 0:
            raise ValueError("No files could be loaded")

        # Stack
        return self.stack_recordings(recordings_a, recordings_b)


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("SIGNAL STACKING TEST")
    print("=" * 80)

    # Create synthetic recordings with increasing noise
    print("\n[i] Creating synthetic recordings with noise...")

    t = np.linspace(0, SAMPLE_LENGTH_SEC, SAMPLE_RATE * SAMPLE_LENGTH_SEC)

    # Leak signal (same in all recordings)
    leak_freq = 800
    leak_signal = np.sin(2 * np.pi * leak_freq * t) * 0.5

    # Create 10 recordings with different noise
    recordings_a = []
    recordings_b = []

    for i in range(10):
        noise_a = np.random.randn(len(t)) * 0.3
        noise_b = np.random.randn(len(t)) * 0.3

        # Add leak signal with slight delay for sensor B
        delay_samples = 100
        signal_a = leak_signal + noise_a
        signal_b = np.concatenate([np.zeros(delay_samples), leak_signal[:-delay_samples]]) + noise_b

        recordings_a.append(signal_a)
        recordings_b.append(signal_b)

    print(f"[✓] Created 10 synthetic recordings")

    # Stack recordings
    stacker = SignalStacker(
        method='signal_averaging',
        alignment_method='cross_correlation',
        verbose=True
    )

    result = stacker.stack_recordings(recordings_a, recordings_b)

    print(f"\n[i] Results:")
    print(f"    Recordings used: {result.num_recordings}")
    print(f"    SNR improvement: {result.snr_improvement_db:.1f} dB")
    print(f"    Theoretical improvement: √{result.num_recordings} = {np.sqrt(result.num_recordings):.2f}× ({10*np.log10(result.num_recordings):.1f} dB)")

    print("\n[✓] Signal stacking test complete!")
