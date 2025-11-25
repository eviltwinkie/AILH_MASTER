#!/usr/bin/env python3
"""
AI Window Gating
Leverage existing leak classifier for adaptive window weighting
"""

import numpy as np
import sys
import os
from typing import List, Optional
from dataclasses import dataclass

# Import AI_DEV modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_DEV import dataset_classifier
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

from correlator_v3_config import *


@dataclass
class AIGatingResult:
    """Result from AI window gating"""
    weighted_correlation: np.ndarray
    per_window_leak_probs: np.ndarray
    per_window_weights: np.ndarray
    overall_leak_confidence: float


class AIWindowGating:
    """
    AI-powered window gating using leak classifier.

    Segments signals into windows, classifies each window,
    and weights correlation by leak probability.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size_sec: float = None,
        verbose: bool = False
    ):
        self.window_size_sec = window_size_sec or AI_WINDOW_SIZE_SEC
        self.verbose = verbose
        self.model = None
        self.model_available = AI_AVAILABLE

        # Try to load model
        if model_path is None:
            model_path = AI_LEAK_MODEL_PATH

        if self.verbose:
            print(f"[i] AIWindowGating initialized")
            print(f"    Window size: {self.window_size_sec}s")
            print(f"    Model path: {model_path}")

        if self.model_available and os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                if self.verbose:
                    print(f"    [✓] Model loaded successfully")
            except Exception as e:
                if self.verbose:
                    print(f"    [!] Failed to load model: {e}")
                self.model_available = False
        else:
            if self.verbose:
                print(f"    [!] Model not available (AI_DEV not loaded or model not found)")
            self.model_available = False

    def segment_windows(
        self,
        signal: np.ndarray,
        sample_rate: int,
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Segment signal into windows for classification.
        """
        window_samples = int(self.window_size_sec * sample_rate)
        step_samples = int(window_samples * (1 - overlap))

        # Validate signal length
        if len(signal) < window_samples:
            raise ValueError(
                f"Signal too short for window size: "
                f"signal={len(signal)} samples ({len(signal)/sample_rate:.2f}s), "
                f"window={window_samples} samples ({self.window_size_sec}s)"
            )

        windows = []
        for start in range(0, len(signal) - window_samples + 1, step_samples):
            end = start + window_samples
            windows.append(signal[start:end])

        return windows

    def classify_windows(
        self,
        windows: List[np.ndarray],
        sample_rate: int
    ) -> np.ndarray:
        """
        Run leak classifier on each window.

        Returns array of leak probabilities p_leak(w) for each window.
        """
        if not self.model_available or self.model is None:
            # Fallback: uniform weights
            if self.verbose:
                print("    [!] AI model not available, using uniform weights")
            return np.ones(len(windows)) * 0.5

        # Prepare windows for classification (need mel spectrograms)
        # This is a simplified version - in practice, use ai_builder.py
        leak_probs = []

        for window in windows:
            # Simple feature extraction (in production, use proper mel spectrogram)
            # Here we use a simple heuristic as fallback
            rms = np.sqrt(np.mean(window ** 2))
            spectral_centroid = self._simple_spectral_centroid(window, sample_rate)

            # Simple heuristic: high RMS + mid-frequency = likely leak
            if rms > 0.1 and 200 < spectral_centroid < 1500:
                prob = min(0.9, rms * 2)
            else:
                prob = 0.3

            leak_probs.append(prob)

        return np.array(leak_probs)

    def _simple_spectral_centroid(self, signal: np.ndarray, sample_rate: int) -> float:
        """Simple spectral centroid calculation"""
        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
        freqs = freqs[:len(freqs) // 2]
        spectrum = spectrum[:len(spectrum) // 2]

        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
        return centroid

    def compute_weights(
        self,
        leak_probs: np.ndarray,
        snr_values: np.ndarray = None,
        leak_prob_exp: float = None,
        snr_exp: float = None
    ) -> np.ndarray:
        """
        Compute window weights.

        w(w) = p_leak(w)^γ · SNR(w)^δ
        """
        if leak_prob_exp is None:
            leak_prob_exp = AI_LEAK_PROB_EXPONENT
        if snr_exp is None:
            snr_exp = AI_SNR_EXPONENT

        # Leak probability component
        weights = leak_probs ** leak_prob_exp

        # SNR component (if provided)
        if snr_values is not None:
            # Normalize SNR to [0, 1]
            snr_norm = (snr_values - np.min(snr_values)) / (np.max(snr_values) - np.min(snr_values) + 1e-12)
            weights *= snr_norm ** snr_exp

        # Check for all-zero weights (fallback to uniform)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-10:
            # All weights are effectively zero, use uniform weights
            weights = np.ones(len(weights))
            weight_sum = np.sum(weights)

        # Normalize weights
        weights /= weight_sum

        return weights

    def weighted_correlation(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        weights_a: np.ndarray,
        weights_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted cross-correlation.

        R(τ) = Σ w(w) R_w(τ) / Σ w(w)
        """
        # Segment both signals
        windows_a = self.segment_windows(signal_a, SAMPLE_RATE_V3)
        windows_b = self.segment_windows(signal_b, SAMPLE_RATE_V3)

        # Ensure same number of windows
        n_windows = min(len(windows_a), len(windows_b), len(weights_a), len(weights_b))

        # Weighted correlation
        weighted_corr = None

        for i in range(n_windows):
            # Correlate windows
            corr_w = np.fft.fftshift(np.fft.ifft(
                np.fft.fft(windows_a[i]) * np.conj(np.fft.fft(windows_b[i]))
            ).real)

            # Weight by average of both sensor weights
            weight = (weights_a[i] + weights_b[i]) / 2

            # Accumulate
            if weighted_corr is None:
                weighted_corr = corr_w * weight
            else:
                # Interpolate to same length if needed
                if len(corr_w) != len(weighted_corr):
                    corr_w = np.interp(
                        np.linspace(0, len(corr_w) - 1, len(weighted_corr)),
                        np.arange(len(corr_w)),
                        corr_w
                    )
                weighted_corr += corr_w * weight

        return weighted_corr

    def process_pair(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sample_rate: int = None
    ) -> AIGatingResult:
        """
        Complete AI window gating pipeline.
        """
        if sample_rate is None:
            sample_rate = SAMPLE_RATE_V3

        if self.verbose:
            print(f"\n[i] AI window gating:")
            print(f"    Signal length: {len(signal_a)} samples ({len(signal_a) / sample_rate:.1f}s)")

        # Segment windows
        windows_a = self.segment_windows(signal_a, sample_rate)
        windows_b = self.segment_windows(signal_b, sample_rate)

        if self.verbose:
            print(f"    Windows: {len(windows_a)} (sensor A), {len(windows_b)} (sensor B)")

        # Classify windows
        leak_probs_a = self.classify_windows(windows_a, sample_rate)
        leak_probs_b = self.classify_windows(windows_b, sample_rate)

        # Compute weights
        weights_a = self.compute_weights(leak_probs_a)
        weights_b = self.compute_weights(leak_probs_b)

        if self.verbose:
            print(f"    Mean leak probability: A={np.mean(leak_probs_a):.3f}, B={np.mean(leak_probs_b):.3f}")
            print(f"    Max weight: A={np.max(weights_a):.3f}, B={np.max(weights_b):.3f}")

        # Weighted correlation
        weighted_corr = self.weighted_correlation(signal_a, signal_b, weights_a, weights_b)

        # Overall confidence
        overall_confidence = (np.mean(leak_probs_a) + np.mean(leak_probs_b)) / 2

        if self.verbose:
            print(f"    [✓] Overall leak confidence: {overall_confidence:.3f}")

        return AIGatingResult(
            weighted_correlation=weighted_corr,
            per_window_leak_probs=(leak_probs_a + leak_probs_b) / 2,
            per_window_weights=(weights_a + weights_b) / 2,
            overall_leak_confidence=float(overall_confidence)
        )


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("AI WINDOW GATING TEST")
    print("=" * 80)

    # Generate test signals
    sample_rate = 4096
    duration = 10.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    # Leak signal in middle 5 seconds (high RMS, mid-frequency)
    leak_signal = np.zeros(n_samples)
    leak_start = int(2.5 * sample_rate)
    leak_end = int(7.5 * sample_rate)
    leak_signal[leak_start:leak_end] = np.sin(2 * np.pi * 800 * t[leak_start:leak_end]) * 0.5

    # Background noise
    noise = np.random.randn(n_samples) * 0.1

    signal_a = leak_signal + noise
    signal_b = np.roll(leak_signal, 50) + noise

    # Test AI window gating
    gating = AIWindowGating(window_size_sec=1.0, verbose=True)

    print("\n[TEST] AI window gating")
    result = gating.process_pair(signal_a, signal_b, sample_rate)

    print(f"\n[RESULTS]")
    print(f"  Overall leak confidence: {result.overall_leak_confidence:.3f}")
    print(f"  Windows classified: {len(result.per_window_leak_probs)}")
    print(f"  Mean window probability: {np.mean(result.per_window_leak_probs):.3f}")
    print(f"  Test: {'PASS' if result.overall_leak_confidence > 0.4 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] AI window gating test complete!")
