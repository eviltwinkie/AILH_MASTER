#!/usr/bin/env python3
"""
Robust Stacking Methods
Weighted trimmed mean, median, Huber M-estimator
"""

import numpy as np
from scipy.stats import trim_mean
from typing import Optional, List

from correlator_v3_config import *


def weighted_trimmed_mean(
    correlations: np.ndarray,
    weights: Optional[np.ndarray] = None,
    trim_percentile: float = None
) -> np.ndarray:
    """
    Trimmed mean with optional weighting.

    Discard top/bottom trim_percentile, then compute weighted mean.

    Args:
        correlations: Array of correlation functions (n_correlations, n_samples)
        weights: Optional weights for each correlation
        trim_percentile: Fraction to trim from each end (0-0.5)
    """
    if trim_percentile is None:
        trim_percentile = TRIMMED_MEAN_PERCENTILE

    # Validate inputs
    if not (0 <= trim_percentile < 0.5):
        raise ValueError(f"Trim percentile must be in [0, 0.5), got {trim_percentile}")

    if weights is None:
        weights = np.ones(correlations.shape[0])

    # Normalize weights
    weights = weights / np.sum(weights)

    # For each sample, sort correlations and trim
    n_corr, n_samples = correlations.shape
    result = np.zeros(n_samples)

    n_trim = int(n_corr * trim_percentile)

    # Ensure at least one value remains after trimming
    if 2 * n_trim >= n_corr:
        raise ValueError(
            f"Trimming {n_trim} from each end removes all {n_corr} correlations. "
            f"Reduce trim_percentile or increase number of correlations."
        )

    for i in range(n_samples):
        values = correlations[:, i]

        # Sort by value
        sorted_indices = np.argsort(values)

        # Trim extremes
        if n_trim > 0:
            trimmed_indices = sorted_indices[n_trim:-n_trim]
        else:
            trimmed_indices = sorted_indices

        # Weighted mean of trimmed values
        trimmed_values = values[trimmed_indices]
        trimmed_weights = weights[trimmed_indices]
        trimmed_weights /= np.sum(trimmed_weights)

        result[i] = np.sum(trimmed_values * trimmed_weights)

    return result


def median_stack(correlations: np.ndarray) -> np.ndarray:
    """
    Median stacking (robust to outliers).

    Args:
        correlations: Array of correlation functions (n_correlations, n_samples)
    """
    return np.median(correlations, axis=0)


def huber_stack(
    correlations: np.ndarray,
    delta: float = None,
    max_iterations: int = 10
) -> np.ndarray:
    """
    Huber M-estimator stacking.

    Adaptive: uses L2 for small residuals, L1 for large.
    Robust to outliers while efficient for inliers.

    Args:
        correlations: Array of correlation functions (n_correlations, n_samples)
        delta: Huber threshold parameter
        max_iterations: Maximum iterations for convergence
    """
    if delta is None:
        delta = HUBER_DELTA

    n_corr, n_samples = correlations.shape

    # Initialize with mean
    estimate = np.mean(correlations, axis=0)

    # Iterative reweighting
    for iteration in range(max_iterations):
        # Compute residuals
        residuals = correlations - estimate

        # Huber weights
        weights = np.zeros_like(residuals)
        for i in range(n_corr):
            for j in range(n_samples):
                r = abs(residuals[i, j])
                if r <= delta:
                    weights[i, j] = 1.0
                else:
                    weights[i, j] = delta / r

        # Weighted mean
        new_estimate = np.sum(correlations * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)

        # Check convergence
        change = np.mean(np.abs(new_estimate - estimate))
        estimate = new_estimate

        if change < 1e-6:
            break

    return estimate


def compute_peak_stability(
    correlation: np.ndarray,
    peak_idx: int,
    window_size: int = 20
) -> float:
    """
    Compute peak stability metric.

    Measures:
    - Peak-to-sidelobe ratio (PSR)
    - Peak curvature (sharpness)

    Returns stability score (0-1, higher = more stable).
    """
    n_samples = len(correlation)

    # Peak value
    peak_val = abs(correlation[peak_idx])

    # Sidelobe values (excluding peak region)
    sidelobe_start = max(0, peak_idx - window_size * 2)
    sidelobe_end = peak_idx - window_size
    if sidelobe_end > sidelobe_start:
        left_sidelobe = np.max(np.abs(correlation[sidelobe_start:sidelobe_end]))
    else:
        left_sidelobe = 0

    sidelobe_start = peak_idx + window_size
    sidelobe_end = min(n_samples, peak_idx + window_size * 2)
    if sidelobe_end > sidelobe_start:
        right_sidelobe = np.max(np.abs(correlation[sidelobe_start:sidelobe_end]))
    else:
        right_sidelobe = 0

    sidelobe_val = max(left_sidelobe, right_sidelobe)

    # Peak-to-sidelobe ratio
    psr = peak_val / (sidelobe_val + 1e-12)
    psr_norm = np.tanh(psr / 3.0)  # Normalize to ~[0, 1]

    # Peak curvature (second derivative)
    if peak_idx > 0 and peak_idx < n_samples - 1:
        curvature = abs(
            correlation[peak_idx] - 0.5 * (correlation[peak_idx - 1] + correlation[peak_idx + 1])
        )
        curvature_norm = np.tanh(curvature * 10)
    else:
        curvature_norm = 0.5

    # Combined stability score
    stability = 0.6 * psr_norm + 0.4 * curvature_norm

    return float(stability)


def multi_band_peak_alignment(
    correlations_per_band: List[np.ndarray],
    tolerance_samples: int = 5
) -> float:
    """
    Check if peaks align across frequency bands.

    Good alignment indicates real leak (not artifact).
    Poor alignment suggests noise or reflection.

    Returns alignment score (0-1).
    """
    # Find peak in each band
    peaks = []
    for corr in correlations_per_band:
        peak_idx = np.argmax(np.abs(corr))
        peaks.append(peak_idx)

    peaks = np.array(peaks)

    # Compute variance of peak locations
    peak_variance = np.var(peaks)

    # Score: high variance = poor alignment
    alignment_score = np.exp(-peak_variance / (tolerance_samples ** 2))

    return float(alignment_score)


def adaptive_stacking(
    correlations: np.ndarray,
    method: str = None,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adaptive stacking dispatcher.

    Selects stacking method based on configuration.
    """
    if method is None:
        method = ROBUST_STACKING_METHOD

    # Validate inputs
    if correlations.size == 0:
        raise ValueError("Correlations array cannot be empty")
    if correlations.ndim != 2:
        raise ValueError(f"Correlations must be 2D array, got {correlations.ndim}D")
    if correlations.shape[0] == 0 or correlations.shape[1] == 0:
        raise ValueError(f"Invalid correlations shape: {correlations.shape}")

    if method == 'mean':
        if weights is not None:
            weights_norm = weights / np.sum(weights)
            return np.sum(correlations * weights_norm[:, np.newaxis], axis=0)
        else:
            return np.mean(correlations, axis=0)

    elif method == 'trimmed_mean':
        return weighted_trimmed_mean(correlations, weights)

    elif method == 'median':
        return median_stack(correlations)

    elif method == 'huber':
        return huber_stack(correlations)

    else:
        raise ValueError(f"Unknown stacking method: {method}")


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("ROBUST STACKING TEST")
    print("=" * 80)

    # Generate test correlations
    n_correlations = 10
    n_samples = 1000
    true_peak_idx = 500
    true_delay = 50

    correlations = []
    for i in range(n_correlations):
        # Most correlations have peak at true location
        corr = np.random.randn(n_samples) * 0.1

        if i < 8:  # 80% inliers
            peak_loc = true_peak_idx + true_delay + np.random.randint(-3, 4)
        else:  # 20% outliers
            peak_loc = true_peak_idx + np.random.randint(-200, 200)

        # Add Gaussian peak
        for j in range(n_samples):
            dist = abs(j - peak_loc)
            corr[j] += 1.0 * np.exp(-(dist / 10) ** 2)

        correlations.append(corr)

    correlations = np.array(correlations)

    # Test each stacking method
    methods = ['mean', 'trimmed_mean', 'median', 'huber']

    for method in methods:
        print(f"\n[TEST] {method.upper()} stacking")

        stacked = adaptive_stacking(correlations, method=method)

        # Find peak
        peak_idx = np.argmax(stacked)
        estimated_delay = peak_idx - true_peak_idx

        error = abs(estimated_delay - true_delay)
        print(f"  True delay: {true_delay}, Estimated: {estimated_delay}, Error: {error}")
        print(f"  Test: {'PASS' if error < 10 else 'FAIL'}")

    # Test peak stability
    print("\n[TEST] Peak stability")
    stacked = adaptive_stacking(correlations, method='huber')
    peak_idx = np.argmax(stacked)
    stability = compute_peak_stability(stacked, peak_idx)
    print(f"  Peak stability: {stability:.3f}")
    print(f"  Test: {'PASS' if stability > 0.5 else 'FAIL'}")

    # Test multi-band alignment
    print("\n[TEST] Multi-band alignment")
    # Create aligned bands
    bands_aligned = [correlations[i] for i in range(5)]
    alignment_good = multi_band_peak_alignment(bands_aligned, tolerance_samples=5)

    # Create misaligned bands
    bands_misaligned = []
    for i in range(5):
        corr = np.random.randn(n_samples) * 0.1
        peak_loc = true_peak_idx + np.random.randint(-100, 100)
        for j in range(n_samples):
            dist = abs(j - peak_loc)
            corr[j] += 1.0 * np.exp(-(dist / 10) ** 2)
        bands_misaligned.append(corr)

    alignment_bad = multi_band_peak_alignment(bands_misaligned, tolerance_samples=5)

    print(f"  Aligned bands: {alignment_good:.3f}")
    print(f"  Misaligned bands: {alignment_bad:.3f}")
    print(f"  Test: {'PASS' if alignment_good > alignment_bad else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] Robust stacking test complete!")
