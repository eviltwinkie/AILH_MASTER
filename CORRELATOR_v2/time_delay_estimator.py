#!/usr/bin/env python3
"""
Leak Detection Correlator - Time Delay Estimation Module

This module estimates time delays from cross-correlation functions using
advanced peak detection and subsample interpolation techniques.

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from correlator_config import *


@dataclass
class TimeDelayEstimate:
    """Results of time delay estimation."""
    delay_samples: float      # Time delay in samples (can be fractional)
    delay_seconds: float      # Time delay in seconds
    confidence: float         # Confidence score (0-1)
    peak_height: float        # Correlation peak height
    peak_sharpness: float     # Ratio of main peak to next highest peak
    snr_db: float            # Signal-to-noise ratio in dB
    peak_index: int          # Integer index of peak in correlation array


class TimeDelayEstimator:
    """
    Advanced time delay estimation from cross-correlation.

    Features:
    - Multiple peak detection
    - Subsample interpolation (parabolic, gaussian, sinc)
    - Quality metrics (SNR, sharpness, confidence)
    - Outlier rejection
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        interpolation_method: str = INTERPOLATION_METHOD,
        verbose: bool = False
    ):
        """
        Initialize time delay estimator.

        Args:
            sample_rate (int): Sample rate in Hz
            interpolation_method (str): Interpolation method ('parabolic', 'gaussian', 'sinc')
            verbose (bool): Print verbose output
        """
        self.sample_rate = sample_rate
        self.interpolation_method = interpolation_method
        self.verbose = verbose

    def estimate(
        self,
        correlation: np.ndarray,
        lags: np.ndarray,
        max_delay_sec: Optional[float] = None
    ) -> TimeDelayEstimate:
        """
        Estimate time delay from correlation function.

        Args:
            correlation (np.ndarray): Cross-correlation values
            lags (np.ndarray): Corresponding lag values (samples)
            max_delay_sec (float, optional): Maximum expected delay (seconds)

        Returns:
            TimeDelayEstimate: Time delay estimate with quality metrics
        """
        if max_delay_sec is None:
            max_delay_sec = MAX_TIME_DELAY_SEC

        # Constrain search range
        max_lag_samples = seconds_to_samples(max_delay_sec, self.sample_rate)
        valid_idx = np.abs(lags) <= max_lag_samples

        if np.sum(valid_idx) == 0:
            raise ValueError("No valid lags within search range")

        corr_search = correlation[valid_idx]
        lags_search = lags[valid_idx]

        # Find peak
        peak_idx_local = np.argmax(corr_search)
        peak_idx_global = np.where(valid_idx)[0][peak_idx_local]

        peak_height = correlation[peak_idx_global]
        peak_lag_samples = lags[peak_idx_global]

        # Subsample interpolation for precision
        if self.interpolation_method == 'parabolic':
            refined_lag = self._parabolic_interpolation(
                correlation, peak_idx_global
            )
        elif self.interpolation_method == 'gaussian':
            refined_lag = self._gaussian_interpolation(
                correlation, peak_idx_global
            )
        elif self.interpolation_method == 'sinc':
            refined_lag = self._sinc_interpolation(
                correlation, lags, peak_idx_global
            )
        else:
            refined_lag = float(peak_lag_samples)  # No interpolation

        # Convert to seconds
        delay_sec = samples_to_seconds(refined_lag, self.sample_rate)

        # Compute quality metrics
        snr_db = self._compute_snr(correlation, peak_idx_global)
        sharpness = self._compute_sharpness(corr_search, peak_idx_local)
        confidence = self._compute_confidence(peak_height, snr_db, sharpness)

        if self.verbose:
            print(f"[i] Time delay estimate:")
            print(f"    Peak index: {peak_idx_global}")
            print(f"    Peak height: {peak_height:.4f}")
            print(f"    Raw delay: {peak_lag_samples} samples ({samples_to_seconds(peak_lag_samples, self.sample_rate):.6f}s)")
            print(f"    Refined delay: {refined_lag:.2f} samples ({delay_sec:.6f}s)")
            print(f"    SNR: {snr_db:.1f} dB")
            print(f"    Sharpness: {sharpness:.2f}")
            print(f"    Confidence: {confidence:.3f}")

        return TimeDelayEstimate(
            delay_samples=refined_lag,
            delay_seconds=delay_sec,
            confidence=confidence,
            peak_height=peak_height,
            peak_sharpness=sharpness,
            snr_db=snr_db,
            peak_index=peak_idx_global
        )

    def estimate_multi_peak(
        self,
        correlation: np.ndarray,
        lags: np.ndarray,
        n_peaks: int = MAX_PEAKS_TO_DETECT,
        max_delay_sec: Optional[float] = None
    ) -> List[TimeDelayEstimate]:
        """
        Detect multiple peaks in correlation (for handling reflections/echoes).

        Args:
            correlation (np.ndarray): Cross-correlation values
            lags (np.ndarray): Corresponding lag values (samples)
            n_peaks (int): Number of peaks to detect
            max_delay_sec (float, optional): Maximum expected delay

        Returns:
            List of TimeDelayEstimate objects, sorted by confidence
        """
        if max_delay_sec is None:
            max_delay_sec = MAX_TIME_DELAY_SEC

        # Constrain search range
        max_lag_samples = seconds_to_samples(max_delay_sec, self.sample_rate)
        valid_idx = np.abs(lags) <= max_lag_samples
        corr_search = correlation[valid_idx]
        lags_search = lags[valid_idx]

        # Find peaks using scipy
        from scipy.signal import find_peaks

        # Peak detection parameters
        height = MIN_PEAK_HEIGHT * np.max(np.abs(corr_search))
        distance = int(0.01 * self.sample_rate)  # Minimum 10ms separation

        peak_indices, properties = find_peaks(
            corr_search,
            height=height,
            distance=distance
        )

        # Sort by height (descending)
        sorted_idx = np.argsort(properties['peak_heights'])[::-1]
        peak_indices = peak_indices[sorted_idx[:n_peaks]]

        # Estimate delay for each peak
        estimates = []
        for local_idx in peak_indices:
            # Map back to global index
            global_idx = np.where(valid_idx)[0][local_idx]

            peak_height = correlation[global_idx]
            peak_lag_samples = lags[global_idx]

            # Subsample interpolation
            if self.interpolation_method == 'parabolic':
                refined_lag = self._parabolic_interpolation(correlation, global_idx)
            else:
                refined_lag = float(peak_lag_samples)

            delay_sec = samples_to_seconds(refined_lag, self.sample_rate)

            # Quality metrics
            snr_db = self._compute_snr(correlation, global_idx)
            sharpness = self._compute_sharpness(corr_search, local_idx)
            confidence = self._compute_confidence(peak_height, snr_db, sharpness)

            estimates.append(TimeDelayEstimate(
                delay_samples=refined_lag,
                delay_seconds=delay_sec,
                confidence=confidence,
                peak_height=peak_height,
                peak_sharpness=sharpness,
                snr_db=snr_db,
                peak_index=global_idx
            ))

        # Sort by confidence
        estimates.sort(key=lambda x: x.confidence, reverse=True)

        if self.verbose:
            print(f"[i] Detected {len(estimates)} peaks")
            for i, est in enumerate(estimates):
                print(f"    Peak {i+1}: τ={est.delay_seconds:.6f}s, conf={est.confidence:.3f}")

        return estimates

    def _parabolic_interpolation(
        self,
        correlation: np.ndarray,
        peak_idx: int
    ) -> float:
        """
        Parabolic interpolation for subsample peak location.

        Fits a parabola to the peak and its neighbors to estimate
        the true peak location with sub-sample precision.

        Args:
            correlation (np.ndarray): Correlation function
            peak_idx (int): Integer index of peak

        Returns:
            float: Refined peak location (fractional samples)
        """
        # Need at least 3 points for parabola fit
        if peak_idx == 0 or peak_idx == len(correlation) - 1:
            return float(peak_idx)

        # Get peak and neighbors
        y1 = correlation[peak_idx - 1]
        y2 = correlation[peak_idx]
        y3 = correlation[peak_idx + 1]

        # Parabolic interpolation formula
        # δ = (y3 - y1) / (2 * (2*y2 - y1 - y3))
        denominator = 2 * (2*y2 - y1 - y3)

        if abs(denominator) < 1e-10:
            # Denominator too small, return integer peak
            return float(peak_idx)

        delta = (y3 - y1) / denominator

        # Refined peak location
        refined_idx = peak_idx + delta

        return refined_idx

    def _gaussian_interpolation(
        self,
        correlation: np.ndarray,
        peak_idx: int
    ) -> float:
        """
        Gaussian interpolation for subsample peak location.

        Assumes correlation peak has Gaussian shape.

        Args:
            correlation (np.ndarray): Correlation function
            peak_idx (int): Integer index of peak

        Returns:
            float: Refined peak location (fractional samples)
        """
        if peak_idx == 0 or peak_idx == len(correlation) - 1:
            return float(peak_idx)

        # Get log-magnitudes (Gaussian becomes parabola in log space)
        y1 = max(abs(correlation[peak_idx - 1]), 1e-10)
        y2 = max(abs(correlation[peak_idx]), 1e-10)
        y3 = max(abs(correlation[peak_idx + 1]), 1e-10)

        log_y1 = np.log(y1)
        log_y2 = np.log(y2)
        log_y3 = np.log(y3)

        # Gaussian peak formula
        denominator = 2 * (2*log_y2 - log_y1 - log_y3)

        if abs(denominator) < 1e-10:
            return float(peak_idx)

        delta = (log_y3 - log_y1) / denominator
        refined_idx = peak_idx + delta

        return refined_idx

    def _sinc_interpolation(
        self,
        correlation: np.ndarray,
        lags: np.ndarray,
        peak_idx: int,
        window: int = 10
    ) -> float:
        """
        Sinc interpolation for subsample peak location.

        More accurate but computationally expensive.

        Args:
            correlation (np.ndarray): Correlation function
            lags (np.ndarray): Lag array
            peak_idx (int): Integer index of peak
            window (int): Window size around peak

        Returns:
            float: Refined peak location (fractional samples)
        """
        # Extract window around peak
        start = max(0, peak_idx - window)
        end = min(len(correlation), peak_idx + window + 1)

        corr_window = correlation[start:end]
        lags_window = lags[start:end]

        # Upsample by factor of 10 using sinc interpolation
        from scipy.interpolate import interp1d

        f_interp = interp1d(
            lags_window,
            corr_window,
            kind='cubic',
            fill_value='extrapolate'
        )

        # Fine grid
        lags_fine = np.linspace(lags_window[0], lags_window[-1], len(lags_window) * 10)
        corr_fine = f_interp(lags_fine)

        # Find peak in fine grid
        peak_fine_idx = np.argmax(corr_fine)
        refined_lag = lags_fine[peak_fine_idx]

        return refined_lag

    def _compute_snr(
        self,
        correlation: np.ndarray,
        peak_idx: int,
        noise_window: int = 100
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio of correlation peak.

        Args:
            correlation (np.ndarray): Correlation function
            peak_idx (int): Index of peak
            noise_window (int): Window size for noise estimation

        Returns:
            float: SNR in dB
        """
        # Peak power
        peak_power = correlation[peak_idx] ** 2

        # Noise power (from tail regions)
        n_corr = len(correlation)
        noise_window = min(noise_window, n_corr // 4)

        noise_left = correlation[:noise_window]
        noise_right = correlation[-noise_window:]
        noise = np.concatenate([noise_left, noise_right])

        noise_power = np.mean(noise ** 2) + 1e-10

        # SNR in dB
        snr_db = 10 * np.log10(peak_power / noise_power)

        return snr_db

    def _compute_sharpness(
        self,
        correlation: np.ndarray,
        peak_idx: int
    ) -> float:
        """
        Compute peak sharpness (ratio to second-highest peak).

        Args:
            correlation (np.ndarray): Correlation function
            peak_idx (int): Index of main peak

        Returns:
            float: Sharpness ratio (>1)
        """
        peak_height = abs(correlation[peak_idx])

        # Exclude region around main peak
        exclude_window = int(0.01 * self.sample_rate)  # 10ms
        mask = np.ones(len(correlation), dtype=bool)
        start = max(0, peak_idx - exclude_window)
        end = min(len(correlation), peak_idx + exclude_window + 1)
        mask[start:end] = False

        # Find second-highest peak
        if np.sum(mask) > 0:
            second_peak_height = np.max(np.abs(correlation[mask]))
        else:
            second_peak_height = 0.0

        if second_peak_height > 0:
            sharpness = peak_height / second_peak_height
        else:
            sharpness = float('inf')

        return sharpness

    def _compute_confidence(
        self,
        peak_height: float,
        snr_db: float,
        sharpness: float
    ) -> float:
        """
        Compute overall confidence score for time delay estimate.

        Confidence is based on:
        - Peak height (higher is better)
        - SNR (higher is better)
        - Sharpness (higher is better)

        Args:
            peak_height (float): Normalized peak height (0-1)
            snr_db (float): SNR in dB
            sharpness (float): Peak sharpness ratio

        Returns:
            float: Confidence score (0-1)
        """
        # Peak height contribution (0-1)
        c_height = np.clip(abs(peak_height), 0, 1)

        # SNR contribution (0-1)
        # Map SNR from [0, 30] dB to [0, 1]
        c_snr = np.clip(snr_db / 30.0, 0, 1)

        # Sharpness contribution (0-1)
        # Map sharpness from [1, 10] to [0, 1]
        c_sharpness = np.clip((sharpness - 1) / 9.0, 0, 1)

        # Weighted average (tunable weights)
        w_height = 0.3
        w_snr = 0.4
        w_sharpness = 0.3

        confidence = (
            w_height * c_height +
            w_snr * c_snr +
            w_sharpness * c_sharpness
        )

        return confidence

    def validate_estimate(
        self,
        estimate: TimeDelayEstimate,
        max_delay_sec: Optional[float] = None,
        min_confidence: Optional[float] = None,
        min_snr_db: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate time delay estimate against quality thresholds.

        Args:
            estimate (TimeDelayEstimate): Estimate to validate
            max_delay_sec (float, optional): Maximum allowed delay
            min_confidence (float, optional): Minimum confidence threshold
            min_snr_db (float, optional): Minimum SNR threshold

        Returns:
            Tuple of (is_valid, message)
        """
        if max_delay_sec is None:
            max_delay_sec = MAX_TIME_DELAY_SEC
        if min_confidence is None:
            min_confidence = MIN_CONFIDENCE
        if min_snr_db is None:
            min_snr_db = MIN_SNR_DB

        # Check delay magnitude
        if abs(estimate.delay_seconds) > max_delay_sec:
            return False, f"Delay ({estimate.delay_seconds:.4f}s) exceeds maximum ({max_delay_sec}s)"

        # Check confidence
        if estimate.confidence < min_confidence:
            return False, f"Confidence ({estimate.confidence:.3f}) below threshold ({min_confidence})"

        # Check SNR
        if estimate.snr_db < min_snr_db:
            return False, f"SNR ({estimate.snr_db:.1f} dB) below threshold ({min_snr_db} dB)"

        # Check sharpness
        if estimate.peak_sharpness < MIN_PEAK_SHARPNESS:
            return False, f"Peak sharpness ({estimate.peak_sharpness:.2f}) below threshold ({MIN_PEAK_SHARPNESS})"

        return True, "Estimate valid"


# ==============================================================================
# MAIN - Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Time Delay Estimator Testing')
    parser.add_argument('--method', default='parabolic',
                       choices=['parabolic', 'gaussian', 'sinc'],
                       help='Interpolation method')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test with synthetic correlation')
    parser.add_argument('--multi-peak', action='store_true',
                       help='Test multi-peak detection')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("TIME DELAY ESTIMATOR TEST")
    print("=" * 80)

    estimator = TimeDelayEstimator(
        interpolation_method=args.method,
        verbose=args.verbose
    )

    print(f"\n[i] Interpolation method: {args.method}")
    print(f"[i] Sample rate: {SAMPLE_RATE} Hz")

    if args.test_synthetic:
        print("\n[i] Testing with synthetic correlation function...")

        # Create synthetic correlation with known peak
        n_samples = SAMPLE_RATE * 2  # 2 seconds
        lags = np.arange(-n_samples//2, n_samples//2)

        # True delay: 0.0234 seconds = 95.7 samples at 4096 Hz
        true_delay_sec = 0.0234
        true_delay_samples = true_delay_sec * SAMPLE_RATE

        # Generate Gaussian peak
        sigma = 50  # samples
        correlation = np.exp(-(lags - true_delay_samples)**2 / (2 * sigma**2))

        # Add noise
        correlation += np.random.randn(len(correlation)) * 0.05

        print(f"[i] True delay: {true_delay_sec}s ({true_delay_samples:.1f} samples)")

        # Estimate delay
        if args.multi_peak:
            estimates = estimator.estimate_multi_peak(correlation, lags, n_peaks=3)
            best_estimate = estimates[0]
        else:
            best_estimate = estimator.estimate(correlation, lags)

        print(f"\n[i] Estimated delay: {best_estimate.delay_seconds:.6f}s ({best_estimate.delay_samples:.2f} samples)")
        print(f"[i] Error: {abs(best_estimate.delay_seconds - true_delay_sec) * 1000:.3f}ms")
        print(f"[i] Confidence: {best_estimate.confidence:.3f}")
        print(f"[i] SNR: {best_estimate.snr_db:.1f} dB")
        print(f"[i] Sharpness: {best_estimate.peak_sharpness:.2f}")

        # Validate
        is_valid, msg = estimator.validate_estimate(best_estimate)
        print(f"\n[i] Validation: {'PASS' if is_valid else 'FAIL'}")
        print(f"    {msg}")

    print("\n[✓] Test complete")
