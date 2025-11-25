#!/usr/bin/env python3
"""
Bayesian Position Estimation
Posterior inference with MAP estimates and credible intervals
"""

import numpy as np
from scipy.stats import norm, entropy
from typing import Optional
from dataclasses import dataclass

from correlator_v3_config import *


@dataclass
class BayesianEstimate:
    """Result from Bayesian estimation"""
    map_position_m: float
    credible_interval_m: tuple
    entropy: float
    posterior_distribution: np.ndarray
    positions: np.ndarray


class BayesianEstimator:
    """
    Bayesian position estimation with posterior inference.

    Given correlation data D and prior knowledge, compute posterior p(x|D).
    """

    def __init__(self, prior_type='uniform', verbose=False):
        self.prior_type = prior_type
        self.verbose = verbose

        if self.verbose:
            print(f"[i] BayesianEstimator initialized (prior={prior_type})")

    def define_prior(
        self,
        sensor_separation_m: float,
        prior_type: str = None,
        mean_m: Optional[float] = None,
        std_m: float = None
    ) -> tuple:
        """
        Define prior p(x) over position space.

        Types:
        - 'uniform': Uniform over [0, L]
        - 'gaussian': Gaussian centered at mean_m with std std_m
        - 'informed': Use domain knowledge (e.g., near valve at 30m)
        """
        if prior_type is None:
            prior_type = self.prior_type
        if std_m is None:
            std_m = BAYESIAN_PRIOR_STD_M

        # Position grid
        positions = np.arange(0, sensor_separation_m + BAYESIAN_GRID_RESOLUTION_M,
                              BAYESIAN_GRID_RESOLUTION_M)

        if prior_type == 'uniform':
            # Uniform prior
            prior = np.ones(len(positions))
            prior /= np.sum(prior)

        elif prior_type == 'gaussian':
            # Gaussian prior
            if mean_m is None:
                mean_m = sensor_separation_m / 2  # Center of pipe

            prior = norm.pdf(positions, loc=mean_m, scale=std_m)
            prior /= np.sum(prior)

        elif prior_type == 'informed':
            # Custom informed prior (example: leak likely near valve at 30m)
            valve_position_m = 30.0
            prior = norm.pdf(positions, loc=valve_position_m, scale=std_m)
            prior /= np.sum(prior)

        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

        if self.verbose:
            print(f"    Prior: {prior_type}, Mean: {mean_m if mean_m else 'N/A'}m, Std: {std_m}m")

        return positions, prior

    def compute_likelihood(
        self,
        correlation: np.ndarray,
        sensor_separation_m: float,
        wave_speed_mps: float,
        sample_rate: int = 4096,
        beta: float = None
    ) -> tuple:
        """
        Compute likelihood p(D|x) from correlation function.

        p(D|x) ∝ exp(β · s(x))

        where s(x) is correlation score at delay corresponding to position x.
        """
        if beta is None:
            beta = BAYESIAN_LIKELIHOOD_BETA

        # Position grid
        positions = np.arange(0, sensor_separation_m + BAYESIAN_GRID_RESOLUTION_M,
                              BAYESIAN_GRID_RESOLUTION_M)

        # For each position, compute expected delay and get correlation value
        n_samples = len(correlation)
        center_idx = n_samples // 2

        likelihood = np.zeros(len(positions))

        for i, x in enumerate(positions):
            # Expected delay for position x
            # τ(x) = (L - 2x) / c
            expected_delay_sec = (sensor_separation_m - 2 * x) / wave_speed_mps
            expected_delay_samples = int(expected_delay_sec * sample_rate)

            # Correlation value at this delay
            delay_idx = center_idx + expected_delay_samples

            if 0 <= delay_idx < n_samples:
                corr_value = correlation[delay_idx]
                # Normalize correlation value to [0, 1]
                corr_value_norm = (corr_value - np.min(correlation)) / (np.max(correlation) - np.min(correlation) + 1e-12)
                likelihood[i] = np.exp(beta * corr_value_norm)
            else:
                likelihood[i] = 1e-10

        # Normalize
        likelihood /= (np.sum(likelihood) + 1e-12)

        if self.verbose:
            print(f"    Likelihood computed (β={beta})")

        return positions, likelihood

    def compute_posterior(
        self,
        prior: np.ndarray,
        likelihood: np.ndarray
    ) -> np.ndarray:
        """
        Compute posterior p(x|D) ∝ p(D|x)p(x).

        Bayes' rule: posterior = likelihood × prior / evidence
        """
        posterior = likelihood * prior

        # Normalize
        evidence = np.sum(posterior)
        if evidence > 0:
            posterior /= evidence

        return posterior

    def extract_estimates(
        self,
        posterior: np.ndarray,
        positions: np.ndarray,
        credible_level: float = None
    ) -> BayesianEstimate:
        """
        Extract MAP estimate and credible interval from posterior.

        MAP (Maximum A Posteriori): position with highest posterior probability
        Credible interval: region containing credible_level of probability mass
        """
        if credible_level is None:
            credible_level = BAYESIAN_CREDIBLE_INTERVAL

        # MAP estimate
        map_idx = np.argmax(posterior)
        map_position_m = positions[map_idx]

        # Credible interval (highest posterior density)
        # Sort by posterior value
        sorted_indices = np.argsort(posterior)[::-1]

        cumsum = 0
        credible_indices = []

        for idx in sorted_indices:
            credible_indices.append(idx)
            cumsum += posterior[idx]
            if cumsum >= credible_level:
                break

        # Get min/max positions in credible set
        credible_positions = positions[credible_indices]
        credible_interval_m = (np.min(credible_positions), np.max(credible_positions))

        # Entropy (uncertainty measure)
        # High entropy = high uncertainty
        post_entropy = entropy(posterior + 1e-12)

        if self.verbose:
            print(f"\n[i] Bayesian estimates:")
            print(f"    MAP position: {map_position_m:.2f}m")
            print(f"    {credible_level * 100:.0f}% credible interval: [{credible_interval_m[0]:.2f}, {credible_interval_m[1]:.2f}]m")
            print(f"    Entropy (uncertainty): {post_entropy:.3f}")

        return BayesianEstimate(
            map_position_m=float(map_position_m),
            credible_interval_m=credible_interval_m,
            entropy=float(post_entropy),
            posterior_distribution=posterior,
            positions=positions
        )

    def estimate(
        self,
        correlation: np.ndarray,
        sensor_separation_m: float,
        wave_speed_mps: float,
        sample_rate: int = 4096
    ) -> BayesianEstimate:
        """
        Complete Bayesian estimation pipeline.

        Steps:
        1. Define prior p(x)
        2. Compute likelihood p(D|x) from correlation
        3. Compute posterior p(x|D)
        4. Extract MAP and credible interval
        """
        if self.verbose:
            print(f"\n[i] Bayesian estimation:")
            print(f"    Sensor separation: {sensor_separation_m:.1f}m")
            print(f"    Wave speed: {wave_speed_mps:.1f} m/s")

        # Step 1: Prior
        positions, prior = self.define_prior(sensor_separation_m)

        # Step 2: Likelihood
        _, likelihood = self.compute_likelihood(
            correlation, sensor_separation_m, wave_speed_mps, sample_rate
        )

        # Step 3: Posterior
        posterior = self.compute_posterior(prior, likelihood)

        # Step 4: Extract estimates
        estimate = self.extract_estimates(posterior, positions)

        return estimate


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BAYESIAN ESTIMATOR TEST")
    print("=" * 80)

    # Generate synthetic correlation with leak at known position
    sample_rate = 4096
    duration = 10.0
    n_samples = int(sample_rate * duration)

    sensor_separation_m = 100.0
    true_leak_position_m = 35.0
    wave_speed_mps = 1400.0

    # Expected delay
    true_delay_sec = (sensor_separation_m - 2 * true_leak_position_m) / wave_speed_mps
    true_delay_samples = int(true_delay_sec * sample_rate)

    # Create synthetic correlation with peak at true delay
    correlation = np.random.randn(n_samples) * 0.1
    center_idx = n_samples // 2
    peak_idx = center_idx + true_delay_samples

    # Add Gaussian peak at true position
    peak_width = 50
    for i in range(max(0, peak_idx - peak_width), min(n_samples, peak_idx + peak_width)):
        distance = abs(i - peak_idx)
        correlation[i] += 1.0 * np.exp(-(distance / 20) ** 2)

    # Test Bayesian estimation
    estimator = BayesianEstimator(prior_type='uniform', verbose=True)

    print("\n[TEST 1] Uniform prior")
    estimate = estimator.estimate(correlation, sensor_separation_m, wave_speed_mps, sample_rate)

    error = abs(estimate.map_position_m - true_leak_position_m)
    in_credible = (estimate.credible_interval_m[0] <= true_leak_position_m <= estimate.credible_interval_m[1])

    print(f"\n[RESULTS]")
    print(f"  True position: {true_leak_position_m:.2f}m")
    print(f"  MAP estimate: {estimate.map_position_m:.2f}m")
    print(f"  Error: {error:.2f}m")
    print(f"  True in credible interval: {in_credible}")
    print(f"  Test: {'PASS' if error < 5.0 and in_credible else 'FAIL'}")

    # Test with Gaussian prior
    print("\n[TEST 2] Gaussian prior (centered)")
    estimator_gauss = BayesianEstimator(prior_type='gaussian', verbose=True)
    estimate_gauss = estimator_gauss.estimate(correlation, sensor_separation_m, wave_speed_mps, sample_rate)

    error_gauss = abs(estimate_gauss.map_position_m - true_leak_position_m)
    print(f"  MAP estimate: {estimate_gauss.map_position_m:.2f}m")
    print(f"  Error: {error_gauss:.2f}m")
    print(f"  Test: {'PASS' if error_gauss < 5.0 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("[✓] Bayesian estimator test complete!")
