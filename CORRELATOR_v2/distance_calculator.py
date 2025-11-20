#!/usr/bin/env python3
"""
Leak Detection Correlator - Distance Calculator and Leak Localizer Module

This module calculates leak distance and position from time delay estimates
and sensor pair configurations.

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, asdict
import json

from correlator_config import *
from time_delay_estimator import TimeDelayEstimate
from sensor_registry import SensorRegistry, SensorPair


@dataclass
class LeakLocation:
    """Results of leak localization."""
    sensor_pair: Tuple[str, str]            # (sensor_a_id, sensor_b_id)
    time_delay_seconds: float               # Measured time delay
    distance_from_sensor_a_meters: float    # Distance from sensor A to leak
    distance_from_sensor_b_meters: float    # Distance from sensor B to leak
    sensor_separation_meters: float         # Total distance between sensors
    leak_position_gps: Optional[Tuple[float, float, float]]  # (lat, lon, elev)
    confidence: float                       # Overall confidence (0-1)
    wave_speed_mps: float                   # Wave speed used
    pipe_material: str                      # Pipe material
    quality_metrics: Dict                   # Additional metrics


class DistanceCalculator:
    """
    Calculate leak distance from time delay estimates.

    Uses the acoustic leak localization formula:
        distance_from_A = (D - v·Δt) / 2

    Where:
        D = sensor separation distance
        v = wave speed in pipe
        Δt = time delay
    """

    def __init__(
        self,
        registry: SensorRegistry,
        verbose: bool = False
    ):
        """
        Initialize distance calculator.

        Args:
            registry (SensorRegistry): Sensor configuration registry
            verbose (bool): Print verbose output
        """
        self.registry = registry
        self.verbose = verbose

    def calculate_distance(
        self,
        sensor_a: str,
        sensor_b: str,
        time_delay_estimate: TimeDelayEstimate,
        wave_speed_override: Optional[float] = None
    ) -> LeakLocation:
        """
        Calculate leak distance from time delay.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID
            time_delay_estimate (TimeDelayEstimate): Time delay estimate
            wave_speed_override (float, optional): Override wave speed from registry

        Returns:
            LeakLocation: Leak location results

        Raises:
            ValueError: If sensor pair not found or distance invalid
        """
        # Get sensor pair configuration
        pair = self.registry.get_sensor_pair(sensor_a, sensor_b)

        if pair is None:
            raise ValueError(
                f"Sensor pair ({sensor_a}, {sensor_b}) not found in registry"
            )

        # Determine wave speed
        if wave_speed_override is not None:
            wave_speed = wave_speed_override
            pipe_material = f"override_{wave_speed}mps"
        else:
            wave_speed = pair.wave_speed_mps
            pipe_material = pair.pipe_segment.material

        # Sensor separation distance
        D = pair.distance_meters

        # Time delay (use sign: positive means signal reaches sensor_b first)
        delta_t = time_delay_estimate.delay_seconds

        # Calculate distance from sensor A
        # Formula: x = (D - v·Δt) / 2
        #
        # Derivation:
        #   Let x = distance from sensor A to leak
        #   Then (D-x) = distance from sensor B to leak
        #   Time for signal to reach A: t_A = x/v
        #   Time for signal to reach B: t_B = (D-x)/v
        #   Time delay: Δt = t_B - t_A = (D-x)/v - x/v = (D-2x)/v
        #   Solving for x: x = (D - v·Δt) / 2

        distance_from_a = (D - wave_speed * delta_t) / 2
        distance_from_b = D - distance_from_a

        if self.verbose:
            print(f"[i] Distance calculation:")
            print(f"    Sensor pair: {sensor_a} ←→ {sensor_b}")
            print(f"    Separation: {D:.2f}m")
            print(f"    Wave speed: {wave_speed} m/s ({pipe_material})")
            print(f"    Time delay: {delta_t:.6f}s")
            print(f"    Distance from {sensor_a}: {distance_from_a:.2f}m")
            print(f"    Distance from {sensor_b}: {distance_from_b:.2f}m")

        # Validate physical constraints
        is_valid, msg = self._validate_distance(distance_from_a, D)

        if not is_valid:
            if self.verbose:
                print(f"[!] Warning: {msg}")

            # Clip to valid range
            distance_from_a = np.clip(distance_from_a, 0, D)
            distance_from_b = D - distance_from_a

        # Calculate GPS position
        try:
            leak_gps = self.registry.interpolate_leak_position(
                sensor_a, sensor_b, distance_from_a
            )
        except Exception as e:
            if self.verbose:
                print(f"[!] Warning: Could not calculate GPS position: {e}")
            leak_gps = None

        # Prepare quality metrics
        quality_metrics = {
            'time_delay_confidence': time_delay_estimate.confidence,
            'time_delay_snr_db': time_delay_estimate.snr_db,
            'peak_sharpness': time_delay_estimate.peak_sharpness,
            'peak_height': time_delay_estimate.peak_height,
            'physical_constraint_satisfied': is_valid,
            'validation_message': msg
        }

        # Overall confidence (use time delay confidence as base)
        confidence = time_delay_estimate.confidence

        # Create result
        location = LeakLocation(
            sensor_pair=(sensor_a, sensor_b),
            time_delay_seconds=delta_t,
            distance_from_sensor_a_meters=distance_from_a,
            distance_from_sensor_b_meters=distance_from_b,
            sensor_separation_meters=D,
            leak_position_gps=leak_gps,
            confidence=confidence,
            wave_speed_mps=wave_speed,
            pipe_material=pipe_material,
            quality_metrics=quality_metrics
        )

        return location

    def _validate_distance(
        self,
        distance: float,
        max_distance: float
    ) -> Tuple[bool, str]:
        """
        Validate that calculated distance is physically valid.

        Args:
            distance (float): Calculated distance
            max_distance (float): Maximum valid distance (sensor separation)

        Returns:
            Tuple of (is_valid, message)
        """
        tolerance = DISTANCE_TOLERANCE_METERS

        if distance < -tolerance:
            return False, f"Distance ({distance:.2f}m) is negative (beyond tolerance)"

        if distance > max_distance + tolerance:
            return False, f"Distance ({distance:.2f}m) exceeds sensor separation ({max_distance:.2f}m)"

        if distance < 0 or distance > max_distance:
            return True, f"Distance ({distance:.2f}m) outside valid range but within tolerance"

        return True, "Distance valid"

    def to_dict(self, location: LeakLocation) -> Dict:
        """
        Convert LeakLocation to dictionary for JSON serialization.

        Args:
            location (LeakLocation): Leak location result

        Returns:
            dict: JSON-serializable dictionary
        """
        return {
            'sensor_pair': {
                'sensor_a': location.sensor_pair[0],
                'sensor_b': location.sensor_pair[1]
            },
            'time_delay_seconds': round(location.time_delay_seconds, TIME_DELAY_PRECISION_DECIMALS),
            'distance_from_sensor_a_meters': round(location.distance_from_sensor_a_meters, DISTANCE_PRECISION_DECIMALS),
            'distance_from_sensor_b_meters': round(location.distance_from_sensor_b_meters, DISTANCE_PRECISION_DECIMALS),
            'sensor_separation_meters': round(location.sensor_separation_meters, DISTANCE_PRECISION_DECIMALS),
            'leak_position_gps': {
                'latitude': location.leak_position_gps[0] if location.leak_position_gps else None,
                'longitude': location.leak_position_gps[1] if location.leak_position_gps else None,
                'elevation': location.leak_position_gps[2] if location.leak_position_gps else None
            } if location.leak_position_gps else None,
            'confidence': round(location.confidence, CONFIDENCE_PRECISION_DECIMALS),
            'wave_speed_mps': location.wave_speed_mps,
            'pipe_material': location.pipe_material,
            'quality_metrics': location.quality_metrics
        }

    def to_json(self, location: LeakLocation, output_file: Optional[str] = None) -> str:
        """
        Convert LeakLocation to JSON string.

        Args:
            location (LeakLocation): Leak location result
            output_file (str, optional): Save to file if specified

        Returns:
            str: JSON string
        """
        data = self.to_dict(location)
        json_str = json.dumps(data, indent=2)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
            if self.verbose:
                print(f"[✓] Results saved to {output_file}")

        return json_str


class LeakLocalizer:
    """
    Multi-sensor leak localization using triangulation.

    Combines estimates from multiple sensor pairs to improve accuracy.
    """

    def __init__(
        self,
        registry: SensorRegistry,
        verbose: bool = False
    ):
        """
        Initialize leak localizer.

        Args:
            registry (SensorRegistry): Sensor configuration registry
            verbose (bool): Print verbose output
        """
        self.registry = registry
        self.calculator = DistanceCalculator(registry, verbose=verbose)
        self.verbose = verbose

    def triangulate(
        self,
        locations: List[LeakLocation]
    ) -> LeakLocation:
        """
        Combine multiple leak location estimates using weighted averaging.

        Args:
            locations (List[LeakLocation]): List of location estimates from different pairs

        Returns:
            LeakLocation: Combined estimate

        Raises:
            ValueError: If insufficient data for triangulation
        """
        if len(locations) < MIN_PAIRS_FOR_TRIANGULATION:
            raise ValueError(
                f"Need at least {MIN_PAIRS_FOR_TRIANGULATION} sensor pairs for triangulation, "
                f"got {len(locations)}"
            )

        # Weight by confidence
        if TRIANGULATION_WEIGHT_METHOD == 'confidence':
            weights = np.array([loc.confidence for loc in locations])
        elif TRIANGULATION_WEIGHT_METHOD == 'inverse_distance':
            # Not applicable here - would need reference point
            weights = np.ones(len(locations))
        else:  # uniform
            weights = np.ones(len(locations))

        weights = weights / np.sum(weights)  # Normalize

        # Weighted average of GPS positions (if available)
        gps_positions = [loc.leak_position_gps for loc in locations if loc.leak_position_gps is not None]

        if len(gps_positions) > 0:
            lats = np.array([gps[0] for gps in gps_positions])
            lons = np.array([gps[1] for gps in gps_positions])
            elevs = np.array([gps[2] for gps in gps_positions])

            # Use weights for available GPS positions
            valid_weights = weights[:len(gps_positions)]
            valid_weights = valid_weights / np.sum(valid_weights)

            avg_lat = np.sum(lats * valid_weights)
            avg_lon = np.sum(lons * valid_weights)
            avg_elev = np.sum(elevs * valid_weights)

            combined_gps = (avg_lat, avg_lon, avg_elev)
        else:
            combined_gps = None

        # Combined confidence (weighted average)
        combined_confidence = np.sum(
            np.array([loc.confidence for loc in locations]) * weights
        )

        # Check consistency (all estimates should be close)
        if combined_gps:
            max_discrepancy = 0
            for loc in locations:
                if loc.leak_position_gps:
                    # Calculate distance between estimates
                    discrepancy = self.registry._haversine_distance(
                        combined_gps[0], combined_gps[1],
                        loc.leak_position_gps[0], loc.leak_position_gps[1]
                    )
                    max_discrepancy = max(max_discrepancy, discrepancy)

            if self.verbose:
                print(f"[i] Triangulation consistency:")
                print(f"    Max discrepancy: {max_discrepancy:.1f}m")
                print(f"    Combined confidence: {combined_confidence:.3f}")

            if max_discrepancy > MAX_TRIANGULATION_DISCREPANCY_M:
                if self.verbose:
                    print(f"[!] Warning: Estimates are inconsistent (>{MAX_TRIANGULATION_DISCREPANCY_M}m apart)")

        # Create combined result (use first location as template)
        combined = LeakLocation(
            sensor_pair=("TRIANGULATED", f"{len(locations)}_PAIRS"),
            time_delay_seconds=np.mean([loc.time_delay_seconds for loc in locations]),
            distance_from_sensor_a_meters=0.0,  # Not meaningful for triangulation
            distance_from_sensor_b_meters=0.0,
            sensor_separation_meters=0.0,
            leak_position_gps=combined_gps,
            confidence=combined_confidence,
            wave_speed_mps=locations[0].wave_speed_mps,
            pipe_material="combined",
            quality_metrics={
                'n_pairs': len(locations),
                'weights': weights.tolist(),
                'individual_confidences': [loc.confidence for loc in locations],
                'method': TRIANGULATION_WEIGHT_METHOD
            }
        )

        return combined


# ==============================================================================
# MAIN - Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    import argparse
    from time_delay_estimator import TimeDelayEstimate

    parser = argparse.ArgumentParser(description='Distance Calculator Testing')
    parser.add_argument('--registry', default='examples/example_sensor_registry.json',
                       help='Sensor registry file')
    parser.add_argument('--test-calculation', action='store_true',
                       help='Test distance calculation')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("DISTANCE CALCULATOR TEST")
    print("=" * 80)

    # Load registry
    try:
        registry = SensorRegistry(args.registry)
        print(f"\n[✓] Loaded registry: {registry}")

        # Validate
        is_valid, errors = registry.validate()
        if not is_valid:
            print(f"\n[!] Registry validation warnings:")
            for error in errors:
                print(f"    - {error}")

    except Exception as e:
        print(f"\n[✗] Error loading registry: {e}")
        exit(1)

    if args.test_calculation:
        print("\n[i] Testing distance calculation...")

        # Get first sensor pair
        if len(registry.sensor_pairs) == 0:
            print("[✗] No sensor pairs in registry")
            exit(1)

        pair = registry.sensor_pairs[0]
        print(f"\n[i] Using sensor pair: {pair.sensor_a} ←→ {pair.sensor_b}")
        print(f"    Separation: {pair.distance_meters}m")
        print(f"    Wave speed: {pair.wave_speed_mps} m/s")

        # Simulate time delay estimate
        # Example: leak at 30% of distance from sensor A
        leak_fraction = 0.3
        actual_distance_from_a = pair.distance_meters * leak_fraction

        # Calculate expected time delay
        # t_A = distance_from_a / v
        # t_B = distance_from_b / v
        # delta_t = t_B - t_A
        t_a = actual_distance_from_a / pair.wave_speed_mps
        t_b = (pair.distance_meters - actual_distance_from_a) / pair.wave_speed_mps
        expected_delay = t_b - t_a

        print(f"\n[i] Simulated leak:")
        print(f"    Actual distance from {pair.sensor_a}: {actual_distance_from_a:.2f}m")
        print(f"    Expected time delay: {expected_delay:.6f}s")

        # Create mock time delay estimate
        estimate = TimeDelayEstimate(
            delay_samples=expected_delay * SAMPLE_RATE,
            delay_seconds=expected_delay,
            confidence=0.95,
            peak_height=0.85,
            peak_sharpness=3.5,
            snr_db=25.0,
            peak_index=1000
        )

        # Calculate distance
        calculator = DistanceCalculator(registry, verbose=args.verbose)
        location = calculator.calculate_distance(
            pair.sensor_a,
            pair.sensor_b,
            estimate
        )

        print(f"\n[i] Calculated leak location:")
        print(f"    Distance from {pair.sensor_a}: {location.distance_from_sensor_a_meters:.2f}m")
        print(f"    Distance from {pair.sensor_b}: {location.distance_from_sensor_b_meters:.2f}m")
        print(f"    Error: {abs(location.distance_from_sensor_a_meters - actual_distance_from_a):.2f}m")
        print(f"    Confidence: {location.confidence:.3f}")

        if location.leak_position_gps:
            print(f"    GPS: ({location.leak_position_gps[0]:.6f}, {location.leak_position_gps[1]:.6f}, {location.leak_position_gps[2]:.1f}m)")

        # Convert to JSON
        json_output = calculator.to_json(location)
        print(f"\n[i] JSON output:")
        print(json_output)

    print("\n[✓] Test complete")
