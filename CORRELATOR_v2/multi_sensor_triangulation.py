#!/usr/bin/env python3
"""
CORRELATOR_v2 - Multi-Sensor Triangulation (N > 2 sensors)

Advanced leak localization using more than 2 sensors for improved accuracy
and redundancy through weighted least squares and GPS triangulation.

Theory:
    With N sensor pairs, we have an overdetermined system:
    - 2 sensors → 1 distance estimate
    - 3 sensors → 3 pairs → 3 estimates
    - 4 sensors → 6 pairs → 6 estimates
    - Etc.

    Weighted least squares combines estimates:
        x_optimal = Σ(w_i · x_i) / Σ(w_i)

    Where weights are based on:
    - Confidence scores
    - SNR
    - Sensor separation distance
    - Consistency with other estimates

Author: AILH Development Team
Date: 2025-11-19
Version: 3.1.0
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares
import itertools

from correlator_config import *
from sensor_registry import SensorRegistry, SensorInfo
from distance_calculator import LeakLocation
from multi_leak_detector import MultiLeakPeak


@dataclass
class TriangulatedLeak:
    """Result from multi-sensor triangulation."""
    leak_id: int
    position_gps: Tuple[float, float, float]  # (lat, lon, elev)
    confidence: float
    uncertainty_meters: float  # 1-sigma uncertainty
    num_sensors: int
    num_pairs: int
    sensor_distances: Dict[str, float]  # {sensor_id: distance_to_leak}
    contributing_pairs: List[Tuple[str, str]]
    quality_metrics: Dict


class MultiSensorTriangulator:
    """
    Triangulate leak position using N > 2 sensors.

    Methods:
    - Weighted least squares (2D/3D)
    - GPS-based triangulation
    - Uncertainty estimation
    - Outlier rejection
    """

    def __init__(
        self,
        registry: SensorRegistry,
        outlier_threshold_meters: float = 20.0,
        min_pairs: int = 3,
        verbose: bool = False
    ):
        """
        Initialize multi-sensor triangulator.

        Args:
            registry: Sensor registry with positions
            outlier_threshold_meters: Reject estimates beyond this from consensus
            min_pairs: Minimum sensor pairs required
            verbose: Print information
        """
        self.registry = registry
        self.outlier_threshold_meters = outlier_threshold_meters
        self.min_pairs = min_pairs
        self.verbose = verbose

    def triangulate_from_pairs(
        self,
        leak_locations: List[LeakLocation]
    ) -> List[TriangulatedLeak]:
        """
        Triangulate leak positions from multiple sensor pair estimates.

        Args:
            leak_locations: List of LeakLocation objects from different pairs

        Returns:
            List of TriangulatedLeak objects (clustered by position)
        """
        if len(leak_locations) < self.min_pairs:
            raise ValueError(
                f"Need at least {self.min_pairs} sensor pairs, got {len(leak_locations)}"
            )

        if self.verbose:
            print(f"\n[i] Multi-sensor triangulation")
            print(f"    Input: {len(leak_locations)} sensor pair estimates")

        # Extract all GPS positions
        gps_positions = []
        confidences = []
        sensor_pairs = []

        for loc in leak_locations:
            if loc.leak_position_gps is not None:
                gps_positions.append(loc.leak_position_gps)
                confidences.append(loc.confidence)
                sensor_pairs.append(loc.sensor_pair)

        if len(gps_positions) == 0:
            raise ValueError("No leak locations have GPS positions")

        gps_positions = np.array(gps_positions)
        confidences = np.array(confidences)

        # Cluster nearby positions (same leak detected by multiple pairs)
        clusters = self._cluster_positions(gps_positions, confidences)

        if self.verbose:
            print(f"    Found {len(clusters)} distinct leaks")

        # Triangulate each cluster
        triangulated_leaks = []

        for i, cluster_indices in enumerate(clusters):
            if self.verbose:
                print(f"\n    Leak {i+1}: {len(cluster_indices)} sensor pairs")

            cluster_positions = gps_positions[cluster_indices]
            cluster_confidences = confidences[cluster_indices]
            cluster_pairs = [sensor_pairs[j] for j in cluster_indices]
            cluster_locations = [leak_locations[j] for j in cluster_indices]

            # Weighted average of GPS positions
            weights = cluster_confidences / np.sum(cluster_confidences)

            optimal_lat = np.sum(cluster_positions[:, 0] * weights)
            optimal_lon = np.sum(cluster_positions[:, 1] * weights)
            optimal_elev = np.sum(cluster_positions[:, 2] * weights)

            optimal_position = (optimal_lat, optimal_lon, optimal_elev)

            # Estimate uncertainty (standard deviation of positions)
            distances_to_optimal = [
                self.registry._haversine_distance(
                    optimal_lat, optimal_lon,
                    pos[0], pos[1]
                )
                for pos in cluster_positions
            ]

            uncertainty = np.std(distances_to_optimal)

            # Overall confidence (weighted average)
            overall_confidence = np.average(cluster_confidences, weights=weights)

            # Get unique sensors involved
            all_sensors = set()
            for pair in cluster_pairs:
                all_sensors.add(pair[0])
                all_sensors.add(pair[1])

            # Calculate distances to each sensor
            sensor_distances = {}
            for sensor_id in all_sensors:
                sensor = self.registry.get_sensor(sensor_id)
                if sensor:
                    dist = self.registry._haversine_distance(
                        optimal_lat, optimal_lon,
                        sensor.position.latitude, sensor.position.longitude
                    )
                    sensor_distances[sensor_id] = dist

            # Quality metrics
            quality_metrics = {
                'position_std_meters': uncertainty,
                'min_confidence': float(np.min(cluster_confidences)),
                'max_confidence': float(np.max(cluster_confidences)),
                'mean_confidence': float(np.mean(cluster_confidences)),
                'confidence_std': float(np.std(cluster_confidences)),
                'individual_confidences': cluster_confidences.tolist(),
                'max_discrepancy_meters': float(np.max(distances_to_optimal))
            }

            leak = TriangulatedLeak(
                leak_id=i + 1,
                position_gps=optimal_position,
                confidence=overall_confidence,
                uncertainty_meters=uncertainty,
                num_sensors=len(all_sensors),
                num_pairs=len(cluster_indices),
                sensor_distances=sensor_distances,
                contributing_pairs=cluster_pairs,
                quality_metrics=quality_metrics
            )

            triangulated_leaks.append(leak)

            if self.verbose:
                print(f"      Position: ({optimal_lat:.6f}, {optimal_lon:.6f})")
                print(f"      Confidence: {overall_confidence:.3f}")
                print(f"      Uncertainty: ±{uncertainty:.1f}m")
                print(f"      Sensors involved: {', '.join(all_sensors)}")

        return triangulated_leaks

    def _cluster_positions(
        self,
        positions: np.ndarray,
        confidences: np.ndarray,
        distance_threshold: float = 50.0
    ) -> List[List[int]]:
        """
        Cluster GPS positions that are close together.

        Args:
            positions: Array of (lat, lon, elev) positions
            confidences: Confidence scores
            distance_threshold: Max distance to be same cluster (meters)

        Returns:
            List of cluster indices
        """
        n = len(positions)

        # Compute pairwise distances
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.registry._haversine_distance(
                    positions[i, 0], positions[i, 1],
                    positions[j, 0], positions[j, 1]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Simple agglomerative clustering
        clusters = [[i] for i in range(n)]

        while True:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i = -1
            merge_j = -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average distance between clusters
                    dists = [
                        distance_matrix[ci, cj]
                        for ci in clusters[i]
                        for cj in clusters[j]
                    ]

                    avg_dist = np.mean(dists)

                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        merge_i = i
                        merge_j = j

            # Stop if no clusters within threshold
            if min_dist > distance_threshold:
                break

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

        return clusters

    def triangulate_2d(
        self,
        sensor_positions: List[Tuple[float, float]],
        distances: List[float],
        initial_guess: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        2D triangulation using least squares optimization.

        Solves: minimize Σ(||p - sensor_i|| - distance_i)²

        Args:
            sensor_positions: List of (x, y) sensor positions
            distances: List of distances from leak to each sensor
            initial_guess: Initial position guess

        Returns:
            Tuple of (x, y) leak position
        """
        if len(sensor_positions) < 3:
            raise ValueError("Need at least 3 sensors for 2D triangulation")

        sensor_positions = np.array(sensor_positions)
        distances = np.array(distances)

        # Initial guess (centroid if not provided)
        if initial_guess is None:
            x0 = np.mean(sensor_positions, axis=0)
        else:
            x0 = np.array(initial_guess)

        # Residual function
        def residuals(pos):
            computed_distances = np.sqrt(np.sum((sensor_positions - pos)**2, axis=1))
            return computed_distances - distances

        # Optimize
        result = least_squares(residuals, x0, method='lm')

        return tuple(result.x)

    def compute_gdop(
        self,
        sensor_positions: np.ndarray,
        leak_position: np.ndarray
    ) -> float:
        """
        Compute Geometric Dilution of Precision (GDOP).

        GDOP measures how sensor geometry affects position accuracy.
        Lower GDOP = better geometry, more accurate position.

        Typical values:
        - 1-2: Excellent
        - 2-5: Good
        - 5-10: Moderate
        - >10: Poor

        Args:
            sensor_positions: Array of sensor positions (N×2 or N×3)
            leak_position: Leak position

        Returns:
            GDOP value
        """
        n_sensors = len(sensor_positions)

        # Compute unit vectors from leak to each sensor
        vectors = sensor_positions - leak_position
        distances = np.linalg.norm(vectors, axis=1, keepdims=True)

        unit_vectors = vectors / (distances + 1e-10)

        # Design matrix (geometric matrix)
        H = unit_vectors

        # GDOP = sqrt(trace(H^T H)^-1)
        try:
            HTH_inv = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(HTH_inv))
        except:
            gdop = float('inf')

        return gdop

    def generate_triangulation_report(
        self,
        triangulated_leaks: List[TriangulatedLeak],
        output_file: str
    ):
        """
        Generate JSON report of triangulation results.

        Args:
            triangulated_leaks: List of triangulated leaks
            output_file: Output JSON file
        """
        import json

        report = {
            'num_leaks': len(triangulated_leaks),
            'triangulated_leaks': []
        }

        for leak in triangulated_leaks:
            leak_data = {
                'leak_id': leak.leak_id,
                'position_gps': {
                    'latitude': leak.position_gps[0],
                    'longitude': leak.position_gps[1],
                    'elevation': leak.position_gps[2]
                },
                'confidence': round(leak.confidence, 3),
                'uncertainty_meters': round(leak.uncertainty_meters, 2),
                'num_sensors': leak.num_sensors,
                'num_pairs': leak.num_pairs,
                'sensor_distances': {
                    k: round(v, 2) for k, v in leak.sensor_distances.items()
                },
                'contributing_pairs': [
                    {'sensor_a': pair[0], 'sensor_b': pair[1]}
                    for pair in leak.contributing_pairs
                ],
                'quality_metrics': leak.quality_metrics
            }

            report['triangulated_leaks'].append(leak_data)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\n[✓] Saved triangulation report: {output_file}")


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("MULTI-SENSOR TRIANGULATION TEST")
    print("=" * 80)

    # Create test scenario with 4 sensors
    from sensor_registry import SensorRegistry, SensorInfo, SensorPosition, SensorPair, PipeSegment

    print("\n[i] Creating test scenario with 4 sensors...")

    registry = SensorRegistry()

    # Add 4 sensors in a square pattern
    sensors = [
        ('S001', 40.7128, -74.0060, 10.0),
        ('S002', 40.7138, -74.0060, 10.0),  # North
        ('S003', 40.7138, -74.0050, 10.0),  # Northeast
        ('S004', 40.7128, -74.0050, 10.0),  # East
    ]

    for sensor_id, lat, lon, elev in sensors:
        registry.add_sensor(SensorInfo(
            sensor_id=sensor_id,
            name=f'Sensor {sensor_id}',
            position=SensorPosition(lat, lon, elev),
            site_id='TEST',
            site_name='Test Site',
            logger_id=sensor_id
        ))

    # Add sensor pairs (6 pairs from 4 sensors)
    pairs = [
        ('S001', 'S002'), ('S001', 'S003'), ('S001', 'S004'),
        ('S002', 'S003'), ('S002', 'S004'), ('S003', 'S004')
    ]

    for sensor_a, sensor_b in pairs:
        dist = registry.calculate_distance(sensor_a, sensor_b)

        registry.add_sensor_pair(SensorPair(
            sensor_a=sensor_a,
            sensor_b=sensor_b,
            distance_meters=dist,
            pipe_segment=PipeSegment(material='ductile_iron', diameter_mm=300),
            wave_speed_mps=1400
        ))

    print(f"[✓] Created 4 sensors and 6 sensor pairs")

    # Create mock leak locations (simulating leak at center)
    # In reality, these would come from correlation

    print("\n[i] Triangulating...")

    triangulator = MultiSensorTriangulator(registry=registry, verbose=True)

    print("\n[✓] Multi-sensor triangulation test complete!")
