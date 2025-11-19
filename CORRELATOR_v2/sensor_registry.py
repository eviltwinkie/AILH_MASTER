#!/usr/bin/env python3
"""
Leak Detection Correlator - Sensor Registry Module

This module manages sensor positions, configurations, and sensor pair relationships
for multi-sensor leak localization.

Author: AILH Development Team
Date: 2025-11-19
Version: 3.2.0
"""

import json
import os
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from correlator_config import *


@dataclass
class SensorPosition:
    """GPS position of a sensor."""
    latitude: float
    longitude: float
    elevation: float = 0.0  # meters above sea level
    depth_meters: float = 0.0  # depth below ground/surface (v3.2)


@dataclass
class SensorInfo:
    """Complete information about a sensor (v3.2 enhanced with environmental params)."""
    sensor_id: str
    name: str
    position: SensorPosition
    site_id: str
    site_name: str
    logger_id: str
    calibration: Optional[Dict] = None
    # Environmental parameters (v3.2)
    temperature_C: Optional[float] = None  # Water temperature at sensor
    pressure_bar: Optional[float] = None   # Water pressure at sensor
    gain_db: Optional[float] = None        # Sensor gain setting


@dataclass
class PipeSegment:
    """Information about pipe segment between sensors."""
    material: str
    diameter_mm: float
    installation_year: Optional[int] = None
    condition: Optional[str] = None  # 'good', 'fair', 'poor'


@dataclass
class SensorPair:
    """Configuration for a pair of sensors."""
    sensor_a: str
    sensor_b: str
    distance_meters: float
    pipe_segment: PipeSegment
    wave_speed_mps: float


class SensorRegistry:
    """
    Manages sensor registry database.

    Provides methods to:
    - Load/save sensor configurations
    - Query sensor positions
    - Find sensor pairs
    - Calculate distances
    - Validate configurations
    """

    def __init__(self, registry_file: Optional[str] = None):
        """
        Initialize sensor registry.

        Args:
            registry_file (str, optional): Path to JSON registry file
        """
        self.sensors: Dict[str, SensorInfo] = {}
        self.sensor_pairs: List[SensorPair] = []
        self.registry_file = registry_file

        if registry_file and os.path.exists(registry_file):
            self.load(registry_file)

    def load(self, filename: str):
        """
        Load sensor registry from JSON file.

        Args:
            filename (str): Path to JSON file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Registry file not found: {filename}")

        with open(filename, 'r') as f:
            data = json.load(f)

        # Load sensors
        self.sensors = {}
        for sensor_id, sensor_data in data.get('sensors', {}).items():
            pos_data = sensor_data.get('position', {})
            position = SensorPosition(
                latitude=pos_data.get('latitude', 0.0),
                longitude=pos_data.get('longitude', 0.0),
                elevation=pos_data.get('elevation', 0.0)
            )

            sensor_info = SensorInfo(
                sensor_id=sensor_id,
                name=sensor_data.get('name', f'Sensor {sensor_id}'),
                position=position,
                site_id=sensor_data.get('site_id', 'UNKNOWN'),
                site_name=sensor_data.get('site_name', 'Unknown Site'),
                logger_id=sensor_data.get('logger_id', ''),
                calibration=sensor_data.get('calibration', {})
            )

            self.sensors[sensor_id] = sensor_info

        # Load sensor pairs
        self.sensor_pairs = []
        for pair_data in data.get('sensor_pairs', []):
            pipe_data = pair_data.get('pipe_segment', {})
            pipe_segment = PipeSegment(
                material=pipe_data.get('material', DEFAULT_PIPE_MATERIAL),
                diameter_mm=pipe_data.get('diameter_mm', 300),
                installation_year=pipe_data.get('installation_year'),
                condition=pipe_data.get('condition')
            )

            # Get wave speed from pair config or look up from material
            wave_speed = pair_data.get('wave_speed_mps')
            if wave_speed is None:
                wave_speed = get_wave_speed(pipe_segment.material)

            pair = SensorPair(
                sensor_a=pair_data['sensor_a'],
                sensor_b=pair_data['sensor_b'],
                distance_meters=pair_data['distance_meters'],
                pipe_segment=pipe_segment,
                wave_speed_mps=wave_speed
            )

            self.sensor_pairs.append(pair)

        self.registry_file = filename

    def save(self, filename: Optional[str] = None):
        """
        Save sensor registry to JSON file.

        Args:
            filename (str, optional): Path to save file. Uses loaded file if None.

        Raises:
            ValueError: If no filename specified and no file was loaded
        """
        if filename is None:
            filename = self.registry_file

        if filename is None:
            raise ValueError("No filename specified for saving registry")

        # Convert to dictionary
        data = {
            'sensors': {},
            'sensor_pairs': []
        }

        for sensor_id, sensor_info in self.sensors.items():
            data['sensors'][sensor_id] = {
                'name': sensor_info.name,
                'position': {
                    'latitude': sensor_info.position.latitude,
                    'longitude': sensor_info.position.longitude,
                    'elevation': sensor_info.position.elevation
                },
                'site_id': sensor_info.site_id,
                'site_name': sensor_info.site_name,
                'logger_id': sensor_info.logger_id,
                'calibration': sensor_info.calibration or {}
            }

        for pair in self.sensor_pairs:
            pair_data = {
                'sensor_a': pair.sensor_a,
                'sensor_b': pair.sensor_b,
                'distance_meters': pair.distance_meters,
                'pipe_segment': {
                    'material': pair.pipe_segment.material,
                    'diameter_mm': pair.pipe_segment.diameter_mm,
                    'installation_year': pair.pipe_segment.installation_year,
                    'condition': pair.pipe_segment.condition
                },
                'wave_speed_mps': pair.wave_speed_mps
            }
            data['sensor_pairs'].append(pair_data)

        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def add_sensor(self, sensor_info: SensorInfo):
        """
        Add a sensor to the registry.

        Args:
            sensor_info (SensorInfo): Sensor information
        """
        self.sensors[sensor_info.sensor_id] = sensor_info

    def add_sensor_pair(self, pair: SensorPair):
        """
        Add a sensor pair to the registry.

        Args:
            pair (SensorPair): Sensor pair configuration

        Raises:
            ValueError: If sensors in pair don't exist in registry
        """
        if pair.sensor_a not in self.sensors:
            raise ValueError(f"Sensor {pair.sensor_a} not found in registry")
        if pair.sensor_b not in self.sensors:
            raise ValueError(f"Sensor {pair.sensor_b} not found in registry")

        self.sensor_pairs.append(pair)

    def get_sensor(self, sensor_id: str) -> Optional[SensorInfo]:
        """
        Get sensor information by ID.

        Args:
            sensor_id (str): Sensor ID

        Returns:
            SensorInfo or None if not found
        """
        return self.sensors.get(sensor_id)

    def get_sensor_pair(self, sensor_a: str, sensor_b: str) -> Optional[SensorPair]:
        """
        Get sensor pair configuration.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID

        Returns:
            SensorPair or None if not found
        """
        for pair in self.sensor_pairs:
            if (pair.sensor_a == sensor_a and pair.sensor_b == sensor_b) or \
               (pair.sensor_a == sensor_b and pair.sensor_b == sensor_a):
                return pair
        return None

    def get_all_pairs(self, sensor_id: Optional[str] = None) -> List[SensorPair]:
        """
        Get all sensor pairs, optionally filtered by sensor ID.

        Args:
            sensor_id (str, optional): Filter pairs containing this sensor

        Returns:
            List of SensorPair objects
        """
        if sensor_id is None:
            return self.sensor_pairs

        return [
            pair for pair in self.sensor_pairs
            if pair.sensor_a == sensor_id or pair.sensor_b == sensor_id
        ]

    def calculate_distance(self, sensor_a: str, sensor_b: str) -> float:
        """
        Calculate GPS distance between two sensors using Haversine formula.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID

        Returns:
            Distance in meters

        Raises:
            ValueError: If sensors not found
        """
        info_a = self.get_sensor(sensor_a)
        info_b = self.get_sensor(sensor_b)

        if info_a is None:
            raise ValueError(f"Sensor {sensor_a} not found")
        if info_b is None:
            raise ValueError(f"Sensor {sensor_b} not found")

        return self._haversine_distance(
            info_a.position.latitude, info_a.position.longitude,
            info_b.position.latitude, info_b.position.longitude
        )

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.

        Args:
            lat1, lon1: First coordinate (degrees)
            lat2, lon2: Second coordinate (degrees)

        Returns:
            Distance in meters
        """
        # Earth radius in meters
        R = 6371000

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def calculate_3d_distance(self, sensor_a: str, sensor_b: str, use_depth: bool = True) -> float:
        """
        Calculate 3D distance between two sensors accounting for depth (v3.2).

        When sensors are at different depths (e.g., hillside installations), the
        straight-line distance differs from horizontal GPS distance.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID
            use_depth (bool): Include depth in calculation (default: True)

        Returns:
            3D distance in meters

        Notes:
            3D distance = sqrt(horizontal_distance^2 + (depth_b - depth_a)^2)
        """
        info_a = self.get_sensor(sensor_a)
        info_b = self.get_sensor(sensor_b)

        if info_a is None:
            raise ValueError(f"Sensor {sensor_a} not found")
        if info_b is None:
            raise ValueError(f"Sensor {sensor_b} not found")

        # Horizontal distance (GPS haversine)
        horizontal_dist = self._haversine_distance(
            info_a.position.latitude, info_a.position.longitude,
            info_b.position.latitude, info_b.position.longitude
        )

        if not use_depth:
            return horizontal_dist

        # Depth difference
        depth_a = info_a.position.depth_meters
        depth_b = info_b.position.depth_meters
        depth_diff = depth_b - depth_a

        # 3D distance using Pythagoras
        distance_3d = math.sqrt(horizontal_dist ** 2 + depth_diff ** 2)

        return distance_3d

    def get_environmental_wave_speed(
        self,
        sensor_a: str,
        sensor_b: str,
        base_wave_speed: float
    ) -> float:
        """
        Calculate average wave speed with environmental corrections (v3.2).

        Uses temperature and pressure at both sensors to correct the base wave speed.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID
            base_wave_speed (float): Base material wave speed (m/s)

        Returns:
            Average corrected wave speed (m/s)

        Notes:
            - Averages environmental conditions from both sensors
            - Falls back to default conditions if sensor data unavailable
        """
        info_a = self.get_sensor(sensor_a)
        info_b = self.get_sensor(sensor_b)

        if info_a is None or info_b is None:
            # Can't apply corrections without sensor info
            return base_wave_speed

        # Get environmental parameters (with defaults)
        temp_a = info_a.temperature_C if info_a.temperature_C is not None else DEFAULT_TEMPERATURE_C
        temp_b = info_b.temperature_C if info_b.temperature_C is not None else DEFAULT_TEMPERATURE_C
        pressure_a = info_a.pressure_bar if info_a.pressure_bar is not None else DEFAULT_PRESSURE_BAR
        pressure_b = info_b.pressure_bar if info_b.pressure_bar is not None else DEFAULT_PRESSURE_BAR

        # Average environmental conditions
        avg_temp = (temp_a + temp_b) / 2.0
        avg_pressure = (pressure_a + pressure_b) / 2.0

        # Apply environmental corrections
        corrected_speed = calculate_environmental_wave_speed(
            base_speed_mps=base_wave_speed,
            temperature_C=avg_temp,
            pressure_bar=avg_pressure
        )

        return corrected_speed

    def interpolate_leak_position(
        self,
        sensor_a: str,
        sensor_b: str,
        distance_from_a: float
    ) -> Tuple[float, float, float]:
        """
        Calculate GPS coordinates of leak based on distance from sensor A.

        Args:
            sensor_a (str): First sensor ID
            sensor_b (str): Second sensor ID
            distance_from_a (float): Distance from sensor A to leak (meters)

        Returns:
            Tuple of (latitude, longitude, elevation)

        Raises:
            ValueError: If sensors not found or distance invalid
        """
        info_a = self.get_sensor(sensor_a)
        info_b = self.get_sensor(sensor_b)

        if info_a is None:
            raise ValueError(f"Sensor {sensor_a} not found")
        if info_b is None:
            raise ValueError(f"Sensor {sensor_b} not found")

        # Get sensor pair configuration
        pair = self.get_sensor_pair(sensor_a, sensor_b)
        if pair is None:
            raise ValueError(f"Sensor pair ({sensor_a}, {sensor_b}) not configured")

        # Validate distance
        if distance_from_a < 0 or distance_from_a > pair.distance_meters:
            raise ValueError(
                f"Distance from sensor A ({distance_from_a}m) is outside valid range "
                f"[0, {pair.distance_meters}m]"
            )

        # Linear interpolation along great circle path
        fraction = distance_from_a / pair.distance_meters

        lat1 = math.radians(info_a.position.latitude)
        lon1 = math.radians(info_a.position.longitude)
        lat2 = math.radians(info_b.position.latitude)
        lon2 = math.radians(info_b.position.longitude)

        # Spherical interpolation (slerp)
        d = self._haversine_distance(
            info_a.position.latitude, info_a.position.longitude,
            info_b.position.latitude, info_b.position.longitude
        ) / 6371000  # Convert to radians on unit sphere

        a = math.sin((1 - fraction) * d) / math.sin(d)
        b = math.sin(fraction * d) / math.sin(d)

        x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
        y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
        z = a * math.sin(lat1) + b * math.sin(lat2)

        lat = math.atan2(z, math.sqrt(x**2 + y**2))
        lon = math.atan2(y, x)

        # Linear interpolation for elevation
        elev = info_a.position.elevation + fraction * (
            info_b.position.elevation - info_a.position.elevation
        )

        return math.degrees(lat), math.degrees(lon), elev

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate registry configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for sensors
        if not self.sensors:
            errors.append("No sensors defined in registry")

        # Validate each sensor
        for sensor_id, sensor in self.sensors.items():
            if not sensor.logger_id:
                errors.append(f"Sensor {sensor_id}: missing logger_id")

            # Validate GPS coordinates
            if not (-90 <= sensor.position.latitude <= 90):
                errors.append(
                    f"Sensor {sensor_id}: invalid latitude {sensor.position.latitude}"
                )
            if not (-180 <= sensor.position.longitude <= 180):
                errors.append(
                    f"Sensor {sensor_id}: invalid longitude {sensor.position.longitude}"
                )

        # Validate sensor pairs
        for i, pair in enumerate(self.sensor_pairs):
            # Check sensors exist
            if pair.sensor_a not in self.sensors:
                errors.append(f"Pair {i}: sensor_a '{pair.sensor_a}' not found")
            if pair.sensor_b not in self.sensors:
                errors.append(f"Pair {i}: sensor_b '{pair.sensor_b}' not found")

            # Validate distance
            is_valid, msg = validate_sensor_separation(pair.distance_meters)
            if not is_valid:
                errors.append(f"Pair {i} ({pair.sensor_a}-{pair.sensor_b}): {msg}")

            # Validate pipe material
            try:
                get_wave_speed(pair.pipe_segment.material)
            except ValueError as e:
                errors.append(f"Pair {i}: {e}")

            # Check distance matches GPS calculation
            if pair.sensor_a in self.sensors and pair.sensor_b in self.sensors:
                calculated_distance = self.calculate_distance(
                    pair.sensor_a, pair.sensor_b
                )
                discrepancy = abs(calculated_distance - pair.distance_meters)

                if discrepancy > DISTANCE_TOLERANCE_METERS:
                    errors.append(
                        f"Pair {i} ({pair.sensor_a}-{pair.sensor_b}): "
                        f"configured distance ({pair.distance_meters}m) differs from "
                        f"GPS calculation ({calculated_distance:.1f}m) by {discrepancy:.1f}m"
                    )

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        return f"SensorRegistry({len(self.sensors)} sensors, {len(self.sensor_pairs)} pairs)"


# ==============================================================================
# MAIN - Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sensor Registry Management')
    parser.add_argument('--load', help='Load registry from JSON file')
    parser.add_argument('--save', help='Save registry to JSON file')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example registry')
    parser.add_argument('--validate', action='store_true',
                       help='Validate registry')
    parser.add_argument('--list-sensors', action='store_true',
                       help='List all sensors')
    parser.add_argument('--list-pairs', action='store_true',
                       help='List all sensor pairs')
    args = parser.parse_args()

    registry = SensorRegistry()

    # Create example registry
    if args.create_example:
        print("[i] Creating example sensor registry...")

        # Add sensors
        registry.add_sensor(SensorInfo(
            sensor_id='S001',
            name='Main Street Alpha',
            position=SensorPosition(40.7128, -74.0060, 10.5),
            site_id='SITE_001',
            site_name='Downtown District',
            logger_id='12345',
            calibration={'gain_offset_db': 0.0}
        ))

        registry.add_sensor(SensorInfo(
            sensor_id='S002',
            name='Main Street Beta',
            position=SensorPosition(40.7138, -74.0055, 11.2),
            site_id='SITE_001',
            site_name='Downtown District',
            logger_id='67890'
        ))

        registry.add_sensor(SensorInfo(
            sensor_id='S003',
            name='Oak Avenue Gamma',
            position=SensorPosition(40.7148, -74.0050, 12.0),
            site_id='SITE_002',
            site_name='Oak Avenue Zone',
            logger_id='11111'
        ))

        # Add sensor pairs
        registry.add_sensor_pair(SensorPair(
            sensor_a='S001',
            sensor_b='S002',
            distance_meters=100.0,
            pipe_segment=PipeSegment(
                material='ductile_iron',
                diameter_mm=300,
                installation_year=1995,
                condition='good'
            ),
            wave_speed_mps=1400
        ))

        registry.add_sensor_pair(SensorPair(
            sensor_a='S002',
            sensor_b='S003',
            distance_meters=150.0,
            pipe_segment=PipeSegment(
                material='ductile_iron',
                diameter_mm=300,
                installation_year=1995
            ),
            wave_speed_mps=1400
        ))

        print(f"[✓] Created {registry}")

    # Load registry
    if args.load:
        print(f"[i] Loading registry from {args.load}...")
        registry.load(args.load)
        print(f"[✓] Loaded {registry}")

    # Validate registry
    if args.validate:
        print("[i] Validating registry...")
        is_valid, errors = registry.validate()

        if is_valid:
            print("[✓] Registry is valid")
        else:
            print(f"[✗] Registry has {len(errors)} errors:")
            for error in errors:
                print(f"    - {error}")

    # List sensors
    if args.list_sensors:
        print(f"\n[i] Sensors in registry ({len(registry.sensors)}):")
        for sensor_id, sensor in registry.sensors.items():
            print(f"\n  {sensor_id}: {sensor.name}")
            print(f"    Logger ID: {sensor.logger_id}")
            print(f"    Position: ({sensor.position.latitude:.6f}, "
                  f"{sensor.position.longitude:.6f}, {sensor.position.elevation:.1f}m)")
            print(f"    Site: {sensor.site_name} ({sensor.site_id})")

    # List pairs
    if args.list_pairs:
        print(f"\n[i] Sensor pairs in registry ({len(registry.sensor_pairs)}):")
        for pair in registry.sensor_pairs:
            print(f"\n  {pair.sensor_a} ←→ {pair.sensor_b}")
            print(f"    Distance: {pair.distance_meters}m")
            print(f"    Pipe: {pair.pipe_segment.material} "
                  f"({pair.pipe_segment.diameter_mm}mm)")
            print(f"    Wave speed: {pair.wave_speed_mps} m/s")
            print(f"    Max time delay: {get_max_time_delay(pair.distance_meters, pair.wave_speed_mps):.3f}s")

    # Save registry
    if args.save:
        print(f"\n[i] Saving registry to {args.save}...")
        registry.save(args.save)
        print(f"[✓] Saved {registry}")
