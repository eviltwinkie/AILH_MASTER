#!/usr/bin/env python3
"""
Example: Correlate a Pair of Sensor Recordings

This example demonstrates how to correlate two WAV files from different sensors
to calculate the leak distance.

Author: AILH Development Team
Date: 2025-11-19
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_registry import SensorRegistry
from correlation_engine import CorrelationEngine
from time_delay_estimator import TimeDelayEstimator
from distance_calculator import DistanceCalculator
from correlator_utils import load_wav, parse_filename


def main():
    """Example workflow."""

    print("=" * 80)
    print("EXAMPLE: Correlate Sensor Pair")
    print("=" * 80)

    # Paths to WAV files (replace with your actual files)
    wav_file_a = "S001~R123~20250118120530~100.wav"
    wav_file_b = "S002~R456~20250118120530~100.wav"

    # Sensor registry
    registry_file = "example_sensor_registry.json"

    print(f"\n[i] Files to correlate:")
    print(f"    Sensor A: {wav_file_a}")
    print(f"    Sensor B: {wav_file_b}")
    print(f"    Registry: {registry_file}")

    # Load sensor registry
    print(f"\n[1] Loading sensor registry...")
    registry = SensorRegistry(registry_file)
    print(f"[✓] Loaded: {registry}")

    # Parse filenames
    print(f"\n[2] Parsing filenames...")
    meta_a = parse_filename(wav_file_a)
    meta_b = parse_filename(wav_file_b)

    sensor_a = meta_a['sensor_id']
    sensor_b = meta_b['sensor_id']

    print(f"[✓] Sensor A ID: {sensor_a}")
    print(f"[✓] Sensor B ID: {sensor_b}")

    # Load WAV files
    print(f"\n[3] Loading WAV files...")

    # NOTE: In this example, the files don't actually exist
    # You would replace this with:
    # audio_a, sr_a = load_wav(wav_file_a)
    # audio_b, sr_b = load_wav(wav_file_b)

    print(f"[i] (Skipping WAV loading - files don't exist in example)")
    print(f"[i] In real usage, load with: audio_a, sr_a = load_wav(wav_file_a)")

    # Initialize components
    print(f"\n[4] Initializing correlator components...")

    correlation_engine = CorrelationEngine(
        method='gcc_phat',
        use_gpu=False,
        verbose=True
    )

    time_estimator = TimeDelayEstimator(
        interpolation_method='parabolic',
        verbose=True
    )

    distance_calculator = DistanceCalculator(
        registry=registry,
        verbose=True
    )

    print(f"[✓] Components initialized")

    # Correlation workflow (conceptual - requires actual audio data)
    print(f"\n[5] Correlation workflow (conceptual):")
    print(f"""
    # Step 5a: Cross-correlate signals
    correlation, lags = correlation_engine.correlate(audio_a, audio_b)

    # Step 5b: Estimate time delay
    time_estimate = time_estimator.estimate(correlation, lags)
    print(f"Time delay: {{time_estimate.delay_seconds:.6f}}s")
    print(f"Confidence: {{time_estimate.confidence:.3f}}")

    # Step 5c: Calculate leak distance
    leak_location = distance_calculator.calculate_distance(
        sensor_a, sensor_b, time_estimate
    )

    # Step 5d: Display results
    print(f"Distance from {{sensor_a}}: {{leak_location.distance_from_sensor_a_meters:.2f}}m")
    print(f"Distance from {{sensor_b}}: {{leak_location.distance_from_sensor_b_meters:.2f}}m")

    if leak_location.leak_position_gps:
        lat, lon, elev = leak_location.leak_position_gps
        print(f"Leak GPS: ({{lat:.6f}}, {{lon:.6f}}, {{elev:.1f}}m)")

    # Step 5e: Save to JSON
    json_output = distance_calculator.to_json(leak_location, "leak_report.json")
    """)

    print(f"\n[✓] Example complete!")
    print(f"\n[i] To run with real data, use the main application:")
    print(f"    python leak_correlator.py \\")
    print(f"        --sensor-a /path/to/sensorA.wav \\")
    print(f"        --sensor-b /path/to/sensorB.wav \\")
    print(f"        --registry {registry_file} \\")
    print(f"        --output leak_report.json \\")
    print(f"        --verbose")


if __name__ == '__main__':
    main()
