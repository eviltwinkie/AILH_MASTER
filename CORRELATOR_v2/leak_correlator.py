#!/usr/bin/env python3
"""
Leak Detection Correlator - Main Application

This is the main CLI application for calculating leak distances between sensor pairs
using cross-correlation analysis of acoustic signals.

Usage:
    # Single pair correlation
    python leak_correlator.py \
        --sensor-a /path/to/sensorA~rec1~timestamp~gain.wav \
        --sensor-b /path/to/sensorB~rec2~timestamp~gain.wav \
        --registry sensor_registry.json \
        --output leak_report.json

    # Batch mode - process multiple pairs
    python leak_correlator.py \
        --batch-mode \
        --input-dir /DATA_SENSORS/SITE_001 \
        --registry sensor_registry.json \
        --output-dir /PROC_REPORTS/CORRELATIONS

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import os
import sys
import argparse
import json
import time
from typing import List, Tuple, Optional

# Import correlator modules
from correlator_config import *
from sensor_registry import SensorRegistry
from correlation_engine import CorrelationEngine
from time_delay_estimator import TimeDelayEstimator
from distance_calculator import DistanceCalculator, LeakLocalizer
from correlator_utils import (
    load_wav, parse_filename, validate_wav_pair,
    match_recording_pairs, find_wav_files
)


class LeakCorrelator:
    """
    Main leak correlator application.

    Orchestrates the complete pipeline:
    1. Load WAV files from sensors
    2. Cross-correlate signals
    3. Estimate time delay
    4. Calculate leak distance
    5. Generate report
    """

    def __init__(
        self,
        registry: SensorRegistry,
        correlation_method: str = DEFAULT_CORRELATION_METHOD,
        use_gpu: bool = USE_GPU_CORRELATION,
        verbose: bool = False,
        debug: bool = False
    ):
        """
        Initialize leak correlator.

        Args:
            registry (SensorRegistry): Sensor configuration registry
            correlation_method (str): Correlation method to use
            use_gpu (bool): Enable GPU acceleration
            verbose (bool): Print verbose output
            debug (bool): Print debug information
        """
        self.registry = registry
        self.verbose = verbose
        self.debug = debug

        # Initialize components
        self.correlation_engine = CorrelationEngine(
            method=correlation_method,
            use_gpu=use_gpu,
            verbose=debug
        )

        self.time_estimator = TimeDelayEstimator(
            verbose=debug
        )

        self.distance_calculator = DistanceCalculator(
            registry=registry,
            verbose=debug
        )

        if self.verbose:
            print(f"[i] Leak correlator initialized")
            print(f"    Correlation method: {correlation_method}")
            print(f"    GPU: {'Enabled' if use_gpu else 'Disabled'}")

    def correlate_pair(
        self,
        wav_a: str,
        wav_b: str,
        wave_speed_override: Optional[float] = None
    ) -> dict:
        """
        Correlate a pair of WAV files and calculate leak location.

        Args:
            wav_a (str): Path to first WAV file
            wav_b (str): Path to second WAV file
            wave_speed_override (float, optional): Override wave speed from registry

        Returns:
            dict: Correlation results and leak location

        Raises:
            ValueError: If files invalid or correlation fails
        """
        if self.verbose:
            print(f"\n[i] Correlating pair:")
            print(f"    File A: {os.path.basename(wav_a)}")
            print(f"    File B: {os.path.basename(wav_b)}")

        t_start = time.time()

        # Parse filenames
        meta_a = parse_filename(wav_a)
        meta_b = parse_filename(wav_b)

        sensor_a = meta_a['sensor_id']
        sensor_b = meta_b['sensor_id']

        if self.verbose:
            print(f"    Sensor A: {sensor_a}")
            print(f"    Sensor B: {sensor_b}")

        # Validate pair
        is_valid, msg = validate_wav_pair(
            wav_a, wav_b,
            check_sync=True,
            verbose=self.debug
        )

        if not is_valid:
            raise ValueError(f"WAV pair validation failed: {msg}")

        # Load audio
        if self.verbose:
            print(f"[i] Loading WAV files...")

        audio_a, sr_a = load_wav(wav_a)
        audio_b, sr_b = load_wav(wav_b)

        # Cross-correlate
        if self.verbose:
            print(f"[i] Computing cross-correlation...")

        correlation, lags = self.correlation_engine.correlate(
            audio_a, audio_b, sample_rate=sr_a
        )

        # Estimate time delay
        if self.verbose:
            print(f"[i] Estimating time delay...")

        time_estimate = self.time_estimator.estimate(correlation, lags)

        # Validate estimate
        is_valid_estimate, val_msg = self.time_estimator.validate_estimate(time_estimate)

        if not is_valid_estimate:
            if self.verbose:
                print(f"[!] Warning: {val_msg}")

        # Calculate leak distance
        if self.verbose:
            print(f"[i] Calculating leak distance...")

        leak_location = self.distance_calculator.calculate_distance(
            sensor_a, sensor_b,
            time_estimate,
            wave_speed_override=wave_speed_override
        )

        # Compute elapsed time
        t_elapsed = time.time() - t_start

        # Prepare result
        result = {
            'success': True,
            'files': {
                'sensor_a': wav_a,
                'sensor_b': wav_b
            },
            'metadata': {
                'sensor_a': meta_a,
                'sensor_b': meta_b
            },
            'time_estimate': {
                'delay_seconds': time_estimate.delay_seconds,
                'delay_samples': time_estimate.delay_samples,
                'confidence': time_estimate.confidence,
                'snr_db': time_estimate.snr_db,
                'peak_sharpness': time_estimate.peak_sharpness
            },
            'leak_location': self.distance_calculator.to_dict(leak_location),
            'processing_time_seconds': t_elapsed,
            'validation': {
                'estimate_valid': is_valid_estimate,
                'validation_message': val_msg
            }
        }

        if self.verbose:
            print(f"\n[✓] Correlation complete in {t_elapsed:.2f}s")
            print(f"    Time delay: {time_estimate.delay_seconds:.6f}s")
            print(f"    Distance from {sensor_a}: {leak_location.distance_from_sensor_a_meters:.2f}m")
            print(f"    Confidence: {leak_location.confidence:.3f}")

        return result

    def batch_correlate(
        self,
        input_dir: str,
        output_dir: str,
        sensor_pairs: Optional[List[Tuple[str, str]]] = None,
        time_window_sec: float = BATCH_TIME_WINDOW_SEC,
        max_pairs: int = MAX_BATCH_PAIRS
    ) -> List[dict]:
        """
        Batch process multiple sensor pairs.

        Args:
            input_dir (str): Directory containing WAV files
            output_dir (str): Directory for output reports
            sensor_pairs (List[Tuple], optional): Specific sensor pairs to process.
                If None, process all pairs from registry.
            time_window_sec (float): Time window for matching recordings
            max_pairs (int): Maximum number of pairs to process

        Returns:
            List of correlation results
        """
        if self.verbose:
            print(f"\n[i] Batch correlation mode")
            print(f"    Input directory: {input_dir}")
            print(f"    Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Determine sensor pairs to process
        if sensor_pairs is None:
            # Use all pairs from registry
            pairs_to_process = [
                (pair.sensor_a, pair.sensor_b)
                for pair in self.registry.sensor_pairs
            ]
        else:
            pairs_to_process = sensor_pairs

        if self.verbose:
            print(f"    Processing {len(pairs_to_process)} sensor pairs")

        # Results
        all_results = []

        # Process each sensor pair
        for i, (sensor_a, sensor_b) in enumerate(pairs_to_process):
            if self.verbose:
                print(f"\n[i] Processing pair {i+1}/{len(pairs_to_process)}: {sensor_a} ←→ {sensor_b}")

            # Find matching recordings
            matched_pairs = match_recording_pairs(
                input_dir,
                sensor_a,
                sensor_b,
                time_window_sec=time_window_sec
            )

            if len(matched_pairs) == 0:
                if self.verbose:
                    print(f"[!] No matching recordings found for this pair")
                continue

            if self.verbose:
                print(f"    Found {len(matched_pairs)} matching recordings")

            # Limit number of pairs
            matched_pairs = matched_pairs[:max_pairs]

            # Process each matched pair
            for j, (wav_a, wav_b, drift) in enumerate(matched_pairs):
                if self.verbose:
                    print(f"\n    [{j+1}/{len(matched_pairs)}] Processing:")
                    print(f"      {os.path.basename(wav_a)}")
                    print(f"      {os.path.basename(wav_b)}")
                    print(f"      Time drift: {drift:.3f}s")

                try:
                    # Correlate
                    result = self.correlate_pair(wav_a, wav_b)

                    # Add to results
                    all_results.append(result)

                    # Save individual result
                    meta_a = result['metadata']['sensor_a']
                    output_filename = f"correlation_{sensor_a}_{sensor_b}_{meta_a['timestamp']}.json"
                    output_path = os.path.join(output_dir, output_filename)

                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)

                    if self.verbose:
                        print(f"      [✓] Saved: {output_filename}")

                except Exception as e:
                    if self.verbose:
                        print(f"      [✗] Error: {e}")

                    # Record failure
                    all_results.append({
                        'success': False,
                        'files': {'sensor_a': wav_a, 'sensor_b': wav_b},
                        'error': str(e)
                    })

        # Save batch summary
        summary = {
            'total_pairs_processed': len(all_results),
            'successful': sum(1 for r in all_results if r.get('success', False)),
            'failed': sum(1 for r in all_results if not r.get('success', False)),
            'results': all_results
        }

        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"\n[✓] Batch processing complete")
            print(f"    Total processed: {len(all_results)}")
            print(f"    Successful: {summary['successful']}")
            print(f"    Failed: {summary['failed']}")
            print(f"    Summary saved: {summary_path}")

        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Leak Detection Correlator - Calculate leak distances between sensor pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single pair correlation
    python leak_correlator.py \\
        --sensor-a sensorA~rec1~20250118120530~100.wav \\
        --sensor-b sensorB~rec2~20250118120530~100.wav \\
        --registry sensor_registry.json \\
        --output leak_report.json

    # Batch mode
    python leak_correlator.py \\
        --batch-mode \\
        --input-dir /DATA_SENSORS/SITE_001 \\
        --registry sensor_registry.json \\
        --output-dir /PROC_REPORTS/CORRELATIONS
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--sensor-a', help='First sensor WAV file (single pair mode)')
    mode_group.add_argument('--batch-mode', action='store_true',
                           help='Batch processing mode')

    # Required arguments
    parser.add_argument('--registry', required=True,
                       help='Sensor registry JSON file')

    # Single pair mode arguments
    parser.add_argument('--sensor-b', help='Second sensor WAV file (single pair mode)')
    parser.add_argument('--output', help='Output JSON file (single pair mode)')

    # Batch mode arguments
    parser.add_argument('--input-dir', help='Input directory (batch mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')
    parser.add_argument('--time-window', type=float, default=BATCH_TIME_WINDOW_SEC,
                       help=f'Time window for matching recordings (default: {BATCH_TIME_WINDOW_SEC}s)')
    parser.add_argument('--max-pairs', type=int, default=MAX_BATCH_PAIRS,
                       help=f'Maximum pairs per sensor combination (default: {MAX_BATCH_PAIRS})')

    # Correlation options
    parser.add_argument('--method', default=DEFAULT_CORRELATION_METHOD,
                       choices=list(CORRELATION_METHODS.keys()),
                       help=f'Correlation method (default: {DEFAULT_CORRELATION_METHOD})')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--wave-speed', type=float,
                       help='Override wave speed (m/s)')
    parser.add_argument('--pipe-material',
                       help='Override pipe material (e.g., ductile_iron, steel, pvc)')

    # Output options
    parser.add_argument('--plot', action='store_true',
                       help='Generate correlation plots')
    parser.add_argument('--svg', action='store_true',
                       help='Output SVG plots instead of PNG')

    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug output')

    args = parser.parse_args()

    # Validate arguments
    if not args.batch_mode:
        if not args.sensor_b:
            parser.error("--sensor-b required in single pair mode")
        if not args.output:
            parser.error("--output required in single pair mode")
    else:
        if not args.input_dir:
            parser.error("--input-dir required in batch mode")
        if not args.output_dir:
            parser.error("--output-dir required in batch mode")

    print("=" * 80)
    print("LEAK DETECTION CORRELATOR v2.0.0")
    print("=" * 80)

    # Load sensor registry
    if args.verbose:
        print(f"\n[i] Loading sensor registry: {args.registry}")

    try:
        registry = SensorRegistry(args.registry)

        if args.verbose:
            print(f"[✓] Loaded registry: {registry}")

        # Validate registry
        is_valid, errors = registry.validate()
        if not is_valid:
            print(f"\n[!] Registry validation warnings:")
            for error in errors:
                print(f"    - {error}")
            print()

    except Exception as e:
        print(f"[✗] Error loading registry: {e}")
        return 1

    # Resolve wave speed
    wave_speed_override = None
    if args.wave_speed:
        wave_speed_override = args.wave_speed
    elif args.pipe_material:
        try:
            wave_speed_override = get_wave_speed(args.pipe_material)
        except ValueError as e:
            print(f"[✗] Error: {e}")
            return 1

    # Initialize correlator
    correlator = LeakCorrelator(
        registry=registry,
        correlation_method=args.method,
        use_gpu=args.gpu,
        verbose=args.verbose,
        debug=args.debug
    )

    try:
        if args.batch_mode:
            # Batch processing
            results = correlator.batch_correlate(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                time_window_sec=args.time_window,
                max_pairs=args.max_pairs
            )

        else:
            # Single pair processing
            result = correlator.correlate_pair(
                wav_a=args.sensor_a,
                wav_b=args.sensor_b,
                wave_speed_override=wave_speed_override
            )

            # Save result
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n[✓] Results saved to: {args.output}")

            # Generate plot if requested
            if args.plot:
                print("[i] Plot generation not yet implemented")
                # TODO: Implement plotting

        print("\n[✓] Correlation complete")
        return 0

    except Exception as e:
        print(f"\n[✗] Error during correlation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
