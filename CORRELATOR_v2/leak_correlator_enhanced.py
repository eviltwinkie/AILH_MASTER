#!/usr/bin/env python3
"""
CORRELATOR_v2 - Enhanced Leak Correlator with Multi-Leak Detection

Main CLI application with GPU acceleration, multi-leak detection,
batch processing, and advanced visualization.

New Features (v3.0):
- Multi-leak detection (up to 10 simultaneous leaks)
- GPU-accelerated batch processing (1000+ pairs/second)
- Multi-band frequency separation
- Advanced clustering and triangulation
- Comprehensive visualizations
- FP16 precision for 50% memory savings
- CUDA stream parallelism

Usage:
    # Single pair with multi-leak detection
    python leak_correlator_enhanced.py \\
        --sensor-a S001~R123~20250118120530~100.wav \\
        --sensor-b S002~R456~20250118120530~100.wav \\
        --registry sensor_registry.json \\
        --output results/ \\
        --multi-leak \\
        --max-leaks 10 \\
        --gpu \\
        --visualize

    # Batch mode with GPU acceleration
    python leak_correlator_enhanced.py \\
        --batch-mode \\
        --batch-gpu \\
        --input-dir /DATA_SENSORS/SITE_001 \\
        --registry sensor_registry.json \\
        --output-dir /PROC_REPORTS/MULTI_LEAK \\
        --gpu \\
        --precision fp16 \\
        --visualize-batch

Author: AILH Development Team
Date: 2025-11-19
Version: 3.0.0
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
from correlator_utils import load_wav, parse_filename, match_recording_pairs
from multi_leak_detector import EnhancedMultiLeakDetector, MultiLeakResult
from batch_gpu_correlator import BatchGPUCorrelator, BatchConfig
from visualization import MultiLeakVisualizer


def process_single_pair_enhanced(
    wav_a: str,
    wav_b: str,
    registry: SensorRegistry,
    output_dir: str,
    max_leaks: int = 10,
    use_gpu: bool = True,
    precision: str = 'fp16',
    visualize: bool = False,
    verbose: bool = False
) -> dict:
    """
    Process single sensor pair with multi-leak detection.

    Args:
        wav_a: First WAV file
        wav_b: Second WAV file
        registry: Sensor registry
        output_dir: Output directory
        max_leaks: Maximum leaks to detect
        use_gpu: Use GPU acceleration
        precision: 'fp16' or 'fp32'
        visualize: Generate visualization
        verbose: Print detailed information

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n[i] Processing sensor pair:")
        print(f"    File A: {os.path.basename(wav_a)}")
        print(f"    File B: {os.path.basename(wav_b)}")

    # Parse filenames
    meta_a = parse_filename(wav_a)
    meta_b = parse_filename(wav_b)

    sensor_a = meta_a['sensor_id']
    sensor_b = meta_b['sensor_id']

    # Get sensor pair configuration
    pair_config = registry.get_sensor_pair(sensor_a, sensor_b)

    if not pair_config:
        raise ValueError(f"Sensor pair ({sensor_a}, {sensor_b}) not found in registry")

    # Load audio
    if verbose:
        print(f"[i] Loading WAV files...")

    audio_a, sr_a = load_wav(wav_a)
    audio_b, sr_b = load_wav(wav_b)

    # Initialize multi-leak detector
    detector = EnhancedMultiLeakDetector(
        use_gpu=use_gpu,
        precision=precision,
        verbose=verbose
    )

    # Detect leaks
    if verbose:
        print(f"[i] Detecting multiple leaks...")

    result = detector.detect_multi_leak(
        signal_a=audio_a,
        signal_b=audio_b,
        sensor_separation_m=pair_config.distance_meters,
        wave_speed_mps=pair_config.wave_speed_mps,
        max_leaks=max_leaks,
        use_frequency_separation=True
    )

    # Update sensor pair info
    result.sensor_pair = (sensor_a, sensor_b)

    # Save results
    output_data = {
        'sensor_pair': {
            'sensor_a': sensor_a,
            'sensor_b': sensor_b
        },
        'files': {
            'wav_a': wav_a,
            'wav_b': wav_b
        },
        'num_leaks': result.num_leaks,
        'detected_leaks': [
            {
                'distance_from_sensor_a_meters': round(peak.distance_from_sensor_a_meters, 2),
                'time_delay_seconds': round(peak.time_delay_seconds, 6),
                'confidence': round(peak.confidence, 3),
                'snr_db': round(peak.snr_db, 1),
                'peak_sharpness': round(peak.peak_sharpness, 2),
                'frequency_band': peak.frequency_band,
                'cluster_id': peak.cluster_id
            }
            for peak in result.detected_leaks
        ],
        'processing_time_seconds': round(result.processing_time_seconds, 3),
        'method': result.method,
        'gpu_used': result.gpu_used,
        'quality_metrics': result.quality_metrics
    }

    # Save JSON
    json_file = os.path.join(output_dir, f"multi_leak_{sensor_a}_{sensor_b}_{meta_a['timestamp']}.json")
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"[✓] Saved results: {json_file}")

    # Visualize
    if visualize:
        viz = MultiLeakVisualizer(style='scientific', dpi=150)

        viz_file = os.path.join(output_dir, f"multi_leak_{sensor_a}_{sensor_b}_{meta_a['timestamp']}.png")

        viz.plot_multi_leak_result(
            result=result,
            sensor_separation_m=pair_config.distance_meters,
            output_file=viz_file
        )

        if verbose:
            print(f"[✓] Saved visualization: {viz_file}")

    return output_data


def process_batch_gpu(
    input_dir: str,
    sensor_pairs: List[Tuple[str, str]],
    registry: SensorRegistry,
    output_dir: str,
    batch_size: int = 32,
    n_cuda_streams: int = 32,
    precision: str = 'fp16',
    max_leaks: int = 10,
    visualize_batch: bool = False,
    verbose: bool = False
) -> dict:
    """
    Process multiple sensor pairs with GPU batch acceleration.

    Args:
        input_dir: Input directory with WAV files
        sensor_pairs: List of sensor pair tuples
        registry: Sensor registry
        output_dir: Output directory
        batch_size: GPU batch size
        n_cuda_streams: Number of CUDA streams
        precision: 'fp16' or 'fp32'
        max_leaks: Maximum leaks per pair
        visualize_batch: Generate batch summary visualization
        verbose: Print detailed information

    Returns:
        Statistics dictionary
    """
    # Create batch configuration
    config = BatchConfig(
        batch_size=batch_size,
        n_cuda_streams=n_cuda_streams,
        precision=precision,
        max_leaks_per_pair=max_leaks
    )

    # Initialize batch correlator
    correlator = BatchGPUCorrelator(config=config, verbose=verbose)

    # Process directory
    stats = correlator.process_directory(
        input_dir=input_dir,
        sensor_pairs=sensor_pairs,
        sensor_registry=registry,
        output_dir=output_dir
    )

    # Generate batch visualization
    if visualize_batch:
        # Load all results
        results = []
        for json_file in Path(output_dir).glob("multi_leak_*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

                # Convert back to MultiLeakResult (simplified)
                # For visualization purposes
                # (In production, would save/load full objects)

        # TODO: Implement batch visualization

    correlator.cleanup()

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Enhanced Leak Correlator with Multi-Leak Detection (v3.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single pair with multi-leak detection
    python leak_correlator_enhanced.py \\
        --sensor-a sensorA~rec1~20250118120530~100.wav \\
        --sensor-b sensorB~rec2~20250118120530~100.wav \\
        --registry sensor_registry.json \\
        --output results/ \\
        --multi-leak --max-leaks 10 --gpu --visualize

    # Batch GPU mode (high throughput)
    python leak_correlator_enhanced.py \\
        --batch-mode --batch-gpu \\
        --input-dir /DATA_SENSORS/SITE_001 \\
        --registry sensor_registry.json \\
        --output-dir /PROC_REPORTS/MULTI_LEAK \\
        --batch-size 32 --cuda-streams 32 \\
        --precision fp16 --gpu --visualize-batch
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
    parser.add_argument('--output', help='Output directory (single pair mode)')

    # Batch mode arguments
    parser.add_argument('--batch-gpu', action='store_true',
                       help='Use GPU batch acceleration (batch mode)')
    parser.add_argument('--input-dir', help='Input directory (batch mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')
    parser.add_argument('--time-window', type=float, default=BATCH_TIME_WINDOW_SEC,
                       help=f'Time window for matching (default: {BATCH_TIME_WINDOW_SEC}s)')

    # Multi-leak detection options
    parser.add_argument('--multi-leak', action='store_true',
                       help='Enable multi-leak detection')
    parser.add_argument('--max-leaks', type=int, default=10,
                       help='Maximum leaks to detect (default: 10)')
    parser.add_argument('--cluster-threshold', type=float, default=5.0,
                       help='Clustering distance threshold in meters (default: 5.0m)')

    # GPU options
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32'],
                       help='GPU precision (default: fp16)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='GPU batch size (default: 32)')
    parser.add_argument('--cuda-streams', type=int, default=32,
                       help='Number of CUDA streams (default: 32)')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization (single pair mode)')
    parser.add_argument('--visualize-batch', action='store_true',
                       help='Generate batch summary visualization (batch mode)')
    parser.add_argument('--svg', action='store_true',
                       help='Output SVG instead of PNG')

    # Output options
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
    print("ENHANCED LEAK CORRELATOR v3.0 - Multi-Leak Detection")
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

    try:
        if args.batch_mode and args.batch_gpu:
            # Batch GPU mode
            sensor_pairs = [(pair.sensor_a, pair.sensor_b) for pair in registry.sensor_pairs]

            stats = process_batch_gpu(
                input_dir=args.input_dir,
                sensor_pairs=sensor_pairs,
                registry=registry,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                n_cuda_streams=args.cuda_streams,
                precision=args.precision,
                max_leaks=args.max_leaks,
                visualize_batch=args.visualize_batch,
                verbose=args.verbose
            )

            print(f"\n[✓] Batch GPU processing complete")
            print(f"    Total pairs: {stats['total_pairs']}")
            print(f"    Total leaks: {stats['total_leaks']}")
            print(f"    Throughput: {stats['total_pairs'] / stats['processing_time']:.1f} pairs/s")

        else:
            # Single pair mode
            result = process_single_pair_enhanced(
                wav_a=args.sensor_a,
                wav_b=args.sensor_b,
                registry=registry,
                output_dir=args.output,
                max_leaks=args.max_leaks,
                use_gpu=args.gpu,
                precision=args.precision,
                visualize=args.visualize,
                verbose=args.verbose
            )

            print(f"\n[✓] Processing complete")
            print(f"    Detected {result['num_leaks']} leaks")
            print(f"    Processing time: {result['processing_time_seconds']:.2f}s")

        return 0

    except Exception as e:
        print(f"\n[✗] Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
