#!/usr/bin/env python3
"""
CORRELATOR_v2 - Enhanced Leak Correlator with Multi-Leak Detection

Main CLI application with GPU acceleration, multi-leak detection,
batch processing, signal stacking, multi-sensor triangulation,
and professional PDF reporting.

Features (v3.1):
- Multi-leak detection (up to 10 simultaneous leaks)
- GPU-accelerated batch processing (1000+ pairs/second)
- Multi-band frequency separation
- Signal stacking for SNR improvement (√N enhancement)
- Multi-sensor triangulation (N > 2 sensors)
- Professional PDF engineering reports
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
        --max-leaks 10 \\
        --gpu \\
        --visualize

    # Signal stacking mode (multiple recordings from same pair)
    python leak_correlator_enhanced.py \\
        --stack-recordings \\
        --stack-files-a S001~R1~*.wav S001~R2~*.wav S001~R3~*.wav \\
        --stack-files-b S002~R1~*.wav S002~R2~*.wav S002~R3~*.wav \\
        --stack-method weighted \\
        --registry sensor_registry.json \\
        --output results/ \\
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
        --visualize-batch \\
        --generate-report \\
        --triangulate

Author: AILH Development Team
Date: 2025-11-19
Version: 3.1.0
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
from signal_stacking import SignalStacker, StackingMethod
from multi_sensor_triangulation import MultiSensorTriangulator, LeakLocation
from professional_report import ProfessionalReportGenerator


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


def process_stacked_recordings(
    wav_files_a: List[str],
    wav_files_b: List[str],
    registry: SensorRegistry,
    output_dir: str,
    stacking_method: str = 'weighted',
    min_quality: float = 0.3,
    max_leaks: int = 10,
    use_gpu: bool = True,
    precision: str = 'fp16',
    visualize: bool = False,
    verbose: bool = False
) -> dict:
    """
    Process stacked recordings from same sensor pair for SNR improvement.

    Args:
        wav_files_a: List of WAV files from sensor A
        wav_files_b: List of WAV files from sensor B
        registry: Sensor registry
        output_dir: Output directory
        stacking_method: 'signal_averaging', 'weighted', or 'correlation_averaging'
        min_quality: Minimum quality threshold for recordings
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
        print(f"\n[i] Processing stacked recordings:")
        print(f"    Sensor A: {len(wav_files_a)} recordings")
        print(f"    Sensor B: {len(wav_files_b)} recordings")
        print(f"    Stacking method: {stacking_method}")

    # Load all recordings
    recordings_a = []
    recordings_b = []

    for wav_a, wav_b in zip(wav_files_a, wav_files_b):
        audio_a, sr_a = load_wav(wav_a)
        audio_b, sr_b = load_wav(wav_b)
        recordings_a.append(audio_a)
        recordings_b.append(audio_b)

    # Stack signals
    stacker = SignalStacker(verbose=verbose)

    if verbose:
        print(f"[i] Stacking {len(recordings_a)} recordings...")

    stack_result = stacker.stack_recordings(
        recordings_a=recordings_a,
        recordings_b=recordings_b,
        method=StackingMethod[stacking_method.upper()],
        min_quality_threshold=min_quality
    )

    if verbose:
        print(f"[✓] Stacking complete:")
        print(f"    Used recordings: {stack_result.num_recordings_used}/{len(recordings_a)}")
        print(f"    SNR improvement: {stack_result.snr_improvement_db:.1f} dB")

    # Get sensor info from first file
    meta_a = parse_filename(wav_files_a[0])
    meta_b = parse_filename(wav_files_b[0])

    sensor_a = meta_a['sensor_id']
    sensor_b = meta_b['sensor_id']

    # Get sensor pair configuration
    pair_config = registry.get_sensor_pair(sensor_a, sensor_b)

    if not pair_config:
        raise ValueError(f"Sensor pair ({sensor_a}, {sensor_b}) not found in registry")

    # Initialize multi-leak detector
    detector = EnhancedMultiLeakDetector(
        use_gpu=use_gpu,
        precision=precision,
        verbose=verbose
    )

    # Detect leaks on stacked signals
    if verbose:
        print(f"[i] Detecting multiple leaks on stacked signals...")

    result = detector.detect_multi_leak(
        signal_a=stack_result.stacked_signal_a,
        signal_b=stack_result.stacked_signal_b,
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
        'stacking_info': {
            'num_recordings_total': len(recordings_a),
            'num_recordings_used': stack_result.num_recordings_used,
            'snr_improvement_db': round(stack_result.snr_improvement_db, 1),
            'stacking_method': stacking_method
        },
        'files': {
            'wav_files_a': wav_files_a,
            'wav_files_b': wav_files_b
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
    timestamp = meta_a['timestamp']
    json_file = os.path.join(output_dir, f"stacked_multi_leak_{sensor_a}_{sensor_b}_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"[✓] Saved results: {json_file}")

    # Visualize
    if visualize:
        viz = MultiLeakVisualizer(style='scientific', dpi=150)

        viz_file = os.path.join(output_dir, f"stacked_multi_leak_{sensor_a}_{sensor_b}_{timestamp}.png")

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

    # Signal stacking options (v3.1)
    parser.add_argument('--stack-recordings', action='store_true',
                       help='Enable signal stacking for multiple recordings')
    parser.add_argument('--stack-files-a', nargs='+',
                       help='List of WAV files from sensor A for stacking')
    parser.add_argument('--stack-files-b', nargs='+',
                       help='List of WAV files from sensor B for stacking')
    parser.add_argument('--stack-method', default='weighted',
                       choices=['signal_averaging', 'weighted', 'correlation_averaging'],
                       help='Stacking method (default: weighted)')
    parser.add_argument('--min-stack-quality', type=float, default=0.3,
                       help='Minimum quality threshold for stacking (default: 0.3)')

    # Multi-sensor triangulation options (v3.1)
    parser.add_argument('--triangulate', action='store_true',
                       help='Enable multi-sensor triangulation')
    parser.add_argument('--triangulate-results', nargs='+',
                       help='List of result JSON files for triangulation')
    parser.add_argument('--min-sensors', type=int, default=3,
                       help='Minimum sensors for triangulation (default: 3)')

    # Professional report options (v3.1)
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate professional PDF report')
    parser.add_argument('--report-output', default='leak_detection_report.pdf',
                       help='PDF report filename (default: leak_detection_report.pdf)')
    parser.add_argument('--project-name', default='Leak Detection Survey',
                       help='Project name for report')
    parser.add_argument('--site-name', default='Unknown Site',
                       help='Site name for report')
    parser.add_argument('--report-author', default='AILH Correlator v3.1',
                       help='Report author name')
    parser.add_argument('--company-name', default='Water Utility',
                       help='Company name for report')

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
    print("ENHANCED LEAK CORRELATOR v3.1 - Multi-Leak Detection + Stacking + Triangulation")
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
        all_results = []  # Collect all results for triangulation and reporting

        # PROCESSING MODES
        if args.stack_recordings:
            # Signal stacking mode (v3.1)
            if not args.stack_files_a or not args.stack_files_b:
                print("[✗] Error: --stack-files-a and --stack-files-b required with --stack-recordings")
                return 1

            if len(args.stack_files_a) != len(args.stack_files_b):
                print("[✗] Error: Number of files in stack-files-a and stack-files-b must match")
                return 1

            result = process_stacked_recordings(
                wav_files_a=args.stack_files_a,
                wav_files_b=args.stack_files_b,
                registry=registry,
                output_dir=args.output,
                stacking_method=args.stack_method,
                min_quality=args.min_stack_quality,
                max_leaks=args.max_leaks,
                use_gpu=args.gpu,
                precision=args.precision,
                visualize=args.visualize,
                verbose=args.verbose
            )

            print(f"\n[✓] Stacked processing complete")
            print(f"    Stacked {result['stacking_info']['num_recordings_used']} recordings")
            print(f"    SNR improvement: {result['stacking_info']['snr_improvement_db']:.1f} dB")
            print(f"    Detected {result['num_leaks']} leaks")

        elif args.batch_mode and args.batch_gpu:
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

            # Load all results for potential triangulation/reporting
            if args.triangulate or args.generate_report:
                from pathlib import Path
                for json_file in Path(args.output_dir).glob("multi_leak_*.json"):
                    with open(json_file, 'r') as f:
                        all_results.append(json.load(f))

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

            all_results = [result]

        # TRIANGULATION (v3.1)
        triangulated_leaks = None
        if args.triangulate:
            if args.triangulate_results:
                # Load results from specified files
                all_results = []
                for json_file in args.triangulate_results:
                    with open(json_file, 'r') as f:
                        all_results.append(json.load(f))

            if len(all_results) == 0:
                print("\n[!] Warning: No results available for triangulation")
            else:
                print(f"\n[i] Performing multi-sensor triangulation...")
                print(f"    Results: {len(all_results)}")

                # Convert results to LeakLocation objects
                leak_locations = []
                for res in all_results:
                    sensor_a = res['sensor_pair']['sensor_a']
                    sensor_b = res['sensor_pair']['sensor_b']

                    # Get sensor positions
                    sensor_a_info = registry.get_sensor(sensor_a)
                    sensor_b_info = registry.get_sensor(sensor_b)

                    if not sensor_a_info or not sensor_b_info:
                        continue

                    for leak in res['detected_leaks']:
                        leak_loc = LeakLocation(
                            sensor_pair=(sensor_a, sensor_b),
                            sensor_a_position=sensor_a_info.position,
                            sensor_b_position=sensor_b_info.position,
                            distance_from_sensor_a_meters=leak['distance_from_sensor_a_meters'],
                            confidence=leak['confidence'],
                            snr_db=leak['snr_db']
                        )
                        leak_locations.append(leak_loc)

                # Triangulate
                triangulator = MultiSensorTriangulator(verbose=args.verbose)
                triangulated_leaks = triangulator.triangulate_from_pairs(
                    leak_locations,
                    cluster_threshold_meters=args.cluster_threshold
                )

                print(f"[✓] Triangulation complete:")
                print(f"    Triangulated {len(triangulated_leaks)} leak locations")
                for i, leak in enumerate(triangulated_leaks):
                    print(f"    Leak {i+1}: ({leak.position.latitude:.6f}, {leak.position.longitude:.6f}) "
                          f"± {leak.uncertainty_meters:.1f}m (from {leak.num_detections} sensors)")

        # PROFESSIONAL REPORT GENERATION (v3.1)
        if args.generate_report:
            if len(all_results) == 0:
                print("\n[!] Warning: No results available for report generation")
            else:
                print(f"\n[i] Generating professional PDF report...")

                # Convert JSON results to MultiLeakResult objects
                from multi_leak_detector import MultiLeakPeak
                multi_leak_results = []

                for res in all_results:
                    peaks = [
                        MultiLeakPeak(
                            distance_from_sensor_a_meters=leak['distance_from_sensor_a_meters'],
                            time_delay_seconds=leak['time_delay_seconds'],
                            confidence=leak['confidence'],
                            snr_db=leak['snr_db'],
                            peak_sharpness=leak.get('peak_sharpness', 0.0),
                            frequency_band=leak.get('frequency_band'),
                            cluster_id=leak.get('cluster_id', 0)
                        )
                        for leak in res['detected_leaks']
                    ]

                    result_obj = MultiLeakResult(
                        num_leaks=res['num_leaks'],
                        detected_leaks=peaks,
                        processing_time_seconds=res['processing_time_seconds'],
                        method=res['method'],
                        gpu_used=res['gpu_used'],
                        quality_metrics=res['quality_metrics']
                    )
                    result_obj.sensor_pair = (res['sensor_pair']['sensor_a'], res['sensor_pair']['sensor_b'])
                    multi_leak_results.append(result_obj)

                # Generate report
                report_gen = ProfessionalReportGenerator(
                    project_name=args.project_name,
                    site_name=args.site_name,
                    report_author=args.report_author,
                    company_name=args.company_name
                )

                output_file = args.report_output
                if args.output_dir and args.batch_mode:
                    output_file = os.path.join(args.output_dir, args.report_output)
                elif args.output and not args.batch_mode:
                    output_file = os.path.join(args.output, args.report_output)

                report_gen.generate_report(
                    results=multi_leak_results,
                    registry=registry,
                    triangulated_leaks=triangulated_leaks,
                    output_file=output_file,
                    include_raw_data=True,
                    include_validation=True
                )

                print(f"[✓] Professional report generated: {output_file}")

        return 0

    except Exception as e:
        print(f"\n[✗] Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
