#!/usr/bin/env python3
"""
CORRELATOR_V3 - Main CLI
Physics-aware, Bayesian, AI-integrated leak correlation
"""

import argparse
import json
import sys
import os
import numpy as np
import wave
from datetime import datetime

# Import V3 modules
from correlator_v3_config import *
from physics_aware_correlator import PhysicsAwareCorrelator, JointSearchResult
from correlation_variants import MultiMethodCorrelator
from coherence_analyzer import CoherenceAnalyzer
from bayesian_estimator import BayesianEstimator
from ai_window_gating import AIWindowGating
from robust_stacking import adaptive_stacking

# Import v2 modules for compatibility
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CORRELATOR_v2'))
from sensor_registry import SensorRegistry
from correlator_utils import load_wav_file, parse_wav_filename


def main():
    parser = argparse.ArgumentParser(
        description='CORRELATOR_V3 - Advanced leak correlation with physics-aware, Bayesian, and AI features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic correlation (V2-compatible)
  %(prog)s --sensor-a A.wav --sensor-b B.wav --registry registry.json --output report.json

  # Full V3 with all features
  %(prog)s --sensor-a A.wav --sensor-b B.wav --registry registry.json \\
    --physics-aware --bayesian --ai-gating --coherence --output report.json

  # Custom correlation methods
  %(prog)s --sensor-a A.wav --sensor-b B.wav --registry registry.json \\
    --correlation-methods gcc_phat,gcc_scot --output report.json
        """
    )

    # Input files
    parser.add_argument('--sensor-a', required=True, help='WAV file from sensor A')
    parser.add_argument('--sensor-b', required=True, help='WAV file from sensor B')
    parser.add_argument('--registry', required=True, help='Sensor registry JSON file')

    # Output
    parser.add_argument('--output', required=True, help='Output JSON report file')

    # V3 Feature toggles
    parser.add_argument('--physics-aware', action='store_true',
                       default=ENABLE_JOINT_POSITION_VELOCITY_SEARCH,
                       help='Enable joint (x,c) search')
    parser.add_argument('--no-physics-aware', dest='physics_aware', action='store_false')

    parser.add_argument('--bayesian', action='store_true',
                       default=ENABLE_BAYESIAN_ESTIMATION,
                       help='Enable Bayesian estimation')
    parser.add_argument('--no-bayesian', dest='bayesian', action='store_false')

    parser.add_argument('--ai-gating', action='store_true',
                       default=ENABLE_AI_WINDOW_GATING,
                       help='Enable AI window gating')
    parser.add_argument('--no-ai-gating', dest='ai_gating', action='store_false')

    parser.add_argument('--coherence', action='store_true',
                       default=ENABLE_COHERENCE_BAND_SELECTION,
                       help='Enable coherence-driven band selection')
    parser.add_argument('--no-coherence', dest='coherence', action='store_false')

    # Correlation methods
    parser.add_argument('--correlation-methods',
                       default=','.join(ENABLED_CORRELATION_METHODS),
                       help='Comma-separated list of methods: gcc_phat,gcc_roth,gcc_scot,classical,wavelet')

    # Stacking method
    parser.add_argument('--stacking-method',
                       default=ROBUST_STACKING_METHOD,
                       choices=['mean', 'trimmed_mean', 'median', 'huber'],
                       help='Robust stacking method')

    # Global flags
    parser.add_argument('--verbose', action='store_true', default=VERBOSE, help='Verbose output')
    parser.add_argument('--debug', action='store_true', default=DEBUG, help='Debug output')
    parser.add_argument('--upscale', action='store_true', default=UPSCALE, help='Upscale audio to 8192 Hz')
    parser.add_argument('--svg', action='store_true', help='Output SVG plots instead of PNG')

    args = parser.parse_args()

    # Override global config with CLI args
    verbose = args.verbose or LOGGING
    debug = args.debug

    if verbose:
        print("=" * 80)
        print("CORRELATOR_V3 - Advanced Leak Correlation")
        print("=" * 80)
        print(f"\n[i] Configuration:")
        print(f"    Sensor A: {args.sensor_a}")
        print(f"    Sensor B: {args.sensor_b}")
        print(f"    Registry: {args.registry}")
        print(f"    Physics-aware: {args.physics_aware}")
        print(f"    Bayesian: {args.bayesian}")
        print(f"    AI gating: {args.ai_gating}")
        print(f"    Coherence: {args.coherence}")
        print(f"    Correlation methods: {args.correlation_methods}")
        print(f"    Stacking: {args.stacking_method}")

    # Load sensor registry
    if verbose:
        print(f"\n[i] Loading sensor registry...")
    registry = SensorRegistry(args.registry)

    # Parse filenames
    info_a = parse_wav_filename(args.sensor_a)
    info_b = parse_wav_filename(args.sensor_b)

    sensor_id_a = info_a['sensor_id']
    sensor_id_b = info_b['sensor_id']

    if verbose:
        print(f"    Sensor pair: {sensor_id_a} <-> {sensor_id_b}")

    # Get sensor pair config
    pair_config = registry.get_sensor_pair(sensor_id_a, sensor_id_b)
    if pair_config is None:
        print(f"[✗] Error: Sensor pair {sensor_id_a} <-> {sensor_id_b} not found in registry")
        sys.exit(1)

    sensor_separation_m = pair_config['distance_meters']
    pipe_material = pair_config.get('pipe_segment', {}).get('material', 'ductile_iron')
    wave_speed_mps = pair_config.get('wave_speed_mps', 1400)

    if verbose:
        print(f"    Separation: {sensor_separation_m:.1f}m")
        print(f"    Pipe material: {pipe_material}")
        print(f"    Wave speed: {wave_speed_mps} m/s")

    # Load audio files
    if verbose:
        print(f"\n[i] Loading audio files...")

    signal_a, sample_rate_a = load_wav_file(args.sensor_a)
    signal_b, sample_rate_b = load_wav_file(args.sensor_b)

    if sample_rate_a != sample_rate_b:
        print(f"[✗] Error: Sample rates don't match ({sample_rate_a} vs {sample_rate_b})")
        sys.exit(1)

    sample_rate = sample_rate_a

    if verbose:
        print(f"    Sample rate: {sample_rate} Hz")
        print(f"    Duration: {len(signal_a) / sample_rate:.1f}s")
        print(f"    Samples: {len(signal_a)}")

    # Upscale if requested
    if args.upscale and sample_rate == 4096:
        if verbose:
            print(f"\n[i] Upscaling audio to 8192 Hz...")
        from scipy.signal import resample
        target_samples = len(signal_a) * 2
        signal_a = resample(signal_a, target_samples)
        signal_b = resample(signal_b, target_samples)
        sample_rate = 8192
        if verbose:
            print(f"    [✓] Upscaled to {sample_rate} Hz")

    # CORRELATE
    start_time = datetime.now()

    # Initialize result dict
    result = {
        'sensor_pair': [sensor_id_a, sensor_id_b],
        'timestamp': start_time.isoformat(),
        'configuration': {
            'physics_aware': args.physics_aware,
            'bayesian': args.bayesian,
            'ai_gating': args.ai_gating,
            'coherence': args.coherence,
            'correlation_methods': args.correlation_methods.split(','),
            'stacking_method': args.stacking_method,
        },
        'sensor_separation_m': sensor_separation_m,
        'pipe_material': pipe_material,
        'wave_speed_mps_assumed': wave_speed_mps,
    }

    # Step 1: AI Window Gating (if enabled)
    if args.ai_gating:
        if verbose:
            print(f"\n[1/5] AI Window Gating...")
        gating = AIWindowGating(verbose=verbose)
        ai_result = gating.process_pair(signal_a, signal_b, sample_rate)

        result['ai_window_gating'] = {
            'overall_leak_confidence': ai_result.overall_leak_confidence,
            'windows_analyzed': len(ai_result.per_window_leak_probs),
            'mean_leak_probability': float(np.mean(ai_result.per_window_leak_probs)),
        }

        if verbose:
            print(f"    [✓] AI confidence: {ai_result.overall_leak_confidence:.3f}")

    # Step 2: Coherence Analysis (if enabled)
    if args.coherence:
        if verbose:
            print(f"\n[2/5] Coherence Analysis...")
        analyzer = CoherenceAnalyzer(sample_rate=sample_rate, verbose=verbose)
        coh_result = analyzer.compute_coherence(signal_a, signal_b)

        result['coherence_analysis'] = {
            'mean_coherence': coh_result.mean_coherence,
            'high_coherence_bands': [[float(low), float(high)] for low, high in coh_result.high_coherence_bands],
        }

    # Step 3: Multi-Method Correlation
    if verbose:
        print(f"\n[3/5] Multi-Method Correlation...")

    methods = args.correlation_methods.split(',')
    # Validate methods list
    methods = [m.strip() for m in methods if m.strip()]  # Remove empty strings
    if not methods:
        print(f"[✗] Error: No correlation methods specified")
        sys.exit(1)

    correlator = MultiMethodCorrelator(methods=methods, verbose=verbose)
    correlations = correlator.correlate(signal_a, signal_b, sample_rate)

    # Fuse correlations
    fused_correlation = adaptive_stacking(
        np.array(list(correlations.values())),
        method=args.stacking_method
    )

    if verbose:
        print(f"    [✓] Fused {len(correlations)} methods with {args.stacking_method} stacking")

    # Step 4: Physics-Aware / Bayesian Estimation
    if args.physics_aware:
        if verbose:
            print(f"\n[4/5] Physics-Aware Correlation...")

        physics_correlator = PhysicsAwareCorrelator(sample_rate=sample_rate, verbose=verbose)
        joint_result = physics_correlator.joint_search(
            signal_a, signal_b, sensor_separation_m
        )

        result['physics_aware'] = {
            'optimal_position_m': joint_result.optimal_position_m,
            'optimal_velocity_mps': joint_result.optimal_velocity_mps,
            'confidence': joint_result.confidence_score,
        }

        # Use physics-aware estimates for Bayesian
        estimated_wave_speed = joint_result.optimal_velocity_mps
        if verbose:
            print(f"    [✓] Physics-aware position: {joint_result.optimal_position_m:.2f}m")
            print(f"    [✓] Estimated velocity: {estimated_wave_speed:.1f} m/s")
    else:
        estimated_wave_speed = wave_speed_mps

    # Step 5: Bayesian Estimation
    if args.bayesian:
        if verbose:
            print(f"\n[5/5] Bayesian Estimation...")

        estimator = BayesianEstimator(prior_type='uniform', verbose=verbose)
        bayesian_result = estimator.estimate(
            fused_correlation, sensor_separation_m, estimated_wave_speed, sample_rate
        )

        result['bayesian_estimation'] = {
            'map_position_m': bayesian_result.map_position_m,
            'credible_interval_m': list(bayesian_result.credible_interval_m),
            'entropy': bayesian_result.entropy,
        }

        final_position_m = bayesian_result.map_position_m
        final_confidence = 1.0 / (1.0 + bayesian_result.entropy)  # Lower entropy = higher confidence

        if verbose:
            print(f"    [✓] MAP estimate: {bayesian_result.map_position_m:.2f}m")
            print(f"    [✓] Credible interval: [{bayesian_result.credible_interval_m[0]:.2f}, {bayesian_result.credible_interval_m[1]:.2f}]m")

    else:
        # Fallback: peak of fused correlation
        center_idx = len(fused_correlation) // 2
        peak_idx = np.argmax(np.abs(fused_correlation))
        delay_samples = peak_idx - center_idx
        delay_sec = delay_samples / sample_rate
        final_position_m = (sensor_separation_m - estimated_wave_speed * delay_sec) / 2
        final_confidence = np.max(np.abs(fused_correlation)) / (np.mean(np.abs(fused_correlation)) + 1e-12)
        final_confidence = min(1.0, final_confidence / 10.0)

    # Final result
    result['leak_detection'] = {
        'leak_detected': True if final_confidence > 0.5 else False,
        'position_from_sensor_a_m': float(final_position_m),
        'position_from_sensor_b_m': float(sensor_separation_m - final_position_m),
        'confidence': float(final_confidence),
        'estimated_wave_speed_mps': float(estimated_wave_speed),
    }

    # Processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    result['processing_time_sec'] = processing_time

    # Write output
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n" + "=" * 80)
        print(f"[✓] CORRELATOR_V3 COMPLETE")
        print(f"=" * 80)
        print(f"\nLeak Detection:")
        print(f"  Detected: {'YES' if result['leak_detection']['leak_detected'] else 'NO'}")
        print(f"  Position from {sensor_id_a}: {result['leak_detection']['position_from_sensor_a_m']:.2f}m")
        print(f"  Position from {sensor_id_b}: {result['leak_detection']['position_from_sensor_b_m']:.2f}m")
        print(f"  Confidence: {result['leak_detection']['confidence']:.3f}")
        if args.physics_aware:
            print(f"  Estimated wave speed: {result['leak_detection']['estimated_wave_speed_mps']:.1f} m/s")
        print(f"\nProcessing time: {processing_time:.2f}s")
        print(f"Output saved to: {args.output}")
        print("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
