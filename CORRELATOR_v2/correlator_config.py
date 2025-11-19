#!/usr/bin/env python3
"""
Leak Detection Correlator - Configuration Module

This module defines all configuration parameters, constants, and settings for the
leak detection correlator system. It extends the global_config.py from AILH_MASTER.

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import os
import sys

# Import global AILH configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_config import *

# ==============================================================================
# CORRELATOR-SPECIFIC CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Wave Speeds by Pipe Material (meters per second)
# ------------------------------------------------------------------------------
WAVE_SPEEDS = {
    'ductile_iron': 1400,      # Default for urban water supply
    'steel': 5000,              # High-pressure transmission mains
    'pvc': 450,                 # Modern residential distribution
    'cast_iron': 3500,          # Legacy infrastructure
    'concrete': 3700,           # Large diameter mains
    'copper': 3560,             # Service connections
    'hdpe': 400,                # High-density polyethylene
    'asbestos_cement': 3100,    # Legacy (being replaced)
    'galvanized_steel': 4900,   # Older service lines
}

# Default pipe material if not specified
DEFAULT_PIPE_MATERIAL = 'ductile_iron'
DEFAULT_WAVE_SPEED = WAVE_SPEEDS[DEFAULT_PIPE_MATERIAL]

# ------------------------------------------------------------------------------
# Correlation Methods
# ------------------------------------------------------------------------------
CORRELATION_METHODS = {
    'basic': 'Time-domain cross-correlation (scipy.signal.correlate)',
    'gcc_phat': 'Generalized Cross-Correlation with Phase Transform (robust to noise)',
    'frequency_domain': 'FFT-based correlation for efficiency',
    'adaptive': 'Adaptive filtering with pre-whitening',
}

DEFAULT_CORRELATION_METHOD = 'gcc_phat'

# ------------------------------------------------------------------------------
# Signal Processing Parameters
# ------------------------------------------------------------------------------

# Bandpass filter (reduce noise outside leak frequency range)
# Typical leak frequencies: 100-1500 Hz for metallic pipes, 50-800 Hz for plastic
BANDPASS_LOW_HZ = 100           # Low frequency cutoff
BANDPASS_HIGH_HZ = 1500         # High frequency cutoff
BANDPASS_ORDER = 4              # Butterworth filter order

# Correlation window (can use subset of 10s recording for efficiency)
CORRELATION_WINDOW_SEC = None   # None = use full recording, or specify seconds (e.g., 5.0)

# Subsample interpolation method
INTERPOLATION_METHOD = 'parabolic'  # Options: 'parabolic', 'gaussian', 'sinc'

# Zero-padding factor for frequency-domain correlation
ZERO_PAD_FACTOR = 2             # 2x zero-padding for better frequency resolution

# ------------------------------------------------------------------------------
# Time Delay Estimation
# ------------------------------------------------------------------------------

# Maximum expected time delay (based on max sensor separation and min wave speed)
# Example: 1000m separation / 400 m/s (PVC) = 2.5 seconds
MAX_TIME_DELAY_SEC = 5.0        # Maximum physical time delay to search

# Minimum correlation peak height (normalized 0-1)
MIN_PEAK_HEIGHT = 0.3           # Reject peaks below this threshold

# Peak sharpness threshold (ratio of main peak to next highest peak)
MIN_PEAK_SHARPNESS = 1.5        # Main peak must be 1.5x higher than next peak

# Number of peaks to detect for multi-path analysis
MAX_PEAKS_TO_DETECT = 5         # Detect up to 5 correlation peaks

# ------------------------------------------------------------------------------
# Quality Control Thresholds
# ------------------------------------------------------------------------------

# Minimum SNR for acceptable correlation (dB)
MIN_SNR_DB = 10.0               # Signal-to-noise ratio threshold

# Minimum confidence score (0-1)
MIN_CONFIDENCE = 0.7            # Reject results below this confidence

# Maximum time synchronization drift allowed between sensors (seconds)
MAX_TIME_SYNC_DRIFT_SEC = 5.0   # 5 seconds max drift (based on timestamp parsing)

# Physical constraint: calculated distance must be within sensor separation
DISTANCE_TOLERANCE_METERS = 5.0 # Allow ±5m tolerance for measurement errors

# ------------------------------------------------------------------------------
# Sensor Configuration
# ------------------------------------------------------------------------------

# Sensor registry file location (can be overridden via CLI)
DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__),
    'examples',
    'example_sensor_registry.json'
)

# Minimum sensor separation for reliable correlation (meters)
MIN_SENSOR_SEPARATION_M = 10.0  # Below this, time delay too small to measure

# Maximum sensor separation (meters) - beyond this, signal too attenuated
MAX_SENSOR_SEPARATION_M = 1000.0

# ------------------------------------------------------------------------------
# GPU Acceleration Settings
# ------------------------------------------------------------------------------

# Use GPU for correlation computation (if available)
USE_GPU_CORRELATION = True

# Batch size for processing multiple sensor pairs
CORRELATION_BATCH_SIZE = 32     # Process 32 pairs at once

# Precision for GPU computation
CORRELATION_PRECISION = 'float32'  # Options: 'float32', 'float16'

# Number of CUDA streams for async processing
CORRELATION_CUDA_STREAMS = 8

# ------------------------------------------------------------------------------
# Multi-Sensor Triangulation
# ------------------------------------------------------------------------------

# Minimum number of sensor pairs for triangulation
MIN_PAIRS_FOR_TRIANGULATION = 3

# Weighting method for combining multiple estimates
TRIANGULATION_WEIGHT_METHOD = 'confidence'  # Options: 'confidence', 'inverse_distance', 'uniform'

# Maximum distance discrepancy between pairs for valid triangulation (meters)
MAX_TRIANGULATION_DISCREPANCY_M = 50.0

# ------------------------------------------------------------------------------
# Batch Processing
# ------------------------------------------------------------------------------

# Time window for matching recordings from different sensors (seconds)
BATCH_TIME_WINDOW_SEC = 60      # Match recordings within 60 seconds

# Maximum number of pairs to process in batch mode
MAX_BATCH_PAIRS = 1000

# Parallel workers for batch processing
BATCH_WORKERS = 4               # CPU threads for parallel processing

# ------------------------------------------------------------------------------
# Output Configuration
# ------------------------------------------------------------------------------

# Default output format
OUTPUT_FORMAT = 'json'          # Options: 'json', 'csv', 'xml'

# Include visualization plots in output
GENERATE_PLOTS = False          # Set to True to generate correlation plots

# Plot format
PLOT_FORMAT = 'png'             # Options: 'png', 'svg', 'pdf'

# Plot DPI
PLOT_DPI = 150

# Verbose output level
VERBOSE_LEVEL = 1               # 0=quiet, 1=normal, 2=verbose, 3=debug

# ------------------------------------------------------------------------------
# File System Paths
# ------------------------------------------------------------------------------

# Default output directory for correlation results
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    'DATA_STORE',
    'PROC_REPORTS',
    'CORRELATIONS'
)

# Cache directory for intermediate results
CORRELATION_CACHE_DIR = os.path.join(
    ROOT_DIR,
    'DATA_STORE',
    'PROC_CACHE',
    'CORRELATIONS'
)

# Log directory
CORRELATION_LOG_DIR = os.path.join(
    ROOT_DIR,
    'DATA_STORE',
    'PROC_LOGS',
    'CORRELATIONS'
)

# Create directories if they don't exist
for directory in [DEFAULT_OUTPUT_DIR, CORRELATION_CACHE_DIR, CORRELATION_LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ------------------------------------------------------------------------------
# Validation Constants
# ------------------------------------------------------------------------------

# Expected sample rate (must match AILH system)
EXPECTED_SAMPLE_RATE = SAMPLE_RATE  # 4096 Hz from global_config

# Expected duration (must match AILH system)
EXPECTED_DURATION_SEC = SAMPLE_LENGTH_SEC  # 10 seconds from global_config

# Expected file naming delimiter
EXPECTED_DELIMITER = DELIMITER  # '~' from global_config

# ------------------------------------------------------------------------------
# Advanced Correlation Parameters
# ------------------------------------------------------------------------------

# GCC-PHAT parameters
PHAT_EPSILON = 1e-10            # Small value to avoid division by zero

# Adaptive filter parameters
ADAPTIVE_FILTER_ORDER = 32      # Number of adaptive filter taps
ADAPTIVE_MU = 0.01              # LMS step size

# Pre-whitening (spectral flattening) parameters
PREWHITEN_ENABLE = True         # Enable pre-whitening
PREWHITEN_ALPHA = 0.9           # Pre-emphasis filter coefficient

# Envelope detection for improved correlation
USE_ENVELOPE_CORRELATION = False # Correlate signal envelopes instead of raw signals

# ------------------------------------------------------------------------------
# Debugging and Profiling
# ------------------------------------------------------------------------------

# Enable detailed timing profiling
ENABLE_PROFILING = False

# Save intermediate correlation functions for debugging
SAVE_CORRELATION_FUNCTIONS = False

# Correlation function save directory
CORRELATION_DEBUG_DIR = os.path.join(
    CORRELATION_CACHE_DIR,
    'debug'
)

if SAVE_CORRELATION_FUNCTIONS:
    os.makedirs(CORRELATION_DEBUG_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# GPS and Coordinate Systems
# ------------------------------------------------------------------------------

# Default coordinate system
COORDINATE_SYSTEM = 'WGS84'     # GPS coordinates

# Altitude reference
ALTITUDE_REFERENCE = 'MSL'      # Mean Sea Level

# Distance calculation method for GPS coordinates
GPS_DISTANCE_METHOD = 'haversine'  # Options: 'haversine', 'vincenty'

# ------------------------------------------------------------------------------
# Reporting Configuration
# ------------------------------------------------------------------------------

# Decimal precision for distance measurements (meters)
DISTANCE_PRECISION_DECIMALS = 2

# Decimal precision for time delays (seconds)
TIME_DELAY_PRECISION_DECIMALS = 6

# Decimal precision for confidence scores
CONFIDENCE_PRECISION_DECIMALS = 3

# Include raw correlation data in output
INCLUDE_RAW_CORRELATION = False

# Include signal quality metrics in output
INCLUDE_QUALITY_METRICS = True

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_wave_speed(pipe_material):
    """
    Get wave speed for a given pipe material.

    Args:
        pipe_material (str): Pipe material name

    Returns:
        float: Wave speed in meters per second

    Raises:
        ValueError: If pipe material not recognized
    """
    material = pipe_material.lower().replace(' ', '_').replace('-', '_')

    if material in WAVE_SPEEDS:
        return WAVE_SPEEDS[material]
    else:
        available = ', '.join(WAVE_SPEEDS.keys())
        raise ValueError(
            f"Unknown pipe material '{pipe_material}'. "
            f"Available materials: {available}"
        )


def validate_sensor_separation(distance_m):
    """
    Validate that sensor separation is within acceptable range.

    Args:
        distance_m (float): Distance between sensors in meters

    Returns:
        tuple: (is_valid, message)
    """
    if distance_m < MIN_SENSOR_SEPARATION_M:
        return False, f"Sensor separation ({distance_m}m) is below minimum ({MIN_SENSOR_SEPARATION_M}m)"

    if distance_m > MAX_SENSOR_SEPARATION_M:
        return False, f"Sensor separation ({distance_m}m) exceeds maximum ({MAX_SENSOR_SEPARATION_M}m)"

    return True, "Sensor separation is valid"


def get_max_time_delay(distance_m, wave_speed_mps):
    """
    Calculate maximum possible time delay for a given sensor separation.

    Args:
        distance_m (float): Distance between sensors in meters
        wave_speed_mps (float): Wave speed in meters per second

    Returns:
        float: Maximum time delay in seconds
    """
    return distance_m / wave_speed_mps


def samples_to_seconds(samples, sample_rate=SAMPLE_RATE):
    """
    Convert samples to seconds.

    Args:
        samples (int): Number of samples
        sample_rate (int): Sample rate in Hz

    Returns:
        float: Time in seconds
    """
    return samples / sample_rate


def seconds_to_samples(seconds, sample_rate=SAMPLE_RATE):
    """
    Convert seconds to samples.

    Args:
        seconds (float): Time in seconds
        sample_rate (int): Sample rate in Hz

    Returns:
        int: Number of samples
    """
    return int(seconds * sample_rate)


def distance_to_time_delay(distance_m, wave_speed_mps):
    """
    Convert distance to time delay.

    Args:
        distance_m (float): Distance in meters
        wave_speed_mps (float): Wave speed in meters per second

    Returns:
        float: Time delay in seconds
    """
    return distance_m / wave_speed_mps


def time_delay_to_distance(time_delay_sec, wave_speed_mps):
    """
    Convert time delay to distance.

    Args:
        time_delay_sec (float): Time delay in seconds
        wave_speed_mps (float): Wave speed in meters per second

    Returns:
        float: Distance in meters
    """
    return time_delay_sec * wave_speed_mps


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_configuration():
    """
    Validate configuration parameters.

    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Validate sample rate
    if EXPECTED_SAMPLE_RATE != SAMPLE_RATE:
        raise ValueError(
            f"EXPECTED_SAMPLE_RATE ({EXPECTED_SAMPLE_RATE}) does not match "
            f"SAMPLE_RATE from global_config ({SAMPLE_RATE})"
        )

    # Validate correlation method
    if DEFAULT_CORRELATION_METHOD not in CORRELATION_METHODS:
        raise ValueError(
            f"DEFAULT_CORRELATION_METHOD '{DEFAULT_CORRELATION_METHOD}' "
            f"is not in CORRELATION_METHODS"
        )

    # Validate bandpass filter
    if BANDPASS_LOW_HZ >= BANDPASS_HIGH_HZ:
        raise ValueError(
            f"BANDPASS_LOW_HZ ({BANDPASS_LOW_HZ}) must be less than "
            f"BANDPASS_HIGH_HZ ({BANDPASS_HIGH_HZ})"
        )

    if BANDPASS_HIGH_HZ > SAMPLE_RATE / 2:
        raise ValueError(
            f"BANDPASS_HIGH_HZ ({BANDPASS_HIGH_HZ}) exceeds Nyquist frequency "
            f"({SAMPLE_RATE / 2})"
        )

    # Validate confidence threshold
    if not 0 <= MIN_CONFIDENCE <= 1:
        raise ValueError(f"MIN_CONFIDENCE must be between 0 and 1")

    # Validate directories exist
    if not os.path.exists(ROOT_DIR):
        raise ValueError(f"ROOT_DIR directory does not exist: {ROOT_DIR}")

    print("[✓] Configuration validated successfully")


# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

if __name__ == '__main__':
    # If run as script, validate configuration and print summary
    print("=" * 80)
    print("LEAK DETECTION CORRELATOR - CONFIGURATION")
    print("=" * 80)

    try:
        validate_configuration()

        print(f"\n[i] Root Directory: {ROOT_DIR}")
        print(f"[i] Sample Rate: {SAMPLE_RATE} Hz")
        print(f"[i] Recording Duration: {SAMPLE_LENGTH_SEC} seconds")
        print(f"\n[i] Correlation Method: {DEFAULT_CORRELATION_METHOD}")
        print(f"[i] Default Pipe Material: {DEFAULT_PIPE_MATERIAL}")
        print(f"[i] Default Wave Speed: {DEFAULT_WAVE_SPEED} m/s")
        print(f"\n[i] Bandpass Filter: {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz")
        print(f"[i] Min Confidence: {MIN_CONFIDENCE}")
        print(f"[i] Max Time Delay: {MAX_TIME_DELAY_SEC} seconds")

        print(f"\n[i] Output Directory: {DEFAULT_OUTPUT_DIR}")
        print(f"[i] Cache Directory: {CORRELATION_CACHE_DIR}")
        print(f"[i] Log Directory: {CORRELATION_LOG_DIR}")

        print(f"\n[i] GPU Acceleration: {'Enabled' if USE_GPU_CORRELATION else 'Disabled'}")
        print(f"[i] Batch Size: {CORRELATION_BATCH_SIZE}")
        print(f"[i] Precision: {CORRELATION_PRECISION}")

        print("\n[i] Available Pipe Materials:")
        for material, speed in sorted(WAVE_SPEEDS.items()):
            print(f"    - {material:20s}: {speed:5d} m/s")

        print("\n[✓] Configuration loaded successfully")

    except Exception as e:
        print(f"\n[✗] Configuration error: {e}")
        sys.exit(1)
