#!/usr/bin/env python3
"""
Leak Detection Correlator - Utilities Module

Utility functions for WAV file handling, filename parsing, and data loading.

Author: AILH Development Team
Date: 2025-11-19
Version: 2.0.0
"""

import os
import numpy as np
import wave
from typing import Tuple, Dict, Optional
from datetime import datetime

from correlator_config import *


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse AILH WAV filename to extract metadata.

    Expected format: sensor_id~recording_id~timestamp~gain_db.wav

    Args:
        filename (str): Filename (with or without path)

    Returns:
        dict: Parsed metadata {
            'sensor_id': str,
            'recording_id': str,
            'timestamp': str,
            'gain_db': str,
            'filepath': str
        }

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Extract basename
    basename = os.path.basename(filename)

    # Remove .wav extension
    if not basename.lower().endswith('.wav'):
        raise ValueError(f"File is not a WAV file: {basename}")

    name_without_ext = basename[:-4]

    # Split by delimiter
    parts = name_without_ext.split(DELIMITER)

    if len(parts) != 4:
        raise ValueError(
            f"Filename '{basename}' does not match expected format: "
            f"sensor_id{DELIMITER}recording_id{DELIMITER}timestamp{DELIMITER}gain_db.wav"
        )

    sensor_id, recording_id, timestamp, gain_db = parts

    return {
        'sensor_id': sensor_id,
        'recording_id': recording_id,
        'timestamp': timestamp,
        'gain_db': gain_db,
        'filepath': filename
    }


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse timestamp string from filename.

    Expected format: YYYYMMDDHHMMSS or YYYY-MM-DD_HH-MM-SS

    Args:
        timestamp_str (str): Timestamp string

    Returns:
        datetime: Parsed timestamp

    Raises:
        ValueError: If timestamp format is invalid
    """
    # Try different formats
    formats = [
        '%Y%m%d%H%M%S',           # 20250118120530
        '%Y-%m-%d_%H-%M-%S',       # 2025-01-18_12-05-30
        '%Y%m%d_%H%M%S',           # 20250118_120530
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse timestamp: {timestamp_str}")


def check_time_sync(
    timestamp_a: str,
    timestamp_b: str,
    max_drift_sec: float = MAX_TIME_SYNC_DRIFT_SEC
) -> Tuple[bool, float]:
    """
    Check if two recordings are synchronized within tolerance.

    Args:
        timestamp_a (str): First timestamp string
        timestamp_b (str): Second timestamp string
        max_drift_sec (float): Maximum allowed drift in seconds

    Returns:
        Tuple of (is_synchronized, drift_seconds)
    """
    try:
        dt_a = parse_timestamp(timestamp_a)
        dt_b = parse_timestamp(timestamp_b)

        drift = abs((dt_a - dt_b).total_seconds())

        is_sync = drift <= max_drift_sec

        return is_sync, drift

    except ValueError as e:
        # Could not parse timestamps
        return False, float('inf')


def load_wav(
    filepath: str,
    expected_sample_rate: int = SAMPLE_RATE,
    expected_duration_sec: Optional[float] = SAMPLE_LENGTH_SEC,
    validate: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load WAV file and return audio data.

    Args:
        filepath (str): Path to WAV file
        expected_sample_rate (int): Expected sample rate (for validation)
        expected_duration_sec (float, optional): Expected duration (for validation)
        validate (bool): Validate sample rate and duration

    Returns:
        Tuple of (audio_data, sample_rate):
            audio_data (np.ndarray): Audio samples (float32, normalized to [-1, 1])
            sample_rate (int): Sample rate in Hz

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WAV file not found: {filepath}")

    try:
        # Open WAV file
        with wave.open(filepath, 'rb') as wav_file:
            # Get parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # Read frames
            frames = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 1:  # 8-bit
            dtype = np.uint8
            audio = np.frombuffer(frames, dtype=dtype)
            audio = (audio.astype(np.float32) - 128) / 128.0  # Normalize to [-1, 1]
        elif sample_width == 2:  # 16-bit
            dtype = np.int16
            audio = np.frombuffer(frames, dtype=dtype)
            audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        elif sample_width == 3:  # 24-bit
            # 24-bit is tricky, need to handle byte order
            audio = np.frombuffer(frames, dtype=np.uint8)
            audio = audio.reshape(-1, 3)
            # Convert to 32-bit
            audio_32 = np.zeros(len(audio), dtype=np.int32)
            audio_32 = (audio[:, 0].astype(np.int32) << 8) | \
                       (audio[:, 1].astype(np.int32) << 16) | \
                       (audio[:, 2].astype(np.int32) << 24)
            audio = audio_32.astype(np.float32) / 2147483648.0  # Normalize
        elif sample_width == 4:  # 32-bit
            dtype = np.int32
            audio = np.frombuffer(frames, dtype=dtype)
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        # Handle multi-channel (take first channel only)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio[:, 0]  # Take first channel

        # Validation
        if validate:
            if sample_rate != expected_sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: expected {expected_sample_rate} Hz, "
                    f"got {sample_rate} Hz"
                )

            if expected_duration_sec is not None:
                actual_duration = len(audio) / sample_rate
                if abs(actual_duration - expected_duration_sec) > 0.1:  # 100ms tolerance
                    raise ValueError(
                        f"Duration mismatch: expected {expected_duration_sec}s, "
                        f"got {actual_duration:.2f}s"
                    )

        return audio, sample_rate

    except wave.Error as e:
        raise ValueError(f"Error reading WAV file: {e}")


def validate_wav_pair(
    filepath_a: str,
    filepath_b: str,
    check_sync: bool = True,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Validate a pair of WAV files for correlation.

    Args:
        filepath_a (str): First WAV file
        filepath_b (str): Second WAV file
        check_sync (bool): Check time synchronization
        verbose (bool): Print validation details

    Returns:
        Tuple of (is_valid, message)
    """
    errors = []

    # Parse filenames
    try:
        meta_a = parse_filename(filepath_a)
        meta_b = parse_filename(filepath_b)
    except ValueError as e:
        return False, f"Filename parsing failed: {e}"

    # Check different sensors
    if meta_a['sensor_id'] == meta_b['sensor_id']:
        errors.append("Both files are from the same sensor")

    # Check time synchronization
    if check_sync:
        is_sync, drift = check_time_sync(meta_a['timestamp'], meta_b['timestamp'])

        if not is_sync:
            errors.append(
                f"Time drift ({drift:.1f}s) exceeds maximum ({MAX_TIME_SYNC_DRIFT_SEC}s)"
            )

        if verbose and is_sync:
            print(f"[✓] Time drift: {drift:.3f}s (within tolerance)")

    # Load and validate files
    try:
        audio_a, sr_a = load_wav(filepath_a, validate=True)
        audio_b, sr_b = load_wav(filepath_b, validate=True)

        if verbose:
            print(f"[✓] File A: {len(audio_a)} samples @ {sr_a} Hz")
            print(f"[✓] File B: {len(audio_b)} samples @ {sr_b} Hz")

    except Exception as e:
        errors.append(f"WAV loading failed: {e}")

    if len(errors) > 0:
        return False, "; ".join(errors)
    else:
        return True, "Pair valid"


def find_wav_files(
    directory: str,
    sensor_id: Optional[str] = None,
    recursive: bool = True
) -> list:
    """
    Find all WAV files in a directory.

    Args:
        directory (str): Directory to search
        sensor_id (str, optional): Filter by sensor ID
        recursive (bool): Search subdirectories

    Returns:
        list: List of WAV file paths
    """
    wav_files = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.wav'):
                    filepath = os.path.join(root, file)

                    # Filter by sensor ID if specified
                    if sensor_id:
                        try:
                            meta = parse_filename(file)
                            if meta['sensor_id'] == sensor_id:
                                wav_files.append(filepath)
                        except ValueError:
                            # Skip files that don't match naming convention
                            continue
                    else:
                        wav_files.append(filepath)
    else:
        for file in os.listdir(directory):
            if file.lower().endswith('.wav'):
                filepath = os.path.join(directory, file)

                if sensor_id:
                    try:
                        meta = parse_filename(file)
                        if meta['sensor_id'] == sensor_id:
                            wav_files.append(filepath)
                    except ValueError:
                        continue
                else:
                    wav_files.append(filepath)

    return sorted(wav_files)


def match_recording_pairs(
    directory: str,
    sensor_a: str,
    sensor_b: str,
    time_window_sec: float = BATCH_TIME_WINDOW_SEC
) -> list:
    """
    Find pairs of synchronized recordings from two sensors.

    Args:
        directory (str): Directory containing WAV files
        sensor_a (str): First sensor ID
        sensor_b (str): Second sensor ID
        time_window_sec (float): Maximum time difference for pairing (seconds)

    Returns:
        list: List of tuples (filepath_a, filepath_b, time_drift)
    """
    # Find files for each sensor
    files_a = find_wav_files(directory, sensor_id=sensor_a)
    files_b = find_wav_files(directory, sensor_id=sensor_b)

    # Parse timestamps
    meta_a = [parse_filename(f) for f in files_a]
    meta_b = [parse_filename(f) for f in files_b]

    # Match pairs
    pairs = []

    for i, ma in enumerate(meta_a):
        try:
            ts_a = parse_timestamp(ma['timestamp'])
        except ValueError:
            continue

        for j, mb in enumerate(meta_b):
            try:
                ts_b = parse_timestamp(mb['timestamp'])
            except ValueError:
                continue

            # Check time difference
            drift = abs((ts_a - ts_b).total_seconds())

            if drift <= time_window_sec:
                pairs.append((files_a[i], files_b[j], drift))

    # Sort by drift (best matches first)
    pairs.sort(key=lambda x: x[2])

    return pairs


def compute_signal_quality(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Compute signal quality metrics.

    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate in Hz

    Returns:
        dict: Quality metrics {
            'rms': RMS level,
            'peak': Peak level,
            'snr_estimate_db': Estimated SNR,
            'clipping_detected': bool,
            'silence_detected': bool
        }
    """
    # RMS level
    rms = np.sqrt(np.mean(audio ** 2))

    # Peak level
    peak = np.max(np.abs(audio))

    # Estimate SNR (simple method: ratio of signal power to noise floor)
    # Assume lowest 10% of power spectrum is noise
    fft = np.fft.rfft(audio)
    power_spectrum = np.abs(fft) ** 2
    sorted_power = np.sort(power_spectrum)
    noise_floor = np.mean(sorted_power[:len(sorted_power)//10])
    signal_power = np.mean(power_spectrum)

    if noise_floor > 0:
        snr_estimate_db = 10 * np.log10(signal_power / noise_floor)
    else:
        snr_estimate_db = float('inf')

    # Clipping detection (signal near ±1.0)
    clipping_threshold = 0.99
    clipping_detected = np.any(np.abs(audio) > clipping_threshold)

    # Silence detection (very low RMS)
    silence_threshold = 0.001
    silence_detected = rms < silence_threshold

    return {
        'rms': float(rms),
        'peak': float(peak),
        'snr_estimate_db': float(snr_estimate_db),
        'clipping_detected': bool(clipping_detected),
        'silence_detected': bool(silence_detected)
    }


# ==============================================================================
# MAIN - Example Usage and Testing
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Correlator Utilities Testing')
    parser.add_argument('--test-parsing', action='store_true',
                       help='Test filename parsing')
    parser.add_argument('--test-load', help='Test loading WAV file')
    parser.add_argument('--test-matching', help='Test finding recording pairs in directory')
    parser.add_argument('--sensor-a', help='First sensor ID for matching')
    parser.add_argument('--sensor-b', help='Second sensor ID for matching')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("CORRELATOR UTILITIES TEST")
    print("=" * 80)

    if args.test_parsing:
        print("\n[i] Testing filename parsing...")

        test_filenames = [
            'S001~R123~20250118120530~100.wav',
            '12345~67890~2025-01-18_12-05-30~95.wav',
            'SENSOR_A~REC001~20250118_120530~110.wav',
        ]

        for filename in test_filenames:
            try:
                meta = parse_filename(filename)
                print(f"\n[✓] Parsed: {filename}")
                for key, value in meta.items():
                    if key != 'filepath':
                        print(f"    {key}: {value}")

                # Try parsing timestamp
                try:
                    ts = parse_timestamp(meta['timestamp'])
                    print(f"    Parsed timestamp: {ts}")
                except ValueError as e:
                    print(f"    [!] Timestamp parsing failed: {e}")

            except ValueError as e:
                print(f"\n[✗] Failed: {filename}")
                print(f"    Error: {e}")

    if args.test_load:
        print(f"\n[i] Testing WAV loading: {args.test_load}")

        try:
            audio, sr = load_wav(args.test_load, validate=False)
            print(f"[✓] Loaded successfully")
            print(f"    Samples: {len(audio)}")
            print(f"    Sample rate: {sr} Hz")
            print(f"    Duration: {len(audio)/sr:.2f}s")
            print(f"    Range: [{np.min(audio):.4f}, {np.max(audio):.4f}]")

            # Compute quality metrics
            quality = compute_signal_quality(audio, sr)
            print(f"\n[i] Signal quality:")
            for key, value in quality.items():
                print(f"    {key}: {value}")

        except Exception as e:
            print(f"[✗] Error loading file: {e}")

    if args.test_matching and args.sensor_a and args.sensor_b:
        print(f"\n[i] Finding recording pairs in: {args.test_matching}")
        print(f"    Sensor A: {args.sensor_a}")
        print(f"    Sensor B: {args.sensor_b}")

        pairs = match_recording_pairs(
            args.test_matching,
            args.sensor_a,
            args.sensor_b
        )

        print(f"\n[✓] Found {len(pairs)} matching pairs:")

        for i, (file_a, file_b, drift) in enumerate(pairs[:10]):  # Show first 10
            print(f"\n  Pair {i+1}:")
            print(f"    File A: {os.path.basename(file_a)}")
            print(f"    File B: {os.path.basename(file_b)}")
            print(f"    Time drift: {drift:.3f}s")

        if len(pairs) > 10:
            print(f"\n  ... and {len(pairs) - 10} more pairs")

    print("\n[✓] Test complete")
