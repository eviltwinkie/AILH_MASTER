# ailh_stacker.py
import os
import sys
import argparse
import asyncio
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, get_window
from datetime import datetime
import matplotlib.pyplot as plt
from signal_utils import *
from signal_plots import plot_stack_report
from signal_processing import hybrid_autocorr_cleaning
from AILH_UTILS.wav_viewer import analyze_wav_to_svg
from wavelet_utils import *

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import SENSOR_DATA
from fileio import sanitize_path_component, create_file_path


def list_wav_files(folder):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.wav')]
    print(f"[✓] Found {len(all_files)} WAV files in {folder}")
    return sorted(all_files)

def extract_info_from_filename(filename):
    base = os.path.basename(filename)  # Strip directory path
    name_parts = base.replace(".wav", "").split("~")
    if len(name_parts) < 4:
        raise ValueError(f"[✗] Invalid filename format: {filename}")
    sensor_id = name_parts[0]
    recording_id = name_parts[1]
    timestamp = name_parts[2]
    gain_db = float(name_parts[3])
    #print(f"[✓] Extracted from filename: sensorId={sensor_id} recordingId={recording_id} timestamp={timestamp} gain_db={gain_db}")
    return sensor_id, recording_id, timestamp, gain_db

def valid_wav_files(folder, output_file):
    output_abspath = os.path.abspath(output_file)
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.wav')]
    print(f"[✓] Found {len(all_files)} WAV files in {folder}")
    file_data = []
    file_path = []
    file_rate = []
    for filename in sorted(all_files):

        if "STACK" in filename:
            #print(f"[!] Skipping output or non-WAV file: {filename}")
            continue
        if os.path.abspath(filename) == output_abspath:
            #print(f"[!] Skipping output file itself to avoid self-stacking: {filename}")
            continue
        sensor_id, recording_id, recording_timestamp, gain_db = extract_info_from_filename(filename)
        if gain_db == 1:
            #print(f"[!] Skipping file with gain == 1: {filename}")
            continue

        try:
            data, rate = sf.read(filename)
            data = stereo_to_mono(data)
            data = preprocess_signal(data, gain_db)
            file_data.append(data)
            file_path.append(filename)
            file_rate.append(rate)
        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")
            continue
    return file_data, file_path, file_rate

# def filter_by_spectral_flatness(raw_file_data, raw_file_path, raw_file_rate, threshold=0.055):
#     filtered_file_data = []
#     filtered_file_path = []
#     filtered_file_rate = []
#     for data, path, samplerate in zip(raw_file_data, raw_file_path, raw_file_rate):
#         #data = rms_normalize(data)
#         #data = normalize_signal(data)
#         f, t, Sxx = spectrogram(rms_normalize(data), fs=samplerate)
#         spectral_flatness = np.exp(np.mean(np.log(Sxx + 1e-12))) / (np.mean(Sxx + 1e-12) + 1e-12)
#         #print(f"[Spectral Flatness] {path} - Spectral Flatness: {spectral_flatness:.4f} (threshold: {threshold:.4f})")
#         if spectral_flatness < threshold:
#             #print(f"[✓] Keeping {path} — Spectral Flatness {spectral_flatness:.4f} below threshold {threshold:.4f}")
#             filtered_file_data.append(data)
#             filtered_file_path.append(path)
#             filtered_file_rate.append(samplerate)
#             #plt.figure()
#             #plt.plot(data)
#             #plt.show()
#         #else:
#         #    print(f"[!] Skipping {path} — Spectral Flatness {spectral_flatness:.4f} below threshold {threshold:.4f}")
#     print(f"[✓] [Filtering - Spectral Flatness] {len(filtered_file_data)} of {len(raw_file_data)} files passed")
#     return filtered_file_data, filtered_file_path, filtered_file_rate

def spectral_flatness_full(x, samplerate):
    """
    Compute spectral flatness over the entire signal.
    Args:
        x (np.ndarray): 1D signal
        samplerate (float): Sample rate in Hz
    Returns:
        flatness (float)
    """
    x = rms_normalize(x)
    x = x - np.mean(x)
    n_fft = len(x)
    window = get_window('hann', n_fft)
    x_win = x * window
    spectrum = np.abs(np.fft.rfft(x_win))**2
    eps = 1e-12
    flatness = np.exp(np.mean(np.log(spectrum + eps))) / (np.mean(spectrum + eps))
    return flatness

def spectral_flatness_segmented(x, samplerate, n_fft=1024, hop_length=256):
    """
    Compute the median spectral flatness across frames (segmenting the signal).
    Args:
        x (np.ndarray): 1D signal
        samplerate (float): Sample rate in Hz
        n_fft (int): FFT/window size
        hop_length (int): Step size between windows
    Returns:
        flatness (float): Median flatness across all frames
    """
    x = rms_normalize(x)
    x = x - np.mean(x)
    window = get_window('hann', n_fft)
    num_frames = 1 + (len(x) - n_fft) // hop_length if len(x) >= n_fft else 0
    flatness_vals = []
    eps = 1e-12

    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft]
        frame_win = frame * window
        spectrum = np.abs(np.fft.rfft(frame_win))**2
        flatness = np.exp(np.mean(np.log(spectrum + eps))) / (np.mean(spectrum + eps))
        flatness_vals.append(flatness)

    if len(flatness_vals) == 0:
        # If signal is too short for even one frame, fall back to full signal
        return spectral_flatness_full(x, samplerate)
    return np.median(flatness_vals)

def filter_by_spectral_flatness(raw_file_data, raw_file_path, raw_file_rate, threshold='auto', keep='below'):
    """
    Filter audio files by spectral flatness, with data-driven adaptive threshold.

    Args:
        raw_file_data (list of np.ndarray): Audio data arrays.
        raw_file_path (list): Corresponding file paths.
        raw_file_rate (list): Corresponding sample rates.
        threshold (float or 'auto'): Threshold for flatness; 'auto' selects adaptively.
        keep (str): 'below' to keep files with flatness below threshold (default); 'above' for above.

    Returns:
        filtered_file_data, filtered_file_path, filtered_file_rate
    """
    spectral_flatness_vals = []
    for data, samplerate in zip(raw_file_data, raw_file_rate):
        #flatness = spectral_flatness_full(data, samplerate)
        flatness = spectral_flatness_segmented(data, samplerate)
        spectral_flatness_vals.append(flatness)

    vals = np.array(spectral_flatness_vals)

    if threshold == 'auto':
        vals = np.array(spectral_flatness_vals)
        # Use the "elbow" in the distribution, or median minus MAD (robust central tendency)
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        threshold = median - mad if keep == 'below' else median + mad
        #threshold = threshold - (0.1 * threshold)
        threshold = max(0.1, float(threshold))  # Ensure non-negative threshold

    filtered_file_data = []
    filtered_file_path = []
    filtered_file_rate = []
    for data, path, samplerate, flatness in zip(raw_file_data, raw_file_path, raw_file_rate, spectral_flatness_vals):
        keep_condition = (flatness < threshold) if keep == 'below' else (flatness > threshold)
        if keep_condition:
            filtered_file_data.append(data)
            filtered_file_path.append(path)
            filtered_file_rate.append(samplerate)

    print(
        f"[✓] [Filtering - Spectral Flatness] {len(filtered_file_data)} of {len(raw_file_data)} files passed "
        f"(threshold={threshold:.4f}, keep='{keep}')"
    )
    return filtered_file_data, filtered_file_path, filtered_file_rate

def preprocess_signal(x, gain_db):

    # 1. Remove DC offset ALWAYS audio files are already normalized to [-1, 1] 
    x = x - np.mean(x)

    #for i in range(len(x)):
    #    print(f"Sample {i}: data = {x[i]}, gain = {gain_db} dB")

    #x = scaled_gain(x, gain_db)

    # 3. (Optional) RMS normalize for ML/feature extraction
    #x = rms_normalize(x) # stacked avg signal higher than stacked
    #x = normalize_signal(x) # stacked louder than the avg signal
    return x

def preprocess_wav(wav_file, wav_file_path, wav_file_rate, filter_type, lowcut, highcut):
    #print(f"[✓] Preprocessing {wav_file_path} at {wav_file_rate} Hz")
    #print("+",end='')
    sensor_id, recording_id, recording_timestamp, gain_db = extract_info_from_filename(wav_file_path)
    data = wav_file

    # Remove large artifacts first
    #data = hybrid_autocorr_cleaning(data, wav_file_rate, verbose=False, window_size=4096)

    data = wavelet_denoise(signal=data, wavelet='bior4.4')
    data = wavelet_packet_denoise(data, wavelet='db4')
   
    # Finally, filter the denoised/cleaned signal
    data = apply_filter(data, wav_file_rate, filter_type, lowcut, highcut, ftype="fir", order=4, numtaps=512)

    return data, sensor_id, recording_id, recording_timestamp

def standard_stack(signal_data):
    print("[*] Starting standard stack...")
    #print(f"[-] Input signal_data: {signal_data}")
    signal_ref = np.mean(np.vstack(signal_data), axis=0)
    #print(f"[-] Signal reference shape: {signal_ref.shape}, dtype: {signal_ref.dtype}")
    signal_snr = compute_snr(signal_data, signal_ref)    
    stack_mask = np.array([True] * len(signal_data))
    print(f"[✓] Files used {sum(stack_mask)}/{len(stack_mask)}, Output SNR: {signal_snr:.2f} dB")
    return signal_data, signal_ref, signal_snr, stack_mask

# def advanced_stack(signal_data):
#     print("[*] Starting advanced stack...")
#     #print(f"[-] Input signal_data: {signal_data}")
#     stack_ref = np.mean(np.vstack(signal_data), axis=0)
#     #print(f"[-] Stack reference shape: {stack_ref.shape}, dtype: {stack_ref.dtype}")
#     signal_data_snr = compute_snr(signal_data, stack_ref)    
#     print(f"[-] Input SNR: {signal_data_snr:.2f} dB")
#     stack_mask, mean_corrs, threshold = correlation_matrix_outlier_rejection(signal_data, mode="auto")
#     if threshold is None:
#         threshold_str = "None"
#     else:
#         threshold_str = f"{threshold:.3f}"
#     print(f"[✓] Auto-correlation threshold: {threshold_str} | {stack_mask.sum()} of {len(stack_mask)} signals kept")
#     print(f"[✓] Mean correlations: {mean_corrs.mean():.3f} (std: {mean_corrs.std():.3f})")
#     used_data_list = [s for s, keep in zip(signal_data, stack_mask) if keep]
#     if len(used_data_list) == 0:
#         print("[✗] No signals passed outlier rejection. Returning zeros.")
#         # Return dummy arrays of the correct shape
#         dummy_shape = signal_data[0].shape if len(signal_data) > 0 else (0,)
#         zero_stack = np.zeros(dummy_shape)
#         return [], zero_stack, 0.0, stack_mask
#     used_stack_ref = np.mean(np.vstack(used_data_list), axis=0)
#     stack_snr = compute_snr(used_data_list, used_stack_ref)
#     print(f"[✓] Files used {sum(stack_mask)}/{len(stack_mask)}, Input SNR: {signal_data_snr:.2f} dB, Output SNR: {stack_snr:.2f} dB")
#     return used_data_list, used_stack_ref, stack_snr, stack_mask

import numpy as np

def advanced_stack(signal_data):
    print("[*] Starting advanced stack...")

    if len(signal_data) == 0:
        print("[✗] No input signals. Returning zeros.")
        return [], np.array([]), 0.0, np.array([])

    stack_ref = np.mean(np.vstack(signal_data), axis=0)
    signal_data_snr = compute_snr(signal_data, stack_ref)
    print(f"[-] Input SNR: {signal_data_snr:.2f} dB")

    # Outlier rejection by correlation matrix
    stack_mask, mean_corrs, threshold = correlation_matrix_outlier_rejection(signal_data, mode="auto")

    # Debugging: print the correlations and check for NaN
    #print(f"[-] Mean correlations array: {mean_corrs}")
    if np.any(np.isnan(mean_corrs)):
        print("[!] Warning: NaN detected in mean correlations!")
    if np.all(np.isnan(mean_corrs)):
        print("[!] All mean correlations are NaN -- check if your signals are constant or zero!")

    if threshold is None:
        threshold_str = "None"
    else:
        threshold_str = f"{threshold:.3f}"
    print(f"[✓] Auto-correlation threshold: {threshold_str} | {stack_mask.sum()} of {len(stack_mask)} signals kept")
    print(f"[✓] Mean correlations: {np.nanmean(mean_corrs):.3f} (std: {np.nanstd(mean_corrs):.3f})")

    used_data_list = [s for s, keep in zip(signal_data, stack_mask) if keep]
    if len(used_data_list) == 0:
        print("[✗] No signals passed outlier rejection. Returning zeros.")
        dummy_shape = signal_data[0].shape if len(signal_data) > 0 else (0,)
        zero_stack = np.zeros(dummy_shape)
        return [], zero_stack, 0.0, stack_mask

    used_stack_ref = np.mean(np.vstack(used_data_list), axis=0)
    stack_snr = compute_snr(used_data_list, used_stack_ref)
    print(f"[✓] Files used {sum(stack_mask)}/{len(stack_mask)}, Input SNR: {signal_data_snr:.2f} dB, Output SNR: {stack_snr:.2f} dB")
    return used_data_list, used_stack_ref, stack_snr, stack_mask


def stack_wav_files(sensor_id, sensor_name, sensor_site_id, sensor_site_name, sensor_station, filter_type='bandpass', lowcut=1, highcut=2047): 
    site_dir = os.path.join(SENSOR_DATA, f"{sensor_site_id}_{sensor_site_name}_{sensor_station}")
    sensor_dir = os.path.join(site_dir, f"{sensor_name}_{sensor_id}")
    file_path = create_file_path(sensor_dir, f"{sensor_id}~STACK.wav")
    if not os.path.isdir(sensor_dir):
        print(f"[✗] Sensor directory does not exist: {sensor_dir}")
        return None
    raw_file_data, raw_file_path, raw_file_rate = valid_wav_files(sensor_dir, file_path)
    if not raw_file_path:
        print(f"[!] No WAV files to process in {sensor_dir}")
        return None
    else:
        print(f"[✓] Processing {len(raw_file_path)} WAV files for stacking...")
        wav_files, wav_files_path, wav_files_rate = filter_by_spectral_flatness(raw_file_data, raw_file_path, raw_file_rate)
        signals_data = []
        signals_path = []
        sensor_id, recording_id, recording_timestamp, samplerate = None, None, None, None
        print(f"[✓] Preprocessing {len(wav_files_path)} files with filter_type={filter_type}, lowcut={lowcut}, highcut={highcut}", end=' ')
        for wav_file, wav_file_path, wav_file_rate in zip(wav_files, wav_files_path, wav_files_rate):
            signal_data, sensor_id, recording_id, recording_timestamp = preprocess_wav(wav_file, wav_file_path, wav_file_rate, filter_type, lowcut, highcut)
            signals_data.append(signal_data)
            signals_path.append(wav_file_path)
            samplerate = wav_file_rate
        print(f"\n[✓] Processed {len(wav_files_path)} files, sensor_id={sensor_id}, recording_id={recording_id}, recording_timestamp={recording_timestamp}, filter_type={filter_type}, lowcut={lowcut}, highcut={highcut}")

        # === Simple STACK (average)
        signals, stacked, stack_snr, stack_mask = standard_stack(signals_data)

        # === Advanced STACK (Outlier rejection)
#        signals, stacked, stack_snr, stack_mask = advanced_stack(signals)
   
        # === Correlation-weighted denoising
#        stacked, weights, corrs = correlation_weighted_denoising(signals, stacked)
#        print(f"[X] Correlation-weighted denoising SNR: {compute_snr(signals, stacked):.2f} dB")

        # === Correlation-weighted stacking
#        stacked = correlation_weighted_stack(signals, stacked)
#        print(f"[X] Correlation-weighted stacking SNR: {compute_snr(signals, stacked):.2f} dB")

        # === Optional: Normalize to -1..1
        stacked = normalize_signal(stacked)        

        plot_stack_report(stacked, signals, wav_files, samplerate, stack_mask, sensor_dir, sensor_id)
        #analyze_wav_to_svg(file_path, sensor_id=sensor_id, recording_id="STACK", recording_timestamp=recording_timestamp, gain_db=0)

        # Save the final stacked waveform
        sf.write(file_path, stacked, samplerate)
        print(f"[✓] Saved stacked output to: {file_path} \n")















def add_arguments(parser):
    parser.add_argument("--sensorId", type=str, default="000000", help="Sensor ID")
    parser.add_argument("--sensorName", type=str, default="DEV", help="Sensor Name")
    parser.add_argument("--siteId", type=str, default="XXX-000", help="Site ID")
    parser.add_argument("--siteName", type=str, default="YY-000", help="Site Name")
    parser.add_argument("--siteStation", type=str, default="0_000.000", help="Station Label")
    parser.add_argument("--filter_type", choices=["none", "bandpass", "lowpass", "highpass"], default="bandpass", help="Type of filter to apply")
    parser.add_argument("--lowcut", type=float, default=1.0, help="Low cut frequency for filter (Hz)")
    parser.add_argument("--highcut", type=float, default=2047.0, help="High cut frequency for filter (Hz)")

async def run(args):
    #print(f"[XXX] Running STACKER with args: {args}")
    sensor_id = sanitize_path_component(args.sensorId)
    sensor_name = sanitize_path_component(args.sensorName)
    sensor_site_id = sanitize_path_component(args.siteId)
    sensor_site_name = sanitize_path_component(args.siteName)
    sensor_station = sanitize_path_component(args.siteStation)
    filter_type = args.filter_type
    lowcut = args.lowcut
    highcut = args.highcut
    stack_wav_files(sensor_id, sensor_name, sensor_site_id, sensor_site_name, sensor_station, filter_type, lowcut, highcut)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor WAV Stacker Engine")
    add_arguments(parser)
    args = parser.parse_args()
    print(f"\n[XXX] Running STACKER with args: {args}")
    asyncio.run(run(args))