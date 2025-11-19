import os
import re
import sys
import pywt
import soundfile as sf
import numpy as np
import librosa
import argparse
import multiprocessing

import matplotlib
if '--wav_folder' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from PyEMD import EMD
from concurrent.futures import ProcessPoolExecutor, as_completed

SILENT_THRESH = 0.01
CLIPPED_THRESH = 0.98
MAX_HIST_BINS = 100
PLOT_WAVEFORM_DOWNSAMPLE = 50000

def safe_skew(x): return skew(x) if np.std(x) > 1e-8 else 0.0
def safe_kurtosis(x): return kurtosis(x) if np.std(x) > 1e-8 else 0.0
def safe_log10(x, eps=1e-12): return np.log10(np.maximum(x, eps))
def safe_autocorr(x):
    ac = np.correlate(x, x, mode='full')
    ac = ac[ac.size // 2:]
    if np.abs(ac[0]) > 1e-8: ac /= ac[0]
    else: ac[:] = 0
    return ac
def safe_plot(func, ax, *args, **kwargs):
    try: func(*args, **kwargs)
    except Exception: ax.text(0.5, 0.5, "Plot error", ha='center', va='center'); ax.set_axis_off()
def downsample_for_plot(arr, max_len=5000):
    if len(arr) > max_len:
        idx = np.linspace(0, len(arr)-1, max_len).astype(int)
        return arr[idx]
    return arr

def compute_wavelet(data_norm, samplerate):
    freqs = np.arange(1, 33)
    cfs, _ = pywt.cwt(data_norm, freqs, 'morl', sampling_period=1/samplerate)
    wt_energy = np.sum(np.abs(cfs)**2, axis=1)
    return cfs, wt_energy, freqs

def compute_wavelet_packet(data_norm):
    wp = pywt.WaveletPacket(data=data_norm, wavelet='db4', maxlevel=4)
    wp_energies = [np.sum(np.abs(node.data)**2) for node in wp.get_level(4, 'freq')]
    return wp_energies

def compute_emd(data_norm, max_imfs_to_plot):
    emd = EMD()
    imfs = emd(data_norm)
    return imfs[:max_imfs_to_plot]

def load_and_normalize_wav(wav_path):
    data, samplerate = sf.read(wav_path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    data_dc = data - np.mean(data)
    max_abs = np.max(np.abs(data_dc))
    is_silent = max_abs < 1e-12
    data_norm = np.zeros_like(data_dc) if is_silent else data_dc / max_abs
    return data, data_dc, data_norm, samplerate, is_silent

def extract_info_from_filename(filename):
    base = os.path.basename(filename)
    name_parts = base.replace(".wav", "").split("~")
    if len(name_parts) < 4:
        raise ValueError(f"[✗] Invalid filename format: {filename}")
    sensor_id, recording_id, timestamp = name_parts[:3]
    gain_db = float(name_parts[3])
    return sensor_id, recording_id, timestamp, gain_db

def compute_stats(data, data_dc, data_norm, envelope):
    return {
        'peak': np.max(np.abs(data)) if len(data) else 0.0,
        'rms': np.sqrt(np.mean(data_dc**2)) if len(data) else 0.0,
        'min': np.min(data) if len(data) else 0.0,
        'max': np.max(data) if len(data) else 0.0,
        'median': np.median(data) if len(data) else 0.0,
        'mean': np.mean(data) if len(data) else 0.0,
        'skew': safe_skew(data),
        'kurtosis': safe_kurtosis(data),
        'snr': (
            float('-inf') if np.std(data_norm - envelope) < 1e-12 else
            20 * safe_log10(np.sqrt(np.mean(data_dc**2)) / (np.std(data_norm - envelope) + 1e-12))
        ),
    }

def generate_plots(
    data, data_dc, data_norm, samplerate, envelope, is_silent, stats, sensor_id, recording_id,
    recording_timestamp, gain_db, max_imfs_to_plot, features, plotfile_prefix, save_svg=False, all_plots=False,
    skip_existing=True
):
    (autocorr, lags, fft_freqs, fft_vals, fft_db, fft_power, fft_power_db, log_freqs, log_vals, log_power,
    energies, energy_times, hist_vals, hist_bins, wt_energy, wp_energies, cfs, freqs, imfs, num_imfs,
    S_db, mel_db, C_db, mfccs) = features

    waveform = downsample_for_plot(data_norm, PLOT_WAVEFORM_DOWNSAMPLE)
    envelope_plot = downsample_for_plot(envelope, PLOT_WAVEFORM_DOWNSAMPLE)
    meta_info = (
        f"Sensor: {sensor_id or 'N/A'} | Recording: {recording_id or 'N/A'}\n"
        f"Timestamp: {recording_timestamp or 'N/A'} | Gain: {gain_db or 'N/A'}\n"
        f"Samples: {len(data)}, Duration: {len(data)/samplerate:.2f}s, Rate: {samplerate} Hz\n"
    )

    output_dir = os.path.dirname(plotfile_prefix)
    prefix = os.path.basename(plotfile_prefix)
    png_path = os.path.join(output_dir, f"{prefix}_wav_analysis.png")
    svg_path = os.path.join(output_dir, f"{prefix}_wav_analysis.svg")

    # --- Skip if PNG exists ---
    if skip_existing and os.path.exists(png_path):
        print(f"[✓] Skipped (already exists): {png_path}")
        return

    if not all_plots:
        fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), constrained_layout=False)
        plt.subplots_adjust(top=0.85, bottom=0.07, left=0.09, right=0.98, hspace=0.45)
        fig.text(0.5, 0.96, meta_info, ha='center', va='top', fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.88, edgecolor='none'))
        axes[0].set_title("Autocorrelation")
        safe_plot(lambda: axes[0].plot(lags[:len(autocorr)//8], autocorr[:len(autocorr)//8], color='blue'), axes[0])
        axes[0].set_xlabel("Lag (s)")
        axes[0].set_ylabel("Correlation")
        axes[0].grid(True)
        axes[1].set_title("Waveform (normalized, DC removed)")
        safe_plot(lambda: axes[1].plot(waveform, label='Waveform', lw=0.8), axes[1])
        silent_mask = np.abs(data_norm) < SILENT_THRESH
        clipped_mask = np.abs(data_norm) > CLIPPED_THRESH
        axes[1].fill_between(np.arange(len(data_norm)), -1, 1, where=silent_mask, color='dodgerblue', alpha=0.12)
        axes[1].fill_between(np.arange(len(data_norm)), -1, 1, where=clipped_mask, color='red', alpha=0.12)
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True)
        axes[2].set_title("Frequency Spectrum")
        safe_plot(lambda: axes[2].plot(fft_freqs, fft_vals, color='purple'), axes[2])
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)
    else: 
        # --- All plots mode ---
        if skip_existing and os.path.exists(svg_path):
            print(f"[✓] Skipped (already exists): {svg_path}")
            return
        
        print(f"[i] Generating full analysis plots for {os.path.basename(png_path)}...")

        # Create a grid of subplots     	
        num_fixed_axes = 20
        num_axes = num_fixed_axes + num_imfs
        height_ratios = [1.5] + [1]*(num_axes-1)
        fig, axes = plt.subplots(num_axes, 1, figsize=(17, 2.3*num_axes), gridspec_kw={'height_ratios': height_ratios}, constrained_layout=False, squeeze=False)
        axes = axes.flatten()
        plt.subplots_adjust(top=1.0, bottom=0.01, left=0.06, right=0.99, hspace=0.65, wspace=0)

        axes[0].axis('off')
        axes[0].text(0.5, 0.98, meta_info, va='top', ha='center', fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.88))

        # Autocorrelation
        safe_plot(lambda: axes[1].plot(lags[:len(autocorr)//8], autocorr[:len(autocorr)//8], color='blue'), axes[1])
        axes[1].set_title("Autocorrelation")
        axes[1].set_xlabel("Lag (s)")
        axes[1].set_ylabel("Correlation")
        axes[1].grid(True)

        # Waveform
        safe_plot(lambda: axes[2].plot(waveform, label='Waveform', lw=0.8), axes[2])
        silent_mask = np.abs(data_norm) < SILENT_THRESH
        clipped_mask = np.abs(data_norm) > CLIPPED_THRESH
        axes[2].fill_between(np.arange(len(data_norm)), -1, 1, where=silent_mask, color='dodgerblue', alpha=0.12, label='Silent')
        axes[2].fill_between(np.arange(len(data_norm)), -1, 1, where=clipped_mask, color='red', alpha=0.12, label='Clipped')
        axes[2].set_title("Waveform (normalized, DC removed) [blue=silent, red=clipped]")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True)

        # Envelope
        safe_plot(lambda: axes[3].plot(envelope_plot, color='dodgerblue'), axes[3])
        axes[3].set_title("Waveform Envelope (Hilbert transform)")
        axes[3].set_ylabel("Amplitude")
        axes[3].grid(True)

        # Frequency Spectrum plots
        safe_plot(lambda: axes[4].plot(fft_freqs, fft_vals, color='purple'), axes[4])
        axes[4].set_title("Frequency Spectrum (Linear Amplitude)")
        axes[4].set_xlabel("Frequency (Hz)")
        axes[4].set_ylabel("Amplitude")
        axes[4].grid(True)

        safe_plot(lambda: axes[5].plot(fft_freqs, fft_db, color='purple'), axes[5])
        axes[5].set_title("Frequency Spectrum (Log Amplitude, dB)")
        axes[5].set_xlabel("Frequency (Hz)")
        axes[5].set_ylabel("dB")
        axes[5].grid(True)

        if len(log_freqs) > 0 and len(log_vals) > 0:
            safe_plot(lambda: axes[6].plot(log_freqs, log_vals, color="purple"), axes[6])
            axes[6].set_xscale("log")
            axes[6].set_yscale("log")
            axes[6].set_title("Frequency Spectrum (Log-Log scale)")
            axes[6].set_xlabel("Frequency (Hz), log")
            axes[6].set_ylabel("Amplitude, log")
            axes[6].grid(True)
        else:
            axes[6].text(0.5, 0.5, "No positive FFT values", ha='center', va='center')
            axes[6].set_axis_off()

        safe_plot(lambda: axes[7].plot(fft_freqs, fft_power, color='orange'), axes[7])
        axes[7].set_title("Power Spectrum (Linear)")
        axes[7].set_xlabel("Frequency (Hz)")
        axes[7].set_ylabel("Power")
        axes[7].grid(True)

        safe_plot(lambda: axes[8].plot(fft_freqs, fft_power_db, color='orange'), axes[8])
        axes[8].set_title("Power Spectrum (Log, dB)")
        axes[8].set_xlabel("Frequency (Hz)")
        axes[8].set_ylabel("dB")
        axes[8].grid(True)

        if len(log_freqs) > 0 and len(log_power) > 0:
            safe_plot(lambda: axes[9].plot(log_freqs, log_power, color="orange"), axes[9])
            axes[9].set_xscale("log")
            axes[9].set_yscale("log")
            axes[9].set_title("Power Spectrum (Log-Log scale)")
            axes[9].set_xlabel("Frequency (Hz), log")
            axes[9].set_ylabel("Power, log")
            axes[9].grid(True)
        else:
            axes[9].text(0.5, 0.5, "No positive FFT values", ha='center', va='center')
            axes[9].set_axis_off()

        if len(energy_times) > 0 and len(energies) > 0:
            safe_plot(lambda: axes[10].plot(energy_times, energies, color='red'), axes[10])
        axes[10].set_title("Short-time Energy (50 ms window)")
        axes[10].set_xlabel("Time (s)")
        axes[10].set_ylabel("Energy")
        axes[10].grid(True)

        axes[11].bar((hist_bins[:-1] + hist_bins[1:]) / 2, hist_vals, width=(hist_bins[1] - hist_bins[0]), color="gray", alpha=0.7)
        axes[11].set_title("Amplitude Histogram")
        axes[11].set_xlabel("Normalized Amplitude")
        axes[11].set_ylabel("Count")
        axes[11].grid(True)

        axes[12].bar(np.arange(1, len(wt_energy) + 1), wt_energy, color='darkcyan', alpha=0.8)
        axes[12].set_title("Wavelet Energy by Scale (CWT, 'morl')")
        axes[12].set_xlabel("Scale")
        axes[12].set_ylabel("Energy")

        axes[13].bar(np.arange(1, len(wp_energies)+1), wp_energies, color='darkcyan', alpha=0.8)
        axes[13].set_title("Wavelet Packet Energy (level=4, 'db4')")
        axes[13].set_xlabel("Node")
        axes[13].set_ylabel("Energy")

        try:
            librosa.display.specshow(S_db, sr=samplerate, hop_length=512, x_axis='time', y_axis='linear', ax=axes[14])
            axes[14].set_title("STFT Spectrogram (Linear, dB)")
        except Exception:
            axes[14].text(0.5, 0.5, "No spectrogram", ha='center', va='center')
            axes[14].set_axis_off()
        try:
            librosa.display.specshow(S_db, sr=samplerate, hop_length=512, x_axis='time', y_axis='log', ax=axes[15])
            axes[15].set_title("STFT Spectrogram (Log-Frequency, dB)")
        except Exception:
            axes[15].text(0.5, 0.5, "No spectrogram", ha='center', va='center')
            axes[15].set_axis_off()
        try:
            librosa.display.specshow(mel_db, sr=samplerate, hop_length=512, x_axis='time', y_axis='mel', ax=axes[16])
            axes[16].set_title("Mel-Spectrogram (dB, 64 bands)")
        except Exception:
            axes[16].text(0.5, 0.5, "No mel-spectrogram", ha='center', va='center')
            axes[16].set_axis_off()
        try:
            librosa.display.specshow(C_db, sr=samplerate, x_axis='time', y_axis='cqt_note', ax=axes[17])
            axes[17].set_title("CQT Spectrogram (Constant-Q, dB)")
        except Exception:
            axes[17].text(0.5, 0.5, "No CQT", ha='center', va='center')
            axes[17].set_axis_off()
        try:
            axes[18].imshow(np.abs(cfs), aspect='auto',
                extent=[0, len(data)/samplerate, freqs[-1], freqs[0]], cmap='magma')
            axes[18].set_title("Wavelet Scalogram ('morl')")
            axes[18].set_ylabel("Frequency (Hz)")
            axes[18].set_xlabel("Time (s)")
        except Exception:
            axes[18].text(0.5, 0.5, "No wavelet scalogram", ha='center', va='center')
            axes[18].set_axis_off()
        try:
            img = axes[19].imshow(mfccs, aspect='auto', origin='lower', cmap='viridis',
                extent=[0, len(data)/samplerate, 0, mfccs.shape[0]])
            axes[19].set_title("MFCCs (Mel-frequency cepstral coefficients)")
            axes[19].set_xlabel("Time (s)")
            axes[19].set_ylabel("MFCC Index")
            fig.colorbar(img, ax=axes[19], format="%+2.0f dB")
        except Exception:
            axes[19].text(0.5, 0.5, "No MFCCs", ha='center', va='center')
            axes[19].set_axis_off()

        # 20+. IMFs from EMD (up to num_imfs)
        for idx in range(num_imfs):
            safe_plot(lambda: axes[20+idx].plot(imfs[idx], lw=0.8), axes[20+idx])
            axes[20+idx].set_title(f"EMD IMF #{idx+1}")
            axes[20+idx].set_ylabel("Value")
            axes[20+idx].set_xlim([0, len(imfs[idx])])




    fig.savefig(png_path, format="png", bbox_inches='tight', pad_inches=0.3, dpi=150)
    print(f"[✓] Saved analysis PNG to: {png_path}")

    if save_svg:
        if skip_existing and os.path.exists(svg_path):
            print(f"[✓] Skipped (already exists): {svg_path}")
        else:
            fig.savefig(svg_path, format="svg", bbox_inches='tight', pad_inches=0.3)
            print(f"[✓] Saved analysis SVG to: {svg_path}")
            with open(svg_path, "r") as f:
                svg = f.read()
            svg = re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"', r'\1 width="100%"', svg, count=1)
            svg = re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"', r'\1 ', svg, count=1)
            with open(svg_path, "w") as f:
                f.write(svg)
            print(f"[✓] Saved responsive SVG to: {svg_path}")

    plt.close(fig)

def analyze_wav_to_svg(args):
    wav_path, max_imfs_to_plot, save_svg, all_plots, skip_existing = args
    try:
        sensor_id, recording_id, recording_timestamp, gain_db = extract_info_from_filename(wav_path)
    except Exception:
        sensor_id = recording_id = recording_timestamp = gain_db = None

    data, data_dc, data_norm, samplerate, is_silent = load_and_normalize_wav(wav_path)
    analytic = hilbert(data_norm)
    envelope = np.abs(np.asarray(analytic))
    if is_silent:
        print(f"[i] File {wav_path} is silent or near-silent. Skipping heavy transforms.")
        features = (
            np.zeros(len(data_norm)),  # autocorr
            np.arange(len(data_norm)),  # lags
            np.zeros(len(data_norm)),  # fft_freqs
            np.zeros(len(data_norm)),  # fft_vals
            np.zeros(len(data_norm)),  # fft_db
            np.zeros(len(data_norm)),  # fft_power
            np.zeros(len(data_norm)),  # fft_power_db
            np.zeros(1), np.zeros(1), np.zeros(1),  # log_freqs, log_vals, log_power
            np.zeros(1), np.zeros(1),  # energies, energy_times
            np.zeros(1), np.zeros(2),  # hist_vals, hist_bins
            np.zeros(1), np.zeros(1), np.zeros((1,1)), np.zeros(1), np.zeros((1,1)),  # wt_energy, wp_energies, cfs, freqs, imfs
            0,  # num_imfs
            np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1))  # S_db, mel_db, C_db, mfccs
        )
        stats = compute_stats(data, data_dc, data_norm, envelope)
        plotfile_prefix = os.path.splitext(wav_path)[0]
        generate_plots(
            data, data_dc, data_norm, samplerate, envelope, is_silent, stats,
            sensor_id, recording_id, recording_timestamp, gain_db, max_imfs_to_plot, features, plotfile_prefix,
            save_svg=save_svg, all_plots=all_plots, skip_existing=skip_existing
        )
        return wav_path

    autocorr = safe_autocorr(data_norm)
    lags = np.arange(autocorr.size) / samplerate
    n = len(data_norm)
    fft_vals = np.abs(rfft(data_norm))
    fft_freqs = rfftfreq(n, d=1/samplerate)
    fft_db = 20 * safe_log10(fft_vals)
    fft_power = fft_vals ** 2
    fft_power_db = 10 * safe_log10(fft_power)
    pos_mask = (fft_freqs > 0) & (fft_vals > 0)
    log_freqs = fft_freqs[pos_mask]
    log_vals = fft_vals[pos_mask]
    log_power = fft_power[pos_mask]
    frame_size = int(samplerate * 0.05)
    step = frame_size // 2
    energies = np.array([np.sum(data_norm[i:i+frame_size]**2)
                         for i in range(0, len(data_norm) - frame_size, step)]) if (len(data_norm) - frame_size) > 0 else np.array([])
    energy_times = np.arange(len(energies)) * (step / samplerate)
    hist_vals, hist_bins = np.histogram(data_norm, bins=MAX_HIST_BINS)

    if all_plots:
        cfs, wt_energy, freqs = compute_wavelet(data_norm, samplerate)
        wp_energies = compute_wavelet_packet(data_norm)
        imfs = compute_emd(data_norm, max_imfs_to_plot)
        num_imfs = min(imfs.shape[0], max_imfs_to_plot) if hasattr(imfs, 'shape') else len(imfs)
        try:
            S = np.abs(librosa.stft(data_norm, n_fft=1024, hop_length=512))**2
            S_db = librosa.power_to_db(S, ref=np.max)
            mel = librosa.feature.melspectrogram(y=data_norm, sr=samplerate, n_fft=1024, hop_length=512, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            C = np.abs(librosa.cqt(data_norm, sr=samplerate, n_bins=64))
            C_db = librosa.amplitude_to_db(C, ref=np.max)
            mfccs = librosa.feature.mfcc(y=data_norm, sr=samplerate, n_mfcc=20)
        except Exception:
            S_db = mel_db = C_db = mfccs = np.zeros((1,1))
    else:
        cfs = wt_energy = freqs = wp_energies = imfs = S_db = mel_db = C_db = mfccs = np.zeros((1,1))
        num_imfs = 0

    stats = compute_stats(data, data_dc, data_norm, envelope)

    features = (
        autocorr, lags, fft_freqs, fft_vals, fft_db, fft_power, fft_power_db, log_freqs, log_vals, log_power,
        energies, energy_times, hist_vals, hist_bins, wt_energy, wp_energies, cfs, freqs, imfs, num_imfs,
        S_db, mel_db, C_db, mfccs
    )
    plotfile_prefix = os.path.splitext(wav_path)[0]
    generate_plots(
        data, data_dc, data_norm, samplerate, envelope, is_silent, stats,
        sensor_id, recording_id, recording_timestamp, gain_db, max_imfs_to_plot, features, plotfile_prefix,
        save_svg=save_svg, all_plots=all_plots, skip_existing=skip_existing
    )
    return wav_path

def find_wav_files(folder):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files if f.lower().endswith(".wav")
    ]

def batch_analyze(folder, max_imfs_to_plot=7, num_workers=None, save_svg=False, all_plots=False, skip_existing=True):
    wavs = find_wav_files(folder)
    print(f"[i] Found {len(wavs)} .wav files in '{folder}'.")
    if num_workers is None or num_workers <= 0:
        num_workers = multiprocessing.cpu_count()
    print(f"[i] Using {num_workers} parallel processes for batch processing.")
    args_list = [(wav, max_imfs_to_plot, save_svg, all_plots, skip_existing) for wav in wavs]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_wav_to_svg, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures)):
            wav = futures[future]
            try:
                future.result()
                print(f"[{i+1}/{len(wavs)}] Processed: {wav}")
            except Exception as e:
                print(f"[!] Error processing {wav}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze WAV files and output PNG (default) or SVG visualizations.")
    parser.add_argument("--wav_path", help="Path to input WAV file")
    parser.add_argument("--wav_folder", help="If set, process all .wav files under this folder recursively")
    parser.add_argument("--max_imfs_to_plot", help="Max number of IMFs to plot (default: 7)", type=int, default=7)
    parser.add_argument("--num_workers", help="Number of parallel workers for batch mode (default: all CPU cores)", type=int, default=None)
    parser.add_argument("--svg", help="Also save SVG and responsive SVG in addition to PNG", action="store_true")
    parser.add_argument("--all_plots", help="Include all advanced plots, not just autocorr/waveform/spectrum", action="store_true")
    parser.add_argument("--no_skip_existing", help="Regenerate image files even if they already exist", action="store_true")
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing

    if args.wav_folder:
        batch_analyze(args.wav_folder, max_imfs_to_plot=args.max_imfs_to_plot, num_workers=args.num_workers,
                      save_svg=args.svg, all_plots=args.all_plots, skip_existing=skip_existing)
    elif args.wav_path:
        analyze_wav_to_svg((args.wav_path, args.max_imfs_to_plot, args.svg, args.all_plots, skip_existing))
    else:
        print("You must supply either --wav_path or --wav_folder.")