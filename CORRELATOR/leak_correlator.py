import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, spectrogram, welch, butter, filtfilt, sosfilt, firwin
from scipy.stats import kurtosis, entropy
import uuid
import os
from datetime import datetime

def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, samplerate

def sanitize(s):
    return str(s).replace(' ', '_').replace('\\', '_').replace('/', '_').replace(':', '_')

def plot_spectrogram(signal, sr, title="Spectrogram", freq_lim=(0, 3000), savepath=None):
    f, t, Sxx = spectrogram(signal, sr, nperseg=2048)
    plt.figure(figsize=(10,6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx+1e-12), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.ylim(freq_lim)
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_cross_correlation(lags, corr, top_lags, top_corrs, wav1, wav2, savepath=None):
    plt.figure(figsize=(12,6))
    plt.plot(lags, corr, label='Cross-correlation')
    plt.scatter(top_lags, top_corrs, color='red', zorder=5, label='Detected Peaks')
    plt.title(f'Cross-correlation between:\nSensor 1: {wav1}\nand Sensor 2: {wav2}')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_waveforms(data1, data2, sr, label1, label2, savepath=None):
    t = np.arange(len(data1)) / sr
    plt.figure(figsize=(12,5))
    plt.plot(t, data1, label=label1)
    plt.plot(t, data2, label=label2, alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Sensor Waveforms")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_histogram(data1, data2, label1, label2, savepath=None):
    plt.figure(figsize=(10,5))
    plt.hist(data1, bins=100, alpha=0.5, label=label1)
    plt.hist(data2, bins=100, alpha=0.5, label=label2)
    plt.xlabel('Amplitude')
    plt.ylabel('Count')
    plt.title('Amplitude Histogram')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_psd(data1, data2, sr, label1, label2, savepath=None):
    plt.figure(figsize=(10,5))
    f1, Pxx1 = welch(data1, sr, nperseg=2048)
    f2, Pxx2 = welch(data2, sr, nperseg=2048)
    plt.semilogy(f1, Pxx1, label=label1)
    plt.semilogy(f2, Pxx2, label=label2)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def plot_frequency_spectrum_with_peaks_and_save(
    data1, data2, sr, label1, label2, 
    leak_lags, leak_corrs, lags, 
    c_avg, sensor_distance, 
    out_dir, filename_root,
    segment_ms=100  # length of window to analyze (in milliseconds)
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    freqs = np.fft.rfftfreq(len(data1), d=1/sr)
    fft1 = np.abs(np.fft.rfft(data1))
    fft2 = np.abs(np.fft.rfft(data2))
    plt.figure(figsize=(12,6))
    plt.plot(freqs, fft1, label=f"{label1} spectrum", alpha=0.8)
    plt.plot(freqs, fft2, label=f"{label2} spectrum", alpha=0.8)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum with Leak Indicators")

    # Filter out zero-lag leaks
    leak_lags = np.array(leak_lags)
    leak_corrs = np.array(leak_corrs)
    nonzero = leak_lags != 0
    leak_lags = leak_lags[nonzero]
    leak_corrs = leak_corrs[nonzero]
    n_leaks = len(leak_lags)
    if n_leaks == 0:
        plt.legend()
        plt.tight_layout()
        spectrum_filename = os.path.join(out_dir, filename_root + "~spectrum-leak.png")
        plt.savefig(spectrum_filename)
        plt.close()
        return spectrum_filename

    y_max = max(fft1.max(), fft2.max())
    x_offsets = np.linspace(120, 500, n_leaks)
    fontsize = max(8, 12 - n_leaks//2)
    segment_len = int((segment_ms / 1000.0) * sr)
    label_offset = y_max * 0.5  # always put label above arrow tip

    for i, (lag, corr_val) in enumerate(zip(leak_lags, leak_corrs)):
        time_lag = lag / sr
        if sensor_distance is not None:
            x = (sensor_distance + c_avg * time_lag) / 2
            x = max(0, min(x, sensor_distance))
            leak_distance = x
        else:
            leak_distance = abs(time_lag) * c_avg
        color = f"C{(i+2)%10}"

        # Center window in data1 at midpoint + lag for this event
        leak_sample = len(data1)//2 + lag
        seg_start = int(max(leak_sample - segment_len//2, 0))
        seg_end = int(min(leak_sample + segment_len//2, len(data1)))
        segment = data1[seg_start:seg_end]

        if len(segment) > 10:
            seg_fft = np.abs(np.fft.rfft(segment))
            seg_freqs = np.fft.rfftfreq(len(segment), d=1/sr)
            peak_idx = np.argmax(seg_fft)
            peak_freq = seg_freqs[peak_idx]
            # Get the spectrum value at the peak frequency for arrow anchor
            arrow_y = np.interp(peak_freq, freqs, fft1)
        else:
            peak_freq = 0
            arrow_y = 0

        # Label always above arrow tip
        xytext_x = min(peak_freq + x_offsets[i], freqs[-1] * 0.98)
        xytext_y = arrow_y + label_offset
        plt.annotate(
            f"Leak {i+1}\nLag: {lag} ({time_lag:.3f}s)\nDist: {leak_distance:.1f}m\nPeak: {peak_freq:.1f}Hz",
            xy=(peak_freq, arrow_y),
            xytext=(xytext_x, xytext_y),
            color=color, fontsize=fontsize, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="w", alpha=0.7, ec=color, lw=2),
            arrowprops=dict(arrowstyle="->", color=color, lw=1, alpha=0.7)
        )
    plt.legend()
    plt.tight_layout()
    spectrum_filename = os.path.join(out_dir, filename_root + "~spectrum-leak.png")
    plt.savefig(spectrum_filename)
    plt.close()
    return spectrum_filename


def save_all_plots(
    data1, data2, sr, corr, lags, 
    wav1_path, wav2_path, sensor1_name, sensor2_name, settings, 
    top_lags, top_corrs, freq_band, c_avg, sensor_distance
):
    out_dir = os.path.dirname(os.path.abspath(wav1_path))
    uuid_str = str(uuid.uuid4())
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    sensor1_info = sanitize(os.path.basename(sensor1_name))
    sensor2_info = sanitize(os.path.basename(sensor2_name))
    settings_info = sanitize(settings)
    filename_root = f"correlation~{uuid_str}~{sensor1_info}~{sensor2_info}~{settings_info}~{now}"

    plot_waveforms(data1, data2, sr, sensor1_info, sensor2_info,
        savepath=os.path.join(out_dir, filename_root + "~waveforms.png"))
    plot_spectrogram(data1, sr, title=f"Spectrogram: {sensor1_info}",
        freq_lim=freq_band, savepath=os.path.join(out_dir, filename_root + "~spectrogram_1.png"))
    plot_spectrogram(data2, sr, title=f"Spectrogram: {sensor2_info}",
        freq_lim=freq_band, savepath=os.path.join(out_dir, filename_root + "~spectrogram_2.png"))
    plot_histogram(data1, data2, sensor1_info, sensor2_info,
        savepath=os.path.join(out_dir, filename_root + "~histogram.png"))
    plot_cross_correlation(lags, corr, top_lags, top_corrs, sensor1_info, sensor2_info,
        savepath=os.path.join(out_dir, filename_root + "~correlation.png"))
    plot_psd(data1, data2, sr, sensor1_info, sensor2_info,
        savepath=os.path.join(out_dir, filename_root + "~psd.png"))
    spectrum_leak_path = plot_frequency_spectrum_with_peaks_and_save(
        data1, data2, sr, sensor1_info, sensor2_info,
        leak_lags=top_lags, leak_corrs=top_corrs, lags=lags,
        c_avg=c_avg, sensor_distance=sensor_distance,
        out_dir=out_dir, filename_root=filename_root)
    return [os.path.join(out_dir, filename_root + suffix) 
            for suffix in ["~waveforms.png", "~spectrogram_1.png", "~spectrogram_2.png", "~histogram.png", "~correlation.png", "~psd.png", "~spectrum-leak.png"]]

def compute_band_power(signal, sr, band=(100, 2000)):
    freqs, psd = welch(signal, sr, nperseg=2048)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.sum(psd[band_mask])
    total_power = np.sum(psd)
    ratio = band_power / total_power if total_power > 0 else 0
    return band_power, total_power, ratio

def compute_statistics(signal):
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    psd = np.abs(np.fft.fft(signal))**2
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    return {'rms': rms, 'kurtosis': kurt, 'spectral_entropy': spec_entropy}

def speed_of_sound_water(pressure_bar, temperature_C=20.0):
    c0 = 1482.0
    dp = (pressure_bar if pressure_bar is not None else 1.013) - 1.013
    c = c0 + 0.045 * dp
    return c

def bandpass_filter(signal, sr, lowcut, highcut, order=4):
    if lowcut is None or highcut is None:
        raise ValueError("Both lowcut and highcut frequencies must be specified for bandpass filtering.")
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    #sos = butter(order, [low, high], btype='bandpass', fs=sr, output='sos')
    #filtered = sosfilt(sos, signal)
    numtaps=1024
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sr)
    filtered = filtfilt(taps, 1.0, signal)
    return filtered

def fir_bandpass_filter(data, lowcut, highcut, fs, numtaps=801):
    """FIR bandpass filter with sharp cutoff and linear phase using filtfilt."""
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return filtfilt(taps, 1.0, data)

def align_signals(sig1, sig2):
    print("[INFO] Performing initial alignment of sensor files ...")
    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig1) + 1, len(sig2))
    max_lag = lags[np.argmax(np.abs(corr))]
    print(f"[INFO] Alignment lag (samples): {max_lag}")
    if max_lag > 0:
        sig1_aligned = sig1[max_lag:]
        sig2_aligned = sig2[:len(sig1_aligned)]
    elif max_lag < 0:
        sig2_aligned = sig2[-max_lag:]
        sig1_aligned = sig1[:len(sig2_aligned)]
    else:
        min_len = min(len(sig1), len(sig2))
        sig1_aligned = sig1[:min_len]
        sig2_aligned = sig2[:min_len]
    final_len = min(len(sig1_aligned), len(sig2_aligned))
    return sig1_aligned[:final_len], sig2_aligned[:final_len]

def preprocess_signal(signal, lowcut, highcut, sr):
    signal = signal - np.mean(signal)
    signal = signal - np.linspace(signal[0], signal[-1], num=len(signal))
    signal = bandpass_filter(signal, sr, lowcut, highcut)
    signal = signal / (np.std(signal) + 1e-12)
    return signal

def leak_detection_full_analysis(
    wav1, wav2,
    top_n=3,
    horizontal_distance=None,
    depth1=None, depth2=None,
    pressure1=None, pressure2=None,
    temperature1=20.0, temperature2=20.0,
    baseline1=None, baseline2=None,
    plot=True,
    freq_band=(100, 2000),
    debug=False
):
    sig1, sr1 = load_wav(wav1)
    sig2, sr2 = load_wav(wav2)
    print(f"[INFO] Sensor 1 file: {wav1} | Sample rate: {sr1} Hz | Samples: {len(sig1)}")
    print(f"[INFO] Sensor 2 file: {wav2} | Sample rate: {sr2} Hz | Samples: {len(sig2)}")
    assert sr1 == sr2, "Sample rates must match"
    sr = sr1

    sig1, sig2 = align_signals(sig1, sig2)
    sig1 = preprocess_signal(sig1, freq_band[0], freq_band[1], sr)
    sig2 = preprocess_signal(sig2, freq_band[0], freq_band[1], sr)

    base1 = base2 = None
    if baseline1:
        base1, sr_base1 = load_wav(baseline1)
        assert sr_base1 == sr, "Baseline 1 sample rate mismatch!"
        base1 = preprocess_signal(base1, freq_band[0], freq_band[1], sr)
    if baseline2:
        base2, sr_base2 = load_wav(baseline2)
        assert sr_base2 == sr, "Baseline 2 sample rate mismatch!"
        base2 = preprocess_signal(base2, freq_band[0], freq_band[1], sr)

    for idx, (signal, base, fname) in enumerate([
        (sig1, base1, wav1),
        (sig2, base2, wav2)
    ], 1):
        band_power, total_power, band_ratio = compute_band_power(signal, sr, band=freq_band)
        stats = compute_statistics(signal)
        print(f"\n[INFO] Sensor {idx} - {fname}")
        print(f"  Band power ({freq_band[0]}-{freq_band[1]} Hz): {band_power:.4e}, total: {total_power:.4e}, ratio: {band_ratio:.3f}")
        print(f"  RMS={stats['rms']:.4f}, Kurtosis={stats['kurtosis']:.3f}, Spectral entropy={stats['spectral_entropy']:.3f}")
        if base is not None:
            base_band_power, base_total_power, base_band_ratio = compute_band_power(base, sr, band=freq_band)
            base_stats = compute_statistics(base)
            print(f"  Baseline band ratio: {base_band_ratio:.3f}")
            print(f"  Baseline RMS={base_stats['rms']:.4f}, Kurtosis={base_stats['kurtosis']:.3f}, Spectral entropy={base_stats['spectral_entropy']:.3f}")
            if plot:
                plot_spectrogram(base, sr, title=f"Spectrogram: Baseline Sensor {idx}")

    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig1) + 1, len(sig2))
    peaks, props = find_peaks(np.abs(corr), distance=sr//10)
    peak_amplitudes = np.abs(corr[peaks])
    top_indices = np.argsort(peak_amplitudes)[::-1][:top_n]
    top_peaks = peaks[top_indices]
    top_lags = lags[top_peaks]
    top_corrs = corr[top_peaks]

    # === FIX: calculate c1, c2, c_avg before using in save_all_plots ===
    if pressure1 is None:
        pressure1 = 1.013
    if pressure2 is None:
        pressure2 = 1.013
    if temperature1 is None:
        temperature1 = 20.0
    if temperature2 is None:
        temperature2 = 20.0
    c1 = speed_of_sound_water(pressure1, temperature1)
    c2 = speed_of_sound_water(pressure2, temperature2)
    c_avg = (c1 + c2) / 2.0

    if horizontal_distance is not None and depth1 is not None and depth2 is not None:
        sensor_distance = np.sqrt(horizontal_distance**2 + (depth2-depth1)**2)
    else:
        sensor_distance = horizontal_distance

    settings_str = f"sr{sr}_band{freq_band[0]}-{freq_band[1]}"
    save_all_plots(
        data1=sig1,
        data2=sig2,
        sr=sr,
        corr=corr,
        lags=lags,
        wav1_path=wav1,
        wav2_path=wav2,
        sensor1_name=wav1,
        sensor2_name=wav2,
        settings=settings_str,
        top_lags=top_lags,
        top_corrs=top_corrs,
        freq_band=freq_band,
        c_avg=c_avg,
        sensor_distance=sensor_distance
    )

    if plot:
        plot_cross_correlation(lags, corr, top_lags, top_corrs, wav1, wav2)

    if debug:
        print(f"[DEBUG] {wav1} sample rate: {sr1}, length: {len(sig1)} samples, duration: {len(sig1)/sr1:.2f} s")
        print(f"[DEBUG] {wav2} sample rate: {sr2}, length: {len(sig2)} samples, duration: {len(sig2)/sr2:.2f} s")
        print(f"[DEBUG] Speed of sound at sensor 1: {c1:.4f} m/s (P={pressure1} bar, T={temperature1}°C)")
        print(f"[DEBUG] Speed of sound at sensor 2: {c2:.4f} m/s (P={pressure2} bar, T={temperature2}°C)")
        print(f"[DEBUG] Average speed of sound: {c_avg:.4f} m/s")

    def compute_confidence(
        lag, time_lag, corr_value, max_corr, leak_distance, sensor_distance,
        band_ratio, base_band_ratio, min_distance=0, band_ratio_thresh=0.4
    ):
        score = 1.0
        if sensor_distance is not None:
            if leak_distance < 0 or leak_distance > 1.5 * sensor_distance:
                return 0.05
        if abs(corr_value) < 0.4 * abs(max_corr):
            score *= 0.3
        if corr_value < 0:
            score *= 0.7
        if abs(lag) < 10:
            score *= 0.2
        if base_band_ratio is not None:
            if band_ratio < 1.2 * base_band_ratio:
                score *= 0.4
        else:
            if band_ratio < band_ratio_thresh:
                score *= 0.5
        return max(0.0, min(score, 1.0))

    summary_table = []
    max_corr = np.max(np.abs(corr))
    print("\n=== Leak Localization Results ===")
    print(f"Sensor 1: {wav1}")
    print(f"Sensor 2: {wav2}")

    if base1 is not None:
        base_band_power, base_total_power, base_band_ratio = compute_band_power(base1, sr, band=freq_band)
    else:
        base_band_ratio = None

    # Compute band_ratio for sig1 (used for all peaks)
    band_power, total_power, band_ratio = compute_band_power(sig1, sr, band=freq_band)

    for i, lag in enumerate(top_lags):
        time_lag = lag / sr
        if sensor_distance is not None:
            x = (sensor_distance + c_avg * time_lag) / 2
            x = max(0, min(x, sensor_distance))
            leak_distance_from_sensor1 = x
            leak_distance_from_sensor2 = sensor_distance - x
            sensor_sep = sensor_distance
            debug_distance_str = f"using 3D separation {sensor_distance:.3f} m"
        else:
            leak_distance_from_sensor1 = abs(time_lag) * c_avg
            leak_distance_from_sensor2 = None
            sensor_sep = None
            debug_distance_str = "no sensor separation provided, distance is abs(time_lag) * avg velocity"

        confidence = compute_confidence(
            lag=lag,
            time_lag=time_lag,
            corr_value=corr[top_peaks[i]],
            max_corr=max_corr,
            leak_distance=leak_distance_from_sensor1,
            sensor_distance=sensor_distance,
            band_ratio=band_ratio,
            base_band_ratio=base_band_ratio,
            min_distance=0
        )

        summary_table.append({
            "signature": i+1,
            "lag_samples": lag,
            "time_lag": time_lag,
            "corr_value": corr[top_peaks[i]],
            "leak_distance_from_sensor1": leak_distance_from_sensor1,
            "leak_distance_from_sensor2": leak_distance_from_sensor2,
            "sensor_sep": sensor_sep,
            "velocity_sensor1": c1,
            "velocity_sensor2": c2,
            "debug_distance_str": debug_distance_str,
            "confidence": confidence
        })

        print(f"Peak {i+1}:")
        print(f"  Lag (samples): {lag}")
        print(f"  Time lag (s): {time_lag:.9f}")
        print(f"  Leak distance (from Sensor 1: {wav1}): {leak_distance_from_sensor1:.5f} m")
        if leak_distance_from_sensor2 is not None:
            print(f"  Leak distance (from Sensor 2: {wav2}): {leak_distance_from_sensor2:.5f} m")
        if sensor_sep is not None:
            print(f"  Sensor separation: {sensor_sep:.5f} m")
        print(f"  Used velocity (sensor 1: {wav1}): {c1:.5f} m/s")
        print(f"  Used velocity (sensor 2: {wav2}): {c2:.5f} m/s")
        print(f"  Correlation value: {corr[top_peaks[i]]:.3f}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  [{debug_distance_str}]")

    filtered_rows = [row for row in summary_table if row["lag_samples"] != 0]
    for idx, row in enumerate(filtered_rows, 1):
        row["signature"] = idx
    if sensor_distance is not None:
        print(f"{'Signature':>10} | {'Lag (samples)':>12} | {'Time lag (s)':>12} | {'Dist from S1 (m) [S1: '+os.path.basename(wav1)+']':>30} | {'Dist from S2 (m) [S2: '+os.path.basename(wav2)+']':>30} | {'Corr. value':>12} | {'Confidence':>11} | {'Sensor Sep (m)':>14} | {'V1 (m/s)':>9} | {'V2 (m/s)':>9}")
        print("-" * 180)
        for row in filtered_rows:
            print(f"{row['signature']:>10} | {row['lag_samples']:>12} | {row['time_lag']:>12.6f} | {row['leak_distance_from_sensor1']:>30.5f} | {row['leak_distance_from_sensor2']:>30.5f} | {row['corr_value']:>12.3f} | {row['confidence']*100:>10.1f}% | {row['sensor_sep']:>14.5f} | {row['velocity_sensor1']:>9.2f} | {row['velocity_sensor2']:>9.2f}")
    else:
        print(f"{'Signature':>10} | {'Lag (samples)':>12} | {'Time lag (s)':>12} | {'Abs Dist from S1 (m) [S1: '+os.path.basename(wav1)+']':>37} | {'Corr. value':>12} | {'Confidence':>11} | {'V1 (m/s)':>9} | {'V2 (m/s)':>9}")
        print("-" * 135)
        for row in filtered_rows:
            print(f"{row['signature']:>10} | {row['lag_samples']:>12} | {row['time_lag']:>12.6f} | {row['leak_distance_from_sensor1']:>37.5f} | {row['corr_value']:>12.3f} | {row['confidence']*100:>10.1f}% | {row['velocity_sensor1']:>9.2f} | {row['velocity_sensor2']:>9.2f}")

    print(f"\n=== Leak Signature Detection (Sensor 1: {wav1}) ===")
    band_power, total_power, band_ratio = compute_band_power(sig1, sr, band=freq_band)
    stats = compute_statistics(sig1)
    leak_detected = False
    reason = ""
    if base1 is not None:
        base_band_power, base_total_power, base_band_ratio = compute_band_power(base1, sr, band=freq_band)
        base_stats = compute_statistics(base1)
        if band_ratio > base_band_ratio * 1.2:
            leak_detected = True
            reason += f"Band power ratio elevated ({band_ratio:.2f} > {base_band_ratio:.2f}). "
        if stats['spectral_entropy'] < base_stats['spectral_entropy']:
            leak_detected = True
            reason += f"Spectral entropy lower than baseline ({stats['spectral_entropy']:.2f} < {base_stats['spectral_entropy']:.2f}). "
        if stats['rms'] > base_stats['rms'] * 1.2:
            leak_detected = True
            reason += f"RMS elevated ({stats['rms']:.2f} > {base_stats['rms']:.2f}). "
    else:
        if band_ratio > 0.4:
            leak_detected = True
            reason += f"Band power ratio high ({band_ratio:.2f}). "

    print(f"Leak detected: {leak_detected} | Reason: {reason if reason else 'No strong evidence.'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Leak Detection and Leak Signature Localization with Plots")
    parser.add_argument("wav1", help="First sensor WAV file")
    parser.add_argument("wav2", help="Second sensor WAV file")
    parser.add_argument("--top", type=int, default=3, help="Top N prominent signals to detect (default: 3)")
    parser.add_argument("--distance", type=float, default=None,
                        help="Horizontal distance between the two sensors (meters)")
    parser.add_argument("--depth1", type=float, default=None, help="Depth of sensor 1 (meters)")
    parser.add_argument("--depth2", type=float, default=None, help="Depth of sensor 2 (meters)")
    parser.add_argument("--pressure1", type=float, default=None, help="Pressure at sensor 1 (bar)")
    parser.add_argument("--pressure2", type=float, default=None, help="Pressure at sensor 2 (bar)")
    parser.add_argument("--temperature1", type=float, default=20.0, help="Water temperature at sensor 1 (°C)")
    parser.add_argument("--temperature2", type=float, default=20.0, help="Water temperature at sensor 2 (°C)")
    parser.add_argument("--baseline1", default=None, help="Baseline (no-leak) WAV file for sensor 1")
    parser.add_argument("--baseline2", default=None, help="Baseline (no-leak) WAV file for sensor 2")
    parser.add_argument("--plot", action="store_true", help="Show spectrograms and plots")
    parser.add_argument("--low-freq", type=float, default=100, help="Low frequency for band power (Hz)")
    parser.add_argument("--high-freq", type=float, default=2000, help="High frequency for band power (Hz)")
    parser.add_argument("--debug", action="store_true", help="Enable debugging prints and plots")
    args = parser.parse_args()

    leak_detection_full_analysis(
        args.wav1, args.wav2,
        top_n=args.top,
        horizontal_distance=args.distance,
        depth1=args.depth1,
        depth2=args.depth2,
        pressure1=args.pressure1,
        pressure2=args.pressure2,
        temperature1=args.temperature1,
        temperature2=args.temperature2,
        baseline1=args.baseline1,
        baseline2=args.baseline2,
        plot=args.plot,
        freq_band=(args.low_freq, args.high_freq),
        debug=args.debug
    )