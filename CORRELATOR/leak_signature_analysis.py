import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, correlate
from scipy.stats import kurtosis, entropy

def load_wav(filename):
    data, samplerate = sf.read(filename)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, samplerate

def plot_spectrogram(signal, sr, title="Spectrogram", freq_lim=(0, 3000)):
    f, t, Sxx = spectrogram(signal, sr, nperseg=2048)
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(Sxx+1e-12), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.ylim(freq_lim)
    plt.colorbar(label='Intensity [dB]')
    plt.show()

def compute_band_power(signal, sr, band=(100, 2000)):
    # Welch's method
    freqs, psd = welch(signal, sr, nperseg=2048)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.sum(psd[band_mask])
    total_power = np.sum(psd)
    ratio = band_power / total_power if total_power > 0 else 0
    return band_power, total_power, ratio

def compute_statistics(signal):
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal)
    # Spectral entropy
    psd = np.abs(np.fft.fft(signal))**2
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    return {'rms': rms, 'kurtosis': kurt, 'spectral_entropy': spec_entropy}

def template_match(signal, template):
    # Cross-correlation, normalized
    template = (template - np.mean(template)) / (np.std(template) + 1e-12)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    corr = correlate(signal, template, mode='valid')
    corr /= len(template)
    max_corr = np.max(np.abs(corr))
    return max_corr, corr

def analyze_leak_signature(
    signal_file,
    baseline_file=None,
    template_file=None,
    plot=True,
    freq_band=(100, 2000)
):
    # --- Load signals
    signal, sr = load_wav(signal_file)
    print(f"[INFO] Primary signal loaded: {signal_file}")
    baseline = None
    if baseline_file:
        baseline, sr_base = load_wav(baseline_file)
        assert sr_base == sr, "Baseline sample rate mismatch!"
        print(f"[INFO] Baseline (no-leak) loaded: {baseline_file}")
    template = None
    if template_file:
        template, sr_t = load_wav(template_file)
        assert sr_t == sr, "Template sample rate mismatch!"
        print(f"[INFO] Leak template loaded: {template_file}")

    # --- Spectrogram visualization
    if plot:
        plot_spectrogram(signal, sr, title=f"Spectrogram: {signal_file}")

    # --- Band power analysis
    band_power, total_power, band_ratio = compute_band_power(signal, sr, band=freq_band)
    print(f"[INFO] Band power ({freq_band[0]}-{freq_band[1]} Hz): {band_power:.4e}, total: {total_power:.4e}, ratio: {band_ratio:.3f}")

    # --- Statistics
    stats = compute_statistics(signal)
    print(f"[INFO] Signal statistics: RMS={stats['rms']:.4f}, Kurtosis={stats['kurtosis']:.3f}, Spectral entropy={stats['spectral_entropy']:.3f}")

    # --- Baseline comparison
    if baseline is not None:
        baseline_band_power, baseline_total_power, baseline_band_ratio = compute_band_power(baseline, sr, band=freq_band)
        baseline_stats = compute_statistics(baseline)
        print(f"[INFO] Baseline band ratio: {baseline_band_ratio:.3f} | Signal ratio: {band_ratio:.3f}")
        print(f"[INFO] Baseline statistics: RMS={baseline_stats['rms']:.4f}, Kurtosis={baseline_stats['kurtosis']:.3f}, Spectral entropy={baseline_stats['spectral_entropy']:.3f}")
        if plot:
            plot_spectrogram(baseline, sr, title=f"Spectrogram: {baseline_file}")

    # --- Template matching
    if template is not None:
        max_corr, corr = template_match(signal, template)
        print(f"[INFO] Maximum normalized correlation with template: {max_corr:.3f}")
        if plot:
            plt.figure()
            plt.title("Template Cross-Correlation")
            plt.plot(corr)
            plt.xlabel("Sample lag")
            plt.ylabel("Normalized correlation")
            plt.show()

    # --- Detection heuristic
    leak_detected = False
    reason = ""
    if baseline is not None:
        if band_ratio > baseline_band_ratio * 1.2:
            leak_detected = True
            reason += f"Band power ratio elevated ({band_ratio:.2f} > {baseline_band_ratio:.2f}). "
        if stats['spectral_entropy'] < baseline_stats['spectral_entropy']:
            leak_detected = True
            reason += f"Spectral entropy lower than baseline ({stats['spectral_entropy']:.2f} < {baseline_stats['spectral_entropy']:.2f}). "
        if stats['rms'] > baseline_stats['rms'] * 1.2:
            leak_detected = True
            reason += f"RMS elevated ({stats['rms']:.2f} > {baseline_stats['rms']:.2f}). "
        if template is not None and max_corr > 0.3:
            leak_detected = True
            reason += f"High template match ({max_corr:.2f}). "
    else:
        # No baseline: rely on absolute thresholds
        if band_ratio > 0.4:
            leak_detected = True
            reason += f"Band power ratio high ({band_ratio:.2f}). "
        if template is not None and max_corr > 0.3:
            leak_detected = True
            reason += f"High template match ({max_corr:.2f}). "

    print(f"[RESULT] Leak detected: {leak_detected} | Reason: {reason if reason else 'No strong evidence.'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Leak Signature Analysis")
    parser.add_argument("signal", help="Signal WAV file (to analyze)")
    parser.add_argument("--baseline", default=None, help="Baseline (no-leak) WAV file")
    parser.add_argument("--template", default=None, help="Leak template WAV file (snippet of known leak sound)")
    parser.add_argument("--plot", action="store_true", help="Show spectrograms and plots")
    parser.add_argument("--low-freq", type=float, default=100, help="Low frequency for band power (Hz)")
    parser.add_argument("--high-freq", type=float, default=2000, help="High frequency for band power (Hz)")
    args = parser.parse_args()

    analyze_leak_signature(
        args.signal,
        baseline_file=args.baseline,
        template_file=args.template,
        plot=args.plot,
        freq_band=(args.low_freq, args.high_freq)
    )