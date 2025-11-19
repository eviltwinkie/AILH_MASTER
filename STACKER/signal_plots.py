# audio_analysis_plots.py

"""
Advanced and Modular Audio Analysis Plotting
============================================
This module provides high-level plotting utilities for WAV stacking,
feature analysis, and anomaly detection pipelines. All plots are saved as SVG
by default (for high quality and easy inclusion in reports).

Author: Your Name
Date: 2025-07-04

Requirements:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - plotly
    - scikit-learn
    - packaging
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo

from numpy.fft import rfft, rfftfreq
import matplotlib.gridspec as gridspec

from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
# from mpl_toolkits.mplot3d import Axes3D  # Removed: unused import
from packaging import version
import sklearn

# ---- Utility functions ----

def _ensure_dir(dir_path):
    """Create directory if it does not exist."""
    os.makedirs(dir_path, exist_ok=True)

def _save_and_close(fig, out_path, format="svg"):
    """Save matplotlib figure and close it."""
    fig.savefig(out_path, format=format, bbox_inches='tight')
    plt.close(fig)

# ---- Basic Plots ----

def plot_amplitudes(data_list, title="Amplitude Ranges", output_dir="plots", filename="amplitude_ranges.svg"):
    """Plot log-scaled peak amplitudes for each signal."""
    _ensure_dir(output_dir)
    peaks = [np.max(np.abs(d)) for d in data_list]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(peaks, marker='o')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_ylabel("Peak Amplitude (log scale)")
    ax.set_xlabel("File Index")
    ax.grid(True)
    _save_and_close(fig, os.path.join(output_dir, filename))

def plot_signals(signals, num_samples=1000, output_dir="plots", filename="signals.svg", aligned=False):
    """Plot the first N samples of a list of signals (before/after alignment)."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    label = "Aligned" if aligned else "Signal"
    for i, sig in enumerate(signals):
        ax.plot(sig[:num_samples], label=f"{label} {i}", alpha=0.7)
    ax.set_title(f"First {num_samples} Samples — {'After' if aligned else 'Before'} Alignment")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, filename))

def plot_stack_weights(weights, output_dir="plots", filename="stack_weights.svg"):
    """Plot bar chart of weights assigned to each signal."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(weights)), weights)
    ax.set_title("Correlation Weights Assigned to Each Signal")
    ax.set_xlabel("Signal Index")
    ax.set_ylabel("Weight")
    ax.grid(True)
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, filename))

def plot_stacked_signal(signal, output_dir="plots", filename="stacked_signal.svg"):
    """Plot the final stacked signal."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal)
    ax.set_title("Final Stacked Signal")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, filename))

# ---- Feature and Anomaly Analysis ----

def perform_pca_and_tsne(data_list, output_dir, prefix="analysis", lags=None):
    """
    Run PCA, t-SNE, and IsolationForest on stacked data.
    Saves PCA 2D/3D, t-SNE, lag histograms, and overlays.
    Returns: dict of results and anomaly mask.
    """
    _ensure_dir(output_dir)
    data_matrix = np.stack(data_list)
    results = {}

    # --- PCA ---
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_matrix)
    results["pca"] = pca_result

    # --- Anomaly detection ---
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    labels = clf.fit_predict(data_matrix)
    anomaly_mask = labels == -1

    # PCA 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, point in enumerate(pca_result[:, :2]):
        ax.scatter(*point, color='red' if anomaly_mask[i] else 'blue', alpha=0.7)
    ax.set(title="PCA Projection with Anomaly Overlay", xlabel="PC1", ylabel="PC2")
    ax.grid(True)
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_pca_2d.svg"))

    # PCA 3D
    from mpl_toolkits.mplot3d import Axes3D  # Import here to avoid unused import warning
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(pca_result)):
        ax.scatter(*pca_result[i, :3], color='red' if anomaly_mask[i] else 'blue', alpha=0.7)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3")
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_pca_3d.svg"))

    tsne = TSNE(n_components=2, perplexity=5, random_state=42, n_iter=1000)
    tsne_result = tsne.fit_transform(data_matrix)
    results["tsne"] = tsne_result

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, point in enumerate(tsne_result):
        ax.scatter(*point, color='red' if anomaly_mask[i] else 'blue', alpha=0.7)
    ax.set(title="t-SNE Projection with Anomaly Overlay", xlabel="t-SNE1", ylabel="t-SNE2")
    ax.grid(True)
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_tsne_2d.svg"))

    # --- Lags ---
    if lags is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(lags, bins=20, color='purple', alpha=0.7)
        ax.set(title="Histogram of Alignment Lags", xlabel="Lag (samples)", ylabel="Count")
        _save_and_close(fig, os.path.join(output_dir, f"{prefix}_lag_histogram.svg"))

        if len(lags) == len(pca_result):
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=lags, cmap="viridis", alpha=0.8)
            fig.colorbar(scatter, label="Lag (samples)")
            ax.set(title="PCA Projection Colored by Alignment Lag", xlabel="PC1", ylabel="PC2")
            _save_and_close(fig, os.path.join(output_dir, f"{prefix}_pca_lag_overlay.svg"))

    return results, anomaly_mask.tolist()

def generate_anomaly_summary(data_list, output_dir, prefix="analysis"):
    """Runs IsolationForest and saves anomaly summary report as txt."""
    _ensure_dir(output_dir)
    data_matrix = np.stack(data_list)
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    labels = clf.fit_predict(data_matrix)
    anomaly_mask = labels == -1
    summary = {
        "num_total": len(data_matrix),
        "num_anomalies": int(np.sum(anomaly_mask)),
        "anomaly_indices": np.where(anomaly_mask)[0].tolist()
    }
    summary_path = os.path.join(output_dir, f"{prefix}_anomaly_report.txt")
    with open(summary_path, "w") as f:
        for key, val in summary.items():
            f.write(f"{key}: {val}\n")
    return summary

# ---- Advanced Audio Plots ----

def plot_correlation_heatmap(data_list, output_dir=".", prefix="plot"):
    """Correlation matrix heatmap of all signals."""
    _ensure_dir(output_dir)
    corr_matrix = np.corrcoef(data_list)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set(title="Correlation Heatmap", xlabel="Sample Index", ylabel="Sample Index")
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_correlation_heatmap.svg"))

def plot_fft_overlay(data_list, samplerate, output_dir=".", prefix="plot"):
    """Plot overlay of FFTs of all signals."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, signal in enumerate(data_list):
        freqs = np.fft.rfftfreq(len(signal), 1/samplerate)
        fft_vals = np.abs(np.fft.rfft(signal))
        ax.plot(freqs, fft_vals, alpha=0.5, label=f"Signal {i+1}")
    ax.set(title="FFT Overlays", xlabel="Frequency (Hz)", ylabel="Amplitude")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_fft_overlay.svg"))

def plot_band_energy_comparison(data_list, samplerate, output_dir=".", prefix="plot", bands=None):
    """Compare energy in frequency bands for all signals."""
    _ensure_dir(output_dir)
    if bands is None:
        bands = [(0, 100), (100, 500), (500, 1000), (1000, 5000)]
    energies = []
    for signal in data_list:
        freqs = np.fft.rfftfreq(len(signal), 1/samplerate)
        fft_vals = np.abs(np.fft.rfft(signal))
        band_energies = [np.sum(fft_vals[(freqs >= low) & (freqs < high)] ** 2) for low, high in bands]
        energies.append(band_energies)
    df = pd.DataFrame(energies, columns=[f"{low}-{high}Hz" for low, high in bands])
    ax = df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    ax.set_title("Band Energy Comparison")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Energy")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_band_energy_comparison.svg"), format='svg')
    plt.close()
    # Use the last signal for the spectrogram
    signal_for_specgram = data_list[-1] if len(data_list) > 0 else np.zeros(1024)
    fig, ax = plt.subplots(figsize=(10, 6))
    _, _, _, im = ax.specgram(signal_for_specgram, Fs=samplerate, NFFT=1024, noverlap=512, cmap="viridis")
    ax.set(title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Intensity (dB)")
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_spectrogram.svg"))
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_spectrogram.svg"))

def plot_cross_correlation_lags(data_list, output_dir=".", prefix="plot"):
    """Plot lag (max correlation offset) of each signal vs first."""
    _ensure_dir(output_dir)
    base = data_list[0]
    lags = [np.argmax(correlate(base, data_list[i], mode='full')) - len(base) + 1 for i in range(1, len(data_list))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(data_list)), lags)
    ax.set(title="Cross-Correlation Lag Metrics", xlabel="Signal Index", ylabel="Lag (samples)")
    ax.grid(True)
    fig.tight_layout()
    _save_and_close(fig, os.path.join(output_dir, f"{prefix}_cross_correlation_lags.svg"))

def classify_signal_quality(snrs, threshold=10.0):
    """Label SNR quality as GOOD or POOR."""
    return ["GOOD" if snr >= threshold else "POOR" for snr in snrs]

def plot_interactive_fft_overlay(data_list, samplerate, output_dir=".", prefix="plot"):
    """Create interactive FFT overlay HTML plot (Plotly)."""
    _ensure_dir(output_dir)
    traces = []
    for i, signal in enumerate(data_list):
        freqs = np.fft.rfftfreq(len(signal), 1/samplerate)
        fft_vals = np.abs(np.fft.rfft(signal))
        traces.append(go.Scatter(x=freqs, y=fft_vals, mode="lines", name=f"Signal {i+1}"))
    layout = go.Layout(title="Interactive FFT Overlay", xaxis=dict(title="Frequency (Hz)"), yaxis=dict(title="Amplitude"))
    fig = go.Figure(data=traces, layout=layout)
    html_path = os.path.join(output_dir, f"{prefix}_interactive_fft_overlay.html")
    pyo.plot(fig, filename=html_path, auto_open=False)

# ---- WAV Stack Quality Summary ----

def plot_stack_report(
    stack_reference,
    stack_data_list,      # List of used signals, len K (K <= N)
    valid_data_list,     # List of candidate signals, len N
    samplerate,
    mask_all,            # np.array, shape (N,) dtype=bool -- True=used, False=outlier
    output_dir,
    prefix
):
    """
    Create a comprehensive stacking report with signal overlays, FFTs, and SNR distributions.
    All output is saved to output_dir with prefix.
    """

    # --- [A] Setup & Defensive Checks ---
    os.makedirs(output_dir, exist_ok=True)
    N = len(valid_data_list)
    K = len(stack_data_list)
    total = N
    used = int(np.sum(mask_all))
    anomalies = int(np.sum(~np.asarray(mask_all)))
    mask_all = np.asarray(mask_all)
    used_indices = np.flatnonzero(mask_all)
    outlier_indices = np.flatnonzero(~mask_all)

    # For consistent plotting, limit signals to the same length as the stack
    stack_len = len(stack_reference)
    valid_data_trimmed = [np.asarray(x)[:stack_len] for x in valid_data_list]
    stack_data_trimmed = [np.asarray(x)[:stack_len] for x in stack_data_list]

    # --- [B] FFT Calculations ---
    fft_freqs = rfftfreq(stack_len, 1.0 / samplerate)
    nonzero_idx = fft_freqs > 0  # exclude DC for log plots

    # All used signals: compute FFT (in dB) for overlays
    input_ffts = []
    for x in stack_data_trimmed:
        fft_vals = np.abs(rfft(x))
        fft_vals[fft_vals == 0] = 1e-12
        fft_db = 20 * np.log10(fft_vals)
        input_ffts.append(fft_db)
    input_ffts = np.array(input_ffts) if len(input_ffts) > 0 else None

    # Median FFT across used signals (in dB)
    if input_ffts is not None and len(input_ffts) > 0:
        median_input_fft_db = np.median(input_ffts, axis=0)
    else:
        median_input_fft_db = None

    # Stacked output FFT (in dB)
    stack_fft_vals = np.abs(rfft(stack_reference[:stack_len]))
    stack_fft_vals[stack_fft_vals == 0] = 1e-12
    stack_fft_db = 20 * np.log10(stack_fft_vals)

    # --- [C] Figure and Grid Layout ---
    nrows = 7
    fig = plt.figure(figsize=(18, 4 * nrows))
    gs = gridspec.GridSpec(nrows, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1])
    row = 0

    # --- [1] All input waveforms, colored by mask, stacked output ---
    ax = fig.add_subplot(gs[row, :])
    for i, x in enumerate(valid_data_trimmed):
        color = 'black' if mask_all[i] else 'red'
        ax.plot(x, color=color, alpha=0.5, linewidth=0.5, zorder=4 if color == 'black' else 5)
    ax.plot(stack_reference[:stack_len], color='dodgerblue', alpha=0.5, linewidth=0.5, zorder=6)
    ax.set_title(f'All Input Waveforms (Used={used}/{total} signals, {anomalies} outlier, total {total})', fontsize=15)
    ax.grid(True, linestyle=':', alpha=0.8)
    row += 1

    # --- [2] Used signals overlay ---
    ax = fig.add_subplot(gs[row, 0])
    for x in stack_data_trimmed:
        ax.plot(x, color='black', alpha=0.5, linewidth=0.5,)
    ax.set_title('Waveforms (Used in Stack)')
    ax.grid(True, linestyle=':', alpha=0.8)

    # --- [3] Output stack ---
    ax = fig.add_subplot(gs[row, 1])
    ax.plot(stack_reference[:stack_len], color='dodgerblue', alpha=0.9, linewidth=0.5)
    ax.set_title('Waveform (Stacked Output)')
    ax.grid(True, linestyle=':', alpha=0.8)
    row += 1

    # --- [4] FFT (median used vs output, linear) ---
    ax = fig.add_subplot(gs[row, :])
    if median_input_fft_db is not None:
        ax.plot(fft_freqs, 10 ** (median_input_fft_db / 20), color='black', alpha=0.8, linewidth=0.5, zorder=2, label='Median Used')
    ax.plot(fft_freqs, 10 ** (stack_fft_db / 20), color='dodgerblue', alpha=0.8, linewidth=0.5, zorder=1, label='Stacked Output')
    ax.set_title('Frequency Spectrum (Median Used + Output) - Linear Scale')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_xlim(0, samplerate / 2)
    ax.grid(True, linestyle=':', alpha=0.8)
    ax.legend()
    row += 1

    # --- [5] FFT overlays, all used signals (linear) ---
    ax = fig.add_subplot(gs[row, 0])
    if input_ffts is not None:
        for fft_db in input_ffts:
            ax.plot(fft_freqs, 10 ** (fft_db / 20), color='gray', alpha=0.32, linewidth=0.8)
        if median_input_fft_db is not None:
            ax.plot(fft_freqs, 10 ** (median_input_fft_db / 20), color='black', linewidth=1.1, label="Median Used")
    ax.set_title('Frequency Spectra (All Used Inputs)')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_xlim(0, samplerate / 2)
    ax.grid(True, linestyle=':', alpha=0.8)

    # --- [6] Output stack FFT (linear) ---
    ax = fig.add_subplot(gs[row, 1])
    ax.plot(fft_freqs, 10 ** (stack_fft_db / 20), color='dodgerblue', linewidth=1.2)
    ax.set_title('Frequency Spectrum (Stacked Output)')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_xlim(0, samplerate / 2)
    ax.grid(True, linestyle=':', alpha=0.8)
    row += 1

    # --- [7] FFT (median/input/output, log scale) ---
    ax = fig.add_subplot(gs[row, :])
    if median_input_fft_db is not None:
        ax.plot(fft_freqs[nonzero_idx], 10 ** (median_input_fft_db[nonzero_idx] / 20), color='black', alpha=0.8, linewidth=0.5, zorder=2, label='Median Used')
    ax.plot(fft_freqs[nonzero_idx], 10 ** (stack_fft_db[nonzero_idx] / 20), color='dodgerblue', alpha=0.8, linewidth=0.5, zorder=1, label='Stacked Output')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(fft_freqs[nonzero_idx][0], samplerate / 2)
    ax.set_title('Frequency Spectrum (Median Used + Output) - Log Scale')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (log scale)")
    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    ax.legend()
    row += 1

    # --- [8] FFT overlays (log) ---
    ax = fig.add_subplot(gs[row, 0])
    if input_ffts is not None:
        for fft_db in input_ffts:
            ax.plot(fft_freqs[nonzero_idx], 10 ** (fft_db[nonzero_idx] / 20), color='gray', alpha=0.13, linewidth=0.7)
        if median_input_fft_db is not None:
            ax.plot(fft_freqs[nonzero_idx], 10 ** (median_input_fft_db[nonzero_idx] / 20), color='black', linewidth=1, label="Median Used")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(fft_freqs[nonzero_idx][0], samplerate / 2)
    ax.set_title('Frequency Spectra (Used Inputs, Log Scale)')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (log scale)")
    ax.grid(True, which='both', linestyle=':', alpha=0.7)

    # --- [9] Output FFT (log) ---
    ax = fig.add_subplot(gs[row, 1])
    ax.plot(fft_freqs[nonzero_idx], 10 ** (stack_fft_db[nonzero_idx] / 20), color='dodgerblue', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(fft_freqs[nonzero_idx][0], samplerate / 2)
    ax.set_title('Frequency Spectrum (Output, Log Scale)')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude (log scale)")
    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    row += 1

    # --- Save and Print ---
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.suptitle(f"{prefix}: WAV Stacking Report\nInput: {output_dir}", fontsize=17, y=1.02)
    out_file_png = os.path.join(output_dir, f"{prefix}_stack_report.png")
    out_file_svg = os.path.join(output_dir, f"{prefix}_stack_report.svg")
    fig.savefig(out_file_png, dpi=50, bbox_inches='tight')
    fig.savefig(out_file_svg, dpi=50, bbox_inches='tight')
    plt.close(fig)

    with open(out_file_svg, "r") as f:
        svg = f.read()
    svg = re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"', r'\1 width="100%"', svg, count=1)
    svg = re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"', r'\1 ', svg, count=1)
    svg_path_responsive = os.path.splitext(out_file_svg)[0] + "_responsive.svg"
    with open(svg_path_responsive, "w") as f:
        f.write(svg)
    print(f"[✓] Saved responsive SVG to: {svg_path_responsive}")
