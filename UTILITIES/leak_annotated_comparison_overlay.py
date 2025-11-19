import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate, butter, sosfiltfilt
from numpy.fft import rfft, rfftfreq

# Parameters
fs = 4096           # Sample rate matches your hydrophone WAV files
duration = 10.0     # 10 seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# LEAK SIGNAL: broadband white noise, bandpassed for "leak" energy (150 Hz–1600 Hz, typical for water leaks)
leak_noise = np.random.normal(0, 1, t.shape)
sos_leak = butter(6, [150, 1600], btype='bandpass', fs=fs, output='sos')
leak_signal = sosfiltfilt(sos_leak, leak_noise)
leak_signal = leak_signal / np.max(np.abs(leak_signal)) * 0.6   # normalize for 16-bit like amplitude

leak_fft = np.abs(rfft(leak_signal))
leak_freqs = rfftfreq(len(leak_signal), 1/fs)
f_leak, t_leak, Sxx_leak = spectrogram(leak_signal, fs, nperseg=512)
autocorr_leak = correlate(leak_signal, leak_signal, mode='full')
lags_leak = np.arange(-len(leak_signal)+1, len(leak_signal))

# QUIET BACKGROUND: low-level filtered noise (below 250 Hz, typical for ambient pipe hum)
quiet_noise = np.random.normal(0, 0.10, t.shape)
sos_quiet = butter(4, [10, 250], btype='bandpass', fs=fs, output='sos')
quiet_signal = sosfiltfilt(sos_quiet, quiet_noise)
quiet_signal = quiet_signal / np.max(np.abs(quiet_signal)) * 0.2

fft_quiet = np.abs(rfft(quiet_signal))
freqs_quiet = rfftfreq(len(quiet_signal), 1/fs)
f_q, t_q, Sxx_q = spectrogram(quiet_signal, fs, nperseg=512)
autocorr_q = correlate(quiet_signal, quiet_signal, mode='full')
lags_q = np.arange(-len(quiet_signal)+1, len(quiet_signal))

# MECHANICAL INTERFERENCE: pulsed low-freq sine (e.g. valve/pump) + some noise, with on/off bursts
mech_freq = 36   # Hz, common for electric pump or valve
mechanical_pulse = np.zeros_like(t)
burst_period = 1.2  # seconds between mechanical bursts
burst_len = 0.3     # duration of each burst (seconds)

# Generate mechanical bursts
burst_samples = int(burst_len * fs)
burst_onsets = np.arange(0, duration, burst_period)
for onset in burst_onsets:
    idx = int(onset * fs)
    if idx + burst_samples < len(t):
        mechanical_pulse[idx:idx+burst_samples] += np.sin(2 * np.pi * mech_freq * t[idx:idx+burst_samples])
mechanical_pulse += np.random.normal(0, 0.06, t.shape)  # add some low-level background noise
mechanical_signal = mechanical_pulse / np.max(np.abs(mechanical_pulse)) * 0.5

fft_mech = np.abs(rfft(mechanical_signal))
freqs_mech = rfftfreq(len(mechanical_signal), 1/fs)
f_m, t_m, Sxx_m = spectrogram(mechanical_signal, fs, nperseg=512)
autocorr_m = correlate(mechanical_signal, mechanical_signal, mode='full')
lags_m = np.arange(-len(mechanical_signal)+1, len(mechanical_signal))

# Store all signals and metadata in a dictionary
all_signals = {
    "Leak Signal": (leak_signal, leak_freqs, leak_fft, f_leak, t_leak, Sxx_leak, lags_leak, autocorr_leak),
    "Quiet Background": (quiet_signal, freqs_quiet, fft_quiet, f_q, t_q, Sxx_q, lags_q, autocorr_q),
    "Mechanical Interference": (mechanical_signal, freqs_mech, fft_mech, f_m, t_m, Sxx_m, lags_m, autocorr_m),
}

# Plot (show the first 2 seconds only for time domain view for clarity)
fig, axs = plt.subplots(4, 3, figsize=(18, 14))
titles = ["Leak Signal", "Quiet Background", "Mechanical Interference"]
signals = [leak_signal, quiet_signal, mechanical_signal]
ffts = [(leak_freqs, leak_fft), (freqs_quiet, fft_quiet), (freqs_mech, fft_mech)]
spectros = [(t_leak, f_leak, Sxx_leak), (t_q, f_q, Sxx_q), (t_m, f_m, Sxx_m)]
autocorrs = [(lags_leak, autocorr_leak), (lags_q, autocorr_q), (lags_m, autocorr_m)]



num_signals = len(titles)
fig, axs = plt.subplots(5, num_signals, figsize=(6*num_signals, 18))  # 5 rows: short time, full time, FFT, spectro, autocorr

for i in range(num_signals):
    # Short time domain (first 2s)
    axs[0][i].plot(t[:2*fs], signals[i][:2*fs])
    axs[0][i].set_title(f"{titles[i]} - Time Domain (First 2s)")
    axs[0][i].set_xlabel("Time [s]")
    axs[0][i].set_ylabel("Amplitude")

    # Full time domain (all 10s)
    axs[1][i].plot(t, signals[i])
    axs[1][i].set_title(f"{titles[i]} - Time Domain (Full 10s)")
    axs[1][i].set_xlabel("Time [s]")
    axs[1][i].set_ylabel("Amplitude")

    # FFT
    axs[2][i].plot(ffts[i][0], 20 * np.log10(ffts[i][1] + 1e-12))
    axs[2][i].set_title(f"{titles[i]} - FFT Spectrum")
    axs[2][i].set_xlim(0, fs//2)
    axs[2][i].set_xlabel("Frequency [Hz]")
    axs[2][i].set_ylabel("dB")
    if i == 0:
        axs[2][i].annotate("Leak band", xy=(700, 0), xytext=(1200, 10),
                           arrowprops=dict(facecolor='red', shrink=0.05), color='red')
    if i == 2:
        axs[2][i].annotate("Mechanical peak", xy=(mech_freq, 10), xytext=(mech_freq+150, 25),
                           arrowprops=dict(facecolor='purple', shrink=0.05), color='purple')

    # Spectrogram
    axs[3][i].pcolormesh(spectros[i][0], spectros[i][1], 10 * np.log10(spectros[i][2] + 1e-12), shading='gouraud')
    axs[3][i].set_ylim(0, fs//2)
    axs[3][i].set_title(f"{titles[i]} - Spectrogram")
    axs[3][i].set_xlabel("Time [s]")
    axs[3][i].set_ylabel("Frequency [Hz]")
    if i == 0:
        axs[3][i].annotate("Broadband leak energy", xy=(3, 600), xytext=(7, 1200),
                           arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')
    if i == 2 and len(burst_onsets) > 2:
        axs[3][i].annotate("Pulsed energy", xy=(burst_onsets[1], mech_freq), xytext=(burst_onsets[2], 400),
                           arrowprops=dict(facecolor='green', shrink=0.05), color='green')

    # Autocorrelation
    axs[4][i].plot(autocorrs[i][0], autocorrs[i][1])
    axs[4][i].set_title(f"{titles[i]} - Autocorrelation")
    axs[4][i].set_xlabel("Lag")
    axs[4][i].set_ylabel("Correlation")
    if i == 2:
        lag_index = len(autocorrs[i][1])//2 + int(fs*burst_period)
        if lag_index < len(autocorrs[i][1]):
            axs[4][i].annotate("Regular peaks (mechanical period)",
                               xy=(int(fs*burst_period), autocorrs[i][1][lag_index]),
                               xytext=(int(fs*burst_period)+5000, 0.5),
                               arrowprops=dict(facecolor='purple', shrink=0.05), color='purple')

plt.tight_layout()
plt.show()

# Optionally: Save figure (uncomment)
# fig.savefig("comparison_overlay_4096hz.png")
