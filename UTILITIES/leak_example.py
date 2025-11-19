import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate, butter, sosfiltfilt
from numpy.fft import rfft, rfftfreq
import pandas as pd
import ace_tools as tools

# Generate synthetic leak-like signal (broadband noise)
fs = 44100  # Sample rate
duration = 2.0  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Simulated leak: broadband white noise filtered into 1-5 kHz
white_noise = np.random.normal(0, 1, t.shape)
sos = butter(6, [1000, 5000], btype='bandpass', fs=fs, output='sos')
leak_signal = sosfiltfilt(sos, white_noise)

# Compute FFT
fft_vals = np.abs(rfft(leak_signal))
fft_freqs = rfftfreq(len(leak_signal), 1/fs)

# Compute spectrogram
f, time_spec, Sxx = spectrogram(leak_signal, fs, nperseg=1024)

# Compute autocorrelation
autocorr = correlate(leak_signal, leak_signal, mode='full')
lags = np.arange(-len(leak_signal)+1, len(leak_signal))

# Plot all
fig, axs = plt.subplots(4, 1, figsize=(12, 14))

# Time domain
axs[0].plot(t[:2000], leak_signal[:2000])
axs[0].set_title("Leak Signal - Time Domain (First 2000 samples)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")

# Frequency domain
axs[1].plot(fft_freqs, 20 * np.log10(fft_vals + 1e-12))
axs[1].set_title("Leak Signal - Frequency Domain (FFT in dB)")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("Amplitude [dB]")
axs[1].set_xlim(0, 10000)
axs[1].grid(True)

# Spectrogram
im = axs[2].pcolormesh(time_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
axs[2].set_title("Leak Signal - Spectrogram")
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Frequency [Hz]")
axs[2].set_ylim(0, 10000)
fig.colorbar(im, ax=axs[2], label='Power [dB]')

# Autocorrelation
axs[3].plot(lags, autocorr)
axs[3].set_title("Leak Signal - Autocorrelation")
axs[3].set_xlabel("Lag")
axs[3].set_ylabel("Correlation")

plt.tight_layout()
plt.show()

# --- Compute some signal summary statistics and display as DataFrame ---
analysis = {
    "Mean": [np.mean(leak_signal)],
    "StdDev": [np.std(leak_signal)],
    "Max": [np.max(leak_signal)],
    "Min": [np.min(leak_signal)],
    "RMS": [np.sqrt(np.mean(leak_signal**2))],
}
df = pd.DataFrame(analysis)
tools.display_dataframe_to_user(name="Leak Signal Analysis", dataframe=df)
