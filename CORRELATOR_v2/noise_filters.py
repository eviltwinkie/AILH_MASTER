#!/usr/bin/env python3
"""
CORRELATOR_v2 - Noise Filtering and Suppression Module

Advanced filtering techniques to eliminate common noise sources:
- Electrical hum (50/60 Hz and harmonics)
- Mechanical vibrations (pumps, motors)
- Traffic noise
- Environmental interference
- Low-frequency rumble
- High-frequency transients

Filter types:
1. Notch filters - Remove specific frequencies (electrical hum)
2. Adaptive filters - Learn and suppress stationary noise
3. Spectral subtraction - Remove noise spectrum
4. Wavelet denoising - Remove non-stationary noise
5. Comb filters - Remove periodic interference

Author: AILH Development Team
Date: 2025-11-19
Version: 3.2.0
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, iirnotch, firwin, lfilter, welch
import warnings


@dataclass
class NoiseProfile:
    """
    Noise profile for spectral subtraction.

    Attributes:
        noise_spectrum: Noise power spectrum
        frequencies: Frequency bins
        alpha: Over-subtraction factor (1.0-2.0)
    """
    noise_spectrum: np.ndarray
    frequencies: np.ndarray
    alpha: float = 1.5


class NoiseFilter:
    """
    Advanced noise filtering and suppression.

    Provides multiple filtering strategies for different noise types.
    Can be applied before correlation to improve leak detection accuracy.
    """

    def __init__(self, sample_rate: int = 4096, verbose: bool = False):
        """
        Initialize noise filter.

        Args:
            sample_rate: Audio sample rate (Hz)
            verbose: Print filtering details
        """
        self.sample_rate = sample_rate
        self.verbose = verbose

    def remove_electrical_hum(
        self,
        signal: np.ndarray,
        line_frequency: float = 60.0,
        n_harmonics: int = 5,
        quality_factor: float = 30.0
    ) -> np.ndarray:
        """
        Remove electrical hum and harmonics using notch filters.

        Common sources:
        - US: 60 Hz (and harmonics: 120, 180, 240, 300 Hz)
        - Europe/Asia: 50 Hz (and harmonics: 100, 150, 200, 250 Hz)

        Args:
            signal: Audio signal
            line_frequency: AC line frequency (50 or 60 Hz)
            n_harmonics: Number of harmonics to remove
            quality_factor: Notch filter Q factor (higher = narrower)

        Returns:
            Filtered signal
        """
        if self.verbose:
            print(f"\n[i] Removing {line_frequency} Hz electrical hum and {n_harmonics} harmonics")

        filtered = signal.copy()

        # Remove fundamental and harmonics
        for harmonic in range(1, n_harmonics + 1):
            freq = line_frequency * harmonic

            # Skip if frequency is above Nyquist
            if freq >= self.sample_rate / 2:
                break

            # Design notch filter
            b, a = iirnotch(freq, quality_factor, fs=self.sample_rate)

            # Apply filter (zero-phase)
            filtered = filtfilt(b, a, filtered)

            if self.verbose:
                print(f"    Removed {freq:.1f} Hz (harmonic {harmonic})")

        return filtered

    def remove_dc_offset(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from signal.

        Args:
            signal: Audio signal

        Returns:
            Zero-mean signal
        """
        return signal - np.mean(signal)

    def highpass_filter(
        self,
        signal: np.ndarray,
        cutoff_hz: float = 50.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency rumble.

        Removes:
        - Building vibrations
        - Traffic rumble
        - Wind noise
        - Microphone handling noise

        Args:
            signal: Audio signal
            cutoff_hz: Cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist

        if normalized_cutoff >= 1.0:
            warnings.warn(f"Cutoff frequency {cutoff_hz} Hz >= Nyquist {nyquist} Hz, skipping")
            return signal

        b, a = butter(order, normalized_cutoff, btype='high')
        filtered = filtfilt(b, a, signal)

        if self.verbose:
            print(f"[i] High-pass filtered @ {cutoff_hz} Hz (order {order})")

        return filtered

    def lowpass_filter(
        self,
        signal: np.ndarray,
        cutoff_hz: float = 2000.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.

        Removes:
        - Sensor electronics noise
        - Aliasing artifacts
        - Ultrasonic interference

        Args:
            signal: Audio signal
            cutoff_hz: Cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist

        if normalized_cutoff >= 1.0:
            warnings.warn(f"Cutoff frequency {cutoff_hz} Hz >= Nyquist {nyquist} Hz, skipping")
            return signal

        b, a = butter(order, normalized_cutoff, btype='low')
        filtered = filtfilt(b, a, signal)

        if self.verbose:
            print(f"[i] Low-pass filtered @ {cutoff_hz} Hz (order {order})")

        return filtered

    def bandstop_filter(
        self,
        signal: np.ndarray,
        low_hz: float,
        high_hz: float,
        order: int = 4
    ) -> np.ndarray:
        """
        Remove specific frequency band (band-stop/notch filter).

        Useful for removing known interference in a frequency range.

        Args:
            signal: Audio signal
            low_hz: Lower cutoff frequency
            high_hz: Upper cutoff frequency
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = self.sample_rate / 2
        low_norm = low_hz / nyquist
        high_norm = high_hz / nyquist

        if low_norm >= 1.0 or high_norm >= 1.0:
            warnings.warn(f"Cutoff frequencies exceed Nyquist, skipping")
            return signal

        b, a = butter(order, [low_norm, high_norm], btype='bandstop')
        filtered = filtfilt(b, a, signal)

        if self.verbose:
            print(f"[i] Band-stop filtered {low_hz}-{high_hz} Hz")

        return filtered

    def spectral_subtraction(
        self,
        signal: np.ndarray,
        noise_profile: NoiseProfile,
        beta: float = 0.001
    ) -> np.ndarray:
        """
        Spectral subtraction noise reduction.

        Estimates and subtracts noise spectrum from signal.
        Good for stationary background noise.

        Args:
            signal: Audio signal
            noise_profile: Noise profile (from noise-only recording)
            beta: Spectral floor parameter (prevents over-subtraction)

        Returns:
            Denoised signal
        """
        if self.verbose:
            print(f"[i] Applying spectral subtraction (alpha={noise_profile.alpha:.2f})")

        # STFT parameters
        n_fft = 512
        hop_length = n_fft // 4

        # Compute STFT of signal
        stft = self._stft(signal, n_fft, hop_length)

        # Get magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Subtract noise spectrum (with over-subtraction)
        noise_mag = np.mean(noise_profile.noise_spectrum)
        magnitude_sub = magnitude - noise_profile.alpha * noise_mag

        # Apply spectral floor
        magnitude_sub = np.maximum(magnitude_sub, beta * magnitude)

        # Reconstruct with original phase
        stft_sub = magnitude_sub * np.exp(1j * phase)

        # Inverse STFT
        denoised = self._istft(stft_sub, hop_length)

        # Match original length
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)))

        return denoised

    def create_noise_profile(
        self,
        noise_signal: np.ndarray,
        alpha: float = 1.5
    ) -> NoiseProfile:
        """
        Create noise profile from noise-only recording.

        Args:
            noise_signal: Recording of noise without leak
            alpha: Over-subtraction factor (1.0-2.0)

        Returns:
            NoiseProfile for spectral subtraction
        """
        # Compute average power spectrum
        freqs, psd = welch(noise_signal, fs=self.sample_rate, nperseg=512)

        return NoiseProfile(
            noise_spectrum=psd,
            frequencies=freqs,
            alpha=alpha
        )

    def adaptive_filter(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        filter_order: int = 32,
        mu: float = 0.01
    ) -> np.ndarray:
        """
        Adaptive LMS filter for noise cancellation.

        Learns noise characteristics from reference signal and removes
        correlated noise from primary signal.

        Args:
            signal: Primary signal (leak + noise)
            reference: Reference signal (noise only, e.g., from second sensor far from leak)
            filter_order: Adaptive filter order
            mu: Learning rate (0.001-0.1)

        Returns:
            Filtered signal
        """
        if self.verbose:
            print(f"[i] Adaptive filtering (order={filter_order}, mu={mu})")

        # Ensure signals are same length
        n = min(len(signal), len(reference))
        signal = signal[:n]
        reference = reference[:n]

        # Initialize filter weights
        w = np.zeros(filter_order)
        output = np.zeros(n)

        # LMS algorithm
        for i in range(filter_order, n):
            # Extract reference window
            x = reference[i - filter_order:i][::-1]

            # Predict noise
            y = np.dot(w, x)

            # Error (desired signal)
            e = signal[i] - y

            # Update weights
            w = w + mu * e * x

            output[i] = e

        return output

    def comb_filter(
        self,
        signal: np.ndarray,
        fundamental_hz: float,
        n_harmonics: int = 10,
        bandwidth_hz: float = 5.0
    ) -> np.ndarray:
        """
        Comb filter to remove periodic interference.

        Removes fundamental frequency and harmonics (e.g., motor hum).

        Args:
            signal: Audio signal
            fundamental_hz: Fundamental frequency to remove
            n_harmonics: Number of harmonics to remove
            bandwidth_hz: Bandwidth of each notch

        Returns:
            Filtered signal
        """
        if self.verbose:
            print(f"[i] Comb filter @ {fundamental_hz} Hz + {n_harmonics} harmonics")

        filtered = signal.copy()

        for i in range(1, n_harmonics + 1):
            freq = fundamental_hz * i

            if freq >= self.sample_rate / 2:
                break

            # Calculate Q factor from bandwidth
            Q = freq / bandwidth_hz

            # Apply notch filter
            b, a = iirnotch(freq, Q, fs=self.sample_rate)
            filtered = filtfilt(b, a, filtered)

        return filtered

    def median_filter(
        self,
        signal: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Median filter for impulse noise removal.

        Removes sharp transients and clicks while preserving edges.

        Args:
            signal: Audio signal
            kernel_size: Filter kernel size (odd number)

        Returns:
            Filtered signal
        """
        from scipy.ndimage import median_filter as scipy_median

        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd

        filtered = scipy_median(signal, size=kernel_size)

        if self.verbose:
            print(f"[i] Median filtered (kernel={kernel_size})")

        return filtered

    def wiener_filter(
        self,
        signal: np.ndarray,
        noise_power_estimate: Optional[float] = None
    ) -> np.ndarray:
        """
        Wiener filtering for optimal noise reduction.

        Minimizes mean square error between clean and noisy signal.

        Args:
            signal: Audio signal
            noise_power_estimate: Estimated noise power (auto-estimated if None)

        Returns:
            Filtered signal
        """
        from scipy.signal import wiener as scipy_wiener

        if noise_power_estimate is None:
            # Estimate noise from quiet sections
            noise_power_estimate = np.var(signal[:int(0.1 * len(signal))])

        filtered = scipy_wiener(signal, mysize=None, noise=noise_power_estimate)

        if self.verbose:
            print(f"[i] Wiener filtered (noise power={noise_power_estimate:.6f})")

        return filtered

    def apply_default_pipeline(
        self,
        signal: np.ndarray,
        line_frequency: float = 60.0,
        remove_dc: bool = True,
        highpass_cutoff: float = 50.0,
        lowpass_cutoff: float = 2000.0
    ) -> np.ndarray:
        """
        Apply default recommended filtering pipeline.

        Pipeline:
        1. Remove DC offset
        2. Remove electrical hum (60 Hz and harmonics)
        3. High-pass filter (remove low-freq rumble)
        4. Low-pass filter (remove high-freq noise)

        Args:
            signal: Audio signal
            line_frequency: AC line frequency (50 or 60 Hz)
            remove_dc: Remove DC offset
            highpass_cutoff: High-pass cutoff (0 to disable)
            lowpass_cutoff: Low-pass cutoff (0 to disable)

        Returns:
            Filtered signal
        """
        if self.verbose:
            print("\n[i] Applying default noise filtering pipeline")

        filtered = signal.copy()

        # 1. Remove DC offset
        if remove_dc:
            filtered = self.remove_dc_offset(filtered)

        # 2. Remove electrical hum
        filtered = self.remove_electrical_hum(
            filtered,
            line_frequency=line_frequency,
            n_harmonics=5
        )

        # 3. High-pass filter
        if highpass_cutoff > 0:
            filtered = self.highpass_filter(filtered, cutoff_hz=highpass_cutoff)

        # 4. Low-pass filter
        if lowpass_cutoff > 0 and lowpass_cutoff < self.sample_rate / 2:
            filtered = self.lowpass_filter(filtered, cutoff_hz=lowpass_cutoff)

        if self.verbose:
            print("[✓] Filtering pipeline complete")

        return filtered

    def _stft(self, signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
        """Short-Time Fourier Transform."""
        # Simple STFT implementation
        n_frames = 1 + (len(signal) - n_fft) // hop_length
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft

            if end > len(signal):
                break

            frame = signal[start:end]
            frame = frame * np.hanning(n_fft)  # Window
            spectrum = np.fft.rfft(frame)
            stft[:, i] = spectrum

        return stft

    def _istft(self, stft: np.ndarray, hop_length: int) -> np.ndarray:
        """Inverse Short-Time Fourier Transform."""
        n_fft = (stft.shape[0] - 1) * 2
        n_frames = stft.shape[1]
        signal_length = (n_frames - 1) * hop_length + n_fft

        signal = np.zeros(signal_length)
        window_sum = np.zeros(signal_length)

        window = np.hanning(n_fft)

        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft

            frame = np.fft.irfft(stft[:, i], n=n_fft)
            signal[start:end] += frame * window
            window_sum[start:end] += window ** 2

        # Normalize by window overlap
        mask = window_sum > 1e-10
        signal[mask] /= window_sum[mask]

        return signal


# ==============================================================================
# MAIN - Testing
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("NOISE FILTERS TEST")
    print("=" * 80)

    # Generate test signal
    sample_rate = 4096
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Clean leak signal (800 Hz)
    leak_signal = np.sin(2 * np.pi * 800 * t) * 0.5

    # Add various noise sources
    noisy_signal = leak_signal.copy()

    # 1. 60 Hz electrical hum
    noisy_signal += 0.2 * np.sin(2 * np.pi * 60 * t)
    noisy_signal += 0.1 * np.sin(2 * np.pi * 120 * t)  # 2nd harmonic

    # 2. DC offset
    noisy_signal += 0.1

    # 3. Low-frequency rumble
    noisy_signal += 0.15 * np.sin(2 * np.pi * 30 * t)

    # 4. White noise
    noisy_signal += np.random.randn(len(t)) * 0.05

    print("\n[i] Test signal created:")
    print(f"    Clean signal: 800 Hz sine wave")
    print(f"    Noise sources: 60 Hz hum, DC offset, rumble, white noise")

    # Test filters
    filter = NoiseFilter(sample_rate=sample_rate, verbose=True)

    print("\n" + "=" * 80)
    print("\n[TEST 1] Remove DC offset")
    dc_removed = filter.remove_dc_offset(noisy_signal)
    print(f"    DC before: {np.mean(noisy_signal):.6f}")
    print(f"    DC after: {np.mean(dc_removed):.6f}")

    print("\n" + "=" * 80)
    print("\n[TEST 2] Remove electrical hum")
    hum_removed = filter.remove_electrical_hum(dc_removed, line_frequency=60.0)

    print("\n" + "=" * 80)
    print("\n[TEST 3] High-pass filter (remove rumble)")
    highpass = filter.highpass_filter(hum_removed, cutoff_hz=50.0)

    print("\n" + "=" * 80)
    print("\n[TEST 4] Default pipeline")
    filtered = filter.apply_default_pipeline(
        noisy_signal,
        line_frequency=60.0,
        highpass_cutoff=50.0,
        lowpass_cutoff=2000.0
    )

    # Calculate SNR improvement
    noise_power_before = np.var(noisy_signal - leak_signal)
    noise_power_after = np.var(filtered - leak_signal)
    snr_improvement = 10 * np.log10(noise_power_before / (noise_power_after + 1e-12))

    print(f"\n[✓] Noise filtering tests complete")
    print(f"    SNR improvement: {snr_improvement:.1f} dB")
