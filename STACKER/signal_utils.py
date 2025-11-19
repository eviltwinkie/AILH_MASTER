import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt, sosfilt, hilbert, stft, istft, medfilt, correlate, firwin, filtfilt
from scipy.optimize import minimize_scalar
from PyEMD import EMD
import matplotlib.pyplot as plt

# === UTILITY FUNCTIONS ===

def stereo_to_mono(data):
    #print(f"[✓] Converted stereo to mono")
    if data.ndim > 1:
        data = np.mean(data, axis=1)        
    return data

def remove_dc_offset(data):
    #print(f"[✓] Remove mean (DC offset) from a signal.")
    data = to_ndarray(data)
    return data - np.mean(data)

def rms_normalize(signal):
    #print(f"[✓] Normalizing signal RMS to {target_rms:.2f}")
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        return signal / (rms + 1e-12)
    else:
        return signal

def normalize_signal(signal):
    #print(f"[✓] Normalizing signal to -1..1 range")
    signal = to_ndarray(signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak
    else:
        return signal.copy()  
    
def compute_snr(signal, reference, eps=1e-12):
    """
    Compute SNR (in dB) between signal and reference.
    Returns 0.0 if signal_power or noise_power is zero or negative, or if inputs are empty.
    """
    signal = np.asarray(signal)
    reference = np.asarray(reference)
    if signal.size == 0 or reference.size == 0:
        return 0.0
    noise = signal - reference
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    if signal_power <= 0 or noise_power <= 0:
        return 0.0
    return 10 * np.log10(signal_power / max(noise_power, eps))

def correct_for_gain_db(data, gain_db):
    """
    Undo hardware gain (dB) applied to audio data.
    Example: data with +80 dB gain -> multiply by 10^(-80/20) = 0.0001
    """
    try:
        gain_db = float(gain_db)
    except ValueError:
        raise ValueError(f"[✗] Invalid gain value (not a float): {gain_db}")
    scaling_factor = 10 ** (-gain_db / 20)
    #print(f"[✓] Correcting for gain: {gain_db} dB -> linear gain factor: {scaling_factor:.10f}")
    return data * scaling_factor

def scaled_gain(data, gain_db):
    # Map gain in [0, 200] to dB in [10, -10]
    # mapped_db = 10 - 0.1 * gain_db
    # Map gain in [0, 200] to dB in [1, -1]
    mapped_db = 1 - 0.01 * gain_db
    #print(f"Mapped gain (dB): {mapped_db:.2f} dB")
    # Convert dB to amplitude scaling factor
    gain_correction = 10 ** (mapped_db / 20)
    #print(f"Amplitude correction: {gain_correction:.8f}")
    return data * gain_correction

def to_ndarray(data):
    """Ensure the input is a numpy ndarray."""
    return np.asarray(data)

def butter_bandpass(lowcut, highcut, fs, order=6):
    """Return SOS for bandpass filter."""
    return butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')

def butter_highpass(cutoff, fs, order=6):
    """Return SOS for highpass filter."""
    return butter(order, cutoff, btype='highpass', fs=fs, output='sos')

def butter_lowpass(cutoff, fs, order=6):
    """Return SOS for lowpass filter."""
    return butter(order, cutoff, btype='lowpass', fs=fs, output='sos')

def fir_bandpass_filter(data, lowcut, highcut, fs, numtaps=801):
    """FIR bandpass filter with sharp cutoff and linear phase using filtfilt."""
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return filtfilt(taps, 1.0, data)

def fir_highpass_filter(data, cutoff, fs, numtaps=801):
    taps = firwin(numtaps, cutoff, pass_zero=False, fs=fs)
    return filtfilt(taps, 1.0, data)

def fir_lowpass_filter(data, cutoff, fs, numtaps=801):
    taps = firwin(numtaps, cutoff, pass_zero=True, fs=fs)
    return filtfilt(taps, 1.0, data)

def apply_filter(
    data, 
    fs, 
    filter_type=None, 
    lowcut=None, 
    highcut=None, 
    order=6, 
    ftype="butter", 
    numtaps=512
):
    """
    Apply bandpass/highpass/lowpass filter to signal data.
    Choose between 'butter' (IIR Butterworth, default) and 'fir' (FIR, linear phase).
    """
    #print(f"[✓] Applying {ftype} {filter_type} filter: lowcut={lowcut}, highcut={highcut}, fs={fs}, order={order}, numtaps={numtaps}")
    data = to_ndarray(data)

    if ftype == "fir":
        if filter_type == 'bandpass' and lowcut is not None and highcut is not None:
            return fir_bandpass_filter(data, lowcut, highcut, fs, numtaps=numtaps)
        elif filter_type == 'highpass' and lowcut is not None:
            return fir_highpass_filter(data, lowcut, fs, numtaps=numtaps)
        elif filter_type == 'lowpass' and highcut is not None:
            return fir_lowpass_filter(data, highcut, fs, numtaps=numtaps)
    else:  # butterworth IIR by default
        if filter_type == 'bandpass' and lowcut is not None and highcut is not None:
            sos = butter_bandpass(lowcut, highcut, fs, order=order)
            return sosfilt(sos, data)
        elif filter_type == 'highpass' and lowcut is not None:
            sos = butter_highpass(lowcut, fs, order=order)
            return sosfilt(sos, data)
        elif filter_type == 'lowpass' and highcut is not None:
            sos = butter_lowpass(highcut, fs, order=order)
            return sosfilt(sos, data)

    # No filter applied
    return data

# === STACKING, ALIGNMENT, OUTLIERS ===

def correlation_matrix_outlier_rejection(
    signals,
    mode="auto",   # "mad", "percentile", or "auto"
    mad_k=2.0,
    perc=10,
    verbose=False
):
    if signals is None or len(signals) == 0:
        return np.array([], dtype=bool), np.array([]), None
    if len(signals) == 1:
        return np.array([True]), np.array([1.0]), 1.0

    X = np.vstack([np.asarray(s) for s in signals])
    n_signals = X.shape[0]
    stds = np.std(X, axis=1)
    constant_mask = stds < 1e-8
    if np.any(constant_mask) and verbose:
        print(f"[!] Warning: {np.sum(constant_mask)} signals are nearly constant (std < 1e-8).")

    X_for_corr = X.copy()
    X_for_corr[constant_mask] = np.nan

    with np.errstate(invalid='ignore'):
        cor_matrix = np.corrcoef(X_for_corr)

    mean_corrs = np.full(n_signals, np.nan)
    for i in range(n_signals):
        row = cor_matrix[i]
        vals = row[np.arange(n_signals) != i]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            mean_corrs[i] = np.mean(vals)
        else:
            mean_corrs[i] = np.nan

    # === NAN-FREE GUARANTEE ===
    nan_mask = ~np.isfinite(mean_corrs)
    if np.any(nan_mask) and verbose:
        print(f"[!] Replacing {np.sum(nan_mask)} NaN mean correlations with -1.0.")
    mean_corrs[nan_mask] = -1.0

    valid_corrs = mean_corrs
    median = np.median(valid_corrs)
    mad = np.median(np.abs(valid_corrs - median))
    mean = np.mean(valid_corrs)
    std = np.std(valid_corrs)
    iqr = np.percentile(valid_corrs, 75) - np.percentile(valid_corrs, 25)

    if verbose:
        print(f"[auto] Mean correlations: mean={mean:.4f}, median={median:.4f}, std={std:.4f}, MAD={mad:.4f}, IQR={iqr:.4f}, min={np.min(valid_corrs):.4f}, max={np.max(valid_corrs):.4f}")

    if mode == "auto":
        if std > 0.15 * mean and iqr > 0.1 * mean:
            selected_mode = "percentile"
        else:
            selected_mode = "mad"
        if verbose:
            print(f"[auto] Selected threshold mode: {selected_mode}")
    else:
        selected_mode = mode

    if selected_mode == "mad":
        threshold = median - mad_k * mad
        if verbose:
            print(f"[auto] Using MAD: threshold = median - {mad_k}*MAD = {threshold:.4f}")
    elif selected_mode == "percentile":
        threshold = np.percentile(valid_corrs, perc)
        if verbose:
            print(f"[auto] Using percentile({perc}): threshold = {threshold:.4f}")
    else:
        raise ValueError("mode must be 'mad', 'percentile', or 'auto'")

    mask = mean_corrs >= threshold

    return mask, mean_corrs, threshold

def correlation_weighted_denoising(signals, reference):
    """
    Stack signals using correlation with the mean reference as weights.

    Args:
        signals: list or np.ndarray, shape (n_signals, n_samples)
        reference: np.ndarray, shape (n_samples,)

    Returns:
        stacked: weighted average (np.ndarray, 1D)
        weights: weights used for each signal (np.ndarray, 1D)
        corrs: correlation values for each signal (np.ndarray, 1D)
    """

    # Convert all signals and reference to numpy arrays, handle NaNs
    signals = [np.nan_to_num(np.asarray(s)) for s in signals]
    reference = np.nan_to_num(np.asarray(reference))

    # Check that all signals and reference are the same length
    signal_lengths = [len(s) for s in signals]
    assert all(l == len(reference) for l in signal_lengths), (
        f"All signals and reference must be the same length. "
        f"Signal lengths: {signal_lengths}, reference length: {len(reference)}"
    )

    X = np.vstack(signals)
    corrs = []
    for s in X:
        try:
            c = np.corrcoef(s, reference)[0, 1]
            if not np.isfinite(c):
                c = 0.0
        except Exception:
            c = 0.0
        corrs.append(c)
    corrs = np.array(corrs)
    # Clip negative correlations to zero
    weights = np.maximum(corrs, 0)
    # Normalize weights
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    stacked = np.average(X, axis=0, weights=weights)

    return stacked, weights, corrs

def correlation_weighted_stack(signals, reference):
    """
    Stack aligned signals using correlation-based weights.
    """
    weights = []
    for s in signals:
        corr = np.corrcoef(s, reference)[0, 1]
        w = corr if np.isfinite(corr) and corr > 0 else 0.0
        weights.append(w)
    weights = np.array(weights)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    stacked = np.average(signals, axis=0, weights=weights)
    print(f"[✓] Correlation-weighted stacking: {len(signals)} signals stacked")    
    # print(f"[✓] Weights ({len(weights)}) used for stacking:")
    # print("-" * 26)
    # sample_weights = ""
    # for idx, w in enumerate(weights):
    #     sample_weights += f"[{idx}: {w*100:3.1f}%]"
    # print(sample_weights)
    # print("-" * 26)
    return stacked
