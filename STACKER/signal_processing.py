import numpy as np
from scipy.signal import find_peaks, correlate
import warnings

def fast_autocorr(x):
    x = np.asarray(x)
    N = len(x)
    if N == 0:
        return np.zeros(0)
    nfft = 1 << (2*N-1).bit_length()
    X = np.fft.fft(x, n=nfft)
    ac = np.fft.ifft(X * np.conj(X)).real
    ac = ac[:N]
    return np.concatenate((ac[::-1][:-1], ac))

def suggest_packet_check_range(
    signal, 
    sample_rate, 
    min_period=0.05, 
    max_period=2.0, 
    min_peak_height_ratio=0.05,
    plot=False,
    verbose=False,
):
    """
    Suggest a (start_lag, end_lag) tuple for packet_check_range based on autocorrelation peaks.
    """
    signal = np.asarray(signal)
    if verbose: print("[*] Running packet_check_range suggestion...")
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("Input signal must be a 1D non-empty array.")
    ac = correlate(signal, signal, mode='full')
    if verbose: print(f"[+] Performed autocorrelation, array shape: {ac.shape}")
    ac = ac[ac.size // 2:]
    min_lag = int(min_period * sample_rate)
    max_lag = min(int(max_period * sample_rate), len(ac) - 1)
    if verbose: print(f"[+] Lag search range: {min_lag} - {max_lag}")
    search_ac = ac[min_lag:max_lag]
    if search_ac.size == 0:
        warnings.warn("Search range for autocorrelation is empty.")
        return (min_lag, max_lag)
    max_ac = np.max(np.abs(ac))
    threshold = max_ac * min_peak_height_ratio
    if verbose:
        print(f"max_ac={max_ac:.4f} threshold={threshold:.4f}")
    if max_ac == 0:
        warnings.warn("Autocorrelation is zero everywhere.")
        return (min_lag, max_lag)
    peaks, _ = find_peaks(search_ac, height=threshold)
    if verbose: print(f"[+] Found {len(peaks)} peaks in autocorrelation.")
    if len(peaks) == 0:
        if plot:
            print("No strong periodicity found; using default range.")
        return (min_lag, max_lag)
    peak_lags = peaks + min_lag
    start_lag = int(np.min(peak_lags))
    end_lag = int(np.max(peak_lags))

    # Ensure a minimum range
    if start_lag == end_lag:
        start_lag = max(min_lag, start_lag - 8)
        end_lag = min(max_lag, end_lag + 8)

    if verbose:
        print(f"[+] Suggested packet_check_range: ({start_lag}, {end_lag})")
    if plot:
        import matplotlib.pyplot as plt
        lags = np.arange(len(ac))
        plt.figure(figsize=(10, 4))
        plt.plot(lags, ac)
        plt.scatter(peak_lags, ac[peak_lags], color='red', label='Detected Peaks')
        plt.title("Autocorrelation with Detected Peaks")
        plt.xlabel("Lag (samples)")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return (start_lag, end_lag)

def _window_autocorr_task(args):
    data, i, window_size, threshold, packet_check_range, verbose = args
    segment = data[i:i+window_size]
    ac = fast_autocorr(segment)
    mid = len(ac) // 2
    lag_start, lag_end = packet_check_range
    peaks = ac[mid+lag_start : mid+lag_end]
    if ac[mid] == 0 or np.isnan(ac[mid]):
        norm_peak = 0
        keep_flag = True
    elif len(peaks) == 0:
        norm_peak = 0.0
        keep_flag = norm_peak < threshold if not np.isnan(norm_peak) else True
    else:
        norm_peak = np.max(np.abs(peaks)) / (ac[mid] + 1e-10)
        keep_flag = norm_peak < threshold if not np.isnan(norm_peak) else True
#    if verbose:
#        print(f"[window {i}] norm_peak={norm_peak:.4f} threshold={threshold:.4f} keep={keep_flag}")
    return (i, segment, keep_flag)

def hybrid_autocorr_cleaning(
    data,
    sample_rate=44100,
    window_size=4096,
    padding_mode="median",
    plot_packet_range=False,
    verbose=False,
    remove_dc=False,
):
    #print("\n[*] Starting hybrid autocorrelation cleaning...")
    data = np.asarray(data)
    N = len(data)
    if N == 0:
        if verbose: print("[!] Empty input signal. Returning input.")
        return data.copy()
    if window_size > N:
        if verbose: print("[!] Signal too short for windowing. Returning input.")
        return data.copy()

    if remove_dc:
        if verbose: print("[*] Removing DC component from the signal.")
        data = data - np.mean(data)

    # Step 1: Automatic packet_check_range detection
    if verbose: print("[*] Auto-detecting packet_check_range...")
    packet_check_range = suggest_packet_check_range(
        data, sample_rate,
        plot=plot_packet_range,
        verbose=verbose,
    )
    if verbose: print(f"[*] Using detected packet_check_range: {packet_check_range}")

    # Step 2: Data-driven packet_noise_ratio
    ac_global = fast_autocorr(data)
    mid = len(ac_global) // 2
    lag_start, lag_end = packet_check_range
    off_center_peaks = ac_global[mid+lag_start:mid+lag_end]
    main_peak = ac_global[mid]
    if main_peak == 0 or np.isnan(main_peak):
        packet_noise_ratio = 0.1  # fallback default
    else:
        ratios = np.abs(off_center_peaks) / (main_peak + 1e-10)
        ratios = ratios[np.isfinite(ratios)]  # Remove NaN and Inf if present

        if len(ratios) == 0:
            # Fallback if no off-center peaks or all invalid
            packet_noise_ratio = 0.05  # or another default you prefer
            if verbose:
                print(f"[auto] No valid off-center peaks; fallback packet_noise_ratio={packet_noise_ratio:.4f}")
        else:
            raw_ratio = np.percentile(ratios, 90)
            packet_noise_ratio = float(np.clip(raw_ratio, 0.01, 0.5))
            if verbose:
                print(f"[auto] 90th percentile off-center ratio: {raw_ratio:.4f}")
                print(f"[auto] Setting packet_noise_ratio={packet_noise_ratio:.4f} based on signal stats")

    # Step 3: Data-driven overlap_ratio
    # Use the spread of the norm_peak distribution to decide overlap.
    # More spread = more overlap (more aggressive cleaning).
    # Less spread = less overlap (faster, less aggressive).
    # Default is 0.5. Clamp between 0.25 and 0.8.
    if verbose: print("[*] Computing autocorrelation peaks for overlap ratio decision...")

    window_autocorr_peaks = []
    for i in range(0, N - window_size + 1, max(1, int(window_size * 0.5))):
        segment = data[i:i+window_size]
        ac = fast_autocorr(segment)
        mid_seg = len(ac) // 2

        peaks = ac[mid_seg+lag_start : mid_seg+lag_end]
        if ac[mid_seg] == 0 or np.isnan(ac[mid_seg]):
            norm_peak = np.nan
        elif len(peaks) == 0:
            norm_peak = 0.0
        else:
            norm_peak = np.max(np.abs(peaks)) / (ac[mid_seg] + 1e-10)
        window_autocorr_peaks.append(norm_peak)

    window_autocorr_peaks = [p for p in window_autocorr_peaks if not np.isnan(p)]
    spread = np.std(window_autocorr_peaks)
    if spread < 0.05:
        overlap_ratio = 0.25
    elif spread < 0.10:
        overlap_ratio = 0.4
    elif spread < 0.20:
        overlap_ratio = 0.6
    else:
        overlap_ratio = 0.8
    overlap_ratio = float(np.clip(overlap_ratio, 0.25, 0.8))
    if verbose:
        print(f"[auto] Setting overlap_ratio={overlap_ratio:.2f} based on norm_peak std={spread:.4f}")

    # Step 4: Global autocorr check for periodic noise (always auto, skip if already DC-free)
    if verbose: print("[*] Performing global autocorrelation check for periodic noise (auto mode).")
    peak_noise = np.max(ac_global[mid+lag_start:mid+lag_end]) if lag_start < lag_end else 0
    do_mean_subtract = (main_peak != 0 and peak_noise > packet_noise_ratio * main_peak)
    if do_mean_subtract and np.abs(np.mean(data)) > 1e-10:
        if verbose:
            print(f"[✓] Mean subtraction triggered (peak_noise={peak_noise:.2f}, main_peak={main_peak:.2f})")
            print("[*] Subtracting mean from data before cleaning (data is not already DC-free).")
        data = data - np.mean(data)
    elif do_mean_subtract and verbose:
        print("[*] Mean subtraction would be triggered, but data is already DC-free.")

    # Step 5: Windowed autocorr cleaning
    step = max(1, int(window_size * (1 - overlap_ratio)))
    cleaned = np.zeros_like(data, dtype=np.float64)
    overlap_counts = np.zeros_like(data, dtype=np.float64)

    if verbose:
        print(f"[*] Starting windowed autocorrelation cleaning: window_size={window_size}, step={step}, total_windows={((N-window_size)//step)+1}")

    # Recompute window_autocorr_peaks for the actual windowing with this step size
    window_autocorr_peaks = []
    for i in range(0, N - window_size + 1, step):
        segment = data[i:i+window_size]
        ac = fast_autocorr(segment)
        mid_seg = len(ac) // 2
        peaks = ac[mid_seg+lag_start:mid_seg+lag_end]
        if ac[mid_seg] == 0 or np.isnan(ac[mid_seg]):
            norm_peak = np.nan
        elif len(peaks) == 0:
            norm_peak = 0.0
        else:
            norm_peak = np.max(np.abs(peaks)) / (ac[mid_seg] + 1e-10)
        window_autocorr_peaks.append(norm_peak)

    window_autocorr_peaks = [p for p in window_autocorr_peaks if not np.isnan(p)]
    # Data-driven threshold fraction
    spread = np.std(window_autocorr_peaks)
    if spread < 0.05:
        target_fraction = 0.95
    elif spread < 0.10:
        target_fraction = 0.85
    elif spread < 0.20:
        target_fraction = 0.70
    else:
        target_fraction = 0.50
    threshold = np.percentile(window_autocorr_peaks, 100 * target_fraction)
    if verbose:
        print(f"[+] Auto-set threshold to keep {int(target_fraction*100)}% of windows: {threshold:.4f}")

    if verbose: print(f"[+] Automatic threshold set to {threshold:.4f} lag_start={packet_check_range[0]}, lag_end={packet_check_range[1]}")

    indices = list(range(0, N - window_size + 1, step))
    window_args = [(data, i, window_size, threshold, packet_check_range, verbose) for i in indices]
    total_windows = len(indices)

    if verbose:
        print("[*] Cleaning windows...")
    results = [_window_autocorr_task(args) for args in window_args]

    kept_windows = sum(1 for _, _, keep_flag in results if keep_flag)
    if verbose:
        print(f"[*] Windows kept after cleaning: {kept_windows} of {total_windows}")

    if kept_windows == 0:
        if verbose:
            print("[!] No windows passed autocorrelation cleaning. Relaxing threshold and retrying...")
        # Relax threshold: e.g., keep all windows
        cleaned = data.copy()
        overlap_counts = np.ones_like(data)
        return cleaned.astype(data.dtype)    

    for i, segment, keep_flag in results:
        if keep_flag:
            cleaned[i:i+window_size] += segment
            overlap_counts[i:i+window_size] += 1

    # Tail segment (if not aligned)
    last_start = N - window_size
    if last_start % step != 0 and N >= window_size:
        if verbose: print("[*] Processing tail segment.")
        i = last_start
        segment = data[i:]
        pad_len = window_size - len(segment)
        valid_pad_modes = [
            'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum',
            'reflect', 'symmetric', 'wrap'
        ]
        pad_mode: str = padding_mode if padding_mode in valid_pad_modes else 'constant'
        padded = np.pad(segment, (0, pad_len), mode=pad_mode) # type: ignore
        ac = fast_autocorr(padded)
        mid_seg = len(ac) // 2
        peaks = ac[mid_seg+lag_start:mid_seg+lag_end]
        if ac[mid_seg] == 0 or np.isnan(ac[mid_seg]):
            norm_peak = np.nan
        elif len(peaks) == 0:
            norm_peak = 0.0
        else:
            norm_peak = np.max(np.abs(peaks)) / (ac[mid_seg] + 1e-10)
        if not np.isnan(norm_peak) and norm_peak < threshold:
            cleaned[i:] += segment
            overlap_counts[i:] += 1

    overlap_counts = np.where(overlap_counts == 0, 1, overlap_counts)
    result = cleaned / overlap_counts
    if verbose: print("[*] Cleaning complete.")
    return result.astype(data.dtype)