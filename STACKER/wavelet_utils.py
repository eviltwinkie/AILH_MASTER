import numpy as np
import pywt
import math


def soft_threshold(data, thresh):
    return np.sign(data) * np.maximum(np.abs(data) - thresh, 0)

def hard_threshold(data, thresh):
    return data * (np.abs(data) > thresh)

def estimate_noise_sigma(detail_coeffs):
    # Median absolute deviation estimator (robust to outliers)
    return np.median(np.abs(detail_coeffs)) / 0.6745

def calc_universal_thresh(sigma, n):
    # VisuShrink threshold (universal threshold)
    return sigma * np.sqrt(2 * np.log(n))

def calc_sure_thresh(coeffs):
    # SURE threshold (minimizes Stein's unbiased risk)
    # Only works for soft thresholding
    from skimage.restoration import denoise_wavelet
    # Use a dummy call to get SURE threshold, then apply manually
    # If skimage is not available, fallback to universal
    try:
        arr = np.asarray(coeffs)
        threshold = denoise_wavelet(arr, method='BayesShrink', rescale_sigma=True, mode='soft', convert2ycbcr=False, channel_axis=None, wavelet='db1', sigma=None, wavelet_levels=None, thresholding_method='SURE', return_threshold=True)[1]
        return threshold
    except Exception:
        return None

def calc_bayesshrink_thresh(coeffs, sigma):
    # BayesShrink threshold
    var_signal = np.var(coeffs)
    thresh = sigma**2 / np.sqrt(max(var_signal - sigma**2, 1e-12))
    return thresh

def fade_edges(x, fade_len=1024):
    n = len(x)
    fade = np.linspace(0, 1, fade_len)
    out = x.copy()
    out[:fade_len] *= fade
    out[-fade_len:] *= fade[::-1]
    return out

def wavelet_denoise_data_driven(
    x, 
    wavelet='bior4.4', 
    level=None, 
    fade_len=1024,
    thresholding='soft', # 'soft', 'hard'
    scheme='universal',  # 'universal', 'SURE', 'BayesShrink'
):
    """
    Data-driven wavelet packet denoising with alternative thresholding schemes.
    - thresholding: 'soft' or 'hard'
    - scheme: 'universal', 'SURE', or 'BayesShrink'
    """
    x = x - np.mean(x)
    pad_width = 2 * pywt.Wavelet(wavelet).dec_len # type: ignore
    x_padded = np.pad(x, pad_width, mode='symmetric')
    wp = pywt.WaveletPacket(data=x_padded, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = wp.get_level(wp.maxlevel, order='freq')
    coeffs = np.array([n.data for n in nodes])
    sigma = estimate_noise_sigma(coeffs[-1])
    n_coeffs = len(x_padded)
    # Choose threshold
    if scheme == 'universal':
        thresh = calc_universal_thresh(sigma, n_coeffs)
    elif scheme == 'BayesShrink':
        thresh = calc_bayesshrink_thresh(coeffs[-1], sigma)
    elif scheme == 'SURE':
        sure_thresh = calc_sure_thresh(coeffs[-1])
        thresh = sure_thresh if sure_thresh is not None else calc_universal_thresh(sigma, n_coeffs)
    else:
        thresh = calc_universal_thresh(sigma, n_coeffs)
    # Threshold
    if thresholding == 'soft':
        coeffs_thresh = [soft_threshold(c, thresh) for c in coeffs]
    elif thresholding == 'hard':
        coeffs_thresh = [hard_threshold(c, thresh) for c in coeffs]
    else:
        coeffs_thresh = [soft_threshold(c, thresh) for c in coeffs]
    for n, c in zip(nodes, coeffs_thresh):
        n.data = c
    x_denoised = wp.reconstruct(update=True)
    x_denoised = x_denoised[pad_width:-pad_width]
#    x_denoised = fade_edges(x_denoised, fade_len=min(fade_len, len(x_denoised)//4))
    return x_denoised


def wavelet_denoise(signal, wavelet='db4', level=None, thresholding='soft'):
    """
    Denoise a 1D signal using wavelet transform.
    Parameters:
        signal (array-like): Input 1D signal to be denoised.
        wavelet (str): Wavelet type to use (default: 'db1').
        level (int or None): Decomposition level (default: None, uses max possible level).
        thresholding (str): 'soft' or 'hard' thresholding (default: 'soft').
    Returns:
        np.ndarray: Denoised signal.
    """
    # Convert input to numpy array
    signal = np.asarray(signal)

    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate the noise sigma using the detail coefficients at the highest level
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    # Threshold detail coefficients
    denoised_coeffs = [coeffs[0]]  # Approximation coefficients untouched
    for c in coeffs[1:]:
        denoised_c = pywt.threshold(c, value=uthresh, mode=thresholding)
        denoised_coeffs.append(denoised_c)

    # Wavelet reconstruction
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    # Match output length to input length
    return denoised_signal[:len(signal)]




def wavelet_packet_denoise(
    signal,
    wavelet='db4',
    maxlevel=None,
    thresholding='soft',
    threshold_factor=1.5
):
    """
    Denoise a 1D signal using wavelet packet decomposition.
    """

    signal = np.asarray(signal)
    orig_len = len(signal)
    #print(f"[✓] Original signal length: {orig_len}")

    mean_val = np.mean(signal)
    signal = signal - mean_val  # Remove DC offset

    exp = int(np.ceil(np.log2(orig_len)))
    new_len = 2 ** exp
    pad_width = new_len - orig_len
    signal_padded = np.pad(signal, (0, pad_width), mode='symmetric')

    maxlevel = math.floor(math.log2(orig_len / 7))
    #print(f"[✓] Max level for wavelet packet: {maxlevel}")

    wp = pywt.WaveletPacket(data=signal_padded, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)

    # Get nodes at the finest level (for noise estimation)
    maxlevel = wp.maxlevel if maxlevel is None else maxlevel
    leaves = [n for n in wp.get_leaf_nodes(True) if n.level == maxlevel]

    # Estimate sigma from finest scale nodes
    detail_coeffs = np.concatenate([n.data for n in leaves])
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Compute threshold
    #uthresh = sigma * np.sqrt(2 * np.log(len(signal_padded)))
    uthresh = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal_padded)))

    # Threshold all nodes except root
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
    for node in wp.get_leaf_nodes(True):
        if node.path == '':
            # root node, do not threshold
            new_data = node.data
        else:
            new_data = pywt.threshold(node.data, value=uthresh, mode=thresholding)
        new_wp[node.path] = new_data

    denoised_signal = new_wp.reconstruct(update=False)
    denoised_signal = denoised_signal[:orig_len] + mean_val  # Restore DC offset
    return denoised_signal[:len(signal)]




# def wavelet_packet_denoise(
#     signal,
#     wavelet='db4',
#     maxlevel=8,
#     thresholding='soft',
#     threshold_factor=1,
#     pad_to_pow2=True
# ):
#     signal = np.asarray(signal)
#     orig_len = len(signal)
#     mean_val = np.mean(signal)
#     signal = signal - mean_val  # Remove DC offset

#     if pad_to_pow2:
#         exp = int(np.ceil(np.log2(orig_len)))
#         new_len = 2 ** exp
#         pad_width = new_len - orig_len
#         signal_padded = np.pad(signal, (0, pad_width), mode='symmetric')
#     else:
#         signal_padded = signal

#     wp = pywt.WaveletPacket(data=signal_padded, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     #maxlevel = wp.maxlevel if maxlevel is None else maxlevel
#     #print(f"Wavelet Packet max level: {maxlevel}")

#     # Use all detail coefficients for sigma estimate (all nodes except root)
#     all_detail_nodes = [n for n in wp.get_leaf_nodes(True) if n.path != '']
#     if len(all_detail_nodes) == 0:
#         all_detail_nodes = wp.get_leaf_nodes(True)
#     detail_coeffs = np.concatenate([n.data for n in all_detail_nodes])
#     sigma = np.median(np.abs(detail_coeffs)) / 0.6745

#     uthresh = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal_padded)))

#     new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
#     for node in wp.get_leaf_nodes(True):
#         if node.path == '':
#             new_data = node.data
#         else:
#             new_data = pywt.threshold(node.data, value=uthresh, mode=thresholding)
#         new_wp[node.path] = new_data

#     denoised_signal = new_wp.reconstruct(update=False)
#     denoised_signal = denoised_signal[:orig_len] + mean_val  # Restore DC offset

#     return denoised_signal

# def wavelet_packet_denoise(
#     signal,
#     wavelet='db4',
#     maxlevel=None,
#     thresholding='soft',
#     threshold_factor=0.5,  # try 1.2 or 1.5 for more aggressive denoising
#     pad_to_pow2=True
# ):
#     import numpy as np
#     import pywt
#     import math

#     signal = np.asarray(signal)
#     orig_len = len(signal)

#     # Optional: Pad to next power of two for better wavelet performance
#     if pad_to_pow2:
#         exp = int(np.ceil(np.log2(orig_len)))
#         new_len = 2 ** exp
#         pad_width = new_len - orig_len
#         signal_padded = np.pad(signal, (0, pad_width), mode='symmetric')
#     else:
#         signal_padded = signal

#     wp = pywt.WaveletPacket(data=signal_padded, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     maxlevel = wp.maxlevel if maxlevel is None else maxlevel

#     # Use all detail coefficients for sigma estimate
#     all_detail_nodes = [n for n in wp.get_leaf_nodes(True) if n.path != '']
#     detail_coeffs = np.concatenate([n.data for n in all_detail_nodes])
#     sigma = np.median(np.abs(detail_coeffs)) / 0.6745

#     # Threshold
#     uthresh = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal_padded)))

#     # Threshold all detail nodes (not just leaves)
#     new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
#     for node in wp.get_leaf_nodes(True):
#         if node.path == '':
#             # root node, do not threshold
#             new_data = node.data
#         else:
#             new_data = pywt.threshold(node.data, value=uthresh, mode=thresholding)
#         new_wp[node.path] = new_data

#     denoised_signal = new_wp.reconstruct(update=False)
#     return denoised_signal[:orig_len]





# def wavelet_packet_denoise(
#     signal,
#     wavelet='db4',
#     maxlevel=None,
#     thresholding='soft',
#     threshold_method='universal'
# ):
#     import numpy as np
#     import pywt
#     import math

#     signal = np.asarray(signal)
#     orig_len = len(signal)
#     # Pad to next power of two for best WPT behavior
#     exp = math.ceil(np.log2(orig_len))
#     new_len = 2 ** exp
#     pad_width = new_len - orig_len
#     if pad_width > 0:
#         signal_padded = np.pad(signal, (0, pad_width), mode='symmetric')
#     else:
#         signal_padded = signal

#     wp = pywt.WaveletPacket(data=signal_padded, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     maxlevel = wp.maxlevel if maxlevel is None else maxlevel
#     leaves = [n for n in wp.get_leaf_nodes(True) if n.level == maxlevel]
#     detail_coeffs = np.concatenate([n.data for n in leaves])
#     sigma = np.median(np.abs(detail_coeffs)) / 0.6745

#     # Compute threshold
#     if threshold_method == 'universal':
#         uthresh = sigma * np.sqrt(2 * np.log(len(signal_padded)))
#     else:
#         raise NotImplementedError('Only universal threshold implemented')

#     new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
#     for node in wp.get_leaf_nodes(True):
#         if node.path == '':
#             new_data = node.data
#         else:
#             new_data = pywt.threshold(node.data, value=uthresh, mode=thresholding)
#         new_wp[node.path] = new_data

#     denoised_signal = new_wp.reconstruct(update=False)
#     # Crop to original length
#     return denoised_signal[:orig_len]




# def wavelet_packet_denoise(
#     signal,
#     wavelet='db4',
#     maxlevel=None,
#     thresholding='soft',
#     threshold_method='sure',
#     sure_scale_range=(0.5, 1.2),  # (min_scale, max_scale)
#     snr_range=(0.5, 2.5)          # (min_SNR, max_SNR) for scaling
# ):
#     import numpy as np
#     import pywt
#     import math

#     def sure_threshold(data, sigma):
#         x = data
#         n = x.size
#         if n == 0:
#             return 0.0
#         x2 = np.sort(x ** 2)
#         risks = (n - 2 * np.arange(1, n+1) +
#                  np.cumsum(x2) +
#                  x2 * np.arange(1, n+1)) / n
#         min_idx = np.argmin(risks)
#         thresh = np.sqrt(x2[min_idx])
#         if np.isnan(thresh) or np.isinf(thresh):
#             return 0.0
#         return thresh * sigma

#     def adaptive_sure_scaling(node_data, sigma, scale_range, snr_range):
#         # Estimate SNR for the node (signal+noise power / noise power)
#         coeff_var = np.var(node_data)
#         snr_est = coeff_var / (sigma**2 + 1e-10)
#         # Map snr_est linearly from snr_range to scale_range (clipped)
#         snr_min, snr_max = snr_range
#         scale_min, scale_max = scale_range
#         # Normalize SNR to [0, 1] within snr_range
#         snr_norm = (snr_est - snr_min) / (snr_max - snr_min)
#         snr_norm = np.clip(snr_norm, 0, 1)
#         # Higher SNR → lower scale (less aggressive)
#         scale = scale_max - (scale_max - scale_min) * snr_norm
#         return scale

#     signal = np.asarray(signal)
#     orig_len = len(signal)
#     exp = math.ceil(np.log2(orig_len))
#     new_len = 2 ** exp
#     pad_width = new_len - orig_len
#     if pad_width > 0:
#         signal_padded = np.pad(signal, (0, pad_width), mode='symmetric')
#     else:
#         signal_padded = signal

#     wp = pywt.WaveletPacket(data=signal_padded, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
#     maxlevel = wp.maxlevel if maxlevel is None else maxlevel
#     leaves = [n for n in wp.get_leaf_nodes(True) if n.level == maxlevel]
#     detail_coeffs = np.concatenate([n.data for n in leaves])
#     sigma = np.median(np.abs(detail_coeffs)) / 0.6745

#     new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
#     for node in wp.get_leaf_nodes(True):
#         if node.path == '':
#             new_data = node.data
#         else:
#             universal_thresh = sigma * np.sqrt(2 * np.log(len(signal_padded)))
#             sure_thresh = sure_threshold(node.data, 1.0)
#             # Fully adaptive, stat-driven scaling for SURE:
#             if np.isfinite(sure_thresh):
#                 # Compute per-node scaling factor based on SNR
#                 scale = adaptive_sure_scaling(
#                     node.data, sigma, sure_scale_range, snr_range
#                 )
#                 scaled_sure_thresh = max(scale * sure_thresh, 0.4 * universal_thresh)
#             else:
#                 scaled_sure_thresh = 0.4 * universal_thresh
#             thresh = scaled_sure_thresh if threshold_method.lower() == 'sure' else universal_thresh
#             new_data = pywt.threshold(node.data, value=thresh, mode=thresholding)
#         new_wp[node.path] = new_data

#     denoised_signal = new_wp.reconstruct(update=False)
#     return denoised_signal[:orig_len]


