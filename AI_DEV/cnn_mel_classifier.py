#!/usr/bin/env python3
"""
CNN Mel Spectrogram Classifier for Acoustic Leak Detection

This script classifies audio files using a trained CNN model with mel spectrogram features.
It processes WAV files through two-stage temporal segmentation and outputs classification
results with confidence scores.

Usage:
    python cnn_mel_classifier.py <wavfile or folder>

Arguments:
    wavfile or folder: Path to a single WAV file or directory containing WAV files

Configuration:
    The script automatically selects between cnn_mel_updated.h5 (if available) or
    cnn_mel_best.h5 as the model file.

Output:
    Prints classification results in the format:
    filename.wav: LABEL (XX.X%), label2 (XX.X%), ...

Note:
    This script uses different parameters than global_config.py:
    - N_FFT: 256 (vs 512 in global_config)
    - HOP_LENGTH: 32 (vs 128 in global_config)
    - N_MELS: 64 (matches global_config)
"""
import os
import sys
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model # type: ignore

# Model file selection (prioritizes updated model if available)
CNN_MODEL_BEST = "cnn_mel_best.h5"
CNN_MODEL_UPDATED = "cnn_mel_updated.h5"

# ==== MODEL AND FEATURE PARAMETERS ====
# WARNING: These parameters differ from global_config.py - verify before modifying!
KERNEL_SIZE = (3, 3)       # CNN kernel size
HOP_LENGTH = 32            # Hop length for STFT (differs from global_config: 128)
LONG_TERM_SEC = 8.0        # Long temporal segment duration in seconds
SHORT_TERM_SEC = 1.75      # Short temporal segment duration in seconds
STRIDE_SEC = 1.7           # Stride between segments in seconds
DROPOUT = 0.15             # Dropout rate for regularization
N_FILTERS = 80             # Number of filters in first CNN layer
N_MELS = 64                # Number of mel frequency bins (matches global_config)
N_FFT = 256                # FFT window size (differs from global_config: 512)
N_CONV_BLOCKS = 3          # Number of convolutional blocks
MAX_POOLINGS = 2           # Number of max pooling layers

# Model selection logic
if os.path.exists(CNN_MODEL_UPDATED):
    MODEL_PATH = os.path.join(CNN_MODEL_UPDATED)
else:
    MODEL_PATH = os.path.join(CNN_MODEL_BEST)

# ---- OTHER PARAMETERS ----
FEATURE_CACHE_DIR = "./mel_cache"

INPUT_SHAPE = (N_MELS, 217, 1)  # Adjust 217 to your max_frames if known/trained otherwise compute as needed

LABELS = ["LEAK", "NORMAL", "MECHANICAL", "QUIET", "RANDOM", "UNCLASSIFIED"]  # adjust if your model differs

# ---- Ensure cache dir exists ----
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# ---- Utility: File Hash ----
import hashlib
def file_hash(filepath):
    """
    Compute SHA1 hash of a file for caching purposes.

    Args:
        filepath (str): Path to the file to hash

    Returns:
        str: Hexadecimal SHA1 hash digest of the file contents

    Note:
        Uses 8KB chunks for memory-efficient hashing of large files
    """
    h = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ---- You must provide these or adapt: ----
# from your_library import split_segments, segment_to_mel

def split_segments(y, sr, long_term=LONG_TERM_SEC, short_term=SHORT_TERM_SEC, stride=STRIDE_SEC):
    """
    Split audio signal into overlapping temporal segments.

    Implements the temporal segmentation strategy described in the research paper.
    Creates overlapping segments with configurable stride for robust feature extraction.

    Args:
        y (np.ndarray): Audio signal array
        sr (int): Sample rate in Hz
        long_term (float): Long temporal segment duration in seconds (default: 8.0)
        short_term (float): Short temporal segment duration in seconds (default: 1.75)
        stride (float): Stride between segments in seconds (default: 1.7)

    Returns:
        list: List of audio segments as numpy arrays

    Note:
        This is a simplified implementation. The full two-stage segmentation
        should be implemented according to the research paper methodology.
    """
    seg_samples = int(short_term * sr)
    stride_samples = int(stride * sr)
    segments = []
    for start in range(0, len(y) - seg_samples + 1, stride_samples):
        segments.append(y[start:start + seg_samples])
    return segments

def segment_to_mel(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert audio segment to Mel spectrogram using scipy (librosa-free).

    Implements mel spectrogram computation using numpy/scipy following project
    guidelines (librosa is not allowed per CLAUDE.md).

    Args:
        y (np.ndarray): Audio segment array
        sr (int): Sample rate in Hz
        n_mels (int): Number of mel frequency bins (default: 64)
        n_fft (int): FFT window size (default: 256)
        hop_length (int): Hop size for STFT (default: 32)

    Returns:
        np.ndarray: Log-scaled mel spectrogram of shape (n_mels, time_frames)

    Note:
        Uses a simplified mel filterbank approximation. For production use,
        consider implementing proper mel filter bank computation.
    """
    from scipy.signal import spectrogram
    # Compute power spectrogram
    f, t, Sxx = spectrogram(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    # Convert to log scale for better numerical stability
    Sxx = np.log(Sxx + 1e-9)
    # Mel filterbank approximation: select n_mels lowest frequencies
    # TODO: Replace with proper mel filterbank implementation
    mel_basis = np.linspace(0, len(f)-1, n_mels).astype(int)
    mel_spec = Sxx[mel_basis, :]
    return mel_spec

# ---- Main feature extraction function ----
def extract_mel_features(wav_path, input_shape):
    """
    Extract mel spectrogram features from WAV file with disk caching.

    Processes audio file through temporal segmentation and mel spectrogram computation.
    Results are cached to disk using file hash for faster re-processing.

    Args:
        wav_path (str): Path to WAV audio file
        input_shape (tuple): Expected input shape (n_mels, max_frames, channels)

    Returns:
        tuple: (wav_path, mel_segments) where mel_segments is np.ndarray of shape
               (num_segments, n_mels, max_frames, 1) or None if processing failed

    Raises:
        Exception: Catches and logs any processing errors, returns None for features

    Caching:
        Cached features are stored in FEATURE_CACHE_DIR using SHA1 hash of file
        content as filename for consistent cache invalidation on file changes.
    """
    cache_fname = os.path.join(FEATURE_CACHE_DIR, file_hash(wav_path) + ".npy")
    # Check if cached features exist
    if os.path.exists(cache_fname):
        mel_segments = np.load(cache_fname, allow_pickle=True)
        return wav_path, mel_segments
    try:
        # Load audio file
        y, sr = sf.read(wav_path)
        # Convert stereo to mono if necessary
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Segment audio into temporal windows
        segments = split_segments(
            y, sr,
            long_term=LONG_TERM_SEC,
            short_term=SHORT_TERM_SEC,
            stride=STRIDE_SEC
        )
        # Extract mel spectrograms for each segment
        mel_segments = []
        n_mels, max_frames = input_shape[0], input_shape[1]
        for seg in segments:
            mel = segment_to_mel(
                seg, sr,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            # Pad or truncate to match expected input shape
            if mel.shape[1] < max_frames:
                pad = np.full((n_mels, max_frames - mel.shape[1]), np.min(mel))
                mel = np.concatenate([mel, pad], axis=1)
            else:
                mel = mel[:, :max_frames]
            mel_segments.append(mel[..., np.newaxis])  # Add channel dimension
        # Stack all segments and convert to float32
        mel_segments = np.stack(mel_segments).astype(np.float32)
        # Cache computed features to disk
        np.save(cache_fname, mel_segments)
        return wav_path, mel_segments
    except Exception as e:
        print(f"[!] Error processing {wav_path}: {e}")
        return wav_path, None

# ---- Load model ----
def load_cnn_model(model_path=MODEL_PATH):
    """
    Load trained CNN model from disk.

    Args:
        model_path (str): Path to model file (.h5 format)

    Returns:
        keras.Model: Loaded and compiled TensorFlow/Keras model

    Note:
        The model is expected to be in HDF5 format (.h5) and must be
        compatible with the current TensorFlow version.
    """
    print(f"[i] Loading model: {model_path}")
    return load_model(model_path)

# ---- Predict one file ----
def predict_file(model, wav_path, input_shape=INPUT_SHAPE):
    """
    Classify a single WAV file and print results.

    Extracts mel spectrogram features, runs inference, and outputs classification
    results with confidence scores sorted by probability.

    Args:
        model (keras.Model): Trained CNN model
        wav_path (str): Path to WAV audio file
        input_shape (tuple): Expected input shape (n_mels, max_frames, channels)

    Returns:
        str: Predicted label (highest confidence class) or None if processing failed

    Output Format:
        filename.wav: LABEL (XX.X%), label2 (YY.Y%), ...

    Prediction Strategy:
        - Extracts mel features for all segments
        - Runs model inference on each segment
        - Averages predictions across all segments for final classification
    """
    _, mels = extract_mel_features(wav_path, input_shape)
    if mels is None or len(mels) == 0:
        print(f"[!] No mel segments for {wav_path}")
        return None
    # Run model prediction on all segments
    preds = model.predict(mels)
    # Average predictions across segments for ensemble decision
    mean_pred = np.mean(preds, axis=0)
    # Sort labels by confidence (descending)
    idx_sorted = np.argsort(mean_pred)[::-1]
    label_sorted = [(LABELS[i], mean_pred[i]) for i in idx_sorted]
    # Format and print output
    filename = os.path.basename(wav_path)
    main_label, main_conf = label_sorted[0]
    output = f"{filename}: {main_label} ({main_conf*100:.1f}%)"
    # Append remaining labels with confidences
    if len(label_sorted) > 1:
        rest = ", ".join(f"{label} ({conf*100:.1f}%)" for label, conf in label_sorted[1:])
        output += f", {rest}"
    print(output)
    return main_label


# ---- Batch predict ----
def batch_predict_folder(model, folder, input_shape=INPUT_SHAPE, exts=(".wav", ".WAV")):
    """
    Recursively classify all WAV files in a directory.

    Walks through directory tree and classifies all WAV files found,
    printing results for each file.

    Args:
        model (keras.Model): Trained CNN model
        folder (str): Root directory to search for WAV files
        input_shape (tuple): Expected input shape (n_mels, max_frames, channels)
        exts (tuple): File extensions to process (default: (".wav", ".WAV"))

    Note:
        Results are printed to stdout in the same format as predict_file().
        No results are returned or saved to file.
    """
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.endswith(exts):
                wav_path = os.path.join(root, fname)
                predict_file(model, wav_path, input_shape=input_shape)

# ---- MAIN ----
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cnn_mel_classifier.py [wavfile or folder]")
        sys.exit(1)
    model = load_cnn_model(MODEL_PATH)
    path = sys.argv[1]
    if os.path.isfile(path):
        predict_file(model, path, input_shape=INPUT_SHAPE)
    elif os.path.isdir(path):
        batch_predict_folder(model, path, input_shape=INPUT_SHAPE)
    else:
        print(f"[!] Path does not exist: {path}")
