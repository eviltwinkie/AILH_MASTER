import os
import sys
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model # type: ignore

CNN_MODEL_BEST = "cnn_mel_best.h5"
CNN_MODEL_UPDATED = "cnn_mel_updated.h5"

# ==== MODEL AND FEATURE PARAMETERS (COPY FROM TRAINER) ====
KERNEL_SIZE = (3, 3)
HOP_LENGTH = 32
LONG_TERM_SEC = 8.0
SHORT_TERM_SEC = 1.75
STRIDE_SEC = 1.7
DROPOUT = 0.15
N_FILTERS = 80
N_MELS = 64
N_FFT = 256
N_CONV_BLOCKS = 3
MAX_POOLINGS = 2

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
    Split audio signal into segments (mock-up). Replace with your actual implementation!
    """
    seg_samples = int(short_term * sr)
    stride_samples = int(stride * sr)
    segments = []
    for start in range(0, len(y) - seg_samples + 1, stride_samples):
        segments.append(y[start:start + seg_samples])
    return segments

def segment_to_mel(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert a segment to a Mel spectrogram using numpy/scipy (no librosa).
    You can replace with your own implementation if you have one.
    """
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    Sxx = np.log(Sxx + 1e-9)
    # Mel filterbank (approximation: select n_mels lowest frequencies)
    mel_basis = np.linspace(0, len(f)-1, n_mels).astype(int)
    mel_spec = Sxx[mel_basis, :]
    return mel_spec

# ---- Main feature extraction function ----
def extract_mel_features(wav_path, input_shape):
    cache_fname = os.path.join(FEATURE_CACHE_DIR, file_hash(wav_path) + ".npy")
    if os.path.exists(cache_fname):
        mel_segments = np.load(cache_fname, allow_pickle=True)
        return wav_path, mel_segments
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        segments = split_segments(
            y, sr,
            long_term=LONG_TERM_SEC,
            short_term=SHORT_TERM_SEC,
            stride=STRIDE_SEC
        )
        mel_segments = []
        n_mels, max_frames = input_shape[0], input_shape[1]
        for seg in segments:
            mel = segment_to_mel(
                seg, sr,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            # Pad/truncate to correct shape
            if mel.shape[1] < max_frames:
                pad = np.full((n_mels, max_frames - mel.shape[1]), np.min(mel))
                mel = np.concatenate([mel, pad], axis=1)
            else:
                mel = mel[:, :max_frames]
            mel_segments.append(mel[..., np.newaxis])  # (n_mels, max_frames, 1)
        mel_segments = np.stack(mel_segments).astype(np.float32)
        np.save(cache_fname, mel_segments)
        return wav_path, mel_segments
    except Exception as e:
        print(f"[!] Error processing {wav_path}: {e}")
        return wav_path, None

# ---- Load model ----
def load_cnn_model(model_path=MODEL_PATH):
    print(f"[i] Loading model: {model_path}")
    return load_model(model_path)

# ---- Predict one file ----
def predict_file(model, wav_path, input_shape=INPUT_SHAPE):
    _, mels = extract_mel_features(wav_path, input_shape)
    if mels is None or len(mels) == 0:
        print(f"[!] No mel segments for {wav_path}")
        return None
    preds = model.predict(mels)
    mean_pred = np.mean(preds, axis=0)  # Average over segments
    idx_sorted = np.argsort(mean_pred)[::-1]
    label_sorted = [(LABELS[i], mean_pred[i]) for i in idx_sorted]
    # Print as requested
    filename = os.path.basename(wav_path)
    main_label, main_conf = label_sorted[0]
    output = f"{filename}: {main_label} ({main_conf*100:.1f}%)"
    # Add rest of the labels/confidences
    if len(label_sorted) > 1:
        rest = ", ".join(f"{label} ({conf*100:.1f}%)" for label, conf in label_sorted[1:])
        output += f", {rest}"
    print(output)
    return main_label


# ---- Batch predict ----
def batch_predict_folder(model, folder, input_shape=INPUT_SHAPE, exts=(".wav", ".WAV")):
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
