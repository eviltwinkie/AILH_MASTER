#!/usr/bin/env python3
"""
CNN Mel Incremental Learning Module

Implements incremental/continual learning for acoustic leak detection models.
Processes new WAV files from UPDATE_DATA directory, standardizes them, and
performs incremental training on the existing model.

Features:
    - Automatic WAV file standardization (resampling, padding, splitting)
    - Incremental model updates without full retraining
    - Label encoder compatibility checking
    - Support for both .h5 and .keras model formats

Directory Structure:
    UPDATE_DATA/RAW/          - Raw input WAV files organized by label
    UPDATE_DATA/PROCESSED/    - Standardized WAV files ready for training
    CNN_CACHE/                - Cached label encoders and features

Note:
    Most of the incremental learning logic is currently commented out.
    Active functionality is limited to WAV file preprocessing.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np
import tensorflow as tf
import soundfile as sf
import scipy.signal
from tensorflow.keras.utils import to_categorical # type: ignore

CNN_MODEL_BEST = "cnn_mel_best.h5"
CNN_MODEL_UPDATED = "cnn_mel_updated.h5"

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

BASE_DIR = os.path.join("..")
CACHE_DIR = os.path.join(BASE_DIR, "CNN_CACHE")
PLOTS_DIR = os.path.join(BASE_DIR, "CNN_PLOTS")
UPDATE_DIR = os.path.join(BASE_DIR, "UPDATE_DATA")
RAW_DATA_DIR = os.path.join(UPDATE_DIR, "RAW")
PROCESSED_DATA_DIR = os.path.join(UPDATE_DIR, "PROCESSED")

def find_wav_files(folder):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files if f.lower().endswith(".wav")
    ]

def resample_audio(y, orig_sr, target_sr):
    """Resample audio array y from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return y
    n_target = int(len(y) * target_sr / orig_sr)
    y_rs = scipy.signal.resample(y, n_target)
    return y_rs

def standardize_wav_lengths_to_processed(input_root, output_root, sample_rate=4096, max_sec=10):
    """
    Recursively standardize all WAV files in input_root for model consumption:
        - Resample to sample_rate if needed.
        - Split >max_sec into non-overlapping max_sec segments.
        - Pad <max_sec to length max_sec.
    """
    max_samples = int(sample_rate * max_sec)
    processed_files = []
    processed_labels = []

    wav_files = find_wav_files(input_root)
    for wav_path in wav_files:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != sample_rate:
            print(f"[i] {wav_path} has sample rate {sr}, resampling to {sample_rate} ...")
            y = resample_audio(y, sr, sample_rate)
            sr = sample_rate
        total_samples = len(y)
        rel_path = os.path.relpath(wav_path, input_root)
        base, ext = os.path.splitext(rel_path)
        label = os.path.normpath(base).split(os.sep)[-2]  # parent folder as label
        dest_dir = os.path.join(output_root, os.path.dirname(base))
        os.makedirs(dest_dir, exist_ok=True)

        if total_samples == max_samples:
            dest_path = os.path.join(output_root, base + ext)
            if os.path.exists(dest_path):
                print(f"[✓] Exists, skipping: {dest_path}")
            else:
                sf.write(dest_path, y, sample_rate)
                print(f"    Copied {wav_path} to {dest_path} ({len(y)} samples)")
            processed_files.append(dest_path)
            processed_labels.append(label)

        elif total_samples > max_samples:
            print(f"[i] Splitting {wav_path} ({total_samples} samples) into 10s segments...")
            num_parts = (total_samples + max_samples - 1) // max_samples
            for i in range(num_parts):
                seg_start = i * max_samples
                seg_end = min((i+1) * max_samples, total_samples)
                seg = y[seg_start:seg_end]
                seg = np.asarray(seg)
                if len(seg) < max_samples:
                    pad = np.zeros(max_samples - len(seg), dtype=seg.dtype)
                    seg = np.concatenate([seg, pad])
                seg_fname = os.path.join(dest_dir, f"{os.path.basename(base)}_part{i+1}{ext}")
                if os.path.exists(seg_fname):
                    print(f"[✓] Exists, skipping: {seg_fname}")
                else:
                    sf.write(seg_fname, seg, sample_rate)
                    print(f"    Saved {seg_fname} ({len(seg)} samples)")
                processed_files.append(seg_fname)
                processed_labels.append(label)

        elif total_samples < max_samples:
            y = np.asarray(y)
            pad = np.zeros(max_samples - total_samples, dtype=y.dtype)
            y_padded = np.concatenate([y, pad])
            padded_fname = os.path.join(dest_dir, os.path.basename(base) + "_padded" + ext)
            if os.path.exists(padded_fname):
                print(f"[✓] Exists, skipping: {padded_fname}")
            else:
                sf.write(padded_fname, y_padded, sample_rate)
                print(f"    Saved {padded_fname} ({len(y_padded)} samples)")
            processed_files.append(padded_fname)
            processed_labels.append(label)

    return processed_files, processed_labels

def load_label_encoder():
    """Load LabelEncoder from cache."""
    le_path = None
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith("cache_train_le") and fname.endswith(".npy"):
            le_path = os.path.join(CACHE_DIR, fname)
            break
    if not le_path:
        raise RuntimeError("LabelEncoder file missing from CACHE. Train must have been run first.")
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return le

def load_best_model():
    """Try loading the .h5 model first, then fallback to .keras if needed."""
    if os.path.exists(CNN_MODEL_BEST):
        print(f"[i] Loading model from {CNN_MODEL_BEST}")
        return tf.keras.models.load_model(CNN_MODEL_BEST)
    elif os.path.exists(CNN_MODEL_BEST.replace(".h5", ".keras")):
        print(f"[i] Loading model from {CNN_MODEL_BEST.replace('.h5', '.keras')}")
        return tf.keras.models.load_model(CNN_MODEL_BEST.replace(".h5", ".keras"))
    else:
        raise RuntimeError("No model found (neither .h5 nor .keras). Please train a model first.")

if __name__ == "__main__":
    # 1. Standardize and preprocess all WAVs in the update folder
    processed_files, processed_labels = standardize_wav_lengths_to_processed(
        input_root=RAW_DATA_DIR,
        output_root=PROCESSED_DATA_DIR,
        sample_rate=4096,   # Should match model
        max_sec=10          # Should match training
    )

    # print(f"[i] Processed {len(processed_files)} files to {PROCESSED_DATA_DIR}. Labels: {set(processed_labels)}")

    # # 2. Load Mel features for update data (no augmentation)
    # X, all_labels, _, max_frames = load_wavs_parallel(
    #     processed_files, processed_labels,
    #     augment=False,
    #     cache_prefix="cache_update",
    #     debug_plot_first=False
    # )

    # print("[DEBUG] Example all_labels[0]:", all_labels[0])
    # print("[DEBUG] all_labels type:", type(all_labels[0]))

    # # 3. Prepare Y labels and categorical
    # le = load_label_encoder()
    # if isinstance(all_labels, np.ndarray) and all_labels.ndim == 2 and all_labels.shape[1] > 1:
    #     y_indices = np.argmax(all_labels, axis=1)
    #     y_labels = [le.classes_[i] for i in y_indices]
    # elif isinstance(all_labels[0], (tuple, list)) and len(all_labels[0]) > 0:
    #     y_labels = [lbl[0] for lbl in all_labels]
    # else:
    #     y_labels = all_labels

    # print("[DEBUG] y_labels shape:", np.shape(y_labels))

    # y = le.transform(y_labels)
    # Y = to_categorical(y, num_classes=len(le.classes_))  # Always matches model!

    # # 4. Load the best trained model (either .h5 or .keras)
    # model = load_best_model()

    # print(f"[i] Incremental training on shape {X.shape} and Y shape {Y.shape}")
    # history = model.fit(X, Y, epochs=20, batch_size=64)

    # # 5. Save the updated model (both .keras and .h5 for compatibility)
    # #model.save("cnn_incremental_updated.keras")
    # model.save("cnn_incremental_updated.h5")
    # print("[✓] Incremental training complete and model saved as cnn_incremental_updated.keras and cnn_incremental_updated.h5")
