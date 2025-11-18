import os

from old_config import BASE_DIR, MIN_BATCH_SIZE, MAX_BATCH_SIZE, HOP_LENGTH, LONG_TERM_SEC, SHORT_TERM_SEC, STRIDE_SEC, N_MELS, N_FFT, SAMPLE_RATE, MAX_THREAD_WORKERS, CPU_COUNT

import sys
import time
import glob
import hashlib
import argparse
import json
import numpy as np
from pyfftw.interfaces.numpy_fft import rfft as fftw_rfft
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D
import traceback


from cnn_mel_processor import compute_max_poolings, load_wavs_parallel, after_trial_cleanup

def find_safe_batch_size(model_fn, X_train, Y_train, X_valid, Y_valid, callbacks):
    # Always clean memory before trying batch sizes
    after_trial_cleanup()

    def make_dataset_gpu(X, Y, bs):
        # Convert NumPy arrays to GPU tensors
        X_gpu = tf.constant(X, dtype=tf.float32)
        Y_gpu = tf.constant(Y, dtype=tf.float32)

        # Create dataset directly from GPU tensors
        ds = tf.data.Dataset.from_tensor_slices((X_gpu, Y_gpu)).batch(bs).prefetch(tf.data.AUTOTUNE)

        # TensorFlow example:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE).batch(bs)


        return ds

    bs = MIN_BATCH_SIZE  # start small to avoid early OOM
    best_bs = bs

    while bs <= MAX_BATCH_SIZE:
        try:
            print(f"[i] Testing batch size {bs} ...")

            train_ds = make_dataset_gpu(X_train, Y_train, bs)
            valid_ds = make_dataset_gpu(X_valid, Y_valid, bs)

            with tf.device('/GPU:0'):  # Force run on GPU
                model = model_fn()
                model.fit(
                    train_ds,
                    epochs=1,
                    validation_data=valid_ds,
                    callbacks=callbacks,
                    verbose=0
                )

            best_bs = bs
            bs *= 2

        except tf.errors.ResourceExhaustedError:
            print(f"[!] OOM at batch size {bs}, stopping search.")
            break
        except tf.errors.InternalError as e:
            if "Dst tensor is not initialized" in str(e):
                print(f"[!] GPU allocator issue at batch size {bs}, stopping search.")
                break
            else:
                raise

    print(f"[✓] Safe batch size determined: {best_bs}")
    return best_bs

def robust_fit(model, X, Y, batch_size, epochs=40, validation_data=None, callbacks=None, patience=2, verbose=1):
    cur_bs = batch_size
    min_bs = MIN_BATCH_SIZE
    while cur_bs >= min_bs:
        try:
            print(f"[i] Attempting to train with batch size {cur_bs} ...")
            return model.fit(
                X, Y,
                batch_size=cur_bs, epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks, verbose=verbose
            )
        except Exception as exc:
            tb = ''.join(traceback.format_exception(None, exc, exc.__traceback__))
            if ("RESOURCE_EXHAUSTED" in tb or "OOM" in tb or "out of memory" in tb is not None):
                new_bs = max(min_bs, int(cur_bs * 0.75))
                print(f"[!] OOM during training at batch size {cur_bs}, retrying with lower ({new_bs}) ...")
                cur_bs = new_bs
                time.sleep(1)
            else:
                print(tb, file=sys.stderr)
                raise
    print(f"[✗] Training failed: batch size reduced below {min_bs}")
    return None

# ======================
# DATA UTILS & PIPELINE
# ======================

def hash_files(file_list):
    hash_md5 = hashlib.md5()
    for fname in sorted(file_list):
        hash_md5.update(fname.encode('utf-8'))
        try:
            stat = os.stat(fname)
            hash_md5.update(str(stat.st_mtime).encode('utf-8'))
            hash_md5.update(str(stat.st_size).encode('utf-8'))
        except Exception:
            hash_md5.update(b"NA")
    return hash_md5.hexdigest()

def validate_parameters():
    if SHORT_TERM_SEC >= LONG_TERM_SEC:
        raise ValueError(f"SHORT_TERM_SEC ({SHORT_TERM_SEC}) must be less than LONG_TERM_SEC ({LONG_TERM_SEC})")
    if STRIDE_SEC >= SHORT_TERM_SEC:
        print(f"[!] Warning: STRIDE_SEC ({STRIDE_SEC}) >= SHORT_TERM_SEC ({SHORT_TERM_SEC}) may cause segment overlap issues")
    if N_FFT < HOP_LENGTH:
        raise ValueError(f"N_FFT ({N_FFT}) must be >= HOP_LENGTH ({HOP_LENGTH})")
    min_samples = int(SHORT_TERM_SEC * SAMPLE_RATE)
    if min_samples < N_FFT:
        raise ValueError(f"SHORT_TERM_SEC too small: need at least {N_FFT/SAMPLE_RATE:.3f} seconds for N_FFT={N_FFT}")

# ======================
# PLOTTING HELPERS
# ======================

def plot_waveform(segment, sr, fname):
    plt.figure(figsize=(8,2))
    t = np.arange(len(segment)) / sr
    plt.plot(t, segment)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_stft(segment, sr, n_fft, hop_length, fname):
    frames = []
    for start in range(0, len(segment) - n_fft + 1, hop_length):
        frame = segment[start:start + n_fft]
        windowed = frame * np.hanning(n_fft)
        spectrum = np.abs(fftw_rfft(windowed, n=n_fft))
        frames.append(spectrum)
    S = np.stack(frames, axis=1)
    plt.figure(figsize=(7,4))
    plt.imshow(20 * np.log10(S + 1e-10), aspect='auto', origin='lower', extent=(0, S.shape[1]*hop_length/sr, 0, sr/2))
    plt.title("STFT (Spectrogram) [dB]")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_mel(mel_db, fname):
    plt.figure(figsize=(7,4))
    plt.imshow(mel_db, aspect='auto', origin='lower')
    plt.title("Mel Spectrogram [dB]")
    plt.xlabel("Frame")
    plt.ylabel("Mel Band")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def build_auto_cnn_mel(input_shape, n_classes, dropout=0.25, kernel_size=(3, 3), max_poolings=8, n_conv_blocks=None, min_freq_dim=8, min_time_dim=1, n_filters=32):
    model = Sequential()
    model.add(Input(shape=input_shape))
    freq_dim, time_dim, _ = input_shape
    filters = n_filters
    num_blocks = max_poolings
    if n_conv_blocks is not None:
        num_blocks = min(max_poolings, n_conv_blocks)
    for i in range(num_blocks):
        if freq_dim // 2 < min_freq_dim or time_dim // 2 < min_time_dim:
            break
        model.add(Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        freq_dim //= 2
        time_dim //= 2
        filters = min(filters * 2, 512)
    if freq_dim >= min_freq_dim and time_dim >= min_time_dim:
        model.add(Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(Dropout(dropout))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_wavs_and_labels(base_dir):
    wav_files = []
    labels = []
    for label in sorted(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path): continue
        files = glob.glob(os.path.join(label_path, '*.wav'))
        wav_files.extend(files)
        labels.extend([label] * len(files))
    return wav_files, labels

# ======================
# MAIN TRAINING
# ======================

def main():
    validate_parameters()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train standard model (not Optuna)')
    parser.add_argument('--cache', choices=['clear', 'force', 'only'], default=None, help='Cache behavior (clear, force, only)')
    parser.add_argument('--plot', action='store_true', help='Save plots for random sample of segments per class')
    parser.add_argument('--paramfile', type=str, default=None, help='JSON file with Optuna best trial parameters')
    parser.add_argument('--batch_fraction', type=float, default=0.7, help='Fraction of free GPU memory to use for data batch (default: 0.7)')
    parser.add_argument('--batch_size', type=str, default='auto', help='Batch size (int) or "auto"')
    args = parser.parse_args()

    print(f"[i] Using {MAX_THREAD_WORKERS} parallel jobs for data processing (CPU count: {CPU_COUNT})")
    auto_param_path = args.paramfile or os.path.join("plots", "best_trial_params.json")
    optuna_params = None
    if os.path.exists(auto_param_path):
        print(f"[✓] Loading Optuna hyperparameters from {auto_param_path}")
        with open(auto_param_path, "r") as f:
            optuna_params = json.load(f)
    else:
        if args.paramfile:
            print(f"[!] Parameter file '{auto_param_path}' not found. Using built-in defaults.")
        else:
            print(f"[i] No Optuna best parameter JSON found at '{auto_param_path}'. Using built-in defaults.")

    def param(name, default):
        return optuna_params[name] if (optuna_params and name in optuna_params) else default

    KERNEL_SIZE = tuple(map(int, str(param("kernel_size", "3x3")).replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',', 'x').split('x')))
    HOP_LENGTH = param("hop_length", 32)
    LONG_TERM_SEC = param("long_term_sec", 8.0)
    SHORT_TERM_SEC = param("short_term_sec", 1.75)
    STRIDE_SEC = param("stride_sec", 1.7)
    DROPOUT = param("dropout", 0.15)
    N_FILTERS = param("n_filters", 80)
    N_MELS = param("n_mels", 64)
    N_FFT = param("n_fft", 256)
    N_CONV_BLOCKS = param("n_conv_blocks", 3)
    MAX_POOLINGS = param("max_poolings", 2)

    print("[i] Using parameters:")
    print(f"KERNEL_SIZE={KERNEL_SIZE}, HOP_LENGTH={HOP_LENGTH}, LONG_TERM_SEC={LONG_TERM_SEC}, SHORT_TERM_SEC={SHORT_TERM_SEC}, STRIDE_SEC={STRIDE_SEC}, DROPOUT={DROPOUT}, N_FILTERS={N_FILTERS}, N_MELS={N_MELS}, N_FFT={N_FFT}, N_CONV_BLOCKS={N_CONV_BLOCKS}, MAX_POOLINGS={MAX_POOLINGS}")

    if args.train:
        train_dir = os.path.join('/mnt/d/', 'DATASET_REFERENCE', 'TRAINING')
        valid_dir = os.path.join('/mnt/d/', 'DATASET_REFERENCE', 'VALIDATION')
        train_wavs, train_labels = get_wavs_and_labels(train_dir)
        valid_wavs, valid_labels = get_wavs_and_labels(valid_dir)

#        X_train, Y_train, train_label_encoder, train_max_frames = load_wavs_parallel(train_wavs, train_labels, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, long_term_sec=LONG_TERM_SEC, short_term_sec=SHORT_TERM_SEC, stride_sec=STRIDE_SEC, sample_rate=SAMPLE_RATE, augment=False, cache_prefix="cache_train", debug_plot_first=True)
#        X_valid, Y_valid, valid_label_encoder, valid_max_frames = load_wavs_parallel(valid_wavs, valid_labels, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, long_term_sec=LONG_TERM_SEC, short_term_sec=SHORT_TERM_SEC, stride_sec=STRIDE_SEC, sample_rate=SAMPLE_RATE, augment=False, cache_prefix="cache_valid", debug_plot_first=True)
        X_train, Y_train, train_label_encoder, train_max_frames = load_wavs_parallel(train_wavs, train_labels, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, long_term_sec=LONG_TERM_SEC, short_term_sec=SHORT_TERM_SEC, stride_sec=STRIDE_SEC, sample_rate=SAMPLE_RATE, cache_prefix="cache_train")
        X_valid, Y_valid, valid_label_encoder, valid_max_frames = load_wavs_parallel(valid_wavs, valid_labels, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, long_term_sec=LONG_TERM_SEC, short_term_sec=SHORT_TERM_SEC, stride_sec=STRIDE_SEC, sample_rate=SAMPLE_RATE, cache_prefix="cache_valid")


                                                                               #def load_wavs_parallel( wav_files, labels, n_mels, n_fft, hop_length, sample_rate, long_term_sec, short_term_sec, stride_sec, cache_prefix, batch_size, device='cuda', cache_dir="/mnt/d/AILH_CACHE", tmp_dir="/mnt/d/AILH_TMP"):
        
        n_classes = Y_train.shape[1]
        input_shape = X_train.shape[1:]
        y_int = np.argmax(Y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}
        freq_dim, time_dim, _ = input_shape
        max_pools = compute_max_poolings(freq_dim, time_dim, min_freq_dim=8, min_time_dim=1)

        def model_factory():
            return build_auto_cnn_mel(
                input_shape, n_classes,
                dropout=DROPOUT,
                kernel_size=KERNEL_SIZE,
                max_poolings=min(MAX_POOLINGS, max_pools),
                n_conv_blocks=N_CONV_BLOCKS,
                n_filters=N_FILTERS
            )

        model = model_factory()
        print(model.summary())
        callbacks = [
            ModelCheckpoint('cnn_model_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1, save_freq='epoch'),
            EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True, verbose=1)
        ]

        SAFE_BATCH_SIZE = find_safe_batch_size(model, X_train, Y_train, X_valid, Y_valid, callbacks)
        print(f"[✓] Using safe batch size: {SAFE_BATCH_SIZE}")

        history = robust_fit(
            model, X_train, Y_train,
            batch_size=SAFE_BATCH_SIZE,
            epochs=100,
            validation_data=(X_valid, Y_valid),
            callbacks=callbacks
        )
        print(f"[✓] Final model training complete. Batch size used: {SAFE_BATCH_SIZE}")

        model.save("cnn_model_full.h5")
        print(f"[✓] Model saved as cnn_model_best.h5 and cnn_model_full.h5")

        if history is not None and hasattr(history, 'history'):
            plt.figure()
            plt.plot(history.history.get('accuracy', []), label='Train Acc')
            plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training Accuracy')
            plt.savefig("train_accuracy.png")
            plt.close()
            plt.figure()
            plt.plot(history.history.get('loss', []), label='Train Loss')
            plt.plot(history.history.get('val_loss', []), label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training Loss')
            plt.savefig("train_loss.png")
            plt.close()
            print("[✓] Plots saved: train_accuracy.png, train_loss.png")
        else:
            print("[!] No training history available to plot.")

    after_trial_cleanup()

if __name__ == "__main__":
    main()