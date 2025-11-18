import os

from old_config import SAMPLE_RATE, CACHE_DIR

from cnn_mel_processor import compute_max_poolings, load_wavs_parallel
from cnn_mel_trainer import get_wavs_and_labels, build_auto_cnn_mel, hash_files, robust_fit, find_safe_batch_size, after_trial_cleanup

import argparse
import json
import numpy as np
import tensorflow as tf
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback
import plotly.io as pio


# try:
#     # Get list of physical GPUs
#     physical_devices = tf.config.list_physical_devices('GPU')
#     if physical_devices:
#         for device in physical_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#             #tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
#         print(f"[✓] Configured {len(physical_devices)} GPU(s) with memory growth enabled")        
#         # try:
#         #     with tf.device('/GPU:0'):
#         #         test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#         #         result = tf.matmul(test_tensor, test_tensor)
#         #         # Test cuDNN specifically
#         #         conv_test = tf.keras.layers.Conv2D(1, (3, 3))
#         #         test_input = tf.random.normal((1, 8, 8, 1))
#         #         conv_result = conv_test(test_input)
#         #     print("[✓] GPU test successful - cuDNN libraries working")
#         #     print("[✓] GPU acceleration enabled for training")
#         # except Exception as gpu_test_error:
#         #     print(f"[!] GPU test failed: {gpu_test_error}")
#         #     print("[!] Falling back to CPU-only mode")
#         #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only  
#     else:
#         print("[!] No GPUs detected, running on CPU")
# except Exception as e:
#     print(f"[!] GPU configuration error: {e}")
#     print("[!] Falling back to CPU-only mode")
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', action='store_true', help='Delete Optuna DB and start fresh for this data')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel Optuna jobs (default: 1, recommended for GPU)')
    args = parser.parse_args()

    train_dir = os.path.join('/mnt/d/', 'DATASET_REFERENCE', 'TRAINING')
    valid_dir = os.path.join('/mnt/d/', 'DATASET_REFERENCE', 'VALIDATION')
    train_wavs, train_labels = get_wavs_and_labels(train_dir)
    valid_wavs, valid_labels = get_wavs_and_labels(valid_dir)

    # data_hash = hash_files(train_wavs + valid_wavs)
    # study_name = f"cnn_mel_study_{data_hash}"
    # storage = f"sqlite:///optuna_study_{data_hash}.db"
    # db_path = f"optuna_study_{data_hash}.db"
    # plots_dir = "OPTUNA_PLOTS"
    #os.makedirs(plots_dir, exist_ok=True)

    # --- Main output paths ---
    data_hash = hash_files(train_wavs + valid_wavs)
    study_name = f"cnn_mel_study_{data_hash}"

    # Make sure all Optuna/plots/db go to CACHE_DIR
    optuna_db_name = f"optuna_study_{data_hash}.db"
    db_path = os.path.join(CACHE_DIR, optuna_db_name)
    storage = f"sqlite:///{db_path}"        # absolute sqlite path

    plots_dir = os.path.join(CACHE_DIR, "OPTUNA_PLOTS")
    os.makedirs(plots_dir, exist_ok=True)

    print("[AUDIT] Optuna DB path:", db_path)
    print("[AUDIT] Plots directory:", plots_dir)

    if args.restart:
        if os.path.exists(db_path):
            print(f"[i] --restart specified: Deleting study database {db_path}")
            os.remove(db_path)
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"[i] Optuna study '{study_name}' deleted from storage.")
        except Exception as e:
            print(f"[i] Could not delete Optuna study from storage (may not exist yet): {e}")

    kernel_size_strs = ['3x3', '5x5', '7x3', '3x7']

    def objective(trial):
        try:
            kernel_size_str = trial.suggest_categorical('kernel_size', kernel_size_strs)
            kernel_size = tuple(map(int, kernel_size_str.split('x')))

            hop_length = trial.suggest_categorical("hop_length", [32, 64, 128, 256, 512, 768, 1024, 2048])
            
            # More conservative parameter ranges to avoid "No segments extracted" errors
            long_term_sec = trial.suggest_float("long_term_sec", 1.5, 5.0, step=0.5)  # Reduced from 1.5-10.0
            short_term_sec = trial.suggest_float("short_term_sec", 0.5, 2.5, step=0.25)  # Reduced from 0.25-5.0
            stride_sec = trial.suggest_float("stride_sec", 0.2, 1.0, step=0.1)  # Reduced from 0.1-2.0
            
            # Ensure parameter relationships are valid
            if short_term_sec >= long_term_sec:
                short_term_sec = long_term_sec * 0.8  # Ensure short < long
            if stride_sec >= short_term_sec:
                stride_sec = short_term_sec * 0.5  # Ensure stride < short
            
            dropout = trial.suggest_float("dropout", 0.15, 0.5, step=0.05)
            n_filters = trial.suggest_categorical("n_filters", [16, 32, 48, 64, 80])
            n_mels = trial.suggest_categorical("n_mels", [64, 80, 96, 128, 160])
            n_fft = trial.suggest_categorical("n_fft", [128, 256, 512, 1024, 2048])
            n_conv_blocks = trial.suggest_int("n_conv_blocks", 2, 5)

            segment_samples = int(long_term_sec * SAMPLE_RATE)
            n_frames = 1 + int((segment_samples - n_fft) / hop_length)
            input_shape = (None, n_mels, n_frames, 1) 

            print(f"[Trial {trial.number}] Testing parameters: long={long_term_sec:.1f}s, short={short_term_sec:.1f}s, stride={stride_sec:.1f}s")

            if n_mels < kernel_size[0] or n_frames < kernel_size[1]:
                print(f"[Trial {trial.number}] ❌ Kernel {kernel_size} doesn't fit for input shape {input_shape} - trial pruned")
                raise optuna.TrialPruned()

            try:
                #X_train, Y_train, train_label_encoder, train_max_frames = load_wavs_parallel(train_wavs, train_labels, hop_length=hop_length, long_term_sec=long_term_sec, short_term_sec=short_term_sec, stride_sec=stride_sec, n_mels=n_mels, n_fft=n_fft, sample_rate=SAMPLE_RATE, augment=True, cache_prefix=f"optuna_train_trial_{trial.number}", debug_plot_first=False)
                X_train, Y_train, train_label_encoder, train_max_frames = load_wavs_parallel(train_wavs, train_labels, hop_length=hop_length, long_term_sec=long_term_sec, short_term_sec=short_term_sec, stride_sec=stride_sec, n_mels=n_mels, n_fft=n_fft, sample_rate=SAMPLE_RATE, cache_prefix=f"optuna_train_trial_{trial.number}")
            except ValueError as ve:
                if "No segments were extracted" in str(ve):
                    print(f"[Trial {trial.number}] ❌ No segments extracted with these parameters - skipping trial")
                    return float("nan")  # Skip this trial
                else:
                    raise ve
            try:
                #X_valid, Y_valid, valid_label_encoder, valid_max_frames = load_wavs_parallel(valid_wavs, valid_labels, hop_length=hop_length, long_term_sec=long_term_sec, short_term_sec=short_term_sec, stride_sec=stride_sec, n_mels=n_mels, n_fft=n_fft, sample_rate=SAMPLE_RATE, augment=False, cache_prefix=f"optuna_valid_trial_{trial.number}", debug_plot_first=False)
                X_valid, Y_valid, valid_label_encoder, valid_max_frames = load_wavs_parallel(valid_wavs, valid_labels, hop_length=hop_length, long_term_sec=long_term_sec, short_term_sec=short_term_sec, stride_sec=stride_sec, n_mels=n_mels, n_fft=n_fft, sample_rate=SAMPLE_RATE, cache_prefix=f"optuna_valid_trial_{trial.number}")
            except ValueError as ve:
                if "No segments were extracted" in str(ve):
                    print(f"[Trial {trial.number}] ❌ No validation segments extracted - skipping trial")
                    return float("nan")  # Skip this trial
                else:
                    raise ve
                
            n_classes = Y_train.shape[1]
            input_shape = X_train.shape[1:]
            freq_dim, time_dim, _ = input_shape
            min_freq_dim, min_time_dim = 8, 1
            max_possible_pools = compute_max_poolings(freq_dim, time_dim, min_freq_dim, min_time_dim)
            max_poolings = trial.suggest_int("max_poolings", 0, max_possible_pools)

            y_int = np.argmax(Y_train, axis=1)
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
            class_weights_dict = {i: w for i, w in enumerate(class_weights)}

            model = build_auto_cnn_mel(
                input_shape, n_classes,
                dropout=dropout,
                max_poolings=max_poolings,
                kernel_size=kernel_size,
                n_conv_blocks=n_conv_blocks,
                n_filters=n_filters,
            )
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            callbacks = [
                TFKerasPruningCallback(trial, 'val_accuracy'),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=0)
            ]

            try:
                batch_size = find_safe_batch_size(
                    lambda: build_auto_cnn_mel(
                        input_shape, n_classes,
                        dropout=dropout,
                        max_poolings=max_poolings,
                        kernel_size=kernel_size,
                        n_conv_blocks=n_conv_blocks,
                        n_filters=n_filters
                    ),
                    X_train, Y_train, X_valid, Y_valid, callbacks
                )
                print(f"[✓] Using safe batch size: {batch_size}")
            except tf.errors.FailedPreconditionError as fpe:
                print(f"[Trial {trial.number}] ❌ cuDNN initialization failed - GPU may be out of memory")
                print(f"    Error: {fpe}")
            except Exception as e:
                print(f"[Trial {trial.number}] ❌ Batch size detection failed: {e}")
                return float("nan")

            try:
                history = robust_fit(model, X_train, Y_train, batch_size=batch_size, epochs=40, validation_data=(X_valid, Y_valid), callbacks=callbacks)
                if history is not None and hasattr(history, "history"):
                    best_val_acc = max(history.history.get('val_accuracy', [0.0]))
                else:
                    best_val_acc = 0.0
            except tf.errors.FailedPreconditionError as fpe:
                print(f"[Trial {trial.number}] ❌ cuDNN error during training: {fpe}")
            except Exception as e:
                print(f"[Trial {trial.number}] ❌ Training failed: {e}")
                return float("nan")
            return best_val_acc
        except ValueError as e:
            print(f"[Optuna] Trial failed: {e}")
            return float("nan")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True
    )
    print(f"[✓] Optuna study loaded: {study_name}, auto-resume enabled for this exact dataset.")

    def wrapped_objective(trial):
        after_trial_cleanup()
        try:
            return objective(trial)
        finally:
            after_trial_cleanup()

    study.optimize(
        wrapped_objective,
        n_trials=args.n_trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
        gc_after_trial=True
    )

    best_json = os.path.join(plots_dir, "best_trial_params.json")
    with open(best_json, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"[✓] Best trial parameters saved to {best_json}")

    plots = {
        "optimization_history.png": vis.plot_optimization_history(study),
        "param_importances.png": vis.plot_param_importances(study),
        "intermediate_values.png": vis.plot_intermediate_values(study)
    }
    for fname, fig in plots.items():
        path = os.path.join(plots_dir, fname)
        pio.write_image(fig, path, format="png", scale=2)
        print(f"[✓] Saved plot: {path}")

    print("\n--- Optuna Hyperparameter Search Complete ---")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Plots saved to: {plots_dir}")

    print("\n[Dashboard]")
    print("To install optuna-dashboard, run:")
    print("    pip install optuna-dashboard")
    print("To launch the dashboard, run:")
    print(f"    optuna-dashboard {storage} --study-name {study_name}")
    print("Then open the provided URL in your browser.")
    print("\n[GPU METRICS]")
    print("If you want to see GPU memory per trial, install pynvml (pip install pynvml).")
    print("These will appear as user attributes in the dashboard (see above).")

if __name__ == "__main__":
    main()