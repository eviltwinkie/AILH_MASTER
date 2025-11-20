# Tuner Comparison: cnn_mel_tuner.py vs dataset_tuner.py

## Overview

Both tuners use Optuna for hyperparameter optimization but target different aspects of the machine learning pipeline:

- **cnn_mel_tuner.py**: TensorFlow/Keras-based tuner optimizing **model architecture** parameters
- **dataset_tuner.py**: PyTorch-based tuner optimizing **training hyperparameters**

## Key Differences

### Framework & Target
| Feature | cnn_mel_tuner.py | dataset_tuner.py |
|---------|------------------|------------------|
| Framework | TensorFlow 2.x + Keras | PyTorch 2.9.1 |
| Model | Custom Keras Sequential | LeakCNNMulti (fixed architecture) |
| Data Source | WAV files (on-the-fly loading) | HDF5 mel spectrograms (preloaded to RAM) |
| Optimization Target | Architecture parameters | Training hyperparameters |

### Parameters Optimized

**cnn_mel_tuner.py** (Architecture):
- `n_filters`: [8, 16, 32, 64] - Base filter count
- `n_mels`: [32, 64, 128, 256] - Mel spectrogram resolution
- `n_fft`: [256, 512, 1024, 2048] - FFT window size
- `kernel_size`: [3, 5, 7] - Convolutional kernel dimensions
- `n_conv_blocks`: [2, 3, 4] - Number of convolutional layers
- `max_poolings`: [0, 1, 2, 3] - Number of max pooling layers
- `dense_units`: [64, 128, 256] - Hidden layer size
- `dropout`: [0.1, 0.3, 0.5] - Dropout rate
- `learning_rate`: [1e-5, 1e-2] - Optimizer learning rate
- `batch_size`: Dynamic (tested via GPU memory)

**dataset_tuner.py** (Training):
- `learning_rate`: [1e-4, 1e-2] (log scale) - AdamW learning rate
- `batch_size`: [4096, 8192, 12288, 16384] - Batch size (categorical)
- `dropout`: [0.1, 0.5] (step 0.05) - Model dropout rate
- `grad_clip_norm`: [0.5, 2.0] - Gradient clipping threshold
- `loss_type`: ["weighted_ce", "focal"] - Loss function type
- `focal_gamma`: [1.0, 3.0] - Focal loss focusing parameter
- `focal_alpha_leak`: [0.5, 0.9] - Focal loss class weighting
- `leak_aux_weight`: [0.2, 0.8] - Auxiliary head loss weight
- `early_stop_patience`: [5, 20] - Early stopping patience

### Memory Management

**cnn_mel_tuner.py** - Robust cleanup:
```python
def after_trial_cleanup():
    """Aggressively clean up memory between trials."""
    K.clear_session()  # Clear Keras session
    tf.compat.v1.reset_default_graph()  # Reset TensorFlow graph
    gc.collect()  # Python garbage collection
```

**dataset_tuner.py** - Basic cleanup:
```python
def cleanup_after_trial():
    """Clean up GPU memory and caches after trial."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()
```

Both use wrapped objectives that cleanup before and after trials:
```python
def wrapped_objective(trial):
    cleanup_after_trial()  # Before
    try:
        return objective(trial, base_cfg, tuning_cfg)
    finally:
        cleanup_after_trial()  # After
```

And enable garbage collection in study.optimize():
```python
study.optimize(..., gc_after_trial=True)
```

### Batch Size Optimization

**cnn_mel_tuner.py** - Dynamic GPU testing:
```python
def find_safe_batch_size(dataset_len, model_fn, initial_batch, max_attempts=10):
    """
    Binary search for largest batch size that fits in GPU memory.
    Tests actual model forward passes to detect OOM conditions.
    """
    # Tests batches from initial_batch down to initial_batch/2^max_attempts
    # Returns largest successful batch size
```

**dataset_tuner.py** - Fixed categories:
```python
cfg.batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 12288, 16384])
```

### Data Pipeline

**cnn_mel_tuner.py**:
- Loads WAV files on-the-fly during training
- Uses `tf.data.Dataset` with caching
- Computes mel spectrograms in real-time
- Cache size: 1GB for spectrogram cache

**dataset_tuner.py**:
- Pre-computed HDF5 mel spectrograms
- **RAM preloading**: Entire dataset loaded to memory (6.5GB train, 1.3GB val)
- Zero I/O latency during training
- Enables 82-92% GPU utilization

### Trial Lifecycle

**Both tuners follow similar patterns:**

1. **Setup Phase**:
   - Generate trial config from suggestions
   - Load/prepare datasets
   - Create model
   - Setup optimizer, loss functions

2. **Training Phase**:
   - Train for N epochs (cnn_mel: 10-30, dataset: 50)
   - Validate after each epoch
   - Report intermediate metric to Optuna
   - Check for pruning (`trial.should_prune()`)

3. **Cleanup Phase**:
   - Return best validation metric
   - Cleanup GPU memory
   - Handle exceptions (OOM, pruning, errors)

### Error Handling

**cnn_mel_tuner.py**:
```python
try:
    # Training logic
    return best_val_f1
except Exception as e:
    logger.error(f"Trial {trial.number} failed: {e}")
    return float("nan")  # Invalid trial marker
finally:
    after_trial_cleanup()
```

**dataset_tuner.py**:
```python
try:
    # Training logic
    return best_val_f1
except optuna.TrialPruned:
    raise  # Let Optuna handle pruning
except torch.cuda.OutOfMemoryError:
    logger.error(f"Trial {trial.number} failed: GPU OOM")
    cleanup_after_trial()
    return 0.0  # Poor score instead of nan
except Exception as e:
    logger.error(f"Trial {trial.number} failed: {e}")
    traceback.print_exc()
    cleanup_after_trial()
    return 0.0
```

### Optimization Strategy

Both use:
- **Sampler**: TPESampler (Tree-structured Parzen Estimator) - Bayesian optimization
- **Pruner**: MedianPruner - Terminates unpromising trials early
  - `n_startup_trials=5`: No pruning for first 5 trials
  - `n_warmup_steps=3`: Wait 3 epochs before pruning
- **Storage**: SQLite database for persistence
- **Direction**: Maximize (file-level leak F1 score)

### Pruning Strategy

**cnn_mel_tuner.py**:
```python
# Uses Keras callback
pruning_callback = TFKerasPruningCallback(trial, "val_leak_f1")
model.fit(..., callbacks=[pruning_callback])
```

**dataset_tuner.py**:
```python
# Manual pruning after each epoch
trial.report(val_f1, epoch)
if epoch >= tuning_cfg.pruning_warmup and trial.should_prune():
    raise optuna.TrialPruned()
```

### Output & Visualization

Both generate:
- `study.db`: SQLite database with trial history
- `best_params.json`: Best parameters found
- Visualization plots:
  - Optimization history (F1 vs trial number)
  - Parameter importances (feature importance)
  - Parallel coordinate plot (parameter relationships)
  - Slice plots (individual parameter effects)

### Performance Considerations

**cnn_mel_tuner.py**:
- Slower per trial (WAV loading + mel computation)
- Tests architectural variations (more exploration)
- Typical trial time: 5-15 minutes

**dataset_tuner.py**:
- Fast per trial (RAM preloading)
- Fixed architecture (focused optimization)
- Typical trial time: 2-5 minutes
- **9x faster** than original pipeline (3 hours → 20 min for 200 epochs)
- GPU utilization: 82-92% (vs 11% before RAM preloading)

## Recommendations

### Use cnn_mel_tuner.py when:
- Exploring different model architectures
- Working with new datasets requiring mel parameter tuning
- Need to optimize FFT/STFT parameters
- Starting a new project from scratch

### Use dataset_tuner.py when:
- Model architecture is fixed
- Optimizing training dynamics (loss functions, learning rate, etc.)
- Fast iteration is critical
- HDF5 datasets are available
- Have sufficient RAM for preloading (8GB+ datasets)

## Integration Pattern

**Typical workflow:**
1. **Phase 1**: Use `cnn_mel_tuner.py` to find optimal architecture
   - Determines best n_mels, n_fft, n_filters, kernel_size, etc.
   - Results in best model architecture

2. **Phase 2**: Use `dataset_tuner.py` to optimize training
   - Fine-tune learning rate, batch size, loss function
   - Optimize dropout, gradient clipping, early stopping
   - Results in best training configuration

3. **Production**: Train final model with combined best parameters

## Future Enhancements

### For dataset_tuner.py:
1. **Add dynamic batch size testing** (port `find_safe_batch_size` from cnn_mel)
2. **Enhance cleanup** (more aggressive memory management)
3. **Add warmup tuning** (optimize learning rate warmup schedule)
4. **Test optimizer types** (AdamW vs Adam vs SGD)
5. **Add augmentation tuning** (SpecAugment time/freq mask parameters)

### For cnn_mel_tuner.py:
1. **Port to PyTorch** for consistency with dataset_trainer
2. **Add RAM preloading** for faster iteration
3. **Test attention mechanisms** (self-attention layers)
4. **Explore EfficientNet-style scaling** (compound scaling)

## Conclusion

Both tuners complement each other:
- **cnn_mel_tuner**: Architectural exploration (what to build)
- **dataset_tuner**: Training optimization (how to train it)

Together they provide comprehensive hyperparameter optimization covering both model design and training dynamics.
