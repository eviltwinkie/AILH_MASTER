# Binary Mode Bug Fix Summary

## Issue
Binary mode training crashed with:
```
IndexError: index 2 is out of bounds for dimension 1 with size 2
```

## Root Cause
When `binary_mode=True`:
- Model outputs 2 classes: [NOLEAK, LEAK] where LEAK is at index 1
- But code was using original dataset `leak_idx=2` (from 5-class dataset)
- Accessing `logits[:, 2]` failed because model only has indices [0, 1]

## Fixes Applied

### 1. Loss Function Weights (lines 1168-1230)
**Problem**: Always computed 5-class weights even in binary mode
**Solution**: Added binary mode handling:
```python
if cfg.binary_mode:
    # Compute weights for [NOLEAK, LEAK]
    leak_count = (labels == leak_idx).sum()
    noleak_count = len(labels) - leak_count
    # ... compute 2-class weights
else:
    # Original 5-class weights
    counts = class_counts_from_labels(ds_tr)
```

### 2. Model vs Dataset Leak Index (lines 2248-2252)
**Problem**: Single `leak_idx` variable used for both dataset labels and model outputs
**Solution**: Created separate indices:
```python
# dataset_leak_idx: Original index in HDF5 labels (2 for LEAK in 5-class)
# model_leak_idx: Index in model output (1 for binary, 2 for multi-class)
model_leak_idx = 1 if cfg.binary_mode else leak_idx
```

### 3. evaluate_file_level() Function (lines 1428-1432)
**Problem**: Used `leak_idx` to index model outputs `logits[:, leak_idx]`
**Solution**: Split into two parameters:
- `dataset_leak_idx`: For comparing file labels
- `model_leak_idx`: For indexing model outputs

Changed:
```python
# Before:
p_cls = torch.softmax(logits, dim=1)[:, leak_idx]  # ❌ Crashes in binary mode

# After:
p_cls = torch.softmax(logits, dim=1)[:, model_leak_idx]  # ✅ Uses correct index
```

## Testing
To test binary mode:
```bash
cd /DEVELOPMENT/ROOT_AILH/REPOS/AILH_MASTER/AI_DEV
python train_both_models.py
```

The script will:
1. Train binary model (LEAK/NOLEAK) - should complete without IndexError
2. Train multi-class model (5 classes) - should still work as before

## What Changed
- ✅ Binary mode: model outputs [NOLEAK, LEAK], uses index 1 for LEAK
- ✅ Multi-class mode: model outputs 5 classes, uses index 2 for LEAK
- ✅ Loss weights match model output size
- ✅ Evaluation uses correct index for leak probability

## Files Modified
- `dataset_trainer.py`: Added `model_leak_idx` parameter throughout
- Lines changed: 1168-1230 (loss), 1428-1432 (evaluate_file_level), 2095-2100 (run_training_loop), 2175 (run_final_test), 2248-2252 (train)
