# Binary Mode Training Failure - Root Cause Analysis

## CRITICAL ISSUE FOUND ⚠️

**Model predicts 0 LEAK files when there are 1687 LEAK files (42.6% of dataset)**

---

## Root Cause

The training script **`train_both_models.py` failed to enable binary mode**. Despite attempting to modify the code dynamically, the model was trained in **5-class multi-class mode** instead of **2-class binary mode**.

### Evidence

```bash
$ python3 -c "import torch; state = torch.load('checkpoints/last.pth'); print(state['config']['binary_mode'], state['config']['num_classes'])"
False 5
```

**Checkpoint shows**:
- `binary_mode: False` (should be True)
- `num_classes: 5` (should be 2)
- `class_names: ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']` (should be ['NOLEAK', 'LEAK'])
- `cls_head.weight: shape=[5, 256]` (should be [2, 256])

---

## Why `train_both_models.py` Failed

**Line 47-49 of `train_both_models.py`**:
```python
modified_content = content.replace(
    "binary_mode: bool = False",
    "binary_mode: bool = True"
)
```

**Actual line in `dataset_trainer.py` (line 368)**:
```python
binary_mode: bool = False           # True: LEAK/NOLEAK binary classification, False: All classes
```

**Problem**: The `replace()` looks for exact string `"binary_mode: bool = False"` but the actual line has trailing comment. The replacement **silently fails**, leaving `binary_mode=False`.

---

## Why Model Predicts 0 LEAK

The 5-class model was trained but evaluated as if it were binary:

1. **Training**: Model learned 5 classes [BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED]
2. **Evaluation**: Code tried to extract `P(LEAK)` from index 1 (binary mode)
3. **Reality**: Index 1 in 5-class model is CRACK, not LEAK!
4. **Result**: Model correctly has low confidence for CRACK → evaluates as NOLEAK

The model was actually trained on 5 classes but the evaluation code was using binary mode logic with `model_leak_idx=1`, which points to the wrong class!

---

## Fix Implemented

### Solution 1: Direct Wrapper Script (RECOMMENDED)

Created **`train_binary.py`** that directly overrides Config:

```python
from pathlib import Path
import dataset_trainer

class BinaryConfig(dataset_trainer.Config):
    def __post_init__(self):
        super().__post_init__()
        self.binary_mode = True
        self.num_classes = 2
        self.model_dir = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS_BINARY")

dataset_trainer.Config = BinaryConfig
dataset_trainer.train()
```

**Usage**:
```bash
cd /DEVELOPMENT/ROOT_AILH/REPOS/AILH_MASTER/AI_DEV
python train_binary.py
```

### Solution 2: Fix `train_both_models.py` (Alternative)

Update line 47-49 to handle comment:

```python
# Use regex for robust replacement
import re
modified_content = re.sub(
    r'binary_mode:\s*bool\s*=\s*False.*',
    'binary_mode: bool = True  # Modified by train_both_models.py',
    content
)
```

---

## Dataset Statistics

The dataset is well-balanced, so the model should be able to learn:

| Split | Total Files | LEAK Files | NOLEAK Files | LEAK % |
|-------|-------------|------------|--------------|--------|
| Train | 39,563 | 17,224 | 22,339 | 43.5% |
| Val | 7,913 | 3,454 | 4,459 | 43.6% |
| Test | 3,958 | 1,687 | 2,271 | 42.6% |

**Imbalance ratio**: 1.30:1 (NOLEAK:LEAK) - very reasonable!

---

## Action Items

### IMMEDIATE (Critical)
1. ✅ Created `train_binary.py` wrapper script
2. ⏳ **Run actual binary training**: `python train_binary.py`
3. ⏳ Verify checkpoint has `binary_mode=True, num_classes=2`

### After Binary Training Completes
1. Check that model learns (LEAK F1 > 0)
2. Compare binary vs multi-class performance
3. Debug if still failing to predict LEAK

### Optional
1. Fix `train_both_models.py` to use regex replacement
2. Add validation check to ensure binary_mode is set correctly
3. Add early warning if checkpoint config doesn't match expected mode

---

## Expected Behavior (After Fix)

### Training Output Should Show:
```
[INFO] [BINARY MODE] Converting to LEAK/NOLEAK classification
[INFO] Model class names: ['NOLEAK', 'LEAK']
[INFO] Model will output 2 classes
```

### Checkpoint Should Have:
```python
config['binary_mode'] = True
config['num_classes'] = 2
class_names = ['NOLEAK', 'LEAK']
model['cls_head.weight'].shape = [2, 256]  # Not [5, 256]!
```

### Evaluation Should Predict:
```
Files predicted as LEAK: ~1500-1800 / 3958  # Not 0!
Leak F1: >0.60  # Not 0.0000!
```

---

## Prevention

### Add Config Validation (Recommended)

Add to `train()` function after dataset setup:

```python
# Verify binary mode is correctly configured
if cfg.binary_mode:
    assert cfg.num_classes == 2, f"binary_mode=True requires num_classes=2, got {cfg.num_classes}"
    assert len(class_names) == 2, f"binary_mode=True requires 2 class names, got {len(class_names)}"
    logger.info("✓ Binary mode validation passed")
```

### Add Checkpoint Verification

Add to checkpoint loading:

```python
if cfg.binary_mode != ckpt['config']['binary_mode']:
    raise ValueError(
        f"Config mismatch! Current binary_mode={cfg.binary_mode}, "
        f"checkpoint binary_mode={ckpt['config']['binary_mode']}"
    )
```

---

## Testing the Fix

```bash
# 1. Train binary model
python train_binary.py

# 2. After a few epochs, check checkpoint
python3 -c "
import torch
state = torch.load('/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS_BINARY/checkpoints/last.pth')
print(f\"binary_mode: {state['config']['binary_mode']}\")
print(f\"num_classes: {state['config']['num_classes']}\")
print(f\"class_names: {state['class_names']}\")
print(f\"cls_head shape: {list(state['model']['cls_head.weight'].shape)}\")
"

# 3. Should output:
# binary_mode: True
# num_classes: 2
# class_names: ['NOLEAK', 'LEAK']
# cls_head shape: [2, 256]
```

---

## Summary

- ❌ **Previous run**: Trained 5-class model, evaluated as binary → 0 LEAK predictions
- ✅ **Solution**: Created `train_binary.py` wrapper that correctly enables binary mode
- ⏳ **Next step**: Run `python train_binary.py` and verify model learns

The issue was **silently failing code modification**, not a fundamental model or dataset problem. The model should train successfully with the corrected configuration.
