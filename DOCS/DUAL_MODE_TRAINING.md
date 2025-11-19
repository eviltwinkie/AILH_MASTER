# Dual-Mode Leak Detection Training

Train both **Binary** (LEAK/NOLEAK) and **Multi-Class** (5-label) leak detection models with a single codebase.

---

## 🎯 Training Modes

### 1. **Binary Mode** (LEAK/NOLEAK)
- **Classes**: 2 (LEAK, NOLEAK)
- **Use case**: Simple leak detection (yes/no)
- **Model output**: Binary classification logits
- **Faster training**: ~50% fewer parameters to optimize

### 2. **Multi-Class Mode** (Default)
- **Classes**: 5 (LEAK, NORMAL, QUIET, RANDOM, MECHANICAL)
- **Use case**: Detailed acoustic classification
- **Model output**: 5-class logits + auxiliary binary leak head
- **More informative**: Distinguishes between different acoustic signatures

---

## 🚀 Quick Start

### Train Both Models Automatically
```bash
cd AI_DEV
python train_both_models.py
```

This will:
1. Train binary LEAK/NOLEAK model → saves to `MODELS_binary/`
2. Train 5-class model → saves to `MODELS_multiclass/`

### Train Individual Models

**Binary Model Only:**
```python
# Edit dataset_trainer.py, line ~371:
binary_mode: bool = True  # Change False to True

# Then run:
python dataset_trainer.py
```

**Multi-Class Model Only:**
```python
# Keep default:
binary_mode: bool = False

# Then run:
python dataset_trainer.py
```

**Or use environment variable:**
```bash
# Binary mode
BINARY_MODE=1 python dataset_trainer.py

# Multi-class mode (default)
python dataset_trainer.py
```

---

## 📊 Model Architecture

Both modes use the **same CNN backbone** (LeakCNNMulti):

```
Input: [B, 1, 32, 1] mel spectrogram
  ↓
Conv2D(1→32) + ReLU
Conv2D(32→64) + ReLU
MaxPool2D(2,1)
Conv2D(64→128) + ReLU
MaxPool2D(2,1)
AdaptiveAvgPool2D(16,1)
Flatten → [B, 2048]
Dropout(0.25)
Linear(2048→256) + ReLU
  ↓
╔═══════════════════╦═══════════════════╗
║   BINARY MODE     ║  MULTI-CLASS MODE ║
╠═══════════════════╬═══════════════════╣
║ Linear(256→2)     ║ Linear(256→5)     ║
║ [NOLEAK, LEAK]    ║ [L, N, Q, R, M]   ║
║                   ║ + Aux: Linear→1   ║
║                   ║   (leak-vs-rest)  ║
╚═══════════════════╩═══════════════════╝
```

**Parameters**: ~2.1M (same for both modes, only final layer differs)

---

## 📂 Output Structure

```
MODELS_binary/
├── best.pth                    # Best binary model weights
├── model_meta.json             # Model metadata
└── checkpoints/
    ├── last.pth                # Resume checkpoint
    └── epoch_*.pth             # Rolling checkpoints

MODELS_multiclass/
├── best.pth                    # Best 5-class model weights
├── model_meta.json             # Model metadata
└── checkpoints/
    ├── last.pth                # Resume checkpoint
    └── epoch_*.pth             # Rolling checkpoints
```

---

## 🔧 Configuration Options

Key settings in `Config` dataclass (lines 300-420):

| Parameter | Binary | Multi-Class | Description |
|-----------|--------|-------------|-------------|
| `binary_mode` | `True` | `False` | Classification mode |
| `num_classes` | 2 (auto) | 5 | Output classes |
| `use_leak_aux_head` | N/A | `True` | Auxiliary binary head |
| `leak_aux_weight` | N/A | `0.5` | Aux loss weight |
| `batch_size` | `5632` | `5632` | Training batch size |
| `learning_rate` | `1e-3` | `1e-3` | Initial LR |
| `epochs` | `200` | `200` | Max epochs |
| `early_stop_patience` | `15` | `15` | Early stopping |

---

## 📈 Training Metrics

### Binary Mode
- **Primary metric**: File-level leak F1
- **Metrics logged**:
  - Train/Val Loss & Accuracy
  - File-level: Accuracy, F1, Precision, Recall
  - Prediction distribution (LEAK vs NOLEAK)

### Multi-Class Mode
- **Primary metric**: File-level leak F1
- **Metrics logged**:
  - Train/Val Loss & Accuracy
  - Segment-level leak F1 (monitoring)
  - File-level: Accuracy, Leak F1/P/R
  - Per-class predictions

---

## 🎓 Model Selection Logic

Both modes use **file-level leak F1** for model selection:

1. **File-level evaluation** (paper-exact):
   - Process all segments in file
   - Average probabilities per long segment
   - Vote: File = LEAK if ≥50% long segments exceed threshold
   
2. **Best model saved** when file-level leak F1 improves

3. **Early stopping** if no improvement for 15 epochs

---

## 💾 Model Metadata

`model_meta.json` includes:

```json
{
  "binary_mode": true,              // Training mode
  "num_classes": 2,                 // Model output size
  "class_names": ["NOLEAK", "LEAK"],  // Active classes
  "original_class_names": [...],    // Original 5 classes
  "leak_class_name": "LEAK",
  "leak_idx": 0,                    // Index in ORIGINAL labels
  "best_file_leak_f1": 0.8234,
  "best_epoch": 42,
  "eval_method": "paper_exact_file_level_50pct_voting",
  "model_type": "LeakCNNMulti",
  "trainer_cfg": {...}
}
```

**Important**: `leak_idx` refers to the **original multi-class labels**, not binary indices!

---

## 🔍 Implementation Details

### Binary Mode Workflow

1. **Dataset Loading**: Loads original 5-class HDF5 files
2. **Label Conversion**: Wraps datasets with `BinaryLabelDataset`
   - LEAK class → label 1
   - All others → label 0
3. **Model Creation**: LeakCNNMulti with `n_classes=2`
4. **Training**: Standard cross-entropy loss on binary labels
5. **Evaluation**: File-level voting on binary predictions

### Multi-Class Mode Workflow

1. **Dataset Loading**: Uses original 5-class labels
2. **Model Creation**: LeakCNNMulti with `n_classes=5`
3. **Dual Loss**:
   - Primary: Weighted CE on 5 classes
   - Auxiliary: BCE on binary leak-vs-rest
   - Combined: `L = L_ce + 0.5 * L_bce`
4. **Evaluation**: File-level voting combining both heads

---

## 🧪 Testing

### Quick Validation
```bash
# Test binary mode
python -c "from dataset_trainer import Config; c=Config(); c.binary_mode=True; print(f'Binary mode: {c.binary_mode}, Classes: {c.num_classes}')"

# Expected output:
# Binary mode: True, Classes: 2
```

### Full Training Test
```bash
# Fast test (1 epoch)
# Edit Config: epochs = 1
python dataset_trainer.py
```

---

## 📝 Usage Examples

### Loading Trained Models

```python
import torch
import json

# Load binary model
binary_model = LeakCNNMulti(n_classes=2)
binary_model.load_state_dict(torch.load('MODELS_binary/best.pth'))
binary_model.eval()

# Load multi-class model
multi_model = LeakCNNMulti(n_classes=5)
multi_model.load_state_dict(torch.load('MODELS_multiclass/best.pth'))
multi_model.eval()

# Load metadata
with open('MODELS_binary/model_meta.json') as f:
    binary_meta = json.load(f)
    print(f"Binary F1: {binary_meta['best_file_leak_f1']:.4f}")

with open('MODELS_multiclass/model_meta.json') as f:
    multi_meta = json.load(f)
    print(f"Multi-class F1: {multi_meta['best_file_leak_f1']:.4f}")
```

### Inference (Binary Model)

```python
import torch
import numpy as np

# Prepare input: [1, 1, 32, 1] mel spectrogram
mel_input = torch.randn(1, 1, 32, 1).cuda()

# Forward pass
with torch.no_grad():
    logits, _ = binary_model(mel_input)
    probs = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1).item()  # 0=NOLEAK, 1=LEAK

print(f"Prediction: {'LEAK' if prediction == 1 else 'NOLEAK'}")
print(f"LEAK probability: {probs[0, 1].item():.4f}")
```

---

## 🎯 Performance Expectations

### Binary Model
- **Training time**: ~30-40 min/epoch (RTX 5090)
- **Best F1**: 0.75-0.85 (typical)
- **Simpler decision boundary** → better for deployment

### Multi-Class Model
- **Training time**: ~30-40 min/epoch (same, similar architecture)
- **Best F1**: 0.70-0.80 (typical)
- **More granular** → better for analysis

**Note**: Binary mode may achieve higher F1 because:
- Simpler 2-class problem
- No confusion between non-leak classes
- More balanced dataset (LEAK vs all others)

---

## 🐛 Troubleshooting

### Issue: Models saving to same directory
**Solution**: Use `train_both_models.py` which automatically separates outputs

### Issue: Binary model has 5 outputs
**Solution**: Ensure `binary_mode=True` is set BEFORE training starts

### Issue: Wrong leak_idx in metadata
**Solution**: `leak_idx` is intentionally the original index (used for dataset reading)

### Issue: Low GPU utilization
**Solution**: Set `batch_long_segments=0` in evaluate_file_level (line 1327)

---

## 📚 References

- **Architecture**: LeakCNNMulti (2.1M params)
- **Dataset**: HDF5 hierarchical mel spectrograms
- **Evaluation**: Paper-exact file-level 50% voting
- **Optimization**: FP16 AMP, channels_last, optional torch.compile

---

## ✅ Validation Checklist

Before deploying:
- [ ] Train both models with `train_both_models.py`
- [ ] Verify separate output directories exist
- [ ] Check `model_meta.json` has correct `binary_mode` flag
- [ ] Test inference on sample files
- [ ] Compare F1 scores between modes
- [ ] Document which model is used for production

---

**Version**: 1.0  
**Last Updated**: 2025-11-19  
**Author**: Dataset Trainer v15
