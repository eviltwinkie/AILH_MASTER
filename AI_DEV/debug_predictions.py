#!/usr/bin/env python3
"""
Debug script to inspect model outputs and understand why it's not predicting LEAK.
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_trainer import LeakMelDataset, LeakCNNMulti, BinaryLabelDataset, prepare_mel_batch

MODEL_PATH = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/best.pth")
VAL_H5 = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/MASTER_DATASET/VALIDATION_DATASET.H5")

def main():
    device = torch.device("cuda")
    
    # Load model
    print("Loading model...")
    model = LeakCNNMulti(n_classes=2, dropout=0.25).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    
    # Load dataset
    print("Loading validation dataset...")
    ds = LeakMelDataset(VAL_H5)
    ds_binary = BinaryLabelDataset(ds, leak_idx=2)  # LEAK is at index 2 in original labels
    
    # Check actual vs predicted labels for first 100 files
    print("\nChecking predictions for first 100 files...")
    print(f"{'File':<6} {'True':<8} {'Pred':<8} {'P(LEAK)':<10} {'P(AUX)':<10} {'Combined':<10}")
    print("=" * 70)
    
    leak_probs = []
    true_labels = []
    
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for file_idx in range(min(100, len(ds))):
            # Get file data
            blk = ds._segs[file_idx]  # [num_long, num_short, n_mels, t_frames]
            true_label = int(ds._labels[file_idx])
            binary_label = 1 if true_label == 2 else 0
            
            # Flatten to segments
            segments = blk.reshape(-1, *blk.shape[-2:])  # [num_long * num_short, n_mels, t_frames]
            
            # Prepare for model
            mel_t = prepare_mel_batch(segments, device, use_channels_last=True)
            
            # Forward pass
            logits, leak_logit = model(mel_t)
            
            # Compute probabilities
            p_cls = torch.softmax(logits, dim=1)[:, 1]  # P(LEAK) from classification head
            p_aux = torch.sigmoid(leak_logit)  # P(LEAK) from auxiliary head
            p_combined = 0.5 * (p_cls + p_aux)
            
            # Average across all segments for file-level prediction
            avg_p_cls = p_cls.mean().item()
            avg_p_aux = p_aux.mean().item()
            avg_combined = p_combined.mean().item()
            
            pred_label = 1 if avg_combined >= 0.5 else 0
            
            leak_probs.append(avg_combined)
            true_labels.append(binary_label)
            
            if file_idx < 20:  # Print first 20
                print(f"{file_idx:<6} {'LEAK' if binary_label == 1 else 'NOLEAK':<8} "
                      f"{'LEAK' if pred_label == 1 else 'NOLEAK':<8} "
                      f"{avg_p_cls:<10.4f} {avg_p_aux:<10.4f} {avg_combined:<10.4f}")
    
    # Statistics
    leak_probs = np.array(leak_probs)
    true_labels = np.array(true_labels)
    
    print("\n" + "=" * 70)
    print("\nProbability Statistics:")
    print(f"  Min P(LEAK): {leak_probs.min():.6f}")
    print(f"  Max P(LEAK): {leak_probs.max():.6f}")
    print(f"  Mean P(LEAK): {leak_probs.mean():.6f}")
    print(f"  Median P(LEAK): {np.median(leak_probs):.6f}")
    
    print("\nFor TRUE LEAK files:")
    leak_mask = true_labels == 1
    if leak_mask.sum() > 0:
        print(f"  Mean P(LEAK): {leak_probs[leak_mask].mean():.6f}")
        print(f"  Max P(LEAK): {leak_probs[leak_mask].max():.6f}")
    
    print("\nFor TRUE NOLEAK files:")
    noleak_mask = true_labels == 0
    if noleak_mask.sum() > 0:
        print(f"  Mean P(LEAK): {leak_probs[noleak_mask].mean():.6f}")
        print(f"  Max P(LEAK): {leak_probs[noleak_mask].max():.6f}")
    
    # Confusion matrix at different thresholds
    print("\nConfusion Matrix at Different Thresholds:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (leak_probs >= thresh).astype(int)
        tp = ((preds == 1) & (true_labels == 1)).sum()
        fp = ((preds == 1) & (true_labels == 0)).sum()
        fn = ((preds == 0) & (true_labels == 1)).sum()
        tn = ((preds == 0) & (true_labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Threshold {thresh:.1f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} "
              f"(TP={tp}, FP={fp}, FN={fn}, TN={tn})")

if __name__ == "__main__":
    main()
