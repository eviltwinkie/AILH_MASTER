#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Both Binary and Multi-Class Models

Convenience script to train both:
1. Binary LEAK/NOLEAK model
2. Multi-class 5-label model

Usage:
    python train_both_models.py
    
    # Or train individually:
    python dataset_trainer.py      # Multi-class (default)
    python train_binary.py         # Binary mode only
"""

import sys
from pathlib import Path
import time

# Import the training module
import dataset_trainer


def train_model(binary_mode: bool, model_suffix: str):
    """
    Train a model in either binary or multi-class mode.
    
    Args:
        binary_mode: True for LEAK/NOLEAK, False for 5-class
        model_suffix: Suffix to add to model directory (e.g., "_binary" or "_multiclass")
    """
    mode_name = "BINARY (LEAK/NOLEAK)" if binary_mode else "MULTI-CLASS (5 labels)"
    print("=" * 80)
    print(f"TRAINING {mode_name} MODEL")
    print("=" * 80)
    
    # Create config instance with appropriate settings
    cfg = dataset_trainer.Config()
    cfg.binary_mode = binary_mode
    cfg.num_classes = 2 if binary_mode else 5
    cfg.model_dir = Path(f"/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS{model_suffix}")
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    print(f"DEBUG: Training with binary_mode={cfg.binary_mode}, num_classes={cfg.num_classes}, model_dir={cfg.model_dir}")
    
    try:
        # Run the training with our custom config
        start_time = time.time()
        
        dataset_trainer.train(cfg)
        
        elapsed = time.time() - start_time
        
        print("=" * 80)
        print(f"✓ {mode_name} MODEL TRAINING COMPLETE")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ {mode_name} MODEL TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return False


def main():
    """Train both models sequentially."""
    print("=" * 80)
    print("TRAINING BOTH BINARY AND MULTI-CLASS LEAK DETECTION MODELS")
    print("=" * 80)
    print()
    print("This will train two models:")
    print("  1. Binary LEAK/NOLEAK classifier (2 classes)")
    print("  2. Multi-class classifier (5 classes)")
    print()
    print("Models will be saved to separate directories:")
    print("  - PROC_MODELS_binary/")
    print("  - PROC_MODELS_multiclass/")
    print()
    
    overall_start = time.time()
    results = {}
    
    # Train binary model first (typically faster)
    print("[1/2] Training binary model...")
    results['binary'] = train_model(binary_mode=True, model_suffix="_binary")
    
    # Train multi-class model
    print("[2/2] Training multi-class model...")
    results['multiclass'] = train_model(binary_mode=False, model_suffix="_multiclass")
    
    # Summary
    overall_elapsed = time.time() - overall_start
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Binary model:      {'✓ SUCCESS' if results['binary'] else '✗ FAILED'}")
    print(f"Multi-class model: {'✓ SUCCESS' if results['multiclass'] else '✗ FAILED'}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print("=" * 80)
    
    # Exit with error if any training failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
