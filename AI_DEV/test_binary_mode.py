#!/usr/bin/env python3
"""Quick test to verify binary mode training works correctly."""

import sys
from pathlib import Path

# Update Config to test binary mode
import dataset_trainer
from dataset_trainer import Config

# Create test config with binary_mode=True
cfg = Config()
cfg.binary_mode = True
cfg.epochs = 2  # Just 2 epochs for quick test
cfg.auto_resume = False  # Don't resume from checkpoint

# Override train() to use our test config
original_train = dataset_trainer.train

def test_train():
    """Test train function with binary mode enabled."""
    # Monkey-patch the Config() call
    import dataset_trainer
    dataset_trainer.Config = lambda: cfg
    
    # Run training
    original_train()

if __name__ == "__main__":
    print("="*80)
    print("TESTING BINARY MODE TRAINING")
    print("="*80)
    print(f"binary_mode = {cfg.binary_mode}")
    print(f"epochs = {cfg.epochs}")
    print("="*80)
    
    test_train()
