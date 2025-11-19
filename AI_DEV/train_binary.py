#!/usr/bin/env python3
"""
Train Binary LEAK/NOLEAK Model

Simple wrapper to train binary classifier without code modification.
"""

from pathlib import Path
import dataset_trainer

# Override Config defaults for binary mode
original_config = dataset_trainer.Config

class BinaryConfig(original_config):
    def __post_init__(self):
        super().__post_init__()
        # Force binary mode
        self.binary_mode = True
        self.num_classes = 2
        # Separate model directory
        self.model_dir = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS_BINARY")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

# Replace Config class
dataset_trainer.Config = BinaryConfig

# Run training
if __name__ == "__main__":
    dataset_trainer.train()
