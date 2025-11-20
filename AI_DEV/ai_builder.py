#!/usr/bin/env python3
"""
AI Builder - Unified Pipeline Manager
======================================

Central command-line interface for managing the entire leak detection ML pipeline.
Orchestrates dataset building, model training, hyperparameter tuning, and classification.

Usage Examples:
    # Build HDF5 datasets from WAV files
    python ai_builder.py --build-dataset
    
    # Train both binary and multi-class models (default)
    python ai_builder.py --train-models
    
    # Train only binary model
    python ai_builder.py --train-binary-model
    
    # Train only multi-class model
    python ai_builder.py --train-multi-model
    
    # Tune both models (default 50 trials, 20 epochs each)
    python ai_builder.py --tune-models

    # Quick tuning test (10 trials, 10 epochs each)
    python ai_builder.py --tune-models --n_trials 10 --max_epochs 10

    # Thorough tuning (100 trials, 30 epochs each)
    python ai_builder.py --full-pipeline --n_trials 100 --max_epochs 30

    # Tune only binary model
    python ai_builder.py --tune-binary-model --n_trials 50
    
    # Tune only multi-class model
    python ai_builder.py --tune-multi-model --n_trials 50
    
    # Run classification with binary model
    python ai_builder.py --classify-binary /path/to/audio/files
    
    # Run classification with multi-class model
    python ai_builder.py --classify-multi /path/to/audio/files
    
    # Full pipeline: build → tune → train (recommended order)
    python ai_builder.py --full-pipeline --n_trials 50
    
    # Or manually: tune then train with best parameters
    python ai_builder.py --tune-and-train-binary --n_trials 50 --fresh

Author: AI Development Team
Version: 1.2
Last Updated: November 20, 2025

Directory Structure:
    PROC_MODELS/
    ├── binary/          - Binary LEAK/NOLEAK models
    │   ├── checkpoints/ - Training state & epoch saves
    │   └── tuning/      - Optuna optimization results
    └── multiclass/      - 5-class models (BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED)
        ├── checkpoints/ - Training state & epoch saves
        └── tuning/      - Optuna optimization results

Notes:
    - All model directories are automatically created on first run
    - Use --fresh flag to delete and recreate all model directories (clean slate)
    - Fresh mode removes all checkpoints and tuning results for both binary and multiclass
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

from global_config import PROC_MODELS

# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}{title.center(80)}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")


def load_best_params(model_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load best hyperparameters from tuning results.
    
    Args:
        model_dir: Model directory containing tuning/best_params.json
    
    Returns:
        Dictionary of best parameters, or None if not found
    """
    best_params_file = model_dir / "tuning" / "best_params.json"
    
    if not best_params_file.exists():
        print_warning(f"No tuned parameters found at {best_params_file}")
        return None
    
    try:
        with open(best_params_file, 'r') as f:
            data = json.load(f)
            return data.get('params', {})
    except Exception as e:
        print_error(f"Failed to load tuned parameters: {e}")
        return None


def apply_tuned_params(cfg, params: Dict[str, Any]):
    """
    Apply tuned hyperparameters to training config.
    
    Args:
        cfg: Config object to modify
        params: Dictionary of hyperparameters from tuning
    """
    if not params:
        return
    
    print(f"{CYAN}Applying tuned hyperparameters:{RESET}")
    
    # Apply each parameter
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            print(f"  {key}: {value}")
        else:
            print_warning(f"Unknown parameter: {key}")


def ensure_directories(fresh: bool = False):
    """
    Ensure all required model directories exist.
    If fresh=True, delete and recreate directories for clean slate.
    
    Args:
        fresh: Whether to delete existing directories first
    """
    from pathlib import Path
    import shutil
    from global_config import PROC_MODELS, TEMP_DIR
    
    base_dir = Path(PROC_MODELS)
    temp_dir = Path(TEMP_DIR)
    
    # Define required directory structure
    required_dirs = [
        base_dir / "binary",
        base_dir / "binary" / "checkpoints",
        base_dir / "binary" / "tuning",
        base_dir / "binary" / "tuning" / "plots",
        base_dir / "multiclass",
        base_dir / "multiclass" / "checkpoints",
        base_dir / "multiclass" / "tuning",
        base_dir / "multiclass" / "tuning" / "plots",
    ]
    
    if fresh:
        print(f"{YELLOW}Fresh start: Cleaning directories...{RESET}")
        
        # Delete binary and multiclass subdirectories (preserve root and any other files)
        for subdir in ["binary", "multiclass"]:
            dir_path = base_dir / subdir
            if dir_path.exists():
                print(f"  Deleting: {dir_path}")
                shutil.rmtree(dir_path)
        
        # Clean PROC_TEMP directory (delete contents but preserve directory)
        if temp_dir.exists():
            print(f"  Cleaning: {temp_dir}")
            for item in temp_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        print_success("Model and temp directories cleaned")
    
    # Create all required directories
    print(f"{CYAN}Setting up model directories...{RESET}")
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print_success(f"Model directories ready: {base_dir}")


def run_dataset_builder() -> bool:
    """
    Build HDF5 datasets from WAV files.
    
    Returns:
        True if successful, False otherwise
    """
    print_header("BUILDING HDF5 DATASETS")
    
    try:
        import dataset_builder
        
        start_time = time.time()
        dataset_builder.main()
        elapsed = time.time() - start_time
        
        print_success(f"Dataset building completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print_error(f"Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_train_binary(use_tuned: bool = False, fresh: bool = False, continue_training: bool = False, checkpoint_path: Optional[str] = None, reset_optimizer: bool = False, reset_scheduler: bool = False) -> bool:
    """
    Train binary LEAK/NOLEAK model.
    
    Args:
        use_tuned: Whether to use tuned hyperparameters
        fresh: Whether to delete old checkpoint before starting
        continue_training: Enable continual learning mode (load existing checkpoint)
        checkpoint_path: Path to checkpoint for continual learning (None=auto-detect)
        reset_optimizer: Reset optimizer state when continuing (fresh optimization)
        reset_scheduler: Reset learning rate scheduler when continuing
    
    Returns:
        True if successful, False otherwise
    """
    print_header("TRAINING BINARY MODEL (LEAK/NOLEAK)")
    
    try:
        import dataset_trainer
        
        start_time = time.time()
        
        # Ensure directories exist (will be cleaned if fresh=True was set globally)
        # Note: fresh cleanup happens once in main() before any operations
        
        # Create binary config
        cfg = dataset_trainer.Config()
        cfg.binary_mode = True
        cfg.num_classes = 2
        cfg.model_dir = Path(PROC_MODELS) / "binary"
        
        # Configure continual learning if requested
        cfg.continual_learning = continue_training
        if continue_training:
            cfg.continual_checkpoint = checkpoint_path
            cfg.continual_reset_optimizer = reset_optimizer
            cfg.continual_reset_scheduler = reset_scheduler
            print(f"{CYAN}Continual Learning Mode: Loading from existing checkpoint{RESET}")
            if checkpoint_path:
                print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Reset optimizer: {reset_optimizer}")
            print(f"  Reset scheduler: {reset_scheduler}")
        
        # Apply tuned parameters if requested
        if use_tuned:
            params = load_best_params(cfg.model_dir)
            if params:
                apply_tuned_params(cfg, params)
            else:
                print_warning("Tuned parameters not found, using default config")
        
        dataset_trainer.train(cfg)
        elapsed = time.time() - start_time
        
        print_success(f"Binary model training completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print_error(f"Binary model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_train_multi(use_tuned: bool = False, fresh: bool = False, continue_training: bool = False, checkpoint_path: Optional[str] = None, reset_optimizer: bool = False, reset_scheduler: bool = False) -> bool:
    """
    Train multi-class (5-class) model.
    
    Args:
        use_tuned: Whether to use tuned hyperparameters
        fresh: Whether to delete old checkpoint before starting
        continue_training: Enable continual learning mode (load existing checkpoint)
        checkpoint_path: Path to checkpoint for continual learning (None=auto-detect)
        reset_optimizer: Reset optimizer state when continuing (fresh optimization)
        reset_scheduler: Reset learning rate scheduler when continuing
    
    Returns:
        True if successful, False otherwise
    """
    print_header("TRAINING MULTI-CLASS MODEL (5 CLASSES)")
    
    try:
        import dataset_trainer
        
        start_time = time.time()
        cfg = dataset_trainer.Config()
        cfg.binary_mode = False
        cfg.num_classes = 5
        cfg.model_dir = Path(PROC_MODELS) / "multiclass"
        
        # Configure continual learning if requested
        cfg.continual_learning = continue_training
        if continue_training:
            cfg.continual_checkpoint = checkpoint_path
            cfg.continual_reset_optimizer = reset_optimizer
            cfg.continual_reset_scheduler = reset_scheduler
            print(f"{CYAN}Continual Learning Mode: Loading from existing checkpoint{RESET}")
            if checkpoint_path:
                print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Reset optimizer: {reset_optimizer}")
            print(f"  Reset scheduler: {reset_scheduler}")
        
        # Apply tuned parameters if requested
        if use_tuned:
            params = load_best_params(cfg.model_dir)
            if params:
                apply_tuned_params(cfg, params)
            else:
                print_warning("Tuned parameters not found, using default config")
        
        dataset_trainer.train(cfg)
        elapsed = time.time() - start_time
        
        print_success(f"Multi-class model training completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print_error(f"Multi-class model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tune_binary(n_trials: int = 50, max_epochs: int = 20, restart: bool = False, fresh: bool = False) -> bool:
    """
    Tune binary model hyperparameters.
    
    Args:
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs per trial
        restart: Whether to restart optimization from scratch
        fresh: Whether to delete old study database before starting
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"TUNING BINARY MODEL ({n_trials} trials, max {max_epochs} epochs/trial)")
    
    try:
        import dataset_tuner
        
        start_time = time.time()
        
        # Configure for binary mode
        tuning_cfg = dataset_tuner.TuningConfig()
        tuning_cfg.n_trials = n_trials
        tuning_cfg.max_epochs_per_trial = max_epochs
        tuning_cfg.restart = restart
        tuning_cfg.binary_mode = True
        tuning_cfg.model_dir = Path(PROC_MODELS) / "binary"
        
        # Delete old study database if fresh start requested
        if fresh:
            study_db = tuning_cfg.model_dir / "tuning" / "study.db"
            if study_db.exists():
                print(f"{YELLOW}Deleting old study database: {study_db}{RESET}")
                study_db.unlink()
                print_success("Study database deleted")
        
        dataset_tuner.run_optimization(tuning_cfg)
        elapsed = time.time() - start_time
        
        print_success(f"Binary model tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print_error(f"Binary model tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tune_multi(n_trials: int = 50, max_epochs: int = 20, restart: bool = False, fresh: bool = False) -> bool:
    """
    Tune multi-class model hyperparameters.
    
    Args:
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs per trial
        restart: Whether to restart optimization from scratch
        fresh: Whether to delete old study database before starting
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"TUNING MULTI-CLASS MODEL ({n_trials} trials, max {max_epochs} epochs/trial)")
    
    try:
        import dataset_tuner
        
        start_time = time.time()
        
        # Configure for multi-class mode
        tuning_cfg = dataset_tuner.TuningConfig()
        tuning_cfg.n_trials = n_trials
        tuning_cfg.max_epochs_per_trial = max_epochs
        tuning_cfg.restart = restart
        tuning_cfg.binary_mode = False
        tuning_cfg.model_dir = Path(PROC_MODELS) / "multiclass"
        
        # Delete old study database if fresh start requested
        if fresh:
            study_db = tuning_cfg.model_dir / "tuning" / "study.db"
            if study_db.exists():
                print(f"{YELLOW}Deleting old study database: {study_db}{RESET}")
                study_db.unlink()
                print_success("Study database deleted")
        
        dataset_tuner.run_optimization(tuning_cfg)
        elapsed = time.time() - start_time
        
        print_success(f"Multi-class model tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return True
        
    except Exception as e:
        print_error(f"Multi-class model tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_classify_binary(input_path: str) -> bool:
    """
    Run classification with binary model.
    
    Args:
        input_path: Path to audio file or directory
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"CLASSIFYING WITH BINARY MODEL: {input_path}")
    
    try:
        import dataset_classifier
        
        # Configure classifier for binary mode
        # This will need to be implemented in dataset_classifier.py
        print_warning("Binary classification not yet implemented in dataset_classifier.py")
        return False
        
    except Exception as e:
        print_error(f"Binary classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tune_and_train_binary(n_trials: int = 50, max_epochs: int = 20, restart: bool = False, fresh: bool = False) -> bool:
    """
    Tune binary model hyperparameters then train with best parameters.
    
    Args:
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs per trial
        restart: Whether to restart optimization from scratch
        fresh: Whether to delete old study database and checkpoint before starting
    
    Returns:
        True if successful, False otherwise
    """
    print_header("TUNE AND TRAIN BINARY MODEL")
    
    # Step 1: Tune
    if not run_tune_binary(n_trials, max_epochs, restart, fresh):
        print_error("Tuning failed, skipping training")
        return False
    
    # Step 2: Train with best params
    return run_train_binary(use_tuned=True, fresh=fresh)


def run_tune_and_train_multi(n_trials: int = 50, max_epochs: int = 20, restart: bool = False, fresh: bool = False) -> bool:
    """
    Tune multi-class model hyperparameters then train with best parameters.
    
    Args:
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs per trial
        restart: Whether to restart optimization from scratch
        fresh: Whether to delete old study database and checkpoint before starting
    
    Returns:
        True if successful, False otherwise
    """
    print_header("TUNE AND TRAIN MULTI-CLASS MODEL")
    
    # Step 1: Tune
    if not run_tune_multi(n_trials, max_epochs, restart, fresh):
        print_error("Tuning failed, skipping training")
        return False
    
    # Step 2: Train with best params
    return run_train_multi(use_tuned=True, fresh=fresh)


def run_classify_multi(input_path: str) -> bool:
    """
    Run classification with multi-class model.
    
    Args:
        input_path: Path to audio file or directory
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"CLASSIFYING WITH MULTI-CLASS MODEL: {input_path}")
    
    try:
        import dataset_classifier
        
        # Run classification
        print_warning("Multi-class classification not yet fully integrated")
        return False
        
    except Exception as e:
        print_error(f"Multi-class classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AI Builder - Unified Pipeline Manager for Leak Detection ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full-pipeline
  %(prog)s --build-dataset
  %(prog)s --train-models
  %(prog)s --train-binary-model
  %(prog)s --train-multi-model
  %(prog)s --tune-models --n_trials 50
  %(prog)s --tune-binary-model --n_trials 30 --restart
  %(prog)s --classify-binary /path/to/audio
  %(prog)s --classify-multi /path/to/audio
  %(prog)s --build-dataset --train-models --tune-models

For detailed help on individual components:
  - Dataset Building: See dataset_builder.py
  - Model Training: See dataset_trainer.py or train_binary.py
  - Hyperparameter Tuning: See dataset_tuner.py
  - Classification: See dataset_classifier.py
        """
    )
    
    # Full pipeline
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete pipeline: build dataset → tune models → train with best params'
    )
    
    # Dataset building
    parser.add_argument(
        '--build-dataset',
        action='store_true',
        help='Build HDF5 datasets from WAV files (runs dataset_builder.py)'
    )
    
    # Model training
    train_group = parser.add_argument_group('Model Training')
    train_group.add_argument(
        '--train-models',
        action='store_true',
        help='Train both binary and multi-class models (default training mode)'
    )
    train_group.add_argument(
        '--train-binary-model',
        action='store_true',
        help='Train only binary LEAK/NOLEAK model'
    )
    train_group.add_argument(
        '--train-multi-model',
        action='store_true',
        help='Train only multi-class (5-class) model'
    )
    train_group.add_argument(
        '--continue-training',
        action='store_true',
        help='Continue training from existing checkpoint (continual learning mode)'
    )
    train_group.add_argument(
        '--checkpoint',
        type=str,
        metavar='PATH',
        help='Path to checkpoint for continual learning (default: auto-detect last.pth or best_model.pth)'
    )
    train_group.add_argument(
        '--reset-optimizer',
        action='store_true',
        help='Reset optimizer state when continuing training (fresh optimization on new data)'
    )
    train_group.add_argument(
        '--reset-scheduler',
        action='store_true',
        help='Reset learning rate scheduler when continuing training'
    )
    
    # Hyperparameter tuning
    tune_group = parser.add_argument_group('Hyperparameter Tuning')
    tune_group.add_argument(
        '--tune-models',
        action='store_true',
        help='Tune both binary and multi-class models (default tuning mode)'
    )
    tune_group.add_argument(
        '--tune-binary-model',
        action='store_true',
        help='Tune only binary model hyperparameters'
    )
    tune_group.add_argument(
        '--tune-multi-model',
        action='store_true',
        help='Tune only multi-class model hyperparameters'
    )
    tune_group.add_argument(
        '--tune-and-train-binary',
        action='store_true',
        help='Tune binary model then train with best parameters'
    )
    tune_group.add_argument(
        '--tune-and-train-multi',
        action='store_true',
        help='Tune multi-class model then train with best parameters'
    )
    tune_group.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials for tuning (default: 50)'
    )
    tune_group.add_argument(
        '--max_epochs',
        type=int,
        default=20,
        help='Maximum epochs per trial for tuning (default: 20)'
    )
    tune_group.add_argument(
        '--restart',
        action='store_true',
        help='Restart optimization study from scratch (delete existing results)'
    )
    tune_group.add_argument(
        '--fresh',
        action='store_true',
        help='Delete old study database and checkpoint files before starting (clean slate)'
    )
    
    # Classification
    classify_group = parser.add_argument_group('Classification')
    classify_group.add_argument(
        '--classify-binary',
        type=str,
        metavar='PATH',
        help='Run classification with binary model on audio file or directory'
    )
    classify_group.add_argument(
        '--classify-multi',
        type=str,
        metavar='PATH',
        help='Run classification with multi-class model on audio file or directory'
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Handle full pipeline flag
    if args.full_pipeline:
        print_header("FULL PIPELINE EXECUTION")
        print(f"{CYAN}Running: Build Dataset → Tune → Train with Best Params{RESET}\n")
        args.build_dataset = True
        args.tune_and_train_binary = True
        args.tune_and_train_multi = True
    
    # Track overall success
    all_success = True
    overall_start = time.time()
    
    # Setup directories first (clean if --fresh flag set)
    if args.fresh:
        ensure_directories(fresh=True)
    else:
        ensure_directories(fresh=False)
    
    # Execute requested operations in logical order
    
    # 1. Dataset building
    if args.build_dataset:
        if not run_dataset_builder():
            all_success = False
            print_error("Stopping pipeline due to dataset building failure")
            sys.exit(1)
    
    # 2. Tune-and-train (combined operations)
    if args.tune_and_train_binary:
        if not run_tune_and_train_binary(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
    
    if args.tune_and_train_multi:
        if not run_tune_and_train_multi(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
    
    # 3. Model training (standalone)
    if args.train_models or (args.train_binary_model and args.train_multi_model):
        # Train both models
        print_header("TRAINING BOTH MODELS")
        
        if not run_train_binary(fresh=args.fresh, continue_training=args.continue_training, 
                               checkpoint_path=args.checkpoint, reset_optimizer=args.reset_optimizer,
                               reset_scheduler=args.reset_scheduler):
            all_success = False
            print_warning("Binary training failed, continuing with multi-class...")
        
        if not run_train_multi(fresh=args.fresh, continue_training=args.continue_training,
                              checkpoint_path=args.checkpoint, reset_optimizer=args.reset_optimizer,
                              reset_scheduler=args.reset_scheduler):
            all_success = False
            print_warning("Multi-class training failed")
    
    elif args.train_binary_model:
        if not run_train_binary(fresh=args.fresh, continue_training=args.continue_training,
                               checkpoint_path=args.checkpoint, reset_optimizer=args.reset_optimizer,
                               reset_scheduler=args.reset_scheduler):
            all_success = False
    
    elif args.train_multi_model:
        if not run_train_multi(fresh=args.fresh, continue_training=args.continue_training,
                              checkpoint_path=args.checkpoint, reset_optimizer=args.reset_optimizer,
                              reset_scheduler=args.reset_scheduler):
            all_success = False
    
    # 4. Hyperparameter tuning (standalone)
    if args.tune_models or (args.tune_binary_model and args.tune_multi_model):
        # Tune both models
        print_header("TUNING BOTH MODELS")
        
        if not run_tune_binary(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
            print_warning("Binary tuning failed, continuing with multi-class...")
        
        if not run_tune_multi(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
            print_warning("Multi-class tuning failed")
    
    elif args.tune_binary_model:
        if not run_tune_binary(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
    
    elif args.tune_multi_model:
        if not run_tune_multi(args.n_trials, args.max_epochs, args.restart, args.fresh):
            all_success = False
    
    # 5. Classification
    if args.classify_binary:
        if not run_classify_binary(args.classify_binary):
            all_success = False
    
    if args.classify_multi:
        if not run_classify_multi(args.classify_multi):
            all_success = False
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    print_header("PIPELINE SUMMARY")
    
    if all_success:
        print_success(f"All operations completed successfully!")
        print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    else:
        print_warning(f"Some operations failed or were skipped")
        print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
        sys.exit(1)


if __name__ == "__main__":
    main()
