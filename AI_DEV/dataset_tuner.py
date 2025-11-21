#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Hyperparameter Tuner - Optuna-based optimization for LeakCNNMulti

Automated hyperparameter optimization using Optuna to find optimal training
configurations for leak detection models. Integrates with dataset_trainer.py
for consistent training pipeline.

Hyperparameters Optimized:
    - Training: learning_rate, batch_size, dropout, grad_clip_norm
    - Loss: loss_type, focal_gamma, focal_alpha_leak, leak_aux_weight
    - Architecture: n_filters (model width scaling)
    - Regularization: weight_decay, early_stop_patience

Search Strategy:
    - Bayesian optimization (TPE sampler) for efficient exploration
    - MedianPruner for early termination of poor trials
    - Persistent SQLite storage for resumable studies
    - GPU memory-aware batch size selection

Features:
    - RAM preloading for fast iteration (requires 8GB+ RAM)
    - Automatic checkpoint cleanup between trials
    - Real-time progress tracking with GPU utilization
    - Best parameters saved with validation metrics
    - Visualization plots (history, importances, parallel coordinates)

Output:
    - Best parameters: MODEL_DIR/tuning/best_params.json
    - Optuna database: MODEL_DIR/tuning/study.db
    - Plots: MODEL_DIR/tuning/plots/

Usage:
    python dataset_tuner.py --n_trials 50 --timeout 3600
    python dataset_tuner.py --restart  # Delete study and start fresh
    
    # View dashboard:
    optuna-dashboard sqlite:///MODEL_DIR/tuning/study.db

Example Integration:
    # After tuning, apply best parameters:
    best = json.load(open('MODEL_DIR/tuning/best_params.json'))
    cfg = Config()
    cfg.learning_rate = best['learning_rate']
    cfg.batch_size = best['batch_size']
    # ... etc
    
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import optuna
    from optuna.trial import Trial, TrialState
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    print("❌ Optuna not installed. Install with: pip install optuna optuna-dashboard")
    sys.exit(1)

# Import from dataset_trainer
from dataset_trainer import (
    Config, LeakMelDataset, BinaryLabelDataset, StatefulSampler,
    train_one_epoch, evaluate_file_level, eval_split, setup_datasets,
    device_setup, set_seed, logger, CYAN, GREEN, YELLOW, RED, RESET,
    LeakCNNMulti, setup_dataloaders, setup_loss_functions,
    SystemMonitor, GPUProfiler
)

# Import from global_config
from global_config import PROC_MODELS

# Constants
TUNING_SUBDIR = "tuning"
PLOTS_SUBDIR = "plots"
STUDY_NAME = "leak_detection_hpo"
STORAGE_FILE = "study.db"
BEST_PARAMS_FILE = "best_params.json"


def cleanup_after_trial():
    """Clean up GPU memory and caches after trial."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()


class TuningConfig:
    """Configuration for hyperparameter optimization."""
    def __init__(self):
        # Base paths - will be set based on binary_mode, but can be overridden
        self.model_dir = Path(PROC_MODELS) / "multiclass"  # Default to multi-class
        
        # Model mode
        self.binary_mode = False  # False: 5-class (default), True: LEAK/NOLEAK
        
        # Optuna settings
        self.n_trials = 50
        self.timeout = None  # seconds
        self.n_jobs = 1  # Parallel trials (1 recommended for GPU)
        self.restart = False
        
        # Trial settings
        self.max_epochs_per_trial = 20  # Early stopping for bad trials
        self.min_epochs_per_trial = 5   # Minimum before pruning
        self.pruning_warmup = 3         # Epochs before pruning starts
        
        # Fixed dataset params (use optimized settings from training)
        self.preload_to_ram = True
        self.num_workers = 12   # Increased for better GPU utilization
        self.prefetch_factor = 4  # Reduced from 12 to avoid RAM bandwidth saturation
        self.persistent_workers = True  # Enable for multiple epochs per trial
    
    @property
    def tuning_dir(self) -> Path:
        """Dynamically compute tuning directory based on current model_dir."""
        return self.model_dir / TUNING_SUBDIR
    
    @property
    def plots_dir(self) -> Path:
        """Dynamically compute plots directory based on current model_dir."""
        return self.tuning_dir / PLOTS_SUBDIR
        
    def setup_dirs(self):
        """Create tuning directories."""
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


def suggest_hyperparameters(trial: Trial, base_cfg: Config, tuning_cfg: TuningConfig) -> Config:
    """
    Suggest hyperparameters for trial.
    
    Returns modified Config with trial suggestions.
    """
    cfg = Config()
    
    # Copy fixed settings from base
    cfg.stage_dir = base_cfg.stage_dir
    cfg.model_dir = base_cfg.model_dir
    cfg.binary_mode = base_cfg.binary_mode
    cfg.num_classes = base_cfg.num_classes
    cfg.preload_to_ram = base_cfg.preload_to_ram
    cfg.num_workers = base_cfg.num_workers
    cfg.prefetch_factor = base_cfg.prefetch_factor
    cfg.persistent_workers = base_cfg.persistent_workers
    cfg.seed = base_cfg.seed
    
    # === Hyperparameters to optimize ===
    
    # Batch size first (needed for LR scaling) - larger for better GPU utilization
    cfg.batch_size = trial.suggest_categorical("batch_size", [4096, 6144, 8192, 10240])
    cfg.val_batch_size = min(cfg.batch_size * 2, 16384)  # 2x training (validation doesn't need gradients)
    
    # Learning rate (adjusted for batch size - larger batches need higher LR)
    # Raised minimum from 1e-4 to 3e-4 to avoid getting stuck with weak gradients
    batch_size_factor = cfg.batch_size / 6144  # Normalize to middle batch size (6144)
    lr_min = 3e-4 * max(1.0, batch_size_factor ** 0.5)  # Scale up min LR with batch size
    lr_max = 1e-2 * max(1.0, batch_size_factor ** 0.5)  # Scale up max LR with batch size
    cfg.learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
    
    # Dropout regularization
    cfg.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    
    # Gradient clipping
    grad_clip_enabled = trial.suggest_categorical("grad_clip_enabled", [True, False])
    if grad_clip_enabled:
        cfg.grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.5, 2.0, step=0.25)
    else:
        cfg.grad_clip_norm = None
    
    # Loss function
    cfg.loss_type = trial.suggest_categorical("loss_type", ["weighted_ce", "focal"])
    
    if cfg.loss_type == "focal":
        cfg.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)
        cfg.focal_alpha_leak = trial.suggest_float("focal_alpha_leak", 0.5, 0.9, step=0.1)
    
    # Auxiliary leak head weight (very conservative to prevent oscillation)
    # Diagnostic evidence shows even weight=0.25 causes training/validation oscillation:
    #   - Training batch 0: 98.8% LEAK predictions (chasing aux head gradient)
    #   - Validation: 0% LEAK predictions (collapsed to class prior)
    # Aux loss magnitude (~0.72) >> cls loss (~0.12), so even low weights dominate
    # Reducing to [0.05, 0.15] to let classification head lead training
    cfg.leak_aux_weight = trial.suggest_float("leak_aux_weight", 0.05, 0.15, step=0.05)

    # Class balancing (critical for preventing collapse)
    # REMOVED oversample_factor from search - causes LEAK collapse when >1
    # Use class weights only (more stable than oversampling)
    cfg.leak_oversample_factor = 1  # Disable oversampling
    # Narrowed range to middle ground between collapse modes:
    #   - boost ≥2.0: 96.5% LEAK predictions (too strong)
    #   - boost ≤1.25: constant 0.5 outputs (too weak)
    cfg.leak_weight_boost = trial.suggest_float("leak_weight_boost", 1.3, 1.8, step=0.1)

    # Small warmup to stabilize early training (helps prevent early collapse)
    cfg.warmup_epochs = trial.suggest_int("warmup_epochs", 2, 5)
    
    # Set epochs for tuning (override default 200)
    cfg.epochs = tuning_cfg.max_epochs_per_trial
    
    return cfg


def objective(trial: Trial, base_cfg: Config, tuning_cfg: TuningConfig) -> float:
    """
    Objective function for Optuna optimization.
    
    Returns:
        Best validation file-level leak F1 score (to maximize)
    """
    trial_start = time.time()
    
    # Generate trial configuration
    cfg = suggest_hyperparameters(trial, base_cfg, tuning_cfg)
    
    logger.info(f"{CYAN}{'='*80}{RESET}")
    logger.info(f"{CYAN}Optuna Trial #{trial.number}{RESET}")
    logger.info(f"{CYAN}{'='*80}{RESET}")
    logger.info(f"Parameters:")
    for key, value in trial.params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  actual_learning_rate: {cfg.learning_rate:.2e}")
    logger.info(f"  actual_batch_size: {cfg.batch_size}")
    logger.info(f"  num_classes: {cfg.num_classes}")
    logger.info(f"  binary_mode: {cfg.binary_mode}")
    logger.info(f"  leak_oversample_factor: {cfg.leak_oversample_factor}")
    logger.info(f"  leak_weight_boost: {cfg.leak_weight_boost}")
    
    try:
        # Setup
        set_seed(cfg.seed)
        device = device_setup()
        
        # Load datasets (with RAM preloading for speed)
        train_ds, val_ds, train_file_ids, val_file_ids, leak_idx, train_indices, val_subset = setup_datasets(cfg)
        
        # Validate dataset sizes
        if len(train_indices) == 0:
            logger.error(f"{RED}Trial {trial.number} failed: No training data{RESET}")
            return 0.0
        
        logger.info(f"Trial {trial.number}: {len(train_indices)} train samples, {len(val_subset)} val samples")
        
        # Create model
        model = LeakCNNMulti(
            n_classes=cfg.num_classes,
            dropout=cfg.dropout
        ).to(device)
        
        # Enable optimizations (use compile for performance)
        if cfg.use_channels_last:
            model = model.to(memory_format=torch.channels_last)  # type: ignore
        
        model_compiled = model
        if cfg.use_compile:
            try:
                model_compiled = torch.compile(model, mode="reduce-overhead")  # More stable mode
                model_compiled = cast(nn.Module, model_compiled)  # type: ignore
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, using uncompiled model")
                model_compiled = model
        
        # Setup training components
        train_loader, val_loader, train_sampler = setup_dataloaders(
            cfg, train_ds, train_indices, val_subset, leak_idx, cfg.binary_mode
        )
        
        # Debug: Check DataLoader sizes
        logger.info(f"Trial {trial.number}: train_loader has {len(train_loader)} batches (batch_size={cfg.batch_size})")
        logger.info(f"Trial {trial.number}: val_loader has {len(val_loader)} batches")
        
        # Debug: Check model is trainable
        initial_params = torch.cat([p.flatten() for p in model.parameters()]).detach().cpu()
        
        if len(train_loader) == 0:
            logger.error(f"{RED}Trial {trial.number} failed: train_loader is empty!{RESET}")
            logger.error(f"  train_indices: {len(train_indices)}, batch_size: {cfg.batch_size}")
            logger.error(f"  Expected batches: {len(train_indices) // cfg.batch_size}")
            return 0.0
        
        cls_loss_fn, leak_bce = setup_loss_functions(cfg, train_ds, leak_idx, device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=1e-4
        )
        
        # Use compiled model for training/inference
        train_model = model_compiled
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=1e-6
        )
        
        scaler = torch.amp.GradScaler('cuda')
        
        # Create monitors for training (reuse across epochs for tuning)
        sys_monitor = SystemMonitor(device=device, enabled=True)  # Enable for monitoring GPU sputtering
        profiler = GPUProfiler(device=device, enabled=False) if torch.cuda.is_available() else None
        
        # Training loop
        best_val_f1 = 0.0
        no_improve_count = 0
        model_leak_idx = 1 if cfg.binary_mode else leak_idx

        for epoch in range(1, cfg.epochs + 1):
            # Reset sampler for new epoch
            train_sampler.on_epoch_start(epoch)

            # Train
            train_loss, train_acc = train_one_epoch(
                epoch=epoch,
                model=train_model,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                cls_loss_fn=cls_loss_fn,
                leak_bce=leak_bce,
                cfg=cfg,
                leak_idx=leak_idx,
                model_leak_idx=model_leak_idx,
                device=device,
                use_ta=False,  # Disable augmentation for tuning speed
                time_mask=None,
                freq_mask=None,
                interrupted={"flag": False},
                sys_monitor=sys_monitor,
                profiler=profiler
            )

            # Debug: Check if model parameters are changing
            if epoch == 1:
                final_params = torch.cat([p.flatten() for p in model.parameters()]).detach().cpu()
                param_diff = torch.abs(final_params - initial_params).mean().item()
                logger.info(f"Trial {trial.number}: Parameter change after epoch 1: {param_diff:.2e}")
                if param_diff < 1e-6:
                    logger.warning(f"{YELLOW}Trial {trial.number}: Model parameters barely changed! Learning may have failed.{RESET}")
                    # Don't return 0.0 immediately, let it continue to see if it improves

            # Validate (use full validation set for accurate F1)
            # CRITICAL: Use model_leak_idx for binary mode (1) instead of leak_idx (2)
            # After BinaryLabelDataset wrapping, labels are 0/1, not 0-4
            # Use full validation to avoid missing LEAK samples (validation is NOT shuffled!)
            val_metrics = eval_split(
                model=train_model,
                loader=val_loader,
                device=device,
                leak_idx=model_leak_idx,  # FIX: Use model_leak_idx (1 for binary, leak_idx for multi-class)
                use_channels_last=cfg.use_channels_last,
                max_batches=None,  # Use full validation set (validation is not shuffled, sampling can miss LEAK!)
                leak_threshold=0.5  # Standard threshold (0.3 was too low for uncertain model outputs)
            )
            
            val_f1 = val_metrics["leak_f1"]
            val_acc = val_metrics["acc"]
            
            logger.info(
                f"Trial {trial.number} Epoch {epoch}/{cfg.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_f1={val_f1:.4f}, val_acc={val_acc:.4f}"
            )
            
            # Report to Optuna for pruning
            trial.report(val_f1, epoch)
            
            # Prune if trial shows no promise
            if epoch >= tuning_cfg.pruning_warmup and trial.should_prune():
                logger.warning(f"{YELLOW}Trial {trial.number} pruned at epoch {epoch}{RESET}")
                raise optuna.TrialPruned()
            
            # Track best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= cfg.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # LR scheduling with optional warmup
            # During warmup: linearly increase LR from 0 to base_lr
            # After warmup: apply cosine annealing decay
            if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
                # Linear warmup: LR = base_lr * (epoch / warmup_epochs)
                warmup_factor = epoch / float(max(1, cfg.warmup_epochs))
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg.learning_rate * warmup_factor
                logger.debug("Warmup LR epoch %d/%d: factor=%.3f, lr=%.6e",
                            epoch, cfg.warmup_epochs, warmup_factor, optimizer.param_groups[0]['lr'])
            else:
                # After warmup: apply cosine annealing
                scheduler.step()
                logger.debug("Cosine LR epoch %d: lr=%.6e", epoch, optimizer.param_groups[0]['lr'])
        
        trial_elapsed = time.time() - trial_start
        logger.info(
            f"{GREEN}Trial {trial.number} completed: best_val_f1={best_val_f1:.4f} "
            f"({trial_elapsed:.1f}s){RESET}"
        )
        
        return best_val_f1
        
    except optuna.TrialPruned:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error(f"{RED}Trial {trial.number} failed: GPU OOM - try smaller batch_size{RESET}")
        cleanup_after_trial()
        return 0.0  # Return poor score instead of failing
    except Exception as e:
        logger.error(f"{RED}Trial {trial.number} failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        cleanup_after_trial()
        return 0.0  # Return poor score instead of failing


def run_optimization(tuning_cfg: TuningConfig):
    """Run Optuna optimization study."""
    
    # Setup
    tuning_cfg.setup_dirs()
    storage_path = tuning_cfg.tuning_dir / STORAGE_FILE
    storage_url = f"sqlite:///{storage_path}"
    
    # Delete existing study if restart requested
    if tuning_cfg.restart and storage_path.exists():
        logger.info(f"{YELLOW}Restarting: deleting {storage_path}{RESET}")
        storage_path.unlink()
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage_url)
        except Exception:
            pass
    
    # Create base configuration
    base_cfg = Config()
    base_cfg.preload_to_ram = tuning_cfg.preload_to_ram
    base_cfg.num_workers = tuning_cfg.num_workers
    base_cfg.prefetch_factor = tuning_cfg.prefetch_factor
    base_cfg.persistent_workers = tuning_cfg.persistent_workers
    base_cfg.binary_mode = tuning_cfg.binary_mode  # Set binary mode from tuning config
    base_cfg.num_classes = 2 if tuning_cfg.binary_mode else 5
    base_cfg.model_dir = tuning_cfg.model_dir
    
    # Create or load study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        load_if_exists=not tuning_cfg.restart,
        direction="maximize",  # Maximize leak F1
        sampler=TPESampler(seed=base_cfg.seed),
        pruner=MedianPruner(
            n_startup_trials=tuning_cfg.min_epochs_per_trial,
            n_warmup_steps=tuning_cfg.pruning_warmup
        )
    )
    
    # Check if study has incompatible trials
    if study.trials:
        logger.warning("⚠️  Existing study found with previous trials.")
        logger.warning("   If you changed hyperparameter ranges, you may need to restart:")
        logger.warning("   python dataset_tuner.py --restart")
        logger.warning("   Or use: python ai_builder.py --tune-binary-model --restart")
    
    logger.info(f"{CYAN}{'='*80}{RESET}")
    logger.info(f"{CYAN}Starting Optuna Hyperparameter Optimization{RESET}")
    logger.info(f"{CYAN}{'='*80}{RESET}")
    logger.info(f"Study: {STUDY_NAME}")
    logger.info(f"Storage: {storage_path}")
    logger.info(f"Trials: {tuning_cfg.n_trials}")
    logger.info(f"Timeout: {tuning_cfg.timeout}s" if tuning_cfg.timeout else "Timeout: None")
    logger.info(f"Max epochs per trial: {tuning_cfg.max_epochs_per_trial}")
    logger.info(f"{CYAN}{'='*80}{RESET}")
    
    # Wrapper for cleanup
    def wrapped_objective(trial: Trial) -> float:
        cleanup_after_trial()  # Cleanup before trial
        try:
            return objective(trial, base_cfg, tuning_cfg)
        finally:
            cleanup_after_trial()  # Cleanup after trial
    
    # Run optimization
    try:
        study.optimize(
            wrapped_objective,
            n_trials=tuning_cfg.n_trials,
            timeout=tuning_cfg.timeout,
            n_jobs=tuning_cfg.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True  # Enable garbage collection
        )
    except KeyboardInterrupt:
        logger.info(f"{YELLOW}Optimization interrupted by user{RESET}")
    
    # Results
    logger.info(f"{GREEN}{'='*80}{RESET}")
    logger.info(f"{GREEN}Optimization Complete!{RESET}")
    logger.info(f"{GREEN}{'='*80}{RESET}")
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    
    logger.info(f"Total trials: {len(study.trials)}")
    logger.info(f"  Completed: {len(completed_trials)}")
    logger.info(f"  Pruned: {len(pruned_trials)}")
    logger.info(f"  Failed: {len(failed_trials)}")
    
    if completed_trials:
        best_trial = study.best_trial
        logger.info(f"\n{GREEN}Best Trial #{best_trial.number}:{RESET}")
        logger.info(f"  Value (leak F1): {best_trial.value:.4f}")
        logger.info(f"  Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
        
        # Save best parameters
        best_params_path = tuning_cfg.tuning_dir / BEST_PARAMS_FILE
        with open(best_params_path, 'w') as f:
            json.dump({
                "trial_number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "datetime": best_trial.datetime_complete.isoformat() if best_trial.datetime_complete else None
            }, f, indent=2)
        logger.info(f"\n{GREEN}✓ Best parameters saved to: {best_params_path}{RESET}")
        
        # Generate plots
        try:
            import plotly.graph_objects as go
            
            logger.info(f"\n{CYAN}Generating visualization plots...{RESET}")
            
            # Optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(str(tuning_cfg.plots_dir / "optimization_history.html"))
            
            # Parameter importances
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(tuning_cfg.plots_dir / "param_importances.html"))
            
            # Parallel coordinates
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(str(tuning_cfg.plots_dir / "parallel_coordinate.html"))
            
            # Slice plot
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(str(tuning_cfg.plots_dir / "slice.html"))
            
            logger.info(f"{GREEN}✓ Plots saved to: {tuning_cfg.plots_dir}{RESET}")
        except Exception as e:
            logger.warning(f"{YELLOW}Failed to generate plots: {e}{RESET}")
    
    logger.info(f"\n{CYAN}View dashboard with:{RESET}")
    logger.info(f"  optuna-dashboard {storage_url}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for leak detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (None for no limit)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel trials (1 recommended for GPU)")
    parser.add_argument("--restart", action="store_true", help="Delete existing study and start fresh")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum epochs per trial")
    parser.add_argument("--binary_mode", action="store_true", default=True, 
                       help="Train binary LEAK/NOLEAK model (default: True)")
    parser.add_argument("--multiclass", action="store_true", 
                       help="Train 5-class model instead of binary")
    
    args = parser.parse_args()
    
    tuning_cfg = TuningConfig()
    tuning_cfg.n_trials = args.n_trials
    tuning_cfg.timeout = args.timeout
    tuning_cfg.n_jobs = args.n_jobs
    tuning_cfg.restart = args.restart
    tuning_cfg.max_epochs_per_trial = args.max_epochs
    
    # Handle mode selection
    if args.multiclass:
        tuning_cfg.binary_mode = False
        tuning_cfg.model_dir = Path(PROC_MODELS) / "multiclass"
        logger.info(f"{CYAN}Mode: Multi-class (5 classes){RESET}")
    else:
        tuning_cfg.binary_mode = args.binary_mode
        if tuning_cfg.binary_mode:
            tuning_cfg.model_dir = Path(PROC_MODELS) / "binary"
            logger.info(f"{CYAN}Mode: Binary (LEAK/NOLEAK){RESET}")
        else:
            tuning_cfg.model_dir = Path(PROC_MODELS) / "multiclass"
            logger.info(f"{CYAN}Mode: Multi-class (5 classes){RESET}")
    
    run_optimization(tuning_cfg)


if __name__ == "__main__":
    main()
