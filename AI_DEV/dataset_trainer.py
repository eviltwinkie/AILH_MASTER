#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Label Dataset Trainer v15 - Production Leak Detection Training

Leak-optimized training script for multi-label acoustic leak detection models.
Implements dual-head architecture with class weighting, auxiliary leak detection,
and paper-exact file-level evaluation metrics.

Key Features:
    - Dual-head architecture (multiclass + binary leak-vs-rest)
    - Class-weighted or Focal loss for imbalanced datasets
    - Optional LEAK class oversampling at segment level
    - **Paper-exact file-level evaluation with 50% voting threshold** (IMPLEMENTED)
    - Step-accurate resume with StatefulSampler
    - GPU optimizations (AMP, TF32, channels_last, torch.compile)
    - Rolling checkpoint management (keep last K)
    - Live GPU/CPU monitoring during evaluation
    - Early stopping based on file-level leak F1 score
    - Optional SpecAugment data augmentation

Architecture - LeakCNNMulti:
    Conv2D(1→32) → Conv2D(32→64) → MaxPool(2,1) →
    Conv2D(64→128) → MaxPool(2,1) → AdaptiveAvgPool2d(16,1) →
    Dropout(0.25) → FC(2048→256) →
    ├── Classification head (256→n_classes)
    └── Leak detection head (256→1)

Training Strategy:
    - Primary loss: Weighted CrossEntropy or Focal Loss
    - Auxiliary loss: BCEWithLogitsLoss (leak-vs-rest)
    - Combined: L_total = L_primary + λ*L_auxiliary (λ=0.5)
    - Optimizer: AdamW with CosineAnnealingLR
    - Mixed precision: FP16 with GradScaler
    - Gradient clipping: 1.0

File-Level Evaluation (Paper-Exact):
    1. For each file, process all long×short segments
    2. Compute leak probability per short segment:
       p = 0.5*softmax(logits)[leak_idx] + 0.5*sigmoid(leak_logit)
    3. Average probabilities within each long segment
    4. File classified as LEAK if ≥50% of long segments have p ≥ threshold

Configuration:
    Defaults optimized for 5-class problem (LEAK, NORMAL, QUIET, RANDOM, MECHANICAL)
    Adjust Config dataclass for 2-class (LEAK/NOLEAK) or 6-class (+UNCLASSIFIED)

Dataset Requirements:
    - TRAINING_DATASET.H5   (built by dataset_builder.py)
    - VALIDATION_DATASET.H5
    - TESTING_DATASET.H5 (optional, for final evaluation)

Output Files:
    MODEL_DIR/best.pth             - Best model weights (by file-level F1)
    MODEL_DIR/model_meta.json      - Metadata with threshold and config
    MODEL_DIR/checkpoints/last.pth - Latest checkpoint for resume
    MODEL_DIR/checkpoints/epoch_*.pth - Rolling checkpoints

Usage:
    Edit Config dataclass paths and hyperparameters, then run:
    python dataset_trainer.py

CTRL-C Handling:
    Gracefully saves checkpoint before exit (SIGINT/SIGTERM handlers)
"""
# =============================================================================
# dataset_trainer.py
# Focused on LEAK performance:
#  - Class-weighted CE (or focal) + auxiliary leak-vs-rest head (BCE)
#  - PR sweep on validation to pick best leak threshold
#  - Step-accurate resume (StatefulSampler), AMP, TF32, channels_last
#  - Checkpoints store best_leak_thr; model_meta.json written alongside
#  - Optional SpecAugment; optional LEAK oversampling (kept off by default)
# 5 LABELS
# =============================================================================

from __future__ import annotations

import os
import json
import logging
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator, Union, cast

import h5py
import numpy as np
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_NVML = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

# Color codes for terminal output
CYAN, GREEN, YELLOW, RED, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"

# HDF5 dataset keys
HDF5_SEGMENTS_KEY = "segments_mel"
HDF5_LABELS_KEY = "labels"
HDF5_LABELS_JSON_ATTR = "labels_json"
HDF5_CONFIG_JSON_ATTR = "config_json"

# Checkpoint filenames
CHECKPOINT_LAST = "last.pth"
CHECKPOINT_BEST = "best.pth"
CHECKPOINT_EPOCH_PREFIX = "epoch_"
MODEL_METADATA_FILE = "model_meta.json"


# ======================================================================
# LOGGING SETUP
# ======================================================================

# Set TRAINER_LOG_LEVEL=DEBUG in your environment for deep diagnostics.
LOG_LEVEL = os.environ.get("TRAINER_LOG_LEVEL", "INFO").upper()

# Custom formatter to show elapsed time instead of timestamp
class ElapsedTimeFormatter(logging.Formatter):
    """Logging formatter that displays elapsed time since initialization."""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.start_time = time.time()
    
    def format(self, record):
        elapsed = time.time() - self.start_time
        record.elapsed = f"{elapsed:7.2f}s"
        return super().format(record)

# Configure logging with elapsed time - clear any existing handlers first
logger = logging.getLogger("dataset_trainer")
logger.handlers.clear()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

handler = logging.StreamHandler()
handler.setFormatter(ElapsedTimeFormatter("[%(elapsed)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


# =============================================================================
# CONSTANTS & GLOBAL CONFIGURATION
# =============================================================================

# Performance optimization settings
OS_THREAD_LIMIT = 1  # Limit CPU thread pool to avoid over-subscription
TF32_ENABLED = True  # Enable TensorFloat-32 for faster matrix operations
CUDNN_BENCHMARK = True  # Enable cuDNN auto-tuner for optimal conv algorithms

# Memory format optimization
CHANNELS_LAST_DEFAULT = True  # Use NHWC layout for 20-30% conv speedup

# Early stopping configuration
EARLY_STOP_METRIC = "file_leak_f1"  # Primary metric for model selection
FILE_LEVEL_VOTE_THRESHOLD = 0.5  # 50% of long segments must exceed threshold

# =============================================================================
# SYSTEM MONITORING & PROFILING
# =============================================================================

class EvalStatus:
    """
    System resource monitor for file-level evaluation.
    
    Provides real-time tracking of:
    - Evaluation progress (files processed, predictions made)
    - System resources (CPU, RAM utilization)
    - GPU resources (GPU utilization, VRAM usage) if available
    
    Architecture:
    -------------
    - Thread-safe counter updates using threading.Lock
    - Optional NVML integration for GPU monitoring
    - Graceful degradation if NVML unavailable
    - Background monitoring disabled to avoid tqdm conflicts
    
    Usage:
    ------
    >>> status = EvalStatus(total_files=1000)
    >>> status.start()
    >>> # ... during evaluation ...
    >>> status.update(done=batch_count, correct=correct_count)
    >>> status.stop()  # Cleanup NVML resources
    
    Thread Safety:
    --------------
    All counter updates are atomic via threading.Lock to ensure
    consistency when called from multiple threads.
    """
    def __init__(self, total_files: int):
        self.total = total_files
        self.done = 0
        self.correct = 0
        self.pred_leaks = 0
        self._stop = threading.Event()
        self._th = None
        self._nvml = None
        self._lock = threading.Lock()  # For atomic counter updates
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()  # type: ignore[possibly-unbound]
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[possibly-unbound]
            except Exception:
                self._nvml = None

    def update(self, done: Optional[int] = None, correct: Optional[int] = None, pred_leaks: Optional[int] = None):
        """Update progress counters atomically (thread-safe)."""
        with self._lock:
            if done is not None:
                self.done = done
            if correct is not None:
                self.correct = correct
            if pred_leaks is not None:
                self.pred_leaks = pred_leaks

    def gpu_line(self) -> str:
        """Get GPU utilization and VRAM usage."""
        if self._nvml is None:
            return "GPU N/A"
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml).gpu  # type: ignore[possibly-unbound]
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml)  # type: ignore[possibly-unbound]
            vram = mem.used / (1024 ** 3)  # type: ignore[operator]
            return f"GPU {util}% | VRAM {vram:.2f} GB"
        except Exception:
            return "GPU N/A"

    def run(self):
        """Background monitoring loop (disabled by default to avoid tqdm conflicts).
        
        Note: This method is intentionally disabled because:
        1. tqdm progress bars provide clearer, real-time feedback
        2. Concurrent printing can interfere with tqdm display
        3. System metrics can be monitored externally (nvidia-smi, htop)
        
        To re-enable, implement polling loop similar to dataset_builder.py
        """
        # Monitoring disabled - tqdm progress bar provides sufficient feedback
        pass

    def start(self):
        """Start background monitoring thread."""
        if self._th is None:
            self._th = threading.Thread(target=self.run, daemon=True)
            self._th.start()

    def stop(self):
        """Stop monitoring thread and cleanup NVML."""
        self._stop.set()
        if self._th:
            try:
                self._th.join(timeout=2.0)
            except Exception:
                pass
        if self._nvml is not None:
            try:
                pynvml.nvmlShutdown()  # type: ignore[possibly-unbound]
            except Exception:
                pass

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================

# Limit CPU threading to prevent over-subscription with GPU workloads
# PyTorch DataLoader workers already provide parallelism
os.environ.setdefault("OMP_NUM_THREADS", str(OS_THREAD_LIMIT))
os.environ.setdefault("MKL_NUM_THREADS", str(OS_THREAD_LIMIT))
torch.set_num_threads(OS_THREAD_LIMIT)

# Enable cuDNN auto-tuner: finds fastest convolution algorithms for this hardware
# Adds ~1-2 min warmup but speeds up training by 10-20%
torch.backends.cudnn.benchmark = CUDNN_BENCHMARK

# Enable TensorFloat-32 (TF32) for Ampere+ GPUs (RTX 30xx/40xx, A100, etc.)
# Provides 8x speedup over FP32 with minimal accuracy impact
torch.backends.cudnn.allow_tf32 = TF32_ENABLED
torch.backends.cuda.matmul.allow_tf32 = TF32_ENABLED

# Set matmul precision hint for PyTorch 2.0+
try:
    torch.set_float32_matmul_precision("high")  # Enables TF32 for matmuls
except Exception:
    pass  # Older PyTorch versions don't have this API

# ------------------------------- CONFIG --------------------------------------

@dataclass
class Config:
    """
    Training configuration for leak detection model.
    
    Organized into logical sections:
    1. Data paths and I/O
    2. Training hyperparameters
    3. Loss functions and class weighting
    4. Auxiliary leak detection head
    5. Data augmentation
    6. Optimization and compilation
    7. Checkpointing and resume
    
    Key Design Decisions:
    ---------------------
    - batch_size=5632: Maximizes GPU utilization on RTX 5090 (24GB VRAM)
    - learning_rate=1e-3: AdamW default, works well with cosine annealing
    - early_stop_patience=15: Balances exploration vs compute cost
    - use_compile=True: PyTorch 2.0+ compilation for 15-20% speedup
    - use_channels_last=True: NHWC memory layout for 20-30% conv speedup
    
    Usage:
    ------
    >>> cfg = Config()
    >>> cfg.batch_size = 4096  # Adjust for different GPU
    >>> cfg.epochs = 100       # Shorter training
    
    Note: Config initialization now happens in __post_init__ to avoid
    import-time side effects.
    """
    
    def __post_init__(self):
        """Load paths from global_config after dataclass initialization.
        
        Defers global_config import to avoid import-time side effects.
        Allows Config to be instantiated before global_config is available.
        """
        # Add parent directory to sys.path for global_config import
        parent_path = str(Path(__file__).parent.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        
        from global_config import MASTER_DATASET, PROC_MODELS, PROC_LOGS
        
        # Update paths if using defaults
        if self.stage_dir == Path("/DEVELOPMENT/DATASET_REFERENCE"):
            self.stage_dir = Path(MASTER_DATASET)
        if self.model_dir == Path("/DEVELOPMENT/DATASET_REFERENCE/MODELS"):
            self.model_dir = Path(PROC_MODELS)
        if self.log_dir == Path("/DEVELOPMENT/DATASET_REFERENCE/LOGS"):
            self.log_dir = Path(PROC_LOGS)

    # ========== DATA PATHS & I/O ==========
    # Paths will be loaded from global_config in __post_init__
    stage_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE")
    hdf5_name: str = "TRAINING_DATASET.H5"       # Training split
    val_hdf5_name: str = "VALIDATION_DATASET.H5" # Validation split
    test_hdf5_name: str = "TESTING_DATASET.H5"   # Test split (optional)
    model_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/MODELS")
    log_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/LOGS")

    # ========== TRAINING HYPERPARAMETERS ==========
    batch_size: int = 16384             # Training batch size (maximized for GPU utilization)
    val_batch_size: int = 8192          # Validation batch size (maximized for throughput)
    epochs: int = 200                   # Maximum training epochs
    learning_rate: float = 1e-3         # Initial learning rate for AdamW
    dropout: float = 0.25               # Dropout probability for regularization
    binary_mode: bool = False           # True: LEAK/NOLEAK binary classification, False: All classes
    num_classes: int = 5                # Number of output classes (auto-set to 2 if binary_mode=True)
    grad_clip_norm: Optional[float] = 1.0  # Gradient clipping threshold (None to disable)
    early_stop_patience: int = 15       # Stop if no improvement for N epochs

    # ========== LOSS FUNCTIONS & CLASS WEIGHTING ==========
    leak_class_name: str = "LEAK"       # Name of leak class in dataset
    loss_type: str = "weighted_ce"      # Loss: "ce", "weighted_ce", "focal"
    focal_gamma: float = 2.0            # Focal loss focusing parameter (if loss_type="focal")
    focal_alpha_leak: float = 0.75      # Focal loss weight for leak class (others: (1-α)/(C-1))

    # ========== AUXILIARY LEAK DETECTION HEAD ==========
    use_leak_aux_head: bool = True      # Enable binary leak-vs-rest auxiliary head
    leak_aux_weight: float = 0.5        # Loss weight: L_total = L_primary + λ*L_aux
    leak_aux_pos_weight: Optional[float] = None  # BCE pos_weight (None=auto from data)

    # ========== DATA BALANCING ==========
    leak_oversample_factor: int = 1     # Duplicate LEAK segments N times (1=disabled)

    # ========== DATALOADER CONFIGURATION ==========
    num_workers: int = 12               # Parallel data loading workers (maximized for throughput)
    prefetch_factor: int = 24           # Batches to prefetch per worker (extreme prefetch)
    persistent_workers: bool = True     # Keep workers alive between epochs
    pin_memory: bool = True             # Pin memory for faster GPU transfer
    preload_to_ram: bool = True         # Preload entire dataset to RAM (requires ~8GB, eliminates I/O)

    # ========== DATA AUGMENTATION ==========
    use_specaugment: bool = False       # Enable SpecAugment (requires torchaudio)
    time_mask_param: int = 6            # Max time steps to mask
    freq_mask_param: int = 8            # Max frequency bins to mask

    # ========== OPTIMIZATION & COMPILATION ==========
    use_compile: bool = True            # PyTorch 2.0+ model compilation (enables kernel fusion)
    compile_mode: Optional[str] = "max-autotune"  # Aggressive optimization for best throughput
    use_channels_last: bool = True      # NHWC memory layout for faster convs

    # ========== CHECKPOINTING & RESUME ==========
    auto_resume: bool = True            # Auto-resume from last.pth if exists
    keep_last_k: int = 3                # Keep last K epoch checkpoints

    # ========== PERFORMANCE PROFILING ==========
    profile_performance: bool = True    # Enable detailed performance profiling
    profile_gpu_util: bool = True       # Monitor GPU utilization during training

    seed: Optional[int] = 1234

    @property
    def train_h5(self) -> Path:
        return self.stage_dir / self.hdf5_name

    @property
    def val_h5(self) -> Path:
        return self.stage_dir / self.val_hdf5_name

    @property
    def test_h5(self) -> Path:
        return self.stage_dir / self.test_hdf5_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.model_dir / "checkpoints"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: Optional[int]):
    """
    Set random seed for reproducibility across all libraries.
    
    Seeds:
    - Python random module
    - NumPy random generator
    - PyTorch CPU generator
    - PyTorch CUDA generators (all GPUs)
    
    Note: Perfect reproducibility requires additional settings:
    - torch.backends.cudnn.deterministic = True (slower)
    - torch.backends.cudnn.benchmark = False (slower)
    - Single-threaded execution (CUDA operations are non-deterministic)
    """
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bytes_human(n: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        n: Number of bytes
    
    Returns:
        Human-readable string (e.g., "1.50 GB")
    
    Example:
        >>> bytes_human(1536)
        '1.50 KB'
        >>> bytes_human(1073741824)
        '1.00 GB'
    """
    n_float = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_float < 1024:
            return f"{n_float:.2f} {unit}"
        n_float /= 1024
    return f"{n_float:.2f} PB"

def ceiling_half(n: int) -> int:
    """
    Return ceiling of n/2 for 50% threshold calculations.
    
    Used for file-level voting: A file is classified as LEAK if
    ≥50% of its long segments exceed the probability threshold.
    
    Args:
        n: Number of items
    
    Returns:
        Ceiling of n/2 (rounds up)
    
    Examples:
        >>> ceiling_half(1)  # 1 → need 1/1 = 100%
        1
        >>> ceiling_half(2)  # 2 → need 1/2 = 50%
        1
        >>> ceiling_half(3)  # 3 → need 2/3 = 67%
        2
        >>> ceiling_half(79) # 79 → need 40/79 = 51%
        40
    """
    return (n + 1) // 2

def device_setup() -> torch.device:
    """
    Initialize CUDA device and verify availability.
    
    Returns:
        torch.device: CUDA device object
    
    Raises:
        RuntimeError: If CUDA is not available
    
    Note: This trainer requires CUDA for:
    - Mixed precision training (AMP)
    - Efficient batch processing
    - Memory-optimized formats (channels_last)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    return torch.device("cuda")


class SystemMonitor:
    """
    Real-time system resource monitor for training.
    
    Monitors:
    - GPU utilization % (via NVML)
    - VRAM usage GB (allocated/total)
    - CPU utilization % (via psutil)
    - RAM usage GB (used/total)
    
    Provides live status updates during training to identify bottlenecks.
    """
    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.nvml_handle = None
        self.process = None
        
        # Initialize NVML for GPU monitoring
        if enabled and _HAS_NVML and pynvml is not None and device.type == 'cuda':
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            except Exception:
                self.nvml_handle = None
        
        # Initialize psutil for CPU/RAM monitoring
        if enabled and _HAS_PSUTIL and psutil is not None:
            try:
                self.process = psutil.Process()
            except Exception:
                self.process = None
    
    def get_stats(self) -> Dict[str, float]:
        """Get current system statistics."""
        stats = {}
        
        # GPU metrics
        if self.nvml_handle is not None and pynvml is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                stats['gpu_util'] = util.gpu  # type: ignore
                mem_used = float(mem_info.used)  # type: ignore
                mem_total = float(mem_info.total)  # type: ignore
                stats['vram_used_gb'] = mem_used / (1024**3)
                stats['vram_total_gb'] = mem_total / (1024**3)
                stats['vram_percent'] = (mem_used / mem_total) * 100
            except Exception:
                stats['gpu_util'] = -1
                stats['vram_used_gb'] = -1
                stats['vram_total_gb'] = -1
                stats['vram_percent'] = -1
        else:
            # Fallback to PyTorch for VRAM
            if self.device.type == 'cuda':
                try:
                    stats['vram_used_gb'] = torch.cuda.memory_allocated(self.device) / (1024**3)
                    stats['vram_total_gb'] = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                    stats['vram_percent'] = (stats['vram_used_gb'] / stats['vram_total_gb']) * 100
                except Exception:
                    pass
        
        # CPU/RAM metrics
        if _HAS_PSUTIL and psutil is not None:
            try:
                stats['cpu_percent'] = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                stats['ram_used_gb'] = mem.used / (1024**3)
                stats['ram_total_gb'] = mem.total / (1024**3)
                stats['ram_percent'] = mem.percent
            except Exception:
                stats['cpu_percent'] = -1
                stats['ram_used_gb'] = -1
                stats['ram_total_gb'] = -1
                stats['ram_percent'] = -1
        
        return stats
    
    def format_stats(self) -> str:
        """Format statistics as human-readable string."""
        stats = self.get_stats()
        parts = []
        
        # GPU info
        if 'gpu_util' in stats and stats['gpu_util'] >= 0:
            parts.append(f"GPU:{stats['gpu_util']:3.0f}%")
        
        # VRAM info
        if 'vram_used_gb' in stats and stats['vram_used_gb'] >= 0:
            parts.append(f"VRAM:{stats['vram_used_gb']:4.1f}/{stats['vram_total_gb']:.1f}GB({stats['vram_percent']:4.1f}%)")
        
        # CPU info
        if 'cpu_percent' in stats and stats['cpu_percent'] >= 0:
            parts.append(f"CPU:{stats['cpu_percent']:5.1f}%")
        
        # RAM info
        if 'ram_used_gb' in stats and stats['ram_used_gb'] >= 0:
            parts.append(f"RAM:{stats['ram_used_gb']:4.1f}/{stats['ram_total_gb']:.1f}GB({stats['ram_percent']:4.1f}%)")
        
        return " | ".join(parts) if parts else "N/A"
    
    def __del__(self):
        """Cleanup NVML resources."""
        if self.nvml_handle is not None and _HAS_NVML and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class GPUProfiler:
    """
    Real-time GPU utilization profiler for performance analysis.
    
    Monitors:
    - GPU utilization %
    - Memory usage (allocated/reserved)
    - SM occupancy
    - Kernel execution time
    """
    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled and device.type == 'cuda'
        self.samples = []
        self.start_time = None
        self.nvml_handle = None
        
        # Initialize nvml if enabled
        if self.enabled and _HAS_NVML and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
            except Exception:
                self.nvml_handle = None
        
    def sample(self, phase: str = ""):
        """Sample current GPU state."""
        if not self.enabled:
            return
            
        try:
            # Use cached handle instead of getting it each time
            if self.nvml_handle is not None and pynvml is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                gpu_util = util.gpu
            else:
                gpu_util = -1
                
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            
            self.samples.append({
                'time': time.perf_counter() - (self.start_time or 0),
                'phase': phase,
                'gpu_util': gpu_util,
                'mem_allocated_gb': allocated,
                'mem_reserved_gb': reserved,
            })
        except Exception:
            pass
    
    def __del__(self):
        """Cleanup nvml resources."""
        if self.enabled and _HAS_NVML and self.nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    
    def start(self):
        """Start profiling session."""
        self.start_time = time.perf_counter()
        self.samples = []
        
    def report(self) -> str:
        """Generate performance report."""
        if not self.samples:
            return "No profiling data"
            
        gpu_utils = [s['gpu_util'] for s in self.samples if s['gpu_util'] >= 0]
        if gpu_utils:
            avg_util = sum(gpu_utils) / len(gpu_utils)
            max_util = max(gpu_utils)
            min_util = min(gpu_utils)
        else:
            avg_util = max_util = min_util = -1
            
        mem_allocated = [s['mem_allocated_gb'] for s in self.samples]
        avg_mem = sum(mem_allocated) / len(mem_allocated)
        peak_mem = max(mem_allocated)
        
        report = f"GPU: {avg_util:.1f}% avg (min={min_util:.1f}%, max={max_util:.1f}%) | "
        report += f"VRAM: {avg_mem:.2f}GB avg, {peak_mem:.2f}GB peak"
        return report


def get_rng_state_safe() -> Dict:
    """
    Capture complete RNG state for checkpoint saving.
    
    Captures state from:
    1. PyTorch CPU generator
    2. PyTorch CUDA generators (all devices)
    3. NumPy random generator
    
    Returns:
        Dictionary with serializable RNG state
    
    Note: Python's random module state is captured via torch.manual_seed
    which seeds Python's random internally.
    """
    t_cpu = torch.get_rng_state()
    t_cuda = torch.cuda.get_rng_state_all()
    algo, keys, pos, has_gauss, cached = np.random.get_state()
    # Convert to JSON-serializable types
    np_state = (
        str(algo),
        [int(x) for x in keys.tolist()],  # type: ignore[union-attr]
        int(pos),
        int(has_gauss),
        float(cached)
    )
    return {"torch_cpu": t_cpu, "torch_cuda": t_cuda, "numpy": np_state}

def set_rng_state_safe(state: Optional[Dict]):
    """
    Restore RNG state from checkpoint.
    
    Args:
        state: Dictionary from get_rng_state_safe() or None
    
    Note: Enables exact training resumption with identical random
    data augmentation, dropout masks, and weight initialization.
    """
    if not state:
        return
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    algo, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((algo, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))

def rotate_checkpoints(ckpt_dir: Path, keep_last_k: int):
    """
    Manage checkpoint storage by keeping only the K most recent epoch checkpoints.
    
    Args:
        ckpt_dir: Directory containing epoch_*.pth checkpoints
        keep_last_k: Number of most recent checkpoints to retain
    
    Behavior:
    ---------
    - Finds all epoch_*.pth files in directory
    - Sorts by filename (lexicographic = chronological for zero-padded names)
    - Deletes oldest checkpoints beyond keep_last_k limit
    - Preserves last.pth and best.pth (different naming pattern)
    
    Performance:
    ------------
    - Only sorts when deletion needed (len > keep_last_k)
    - Silent failure on individual file deletions (e.g., permission errors)
    - O(N log N) where N = number of checkpoint files
    
    Example:
    --------
    Files: epoch_001.pth, epoch_002.pth, epoch_003.pth, epoch_004.pth
    keep_last_k=2 → Deletes epoch_001.pth, epoch_002.pth
    """
    ckpts = list(ckpt_dir.glob(f"{CHECKPOINT_EPOCH_PREFIX}*.pth"))
    if len(ckpts) > keep_last_k:
        ckpts.sort()  # Lexicographic sort works for zero-padded numbers
        for p in ckpts[:-keep_last_k]:
            try:
                p.unlink()
            except Exception:
                pass  # Ignore deletion errors (file in use, permissions, etc.)


def prepare_mel_batch(
    mel: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    use_channels_last: bool = True,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Prepare mel spectrogram batch for model input.
    
    Handles:
    - NumPy → Tensor conversion
    - Adding channel dimension (unsqueeze)
    - Channels-last memory layout (NHWC) for faster convolutions
    - Device transfer with optional type casting
    
    Args:
        mel: Input mel spectrogram [B, n_mels, t_frames] or [B, C, n_mels, t_frames]
        device: Target device (cuda/cpu)
        use_channels_last: Convert to channels_last memory format
        dtype: Target dtype (default: float16 for AMP)
    
    Returns:
        Prepared tensor [B, 1, n_mels, t_frames] on device with correct dtype
    
    Performance:
    -----------
    - Channels-last provides 20-30% speedup for conv operations
    - Non-blocking transfer overlaps CPU→GPU copy with compute
    - Single memory allocation (no intermediate copies)
    
    Example:
    --------
    >>> mel_np = np.random.randn(32, 128, 100)  # [B, n_mels, t_frames]
    >>> mel_t = prepare_mel_batch(mel_np, torch.device('cuda'))
    >>> mel_t.shape
    torch.Size([32, 1, 128, 100])
    """
    # Convert to tensor if needed
    if isinstance(mel, np.ndarray):
        mel_t = torch.from_numpy(mel)
    else:
        mel_t = mel
    
    # Add channel dimension if missing: [B, n_mels, t_frames] → [B, 1, n_mels, t_frames]
    if mel_t.ndim == 3:
        mel_t = mel_t.unsqueeze(1)
    
    # Convert to channels-last memory layout for faster conv operations
    if use_channels_last:
        mel_t = mel_t.contiguous(memory_format=torch.channels_last)
    
    # Transfer to device with type casting
    mel_t = mel_t.to(device, dtype=dtype, non_blocking=True)
    
    return mel_t


# =============================================================================
# DATASET & DATA LOADING
# =============================================================================

class BinaryLabelDataset(Dataset):
    """
    Transparent wrapper that converts multi-class labels to binary LEAK/NOLEAK.
    
    This wrapper enables binary classification training using datasets originally
    designed for multi-class problems. It transparently forwards all attributes
    and methods to the underlying dataset while converting labels on-the-fly.
    
    Label Conversion:
    -----------------
    - Original LEAK class (leak_idx) → 1 (LEAK)
    - All other classes → 0 (NOLEAK)
    
    The wrapper maintains compatibility with file-level evaluation by preserving
    access to the original file labels in the HDF5 dataset. During evaluation,
    these original labels are used to determine ground truth LEAK files.
    
    Implementation:
    ---------------
    Uses __getattr__ to forward all attribute access to base_dataset, making
    this a truly transparent wrapper. This eliminates the need to manually
    forward every HDF5 attribute (num_files, _segs, _labels, etc.).
    
    Args:
        base_dataset: LeakMelDataset with multi-class labels
        leak_idx: Index of LEAK class in original labels (e.g., 2 for 5-class)
    
    Example:
        >>> # 5-class dataset with LEAK at index 2
        >>> base_ds = LeakMelDataset('train.h5')
        >>> binary_ds = BinaryLabelDataset(base_ds, leak_idx=2)
        >>> mel, label = binary_ds[0]  # label is 0 or 1
        >>> binary_ds.num_files  # Forwarded from base_ds
    """
    def __init__(self, base_dataset: Dataset, leak_idx: int):
        self.base_dataset = base_dataset
        self.leak_idx = leak_idx
    
    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore
    
    def __getitem__(self, index: int):
        """Convert segment label to binary on-the-fly."""
        mel_t, label_t = self.base_dataset[index]
        # Efficient in-place conversion: 1 if LEAK, 0 otherwise
        binary_label = torch.tensor(1 if label_t.item() == self.leak_idx else 0, dtype=torch.long)
        return mel_t, binary_label
    
    def __getattr__(self, name):
        """
        Transparent attribute forwarding to base dataset.
        
        This magic method forwards ALL attribute access to the underlying dataset,
        including HDF5 attributes needed for file-level evaluation:
        - num_files, num_long, num_short (dataset structure)
        - _segs, _labels, h5f (HDF5 datasets)
        - _ensure_open, _has_channel (methods)
        
        This eliminates the need for manual forwarding and makes the wrapper
        truly transparent to downstream code.
        """
        return getattr(self.base_dataset, name)
    
    def __enter__(self):
        if hasattr(self.base_dataset, '__enter__'):
            self.base_dataset.__enter__()  # type: ignore
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.base_dataset, '__exit__'):
            return self.base_dataset.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
        return False


class LeakMelDataset(Dataset):
    """
    HDF5-backed dataset for hierarchically segmented mel spectrograms.
    
    Data Structure:
    ---------------
    HDF5 file contains:
    - /segments_mel: [num_files, num_long, num_short, (C,) n_mels, t_frames]
    - /labels: [num_files] - file-level labels
    - attributes: labels_json (class names), config_json (builder config)
    
    Segmentation Hierarchy:
    ----------------------
    File → Long segments (1024 samples, 512 hop) → Short segments (512 samples, 256 hop)
    Example: 10s @ 4096Hz → 79 long × 3 short = 237 segments per file
    
    Caching Strategy:
    -----------------
    Caches entire file blocks (all long×short segments for a file) in memory.
    LRU eviction when cache exceeds cache_files limit.
    Default: 256 files cached (~6GB for typical 32×1 mel specs).
    
    Performance:
    ------------
    - SWMR mode: Allows concurrent reads from multiple workers
    - Zero-copy: torch.from_numpy() creates view into HDF5 mmap
    - Lazy loading: HDF5 file opened on first __getitem__ call (per worker)
    
    Returns:
    --------
    Per-segment basis:
    - mel_t: [n_mels, t_frames] mel spectrogram (float32)
    - lbl_t: scalar label (long) - file-level label replicated for all segments
    
    Note:
    -----
    Labels are file-level but returned per-segment for training convenience.
    File-level evaluation (evaluate_file_level) aggregates predictions back to files.
    """
    def __init__(self, h5_path: Path, cache_files: int = 1024, preload_to_ram: bool = False):
        self.h5_path = str(h5_path)
        self.h5f = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_files = cache_files
        self._preloaded = False
        self._preloaded_segs = None
        self._preloaded_labels = None
        
        with h5py.File(self.h5_path, "r") as f:  # type: ignore[misc]
            segs = f[HDF5_SEGMENTS_KEY]; shp = tuple(segs.shape)  # type: ignore[union-attr]
            if len(shp) == 5:
                self.num_files, self.num_long, self.num_short, self.n_mels, self.t_frames = shp
                self._has_channel = False
            elif len(shp) == 6:
                self.num_files, self.num_long, self.num_short, _, self.n_mels, self.t_frames = shp
                self._has_channel = True
            else:
                raise RuntimeError(f"Unsupported segments_mel shape: {shp}")
            self._dtype = segs.dtype  # type: ignore[union-attr]
            self._class_names = None
            try:
                lbls = f.attrs.get(HDF5_LABELS_JSON_ATTR)
                if isinstance(lbls, (bytes, bytearray)): lbls = lbls.decode("utf-8")
                if lbls: self._class_names = json.loads(lbls)
            except Exception: pass
            self.builder_cfg = None
            try:
                cj = f.attrs.get(HDF5_CONFIG_JSON_ATTR)
                if isinstance(cj, (bytes, bytearray)): cj = cj.decode("utf-8")
                if cj: self.builder_cfg = json.loads(cj)
            except Exception: pass
            
            # Preload entire dataset to RAM if requested
            if preload_to_ram:
                logger.info("Preloading dataset to RAM...")
                import time
                start = time.time()
                self._preloaded_segs = segs[:]  # type: ignore[index] # Load entire array to RAM
                self._preloaded_labels = f[HDF5_LABELS_KEY][:]  # type: ignore[index] # Load labels
                elapsed = time.time() - start
                size_gb = self._preloaded_segs.nbytes / (1024**3)  # type: ignore[union-attr]
                logger.info("✓ Preloaded %.2f GB to RAM in %.1fs (%.0f MB/s)", 
                           size_gb, elapsed, (size_gb * 1024) / elapsed)
                self._preloaded = True
        
        self.total_segments = self.num_files * self.num_long * self.num_short

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    def _ensure_open(self):
        if self.h5f is None:
            # Open with larger chunk cache for better read performance
            # Use rdcc_nbytes=50MB (default is 1MB) to cache more file blocks
            self.h5f = h5py.File(
                self.h5_path, "r", 
                libver="latest",
                rdcc_nbytes=50 * 1024 * 1024,  # 50MB chunk cache
                rdcc_nslots=5003  # Prime number for better hash distribution
            )  # type: ignore[misc]
            self._segs = self.h5f[HDF5_SEGMENTS_KEY]  # type: ignore[misc]
            self._labels = self.h5f[HDF5_LABELS_KEY]  # type: ignore[misc]

    def _get_file_block(self, file_idx: int) -> np.ndarray:
        blk = self._cache.get(file_idx)
        if blk is not None:
            # Move to end (LRU: most recently used) - O(1) operation
            self._cache.move_to_end(file_idx)
            return blk
        blk = self._segs[file_idx]  # type: ignore[index] # numpy view
        self._cache[file_idx] = blk  # type: ignore[assignment]
        if len(self._cache) > self._cache_files:
            # Remove oldest (first) item - O(1) operation
            self._cache.popitem(last=False)
        return blk  # type: ignore[return-value]

    def __len__(self): return self.total_segments

    def __getitem__(self, index: int):
        LxS = self.num_long * self.num_short
        fidx = index // LxS
        rem  = index %  LxS
        li   = rem // self.num_short
        si   = rem %  self.num_short
        
        # Use preloaded RAM data if available (ZERO I/O latency)
        if self._preloaded:
            mel = self._preloaded_segs[fidx, li, si]  # type: ignore[index]
            if self._has_channel: mel = mel[0]  # type: ignore[index]
            mel_t = torch.from_numpy(mel)  # zero-copy view
            lbl_t = torch.tensor(int(self._preloaded_labels[fidx]), dtype=torch.long)  # type: ignore[index,arg-type]
            return mel_t, lbl_t
        
        # Fall back to HDF5 I/O with caching
        self._ensure_open()
        blk = self._get_file_block(fidx)  # [num_long,num_short,(C,)n_mels,t_frames]
        mel = blk[li, si]
        if self._has_channel: mel = mel[0]
        mel_t = torch.from_numpy(mel)  # zero-copy
        lbl_t = torch.tensor(int(self._labels[fidx]), dtype=torch.long)  # type: ignore[index,arg-type]
        return mel_t, lbl_t

    def __enter__(self):
        """Context manager entry: ensure HDF5 file is open."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close HDF5 file reliably."""
        if self.h5f is not None:
            try:
                self.h5f.close()  # type: ignore[misc]
            except Exception:
                pass
            self.h5f = None
        return False

    def __del__(self):
        """Fallback cleanup (unreliable, use context manager instead)."""
        try:
            if self.h5f is not None: self.h5f.close()  # type: ignore[misc]
        except Exception: pass


# --------------------------- Sampler (resumeable) -----------------------------

class StatefulSampler(Sampler[int]):
    """
    Stateful sampler supporting exact resume from any training step.
    
    Functionality:
    --------------
    - Maintains epoch number, position within epoch, and permutation state
    - Supports deterministic shuffling with seed + epoch mixing
    - Can serialize/deserialize complete state for checkpoint resume
    
    Resume Guarantees:
    ------------------
    When resuming from checkpoint:
    1. Same epoch → continues from exact position (no repeated/skipped samples)
    2. Same permutation → deterministic order preserved
    3. RNG state → synchronized with training loop's RNG
    
    Shuffling:
    ----------
    - Uses numpy.random.default_rng(seed ^ epoch) for deterministic shuffling
    - Each epoch gets different permutation but reproducible with same seed
    - Non-shuffle mode: maintains original order
    
    Usage:
    ------
    >>> sampler = StatefulSampler(indices=[0,1,2,3], shuffle=True, seed=42)
    >>> sampler.on_epoch_start(1)  # Call at start of each epoch
    >>> loader = DataLoader(dataset, sampler=sampler)
    >>> # ... training ...
    >>> state = sampler.state_dict()  # Save in checkpoint
    >>> sampler.load_state_dict(state)  # Restore from checkpoint
    
    Performance:
    ------------
    - Permutation computed once per epoch (O(N) time, O(N) memory)
    - Iterator overhead: O(1) per sample
    - Negligible impact vs shuffle=True in standard DataLoader
    """
    def __init__(self, indices: List[int], shuffle: bool = True, seed: Optional[int] = None):
        self.indices = list(indices)
        self.shuffle = bool(shuffle)
        self.seed = int(seed or 0)
        self.epoch = 1
        self.pos = 0
        self._perm = np.arange(len(self.indices), dtype=np.int64)
        if self.shuffle: self._regen_perm()

    def __len__(self) -> int: return len(self.indices) - self.pos

    def _regen_perm(self):
        rng = np.random.default_rng(self.seed ^ self.epoch)
        self._perm = rng.permutation(len(self.indices)).astype(np.int64, copy=False)
        self.pos = 0

    def on_epoch_start(self, epoch: int):
        if epoch != self.epoch and self.shuffle:
            self.epoch = int(epoch); self._regen_perm()
        else:
            self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        N = len(self.indices)
        p = self._perm if self.shuffle else np.arange(N, dtype=np.int64)
        for i in range(self.pos, N):
            idx = self.indices[int(p[i])]
            self.pos = i + 1
            yield idx

    def state_dict(self) -> Dict:
        return {
            "epoch": int(self.epoch),
            "pos": int(self.pos),
            "perm": self._perm.astype(np.int64).tolist(),
            "seed": int(self.seed),
            "shuffle": bool(self.shuffle),
            "indices": list(self.indices),
        }

    def load_state_dict(self, state: Dict):
        self.epoch = int(state["epoch"]); self.pos = int(state["pos"])
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.indices = list(state.get("indices", self.indices))
        perm = state.get("perm")
        self._perm = np.asarray(perm, dtype=np.int64) if perm is not None else np.arange(len(self.indices))


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class LeakCNNMulti(nn.Module):
    """
    Dual-head CNN for leak detection with auxiliary binary classification.
    
    Supports two modes:
    - Binary mode (n_classes=2): LEAK vs NOLEAK classification
    - Multi-class mode (n_classes=5): LEAK, NORMAL, QUIET, RANDOM, MECHANICAL
    
    Architecture:
    -------------
    Input: [B, 1, 32, 1] - Mel spectrogram (batch, channel, mels, time)
    
    Backbone:
    - Conv2D(1→32, 3×3, pad=1) + ReLU
    - Conv2D(32→64, 3×3, pad=1) + ReLU
    - MaxPool2D((2,1)) → Reduce freq, keep time
    - Conv2D(64→128, 3×3, pad=1) + ReLU
    - MaxPool2D((2,1)) → Further freq reduction
    - AdaptiveAvgPool2D(16,1) → Fixed size regardless of input
    - Flatten → [B, 2048]
    - Dropout(0.25)
    - Linear(2048→256) + ReLU
    
    Heads:
    - Classification: Linear(256→n_classes) → Binary or multiclass logits
    - Leak detection: Linear(256→1) → Binary leak-vs-rest logit (optional)
    
    Design Decisions:
    -----------------
    1. **MaxPool (2,1)**: Preserves temporal granularity while reducing frequency.
       Rationale: Time contains critical leak signature info (periodic patterns).
    
    2. **AdaptiveAvgPool**: Handles variable-length inputs if needed.
       Current: Fixed 32×1 input, but flexible for future changes.
    
    3. **Dual heads**: 
       - Binary mode: Main head predicts LEAK/NOLEAK
       - Multi-class: Main head predicts all 5 classes, aux head focuses on LEAK
       - Combined in loss: L = L_ce + λ*L_bce (λ=0.5) for multi-class
    
    4. **Dropout 0.25**: Regularization for ~2M parameters, prevents overfitting.
    
    Parameters:
    -----------
    n_classes: Number of classes (2 for binary, 5 for multi-class)
    dropout: Dropout probability (default: 0.25)
    
    Returns:
    --------
    Tuple[Tensor, Tensor]:
    - logits: [B, n_classes] - Binary or multiclass logits (raw, no softmax)
    - leak_logit: [B] - Binary leak-vs-rest logit (raw, no sigmoid)
    
    Performance:
    ------------
    - Parameters: ~2.1M (mostly in fc1: 2048×256 = 524K)
    - FLOPs: ~15M per sample (dominated by convolutions)
    - Memory: ~8MB per batch of 5632 samples (FP16)
    - Throughput: ~50K samples/sec on RTX 5090 (FP16, channels_last, compiled)
    
    Training:
    ---------
    - Use channels_last memory format for 20-30% speedup
    - torch.compile with mode='reduce-overhead' for 15-20% additional speedup
    - Mixed precision (FP16) with GradScaler for 2x speedup + 50% memory reduction
    """
    def __init__(self, n_classes: int = 5, dropout: float = 0.25):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.adapt = nn.AdaptiveAvgPool2d((16, 1))
        self.fc1 = nn.Linear(128 * 16 * 1, 256)
        self.cls_head = nn.Linear(256, n_classes)
        self.leak_head = nn.Linear(256, 1)  # auxiliary leak vs rest

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = self.adapt(x); x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.cls_head(x)
        leak_logit = self.leak_head(x).squeeze(1)
        return logits, leak_logit


# =============================================================================
# LOSS FUNCTIONS & CLASS WEIGHTING
# =============================================================================
# Supports three loss strategies for imbalanced leak detection:
# 1. CrossEntropy (CE): Standard unweighted loss
# 2. Weighted CE: Inverse frequency class weights
# 3. Focal Loss: Down-weights easy examples, focuses on hard cases
# =============================================================================

def class_counts_from_labels(ds: LeakMelDataset) -> np.ndarray:
    # Count per-file labels then expand by segments per file
    with h5py.File(ds.h5_path, "r") as f:  # type: ignore[misc]
        labels = np.asarray(f[HDF5_LABELS_KEY][:], dtype=np.int64)  # type: ignore[index]
    minlength = len(ds.class_names) if ds.class_names else 0
    counts_files = np.bincount(labels, minlength=minlength)
    # Each file contributes num_long * num_short segments
    segs_per_file = ds.num_long * ds.num_short
    # Ensure no integer overflow for large datasets
    counts_segments = counts_files.astype(np.int64) * np.int64(segs_per_file)
    if (counts_segments < 0).any():
        raise OverflowError(f"Class count overflow detected. Max count: {counts_segments.max()}, "
                            f"files per class: {counts_files}, segments per file: {segs_per_file}")
    return counts_segments

class FocalLoss(nn.Module):
    def __init__(self, class_alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", class_alpha)  # shape [C]
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # CE per sample
        ce = F.cross_entropy(logits, targets, reduction="none")
        # p_t
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1,1)).squeeze(1).clamp(1e-8, 1-1e-8)
        alpha_t = self.alpha[targets]  # type: ignore[index]
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# =============================================================================
# MODEL & TRAINING SETUP
# =============================================================================
# Factory functions for creating and configuring:
# - Model with compilation and memory format optimizations
# - Loss functions with automatic class weighting
# - Data augmentation transforms (SpecAugment)
# =============================================================================

def create_model(cfg: Config, device: torch.device) -> nn.Module:
    """
    Create and configure LeakCNNMulti model with optimizations.
    
    Args:
        cfg: Training configuration
        device: Target device (cuda/cpu)
    
    Returns:
        Configured model ready for training (may be compiled or uncompiled)
    """
    model = LeakCNNMulti(n_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    
    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(
                model,
                mode=(cfg.compile_mode or "default")
            )  # type: ignore[assignment]
        except Exception as e:
            logger.warning("%storch.compile failed: %s - Using uncompiled model (15-20%% slower)%s", 
                         YELLOW, e, RESET)
    
    # Explicit cast for static type checkers: torch.compile may return a wrapper typed as Callable.
    model = cast(nn.Module, model)
    return model


def setup_loss_functions(
    cfg: Config,
    ds_tr: LeakMelDataset,
    leak_idx: int,
    device: torch.device
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """
    Setup primary and auxiliary loss functions with class weighting.
    
    Args:
        cfg: Training configuration
        ds_tr: Training dataset (for computing class frequencies)
        leak_idx: Index of leak class in original labels
        device: Target device
    
    Returns:
        Tuple of (primary_loss_fn, auxiliary_loss_fn)
    """
    class_weights_t: Optional[torch.Tensor] = None
    
    if cfg.loss_type in ("weighted_ce", "focal"):
        if cfg.binary_mode:
            # Binary mode: compute weights for LEAK vs NOLEAK
            with h5py.File(ds_tr.h5_path, "r") as f:
                labels = np.asarray(f[HDF5_LABELS_KEY][:], dtype=np.int64)  # type: ignore
            
            # Count LEAK and NOLEAK files
            leak_count = (labels == leak_idx).sum()
            noleak_count = len(labels) - leak_count
            
            # Convert to segment counts
            segs_per_file = ds_tr.num_long * ds_tr.num_short
            leak_segs = leak_count * segs_per_file
            noleak_segs = noleak_count * segs_per_file
            
            # Compute inverse frequency weights for 2 classes [NOLEAK, LEAK]
            total_segs = leak_segs + noleak_segs
            counts_binary = np.array([noleak_segs, leak_segs], dtype=np.float64)
            counts_binary = np.maximum(counts_binary, 1)
            inv = (total_segs / counts_binary)
            inv = inv / inv.mean()
            class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)
            logger.info("Binary class weights: NOLEAK=%.4f, LEAK=%.4f (from %d NOLEAK segs, %d LEAK segs)",
                       inv[0], inv[1], noleak_segs, leak_segs)
        else:
            # Multi-class mode: compute weights for all classes
            counts = class_counts_from_labels(ds_tr)
            counts = np.maximum(counts, 1)
            inv = (counts.sum() / counts).astype(np.float64)
            inv = inv / inv.mean()
            class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)
    
    if cfg.loss_type == "ce":
        cls_loss_fn = nn.CrossEntropyLoss()
    elif cfg.loss_type == "weighted_ce":
        cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    else:  # focal
        C = cfg.num_classes
        if cfg.binary_mode:
            # Binary mode: alpha for [NOLEAK, LEAK]
            alpha = torch.tensor([1.0 - cfg.focal_alpha_leak, cfg.focal_alpha_leak], 
                                device=device, dtype=torch.float32)
        else:
            # Multi-class mode: leak class gets focal_alpha_leak, others share remainder
            alpha = torch.ones(C, device=device, dtype=torch.float32) * ((1.0 - cfg.focal_alpha_leak) / (C - 1))
            alpha[leak_idx] = cfg.focal_alpha_leak
        cls_loss_fn = FocalLoss(alpha, gamma=cfg.focal_gamma)
    
    # Auxiliary loss (binary leak-vs-rest)
    leak_bce = None
    if cfg.use_leak_aux_head:
        if cfg.leak_aux_pos_weight is not None:
            bce_pos_weight = torch.tensor([cfg.leak_aux_pos_weight], device=device)
        else:
            # Derive from class frequency
            with h5py.File(ds_tr.h5_path, "r") as f:  # type: ignore[misc]
                labels = np.asarray(f[HDF5_LABELS_KEY][:], dtype=np.int64)  # type: ignore[index]
            pos = (labels == leak_idx).sum()
            neg = max(len(labels) - pos, 1)
            segs_per_file = ds_tr.num_long * ds_tr.num_short
            pos *= segs_per_file
            neg *= segs_per_file
            bce_pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        leak_bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
    
    return cls_loss_fn, leak_bce


def setup_augmentation(cfg: Config) -> Tuple[bool, Optional[nn.Module], Optional[nn.Module]]:
    """
    Setup SpecAugment transforms if requested.
    
    Args:
        cfg: Training configuration
    
    Returns:
        Tuple of (use_ta, time_mask, freq_mask)
    """
    use_ta = False
    time_mask = None
    freq_mask = None
    
    try:
        import torchaudio
        time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask_param) if cfg.use_specaugment else None
        freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask_param) if cfg.use_specaugment else None
        use_ta = True
    except Exception as e:
        if cfg.use_specaugment:
            logger.warning("%sSpecAugment enabled but torchaudio unavailable: %s - Training WITHOUT data augmentation!%s",
                          YELLOW, e, RESET)
    
    return use_ta, time_mask, freq_mask


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
# Core training loop components:
# - build_indices: Dataset sampling with optional oversampling
# - eval_split: Segment-level validation metrics
# - evaluate_file_level: Paper-exact file-level evaluation (primary metric)
# - train_one_epoch: Single epoch training with mixed precision
# - Checkpoint management: Save/load with RNG state preservation
# =============================================================================

def convert_labels_to_binary(labels: np.ndarray, leak_idx: int) -> np.ndarray:
    """
    Convert multi-class labels to binary LEAK/NOLEAK.
    
    Args:
        labels: Multi-class labels [N]
        leak_idx: Index of LEAK class
    
    Returns:
        Binary labels where 1=LEAK, 0=NOLEAK
    """
    return (labels == leak_idx).astype(np.int64)


def build_indices(ds: LeakMelDataset, leak_idx: int, oversample_factor: int) -> List[int]:
    """
    Build segment-level indices with optional LEAK class oversampling.
    
    Oversampling Strategy:
    ----------------------
    Default (oversample_factor=1): All segments appear once.
    
    Oversampling (oversample_factor>1):
    1. Identify all LEAK files from dataset labels
    2. Compute segment ranges for each LEAK file
    3. Duplicate LEAK segment indices (oversample_factor-1) times
    4. Concatenate: [all_segments] + [leak_segments] × (factor-1)
    
    Example:
    --------
    Dataset: 100 files (10 LEAK, 90 non-LEAK), 237 segments/file
    - oversample_factor=1: 23,700 segment indices
    - oversample_factor=3: 23,700 + 2×(10×237) = 28,440 indices
      → LEAK segments appear 3x, others appear 1x
    
    Impact:
    -------
    - Helps with class imbalance (more LEAK examples per epoch)
    - Increases effective dataset size without data augmentation
    - May cause overfitting if factor too high (monitor validation)
    
    Args:
    -----
    ds: Dataset with num_long, num_short attributes and HDF5 labels
    leak_idx: Index of LEAK class in class_names
    oversample_factor: Multiplier for LEAK segments (1=no oversampling)
    
    Returns:
    --------
    List of segment indices (may contain duplicates if oversampling)
    
    Performance:
    ------------
    - Time: O(num_files) to read labels + O(leak_files×segs_per_file) to build ranges
    - Memory: O(total_segments × oversample_factor) for index list
    - Typical: <100ms for 40K files, <500MB memory
    """
    N = len(ds)
    idx = np.arange(N, dtype=np.int64)
    if oversample_factor <= 1:
        return idx.tolist()
    # expand leak indices
    LxS = ds.num_long * ds.num_short
    with h5py.File(ds.h5_path, "r") as f:  # type: ignore[misc]
        labels = np.asarray(f[HDF5_LABELS_KEY][:], dtype=np.int64)  # type: ignore[index]
    leak_file_mask = (labels == leak_idx)
    leak_file_ids = np.nonzero(leak_file_mask)[0]
    leak_seg_ranges = [np.arange(fid*LxS, (fid+1)*LxS, dtype=np.int64) for fid in leak_file_ids]
    leak_seg_ids = np.concatenate(leak_seg_ranges) if leak_seg_ranges else np.empty(0, dtype=np.int64)
    expanded = [idx]
    for _ in range(oversample_factor - 1):
        expanded.append(leak_seg_ids.copy())
    return np.concatenate(expanded).tolist()


def eval_split(model: nn.Module,
               loader: DataLoader,  # type: ignore[type-arg]
               device: torch.device,
               leak_idx: int,
               use_channels_last: bool) -> Dict[str, float]:
    """Evaluate model on segment-level data.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        device: Device for computation
        leak_idx: Index of leak class
        use_channels_last: Whether to use channels_last memory format
    
    Returns:
        Dictionary with loss, accuracy, and leak metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct_count = torch.tensor(0, device=device, dtype=torch.int64)  # Accumulate on GPU
    leak_scores = []
    leak_targets = []
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for mel_batch, labels in loader:
            # Use helper function for consistent mel preparation
            mel_batch = prepare_mel_batch(mel_batch, device, use_channels_last)
            labels = labels.to(device, non_blocking=True)
            logits, leak_logit = model(mel_batch)
            loss = criterion(logits, labels)
            total_loss += loss.item()  # Already float, no need for float()
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            correct_count += (preds == labels).sum()  # Keep on GPU
            # collect leak scores
            rel = (labels == leak_idx).to(torch.float32)
            leak_targets.append(rel.detach().cpu().numpy())
            leak_scores.append(torch.sigmoid(leak_logit).detach().cpu().numpy())
    
    # Single CPU↔GPU sync at end (optimization)
    correct = int(correct_count.item())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    leak_targets = np.concatenate(leak_targets) if leak_scores else np.array([])
    leak_scores  = np.concatenate(leak_scores) if leak_scores else np.array([])
    out = {"loss": avg_loss, "acc": acc}
    # Note: Segment-level leak metrics not used (file-level metrics are primary)
    # Kept for monitoring purposes only
    if leak_scores.size:
        # Compute leak metrics at fixed 0.5 threshold (consistent with file-level)
        preds = (leak_scores >= 0.5).astype(np.int32)
        tp = int(((preds == 1) & (leak_targets == 1)).sum())
        fp = int(((preds == 1) & (leak_targets == 0)).sum())
        fn = int(((preds == 0) & (leak_targets == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p + r, 1e-12)
        out.update({"leak_f1": f1, "leak_p": p, "leak_r": r})
    return out


def evaluate_file_level(
    model: nn.Module,
    ds,  # LeakMelDataset or BinaryLabelDataset
    device: torch.device,
    dataset_leak_idx: int,
    model_leak_idx: int,
    use_channels_last: bool = True,
    threshold: float = 0.5,
    batch_long_segments: int = 0,  # 0 = process entire file at once (best GPU util)
) -> Tuple[float, Dict[str, float]]:
    """
    Paper-exact file-level evaluation with 50% voting threshold.
    
    Algorithm:
    ----------
    1. For each FILE:
       a. For each LONG segment (batched for GPU efficiency):
          - Process all SHORT segments within that long segment
          - Compute leak probability per short segment:
            p_short = 0.5 * softmax(logits)[model_leak_idx] + 0.5 * sigmoid(leak_logit)
          - Average probabilities across all short segments → p_long
       b. Count how many long segments have p_long ≥ threshold
       c. FILE is classified as LEAK if ≥50% of long segments exceed threshold
    
    2. Compute file-level accuracy, precision, recall, F1 for leak class
    
    Args:
        model: Trained model with dual heads (logits, leak_logit)
        ds: Dataset with file-level structure
        device: CUDA device
        dataset_leak_idx: Index of LEAK class in original dataset labels
        model_leak_idx: Index of LEAK class in model output (1 for binary, dataset_leak_idx for multi-class)
        use_channels_last: Use channels_last memory format
        threshold: Probability threshold for long segment (default: 0.5)
        batch_long_segments: Number of long segments to batch (0=entire file, best GPU util)
    
    Returns:
        Tuple of (file_accuracy, metrics_dict) where metrics_dict contains:
        - acc: File-level accuracy
        - leak_f1: Leak F1 score at file level
        - leak_p: Leak precision at file level
        - leak_r: Leak recall at file level
        - pred_leaks: Number of files predicted as leak
    
    Note:
        This implements the exact algorithm described in the research paper.
        Uses tqdm progress bar for visual feedback during evaluation.
        Batches multiple long segments together for better GPU utilization.
    """
    # Validate threshold
    if not (0 < threshold < 1):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    
    model.eval()
    
    # Ensure dataset file handle is open
    if getattr(ds, "h5f", None) is None:
        ds._ensure_open()
    
    correct = 0
    total = 0
    pred_leaks = 0
    true_leak_count = 0
    true_positives = 0  # Predicted leak AND actually leak
    false_positives = 0  # Predicted leak BUT not leak
    false_negatives = 0  # Predicted not leak BUT actually leak
    
    # Initialize monitoring
    status = EvalStatus(ds.num_files)
    status.start()
    
    # Performance profiling
    eval_start = time.perf_counter()
    forward_times = []
    
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            pbar = tqdm(
                range(ds.num_files),
                desc="[File-Level Eval]",
                unit="file",
                leave=True,
                position=0,
                dynamic_ncols=True,
                mininterval=0.2,
                file=sys.stdout,
            )
            
            for fidx in pbar:
                # Load entire file block: [num_long, num_short, (C,) n_mels, t_frames]
                blk = ds._segs[fidx]  # type: ignore[index]
                true_label = int(ds._labels[fidx])  # type: ignore[index,arg-type]
                true_is_leak = 1 if true_label == dataset_leak_idx else 0
                true_leak_count += true_is_leak
                
                # Process each long segment (pre-allocate numpy array for efficiency)
                probs_long = np.zeros(ds.num_long, dtype=np.float32)
                
                # Batch process long segments for better GPU utilization
                # If batch_long_segments == 0, process entire file at once for maximum GPU efficiency
                effective_batch_size = ds.num_long if batch_long_segments == 0 else batch_long_segments
                
                for batch_start in range(0, ds.num_long, effective_batch_size):
                    batch_end = min(batch_start + effective_batch_size, ds.num_long)
                    batch_size = batch_end - batch_start
                    
                    # Collect all short segments for this batch of long segments
                    mel_batch_list = []
                    for li in range(batch_start, batch_end):
                        # Get all short segments within this long segment
                        mel = blk[li]  # type: ignore[index] # [num_short, (C,) n_mels, t_frames]
                        
                        # Handle channel dimension if present
                        if ds._has_channel:
                            mel = mel[:, 0]  # type: ignore[index] # Take first channel: [num_short, n_mels, t_frames]
                        
                        mel_batch_list.append(mel)
                    
                    # Stack into single batch: [batch_size * num_short, n_mels, t_frames]
                    mel_batch_np = np.concatenate(mel_batch_list, axis=0)  # type: ignore[arg-type]
                    
                    # Prepare batch for model input
                    mel_t = prepare_mel_batch(
                        mel_batch_np,  # type: ignore[arg-type]
                        device,
                        use_channels_last,
                        dtype=torch.float16
                    )
                    
                    # Forward pass for entire batch
                    fwd_start = time.perf_counter()
                    logits, leak_logit = model(mel_t)  # logits: [batch_size * num_short, n_classes]
                    forward_times.append(time.perf_counter() - fwd_start)
                    
                    # Compute combined leak probability
                    p_cls = torch.softmax(logits, dim=1)[:, model_leak_idx]  # [batch_size * num_short]
                    p_aux = torch.sigmoid(leak_logit)  # [batch_size * num_short]
                    p_combined = 0.5 * (p_cls + p_aux)  # [batch_size * num_short]
                    
                    # Reshape to [batch_size, num_short] and average across short segments
                    p_combined = p_combined.view(batch_size, ds.num_short)
                    p_long_batch = p_combined.mean(dim=1)  # [batch_size]
                    
                    # Store results for each long segment in this batch
                    for i, li in enumerate(range(batch_start, batch_end)):
                        probs_long[li] = float(p_long_batch[i].item())
                
                # File-level decision: ≥50% of long segments must exceed threshold
                # Use ceiling_half helper for proper 50% threshold (handles odd num_long)
                num_long_above_threshold = sum(1 for p in probs_long if p >= threshold)
                vote_threshold = ceiling_half(ds.num_long)
                pred_is_leak = 1 if num_long_above_threshold >= vote_threshold else 0
                
                # Update metrics
                if pred_is_leak == true_is_leak:
                    correct += 1
                
                if pred_is_leak == 1:
                    pred_leaks += 1
                    if true_is_leak == 1:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if true_is_leak == 1:
                        false_negatives += 1
                
                total += 1
                
                # Update status
                status.update(done=total, correct=correct, pred_leaks=pred_leaks)
                
                # Update progress bar
                pbar.set_postfix({
                    "acc": f"{(correct / max(total, 1)):.4f}",
                    "leak%": f"{(100.0 * pred_leaks / max(total, 1)):.1f}",
                    "files": f"{total}/{ds.num_files}",
                })
                pbar.refresh()
            
            pbar.close()
    finally:
        # Ensure NVML cleanup even if evaluation fails
        status.stop()
    
    # Compute final metrics
    file_acc = correct / max(total, 1)
    
    # Leak-specific metrics
    leak_precision = true_positives / max(true_positives + false_positives, 1)
    leak_recall = true_positives / max(true_positives + false_negatives, 1)
    leak_f1 = 2 * leak_precision * leak_recall / max(leak_precision + leak_recall, 1e-12)
    
    # Performance report
    eval_elapsed = time.perf_counter() - eval_start
    if forward_times:
        avg_fwd = sum(forward_times) / len(forward_times)
        total_fwd = sum(forward_times)
        overhead = eval_elapsed - total_fwd
        logger.debug("%s[EVAL PROFILE] Total: %.2fs | Forward: %.2fs (%.1f%%) | Overhead: %.2fs (%.1f%%)%s",
                    CYAN, eval_elapsed, total_fwd, 100*total_fwd/eval_elapsed, 
                    overhead, 100*overhead/eval_elapsed, RESET)
        effective_batch = ds.num_long if batch_long_segments == 0 else batch_long_segments
        logger.debug("%s[EVAL PROFILE] Avg forward: %.3fs | Batch size: %d long segs | Files/s: %.1f%s",
                    CYAN, avg_fwd, effective_batch, total/eval_elapsed, RESET)
    
    metrics = {
        "acc": file_acc,
        "leak_f1": leak_f1,
        "leak_p": leak_precision,
        "leak_r": leak_recall,
        "pred_leaks": pred_leaks,
        "true_leaks": true_leak_count,
    }
    
    return file_acc, metrics


def train_one_epoch(
    epoch: int,
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cls_loss_fn: nn.Module,
    leak_bce: Optional[nn.Module],
    cfg: Config,
    leak_idx: int,
    model_leak_idx: int,
    device: torch.device,
    use_ta: bool,
    time_mask: Optional[nn.Module],
    freq_mask: Optional[nn.Module],
    interrupted: Dict[str, bool]
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (train_loss, train_acc)
    """
    model.train()  # type: ignore[attr-defined]
    running_loss = 0.0
    correct_count = torch.tensor(0, device=device, dtype=torch.int64)  # Accumulate on GPU
    seen = 0
    
    # System monitoring
    sys_monitor = SystemMonitor(device, enabled=True)
    
    # Performance profiling
    profiler = GPUProfiler(device, enabled=cfg.profile_gpu_util) if cfg.profile_performance else None
    if profiler:
        profiler.start()
    
    pbar = tqdm(total=len(train_loader), desc=f"[Train] Epoch {epoch}/{cfg.epochs}", unit="batch")
    batch_times = []
    dataloader_times = []
    prev_batch_end = time.perf_counter()
    
    # Sample system stats every N batches
    report_interval = max(1, len(train_loader) // 10)  # 10 reports per epoch
    
    for batch_idx, (mel_batch, labels) in enumerate(train_loader):
        # Measure DataLoader iterator time (CPU→GPU data pipeline)
        iter_time = time.perf_counter() - prev_batch_end
        if batch_idx > 0:  # Skip first batch (includes warmup)
            dataloader_times.append(iter_time)
        
        batch_start = time.perf_counter()
        if interrupted["flag"]:
            break
        
        # Prepare inputs
        mel_batch = prepare_mel_batch(mel_batch, device, cfg.use_channels_last)
        labels = labels.to(device, non_blocking=True)
        
        if profiler and batch_idx % 50 == 0:
            profiler.sample(f"batch_{batch_idx}")
        
        # Optional SpecAugment
        if cfg.use_specaugment and use_ta:
            with torch.no_grad():
                if time_mask: mel_batch = time_mask(mel_batch)
                if freq_mask: mel_batch = freq_mask(mel_batch)
        
        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits, leak_logit = model(mel_batch)
            loss_cls = cls_loss_fn(logits, labels)
            
            if cfg.use_leak_aux_head:
                # Auxiliary head: Binary LEAK detection across both modes
                # 
                # Binary mode (n_classes=2):
                #   - labels: 0 (NOLEAK) or 1 (LEAK) from BinaryLabelDataset
                #   - model_leak_idx: 1 (LEAK class in binary output)
                #   - leak_target: 1.0 if label==1, else 0.0
                #
                # Multi-class mode (n_classes=5):
                #   - labels: 0-4 (BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED)
                #   - model_leak_idx: 2 (LEAK class in original labels)
                #   - leak_target: 1.0 if label==2, else 0.0
                #
                # The auxiliary head learns to detect LEAK regardless of mode,
                # providing additional gradient signal during training and
                # averaging with the main classifier during inference.
                leak_target = (labels == model_leak_idx).to(torch.float32)
                loss_leak = leak_bce(leak_logit, leak_target)  # type: ignore[misc]
                loss = loss_cls + cfg.leak_aux_weight * loss_leak
            else:
                loss = loss_cls
        
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)  # type: ignore[attr-defined]
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics (accumulate on GPU to reduce sync overhead)
        bs = labels.size(0)
        running_loss += loss.item() * bs  # Already scalar, no float() needed
        preds = logits.argmax(dim=1)
        correct_count += (preds == labels).sum()  # Keep on GPU
        seen += bs
        
        batch_elapsed = time.perf_counter() - batch_start
        batch_times.append(batch_elapsed)
        prev_batch_end = time.perf_counter()  # For next iteration's DataLoader timing
        
        # Update progress bar with system stats periodically
        if batch_idx % report_interval == 0 or batch_idx == len(train_loader) - 1:
            stats_str = sys_monitor.format_stats()
            pbar.set_postfix_str(stats_str)
        
        pbar.update(1)
    
    pbar.close()
    
    # Single CPU↔GPU sync at end (optimization)
    correct = int(correct_count.item())
    train_loss = running_loss / max(seen, 1)
    train_acc = correct / max(seen, 1)
    
    # Collect final system stats for epoch summary
    final_stats = sys_monitor.get_stats()
    
    # Performance report
    if profiler and cfg.profile_performance:
        logger.debug("%s[TRAIN PROFILE] %s%s", CYAN, profiler.report(), RESET)
        if batch_times:
            avg_batch = sum(batch_times) / len(batch_times)
            logger.debug("%s[TRAIN PROFILE] Avg batch time: %.3fs (%.1f samples/s)%s",
                        CYAN, avg_batch, seen / sum(batch_times), RESET)
        
        # DataLoader timing diagnostics (identify CPU bottlenecks)
        if dataloader_times:
            avg_iter = sum(dataloader_times) / len(dataloader_times)
            max_iter = max(dataloader_times)
            logger.debug("%s[TRAIN PROFILE] DataLoader avg: %.3fms, max: %.3fms%s",
                        CYAN, avg_iter * 1000, max_iter * 1000, RESET)
            if avg_iter > 0.050:  # > 50ms is a bottleneck
                logger.warning("%s⚠️  DataLoader is slow (%.1fms avg)! Consider increasing prefetch_factor or num_workers%s",
                              YELLOW, avg_iter * 1000, RESET)
    
    # Print epoch summary with resource utilization
    logger.info("%s[EPOCH %d SUMMARY] Loss: %.4f | Acc: %.2f%% | %s%s",
               CYAN, epoch, train_loss, train_acc * 100, sys_monitor.format_stats(), RESET)
    
    # Alert if GPU utilization is low (bottleneck detection)
    if 'gpu_util' in final_stats and final_stats['gpu_util'] >= 0 and final_stats['gpu_util'] < 30:
        logger.warning("%s⚠️  LOW GPU UTILIZATION (%.1f%%)! Possible bottlenecks:%s", 
                      YELLOW, final_stats['gpu_util'], RESET)
        logger.warning("%s   - Increase batch_size (current: %d)%s", YELLOW, cfg.batch_size, RESET)
        logger.warning("%s   - Increase num_workers (current: %d)%s", YELLOW, cfg.num_workers, RESET)
        logger.warning("%s   - Increase prefetch_factor (current: %d)%s", YELLOW, cfg.prefetch_factor, RESET)
        logger.warning("%s   - Check DataLoader timing above%s", YELLOW, RESET)
    
    return train_loss, train_acc


def save_checkpoint(
    cfg: Config,
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    train_sampler: StatefulSampler,
    best_leak_f1: float,
    best_leak_thr: float,
    best_epoch: int,
    class_names: List[str],
    leak_idx: int
) -> bool:
    """
    Save training checkpoint.
    
    Returns:
        True if successful, False otherwise
    """
    ckpt = {
        "epoch": epoch + 1,
        "model": model.state_dict(),  # type: ignore[attr-defined]
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": get_rng_state_safe(),
        "train_sampler_state": train_sampler.state_dict(),
        "best_leak_f1": best_leak_f1,
        "best_leak_thr": best_leak_thr,
        "best_epoch": best_epoch,
        "config": asdict(cfg),
        "class_names": class_names,
        "leak_idx": leak_idx,
    }
    
    try:
        torch.save(ckpt, cfg.checkpoints_dir / CHECKPOINT_LAST)
        torch.save(ckpt, cfg.checkpoints_dir / f"{CHECKPOINT_EPOCH_PREFIX}{epoch:03d}.pth")
        rotate_checkpoints(cfg.checkpoints_dir, keep_last_k=cfg.keep_last_k)
        return True
    except Exception as e:
        logger.error("%sFailed to save checkpoint: %s%s", RED, e, RESET)
        logger.error("%sTraining will continue but progress may be lost if interrupted!%s", RED, RESET)
        return False


def save_best_model(
    cfg: Config,
    model,
    class_names: List[str],
    original_class_names: List[str],
    leak_idx: int,
    best_leak_f1: float,
    best_leak_thr: float,
    best_epoch: int,
    val_file_acc: float,
    train_ds: LeakMelDataset
):
    """
    Save best model weights and metadata.
    """
    torch.save(model.state_dict(), cfg.model_dir / CHECKPOINT_BEST)
    
    meta = {
        "class_names": class_names,
        "original_class_names": original_class_names,
        "binary_mode": cfg.binary_mode,
        "num_classes": cfg.num_classes,
        "leak_class_name": cfg.leak_class_name,
        "leak_idx": leak_idx,
        "best_leak_threshold": best_leak_thr,
        "best_file_leak_f1": best_leak_f1,
        "best_file_acc": val_file_acc,
        "best_epoch": best_epoch,
        "eval_method": "paper_exact_file_level_50pct_voting",
        "model_type": "LeakCNNMulti",
        "builder_cfg": train_ds.builder_cfg,
        "trainer_cfg": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool))},
    }
    with open(cfg.model_dir / MODEL_METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def validate_config(cfg: Config):
    """
    Validate all critical configuration parameters.
    """
    if cfg.early_stop_patience <= 0:
        raise ValueError(f"early_stop_patience must be > 0, got {cfg.early_stop_patience}")
    if cfg.batch_size <= 0 or cfg.val_batch_size <= 0:
        raise ValueError(f"Batch sizes must be > 0, got batch_size={cfg.batch_size}, val_batch_size={cfg.val_batch_size}")
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.epochs}")
    if cfg.binary_mode and cfg.num_classes != 2:
        logger.warning("%sbinary_mode=True but num_classes=%d, will be overridden to 2%s", YELLOW, cfg.num_classes, RESET)
    if cfg.num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {cfg.num_classes}")
    if cfg.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {cfg.learning_rate}")
    if not (0 <= cfg.dropout < 1):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")
    if not (0 < cfg.focal_alpha_leak <= 1):
        raise ValueError(f"focal_alpha_leak must be in (0, 1], got {cfg.focal_alpha_leak}")
    if cfg.loss_type == "focal" and not (0 < cfg.focal_gamma <= 5):
        raise ValueError(f"focal_gamma should be in (0, 5] for stable training, got {cfg.focal_gamma}")
    if cfg.keep_last_k < 1:
        raise ValueError(f"keep_last_k must be >= 1, got {cfg.keep_last_k}")
    if cfg.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}")
    if cfg.num_workers > 0 and cfg.prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1 when num_workers > 0, got {cfg.prefetch_factor}")
    if cfg.leak_oversample_factor < 1:
        raise ValueError(f"leak_oversample_factor must be >= 1, got {cfg.leak_oversample_factor}")
    if cfg.use_leak_aux_head and cfg.leak_aux_weight < 0:
        raise ValueError(f"leak_aux_weight must be >= 0, got {cfg.leak_aux_weight}")
    if cfg.use_specaugment and (cfg.time_mask_param < 0 or cfg.freq_mask_param < 0):
        raise ValueError(f"Augmentation params must be >= 0, got time_mask_param={cfg.time_mask_param}, freq_mask_param={cfg.freq_mask_param}")
    
    # Validate dataset files exist
    if not cfg.train_h5.exists():
        raise FileNotFoundError(f"Training dataset not found: {cfg.train_h5}")
    if not cfg.val_h5.exists():
        raise FileNotFoundError(f"Validation dataset not found: {cfg.val_h5}")


def setup_datasets(cfg: Config) -> Tuple[LeakMelDataset, LeakMelDataset, List[str], List[str], int, List[int], Subset]:
    """
    Load HDF5 datasets and configure for binary or multi-class training.
    
    This function handles the critical setup of dataset structure that differs
    between binary and multi-class modes:
    
    Binary Mode (cfg.binary_mode=True):
    ------------------------------------
    - Datasets remain in original multi-class format (HDF5 unchanged)
    - class_names set to ["NOLEAK", "LEAK"] for model output
    - cfg.num_classes forced to 2
    - leak_idx keeps original value (e.g., 2 for 5-class datasets)
    - BinaryLabelDataset wrapper applied later to convert labels 0/1
    - File-level labels stay original (needed for evaluation)
    
    Multi-class Mode (cfg.binary_mode=False):
    ------------------------------------------
    - Datasets used as-is
    - class_names from HDF5 metadata
    - cfg.num_classes matches dataset
    - leak_idx identifies LEAK class in original labels
    
    The key insight: We don't modify the HDF5 files. The BinaryLabelDataset
    wrapper handles label conversion at training time, while file-level
    evaluation uses original labels for ground truth.
    
    Returns:
        Tuple of (train_ds, val_ds, class_names, original_class_names, 
                  leak_idx, train_indices, val_subset)
    """
    logger.info("="*80)
    logger.info("DATASET PREPARATION")
    logger.info("="*80)
    
    logger.info("Loading training dataset: %s", cfg.train_h5)
    train_ds = LeakMelDataset(cfg.train_h5, preload_to_ram=cfg.preload_to_ram)
    logger.info("  Files: %d, Segments: %d (long=%d, short=%d per long)",
               train_ds.num_files, train_ds.total_segments, train_ds.num_long, train_ds.num_short)
    logger.info("  Mel shape: [%d, %d] (n_mels × t_frames)", train_ds.n_mels, train_ds.t_frames)
    
    logger.info("Loading validation dataset: %s", cfg.val_h5)
    val_ds = LeakMelDataset(cfg.val_h5, preload_to_ram=cfg.preload_to_ram)
    logger.info("  Files: %d, Segments: %d", val_ds.num_files, val_ds.total_segments)
    
    original_class_names = train_ds.class_names or [f"C{i}" for i in range(cfg.num_classes)]
    logger.info("Original class names: %s", original_class_names)
    
    try:
        leak_idx = original_class_names.index(cfg.leak_class_name)
        logger.info("%sLeak class '%s' at index %d%s", GREEN, cfg.leak_class_name, leak_idx, RESET)
    except ValueError:
        raise ValueError(
            f"Leak class '{cfg.leak_class_name}' not found in dataset classes: {original_class_names}"
        )
    
    # Configure class names and model output dimensionality based on mode
    if cfg.binary_mode:
        logger.info("%s[BINARY MODE] Converting to LEAK/NOLEAK classification%s", CYAN, RESET)
        class_names = ["NOLEAK", "LEAK"]
        cfg.num_classes = 2  # Force binary output
        # leak_idx stays original (e.g., 2) - needed for file-level eval
        # model_leak_idx will be set to 1 in train() function
    else:
        logger.info("%s[MULTI-CLASS MODE] Using all %d classes%s", CYAN, len(original_class_names), RESET)
        class_names = original_class_names
    
    logger.info("Model class names: %s", class_names)
    logger.info("Model will output %d classes", cfg.num_classes)

    logger.info("Building training indices (oversample_factor=%d)...", cfg.leak_oversample_factor)
    train_indices = build_indices(train_ds, leak_idx, cfg.leak_oversample_factor)
    logger.info("  Training indices: %d segments", len(train_indices))
    if cfg.leak_oversample_factor > 1:
        logger.info("  %sLEAK segments oversampled %dx%s", YELLOW, cfg.leak_oversample_factor, RESET)
    
    val_subset = Subset(val_ds, list(range(len(val_ds))))
    
    return train_ds, val_ds, class_names, original_class_names, leak_idx, train_indices, val_subset


def setup_dataloaders(cfg: Config, train_ds: Dataset, train_indices: List[int], val_subset: Subset, leak_idx: int, binary_mode: bool) -> Tuple[DataLoader, DataLoader, StatefulSampler]:
    """
    Create training and validation DataLoaders.
    
    Args:
        cfg: Configuration
        train_ds: Training dataset
        train_indices: Training indices
        val_subset: Validation subset
        leak_idx: Index of LEAK class in original labels
        binary_mode: Whether to use binary classification
    
    Returns:
        Tuple of (train_loader, val_loader, train_sampler)
    """
    logger.info("="*80)
    logger.info("DATALOADER CONFIGURATION")
    logger.info("="*80)
    
    # Wrap datasets for binary mode
    if binary_mode:
        logger.info("Wrapping datasets for binary LEAK/NOLEAK classification")
        train_ds_wrapped = BinaryLabelDataset(train_ds, leak_idx)
        # Wrap the underlying dataset in val_subset
        val_ds_base = val_subset.dataset
        val_ds_wrapped = BinaryLabelDataset(val_ds_base, leak_idx)
        val_subset = Subset(val_ds_wrapped, val_subset.indices)
        train_ds = train_ds_wrapped
    
    train_sampler = StatefulSampler(train_indices, shuffle=True, seed=cfg.seed)
    
    logger.info("Training DataLoader:")
    logger.info("  Batch size: %d", cfg.batch_size)
    logger.info("  Workers: %d", cfg.num_workers)
    logger.info("  Prefetch factor: %d", cfg.prefetch_factor if cfg.num_workers > 0 else 0)
    logger.info("  Pin memory: %s", cfg.pin_memory)
    logger.info("  Persistent workers: %s", cfg.persistent_workers and cfg.num_workers > 0)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=False,
    )
    
    logger.info("Validation DataLoader:")
    val_workers = max(1, cfg.num_workers // 2) if cfg.num_workers > 0 else 0
    logger.info("  Batch size: %d", cfg.val_batch_size)
    logger.info("  Workers: %d", val_workers)
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and val_workers > 0),
        prefetch_factor=max(2, cfg.prefetch_factor - 1),
        drop_last=False,
    )
    
    return train_loader, val_loader, train_sampler


def setup_training_components(cfg: Config, train_ds: LeakMelDataset, leak_idx: int, device: torch.device):
    """
    Create model, optimizer, scheduler, and loss functions.
    
    Returns:
        Tuple of (model, optimizer, scheduler, scaler, cls_loss_fn, leak_bce, use_ta, time_mask, freq_mask)
    """
    logger.info("="*80)
    logger.info("MODEL & TRAINING SETUP")
    logger.info("="*80)
    
    logger.info("Initializing LeakCNNMulti (n_classes=%d, dropout=%.2f)...", cfg.num_classes, cfg.dropout)
    model = create_model(cfg, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Total parameters: %s", f"{total_params:,}")
    logger.info("  Trainable parameters: %s", f"{trainable_params:,}")
    
    logger.info("Loss configuration:")
    logger.info("  Primary loss: %s", cfg.loss_type)
    if cfg.loss_type == "focal":
        logger.info("  Focal gamma: %.2f", cfg.focal_gamma)
    logger.info("  Auxiliary leak head: %s", "Enabled" if cfg.use_leak_aux_head else "Disabled")
    if cfg.use_leak_aux_head:
        logger.info("  Auxiliary weight: %.2f", cfg.leak_aux_weight)
    
    cls_loss_fn, leak_bce = setup_loss_functions(cfg, train_ds, leak_idx, device)
    
    use_ta, time_mask, freq_mask = setup_augmentation(cfg)
    if cfg.use_specaugment and use_ta:
        logger.info("Data augmentation: SpecAugment (time_mask=%d, freq_mask=%d)",
                   cfg.time_mask_param, cfg.freq_mask_param)
    
    logger.info("Optimizer: AdamW (lr=%.4f)", cfg.learning_rate)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, fused=True)
        logger.debug("  Using fused AdamW optimizer")
    except Exception as e:
        logger.warning("%sFused AdamW failed (%s), using standard AdamW%s", YELLOW, e, RESET)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    logger.info("Scheduler: CosineAnnealingLR (T_max=%d, eta_min=1e-6)", cfg.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    logger.info("Mixed precision: FP16 with GradScaler")
    scaler = torch.amp.GradScaler('cuda')
    
    return model, optimizer, scheduler, scaler, cls_loss_fn, leak_bce, use_ta, time_mask, freq_mask


def resume_from_checkpoint(cfg: Config, model, optimizer, scheduler, scaler, train_sampler: StatefulSampler, device: torch.device) -> Tuple[int, float, float, int]:
    """
    Attempt to resume training from checkpoint.
    
    Returns:
        Tuple of (start_epoch, best_leak_f1, best_leak_thr, best_epoch)
    """
    start_epoch = 1
    best_leak_f1 = -1.0
    best_leak_thr = 0.55
    best_epoch = 0

    last_ckpt = cfg.checkpoints_dir / CHECKPOINT_LAST
    if cfg.auto_resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 1))
            set_rng_state_safe(ckpt.get("rng_state"))
            
            sstate = ckpt.get("train_sampler_state")
            if sstate:
                train_sampler.load_state_dict(sstate)
            
            best_leak_f1 = float(ckpt.get("best_leak_f1", best_leak_f1))
            best_leak_thr = float(ckpt.get("best_leak_thr", best_leak_thr))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            
            logger.info("%sRESUME: epoch=%d best_leak_f1=%.4f@%d thr=%.4f%s",
                       GREEN, start_epoch, best_leak_f1, best_epoch, best_leak_thr, RESET)
        except Exception as e:
            logger.warning("%sRESUME failed: %s — starting fresh%s", YELLOW, e, RESET)
    
    return start_epoch, best_leak_f1, best_leak_thr, best_epoch


def run_training_loop(cfg: Config, model, train_loader: DataLoader, val_loader: DataLoader,
                     optimizer, scheduler, scaler, train_sampler: StatefulSampler,
                     cls_loss_fn, leak_bce, leak_idx: int, model_leak_idx: int, device: torch.device,
                     use_ta: bool, time_mask, freq_mask, class_names: List[str],
                     original_class_names: List[str],
                     train_ds, val_ds,
                     start_epoch: int, best_leak_f1: float, best_leak_thr: float, best_epoch: int) -> Tuple[float, int]:
    """
    Execute main training loop with validation and checkpointing.
    
    Returns:
        Tuple of (best_leak_f1, best_epoch)
    """
    interrupted = {"flag": False}
    
    def _sig(_sig, _frame):  # noqa: ARG001
        interrupted["flag"] = True
        logger.info("%sCTRL-C detected: Will save checkpoint at end of epoch…%s", YELLOW, RESET)
    
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    no_improve = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.on_epoch_start(epoch)
        
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, scaler,
            cls_loss_fn, leak_bce, cfg, leak_idx, model_leak_idx, device,
            use_ta, time_mask, freq_mask, interrupted
        )

        metrics = eval_split(model, val_loader, device, leak_idx, cfg.use_channels_last)
        val_loss = metrics["loss"]
        val_acc = metrics["acc"]
        seg_leak_f1 = metrics.get("leak_f1", -1.0)
        
        val_file_acc, file_metrics = evaluate_file_level(
            model, val_ds, device, leak_idx, model_leak_idx, cfg.use_channels_last, threshold=0.5
        )
        file_leak_f1 = file_metrics["leak_f1"]
        file_leak_p = file_metrics["leak_p"]
        file_leak_r = file_metrics["leak_r"]
        
        logger.info("Epoch %03d │ train_loss=%.4f │ train_acc=%.4f │ val_loss=%.4f │ val_acc=%.4f",
                   epoch, train_loss, train_acc, val_loss, val_acc)
        logger.info("          │ seg_leak_f1=%.4f │ file_acc=%.4f │ file_leak_f1=%.4f (P=%.4f, R=%.4f)",
                   seg_leak_f1, val_file_acc, file_leak_f1, file_leak_p, file_leak_r)

        scheduler.step()
        
        save_checkpoint(
            cfg, epoch, model, optimizer, scheduler, scaler, train_sampler,
            best_leak_f1, best_leak_thr, best_epoch, class_names, leak_idx
        )
        
        improved = file_leak_f1 > best_leak_f1
        if improved:
            best_leak_f1 = file_leak_f1
            best_leak_thr = 0.5
            best_epoch = epoch
            save_best_model(
                cfg, model, class_names, original_class_names, leak_idx,
                best_leak_f1, best_leak_thr, best_epoch, val_file_acc, train_ds
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                logger.info("%sEarlyStop: No file-level LEAK F1 improvement for %d epochs%s",
                           YELLOW, cfg.early_stop_patience, RESET)
                logger.info("%sEarlyStop: Best: file_leak_f1=%.4f at epoch %d%s",
                           YELLOW, best_leak_f1, best_epoch, RESET)
                logger.info("%sEarlyStop: Current: file_leak_f1=%.4f, file_acc=%.4f%s",
                           YELLOW, file_leak_f1, val_file_acc, RESET)
                break
        
        if interrupted["flag"]:
            logger.info("%sINTERRUPTED: Checkpoint saved at epoch %d. Exiting training loop.%s",
                       YELLOW, epoch, RESET)
            break
    
    return best_leak_f1, best_epoch


def run_final_test(cfg: Config, model, leak_idx: int, model_leak_idx: int, device: torch.device):
    """
    Evaluate best model on test set if available.
    """
    if not cfg.test_h5.exists():
        return
        
    logger.info("="*80)
    logger.info("FINAL TEST EVALUATION (Paper-Exact File-Level Metrics)")
    logger.info("="*80)
    
    test_ds = LeakMelDataset(cfg.test_h5)
    state = torch.load(cfg.model_dir / CHECKPOINT_BEST, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    
    test_file_acc, test_file_metrics = evaluate_file_level(
        model, test_ds, device, leak_idx, model_leak_idx, cfg.use_channels_last, threshold=0.5
    )
    
    logger.info("%s[TEST RESULTS - File Level]%s", CYAN, RESET)
    logger.info("  File Accuracy: %.4f", test_file_acc)
    logger.info("  Leak F1:       %.4f", test_file_metrics['leak_f1'])
    logger.info("  Leak Precision: %.4f", test_file_metrics['leak_p'])
    logger.info("  Leak Recall:    %.4f", test_file_metrics['leak_r'])
    logger.info("  Files predicted as LEAK: %d/%d", test_file_metrics['pred_leaks'], test_ds.num_files)
    logger.info("  Actual LEAK files: %d/%d", test_file_metrics['true_leaks'], test_ds.num_files)
    logger.info("="*80)


def train(cfg=None):
    """
    Main training entry point for leak detection model.
    
    Orchestrates the complete training pipeline:
    1. Configuration & initialization
    2. Dataset preparation
    3. DataLoader setup
    4. Model & optimizer creation
    5. Checkpoint resume
    6. Training loop with validation
    7. Final test evaluation
    
    Args:
        cfg: Optional Config instance. If None, creates default Config().
    """
    if cfg is None:
        cfg = Config()
    
    logger.info("="*80)
    logger.info("%sMulti-Label Dataset Trainer v15 - Production Leak Detection Training%s", CYAN, RESET)
    logger.info("="*80)
    
    set_seed(cfg.seed)
    device = device_setup()
    
    if torch.cuda.is_available():
        logger.info("%sCUDA Device: %s%s", GREEN, torch.cuda.get_device_name(0), RESET)
        logger.info("CUDA Version: %s", torch.version.cuda)
        logger.info("PyTorch Version: %s", torch.__version__)
        mem_total = torch.cuda.get_device_properties(0).total_memory
        logger.info("GPU Memory: %s", bytes_human(mem_total))
    
    logger.info("Optimizations: cudnn.benchmark=%s, allow_tf32=%s, channels_last=%s",
               CUDNN_BENCHMARK, TF32_ENABLED, cfg.use_channels_last)
    logger.info("Mixed Precision: FP16 (GradScaler enabled)")
    logger.info("Model Compilation: %s", "Enabled" if cfg.use_compile else "Disabled")
    
    validate_config(cfg)
    
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Model directory: %s", cfg.model_dir)
    logger.info("Checkpoint directory: %s", cfg.checkpoints_dir)

    train_ds, val_ds, class_names, original_class_names, leak_idx, train_indices, val_subset = setup_datasets(cfg)
    
    # Determine model output index for LEAK class based on mode
    # Binary mode: model outputs [NOLEAK, LEAK], so LEAK is at index 1
    # Multi-class: model outputs original classes, LEAK at original leak_idx
    model_leak_idx = 1 if cfg.binary_mode else leak_idx
    
    # Setup DataLoaders with optional BinaryLabelDataset wrapping for training
    train_loader, val_loader, train_sampler = setup_dataloaders(cfg, train_ds, train_indices, val_subset, leak_idx, cfg.binary_mode)
    
    # Critical: Wrap val_ds for file-level evaluation in binary mode
    # The file-level eval accesses ds._labels directly from HDF5, which are
    # still multi-class (0-4). But it uses model_leak_idx=1 to extract LEAK
    # probabilities from the model's binary output. This wrapper ensures
    # attribute forwarding works correctly for HDF5 access.
    if cfg.binary_mode:
        val_ds = BinaryLabelDataset(val_ds, leak_idx)
    
    model, optimizer, scheduler, scaler, cls_loss_fn, leak_bce, use_ta, time_mask, freq_mask = setup_training_components(cfg, train_ds, leak_idx, device)
    start_epoch, best_leak_f1, best_leak_thr, best_epoch = resume_from_checkpoint(cfg, model, optimizer, scheduler, scaler, train_sampler, device)
    
    best_leak_f1, best_epoch = run_training_loop(
        cfg, model, train_loader, val_loader, optimizer, scheduler, scaler, train_sampler,
        cls_loss_fn, leak_bce, leak_idx, model_leak_idx, device, use_ta, time_mask, freq_mask,
        class_names, original_class_names, train_ds, val_ds, start_epoch, best_leak_f1, best_leak_thr, best_epoch
    )
    
    logger.info("%sTraining complete. Best file_leak_f1=%.4f @ epoch %d (paper-exact file-level metric)%s",
               GREEN, best_leak_f1, best_epoch, RESET)
    
    run_final_test(cfg, model, leak_idx, model_leak_idx, device)

if __name__ == "__main__":
    train()
