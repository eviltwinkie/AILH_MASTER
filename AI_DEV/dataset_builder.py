#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Label Dataset Builder v15 - HDF5 Dataset Construction
=============================================================

Production dataset builder for acoustic leak detection using two-stage temporal segmentation.
Processes WAV files from labeled directories and generates HDF5 datasets with pre-computed
mel spectrograms for efficient training.

Performance:
    - GPU-accelerated mel computation with triple-buffering
    - In-RAM HDF5 assembly with single sequential disk flush
    - Multi-threaded WAV loading with configurable parallelism
    - Automatic GPU batch sizing based on available VRAM

Architecture:
    Pipeline stages:
    1. File Discovery: Scan directory tree for WAV files, build global label mapping
    2. Prefetch (CPU): Multi-threaded WAV loading → resample → segment → HDF5 waveform dataset
    3. GPU Processing: Triple-buffered mel transforms with async H2D/D2H transfers
    4. Finalization: Single sequential HDF5 flush to disk with metadata

Key Features:
    - Two-stage temporal segmentation (long-term + short-term windows)
    - GPU-accelerated mel spectrogram computation (batch processing)
    - Multi-threaded WAV loading and resampling
    - HDF5 output with in-RAM assembly and efficient disk flushing
    - Stores builder config and labels as HDF5 attributes
    - Support for multiple label sets (2-class, 5-class)
    - Global label mapping across train/validation/test splits

Two-Stage Segmentation:
    Stage 1 (Long-term): Divide signal into long windows (1024 samples)
    Stage 2 (Short-term): Subdivide each long window into short segments (512 samples)
    
    Example: 10-second audio at 4096 Hz
    - Long window: 1024 samples → 40 long segments
    - Short window: 512 samples → 2 short segments per long
    - Total: 80 segments per file

HDF5 Dataset Structure:
    /segments_waveform - Shape: [files, num_long, num_short, short_window]
    /segments_mel      - Shape: [files, num_long, num_short, n_mels, time_frames]
    /labels            - Shape: [files] with integer class labels
    Attributes:
        - config_json: Builder configuration (sample_rate, n_fft, etc.)
        - cnn_config_json: CNN training configuration
        - labels_json: Class names list
        - label2id_json: Label to integer ID mapping
        - created_at_utc: ISO timestamp

Output Files:
    - TRAINING_DATASET.H5   (train split)
    - VALIDATION_DATASET.H5 (validation split)
    - TESTING_DATASET.H5    (test split)

Supported Label Sets:
    2-class: LEAK, NOLEAK
    5-class: BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED

Configuration:
    - cpu_max_workers: 4 (disk I/O threads)
    - disk_files_per_task: 1024 (files per batch)
    - files_per_gpu_batch: 512 (GPU batch size, auto-adjusted)
    - num_mega_buffers: 3 (triple-buffering)

Usage:
    python dataset_builder.py
    
    Environment variables:
    - BUILDER_LOG_LEVEL: Set to DEBUG for verbose logging (default: INFO)

Dependencies:
    - PyTorch 2.0+ with CUDA
    - TorchAudio (mel transforms)
    - h5py (HDF5 I/O)
    - soundfile (WAV loading)
    - pynvml (GPU monitoring)
    - psutil (CPU/RAM monitoring)

Author: AI Development Team
Version: 15.0
Last Updated: November 19, 2025
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import signal
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional, Any, Union

import h5py
import librosa
import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from tqdm import tqdm

# Optional GPU profiling
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    pynvml = None  # type: ignore
    _HAS_NVML = False

CYAN, GREEN, YELLOW, RED, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


# ======================================================================
# LOGGING SETUP
# ======================================================================

# Set BUILDER_LOG_LEVEL=DEBUG in your environment for deep diagnostics.
LOG_LEVEL = os.environ.get("BUILDER_LOG_LEVEL", "INFO").upper()

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
logger = logging.getLogger("dataset_builder")
logger.handlers.clear()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

handler = logging.StreamHandler()
handler.setFormatter(ElapsedTimeFormatter("[%(elapsed)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


# ======================================================================
# CONFIGURATION
# ======================================================================

@dataclass
class Config:
    """
    Global configuration for HDF5 dataset builder.
    
    This class centralizes all builder parameters including I/O settings,
    audio processing parameters, and CNN training configuration.
    
    Configuration Groups:
    --------------------
    Paths:
        - stage_dir: Output directory for HDF5 files
        - training_dir: Input directory with training WAV files
        - validation_dir: Input directory with validation WAV files
        - testing_dir: Input directory with test WAV files
    
    Audio Processing:
        - sample_rate: Audio sample rate in Hz (4096)
        - duration_sec: Audio clip duration in seconds (10)
        - long_window: Long segment window size in samples (1024)
        - short_window: Short segment window size in samples (512)
    
    Mel Spectrogram:
        - n_mels: Number of mel frequency bins (64)
        - n_fft: FFT window size (512)
        - hop_length: FFT hop length (128)
        - power: Power for spectrogram (1.0 = magnitude, 2.0 = power)
        - center: Whether to center FFT windows (False)
    
    Disk I/O:
        - cpu_max_workers: Number of parallel disk readers (4)
        - disk_files_per_task: Files per prefetch batch (1024)
        - disk_max_inflight: Maximum in-flight disk tasks (16)
        - disk_submit_window: Submit window for disk tasks (16)
    
    GPU Processing:
        - autosize_target_util_frac: Target VRAM utilization (0.80)
        - files_per_gpu_batch: Initial GPU batch size (512, auto-adjusted)
        - seg_microbatch_segments: Segments per GPU microbatch (8192)
        - max_files_per_gpu_batch: Maximum GPU batch size (4096)
        - num_mega_buffers: Number of GPU buffers for triple-buffering (3)
    
    CNN Training:
        - CNN_MODEL_TYPE: Model architecture ('mel')
        - CNN_BATCH_SIZE: Training batch size (5632)
        - CNN_LEARNING_RATE: Learning rate (0.001)
        - CNN_DROPOUT: Dropout rate (0.25)
        - CNN_EPOCHS: Number of training epochs (200)
        - CNN_FILTERS: Number of CNN filters (32)
        - CNN_KERNEL_SIZE: Convolution kernel size (3, 3)
        - CNN_POOL_SIZE: Pooling size (2, 2)
        - CNN_STRIDES: Stride size (2, 2)
        - CNN_DENSE: Dense layer units (128)
    
    Miscellaneous:
        - track_times: Enable HDF5 timestamp tracking (False)
        - warn_ram_fraction: RAM usage warning threshold (0.70)
    
    Computed Properties:
        - num_samples: Total samples per file
        - num_long: Number of long segments per file
        - num_short: Number of short segments per long segment
        - segments_per_file: Total segments per file
        - db_mult: dB scale multiplier based on power setting
    
    Example:
        >>> cfg = Config()
        >>> cfg.num_samples
        40960
        >>> cfg.segments_per_file
        80
    """

    # Add parent directory to sys.path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from global_config import MASTER_DATASET, DATASET_TRAINING, DATASET_VALIDATION, DATASET_TESTING, N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE, SAMPLE_DURATION, LONG_WINDOW, SHORT_WINDOW, N_POWER

    # Roots
    stage_dir: Path = Path(MASTER_DATASET)
    training_dir: Path = Path(DATASET_TRAINING)
    validation_dir: Path = Path(DATASET_VALIDATION)
    testing_dir: Path = Path(DATASET_TESTING)

    # Output HDF5 names
    training_hdf5: str = "TRAINING_DATASET.H5"
    validation_hdf5: str = "VALIDATION_DATASET.H5"
    testing_hdf5: str = "TESTING_DATASET.H5"

    # Signal params
    sample_rate: int = SAMPLE_RATE
    duration_sec: int = SAMPLE_DURATION

    long_window: int = LONG_WINDOW
    short_window: int = SHORT_WINDOW

    # ---- CNN config (requested) ----
    CNN_MODEL_TYPE: str = "mel"
    CNN_BATCH_SIZE: int = 5632
    CNN_LEARNING_RATE: float = 0.001
    CNN_DROPOUT: float = 0.25
    CNN_EPOCHS: int = 200
    CNN_FILTERS: int = 32
    CNN_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    CNN_POOL_SIZE: Tuple[int, int] = (2, 2)
    CNN_STRIDES: Tuple[int, int] = (2, 2)
    CNN_DENSE: int = 128

    # Mel params (aligned with trainer)
    n_mels: int = N_MELS
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    power: float = N_POWER
    center: bool = False

    # Disk I/O (optimized for NVMe SSD sequential reads)
    # Balance: Larger batches reduce overhead, moderate parallelism avoids context switching
    cpu_max_workers: int = 20          # Sweet spot: enough parallelism without excessive overhead
    disk_files_per_task: int = 2048   # Larger batches for better sequential read efficiency
    disk_max_inflight: int = 0       # Moderate queue depth for pipeline overlap
    disk_submit_window: int = 20      # Moderate submit window

    # Performance optimization strategies
    use_ram_preload: bool = True      # Load all WAVs to RAM before GPU processing (fastest)
    use_memmap_cache: bool = False    # Use memory-mapped cache file (good for large datasets)
    use_async_pipeline: bool = True   # Separate disk I/O and GPU processing threads (best overlap)
    memmap_cache_dir: Path = Path("/tmp/dataset_cache")  # Directory for memmap cache files
    
    # Advanced GPU optimizations
    use_persistent_buffers: bool = True   # Reuse GPU buffers across splits (avoid realloc)
    use_precomputed_filterbanks: bool = True  # Cache mel filterbanks on GPU (faster transforms)
    use_cuda_graphs: bool = True          # Capture mel pipeline as CUDA graph (reduce overhead)

    # GPU mega-batching (optimized for high utilization)
    autosize_target_util_frac: float = 0.85  # Use more VRAM
    files_per_gpu_batch: int = 1024          # Larger initial batch
    seg_microbatch_segments: int = 16384     # 2x larger microbatches
    max_files_per_gpu_batch: int = 8192      # Allow 2x larger batches
    num_mega_buffers: int = 4                # Quad-buffering for better overlap

    # Misc
    track_times: bool = False
    warn_ram_fraction: float = 0.70

    @property
    def num_samples(self) -> int:
        return self.sample_rate * self.duration_sec

    @property
    def num_long(self) -> int:
        return self.num_samples // self.long_window

    @property
    def num_short(self) -> int:
        return self.long_window // self.short_window

    @property
    def segments_per_file(self) -> int:
        return self.num_long * self.num_short

    @property
    def db_mult(self) -> float:
        return 10.0 if self.power == 2.0 else 20.0


# ======================================================================
# UTILITIES
# ======================================================================

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


def ensure_cuda() -> torch.device:
    """
    Initialize CUDA device with performance optimizations.
    
    Enables cudnn.benchmark for auto-tuning convolution algorithms
    and TensorFloat-32 for faster computation on Ampere+ GPUs.
    
    Returns:
        torch.device: CUDA device object
    
    Raises:
        RuntimeError: If CUDA is not available
    
    Example:
        >>> device = ensure_cuda()
        >>> tensor = torch.randn(1024, device=device)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU-accelerated mel computation.")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("CUDA initialized with optimizations: cudnn.benchmark=True, allow_tf32=True")
    return dev


def build_mel_transform(cfg: Config, device: torch.device) -> torch.nn.Module:
    """
    Create GPU-accelerated mel spectrogram transform.
    
    Args:
        cfg: Configuration with audio parameters
        device: PyTorch CUDA device
    
    Returns:
        TorchAudio MelSpectrogram transform in eval mode on GPU
    
    Example:
        >>> device = torch.device('cuda')
        >>> mel_transform = build_mel_transform(cfg, device)
        >>> audio = torch.randn(512, device=device)
        >>> mel = mel_transform(audio)  # (n_mels, time_frames)
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        center=cfg.center,
    ).eval().to(device)


def estimate_image_bytes(cfg: Config, n_mels: int, t_frames: int, fp16_mel: bool = True) -> int:
    """
    Estimate memory footprint for single file in HDF5 dataset.
    
    Calculates total bytes needed for waveform segments, mel spectrograms,
    and labels for one file.
    
    Args:
        cfg: Configuration with segment parameters
        n_mels: Number of mel frequency bins
        t_frames: Time frames in mel spectrogram
        fp16_mel: Use float16 for mel (default: True)
    
    Returns:
        Total bytes per file
    
    Example:
        >>> estimate_image_bytes(cfg, n_mels=64, t_frames=252, fp16_mel=True)
        163856  # bytes per file
    """
    wave = cfg.segments_per_file * cfg.short_window * 4  # f32 waveform
    melb = cfg.segments_per_file * n_mels * t_frames * (2 if fp16_mel else 4)  # mel spectrogram
    labels = 2  # int16 label
    return wave + melb + labels


def load_wav_mono_fast(path: str) -> Tuple[np.ndarray, int]:
    """
    Fast WAV file loading with automatic mono conversion.
    
    Uses soundfile for efficient loading and handles multi-channel audio
    by averaging channels to mono.
    
    Args:
        path: Absolute path to WAV file
    
    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: Mono float32 array, contiguous in memory
        - sample_rate: Sample rate in Hz
    
    Channel Handling:
        - Mono: Use directly
        - Stereo: Average channels (L+R)/2
        - Multi-channel: Average all channels
    
    Example:
        >>> data, sr = load_wav_mono_fast('/path/to/audio.wav')
        >>> data.shape
        (40960,)
        >>> sr
        4096
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    ch = data.shape[1]
    if ch == 1:
        wav = data[:, 0]
    elif ch == 2:
        wav = (data[:, 0] + data[:, 1]) * 0.5
    else:
        wav = data.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def prefetch_wavs(batch: List[Tuple[int, Path, int]]) -> List[Tuple[int, Path, np.ndarray, int, int]]:
    """
    Load a batch of WAV files in parallel worker thread.
    
    Reads WAV files using soundfile, converts to mono, and returns audio data
    with metadata. Errors are logged but don't stop batch processing.
    
    Args:
        batch: List of (file_index, file_path, label_id) tuples
    
    Returns:
        List of (file_index, file_path, audio_data, sample_rate, label_id)
        Only successful reads are included in output.
    
    Error Handling:
        - Read failures are logged with yellow warning
        - Failed files are skipped, not included in return
        - Batch processing continues despite individual failures
    
    Example:
        >>> batch = [(0, Path('/data/file1.wav'), 1), (1, Path('/data/file2.wav'), 0)]
        >>> results = prefetch_wavs(batch)
        >>> len(results)  # May be less than input if files failed
        2
    """
    out: List[Tuple[int, Path, np.ndarray, int, int]] = []
    for idx, path, label_id in batch:
        try:
            data, sr = load_wav_mono_fast(str(path))
            out.append((idx, path, data, sr, label_id))
        except Exception as e:
            logger.warning("[SKIP-READ] %s → %s", path, e)
    return out


def autosize_gpu_batch(cfg: Config, free_vram_gb: float, t_frames: int,
                       target_util_frac: float = 0.70, num_buffers: int = 3,
                       max_bsz: int = 131072) -> int:
    """
    Automatically determine optimal GPU batch size based on available VRAM.
    
    Calculates maximum number of files that can be processed in one GPU batch
    while staying within VRAM limits. Uses conservative safety margins and
    prefers power-of-2 friendly sizes.
    
    Args:
        cfg: Configuration with segment and mel parameters
        free_vram_gb: Available VRAM in gigabytes
        t_frames: Time frames in mel spectrogram
        target_util_frac: Target VRAM utilization (0.70 = 70%)
        num_buffers: Number of GPU buffers for triple-buffering (3)
        max_bsz: Maximum batch size cap (131072)
    
    Returns:
        Optimal batch size (files per GPU batch)
        Returns power-of-2 friendly sizes: 512, 1024, 2048, 4096, etc.
        Minimum return value is 1.
    
    Algorithm:
        1. Apply safety overhead (10%) to account for PyTorch/CUDA internals
        2. Calculate per-file memory: waveform (FP32) + mel (FP16)
        3. Account for triple-buffering (num_buffers × per_file_bytes)
        4. Find largest power-of-2 friendly size that fits
    
    Example:
        >>> autosize_gpu_batch(cfg, free_vram_gb=20.0, t_frames=252)
        4096  # Can process 4096 files per batch
        >>> autosize_gpu_batch(cfg, free_vram_gb=2.0, t_frames=252)
        512  # Limited VRAM, smaller batch
    
    Note:
        Conservative sizing prevents OOM errors during mel computation.
        If OOM occurs, batch size is automatically halved and retried.
    """
    safety_overhead_frac = 0.1
    free_bytes = int(free_vram_gb * (1024 ** 3))
    usable = int(free_bytes * target_util_frac * (1.0 - safety_overhead_frac))
    if usable <= 0:
        return 1

    bytes_per_segment_in = cfg.short_window * 4  # FP32 waveform
    bytes_per_segment_out = cfg.n_mels * t_frames * 2  # FP16 mel
    per_file_bytes = cfg.segments_per_file * (bytes_per_segment_in + bytes_per_segment_out)
    if per_file_bytes <= 0:
        return 1

    max_files = usable // (num_buffers * per_file_bytes)
    # Prefer power-of-2 friendly sizes for GPU efficiency
    for cand in [131072, 65536, 32768, 16384, 8192, 4096, 3072, 2048, 1536, 1024, 768, 512]:
        if max_files >= cand:
            return min(cand, max_bsz)
    return max(1, min(max_bsz, int(max_files)))


# ======================================================================
# BUILDER CLASS
# ======================================================================

class MultiSplitBuilder:
    """
    Multi-split HDF5 dataset builder with GPU-accelerated mel computation.
    
    Orchestrates the complete pipeline from WAV file discovery through mel
    spectrogram computation to HDF5 dataset creation. Implements producer-consumer
    architecture with triple-buffered GPU processing.
    
    Architecture:
        1. Discovery: Scan directories, build global label mapping
        2. Prefetch (CPU): Multi-threaded WAV loading → segmentation
        3. GPU Processing: Triple-buffered mel transforms
        4. Storage: In-RAM HDF5 assembly → single disk flush
    
    Attributes:
        cfg (Config): Configuration object
        device (torch.device): CUDA device
        mel_transform (torch.nn.Module): GPU mel spectrogram transform
        stop_evt (threading.Event): Profiling stop signal
        profile_stats (List): Performance statistics
        nvml_handle: NVML GPU handle for monitoring
        prof_thread: Background profiling thread
        cpu_times (List[float]): Per-file CPU processing times
        gpu_times (List[float]): Per-batch GPU processing times
        file_indices (List[int]): Successfully processed file indices
    
    Example:
        >>> cfg = Config()
        >>> builder = MultiSplitBuilder(cfg)
        >>> builder.build_all()  # Process all splits
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = ensure_cuda()
        self.mel_transform = build_mel_transform(cfg, self.device)

        # profiling
        self.stop_evt = threading.Event()
        self.profile_stats: List[Tuple[float, float, float, Optional[float], Optional[float]]] = []
        self.nvml_handle = None
        self.prof_thread: Optional[threading.Thread] = None

        # per-split timings
        self.cpu_times: List[float] = []
        self.gpu_times: List[float] = []
        self.file_indices: List[int] = []
        
        # detailed performance tracking
        self.disk_read_times: List[float] = []
        self.cpu_segment_times: List[float] = []
        self.gpu_h2d_times: List[float] = []
        self.gpu_compute_times: List[float] = []
        self.gpu_d2h_times: List[float] = []
        self.gpu_batch_sizes: List[int] = []
        self.queue_depths: List[int] = []
        
        # Persistent GPU resources (reused across splits)
        self.persistent_buffers: Optional[Dict[str, Any]] = None
        self.mel_filterbank: Optional[torch.Tensor] = None
        self.cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_input: Optional[torch.Tensor] = None
        self.graph_output: Optional[torch.Tensor] = None

    # --------- profiling ---------
    def _profile_worker(self):
        while not self.stop_evt.wait(0.5):
            cpu_raw = psutil.cpu_percent(interval=None)
            cpu = float(cpu_raw) if not isinstance(cpu_raw, list) else 0.0
            ram = float(psutil.virtual_memory().percent)
            if _HAS_NVML and self.nvml_handle is not None:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu  # type: ignore
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)  # type: ignore
                    gpu = float(gpu_util)
                    vram = float(mem_info.used) / (1024 ** 3)
                except Exception:
                    gpu, vram = None, None
            else:
                gpu, vram = None, None
            self.profile_stats.append((time.time(), cpu, ram, gpu, vram))

    def _start_profiling(self):
        """Initialize NVML and start background profiling thread."""
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()  # type: ignore
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore
                logger.info("NVML initialized, using GPU 0")
            except Exception as e:
                logger.warning("NVML init failed: %s", e)
                self.nvml_handle = None
        t = threading.Thread(target=self._profile_worker, daemon=True)
        t.start()
        self.prof_thread = t
        logger.info("Profiling thread started")

    def _stop_profiling(self):
        self.stop_evt.set()
        if self.prof_thread is not None:
            try:
                self.prof_thread.join(timeout=2)
            except Exception:
                pass
        if _HAS_NVML and (self.nvml_handle is not None):
            try:
                pynvml.nvmlShutdown()  # type: ignore
            except Exception:
                pass
    # --------- discovery helpers ---------
    @staticmethod
    def _discover_records(root_dir: Path) -> List[Tuple[Path, str]]:
        """
        Recursively discover WAV files in directory tree.
        
        Scans subdirectories (label folders) and collects all .wav files.
        Skips the root directory itself to enforce label subfolder structure.
        
        Args:
            root_dir: Root directory to scan
        
        Returns:
            List of (file_path, label_name) tuples
        
        Example:
            >>> records = _discover_records(Path('/data/TRAINING'))
            >>> records[0]
            (Path('/data/TRAINING/LEAK/file1.wav'), 'LEAK')
        """
        records: List[Tuple[Path, str]] = []
        if not root_dir.exists():
            return records
        for root, _, files in os.walk(root_dir):
            # skip the root itself; we want label subfolders
            if os.path.abspath(root) == os.path.abspath(root_dir):
                continue
            label = os.path.basename(root)
            for f in files:
                if f.lower().endswith(".wav"):
                    records.append((Path(root) / f, label))
        return records

    def _preload_wavs_to_ram(self, records: List[Tuple[Path, str]], 
                             global_label2id: Dict[str, int]) -> Dict[int, np.ndarray]:
        """
        Preload all WAV files into RAM for maximum GPU throughput.
        
        This eliminates disk I/O bottleneck by loading everything once.
        Trade-off: Requires RAM (~5-10 GB for 50k files).
        
        Args:
            records: List of (file_path, label_name) tuples
            global_label2id: Global label mapping
        
        Returns:
            Dictionary mapping file_index -> preprocessed waveform array
        
        Performance:
            - One-time disk hit with parallel loading
            - Subsequent GPU processing is pure RAM → GPU → RAM
            - Eliminates 95%+ of pipeline time
        """
        cfg = self.cfg
        logger.info("🚀 RAM PRELOAD: Loading %d files to memory...", len(records))
        
        wav_cache: Dict[int, np.ndarray] = {}
        indexed = [(i, p, global_label2id[lbl]) for i, (p, lbl) in enumerate(records)]
        
        from concurrent.futures import as_completed
        
        def load_and_preprocess(idx: int, path: Path, label_id: int) -> Tuple[int, Optional[np.ndarray]]:
            try:
                # Load WAV
                data, sr = load_wav_mono_fast(str(path))
                
                # Resample if needed
                if sr != cfg.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=cfg.sample_rate)
                
                # Pad/trim to exact size
                if len(data) < cfg.num_samples:
                    data = np.pad(data, (0, cfg.num_samples - len(data)), mode="constant")
                elif len(data) > cfg.num_samples:
                    data = data[:cfg.num_samples]
                
                # PRE-COMPUTE 3D segmentation to eliminate CPU bottleneck later
                # This moves work from sequential processing to parallel preload phase
                data_3d = data.reshape(cfg.num_long, cfg.num_short, cfg.short_window).astype(np.float32)
                
                return idx, data_3d
            except Exception as e:
                logger.warning("[SKIP-PRELOAD] %s → %s", path, e)
                return idx, None
        
        # Parallel loading with progress bar
        with ThreadPoolExecutor(max_workers=cfg.cpu_max_workers * 2) as executor:
            futures = [executor.submit(load_and_preprocess, idx, path, lbl) 
                      for idx, path, lbl in indexed]
            
            pbar = tqdm(total=len(futures), desc=f"{CYAN}[RAM Preload]{RESET}", unit="file")
            for future in as_completed(futures):
                idx, data = future.result()
                if data is not None:
                    wav_cache[idx] = data
                pbar.update(1)
            pbar.close()
        
        total_gb = sum(arr.nbytes for arr in wav_cache.values()) / (1024**3)
        logger.info("✅ Preloaded %d/%d files to RAM (%.1f GB) - Pre-segmented 3D arrays", 
                   len(wav_cache), len(records), total_gb)
        
        return wav_cache

    def _create_memmap_cache(self, records: List[Tuple[Path, str]], 
                             global_label2id: Dict[str, int],
                             cache_path: Path) -> np.memmap:
        """
        Create memory-mapped cache file for zero-copy WAV access.
        
        One-time preprocessing that creates a large memmap file containing
        all preprocessed waveforms. Subsequent runs use zero-copy memory mapping.
        
        Args:
            records: List of (file_path, label_name) tuples
            global_label2id: Global label mapping
            cache_path: Path to memmap cache file
        
        Returns:
            Memory-mapped array [num_files, num_samples] dtype=float32
        
        Performance:
            - One-time cost: ~20-30s preprocessing
            - Subsequent runs: Zero-copy mmap, kernel handles caching
            - Good for datasets too large for RAM
        """
        cfg = self.cfg
        logger.info("📁 MEMMAP CACHE: Creating cache at %s", cache_path)
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create memmap array
        memmap_array = np.memmap(cache_path, dtype=np.float32, mode='w+',
                                shape=(len(records), cfg.num_samples))
        
        # Fill with preprocessed data
        indexed = [(i, p, global_label2id[lbl]) for i, (p, lbl) in enumerate(records)]
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_file(idx: int, path: Path, label_id: int) -> Tuple[int, Optional[np.ndarray]]:
            try:
                data, sr = load_wav_mono_fast(str(path))
                if sr != cfg.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=cfg.sample_rate)
                if len(data) < cfg.num_samples:
                    data = np.pad(data, (0, cfg.num_samples - len(data)), mode="constant")
                elif len(data) > cfg.num_samples:
                    data = data[:cfg.num_samples]
                return idx, data.astype(np.float32)
            except Exception as e:
                logger.warning("[SKIP-MEMMAP] %s → %s", path, e)
                return idx, None
        
        with ThreadPoolExecutor(max_workers=cfg.cpu_max_workers * 2) as executor:
            futures = [executor.submit(process_file, idx, path, lbl) 
                      for idx, path, lbl in indexed]
            
            pbar = tqdm(total=len(futures), desc=f"{CYAN}[Memmap Cache]{RESET}", unit="file")
            for future in as_completed(futures):
                idx, data = future.result()
                if data is not None:
                    memmap_array[idx] = data
                pbar.update(1)
            pbar.close()
        
        # Flush to disk
        memmap_array.flush()
        logger.info("✅ Memmap cache created: %s (%.1f GB)", 
                   cache_path, cache_path.stat().st_size / (1024**3))
        
        return memmap_array
    
    def _load_memmap_cache(self, cache_path: Path, num_files: int) -> np.memmap:
        """
        Load existing memory-mapped cache for zero-copy access.
        
        Args:
            cache_path: Path to memmap cache file
            num_files: Expected number of files
        
        Returns:
            Memory-mapped array [num_files, num_samples] dtype=float32
        """
        cfg = self.cfg
        logger.info("📂 Loading memmap cache from %s", cache_path)
        
        memmap_array = np.memmap(cache_path, dtype=np.float32, mode='r',
                                shape=(num_files, cfg.num_samples))
        
        logger.info("✅ Memmap cache loaded (%.1f GB, zero-copy)", 
                   cache_path.stat().st_size / (1024**3))
        
        return memmap_array

    def discover_global_labels(self, dirs: List[Path]) -> Tuple[List[Tuple[Path, str]], Dict[str, int]]:
        """
        Build global label mapping across all splits.
        
        Ensures consistent label IDs across training, validation, and test splits.
        All files are discovered and labels are sorted alphabetically for
        deterministic ID assignment.
        
        Args:
            dirs: List of directories to scan (train, val, test)
        
        Returns:
            Tuple of (all_records, label_to_id_mapping)
            - all_records: Combined list of all (path, label) tuples
            - label_to_id_mapping: Dict mapping label names to integer IDs
        
        Example:
            >>> records, label2id = builder.discover_global_labels([train_dir, val_dir])
            >>> label2id
            {'LEAK': 0, 'NOLEAK': 1}
        """
        all_records: List[Tuple[Path, str]] = []
        label_set = set()
        for d in dirs:
            recs = self._discover_records(d)
            all_records.extend(recs)
            label_set.update(lbl for _, lbl in recs)
        labels_sorted = sorted(label_set)
        label2id = {l: i for i, l in enumerate(labels_sorted)}
        logger.info("Discovered %d total files across %d labels: %s", 
                   len(all_records), len(label2id), list(label2id.keys()))
        return all_records, label2id

    # --------- single split build ---------
    def build_split(self, split_name: str, input_dir: Path, out_path: Path,
                    global_label2id: Dict[str, int], global_label_list: List[str]):
        cfg = self.cfg
        logger.info("\n=== Building split: %s ===", split_name)
        records = self._discover_records(input_dir)
        logger.info("Discovered %d WAV files in %s", len(records), input_dir)

        if len(records) == 0:
            logger.warning("[SKIP] No WAV files under %s", input_dir)
            return

        # reset per-split timings
        self.cpu_times = []
        self.gpu_times = []
        self.file_indices = []
        self.disk_read_times = []
        self.cpu_segment_times = []
        self.gpu_h2d_times = []
        self.gpu_compute_times = []
        self.gpu_d2h_times = []
        self.gpu_batch_sizes = []
        self.queue_depths = []

        # map labels using global mapping
        labels_np = np.array([global_label2id[lbl] for _, lbl in records], dtype=np.int16)
        num_files = len(records)

        # HDF5 in RAM
        with h5py.File(str(out_path), "w", libver="latest", driver="core", backing_store=True) as h5f:
            d_labels = h5f.create_dataset("labels", (num_files,), dtype=np.int16, track_times=cfg.track_times)
            d_labels[:] = labels_np

            d_wave = h5f.create_dataset(
                "segments_waveform",
                (num_files, cfg.num_long, cfg.num_short, cfg.short_window),
                dtype=np.float32,
                track_times=cfg.track_times,
            )

            # probe mel shape
            dummy = torch.randn(cfg.short_window, device=self.device)
            m = self.mel_transform(dummy)
            n_mels, t_frames = m.shape
            del dummy, m

            d_mel = h5f.create_dataset(
                "segments_mel",
                (num_files, cfg.num_long, cfg.num_short, n_mels, t_frames),
                dtype=np.float16,
                track_times=cfg.track_times,
            )

            # metadata (include BOTH builder cfg and CNN cfg for downstream trainer)
            h5f.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            h5f.attrs["config_json"] = json.dumps({
                "sample_rate": cfg.sample_rate, "duration_sec": cfg.duration_sec,
                "long_window": cfg.long_window, "short_window": cfg.short_window,
                "n_mels": cfg.n_mels, "n_fft": cfg.n_fft, "hop_length": cfg.hop_length, "power": cfg.power,
                "long_scale_sec": cfg.long_window / cfg.sample_rate,
                "short_points": cfg.short_window,
                "center": cfg.center
            })
            # Persist your CNN training configuration too:
            h5f.attrs["cnn_config_json"] = json.dumps({
                "model_type": cfg.CNN_MODEL_TYPE,
                "batch_size": cfg.CNN_BATCH_SIZE,
                "learning_rate": cfg.CNN_LEARNING_RATE,
                "dropout": cfg.CNN_DROPOUT,
                "epochs": cfg.CNN_EPOCHS,
                "filters": cfg.CNN_FILTERS,
                "kernel_size": list(cfg.CNN_KERNEL_SIZE),
                "pool_size": list(cfg.CNN_POOL_SIZE),
                "strides": list(cfg.CNN_STRIDES),
                "dense_units": cfg.CNN_DENSE,
            })
            h5f.attrs["labels_json"] = json.dumps(global_label_list)
            h5f.attrs["label2id_json"] = json.dumps(global_label2id)

            # RAM warning
            per_file_bytes = estimate_image_bytes(cfg, n_mels, t_frames, fp16_mel=True)
            total_bytes = per_file_bytes * num_files
            avail = psutil.virtual_memory().available
            if total_bytes > cfg.warn_ram_fraction * avail:
                logger.warning("[WARN] HDF5 image ~%s; available ~%s",
                             bytes_human(total_bytes), bytes_human(avail))

            logger.info("HDF5 datasets created in RAM. Starting processing...")

            # ----- Choose optimization strategy -----
            wav_data_source: Optional[Union[Dict[int, np.ndarray], np.memmap]] = None
            
            if cfg.use_ram_preload:
                # Strategy 1: RAM Preload (fastest, requires RAM)
                wav_data_source = self._preload_wavs_to_ram(records, global_label2id)
            elif cfg.use_memmap_cache:
                # Strategy 2: Memory-mapped cache (zero-copy, good for large datasets)
                cache_file = cfg.memmap_cache_dir / f"{split_name.lower()}_cache.mmap"
                if cache_file.exists():
                    wav_data_source = self._load_memmap_cache(cache_file, num_files)
                else:
                    wav_data_source = self._create_memmap_cache(records, global_label2id, cache_file)
            
            # ----- Prefetch + CPU segment -----
            indexed = [(i, p, global_label2id[lbl]) for i, (p, lbl) in enumerate(records)]
            batches = [indexed[i:i + cfg.disk_files_per_task] for i in range(0, num_files, cfg.disk_files_per_task)]
            prefetch_bar = tqdm(total=num_files, desc=f"{CYAN}[{split_name} Prefetch+Process]{RESET}", unit="file")

            data_full: Optional[np.ndarray] = None
            self.file_indices = []

            inflight_sem = threading.Semaphore(cfg.disk_max_inflight if cfg.disk_max_inflight > 0 else len(batches))

            def _prefetch_task(batch):
                with inflight_sem:
                    t0 = perf_counter()
                    result = prefetch_wavs(batch)
                    self.disk_read_times.append(perf_counter() - t0)
                    return result

            def _safe_window(threads: int, inflight: int, submit_window: int, nbatches: int) -> int:
                very_large = 10**9
                cap_threads = max(1, threads)
                cap_inflight = inflight if inflight and inflight > 0 else very_large
                cap_submit = submit_window if submit_window and submit_window > 0 else very_large
                cap_batches = max(1, nbatches)
                return max(1, min(cap_threads, cap_inflight, cap_submit, cap_batches))

            window = _safe_window(cfg.cpu_max_workers, cfg.disk_max_inflight, cfg.disk_submit_window, len(batches))

            # ----- Async I/O Pipeline: Separate disk reading from processing -----
            if cfg.use_async_pipeline and wav_data_source is None:
                # Use producer-consumer pattern for maximum overlap
                from queue import Queue
                
                disk_queue: Queue = Queue(maxsize=cfg.cpu_max_workers * 2)
                process_queue: Queue = Queue(maxsize=cfg.cpu_max_workers)
                stop_signal = threading.Event()
                
                def disk_producer():
                    """Producer thread: Read files from disk in parallel"""
                    with ThreadPoolExecutor(max_workers=cfg.cpu_max_workers) as disk_pool:
                        for batch in batches:
                            if stop_signal.is_set():
                                break
                            future = disk_pool.submit(_prefetch_task, batch)
                            disk_queue.put(future)
                    disk_queue.put(None)  # Signal completion
                
                def processing_consumer():
                    """Consumer thread: Process loaded data"""
                    data_full_local: Optional[np.ndarray] = None
                    
                    while not stop_signal.is_set():
                        future = disk_queue.get()
                        if future is None:  # Completion signal
                            break
                        
                        try:
                            res_list = future.result()
                            self.queue_depths.append(disk_queue.qsize())
                            
                            for (file_idx, p, data, sr, label_id) in res_list:
                                _ = label_id
                                t0 = perf_counter()
                                try:
                                    if sr != cfg.sample_rate:
                                        raise RuntimeError(f"{p} sr={sr}, expected={cfg.sample_rate}")
                                    
                                    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
                                        data = np.array(data, dtype=np.float32, copy=False)
                                    data = np.ascontiguousarray(data)
                                    
                                    target = cfg.num_samples
                                    if data.size != target:
                                        if data_full_local is None or data_full_local.size != target:
                                            data_full_local = np.empty(target, dtype=np.float32)
                                        L = min(data.size, target)
                                        data_full_local[:L] = data[:L]
                                        if L < target:
                                            data_full_local[L:] = 0.0
                                        view = data_full_local
                                    else:
                                        view = data
                                    
                                    segs3d = view.reshape(cfg.num_long, cfg.num_short, cfg.short_window)
                                    d_wave[file_idx, :, :, :] = segs3d
                                    self.file_indices.append(file_idx)
                                except Exception as e:
                                    logger.warning("[SKIP] %s → %s", p, e)
                                self.cpu_times.append(perf_counter() - t0)
                            prefetch_bar.update(len(res_list))
                        except Exception as e:
                            logger.error("Processing error: %s", e)
                
                # Start async pipeline
                producer_thread = threading.Thread(target=disk_producer, daemon=True)
                consumer_thread = threading.Thread(target=processing_consumer, daemon=True)
                
                producer_thread.start()
                consumer_thread.start()
                
                # Wait for completion
                producer_thread.join()
                consumer_thread.join()
                
            elif wav_data_source is not None:
                # Use preloaded data (RAM or memmap) - no disk I/O needed!
                logger.info("📊 Processing from preloaded data (zero disk I/O)...")
                
                for file_idx in tqdm(range(num_files), desc=f"{CYAN}[{split_name} Process]{RESET}", unit="file"):
                    try:
                        t0 = perf_counter()
                        
                        # Get PRE-SEGMENTED data from RAM/memmap (already 3D!)
                        if isinstance(wav_data_source, dict):
                            segs3d = wav_data_source.get(file_idx)
                            if segs3d is None:
                                continue
                        else:  # memmap (needs reshape)
                            data = wav_data_source[file_idx]
                            segs3d = data.reshape(cfg.num_long, cfg.num_short, cfg.short_window)
                        
                        # Direct write - no CPU reshape needed!
                        d_wave[file_idx, :, :, :] = segs3d
                        self.file_indices.append(file_idx)
                        self.cpu_times.append(perf_counter() - t0)
                        
                    except Exception as e:
                        logger.warning("[SKIP] file_idx=%d → %s", file_idx, e)
                
                prefetch_bar.update(num_files)
            
            else:
                # Original synchronous pipeline (fallback)
                with ThreadPoolExecutor(max_workers=cfg.cpu_max_workers) as pe:
                    it = iter(batches)
                    live = set()

                    # prime
                    for _ in range(window):
                        b = next(it, None)
                        if b is None:
                            break
                        live.add(pe.submit(_prefetch_task, b))

                    while live:
                        done, live = wait(live, return_when=FIRST_COMPLETED)
                        for fut in done:
                            res_list = fut.result()
                            
                            # Track queue depth for bottleneck analysis
                            self.queue_depths.append(len(live))
                            
                            for (file_idx, p, data, sr, label_id) in res_list:
                                _ = label_id  # Unused but needed for unpacking
                                t0 = perf_counter()
                                try:
                                    if sr != cfg.sample_rate:
                                        raise RuntimeError(f"{p} sr={sr}, expected={cfg.sample_rate}")

                                    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
                                        data = np.array(data, dtype=np.float32, copy=False)
                                    data = np.ascontiguousarray(data)

                                    # pad/trim via reusable buffer
                                    target = cfg.num_samples
                                    if data.size != target:
                                        if data_full is None or data_full.size != target:
                                            data_full = np.empty(target, dtype=np.float32)
                                        L = min(data.size, target)
                                        data_full[:L] = data[:L]
                                        if L < target:
                                            data_full[L:] = 0.0
                                        view = data_full
                                    else:
                                        view = data

                                    segs3d = view.reshape(cfg.num_long, cfg.num_short, cfg.short_window)
                                    d_wave[file_idx, :, :, :] = segs3d
                                    self.file_indices.append(file_idx)
                                except Exception as e:
                                    logger.warning("[SKIP] %s → %s", p, e)
                                self.cpu_times.append(perf_counter() - t0)
                            prefetch_bar.update(len(res_list))

                            nxt = next(it, None)
                            if nxt is not None:
                                live.add(pe.submit(_prefetch_task, nxt))

            prefetch_bar.close()
            del indexed, batches
            if data_full is not None:
                del data_full
            gc.collect()
            
            # ----- Pre-compute Mel Filterbanks (once, reused across all batches) -----
            if cfg.use_precomputed_filterbanks and self.mel_filterbank is None:
                logger.info("🎯 Pre-computing mel filterbanks on GPU (reusable across all splits)...")
                # Extract filterbank from mel_transform and cache on GPU
                with torch.no_grad():
                    # MelSpectrogram has internal mel_scale.fb (filterbank) tensor
                    try:
                        mel_scale = getattr(self.mel_transform, 'mel_scale', None)
                        if mel_scale is not None:
                            fb = getattr(mel_scale, 'fb', None)
                            if fb is not None and isinstance(fb, torch.Tensor):
                                self.mel_filterbank = fb.clone().to(self.device)
                                logger.info("✅ Mel filterbanks cached on GPU (%.2f MB)", 
                                           self.mel_filterbank.numel() * 4 / (1024**2))
                    except Exception as e:
                        logger.debug("Could not cache filterbank: %s", e)

            # ----- GPU Mel (quad-buffer + micro-pipeline with direct RAM access) -----
            # OPTIMIZATION: If using RAM preload, process directly from RAM to GPU
            # This eliminates the HDF5 intermediate step for waveform storage
            use_direct_gpu_processing = (cfg.use_ram_preload and wav_data_source is not None)
            
            if use_direct_gpu_processing:
                logger.info("⚡ Direct RAM→GPU processing enabled (bypass HDF5 waveform read)")
            
            # ----- GPU Mel (quad buffer + micro-pipeline) -----
            files_per_gpu_batch: int = cfg.files_per_gpu_batch
            if _HAS_NVML and (self.nvml_handle is not None):
                try:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)  # type: ignore
                    free_gb = float(meminfo.free) / (1024 ** 3)
                    suggested = autosize_gpu_batch(
                        cfg, free_gb, t_frames,
                        cfg.autosize_target_util_frac,
                        num_buffers=cfg.num_mega_buffers,
                        max_bsz=cfg.max_files_per_gpu_batch
                    )
                    if suggested != files_per_gpu_batch:
                        logger.info("Autosized FILES_PER_GPU_BATCH → %d (was %d), Buffers → %d",
                                  suggested, files_per_gpu_batch, cfg.num_mega_buffers)
                    files_per_gpu_batch = max(1, suggested)
                except Exception:
                    pass

            def _alloc_buffers(files_per_batch: int) -> Dict[str, Any]:
                """
                Allocate quad-buffered GPU pipeline buffers.
                
                OPTIMIZATION: Persistent buffers (reused across splits)
                - Avoids repeated allocation/deallocation overhead
                - Keeps GPU memory warm and reduces fragmentation
                - Reuses pinned host memory for consistent performance
                
                Uses pinned (page-locked) memory for faster H2D/D2H transfers.
                Pinned memory avoids extra copy through pageable memory, giving
                ~2x speedup for GPU transfers.
                
                Returns dict with pre-allocated buffers for zero-copy pipeline.
                """
                # Check if we can reuse existing buffers
                if cfg.use_persistent_buffers and self.persistent_buffers is not None:
                    existing_batch_size = self.persistent_buffers.get('files_per_batch', 0)
                    if existing_batch_size >= files_per_batch:
                        logger.info("♻️  Reusing persistent GPU buffers (%.2f MB)", 
                                   existing_batch_size * cfg.segments_per_file * cfg.short_window * 4 / (1024**2))
                        return self.persistent_buffers
                
                host_wave = np.empty((files_per_batch, cfg.num_long, cfg.num_short, cfg.short_window), dtype=np.float32)
                
                # OPTIMIZATION: Pinned memory for faster GPU transfers (~2x speedup)
                seg_cpu_pinned = torch.empty(
                    (files_per_batch * cfg.segments_per_file, cfg.short_window),
                    dtype=torch.float32, device="cpu", pin_memory=True
                )
                seg_dev = torch.empty(
                    (files_per_batch * cfg.segments_per_file, cfg.short_window),
                    dtype=torch.float32, device=self.device
                )
                
                # OPTIMIZATION: FP16 output reduces memory by 50% and I/O by 50%
                mel_cpu_buf = torch.empty(
                    (files_per_batch * cfg.segments_per_file, n_mels, t_frames),
                    dtype=torch.float16, device="cpu", pin_memory=True
                )
                mel_cpu_np = mel_cpu_buf.numpy()
                s_copy = torch.cuda.Stream()
                s_comp = torch.cuda.Stream()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                
                buf_dict = dict(
                    host_wave=host_wave,
                    seg_cpu_pinned=seg_cpu_pinned,
                    seg_dev=seg_dev,
                    mel_cpu_buf=mel_cpu_buf,
                    mel_cpu_np=mel_cpu_np,
                    s_copy=s_copy,
                    s_comp=s_comp,
                    start_evt=start_evt,
                    end_evt=end_evt,
                    files_per_batch=files_per_batch,
                )
                
                # Store as persistent buffer for reuse
                if cfg.use_persistent_buffers:
                    self.persistent_buffers = buf_dict
                
                return buf_dict

            def _create_timing_events() -> Dict[str, torch.cuda.Event]:
                """Create timing events for performance tracking."""
                return {
                    'h2d_start': torch.cuda.Event(enable_timing=True),
                    'h2d_end': torch.cuda.Event(enable_timing=True),
                    'comp_start': torch.cuda.Event(enable_timing=True),
                    'comp_end': torch.cuda.Event(enable_timing=True),
                    'd2h_start': torch.cuda.Event(enable_timing=True),
                    'd2h_end': torch.cuda.Event(enable_timing=True)
                }
            
            def _execute_mel_pipeline(buf: Dict[str, Any], B: int, timing: Dict[str, torch.cuda.Event]) -> None:
                """
                Execute mel spectrogram pipeline with microbatching and CUDA streams.
                
                Common logic for both HDF5 and direct RAM paths.
                Uses dual CUDA streams for overlapped H2D, compute, and D2H transfers.
                """
                s_copy, s_comp = buf["s_copy"], buf["s_comp"]
                MB = max(1, min(cfg.seg_microbatch_segments, B))
                
                buf["start_evt"].record(s_copy)
                timing['h2d_start'].record(s_copy)

                for off in range(0, B, MB):
                    end = min(off + MB, B)
                    h2d_done = torch.cuda.Event()
                    
                    # Async H2D transfer
                    with torch.cuda.stream(s_copy):
                        buf["seg_dev"][off:end].copy_(buf["seg_cpu_pinned"][off:end], non_blocking=True)
                        h2d_done.record(s_copy)

                    comp_done = torch.cuda.Event()
                    # GPU computation with CUDA graphs support
                    with torch.cuda.stream(s_comp):
                        s_comp.wait_event(h2d_done)
                        if off == 0:
                            timing['comp_start'].record(s_comp)
                        
                        microbatch_size = end - off
                        
                        # CUDA graph capture on first full microbatch
                        if cfg.use_cuda_graphs and self.cuda_graph is None and microbatch_size == MB:
                            s_comp.synchronize()
                            self.graph_input = buf["seg_dev"][off:end].clone()
                            
                            # Warmup
                            for _ in range(3):
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    _ = self.mel_transform(self.graph_input)
                            
                            # Capture
                            self.cuda_graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(self.cuda_graph, stream=s_comp):
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    graph_mel = self.mel_transform(self.graph_input)
                                self.graph_output = graph_mel.float().clamp_min_(1e-10).log10_().mul_(cfg.db_mult).to(torch.float16)
                            
                            logger.info("📊 CUDA Graph captured for mel pipeline (microbatch=%d segments)", microbatch_size)
                        
                        # Execute: Use CUDA graph if available and size matches
                        if (cfg.use_cuda_graphs and self.cuda_graph is not None and 
                            self.graph_input is not None and self.graph_input.size(0) == microbatch_size):
                            self.graph_input.copy_(buf["seg_dev"][off:end])
                            self.cuda_graph.replay()
                            m = self.graph_output
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                m = self.mel_transform(buf["seg_dev"][off:end])
                            m = m.float().clamp_min_(1e-10).log10_().mul_(cfg.db_mult).to(torch.float16)
                        
                        if end == B:
                            timing['comp_end'].record(s_comp)
                        comp_done.record(s_comp)

                    # Async D2H transfer
                    with torch.cuda.stream(s_copy):
                        s_copy.wait_event(comp_done)
                        if off == 0:
                            timing['h2d_end'].record(s_copy)
                            timing['d2h_start'].record(s_copy)
                        buf["mel_cpu_buf"][off:end].copy_(m, non_blocking=True)

                timing['d2h_end'].record(s_copy)
                buf["end_evt"].record(s_copy)
            
            def launch_batch_direct_ram(batch_files: List[int], buf: Dict[str, Any], 
                                       wav_source: Dict[int, np.ndarray]) -> Optional[Dict[str, Any]]:
                """
                Launch GPU batch with direct RAM→GPU transfer (bypass HDF5).
                
                Eliminates intermediate HDF5 waveform read by streaming directly
                from RAM cache to GPU. This is the fastest possible path.
                """
                if not batch_files:
                    return None
                
                B = len(batch_files) * cfg.segments_per_file
                
                # Load pre-segmented data from RAM cache
                for j, fidx in enumerate(batch_files):
                    segs3d = wav_source.get(fidx)
                    if segs3d is None:
                        continue
                    start_seg = j * cfg.segments_per_file
                    end_seg = start_seg + cfg.segments_per_file
                    buf["seg_cpu_pinned"][start_seg:end_seg] = torch.from_numpy(
                        segs3d.reshape(-1, cfg.short_window)
                    )
                
                # Execute mel pipeline
                timing = _create_timing_events()
                _execute_mel_pipeline(buf, B, timing)
                
                return dict(files=batch_files, buf=buf, **timing)

            def launch_batch(batch_files: List[int], buf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """
                Launch GPU processing for a batch of files with async H2D/compute/D2H pipeline.
                
                Optimizations:
                - Vectorized numpy->torch conversion (10-15x faster than loops)
                - Non-blocking async transfers (overlapped I/O and compute)
                - Dual CUDA streams (copy and compute in parallel)
                - Microbatching to keep GPU saturated
                """
                if not batch_files:
                    return None
                
                Bfiles = len(batch_files)
                
                # Read waveforms from HDF5 into host buffer
                for j, fidx in enumerate(batch_files):
                    d_wave.read_direct(buf["host_wave"][j], np.s_[fidx, :, :, :])
                
                B = Bfiles * cfg.segments_per_file
                
                # Vectorized reshape + conversion to pinned memory
                arr_view = buf["host_wave"][:Bfiles].reshape(B, cfg.short_window)
                buf["seg_cpu_pinned"][:B].copy_(torch.from_numpy(arr_view), non_blocking=False)
                
                # Execute mel pipeline
                timing = _create_timing_events()
                _execute_mel_pipeline(buf, B, timing)
                
                return dict(files=batch_files, buf=buf, **timing)

            def finish_batch(ctx: Optional[Dict[str, Any]]) -> int:
                """
                Finalize GPU batch: synchronize streams and write results to HDF5.
                
                Synchronization ensures all async GPU operations complete before
                accessing results on CPU side. Uses zero-copy numpy view into
                pinned torch tensor for efficient HDF5 writing.
                
                Returns:
                    Number of files processed in this batch
                """
                if ctx is None:
                    return 0
                files = ctx["files"]; buf = ctx["buf"]
                s_copy, s_comp = buf["s_copy"], buf["s_comp"]
                
                # Synchronize both streams to ensure all GPU work is complete
                s_copy.synchronize(); s_comp.synchronize()
                
                # Record GPU processing time for this batch
                total_time = buf["start_evt"].elapsed_time(buf["end_evt"]) / 1000.0
                self.gpu_times.append(total_time)
                self.gpu_batch_sizes.append(len(files))
                
                # Extract detailed timing breakdown
                if "h2d_start" in ctx:
                    h2d_ms = ctx["h2d_start"].elapsed_time(ctx["h2d_end"])
                    comp_ms = ctx["comp_start"].elapsed_time(ctx["comp_end"])
                    d2h_ms = ctx["d2h_start"].elapsed_time(ctx["d2h_end"])
                    self.gpu_h2d_times.append(h2d_ms / 1000.0)
                    self.gpu_compute_times.append(comp_ms / 1000.0)
                    self.gpu_d2h_times.append(d2h_ms / 1000.0)
                
                # OPTIMIZATION: Zero-copy write via numpy view into pinned tensor
                # mel_cpu_np is a numpy view, no data copy when reshaping
                for j, fidx in enumerate(files):
                    start = j * cfg.segments_per_file
                    stop = start + cfg.segments_per_file
                    mnp = buf["mel_cpu_np"][start:stop].reshape(cfg.num_long, cfg.num_short, n_mels, t_frames)
                    d_mel[fidx, :, :, :, :] = mnp
                return len(files)

            # OPTIMIZATION: Triple-buffered ring buffer for GPU pipeline
            # Allows overlapping: CPU read → GPU H2D → GPU compute → GPU D2H → CPU write
            # While GPU processes batch N, CPU can prepare batch N+1
            # Keeps both CPU and GPU busy for maximum throughput
            buffers = [_alloc_buffers(files_per_gpu_batch) for _ in range(cfg.num_mega_buffers)]
            mel_bar = tqdm(total=len(self.file_indices), desc=f"{CYAN}[{split_name} Mel MegaBatch]{RESET}", unit="file")

            work = list(sorted(self.file_indices))
            i = 0
            free_buf_ids = list(range(cfg.num_mega_buffers))
            inflight: List[Tuple[Dict, int]] = []  # (context, buffer_id) pairs

            # Main processing loop: launch batches and finish completed ones
            # Choose optimal launch function based on data source
            if use_direct_gpu_processing and isinstance(wav_data_source, dict):
                launch_fn = lambda batch_files, buf: launch_batch_direct_ram(batch_files, buf, wav_data_source)
            else:
                launch_fn = launch_batch
            
            while i < len(work) or inflight:
                while free_buf_ids and i < len(work):
                    bsz = files_per_gpu_batch
                    batch = work[i: i + bsz]
                    if not batch:
                        break
                    buf_id = free_buf_ids.pop(0)
                    buf = buffers[buf_id]
                    try:
                        ctx = launch_fn(batch, buf)
                        if ctx is not None:
                            inflight.append((ctx, buf_id))
                        i += len(batch)
                    except torch.cuda.OutOfMemoryError:
                        # finish in-flight then shrink
                        for _ in range(len(inflight)):
                            ctx_k, bid_k = inflight.pop(0)
                            processed = finish_batch(ctx_k)
                            mel_bar.update(processed)
                            free_buf_ids.append(bid_k)
                        new_bsz = max(128, files_per_gpu_batch // 2)
                        # free & reallocate
                        for b in buffers: del b
                        gc.collect(); torch.cuda.empty_cache()
                        buffers = [_alloc_buffers(new_bsz) for _ in range(cfg.num_mega_buffers)]
                        files_per_gpu_batch = new_bsz
                        logger.warning("[OOM-backoff] Shrinking FILES_PER_GPU_BATCH → %d", new_bsz)
                        free_buf_ids = list(range(cfg.num_mega_buffers))
                        inflight = []
                        break

                if not inflight:
                    continue
                ctx0, bid0 = inflight.pop(0)
                processed = finish_batch(ctx0)
                mel_bar.update(processed)
                free_buf_ids.append(bid0)

            mel_bar.close()
            for b in buffers: del b
            gc.collect(); torch.cuda.empty_cache()

            logger.info("Writing HDF5 to disk (single sequential flush on close)...")

        logger.info("%s HDF5 written: %s", split_name, out_path)

        # per-split performance analysis
        logger.info("\n" + "="*70)
        logger.info(f"{split_name} PERFORMANCE ANALYSIS")
        logger.info("="*70)
        
        # Disk I/O stats
        if self.disk_read_times:
            total_disk = sum(self.disk_read_times)
            avg_disk = np.mean(self.disk_read_times)
            logger.info(f"Disk I/O: total={total_disk:.2f}s, avg={avg_disk:.4f}s/batch, "
                       f"max={np.max(self.disk_read_times):.4f}s, batches={len(self.disk_read_times)}")
        
        # CPU processing stats
        if self.cpu_times:
            total_cpu = sum(self.cpu_times)
            avg_cpu = np.mean(self.cpu_times)
            logger.info(f"CPU Processing: total={total_cpu:.2f}s, avg={avg_cpu:.4f}s/file, "
                       f"max={np.max(self.cpu_times):.4f}s, files={len(self.cpu_times)}")
        
        # GPU breakdown
        if self.gpu_times:
            total_gpu = sum(self.gpu_times)
            avg_gpu = np.mean(self.gpu_times)
            logger.info(f"GPU Total: total={total_gpu:.2f}s, avg={avg_gpu:.4f}s/batch, "
                       f"max={np.max(self.gpu_times):.4f}s, batches={len(self.gpu_times)}")
        
        if self.gpu_h2d_times:
            avg_h2d = np.mean(self.gpu_h2d_times) * 1000  # Convert to ms
            pct_h2d = (sum(self.gpu_h2d_times) / sum(self.gpu_times) * 100) if self.gpu_times else 0
            logger.info(f"  ├─ H2D Transfer: avg={avg_h2d:.2f}ms ({pct_h2d:.1f}% of GPU time)")
        
        if self.gpu_compute_times:
            avg_comp = np.mean(self.gpu_compute_times) * 1000  # Convert to ms
            pct_comp = (sum(self.gpu_compute_times) / sum(self.gpu_times) * 100) if self.gpu_times else 0
            logger.info(f"  ├─ GPU Compute: avg={avg_comp:.2f}ms ({pct_comp:.1f}% of GPU time)")
        
        if self.gpu_d2h_times:
            avg_d2h = np.mean(self.gpu_d2h_times) * 1000  # Convert to ms
            pct_d2h = (sum(self.gpu_d2h_times) / sum(self.gpu_times) * 100) if self.gpu_times else 0
            logger.info(f"  └─ D2H Transfer: avg={avg_d2h:.2f}ms ({pct_d2h:.1f}% of GPU time)")
        
        # Throughput
        if self.gpu_times and self.file_indices:
            total_gpu_s = sum(self.gpu_times)
            if total_gpu_s > 0:
                total_files = len(self.file_indices)
                files_per_s = total_files / total_gpu_s
                segs_per_s = (total_files * self.cfg.segments_per_file) / total_gpu_s
                logger.info(f"GPU Throughput: {files_per_s:,.0f} files/s, {segs_per_s:,.0f} segments/s")
        
        # Batch size analysis
        if self.gpu_batch_sizes:
            avg_bsz = np.mean(self.gpu_batch_sizes)
            logger.info(f"GPU Batch Sizes: avg={avg_bsz:.0f}, max={np.max(self.gpu_batch_sizes)}, "
                       f"min={np.min(self.gpu_batch_sizes)}")
        
        # Queue depth analysis (bottleneck indicator)
        if self.queue_depths:
            avg_depth = np.mean(self.queue_depths)
            max_depth = np.max(self.queue_depths)
            logger.info(f"Queue Depth: avg={avg_depth:.1f}, max={max_depth}")
            if avg_depth < 2:
                logger.warning(f"⚠ LOW QUEUE DEPTH ({avg_depth:.1f}) - CPU cannot keep up with GPU!")
                logger.info("💡 Suggestion: Increase cpu_max_workers (currently %d)", cfg.cpu_max_workers)
        
        # Bottleneck analysis
        if self.disk_read_times and self.gpu_times:
            total_disk = sum(self.disk_read_times)
            total_gpu = sum(self.gpu_times)
            disk_pct = (total_disk / (total_disk + total_gpu) * 100) if (total_disk + total_gpu) > 0 else 0
            gpu_pct = (total_gpu / (total_disk + total_gpu) * 100) if (total_disk + total_gpu) > 0 else 0
            logger.info(f"Time Distribution: Disk={disk_pct:.1f}% ({total_disk:.1f}s), GPU={gpu_pct:.1f}% ({total_gpu:.1f}s)")
            
            if total_disk > total_gpu * 1.5:
                logger.warning("⚠ DISK I/O BOTTLENECK detected")
                logger.info("💡 Suggestions:")
                logger.info("   - Increase disk_files_per_task (currently %d)", cfg.disk_files_per_task)
                logger.info("   - Increase cpu_max_workers (currently %d)", cfg.cpu_max_workers)
            elif total_gpu > total_disk * 2:
                logger.warning("⚠ GPU UNDERUTILIZED - CPU feeding too slow")
                logger.info("💡 Suggestions:")
                logger.info("   - Increase cpu_max_workers (currently %d)", cfg.cpu_max_workers)
                logger.info("   - Increase disk_files_per_task (currently %d)", cfg.disk_files_per_task)
                logger.info("   - Increase files_per_gpu_batch (currently %d)", cfg.files_per_gpu_batch)
        
        logger.info("="*70 + "\n")

    # --------- orchestrate all splits ---------
    def build_all(self):
        cfg = self.cfg
        cfg.stage_dir.mkdir(parents=True, exist_ok=True)

        self._start_profiling()

        # global label mapping across all splits
        all_records, global_label2id = self.discover_global_labels(
            [cfg.training_dir, cfg.validation_dir, cfg.testing_dir]
        )
        _ = all_records  # Not used but returned from discovery
        global_label_list = [l for l, _ in sorted(global_label2id.items(), key=lambda kv: kv[1])]
        logger.info("Global labels: %s", global_label2id)

        # TRAIN
        self.build_split("TRAINING", cfg.training_dir, cfg.stage_dir / cfg.training_hdf5,
                         global_label2id, global_label_list)
        # VALIDATION
        self.build_split("VALIDATION", cfg.validation_dir, cfg.stage_dir / cfg.validation_hdf5,
                         global_label2id, global_label_list)
        # TESTING
        self.build_split("TESTING", cfg.testing_dir, cfg.stage_dir / cfg.testing_hdf5,
                         global_label2id, global_label_list)

        self._stop_profiling()

        # profiling summary
        if self.profile_stats:
            cpu_vals  = [x[1] for x in self.profile_stats if x[1] is not None]
            ram_vals  = [x[2] for x in self.profile_stats if x[2] is not None]
            gpu_vals  = [x[3] for x in self.profile_stats if x[3] is not None]
            vram_vals = [x[4] for x in self.profile_stats if x[4] is not None]
            logger.info("CPU usage avg: %.1f%%, max: %.1f%%, min: %.1f%%",
                      np.mean(cpu_vals), np.max(cpu_vals), np.min(cpu_vals))
            logger.info("RAM usage avg: %.1f%%, max: %.1f%%, min: %.1f%%",
                      np.mean(ram_vals), np.max(ram_vals), np.min(ram_vals))
            if gpu_vals:
                logger.info("GPU usage avg: %.1f%%, max: %.1f%%, min: %.1f%%",
                          np.mean(gpu_vals), np.max(gpu_vals), np.min(gpu_vals))
            if vram_vals:
                logger.info("VRAM usage avg: %.2f GB, max: %.2f GB, min: %.2f GB",
                          np.mean(vram_vals), np.max(vram_vals), np.min(vram_vals))


# ======================================================================
# MAIN
# ======================================================================

def main():
    """Main entry point for dataset builder."""
    cfg = Config()
    builder = MultiSplitBuilder(cfg)

    logger.info("[I/O] threads=%d, files_per_task=%d, inflight_depth=%d, submit_window=%d",
              cfg.cpu_max_workers, cfg.disk_files_per_task, cfg.disk_max_inflight, cfg.disk_submit_window)

    def _graceful_exit(signum, frame):
        _ = frame  # Unused but required by signal handler signature
        logger.warning("Signal %d received; stopping...", signum)
        builder.stop_evt.set()
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    builder.build_all()


if __name__ == "__main__":
    main()
