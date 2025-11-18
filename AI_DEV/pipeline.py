#!/usr/bin/env python3
"""
High-throughput audio → Mel-spectrogram feature extraction pipeline.

This version adds detailed diagnostics and logging to help debug
end-of-run hangs and verify that all files, batches, and segments
are being processed and flushed correctly.
"""

import os
import sys
import time
import queue
import threading
import fcntl
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torchaudio
import psutil
import pynvml
import logging

# Optional TensorRT support
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from global_config import CPU_COUNT, DRIVE_BUFFERSIZE, PREFETCH_DEPTH, PREFETCH_THREADS, FILES_PER_TASK, DATASET_TRAINING, PROC_OUTPUT, SAMPLE_RATE, SAMPLE_LENGTH_SEC, HOP_LENGTH, N_MELS, N_FFT, LONG_SEGMENT_SCALE_SEC, SHORT_SEGMENT_POINTS

# ======================================================================
# LOGGING SETUP
# ======================================================================

# Set PIPELINE_LOG_LEVEL=DEBUG in your environment for deep diagnostics.
LOG_LEVEL = os.environ.get("PIPELINE_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    #format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    format="[%(levelname)s] %(threadName)s: %(message)s",
)
logger = logging.getLogger("mel_pipeline")


# ======================================================================
# CONFIGURATION
# ======================================================================


class Config:
    """
    Global configuration for the audio processing pipeline.

    Attributes are grouped into:
      - CPU / GPU parallelism tuning
      - Disk / RAM prefetch buffering
      - Audio geometry and segmentation parameters
      - Output paths and device setup
    """

    def __init__(self) -> None:
        # ---------- CPU settings ----------
        self.CPU_THREADS: int = CPU_COUNT
        logger.info(f"Detected CPU_COUNT={self.CPU_THREADS}")

        # ---------- GPU settings ----------
        self.GPU_BATCH_SIZE: int = 4096
        self.GPU_MEMORY_FRACTION: float = 0.85

        # ---------- CUDA streams ----------
        self.CUDA_STREAMS: int = 8  # Increased from 1 for better GPU parallelism

        # ---------- Precision settings ----------
        self.PRECISION: str = "fp16"  # Options: "fp16", "bf16", "fp8"
        self.USE_TENSORRT: bool = False  # TensorRT engine for mel transforms

        # ---------- RAM prefetch settings ----------
        self.RAM_PREFETCH_DEPTH: int = 24
        self.RAM_AUDIO_Q_SIZE: int = self.RAM_PREFETCH_DEPTH

        # ---------- Disk prefetch settings ----------
        self.PREFETCH_THREADS: int = PREFETCH_THREADS
        self.PREFETCH_DEPTH: int = PREFETCH_DEPTH
        self.FILES_PER_TASK: int = FILES_PER_TASK
        self.DRIVE_BUFFERSIZE: int = DRIVE_BUFFERSIZE

        # ---------- Audio / segmentation parameters ----------
        self.SAMPLE_RATE: int = SAMPLE_RATE
        self.SAMPLE_LENGTH_SEC: int = SAMPLE_LENGTH_SEC

        self.LONG_SEGMENT_SCALE_SEC: float = LONG_SEGMENT_SCALE_SEC
        self.SHORT_SEGMENT_POINTS: int = SHORT_SEGMENT_POINTS

        self.N_FFT: int = N_FFT
        self.HOP_LENGTH: int = HOP_LENGTH
        self.N_MELS: int = N_MELS

        # ---------- Output / dataset paths ----------
        self.OUTPUT_DIR: Path = Path(PROC_OUTPUT)
        self.DATASET_PATH: Path = Path(DATASET_TRAINING)

        # ---------- Device / CUDA optimization ----------
        self.DEVICE: torch.device = torch.device("cuda")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        try:
            torch.cuda.set_per_process_memory_fraction(self.GPU_MEMORY_FRACTION)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to set per-process memory fraction: %s", exc
            )
        torch.cuda.empty_cache()

        self.compute_streams = [
            torch.cuda.Stream() for _ in range(self.CUDA_STREAMS)
        ]


cfg = Config()


# ======================================================================
# HELPER UTILITIES
# ======================================================================


def init_nvml() -> Optional[Any]:
    """
    Initialize NVML and return the handle for GPU index 0.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        logger.info("NVML initialized, using GPU 0")
        return handle
    except Exception as exc:  # noqa: BLE001
        logger.warning("NVML init failed: %s", exc)
        return None


def get_gpu_occupancy(gpu_handle: Optional[Any]) -> Tuple[int, int]:
    """
    Get GPU utilization percentage and SM count.
    Returns (gpu_utilization_percent, total_sm_count)
    """
    if gpu_handle is None:
        return 0, 82  # RTX 5090 has 82 SMs
    
    try:
        # Get GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        # RTX 5090 has 82 SMs (from test results)
        total_sms = 82
        return int(util), total_sms
    except Exception as exc:  # noqa: BLE001
        logger.debug("GPU occupancy query failed: %s", exc)
        return 0, 82


def safe_init_memmap(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.dtype(np.float32),
    mode: Literal["r+", "r", "w+", "c"] = "w+",
) -> np.memmap:
    """
    Create a NumPy memmap ensuring the requested shape is non-empty.
    """
    total_elems = int(np.prod(shape))
    if total_elems <= 0:
        msg = (
            f"Attempted to create zero-sized memmap at {path} "
            f"with shape={shape}, total_elems={total_elems}. "
            "Likely cause: no WAV files / segments discovered."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Allocating memmap at %s with shape=%s, dtype=%s", path, shape, dtype)
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)


class AtomicCounter:
    """
    Thread-safe integer counter for cross-thread statistics.
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """Atomically increment the counter by `delta`."""
        with self._lock:
            self._value += delta
            return self._value

    def get(self) -> int:
        """Atomically read the counter value."""
        with self._lock:
            return self._value


def build_tensorrt_mel_engine(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    segment_size: int,
    precision: str = "fp16",
) -> Optional[Any]:
    """
    Build a TensorRT engine for mel spectrogram computation (optional).
    Returns engine if TensorRT available and enabled, else None.
    """
    if not TENSORRT_AVAILABLE:
        logger.info("TensorRT not available; skipping engine build")
        return None
    
    try:
        logger.info("Building TensorRT mel spectrogram engine (%s precision)...", precision)
        
        # For now, return None as TensorRT for mel requires complex setup
        # In production, you'd define a TRT network with custom CUDA kernels
        # or use TRT's built-in layers to approximate mel spectrogram
        logger.info("TensorRT mel engine support deferred (requires custom kernel)")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorRT engine build failed: %s; falling back to PyTorch", exc)
        return None


# ======================================================================
# MAIN PIPELINE
# ======================================================================


def run_pipeline() -> None:
    """
    Main entry point for the NVMe → RAM → GPU Mel-spectrogram pipeline.

    Steps
    -----
    1. Discover all WAV files in the dataset tree.
    2. Compute segmentation geometry and total segment counts.
    3. Allocate memmaps for:
         - Mel features: (total_segments, N_MELS, time_frames)
         - Mapping array: (total_segments, 6)
    4. Launch:
         - Status reporter thread.
         - GPU consumer thread.
         - Thread pool for NVMe prefetch → pinned RAM buffers.
    5. Stream data through the pipeline until all files are processed.
    6. Flush memmaps / mapping array to disk.
    """

    gpu_handle = init_nvml()

    HEADER_SIZE = 44  # Standard WAV header size (bytes)
    NUM_SAMPLES = cfg.SAMPLE_RATE * cfg.SAMPLE_LENGTH_SEC
    BYTES_PER_SAMPLE = np.dtype(np.int16).itemsize
    
    # Log TensorRT availability
    if cfg.USE_TENSORRT and TENSORRT_AVAILABLE:
        logger.info("TensorRT enabled for mel transforms")
    elif cfg.USE_TENSORRT:
        logger.warning("TensorRT requested but not available; using PyTorch")
        cfg.USE_TENSORRT = False
    else:
        logger.info("Using PyTorch for mel transforms (TensorRT disabled)")

    # ------------------------------------------------------------------
    # FILE DISCOVERY
    # ------------------------------------------------------------------
    logger.info("Scanning for WAV files under %s", cfg.DATASET_PATH)
    wav_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(cfg.DATASET_PATH)
        for filename in files
        if filename.lower().endswith(".wav")
    ]
    total_files = len(wav_files)
    logger.info("Discovered %d WAV files", total_files)

    if total_files == 0:
        logger.error("No WAV files found under %s. Aborting.", cfg.DATASET_PATH)
        return

    # ------------------------------------------------------------------
    # SEGMENTATION GEOMETRY
    # ------------------------------------------------------------------
    long_win = int(cfg.SAMPLE_RATE * cfg.LONG_SEGMENT_SCALE_SEC)
    long_hop = long_win // 2

    short_win = cfg.SHORT_SEGMENT_POINTS
    short_hop = short_win // 2

    num_long_segments = 1 + (NUM_SAMPLES - long_win) // long_hop
    num_short_segments_per_long = 1 + (long_win - short_win) // short_hop
    total_short_segments = (
        total_files * num_long_segments * num_short_segments_per_long
    )

    if total_short_segments <= 0:
        logger.error(
            "Computed non-positive total_short_segments=%d "
            "(files=%d, num_long_segments=%d, num_short_segments_per_long=%d). "
            "Aborting.",
            total_short_segments,
            total_files,
            num_long_segments,
            num_short_segments_per_long,
        )
        return

    mel_time_frames = (short_win - cfg.N_FFT) // cfg.HOP_LENGTH + 1
    mel_shape = (total_short_segments, cfg.N_MELS, mel_time_frames)

    logger.info("Segmentation geometry:")
    logger.info("  NUM_SAMPLES=%d", NUM_SAMPLES)
    logger.info("  long_win=%d, long_hop=%d", long_win, long_hop)
    logger.info("  short_win=%d, short_hop=%d", short_win, short_hop)
    logger.info("  num_long_segments=%d", num_long_segments)
    logger.info("  num_short_segments_per_long=%d", num_short_segments_per_long)
    logger.info("  total_short_segments=%d", total_short_segments)
    logger.info("  mel_shape=%s", mel_shape)

    # ------------------------------------------------------------------
    # MEMMAP ALLOCATION
    # ------------------------------------------------------------------
    mel_memmap_path = cfg.OUTPUT_DIR / "PIPELINE_FEATURES.DAT"
    mel_memmap = safe_init_memmap(
        mel_memmap_path, mel_shape, dtype=np.dtype(np.float32), mode="w+"
    )

    mapping_path = cfg.OUTPUT_DIR / "PIPELINE_MEMMAP.NPY"

    if total_short_segments < 10_000_000:
        mapping_array: Union[np.ndarray, np.memmap] = np.empty(
            (total_short_segments, 6), dtype=np.int64
        )
        use_memmap_mapping = False
        logger.info(
            "Using in-RAM mapping array of shape %s", mapping_array.shape
        )
    else:
        mapping_memmap_path = str(mapping_path).replace(".npy", "_temp.dat")
        mapping_array = np.memmap(
            mapping_memmap_path,
            dtype=np.int64,
            mode="w+",
            shape=(total_short_segments, 6),
        )
        use_memmap_mapping = True
        logger.info(
            "Using memmap mapping array at %s with shape %s",
            mapping_memmap_path,
            mapping_array.shape,
        )

    # ------------------------------------------------------------------
    # QUEUES, FLAGS, COUNTERS
    # ------------------------------------------------------------------
    ram_audio_q: "queue.Queue[tuple[int, torch.Tensor]]" = queue.Queue( maxsize=cfg.RAM_PREFETCH_DEPTH)

    done_flag = threading.Event()
    producer_complete = threading.Event()

    nvme_bytes_read = AtomicCounter()
    gpu_bytes_processed = AtomicCounter()
    files_processed = AtomicCounter()
    batches_processed = AtomicCounter()

    # ------------------------------------------------------------------
    # GPU TRANSFORMS
    # ------------------------------------------------------------------
    logger.info("Initializing GPU transforms on device %s", cfg.DEVICE)
    logger.info("Precision mode: %s, CUDA Streams: %d", cfg.PRECISION, cfg.CUDA_STREAMS)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        center=False,
        power=2.0,
        norm="slaney",
    ).to(cfg.DEVICE)

    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80.0,
    ).to(cfg.DEVICE)
    
    # Determine autocast dtype based on precision setting
    if cfg.PRECISION == "fp8":
        # FP8 requires compute capability 8.9+ (Blackwell, Ada, etc.)
        autocast_dtype = torch.float8_e4m3fn
        logger.info("Using FP8 (E4M3) precision for mel transforms")
    elif cfg.PRECISION == "bf16":
        autocast_dtype = torch.bfloat16
        logger.info("Using BF16 precision for mel transforms")
    else:  # Default to fp16
        autocast_dtype = torch.float16
        logger.info("Using FP16 precision for mel transforms")

    # ==================================================================
    # STATUS REPORTER THREAD
    # ==================================================================

    def status_reporter(start_time: float) -> None:
        last_nvme = nvme_bytes_read.get()
        last_gpu = gpu_bytes_processed.get()

        logger.info("Status reporter thread started")
        while not done_flag.is_set():
            elapsed = time.time() - start_time

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            if gpu_handle is not None:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(
                        gpu_handle
                    ).gpu
                    vram_used = (
                        int(
                            pynvml.nvmlDeviceGetMemoryInfo(
                                gpu_handle
                            ).used
                        )
                        / (1024**2)
                    )
                    # GPU occupancy (approximate via utilization)
                    occupancy, total_sms = get_gpu_occupancy(gpu_handle)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("NVML query failed: %s", exc)
                    gpu_util = 0
                    vram_used = 0
                    occupancy = 0
                    total_sms = 82
            else:
                gpu_util = 0
                vram_used = 0
                occupancy = 0
                total_sms = 82

            current_nvme = nvme_bytes_read.get()
            current_gpu = gpu_bytes_processed.get()
            current_files = files_processed.get()

            nvme_rate = (current_nvme - last_nvme) / (1024**3)
            gpu_rate = (current_gpu - last_gpu) / (1024**3)

            last_nvme = current_nvme
            last_gpu = current_gpu

            logger.info(
                "%6.1fs | CPU %5.1f%% | GPU %3d%% | Occ %3d%% | RAM %5.1f%% | "
                "VRAM %6.0fMB | Buff %d/%d | Files %d/%d | "
                "NVMe %5.2f GB/s | GPU %5.2f GB/s | Streams %d | Prec %s",
                elapsed,
                cpu,
                gpu_util,
                occupancy,
                ram,
                vram_used,
                ram_audio_q.qsize(),
                ram_audio_q.maxsize,
                current_files,
                total_files,
                nvme_rate,
                gpu_rate,
                cfg.CUDA_STREAMS,
                cfg.PRECISION,
            )

            time.sleep(1.0)

        logger.info("Status reporter thread exiting")

    # ==================================================================
    # DISK → RAM PREFETCH (PRODUCER)
    # ==================================================================

    def prefetch_audio(start_idx: int) -> None:
        """
        Read a contiguous block of WAV files optimized with tuned disk settings.
        
        Uses:
        - O_NOATIME flag for reduced syscalls
        - Configurable buffer size (131KB optimal per tuning)
        - Sequential read hints via posix_fadvise
        """
        end_idx = min(start_idx + cfg.FILES_PER_TASK, total_files)
        batch_size = end_idx - start_idx

        logger.debug(
            "Prefetch task starting for files [%d, %d) (batch_size=%d)",
            start_idx,
            end_idx,
            batch_size,
        )

        buf = torch.empty(
            (batch_size, NUM_SAMPLES),
            dtype=torch.float32,
        ).pin_memory()

        for i, file_idx in enumerate(range(start_idx, end_idx)):
            file_path = wav_files[file_idx]
            try:
                # Open with O_NOATIME to avoid inode updates (tuned parameter)
                fd = os.open(file_path, os.O_RDONLY | os.O_NOATIME)
                try:
                    # Apply posix_fadvise for sequential access (tuning: NONE = let kernel decide)
                    # Fadvise hint was NONE in tuning, but we can use SEQUENTIAL for clarity
                    try:
                        fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    except (AttributeError, OSError):
                        pass  # posix_fadvise not available on all systems
                    
                    # Seek to WAV header
                    os.lseek(fd, HEADER_SIZE, os.SEEK_SET)
                    
                    # Read using tuned buffer size for optimal throughput
                    bytes_to_read = NUM_SAMPLES * BYTES_PER_SAMPLE
                    raw = b""
                    remaining = bytes_to_read
                    
                    while remaining > 0:
                        chunk_size = min(cfg.DRIVE_BUFFERSIZE, remaining)
                        chunk = os.read(fd, chunk_size)
                        if not chunk:
                            break
                        raw += chunk
                        remaining -= len(chunk)
                    
                    if len(raw) == bytes_to_read:
                        audio_np = (
                            np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                    else:
                        audio_data = (
                            np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        audio_np = np.zeros(NUM_SAMPLES, dtype=np.float32)
                        audio_np[: len(audio_data)] = audio_data
                        logger.debug(
                            "Short read for %s (got %d bytes), padded to %d samples",
                            file_path,
                            len(raw),
                            NUM_SAMPLES,
                        )
                    
                    buf[i] = torch.from_numpy(audio_np)
                    nvme_bytes_read.increment(len(raw))
                    files_processed.increment()
                
                finally:
                    os.close(fd)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", file_path, exc)
                buf[i] = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
                files_processed.increment()

        logger.debug(
            "Prefetch task finished reading [%d, %d), pushing batch to queue",
            start_idx,
            end_idx,
        )

        try:
            ram_audio_q.put((start_idx, buf), timeout=60)
            logger.debug(
                "Prefetch task enqueued batch for [%d, %d)", start_idx, end_idx
            )
        except queue.Full:
            logger.error(
                "ram_audio_q full while enqueuing batch [%d, %d); dropping",
                start_idx,
                end_idx,
            )

    # ==================================================================
    # RAM → GPU CONSUMER
    # ==================================================================

    def gpu_consumer() -> None:
        """
        Consume batches from `ram_audio_q`, compute Mel features on GPU,
        and write them into the memmap together with the mapping array.
        """
        logger.info("GPU consumer thread started")
        mel_index = 0
        next_batch: Optional[tuple[int, torch.Tensor]] = None

        while True:
            if next_batch is None:
                try:
                    logger.debug(
                        "GPU consumer waiting for batch; producer_complete=%s, qsize=%d",
                        producer_complete.is_set(),
                        ram_audio_q.qsize(),
                    )
                    next_batch = ram_audio_q.get(timeout=5.0)
                    logger.debug(
                        "GPU consumer acquired batch from queue (start_idx=%d)",
                        next_batch[0],
                    )
                except queue.Empty:
                    if producer_complete.is_set() and ram_audio_q.empty():
                        logger.info(
                            "GPU consumer: producer_complete set and queue empty; exiting loop"
                        )
                        break
                    logger.debug(
                        "GPU consumer timeout waiting for batch; "
                        "producer_complete=%s, qsize=%d",
                        producer_complete.is_set(),
                        ram_audio_q.qsize(),
                    )
                    continue

            start_file_idx, buf = next_batch
            batch_size = buf.size(0)

            stream_id = batches_processed.get() % cfg.CUDA_STREAMS
            stream = cfg.compute_streams[stream_id]

            logger.debug(
                "GPU consumer processing batch (start_file_idx=%d, batch_size=%d, "
                "stream_id=%d, current_mel_index=%d)",
                start_file_idx,
                batch_size,
                stream_id,
                mel_index,
            )

            with torch.cuda.stream(stream):
                gpu_buf = buf.to(cfg.DEVICE, non_blocking=True)

                long_segments = gpu_buf.unfold(1, long_win, long_hop)
                short_segments = long_segments.unfold(2, short_win, short_hop)
                batch_segments = short_segments.reshape(-1, short_win)
                total_segments = batch_segments.size(0)

                if total_segments == 0:
                    logger.warning(
                        "Batch from start_file_idx=%d produced zero segments; skipping",
                        start_file_idx,
                    )
                    next_batch = None
                    continue

                if mel_index + total_segments > total_short_segments:
                    logger.error(
                        "Segment overflow: mel_index=%d, total_segments=%d, "
                        "total_short_segments=%d; clipping",
                        mel_index,
                        total_segments,
                        total_short_segments,
                    )
                    total_segments = max(
                        0, total_short_segments - mel_index
                    )
                    batch_segments = batch_segments[:total_segments]

                idx_range = np.arange(mel_index, mel_index + total_segments)

                file_indices = np.repeat(
                    np.arange(start_file_idx, start_file_idx + batch_size),
                    num_long_segments * num_short_segments_per_long,
                )[:total_segments]

                long_indices = np.tile(
                    np.repeat(
                        np.arange(num_long_segments),
                        num_short_segments_per_long,
                    ),
                    batch_size,
                )[:total_segments]

                short_indices = np.tile(
                    np.arange(num_short_segments_per_long),
                    batch_size * num_long_segments,
                )[:total_segments]

                start_samples = long_indices * long_hop + short_indices * short_hop
                end_samples = start_samples + short_win

                mapping_array[mel_index : mel_index + total_segments, :] = (
                    np.column_stack(
                        (
                            idx_range,
                            file_indices,
                            long_indices,
                            short_indices,
                            start_samples,
                            end_samples,
                        )
                    )
                )

                # Use selected precision for mel transform
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    mel_spec = mel_transform(batch_segments)
                    mel_spec_db = amplitude_to_db(mel_spec)

                mel_result = mel_spec_db.float().contiguous().cpu().numpy()
                mel_memmap[mel_index : mel_index + total_segments] = mel_result

                gpu_bytes_processed.increment(
                    batch_segments.numel() * batch_segments.element_size()
                )

            stream.synchronize()

            mel_index += total_segments
            batches_processed.increment()

            logger.debug(
                "GPU consumer finished batch (start_file_idx=%d, total_segments=%d, "
                "new_mel_index=%d)",
                start_file_idx,
                total_segments,
                mel_index,
            )

            if not producer_complete.is_set() or not ram_audio_q.empty():
                try:
                    next_batch = ram_audio_q.get_nowait()
                    logger.debug(
                        "GPU consumer immediately acquired next batch (start_idx=%d)",
                        next_batch[0],
                    )
                except queue.Empty:
                    next_batch = None
            else:
                next_batch = None

        if mel_index != total_short_segments:
            logger.warning(
                "GPU consumer exiting with mel_index=%d but expected total_short_segments=%d",
                mel_index,
                total_short_segments,
            )
        else:
            logger.info(
                "GPU consumer processed all segments: mel_index=%d",
                mel_index,
            )

        logger.info("GPU consumer thread exiting")

    # ==================================================================
    # PIPELINE EXECUTION
    # ==================================================================
    logger.info("Starting VRAM-optimized Mel pipeline")
    start_time = time.time()

    status_thread = threading.Thread(
        target=status_reporter,
        args=(start_time,),
        daemon=True,
        name="[STATUS]",
    )
    status_thread.start()

    gpu_thread = threading.Thread(
        target=gpu_consumer,
        daemon=True,
        name="gpu_consumer",
    )
    gpu_thread.start()

    # Launch disk prefetchers
    logger.info(
        "Launching prefetchers: threads=%d, FILES_PER_TASK=%d",
        cfg.PREFETCH_THREADS,
        cfg.FILES_PER_TASK,
    )
    with ThreadPoolExecutor(max_workers=cfg.PREFETCH_THREADS) as pool:
        futures = [
            pool.submit(prefetch_audio, start_idx)
            for start_idx in range(0, total_files, cfg.FILES_PER_TASK)
        ]

        # Warmup: wait until queue has some data or all futures done
        logger.info(
            "Waiting for RAM queue warmup (target size >= %d or all futures done)",
            cfg.RAM_AUDIO_Q_SIZE,
        )
        while (
            ram_audio_q.qsize() < cfg.RAM_AUDIO_Q_SIZE
            and not all(f.done() for f in futures)
        ):
            logger.debug(
                "Warmup: qsize=%d, target=%d",
                ram_audio_q.qsize(),
                cfg.RAM_AUDIO_Q_SIZE,
            )
            time.sleep(0.1)

        logger.info(
            "Warmup complete: qsize=%d, all_futures_done=%s",
            ram_audio_q.qsize(),
            all(f.done() for f in futures),
        )

        # Propagate any exceptions and ensure all prefetch tasks completed
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Prefetch future raised an exception: %s", exc)
                raise

    logger.info("All prefetch futures completed; setting producer_complete")
    producer_complete.set()

    logger.info("Waiting for GPU consumer thread to join")
    gpu_thread.join(timeout=600.0)

    if gpu_thread.is_alive():
        logger.error(
            "gpu_consumer thread is still alive after join timeout; "
            "producer_complete=%s, qsize=%d",
            producer_complete.is_set(),
            ram_audio_q.qsize(),
        )
    else:
        logger.info("gpu_consumer thread joined successfully")

    logger.info("Flushing memmaps and mapping array to disk")
    mel_memmap.flush()
    if use_memmap_mapping:
        if isinstance(mapping_array, np.memmap):
            mapping_array.flush()
    else:
        np.save(mapping_path, mapping_array)

    done_flag.set()
    logger.info("Pipeline done_flag set; waiting briefly for status thread to exit")
    time.sleep(1.5)
    logger.info("[DONE] Processing complete.")


if __name__ == "__main__":
    run_pipeline()
