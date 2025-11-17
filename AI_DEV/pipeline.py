#!/usr/bin/env python3
"""
High-throughput audio → Mel-spectrogram feature extraction pipeline.

This version adds detailed diagnostics and logging to help debug
end-of-run hangs and verify that all files, batches, and segments
are being processed and flushed correctly.
"""

import os
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torchaudio
import psutil
import pynvml
import logging


# ======================================================================
# LOGGING SETUP
# ======================================================================

# Set PIPELINE_LOG_LEVEL=DEBUG in your environment for deep diagnostics.
LOG_LEVEL = os.environ.get("PIPELINE_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
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
        self.CPU_THREADS: int = 12

        # ---------- GPU settings ----------
        self.GPU_BATCH_SIZE: int = 256
        self.CUDA_STREAMS: int = 4
        self.GPU_MEMORY_FRACTION: float = 0.85

        # ---------- CPU↔GPU buffer settings ----------
        self.CPU_GPU_BUFFER_SIZE: int = 240  # in number of audio files
        self.CPU_GPU_BUFFER_TIMEOUT_MS: int = 25  # in milliseconds

        # ---------- RAM prefetch settings ----------
        self.RAM_PREFETCH_DEPTH: int = 12
        self.RAM_AUDIO_Q_SIZE: int = 4

        # ---------- Disk prefetch settings ----------
        self.PREFETCH_THREADS: int = 1
        self.PREFETCH_DEPTH: int = 1
        self.FILES_PER_TASK: int = 4096

        # ---------- Audio / segmentation parameters ----------
        self.SAMPLE_RATE: int = 4096
        self.SAMPLE_LENGTH_SEC: int = 10

        self.LONG_SEGMENT_SCALE_SEC: float = 0.25
        self.SHORT_SEGMENT_POINTS: int = 512

        self.N_FFT: int = 512
        self.HOP_LENGTH: int = 128
        self.N_MELS: int = 32

        # ---------- Output / dataset paths ----------
        self.OUTPUT_DIR: Path = Path(
            "/DEVELOPMENT/ROOT_AILH/DATA_STORE/MEMMAPS"
        )
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.DATASET_PATH: Path = Path(
            "/DEVELOPMENT/ROOT_AILH/DATA_STORE/TRAINING"
        )

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

    mel_time_frames = short_win // cfg.HOP_LENGTH
    mel_shape = (total_short_segments, cfg.N_MELS, mel_time_frames)

    logger.info("Segmentation geometry:")
    logger.info("  NUM_SAMPLES=%d", NUM_SAMPLES)
    logger.info("  long_win=%d, long_hop=%d", long_win, long_hop)
    logger.info("  short_win=%d, short_hop=%d", short_win, short_hop)
    logger.info("  num_long_segments=%d", num_long_segments)
    logger.info(
        "  num_short_segments_per_long=%d", num_short_segments_per_long
    )
    logger.info("  total_short_segments=%d", total_short_segments)
    logger.info("  mel_shape=%s", mel_shape)

    # ------------------------------------------------------------------
    # MEMMAP ALLOCATION
    # ------------------------------------------------------------------
    mel_memmap_path = cfg.OUTPUT_DIR / "mel_features.dat"
    mel_memmap = safe_init_memmap(
        mel_memmap_path, mel_shape, dtype=np.dtype(np.float32), mode="w+"
    )

    mapping_path = cfg.OUTPUT_DIR / "mel_mapping.npy"

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
    ram_audio_q: "queue.Queue[tuple[int, torch.Tensor]]" = queue.Queue(
        maxsize=cfg.RAM_PREFETCH_DEPTH
    )

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
                except Exception as exc:  # noqa: BLE001
                    logger.debug("NVML query failed: %s", exc)
                    gpu_util = 0
                    vram_used = 0
            else:
                gpu_util = 0
                vram_used = 0

            current_nvme = nvme_bytes_read.get()
            current_gpu = gpu_bytes_processed.get()
            current_files = files_processed.get()

            nvme_rate = (current_nvme - last_nvme) / (1024**3)
            gpu_rate = (current_gpu - last_gpu) / (1024**3)

            last_nvme = current_nvme
            last_gpu = current_gpu

            logger.info(
                "[STATUS] %6.1fs | CPU %5.1f%% | GPU %3d%% | RAM %5.1f%% | "
                "VRAM %6.0fMB | Buff %d/%d | Files %d/%d | "
                "NVMe %5.2f GB/s | GPU %5.2f GB/s | Thr %d | FPT %d | Depth %d",
                elapsed,
                cpu,
                gpu_util,
                ram,
                vram_used,
                ram_audio_q.qsize(),
                ram_audio_q.maxsize,
                current_files,
                total_files,
                nvme_rate,
                gpu_rate,
                cfg.PREFETCH_THREADS,
                cfg.FILES_PER_TASK,
                cfg.RAM_PREFETCH_DEPTH,
            )

            time.sleep(1.0)

        logger.info("Status reporter thread exiting")

    # ==================================================================
    # DISK → RAM PREFETCH (PRODUCER)
    # ==================================================================

    def prefetch_audio(start_idx: int) -> None:
        """
        Read a contiguous block of WAV files, decode to float32, and push to RAM.
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
                with open(file_path, "rb") as f:
                    f.seek(HEADER_SIZE)
                    raw = f.read(NUM_SAMPLES * BYTES_PER_SAMPLE)

                if len(raw) == NUM_SAMPLES * BYTES_PER_SAMPLE:
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

                with torch.amp.autocast("cuda", dtype=torch.float16):
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
        name="status_reporter",
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
