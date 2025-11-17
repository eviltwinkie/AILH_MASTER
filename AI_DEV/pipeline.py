#!/usr/bin/env python3
import os
import time
import queue
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor , as_completed

import torch
import torchaudio
import psutil

import pynvml

# ===== CONFIG =====
class Config:
    def __init__(self):

        # CPU settings
        self.CPU_THREADS = 12

        # GPU settings
        self.GPU_BATCH_SIZE = 256
        self.CUDA_STREAMS = 4
        self.GPU_MEMORY_FRACTION = 0.85

        # CPU-GPU buffer settings
        self.CPU_GPU_BUFFER_SIZE = 240  # in number of audio files
        self.CPU_GPU_BUFFER_TIMEOUT_MS = 25  # in milliseconds

        # RAM buffer settings
        self.RAM_PREFETCH_DEPTH = 12
        self.RAM_AUDIO_Q_SIZE = 4 # minimum number of files loaded into the shared queue before the pipeline proceeds

        # Disk prefetch settings
        self.PREFETCH_THREADS = 1
        self.PREFETCH_DEPTH = 1
        self.FILES_PER_TASK = 4096

        # Audio / segmentation
        self.SAMPLE_RATE = 4096
        self.SAMPLE_LENGTH_SEC = 10
        self.LONG_SEGMENT_SCALE_SEC = 0.25
        self.SHORT_SEGMENT_POINTS = 512
        self.N_FFT = 512
        self.HOP_LENGTH = 128
        self.N_MELS = 32

        # Output / dataset paths
        self.OUTPUT_DIR = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/MEMMAPS")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASET_PATH = Path("/DEVELOPMENT/ROOT_AILH/DATA_STORE/TRAINING")

        # Device optimization
        self.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(self.GPU_MEMORY_FRACTION)
        torch.cuda.empty_cache()

        # CUDA streams
        self.compute_streams = [torch.cuda.Stream() for _ in range(self.CUDA_STREAMS)]


cfg = Config()


# ===== HELPERS =====
def init_nvml():
    """Initialize NVML and return a handle for GPU0, or None if unavailable."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return handle
    except Exception as e:
        print(f"[WARN] NVML init failed: {e}")
        return None


def safe_init_memmap(path: Path, shape, dtype=np.float32, mode="w+"):
    """Create a memmap only if shape has > 0 elements."""
    total_elems = int(np.prod(shape))
    if total_elems <= 0:
        raise RuntimeError(
            f"[FATAL] Attempted to create zero-sized memmap at {path} "
            f"with shape={shape}, total_elems={total_elems}. "
            "Likely cause: no WAV files / segments discovered."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)  # type: ignore[call-overload]


class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta=1):
        with self._lock:
            self._value += delta
            return self._value

    def get(self):
        with self._lock:
            return self._value


def run_pipeline():
    # ===== INIT =====
    gpu_handle = init_nvml()

    HEADER_SIZE = 44
    NUM_SAMPLES = cfg.SAMPLE_RATE * cfg.SAMPLE_LENGTH_SEC
    BYTES_PER_SAMPLE = np.dtype(np.int16).itemsize

    # Discover WAV files
    wav_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(cfg.DATASET_PATH)
        for f in files
        if f.lower().endswith(".wav")
    ]
    total_files = len(wav_files)

    if total_files == 0:
        print(f"[FATAL] No WAV files found under {cfg.DATASET_PATH}. Aborting.")
        return

    # Segmentation geometry
    long_win = int(cfg.SAMPLE_RATE * cfg.LONG_SEGMENT_SCALE_SEC)
    long_hop = long_win // 2
    short_win = cfg.SHORT_SEGMENT_POINTS
    short_hop = short_win // 2

    num_long_segments = 1 + (NUM_SAMPLES - long_win) // long_hop
    num_short_segments_per_long = 1 + (long_win - short_win) // short_hop
    total_short_segments = total_files * num_long_segments * num_short_segments_per_long

    if total_short_segments <= 0:
        print(
            f"[FATAL] total_short_segments={total_short_segments} "
            f"(files={total_files}, num_long_segments={num_long_segments}, "
            f"num_short_segments_per_long={num_short_segments_per_long}). Aborting."
        )
        return

    mel_time_frames = short_win // cfg.HOP_LENGTH
    mel_shape = (total_short_segments, cfg.N_MELS, mel_time_frames)

    print(f"[INFO] total_files={total_files}")
    print(f"[INFO] long_win={long_win}, long_hop={long_hop}")
    print(f"[INFO] short_win={short_win}, short_hop={short_hop}")
    print(f"[INFO] num_long_segments={num_long_segments}, "
          f"num_short_segments_per_long={num_short_segments_per_long}")
    print(f"[INFO] total_short_segments={total_short_segments}")
    print(f"[INFO] mel_shape={mel_shape}")

    # Memmaps
    mel_memmap_path = cfg.OUTPUT_DIR / "mel_features.dat"
    mel_memmap = safe_init_memmap(mel_memmap_path, mel_shape, dtype=np.float32, mode="w+")

    mapping_path = cfg.OUTPUT_DIR / "mel_mapping.npy"

    if total_short_segments < 10_000_000:
        mapping_array = np.empty((total_short_segments, 6), dtype=np.int64)
        use_memmap_mapping = False
    else:
        mapping_memmap_path = str(mapping_path).replace(".npy", "_temp.dat")
        mapping_array = np.memmap(
            mapping_memmap_path,
            dtype=np.int64,
            mode="w+",
            shape=(total_short_segments, 6),
        )
        use_memmap_mapping = True

    # Queues & flags
    ram_audio_q = queue.Queue(maxsize=cfg.RAM_PREFETCH_DEPTH)
    done_flag = threading.Event()
    producer_complete = threading.Event()

    # Counters
    nvme_bytes_read = AtomicCounter()
    gpu_bytes_processed = AtomicCounter()
    files_processed = AtomicCounter()
    batches_processed = AtomicCounter()

    # Transforms
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

    # ===== STATUS REPORTER =====
    def status_reporter(start_time):
        last_nvme = nvme_bytes_read.get()
        last_gpu = gpu_bytes_processed.get()
        while not done_flag.is_set():
            elapsed = time.time() - start_time
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            if gpu_handle is not None:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    vram_used = (int(pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used) / (1024**2))
                except Exception:
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

            print(
                f"[STATUS] {elapsed:6.1f}s | CPU {cpu:5.1f}% | GPU {gpu_util:3d}% "
                f"| RAM {ram:5.1f}% | VRAM {vram_used:6.0f}MB "
                f"| Buff {ram_audio_q.qsize()}/{ram_audio_q.maxsize} "
                f"| Files {current_files}/{total_files} "
                f"| NVMe {nvme_rate:5.2f} GB/s | GPU {gpu_rate:5.2f} GB/s "
                f"| Thr {cfg.PREFETCH_THREADS} | FPT {cfg.FILES_PER_TASK} "
                f"| Depth {cfg.RAM_PREFETCH_DEPTH}"
            )
            time.sleep(1)

    # ===== AUDIO PREFETCH =====
    def prefetch_audio(start_idx):
        end_idx = min(start_idx + cfg.FILES_PER_TASK, total_files)
        buf = torch.empty(
            (end_idx - start_idx, NUM_SAMPLES), dtype=torch.float32
        ).pin_memory()
        for i, file_idx in enumerate(range(start_idx, end_idx)):
            try:
                with open(wav_files[file_idx], "rb") as f:
                    f.seek(HEADER_SIZE)
                    raw = f.read(NUM_SAMPLES * BYTES_PER_SAMPLE)

                if len(raw) == NUM_SAMPLES * BYTES_PER_SAMPLE:
                    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_np = np.zeros(NUM_SAMPLES, dtype=np.float32)
                    audio_np[: len(audio_data)] = audio_data

                buf[i] = torch.from_numpy(audio_np)
                nvme_bytes_read.increment(len(raw))
                files_processed.increment()
            except Exception as e:
                print(f"[WARN] Failed to read {wav_files[file_idx]}: {e}")
                buf[i] = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
                files_processed.increment()

        try:
            ram_audio_q.put((start_idx, buf), timeout=30)
        except queue.Full:
            print("[WARN] ram_audio_q full; dropping prefetch batch.")

    # ===== GPU CONSUMER =====
    def gpu_consumer():
        mel_index = 0
        next_batch = None
        while True:
            if next_batch is None:
                try:
                    next_batch = ram_audio_q.get(timeout=5)
                except queue.Empty:
                    if producer_complete.is_set() and ram_audio_q.empty():
                        break
                    continue

            start_file_idx, buf = next_batch
            batch_size = buf.size(0)
            stream_id = batches_processed.get() % cfg.CUDA_STREAMS

            with torch.cuda.stream(cfg.compute_streams[stream_id]):
                gpu_buf = buf.to(cfg.DEVICE, non_blocking=True)
                long_segments = gpu_buf.unfold(1, long_win, long_hop)
                short_segments = long_segments.unfold(2, short_win, short_hop)
                batch_segments = short_segments.reshape(-1, short_win)
                total_segments = batch_segments.size(0)

                # Vectorized mapping write
                idx_range = np.arange(mel_index, mel_index + total_segments)
                mapping_array[mel_index : mel_index + total_segments, :] = np.column_stack(
                    (
                        idx_range,
                        np.repeat(
                            np.arange(start_file_idx, start_file_idx + batch_size),
                            num_long_segments * num_short_segments_per_long,
                        ),
                        np.tile(
                            np.repeat(np.arange(num_long_segments), num_short_segments_per_long),
                            batch_size,
                        ),
                        np.tile(
                            np.arange(num_short_segments_per_long),
                            batch_size * num_long_segments,
                        ),
                        (
                            np.tile(
                                np.repeat(np.arange(num_long_segments), num_short_segments_per_long),
                                batch_size,
                            )
                            * long_hop
                        )
                        + (
                            np.tile(
                                np.arange(num_short_segments_per_long),
                                batch_size * num_long_segments,
                            )
                            * short_hop
                        ),
                        (
                            np.tile(
                                np.repeat(np.arange(num_long_segments), num_short_segments_per_long),
                                batch_size,
                            )
                            * long_hop
                        )
                        + (
                            np.tile(
                                np.arange(num_short_segments_per_long),
                                batch_size * num_long_segments,
                            )
                            * short_hop
                        )
                        + short_win,
                    )
                )

                # Vectorized mel-spectrogram computation
                use_autocast = batch_segments.size(0) >= 1024
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    mel_spec = mel_transform(batch_segments)
                    mel_spec_db = amplitude_to_db(mel_spec)
                mel_result = mel_spec_db.float().contiguous().cpu().numpy()
                mel_memmap[mel_index : mel_index + total_segments] = mel_result

                gpu_bytes_processed.increment(
                    batch_segments.numel() * batch_segments.element_size()
                )

            cfg.compute_streams[stream_id].synchronize()
            mel_index += total_segments
            batches_processed.increment()

            # Prefetch next batch asynchronously
            if not producer_complete.is_set() or not ram_audio_q.empty():
                try:
                    next_batch = ram_audio_q.get_nowait()
                except queue.Empty:
                    next_batch = None
            else:
                next_batch = None

    # ===== RUN =====
    print("[INFO] Starting VRAM-optimized pipeline...")
    start_time = time.time()

    status_thread = threading.Thread(
        target=status_reporter, args=(start_time,), daemon=True
    )
    status_thread.start()

    gpu_thread = threading.Thread(target=gpu_consumer, daemon=True)
    gpu_thread.start()

    with ThreadPoolExecutor(max_workers=cfg.PREFETCH_THREADS) as pool:
        futures = [
            pool.submit(prefetch_audio, i)
            for i in range(0, total_files, cfg.FILES_PER_TASK)
        ]
        # Warmup
        while ram_audio_q.qsize() < cfg.RAM_AUDIO_Q_SIZE:
            time.sleep(0.1)
        for f in as_completed(futures):
            f.result()


    producer_complete.set()
    gpu_thread.join()

    mel_memmap.flush()
    if use_memmap_mapping:
        if isinstance(mapping_array, np.memmap):
            mapping_array.flush()
    else:
        np.save(mapping_path, mapping_array)

    done_flag.set()
    print("[DONE] Processing complete.")


if __name__ == "__main__":
    run_pipeline()
