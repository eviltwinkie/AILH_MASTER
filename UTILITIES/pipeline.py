import os
import time
import queue
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import torchaudio
import psutil
import pynvml
import json

# ===== CONFIG =====
class Config:
    def __init__(self):
        self.SAMPLE_RATE = 4096
        self.SAMPLE_LENGTH_SEC = 10
        self.LONG_SEGMENT_SCALE_SEC = 0.25
        self.SHORT_SEGMENT_POINTS = 512
        self.N_FFT = 512
        self.HOP_LENGTH = 128
        self.N_MELS = 32

        # Prefetch settings
        self.PREFETCH_THREADS = 8
        self.FILES_PER_TASK = 192
        self.RAM_PREFETCH_DEPTH = 12
        self.WARMUP_THRESHOLD = 4

        # GPU settings
        self.GPU_BATCH_SIZE = 256
        self.CUDA_STREAMS = 4
        self.GPU_MEMORY_FRACTION = 0.85

        # Output
        self.OUTPUT_DIR = Path("/mnt/d/DEV/mel_memmaps_gpu_seg")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASET_PATH = Path("/mnt/d/DATASET_REFERENCE/TRAINING")

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

# ===== INIT =====
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

HEADER_SIZE = 44
NUM_SAMPLES = cfg.SAMPLE_RATE * cfg.SAMPLE_LENGTH_SEC
BYTES_PER_SAMPLE = np.dtype(np.int16).itemsize

wav_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(cfg.DATASET_PATH)
    for f in files if f.lower().endswith(".wav")
]
total_files = len(wav_files)

long_win = int(cfg.SAMPLE_RATE * cfg.LONG_SEGMENT_SCALE_SEC)
long_hop = long_win // 2
short_win = cfg.SHORT_SEGMENT_POINTS
short_hop = short_win // 2
num_long_segments = 1 + (NUM_SAMPLES - long_win) // long_hop
num_short_segments_per_long = 1 + (long_win - short_win) // short_hop
total_short_segments = total_files * num_long_segments * num_short_segments_per_long
mel_time_frames = (short_win // cfg.HOP_LENGTH)
mel_shape = (total_short_segments, cfg.N_MELS, mel_time_frames)

# Memmaps
mel_memmap_path = cfg.OUTPUT_DIR / "mel_features.dat"
mel_memmap = np.memmap(mel_memmap_path, dtype=np.float32, mode="w+", shape=mel_shape)
mapping_path = cfg.OUTPUT_DIR / "mel_mapping.npy"

if total_short_segments < 10_000_000:
    mapping_array = np.empty((total_short_segments, 6), dtype=np.int64)
    use_memmap_mapping = False
else:
    mapping_array = np.memmap(
        str(mapping_path).replace('.npy', '_temp.dat'),
        dtype=np.int64, mode="w+",
        shape=(total_short_segments, 6)
    )
    use_memmap_mapping = True

ram_audio_q = queue.Queue(maxsize=cfg.RAM_PREFETCH_DEPTH)
done_flag = threading.Event()
processing_complete = threading.Event()
producer_complete = threading.Event()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.SAMPLE_RATE,
    n_fft=cfg.N_FFT,
    hop_length=cfg.HOP_LENGTH,
    n_mels=cfg.N_MELS,
    center=False,
    power=2.0,
    norm='slaney'
).to(cfg.DEVICE)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
    stype='power',
    top_db=80.0
).to(cfg.DEVICE)

# ===== ATOMIC COUNTER =====
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

nvme_bytes_read = AtomicCounter()
gpu_bytes_processed = AtomicCounter()
files_processed = AtomicCounter()
batches_processed = AtomicCounter()

counter_lock = threading.Lock()

# ===== STATUS =====
def status_reporter(start_time):
    last_nvme = nvme_bytes_read.get()
    last_gpu = gpu_bytes_processed.get()
    while not done_flag.is_set():
        elapsed = time.time() - start_time
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        vram_used = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**2)

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
            f"| Thr {cfg.PREFETCH_THREADS} | FPT {cfg.FILES_PER_TASK} | Depth {cfg.RAM_PREFETCH_DEPTH}"
        )
        time.sleep(1)

# ===== AUDIO PREFETCH =====
def prefetch_audio(start_idx):
    end_idx = min(start_idx + cfg.FILES_PER_TASK, total_files)
    buf = torch.empty((end_idx - start_idx, NUM_SAMPLES), dtype=torch.float32).pin_memory()
    for i, file_idx in enumerate(range(start_idx, end_idx)):
        try:
            with open(wav_files[file_idx], 'rb') as f:
                f.seek(HEADER_SIZE)
                raw = f.read(NUM_SAMPLES * BYTES_PER_SAMPLE)
            if len(raw) == NUM_SAMPLES * BYTES_PER_SAMPLE:
                buf[i] = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
            else:
                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                padded = np.zeros(NUM_SAMPLES, dtype=np.float32)
                padded[:len(audio_data)] = audio_data
                buf[i] = torch.from_numpy(padded)
            nvme_bytes_read.increment(len(raw))
            files_processed.increment()
        except:
            buf[i] = torch.zeros(NUM_SAMPLES, dtype=np.float32)
            files_processed.increment()
    try:
        ram_audio_q.put((start_idx, buf), timeout=30)
    except queue.Full:
        pass

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
            mapping_array[mel_index:mel_index + total_segments, :] = np.column_stack((
                idx_range,
                np.repeat(np.arange(start_file_idx, start_file_idx + batch_size), num_long_segments * num_short_segments_per_long),
                np.tile(np.repeat(np.arange(num_long_segments), num_short_segments_per_long), batch_size),
                np.tile(np.arange(num_short_segments_per_long), batch_size * num_long_segments),
                (np.tile(np.repeat(np.arange(num_long_segments), num_short_segments_per_long), batch_size) * long_hop) +
                (np.tile(np.arange(num_short_segments_per_long), batch_size * num_long_segments) * short_hop),
                ((np.tile(np.repeat(np.arange(num_long_segments), num_short_segments_per_long), batch_size) * long_hop) +
                 (np.tile(np.arange(num_short_segments_per_long), batch_size * num_long_segments) * short_hop)) + short_win
            ))

            # Vectorized mel-spectrogram computation
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=batch_segments.size(0) >= 1024):
                mel_spec = mel_transform(batch_segments)
                mel_spec_db = amplitude_to_db(mel_spec)
            mel_result = mel_spec_db.float().contiguous().cpu().numpy()
            mel_memmap[mel_index:mel_index + total_segments] = mel_result
            gpu_bytes_processed.increment(batch_segments.numel() * batch_segments.element_size())

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
threading.Thread(target=status_reporter, args=(start_time,), daemon=True).start()
gpu_thread = threading.Thread(target=gpu_consumer, daemon=True)
gpu_thread.start()

with ThreadPoolExecutor(max_workers=cfg.PREFETCH_THREADS) as pool:
    futures = [pool.submit(prefetch_audio, i) for i in range(0, total_files, cfg.FILES_PER_TASK)]
    while ram_audio_q.qsize() < cfg.WARMUP_THRESHOLD:
        time.sleep(0.1)
    for f in futures:
        f.result()

producer_complete.set()
gpu_thread.join()
mel_memmap.flush()
if not use_memmap_mapping:
    np.save(mapping_path, mapping_array)
done_flag.set()

print("[DONE] Processing complete.")
