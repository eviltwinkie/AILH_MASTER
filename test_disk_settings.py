import os
import time
import queue
import itertools
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ===== CONFIG =====
DATASET_PATH = Path("/DEVELOPMENT/DATASET_REFERENCE/TRAINING")
HEADER_SIZE = 44
SAMPLE_RATE = 4096
SAMPLE_LENGTH_SEC = 10
NUM_SAMPLES = SAMPLE_RATE * SAMPLE_LENGTH_SEC
BYTES_PER_SAMPLE = 2  # int16

# Candidate values to test
PREFETCH_THREADS_VALUES = [ 64, 32, 16, 8, 4 ]
FILES_PER_TASK_VALUES = [ 1024, 512, 256, 128, 96, 64, 32, 16, 8, 4, 2, 1 ]
PREFETCH_DEPTH_VALUES = [ 128, 64, 32, 16, 8, 4 ]

# ===== GET FILE LIST =====
wav_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(DATASET_PATH)
    for f in files if f.lower().endswith(".wav")
]
total_files = len(wav_files)
print(f"[INFO] Found {total_files} WAV files.")

# ===== PRODUCER =====
def read_wav_batch(file_indices, prefetch_q, nvme_counter):
    bytes_read = 0
    for idx in file_indices:
        with open(wav_files[idx], "rb") as f:
            f.seek(HEADER_SIZE)
            raw = f.read(NUM_SAMPLES * BYTES_PER_SAMPLE)
        bytes_read += len(raw)
    nvme_counter["bytes"] += bytes_read
    prefetch_q.put(1)  # simulate 1 batch produced

# ===== CONSUMER =====
def consumer(prefetch_q, done_flag):
    while not done_flag.is_set() or not prefetch_q.empty():
        try:
            prefetch_q.get(timeout=0.5)
            prefetch_q.task_done()
        except queue.Empty:
            continue

# ===== TEST ONE CONFIG =====
def test_config(prefetch_threads, files_per_task, prefetch_depth):
    prefetch_q = queue.Queue(maxsize=prefetch_depth)
    nvme_counter = {"bytes": 0}
    done_flag = threading.Event()

    # Prepare batches
    file_indices = list(range(total_files))
    batches = [
        file_indices[i:i+files_per_task]
        for i in range(0, total_files, files_per_task)
    ]

    # Start consumer thread (simulates GPU drain)
    consumer_thread = threading.Thread(target=consumer, args=(prefetch_q, done_flag))
    consumer_thread.start()

    start_time = time.time()

    # Run producers with limited prefetch
    with ThreadPoolExecutor(max_workers=prefetch_threads) as pool:
        for batch in batches:
            while prefetch_q.full():
                time.sleep(0.001)
            pool.submit(read_wav_batch, batch, prefetch_q, nvme_counter)

    prefetch_q.join()  # Wait for all queued batches to drain
    done_flag.set()
    consumer_thread.join()

    elapsed = time.time() - start_time
    throughput_gbps = (nvme_counter["bytes"] / elapsed) / (1024**3)
    return elapsed, throughput_gbps

# ===== MAIN TEST LOOP =====
results = []
for prefetch_threads, files_per_task, prefetch_depth in itertools.product(
    PREFETCH_THREADS_VALUES,
    FILES_PER_TASK_VALUES,
    PREFETCH_DEPTH_VALUES
):
    print(f"[TEST] Thr={prefetch_threads} | FPT={files_per_task} | Depth={prefetch_depth}")
    elapsed, gbps = test_config(prefetch_threads, files_per_task, prefetch_depth)
    results.append((prefetch_threads, files_per_task, prefetch_depth, elapsed, gbps))

# ===== SORT & PRINT RESULTS =====
results.sort(key=lambda x: (-x[4], x[3]))  # sort by GB/s desc, then time
print("\n=== RESULTS (Best throughput first) ===")
print(f"{'Thr':>4} {'FPT':>5} {'Depth':>5} {'Time(s)':>8} {'GB/s':>8}")
for r in results:
    print(f"{r[0]:>4} {r[1]:>5} {r[2]:>5} {r[3]:>8.2f} {r[4]:>8.2f}")
