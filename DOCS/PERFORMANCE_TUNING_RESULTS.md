# Performance Tuning Results & Final Recommendations

## Test Results Comparison

### Before Optimization
```
Configuration:
- cpu_max_workers: 4
- disk_files_per_task: 1024
- disk_max_inflight: 16

Results (TRAINING):
- Disk I/O: 11.5s (95.6%)
- GPU: 0.5s (4.4%)
- Queue Depth: avg=2.2
- GPU Usage: 2.9%
```

### After Initial Optimization
```
Configuration:
- cpu_max_workers: 12  (+200%)
- disk_files_per_task: 512  (-50%)
- disk_max_inflight: 32  (+100%)

Results (TRAINING):
- Disk I/O: 29.6s (98.3%) ⬆️ WORSE
- GPU: 0.5s (1.7%)
- Queue Depth: avg=5.9  ✅ IMPROVED
- GPU Usage: 4.9%  ✅ IMPROVED
```

## Root Cause Analysis

### Why Disk I/O Got Worse
The optimization **increased thread count but decreased batch size**, resulting in:
- **More batches**: 39 → 78 batches (2x increase)
- **More overhead**: Each batch has setup/teardown cost
- **Smaller batches**: 1024 → 512 files/batch
- **Result**: More context switching, worse disk throughput

### Why Queue Depth Improved
- More threads (4 → 12) = more in-flight work
- Deeper queue (16 → 32) = better pipeline overlap
- **Result**: GPU gets fed more consistently (2.2 → 5.9 avg depth)

### Why GPU Usage Improved Slightly
- Better queue depth means less idle time
- 2.9% → 4.9% (1.7x improvement)
- Still severely underutilized

---

## Optimal Configuration (Data-Driven)

### Analysis
The bottleneck is NOT thread count or queue depth - it's **inherent disk latency** and **sequential WAV file reading**.

### Key Insight
```
GPU Processing Speed: 77,000-121,000 files/s
Disk I/O Speed: ~1,300-4,000 files/s (depending on access pattern)

Ratio: GPU is 20-90x faster than disk I/O
```

The GPU can process files **20-90x faster** than the disk can supply them. No amount of threading will fix this - you need architectural changes.

---

## Recommended Configuration

### Balanced Approach
```python
@dataclass
class Config:
    # OPTIMAL: Balance batch size with parallelism
    cpu_max_workers: int = 8          # Sweet spot: 2x original, not excessive
    disk_files_per_task: int = 2048   # Larger batches for better disk efficiency
    disk_max_inflight: int = 24       # Moderate queue depth
    disk_submit_window: int = 20      # Moderate submit window
```

### Rationale
- **cpu_max_workers=8**: 2x original, balances parallelism without overhead
- **disk_files_per_task=2048**: LARGER batches reduce overhead, better for sequential reads
- **Moderate queue**: Enough pipelining without excessive context switching

---

## Architectural Solutions (Real Fix)

### Priority 1: Pre-load Entire Dataset to RAM ⭐⭐⭐
```python
# Load ALL WAV files into RAM first, then process
def preload_dataset_to_ram(files: List[Path]) -> Dict[int, np.ndarray]:
    """Load entire dataset to RAM in one pass"""
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(load_wav, f): i for i, f in enumerate(files)}
        return {i: fut.result() for i, fut in futures.items()}

# Then GPU processing becomes pure RAM → GPU → RAM (no disk)
wav_cache = preload_dataset_to_ram(all_files)  # One-time disk hit
for batch in batches:
    data = [wav_cache[i] for i in batch]  # RAM access only
    gpu_process(data)
```

**Impact**: Eliminates disk bottleneck entirely after initial load
**Trade-off**: Requires RAM for entire dataset (~5-10 GB for 51k files)

---

### Priority 2: Memory-Mapped Dataset ⭐⭐
```python
# Use numpy memmap for zero-copy access
wav_memmap = np.memmap('dataset.mmap', dtype=np.float32, 
                       shape=(num_files, num_samples), mode='r')

# GPU processing reads directly from mmap (kernel handles caching)
for batch in batches:
    data = wav_memmap[batch_indices]  # Zero-copy slice
    gpu_process(data)
```

**Impact**: Kernel manages I/O, automatic caching, zero-copy
**Trade-off**: One-time preprocessing to create memmap file

---

### Priority 3: Async I/O with io_uring ⭐⭐
```python
import liburing  # Linux io_uring for async I/O

# Submit multiple reads asynchronously
ring = liburing.io_uring()
for f in files:
    ring.prep_read(f)
ring.submit()

# GPU processes while kernel handles I/O in background
while ring.has_completions():
    data = ring.get_completion()
    gpu_process(data)
```

**Impact**: True async I/O, overlaps disk and GPU
**Trade-off**: Linux-only, requires liburing

---

### Priority 4: Separate I/O and Processing Threads ⭐
```python
from queue import Queue
from threading import Thread

# Producer: Disk I/O only
def disk_reader_thread(file_queue, data_queue):
    while True:
        files = file_queue.get()
        if files is None: break
        data = [load_wav(f) for f in files]
        data_queue.put(data)

# Consumer: GPU processing only  
def gpu_processor_thread(data_queue):
    while True:
        data = data_queue.get()
        if data is None: break
        gpu_process(data)

# Separate concerns, each runs at full speed
disk_thread = Thread(target=disk_reader_thread)
gpu_thread = Thread(target=gpu_processor_thread)
```

**Impact**: True pipelining, each stage runs independently
**Trade-off**: More complex architecture

---

## Performance Projections

### Current Architecture (Best Case)
```
With optimal threading (cpu_max_workers=8, disk_files_per_task=2048):
- Disk I/O: ~8-10s
- GPU: 0.5s
- Total: ~8.5-10.5s
- GPU Utilization: ~5-10%
```

### With RAM Preload
```
- Preload: ~10-15s (one-time)
- GPU Processing: 0.5s (from RAM)
- Total per epoch: 0.5s (after first preload)
- GPU Utilization: 80-90%
```

### With Memory-Mapped Dataset
```
- Preprocessing: ~20s (one-time, create memmap)
- GPU Processing: ~2-3s (kernel caching helps)
- Total per run: ~2-3s
- GPU Utilization: 20-30%
```

---

## Immediate Action Plan

### Step 1: Test Optimal Threading (5 min)
```python
cpu_max_workers: int = 8
disk_files_per_task: int = 2048
disk_max_inflight: int = 24
disk_submit_window: int = 20
```

### Step 2: Measure Improvement
Run builder and check:
- Disk I/O time (target: < 8s)
- Queue depth (target: > 4)
- GPU usage (target: > 10%)

### Step 3: If Still Bottlenecked...
Implement RAM preload strategy:
```python
def build_split_with_preload(self, split_name, input_dir, ...):
    # Phase 1: Load ALL files to RAM
    logger.info("Preloading %d files to RAM...", len(records))
    wav_cache = self._preload_wavs_parallel(records)
    
    # Phase 2: GPU processing (pure RAM → GPU)
    for batch in batches:
        data = [wav_cache[i] for i in batch]
        gpu_process(data)
```

---

## Conclusion

**The Fundamental Problem**: 
Your GPU can process 77,000 files/sec but disk delivers ~1,300-4,000 files/sec. The gap is too large for threading to bridge.

**The Solution**: 
Change the architecture to decouple disk I/O from GPU processing. Either:
1. Load everything to RAM once (best for training loops)
2. Use memory-mapped files (good for large datasets)
3. Implement true async I/O with separate threads

**Quick Win**:
```python
cpu_max_workers: int = 8
disk_files_per_task: int = 2048
```
This will get you ~10-15% GPU utilization. For 80%+, you need architectural changes.

---

## Test Script

```bash
#!/bin/bash
# Test different configurations

echo "Testing config 1: workers=8, batch=2048"
sed -i 's/cpu_max_workers: int = [0-9]*/cpu_max_workers: int = 8/' dataset_builder.py
sed -i 's/disk_files_per_task: int = [0-9]*/disk_files_per_task: int = 2048/' dataset_builder.py
python dataset_builder.py > results_8_2048.log 2>&1

echo "Testing config 2: workers=16, batch=1024"  
sed -i 's/cpu_max_workers: int = [0-9]*/cpu_max_workers: int = 16/' dataset_builder.py
sed -i 's/disk_files_per_task: int = [0-9]*/disk_files_per_task: int = 1024/' dataset_builder.py
python dataset_builder.py > results_16_1024.log 2>&1

# Compare
grep "Disk I/O:" results_*.log
grep "GPU Throughput:" results_*.log
```

---

## Hardware Utilization Analysis

From test_gpu_cuda_results.txt:
- **CPU**: 24 cores @ x86_64 → Only using 6.4% avg (1.5 cores)
- **RAM**: Plenty available → Using 18.2% avg
- **GPU**: CUDA 13.0 capable → Only 4.9% utilized
- **Storage**: NVMe SSD → Sequential reads bottlenecked

**Verdict**: Hardware is massively underutilized due to serial I/O pattern.
