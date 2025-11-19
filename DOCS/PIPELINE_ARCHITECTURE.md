# Pipeline Architecture Documentation

## Overview

High-throughput audio-to-Mel-spectrogram feature extraction pipeline optimized for NVIDIA RTX 5090 GPU with zero-copy I/O and FP16 end-to-end processing.

**Performance:** ~8 seconds to process 39,563 WAV files → 9,376,431 mel spectrograms

---

## Code Review Summary

### ✅ Strengths

1. **Zero-Copy mmap I/O** - Memory-mapped file reads with kernel hints (MADV_SEQUENTIAL, MADV_WILLNEED)
2. **FP16 End-to-End** - 50% memory reduction, 50% faster I/O throughout pipeline
3. **Async GPU Transfers** - Overlapped H2D/D2H with 32 CUDA streams
4. **Vectorized Operations** - Batch conversions (int16→float16), batch memmap writes
5. **Thread Safety** - AtomicCounter for cross-thread statistics
6. **Error Handling** - Narrow exception handling, future cancellation on error
7. **Durability** - fsync after memmap flush for guaranteed disk writes
8. **NVMath Acceleration** - 4.1x faster FP16 GEMM for mel computation

### ⚠️ Configuration Issues

1. **FILES_PER_TASK = 356** - Comment says 340 is optimal (empirical testing)
2. **RAM_QUEUE_SIZE = 2048** - Extremely high, wastes ~57GB RAM (should be 96)
3. **BATCH_ACCUMULATION = 1** - Good (empirical optimal), but was 20 previously

### 🐛 Potential Issues

1. **Broad Exception Handling (line 518)** - `except Exception` for NVML queries
2. **Magic Number (line 451)** - `10_000_000` threshold for memmap vs in-RAM
3. **Race Condition Risk** - GPU consumer and prefetchers start without synchronization

---

## Architecture

### Pipeline Stages

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│   Disk I/O  │────▶│ RAM Queue│────▶│  GPU Compute│────▶│  Memmap  │
│  (mmap)     │     │  (2048)  │     │  (32 streams)│     │ (fsync)  │
└─────────────┘     └──────────┘     └─────────────┘     └──────────┘
  2 threads           FP16 buffers      Async H2D/D2H      FP16 storage
  356 files/batch     ~57GB capacity    Batch accum=1      Zero-copy writes
```

### Data Flow

1. **Discovery** - Scan dataset for WAV files
2. **Prefetch** (Disk → RAM)
   - 2 worker threads
   - 356 files per task (~21MB)
   - Memory-mapped zero-copy reads
   - Vectorized int16→float16 conversion
   - Pinned memory buffers

3. **GPU Processing** (RAM → GPU)
   - Batch accumulation (1x = 356 files)
   - Async H2D transfer via dedicated stream
   - Segment extraction (unfold operations)
   - Mel spectrogram computation (FP16 autocast)
   - Async D2H transfer overlapped with CPU work

4. **Storage** (GPU → Disk)
   - Vectorized mapping array construction
   - FP16 clamping (-65504 to 65504)
   - Single memmap write operations
   - fsync for durability

---

## Performance Characteristics

### Memory Usage

| Component | Size | Details |
|-----------|------|---------|
| Single file | 81 KB | 40,960 samples × 2 bytes (int16) |
| Disk batch | ~28 MB | 356 files × 81 KB |
| RAM queue | ~57 GB | 2048 slots × 28 MB |
| GPU batch | ~28 MB | 356 files in FP16 |
| Mel output | ~282 MB | 9.3M segments × 32 mels × 1 frame × 2 bytes (FP16) |

### Timing Breakdown (Typical Run)

| Phase | Duration | Activity |
|-------|----------|----------|
| Init | 0-2s | CUDA init, file discovery, memmap allocation |
| Disk I/O | 2-4s | Read all 39,563 WAV files (2 threads) |
| GPU Compute | 3-8s | Process all segments, overlapped with disk |
| Flush | 8-9s | Write memmaps to disk with fsync |

### Bottlenecks

1. **Disk I/O (2-4s)** - Only 2 threads reading 356 files/batch
2. **GPU Idle (2-3s)** - Waits for first batches to arrive in queue
3. **fsync (0.5-1s)** - Blocking write-to-disk at completion

---

## Configuration

### Disk I/O Settings

```python
PREFETCH_THREADS = 2      # Number of parallel disk readers
FILES_PER_TASK = 356       # Files per batch (should be 340)
RAM_QUEUE_SIZE = 2048      # Queue slots (should be 96)
HEADER_SIZE = 44           # WAV header skip bytes
```

**Empirical Optimal:**
- `PREFETCH_THREADS = 2`
- `FILES_PER_TASK = 340`
- `RAM_QUEUE_SIZE = 96`

### GPU Settings

```python
BATCH_ACCUMULATION = 1     # Batches to combine before GPU processing
CUDA_STREAMS = 32          # Concurrent GPU streams
ASYNC_COPIES = True        # Enable async H2D/D2H transfers
PRECISION = "fp16"         # FP16 end-to-end
```

### Audio Parameters

```python
SAMPLE_RATE = 4096         # Hz
SAMPLE_LENGTH_SEC = 10     # seconds per file
N_FFT = 64                 # FFT window size
HOP_LENGTH = 16            # Hop between frames
N_MELS = 32                # Mel frequency bins
```

### Segmentation Geometry

```python
NUM_SAMPLES = 40,960           # Samples per file
long_win = 1024                # Long segment window
long_hop = 512                 # Long segment stride
short_win = 512                # Short segment window
short_hop = 256                # Short segment stride
num_long_segments = 79         # Per file
num_short_segments_per_long = 3
total_segments = 9,376,431     # Total output segments
```

---

## Key Components

### 1. Config Class (Lines 95-158)

Centralized configuration with automatic CUDA optimization initialization.

**Features:**
- Hardware detection (CPU threads, GPU device)
- CUDA backend optimization (cudnn.benchmark, allow_tf32)
- 32 CUDA compute streams pre-allocated
- Precision mode selection (fp16/bf16/fp8)

### 2. safe_init_memmap (Lines 200-222)

Creates numpy memmaps with validation to prevent "cannot mmap empty file" errors.

**Parameters:**
- `path`: Output file path
- `shape`: Array dimensions
- `dtype`: Data type (default float32, pipeline uses float16)
- `mode`: File mode (w+, r+, r, c)

**Validation:**
- Checks shape is non-empty
- Creates parent directories
- Logs allocation details

### 3. AtomicCounter (Lines 225-243)

Thread-safe counter for cross-thread statistics.

**Methods:**
- `increment(delta)`: Atomic add and return new value
- `get()`: Atomic read current value

**Used for:**
- `nvme_bytes_read`: Disk throughput tracking
- `gpu_bytes_processed`: GPU throughput tracking
- `files_processed`: Progress tracking
- `batches_processed`: Batch rate calculation

### 4. prefetch_audio (Lines 589-697)

Disk I/O worker function running in thread pool.

**Process:**
1. Pre-allocate pinned FP16 buffer for batch
2. Pre-allocate int16 numpy buffer
3. For each file:
   - Open file descriptor
   - Get file size (fstat)
   - Memory-map entire file (mmap.mmap)
   - Apply kernel hints (MADV_SEQUENTIAL, MADV_WILLNEED)
   - Zero-copy numpy view into int16_buffer
   - Handle short files with padding
4. Vectorized batch conversion: int16→float16
5. Copy to pinned PyTorch tensor
6. Enqueue batch to RAM queue

**Optimizations:**
- Zero-copy mmap (no intermediate buffers)
- Kernel prefetch hints
- Vectorized type conversion
- Pinned memory for fast GPU transfer

### 5. gpu_consumer (Lines 700-896)

GPU processing worker running in dedicated thread.

**Process:**
1. Accumulate batches from RAM queue (batch_accumulation count)
2. Concatenate accumulated buffers
3. Async H2D transfer to GPU
4. Segment extraction via unfold operations
5. Mel spectrogram computation (FP16 autocast)
6. Async D2H transfer (overlapped with CPU work)
7. Build mapping array (vectorized PyTorch operations)
8. Clamp FP16 values to valid range
9. Vectorized memmap writes

**Optimizations:**
- Batch accumulation reduces kernel launch overhead
- Async transfers overlap with compute
- Pre-allocated mapping buffer
- Vectorized numpy stack for mapping construction
- Stream cycling (32 streams)

### 6. status_reporter (Lines 494-586)

Background monitoring thread logging pipeline status every second.

**Metrics:**
- CPU/GPU/RAM utilization
- VRAM usage
- Queue depth
- File/batch progress
- Disk/GPU throughput (GB/s)
- Pipeline phase (INIT/RUN/DRAIN/DONE)

### 7. run_pipeline (Lines 397-985)

Main orchestrator function.

**Phases:**
1. **Initialization:**
   - Initialize NVML (GPU monitoring)
   - Discover WAV files
   - Calculate segmentation geometry
   - Allocate memmaps (features + mapping)
   - Initialize GPU transforms

2. **Execution:**
   - Start status reporter thread
   - Start GPU consumer thread
   - Launch prefetch thread pool
   - Wait for completion

3. **Cleanup:**
   - Join GPU consumer thread
   - Flush memmaps
   - fsync for durability
   - Signal completion

---

## Thread Safety

### Synchronization Primitives

| Primitive | Purpose |
|-----------|---------|
| `ram_audio_q` | Queue for disk→GPU data flow (thread-safe) |
| `done_flag` | Event signaling pipeline completion |
| `producer_complete` | Event signaling disk I/O finished |
| `AtomicCounter._lock` | Mutex for counter operations |

### Thread Lifecycle

```
Main Thread
├─ Status Reporter (daemon)
├─ GPU Consumer (daemon)
└─ Thread Pool (PREFETCH_THREADS workers)
   ├─ prefetch_audio[0]
   └─ prefetch_audio[1]
```

All daemon threads automatically terminate when main thread exits.

---

## Error Handling

### Exception Strategy

1. **Disk I/O** - Narrow exceptions: `(IOError, OSError, ValueError)`
   - Log warning, fill with zeros, continue processing
   
2. **Prefetch Futures** - Narrow exceptions: `(IOError, OSError, RuntimeError)`
   - Log error, cancel remaining futures, raise
   
3. **KeyboardInterrupt** - Graceful shutdown
   - Cancel all futures, propagate interrupt

4. **GPU Consumer** - Validation checks
   - Zero segment detection: skip batch
   - Segment overflow: clip to valid range
   - Index mismatch: log warning

### Durability Guarantees

1. **memmap.flush()** - Writes to OS page cache
2. **os.fsync()** - Forces OS to write to physical disk
3. Applied to:
   - Mel features memmap
   - Mapping array memmap/npy
   
Without fsync, data could be lost on system crash before OS writes cache.

---

## Optimization Opportunities

### High Impact

1. **Fix Configuration** ✅
   - `FILES_PER_TASK = 340` (proven optimal)
   - `RAM_QUEUE_SIZE = 96` (saves 55GB RAM)
   
2. **Increase Prefetch Threads** (Test 3-4 threads)
   - Currently only 2 threads on 24-core system
   - More parallelism could reduce 2-4s disk phase
   
3. **Batch Prefetching** (Start GPU consumer with 2-3 batches ready)
   - Eliminates 2-3s GPU idle time at startup

### Medium Impact

4. **Async fsync** (Non-blocking disk sync)
   - Current fsync blocks for 0.5-1s
   - Could use separate thread for final writes

5. **Larger Batch Accumulation** (Test BA=2-4)
   - Current BA=1 with 356 files
   - BA=2 would process 712 files/GPU batch
   - Better kernel efficiency but higher latency

6. **Direct GPU Storage** (GPUDirect if available)
   - Skip CPU entirely: Disk → GPU direct
   - Requires special drivers and NVMe setup

### Low Impact

7. **Narrow NVML Exception** (Line 518)
8. **Pre-compile CUDA Kernels** (First-run warmup)
9. **Tune CUDA Stream Count** (Test 16, 64, 128)

---

## Dependencies

### Required

- **Python 3.12+**
- **PyTorch 2.9.1+** - CUDA backend, autocast, streams
- **TorchAudio** - Mel spectrogram transforms
- **NumPy 2.1.3+** - Memory-mapped I/O, array operations
- **psutil** - CPU/RAM monitoring
- **pynvml** - GPU monitoring (nvidia-ml-py)

### Optional

- **NVMath** - 4.1x faster FP16 GEMM (highly recommended)
- **CuPy** - Custom GPU kernels (available but unused)
- **TensorRT** - Optimized inference (disabled)

### System Requirements

- **GPU:** NVIDIA RTX 5090 (24GB VRAM, 82 SMs, CUDA 12.8+)
- **RAM:** 16GB+ (pipeline uses ~13-15GB with correct config)
- **Disk:** Fast NVMe SSD (2+ GB/s sequential read)
- **OS:** Linux (WSL2 Ubuntu tested), madvise support

---

## File Outputs

### 1. PIPELINE_FEATURES.DAT

- **Format:** NumPy memmap, dtype=float16
- **Shape:** (9,376,431, 32, 1)
- **Size:** ~282 MB (50% smaller than float32)
- **Content:** Mel spectrograms in dB scale
- **Range:** Clamped to [-65504, 65504] for FP16 safety

### 2. PIPELINE_MEMMAP.npy or _temp.dat

- **Format:** NumPy array or memmap, dtype=int64
- **Shape:** (9,376,431, 6)
- **Size:** ~428 MB
- **Columns:**
  1. `idx`: Global segment index
  2. `file_idx`: Source file index
  3. `long_idx`: Long segment index within file
  4. `short_idx`: Short segment index within long segment
  5. `start_sample`: Sample offset in file
  6. `end_sample`: Sample offset + segment length

- **Usage:** Maps each mel spectrogram back to source WAV file location

---

## Logging

### Log Levels

- `INFO` - Pipeline progress, configuration, milestones
- `DEBUG` - Per-batch details, file counts, debugging info
- `WARNING` - Short files, failed reads, non-fatal errors
- `ERROR` - Fatal errors, invalid configuration

### Environment Variable

```bash
export PIPELINE_LOG_LEVEL=DEBUG  # Enable verbose logging
python pipeline.py
```

### Status Format

```
[ TIME ] | PHASE | CPU | GPU | Occ | RAM | VRAM | Buff | Files | Batch | NVMe | GPU | BA | Th | FpT
```

Example:
```
[   4.0s] | RUN | CPU  20.0% | GPU   1% | Occ   1% | RAM  11.8% | 
VRAM   2722MB | Buff 90/96 | Files 38247/39563 | Batch 0 (0/s) | 
NVMe  1.47 GB/s | GPU  0.00 GB/s | BA 1 | Th 2 | FpT 356
```

---

## Performance Tuning Guide

### Disk I/O Phase (Target: <2s)

**Current:** 2-4 seconds

**Tuning:**
1. Increase `PREFETCH_THREADS` to 3-4
2. Ensure `FILES_PER_TASK = 340` (empirical optimal)
3. Reduce `RAM_QUEUE_SIZE = 96` (less contention)
4. Use faster NVMe SSD if available

### GPU Compute Phase (Target: <5s)

**Current:** 3-8 seconds

**Tuning:**
1. Prefetch 2-3 batches before starting GPU consumer
2. Increase `BATCH_ACCUMULATION` to 2-4 for larger GPU batches
3. Tune `CUDA_STREAMS` (try 16, 64, 128)
4. Ensure NVMath is active (4.1x speedup)

### Memory Usage (Target: <16GB)

**Current:** ~13-15GB with correct config, ~70GB with current config

**Tuning:**
1. Fix `RAM_QUEUE_SIZE = 96` (saves 55GB)
2. Lower `BATCH_ACCUMULATION` if memory pressure
3. Use memmap for mapping array if >10M segments

### End-to-End Runtime (Target: <7s)

**Current:** ~8-9 seconds

**Achievable:** 6-7 seconds with:
- 3-4 prefetch threads
- Batch prefetching (2-3 batches ready)
- Async fsync
- Optimal configuration values

---

## Known Issues

### 1. Configuration Drift

**Issue:** Constants don't match empirical optimal values
- `FILES_PER_TASK = 356` (should be 340)
- `RAM_QUEUE_SIZE = 2048` (should be 96)

**Impact:** Slower runtime, 55GB excess RAM usage

**Fix:** Update constants to match empirical testing

### 2. GPU Startup Latency

**Issue:** GPU sits idle 2-3 seconds waiting for first batches

**Impact:** 20-30% of total runtime wasted

**Fix:** Prefetch 2-3 batches before starting GPU consumer

### 3. Blocking fsync

**Issue:** fsync blocks main thread for 0.5-1 second

**Impact:** 10-15% of total runtime in I/O wait

**Fix:** Move fsync to background thread or accept trade-off

### 4. Broad Exception Handling

**Issue:** `except Exception` catches too much (line 518)

**Impact:** Could mask important errors

**Fix:** Narrow to specific exception types

---

## Testing

### Unit Test Checklist

- [ ] `safe_init_memmap` with empty shape
- [ ] `safe_init_memmap` with valid shape
- [ ] `AtomicCounter` thread safety
- [ ] Short file handling (<40960 samples)
- [ ] FP16 clamping at boundaries
- [ ] Queue full behavior
- [ ] Future cancellation on error
- [ ] KeyboardInterrupt handling

### Integration Test Checklist

- [ ] Small dataset (10 files)
- [ ] Medium dataset (1000 files)
- [ ] Large dataset (39563 files)
- [ ] Different batch sizes (256, 340, 512)
- [ ] Different thread counts (1, 2, 4, 8)
- [ ] Different batch accumulation (1, 2, 4, 8)
- [ ] Verify output shapes match expected
- [ ] Verify mapping array correctness
- [ ] Memory leak detection (run multiple times)
- [ ] Crash recovery (kill mid-run, verify no corruption)

### Performance Benchmarks

- [ ] Disk I/O throughput (GB/s)
- [ ] GPU utilization (%)
- [ ] Memory usage (peak GB)
- [ ] End-to-end runtime (seconds)
- [ ] Scalability (files vs runtime)

---

## Maintenance

### Adding New Precision Modes

1. Add dtype to `Config.PRECISION` options
2. Add case to `init_mel_transforms()` autocast logic
3. Update documentation
4. Test with sample dataset

### Adding New Audio Formats

1. Detect format in `prefetch_audio()`
2. Calculate new HEADER_SIZE for format
3. Adjust bytes_per_file calculation
4. Add format-specific error handling
5. Update documentation

### Upgrading PyTorch/CUDA

1. Check CUDA stream API compatibility
2. Verify autocast dtype support
3. Test async copy behavior
4. Validate NVMath integration
5. Re-benchmark performance

---

## References

- **madvise man page:** https://man7.org/linux/man-pages/man2/madvise.2.html
- **PyTorch CUDA Streams:** https://pytorch.org/docs/stable/notes/cuda.html
- **NumPy memmap:** https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
- **TorchAudio Transforms:** https://pytorch.org/audio/stable/transforms.html
- **NVMath:** https://docs.nvidia.com/nvmath-python/
- **FP16 Range:** https://en.wikipedia.org/wiki/Half-precision_floating-point_format

---

**Last Updated:** November 19, 2025  
**Version:** 2.0  
**Author:** AI Development Team
