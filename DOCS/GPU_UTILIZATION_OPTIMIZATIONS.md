# GPU Utilization Optimizations - Phase 2

## Problem Statement
GPU utilization was only **11%** during training, indicating severe CPU→GPU data pipeline bottleneck.

## Root Cause Analysis
1. **Batch size too small** - GPU finishes computation quickly, waits for next batch
2. **DataLoader not prefetching enough** - CPU not keeping up with GPU demand
3. **Insufficient parallelism** - Not enough workers to saturate data pipeline

## Optimizations Applied ✅

### 1. System Monitoring Added
**New Class**: `SystemMonitor` 
- **Location**: Lines 527-644
- **Features**:
  - Real-time GPU utilization tracking (via NVML)
  - VRAM usage monitoring (allocated/total/percentage)
  - CPU utilization tracking (via psutil)
  - RAM usage monitoring (used/total/percentage)

**Integration**:
- Progress bar updates every 10% of epoch with live stats
- Epoch summary includes final resource utilization
- Automatic warnings if GPU util < 30%

**Output Format**:
```
GPU: 85% | VRAM:12.3/24.0GB(51.2%) | CPU: 45.2% | RAM:18.4/64.0GB(28.8%)
```

---

### 2. Increased Batch Size
**Change**: 5632 → **8192** (+45% increase)

**Location**: Line 370
```python
batch_size: int = 8192  # Was 5632
```

**Impact**:
- GPU works longer per batch (more compute per transfer)
- Reduces CPU↔GPU transfer overhead
- Better amortization of kernel launch overhead

**Validation batch size**: 2048 → **4096** (doubled)

---

### 3. Increased DataLoader Parallelism
**Changes**:
- `num_workers`: 4 → **8** (2x workers)
- `prefetch_factor`: 8 → **16** (2x prefetch depth)

**Location**: Lines 395-396
```python
num_workers: int = 8        # Was 4
prefetch_factor: int = 16   # Was 8
```

**Impact**:
- **Total prefetch queue**: 8 workers × 16 batches = **128 batches** ready
- At batch_size=8192: ~1GB of data queued in RAM
- Ensures GPU never starves waiting for data

---

### 4. Automated Bottleneck Detection
**New Feature**: Automatic warnings when GPU util < 30%

**Location**: Lines 1972-1978

**Output Example**:
```
⚠️  LOW GPU UTILIZATION (11.0%)! Possible bottlenecks:
   - Increase batch_size (current: 8192)
   - Increase num_workers (current: 8)
   - Increase prefetch_factor (current: 16)
   - Check DataLoader timing above
```

---

## Expected Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | 11% | **70-85%** | +7.0x |
| Batch Size | 5632 | 8192 | +45% |
| Prefetch Queue | 32 batches | 128 batches | +4x |
| Throughput | ~30 batch/s | ~50+ batch/s | +67% |
| Training Time | 11.5 min | ~7-8 min | -30% |

---

## Validation & Testing

### Run Training with Monitoring
```bash
cd /DEVELOPMENT/ROOT_AILH/REPOS/AILH_MASTER/AI_DEV
python train_binary.py
```

**Monitor Output**:
1. Progress bar shows live GPU/CPU/VRAM/RAM stats
2. Epoch summary includes utilization percentages
3. Warnings if GPU util remains low

### External Monitoring
```bash
# Terminal 2: Watch GPU utilization
watch -n 1 nvidia-smi

# Terminal 3: Watch CPU/RAM
htop
```

---

## Troubleshooting Low GPU Utilization

### If GPU util still < 50% after these changes:

#### 1. Check VRAM Headroom
```python
# If VRAM usage < 18GB, increase batch_size further
cfg.batch_size = 10240  # or 12288
```

#### 2. Profile DataLoader Speed
Enable debug logging to see DataLoader timing:
```bash
TRAINER_LOG_LEVEL=DEBUG python train_binary.py
```

Look for:
```
[TRAIN PROFILE] DataLoader avg: 25.3ms, max: 78.1ms
```

If avg > 30ms:
- Increase `num_workers` to 12
- Increase `prefetch_factor` to 20
- Check disk I/O with `iostat -x 1`

#### 3. Check HDF5 Read Performance
If disk is bottleneck (HDD or slow NVMe):
```python
# Add to Config:
cfg.pin_memory = True          # Already enabled
cfg.persistent_workers = True  # Already enabled
```

Consider loading dataset into tmpfs (RAM disk):
```bash
sudo mkdir /mnt/ramdisk
sudo mount -t tmpfs -o size=40G tmpfs /mnt/ramdisk
cp /DEVELOPMENT/DATASET_REFERENCE/*.H5 /mnt/ramdisk/
# Update cfg.stage_dir to /mnt/ramdisk
```

#### 4. Disable Augmentation (if enabled)
SpecAugment adds CPU overhead:
```python
cfg.use_specaugment = False  # Already disabled by default
```

#### 5. Profile with PyTorch Profiler
```python
cfg.profile_performance = True  # Already enabled
cfg.profile_gpu_util = True     # Already enabled
```

---

## Hardware-Specific Tuning

### RTX 5090 24GB (Current System)
```python
batch_size = 8192           # ✅ Good starting point
num_workers = 8             # ✅ Good for 16-core CPU
prefetch_factor = 16        # ✅ Aggressive prefetch

# If still underutilized, try:
batch_size = 10240          # Push to 20GB VRAM
num_workers = 12            # Max out CPU cores
```

### RTX 4090 24GB
```python
batch_size = 7168           # Slightly less powerful than 5090
num_workers = 8
prefetch_factor = 16
```

### RTX 3090 24GB
```python
batch_size = 6144           # Older arch, less bandwidth
num_workers = 8
prefetch_factor = 12
```

### RTX 4080 16GB
```python
batch_size = 5120           # Limited VRAM
num_workers = 6
prefetch_factor = 12
```

---

## Code Changes Summary

### Files Modified
1. **dataset_trainer.py** (Lines 91-644, 370-396, 1845-1978)
   - Added `psutil` import for CPU/RAM monitoring
   - Created `SystemMonitor` class (115 lines)
   - Increased `batch_size` 5632 → 8192
   - Increased `val_batch_size` 2048 → 4096
   - Increased `num_workers` 4 → 8
   - Increased `prefetch_factor` 8 → 16
   - Integrated monitoring into training loop
   - Added epoch summary with utilization stats
   - Added automatic low-GPU-util warnings

### New Dependencies
- **psutil** (already installed: 7.1.3)
- **py-cpuinfo** (already installed: 9.0.0)

### Backward Compatibility
- All changes are config-based, users can revert:
  ```python
  cfg.batch_size = 5632
  cfg.num_workers = 4
  cfg.prefetch_factor = 8
  ```

---

## Performance Baseline

### Before Optimizations
```
Epoch 1: 11.5 minutes
GPU Utilization: 11%
VRAM Usage: ~8GB
Throughput: ~30 batch/s
```

### Expected After Optimizations
```
Epoch 1: ~7-8 minutes
GPU Utilization: 70-85%
VRAM Usage: ~16-18GB
Throughput: ~50 batch/s
```

---

## Next Steps

1. **Run test training**: `python train_binary.py`
2. **Monitor first epoch**: Check GPU util in progress bar
3. **Verify epoch summary**: Confirm GPU% > 70%
4. **Adjust if needed**: Increase batch_size if VRAM < 18GB
5. **Document results**: Record actual GPU util achieved

---

## References
- PyTorch DataLoader tuning: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
- CUDA best practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- torch.compile guide: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
