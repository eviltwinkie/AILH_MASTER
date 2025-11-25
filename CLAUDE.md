# AILH_MASTER - AI Acoustic Leak Detection System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Hardware Environment](#hardware-environment)
3. [Codebase Structure](#codebase-structure)
4. [Development Guidelines](#development-guidelines)
5. [Data Organization](#data-organization)
6. [Configuration Management](#configuration-management)
7. [Script Types & Purposes](#script-types--purposes)
8. [Development Workflows](#development-workflows)
9. [Technical Architecture](#technical-architecture)
10. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

**AILH (Acoustic Inspection for Leak & Hydrophone)** is a machine learning system for acoustic leak detection in urban water supply networks using two-stage temporal segmentation and CNN-based classification.

### Core Concept
The system uses hydrophone sensors to capture acoustic signals from water pipelines and employs a two-stage temporal segmentation approach combined with CNN models to identify leaks despite dynamic interference noise.

### Key Features
- Two-stage temporal segmentation (long-term + short-term)
- Mel spectrogram feature extraction
- CNN-based classification with incremental learning
- GPU-accelerated processing pipeline
- Adaptive and continuous learning model
- Multi-threaded data processing

---

## Hardware Environment

### Development Platform
- **OS**: Windows 11 + WSL2 Ubuntu 24.04
- **Architecture**: x86_64
- **CPU**: Intel Core Ultra 9 275HX (24 cores)
- **Filesystem**: ext4 (optimized for small files: 24,801.7 files/s @ 2032.8 MB/s)

### GPU Configuration
- **GPU**: NVIDIA GeForce RTX 5090 Laptop (Blackwell Architecture)
- **VRAM**: 24GB
- **Compute Capability**: 12.0
- **CUDA**: 12.8.1
- **cuDNN**: 9.8.0
- **Multi-Processor Count**: 82
- **Max Threads per SM**: 1536

### Deep Learning Stack
- **TensorFlow**: 2.20.0-dev0+selfbuilt (CUDA 12.8.1, cuDNN 9)
- **PyTorch**: 2.9.1+cu128
- **CuPy**: 13.6.0
- **TensorRT**: 10.14.1.48
- **NVMath**: 0.6.0
- **Numba CUDA**: Available
- **FP8 Support**: ✅ (E4M3, E5M2)

### Performance Benchmarks
- **GPU FP32 GEMM**: 26.03 TFLOP/s (PyTorch), 25.86 TFLOP/s (NVMath)
- **GPU FP16 GEMM**: 73.06 TFLOP/s (NVMath), 17.68 TFLOP/s (PyTorch)
- **GPU FP8 GEMM**: ~28.6 TFLOP/s
- **Disk I/O**: 24,801 files/s, 2032 MB/s (small files)

---

## Codebase Structure

### Repository Layout
```
AILH_MASTER/
├── AI_DEV/                  # Main development scripts
│   ├── global_config.py         # Global configuration
│   ├── dataset_builder.py
│   ├── dataset_classifier.py
│   ├── dataset_trainer.py
│   ├── dataset_tuner.py
│   └── ai_builder.py        
│
├── CORRELATOR_V2/                    # 
│   └── ...             # Correlator v2.0 source code
│
├── DOCS/                    # Documentation
│   ├── AILH.md             # Core requirements & specifications
│   ├── ... # All other misc information for loose project guidance 
│   ├── OPTIMIZATION_GUIDE.md # Misc information
│   ├── test_disk_tune_results.json # Local system hard drive performance test results/settings
│   ├── test_disk_tune_results.txt # Local system hard drive performance test results/settings
│   ├── test_gpu_cuda_results.txt # Local system GPU/CUDA and support/hardware information
│   └── LeakDetectionTwoStageSegmentation.pdf # Reference information for project guidelines
│
├── FCS_TOOLS/               # Field Control System tools
│   ├── datagate_client.py   # Credential rotation client
│   └── datagate_sync.py     # Data synchronization
│
├── UTILITIES/               # Utility scripts
│   ├── gen_requirements.py
│   ├── ... # Other misc utilities
│   ├── normalize_wav_files.py
│   ├── shuffle_data_for_training.py
│   ├── test_disk_tune.py
│   ├── test_gpu_cuda.py
│   ├── test_wav_files.py
│   └── test_wav_files_v2.py
│
└── CLAUDE.md               # This file
```

### External Data Structure (ROOT_AILH)
The actual data is stored outside the repository at `/DEVELOPMENT/ROOT_AILH/`:

```
/DEVELOPMENT/ROOT_AILH/
├── ATTIC/                   # Storage files/docs
├── DATA_SENSORS/            # Incoming sensor data folders
├── DATA_STORE/              # Primary data storage
│   ├── DATASET_DEV/        # Development/testing dataset
│   ├── DATASET_LEARNING/   # Incremental learning data
│   ├── DATASET_TESTING/    # Testing dataset (10%)
│   ├── DATASET_TRAINING/   # Training dataset (70%)
│   ├── DATASET_VALIDATION/ # Validation dataset (20%)
│   ├── MASTER_DATASET/     # ⭐ Source of truth (all labeled data)
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   ├── PROC_CACHE/         # Memmaps, temp files, model caches
│   ├── PROC_LOGS/          # Processing logs
│   ├── PROC_MODELS/        # Trained models
│   ├── PROC_OUTPUT/        # Processing output
│   └── PROC_REPORTS/       # Classification reports
└── REPOS/                   # Source code repositories
    └── AILH_MASTER/        # This repository
```

---

## Development Guidelines

### Mandatory Requirements

#### 1. File Naming Convention
**ALL input WAV files MUST follow this nomenclature:**
```
sensor_id~recording_id~timestamp~gain_db.wav
```
- Delimiter: `~` (tilde)
- Example: `S001~R123~20250118120530~100.wav`
- Gain range: 0-200dB (1dB indicates hydrophone not touching water)

#### 2. Data Specifications
- **Sample Rate**: 4096 Hz
- **Sample Upscale Rate**: 8192 Hz
- **Sample Duration**: 10 seconds
- **Sample File Format**: WAV

#### 3. Command-Line Interface
All scripts MUST support:
- `--verbose` and VERBOSE=YES: Print information during processing steps
- `--debug` and DEBUG=YES: Print detailed debugging information

#### 4. Output Formats
- **Data files**: JSON format
- **Graphs/plots**: PNG format (with `--svg` flag for SVG)
- **Console reports**: For all end results and debugging

#### 5. Code Quality
- **Document all code** with docstrings and comments
- **Optimize for performance**: Use multiprocessing/multithreading where it makes performance sense, vector where possible, use RAM where possible/available
- **Error handling**: Robust exception handling and logging
- **Memory efficiency**: Use memory-mapped files for large datasets

#### 6. Library Restrictions
- **Prefer DO NOT use librosa** - Use numpy, matplotlib, scipy instead
- **Prefer**: numpy, scipy, matplotlib, tensorflow, keras, cupy, pyfftw

---

## Data Organization

### Classification Categories

**CRITICAL**: Active label set:

DATA_LABELS = ['LEAK', 'NORMAL', 'QUIET', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']

### Dataset Split
- **MASTER_DATASET**: 100% - Source of truth (manually labeled)
- **DATASET_TRAINING**: 70% - Training data
- **DATASET_VALIDATION**: 20% - Validation data
- **DATASET_TESTING**: 10% - Testing data
- **DATASET_LEARNING**: Incremental learning data (pseudo-labeled + verified)
- **DATASET_DEV**: Development/testing only (not for production)

### Data Flow
1. Raw sensor data → `DATA_SENSORS/`
2. Manual labeling → `MASTER_DATASET/`
3. Initial split → `DATASET_TRAINING/`, `DATASET_VALIDATION/`, `DATASET_TESTING/`
4. Model predictions → `DATASET_LEARNING/`
5. Manual verification → Back to `MASTER_DATASET/`
6. Reshuffle for next training round

---

## Configuration Management

### Global Configuration (`global_config.py`)

⚠️ **CRITICAL WARNING**: Many scripts DO NOT use global_config.py parameters!

#### Common Discrepancies
| Parameter | global_config.py | Actual Scripts |
|-----------|-----------------|----------------|
| N_FFT | 256 | 512 (most scripts) |
| HOP_LENGTH | 32 | 128 (most scripts) |
| N_MELS | 64 | 32-64 (varies) |

**Always verify the actual parameters used in each script!**

### Key Parameters

#### Audio Processing
```python
SAMPLE_RATE = 4096              # Hz
SAMPLE_UPSCALE_RATE = 8192              # Hz
SAMPLE_DURATION = 10.0          # 10 seconds
LONG_SEGMENTS = [0.125, 0.25, 0.5, 0.75, 1.0]  # seconds
SHORT_SEGMENTS = [64, 128, 256, 512, 1024]      # points
DELIMITER = '~'                 # File naming delimiter
```

#### CNN Hyperparameters (from research paper)
```python
# CNN(Mel) - Recommended configuration
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT = 0.25
CNN_EPOCHS = 200
CNN_FILTERS = 32
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_STRIDES = (2, 2)

# Mel Spectrogram
N_FFT = 512                     # Most scripts use this
HOP_LENGTH = 128                # Most scripts use this
N_MELS = 64                     # Varies by script
```

#### Performance Tuning
```python
MAX_THREAD_WORKERS = 8          # CPU thread pool size
MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 512
```

#### Environment Variables
```python
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
```

---

## Script Types & Purposes

### AI_DEV/ Scripts

#### Core Processing
- **`dataset_processor.py`**: Mel spectrogram computation, data preprocessing
- **`ai_builder.py`**: Optimized GPU-accelerated processing pipeline (5-20x speedup)

#### Training
- **`dataset_trainer.py`**: General dataset training utilities

#### Tuning
- **`cnn_mel_tuner.py`**: Hyperparameter tuning (Keras Tuner / Optuna)

#### Classification
- **`dataset_classifier.py`**: Batch classification utilities

#### Incremental Learning
- **`dataset_learner.py`**: Dataset-based incremental learning

#### Dataset Management
- **`dataset_builder.py`**: Build and organize datasets

### UTILITIES/ Scripts

- **`test_gpu_cuda.py`**: Comprehensive GPU/CUDA diagnostics
- **`test_disk_tune.py`**: Disk I/O performance tuning (24,801 files/s achieved)
- **`normalize_wav_files.py`**: WAV file normalization
- **`shuffle_data_for_training.py`**: Dataset shuffling for training/validation
- **`test_wav_files.py`**: WAV file validation
- **`test_wav_files_v2.py`**: Enhanced WAV validation
- **`gen_requirements.py`**: Generate requirements.txt

### FCS_TOOLS/ Scripts

- **`datagate_client.py`**: Credential rotation client for secure data access
- **`datagate_sync.py`**: Synchronize sensor data from field control systems

---

## Development Workflows

### 1. Initial Model Training

```bash
# Step 1: Prepare MASTER_DATASET with labeled data
# Ensure data follows naming convention: sensor_id~recording_id~timestamp~gain_db.wav

# Step 2: Shuffle and split data (70/20/10)
python UTILITIES/shuffle_data_for_training.py \
    --input /DEVELOPMENT/ROOT_AILH/DATA_STORE/MASTER_DATASET \
    --output-train /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_TRAINING \
    --output-valid /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_VALIDATION \
    --output-test /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_TESTING \
    --verbose

# Step 3: Train initial model
python AI_DEV/cnn_mel_trainer.py \
    --training-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_TRAINING \
    --validation-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_VALIDATION \
    --output-model /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/initial_model.keras \
    --epochs 200 \
    --batch-size 64 \
    --verbose

# Step 4: Evaluate on test set
python AI_DEV/cnn_mel_classifier.py \
    --model /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/initial_model.keras \
    --input-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_TESTING \
    --output-report /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_REPORTS/test_results.json \
    --verbose
```

### 2. Hyperparameter Tuning

```bash
# Use automated hyperparameter search
python AI_DEV/cnn_mel_tuner.py \
    --training-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_TRAINING \
    --validation-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_VALIDATION \
    --tuner keras  # or 'optuna'
    --max-trials 100 \
    --verbose
```

### 3. Incremental Learning

```bash
# Step 1: Classify new real-world data
python AI_DEV/cnn_mel_classifier.py \
    --model /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/current_model.keras \
    --input-dir /DEVELOPMENT/ROOT_AILH/DATA_SENSORS/NEW_BATCH \
    --output-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_LEARNING \
    --confidence-threshold 0.8 \
    --verbose

# Step 2: Manual verification and labeling of predictions

# Step 3: Incremental learning with verified data
python AI_DEV/cnn_mel_learner.py \
    --base-model /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/current_model.keras \
    --learning-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_LEARNING \
    --output-model /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_MODELS/updated_model.keras \
    --confidence-threshold 0.8 \
    --rounds 2 \
    --verbose
```

### 4. GPU Performance Testing

```bash
# Test GPU/CUDA stack
python UTILITIES/test_gpu_cuda.py

# Optimize disk I/O
python UTILITIES/test_disk_tune.py \
    --root /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_DEV \
    --pattern "*.wav" \
    --verbose
```

---

## Technical Architecture

### Two-Stage Temporal Segmentation

#### Stage 1: Long Temporal Segmentation
Subdivide signals into extended segments to isolate non-stationary characteristics:
- Scales: [0.125, 0.25, 0.5, 0.75, 1.0] seconds
- Purpose: Capture long-scale perturbative factors
- Each segment represents partial information about the original signal

#### Stage 2: Short Temporal Segmentation
Partition long-term segments into short-term segments for quasi-stationary features:
- Points: [64, 128, 256, 512, 1024]
- Purpose: Extract quasi-stationary signal characteristics
- Each short segment can be treated as stationary

#### Data Transformation Flow
```
Original Signal X (10s @ 4096 Hz)
    ↓
Long Segmentation → [X1, X2, X3, ..., Xm]
    ↓
Short Segmentation → [[X11, X12, ..., X1n], [X21, X22, ..., X2n], ..., [Xm1, Xm2, ..., Xmn]]
    ↓
Mel Spectrogram (per Xij) → 2D frequency-time matrix
    ↓
3D Matrix [Xi] → [n_short_segments, n_mels, time_frames]
    ↓
4D Matrix [X] → [n_long_segments, n_short_segments, n_mels, time_frames]
    ↓
CNN Classification
    ↓
Voting Mechanism (≥50% segments → leak detected)
```

### CNN Architecture (from research paper)

```python
Model: CNN(Mel)
- Input: Mel spectrogram (n_mels x time_frames)
- Conv2D: filters=32, kernel_size=(3,3), activation='relu'
- MaxPooling2D: pool_size=(2,2), strides=(2,2)
- Dropout: 0.25
- Conv2D: filters=32, kernel_size=(3,3), activation='relu'
- MaxPooling2D: pool_size=(2,2), strides=(2,2)
- GlobalAveragePooling2D
- Dense: 128, activation='relu'
- Dropout: 0.25
- Dense: n_classes, activation='softmax'

Optimizer: Adam (learning_rate=0.001)
Batch Size: 64
Epochs: 200
```

### Incremental Learning Rules

#### Pseudo-labeled Data
- **Leak predictions**: Select top 50% of long segments with highest probabilities
- **Normal predictions**: Select bottom 50% of long segments with lowest probabilities

#### True-labeled Data (Manual Verification)
- **True Positive (TP)**: Top 50% highest probability segments → leak samples
- **False Positive (FP)**: Bottom 50% lowest probability segments → normal samples
- **True Negative (TN)**: Bottom 50% lowest probability segments → normal samples
- **False Negative (FN)**: Top 50% highest probability segments → leak samples

### Performance Metrics

```python
Accuracy = (TP + TN) / (TP + FP + FN + TN)
Precision = TP / (TP + FP)
Sensitivity (Recall) = TP / (TP + FN)
Specificity = TN / (FP + TN)
AUC = Area Under ROC Curve
```

---

## AI Assistant Guidelines

### When Working on This Codebase

#### 1. Always Verify Configuration
- **DON'T** assume `global_config.py` values are used everywhere
- **DO** check actual parameter values in the script you're modifying
- **DO** document any discrepancies you find

#### 2. Respect Data Organization
- **DON'T** modify `MASTER_DATASET` directly
- **DO** use `DATASET_DEV` for testing
- **DO** maintain the 70/20/10 split convention
- **DO** verify label set matches current branch

#### 3. Follow Naming Conventions
- **DO** use `sensor_id~recording_id~timestamp~gain_db.wav` format
- **DO** use `~` (tilde) as delimiter everywhere
- **DO** validate input files follow conventions

#### 4. Code Quality Standards
- **DO** add docstrings to all functions/classes
- **DO** include `--verbose` and `--debug` flags
- **DO** support command-line configuration
- **DO** generate JSON output for data files
- **DO** create both PNG and SVG (with flag) for plots

#### 5. Performance Considerations
- **DO** use GPU acceleration when possible
- **DO** implement multiprocessing/multithreading
- **DO** use memory-mapped files for large datasets
- **DO** test with `test_gpu_cuda.py` after GPU code changes
- **DO** profile disk I/O with `test_disk_tune.py`

#### 6. Testing Workflow
```bash
# Before committing changes:
1. Test with DATASET_DEV (not production data)
2. Run test_gpu_cuda.py to verify GPU functionality
3. Validate WAV files with test_wav_files.py
4. Check performance impact
5. Document changes in commit message
```

#### 7. Common Pitfalls to Avoid
- ❌ Hardcoding paths (use `global_config.py` constants)
- ❌ Ignoring file naming convention
- ❌ Modifying production datasets during testing
- ❌ Not handling OOM errors gracefully
- ❌ Forgetting `--verbose` and `--debug` flags

#### 8. When Adding New Scripts
Create scripts that follow this template pattern:

```python
#!/usr/bin/env python3
"""
Script Description

Usage:
    python script_name.py --input INPUT --output OUTPUT [--verbose] [--debug]

Arguments:
    --input: Input directory/file
    --output: Output directory/file
    --verbose: Print processing information
    --debug: Print debug information
"""

import argparse
import sys
from global_config import *

def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--input', required=True, help='Input path')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Debug output')
    args = parser.parse_args()

    if args.verbose:
        print(f"[i] Processing input: {args.input}")

    # Your code here

    if args.verbose:
        print(f"[✓] Output saved to: {args.output}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[✗] Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
```

#### 9. Documentation Updates
When modifying code:
- Update docstrings
- Update `CLAUDE.md` if architecture changes
- Update `DOCS/AILH.md` if requirements change
- Document configuration discrepancies

#### 10. Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/description
# Make changes
git add .
git commit -m "feat: descriptive commit message"
git push -u origin feature/description
# Create pull request
```

### Quick Reference Commands

```bash
# Validate environment
python UTILITIES/test_gpu_cuda.py

# Check disk performance
python UTILITIES/test_disk_tune.py --root /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_DEV

# Validate WAV files
python UTILITIES/test_wav_files.py --directory /path/to/wav/files

# Normalize WAV files
python UTILITIES/normalize_wav_files.py --input /input/dir --output /output/dir

# Shuffle dataset
python UTILITIES/shuffle_data_for_training.py --input MASTER_DATASET --output-train DATASET_TRAINING

# Train model
python AI_DEV/cnn_mel_trainer.py --training-dir DATASET_TRAINING --validation-dir DATASET_VALIDATION

# Classify audio
python AI_DEV/cnn_mel_classifier.py --model model.keras --input-dir /audio/files

# Incremental learning
python AI_DEV/cnn_mel_learner.py --base-model model.keras --learning-dir DATASET_LEARNING
```

---

## Additional Resources

### Documentation
- **DOCS/AILH.md**: Core requirements and specifications
- **DOCS/OPTIMIZATION_GUIDE.md**: GPU and performance optimization
- **DOCS/LeakDetectionTwoStageSegmentation.pdf**: Research paper
- **global_config.py**: Configuration reference

### External References
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- CuPy Documentation: https://docs.cupy.dev/

---

## Changelog

### 2025-11-18
- Initial CLAUDE.md creation
- Documented complete codebase structure
- Added comprehensive development guidelines
- Documented hardware environment and benchmarks
- Added configuration management section
- Included AI assistant guidelines

---

**Last Updated**: 2025-11-18
**Maintainer**: AILH Development Team
**Repository**: eviltwinkie/AILH_MASTER
**Branch**: claude/add-claude-documentation-012RxEwbcA8qApBtvt6XByaN
