# CLAUDE.md - AI Assistant Guide for AILH_MASTER

## Project Overview

**AILH_MASTER** is a sophisticated **Acoustic Leak Detection System for Urban Water Supply Networks** using deep learning. The system employs Convolutional Neural Networks (CNN) with Mel spectrograms to identify leaks in water pipelines by analyzing acoustic signals from hydrophone sensors.

### Core Technology
- **Purpose**: Detect water leaks in urban pipeline networks using acoustic signal analysis
- **Approach**: Two-stage temporal segmentation + CNN with Mel spectrograms
- **Key Feature**: Adaptive incremental learning model that improves with real-world data
- **Primary Language**: Python 3 (51 files, ~20,000 lines of code)
- **ML Frameworks**: TensorFlow 2.20.0 (self-built!), PyTorch 2.9.1, Keras
- **Optimization Level**: Production-grade with extensive GPU acceleration (CUDA)

### ⚠️ CRITICAL: Configuration Discrepancies Detected

**MAJOR ISSUE**: `global_config.py` parameters are NOT used by most scripts! Each script has hardcoded parameters:

| Parameter | global_config.py | Actual Implementation |
|-----------|------------------|---------------------|
| **N_FFT** | 256 | **512** (classifiers, pipeline) |
| **HOP_LENGTH** | 32 | **128** (classifiers, pipeline) |
| **N_MELS** | 64 | **32-64** (varies by script) |
| **power** | Not defined | **1.0 or 2.0** (affects dB calculation!) |

**⚠️ BROKEN CODE**: `cnn_mel_processor.py` imports variables from `old_config.py` that don't exist - **will crash on import!**

### Signal Processing Workflow
1. **Input**: 10-second audio samples at 4096 Hz from hydrophone sensors (0-200dB gain)
2. **Segmentation**: Overlapping windowing approach (NOT multiple segment sizes!)
   - Long window: 1024 points (0.25s @ 4096Hz) with 50% overlap
   - Short window: 512 points with 50% overlap
   - **Note**: Documentation in README.md describes multi-size segmentation, but code uses single-size overlapping windows
3. **Transform**: Convert to Mel spectrograms
   - **Actual parameters**: N_FFT=512, HOP_LENGTH=128, N_MELS=32-64, power=1.0-2.0
   - norm='slaney', mel_scale='htk', fmin=20Hz, center=False
4. **Classify**: CNN model outputs classification probabilities
5. **Decision**: Multiple voting strategies available (mean, long_vote, any_long, frac_vote)
6. **Learn**: Incremental learning with confidence thresholds (0.8)

### Classification Categories

**⚠️ IMPORTANT: Multiple Label Sets Exist**

The repository contains **FOUR different label sets** - verify which is active before modifying:

1. **DOCS/AILH.md (OFFICIAL DOCUMENTATION - Main Branch)**:
   ```python
   DATA_LABELS = ['LEAK', 'NORMAL', 'QUIET', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']
   ```
   - This is the **authoritative specification** from the main branch documentation
   - Includes 6 categories covering all acoustic signal types
   - Recommended for new deployments

2. **AI_DEV/global_config.py (ACTIVE - Feature Branch)**:
   ```python
   DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']
   ```
   - Currently active in the feature branch
   - Simplified to 5 categories
   - Different naming: 'BACKGROUND' instead of 'QUIET', adds 'CRACK', removes 'RANDOM'/'MECHANICAL'

3. **AI_ATTIC/README.md (ARCHIVE DOCUMENTATION)**:
   ```python
   DATA_LABELS = ['LEAK', 'NORMAL', 'QUIET', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']
   ```
   - Matches the official DOCS/AILH.md specification
   - Archived documentation (same as official)

4. **UTILITIES/old_config.py (LEGACY)**:
   ```python
   LABELS = ['LEAK', 'NORMAL', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']
   ```
   - Legacy configuration (5 categories, missing 'QUIET')
   - **DO NOT USE** - contains outdated configurations

**⚠️ LABEL SET MISMATCH WARNING**:
- **Official documentation** (main branch): 6 categories with QUIET/RANDOM/MECHANICAL
- **Current implementation** (feature branch): 5 categories with BACKGROUND/CRACK
- This discrepancy can cause:
  - Training/inference mismatches
  - Model architecture incompatibilities (wrong output layer size)
  - Dataset organization issues

**Recommendation**:
1. **Before training**: Verify which label set matches your MASTER_DATASET
2. **For new projects**: Use the official DOCS/AILH.md labels
3. **For existing models**: Maintain consistency with the label set used during initial training
4. **When merging branches**: Reconcile label set differences carefully

---

## Hardware Environment

### Production System Specifications

**⚠️ IMPORTANT**: This project runs on **Windows 11 with WSL2 Ubuntu** environment.

**GPU Configuration**:
- **Model**: NVIDIA GeForce RTX 5090 Laptop GPU
- **VRAM**: 24GB GDDR7
- **Compute Capability**: 12.0 (Ada Lovelace architecture)
- **CUDA Version**: 12.8
- **cuDNN Version**: 9.x
- **Driver Version**: Latest NVIDIA drivers

**CPU Configuration**:
- Multiple cores available (optimal: 4-12 threads)
- OMP_NUM_THREADS=4 (configured)

**Storage Optimization**:
- **Filesystem**: ext4 (optimized for small files)
- **Performance**: 24,801.7 files/s (measured with bench_smallfiles_ext4.py)
- **Optimal Settings**: 6 threads, 131KB buffer, osread method
- **Dataset Path**: /DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_DEV

**Memory**:
- 8GB+ RAM recommended
- GPU memory management with TF_GPU_ALLOCATOR=cuda_malloc_async

### Sample Rate Upscaling Requirement

**⚠️ CRITICAL**: The system is designed for **4096 Hz sampling REQUIRED TO UPSCALE TO 8192 Hz**.

This is a fundamental design requirement documented in DOCS/AILH.md:
```python
SAMPLE_RATE = 4096      # 4096 Hz REQUIRED TO UPSCALE TO 8192
```

The 2x upscaling is likely used for:
- Enhanced frequency resolution
- Better Mel spectrogram quality
- Improved CNN feature extraction

**AI Assistant Note**: When processing audio, ensure the upscaling step is preserved in the pipeline.

---

## Repository Structure

### Complete File Inventory

```
AILH_MASTER/                   (51 Python files, ~20,000 lines)
├── main.py                    # Empty entry point (0 bytes - placeholder)
├── CLAUDE.md                  # This file (30KB, 1,004+ lines)
│
├── AI_DEV/                    # ⭐ ACTIVE DEVELOPMENT (24 files, 9,708 lines)
│   ├── global_config.py       # 🔧 Primary config (132 lines, 3.9KB)
│   │
│   ├── Dataset Building:
│   │   ├── dataset_builder.py              # 688 lines, 29KB - HDF5 builder
│   │   └── leak_dataset_builder_v15.py     # 676 lines, 28KB - Multi-split v15
│   │
│   ├── Model Training:
│   │   ├── dataset_trainer.py              # 780 lines, 31KB - CNN trainer
│   │   ├── leak_dataset_trainer_v15.py     # 711 lines, 29KB - Trainer v15
│   │   ├── cnn_mel_trainer.py              # 339 lines, 15KB - Mel trainer
│   │   └── cnn_mel_tuner.py                # 267 lines, 14KB - Hyperparameter tuning
│   │
│   ├── Classification:
│   │   ├── dataset_classifier.py           # 412 lines, 17KB - Single file
│   │   ├── leak_directory_classifier.py    # 314 lines, 12KB - Batch directory
│   │   └── cnn_mel_classifier.py           # 164 lines, 5.6KB - Mel-based
│   │
│   ├── Incremental Learning:
│   │   ├── dataset_learner.py              # 470 lines, 20KB - Base learner
│   │   ├── leak_dataset_learner.py         # 313 lines, 15KB - Leak-specific
│   │   └── cnn_mel_learner.py              # 189 lines, 7.5KB - Mel-based
│   │
│   ├── Feature Processing:
│   │   ├── cnn_mel_processor.py            # 351 lines, 14KB - Mel processor
│   │   └── pipeline.py                     # 417 lines, 15KB - ⚡ GPU pipeline
│   │
│   ├── Utilities:
│   │   ├── normalize_wav_files.py          # 175 lines, 6.6KB - WAV norm
│   │   └── shuffle_data_for_training.py    # 217 lines, 6.9KB - Data shuffle
│   │
│   └── Testing/Benchmarking:
│       ├── test_gpu_cuda.py                # 1,590 lines, 54KB - 🔥 MASSIVE GPU suite
│       ├── test_cpu_fixes.py               # 36 lines, 1.4KB - CPU tests
│       ├── test_optimizations.py           # 95 lines, 2.6KB - Performance tests
│       ├── test_disk_settings.py           # 115 lines, 4.0KB - Filesystem tests
│       ├── test_wav_files.py               # 170 lines, 6.4KB - WAV validation
│       ├── test_wav_files_v2.py            # 119 lines, 4.1KB - WAV validation v2
│       └── bench_smallfiles_ext4.py        # 968 lines, 37KB - Filesystem benchmark
│
├── AI_ATTIC/                  # 📦 ARCHIVE (24 files + docs, 9,532 lines)
│   ├── README.md              # 📄 171 lines, 16KB - Project requirements
│   ├── OPTIMIZATION_GUIDE.md  # 📄 253 lines, 8KB - Performance guide
│   ├── requirements.txt       # 22 lines - Python dependencies (auto-generated)
│   ├── admin_config.py        # 132 lines, 3.9KB - Archived config (= global_config.py)
│   ├── leak_report.csv        # 19 lines - Sample classification results
│   ├── pipeline.py            # ⚠️ 271 lines - OLDER VERSION (146 lines shorter!)
│   └── [same file structure as AI_DEV - older/archived versions]
│
├── FCS_UTILS/                 # 🌐 External Data Integration (2 files, ~334 lines)
│   ├── datagate_client.py     # 42 lines, 1.5KB - Async HTTP client
│   └── datagate_sync.py       # 292 lines, 13KB - FCS DataGate API sync
│
└── UTILITIES/                 # 🛠️ Shared Utilities (2 files, ~472 lines)
    ├── old_config.py          # ⚠️ 88 lines, 2.6KB - CONTAINS CREDENTIALS!
    └── gen_requirements.py    # 384 lines, 12KB - Auto-gen requirements.txt
```

### Key Directory Relationships
- **AI_DEV**: Active development - make ALL code changes here
- **AI_ATTIC**: Archive/backup - reference only, DO NOT modify
  - **Note**: `pipeline.py` in AI_ATTIC is **146 lines shorter** - use AI_DEV version
- **FCS_UTILS**: External data fetching - standalone module
  - **⚠️ NAMING DISCREPANCY**: Feature branch uses `FCS_UTILS/`, but main branch may use `FCS_TOOLS/`
  - Verify the correct naming when merging branches
- **UTILITIES**: Shared tools - used by both AI_DEV and FCS_UTILS

---

## Development Environment Paths

**IMPORTANT**: Multiple scripts use hardcoded `/DEVELOPMENT/` paths not configurable via `global_config.py`.

### Documented Folder Structure (from DOCS/AILH.md)

The **official documented structure** for the project is shown below.

**⚠️ IMPORTANT**: In the actual system, `ROOT_AILH` is located at **`/DEVELOPMENT/ROOT_AILH/`**

The documentation shows relative paths from ROOT_AILH for clarity:

```
ROOT_AILH/                             # Actual path: /DEVELOPMENT/ROOT_AILH/
│
├── ATTIC/                             # Storage files/docs (archived)
│
├── DATA_SENSORS/                      # Incoming sensor data (multiple folders)
│   ├── DATA_FOLDER/                   # Site/Logger specific folders
│   ⋮
│   └── DATA_FOLDER/
│
├── DATA_STORE/                        # Datasets, logs, output, etc.
│   │
│   ├── DATASET_DEV/                   # Temp development use for code testing
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── DATASET_LEARNING/              # Incremental learning data
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── DATASET_TESTING/               # Test dataset
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── DATASET_TRAINING/              # Training dataset
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── DATASET_VALIDATION/            # Validation dataset
│   │   ├── BACKGROUND/
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── MASTER_DATASET/                # ⭐ SOURCE OF ALL DATA
│   │   ├── BACKGROUND/                # (Master copy - do not modify directly)
│   │   ├── CRACK/
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   └── UNCLASSIFIED/
│   │
│   ├── PROC_CACHE/                    # Memmaps, temp files, etc.
│   ├── PROC_LOGS/                     # Logs
│   ├── PROC_MODELS/                   # Models
│   ├── PROC_OUTPUT/                   # Output
│   └── PROC_REPORTS/                  # Reports
│
└── REPOS/                             # Source code repositories
    └── AILH_MASTER/                   # Current working code

```

**⚠️ CRITICAL: MASTER_DATASET Concept**

The `MASTER_DATASET` is the **single source of truth** for all training data:
- All labeled and verified audio samples are stored here
- Other dataset folders (TRAINING, VALIDATION, TESTING, etc.) are **derived** from MASTER_DATASET
- **Never modify MASTER_DATASET directly** during training - only add verified samples
- Use data splitting scripts to populate other dataset folders from MASTER_DATASET

### Actual Implementation Paths (Current Scripts)

The **current scripts** use this structure under `/DEVELOPMENT/`:

```
/DEVELOPMENT/
├── ROOT_AILH/
│   ├── AILH_LOGS/              # Logs directory (FCS, old_config.py)
│   ├── AILH_CACHE/             # Cache directory (old_config.py)
│   ├── AILH_TMP/               # Temp directory (old_config.py)
│   ├── DATA_SENSORS/           # FCS DataGate downloads
│   └── DATA_STORE/             # Training data and memmaps
│       ├── MEMMAPS/            # Memory-mapped arrays (pipeline.py)
│       ├── TRAINING/           # Training datasets (pipeline.py)
│       └── DATASET_DEV/        # Development dataset (test_disk_settings.py)
│
└── DATASET_REFERENCE/          # Dataset and model storage
    ├── INFERENCE/              # Inference inputs
    ├── TESTING/                # Test datasets
    ├── MODELS/                 # Model checkpoints
    │   ├── best.pth
    │   ├── cnn_model_best.h5
    │   └── cnn_model_full.h5
    ├── reports/                # Classification reports
    ├── TRAINING_DATASET.H5
    ├── VALIDATION_DATASET.H5
    └── TESTING_DATASET.H5
```

**⚠️ PATH DISCREPANCY WARNING**:
- **Documented structure** (DOCS/AILH.md): Uses `ROOT_AILH/` as base with comprehensive `DATA_STORE/` subdirectories
- **Current implementation**: Uses `/DEVELOPMENT/ROOT_AILH/` and `/DEVELOPMENT/DATASET_REFERENCE/`
- **Recommendation**: Migrate to the documented structure for consistency

**Scripts using hardcoded paths**:
- `pipeline.py` → `/DEVELOPMENT/ROOT_AILH/DATA_STORE/`
- `datagate_sync.py` → `/DEVELOPMENT/ROOT_AILH/AILH_LOGS`, `/DEVELOPMENT/ROOT_AILH/DATA_SENSORS`
- `old_config.py` → `/DEVELOPMENT/ROOT_AILH/` (all subdirectories)
- `dataset_classifier.py` → `/DEVELOPMENT/DATASET_REFERENCE` (default)
- `leak_directory_classifier.py` → `/DEVELOPMENT/DATASET_REFERENCE` (hardcoded)
- `test_disk_settings.py` → `/DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_DEV`

---

## Development Guidelines

### Critical Conventions (MUST FOLLOW)

1. **File Delimiter**: Use `~` (tilde) for all file field separators
   ```
   Format: sensor_id~recording_id~timestamp~gain_db.wav
   Example: sensor_001~rec_12345~20240118_143022~45.wav

   FCS DataGate Format: loggerId~recordingId~timestamp~gain.wav
   Local Format: sensor_id~recording_id~timestamp~gain_db.wav
   ```

2. **Data Files**: Save all output data in JSON format
   ```python
   import json
   with open("output.json", "w") as f:
       json.dump(data, f, indent=2)
   ```

3. **Plots/Graphs**: Save in PNG format by default, SVG with `--svg` flag
   ```python
   plt.savefig("plot.png", dpi=300, bbox_inches='tight')
   if args.svg:
       plt.savefig("plot.svg", format='svg')
   ```

4. **Command-line Arguments**: ALL scripts must be configurable via CLI
   ```python
   parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
   parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
   parser.add_argument("--svg", action="store_true", help="Also save plots as SVG")
   ```

5. **Verbosity Levels**:
   - Standard: Essential information only
   - `--verbose` (`-v`): Detailed step-by-step progress
   - `--debug` (`-d`): Full debugging information, intermediate values

6. **Audio Processing**:
   - **DO NOT** use `librosa` - use `numpy`, `scipy`, `torchaudio`, `soundfile` instead
   - Default: 4096 Hz sample rate, 10s duration
   - Always verify WAV files are normalized (check even if pre-normalized)
   - Hydrophone gain 1dB = sensor not in water (filter these out)

7. **Code Quality**:
   - Document ALL code with clear comments
   - Use type hints where practical (newer files use them)
   - Optimize for multiprocessing/multithreading
   - Clean up GPU memory explicitly
   - Handle errors gracefully with informative messages

8. **Hyperparameter Tuning**:
   - Enabled by default with auto-retraining
   - Use `--no-tuning` to disable
   - Support both Keras Tuner and Optuna (auto-select or `--tuner` flag)

9. **Console Reports**:
   - Always generate console reports with summary statistics
   - Include plots/graphs for visualizations
   - Both for results AND debugging output

10. **Data Shuffling**:
    - During initial/manual training, randomly shuffle reference and validation data
    - Swap evenly between TRAINING and VALIDATION folders

---

## Configuration and Environment

### Global Configuration (`AI_DEV/global_config.py`)

**Environment Variables** (automatically set):
```python
CACHE_DIR = "/mnt/d/AILH_CACHE"          # Cache directory
TMPDIR = "/mnt/d/AILH_TMP"                # Temporary files
TEMP = "/mnt/d/AILH_TMP"                  # Temp (Windows compat)
TMP = "/mnt/d/AILH_TMP"                   # Tmp (Unix compat)
KERAS_HOME = "/mnt/d/AILH_CACHE/KERAS"   # Keras cache
TORCH_HOME = "/mnt/d/AILH_CACHE/TORCH"   # PyTorch cache
XDG_CACHE_HOME = "/mnt/d/AILH_CACHE/XDG" # XDG cache
```

**GPU/CUDA Settings**:
```python
KERAS_BACKEND = "tensorflow"
TF_GPU_ALLOCATOR = "cuda_malloc_async"
CUDA_VISIBLE_DEVICES = "0"                # Use only first GPU
XLA_FLAGS = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
TF_FORCE_GPU_ALLOW_GROWTH = "true"
TF_ENABLE_ONEDNN_OPTS = "0"
TF_GPU_THREAD_MODE = "gpu_private"
CUDA_MODULE_LOADING = "EAGER"
TF_CUDNN_DETERMINISTIC = "1"              # Deterministic cuDNN
```

**Threading Configuration**:
```python
CPU_COUNT = os.cpu_count()
MAX_THREAD_WORKERS = 8
OMP_NUM_THREADS = "4"
TF_NUM_INTEROP_THREADS = "2"
TF_NUM_INTRAOP_THREADS = "4"
```

**Signal Processing Parameters**:
```python
SAMPLE_RATE = 4096                        # 4096 Hz
SAMPLE_DURATION = 10.0                    # 10 seconds
INCREMENTAL_CONFIDENCE_THRESHOLD = 0.8    # Incremental learning threshold
INCREMENTAL_ROUNDS = 2

# Two-stage temporal segmentation
LONG_SEGMENTS = [0.125, 0.25, 0.5, 0.75, 1.0]     # seconds
SHORT_SEGMENTS = [64, 128, 256, 512, 1024]        # points

# Mel spectrogram
N_MELS = 64
N_FFT = 256
HOP_LENGTH = 32
N_FILTERS = 80
```

**CNN Model Hyperparameters** (from grid search):
```python
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 200
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT = 0.25
CNN_FILTERS = 32
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_STRIDES = (2, 2)
CNN_DENSE = 128
```

**Advanced Configuration** (pipeline.py):
```python
CPU_THREADS = 12
GPU_BATCH_SIZE = 256
CUDA_STREAMS = 4
CPU_GPU_BUFFER_SIZE = 240
RAM_PREFETCH_DEPTH = 12
FILES_PER_TASK = 4096        # AI_DEV (vs 192 in AI_ATTIC)
```

**Batch Size Variations Across Scripts**:
```python
CNN_BATCH_SIZE = 64                    # global_config.py - Training
batch_segments = 2048                  # dataset_classifier.py - Classification
BATCH_SEGMENTS = 16384                 # leak_directory_classifier.py - Batch classification (8x larger!)
GPU_BATCH_SIZE = 256                   # pipeline.py - GPU mel computation
batch_size = 512                       # cnn_mel_processor.py - Streaming extraction
```

**⚠️ Note**: Each script has different optimal batch sizes based on its specific memory/performance requirements. Do NOT assume they should all use the same value.

**File Delimiter**:
```python
DELIMITER = '~'  # Used in all filename parsing
```

---

## Data Directory Structure

### Official Structure (DOCS/AILH.md)

The **official documented structure** uses the ROOT_AILH/DATA_STORE hierarchy with MASTER_DATASET as the source of truth.

**⚠️ NOTE**: `ROOT_AILH` = `/DEVELOPMENT/ROOT_AILH/` in the actual filesystem

```
ROOT_AILH/DATA_STORE/                  # Actual: /DEVELOPMENT/ROOT_AILH/DATA_STORE/
│
├── MASTER_DATASET/            # ⭐ SINGLE SOURCE OF TRUTH
│   ├── BACKGROUND/            # (or QUIET - depending on label set)
│   ├── CRACK/                 # (if using feature branch labels)
│   ├── LEAK/
│   ├── NORMAL/
│   ├── RANDOM/                # (if using official labels)
│   ├── MECHANICAL/            # (if using official labels)
│   └── UNCLASSIFIED/
│
├── DATASET_TRAINING/          # Derived from MASTER_DATASET (70%)
│   └── [same label subdirectories]
│
├── DATASET_VALIDATION/        # Derived from MASTER_DATASET (20%)
│   └── [same label subdirectories]
│
├── DATASET_TESTING/           # Derived from MASTER_DATASET (10%)
│   └── [same label subdirectories]
│
├── DATASET_LEARNING/          # Incremental learning data
│   └── [same label subdirectories]
│
└── DATASET_DEV/               # Development/testing (not for production training)
    └── [same label subdirectories]
```

**⚠️ CRITICAL WORKFLOW**:
1. All verified audio samples go into **MASTER_DATASET** first
2. Use data splitting scripts to populate TRAINING/VALIDATION/TESTING from MASTER_DATASET
3. **Never train directly on MASTER_DATASET** - it's the archive
4. DATASET_LEARNING receives incremental learning data from production models
5. DATASET_DEV is for code testing only

### Legacy Structure (AI_ATTIC/README.md)

The older documentation describes this alternative structure:

```
BASE_DIR/
├── SENSOR_DATA/
│   ├── RAW_SIGNALS/           # Unprocessed sensor recordings
│   └── LABELED_SEGMENTS/      # Manually labeled segments
│
├── REFERENCE_DATA/            # Initial training/validation data
│   ├── TRAINING/              # Training dataset (70%)
│   │   └── [label subdirectories]
│   └── VALIDATION/            # Validation dataset (20%)
│       └── [label subdirectories]
│
└── UPDATE_DATA/               # Incremental learning data
    ├── POSITIVE/              # True positive labeled data
    │   └── [label subdirectories]
    └── NEGATIVE/              # False negative labeled data
        └── [label subdirectories]
```

**Note**: Some scripts still reference this legacy structure. When working with older scripts, verify which directory structure they expect.

### Data Split Ratios
- **Training**: 70% (DATASET_TRAINING or REFERENCE_DATA/TRAINING)
- **Validation**: 20% (DATASET_VALIDATION or REFERENCE_DATA/VALIDATION)
- **Test**: 10% (DATASET_TESTING or held-out from validation)

### Incremental Learning Data Flow

**Pseudo-labeled Data** (model predictions):
- Leak predictions → Select top 50% confidence segments → UPDATE_DATA/POSITIVE/LEAK
- Normal predictions → Select bottom 50% confidence segments → UPDATE_DATA/NEGATIVE/NORMAL

**True-labeled Data** (manually verified):
- True Positive (TP): Top 50% confidence → UPDATE_DATA/POSITIVE/LEAK
- False Positive (FP): Bottom 50% confidence → UPDATE_DATA/NEGATIVE/NORMAL
- True Negative (TN): Bottom 50% confidence → UPDATE_DATA/NEGATIVE/NORMAL
- False Negative (FN): Top 50% confidence → UPDATE_DATA/POSITIVE/LEAK

---

## Common Workflows

### 1. Building a Dataset

**Purpose**: Convert WAV files into HDF5 dataset with Mel spectrograms

```bash
# Basic usage
cd AI_DEV
python dataset_builder.py --input ../REFERENCE_DATA/TRAINING --output ../OUTPUT/training.h5

# With all options
python dataset_builder.py \
    --input ../REFERENCE_DATA/TRAINING \
    --output ../OUTPUT/training.h5 \
    --verbose \
    --debug \
    --workers 8
```

**Alternative (v15 multi-split builder)**:
```bash
python leak_dataset_builder_v15.py \
    --input ../REFERENCE_DATA \
    --output ../OUTPUT \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --verbose
```

### 2. Training a Model

**Purpose**: Train CNN model on HDF5 dataset

```bash
# Basic training
python dataset_trainer.py \
    --dataset ../OUTPUT/training.h5 \
    --model ../OUTPUT/model.keras \
    --epochs 200 \
    --batch-size 64

# With hyperparameter tuning
python dataset_trainer.py \
    --dataset ../OUTPUT/training.h5 \
    --model ../OUTPUT/model.keras \
    --epochs 200 \
    --batch-size 64 \
    --tuner optuna \
    --verbose

# Disable tuning
python dataset_trainer.py \
    --dataset ../OUTPUT/training.h5 \
    --model ../OUTPUT/model.keras \
    --no-tuning \
    --verbose
```

**Alternative (Mel-based trainer)**:
```bash
python cnn_mel_trainer.py \
    --input ../REFERENCE_DATA/TRAINING \
    --model ../OUTPUT/cnn_mel_model.keras \
    --verbose
```

### 3. Classifying Audio Files

**Purpose**: Classify WAV files using trained model

#### A. dataset_classifier.py (CLI-configurable, advanced options)

```bash
# Recommended settings for maximum recall
python dataset_classifier.py \
    --stage-dir /DEVELOPMENT/DATASET_REFERENCE \
    --in-dir /DEVELOPMENT/DATASET_REFERENCE/INFERENCE/LEAK \
    --prob softmax \
    --decision frac_vote \
    --long-frac 0.25 \
    --thr 0.35 \
    --out leak_report.csv \
    --batch-segments 2048

# Single file classification
python dataset_classifier.py \
    --stage-dir /DEVELOPMENT/DATASET_REFERENCE \
    --in-dir /path/to/file.wav \
    --out results.csv
```

**Decision Rules** (lines 284-312 of dataset_classifier.py):
1. **`mean`**: Average leak probability across all long segments ≥ threshold
   - Simple averaging approach
   - Good for balanced precision/recall

2. **`long_vote`**: Majority voting (≥50%) on long segments
   - Paper-style approach
   - Each long segment votes if mean(short segments) ≥ threshold

3. **`any_long`**: ANY long segment exceeds threshold → LEAK
   - Maximum recall, lower precision
   - Detects even brief leak signatures

4. **`frac_vote`** ⭐ **(RECOMMENDED)**: Fractional threshold voting
   - Leak if `(segments ≥ thr) / total_segments ≥ --long-frac`
   - Default: `--long-frac 0.25` (25% of segments must exceed threshold)
   - Default threshold: `--thr 0.35`
   - **Best balance for recall-first pipelines**

**Probability Heads**:
- `softmax` (default): Standard softmax over class logits, uses P(LEAK)
- `blend`: Average of softmax P(LEAK) + sigmoid(auxiliary_leak_logit)

**CSV Output Format**:
```
filepath,is_leak,leak_conf_mean,per_long_probs_json,long_pos_frac,notes
```

**Config Discovery**: Automatically reads HDF5 metadata or uses fallback defaults

#### B. leak_directory_classifier.py (Hardcoded config, batch processing)

**⚠️ NO CLI ARGUMENTS** - Edit source code to configure:
```python
STAGE_DIR = Path("/DEVELOPMENT/DATASET_REFERENCE")  # Line 34
INPUT_DIR = STAGE_DIR / "INFERENCE"  # Line 35
OUTPUT_CSV = STAGE_DIR / "reports" / "classification_report.csv"  # Line 36
BATCH_SEGMENTS = 16384  # Line 39 (vs 2048 in dataset_classifier!)
```

**CSV Output Format**:
```
filepath,predicted_label,predicted_confidence,prob_BACKGROUND,prob_CRACK,prob_LEAK,prob_NORMAL,prob_UNCLASSIFIED
```

**Differences from dataset_classifier.py**:
- Larger batch size (16384 vs 2048)
- Simpler CNN architecture (single-head vs dual-head)
- Uses `soundfile` instead of `torchaudio`
- Outputs per-class probabilities instead of voting results
- No decision rules - direct top-1 prediction

### 4. Incremental Learning

**Purpose**: Update model with new real-world data

```bash
python dataset_learner.py \
    --model ../OUTPUT/model.keras \
    --update-data ../UPDATE_DATA \
    --output ../OUTPUT/model_updated.keras \
    --confidence 0.8 \
    --rounds 2 \
    --verbose
```

### 5. Hyperparameter Tuning

**Purpose**: Optimize CNN architecture and parameters

```bash
python cnn_mel_tuner.py \
    --input ../REFERENCE_DATA/TRAINING \
    --validation ../REFERENCE_DATA/VALIDATION \
    --trials 100 \
    --output ../OUTPUT/best_model.keras \
    --verbose

# Restart from previous tuning session
python cnn_mel_tuner.py \
    --input ../REFERENCE_DATA/TRAINING \
    --validation ../REFERENCE_DATA/VALIDATION \
    --restart \
    --n_trials 100 \
    --verbose
```

### 6. Data Preparation

**Normalize WAV files**:
```bash
python normalize_wav_files.py \
    --input ../SENSOR_DATA/RAW_SIGNALS \
    --output ../SENSOR_DATA/NORMALIZED \
    --verbose
```

**Shuffle training/validation data**:
```bash
python shuffle_data_for_training.py \
    --training ../REFERENCE_DATA/TRAINING \
    --validation ../REFERENCE_DATA/VALIDATION \
    --verbose
```

### 7. Fetching External Data (FCS DataGate)

```bash
cd ../FCS_UTILS
python datagate_sync.py \
    --output ../SENSOR_DATA/RAW_SIGNALS \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --verbose
```

---

## Testing and Diagnostics

### GPU/CUDA Testing

```bash
# Comprehensive GPU diagnostics (1,590 lines of tests!)
python test_gpu_cuda.py
```

**What it tests:**
- System diagnostics (NVIDIA driver, CPU info)
- GPU inventory via nvidia-smi
- TensorFlow & PyTorch functional tests
- Conv2D, matmul, gradient operations
- CPU vs GPU timing comparisons
- Medium-stress GPU load tests
- GEMM benchmarks (NumPy/PyTorch/NVMath)
- PTXAS diagnostics
- Resource usage recommendations

```bash
# CPU optimization tests
python test_cpu_fixes.py

# General optimization tests
python test_optimizations.py

# Disk/filesystem performance
python test_disk_settings.py
python bench_smallfiles_ext4.py
```

### WAV File Validation
```bash
python test_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS
python test_wav_files_v2.py --input ../SENSOR_DATA/RAW_SIGNALS --verbose
```

---

## Dependencies

### Core ML/Data Science Stack
```
tensorflow==2.20.0.dev0+selfbuilt   # ⚡ Self-built TensorFlow with custom optimizations!
torch==2.9.1                        # PyTorch
torchaudio==2.9.1                   # Audio processing
keras                               # High-level neural networks API
```

### Audio Processing
```
soundfile==0.13.1                   # WAV file I/O
pyfftw                              # Fast FFT (FFTW wrapper)
scipy==1.16.3                       # Signal processing
numpy==2.1.3                        # Numerical operations
```

### Machine Learning Tools
```
scikit-learn==1.7.2                 # ML utilities, metrics
optuna                              # Hyperparameter optimization
h5py==3.15.1                        # HDF5 dataset storage
```

### Visualization
```
matplotlib==3.10.7                  # Plotting
plotly                              # Interactive plots
```

### GPU/Performance
```
nvmath                              # NVIDIA math libraries
pynvml                              # NVIDIA GPU monitoring
cupy-cuda11x or cupy-cuda12x        # GPU acceleration (install separately)
```

### System Utilities
```
psutil==7.1.3                       # System monitoring
cpuinfo                             # CPU information
tqdm                                # Progress bars
```

### FCS DataGate Integration (not in requirements.txt)
```
httpx                               # Async HTTP client
tenacity                            # Retry logic with exponential backoff
xmltodict                           # XML parsing for API responses
```

### Installation

**From requirements.txt**:
```bash
cd AI_ATTIC
pip install -r requirements.txt
```

**⚠️ Note**: TensorFlow is self-built (`tensorflow==2.20.0.dev0+selfbuilt`). Standard TensorFlow can be installed with:
```bash
pip install tensorflow-gpu  # or tensorflow for CPU-only
```

**For GPU acceleration** (CuPy):
```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

**For FCS DataGate integration**:
```bash
pip install httpx tenacity xmltodict
```

---

## Performance Optimization

### Expected Speedups (from OPTIMIZATION_GUIDE.md)
- **GPU vs CPU**: 3-10x speedup for mel spectrogram computation
- **Batch vs Individual**: 2-5x speedup through vectorization
- **Threading**: 1.5-3x speedup with optimal thread count
- **Overall**: 5-20x total improvement depending on hardware

### Hardware Recommendations

**GPU Acceleration**:
- NVIDIA GPU: CUDA-compatible (GTX 1060 or better)
- CUDA: 11.x or 12.x
- GPU Memory: 4GB+ for optimal performance (200MB-2GB depending on batch size)
- CuPy: Install matching CUDA version

**CPU Optimization**:
- Cores: 4+ CPU cores recommended
- RAM: 8GB+ for large datasets
- Storage: SSD for faster I/O

### Memory Requirements
- **Small dataset** (10s audio): 50-200 MB RAM
- **Medium dataset** (60s audio): 200-800 MB RAM
- **Large dataset** (300s audio): 1-4 GB RAM
- **GPU memory**: 200MB-2GB depending on batch size

### Memory Management
- Adaptive batch sizes based on available memory
- Strategic garbage collection to prevent leaks
- HDF5 compression for dataset storage
- GPU memory cleanup after operations

---

## Security and Best Practices

### ⚠️ CRITICAL Security Warnings

1. **TWO Sets of Credentials in Plaintext**:
   - `UTILITIES/old_config.py` contains **TWO sets** of plaintext API credentials:
     ```python
     # Commented out (but still in git history!)
     #DATAGATE_USERNAME = "sbartal"
     #DATAGATE_PASSWORD = "Sb749499houstonTX"

     # Active credentials
     DATAGATE_USERNAME = "emartinez"
     DATAGATE_PASSWORD = "letmein2Umeow!!!"
     ```
   - **CRITICAL**: Even commented credentials are recoverable from git history!
   - **IMMEDIATELY** rotate ALL credentials if this repository has been shared
   - **NEVER** commit this file to public repositories
   - Use environment variables or secrets management instead
   - Consider using `.env` files with python-dotenv (add `.env` to `.gitignore`)

2. **Missing .gitignore**:
   - Repository lacks `.gitignore`
   - Risk of committing cache/temp files, credentials, large data files
   - **URGENT RECOMMENDATION**: Create `.gitignore` immediately:
     ```gitignore
     # Python
     __pycache__/
     *.pyc
     *.pyo
     *.pyd
     .Python
     *.so

     # Data files
     *.h5
     *.hdf5
     *.keras
     *.npz
     *.npy

     # Cache directories
     /AILH_CACHE/
     /AILH_TMP/

     # Credentials (CRITICAL!)
     UTILITIES/old_config.py
     *.env
     .env
     credentials.json

     # IDE
     .vscode/
     .idea/
     *.swp
     *.swo

     # Logs
     *.log

     # OS
     .DS_Store
     Thumbs.db
     ```

### Recommended Security Practices

1. **Use Environment Variables for Credentials**:
   ```python
   import os

   DATAGATE_USERNAME = os.environ.get('DATAGATE_USERNAME')
   DATAGATE_PASSWORD = os.environ.get('DATAGATE_PASSWORD')

   if not DATAGATE_USERNAME or not DATAGATE_PASSWORD:
       raise ValueError("API credentials not found in environment variables")
   ```

2. **Credential Rotation**:
   - Rotate API credentials immediately if code is shared
   - Use temporary tokens instead of passwords where possible
   - Implement credential expiration policies

3. **Always work in AI_DEV**: Make all code changes in `AI_DEV/`, not `AI_ATTIC/`

4. **Test before committing**:
   ```bash
   # Run GPU tests
   python test_gpu_cuda.py

   # Validate WAV files
   python test_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS
   ```

5. **Memory cleanup in scripts**:
   ```python
   import gc
   import tensorflow as tf
   import torch

   # After GPU operations
   tf.keras.backend.clear_session()
   torch.cuda.empty_cache()
   gc.collect()
   ```

6. **Error handling with GPU fallback**:
   ```python
   try:
       # GPU operations
       result = process_on_gpu(data)
   except Exception as e:
       logger.error(f"GPU processing failed: {e}")
       # Fallback to CPU
       result = process_on_cpu(data)
   ```

7. **Logging instead of print**:
   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   logger.info("Processing started")
   logger.debug("Detailed debug info")
   logger.error("Error occurred")
   ```

---

## File Naming Conventions

### Input WAV Files
```
Local Format: sensor_id~recording_id~timestamp~gain_db.wav
Example: sensor_042~rec_98765~20240118_143022~45.wav

FCS DataGate Format: loggerId~recordingId~timestamp~gain.wav
Example: logger_001~rec_98765~20240118_143022~45.wav

Fields:
  - sensor_id/loggerId: Unique sensor/logger identifier
  - recording_id/recordingId: Unique recording session ID
  - timestamp: YYYYMMdd_HHmmss format
  - gain_db/gain: Hydrophone gain (0-200dB, 1dB = not in water)
```

### Output Files
```
Models: model_name.keras or model_name.h5
Datasets: dataset_name.h5 or dataset_name.hdf5
Results: results.json or leak_report.csv
Plots: plot_name.png (default) or plot_name.svg (with --svg)
Logs: process_name.log
```

---

## FCS DataGate Integration

### API Overview
- **Base URL**: `https://api.omnicoll.net/datagate/`
- **API Endpoints**:
  - `LOGGER_URL`: `/api/loggerapi.ashx` - Get loggers and sites
  - `RECORDINGS_URL`: `/api/recordingsapi.ashx` - List recordings
  - `GETRECORDINGS_URL`: `/api/getrecordingsapi.ashx` - Download recordings
- **Authentication**: Basic Auth (username/password from `old_config.py`)
- **Response Format**: XML (parsed to dict via xmltodict)
- **Data Organization**: Account → Site → Logger → Recordings

### Data Fetching Workflow

**⚠️ Note**: `datagate_sync.py` uses hardcoded paths - no CLI arguments for configuration.

```bash
cd FCS_UTILS
python datagate_sync.py
```

**Hardcoded Configuration** (lines 10-14):
```python
LOGS_DIR = "/DEVELOPMENT/ROOT_AILH/AILH_LOGS"
DATA_SENSORS = "/DEVELOPMENT/ROOT_AILH/DATA_SENSORS"
```

**What it does:**
1. Connects to FCS DataGate API with credentials from `old_config.py`
2. Fetches account hierarchy with parameters:
   - `ShowAssociations=true`
   - `ShowNestedLevels=true`
   - `ShowSubAccounts=true`
   - `SummaryOnly=false`
3. Downloads WAV files + JSON metadata
4. Filters for `recordingType=="3"` (audio recordings only)
5. Date range: Last 365 days from today (UTC)
6. Uses async/await with retry logic (exponential backoff)
7. Incremental fetching (tracks `lastId`, deduplicates)
8. Skips existing files automatically
9. Saves with format: `{loggerId}~{recordingId}~{timestamp}~{gain}.wav`

**Directory Structure Created**:
```
DATA_SENSORS/
└── {siteId}_{siteName}_{siteStation}/
    └── {loggerName}_{loggerId}/
        ├── {loggerId}~{recordingId}~{timestamp}~{gain}.wav
        └── {loggerId}~{recordingId}.json (metadata)
```

**JSON Output Files** (saved to LOGS_DIR):
- `fcs_accounts.json`: All subaccounts
- `fcs_loggers.json`: All loggers
- `fcs_logger_{logger_id}_recordings.json`: Recordings per logger (incremental)
- `fcs_api_logger.json`: Full API response

**Path Sanitization**:
```python
re.sub(r'[<>:"/\\|?*\s]+', "_", value)  # Replaces unsafe chars with underscore
```

**Siteidtext Parsing Logic** (lines 147-177):
Parses format: `{loggerName} {siteId} {siteName} {siteStation}`
- Handles missing fields by replacing with "OFF"
- Normalizes null/0/"" values

### Module Architecture

**datagate_client.py** (42 lines):
```python
@retry(
    stop=stop_after_attempt(5),          # 5 attempts
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s, 10s, 10s
    reraise=True
)
async def fetch_data(url, params=None):
    # HTTP timeout: 60 seconds
    # Returns: text for text/xml/json, bytes for binary (WAV)
```

**datagate_sync.py** (292 lines):
- `get_accounts()`: Fetches account hierarchy
- `get_loggers()`: Extracts logger list
- `get_logger_recordings_list()`: Fetches recording metadata per logger
- `process_recording()`: Downloads and saves individual WAV + JSON
- **Incremental update logic**: Only fetches recordings with `id > lastId`
- **Deduplication**: Tracks seen recording IDs to avoid duplicates

---

## Incremental Learning Methodology

### Overview
The system implements adaptive continuous learning by integrating detection outcomes and newly labeled data into the training dataset.

### Workflow
1. **Initial Training**: Train on ideal reference data (REFERENCE_DATA/TRAINING)
2. **Classification**: Classify real-world signals
3. **Voting Mechanism**: If ≥50% long-term segments predict leak → classify as LEAK
4. **Filtering**: Select top/bottom 50% confidence segments based on rules
5. **Update Dataset**: Add filtered segments to UPDATE_DATA/
6. **Retrain**: Incrementally update model with new data
7. **Iterate**: Repeat classification with improved model

### Confidence Threshold
- Default: 0.8 (configurable via `INCREMENTAL_CONFIDENCE_THRESHOLD`)
- Only segments with confidence ≥ threshold are used for incremental learning
- Prevents low-quality predictions from degrading model performance

### Incremental Learning Rules

**For Pseudo-labeled Data** (model predictions without manual verification):
- **Predicted LEAK**: Select top 50% confidence → UPDATE_DATA/POSITIVE/LEAK
- **Predicted NORMAL**: Select bottom 50% confidence → UPDATE_DATA/NEGATIVE/NORMAL

**For True-labeled Data** (manually verified):
- **True Positive (TP)**: Top 50% confidence → UPDATE_DATA/POSITIVE/LEAK
- **False Positive (FP)**: Bottom 50% confidence → UPDATE_DATA/NEGATIVE/NORMAL
- **True Negative (TN)**: Bottom 50% confidence → UPDATE_DATA/NEGATIVE/NORMAL
- **False Negative (FN)**: Top 50% confidence → UPDATE_DATA/POSITIVE/LEAK

---

## Model Performance Metrics

### Evaluation Metrics (from README.md)
- **Accuracy**: (TP + TN) / (TP + FP + FN + TN)
- **Precision**: TP / (TP + FP)
- **Sensitivity (Recall)**: TP / (TP + FN)
- **Specificity**: TN / (FP + TN)
- **AUC**: Area Under ROC Curve

### Expected Performance
- Monitor all metrics during training/validation
- Track changes across incremental learning rounds
- Generate confusion matrices for detailed analysis
- Plot ROC curves for threshold tuning

---

## Error Reference

### Common Errors and Solutions

**1. Tensor Rank Mismatch**:
```
RuntimeError: required rank 4 tensor to use channels_last format
```
**Cause**: Tensor shape doesn't match expected CNN input dimensions
**Solution**: Ensure input shape is `(batch, height, width, channels)` or `(batch, channels, height, width)`

**2. GPU Out of Memory**:
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solutions**:
```python
# Reduce batch size
CNN_BATCH_SIZE = 32  # or 16

# Enable memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Clear session between runs
tf.keras.backend.clear_session()
```

**3. CUDA/cuDNN Errors**:
```bash
# Check GPU status
python test_gpu_cuda.py

# Verify CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**4. WAV File Issues**:
```bash
# Validate WAV files
python test_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS

# Normalize if needed
python normalize_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS --output ../SENSOR_DATA/NORMALIZED
```

**5. Import Errors**:
```bash
# Reinstall dependencies
cd AI_ATTIC
pip install -r requirements.txt --force-reinstall

# Install missing FCS dependencies
pip install httpx tenacity xmltodict
```

**6. Performance Issues**:
```bash
# Run optimization tests
python test_optimizations.py

# Check disk performance
python test_disk_settings.py
python bench_smallfiles_ext4.py
```

---

## Module Relationships & Dependencies

### Core Processing Chain
```
WAV Files (sensor_id~recording_id~timestamp~gain_db.wav)
    ↓
normalize_wav_files.py → Normalized WAV
    ↓
pipeline.py / cnn_mel_processor.py → Mel Spectrograms
    ↓
dataset_builder.py / leak_dataset_builder_v15.py → HDF5 Datasets (.h5)
    ↓
dataset_trainer.py / leak_dataset_trainer_v15.py → Trained Models (.keras)
    ↓
dataset_classifier.py / leak_directory_classifier.py → Classifications (JSON/CSV)
    ↓
dataset_learner.py / leak_dataset_learner.py → Improved Models
    ↓
[Loop back to classification with updated model]
```

### External Data Flow
```
FCS DataGate API (api.omnicoll.net)
    ↓
datagate_client.py (HTTP + retry logic)
    ↓
datagate_sync.py (Sync logic + file organization)
    ↓
WAV + JSON metadata saved to SENSOR_DATA/RAW_SIGNALS/
    ↓
AI_DEV processing pipeline
```

### Configuration Hierarchy
```
UTILITIES/old_config.py (Legacy + credentials - DO NOT USE)
    ↓
AI_DEV/global_config.py (Active configuration)
    ↓
Individual scripts import global_config
    ↓
Runtime environment variables
```

---

## Key Technical Insights for AI Assistants

### When Modifying Code

1. **Always preserve two-stage temporal segmentation**:
   - Long segments isolate non-stationary characteristics
   - Short segments capture quasi-stationary features
   - This is CORE to the approach - do not remove

2. **Maintain Mel spectrogram generation**:
   - Frame segmentation → Windowing → FFT → Mel filter bank
   - One-dimensional matrix per short segment (Xij)
   - Two-dimensional matrix per long segment (Xi)
   - Three-dimensional matrix for full signal (X)

3. **Respect voting mechanism**:
   - Each long segment gets a probability [0, 1]
   - ≥50% segments predict leak → final classification = LEAK
   - This reduces non-stationary noise impact

4. **GPU optimization is critical**:
   - Use batch processing where possible
   - Clean up GPU memory explicitly
   - Provide CPU fallback for compatibility
   - Monitor memory usage during operations

5. **Thread safety**:
   - Audio pipeline uses ThreadPoolExecutor
   - Manage worker count based on CPU_COUNT
   - Avoid race conditions in shared resources

### Common Pitfalls to Avoid

1. **DO NOT use librosa**: Use numpy, scipy, torchaudio instead (per guidelines)
2. **DO NOT hardcode paths**: Use global_config.py paths or CLI arguments
3. **DO NOT skip normalization checks**: Even if files are pre-normalized
4. **DO NOT ignore gain 1dB files**: These indicate sensor not in water - filter out
5. **DO NOT modify AI_ATTIC**: This is archive only - work in AI_DEV
6. **DO NOT commit credentials**: Check UTILITIES/old_config.py before committing
7. **DO NOT skip error handling**: GPU operations can fail - always have fallback
8. **DO NOT mix label sets**: Verify which label set matches your dataset before training

### When Adding New Features

1. Add to `AI_DEV/`, not `AI_ATTIC/`
2. Support `--verbose`, `--debug`, `--svg` flags
3. Generate JSON output for data
4. Generate PNG (and optionally SVG) for plots
5. Use `~` delimiter for filenames
6. Document code thoroughly
7. Add command-line argument parsing
8. Include console reports with summary statistics
9. Optimize for multiprocessing/threading
10. Test with `test_gpu_cuda.py` if using GPU
11. Implement GPU memory cleanup
12. Add CPU fallback for GPU operations

### Documentation Requirements

When creating new scripts, always include:
```python
"""
Script Name: script_name.py
Purpose: Brief description of what this script does
Input: Expected input format/files
Output: Expected output format/files
Usage: python script_name.py --arg1 value1 --arg2 value2

Author: [Name]
Date: YYYY-MM-DD
"""

# Import statements
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function with clear logic flow"""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--input", "-i", required=True, help="Input path")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    parser.add_argument("--svg", action="store_true", help="Save plots as SVG")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Implementation here

    # GPU cleanup before exit
    if using_gpu:
        tf.keras.backend.clear_session()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
```

---

## Version History and Variants

The codebase contains multiple versions of key scripts:

### v15 Variants
- **leak_dataset_builder_v15.py** (676 lines) - Multi-split dataset support
- **leak_dataset_trainer_v15.py** (711 lines) - Enhanced training features
- Improvements over base versions (exact changes not documented)

### cnn_mel_* Variants
Mel spectrogram-specific implementations:
- `cnn_mel_trainer.py` (339 lines) - Mel-optimized trainer
- `cnn_mel_classifier.py` (164 lines) - Mel-optimized classifier
- `cnn_mel_learner.py` (189 lines) - Mel-optimized incremental learner
- `cnn_mel_processor.py` (351 lines) - Mel spectrogram processing
- `cnn_mel_tuner.py` (267 lines) - Hyperparameter tuning

### test_* Variants
- `test_wav_files.py` (170 lines) - Original WAV validation
- `test_wav_files_v2.py` (119 lines) - Updated validation

### pipeline.py Variants
- **AI_DEV/pipeline.py** (417 lines) - Current optimized version
- **AI_ATTIC/pipeline.py** (271 lines) - Older version (**146 lines shorter**)
- Key differences: Enhanced CPU-GPU buffering, better prefetch strategy

**AI Assistant Note**: When asked to modify a script, clarify with the user which version they want updated, or update all relevant variants to maintain consistency.

---

## Git History & Development Status

### Recent Activity
```
Latest Commits:
5aa97d8 (8 min ago)   - docs: add comprehensive CLAUDE.md
c64afd5 (21 hours ago) - massive updates to AI_ATTIC and AI_DEV modules
                         → 65 files changed, 17,628 insertions(+), 840 deletions
fe17372 (2 days ago)   - chore(config): update directory paths
bda01eb (2 days ago)   - Delete FCS_UTILS/__pycache__
0c2b764 (2 days ago)   - Delete UTILITIES/__pycache__
```

### Development Activity
- **Massive refactoring** 21 hours ago with **17,628 lines added**
- Active cleanup of cache directories
- Recent documentation improvements
- Current branch: `claude/claude-md-mi4dhprryhqr7c6w-01LgGSbMKZSwDXhbyNs3VtR4`

---

## Quick Reference

### Essential Files
| File | Purpose | Location |
|------|---------|----------|
| `global_config.py` | Main configuration | `AI_DEV/global_config.py` |
| `README.md` | Project requirements | `AI_ATTIC/README.md` |
| `OPTIMIZATION_GUIDE.md` | Performance guide | `AI_ATTIC/OPTIMIZATION_GUIDE.md` |
| `requirements.txt` | Python dependencies | `AI_ATTIC/requirements.txt` |
| `pipeline.py` | Optimized audio processing | `AI_DEV/pipeline.py` (417 lines) |
| `cnn_mel_processor.py` | Mel spectrogram processing | `AI_DEV/cnn_mel_processor.py` |
| `test_gpu_cuda.py` | GPU diagnostics suite | `AI_DEV/test_gpu_cuda.py` (1,590 lines!) |

### Common Commands
```bash
# Build dataset
python dataset_builder.py --input ../REFERENCE_DATA/TRAINING --output ../OUTPUT/training.h5 -v

# Train model
python dataset_trainer.py --dataset ../OUTPUT/training.h5 --model ../OUTPUT/model.keras -v

# Classify directory
python leak_directory_classifier.py --model ../OUTPUT/model.keras --input ../SENSOR_DATA/RAW_SIGNALS -v

# Incremental learning
python dataset_learner.py --model ../OUTPUT/model.keras --update-data ../UPDATE_DATA -v

# Test GPU
python test_gpu_cuda.py

# Normalize WAV files
python normalize_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS --output ../SENSOR_DATA/NORMALIZED -v

# Fetch FCS data
cd ../FCS_UTILS && python datagate_sync.py --output ../SENSOR_DATA/RAW_SIGNALS -v
```

### Important Constants
```python
SAMPLE_RATE = 4096 Hz
SAMPLE_DURATION = 10.0 s
DELIMITER = '~'
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 200
CNN_LEARNING_RATE = 0.001
INCREMENTAL_CONFIDENCE_THRESHOLD = 0.8
LONG_SEGMENTS = [0.125, 0.25, 0.5, 0.75, 1.0]
SHORT_SEGMENTS = [64, 128, 256, 512, 1024]
```

---

## Troubleshooting Checklist

### Before Starting Development

- [ ] Verify which label set matches your dataset
- [ ] Check that `AI_DEV/pipeline.py` is being used (not AI_ATTIC version)
- [ ] Ensure `.gitignore` is created to prevent credential leaks
- [ ] Verify GPU is available and working (`python test_gpu_cuda.py`)
- [ ] Check that all dependencies are installed
- [ ] Confirm WAV files are normalized
- [ ] Verify hydrophone gain values (exclude 1dB files)

### Common Issues Checklist

- [ ] GPU out of memory → Reduce batch size
- [ ] Tensor rank mismatch → Check input dimensions
- [ ] Import errors → Reinstall dependencies
- [ ] Performance issues → Run optimization tests
- [ ] WAV file errors → Run validation scripts
- [ ] CUDA errors → Check CUDA installation
- [ ] Label mismatch → Verify dataset labels match config

---

## Contact and Resources

### Documentation
- **This Guide**: `CLAUDE.md` (1,004+ lines)
- **Project Requirements**: `AI_ATTIC/README.md` (171 lines)
- **Optimization Guide**: `AI_ATTIC/OPTIMIZATION_GUIDE.md` (253 lines)

### Code Generation Guidelines
When generating or modifying code:
1. Follow all conventions in this document
2. Reference `AI_ATTIC/README.md` for technical requirements
3. Check `AI_DEV/global_config.py` for current parameters
4. Test with appropriate `test_*.py` scripts
5. Document thoroughly with docstrings
6. Optimize for performance (GPU/threading)
7. Handle errors gracefully with fallbacks
8. Verify label set consistency
9. Implement GPU memory cleanup
10. Add progress bars (tqdm) for long operations

---

## Project Status

**Current State**: Active development
**Last Major Update**: 2024-11-18 (massive updates: 17,628 lines added)
**Code Size**: 51 Python files, ~20,000 lines
**Active Branch**: AI_DEV
**Archive Branch**: AI_ATTIC
**Git Status**: Clean
**Production Ready**: Yes (highly optimized)
**Security Status**: ⚠️ NEEDS ATTENTION (missing .gitignore, exposed credentials)

---

## Known Issues and Bugs

### 🐛 Critical Bugs

1. **cnn_mel_processor.py - Broken Imports** (CRITICAL)
   - **Line 1**: Imports variables from `old_config.py` that don't exist
   - Missing variables: `SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, `N_MELS`, `LONG_TERM_SEC`, `SHORT_TERM_SEC`, `STRIDE_SEC`, `CNN_BATCH_SIZE`, `MAX_THREAD_WORKERS`, `TMPDIR`
   - **Impact**: Script will crash immediately on import
   - **Fix**: Either update `old_config.py` with missing variables or change imports to use `global_config.py`

2. **Configuration Fragmentation**
   - Parameters defined in `global_config.py` are NOT used by most scripts
   - Each script has its own hardcoded FFT/Mel parameters
   - **Impact**: Inconsistent mel spectrograms across different parts of the pipeline
   - **Fix**: Centralize configuration and enforce imports from single source

3. **Sample Rate Mismatch in old_config.py**
   - `old_config.py` defines `DEFAULT_SAMPLING_RATE = 44100`
   - All other code uses `SAMPLE_RATE = 4096`
   - **Impact**: If `old_config.py` is used, severe audio processing errors
   - **Fix**: Update `old_config.py` or deprecate it entirely

### ⚠️ Runtime Errors

**From leak_report.csv** (AI_ATTIC):
```
RuntimeError: required rank 4 tensor to use channels_last format
```
- All 18 test files in leak_report.csv failed with this error
- **Cause**: Tensor shape mismatch - expecting `(batch, height, width, channels)`
- **Common fix**: Ensure input tensors are properly reshaped before CNN forward pass

### 📋 Configuration Issues

1. **Hardcoded Paths Everywhere**
   - Most scripts use `/DEVELOPMENT/` paths
   - Not configurable via environment or CLI
   - Breaks portability

2. **Multiple Label Sets**
   - Three different label definitions across the codebase
   - Can cause training/inference mismatch

3. **Batch Size Proliferation**
   - Five different batch sizes across scripts (16, 64, 256, 512, 2048, 16384)
   - No clear documentation on when to use which

---

## Critical Actions Required

### Immediate (Security)
1. **Create `.gitignore`** to prevent credential leaks
2. **Rotate ALL FCS DataGate credentials** (both sets!) if code has been shared
3. **Move credentials to environment variables**
4. **Remove commented credentials from old_config.py** (still in git history!)

### Immediate (Code Fixes)
5. **Fix cnn_mel_processor.py imports** - currently broken
6. **Standardize Mel spectrogram parameters** across all scripts
7. **Fix or remove old_config.py** sample rate (44100 vs 4096)

### Immediate (Branch Reconciliation) ⭐ NEW
8. **Reconcile label sets** between main and feature branch
   - Main: 6 categories (LEAK, NORMAL, QUIET, RANDOM, MECHANICAL, UNCLASSIFIED)
   - Feature: 5 categories (BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED)
   - **Impact**: Affects model architecture and dataset organization
9. **Merge DOCS/ folder** from main branch to feature branch
   - Critical documentation missing from feature branch
   - Includes official requirements, hardware specs, test results
10. **Document sample rate upscaling** requirement (4096 → 8192 Hz)
    - Currently only mentioned in main branch DOCS/AILH.md
    - Needs to be in code comments and configuration

### Short-term (Documentation)
11. **Clarify active label set** - update all configs to match chosen standard
12. **Document actual vs documented segmentation** (overlapping windows, not multi-size)
13. **Document version differences** (base vs v15)
14. **Document pipeline.py evolution** (271 → 417 lines)
15. **Add configuration source-of-truth** guide
16. **Implement MASTER_DATASET workflow** from main branch documentation
    - Create data migration scripts
    - Update all scripts to use MASTER_DATASET structure
17. **Clarify FCS_UTILS vs FCS_TOOLS naming** discrepancy

### Medium-term (Quality)
18. **Centralize configuration** - single source of truth
19. **Add unit tests** (currently only integration/diagnostic tests)
20. **Centralize logging configuration**
21. **Create proper package structure** (add `__init__.py` files)
22. **Add example data/sample files** for testing
23. **Make paths configurable** (remove hardcoded /DEVELOPMENT/ paths)
24. **Migrate folder structure** to official ROOT_AILH layout from DOCS/AILH.md

---

---

## Branch Differences Summary

### Main Branch vs Feature Branch

**⚠️ IMPORTANT**: There are significant differences between the main branch and the current feature branch:

| Aspect | Main Branch (DOCS/) | Feature Branch (Current) |
|--------|---------------------|--------------------------|
| **Documentation** | Has DOCS/ folder with AILH.md, OPTIMIZATION_GUIDE.md, test results | No DOCS/ folder |
| **Label Set** | 6 categories: LEAK, NORMAL, QUIET, RANDOM, MECHANICAL, UNCLASSIFIED | 5 categories: BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED |
| **Folder Structure** | ROOT_AILH with comprehensive DATA_STORE subdirs | /DEVELOPMENT/ROOT_AILH with simpler structure |
| **Dataset Concept** | MASTER_DATASET as source of truth | No MASTER_DATASET concept documented |
| **FCS Module Name** | May use FCS_TOOLS | Uses FCS_UTILS |
| **Hardware Specs** | Documented (RTX 5090, CUDA 12.8) | Not in code, only in DOCS |
| **Sample Rate Note** | "4096 Hz REQUIRED TO UPSCALE TO 8192" | Just "4096 Hz" |

**Recommendation for Merging**:
1. Reconcile label sets first (critical for model compatibility)
2. Migrate to MASTER_DATASET structure from main branch
3. Adopt the comprehensive DATA_STORE folder organization
4. Preserve hardware documentation in merged branch
5. Clarify FCS_UTILS vs FCS_TOOLS naming
6. Document the upscaling requirement in code comments

---

## DOCS Folder Contents (Main Branch Only)

The main branch contains a `DOCS/` folder with critical project information not present in the feature branch:

- **AILH.md** (189 lines): Official requirements, folder structure, label sets
- **OPTIMIZATION_GUIDE.md** (253 lines): Performance tuning guide
- **LeakDetectionTwoStageSegmentation.pdf**: Academic paper/methodology reference
- **test_gpu_cuda_results.txt**: RTX 5090 GPU specifications and test results
- **test_disk_tune_results.txt**: Filesystem optimization results (24,801.7 files/s)

**⚠️ Action Required**: When merging branches, ensure DOCS/ folder is preserved and updated in the final branch.

---

*This CLAUDE.md file is maintained for AI assistants working on the AILH_MASTER codebase. Keep it updated as the project evolves.*

**Last Updated**: 2024-11-18 (DOCS Integration v4.0)
**Maintained By**: AI Assistant (Claude)
**Repository**: AILH_MASTER
**Branch**: claude/claude-md-mi4dhprryhqr7c6w-01LgGSbMKZSwDXhbyNs3VtR4
**Total Files**: 51 Python files (~20,000 lines)
**Documentation Version**: 4.0 (DOCS Folder Integration - Main Branch Analysis)
**Hardware**: Windows 11 + WSL2 Ubuntu, RTX 5090 (24GB), CUDA 12.8, 24,801.7 files/s I/O
