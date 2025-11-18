# CLAUDE.md - AI Assistant Guide for AILH_MASTER

## Project Overview

**AILH_MASTER** is a sophisticated **Acoustic Leak Detection System for Urban Water Supply Networks** using deep learning. The system employs Convolutional Neural Networks (CNN) with Mel spectrograms to identify leaks in water pipelines by analyzing acoustic signals from hydrophone sensors.

### Core Technology
- **Purpose**: Detect water leaks in urban pipeline networks using acoustic signal analysis
- **Approach**: Two-stage temporal segmentation + CNN with Mel spectrograms
- **Key Feature**: Adaptive incremental learning model that improves with real-world data
- **Primary Language**: Python 3
- **ML Frameworks**: TensorFlow 2.20.0 (self-built), PyTorch 2.9.1, Keras
- **Optimization Level**: Production-grade with extensive GPU acceleration (CUDA)

### Signal Processing Workflow
1. **Input**: 10-second audio samples at 4096 Hz from hydrophone sensors (0-200dB gain)
2. **Segmentation**: Two-stage temporal segmentation
   - Long-term: 0.125s, 0.25s, 0.5s, 0.75s, 1.0s
   - Short-term: 64, 128, 256, 512, 1024 points
3. **Transform**: Convert to Mel spectrograms (64 mels, 256 FFT, hop=32)
4. **Classify**: CNN model outputs classification probabilities
5. **Decision**: Voting mechanism on long-term segments (>50% = leak)
6. **Learn**: Incremental learning with confidence thresholds (0.8)

### Classification Categories
- **LEAK**: Confirmed leak signatures
- **NORMAL**: Normal pipeline operation sounds
- **QUIET**: Minimal/background noise
- **RANDOM**: Random interference
- **MECHANICAL**: Mechanical noise (pumps, valves)
- **UNCLASSIFIED**: Unknown/ambiguous signals

---

## Repository Structure

```
AILH_MASTER/
├── main.py                    # Empty entry point (placeholder)
│
├── AI_DEV/                    # ⭐ ACTIVE DEVELOPMENT DIRECTORY
│   ├── global_config.py       # 🔧 Primary configuration file
│   │
│   ├── Dataset Building:
│   │   ├── dataset_builder.py              # HDF5 dataset builder
│   │   └── leak_dataset_builder_v15.py     # Multi-split dataset builder v15
│   │
│   ├── Model Training:
│   │   ├── dataset_trainer.py              # CNN trainer
│   │   ├── leak_dataset_trainer_v15.py     # Trainer v15
│   │   ├── cnn_mel_trainer.py              # Mel spectrogram trainer
│   │   └── cnn_mel_tuner.py                # Hyperparameter tuning (Optuna)
│   │
│   ├── Classification:
│   │   ├── dataset_classifier.py           # Audio file classifier
│   │   ├── leak_directory_classifier.py    # Batch directory classifier
│   │   └── cnn_mel_classifier.py           # Mel-based classifier
│   │
│   ├── Incremental Learning:
│   │   ├── dataset_learner.py              # Incremental learner
│   │   ├── leak_dataset_learner.py         # Leak-specific learner
│   │   └── cnn_mel_learner.py              # Mel-based learner
│   │
│   ├── Feature Processing:
│   │   ├── cnn_mel_processor.py            # Mel spectrogram processor
│   │   └── pipeline.py                     # Optimized audio pipeline (GPU accelerated)
│   │
│   ├── Utilities:
│   │   ├── normalize_wav_files.py          # WAV normalization
│   │   └── shuffle_data_for_training.py    # Data shuffling utility
│   │
│   └── Testing/Benchmarking:
│       ├── test_gpu_cuda.py                # Comprehensive GPU diagnostics (1,590 lines!)
│       ├── test_cpu_fixes.py               # CPU configuration tests
│       ├── test_optimizations.py           # Performance optimization tests
│       ├── test_disk_settings.py           # Filesystem performance tests
│       ├── test_wav_files.py               # WAV file validation
│       ├── test_wav_files_v2.py            # WAV validation v2
│       └── bench_smallfiles_ext4.py        # Filesystem benchmarking (37KB)
│
├── AI_ATTIC/                  # 📦 ARCHIVE DIRECTORY
│   ├── README.md              # 📄 Comprehensive project requirements (172 lines)
│   ├── OPTIMIZATION_GUIDE.md  # 📄 Detailed optimization documentation (254 lines)
│   ├── requirements.txt       # Python dependencies (auto-generated)
│   ├── admin_config.py        # Archived configuration
│   └── [same file structure as AI_DEV - older versions]
│
├── FCS_UTILS/                 # 🌐 External Data Integration
│   ├── datagate_client.py     # Async HTTP client for FCS DataGate API
│   └── datagate_sync.py       # Fetches sensor data from cloud (api.omnicoll.net)
│
└── UTILITIES/                 # 🛠️ Shared Utilities
    ├── old_config.py          # ⚠️ Legacy config (contains credentials - DO NOT COMMIT)
    └── gen_requirements.py    # Auto-generates requirements.txt
```

### Key Directory Relationships
- **AI_DEV**: Active development - make ALL code changes here
- **AI_ATTIC**: Archive/backup - reference only, DO NOT modify
- **FCS_UTILS**: External data fetching - standalone module
- **UTILITIES**: Shared tools - used by both AI_DEV and FCS_UTILS

---

## Development Guidelines

### Critical Conventions (MUST FOLLOW)

1. **File Delimiter**: Use `~` (tilde) for all file field separators
   ```
   Format: sensor_id~recording_id~timestamp~gain_db.wav
   Example: sensor_001~rec_12345~20240118_143022~45.wav
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
   - Use type hints where practical
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
KERAS_HOME = "/mnt/d/AILH_CACHE/KERAS"   # Keras cache
TORCH_HOME = "/mnt/d/AILH_CACHE/TORCH"   # PyTorch cache
```

**GPU/CUDA Settings**:
```python
KERAS_BACKEND = "tensorflow"
TF_GPU_ALLOCATOR = "cuda_malloc_async"
CUDA_VISIBLE_DEVICES = "0"                # Use only first GPU
TF_FORCE_GPU_ALLOW_GROWTH = "true"
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

**Data Labels**:
```python
DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']
# NOTE: AI_ATTIC/README.md lists different categories:
# ['LEAK', 'NORMAL', 'QUIET', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']
# Verify which set is currently active before modifying
```

**File Delimiter**:
```python
DELIMITER = '~'  # Used in all filename parsing
```

---

## Data Directory Structure

The system expects a specific folder layout (relative to BASE_DIR):

```
BASE_DIR/
├── SENSOR_DATA/
│   ├── RAW_SIGNALS/           # Unprocessed sensor recordings
│   └── LABELED_SEGMENTS/      # Manually labeled segments
│
├── REFERENCE_DATA/            # Initial training/validation data
│   ├── TRAINING/              # Training dataset (70%)
│   │   ├── LEAK/
│   │   ├── NORMAL/
│   │   ├── QUIET/
│   │   ├── RANDOM/
│   │   ├── MECHANICAL/
│   │   └── UNCLASSIFIED/
│   └── VALIDATION/            # Validation dataset (20%)
│       ├── LEAK/
│       ├── NORMAL/
│       ├── QUIET/
│       ├── RANDOM/
│       ├── MECHANICAL/
│       └── UNCLASSIFIED/
│
└── UPDATE_DATA/               # Incremental learning data
    ├── POSITIVE/              # True positive labeled data
    │   ├── LEAK/
    │   ├── NORMAL/
    │   ├── QUIET/
    │   ├── RANDOM/
    │   ├── MECHANICAL/
    │   └── UNCLASSIFIED/
    └── NEGATIVE/              # False negative labeled data
        ├── LEAK/
        ├── NORMAL/
        ├── QUIET/
        ├── RANDOM/
        ├── MECHANICAL/
        └── UNCLASSIFIED/
```

### Data Split Ratios
- **Training**: 70% (REFERENCE_DATA/TRAINING)
- **Validation**: 20% (REFERENCE_DATA/VALIDATION)
- **Test**: 10% (typically held-out from validation)

### Incremental Learning Data Flow

**Pseudo-labeled Data** (model predictions):
- Leak predictions → Select top 50% confidence segments → UPDATE_DATA/POSITIVE/LEAK
- Normal predictions → Select bottom 50% confidence segments → UPDATE_DATA/NEGATIVE/NORMAL

**True-labeled Data** (manually verified):
- True Positive (TP): Top 50% → UPDATE_DATA/POSITIVE/LEAK
- False Positive (FP): Bottom 50% → UPDATE_DATA/NEGATIVE/NORMAL
- True Negative (TN): Bottom 50% → UPDATE_DATA/NEGATIVE/NORMAL
- False Negative (FN): Top 50% → UPDATE_DATA/POSITIVE/LEAK

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

```bash
# Single file classification
python dataset_classifier.py \
    --model ../OUTPUT/model.keras \
    --input sensor_001~rec_123~20240118_143022~45.wav \
    --output results.json \
    --verbose

# Batch directory classification
python leak_directory_classifier.py \
    --model ../OUTPUT/model.keras \
    --input ../SENSOR_DATA/RAW_SIGNALS \
    --output ../OUTPUT/classifications.json \
    --workers 8 \
    --verbose \
    --svg
```

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
tensorflow==2.20.0.dev0+selfbuilt   # Self-built TensorFlow with optimizations
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

### Installation

**From requirements.txt**:
```bash
cd AI_ATTIC
pip install -r requirements.txt
```

**Note**: TensorFlow is self-built. Standard TensorFlow can be installed with:
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
- GPU Memory: 4GB+ for optimal performance
- CuPy: Install matching CUDA version

**CPU Optimization**:
- Cores: 4+ CPU cores recommended
- RAM: 8GB+ for large datasets
- Storage: SSD for faster I/O

### Memory Management
- Adaptive batch sizes based on available memory
- Strategic garbage collection to prevent leaks
- HDF5 compression for dataset storage
- GPU memory cleanup after operations

---

## Security and Best Practices

### ⚠️ Security Warnings

1. **Credentials in Code**:
   - `UTILITIES/old_config.py` contains plaintext API credentials
   - **NEVER** commit this file to public repositories
   - Consider using environment variables or secrets management
   - Create `.gitignore` to exclude sensitive files

2. **Missing .gitignore**:
   - Repository lacks `.gitignore`
   - Risk of committing cache/temp files, credentials
   - **RECOMMENDATION**: Create `.gitignore` with:
     ```
     __pycache__/
     *.pyc
     *.pyo
     *.pyd
     .Python
     *.so
     *.h5
     *.hdf5
     /AILH_CACHE/
     /AILH_TMP/
     UTILITIES/old_config.py
     *.env
     .env
     ```

### Best Practices

1. **Always work in AI_DEV**: Make all code changes in `AI_DEV/`, not `AI_ATTIC/`

2. **Test before committing**:
   ```bash
   # Run GPU tests
   python test_gpu_cuda.py

   # Validate WAV files
   python test_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS
   ```

3. **Memory cleanup in scripts**:
   ```python
   import gc
   import tensorflow as tf

   # After GPU operations
   tf.keras.backend.clear_session()
   gc.collect()
   ```

4. **Error handling**:
   ```python
   try:
       # GPU operations
       result = process_on_gpu(data)
   except Exception as e:
       logger.error(f"GPU processing failed: {e}")
       # Fallback to CPU
       result = process_on_cpu(data)
   ```

5. **Logging instead of print**:
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
Format: sensor_id~recording_id~timestamp~gain_db.wav
Example: sensor_042~rec_98765~20240118_143022~45.wav

Fields:
  - sensor_id: Unique sensor identifier
  - recording_id: Unique recording session ID
  - timestamp: YYYYMMdd_HHmmss format
  - gain_db: Hydrophone gain (0-200dB, 1dB = not in water)
```

### Output Files
```
Models: model_name.keras or model_name.h5
Datasets: dataset_name.h5 or dataset_name.hdf5
Results: results.json
Plots: plot_name.png (default) or plot_name.svg (with --svg)
Logs: process_name.log
```

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

if __name__ == "__main__":
    main()
```

---

## Quick Reference

### Essential Files
| File | Purpose | Location |
|------|---------|----------|
| `global_config.py` | Main configuration | `AI_DEV/global_config.py` |
| `README.md` | Project requirements | `AI_ATTIC/README.md` |
| `OPTIMIZATION_GUIDE.md` | Performance guide | `AI_ATTIC/OPTIMIZATION_GUIDE.md` |
| `requirements.txt` | Python dependencies | `AI_ATTIC/requirements.txt` |
| `pipeline.py` | Optimized audio processing | `AI_DEV/pipeline.py` |
| `cnn_mel_processor.py` | Mel spectrogram processing | `AI_DEV/cnn_mel_processor.py` |

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
```

---

## Version History and Variants

The codebase contains multiple versions of key scripts:

- **v15 variants**: `leak_dataset_builder_v15.py`, `leak_dataset_trainer_v15.py`
  - Multi-split dataset support
  - Enhanced features

- **cnn_mel_* variants**: Mel spectrogram-specific implementations
  - `cnn_mel_trainer.py`, `cnn_mel_classifier.py`, `cnn_mel_learner.py`
  - Optimized for Mel features

- **test_* variants**: Multiple test versions
  - `test_wav_files.py` vs `test_wav_files_v2.py`
  - Incremental improvements

**AI Assistant Note**: When asked to modify a script, clarify with the user which version they want updated, or update all relevant variants to maintain consistency.

---

## External Dependencies and APIs

### FCS DataGate API
- **Purpose**: Fetch sensor recordings from Fracta Control Systems cloud
- **Base URL**: `https://api.omnicoll.net/datagate/`
- **Credentials**: Stored in `UTILITIES/old_config.py` (⚠️ plaintext)
- **Files**:
  - `datagate_client.py`: Async HTTP client
  - `datagate_sync.py`: Synchronization script
- **Data**: WAV files + JSON metadata

### Cloud Integration
The system can fetch data from external cloud services. Credentials are required and should be managed securely (environment variables recommended).

---

## Troubleshooting

### Common Issues

**GPU Out of Memory**:
```python
# Reduce batch size
CNN_BATCH_SIZE = 32  # or 16

# Enable memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**CUDA/cuDNN Errors**:
```bash
# Check GPU status
python test_gpu_cuda.py

# Verify CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**WAV File Issues**:
```bash
# Validate WAV files
python test_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS

# Normalize if needed
python normalize_wav_files.py --input ../SENSOR_DATA/RAW_SIGNALS --output ../SENSOR_DATA/NORMALIZED
```

**Import Errors**:
```bash
# Reinstall dependencies
cd AI_ATTIC
pip install -r requirements.txt --force-reinstall
```

**Performance Issues**:
```bash
# Run optimization tests
python test_optimizations.py

# Check disk performance
python test_disk_settings.py
python bench_smallfiles_ext4.py
```

---

## Contact and Resources

### Documentation
- **Project Requirements**: `AI_ATTIC/README.md`
- **Optimization Guide**: `AI_ATTIC/OPTIMIZATION_GUIDE.md`
- **This Guide**: `CLAUDE.md`

### Code Generation Guidelines
When generating or modifying code:
1. Follow all conventions in this document
2. Reference `AI_ATTIC/README.md` for technical requirements
3. Check `AI_DEV/global_config.py` for current parameters
4. Test with appropriate `test_*.py` scripts
5. Document thoroughly
6. Optimize for performance
7. Handle errors gracefully

---

## Project Status

**Current State**: Active development
**Last Major Update**: 2024-11 (massive updates and refactoring)
**Active Branch**: AI_DEV
**Archive Branch**: AI_ATTIC
**Git Status**: Clean
**Production Ready**: Yes (highly optimized)

---

*This CLAUDE.md file is maintained for AI assistants working on the AILH_MASTER codebase. Keep it updated as the project evolves.*

**Last Updated**: 2024-11-18
**Maintained By**: AI Assistant (Claude)
**Repository**: AILH_MASTER
