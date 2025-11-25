# AILH_MASTER - AI Acoustic Leak Detection System

## Table of Contents
1. [AI Behavior Guidelines](#ai-behavior-guidelines)
2. [Project Overview](#project-overview)
3. [Hardware Environment](#hardware-environment)
4. [Codebase Structure](#codebase-structure)
5. [Development Guidelines](#development-guidelines)
6. [Data Organization](#data-organization)
7. [Configuration Management](#configuration-management)
8. [Script Types & Purposes](#script-types--purposes)
9. [Development Workflows](#development-workflows)
10. [Technical Architecture](#technical-architecture)
11. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## AI Behavior Guidelines

**CRITICAL**: When working on this codebase, AI assistants MUST follow these protocols:

### Interaction Protocols
- **ALWAYS** wait for plan discussion and explicit GO_AHEAD before implementing changes
- **ALWAYS** ask before committing or pushing to repository
- **ALWAYS** present information, options, and recommendations for user approval
- **NEVER** make assumptions about user intent - ask clarifying questions
- Explain reasoning and trade-offs for technical decisions
- Provide multiple implementation options when applicable

### Code Quality Standards
- Follow all guidelines in Development Guidelines section below
- Test changes before committing
- Document all modifications with clear commit messages
- Verify file locations and paths match actual repository structure

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
- **RAM**: 94GB
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

### GPU Acceleration Strategy

The AILH system uses a **hybrid CPU/GPU architecture** to maximize performance across the entire pipeline:

#### Library Usage Guidelines

**NumPy** (CPU - Control Plane):
- File I/O and WAV loading
- Metadata, indexing, small CPU-side transforms
- Pipe graph, positions, configuration data
- HDF5/memmap dataset building

**CuPy** (GPU - DSP Operations):
- GPU-accelerated DSP: FFTs, correlations, filters, signal stacking
- Operations like "FFT + multiply + IFFT" or large array math
- SciPy-like DSP APIs on GPU via `cupyx.scipy.signal`
- Raw FFT/correlation performance without gradients

**PyTorch** (GPU - Machine Learning):
- All ML models: CNNs, leak window classifier, corrgram CNN, correlation referee MLP
- STFT/Mel feature extraction feeding directly into models
- Any operations requiring automatic differentiation
- End-to-end trainable pipelines

**NVMath / cuBLAS** (GPU - High-Performance Computing):
- Heavy GEMMs and batched FFT operations
- Maximum FLOP/s for linear algebra
- Benchmarking and performance-critical sections

**TensorRT** (GPU - Production Inference):
- Deployed inference for leak CNN + corrgram CNN in production
- Optimized model execution with INT8/FP16 quantization

#### Pipeline Stage Assignments

| Stage | Library | GPU Accelerated? | Notes |
|-------|---------|-----------------|-------|
| WAV load | NumPy + soundfile | ❌ CPU only | File I/O |
| Resampling | CuPy / cuSignal | ✅ GPU | Signal processing |
| Filtering (adaptive bandpass) | CuPy / cuSignal | ✅ GPU | DSP operations |
| Window segmentation | PyTorch | Optional GPU | Tensor ops |
| STFT / Mel | PyTorch | ✅ GPU | Feature extraction |
| Leak CNN | PyTorch / TensorRT | ✅ GPU | Model inference |
| Per-window correlation | CuPy or PyTorch FFT | ✅ GPU | Cross-correlation |
| Robust stacking | CuPy or PyTorch | Optional GPU | Signal averaging |
| Corrgram CNN | PyTorch | ✅ GPU | Model inference |
| Bayesian estimator | PyTorch MLP | Optional GPU | Statistical inference |
| HDF5 / memmap writer | NumPy + h5py | ❌ CPU only | File I/O |
| Metadata, configs | NumPy | ❌ CPU only | Control plane |

#### GPU-Accelerated Libraries Available

**Core Array / Math**:
- CuPy - GPU arrays (NumPy-compatible API)
- PyTorch - Tensors with autograd
- JAX - Composable transformations
- NVMath / cuBLAS / cuFFT / cuSPARSE - Low-level CUDA
- Numba CUDA - JIT compilation
- Triton - Custom CUDA kernel generation

**DSP & Signal Processing**:
- cuSignal / cupyx.scipy.signal - GPU-accelerated scipy.signal
- torchaudio - Audio processing in PyTorch
- PyTorch FFT - GPU FFT operations

**Machine Learning**:
- PyTorch - Primary ML framework
- TensorRT - Optimized inference
- cuML (RAPIDS) - GPU scikit-learn algorithms

**DataFrames / ETL**:
- cuDF (RAPIDS) - GPU-accelerated Pandas
- Polars GPU Engine - Fast DataFrame operations
- Dask + CuPy - Distributed GPU arrays
- cuIO - Fast data loading

**Computer Vision / Imaging** (for visualization):
- CV-CUDA - GPU OpenCV operations
- Torchvision - PyTorch vision utilities
- OpenCV CUDA builds - GPU-accelerated OpenCV

**Graph Algorithms** (for pipe networks):
- cuGraph (RAPIDS) - GPU graph algorithms
- PyTorch Geometric - Graph neural networks

**Visualization** (for reports):
- cuXfilter - GPU dashboards
- Datashader GPU - Large dataset visualization
- VisPy - OpenGL-based visualization

**PDE / Simulation** (for acoustic modeling):
- JAX - Automatic differentiation + GPU
- Warp (NVIDIA) - Physical simulation
- PyTorch + Triton - Custom kernels

#### Migration Strategy from SciPy

Traditional SciPy operations can be accelerated:

| SciPy Function | GPU Alternative | Speedup |
|----------------|----------------|---------|
| scipy.signal.fft | CuPy FFT / PyTorch FFT | 10-100× |
| scipy.signal.correlate | CuPy correlate | 20-50× |
| scipy.signal.filtfilt | cupyx.scipy.signal.filtfilt | 15-40× |
| scipy.signal.welch | Custom CuPy/PyTorch | 10-30× |
| scipy.sparse | cupy.sparse | 5-20× |
| scipy.linalg | cupy.linalg | 10-50× |

#### Performance Optimization Guidelines

1. **Keep data on GPU**: Minimize CPU↔GPU transfers
2. **Batch operations**: Process multiple signals simultaneously
3. **Use async transfers**: Overlap computation with I/O
4. **Profile hotspots**: Use NVIDIA Nsight Systems
5. **FP16 when possible**: 4× faster than FP32 on tensor cores
6. **Persistent GPU allocations**: Reuse memory buffers

---

## Codebase Structure

### Repository Layout
```
AILH_MASTER/
├── AI_DEV/                  # Main development scripts (6 files)
│   ├── global_config.py         # ⭐ Global configuration
│   ├── ai_builder.py            # Optimized GPU-accelerated processing pipeline
│   ├── dataset_builder.py       # Build and organize datasets
│   ├── dataset_classifier.py    # Batch classification utilities
│   ├── dataset_trainer.py       # General dataset training utilities
│   └── dataset_tuner.py         # Hyperparameter tuning (Keras Tuner / Optuna)
│
├── CORRELATOR_v2/           # Acoustic leak correlation system (17 Python files)
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── leak_correlator.py
│   ├── correlation_engine.py
│   ├── batch_gpu_correlator.py
│   ├── multi_leak_detector.py
│   ├── sensor_registry.py
│   ├── visualization.py
│   ├── examples/
│   └── ... (additional modules)
│
├── DOCS/                    # Documentation (33 files)
│   ├── AILH.md                  # Core requirements & specifications
│   ├── LeakDetectionTwoStageSegmentation.pdf  # Reference paper
│   ├── OPTIMIZATION_GUIDE.md
│   ├── test_disk_tune_results.json
│   ├── test_disk_tune_results.txt
│   ├── test_gpu_cuda_results.txt
│   ├── CODE_REVIEW_*.md         # Code review documents
│   ├── PERFORMANCE_*.md         # Performance analysis
│   └── ... (additional documentation)
│
├── FCS_TOOLS/               # Field Control System tools (2 files)
│   ├── datagate_client.py       # Credential rotation client
│   └── datagate_sync.py         # Data synchronization
│
├── UTILITIES/               # Utility scripts (17 files)
│   ├── test_gpu_cuda.py         # GPU/CUDA diagnostics
│   ├── test_disk_tune.py        # Disk I/O performance tuning
│   ├── test_wav_files.py        # WAV file validation
│   ├── test_wav_files_v2.py     # Enhanced WAV validation
│   ├── normalize_wav_files.py   # WAV file normalization
│   ├── shuffle_data_for_training.py  # Dataset shuffling
│   ├── gen_requirements.py      # Generate requirements.txt
│   ├── dataconv.py              # Data conversion utilities
│   ├── synthetic_leakgen.py     # Synthetic leak data generator
│   ├── wav_viewer.py            # WAV file viewer
│   └── ... (additional utilities)
│
├── .gitignore               # Git ignore rules
├── global_vars              # ⭐ Runtime configuration (LOGGING, PERFMON, VERBOSE, DEBUG)
└── CLAUDE.md                # This file
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
- Delimiter: ~ (tilde)
- ALL input WAV files MUST follow this nomenclature:
- sensor_id~recording_id~timestamp~gain_db.wav
- Example: S001~R123~20250118120530~100.wav
- Gain range: 0-200dB (1dB indicates hydrophone not touching water)

#### 2. Data Specifications
- **Sample Rate**: 4096 Hz
- **Sample Upscale Rate**: 8192 Hz
- **Sample Duration**: 10 seconds
- **Sample File Format**: WAV

#### 3. Command-Line Interface
All scripts MUST support:
- `--verbose`: Print information during processing steps
- `--debug`: Print detailed debugging information

#### 4. Script Verbosity, Logging, Debugging
All scripts MUST support the below global variables:
- **LOGGING=YES**: Print information during processing steps
- **PERFMON=NO**: Print detailed debugging information
- **VERBOSE=NO**: Print information during processing steps
- **DEBUG=NO**: Print detailed debugging information
- **From file**: /global_vars

#### 5. Optional variables
- **UPSCALE=NO**: Upscale audio data to 8192 Hz
- `--upscale`: Upscale audio data to 8192 Hz

#### 6. Output Formats
- **Data files**: JSON format
- **Graphs/plots**: PNG format (with --svg flag for (interactive where possible) SVG)
- **Console reports**: For all end results and debugging

#### 7. Code Quality
- **Document all code**: docstrings and comments
- **Optimize for performance**: Use multiprocessing/multithreading where it makes performance sense, parallel processing, vector where possible, use RAM/VRAM where possible/available
- **Memory efficiency**: Use memory-mapped files for large datasets
- **Error handling**: Robust exception handling and logging
- **Performance Monitoring**: Always time functions for debugging, add CPU/RAM GPU/VRAM DISK_IO/DISK_READ/DISK_WRITE utilization over time and as summary output

#### 8. Version Control and Revision Numbers

**CRITICAL**: All Python files and documentation MUST include version headers.

**Python File Header Format:**
```python
#!/usr/bin/env python3
"""
Module Name
Brief description

Version: X.Y.Z
Revision: N
Date: YYYY-MM-DD
Status: Production|Development|Deprecated
"""
```

**Version Numbering (Semantic Versioning):**
- **Major.Minor.Patch** (e.g., 3.0.0)
- **Major**: Breaking changes, major feature additions
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

**Revision Numbering:**
- **Revision**: Integer counter (1, 2, 3, ...)
- **ALWAYS increment** on ANY code change
- Reset to 1 when version number changes

**When to Update:**
- ✅ **ANY code modification**: Increment revision
- ✅ **Bug fixes**: Increment revision
- ✅ **Feature additions**: Increment revision AND version
- ✅ **Documentation changes in file**: Increment revision
- ❌ **External documentation only**: No revision change needed

**Examples:**
```python
# Initial release
Version: 3.0.0
Revision: 1

# Bug fix (same version, new revision)
Version: 3.0.0
Revision: 2

# New feature (new version, reset revision)
Version: 3.1.0
Revision: 1

# Breaking change (new major version, reset revision)
Version: 4.0.0
Revision: 1
```

**Markdown File Header Format (README.md, etc.):**
```markdown
**Version:** X.Y.Z
**Revision:** N
**Status:** ✅ PRODUCTION READY | 🚧 IN DEVELOPMENT
**Date:** YYYY-MM-DD HH:MM UTC
```

**Commit Message Format:**
- Include version/revision in commit messages for releases
- Example: `"feat: Add feature X (v3.1.0-r1)"`
- Example: `"fix: Bug fix Y (v3.0.0-r2)"`

#### 9. Professional Reporting Requirements

All analysis outputs and reports MUST include:

**Executive Summary**:
- High-level findings and recommendations
- Detection confidence levels
- Actionable next steps

**Publication-Quality Graphics**:
- Spectrograms with leak annotations and markers
- Correlation plots with confidence intervals and statistical bounds
- Time-series analysis with trend lines and anomaly markers
- Multi-sensor visualization (when applicable)
- Format: PNG (default) + SVG (with `--svg` flag for interactive/publication use)

**Engineering Validation Data**:
- Detection confidence scores (0.0 - 1.0)
- SNR (Signal-to-Noise Ratio) measurements in dB
- Cross-correlation coefficients with peak locations
- Time-delay estimates with uncertainty bounds
- Frequency domain analysis (peak frequencies, bandwidth)
- Sensor metadata (location, timestamp, gain settings, calibration data)

**Export Formats**:
- **JSON**: Raw data and metrics for further processing
- **PNG/SVG**: Graphics and visualizations
- **PDF**: Complete professional reports with graphics + data tables
- **CSV**: Tabular results for spreadsheet analysis

**Report Structure Template**:
```
1. Executive Summary
2. Sensor Configuration & Metadata
3. Signal Quality Metrics
4. Detection Results
   - Binary Classification (LEAK/NOLEAK)
   - Multi-Class Classification (if LEAK detected)
5. Correlation Analysis (for multi-sensor setups)
6. Visualizations
7. Engineering Validation Data
8. Recommendations & Next Steps
```

---

## Data Organization

### Classification Categories

**CRITICAL**: The system uses TWO separate AI models with different label sets:

#### Model 1: Binary Leak Detection
- **Purpose**: Fast initial screening for leak presence
- **Labels**: `BINARY_LABELS = ['LEAK', 'NOLEAK']`
- **Output**: Binary classification (leak detected / no leak detected)
- **Use Case**: First-stage screening of all incoming acoustic signals

#### Model 2: Multi-Class Classification
- **Purpose**: Detailed acoustic signature classification
- **Official Labels**: `DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']`
- **Output**: Specific acoustic event type with confidence score
- **Use Case**: Detailed analysis of suspected leak signals

**Typical Workflow**:
1. Binary model screens all signals for potential leaks
2. Signals classified as LEAK → Multi-class model for detailed classification
3. Multi-class results (BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED) provide actionable intelligence
4. Engineering reports generated with validation data

### Dataset Split
- **MASTER_DATASET**: 100% - Source of truth (manually labeled with 5-class labels)
- **DATASET_TRAINING**: 70% - Training data (for both binary and multi-class models)
- **DATASET_VALIDATION**: 20% - Validation data (for both models)
- **DATASET_TESTING**: 10% - Testing data (for both models)
- **DATASET_LEARNING**: Incremental learning data (pseudo-labeled + verified)
- **DATASET_DEV**: Development/testing only (not for production)

**Note**: Binary model training uses MASTER_DATASET with labels collapsed to LEAK/NOLEAK (LEAK remains LEAK, all others become NOLEAK).

### Data Flow
1. Raw sensor data → `DATA_SENSORS/`
2. Manual labeling (5-class) → `MASTER_DATASET/`
3. Initial split → `DATASET_TRAINING/`, `DATASET_VALIDATION/`, `DATASET_TESTING/`
4. Train both models:
   - Binary model: Uses collapsed labels (LEAK/NOLEAK)
   - Multi-class model: Uses full 5 categories
5. Inference: Binary screening → Multi-class classification (if LEAK detected)
6. High-confidence predictions → `DATASET_LEARNING/`
7. Manual verification → Back to `MASTER_DATASET/`
8. Reshuffle for next training round

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

The core development scripts (6 Python files):

- **`global_config.py`**: Global configuration (paths, hyperparameters, environment variables)
- **`ai_builder.py`**: Optimized GPU-accelerated processing pipeline (5-20x speedup)
- **`dataset_builder.py`**: Build and organize datasets from raw audio files
- **`dataset_classifier.py`**: Batch classification utilities for acoustic signals
- **`dataset_trainer.py`**: General dataset training utilities for CNN models
- **`dataset_tuner.py`**: Hyperparameter tuning using Keras Tuner or Optuna

### UTILITIES/ Scripts

Utility and testing scripts (17 Python files):

#### Testing & Validation
- **`test_gpu_cuda.py`**: Comprehensive GPU/CUDA diagnostics and benchmarking
- **`test_disk_tune.py`**: Disk I/O performance tuning (achieved: 24,801 files/s)
- **`test_gpu_monitoring.py`**: Real-time GPU resource monitoring
- **`test_wav_files.py`**: WAV file validation and integrity checking
- **`test_wav_files_v2.py`**: Enhanced WAV validation with detailed analysis

#### Data Processing
- **`normalize_wav_files.py`**: WAV file normalization and preprocessing
- **`shuffle_data_for_training.py`**: Dataset shuffling for training/validation splits
- **`dataconv.py`**: Data format conversion utilities
- **`dataframe.py`**: DataFrame manipulation and analysis tools
- **`process_acoustic_data.py`**: Acoustic signal processing utilities

#### Synthetic Data & Visualization
- **`synthetic_data.py`**: Generate synthetic acoustic signals for testing
- **`synthetic_leakgen.py`**: Generate synthetic leak signatures
- **`wav_viewer.py`**: Interactive WAV file viewer and analyzer
- **`leak_annotated_comparison_overlay.py`**: Overlay leak annotations on spectrograms
- **`leak_example.py`**: Example scripts for leak detection workflows

#### System Utilities
- **`gen_requirements.py`**: Automatically generate requirements.txt from imports
- **`prepmp3towav.py`**: Convert MP3 files to WAV format for processing

### FCS_TOOLS/ Scripts

Field Control System integration (2 Python files):

- **`datagate_client.py`**: Credential rotation client for secure data access
- **`datagate_sync.py`**: Synchronize sensor data from field control systems

---

## Development Workflows

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

---

### Multi-Sensor Correlation (CORRELATOR_v2)

The leak correlation system supports **multiple sensors**, not just sensor pairs:

#### Sensor Configurations

**2-Sensor Mode** (Standard Pair-wise):
- Cross-correlation between two hydrophones
- Time-delay estimation for leak localization
- Distance calculation along pipeline
- Minimum viable configuration

**Multi-Sensor Mode** (3+ sensors):
- Multiple sensors deployed in same pipeline area
- Enhanced triangulation and localization
- Spatial filtering for noise rejection
- Redundancy and validation through sensor agreement
- Improved confidence through consensus detection

#### Benefits of Multi-Sensor Deployments

- **Improved Accuracy**: Triangulation from multiple positions reduces localization uncertainty
- **Noise Rejection**: Spatially uncorrelated noise filtered through sensor fusion
- **Redundancy**: System continues operation if one sensor fails
- **Validation**: Cross-validation between sensor pairs confirms leak presence
- **Coverage**: Larger pipeline sections monitored with overlapping detection zones

#### Implementation Notes

- See `CORRELATOR_v2/multi_sensor_triangulation.py` for multi-sensor algorithms
- See `CORRELATOR_v2/sensor_registry.py` for sensor configuration management
- Professional reports include multi-sensor visualization when applicable

---

### Signal Stacking (Temporal Enhancement)

For **repeated measurements** from the same sensor pair over time (e.g., hourly 10-second recordings):

#### Purpose
Enhance Signal-to-Noise Ratio (SNR) by coherently averaging multiple recordings of the same leak signal.

#### Use Case
- **Scenario**: Days of data from same sensor pair (24 samples/day × N days)
- **Goal**: Strengthen weak leak signals buried in noise
- **Method**: Temporal stacking through coherent averaging

#### Implementation Method

```
1. Signal Alignment:
   - Perform cross-correlation between recordings
   - Align signals temporally to compensate for phase shifts
   - Account for clock drift and timing variations

2. Quality Control:
   - Reject outlier signals (e.g., >3σ from median SNR)
   - Verify signal stationarity (leak characteristics stable over stacking period)
   - Check correlation coefficients (reject poorly correlated signals)

3. Coherent Stacking:
   - Stack N aligned signals: S_stacked = (1/N) × Σ(S_i)
   - SNR improvement: √N factor (e.g., 10 signals → 3.16× SNR boost)
   - Preserve phase information for correlation analysis

4. Validation:
   - Compare stacked vs. individual signal correlations
   - Verify SNR improvement matches theoretical √N
   - Check for artifacts or distortion from stacking
```

#### Pros & Cons

**Advantages**:
- ✅ Stronger correlation peaks (easier leak detection)
- ✅ Better SNR (√N improvement with N signals)
- ✅ Reduced false positives (noise averaged out)
- ✅ More confident time-delay estimates

**Limitations**:
- ⚠️ **Assumes leak signal is stationary** over stacking period
- ⚠️ Non-stationary leaks (varying flow, intermittent) may degrade
- ⚠️ Time-varying environmental noise may not average out
- ⚠️ Requires consistent sensor positions and configurations

#### Recommended Practice

**Best Results**:
- Stack signals from **same time of day** across multiple days (e.g., 2:00 AM samples for 7 days)
- Minimizes daily environmental variations (traffic, industrial noise)
- Works well for steady-state leaks

**Quality Checks**:
- Verify signal stability before stacking (check correlation between individual signals)
- Monitor SNR improvement (should approach √N)
- Compare stacked correlation with individual best case

**Implementation**:
- See `CORRELATOR_v2/signal_stacking.py` for stacking algorithms
- Professional reports include stacking statistics and validation metrics

---

## AI Assistant Guidelines

### When Working on This Codebase

#### 1. Always Verify Configuration
- **DON'T** assume `global_config.py` or `global_vars` values are used everywhere
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

#### 5. Version Control Requirements
- **DO** add version headers to ALL new Python files
- **DO** increment revision number on EVERY code change
- **DO** update version number for new features or breaking changes
- **DO** include version info in file docstrings (see Development Guidelines section 8)
- **DO** maintain version/revision history in commit messages
- **DON'T** skip revision updates - even for minor changes

#### 6. Performance Considerations
- **DO** use CPU acceleration when possible
- **DO** use GPU acceleration when possible
- **DO** use RAM acceleration when possible
- **DO** use VRAM acceleration when possible
- **DO** use vector acceleration when possible
- **DO** use parallel processing
- **DO** use memory-mapped files for large datasets

#### 7. Testing Workflow
```bash
# Before committing changes:
1. Test with DATASET_DEV (not production data)
2. Run test_gpu_cuda.py to verify GPU functionality
3. Validate WAV files with test_wav_files.py
4. Check performance impact
5. Update revision number in file header
6. Document changes in commit message with version/revision
```

#### 8. Common Pitfalls to Avoid
- ❌ Hardcoding paths (use `global_config.py` constants)
- ❌ Ignoring file naming convention
- ❌ Modifying production datasets during testing
- ❌ Not handling OOM errors gracefully
- ❌ Forgetting `--verbose` and `--debug` flags
- ❌ Forgetting to increment revision numbers on code changes
- ❌ Omitting version headers in new files

#### 9. When Adding New Scripts
Create scripts that follow this template pattern:

```python
#!/usr/bin/env python3
"""
Script Description

Version: 1.0.0
Revision: 1
Date: YYYY-MM-DD
Status: Development

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

# Classify audio

# Incremental learning

```

---

## Additional Resources

### Documentation
- **DOCS/AILH.md**: Core requirements and specifications
- **DOCS/OPTIMIZATION_GUIDE.md**: GPU and performance optimization
- **DOCS/LeakDetectionTwoStageSegmentation.pdf**: Research paper
- **AI_DEV/global_config.py**: Configuration reference (paths, hyperparameters)
- **global_vars** (repository root): Runtime configuration (LOGGING, PERFMON, VERBOSE, DEBUG)

### External References
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- CuPy Documentation: https://docs.cupy.dev/

---
