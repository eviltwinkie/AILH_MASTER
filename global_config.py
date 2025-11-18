# Global Configuration for AILH_MASTER - Acoustic Leak Detection System
#
# HARDWARE ENVIRONMENT (see CLAUDE.md for full specs):
# - GPU: NVIDIA GeForce RTX 5090 Laptop (24GB VRAM, Compute Capability 12.0)
# - CUDA: 12.8, cuDNN: 9.x
# - OS: Windows 11 + WSL2 Ubuntu
# - Filesystem: ext4 optimized for small files (24,801.7 files/s)
#
# ⚠️ CRITICAL WARNING:
# Many scripts DO NOT use these parameters and have hardcoded values!
# - N_FFT: Scripts use 512 (not 256 as defined here)
# - HOP_LENGTH: Scripts use 128 (not 32 as defined here)
# - N_MELS: Scripts use 32-64 (varies, not always 64)
# See CLAUDE.md "Configuration Discrepancies" section for full details.

import os

CACHE_DIR = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_CACHE"
os.environ["CACHE_DIR"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

TEMP_DIR = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_TEMP"
os.environ["TEMP"] = TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=True)

KERAS_HOME = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_CACHE/KERAS"
os.environ["KERAS_HOME"] = KERAS_HOME
os.makedirs(KERAS_HOME, exist_ok=True)

TORCH_HOME = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_CACHE/TORCH"
os.environ["TORCH_HOME"] = TORCH_HOME
os.makedirs(TORCH_HOME, exist_ok=True)

XDG_CACHE_HOME = "/DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_CACHE/XDG"
os.environ["XDG_CACHE_HOME"] = XDG_CACHE_HOME
os.makedirs(XDG_CACHE_HOME, exist_ok=True)

os.environ["NUMPY_TEMP"] = TEMP_DIR

print("[AUDIT] Current Working Dir:", os.getcwd())

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU to avoid conflicts
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

# Less aggressive optimization settings to prevent cuDNN issues
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit=false --tf_xla_auto_jit=0"
os.environ["TF_DISABLE_MKL"] = "0"  # Re-enable MKL for stability

# cuDNN specific settings
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # Force deterministic cuDNN

# ======================
# GLOBAL PARAMETERS
# ======================

CPU_COUNT = os.cpu_count() or 1
#MAX_THREAD_WORKERS = max(1, min(CPU_COUNT - 2, CPU_COUNT * 3 // 4))
MAX_THREAD_WORKERS = 8

os.environ["OMP_NUM_THREADS"] = "4" #4
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4" #4

MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 512
SAFE_BATCH_SIZE = MIN_BATCH_SIZE
KERNEL_SIZE = (3, 3)

# ⚠️ WARNING: These parameters are often OVERRIDDEN by individual scripts!
HOP_LENGTH = 32        # ⚠️ Most scripts use 128, not this value!
N_MELS = 64            # ⚠️ Scripts vary: 32-64, not always this value!
N_FFT = 256            # ⚠️ Most scripts use 512, not this value!

LONG_TERM_SEC = 8.0
SHORT_TERM_SEC = 1.75
STRIDE_SEC = 1.7
DROPOUT = 0.15
N_FILTERS = 80
N_CONV_BLOCKS = 3
MAX_POOLINGS = 2

SAMPLE_RATE = 4096      # 4096 Hz REQUIRED TO UPSCALE TO 8192 Hz (see DOCS/AILH.md)
SAMPLE_DURATION = 10.0  # 10 seconds
INCREMENTAL_CONFIDENCE_THRESHOLD = 0.8  # Threshold for confidence in incremental learning
INCREMENTAL_ROUNDS = 2
LONG_SEGMENTS = [0.125, 0.25, 0.5, 0.75, 1.0]
SHORT_SEGMENTS = [64, 128, 256, 512, 1024]

CNN_BATCH_SIZE = 64
CNN_DROPOUT = 0.25
CNN_EPOCHS = 200
CNN_FILTERS = 32
CNN_POOL_SIZE = (2, 2)
CNN_STRIDES = (2, 2)    
CNN_LEARNING_RATE = 0.001
CNN_KERNEL_SIZE = (3, 3)
CNN_DENSE = 128

DELIMITER = '~'

# ⚠️ LABEL SET NOTE:
# This is the ACTIVE label set for the feature branch (5 categories)
# Ensure your MASTER_DATASET matches this label set before training!
# See CLAUDE.md "Classification Categories" for full details.
DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']

# ======================
# PATH CONFIGURATION
# ======================

# Official ROOT_AILH structure (from DOCS/AILH.md)
# ROOT_AILH is located at /DEVELOPMENT/ROOT_AILH/ in the actual filesystem
ROOT_AILH = "/DEVELOPMENT/ROOT_AILH"
DATA_STORE = os.path.join(ROOT_AILH, "DATA_STORE")
DATA_SENSORS = os.path.join(ROOT_AILH, "DATA_SENSORS")

# Dataset directories (derived from MASTER_DATASET)
MASTER_DATASET = os.path.join(DATA_STORE, "MASTER_DATASET")      # ⭐ Source of truth
DATASET_TRAINING = os.path.join(DATA_STORE, "DATASET_TRAINING")  # 70% of MASTER_DATASET
DATASET_VALIDATION = os.path.join(DATA_STORE, "DATASET_VALIDATION")  # 20% of MASTER_DATASET
DATASET_TESTING = os.path.join(DATA_STORE, "DATASET_TESTING")    # 10% of MASTER_DATASET
DATASET_LEARNING = os.path.join(DATA_STORE, "DATASET_LEARNING")  # Incremental learning data
DATASET_DEV = os.path.join(DATA_STORE, "DATASET_DEV")            # Development/testing only

# Processing directories
PROC_CACHE = os.path.join(DATA_STORE, "PROC_CACHE")      # Memmaps, temp files
PROC_LOGS = os.path.join(DATA_STORE, "PROC_LOGS")        # Logs
PROC_MODELS = os.path.join(DATA_STORE, "PROC_MODELS")    # Models
PROC_OUTPUT = os.path.join(DATA_STORE, "PROC_OUTPUT")    # Output files
PROC_REPORTS = os.path.join(DATA_STORE, "PROC_REPORTS")  # Classification reports

# Legacy paths (for backward compatibility with older scripts)
BASE_DIR = ".."
OUTPUT_DIR = "OUTPUT"

DEBUG = False


def set_base_dir(base_path):
    """Set the base directory for all data paths."""
    global BASE_DIR, SENSOR_DATA_DIR, RAW_SIGNALS_DIR, LABELED_SEGMENTS_DIR
    global REFERENCE_DIR, REFERENCE_TRAINING_DIR, REFERENCE_VALIDATION_DIR
    global UPDATE_DIR, UPDATE_POSITIVE_DIR, UPDATE_NEGATIVE_DIR
    
    BASE_DIR = base_path
    SENSOR_DATA_DIR = os.path.join(BASE_DIR, "SENSOR_DATA")
    RAW_SIGNALS_DIR = os.path.join(SENSOR_DATA_DIR, "RAW_SIGNALS")
    LABELED_SEGMENTS_DIR = os.path.join(SENSOR_DATA_DIR, "LABELED_SEGMENTS")
    REFERENCE_DIR = os.path.join(BASE_DIR, "REFERENCE_DATA")
    REFERENCE_TRAINING_DIR = os.path.join(REFERENCE_DIR, "TRAINING")
    REFERENCE_VALIDATION_DIR = os.path.join(REFERENCE_DIR, "VALIDATION")
    UPDATE_DIR = os.path.join(BASE_DIR, "UPDATE_DATA")
    UPDATE_POSITIVE_DIR = os.path.join(UPDATE_DIR, "POSITIVE")
    UPDATE_NEGATIVE_DIR = os.path.join(UPDATE_DIR, "NEGATIVE")

# Initialize with default paths
set_base_dir(BASE_DIR)

# try:
#     import pyfftw
#     pyfftw.interfaces.cache.enable()
#     print("[✓] pyFFTW cache enabled for FFT acceleration")
# except ImportError:
#     print("[i] pyFFTW not installed; FFTW acceleration skipped.")

