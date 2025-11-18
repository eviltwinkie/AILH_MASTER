# Global Configuration for AILH_MASTER - Acoustic Leak Detection System
#
# HARDWARE ENVIRONMENT (see CLAUDE.md for full specs):
# - GPU: NVIDIA GeForce RTX 5090 Laptop (24GB VRAM, Compute Capability 12.0)
# - CUDA: 12.8, cuDNN: 9.x
# - OS: Windows 11 + WSL2 Ubuntu
# - Filesystem: ext4 optimized for small files (24,801.7 files/s)

import os

DEBUG = False

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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

#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["TF_NUM_INTEROP_THREADS"] = "2"
#os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

# ======================
# PATH CONFIGURATION
# ======================

ROOT_DIR = "/DEVELOPMENT/ROOT_AILH"
DATA_STORE = os.path.join(ROOT_DIR, "DATA_STORE")
DATA_SENSORS = os.path.join(ROOT_DIR, "DATA_SENSORS")

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

CACHE_DIR = os.path.join(DATA_STORE, "PROC_CACHE")
os.environ["CACHE_DIR"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

TEMP_DIR = os.path.join(DATA_STORE, "PROC_TEMP")
os.environ["TEMP"] = TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=True)

KERAS_CACHE = os.path.join(DATA_STORE, "PROC_CACHE", "KERAS")
os.environ["KERAS_CACHE"] = KERAS_CACHE
os.makedirs(KERAS_CACHE, exist_ok=True)

TORCH_CACHE = os.path.join(DATA_STORE, "PROC_CACHE", "TORCH")
os.environ["TORCH_CACHE"] = TORCH_CACHE
os.makedirs(TORCH_CACHE, exist_ok=True)

XDG_CACHE = os.path.join(DATA_STORE, "PROC_CACHE", "XDG")
os.environ["XDG_CACHE"] = XDG_CACHE
os.makedirs(XDG_CACHE, exist_ok=True)

NUMPY_CACHE = os.path.join(DATA_STORE, "PROC_CACHE", "NUMPY")
os.environ["NUMPY_CACHE"] = NUMPY_CACHE
os.makedirs(NUMPY_CACHE, exist_ok=True)



# ======================
# GLOBAL PARAMETERS
# ======================

SAMPLE_RATE = 4096
SAMPLE_UPSCALE = 8192
SAMPLE_LENGTH_SEC = 10

CPU_COUNT = os.cpu_count() or 1

DRIVE_BUFFERSIZE = 131072
PREFETCH_THREADS = 6
PREFETCH_DEPTH = 16
FILES_PER_TASK = 768

DELIMITER = '~'
DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']

LONG_SEGMENT_SCALE_SEC = 0.25
SHORT_SEGMENT_POINTS = 512

N_FFT = SHORT_SEGMENT_POINTS # typically set equal to short segment length           
N_MELS = 32            
HOP_LENGTH = 128        

CNN_BATCH_SIZE = 64
CNN_DROPOUT = 0.25
CNN_EPOCHS = 200
CNN_FILTERS = 32
CNN_POOL_SIZE = (2, 2)
CNN_STRIDES = (2, 2)    
CNN_LEARNING_RATE = 0.001
CNN_KERNEL_SIZE = (3, 3)
CNN_DENSE = 128

INCREMENTAL_CONFIDENCE_THRESHOLD = 0.8  # Threshold for confidence in incremental learning
INCREMENTAL_ROUNDS = 2



