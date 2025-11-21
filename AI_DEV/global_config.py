# Global Configuration for AILH_MASTER - Acoustic Leak Detection System
#
# HARDWARE ENVIRONMENT (see CLAUDE.md for full specs):
# - GPU: NVIDIA GeForce RTX 5090 Laptop (24GB VRAM, Compute Capability 12.0)
# - CUDA: 12.8, cuDNN: 9.x
# - OS: Windows 11 + WSL2 Ubuntu
# - Filesystem: ext4 optimized for small files (24,801.7 files/s)

import os

DEBUG = False

# PyTorch/CUDA Configuration (TensorFlow removed - project uses PyTorch exclusively)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
os.environ["TEMP_DIR"] = TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=True)




# ======================
# GLOBAL PARAMETERS
# ======================

SAMPLE_RATE = 4096
SAMPLE_UPSCALE = 8192
SAMPLE_DURATION = 10

CPU_COUNT = os.cpu_count() or 1

DRIVE_BUFFERSIZE = 131072
PREFETCH_THREADS = 6
FILES_PER_TASK = 740
RAM_QUEUE_SIZE = 128

DELIMITER = '~'
DATA_LABELS = ['BACKGROUND', 'CRACK', 'LEAK', 'NORMAL', 'UNCLASSIFIED']

LONG_WINDOW = 1024
SHORT_WINDOW = 512

N_FFT = SHORT_WINDOW # typically set equal to short segment length           
N_MELS = 64            
HOP_LENGTH = 128
N_POWER = 1.0

# CNN Training Hyperparameters (Legacy defaults for reference only)
# NOTE: Actual training uses Optuna-tuned values from PROC_MODELS/{binary,multiclass}/tuning/best_params.json
#       Production values: batch_size=32768, epochs=120, lr=0.0015, dropout=0.1, focal_alpha=0.25, focal_gamma=2.0        
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



