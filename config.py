# config.py

import os

# Credentials
#DATAGATE_USERNAME = "sbartal"
#DATAGATE_PASSWORD = "Sb749499houstonTX"
DATAGATE_USERNAME = "emartinez"
DATAGATE_PASSWORD = "letmein2Umeow!!!"

# Logging configuration
LOGS_DIR = "/DEVELOPMENT/ROOT_AILH/AILH_LOGS"
AILH_CACHE = "/DEVELOPMENT/ROOT_AILH/AILH_CACHE"
AILH_TMP = "/DEVELOPMENT/ROOT_AILH/AILH_TMP"
DATA_SENSORS = "/DEVELOPMENT/ROOT_AILH/DATA_SENSORS"
DATA_STORE = "/DEVELOPMENT/ROOT_AILH/DATA_STORE"

# Output log file
LOG_FILE = os.path.join(LOGS_DIR, "all.log")

# Output file for storing the last run timestamp
LAST_TIMESTAMP_FILE = os.path.join(LOGS_DIR, "last_run.json")

# Ensure the log directory exists at import time
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(AILH_CACHE, exist_ok=True)
os.makedirs(AILH_TMP, exist_ok=True)
os.makedirs(DATA_SENSORS, exist_ok=True)
os.makedirs(DATA_STORE, exist_ok=True)

##### Constants for the AMSYS AILHv2 Leak Detection System #####

LABELS = ['LEAK', 'NORMAL', 'RANDOM', 'MECHANICAL', 'UNCLASSIFIED']

DEFAULT_SEGMENT_LENGTHS = [0.125, 0.25, 0.5, 0.75, 1.00]  # seconds (long-term)
DEFAULT_SHORT_SEG_POINTS = [64, 128, 256, 512, 1024]      # short-term segment samples
DEFAULT_MEL_BINS = 64
DEFAULT_SAMPLING_RATE = 44100
DEFAULT_RANDOM_SEED = 42

DEFAULT_CNN_PARAMS = {
    'batch_size': 64,
    'epochs': 200,
    'learning_rate': 0.001,
    'dropout_rate': 0.25,
    'filters': 32,
    'kernel_size': (3, 3),
    'pooling_size': (2, 2),
    'strides': (2, 2)
}

FOLDER_STRUCTURE = {
    "SENSOR_DATA/": {
        "RAW_SIGNALS/": {},
        "LABELED_SEGMENTS/": {}
    },
    "AMSYS_AILH/": {
        "config.py": "",
        "requirements.txt": "",
        "leak_cnn_model.h5": "",
        "AILH_CLASSIFIER_V2/": {
            "leak_classifier.py": ""
        },
        "AILH_CLASSIFIER_TRAINER_V2/": {
            "leak_classifier_trainer.py": "",
            "REFERENCE_DATA/": {
                "TRAINING/": {label + "/" : {} for label in LABELS},
                "VALIDATION/": {label + "/" : {} for label in LABELS}
            },
            "UPDATE_DATA/": {
                "POSITIVE/": {label + "/" : {} for label in LABELS},
                "NEGATIVE/": {label + "/" : {} for label in LABELS}
            },
            "RESULTS/": {},
            "main.py": ""
        }
    }
}





