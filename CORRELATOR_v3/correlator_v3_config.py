#!/usr/bin/env python3
"""
CORRELATOR_V3 Configuration
Extends CORRELATOR_v2 with physics-aware, Bayesian, AI, and advanced correlation features.

Version: 3.0.0
Revision: 1
Date: 2025-11-25
Status: Production
"""

import os
import sys

# Import global AILH configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_DEV.global_config import *

# Import v2 config as base
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CORRELATOR_v2'))
from correlator_config import *

# ==============================================================================
# V3 GLOBAL VARIABLES SUPPORT
# ==============================================================================

LOGGING = os.getenv('LOGGING', 'YES') == 'YES'
PERFMON = os.getenv('PERFMON', 'NO') == 'YES'
VERBOSE = os.getenv('VERBOSE', 'NO') == 'YES'
DEBUG = os.getenv('DEBUG', 'NO') == 'YES'
UPSCALE = os.getenv('UPSCALE', 'NO') == 'YES'

SAMPLE_RATE_V3 = SAMPLE_UPSCALE if UPSCALE else SAMPLE_RATE

# ==============================================================================
# V3 PHYSICS-AWARE CORRELATION
# ==============================================================================

ENABLE_JOINT_POSITION_VELOCITY_SEARCH = os.getenv('V3_PHYSICS_AWARE', 'YES') == 'YES'
VELOCITY_SEARCH_MIN_MPS = 200
VELOCITY_SEARCH_MAX_MPS = 6000
VELOCITY_SEARCH_STEP_MPS = 50
POSITION_SEARCH_RESOLUTION_M = 0.5

ENABLE_DISPERSION_MODEL = os.getenv('V3_DISPERSION', 'YES') == 'YES'
DISPERSION_ALPHA_MIN = -0.5
DISPERSION_ALPHA_MAX = 0.5
DISPERSION_ALPHA_STEP = 0.05

# ==============================================================================
# V3 CORRELATION VARIANTS
# ==============================================================================

CORRELATION_METHODS_V3 = ['gcc_phat', 'gcc_roth', 'gcc_scot', 'classical', 'wavelet']

# Parse from environment or use defaults
env_methods = os.getenv('V3_CORRELATION_METHODS', 'gcc_phat,gcc_scot,wavelet')
ENABLED_CORRELATION_METHODS = [m.strip() for m in env_methods.split(',')]

ENABLE_GCC_PHAT = 'gcc_phat' in ENABLED_CORRELATION_METHODS
ENABLE_GCC_ROTH = 'gcc_roth' in ENABLED_CORRELATION_METHODS
ENABLE_GCC_SCOT = 'gcc_scot' in ENABLED_CORRELATION_METHODS
ENABLE_CLASSICAL = 'classical' in ENABLED_CORRELATION_METHODS
ENABLE_WAVELET = 'wavelet' in ENABLED_CORRELATION_METHODS

CORRELATION_FUSION_METHOD = 'weighted_average'  # 'weighted_average', 'best_snr', 'majority_vote'

# ==============================================================================
# V3 COHERENCE ANALYSIS
# ==============================================================================

ENABLE_COHERENCE_BAND_SELECTION = os.getenv('V3_COHERENCE', 'YES') == 'YES'
COHERENCE_THRESHOLD = 0.7
MIN_BAND_WIDTH_HZ = 50
MAX_COHERENCE_BANDS = 5
COHERENCE_WEIGHT_EXPONENT = 1.0

# ==============================================================================
# V3 BAYESIAN ESTIMATION
# ==============================================================================

ENABLE_BAYESIAN_ESTIMATION = os.getenv('V3_BAYESIAN', 'YES') == 'YES'
BAYESIAN_PRIOR_TYPE = 'uniform'  # 'uniform', 'gaussian', 'informed'
BAYESIAN_PRIOR_MEAN_M = None
BAYESIAN_PRIOR_STD_M = 50.0
BAYESIAN_LIKELIHOOD_BETA = 5.0
BAYESIAN_GRID_RESOLUTION_M = 0.1
BAYESIAN_CREDIBLE_INTERVAL = 0.95

# ==============================================================================
# V3 AI WINDOW GATING
# ==============================================================================

ENABLE_AI_WINDOW_GATING = os.getenv('V3_AI_GATING', 'YES') == 'YES'
AI_WINDOW_SIZE_SEC = 1.0
AI_LEAK_MODEL_PATH = os.path.join(PROC_MODELS, 'leak_classifier_latest.keras')
AI_LEAK_PROB_EXPONENT = 0.5  # γ
AI_SNR_EXPONENT = 0.3  # δ
AI_MIN_LEAK_PROBABILITY = 0.3

# ==============================================================================
# V3 ROBUST STACKING
# ==============================================================================

env_stacking = os.getenv('V3_STACKING_METHOD', 'huber')
ROBUST_STACKING_METHOD = env_stacking  # 'mean', 'trimmed_mean', 'median', 'huber'
TRIMMED_MEAN_PERCENTILE = 0.1
HUBER_DELTA = 1.35
PEAK_STABILITY_THRESHOLD = 0.7

# ==============================================================================
# V3 NOISE FILTER TOGGLES
# ==============================================================================

env_filters = os.getenv('V3_NOISE_FILTERS', 'dc_removal,hum_removal,highpass,bandpass')
enabled_filters = [f.strip() for f in env_filters.split(',')]

ENABLE_DC_OFFSET_REMOVAL = 'dc_removal' in enabled_filters
ENABLE_ELECTRICAL_HUM_REMOVAL = 'hum_removal' in enabled_filters
ENABLE_HIGHPASS_FILTER = 'highpass' in enabled_filters
ENABLE_BANDPASS_FILTER = 'bandpass' in enabled_filters
ENABLE_SPECTRAL_SUBTRACTION = 'spectral_subtraction' in enabled_filters
ENABLE_ADAPTIVE_FILTERING = 'adaptive' in enabled_filters
ENABLE_WAVELET_DENOISING = 'wavelet_denoise' in enabled_filters

ELECTRICAL_HUM_FREQUENCY_HZ = 60.0
ELECTRICAL_HUM_HARMONICS = 5
ELECTRICAL_HUM_Q_FACTOR = 30.0

# ==============================================================================
# V3 PERFORMANCE
# ==============================================================================

V3_GPU_BATCH_SIZE = 32
V3_CUDA_STREAMS = 32
V3_PRECISION = 'float32'
V3_PARALLEL_WORKERS = 4

V3_CACHE_DIR = os.path.join(CACHE_DIR, 'correlator_v3')
os.makedirs(V3_CACHE_DIR, exist_ok=True)

# ==============================================================================
# V3 OUTPUT
# ==============================================================================

V3_OUTPUT_DIR = os.path.join(ROOT_DIR, 'DATA_STORE', 'PROC_REPORTS', 'CORRELATOR_V3')
os.makedirs(V3_OUTPUT_DIR, exist_ok=True)

if VERBOSE or LOGGING:
    print("[i] CORRELATOR_V3 Configuration Loaded")
    print(f"    Physics-Aware: {ENABLE_JOINT_POSITION_VELOCITY_SEARCH}")
    print(f"    Bayesian: {ENABLE_BAYESIAN_ESTIMATION}")
    print(f"    AI Gating: {ENABLE_AI_WINDOW_GATING}")
    print(f"    Coherence: {ENABLE_COHERENCE_BAND_SELECTION}")
    print(f"    Correlation Methods: {ENABLED_CORRELATION_METHODS}")
    print(f"    Stacking: {ROBUST_STACKING_METHOD}")
    print(f"    Sample Rate: {SAMPLE_RATE_V3} Hz")
