# CORRELATOR_V3 Implementation Plan

**Project:** AILH Acoustic Leak Detection System
**Version:** 3.0 (Advanced Physics-Aware Correlation with AI Integration)
**Date:** 2025-11-25
**Status:** ⏸️ AWAITING APPROVAL - DO NOT IMPLEMENT UNTIL GO_AHEAD

---

## Executive Summary

CORRELATOR_V3 will integrate advanced physics-aware correlation algorithms, multiple correlation variants, coherence-driven band selection, Bayesian position estimation, and AI-powered window gating into the existing CORRELATOR_v2 codebase. This upgrade transforms the system from a pure signal-processing correlator into a hybrid physics + ML correlation engine.

**Key Enhancements:**
1. ✅ Physics-Aware Correlation (joint position/velocity estimation, dispersion modeling)
2. ✅ Multiple Correlation Variants (GCC-PHAT, GCC-Roth, GCC-SCOT, classical, wavelet)
3. ✅ Coherence-Driven Band Selection (auto-detect leak-relevant frequencies)
4. ✅ Bayesian Position Estimation (MAP estimates, credible intervals, quality metrics)
5. ✅ AI Level 1 Integration (window gating using existing leak classifier)
6. ✅ Robust Stacking (weighted trimmed mean, Huber M-estimator)
7. ✅ Enhanced Configurability (all features enable/disable via config)
8. ✅ Global Configuration Consolidation (remove all old_config.py dependencies)

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Requirements Checklist](#2-requirements-checklist)
3. [Architecture Overview](#3-architecture-overview)
4. [Implementation Phases](#4-implementation-phases)
5. [Module Specifications](#5-module-specifications)
6. [Configuration Management](#6-configuration-management)
7. [Testing Strategy](#7-testing-strategy)
8. [Migration Path](#8-migration-path)
9. [Timeline & Deliverables](#9-timeline--deliverables)
10. [Risk Assessment](#10-risk-assessment)

---

## 1. Current State Analysis

### 1.1 CORRELATOR_v2 Capabilities (Current)

**Strengths:**
- ✅ Multi-leak detection (up to 10 leaks)
- ✅ GPU-accelerated batch processing (1000+ pairs/sec)
- ✅ GCC-PHAT correlation
- ✅ Single-sensor detection with 5 criteria
- ✅ Noise filtering (electrical hum, spectral subtraction, adaptive)
- ✅ Multi-sensor triangulation
- ✅ Professional reporting and visualization
- ✅ Environmental wave speed corrections (temperature, pressure)

**Modules (17 Python files, ~350KB):**
```
correlation_engine.py (22K)          # Core correlation
time_delay_estimator.py (20K)        # Time delay estimation
distance_calculator.py (18K)         # Distance calculations
multi_leak_detector.py (28K)         # Multi-leak detection
batch_gpu_correlator.py (18K)        # GPU batch processing
single_sensor_detector.py (15K)      # 5-criteria detection
noise_filters.py (18K)               # Noise suppression
signal_stacking.py (16K)             # Signal stacking
statistical_features.py (12K)        # Feature extraction
correlator_config.py (20K)           # Configuration
sensor_registry.py (24K)             # Sensor database
visualization.py (17K)               # Plotting
professional_report.py (37K)         # Report generation
+ 4 more utility modules
```

**Configuration:**
- ✅ `correlator_config.py` imports from `AI_DEV/global_config.py`
- ✅ No `old_config.py` dependencies found
- ✅ Environmental corrections already implemented

**Gaps for V3:**
- ❌ No joint (x,c) velocity search
- ❌ No dispersion-aware multi-band model
- ❌ Only GCC-PHAT (missing GCC-Roth, GCC-SCOT, wavelet)
- ❌ No coherence-driven band selection
- ❌ No Bayesian position estimation
- ❌ No AI window gating integration
- ❌ Limited robust stacking options
- ❌ No configurable filter enable/disable flags

### 1.2 AI_DEV Integration Points

**Available Resources:**
- ✅ `dataset_classifier.py` - Can classify 10s WAV files as LEAK/NORMAL/etc.
- ✅ `ai_builder.py` - GPU-accelerated mel spectrogram builder
- ✅ Trained models in `/DATA_STORE/PROC_MODELS/`
- ✅ Global configuration in `global_config.py`

**AI Level 1 Implementation:**
- Use existing leak classifier to generate p_leak(window) per 1-second window
- Weight correlation windows by leak probability
- Combine with SNR/coherence for adaptive weighting

---

## 2. Requirements Checklist

### 2.1 Mandatory Requirements ✅

- [x] **File naming:** `sensor_id~recording_id~timestamp~gain_db.wav`
- [x] **Sample rate:** 4096 Hz base, 8192 Hz upscale option
- [x] **Sample duration:** 10 seconds
- [x] **Delimiter:** `~` (tilde)
- [x] **Command-line flags:** `--verbose`, `--debug`, `--upscale`, `--svg`
- [x] **Global variables:** `LOGGING`, `PERFMON`, `VERBOSE`, `DEBUG`, `UPSCALE`
- [x] **Output formats:** JSON data, PNG/SVG plots
- [x] **Code quality:** Docstrings, comments, error handling, memory efficiency

### 2.2 V3 Enhancement Requirements ✅

#### Physics-Aware Correlation
- [ ] **Joint (x,c) search:** Search over (position, velocity) space instead of assuming c
- [ ] **Dispersion model:** `c_k = c_0 + α·f_k` (frequency-dependent velocity)
- [ ] **Multi-band velocity estimation:** Fit dispersion parameters from multi-band correlation

#### Multiple Correlation Variants
- [x] **GCC-PHAT** (already implemented)
- [ ] **GCC-Roth:** Roth weighting
- [ ] **GCC-SCOT:** Smoothed Coherence Transform
- [ ] **Classical cross-correlation:** Unweighted
- [ ] **Multi-resolution correlation:** Wavelet-based
- [ ] **Configurable selection:** Enable/disable each method via config

#### Coherence-Driven Band Selection
- [ ] **Magnitude-squared coherence:** `C_xy(f) = |E[X(f)Y*(f)]|² / (E[|X(f)|²]E[|Y(f)|²])`
- [ ] **Auto-detect leak bands:** Identify high-coherence frequency ranges
- [ ] **Coherence weighting:** Weight correlations by frequency-band coherence
- [ ] **Adaptive band selection:** Dynamically select bands per sensor pair

#### Bayesian Position Estimation
- [ ] **Prior definition:** `p(x)` - uniform or informed prior
- [ ] **Likelihood model:** `p(D|x) ∝ exp(β·s(x))` where s(x) is correlation score
- [ ] **Posterior computation:** `p(x|D) ∝ p(D|x)p(x)`
- [ ] **MAP estimate:** Maximum a posteriori position
- [ ] **Credible intervals:** 95% confidence bounds
- [ ] **Entropy-based quality:** H(p(x|D)) measures uncertainty

#### AI Integration (Level 1)
- [ ] **Window-level leak probability:** Run CNN on 1-second windows
- [ ] **Probability weighting:** `w(w) = p_leak(w)^γ · SNR(w)^δ`
- [ ] **Adaptive correlation:** `R(τ) = Σ w(w) R_w(τ) / Σ w(w)`
- [ ] **Integration with AI_DEV:** Import and use existing classifier

#### Robust Stacking
- [ ] **Weighted trimmed mean:** Discard outliers, weight by confidence
- [ ] **Median stacking:** Robust to outliers
- [ ] **Huber M-estimator:** Adaptive robust stacking
- [ ] **Peak stability metrics:** PSR, curvature, multi-band alignment

### 2.3 Configuration Requirements ✅

- [ ] **Consolidate to global_config.py:** All coefficients, environmental constants
- [ ] **Filter enable/disable flags:** Per-filter configuration
  - `ENABLE_ELECTRICAL_HUM_REMOVAL`
  - `ENABLE_SPECTRAL_SUBTRACTION`
  - `ENABLE_ADAPTIVE_FILTERING`
  - `ENABLE_HIGHPASS_FILTER`
  - `ENABLE_BANDPASS_FILTER`
- [ ] **Correlation method selection:** Runtime selection of variants
- [ ] **Physics-aware toggles:** Enable/disable joint search, dispersion
- [ ] **Bayesian estimation toggles:** Enable/disable Bayesian mode
- [ ] **AI integration toggles:** Enable/disable AI window gating
- [ ] **Remove old_config.py:** Already verified ✅

---

## 3. Architecture Overview

### 3.1 CORRELATOR_V3 Module Structure

```
CORRELATOR_v3/                        # New directory
├── correlator_v3_config.py           # V3 configuration (extends global_config)
├── physics_aware_correlator.py       # Joint (x,c) search, dispersion
├── correlation_variants.py           # GCC-Roth, GCC-SCOT, wavelet
├── coherence_analyzer.py             # Coherence computation, band selection
├── bayesian_estimator.py             # Bayesian position estimation
├── ai_window_gating.py               # AI Level 1 integration
├── robust_stacking.py                # Advanced stacking methods
├── leak_correlator_v3.py             # Main CLI (orchestrates all features)
├── batch_gpu_correlator_v3.py        # GPU batch with V3 features
└── examples/
    ├── example_physics_aware.py
    ├── example_bayesian.py
    └── example_ai_gating.py

# Enhanced existing modules (copy to v3, modify):
CORRELATOR_v3/
├── correlation_engine_v3.py          # Enhanced with new variants
├── noise_filters_v3.py               # Configurable enable/disable
├── multi_leak_detector_v3.py         # Integrated with Bayesian/AI
├── visualization_v3.py               # New plots (coherence, posterior)
└── professional_report_v3.py         # Enhanced reports
```

### 3.2 Data Flow (V3)

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: WAV Pair (sensor_id~recording_id~timestamp~gain.wav)   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Preprocessing (noise_filters_v3.py)                        │
│    - Configurable noise removal (flags)                        │
│    - DC removal, highpass, bandpass                            │
│    - Electrical hum removal (if enabled)                       │
│    - Spectral subtraction (if enabled)                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. AI Window Gating (ai_window_gating.py) [OPTIONAL]          │
│    - Segment signal into 1-second windows                      │
│    - Run leak classifier (AI_DEV) → p_leak(w)                 │
│    - Compute SNR(w), coherence(w)                              │
│    - Calculate weights: w(w) = p_leak^γ · SNR^δ               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Coherence Analysis (coherence_analyzer.py)                 │
│    - Compute magnitude-squared coherence C_xy(f)               │
│    - Auto-detect high-coherence bands                          │
│    - Select leak-relevant frequency bands                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Multi-Method Correlation (correlation_variants.py)         │
│    - Run multiple methods:                                      │
│      * GCC-PHAT (existing)                                     │
│      * GCC-Roth (new)                                          │
│      * GCC-SCOT (new)                                          │
│      * Classical (new)                                         │
│      * Wavelet (new)                                           │
│    - Per-band correlation (multi-band analysis)                │
│    - Coherence-weighted combination                            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Robust Stacking (robust_stacking.py)                       │
│    - Weighted trimmed mean                                      │
│    - Huber M-estimator                                         │
│    - Median stacking                                           │
│    - Combine correlation functions → R(τ)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Physics-Aware Estimation (physics_aware_correlator.py)     │
│    - Joint (x,c) search over position × velocity grid          │
│    - Dispersion-aware: Fit c_k = c_0 + α·f_k                  │
│    - Multi-band velocity estimation                            │
│    - Return: (x_opt, c_opt, α_opt)                            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Bayesian Position Estimation (bayesian_estimator.py)       │
│    - Define prior p(x) (uniform or informed)                   │
│    - Compute likelihood p(D|x) from correlation scores         │
│    - Calculate posterior p(x|D) ∝ p(D|x)p(x)                  │
│    - Extract:                                                   │
│      * MAP estimate (peak of posterior)                        │
│      * 95% credible interval                                   │
│      * Entropy H(p(x|D)) (quality metric)                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Multi-Leak Detection (multi_leak_detector_v3.py)           │
│    - Peak detection with Bayesian refinement                   │
│    - Clustering to remove duplicates                           │
│    - Per-leak confidence scoring                               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: Enhanced Leak Report (JSON)                            │
│ {                                                               │
│   "sensor_pair": ["S001", "S002"],                            │
│   "leaks": [                                                    │
│     {                                                           │
│       "position_map_m": 23.5,  // MAP estimate                │
│       "position_credible_interval_m": [21.2, 25.8],           │
│       "velocity_estimated_mps": 1420,                          │
│       "dispersion_coefficient": 0.15,                          │
│       "confidence_bayesian": 0.92,                             │
│       "entropy_quality": 1.2,                                  │
│       "ai_leak_probability": 0.88,                             │
│       "coherence_bands": [[100, 400], [800, 1200]],           │
│       "correlation_methods_used": ["GCC-PHAT", "GCC-SCOT"]    │
│     }                                                           │
│   ],                                                            │
│   "processing_metadata": { ... }                               │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Phases

### Phase 1: Foundation & Configuration (Week 1)
**Goal:** Set up V3 structure, consolidate configuration, enhance existing modules

**Tasks:**
1. Create `CORRELATOR_v3/` directory structure
2. Implement `correlator_v3_config.py` with all V3 parameters
3. Add filter enable/disable flags to global_config.py
4. Copy and enhance `noise_filters_v3.py` with configurable flags
5. Update all modules to support global variables (LOGGING, PERFMON, VERBOSE, DEBUG, UPSCALE)
6. Verify no old_config.py dependencies remain

**Deliverables:**
- [ ] `CORRELATOR_v3/` directory created
- [ ] `correlator_v3_config.py` complete
- [ ] Filter configuration flags implemented
- [ ] Global variable support in all modules
- [ ] Unit tests for configuration loading

**Success Criteria:**
- All filters can be enabled/disabled via config
- All coefficients centralized in global_config.py
- Clean import from AI_DEV/global_config.py

---

### Phase 2: Correlation Variants (Week 2)
**Goal:** Implement GCC-Roth, GCC-SCOT, classical, wavelet correlation

**Tasks:**
1. Implement `correlation_variants.py` module
   - `gcc_roth()` - Roth weighting
   - `gcc_scot()` - Smoothed Coherence Transform
   - `classical_correlation()` - Unweighted
   - `wavelet_correlation()` - Multi-resolution
2. Enhance `correlation_engine_v3.py`
   - Add method selection via config
   - Support multi-method ensemble
3. GPU acceleration for new methods (CuPy/PyTorch)
4. Benchmarking and validation

**Deliverables:**
- [ ] `correlation_variants.py` module (500+ lines)
- [ ] GCC-Roth implementation
- [ ] GCC-SCOT implementation
- [ ] Classical correlation
- [ ] Wavelet correlation
- [ ] GPU-accelerated versions
- [ ] Unit tests for each method
- [ ] Performance benchmarks

**Success Criteria:**
- All 5 correlation methods functional
- GPU acceleration working
- Method selection via config
- Performance comparable to v2 (within 20%)

---

### Phase 3: Coherence Analysis (Week 3)
**Goal:** Implement coherence-driven band selection

**Tasks:**
1. Implement `coherence_analyzer.py` module
   - Magnitude-squared coherence computation
   - Auto-detect high-coherence bands
   - Adaptive band selection algorithm
   - Coherence-weighted correlation
2. Integration with correlation_variants
3. Visualization of coherence spectra
4. Validation with synthetic and real data

**Deliverables:**
- [ ] `coherence_analyzer.py` module (400+ lines)
- [ ] Coherence computation (CPU + GPU)
- [ ] Band selection algorithm
- [ ] Coherence weighting integration
- [ ] Visualization functions
- [ ] Unit tests
- [ ] Validation report

**Success Criteria:**
- Coherence accurately computed
- Auto-detects leak-relevant bands (100-1500 Hz typical)
- Improves SNR by 3-5 dB in noisy conditions
- GPU-accelerated coherence computation

---

### Phase 4: Physics-Aware Correlation (Week 4)
**Goal:** Implement joint (x,c) search and dispersion modeling

**Tasks:**
1. Implement `physics_aware_correlator.py` module
   - Joint (x, c) grid search
   - Dispersion model: c_k = c_0 + α·f_k
   - Multi-band velocity fitting
   - Optimization algorithms (scipy.optimize)
2. Integration with existing distance_calculator
3. Performance optimization (vectorization, GPU)
4. Validation with known pipe properties

**Deliverables:**
- [ ] `physics_aware_correlator.py` module (600+ lines)
- [ ] Joint (x,c) search implementation
- [ ] Dispersion modeling
- [ ] Multi-band velocity estimation
- [ ] GPU-accelerated search
- [ ] Unit tests
- [ ] Validation report

**Success Criteria:**
- Joint search finds correct (x, c) within 2%
- Dispersion parameter α estimated accurately
- Handles unknown/variable wave speed
- Computation time < 500ms per pair (GPU)

---

### Phase 5: Bayesian Estimation (Week 5)
**Goal:** Implement Bayesian position estimation with uncertainty quantification

**Tasks:**
1. Implement `bayesian_estimator.py` module
   - Prior definition (uniform, Gaussian, informed)
   - Likelihood model from correlation scores
   - Posterior computation (grid-based or MCMC)
   - MAP estimate extraction
   - Credible interval calculation
   - Entropy-based quality metric
2. Integration with physics-aware correlator
3. Visualization of posteriors
4. Validation with ground truth data

**Deliverables:**
- [ ] `bayesian_estimator.py` module (500+ lines)
- [ ] Prior/likelihood/posterior implementations
- [ ] MAP estimator
- [ ] Credible interval calculator
- [ ] Entropy quality metric
- [ ] Visualization functions
- [ ] Unit tests
- [ ] Validation report

**Success Criteria:**
- Posterior distributions realistic
- MAP estimates accurate (within 5% of ground truth)
- Credible intervals calibrated (95% coverage)
- Entropy correlates with estimation quality

---

### Phase 6: AI Window Gating (Week 6)
**Goal:** Integrate AI_DEV leak classifier for adaptive window weighting

**Tasks:**
1. Implement `ai_window_gating.py` module
   - Interface with AI_DEV/dataset_classifier.py
   - 1-second window segmentation
   - Per-window leak probability p_leak(w)
   - Weight calculation: w(w) = p_leak^γ · SNR^δ
   - Weighted correlation: R(τ) = Σ w(w) R_w(τ)
2. Load trained models from PROC_MODELS
3. GPU batch inference for windows
4. Integration with correlation pipeline
5. Validation: compare gated vs. non-gated

**Deliverables:**
- [ ] `ai_window_gating.py` module (400+ lines)
- [ ] AI_DEV integration
- [ ] Window segmentation
- [ ] Probability-based weighting
- [ ] GPU batch inference
- [ ] Unit tests
- [ ] A/B validation report

**Success Criteria:**
- Successfully loads and runs leak classifier
- p_leak(w) computed for all windows
- Improves detection confidence by 10-15%
- Reduces false positives in noisy segments

---

### Phase 7: Robust Stacking (Week 7)
**Goal:** Implement advanced robust stacking methods

**Tasks:**
1. Implement `robust_stacking.py` module
   - Weighted trimmed mean (discard outliers)
   - Median stacking
   - Huber M-estimator
   - Peak stability metrics (PSR, curvature)
   - Multi-band peak alignment
2. Integration with correlation pipeline
3. Comparison with existing mean stacking
4. Validation with synthetic outliers

**Deliverables:**
- [ ] `robust_stacking.py` module (400+ lines)
- [ ] Weighted trimmed mean
- [ ] Median stacking
- [ ] Huber M-estimator
- [ ] Peak stability metrics
- [ ] Unit tests
- [ ] Comparative benchmarks

**Success Criteria:**
- Robust to 20% outlier windows
- Improves peak sharpness by 15-20%
- Computational overhead < 10%
- Integrates seamlessly with existing stacking

---

### Phase 8: Multi-Leak Enhancement (Week 8)
**Goal:** Enhance multi-leak detector with V3 features

**Tasks:**
1. Update `multi_leak_detector_v3.py`
   - Integrate Bayesian per-leak estimates
   - Use coherence for leak separation
   - Apply AI gating for leak validation
   - Enhanced clustering with physics constraints
2. Validation with multi-leak synthetic data
3. Real-world testing

**Deliverables:**
- [ ] `multi_leak_detector_v3.py` enhanced
- [ ] Bayesian multi-leak detection
- [ ] Coherence-based separation
- [ ] AI-validated detections
- [ ] Unit tests
- [ ] Validation report

**Success Criteria:**
- Detects up to 10 leaks (maintained from v2)
- Improved separation of closely-spaced leaks (< 2m apart)
- Bayesian confidence per leak
- False detection rate < 5%

---

### Phase 9: CLI and Batch Processing (Week 9)
**Goal:** Create V3 CLI and batch GPU processor

**Tasks:**
1. Implement `leak_correlator_v3.py` (main CLI)
   - All V3 features accessible via command-line
   - Feature toggles (--physics-aware, --bayesian, --ai-gating, etc.)
   - Backward compatible with v2 arguments
2. Implement `batch_gpu_correlator_v3.py`
   - GPU batch processing with V3 features
   - Maintain 1000+ pairs/sec throughput
3. Enhanced help documentation
4. Configuration file support (JSON/YAML)

**Deliverables:**
- [ ] `leak_correlator_v3.py` CLI (800+ lines)
- [ ] `batch_gpu_correlator_v3.py` batch processor
- [ ] Command-line documentation
- [ ] Configuration file templates
- [ ] User guide updates

**Success Criteria:**
- All V3 features accessible via CLI
- Backward compatible with v2
- Batch processing maintains performance
- Clear documentation and examples

---

### Phase 10: Visualization & Reporting (Week 10)
**Goal:** Enhanced visualizations and professional reports for V3

**Tasks:**
1. Update `visualization_v3.py`
   - Coherence spectra plots
   - Posterior distribution plots
   - Multi-method correlation comparison
   - Physics-aware (x,c) search heatmaps
   - AI window gating weight plots
2. Update `professional_report_v3.py`
   - Include all V3 metrics
   - Bayesian uncertainty reporting
   - AI confidence scores
   - Multi-method comparison tables
3. Export to PDF, HTML, JSON

**Deliverables:**
- [ ] `visualization_v3.py` enhanced (800+ lines)
- [ ] `professional_report_v3.py` enhanced (1000+ lines)
- [ ] New plot types (5+)
- [ ] Enhanced report templates
- [ ] Example reports

**Success Criteria:**
- All V3 features visualized
- Professional-quality reports
- Export to multiple formats
- Clear, interpretable plots

---

### Phase 11: Testing & Validation (Week 11)
**Goal:** Comprehensive testing and validation

**Tasks:**
1. Unit tests for all new modules (pytest)
2. Integration tests (full pipeline)
3. Synthetic data validation
   - Known leak positions
   - Known wave speeds
   - Known dispersion
   - Multi-leak scenarios
4. Real-world data validation
   - Field recordings
   - Ground truth verification
5. Performance benchmarking
6. Stress testing (edge cases)

**Deliverables:**
- [ ] Unit test suite (90%+ coverage)
- [ ] Integration test suite
- [ ] Synthetic validation report
- [ ] Real-world validation report
- [ ] Performance benchmark report
- [ ] Stress test report

**Success Criteria:**
- All tests passing
- Validation accuracy > 90% (synthetic)
- Real-world performance meets v2 or better
- No memory leaks or crashes
- GPU acceleration verified

---

### Phase 12: Documentation & Deployment (Week 12)
**Goal:** Complete documentation and production deployment

**Tasks:**
1. Update README.md
2. Create CORRELATOR_V3_USER_GUIDE.md
3. API documentation (Sphinx)
4. Migration guide (v2 → v3)
5. Configuration reference
6. Example scripts and tutorials
7. Performance tuning guide
8. Deployment checklist

**Deliverables:**
- [ ] Updated README.md
- [ ] V3 User Guide (50+ pages)
- [ ] API documentation (HTML)
- [ ] Migration guide
- [ ] Configuration reference
- [ ] 10+ example scripts
- [ ] Performance tuning guide
- [ ] Deployment checklist

**Success Criteria:**
- Complete, clear documentation
- Easy migration path from v2
- Examples cover all features
- Production-ready deployment

---

## 5. Module Specifications

### 5.1 `correlator_v3_config.py`

**Purpose:** Central configuration for all V3 features

**Key Sections:**
```python
# ==============================================================================
# V3 PHYSICS-AWARE CORRELATION
# ==============================================================================

# Enable/disable joint (x,c) search
ENABLE_JOINT_POSITION_VELOCITY_SEARCH = True

# Velocity search range (m/s)
VELOCITY_SEARCH_MIN_MPS = 200
VELOCITY_SEARCH_MAX_MPS = 6000
VELOCITY_SEARCH_STEP_MPS = 50

# Enable/disable dispersion modeling
ENABLE_DISPERSION_MODEL = True

# Dispersion coefficient bounds
DISPERSION_ALPHA_MIN = -0.5
DISPERSION_ALPHA_MAX = 0.5

# ==============================================================================
# V3 CORRELATION VARIANTS
# ==============================================================================

# Available correlation methods
CORRELATION_METHODS_V3 = [
    'gcc_phat',      # Generalized Cross-Correlation with Phase Transform
    'gcc_roth',      # Roth weighting
    'gcc_scot',      # Smoothed Coherence Transform
    'classical',     # Unweighted cross-correlation
    'wavelet',       # Multi-resolution wavelet correlation
]

# Enable/disable each method
ENABLE_GCC_PHAT = True
ENABLE_GCC_ROTH = True
ENABLE_GCC_SCOT = True
ENABLE_CLASSICAL = False  # Usually not needed if GCC methods enabled
ENABLE_WAVELET = True

# Method fusion strategy
CORRELATION_FUSION_METHOD = 'weighted_average'  # 'weighted_average', 'best_snr', 'all'

# ==============================================================================
# V3 COHERENCE ANALYSIS
# ==============================================================================

# Enable coherence-driven band selection
ENABLE_COHERENCE_BAND_SELECTION = True

# Coherence threshold for band selection
COHERENCE_THRESHOLD = 0.7

# Minimum band width (Hz)
MIN_BAND_WIDTH_HZ = 50

# Maximum number of bands
MAX_COHERENCE_BANDS = 5

# Coherence weighting exponent
COHERENCE_WEIGHT_EXPONENT = 1.0

# ==============================================================================
# V3 BAYESIAN ESTIMATION
# ==============================================================================

# Enable Bayesian position estimation
ENABLE_BAYESIAN_ESTIMATION = True

# Prior type
BAYESIAN_PRIOR_TYPE = 'uniform'  # 'uniform', 'gaussian', 'informed'

# Prior parameters (for Gaussian prior)
BAYESIAN_PRIOR_MEAN_M = None  # None = pipe midpoint
BAYESIAN_PRIOR_STD_M = 50.0   # Standard deviation

# Likelihood temperature (β parameter)
BAYESIAN_LIKELIHOOD_BETA = 5.0

# Grid resolution for posterior (meters)
BAYESIAN_GRID_RESOLUTION_M = 0.1

# Credible interval level
BAYESIAN_CREDIBLE_INTERVAL = 0.95

# ==============================================================================
# V3 AI WINDOW GATING
# ==============================================================================

# Enable AI-powered window gating
ENABLE_AI_WINDOW_GATING = True

# Window size for AI classification (seconds)
AI_WINDOW_SIZE_SEC = 1.0

# AI model path (relative to PROC_MODELS)
AI_LEAK_MODEL_PATH = 'leak_classifier_latest.keras'

# Window weighting parameters
AI_LEAK_PROB_EXPONENT = 0.5  # γ parameter
AI_SNR_EXPONENT = 0.3          # δ parameter

# Minimum leak probability threshold
AI_MIN_LEAK_PROBABILITY = 0.3

# ==============================================================================
# V3 ROBUST STACKING
# ==============================================================================

# Stacking method
ROBUST_STACKING_METHOD = 'huber'  # 'mean', 'trimmed_mean', 'median', 'huber'

# Trimmed mean parameters
TRIMMED_MEAN_PERCENTILE = 0.1  # Trim 10% from each end

# Huber M-estimator parameter
HUBER_DELTA = 1.35

# Peak stability threshold
PEAK_STABILITY_THRESHOLD = 0.7

# ==============================================================================
# V3 NOISE FILTER TOGGLES
# ==============================================================================

# Enable/disable individual noise filters
ENABLE_DC_OFFSET_REMOVAL = True
ENABLE_ELECTRICAL_HUM_REMOVAL = True
ENABLE_HIGHPASS_FILTER = True
ENABLE_BANDPASS_FILTER = True
ENABLE_SPECTRAL_SUBTRACTION = False  # Computationally expensive
ENABLE_ADAPTIVE_FILTERING = False    # Computationally expensive
ENABLE_WAVELET_DENOISING = False     # Experimental

# Electrical hum parameters
ELECTRICAL_HUM_FREQUENCY_HZ = 60.0  # 60 Hz (US) or 50 Hz (EU/Asia)
ELECTRICAL_HUM_HARMONICS = 5
ELECTRICAL_HUM_Q_FACTOR = 30.0

# ==============================================================================
# V3 GLOBAL VARIABLES SUPPORT
# ==============================================================================

# Environment variables (read from global_vars or command-line)
import os

LOGGING = os.getenv('LOGGING', 'YES') == 'YES'
PERFMON = os.getenv('PERFMON', 'NO') == 'YES'
VERBOSE = os.getenv('VERBOSE', 'NO') == 'YES'
DEBUG = os.getenv('DEBUG', 'NO') == 'YES'
UPSCALE = os.getenv('UPSCALE', 'NO') == 'YES'

# Sample rates (upscale support)
SAMPLE_RATE_BASE = 4096
SAMPLE_RATE_UPSCALE = 8192
SAMPLE_RATE = SAMPLE_RATE_UPSCALE if UPSCALE else SAMPLE_RATE_BASE

# ==============================================================================
# V3 PERFORMANCE TUNING
# ==============================================================================

# GPU batch size for V3 features
V3_GPU_BATCH_SIZE = 32

# Number of CUDA streams
V3_CUDA_STREAMS = 32

# Precision
V3_PRECISION = 'float32'  # 'float32' or 'float16'

# Parallel workers for multi-method correlation
V3_PARALLEL_WORKERS = 4

# Cache directory for V3 intermediate results
V3_CACHE_DIR = os.path.join(CACHE_DIR, 'correlator_v3')
os.makedirs(V3_CACHE_DIR, exist_ok=True)
```

**Lines of Code:** ~500

---

### 5.2 `physics_aware_correlator.py`

**Purpose:** Joint (x,c) search and dispersion modeling

**Key Classes:**
```python
class PhysicsAwareCorrelator:
    """
    Physics-aware correlation with joint position/velocity estimation.

    Features:
    - Joint (x, c) grid search
    - Dispersion-aware multi-band modeling
    - Frequency-dependent velocity: c_k = c_0 + α·f_k
    """

    def __init__(self, sample_rate=4096, verbose=False):
        pass

    def joint_search(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sensor_separation_m: float,
        velocity_range: Tuple[float, float] = (200, 6000),
        position_resolution_m: float = 0.5,
        velocity_resolution_mps: float = 50
    ) -> JointSearchResult:
        """
        Search over (x, c) space jointly.

        Returns:
            JointSearchResult with:
            - optimal_position_m
            - optimal_velocity_mps
            - confidence_score
            - search_heatmap (for visualization)
        """
        pass

    def dispersion_aware_search(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sensor_separation_m: float,
        frequency_bands: List[Tuple[float, float]]
    ) -> DispersionResult:
        """
        Fit dispersion model: c_k = c_0 + α·f_k

        Returns:
            DispersionResult with:
            - base_velocity_mps (c_0)
            - dispersion_coefficient (α)
            - per_band_velocities
            - fit_quality
        """
        pass

    def multi_band_velocity_estimation(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sensor_separation_m: float,
        n_bands: int = 5
    ) -> MultiBandVelocityResult:
        """
        Estimate velocity in multiple frequency bands.
        """
        pass
```

**Lines of Code:** ~600

---

### 5.3 `correlation_variants.py`

**Purpose:** Multiple correlation method implementations

**Key Functions:**
```python
def gcc_roth(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sample_rate: int = 4096,
    use_gpu: bool = True
) -> np.ndarray:
    """
    GCC with Roth weighting.

    Roth weighting: W(f) = 1 / |X(f)|²
    Good for colored noise.
    """
    pass

def gcc_scot(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sample_rate: int = 4096,
    use_gpu: bool = True
) -> np.ndarray:
    """
    GCC with Smoothed Coherence Transform.

    SCOT weighting: W(f) = 1 / sqrt(|X(f)|² · |Y(f)|²)
    Robust to reverberant environments.
    """
    pass

def classical_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Unweighted time-domain cross-correlation.
    """
    pass

def wavelet_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    wavelet: str = 'db4',
    levels: int = 5
) -> np.ndarray:
    """
    Multi-resolution wavelet-based correlation.

    Performs correlation at multiple wavelet decomposition levels.
    """
    pass

def fuse_correlations(
    correlations: Dict[str, np.ndarray],
    method: str = 'weighted_average',
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Fuse multiple correlation functions.

    Methods:
    - 'weighted_average': Weighted average of all methods
    - 'best_snr': Select method with highest SNR
    - 'majority_vote': Vote on peak location
    """
    pass
```

**Lines of Code:** ~500

---

### 5.4 `coherence_analyzer.py`

**Purpose:** Coherence computation and band selection

**Key Classes:**
```python
class CoherenceAnalyzer:
    """
    Compute coherence and auto-select leak-relevant frequency bands.
    """

    def __init__(self, sample_rate=4096, verbose=False):
        pass

    def compute_coherence(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        nperseg: int = 512
    ) -> CoherenceResult:
        """
        Compute magnitude-squared coherence.

        C_xy(f) = |E[X(f)Y*(f)]|² / (E[|X(f)|²]E[|Y(f)|²])

        Returns:
            CoherenceResult with:
            - frequencies
            - coherence (0-1 per frequency)
            - high_coherence_bands (auto-detected)
        """
        pass

    def auto_select_bands(
        self,
        coherence: np.ndarray,
        frequencies: np.ndarray,
        threshold: float = 0.7,
        min_width_hz: float = 50
    ) -> List[Tuple[float, float]]:
        """
        Auto-detect high-coherence frequency bands.

        Returns:
            List of (low_hz, high_hz) tuples for each band.
        """
        pass

    def coherence_weighted_correlation(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        bands: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Perform correlation weighted by coherence.

        Each band weighted by its mean coherence value.
        """
        pass
```

**Lines of Code:** ~400

---

### 5.5 `bayesian_estimator.py`

**Purpose:** Bayesian position estimation with uncertainty

**Key Classes:**
```python
class BayesianEstimator:
    """
    Bayesian position estimation with posterior inference.
    """

    def __init__(self, prior_type='uniform', verbose=False):
        pass

    def define_prior(
        self,
        sensor_separation_m: float,
        prior_type: str = 'uniform',
        mean_m: Optional[float] = None,
        std_m: float = 50.0
    ) -> np.ndarray:
        """
        Define prior p(x).

        Types:
        - 'uniform': Uniform over [0, L]
        - 'gaussian': Gaussian centered at mean_m
        - 'informed': Based on prior knowledge (e.g., near valve)
        """
        pass

    def compute_likelihood(
        self,
        correlation: np.ndarray,
        sensor_separation_m: float,
        wave_speed_mps: float,
        sample_rate: int = 4096,
        beta: float = 5.0
    ) -> np.ndarray:
        """
        Compute likelihood p(D|x) from correlation function.

        p(D|x) ∝ exp(β · s(x))

        where s(x) is correlation score at delay corresponding to position x.
        """
        pass

    def compute_posterior(
        self,
        prior: np.ndarray,
        likelihood: np.ndarray
    ) -> np.ndarray:
        """
        Compute posterior p(x|D) ∝ p(D|x)p(x).
        """
        pass

    def extract_estimates(
        self,
        posterior: np.ndarray,
        positions: np.ndarray,
        credible_level: float = 0.95
    ) -> BayesianEstimate:
        """
        Extract MAP estimate and credible interval.

        Returns:
            BayesianEstimate with:
            - map_position_m (Maximum A Posteriori)
            - credible_interval_m (95% by default)
            - entropy (quality metric)
            - posterior_distribution (for visualization)
        """
        pass

    def estimate(
        self,
        correlation: np.ndarray,
        sensor_separation_m: float,
        wave_speed_mps: float,
        sample_rate: int = 4096
    ) -> BayesianEstimate:
        """
        Complete Bayesian estimation pipeline.
        """
        pass
```

**Lines of Code:** ~500

---

### 5.6 `ai_window_gating.py`

**Purpose:** AI-powered window weighting

**Key Classes:**
```python
class AIWindowGating:
    """
    AI-powered window gating using leak classifier.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size_sec: float = 1.0,
        verbose: bool = False
    ):
        """
        Load leak classifier model from AI_DEV.
        """
        pass

    def segment_windows(
        self,
        signal: np.ndarray,
        sample_rate: int = 4096,
        window_size_sec: float = 1.0,
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Segment signal into windows for classification.
        """
        pass

    def classify_windows(
        self,
        windows: List[np.ndarray],
        use_gpu: bool = True
    ) -> np.ndarray:
        """
        Run leak classifier on each window.

        Returns:
            Array of leak probabilities p_leak(w) for each window.
        """
        pass

    def compute_weights(
        self,
        leak_probs: np.ndarray,
        snr_values: np.ndarray,
        leak_prob_exp: float = 0.5,
        snr_exp: float = 0.3
    ) -> np.ndarray:
        """
        Compute window weights.

        w(w) = p_leak(w)^γ · SNR(w)^δ
        """
        pass

    def weighted_correlation(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        weights_a: np.ndarray,
        weights_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted cross-correlation.

        R(τ) = Σ w(w) R_w(τ) / Σ w(w)
        """
        pass

    def process_pair(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        sample_rate: int = 4096
    ) -> AIGatingResult:
        """
        Complete AI window gating pipeline.

        Returns:
            AIGatingResult with:
            - weighted_correlation
            - per_window_leak_probs
            - per_window_weights
            - overall_leak_confidence
        """
        pass
```

**Lines of Code:** ~400

---

### 5.7 `robust_stacking.py`

**Purpose:** Advanced robust stacking methods

**Key Functions:**
```python
def weighted_trimmed_mean(
    correlations: np.ndarray,
    weights: Optional[np.ndarray] = None,
    trim_percentile: float = 0.1
) -> np.ndarray:
    """
    Trimmed mean with optional weighting.

    Discard top/bottom trim_percentile, then compute weighted mean.
    """
    pass

def median_stack(
    correlations: np.ndarray
) -> np.ndarray:
    """
    Median stacking (robust to outliers).
    """
    pass

def huber_stack(
    correlations: np.ndarray,
    delta: float = 1.35
) -> np.ndarray:
    """
    Huber M-estimator stacking.

    Adaptive: uses L2 for small residuals, L1 for large.
    """
    pass

def compute_peak_stability(
    correlation: np.ndarray,
    peak_idx: int,
    window_size: int = 20
) -> float:
    """
    Compute peak stability metric.

    Measures peak-to-sidelobe ratio and peak curvature.
    """
    pass

def multi_band_peak_alignment(
    correlations_per_band: List[np.ndarray],
    tolerance_samples: int = 5
) -> float:
    """
    Check if peaks align across frequency bands.

    Returns alignment score (0-1).
    """
    pass
```

**Lines of Code:** ~400

---

## 6. Configuration Management

### 6.1 Global Configuration Consolidation

**Objective:** Centralize ALL coefficients and constants in `AI_DEV/global_config.py`

**Changes to `AI_DEV/global_config.py`:**

```python
# Add V3-specific constants at end of file

# ==============================================================================
# CORRELATOR V3 CONSTANTS (imported by correlator_v3_config.py)
# ==============================================================================

# Physics-aware correlation
V3_VELOCITY_SEARCH_MIN_MPS = 200
V3_VELOCITY_SEARCH_MAX_MPS = 6000
V3_VELOCITY_SEARCH_STEP_MPS = 50
V3_DISPERSION_ALPHA_MIN = -0.5
V3_DISPERSION_ALPHA_MAX = 0.5

# Coherence analysis
V3_COHERENCE_THRESHOLD = 0.7
V3_MIN_BAND_WIDTH_HZ = 50
V3_MAX_COHERENCE_BANDS = 5
V3_COHERENCE_WEIGHT_EXPONENT = 1.0

# Bayesian estimation
V3_BAYESIAN_PRIOR_STD_M = 50.0
V3_BAYESIAN_LIKELIHOOD_BETA = 5.0
V3_BAYESIAN_GRID_RESOLUTION_M = 0.1
V3_BAYESIAN_CREDIBLE_INTERVAL = 0.95

# AI window gating
V3_AI_WINDOW_SIZE_SEC = 1.0
V3_AI_LEAK_PROB_EXPONENT = 0.5
V3_AI_SNR_EXPONENT = 0.3
V3_AI_MIN_LEAK_PROBABILITY = 0.3

# Robust stacking
V3_TRIMMED_MEAN_PERCENTILE = 0.1
V3_HUBER_DELTA = 1.35
V3_PEAK_STABILITY_THRESHOLD = 0.7

# Noise filter parameters
V3_ELECTRICAL_HUM_FREQUENCY_HZ = 60.0
V3_ELECTRICAL_HUM_HARMONICS = 5
V3_ELECTRICAL_HUM_Q_FACTOR = 30.0
```

### 6.2 Environment Variable Support

**Update `global_vars` file:**

```bash
# Existing
LOGGING=YES
PERFMON=NO
VERBOSE=NO
DEBUG=NO

# New for V3
UPSCALE=NO                          # Upscale audio to 8192 Hz

# V3 Feature Toggles (YES/NO)
V3_PHYSICS_AWARE=YES
V3_BAYESIAN=YES
V3_AI_GATING=YES
V3_COHERENCE=YES
V3_DISPERSION=YES

# V3 Correlation Methods (comma-separated)
V3_CORRELATION_METHODS=gcc_phat,gcc_scot,wavelet

# V3 Noise Filters (comma-separated)
V3_NOISE_FILTERS=dc_removal,hum_removal,highpass,bandpass

# V3 Stacking Method
V3_STACKING_METHOD=huber
```

### 6.3 Command-Line Interface

**All V3 scripts support:**

```bash
# Global flags (from CLAUDE.md requirements)
--verbose           # Print processing steps
--debug             # Print debug information
--upscale           # Upscale audio to 8192 Hz
--svg               # Output SVG instead of PNG

# V3 feature toggles
--physics-aware / --no-physics-aware
--bayesian / --no-bayesian
--ai-gating / --no-ai-gating
--coherence / --no-coherence
--dispersion / --no-dispersion

# V3 method selection
--correlation-methods gcc_phat,gcc_scot,wavelet
--stacking-method huber

# V3 noise filters
--enable-filters dc_removal,hum_removal,highpass
--disable-filters spectral_subtraction,adaptive

# V3 configuration file
--config config_v3.json
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Coverage Target:** 90%+

**Test Structure:**
```
CORRELATOR_v3/tests/
├── test_physics_aware_correlator.py
├── test_correlation_variants.py
├── test_coherence_analyzer.py
├── test_bayesian_estimator.py
├── test_ai_window_gating.py
├── test_robust_stacking.py
├── test_config_loading.py
└── test_integration.py
```

**Key Test Cases:**

1. **Physics-Aware Correlator:**
   - Known (x, c) → recover accurately
   - Dispersion: synthetic c(f) → fit α
   - Grid search: peak at correct location
   - Edge cases: x=0, x=L, unknown c

2. **Correlation Variants:**
   - Each method produces valid output
   - GPU vs. CPU equivalence
   - Performance benchmarks
   - Noise robustness

3. **Coherence Analyzer:**
   - Known coherence → compute correctly
   - Auto-detect bands → find leak frequencies
   - Weighting → improves SNR

4. **Bayesian Estimator:**
   - Prior × likelihood → posterior
   - MAP estimate → ground truth
   - Credible intervals → calibrated coverage
   - Entropy → correlates with quality

5. **AI Window Gating:**
   - Model loading → success
   - Window classification → probabilities
   - Weighting → improves detection

6. **Robust Stacking:**
   - Outlier rejection → robust mean
   - Median stacking → correct
   - Huber M-estimator → adaptive

### 7.2 Integration Tests

**Full Pipeline Tests:**

1. **End-to-end V3 pipeline:**
   - Input: WAV pair
   - Enable all V3 features
   - Output: Enhanced leak report
   - Verify: All V3 fields present

2. **Backward compatibility:**
   - Run v2 test cases with v3 code
   - Verify: Same results (within tolerance)

3. **Performance regression:**
   - V3 vs. V2 throughput
   - Target: V3 ≥ 80% of V2 speed

4. **GPU acceleration:**
   - V3 GPU vs. CPU
   - Verify: 5-10x speedup

### 7.3 Validation Tests

**Synthetic Data:**

```python
# Generate synthetic leak signals
def generate_synthetic_leak(
    sensor_separation_m=100,
    leak_position_m=30,
    wave_speed_mps=1400,
    dispersion_alpha=0.15,
    snr_db=20,
    duration_sec=10,
    sample_rate=4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sensor pair signals with known parameters.
    """
    pass
```

**Test Scenarios:**

1. **Single leak, known position:**
   - Generate: x=30m, c=1400 m/s
   - V3 estimate: x_est, c_est
   - Verify: |x_est - 30| < 1m, |c_est - 1400| < 50 m/s

2. **Dispersion present:**
   - Generate: α=0.15
   - V3 estimate: α_est
   - Verify: |α_est - 0.15| < 0.05

3. **Multi-leak:**
   - Generate: leaks at 20m, 50m, 80m
   - V3 detect: all 3 leaks
   - Verify: positions within 2m

4. **Noisy conditions:**
   - Generate: SNR = 5 dB
   - V3 detect: leak present
   - Verify: confidence > 0.6

5. **Unknown wave speed:**
   - Generate: c=unknown
   - V3 joint search: recover (x, c)
   - Verify: both accurate

**Real-World Validation:**

- Field recordings with ground truth
- Compare V3 estimates to verified leak locations
- Target accuracy: 90%+ within 5m

---

## 8. Migration Path (v2 → v3)

### 8.1 Backward Compatibility

**V3 Design Principles:**
- ✅ All v2 functionality preserved
- ✅ V3 features opt-in (disabled by default initially)
- ✅ Same API for basic operations
- ✅ Configuration backward compatible

**Compatibility Matrix:**

| Feature | V2 | V3 (default) | V3 (full) |
|---------|----|--------------|-----------|
| Basic correlation | ✅ | ✅ | ✅ |
| Multi-leak detection | ✅ | ✅ | ✅ (enhanced) |
| GPU acceleration | ✅ | ✅ | ✅ |
| Single-sensor | ✅ | ✅ | ✅ |
| Noise filtering | ✅ | ✅ | ✅ (configurable) |
| Physics-aware | ❌ | ❌ (opt-in) | ✅ |
| Bayesian | ❌ | ❌ (opt-in) | ✅ |
| AI gating | ❌ | ❌ (opt-in) | ✅ |
| Multi-method | GCC-PHAT only | GCC-PHAT | All 5 methods |

### 8.2 Migration Steps

**For Users:**

1. **Install V3:**
   ```bash
   cd CORRELATOR_v3
   # No installation needed (Python scripts)
   ```

2. **Test with v2 workflow:**
   ```bash
   # This should work exactly like v2
   python leak_correlator_v3.py \
       --sensor-a A.wav \
       --sensor-b B.wav \
       --registry registry.json \
       --output report.json
   ```

3. **Enable V3 features gradually:**
   ```bash
   # Add Bayesian estimation
   python leak_correlator_v3.py ... --bayesian

   # Add AI gating
   python leak_correlator_v3.py ... --bayesian --ai-gating

   # Full V3
   python leak_correlator_v3.py ... --bayesian --ai-gating --physics-aware --coherence
   ```

4. **Update configuration:**
   ```bash
   # Copy v2 config
   cp CORRELATOR_v2/correlator_config.py my_config.py

   # Add V3 settings
   # Edit my_config.py to enable V3 features
   ```

**For Developers:**

1. **Import v3 modules:**
   ```python
   # Old (v2)
   from CORRELATOR_v2.correlation_engine import CorrelationEngine

   # New (v3)
   from CORRELATOR_v3.correlation_engine_v3 import CorrelationEngineV3

   # V3 is backward compatible
   engine = CorrelationEngineV3(method='gcc_phat')  # Works like v2
   ```

2. **Use v3 features:**
   ```python
   # Enable physics-aware
   engine = CorrelationEngineV3(
       method='gcc_phat',
       physics_aware=True,
       bayesian=True
   )
   ```

### 8.3 Deprecation Plan

**V2 Support:**
- V2 code remains in `CORRELATOR_v2/` (read-only)
- V3 is separate in `CORRELATOR_v3/`
- No breaking changes to v2
- Users can run both side-by-side

**Timeline:**
- Months 1-3: V3 beta, v2 primary
- Months 4-6: V3 stable, v2 maintenance
- Months 7+: V3 primary, v2 archived

---

## 9. Timeline & Deliverables

### 9.1 Implementation Schedule (12 Weeks)

```
Week 1:  Foundation & Configuration
         └─ Deliverable: V3 structure, config system

Week 2:  Correlation Variants
         └─ Deliverable: 5 correlation methods working

Week 3:  Coherence Analysis
         └─ Deliverable: Coherence-driven band selection

Week 4:  Physics-Aware Correlation
         └─ Deliverable: Joint (x,c) search, dispersion

Week 5:  Bayesian Estimation
         └─ Deliverable: Posterior inference, MAP estimates

Week 6:  AI Window Gating
         └─ Deliverable: AI_DEV integration, weighting

Week 7:  Robust Stacking
         └─ Deliverable: Advanced stacking methods

Week 8:  Multi-Leak Enhancement
         └─ Deliverable: V3 multi-leak detector

Week 9:  CLI & Batch Processing
         └─ Deliverable: V3 CLI, batch GPU processor

Week 10: Visualization & Reporting
         └─ Deliverable: Enhanced plots, reports

Week 11: Testing & Validation
         └─ Deliverable: Test suite, validation reports

Week 12: Documentation & Deployment
         └─ Deliverable: User guide, production release
```

### 9.2 Milestones

**M1 (Week 4):** Core V3 algorithms implemented
- Physics-aware, correlation variants, coherence

**M2 (Week 6):** Advanced features complete
- Bayesian estimation, AI gating

**M3 (Week 9):** User-facing tools ready
- CLI, batch processor, backward compatible

**M4 (Week 12):** Production release
- Tested, documented, validated

### 9.3 Success Metrics

**Technical:**
- ✅ All V3 features implemented
- ✅ Unit test coverage > 90%
- ✅ Validation accuracy > 90% (synthetic)
- ✅ Real-world performance meets or exceeds v2
- ✅ GPU throughput ≥ 800 pairs/sec (vs. v2: 1000)

**Quality:**
- ✅ Improved position accuracy (±1m vs. v2: ±2m)
- ✅ Better uncertainty quantification (credible intervals)
- ✅ Reduced false positives (AI gating)
- ✅ Handles unknown wave speeds (joint search)

**Usability:**
- ✅ Backward compatible with v2
- ✅ Clear documentation
- ✅ Easy migration path
- ✅ Configurable features (enable/disable)

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU memory overflow with V3 features | Medium | High | Batch size tuning, FP16, profiling |
| AI model integration failures | Low | Medium | Extensive testing, fallback to v2 |
| Joint (x,c) search too slow | Medium | Medium | GPU acceleration, coarse-to-fine |
| Bayesian posterior computation expensive | Low | Low | Grid-based (not MCMC), GPU |
| Coherence band selection unstable | Low | Medium | Validation, parameter tuning |
| Multi-method fusion degrades performance | Low | Low | Configurable, can disable |

### 10.2 Project Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 12-week timeline too aggressive | Medium | Medium | Prioritize core features, defer Level 2-4 AI |
| Testing insufficient | Low | High | Allocate 2 weeks for testing, continuous testing |
| Documentation incomplete | Low | Medium | Parallel documentation during dev |
| V2 compatibility broken | Low | High | Extensive regression testing |
| User adoption slow | Medium | Low | Clear migration guide, backward compatible |

### 10.3 Mitigation Strategies

**Performance:**
- Benchmark continuously
- Profile GPU usage
- Optimize hotspots early

**Quality:**
- Write tests alongside code
- Code reviews
- Validation with synthetic and real data

**Schedule:**
- Weekly milestones
- Buffer time in week 11-12
- Prioritize must-have features

**User Adoption:**
- Beta program with early users
- Detailed migration guide
- Backward compatibility as priority

---

## 11. Open Questions & Decisions Needed

### 11.1 Implementation Decisions

**❓ Question 1: Bayesian posterior computation method?**
- Option A: Grid-based (fast, approximate)
- Option B: MCMC sampling (slow, exact)
- **Recommendation:** Grid-based for real-time, MCMC optional

**❓ Question 2: AI model format?**
- Option A: Use existing Keras models from AI_DEV
- Option B: Convert to ONNX for portability
- **Recommendation:** Use Keras (already available), ONNX future

**❓ Question 3: Wavelet family for correlation?**
- Option A: Daubechies (db4)
- Option B: Symlets (sym8)
- **Recommendation:** Daubechies (standard, well-tested)

**❓ Question 4: Default V3 mode?**
- Option A: V3 features enabled by default
- Option B: V3 features opt-in (backward compatible default)
- **Recommendation:** Opt-in initially, default later after validation

### 11.2 Configuration Decisions

**❓ Question 5: Configuration file format?**
- Option A: Python module (current v2 approach)
- Option B: JSON/YAML (more portable)
- **Recommendation:** Both - Python for defaults, JSON/YAML for overrides

**❓ Question 6: Feature toggle granularity?**
- Option A: Coarse (enable/disable V3 as whole)
- Option B: Fine (per-feature toggles)
- **Recommendation:** Fine-grained (as specified in requirements)

---

## 12. Appendices

### Appendix A: Reference Papers

1. **Generalized Cross-Correlation:**
   - Knapp & Carter (1976) - GCC-PHAT foundation
   - Brandstein & Silverman (1997) - Robust methods

2. **Acoustic Leak Detection:**
   - Fuchs & Riehle (1991) - Ten years of experience
   - Brennan et al. (2018) - Soil properties effects

3. **Bayesian Signal Processing:**
   - Kay (1993) - Fundamentals of Statistical Signal Processing
   - Gelman et al. (2013) - Bayesian Data Analysis

4. **Dispersion in Pipes:**
   - Fuller & Fahy (1982) - Characteristics of wave propagation
   - Long et al. (2003) - Acoustic wave propagation in buried pipes

### Appendix B: Glossary

| Term | Definition |
|------|------------|
| GCC-PHAT | Generalized Cross-Correlation with Phase Transform |
| GCC-Roth | Generalized Cross-Correlation with Roth weighting |
| GCC-SCOT | Generalized Cross-Correlation with Smoothed Coherence Transform |
| MAP | Maximum A Posteriori (Bayesian point estimate) |
| PSR | Peak-to-Sidelobe Ratio |
| Coherence | Magnitude-squared coherence, measures correlation in frequency domain |
| Dispersion | Frequency-dependent wave velocity |
| Huber M-estimator | Robust estimator using adaptive loss function |

### Appendix C: Contact & Support

**Project Team:**
- Lead Developer: [TBD]
- AI Integration: [TBD]
- Testing & QA: [TBD]

**Resources:**
- Repository: `eviltwinkie/AILH_MASTER`
- Branch: `claude/build-correlator-v3-01SUJVGAniBmqp1Ykh1qUEJM`
- Documentation: `CORRELATOR_v3/docs/`

---

## ⏸️ IMPLEMENTATION STATUS: AWAITING GO_AHEAD

**This is a comprehensive implementation plan. No implementation has started.**

**Next Steps:**
1. Review this plan
2. Discuss priorities and preferences
3. Answer open questions (Section 11)
4. Provide explicit **GO_AHEAD** to begin implementation

**Questions for discussion:**
1. ✅ Do you approve the overall architecture?
2. ✅ Are the 12-week timeline and phasing acceptable?
3. ✅ Which features should be prioritized if timeline is tight?
4. ✅ Any specific implementation concerns?
5. ✅ Ready to proceed with Phase 1?

---

**Plan Version:** 1.0
**Date:** 2025-11-25
**Status:** 📋 DRAFT - Awaiting Approval
**Next Review:** After stakeholder discussion

