# CORRELATOR_v2 - Quick Start Guide

## Overview

The Leak Detection Correlator calculates the distance and position of a leak between two hydrophone sensors using cross-correlation analysis.

## Installation

No installation required! The correlator uses the existing AILH environment.

```bash
cd /home/user/AILH_MASTER/CORRELATOR_v2
```

## Prerequisites

1. **Sensor Registry**: JSON file defining sensor positions and pairs
2. **WAV Files**: Synchronized recordings from two sensors
3. **File Naming**: Must follow AILH convention: `sensor_id~recording_id~timestamp~gain_db.wav`

## Quick Examples

### 1. Create Sensor Registry

```bash
# Create example registry
python sensor_registry.py --create-example --save my_registry.json

# Validate registry
python sensor_registry.py --load my_registry.json --validate --list-pairs
```

Edit `my_registry.json` to add your actual sensor positions and pipe configurations.

### 2. Correlate Two Sensors (Single Pair)

```bash
python leak_correlator.py \
    --sensor-a /path/to/S001~R123~20250118120530~100.wav \
    --sensor-b /path/to/S002~R456~20250118120530~100.wav \
    --registry my_registry.json \
    --output leak_report.json \
    --verbose
```

**Output**: JSON file with leak distance, GPS position, and confidence metrics

### 3. Batch Process Multiple Pairs

```bash
python leak_correlator.py \
    --batch-mode \
    --input-dir /DEVELOPMENT/ROOT_AILH/DATA_SENSORS/SITE_001 \
    --registry my_registry.json \
    --output-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/PROC_REPORTS/CORRELATIONS \
    --time-window 60 \
    --verbose
```

**Output**: Individual JSON reports + batch summary

## Understanding the Output

Example output (`leak_report.json`):

```json
{
  "sensor_pair": {
    "sensor_a": "S001",
    "sensor_b": "S002"
  },
  "time_delay_seconds": 0.0234,
  "distance_from_sensor_a_meters": 16.38,
  "distance_from_sensor_b_meters": 83.62,
  "sensor_separation_meters": 100.0,
  "leak_position_gps": {
    "latitude": 40.712912,
    "longitude": -74.005895,
    "elevation": 10.6
  },
  "confidence": 0.892,
  "wave_speed_mps": 1400,
  "pipe_material": "ductile_iron",
  "quality_metrics": {
    "time_delay_confidence": 0.892,
    "time_delay_snr_db": 18.5,
    "peak_sharpness": 3.2
  }
}
```

### Key Fields

- **time_delay_seconds**: Measured time difference between sensors (τ)
- **distance_from_sensor_a_meters**: Calculated distance from sensor A to leak
- **confidence**: Overall confidence score (0-1), higher is better
- **leak_position_gps**: Estimated GPS coordinates of leak
- **quality_metrics.time_delay_snr_db**: Signal-to-noise ratio (>10 dB recommended)

## Advanced Usage

### Specify Correlation Method

```bash
# Use GCC-PHAT (recommended for noisy environments)
python leak_correlator.py ... --method gcc_phat

# Use basic correlation (faster, less robust)
python leak_correlator.py ... --method basic

# Use frequency-domain (good for long signals)
python leak_correlator.py ... --method frequency_domain
```

### Enable GPU Acceleration

```bash
python leak_correlator.py ... --gpu
```

Requires CuPy installed. Provides 2-5x speedup for correlation.

### Override Wave Speed

```bash
# Specify custom wave speed
python leak_correlator.py ... --wave-speed 1400

# Or specify pipe material
python leak_correlator.py ... --pipe-material steel
```

Available materials: `ductile_iron` (1400 m/s), `steel` (5000 m/s), `pvc` (450 m/s), `cast_iron` (3500 m/s), etc.

## Troubleshooting

### "Sensor pair not found in registry"

**Solution**: Add the sensor pair to your registry JSON file.

```json
{
  "sensor_pairs": [
    {
      "sensor_a": "S001",
      "sensor_b": "S002",
      "distance_meters": 100.0,
      "pipe_segment": {
        "material": "ductile_iron",
        "diameter_mm": 300
      },
      "wave_speed_mps": 1400
    }
  ]
}
```

### "Time drift exceeds maximum"

**Solution**: Increase time window or check if recordings are actually synchronized.

```bash
python leak_correlator.py ... --time-window 120  # Allow 120s drift
```

### "Low confidence scores"

**Possible causes**:
- Background noise too high
- Leak signal weak
- Incorrect wave speed
- Sensors too far apart

**Solutions**:
- Try GCC-PHAT method: `--method gcc_phat`
- Check signal quality in WAV files
- Verify pipe material and wave speed
- Ensure sensors are on same pipe segment

### "Distance is negative or exceeds sensor separation"

**Cause**: Time delay estimate is incorrect, possibly due to:
- Wrong wave speed
- Reflections/echoes
- Poor signal quality

**Solutions**:
- Verify wave speed is correct for pipe material
- Check sensor pair configuration (distance, wave speed)
- Examine quality metrics (SNR, sharpness)

## Configuration

All parameters can be customized in `correlator_config.py`:

```python
# Bandpass filter (Hz)
BANDPASS_LOW_HZ = 100
BANDPASS_HIGH_HZ = 1500

# Quality thresholds
MIN_CONFIDENCE = 0.7
MIN_SNR_DB = 10.0

# Time synchronization
MAX_TIME_SYNC_DRIFT_SEC = 5.0

# GPU settings
USE_GPU_CORRELATION = True
CORRELATION_PRECISION = 'float32'
```

## Integration with AILH System

### After Leak Classification

```bash
# Step 1: Classify all sensors for leaks
python AI_DEV/leak_directory_classifier.py \
    --input-dir /DATA_SENSORS/SITE_001 \
    --model /DATA_STORE/PROC_MODELS/leak_model.keras \
    --output-report classifications.json

# Step 2: For files classified as LEAK, run correlator
python CORRELATOR_v2/leak_correlator.py \
    --batch-mode \
    --input-dir /DATA_SENSORS/SITE_001 \
    --registry sensor_registry.json \
    --output-dir /DATA_STORE/PROC_REPORTS/LEAK_LOCATIONS
```

## Module Reference

### `leak_correlator.py`
Main application - complete pipeline from WAV files to leak location

### `correlation_engine.py`
Cross-correlation algorithms (basic, GCC-PHAT, frequency-domain)

### `time_delay_estimator.py`
Peak detection and subsample interpolation for precise time delays

### `distance_calculator.py`
Distance calculation and GPS position interpolation

### `sensor_registry.py`
Sensor database management

### `correlator_utils.py`
WAV file loading, filename parsing, validation

### `correlator_config.py`
Configuration parameters and constants

## Next Steps

1. **Create your sensor registry** with actual GPS coordinates
2. **Test with a known leak** to validate wave speed
3. **Calibrate thresholds** based on your pipe network and sensors
4. **Integrate with AILH** classification pipeline

## Support

For detailed documentation, see:
- `README.md` - Full system documentation
- `examples/` - Example scripts
- Module docstrings - Inline API documentation

For issues or questions, consult the main AILH repository documentation.

---

**Version**: 2.0.0
**Last Updated**: 2025-11-19
