"""
Ultra-realistic synthetic leak signal generator for underground pipes.

Provides a function to create a labeled synthetic dataset mimicking acoustic signals from pipe leaks and other classes,
including 'LEAK', 'QUIET', 'MECHANICAL', and 'RANDOM'.
Leak signals simulate plausible physics-inspired noise, burst, and frequency modulation for ductile iron pipes,
with advanced modeling for diameter, wall thickness, material, leak position, pressure, leak size, fluid type, temperature, and environmental noise.

Usage:
    from synthetic_data import generate_labeled_synthetic_dataset
    signals, labels, meta = generate_labeled_synthetic_dataset(n=500, sample_rate=4096, duration=10)
    # meta is a dict with keys: gain, diameter, wall_thickness, material, position, pressure, leak_size, fluid, temperature, env_noise

"""

import numpy as np
from scipy.signal import butter, sosfilt

from config import *


rng = np.random.default_rng()  # Consistent random generator

# Physical/material parameters
MATERIALS = {
    "ductile_iron": {"speed": 5200, "density": 7200},
    "gray_iron": {"speed": 5000, "density": 7100},
    "steel": {"speed": 5900, "density": 7850},
    "polyethylene": {"speed": 1950, "density": 950}
}
FLUIDS = {
    "water": {"speed": 1480, "density": 1000},
    "oil": {"speed": 1400, "density": 870},
    "air": {"speed": 343, "density": 1.2}
}
ENV_NOISE_LEVELS = ["low", "medium", "high"]

def _pipe_resonant_freq(diameter_mm, wall_thickness_mm, material="ductile_iron", pressure_bar=3.0, temperature_C=15.0):
    mat = MATERIALS.get(material, MATERIALS["ductile_iron"])
    speed = mat["speed"]
    speed *= (1 + 0.003 * (pressure_bar - 3.0)) * (1 - 0.001 * (temperature_C - 15.0))
    diameter_m = diameter_mm / 1000.0
    wall_m = wall_thickness_mm / 1000.0
    correction = 1 - (wall_m / diameter_m) * 0.8
    freq = speed / (np.pi * diameter_m) * correction
    freq = np.clip(freq, 80, 4000)
    return freq

def _leak_position_factor(leak_position_m, diameter_mm, env_noise="medium"):
    diameter_m = diameter_mm / 1000.0
    attenuation = np.exp(-leak_position_m / (3 * diameter_m + 0.1))
    reverberation = min(leak_position_m / 40.0, 1.0)
    noise_floor = {"low": 0.005, "medium": 0.02, "high": 0.05}[env_noise]
    return attenuation, reverberation, noise_floor

def _leak_flow_modulation(t, leak_size_mm, pressure_bar, fluid="water"):
    freq = 1 + 8 * np.exp(-leak_size_mm / 10)
    flow_mod = np.sin(2 * np.pi * freq * t) * (pressure_bar / 5.0) * (leak_size_mm / 50.0)
    flow_mod *= FLUIDS.get(fluid, FLUIDS["water"])["density"] / 1000.0
    return flow_mod

def _burst_modulation(t, burst_freq, burst_width):
    burst = np.sin(2 * np.pi * burst_freq * t) * np.exp(-((t - burst_width) ** 2) / (2 * (burst_width / 2) ** 2))
    return burst

def _broadband_noise(n_samples, sample_rate, low=100.0, high=2000.0, env_noise=0.02):
    nyq = sample_rate / 2.0
    low = max(5.0, min(low, nyq * 0.95))
    high = max(low + 5.0, min(high, nyq * 0.99))
    if not (0 < low < high < nyq):
        low = 100.0
        high = nyq * 0.95
    noise = rng.normal(0, 1, n_samples)
    noise *= rng.uniform(0.9, 1.1)
    sos = butter(3, [low/nyq, high/nyq], btype='band', output='sos')
    filtered = sosfilt(sos, noise)
    filtered += rng.normal(0, env_noise, n_samples)
    return filtered

def _resonance_component(t, freq, amplitude=1.0, decay=0.995):
    return amplitude * np.sin(2 * np.pi * freq * t) * (decay ** t)

def augment_signal(sig):
    sig = sig * rng.uniform(0.9, 1.1)
    shift = rng.integers(0, len(sig)//50)
    sig = np.roll(sig, shift)
    sig += rng.normal(0, 0.01, len(sig))
    sig /= np.max(np.abs(sig)) + 1e-8
    return sig.astype(np.float32)

def generate_labeled_synthetic_dataset(
    n: int = 500,
    sample_rate: int = SAMPLE_RATE,
    duration: float = SAMPLE_DURATION,
    gain_range=(0, 200),
    diameter_range=(10, 2000),
    wall_range=(4, 30),
    position_range=(0.1, 30.0),
    pressure_range=(1.0, 10.0),
    leak_size_range=(1.0, 80.0),
    fluid_types=None,
    temperature_range=(4.0, 30.0),
    env_noise_levels=ENV_NOISE_LEVELS
):
    if fluid_types is None:
        fluid_types = list(FLUIDS.keys())
    NUM_SAMPLES = int(sample_rate * duration)
    t = np.linspace(0, duration, NUM_SAMPLES)
    signals, labels = [], []
    meta = {k: [] for k in ["gain", "diameter", "wall_thickness", "material", "position", "pressure", "leak_size", "fluid", "temperature", "env_noise"]}

    for _ in range(n):
        diameter_mm = rng.uniform(*diameter_range)
        wall_mm = rng.uniform(*wall_range)
        material = rng.choice(list(MATERIALS.keys()))
        temperature_C = rng.uniform(*temperature_range)
        env_noise = rng.choice(env_noise_levels)

        ### LEAK
        pressure_bar = rng.uniform(*pressure_range)
        leak_position_m = rng.uniform(*position_range)
        leak_size_mm = rng.uniform(*leak_size_range)
        fluid = rng.choice(fluid_types)

        freq = _pipe_resonant_freq(diameter_mm, wall_mm, material, pressure_bar, temperature_C)
        freq += rng.uniform(-100, 100)
        attenuation, reverberation, noise_floor = _leak_position_factor(leak_position_m, diameter_mm, env_noise)
        amp = rng.uniform(0.8, 1.2) * attenuation
        noise = _broadband_noise(NUM_SAMPLES, sample_rate, low=100, high=min(freq*2, sample_rate//2 * 0.99), env_noise=noise_floor)
        band_noise = amp * noise * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        burst_freq = rng.uniform(3, 25)
        burst_width = rng.uniform(0.2, duration - 0.2)
        burst = _burst_modulation(t, burst_freq, burst_width) * attenuation
        resonance = _resonance_component(t, freq + rng.uniform(-50, 50), amplitude=0.5 * attenuation, decay=0.99 - reverberation * 0.02)
        leak_flow = _leak_flow_modulation(t, leak_size_mm, pressure_bar, fluid)
        leak_signal = band_noise + burst + resonance + leak_flow
        if reverberation > 0.1:
            leak_signal += rng.normal(0, reverberation * 0.05, NUM_SAMPLES)
        leak_signal = augment_signal(leak_signal)
        signals.append(leak_signal)
        labels.append("LEAK")
        meta["gain"].append(int(rng.uniform(40, 200)))
        meta["diameter"].append(diameter_mm)
        meta["wall_thickness"].append(wall_mm)
        meta["material"].append(material)
        meta["position"].append(leak_position_m)
        meta["pressure"].append(pressure_bar)
        meta["leak_size"].append(leak_size_mm)
        meta["fluid"].append(fluid)
        meta["temperature"].append(temperature_C)
        meta["env_noise"].append(env_noise)

        ### QUIET
        quiet_std = rng.uniform(0.005, 0.02)
        quiet = rng.normal(0, quiet_std + noise_floor, NUM_SAMPLES)
        quiet = augment_signal(quiet)
        signals.append(quiet)
        labels.append("QUIET")
        meta["gain"].append(1)
        meta["diameter"].append(diameter_mm)
        meta["wall_thickness"].append(wall_mm)
        meta["material"].append(material)
        meta["position"].append(None)
        meta["pressure"].append(None)
        meta["leak_size"].append(None)
        meta["fluid"].append(None)
        meta["temperature"].append(temperature_C)
        meta["env_noise"].append(env_noise)

        ### NORMAL - typical operating sounds without leaks
        normal_base_freq = rng.uniform(20, 60)  # Low frequency operational hum
        normal_amp = rng.uniform(0.1, 0.4)
        normal_signal = normal_amp * np.sin(2 * np.pi * normal_base_freq * t + rng.uniform(0, 2 * np.pi))
        # Add some higher harmonics for realism
        for harmonic in [2, 3, 4]:
            harm_amp = normal_amp / (harmonic * 2)
            normal_signal += harm_amp * np.sin(2 * np.pi * normal_base_freq * harmonic * t + rng.uniform(0, 2 * np.pi))
        # Add operational noise
        normal_signal += rng.normal(0, noise_floor * 2, NUM_SAMPLES)
        normal_signal = augment_signal(normal_signal)
        signals.append(normal_signal)
        labels.append("NORMAL")
        meta["gain"].append(int(rng.uniform(*gain_range)))
        meta["diameter"].append(diameter_mm)
        meta["wall_thickness"].append(wall_mm)
        meta["material"].append(material)
        meta["position"].append(None)
        meta["pressure"].append(None)
        meta["leak_size"].append(None)
        meta["fluid"].append(None)
        meta["temperature"].append(temperature_C)
        meta["env_noise"].append(env_noise)

        ### MECHANICAL
        mech_freq = rng.uniform(6, 12)
        mech_amp = rng.uniform(0.3, 0.7)
        mech_mask = rng.random(NUM_SAMPLES) > rng.uniform(0.96, 0.99)
        phase = rng.uniform(0, 2 * np.pi)
        mech = mech_amp * np.sin(2 * np.pi * mech_freq * t + phase) * mech_mask
        mech += rng.normal(0, noise_floor, NUM_SAMPLES)
        mech = augment_signal(mech)
        signals.append(mech)
        labels.append("MECHANICAL")
        meta["gain"].append(int(rng.uniform(*gain_range)))
        meta["diameter"].append(diameter_mm)
        meta["wall_thickness"].append(wall_mm)
        meta["material"].append(material)
        meta["position"].append(None)
        meta["pressure"].append(None)
        meta["leak_size"].append(None)
        meta["fluid"].append(None)
        meta["temperature"].append(temperature_C)
        meta["env_noise"].append(env_noise)

        ### RANDOM
        rand_type = rng.choice(['burst', 'hum', 'pop', 'mix'])
        if rand_type == 'burst':
            rand = rng.normal(0, 1, NUM_SAMPLES) * (rng.random(NUM_SAMPLES) > 0.98)
        elif rand_type == 'hum':
            hum_freq = rng.uniform(50, 400)
            rand = rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * hum_freq * t + rng.uniform(0, 2 * np.pi))
            rand += rng.normal(0, noise_floor, NUM_SAMPLES)
        elif rand_type == 'pop':
            rand = np.zeros(NUM_SAMPLES)
            pop_idx = rng.integers(0, NUM_SAMPLES, size=rng.integers(1, 10))
            rand[pop_idx] = rng.uniform(-1, 1, size=len(pop_idx))
            rand += rng.normal(0, noise_floor, NUM_SAMPLES)
        else:  # 'mix'
            rand = (0.5 * rng.normal(0, 1, NUM_SAMPLES) +
                    0.2 * np.sin(2 * np.pi * rng.uniform(100, 1000) * t) +
                    0.3 * np.sign(np.sin(2 * np.pi * rng.uniform(10, 50) * t)))
            rand += rng.normal(0, noise_floor, NUM_SAMPLES)
        rand = augment_signal(rand)
        signals.append(rand)
        labels.append("RANDOM")
        meta["gain"].append(int(rng.uniform(*gain_range)))
        meta["diameter"].append(diameter_mm)
        meta["wall_thickness"].append(wall_mm)
        meta["material"].append(material)
        meta["position"].append(None)
        meta["pressure"].append(None)
        meta["leak_size"].append(None)
        meta["fluid"].append(None)
        meta["temperature"].append(temperature_C)
        meta["env_noise"].append(env_noise)

    return signals, labels, meta

def save_labeled_synthetic_dataset_by_label(signals, labels, meta, out_base_dir, sample_rate=SAMPLE_RATE):
    """
    Save synthetic signals with all metadata into folders by label.
    Uses ~ as delimiter in filenames.
    """
    import os
    from scipy.io.wavfile import write
    for i, sig in enumerate(signals):
        label = labels[i]
        label_folder = label
        if label not in DATA_LABELS:
            label_folder = "UNCLASSIFIED"
        out_dir = os.path.join(out_base_dir, label_folder)
        os.makedirs(out_dir, exist_ok=True)
        gain = meta["gain"][i]
        diam = meta["diameter"][i]
        wall = meta["wall_thickness"][i]
        material = meta["material"][i]
        position = meta["position"][i]
        pressure = meta["pressure"][i]
        leak_size = meta["leak_size"][i]
        fluid = meta["fluid"][i]
        temp = meta["temperature"][i]
        env_noise = meta["env_noise"][i]
        pos_str = f"{position:.2f}" if position is not None else ""
        prs_str = f"{pressure:.1f}" if pressure is not None else ""
        lks_str = f"{leak_size:.1f}" if leak_size is not None else ""
        fld_str = f"{fluid}" if fluid is not None else "none"
        fname = (
            f"synthetic~{label.lower()}~gain{gain:03d}~diam{int(diam):04d}~wall{int(wall):02d}~{material}~pos{pos_str}~prs{prs_str}~lks{lks_str}~{fld_str}~temp{int(temp):02d}~env{env_noise}~{i:04d}.wav"
        )
        write(os.path.join(out_dir, fname), sample_rate, (sig * 32767).astype(np.int16))

def generate_and_save_train_val_sets(train_n=10, val_n=5, sample_rate=SAMPLE_RATE, duration=SAMPLE_DURATION):
    """
    Generates and saves two labeled sets: one to TRAINING and one to VALIDATION folders.
    """
    # Training set
    signals_train, labels_train, meta_train = generate_labeled_synthetic_dataset(n=train_n, sample_rate=sample_rate, duration=duration)
    save_labeled_synthetic_dataset_by_label(signals_train, labels_train, meta_train, REFERENCE_TRAINING_DIR, sample_rate=sample_rate)
    # Validation set
    signals_val, labels_val, meta_val = generate_labeled_synthetic_dataset(n=val_n, sample_rate=sample_rate, duration=duration)
    save_labeled_synthetic_dataset_by_label(signals_val, labels_val, meta_val, REFERENCE_VALIDATION_DIR, sample_rate=sample_rate)

if __name__ == "__main__":
    generate_and_save_train_val_sets(train_n=100, val_n=10)
