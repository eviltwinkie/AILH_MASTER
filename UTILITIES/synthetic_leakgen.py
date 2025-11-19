import os
import glob
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.join("..")
LEAK_DIR = os.path.join(BASE_DIR, "REFERENCE_DATA", "TRAINING", "LEAK")
UNCLASSIFIED_DIR = os.path.join(BASE_DIR, "REFERENCE_DATA", "TRAINING", "UNCLASSIFIED")
NORMAL_DIR = os.path.join(BASE_DIR, "REFERENCE_DATA", "TRAINING", "NORMAL")


def generate_one_synth(src_wav, out_dir, duration_sec, sr, overlay_folder, label, j):
    y, _ = librosa.load(src_wav, sr=sr, mono=True)
    if len(y) < int(sr * duration_sec):
        y = np.pad(y, (0, int(sr*duration_sec) - len(y)), mode='constant')
    elif len(y) > int(sr * duration_sec):
        y = y[:int(sr*duration_sec)]
    base = os.path.splitext(os.path.basename(src_wav))[0]
    this_fft = np.abs(np.fft.rfft(y))
    noise = np.random.randn(len(y))
    noise_fft = np.fft.rfft(noise)
    phase = np.angle(noise_fft)
    synth_fft = this_fft[:len(phase)] * np.exp(1j * phase)
    synth = np.fft.irfft(synth_fft)
    synth /= np.max(np.abs(synth)) + 1e-10
    # Optionally add background from the other class
    if overlay_folder:
        bg_files = glob.glob(os.path.join(overlay_folder, "*.wav"))
        if bg_files:
            import random
            bg, _ = librosa.load(random.choice(bg_files), sr=sr, mono=True)
            if len(bg) < len(synth):
                bg = np.pad(bg, (0, len(synth)-len(bg)), mode='wrap')
            else:
                bg = bg[:len(synth)]
            synth = 0.8 * synth + 0.2 * bg
            synth /= np.max(np.abs(synth)) + 1e-10
    out_name = os.path.join(out_dir, f"{base}_synthetic_{label}_{j+1:03d}.wav")
    sf.write(out_name, synth.astype(np.float32), sr)
    print(f"[✓] Saved: {out_name}")
    return out_name

def generate_synthetic_batch(src_files, out_dir, n_to_generate, duration_sec, sr, overlay_folder, label, max_workers=8):
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for src_wav in src_files:
            for j in range(n_to_generate):
                tasks.append(
                    executor.submit(
                        generate_one_synth, src_wav, out_dir, duration_sec, sr, overlay_folder, label, j
                    )
                )
        for future in as_completed(tasks):
            _ = future.result()  # will print as they're saved

if __name__ == "__main__":
    sr = 4096
    duration_sec = 10.0
    n_to_generate = 3
    max_workers = os.cpu_count() or 8

    # --- SYNTHETIC LEAKS ---
    leak_files = glob.glob(os.path.join(LEAK_DIR, "*.wav"))
    if not leak_files:
        raise RuntimeError("No leak WAVs found!")
    print(f"[i] Generating synthetic leaks using {max_workers} threads...")
    generate_synthetic_batch( leak_files, LEAK_DIR, n_to_generate, duration_sec, sr, UNCLASSIFIED_DIR, "LEAK", max_workers )

    # # --- SYNTHETIC NORMALS ---
    # normal_files = glob.glob(os.path.join(NORMAL_DIR, "*.wav"))
    # if not normal_files:
    #     raise RuntimeError("No normal WAVs found!")
    # print(f"[i] Generating synthetic normals using {max_workers} threads...")
    # generate_synthetic_batch( normal_files, NORMAL_DIR, n_to_generate, duration_sec, sr, UNCLASSIFIED_DIR, "NORMAL", max_workers )
