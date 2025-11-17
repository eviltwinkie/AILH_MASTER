#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leak_directory_classifier_v2.py
-------------------------------------------------------------------
No command-line args. Configure paths below.

- Reads builder config & labels from an HDF5 header (config_json, labels_json).
- Resamples WAVs on the fly to the HDF5's sample_rate when needed.
- Loads model weights from cnn_model_best.h5 (state_dict) or cnn_model_full.h5 (ckpt with "model").
- Classifies every WAV under INPUT_DIR recursively and writes a CSV with
  top-1 prediction and per-class confidences.

Edit the three PATHS below to suit your layout if needed.
"""

from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ===================== USER CONFIG (no CLI) =====================

STAGE_DIR   = Path("/DEVELOPMENT/DATASET_REFERENCE")
INPUT_DIR   = STAGE_DIR / "INFERENCE"      # will fall back to STAGE_DIR/TESTING if missing
OUTPUT_CSV  = STAGE_DIR / "reports" / "classification_report.csv"

# Segments per micro-batch on GPU (tune if you have more/less VRAM)
BATCH_SEGMENTS = 16384

# ================================================================

# Torch perf knobs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ------------------------ Model (same as trainer) ------------------------

class LeakCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.adapt = nn.AdaptiveAvgPool2d((16, 1))
        self.fc1 = nn.Linear(128 * 16 * 1, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------ HDF5 config loader ------------------------

def _pick_h5_for_config(stage_dir: Path) -> Optional[Path]:
    # Try these in order
    candidates = [
        stage_dir / "TRAINING_DATASET.H5",
        stage_dir / "VALIDATION_DATASET.H5",
        stage_dir / "TESTING_DATASET.H5",
        stage_dir / "LEAK_DATASET.H5",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_builder_config_and_labels(stage_dir: Path) -> Tuple[Dict, List[str]]:
    h5_path = _pick_h5_for_config(stage_dir)
    if h5_path is None:
        # Minimal sane defaults if no HDF5 header is available
        cfg = dict(sample_rate=4096, n_fft=512, hop_length=128,
                   n_mels=64, power=1.0, center=False,
                   long_window=1024, short_window=512, duration_sec=10)
        labels = ["BACKGROUND", "CRACK", "LEAK", "NORMAL", "UNCLASSIFIED"]
        print("[CFG] No HDF5 found; using built-in defaults.")
        return cfg, labels

    with h5py.File(str(h5_path), "r") as f:
        # config_json
        cfg_s = f.attrs.get("config_json", None)
        if isinstance(cfg_s, (bytes, bytearray)):
            cfg_s = cfg_s.decode("utf-8")
        cfg = json.loads(cfg_s) if cfg_s else {}
        # labels_json
        lbl_s = f.attrs.get("labels_json", None)
        if isinstance(lbl_s, (bytes, bytearray)):
            lbl_s = lbl_s.decode("utf-8")
        labels = json.loads(lbl_s) if lbl_s else None

    # Fill any missing keys with sensible defaults
    cfg.setdefault("sample_rate", 4096)
    cfg.setdefault("n_fft", 512)
    cfg.setdefault("hop_length", 128)
    cfg.setdefault("n_mels", 64)
    cfg.setdefault("power", 1.0)
    cfg.setdefault("center", False)
    cfg.setdefault("long_window", 1024)
    cfg.setdefault("short_window", 512)
    cfg.setdefault("duration_sec", 10)

    if not labels:
        labels = ["BACKGROUND", "CRACK", "LEAK", "NORMAL", "UNCLASSIFIED"]

    print(f"[CFG] From {h5_path.name}: {cfg}")
    print(f"[CFG] Classes: {labels}")
    return cfg, labels

# ------------------------ Weight loading ------------------------

def find_model_paths(stage_dir: Path) -> List[Path]:
    candidates = [
        stage_dir / "MODELS" / "cnn_model_best.h5",
        stage_dir / "MODELS" / "cnn_model_full.h5",
        Path("cnn_model_best.h5"),
        Path("cnn_model_full.h5"),
    ]
    return [p for p in candidates if p.exists()]

def load_model_from_paths(paths: List[Path], n_classes: int, device: torch.device) -> LeakCNN:
    model = LeakCNN(n_classes=n_classes, dropout=0.25).to(device)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    for p in paths:
        try:
            ck = torch.load(str(p), map_location=device, weights_only=False)
            if isinstance(ck, dict) and "model" in ck:
                model.load_state_dict(ck["model"])
            else:
                model.load_state_dict(ck)
            print(f"[MODEL] Loaded: {p}")
            return model
        except Exception as e:
            print(f"[MODEL] Failed to load {p}: {e}")
    raise FileNotFoundError("No usable model weights found. Expected cnn_model_best.h5 or cnn_model_full.h5.")

# ------------------------ Audio utils ------------------------

def load_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    ch = data.shape[1]
    if ch == 1:
        wav = data[:, 0]
    elif ch == 2:
        wav = (data[:, 0] + data[:, 1]) * 0.5
    else:
        wav = data.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)

def pad_or_trim(wav: np.ndarray, target_samples: int) -> np.ndarray:
    n = wav.shape[0]
    if n == target_samples:
        return wav
    out = np.zeros(target_samples, dtype=np.float32)
    L = min(n, target_samples)
    if L:
        out[:L] = wav[:L]
    return out

def segment_short_windows(x: np.ndarray, long_window: int, short_window: int) -> np.ndarray:
    num_long  = x.shape[0] // long_window
    num_short = long_window // short_window
    return x[: num_long * long_window].reshape(num_long, long_window)\
             .reshape(num_long, num_short, short_window)\
             .reshape(-1, short_window)

def build_mel_transform_from_cfg(cfg: Dict, device: torch.device) -> torch.nn.Module:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        n_mels=int(cfg["n_mels"]),
        power=float(cfg["power"]),
        center=bool(cfg.get("center", False)),
    ).eval().to(device)
    return mel

# ------------------------ Inference core ------------------------

@torch.inference_mode()
def classify_wav(
    path: Path,
    model: LeakCNN,
    mel_transform: torch.nn.Module,
    cfg: Dict,
    device: torch.device,
    batch_segments: int = 16384,
) -> np.ndarray:
    sr_cfg   = int(cfg["sample_rate"])
    n_long   = int(cfg["duration_sec"]) * sr_cfg // int(cfg["long_window"])  # not needed explicitly, just consistency
    power    = float(cfg["power"])
    db_mult  = 10.0 if abs(power - 2.0) < 1e-9 else 20.0

    wav, sr = load_wav_mono(path)
    if sr != sr_cfg:
        # Resample on the fly to match builder config
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_r = torchaudio.functional.resample(wav_t, sr, sr_cfg)
        wav = wav_r.squeeze(0).to(torch.float32).cpu().numpy()

    # exact length
    target_samples = sr_cfg * int(cfg["duration_sec"])
    wav = pad_or_trim(wav, target_samples)

    # segment into short windows
    segs = segment_short_windows(wav, int(cfg["long_window"]), int(cfg["short_window"]))  # [B, short_window]
    B = segs.shape[0]

    # accumulate probabilities across segments
    probs_sum = None

    for off in range(0, B, batch_segments):
        end = min(off + batch_segments, B)
        batch = torch.from_numpy(segs[off:end]).to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mels = mel_transform(batch)  # [b, n_mels, t_frames]
        mels = mels.float().clamp_min_(1e-10).log10_().mul_(db_mult)  # dB
        logits = model(mels.unsqueeze(1).contiguous(memory_format=torch.channels_last))
        p = torch.softmax(logits.float(), dim=1)  # [b, C]

        # sum over segments (better numeric stability vs avg each time)
        s = p.sum(dim=0).detach().cpu().numpy()
        probs_sum = s if probs_sum is None else (probs_sum + s)

    probs = probs_sum / float(B)
    return probs  # [C] numpy

# ------------------------ File walking & report ------------------------

def find_wavs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.wav") if p.is_file()]

def ensure_paths():
    out_dir = OUTPUT_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Choose input dir fallback
    if not INPUT_DIR.exists():
        alt = STAGE_DIR / "TESTING"
        if alt.exists():
            print(f"[INFO] INPUT_DIR not found; using fallback: {alt}")
            return alt
        else:
            raise FileNotFoundError(f"No INPUT_DIR found at {INPUT_DIR} and fallback {alt} missing.")
    return INPUT_DIR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, labels = load_builder_config_and_labels(STAGE_DIR)
    n_classes = len(labels)

    model_paths = find_model_paths(STAGE_DIR)
    model = load_model_from_paths(model_paths, n_classes=n_classes, device=device)
    mel_transform = build_mel_transform_from_cfg(cfg, device)

    root = ensure_paths()
    wavs = find_wavs(root)
    if not wavs:
        print(f"[INFO] No WAV files under: {root}")
        return

    print(f"[RUN] Files: {len(wavs)}  →  CSV: {OUTPUT_CSV}")
    prob_cols = [f"prob_{name}" for name in labels]
    header = ["filepath", "predicted_label", "predicted_confidence"] + prob_cols

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for p in wavs:
            try:
                probs = classify_wav(
                    p, model, mel_transform, cfg, device,
                    batch_segments=BATCH_SEGMENTS
                )  # [C]
                top = int(np.argmax(probs))
                top_label = labels[top] if top < len(labels) else f"C{top}"
                top_conf = float(probs[top])
                row = [str(p), top_label, f"{top_conf:.6f}"] + [f"{float(x):.6f}" for x in probs.tolist()]
                w.writerow(row)
            except Exception as e:
                print(f"[WARN] {p}: {e}")
                row = [str(p), "ERROR", "0.000000"] + ["0.000000"] * n_classes
                w.writerow(row)

    print(f"[DONE] Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
