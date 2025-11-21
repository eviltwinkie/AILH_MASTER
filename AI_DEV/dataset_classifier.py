#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Classifier - PyTorch Leak Detection Inference

High-performance inference script for classifying acoustic leak detection datasets using
PyTorch models. Supports both binary (LEAK/NOLEAK) and multiclass (5-class) models.

Key Features:
    - Dual model support: --model-binary or --model-multi
    - File-level classification with two-stage temporal segmentation
    - Multiple decision rules (mean, long_vote, any_long, frac_vote)
    - CSV output with per-class probabilities
    - Batch processing for memory efficiency
    - GPU acceleration with FP16 inference
    - Resume capability (skips already classified files)

Model Modes:
    Binary (--model-binary):
        - 2 classes: LEAK, NOLEAK
        - Model path: PROC_MODELS/binary/best.pth
        - Metadata: PROC_MODELS/binary/model_meta.json
    
    Multiclass (--model-multi):
        - 5 classes: BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED
        - Model path: PROC_MODELS/multiclass/best.pth
        - Metadata: PROC_MODELS/multiclass/model_meta.json

Decision Rules:
    - mean: Average leak prob across all long segments >= threshold
    - long_vote: Majority of long segments have mean(short) >= threshold (paper-style)
    - any_long: Any long segment has mean(short) >= threshold → LEAK
    - frac_vote: Fraction of long segments >= threshold meets --long-frac requirement

Probability Modes:
    - softmax: Use softmax over class logits, take P(LEAK) only
    - blend: Average 0.5*P_leak_softmax + 0.5*sigmoid(aux_leak)

CSV Output:
    filepath,is_leak,leak_conf_mean,per_long_probs_json,long_pos_frac,notes

Usage Examples:
    # Binary model (LEAK/NOLEAK)
    python dataset_classifier.py --model-binary --in-dir /path/to/audio --out leak_report.csv
    
    # Multiclass model (5 classes)
    python dataset_classifier.py --model-multi --in-dir /path/to/audio --out classification.csv
    
    # Custom decision rule
    python dataset_classifier.py --model-binary --in-dir /path/to/audio \\
        --decision frac_vote --long-frac 0.25 --thr 0.35 --out results.csv

Author: AI Development Team
Version: 2.0
Last Updated: November 20, 2025
"""
import os, json, csv, math, argparse, sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np

from global_config import PROC_MODELS, DATASET_TRAINING, LONG_WINDOW, SHORT_WINDOW, SAMPLE_RATE, SAMPLE_DURATION, N_MELS, N_FFT, HOP_LENGTH, N_POWER

try:
    import h5py
except Exception as _e:
    h5py = None

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchaudio
except Exception as e:
    torchaudio = None

# ------------------------ Utilities ------------------------

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd

def _coerce_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise TypeError("Unsupported checkpoint format for state_dict")
    # Strip common wrappers
    for pref in ("_orig_mod.", "module."):
        sd = _strip_prefix(sd, pref)
    return sd

def _pick_h5_for_config(stage_dir: Path) -> Optional[Path]:
    """Find first available HDF5 dataset file for config extraction."""
    for name in ["TRAINING_DATASET.H5", "VALIDATION_DATASET.H5", "TESTING_DATASET.H5"]:
        p = stage_dir / name
        if p.exists():
            return p
    return None

def load_builder_config(stage_dir: Path) -> Dict:
    """Load builder configuration from HDF5 attrs or use defaults."""
    # Default values aligned with dataset_builder.py
    cfg = {
        "sample_rate": SAMPLE_RATE,
        "duration_sec": SAMPLE_DURATION,
        "long_window": LONG_WINDOW,
        "short_window": SHORT_WINDOW,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "power": N_POWER,
        "center": False
    }
    if h5py is None:
        return cfg

    h5p = _pick_h5_for_config(stage_dir)
    if h5p is None:
        return cfg
    try:
        with h5py.File(str(h5p), "r") as h:
            if "config_json" in h.attrs:
                config_val = h.attrs["config_json"]
                if isinstance(config_val, bytes):
                    config_str = config_val.decode('utf-8')
                elif isinstance(config_val, str):
                    config_str = config_val
                else:
                    config_str = str(config_val)
                j = json.loads(config_str)
                cfg.update({k: j[k] for k in cfg.keys() if k in j})
    except Exception as e:
        pass
    return cfg

def load_model_meta(model_type: str) -> Dict:
    """Load model metadata from model_meta.json."""
    meta_path = Path(PROC_MODELS) / model_type / "model_meta.json"
    meta = {}
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        pass
    return meta

def list_wavs(in_dir: Path) -> List[Path]:
    wavs = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac")):
                wavs.append(Path(root) / f)
    return sorted(wavs)

def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio not available; cannot read WAV.")
    data, sr = torchaudio.load(str(path))  # [C, T], float32 in [-1,1]
    data = data.numpy()
    if data.shape[0] > 1:
        data = data.mean(axis=0, dtype=np.float32)[None, :]
    return data[0], int(sr)

def pad_or_trim(x: np.ndarray, target: int) -> np.ndarray:
    n = x.shape[0]
    if n == target:
        return x.astype(np.float32, copy=False)
    out = np.zeros(target, dtype=np.float32)
    out[:min(n, target)] = x[:min(n, target)]
    return out

def two_stage_segments(x: np.ndarray, long_window: int, short_window: int) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      segs: [num_long*num_short, short_window] float32
      num_long, num_short
    """
    T = x.shape[0]
    num_long = T // long_window
    if num_long <= 0:
        return np.zeros((0, short_window), dtype=np.float32), 0, 0
    num_short = long_window // short_window
    if num_short <= 0:
        return np.zeros((0, short_window), dtype=np.float32), 0, 0
    view = x[:num_long*long_window].reshape(num_long, long_window)
    segs = view.reshape(num_long, num_short, short_window).reshape(num_long*num_short, short_window)
    return segs.astype(np.float32, copy=False), num_long, num_short

# ------------------------ Tiny CNN compatible head ------------------------

class LeakCNNMulti(nn.Module):
    """
    Dual-head CNN for leak detection with auxiliary binary classification.
    
    Supports two modes:
    - Binary mode (n_classes=2): LEAK vs NOLEAK classification
    - Multi-class mode (n_classes=5): BACKGROUND, CRACK, LEAK, NORMAL, UNCLASSIFIED
    
    Architecture matches dataset_trainer.py exactly.
    """
    def __init__(self, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.adapt = nn.AdaptiveAvgPool2d((16, 1))
        self.fc1 = nn.Linear(128 * 16 * 1, 256)
        self.cls_head = nn.Linear(256, n_classes)
        self.leak_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, n_mels, t_frames] (channels_last ok)
        """
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = self.adapt(x); x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.cls_head(x)
        leak_logit = self.leak_head(x).squeeze(1)
        return logits, leak_logit

# ------------------------ Inference helpers ------------------------

def build_mel(cfg: Dict) -> nn.Module:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for Mel computation.")
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=cfg["power"],
        center=cfg["center"],
        norm=None
    )
    return nn.Sequential(
        m,
        torchaudio.transforms.AmplitudeToDB(stype="power")
    )

def load_weights_and_meta(model_dir: Path, n_classes: int, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load model weights and metadata from model directory.
    
    Args:
        model_dir: Path to model directory (e.g., PROC_MODELS/binary or PROC_MODELS/multiclass)
        n_classes: Number of classes (2 for binary, 5 for multiclass)
        device: PyTorch device
    
    Returns:
        Tuple of (model, metadata_dict)
    """
    # Try to load metadata
    meta_path = model_dir / "model_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                print(f"[META] Loaded metadata from {meta_path}")
        except Exception as e:
            print(f"[WARN] Could not load metadata: {e}")
    
    # Get dropout from metadata, fallback to default
    dropout = meta.get('trainer_cfg', {}).get('dropout', 0.25)
    print(f"[MODEL] Using dropout={dropout} from metadata")
    
    # Initialize model
    model = LeakCNNMulti(n_classes=n_classes, dropout=dropout)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    model = model.eval()
    
    # Try to load weights
    candidates = [
        model_dir / "best.pth",
        model_dir / "checkpoints" / "last.pth",
    ]
    
    for ck in candidates:
        if not ck.exists():
            continue
        try:
            obj = torch.load(str(ck), map_location=device, weights_only=False)
            sd = _coerce_state_dict(obj)
            model.load_state_dict(sd, strict=False)
            print(f"[MODEL] Loaded weights from {ck}")
            return model, meta
        except Exception as e:
            print(f"[WARN] Could not load {ck}: {e}")
    
    raise FileNotFoundError(f"No valid checkpoint found in {model_dir}")

def infer_one_wav(model: nn.Module, mel_db: nn.Module, wav: np.ndarray, cfg: Dict,
                  device: torch.device, batch_segments: int = 2048,
                  prob: str = "softmax", leak_idx: int = 0) -> Tuple[List[float], float]:
    """
    Returns:
      per_long_probs: list[float]   (length = num_long) 
      avg_prob: float               (mean of per_long)
    """
    # segment into short chunks
    segs, num_long, num_short = two_stage_segments(wav, int(cfg["long_window"]), int(cfg["short_window"]))
    if num_long == 0 or num_short == 0:
        return [], 0.0

    B = segs.shape[0]
    probs_short = np.empty((B,), dtype=np.float32)

    # batch through Mel+model
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=True):
        for off in range(0, B, batch_segments):
            end = min(off + batch_segments, B)
            batch = torch.from_numpy(segs[off:end]).to(device, non_blocking=True)  # [b, Tshort]
            batch = batch.unsqueeze(1)  # [b, 1, T]
            m = mel_db(batch)           # [b, n_mels, t_frames]
            m = m.unsqueeze(1)          # [b, 1, n_mels, t_frames]
            m = m.to(memory_format=torch.channels_last)

            logits, aux = model(m)
            if prob == "softmax":
                p = F.softmax(logits, dim=1)[..., leak_idx]  # assumes LEAK is index 0 (enforced below)
            elif prob == "blend":
                p = 0.5*F.softmax(logits, dim=1)[..., leak_idx] + 0.5*torch.sigmoid(aux)
            else:
                raise ValueError("--prob must be softmax or blend")
            probs_short[off:end] = p.detach().float().cpu().numpy()

    # Reduce to per-long by averaging num_short chunks
    per_long = probs_short.reshape(num_long, num_short).mean(axis=1)
    return per_long.tolist(), float(per_long.mean())

def decide_label(per_long: List[float], decision: str, thr: float, long_frac: float, min_long: int) -> Tuple[int, float, float, str]:
    """
    Returns (is_leak, leak_conf_mean, long_pos_frac, notes)
    """
    if len(per_long) == 0:
        return 0, 0.0, 0.0, "no_long_segments"

    per_long_arr = np.asarray(per_long, dtype=np.float32)
    long_hits = (per_long_arr >= thr).astype(np.float32)
    long_pos_frac = float(long_hits.mean())
    leak_conf_mean = float(per_long_arr.mean())

    notes = []
    if decision == "mean":
        is_leak = int(leak_conf_mean >= thr)
        notes.append("mean>=thr")
    elif decision == "long_vote":
        is_leak = int(long_pos_frac >= 0.5)
        notes.append("majority_long")
    elif decision == "any_long":
        is_leak = int(long_hits.max() > 0.5)
        notes.append("any_long")
    elif decision == "frac_vote":
        is_leak = int((long_pos_frac >= long_frac) and (int(long_hits.sum()) >= min_long))
        notes.append(f"frac>={long_frac:.2f}&min_long>={min_long}")
    else:
        raise ValueError("invalid decision rule")

    return is_leak, leak_conf_mean, long_pos_frac, ";".join(notes)

# ------------------------ Main CLI ------------------------

def main():
    p = argparse.ArgumentParser(description="Leak detection inference with binary or multiclass models")
    
    # Model selection (mutually exclusive)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-binary", action="store_true", help="Use binary model (LEAK/NOLEAK)")
    model_group.add_argument("--model-multi", action="store_true", help="Use multiclass model (5 classes)")
    
    # Input/output
    p.add_argument("--stage-dir", type=str, default=None, help="Dataset stage root (default: MASTER_DATASET from global_config)")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--in-dir", type=str, help="Directory of audio files (recursively scanned)")
    input_group.add_argument("--filelist", type=str, help="Text file with one audio path per line")
    p.add_argument("--out", type=str, default="leak_report.csv", help="Output CSV")
    
    # Inference settings
    p.add_argument("--prob", type=str, default="softmax", choices=["softmax", "blend"], 
                   help="Probability head (softmax or blend with aux leak head)")
    p.add_argument("--decision", type=str, default="frac_vote",
                   choices=["mean", "long_vote", "any_long", "frac_vote"], help="Decision rule")
    p.add_argument("--long-frac", type=float, default=0.25, 
                   help="For frac_vote: required fraction of long segments >= threshold")
    p.add_argument("--min-long", type=int, default=1, 
                   help="For frac_vote: require at least N long segments to be positive")
    p.add_argument("--thr", type=float, default=0.35, help="Leak probability threshold (on per-long means)")
    p.add_argument("--batch-segments", type=int, default=2048, help="Short segments per forward micro-batch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = p.parse_args()
    
    # Determine model type and paths
    if args.model_binary:
        model_type = "binary"
    else:  # model_multi
        model_type = "multiclass"
    
    model_dir = Path(PROC_MODELS) / model_type
    stage_dir = Path(args.stage_dir) if args.stage_dir else Path(DATASET_TRAINING)
    
    # Load model metadata first to get correct class names and parameters
    meta_path = model_dir / "model_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                temp_meta = json.load(f)
            class_names = temp_meta.get("class_names", [])
            n_classes = temp_meta.get("num_classes", len(class_names))
            print(f"[CONFIG] Loaded class_names from metadata: {class_names}")
        except Exception as e:
            print(f"[WARN] Could not load metadata: {e}, using defaults")
            if args.model_binary:
                class_names = ["LEAK", "NOLEAK"]  # fallback
                n_classes = 2
            else:
                class_names = ["BACKGROUND", "CRACK", "LEAK", "NORMAL", "UNCLASSIFIED"]  # fallback
                n_classes = 5
    else:
        print(f"[WARN] No metadata found, using defaults")
        if args.model_binary:
            class_names = ["LEAK", "NOLEAK"]  # fallback
            n_classes = 2
        else:
            class_names = ["BACKGROUND", "CRACK", "LEAK", "NORMAL", "UNCLASSIFIED"]  # fallback
            n_classes = 5
    
    print(f"[CONFIG] Model type: {model_type}")
    print(f"[CONFIG] Model directory: {model_dir}")
    print(f"[CONFIG] Stage directory: {stage_dir}")
    print(f"[CONFIG] Classes: {class_names} ({n_classes} total)")
    
    # Load builder config
    cfg = load_builder_config(stage_dir)
    device = torch.device(args.device)
    
    # Build mel transform
    mel = build_mel(cfg).to(device).eval()
    
    # Load model and metadata
    try:
        model, meta = load_weights_and_meta(model_dir, n_classes=n_classes, device=device)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get leak index from metadata (not hardcoded)
    leak_idx = meta.get("leak_idx", class_names.index("LEAK") if "LEAK" in class_names else 0)
    print(f"[CONFIG] Using leak_idx={leak_idx} from model metadata")
    
    # Verify leak index is reasonable
    if leak_idx >= n_classes:
        print(f"[WARN] leak_idx={leak_idx} >= n_classes={n_classes}, using class_names.index('LEAK')")
        leak_idx = class_names.index("LEAK") if "LEAK" in class_names else 0
        print(f"[CONFIG] Corrected leak_idx={leak_idx}")
    
    # Collect files
    if args.in_dir:
        files = list_wavs(Path(args.in_dir))
    else:
        with open(args.filelist, "r") as f:
            files = [Path(line.strip()) for line in f if line.strip()]
    
    if not files:
        print("[ERROR] No audio files found.", file=sys.stderr)
        sys.exit(2)
    
    total = len(files)
    leak_ct = 0
    
    print(f"[INFO] Processing {total} files...")
    
    # Prepare CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["filepath", "is_leak", "leak_conf_mean", "per_long_probs_json", "long_pos_frac", "notes"])
        
        for idx, fp in enumerate(files, 1):
            try:
                wav, sr = read_audio(fp)
                if sr != int(cfg["sample_rate"]):
                    # resample to cfg SR
                    if torchaudio is None:
                        raise RuntimeError("torchaudio required for resampling")
                    wav_t, _ = torchaudio.load(str(fp))
                    wav_t = torchaudio.functional.resample(wav_t, sr, int(cfg["sample_rate"]))
                    wav = wav_t.mean(dim=0).numpy().astype(np.float32)
                    sr = int(cfg["sample_rate"])
                
                # pad/trim to full duration
                target = int(cfg["sample_rate"] * cfg["duration_sec"])
                wav = pad_or_trim(wav, target)
                
                per_long, leak_mean = infer_one_wav(model, mel, wav, cfg, device,
                                                    batch_segments=args.batch_segments, 
                                                    prob=args.prob, leak_idx=leak_idx)
                is_leak, leak_conf_mean, long_pos_frac, notes = decide_label(
                    per_long, decision=args.decision, thr=float(args.thr),
                    long_frac=float(args.long_frac), min_long=int(args.min_long)
                )
                
                if is_leak:
                    leak_ct += 1
                
                w.writerow([str(fp), is_leak, f"{leak_conf_mean:.6f}", 
                           json.dumps(per_long), f"{long_pos_frac:.6f}", notes])
                
                if idx % 100 == 0:
                    print(f"[PROGRESS] {idx}/{total} files processed ({leak_ct} leaks detected)")
                    
            except Exception as e:
                w.writerow([str(fp), 0, f"0.0", "[]", "0.0", f"error:{type(e).__name__}:{e}"])
                print(f"[ERROR] Failed to process {fp}: {e}")
    
    print(f"\n[DONE] Results written to: {out_path}")
    print(f"[STATS] Leaks detected: {leak_ct}/{total} ({(leak_ct/total*100.0):.1f}%)")
    print(f"[STATS] Decision rule: {args.decision}, threshold: {args.thr}, long_frac: {args.long_frac}, min_long: {args.min_long}")
    print(f"[STATS] Model: {model_type}, SR: {cfg['sample_rate']}Hz, "
          f"windows: long={cfg['long_window']} short={cfg['short_window']}")


if __name__ == "__main__":
    main()
