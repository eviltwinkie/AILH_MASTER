#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_classifier_new.py — pragmatic recall-first leak classifier

Highlights
----------
• Standalone & explicit: defines its own tiny CNN head (compatible with trainer defaults) and
  can also load state_dicts from common checkpoint formats: raw state_dict, {"model": ...},
  DDP "module."-prefixed, or torch.compile "_orig_mod."-prefixed.
• Robust config discovery:
    - Reads mel/segmentation parameters from any of the builder HDF5 files under STAGE_DIR
      (TRAINING/VALIDATION/TESTING_DATASET.H5) via HDF5 attrs["config_json"].
    - Falls back to sensible defaults matching dataset_builder.py if HDF5 not found.
• Two-stage segmentation to mirror the paper: long_window → short_window segments.
• Multiple decision rules that favor recall in inference-only pipelines:
    - mean         : average leak prob across all *long* segments >= thr
    - long_vote    : majority of *long* segments have mean(short) >= thr  (paper-style)
    - any_long     : if any long segment has mean(short) >= thr → LEAK
    - frac_vote    : fraction of long segments with mean(short) >= thr >= --long-frac
• Probability heads:
    - softmax (default): use softmax over class logits, take P(LEAK) only
    - blend            : if aux leak head exists, average 0.5*P_leak_softmax + 0.5*sigmoid(aux_leak)
• Sensible defaults to maximize recall safely:
    --decision frac_vote --long-frac 0.25 --thr 0.35

CSV Output
----------
filepath,is_leak,leak_conf_mean,per_long_probs_json,long_pos_frac,notes

Usage Example
-------------
python dataset_classifier_new.py \
  --stage-dir /DEVELOPMENT/ROOT_AILH/DATA_STORE/MASTER_DATASET \
  --in-dir   /DEVELOPMENT/ROOT_AILH/DATA_STORE/MASTER_DATASET/INFERENCE/LEAK \
  --prob softmax \
  --decision frac_vote --long-frac 0.25 --thr 0.35 \
  --out leak_report.csv
"""
import os, json, csv, math, argparse, sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np

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
    for name in ["TRAINING_DATASET.H5","VALIDATION_DATASET.H5","TESTING_DATASET.H5",
                 "LEAK_DATASET.H5","INCREMENTAL_DATASET.H5"]:
        p = stage_dir / name
        if p.exists():
            return p
    return None

def load_builder_config(stage_dir: Path) -> Dict:
    # Default values aligned with dataset_builder.py
    cfg = {
        "sample_rate": 4096,
        "duration_sec": 10,
        "long_window": 1024,
        "short_window": 512,
        "n_mels": 64,
        "n_fft": 512,
        "hop_length": 128,
        "power": 1.0,
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
                j = json.loads(h.attrs["config_json"])
                cfg.update({k: j[k] for k in cfg.keys() if k in j})
    except Exception as e:
        pass
    return cfg

def load_model_meta(stage_dir: Path) -> Dict:
    md = stage_dir / "MODELS" / "model_meta.json"
    meta = {}
    try:
        with open(md, "r") as f:
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
    Feature trunk -> (multiclass logits, aux leak logit)
    Lightweight trunk to remain compatible with training defaults.
    """
    def __init__(self, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d((2,1), (2,1))
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d((2,1), (2,1))
        self.drop  = nn.Dropout(dropout)
        self.adapt = nn.AdaptiveAvgPool2d((16,1))
        self.fc1   = nn.Linear(128*16*1, 256)
        self.cls   = nn.Linear(256, n_classes)
        self.leak  = nn.Linear(256, 1)

    def forward(self, mel_db: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mel_db: [B, 1, n_mels, t_frames] (channels_last ok)
        """
        x = F.gelu(self.conv1(mel_db))
        x = self.pool1(F.gelu(self.conv2(x)))
        x = self.pool2(F.gelu(self.conv3(x)))
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.gelu(self.fc1(x)))
        return self.cls(x), self.leak(x).squeeze(1)

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

def load_weights_and_meta(stage_dir: Path, n_classes: int, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Search typical locations under STAGE_DIR/MODELS for weights and meta.
    """
    model_dir = stage_dir / "MODELS"
    candidates = [model_dir/"best.pth", model_dir/"cnn_model_best.h5", model_dir/"cnn_model_full.h5",
                  Path("best.pth"), Path("cnn_model_best.h5"), Path("cnn_model_full.h5")]
    meta = load_model_meta(stage_dir)
    model = LeakCNNMulti(n_classes=n_classes, dropout=0.25).to(device).eval().to(memory_format=torch.channels_last)
    last_err = None
    for ck in candidates:
        if not ck.exists():
            continue
        try:
            obj = torch.load(str(ck), map_location=device, weights_only=False)
            sd  = _coerce_state_dict(obj)
            model.load_state_dict(sd, strict=False)
            print(f"[MODEL] loaded {ck}")
            return model, meta
        except Exception as e:
            print(f"[WARN] could not load {ck}: {e}")
            last_err = e
    if last_err is not None:
        print(f"[WARN] no checkpoint could be loaded; using randomly initialized model.", file=sys.stderr)
    return model, meta

def infer_one_wav(model: nn.Module, mel_db: nn.Module, wav: np.ndarray, cfg: Dict,
                  device: torch.device, batch_segments: int = 2048,
                  prob: str = "softmax") -> Tuple[List[float], float]:
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

    per_long = np.asarray(per_long, dtype=np.float32)
    long_hits = (per_long >= thr).astype(np.float32)
    long_pos_frac = float(long_hits.mean())
    leak_conf_mean = float(per_long.mean())

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
    p = argparse.ArgumentParser(description="Recall-first two-stage leak classifier")
    p.add_argument("--stage-dir", type=str, default="/DEVELOPMENT/DATASET_REFERENCE", help="Dataset stage root")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in-dir", type=str, help="Directory of audio files (recursively scanned)")
    g.add_argument("--filelist", type=str, help="Text file with one audio path per line")
    p.add_argument("--out", type=str, default="leak_report.csv", help="Output CSV")
    p.add_argument("--prob", type=str, default="softmax", choices=["softmax","blend"], help="Probability head")
    p.add_argument("--decision", type=str, default="frac_vote",
                   choices=["mean","long_vote","any_long","frac_vote"], help="Decision rule")
    p.add_argument("--long-frac", type=float, default=0.25, help="For frac_vote: required fraction of long segments >= thr")
    p.add_argument("--min-long", type=int, default=1, help="For frac_vote: require at least N long segments to be positive")
    p.add_argument("--thr", type=float, default=0.35, help="Leak probability threshold (on per-long means)")
    p.add_argument("--batch-segments", type=int, default=2048, help="Short segments per forward micro-batch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    stage_dir = Path(args.stage_dir)
    cfg = load_builder_config(stage_dir)
    device = torch.device(args.device)

    # build mel transform
    mel = build_mel(cfg).to(device).eval()

    # Resolve class order: ensure LEAK is at index 0 for softmax route.
    meta = load_model_meta(stage_dir)
    class_names = meta.get("class_names") or ["LEAK", "NOLEAK"]
    if "LEAK" not in class_names:
        class_names = ["LEAK"] + [c for c in class_names if c != "LEAK"]
    leak_idx = class_names.index("LEAK")
    n_classes = max(2, len(class_names))

    # Instantiate model & try to load weights from STAGE_DIR/MODELS
    model, meta2 = load_weights_and_meta(stage_dir, n_classes=n_classes, device=device)
    meta.update(meta2)

    # If LEAK is not at index 0 in the checkpoint's notion, we will reorder logits during inference.
    # Easiest: assume LEAK is at index 0 here and warn otherwise.
    if leak_idx != 0:
        print(f"[WARN] Expected 'LEAK' at index 0, but it is at {leak_idx}. "
              f"This script will still index 0 as LEAK; ensure your checkpoint uses that order.", file=sys.stderr)

    # Collect files
    if args.in_dir:
        files = list_wavs(Path(args.in_dir))
    else:
        with open(args.filelist, "r") as f:
            files = [Path(line.strip()) for line in f if line.strip()]
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(2)

    total = len(files)
    leak_ct = 0

    # Prepare CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["filepath","is_leak","leak_conf_mean","per_long_probs_json","long_pos_frac","notes"])

        for fp in files:
            try:
                wav, sr = read_audio(fp)
                if sr != int(cfg["sample_rate"]):
                    # resample to cfg SR
                    wav_t, _ = torchaudio.load(str(fp))
                    wav_t = torchaudio.functional.resample(wav_t, sr, int(cfg["sample_rate"]))
                    wav = wav_t.mean(dim=0).numpy().astype(np.float32)
                    sr = int(cfg["sample_rate"])

                # pad/trim to full duration
                target = int(cfg["sample_rate"] * cfg["duration_sec"])
                wav = pad_or_trim(wav, target)

                per_long, leak_mean = infer_one_wav(model, mel, wav, cfg, device,
                                                    batch_segments=args.batch_segments, prob=args.prob)
                is_leak, leak_conf_mean, long_pos_frac, notes = decide_label(
                    per_long, decision=args.decision, thr=float(args.thr),
                    long_frac=float(args.long_frac), min_long=int(args.min_long)
                )

                if is_leak:
                    leak_ct += 1

                w.writerow([str(fp), is_leak, f"{leak_conf_mean:.6f}", json.dumps(per_long), f"{long_pos_frac:.6f}", notes])
            except Exception as e:
                w.writerow([str(fp), 0, f"0.0", "[]", "0.0", f"error:{type(e).__name__}:{e}"])

    print(f"[DONE] wrote {out_path} | leaks={leak_ct}/{total} "
          f"({(leak_ct/total*100.0):.1f}%) using decision={args.decision}, thr={args.thr}, "
          f"long_frac={args.long_frac}, min_long={args.min_long}")
    print(f"Stage dir: {stage_dir} | SR={cfg['sample_rate']}Hz | windows long={cfg['long_window']} short={cfg['short_window']}")

if __name__ == "__main__":
    main()
