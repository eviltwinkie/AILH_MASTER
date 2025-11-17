#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leak_dataset_learner.py (NO-CLI)
• Continual learning on WAVs under STAGE_DIR/LEARNING (same label folder structure).
• Reads builder config + class names from MODEL_DIR/model_meta.json to guarantee
  segmentation/Mel alignment with the original build.
• Uses existing VALIDATION_DATASET.H5 for evaluation and leak-threshold sweep.
• Starts from MODEL_DIR/best.pth and updates weights; overwrites best.pth + model_meta.json.
"""

from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========================= USER VARIABLES (EDIT HERE) =========================
STAGE_DIR   = Path("/DEVELOPMENT/DATASET_REFERENCE")
LEARN_DIR   = STAGE_DIR / "LEARNING"
VAL_H5      = STAGE_DIR / "VALIDATION_DATASET.H5"
MODEL_DIR   = STAGE_DIR / "MODELS"

EPOCHS      = 30
BATCH_SEGMENTS = 8192
VAL_BATCH_SIZE = 2048
LEARNING_RATE   = 5e-4
DROPOUT         = 0.25
SEED            = 1234
NUM_WORKERS     = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
PIN_MEMORY      = True
USE_CHANNELS_LAST = True
USE_COMPILE       = True
COMPILE_MODE      = "reduce-overhead"
GRAD_CLIP_NORM    = 1.0
# =============================== END VARIABLES ================================

def set_seed(seed: Optional[int]):
    if seed is None: return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    return torch.device("cuda")

def load_meta():
    with open(MODEL_DIR / "model_meta.json", "r") as f:
        return json.load(f)

def load_wav_mono(path: Path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1: wav = data[:, 0]
    elif data.shape[1] == 2: wav = (data[:, 0] + data[:, 1]) * 0.5
    else: wav = data.mean(axis=1, dtype=np.float32)
    return wav.astype(np.float32, copy=False), sr

def resample_if_needed(wav: np.ndarray, sr: int, target_sr: int):
    if sr == target_sr: return wav, sr
    wav_t = torch.from_numpy(wav.astype(np.float32, copy=False))
    if wav_t.ndim == 1: wav_t = wav_t.unsqueeze(0)
    res = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav_t)
    res = res.squeeze(0).contiguous().numpy()
    return res.astype(np.float32, copy=False), target_sr

def segment_wave(wav: np.ndarray, sample_rate: int, duration_sec: int, long_window: int, short_window: int):
    num_samples = sample_rate * duration_sec
    if wav.size > num_samples: wav = wav[:num_samples]
    elif wav.size < num_samples:
        out = np.zeros(num_samples, dtype=np.float32); out[:wav.size] = wav; wav = out
    num_long = num_samples // long_window; num_short = long_window // short_window
    segs = wav.reshape(num_long, long_window).reshape(num_long, num_short, short_window).reshape(-1, short_window)
    return segs.astype(np.float32, copy=False), int(num_long), int(num_short)

class LearningWavDataset(Dataset):
    """Streams segments from WAVs under LEARN_DIR; labels from subfolder names."""
    def __init__(self, learn_dir: Path, bcfg: Dict, class_names: List[str]):
        self.learn_dir = learn_dir
        self.sample_rate = int(bcfg.get("sample_rate", 8192))
        self.duration_sec = int(bcfg.get("duration_sec", 5))
        self.long_window = int(bcfg.get("long_window", 2048))
        self.short_window = int(bcfg.get("short_window", 512))
        self.class_names = class_names
        # index files
        pairs: List[Tuple[Path, int]] = []
        for root, _, files in os.walk(learn_dir):
            label = os.path.basename(root)
            if label == os.path.basename(str(learn_dir)):  # skip root
                continue
            if label not in class_names:  # ignore unknown labels
                continue
            y = class_names.index(label)
            for f in files:
                if f.lower().endswith(".wav"):
                    pairs.append((Path(root)/f, y))
        self.pairs = pairs
        # Precompute segment counts per file
        self.num_long = self.duration_sec * self.sample_rate // self.long_window
        self.num_short = self.long_window // self.short_window
        self.segs_per_file = self.num_long * self.num_short
        self.total_segments = len(self.pairs) * self.segs_per_file

        # Mel
        self.n_fft = int(bcfg.get("n_fft", 512))
        self.hop_length = int(bcfg.get("hop_length", 128))
        self.n_mels = int(bcfg.get("n_mels", 64))
        self.power = float(bcfg.get("power", 1.0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, power=self.power, center=False
        ).eval().to(self.device)

    def __len__(self): return self.total_segments

    def __getitem__(self, index: int):
        file_idx = index // self.segs_per_file
        seg_idx  = index %  self.segs_per_file
        path, y = self.pairs[file_idx]
        wav, sr = load_wav_mono(path)
        wav, _  = resample_if_needed(wav, sr, self.sample_rate)
        segs, _, _ = segment_wave(wav, self.sample_rate, self.duration_sec, self.long_window, self.short_window)
        seg = segs[seg_idx]  # [short_window]
        x = torch.from_numpy(seg).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            m = self.mel(x).float().clamp_min_(1e-10).log10_()
            m.mul_(20.0 if self.power == 1.0 else 10.0)
        mel_np = m.detach().cpu().numpy()
        return torch.from_numpy(mel_np), torch.tensor(int(y), dtype=torch.long)

class LeakCNNMulti(nn.Module):
    def __init__(self, n_classes: int = 5, dropout: float = 0.25):
        super().__init__()
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
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = self.adapt(x); x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.cls_head(x), self.leak_head(x).squeeze(1)

class LeakMelDataset(Dataset):
    """Validation loader from existing VAL_H5 (reused from trainer)."""
    def __init__(self, h5_path: Path):
        self.h5_path = str(h5_path); self.h5f = None
        with h5py.File(self.h5_path, "r") as f:
            segs = f["segments_mel"]; shp = tuple(segs.shape)
            if len(shp) == 5:
                self.num_files, self.num_long, self.num_short, self.n_mels, self.t_frames = shp; self._has_channel=False
            elif len(shp) == 6:
                self.num_files, self.num_long, self.num_short, _, self.n_mels, self.t_frames = shp; self._has_channel=True
            else:
                raise RuntimeError(f"Unsupported segments_mel shape: {shp}")
        self.total_segments = self.num_files * self.num_long * self.num_short
    def _ensure(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._segs = self.h5f["segments_mel"]; self._labels = self.h5f["labels"]
    def __len__(self): return self.total_segments
    def __getitem__(self, index: int):
        self._ensure()
        LxS = self.num_long * self.num_short
        fidx = index // LxS; rem = index % LxS; li = rem // self.num_short; si = rem % self.num_short
        blk = self._segs[fidx]; mel = blk[li, si]; 
        if self._has_channel: mel = mel[0]
        return torch.from_numpy(mel), torch.tensor(int(self._labels[fidx]), dtype=torch.long)

def eval_split(model: nn.Module, loader: DataLoader, device: torch.device, leak_idx: int, use_channels_last: bool):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total, correct = 0.0, 0, 0
    leak_scores = []; leak_targets = []
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for mel_batch, labels in loader:
            mel_batch = mel_batch.unsqueeze(1)
            if use_channels_last: mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            logits, leak_logit = model(mel_batch)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            total += labels.size(0)
            preds = logits.argmax(dim=1); correct += int((preds == labels).sum().item())
            rel = (labels == leak_idx).to(torch.float32)
            leak_targets.append(rel.detach().cpu().numpy())
            leak_scores.append(torch.sigmoid(leak_logit).detach().cpu().numpy())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    leak_targets = np.concatenate(leak_targets) if leak_scores else np.array([])
    leak_scores  = np.concatenate(leak_scores) if leak_scores else np.array([])
    out = {"loss": avg_loss, "acc": acc}
    if leak_scores.size:
        best_f1, best_p, best_r, best_thr = -1, 0, 0, 0.5
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (leak_scores >= thr).astype(np.int32)
            tp = int(((preds == 1) & (leak_targets == 1)).sum())
            fp = int(((preds == 1) & (leak_targets == 0)).sum())
            fn = int(((preds == 0) & (leak_targets == 1)).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            f1 = 2*p*r / max(p + r, 1e-12)
            if f1 > best_f1: best_f1, best_p, best_r, best_thr = f1, p, r, float(thr)
        out.update({"leak_f1": best_f1, "leak_p": best_p, "leak_r": best_r, "leak_thr": best_thr})
    return out

def learn():
    set_seed(SEED)
    device = device_setup()
    meta = load_meta()
    class_names: List[str] = meta["class_names"]
    leak_idx = int(meta.get("leak_idx", class_names.index("LEAK") if "LEAK" in class_names else 0))
    bcfg = meta.get("builder_cfg", {}) or {}

    # Datasets
    ds_learn = LearningWavDataset(LEARN_DIR, bcfg, class_names)
    ds_val   = LeakMelDataset(VAL_H5)

    if len(ds_learn) == 0:
        print("[LEARN] No segments found under LEARNING. Nothing to do."); return

    train_loader = DataLoader(
        ds_learn, batch_size=BATCH_SEGMENTS, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS>0),
        prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS>0 else None),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=max(1, NUM_WORKERS//2), pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS>0),
        prefetch_factor=max(2, PREFETCH_FACTOR-1) if NUM_WORKERS>0 else None,
        drop_last=False,
    )

    # Model
    model = LeakCNNMulti(n_classes=len(class_names), dropout=DROPOUT).to(device)
    if USE_CHANNELS_LAST: model = model.to(memory_format=torch.channels_last)
    if USE_COMPILE and hasattr(torch, "compile"):
        try: model = torch.compile(model, mode=COMPILE_MODE)
        except Exception: pass

    state = torch.load(MODEL_DIR / "best.pth", map_location=device)
    model.load_state_dict(state)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    cls_loss_fn = nn.CrossEntropyLoss()

    best_leak_f1, best_leak_thr, best_epoch = -1.0, float(meta.get("best_leak_threshold", 0.55)), -1

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        pbar = tqdm(total=len(train_loader), desc=f"[Learn] {epoch}/{EPOCHS}", unit="batch")
        for mel_batch, labels in train_loader:
            mel_batch = mel_batch.unsqueeze(1)
            if USE_CHANNELS_LAST: mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits, leak_logit = model(mel_batch)
                loss = cls_loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            preds = logits.argmax(dim=1); correct += int((preds == labels).sum().item()); seen += bs
            pbar.update(1)
        pbar.close()
        sch.step()

        # Evaluate on validation
        metrics = eval_split(model, val_loader, device, leak_idx, USE_CHANNELS_LAST)
        leak_f1  = metrics.get("leak_f1", -1.0); leak_thr = metrics.get("leak_thr", best_leak_thr)
        if leak_f1 > best_leak_f1:
            best_leak_f1, best_leak_thr, best_epoch = leak_f1, leak_thr, epoch
            torch.save(model.state_dict(), MODEL_DIR / "best.pth")
            # update meta in-place
            meta["best_leak_threshold"] = float(best_leak_thr)
            with open(MODEL_DIR / "model_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        print(f"[VAL] epoch={epoch:03d} leak_f1={leak_f1:.4f}@{leak_thr:.2f} (best {best_leak_f1:.4f}@{best_leak_thr:.2f})")

    print(f"[Done] best_learn_f1={best_leak_f1:.4f} @ thr={best_leak_thr:.2f} (epoch {best_epoch})")

if __name__ == "__main__":
    learn()
