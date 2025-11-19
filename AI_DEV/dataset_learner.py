#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Continual Learning Module

Implements continual/incremental learning for PyTorch leak detection models.
Processes new WAV files from LEARNING directory and updates existing model weights
while preserving previously learned knowledge.

Key Features:
    - Continual learning from LEARNING/ directory WAV files
    - On-the-fly WAV resampling and mel spectrogram extraction
    - Reads builder config from model metadata for consistency
    - Uses existing VALIDATION_DATASET.H5 for evaluation
    - Leak threshold sweep for optimal F1 score
    - Checkpoint management with auto-resume capability

Workflow:
    1. Load model metadata and builder configuration
    2. Stream and process WAV files from LEARNING/ directory
    3. Perform incremental training epochs (default: 30)
    4. Evaluate on validation set with leak threshold optimization
    5. Update best.pth and model_meta.json when F1 improves

Directory Structure:
    STAGE_DIR/LEARNING/           - New WAV files organized by label (LEAK, NOLEAK)
    STAGE_DIR/VALIDATION_DATASET.H5 - Validation set for evaluation
    MODEL_DIR/best.pth            - Model weights (updated in-place)
    MODEL_DIR/model_meta.json     - Model metadata (updated with new threshold)

Note:
    Only supports LEAK and NOLEAK labels. Other labels are ignored.
"""
leak_dataset_learner.py (NO-CLI)
• Continual learning on WAVs under STAGE_DIR/LEARNING (same label folder structure).
• Reads builder config + class names from MODEL_DIR/model_meta.json to guarantee
  segmentation/Mel alignment with the original build.
• Uses existing VALIDATION_DATASET.H5 for evaluation and leak-threshold sweep.
• Starts from MODEL_DIR/best.pth and updates weights; overwrites best.pth + model_meta.json.
"""
from __future__ import annotations
import sys
import os, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pynvml
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ALLOWED_LABELS = ("LEAK", "NOLEAK")

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

CHECKPOINTS_DIR  = MODEL_DIR / "checkpoints"
AUTO_RESUME     = True
KEEP_LAST_K     = 3
COMPILE_MODE      = "reduce-overhead"
GRAD_CLIP_NORM    = 1.0
# =============================== END VARIABLES ================================

class EvalStatus:
    def __init__(self, total_files:int):
        self.total = total_files
        self.done = 0
        self.correct = 0
        self.pred_leaks = 0
        self._stop = threading.Event()
        self._th = None
        self._nvml = None
        if globals().get("_HAS_NVML", False):
            try:
                pynvml.nvmlInit()
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml = None

    def update(self, done:int=None, correct:int=None, pred_leaks:int=None):
        if done is not None: self.done = done
        if correct is not None: self.correct = correct
        if pred_leaks is not None: self.pred_leaks = pred_leaks

    def gpu_line(self):
        if self._nvml is None: return "GPU N/A"
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml)
            vram = mem.used/(1024**3)
            return f"GPU {util}% | VRAM {vram:.2f} GB"
        except Exception:
            return "GPU N/A"

    def run(self):
        pass
        #print(f"[Paper Eval] Starting status thread. Total files: {self.total}", file=sys.stdout)
        # while not self._stop.wait(2.0):
        #     acc = (self.correct / max(1,self.done))
        #     leak_frac = (self.pred_leaks / max(1,self.done))*100.0
        #     cpu = psutil.cpu_percent(interval=None)
        #     ram = psutil.virtual_memory().percent
        #     tqdm.write(f"[Paper Eval] {self.done}/{self.total} files | acc@file {acc:.4f} | leak% {leak_frac:.1f} | CPU {cpu:.1f}% | RAM {ram:.1f}% | {self.gpu_line()}", file=sys.stdout)

    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self.run, daemon=True)
            self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            try: self._th.join(timeout=2.0)
            except Exception: pass
        if self._nvml is not None:
            try: pynvml.nvmlShutdown()
            except Exception: pass

import threading, time, psutil, sys

def evaluate_file_level(model: nn.Module,
                        ds: LeakMelDataset,
                        device: torch.device,
                        threshold: float = 0.5) -> float:
    """
    Paper-exact file-level metric with live status:
      • For each FILE, for each LONG segment, average per-SHORT leak probabilities where
        per-SHORT = 0.5*softmax(leak_class) + 0.5*sigmoid(aux_leak).
      • A FILE is LEAK iff ≥ 50% of LONG segments have prob ≥ threshold (0.5).
      • Returns overall file-level accuracy.
    Status:
      • tqdm progress bar (position=0, dynamic_ncols, mininterval=0.2)
      • Heartbeat every ~2s with CPU/RAM and GPU util/VRAM.
    """
    model.eval()
    if getattr(ds, "h5f", None) is None:
        ds._ensure_open()
    class_names = ds.class_names or [f"C{i}" for i in range(2)]
    leak_idx = class_names.index("LEAK") if "LEAK" in class_names else 0
    correct, total, pred_leaks = 0, 0, 0

    status = EvalStatus(ds.num_files); status.start()
    from tqdm import tqdm
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        pbar = tqdm(range(ds.num_files), desc="[Paper Eval] files", unit="file", leave=True, position=0, dynamic_ncols=True, mininterval=0.2, file=sys.stdout)
        for fidx in pbar:
            blk = ds._segs[fidx]   # [num_long,num_short,(C,)F,T]
            lbl = int(ds._labels[fidx])
            num_long, num_short = ds.num_long, ds.num_short
            probs_long = []
            for li in range(num_long):
                mel = blk[li]
                if getattr(ds, "_has_channel", False):
                    mel = mel[:,0]
                mel_t = torch.from_numpy(mel).unsqueeze(1).to(device, dtype=torch.float16, non_blocking=True)
                logits, leak_logit = model(mel_t)
                p_cls = torch.softmax(logits, dim=1)[:, leak_idx]
                p_aux = torch.sigmoid(leak_logit)
                p = 0.5*(p_cls + p_aux)
                probs_long.append(float(p.mean().item()))
            num_pos = sum(1 for q in probs_long if q >= threshold)
            is_leak = (num_pos >= max(1, int((0.5*ds.num_long + 1e-9))))
            pred_is_leak = 1 if is_leak else 0
            true_is_leak = 1 if lbl == leak_idx else 0
            if pred_is_leak == true_is_leak:
                correct += 1
            if is_leak:
                pred_leaks += 1
            total += 1
            status.update(done=total, correct=correct, pred_leaks=pred_leaks)
            pbar.set_postfix({
                "acc@file": f"{(correct/max(total,1)):.4f}",
                "leak%": f"{(100.0*pred_leaks/max(total,1)):.1f}",
                "files": f"{total}/{ds.num_files}"
            })
            pbar.refresh()
    status.stop()
    return correct / max(total, 1)

def set_seed(seed: Optional[int]):
    if seed is None: return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    return torch.device("cuda")


def rotate_checkpoints(ckpt_dir: Path, keep_last_k: int = 3):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = sorted(ckpt_dir.glob("epoch_*.pth"), key=lambda q: q.stat().st_mtime, reverse=True)
        for q in files[keep_last_k:]:
            try: q.unlink()
            except Exception: pass
    except Exception:
        pass

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
            if label not in ALLOWED_LABELS:
                continue
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
    def __init__(self, n_classes: int = 2, dropout: float = 0.25):
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
    total_loss = 0.0
    correct = 0
    seen = 0
    cls_loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for mel_batch, labels in loader:
            mel_batch = mel_batch.unsqueeze(1).to(device, non_blocking=True)
            if use_channels_last:
                mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            logits, leak_logit = model(mel_batch)
            loss = cls_loss_fn(logits, labels)
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            seen += bs
    return {"loss": total_loss / max(1, seen), "acc": correct / max(1, seen)}



def learn():
    set_seed(SEED)
    device = device_setup()
    meta = load_meta()
    class_names: List[str] = meta["class_names"]
    leak_idx = int(meta.get("leak_idx", class_names.index("LEAK") if "LEAK" in class_names else 0))
    bcfg = meta.get("builder_cfg", {}) or {}

    # Datasets
    ds_learn = LearnDataset(STAGE_DIR / "LEARNING", class_names, bcfg)
    ds_val = MelH5Dataset(H5_VAL)
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

    # ---------- Auto-resume ----------
    start_epoch = 1
    last_ck = CHECKPOINTS_DIR / "last.pth"
    if AUTO_RESUME and last_ck.exists():
        try:
            ck = torch.load(last_ck, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sch.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck.get("epoch", 1))
            print(f"[RESUME] Loaded {last_ck} → epoch {start_epoch}")
        except Exception as e:
            print(f"[RESUME] Failed to load {last_ck}: {e}")

    best_file_acc, best_epoch = -1.0, -1

    for epoch in range(start_epoch, EPOCHS+1):
        # Train
        model.train()
        running_loss = 0.0; correct = 0; seen = 0
        pbar = tqdm(total=len(train_loader), desc=f"[Learn] Epoch {epoch}/{EPOCHS}", unit="batch")
        for mel_batch, labels in train_loader:
            mel_batch = mel_batch.unsqueeze(1)
            if USE_CHANNELS_LAST:
                mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
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
        val_loss = metrics.get("loss", 0.0); val_acc = metrics.get("acc", 0.0)
        file_acc = evaluate_file_level(model, ds_val, device, class_names, threshold=0.5)
        print(f"[VAL] loss={val_loss:.4f} acc={val_acc:.4f} | file_acc(paper)={file_acc:.4f}")

        # Save rolling/numbered checkpoints
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "scaler": scaler.state_dict(),
            "meta": meta,
        }
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, CHECKPOINTS_DIR / "last.pth")
        torch.save(ckpt, CHECKPOINTS_DIR / f"epoch_{epoch:03d}.pth")
        rotate_checkpoints(CHECKPOINTS_DIR, keep_last_k=KEEP_LAST_K)

        # Track best by paper metric
        if file_acc > best_file_acc:
            best_file_acc, best_epoch = file_acc, epoch
            torch.save(model.state_dict(), MODEL_DIR / "best.pth")
            meta["leak_threshold"] = 0.5
            with open(MODEL_DIR / "model_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    print(f"[Done] best_file_acc(paper)={best_file_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    learn()