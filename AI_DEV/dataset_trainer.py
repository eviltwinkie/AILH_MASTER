#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Trainer - PyTorch Leak Detection Model Training

Production training script for two-stage temporal segmentation leak detection models.
Implements leak-focused training with auxiliary heads, class weighting, and paper-exact
file-level evaluation metrics.

Key Features:
    - Leak-optimized training (weighted CE + auxiliary leak-vs-rest head)
    - Paper-exact file-level evaluation (50% voting threshold)
    - Step-accurate resume with StatefulSampler
    - GPU optimization (AMP, TF32, channels_last, torch.compile)
    - Checkpoint rotation with configurable keep_last_k
    - Live GPU monitoring during evaluation
    - Early stopping based on file-level leak F1

Architecture:
    - LeakCNNMulti: Dual-head model (multiclass + binary leak detection)
    - Pool keeps time dimension (2,1) for temporal granularity
    - AdaptiveAvgPool2d for variable-length inputs

Training Configuration:
    - Loss: Weighted CrossEntropy + BCE auxiliary leak head
    - Optimizer: AdamW with CosineAnnealingLR
    - Batch size: 5120 (training), 2048 (validation)
    - Mixed precision: FP16 with GradScaler
    - Gradient clipping: 1.0

Evaluation Metrics:
    - Segment-level: CrossEntropy loss, top-1 accuracy
    - File-level (paper-exact): Per-file voting with 50% threshold
    - Leak metrics: F1, Precision, Recall with threshold sweep

Usage:
    Edit Config dataclass paths, then run: python dataset_trainer.py

Note:
    Supports both 2-class (LEAK/NOLEAK) and 5-class configurations.
    Current default: 2 classes as per research paper.
"""
# =============================================================================
# leak_dataset_trainer_v15.py
# Focused on LEAK performance:
#  - Class-weighted CE (or focal) + auxiliary leak-vs-rest head (BCE)
#  - PR sweep on validation to pick best leak threshold
#  - Step-accurate resume (StatefulSampler), AMP, TF32, channels_last
#  - Checkpoints store best_leak_thr; model_meta.json written alongside
#  - Optional SpecAugment; optional LEAK oversampling (kept off by default)
# =============================================================================

from __future__ import annotations

import os, json, signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator

import h5py
import numpy as np
try:
    import pynvml
    _HAS_NVML=True
except Exception:
    _HAS_NVML=False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

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
import psutil
import threading, time, psutil, sys

# --- perf knobs ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------------- CONFIG --------------------------------------

@dataclass
class Config:
    # Data / paths
    stage_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE")
    hdf5_name: str = "TRAINING_DATASET.H5"       # point at TRAINING by default
    val_hdf5_name: str = "VALIDATION_DATASET.H5" # separate validation set
    test_hdf5_name: str = "TESTING_DATASET.H5"   # separate test set
    model_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/MODELS")
    log_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/LOGS")

    # Training
    batch_size: int = 5120
    val_batch_size: int = 2048
    epochs: int = 200
    learning_rate: float = 1e-3
    dropout: float = 0.25
    num_classes: int = 2
    grad_clip_norm: Optional[float] = 1.0
    early_stop_patience: int = 15

    # Emphasis on LEAK
    leak_class_name: str = "LEAK"
    # Loss choice: "ce", "weighted_ce", "focal"
    loss_type: str = "weighted_ce"
    focal_gamma: float = 2.0
    focal_alpha_leak: float = 0.75           # α for leak; others get (1-α)/(C-1)

    # Auxiliary leak-vs-rest head
    use_leak_aux_head: bool = True
    leak_aux_weight: float = 0.5             # λ in total loss = CE/Focal + λ*BCE
    leak_aux_pos_weight: Optional[float] = None  # if None, auto from label freq

    # Optional oversampling (segment-level index expansion)
    leak_oversample_factor: int = 1          # >1 duplicates LEAK segments in sampler

    # Dataloader
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    pin_memory_device: str = "cuda"

    # Augment
    use_specaugment: bool = False
    time_mask_param: int = 6
    freq_mask_param: int = 8

    # Compile / memory format
    use_compile: bool = True
    compile_mode: Optional[str] = "reduce-overhead"
    use_channels_last: bool = True

    # Resume / checkpoints
    auto_resume: bool = True
    keep_last_k: int = 3

    seed: Optional[int] = 1234

    @property
    def train_h5(self) -> Path:
        return self.stage_dir / self.hdf5_name

    @property
    def val_h5(self) -> Path:
        return self.stage_dir / self.val_hdf5_name

    @property
    def test_h5(self) -> Path:
        return self.stage_dir / self.test_hdf5_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.model_dir / "checkpoints"


# ------------------------------- UTIL ----------------------------------------

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    return torch.device("cuda")

def get_rng_state_safe():
    t_cpu = torch.get_rng_state()
    t_cuda = torch.cuda.get_rng_state_all()
    algo, keys, pos, has_gauss, cached = np.random.get_state()
    np_state = (str(algo), [int(x) for x in keys.tolist()], int(pos), int(has_gauss), float(cached))
    return {"torch_cpu": t_cpu, "torch_cuda": t_cuda, "numpy": np_state}

def set_rng_state_safe(state):
    if not state: return
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    algo, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((algo, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))

def rotate_checkpoints(ckpt_dir: Path, keep_last_k: int):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
    if len(ckpts) > keep_last_k:
        for p in ckpts[:-keep_last_k]:
            try: p.unlink()
            except Exception: pass

# ------------------------------- DATASET -------------------------------------

class LeakMelDataset(Dataset):
    """Reads /segments_mel and /labels; returns per-segment mel + label."""
    def __init__(self, h5_path: Path, cache_files: int = 256):
        self.h5_path = str(h5_path)
        self.h5f = None
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._cache_files = cache_files
        with h5py.File(self.h5_path, "r") as f:
            segs = f["segments_mel"]; shp = tuple(segs.shape)
            if len(shp) == 5:
                self.num_files, self.num_long, self.num_short, self.n_mels, self.t_frames = shp
                self._has_channel = False
            elif len(shp) == 6:
                self.num_files, self.num_long, self.num_short, _, self.n_mels, self.t_frames = shp
                self._has_channel = True
            else:
                raise RuntimeError(f"Unsupported segments_mel shape: {shp}")
            self._dtype = segs.dtype
            self._class_names = None
            try:
                lbls = f.attrs.get("labels_json")
                if isinstance(lbls, (bytes, bytearray)): lbls = lbls.decode("utf-8")
                if lbls: self._class_names = json.loads(lbls)
            except Exception: pass
            self.builder_cfg = None
            try:
                cj = f.attrs.get("config_json")
                if isinstance(cj, (bytes, bytearray)): cj = cj.decode("utf-8")
                if cj: self.builder_cfg = json.loads(cj)
            except Exception: pass
        self.total_segments = self.num_files * self.num_long * self.num_short

    @property
    def class_names(self) -> List[str] | None:
        return self._class_names

    def _ensure_open(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._segs = self.h5f["segments_mel"]
            self._labels = self.h5f["labels"]

    def _get_file_block(self, file_idx: int) -> np.ndarray:
        blk = self._cache.get(file_idx)
        if blk is not None: return blk
        blk = self._segs[file_idx]  # numpy view
        self._cache[file_idx] = blk
        self._cache_order.append(file_idx)
        if len(self._cache_order) > self._cache_files:
            old = self._cache_order.pop(0); self._cache.pop(old, None)
        return blk

    def __len__(self): return self.total_segments

    def __getitem__(self, index: int):
        self._ensure_open()
        LxS = self.num_long * self.num_short
        fidx = index // LxS
        rem  = index %  LxS
        li   = rem // self.num_short
        si   = rem %  self.num_short
        blk = self._get_file_block(fidx)  # [num_long,num_short,(C,)n_mels,t_frames]
        mel = blk[li, si]
        if self._has_channel: mel = mel[0]
        mel_t = torch.from_numpy(mel)  # zero-copy
        lbl_t = torch.tensor(int(self._labels[fidx]), dtype=torch.long)
        return mel_t, lbl_t

    def __del__(self):
        try:
            if self.h5f is not None: self.h5f.close()
        except Exception: pass


# --------------------------- Sampler (resumeable) -----------------------------

class StatefulSampler(Sampler[int]):
    """Keeps epoch/position/permutation; resume exact step."""
    def __init__(self, indices: List[int], shuffle: bool = True, seed: Optional[int] = None):
        self.indices = list(indices)
        self.shuffle = bool(shuffle)
        self.seed = int(seed or 0)
        self.epoch = 1
        self.pos = 0
        self._perm = np.arange(len(self.indices), dtype=np.int64)
        if self.shuffle: self._regen_perm()

    def __len__(self) -> int: return len(self.indices) - self.pos

    def _regen_perm(self):
        rng = np.random.default_rng(self.seed ^ self.epoch)
        self._perm = rng.permutation(len(self.indices)).astype(np.int64, copy=False)
        self.pos = 0

    def on_epoch_start(self, epoch: int):
        if epoch != self.epoch and self.shuffle:
            self.epoch = int(epoch); self._regen_perm()
        else:
            self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        N = len(self.indices)
        p = self._perm if self.shuffle else np.arange(N, dtype=np.int64)
        for i in range(self.pos, N):
            idx = self.indices[int(p[i])]
            self.pos = i + 1
            yield idx

    def state_dict(self) -> Dict:
        return {
            "epoch": int(self.epoch),
            "pos": int(self.pos),
            "perm": self._perm.astype(np.int64).tolist(),
            "seed": int(self.seed),
            "shuffle": bool(self.shuffle),
            "indices": list(self.indices),
        }

    def load_state_dict(self, state: Dict):
        self.epoch = int(state["epoch"]); self.pos = int(state["pos"])
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.indices = list(state.get("indices", self.indices))
        perm = state.get("perm")
        self._perm = np.asarray(perm, dtype=np.int64) if perm is not None else np.arange(len(self.indices))


# ------------------------------- MODEL ---------------------------------------

class LeakCNNMulti(nn.Module):
    """
    Backbone -> (multiclass logits, leak_logit)
    Pool keeps time (2,1) to preserve temporal granularity.
    """
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
        self.leak_head = nn.Linear(256, 1)  # auxiliary leak vs rest

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = self.pool2(x)
        x = self.adapt(x); x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.cls_head(x)
        leak_logit = self.leak_head(x).squeeze(1)
        return logits, leak_logit


# ----------------------------- LOSSES / WEIGHTS -------------------------------

def class_counts_from_labels(ds: LeakMelDataset) -> np.ndarray:
    # Count per-file labels then expand by segments per file
    with h5py.File(ds.h5_path, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.int64)
    counts_files = np.bincount(labels, minlength=ds.class_names and len(ds.class_names) or 0)
    # Each file contributes num_long * num_short segments
    segs_per_file = ds.num_long * ds.num_short
    counts_segments = counts_files * segs_per_file
    return counts_segments

class FocalLoss(nn.Module):
    def __init__(self, class_alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", class_alpha)  # shape [C]
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # CE per sample
        ce = F.cross_entropy(logits, targets, reduction="none")
        # p_t
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1,1)).squeeze(1).clamp(1e-8, 1-1e-8)
        alpha_t = self.alpha[targets]
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# ------------------------------- TRAIN / EVAL ---------------------------------

def build_indices(ds: LeakMelDataset, leak_idx: int, oversample_factor: int, val_subset: bool = False) -> List[int]:
    """
    Build segment-level indices; optional leak oversampling (duplicates).
    """
    N = len(ds)
    idx = np.arange(N, dtype=np.int64)
    if oversample_factor <= 1:
        return idx.tolist()
    # expand leak indices
    LxS = ds.num_long * ds.num_short
    with h5py.File(ds.h5_path, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.int64)
    leak_file_mask = (labels == leak_idx)
    leak_file_ids = np.nonzero(leak_file_mask)[0]
    leak_seg_ranges = [np.arange(fid*LxS, (fid+1)*LxS, dtype=np.int64) for fid in leak_file_ids]
    leak_seg_ids = np.concatenate(leak_seg_ranges) if leak_seg_ranges else np.empty(0, dtype=np.int64)
    expanded = [idx]
    for _ in range(oversample_factor - 1):
        expanded.append(leak_seg_ids.copy())
    return np.concatenate(expanded).tolist()


def eval_split(model: nn.Module,
               loader: DataLoader,
               device: torch.device,
               num_classes: int,
               leak_idx: int,
               use_channels_last: bool,
               aux_leak: bool,
               tb_collect: bool = False) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    leak_scores = []
    leak_targets = []
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for mel_batch, labels in loader:
            mel_batch = mel_batch.unsqueeze(1)
            if use_channels_last:
                mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, leak_logit = model(mel_batch)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            # collect leak scores
            rel = (labels == leak_idx).to(torch.float32)
            leak_targets.append(rel.detach().cpu().numpy())
            leak_scores.append(torch.sigmoid(leak_logit).detach().cpu().numpy())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    leak_targets = np.concatenate(leak_targets) if leak_scores else np.array([])
    leak_scores  = np.concatenate(leak_scores) if leak_scores else np.array([])
    out = {"loss": avg_loss, "acc": acc}
    if leak_scores.size:
        # sweep thresholds to get best F1 for leak
        best_f1, best_p, best_r, best_thr = -1, 0, 0, 0.5
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (leak_scores >= thr).astype(np.int32)
            tp = int(((preds == 1) & (leak_targets == 1)).sum())
            fp = int(((preds == 1) & (leak_targets == 0)).sum())
            fn = int(((preds == 0) & (leak_targets == 1)).sum())
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2*p*r / max(p + r, 1e-12)
            if f1 > best_f1:
                best_f1, best_p, best_r, best_thr = f1, p, r, float(thr)
        out.update({"leak_f1": best_f1, "leak_p": best_p, "leak_r": best_r, "leak_thr": best_thr})
    return out



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
    cfg = Config()
    model.eval()
    if getattr(ds, "h5f", None) is None:
        ds._ensure_open()
    class_names = ds.class_names or [f"C{i}" for i in range(cfg.num_classes)]
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


def train():
    cfg = Config()
    set_seed(cfg.seed)
    device = device_setup()

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    ds_tr = LeakMelDataset(cfg.train_h5)
    ds_va = LeakMelDataset(cfg.val_h5)
    class_names = ds_tr.class_names or [f"C{i}" for i in range(cfg.num_classes)]
    try:
        leak_idx = class_names.index(cfg.leak_class_name)
    except ValueError:
        leak_idx = 2  # fallback

    # Build train indices (optionally oversample LEAK)
    tr_indices = build_indices(ds_tr, leak_idx, cfg.leak_oversample_factor)
    va_subset = Subset(ds_va, list(range(len(ds_va))))

    # Sampler & loaders
    train_sampler = StatefulSampler(tr_indices, shuffle=True, seed=cfg.seed)
    train_loader = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        pin_memory_device=cfg.pin_memory_device,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=False,
    )
    val_loader = DataLoader(
        va_subset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        pin_memory_device=cfg.pin_memory_device,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=max(2, cfg.prefetch_factor - 1),
        drop_last=False,
    )

    # Model
    model = LeakCNNMulti(n_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=(cfg.compile_mode or "default"))
        except Exception:
            pass

    # Losses
    class_weights_t: Optional[torch.Tensor] = None
    if cfg.loss_type in ("weighted_ce", "focal"):
        counts = class_counts_from_labels(ds_tr)
        counts = np.maximum(counts, 1)
        inv = (counts.sum() / counts).astype(np.float64)
        inv = inv / inv.mean()
        class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)

    if cfg.loss_type == "ce":
        cls_loss_fn = nn.CrossEntropyLoss()
    elif cfg.loss_type == "weighted_ce":
        cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        C = cfg.num_classes
        alpha = torch.ones(C, device=device, dtype=torch.float32) * ((1.0 - cfg.focal_alpha_leak) / max(C-1, 1))
        alpha[leak_idx] = cfg.focal_alpha_leak
        cls_loss_fn = FocalLoss(alpha, gamma=cfg.focal_gamma)

    # Aux leak head loss
    if cfg.use_leak_aux_head:
        with h5py.File(ds_tr.h5_path, "r") as f:
            labels = np.asarray(f["labels"][:], dtype=np.int64)
        pos_files = (labels == leak_idx).sum()
        neg_files = max(len(labels) - pos_files, 1)
        segs_per_file = ds_tr.num_long * ds_tr.num_short
        pos = pos_files * segs_per_file; neg = neg_files * segs_per_file
        pos_w = torch.tensor([neg / max(pos, 1)], device=device)
        leak_bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        leak_bce = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 1
    last_ck = cfg.checkpoints_dir / 'last.pth'
    if cfg.auto_resume and last_ck.exists():
        try:
            
            ck = torch.load(last_ck, map_location=device, weights_only=False)
            ck_model = ck.get('model', {})
            def strip_prefix(k: str) -> str:
                if k.startswith('module.'): k = k[7:]
                if k.startswith('_orig_mod.'): k = k[10:]
                return k
            ck_model = {strip_prefix(k): v for k, v in ck_model.items()}
            new_state = {}
            mismatched = []
            cur_sd = model.state_dict()
            for k, v in ck_model.items():
                if k in ('cls_head.weight', 'cls_head.bias'):
                    mismatched.append(k); continue
                if k in cur_sd and cur_sd[k].shape == v.shape:
                    new_state[k] = v
                else:
                    mismatched.append(k)
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"[RESUME] Loaded backbone with {len(new_state)} tensors; skipped {len(mismatched)}")

        except Exception as e:
            print(f"[RESUME] Failed to load {last_ck}: {e}")

    # Optional SpecAugment
    use_ta = False
    try:
        import torchaudio
        time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask_param) if cfg.use_specaugment else None
        freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask_param) if cfg.use_specaugment else None
        use_ta = True
    except Exception:
        time_mask = None; freq_mask = None

    best_val_file_acc = -1.0
    best_epoch = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.on_epoch_start(epoch)
        model.train()
        running_loss = 0.0; correct = 0; seen = 0

        pbar = tqdm(total=len(train_loader), desc=f"[Train] Epoch {epoch}/{cfg.epochs}", unit="batch")
        for mel_batch, labels in train_loader:
            mel_batch = mel_batch.unsqueeze(1)
            if cfg.use_channels_last:
                mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if cfg.use_specaugment and use_ta:
                with torch.no_grad():
                    if time_mask: mel_batch = time_mask(mel_batch)
                    if freq_mask: mel_batch = freq_mask(mel_batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, leak_logit = model(mel_batch)
                loss_cls = cls_loss_fn(logits, labels)
                if leak_bce is not None:
                    leak_target = (labels == leak_idx).to(torch.float32)
                    loss = loss_cls + cfg.leak_aux_weight * leak_bce(leak_logit, leak_target)
                else:
                    loss = loss_cls

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer); scaler.update()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            seen += bs
            pbar.update(1)
        pbar.close()

        train_loss = running_loss / max(seen, 1)
        train_acc  = correct / max(seen, 1)

        # Segment-level monitoring
        metrics = eval_split(model, val_loader, device, cfg.num_classes, leak_idx, cfg.use_channels_last, cfg.use_leak_aux_head)
        val_loss = metrics["loss"]; val_acc = metrics["acc"]

        # Paper-exact validation file-level accuracy
        val_file_acc = evaluate_file_level(model, ds_va, device, threshold=0.5)

        print(f"Epoch {epoch:03d} │ train_loss={format(train_loss,'.4f')} │ train_acc={format(train_acc,'.4f')} │ "
              f"val_loss={format(val_loss,'.4f')} │ val_acc={format(val_acc,'.4f')} │ file_acc(paper)={format(val_file_acc,'.4f')}")

        scheduler.step()

        # Save 'last' checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
            "class_names": class_names,
        }
        ckpt["sampler"] = train_sampler.state_dict()
        torch.save(ckpt, cfg.checkpoints_dir / "last.pth")
        torch.save(ckpt, cfg.checkpoints_dir / f"epoch_{epoch:03d}.pth")
        rotate_checkpoints(cfg.checkpoints_dir, keep_last_k=cfg.keep_last_k)

        # Track best by paper metric
        if val_file_acc > best_val_file_acc:
            best_val_file_acc = val_file_acc; best_epoch = epoch
            torch.save(model.state_dict(), cfg.model_dir / "best.pth")
            torch.save({"model": model.state_dict()}, cfg.model_dir / "cnn_model_best.h5")
            meta = {
                "class_names": class_names,
                "leak_class_name": cfg.leak_class_name,
                "builder_cfg": ds_tr.builder_cfg,
                "best_val_file_acc": float(best_val_file_acc),
                "best_epoch": int(best_epoch),
                "leak_threshold": 0.5,
                "model_type": "LeakCNNMulti"
            }
            with open(cfg.model_dir / "model_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    print(f"Training complete. Best file_acc(paper)={best_val_file_acc:.4f} @ epoch {best_epoch}.")

    # ---------------- Test using paper voting ----------------
    te_path = cfg.test_h5
    if te_path.exists():
        ds_te = LeakMelDataset(te_path)
        state = torch.load(cfg.model_dir / "best.pth", map_location=device, weights_only=False)
        model.load_state_dict(state)
        test_file_acc = evaluate_file_level(model, ds_te, device, threshold=0.5)
        print(f"[TEST] file-level accuracy (paper voting): {test_file_acc:.4f}")


if __name__ == "__main__":
    train()
