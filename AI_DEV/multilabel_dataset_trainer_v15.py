#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# leak_dataset_trainer_v15.py
# Focused on LEAK performance:
#  - Class-weighted CE (or focal) + auxiliary leak-vs-rest head (BCE)
#  - PR sweep on validation to pick best leak threshold
#  - Step-accurate resume (StatefulSampler), AMP, TF32, channels_last
#  - Checkpoints store best_leak_thr; model_meta.json written alongside
#  - Optional SpecAugment; optional LEAK oversampling (kept off by default)
# 5 LABELS
# =============================================================================

from __future__ import annotations

import os, json, signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

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
    batch_size: int = 5632
    val_batch_size: int = 2048
    epochs: int = 200
    learning_rate: float = 1e-3
    dropout: float = 0.25
    num_classes: int = 5
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
        # inverse-frequency weights
        counts = np.maximum(counts, 1)
        inv = (counts.sum() / counts).astype(np.float64)
        inv = inv / inv.mean()
        class_weights_t = torch.tensor(inv, dtype=torch.float32, device=device)

    if cfg.loss_type == "ce":
        cls_loss_fn = nn.CrossEntropyLoss()
    elif cfg.loss_type == "weighted_ce":
        cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        # focal
        C = cfg.num_classes
        alpha = torch.ones(C, device=device, dtype=torch.float32) * ((1.0 - cfg.focal_alpha_leak) / max(C-1, 1))
        alpha[leak_idx] = cfg.focal_alpha_leak
        cls_loss_fn = FocalLoss(alpha, gamma=cfg.focal_gamma)

    # Aux leak head loss
    bce_pos_weight = None
    if cfg.use_leak_aux_head:
        if cfg.leak_aux_pos_weight is not None:
            bce_pos_weight = torch.tensor([cfg.leak_aux_pos_weight], device=device)
        else:
            # derive from class frequency (files→segments)
            with h5py.File(ds_tr.h5_path, "r") as f:
                labels = np.asarray(f["labels"][:], dtype=np.int64)
            pos = (labels == leak_idx).sum()
            neg = max(len(labels) - pos, 1)
            segs_per_file = ds_tr.num_long * ds_tr.num_short
            pos *= segs_per_file; neg *= segs_per_file
            bce_pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        leak_bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Optional SpecAugment
    use_ta = False
    try:
        import torchaudio
        time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask_param) if cfg.use_specaugment else None
        freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask_param) if cfg.use_specaugment else None
        use_ta = True
    except Exception:
        time_mask = None; freq_mask = None

    start_epoch = 1
    best_leak_f1 = -1.0
    best_leak_thr = 0.55
    best_epoch = 0

    # Resume
    last_ckpt = cfg.checkpoints_dir / "last.pth"
    if cfg.auto_resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 1))
            set_rng_state_safe(ckpt.get("rng_state"))
            sstate = ckpt.get("train_sampler_state")
            if sstate: train_sampler.load_state_dict(sstate)
            best_leak_f1 = float(ckpt.get("best_leak_f1", best_leak_f1))
            best_leak_thr = float(ckpt.get("best_leak_thr", best_leak_thr))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            print(f"[RESUME] epoch={start_epoch} best_leak_f1={best_leak_f1:.4f}@{best_epoch} thr={best_leak_thr}")
        except Exception as e:
            print(f"[RESUME] failed: {e} — starting fresh.")

    interrupted = {"flag": False}
    def _sig(sig, frame):
        interrupted["flag"] = True
        print("\n[CTRL-C] Will save checkpoint at end of epoch…")
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    no_improve = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.on_epoch_start(epoch)
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(total=len(train_loader), desc=f"[Train] Epoch {epoch}/{cfg.epochs}", unit="batch")
        for mel_batch, labels in train_loader:
            if interrupted["flag"]: break
            mel_batch = mel_batch.unsqueeze(1)
            if cfg.use_channels_last:
                mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
            mel_batch = mel_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # (optional) light SpecAugment on mel in-place (train only)
            if cfg.use_specaugment and use_ta:
                with torch.no_grad():
                    if time_mask: mel_batch = time_mask(mel_batch)
                    if freq_mask: mel_batch = freq_mask(mel_batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits, leak_logit = model(mel_batch)
                loss_cls = cls_loss_fn(logits, labels)
                if cfg.use_leak_aux_head:
                    leak_target = (labels == leak_idx).to(torch.float32)
                    loss_leak = leak_bce(leak_logit, leak_target)
                    loss = loss_cls + cfg.leak_aux_weight * loss_leak
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

        # Validation
        metrics = eval_split(model, val_loader, device, cfg.num_classes, leak_idx, cfg.use_channels_last, cfg.use_leak_aux_head)
        val_loss = metrics["loss"]; val_acc = metrics["acc"]
        leak_f1  = metrics.get("leak_f1", -1.0)
        leak_thr = metrics.get("leak_thr", best_leak_thr)
        print(f"Epoch {epoch:03d} │ train_loss={train_loss:.4f} │ train_acc={train_acc:.4f} │ "
              f"val_loss={val_loss:.4f} │ val_acc={val_acc:.4f} │ "
              f"leak_f1={leak_f1:.4f}@thr={leak_thr:.2f}")

        scheduler.step()

        # Save 'last' checkpoint each epoch
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": get_rng_state_safe(),
            "train_sampler_state": train_sampler.state_dict(),
            "best_leak_f1": best_leak_f1,
            "best_leak_thr": best_leak_thr,
            "best_epoch": best_epoch,
            "config": asdict(cfg),
            "class_names": class_names,
            "leak_idx": leak_idx,
        }
        torch.save(ckpt, cfg.checkpoints_dir / "last.pth")

        # Track best by LEAK F1
        improved = leak_f1 > best_leak_f1
        if improved:
            best_leak_f1 = leak_f1
            best_leak_thr = leak_thr
            best_epoch = epoch
            # Save best weights & meta JSON
            torch.save(model.state_dict(), cfg.model_dir / "best.pth")
            meta = {
                "class_names": class_names,
                "leak_class_name": cfg.leak_class_name,
                "leak_idx": leak_idx,
                "best_leak_threshold": best_leak_thr,
                "best_leak_f1": best_leak_f1,
                "best_epoch": best_epoch,
                "builder_cfg": ds_tr.builder_cfg,
                "trainer_cfg": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool))},
            }
            with open(cfg.model_dir / "model_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"[EarlyStop] No LEAK F1 improvement for {cfg.early_stop_patience} epochs. "
                      f"Best leak_f1={best_leak_f1:.4f} at epoch {best_epoch}.")
                break

    print(f"Training complete. Best leak_f1={best_leak_f1:.4f} @ epoch {best_epoch}. Best thr={best_leak_thr:.2f}")

    # ---------------- Test on TESTING_DATASET.H5 ----------------
    if (cfg.stage_dir / cfg.test_h5.name).exists():
        ds_te = LeakMelDataset(cfg.test_h5)
        te_loader = DataLoader(
            ds_te, batch_size=cfg.val_batch_size, shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=cfg.pin_memory, pin_memory_device=cfg.pin_memory_device,
            persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
            prefetch_factor=max(2, cfg.prefetch_factor - 1),
            drop_last=False,
        )
        # load best weights
        state = torch.load(cfg.model_dir / "best.pth", map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        # standard metrics
        metrics = eval_split(model, te_loader, device, cfg.num_classes, leak_idx, cfg.use_channels_last, cfg.use_leak_aux_head)
        test_loss = metrics["loss"]; test_acc = metrics["acc"]
        # Evaluate leak with chosen threshold
        # Re-run collecting raw leak scores
        leak_scores, leak_targets = [], []
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            for mel_batch, labels in te_loader:
                mel_batch = mel_batch.unsqueeze(1)
                if cfg.use_channels_last:
                    mel_batch = mel_batch.contiguous(memory_format=torch.channels_last)
                mel_batch = mel_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, leak_logit = model(mel_batch)
                leak_scores.append(torch.sigmoid(leak_logit).detach().cpu().numpy())
                leak_targets.append((labels == leak_idx).float().cpu().numpy())
        leak_scores = np.concatenate(leak_scores); leak_targets = np.concatenate(leak_targets)
        thr = best_leak_thr
        preds = (leak_scores >= thr).astype(np.int32)
        tp = int(((preds == 1) & (leak_targets == 1)).sum())
        fp = int(((preds == 1) & (leak_targets == 0)).sum())
        fn = int(((preds == 0) & (leak_targets == 1)).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        F1 = 2*P*R / max(P + R, 1e-12)
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} | leak@thr={thr:.2f} → F1={F1:.4f} (P={P:.4f}, R={R:.4f})")

if __name__ == "__main__":
    train()
