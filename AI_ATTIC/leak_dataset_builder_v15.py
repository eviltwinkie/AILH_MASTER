#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# # leak_dataset_builder_v15.py
# Multi-split leak dataset builder:
# - Builds TRAINING_DATASET.H5, VALIDATION_DATASET.H5, TESTING_DATASET.H5
# - WAVs are read from label subfolders under TRAINING / VALIDATION / TESTING dirs
# - One global label mapping (consistent ids across splits)
# - In-RAM HDF5 (h5py core driver) → single sequential flush to ext4 on close
# - Streaming prefetch + CPU segmentation → GPU Mel (triple-buffered, micro-batched)
# =============================================================================

from __future__ import annotations

import os
import gc
import json
import time
import signal
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from tqdm import tqdm

# Optional GPU profiling
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

CYAN, GREEN, YELLOW, RED, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


# ============================== CONFIG =======================================

@dataclass
class Config:
    # Roots
    stage_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE")
    training_dir: Path = Path("DEVELOPMENT\ROOT_AILH\DATA_STORE/TRAINING")
    validation_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/VALIDATION")
    testing_dir: Path = Path("/DEVELOPMENT/DATASET_REFERENCE/TESTING")

    # Output HDF5 names
    training_hdf5: str = "TRAINING_DATASET.H5"
    validation_hdf5: str = "VALIDATION_DATASET.H5"
    testing_hdf5: str = "TESTING_DATASET.H5"

    # Signal params
    sample_rate: int = 4096
    duration_sec: int = 10

    long_window: int = 1024
    short_window: int = 512

    # ---- CNN config (requested) ----
    CNN_MODEL_TYPE: str = "mel"
    CNN_BATCH_SIZE: int = 5632
    CNN_LEARNING_RATE: float = 0.001
    CNN_DROPOUT: float = 0.25
    CNN_EPOCHS: int = 200
    CNN_FILTERS: int = 32
    CNN_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    CNN_POOL_SIZE: Tuple[int, int] = (2, 2)
    CNN_STRIDES: Tuple[int, int] = (2, 2)

    # Mel params (aligned with trainer)
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 128
    power: float = 1.0
    center: bool = False

    # Disk I/O (tuned from NVMe benchmarks: Thr=4, FPT=1024, Depth=16)
    cpu_max_workers: int = 4
    disk_files_per_task: int = 1024
    disk_max_inflight: int = 16
    disk_submit_window: int = 16

    # GPU mega-batching
    autosize_target_util_frac: float = 0.80
    files_per_gpu_batch: int = 512
    seg_microbatch_segments: int = 8192
    max_files_per_gpu_batch: int = 4096
    num_mega_buffers: int = 3  # triple-buffering

    # Misc
    track_times: bool = False
    warn_ram_fraction: float = 0.70

    @property
    def num_samples(self) -> int:
        return self.sample_rate * self.duration_sec

    @property
    def num_long(self) -> int:
        return self.num_samples // self.long_window

    @property
    def num_short(self) -> int:
        return self.long_window // self.short_window

    @property
    def segments_per_file(self) -> int:
        return self.num_long * self.num_short

    @property
    def db_mult(self) -> float:
        return 10.0 if self.power == 2.0 else 20.0


# ============================== UTILITIES ====================================

def bytes_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return dev


def build_mel_transform(cfg: Config, device: torch.device) -> torch.nn.Module:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        center=cfg.center,
    ).eval().to(device)


def estimate_image_bytes(cfg: Config, n_mels: int, t_frames: int, fp16_mel: bool = True) -> int:
    wave = cfg.segments_per_file * cfg.short_window * 4  # f32
    melb = cfg.segments_per_file * n_mels * t_frames * (2 if fp16_mel else 4)
    labels = 2
    return wave + melb + labels


def load_wav_mono_fast(path: str) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    ch = data.shape[1]
    if ch == 1:
        wav = data[:, 0]
    elif ch == 2:
        wav = (data[:, 0] + data[:, 1]) * 0.5
    else:
        wav = data.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def prefetch_wavs(batch: List[Tuple[int, Path, int]]) -> List[Tuple[int, Path, np.ndarray, int, int]]:
    out: List[Tuple[int, Path, np.ndarray, int, int]] = []
    for idx, path, label_id in batch:
        try:
            data, sr = load_wav_mono_fast(str(path))
            out.append((idx, path, data, sr, label_id))
        except Exception as e:
            print(f"{YELLOW}[SKIP-READ]{RESET} {path} → {e}")
    return out


def autosize_gpu_batch(cfg: Config, free_vram_gb: float, t_frames: int,
                       target_util_frac: float = 0.70, num_buffers: int = 3,
                       max_bsz: int = 131072) -> int:
    safety_overhead_frac = 0.1
    free_bytes = int(free_vram_gb * (1024 ** 3))
    usable = int(free_bytes * target_util_frac * (1.0 - safety_overhead_frac))
    if usable <= 0:
        return 1

    bytes_per_segment_in = cfg.short_window * 4
    bytes_per_segment_out = cfg.n_mels * t_frames * 2
    per_file_bytes = cfg.segments_per_file * (bytes_per_segment_in + bytes_per_segment_out)
    if per_file_bytes <= 0:
        return 1

    max_files = usable // (num_buffers * per_file_bytes)
    for cand in [131072, 65536, 32768, 16384, 8192, 4096, 3072, 2048, 1536, 1024, 768, 512]:
        if max_files >= cand:
            return min(cand, max_bsz)
    return max(1, min(max_bsz, int(max_files)))


# ============================== BUILDER ======================================

class MultiSplitBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = ensure_cuda()
        self.mel_transform = build_mel_transform(cfg, self.device)

        # profiling
        self.stop_evt = threading.Event()
        self.profile_stats: List[Tuple[float, float, float, float, float]] = []
        self.nvml_handle = None

        # per-split timings
        self.cpu_times: List[float] = []
        self.gpu_times: List[float] = []
        self.file_indices: List[int] = []

    # --------- profiling ---------
    def _profile_worker(self):
        while not self.stop_evt.wait(0.5):
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            if self.nvml_handle is not None:
                try:
                    gpu = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
                    vram = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024 ** 3)
                except Exception:
                    gpu, vram = None, None
            else:
                gpu, vram = None, None
            self.profile_stats.append((time.time(), cpu, ram, gpu, vram))

    def _start_profiling(self):
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml_handle = None
        t = threading.Thread(target=self._profile_worker, daemon=True)
        t.start()
        self.prof_thread = t
        print("Profiling Started...")

    def _stop_profiling(self):
        self.stop_evt.set()
        try:
            self.prof_thread.join(timeout=2)
        except Exception:
            pass
        if _HAS_NVML and (self.nvml_handle is not None):
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # --------- discovery helpers ---------
    @staticmethod
    def _discover_records(root_dir: Path) -> List[Tuple[Path, str]]:
        records: List[Tuple[Path, str]] = []
        if not root_dir.exists():
            return records
        for root, _, files in os.walk(root_dir):
            if os.path.abspath(root) == os.path.abspath(root_dir):
                continue
            label = os.path.basename(root)
            for f in files:
                if f.lower().endswith(".wav"):
                    records.append((Path(root) / f, label))
        return records

    def discover_global_labels(self, dirs: List[Path]) -> Tuple[List[Tuple[Path, str]], Dict[str, int]]:
        all_records: List[Tuple[Path, str]] = []
        label_set = set()
        for d in dirs:
            recs = self._discover_records(d)
            all_records.extend(recs)
            label_set.update(lbl for _p, lbl in recs)
        labels_sorted = sorted(label_set)
        label2id = {l: i for i, l in enumerate(labels_sorted)}
        return all_records, label2id

    # --------- single split build ---------
    def build_split(self, split_name: str, input_dir: Path, out_path: Path,
                    global_label2id: Dict[str, int], global_label_list: List[str]):
        cfg = self.cfg
        print(f"\n{CYAN}=== Building split: {split_name} ==={RESET}")
        records = self._discover_records(input_dir)
        print(f"Discovered {len(records)} WAV files in {input_dir}.")

        if len(records) == 0:
            print(f"{YELLOW}[SKIP]{RESET} No WAV files under {input_dir}.")
            return

        # reset per-split timings
        self.cpu_times = []
        self.gpu_times = []
        self.file_indices = []

        # map labels using global mapping
        labels_np = np.array([global_label2id[lbl] for _, lbl in records], dtype=np.int16)
        num_files = len(records)

        # HDF5 in RAM
        with h5py.File(str(out_path), "w", libver="latest", driver="core", backing_store=True) as h5f:
            d_labels = h5f.create_dataset("labels", (num_files,), dtype=np.int16, track_times=cfg.track_times)
            d_labels[:] = labels_np

            d_wave = h5f.create_dataset(
                "segments_waveform",
                (num_files, cfg.num_long, cfg.num_short, cfg.short_window),
                dtype=np.float32,
                track_times=cfg.track_times,
            )

            # probe mel shape
            dummy = torch.randn(cfg.short_window, device=self.device)
            m = self.mel_transform(dummy)
            n_mels, t_frames = m.shape
            del dummy, m

            d_mel = h5f.create_dataset(
                "segments_mel",
                (num_files, cfg.num_long, cfg.num_short, n_mels, t_frames),
                dtype=np.float16,
                track_times=cfg.track_times,
            )

            # metadata (include BOTH builder cfg and CNN cfg for downstream trainer)
            h5f.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            h5f.attrs["config_json"] = json.dumps({
                "sample_rate": cfg.sample_rate, "duration_sec": cfg.duration_sec,
                "long_window": cfg.long_window, "short_window": cfg.short_window,
                "n_mels": cfg.n_mels, "n_fft": cfg.n_fft, "hop_length": cfg.hop_length, "power": cfg.power
            })
            # Persist your CNN training configuration too:
            h5f.attrs["cnn_config_json"] = json.dumps({
                "model_type": cfg.CNN_MODEL_TYPE,
                "batch_size": cfg.CNN_BATCH_SIZE,
                "learning_rate": cfg.CNN_LEARNING_RATE,
                "dropout": cfg.CNN_DROPOUT,
                "epochs": cfg.CNN_EPOCHS,
                "filters": cfg.CNN_FILTERS,
                "kernel_size": list(cfg.CNN_KERNEL_SIZE),
                "pool_size": list(cfg.CNN_POOL_SIZE),
                "strides": list(cfg.CNN_STRIDES),
            })
            h5f.attrs["labels_json"] = json.dumps(global_label_list)
            h5f.attrs["label2id_json"] = json.dumps(global_label2id)

            # RAM warning
            per_file_bytes = estimate_image_bytes(cfg, n_mels, t_frames, fp16_mel=True)
            total_bytes = per_file_bytes * num_files
            avail = psutil.virtual_memory().available
            if total_bytes > cfg.warn_ram_fraction * avail:
                print(f"{YELLOW}[WARN]{RESET} HDF5 image ~{bytes_human(total_bytes)}; "
                      f"available ~{bytes_human(avail)}.")

            print(f"{GREEN}HDF5 datasets created in RAM. Starting processing...{RESET}")

            # ----- Prefetch + CPU segment -----
            indexed = [(i, p, global_label2id[lbl]) for i, (p, lbl) in enumerate(records)]
            batches = [indexed[i:i + cfg.disk_files_per_task] for i in range(0, num_files, cfg.disk_files_per_task)]
            prefetch_bar = tqdm(total=num_files, desc=f"{CYAN}[{split_name} Prefetch+Process]{RESET}", unit="file")

            data_full: Optional[np.ndarray] = None
            self.file_indices = []

            inflight_sem = threading.Semaphore(cfg.disk_max_inflight if cfg.disk_max_inflight > 0 else len(batches))

            def _prefetch_task(batch):
                with inflight_sem:
                    return prefetch_wavs(batch)

            def _safe_window(threads: int, inflight: int, submit_window: int, nbatches: int) -> int:
                very_large = 10**9
                cap_threads = max(1, threads)
                cap_inflight = inflight if inflight and inflight > 0 else very_large
                cap_submit = submit_window if submit_window and submit_window > 0 else very_large
                cap_batches = max(1, nbatches)
                return max(1, min(cap_threads, cap_inflight, cap_submit, cap_batches))

            window = _safe_window(cfg.cpu_max_workers, cfg.disk_max_inflight, cfg.disk_submit_window, len(batches))

            with ThreadPoolExecutor(max_workers=cfg.cpu_max_workers) as pe:
                it = iter(batches)
                live = set()

                # prime
                for _ in range(window):
                    b = next(it, None)
                    if b is None:
                        break
                    live.add(pe.submit(_prefetch_task, b))

                while live:
                    done, live = wait(live, return_when=FIRST_COMPLETED)
                    for fut in done:
                        res_list = fut.result()
                        for (file_idx, p, data, sr, _label_id) in res_list:
                            t0 = perf_counter()
                            try:
                                if sr != cfg.sample_rate:
                                    raise RuntimeError(f"{p} sr={sr}, expected={cfg.sample_rate}")

                                if not isinstance(data, np.ndarray) or data.dtype != np.float32:
                                    data = np.array(data, dtype=np.float32, copy=False)
                                data = np.ascontiguousarray(data)

                                # pad/trim via reusable buffer
                                target = cfg.num_samples
                                if data.size != target:
                                    if data_full is None or data_full.size != target:
                                        data_full = np.empty(target, dtype=np.float32)
                                    L = min(data.size, target)
                                    data_full[:L] = data[:L]
                                    if L < target:
                                        data_full[L:] = 0.0
                                    view = data_full
                                else:
                                    view = data

                                segs3d = view.reshape(cfg.num_long, cfg.num_short, cfg.short_window)
                                d_wave[file_idx, :, :, :] = segs3d
                                self.file_indices.append(file_idx)
                            except Exception as e:
                                print(f"{YELLOW}[SKIP]{RESET} {p} → {e}")
                            self.cpu_times.append(perf_counter() - t0)
                        prefetch_bar.update(len(res_list))

                        nxt = next(it, None)
                        if nxt is not None:
                            live.add(pe.submit(_prefetch_task, nxt))

            prefetch_bar.close()
            del indexed, batches
            if data_full is not None:
                del data_full
            gc.collect()

            # ----- GPU Mel (triple buffer + micro-pipeline) -----
            files_per_gpu_batch = cfg.files_per_gpu_batch
            if _HAS_NVML and (self.nvml_handle is not None):
                try:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    free_gb = meminfo.free / (1024 ** 3)
                    suggested = autosize_gpu_batch(
                        cfg, free_gb, t_frames,
                        cfg.autosize_target_util_frac,
                        num_buffers=cfg.num_mega_buffers,
                        max_bsz=cfg.max_files_per_gpu_batch
                    )
                    if suggested != files_per_gpu_batch:
                        print(f"{YELLOW}Autosized FILES_PER_GPU_BATCH → {suggested} (was {files_per_gpu_batch}), Buffers → {cfg.num_mega_buffers}{RESET}")
                    files_per_gpu_batch = max(1, suggested)
                except Exception:
                    pass

            def _alloc_buffers(files_per_batch: int):
                host_wave = np.empty((files_per_batch, cfg.num_long, cfg.num_short, cfg.short_window), dtype=np.float32)
                seg_cpu_pinned = torch.empty(
                    (files_per_batch * cfg.segments_per_file, cfg.short_window),
                    dtype=torch.float32, device="cpu", pin_memory=True
                )
                seg_dev = torch.empty(
                    (files_per_batch * cfg.segments_per_file, cfg.short_window),
                    dtype=torch.float32, device=self.device
                )
                mel_cpu_buf = torch.empty(
                    (files_per_batch * cfg.segments_per_file, n_mels, t_frames),
                    dtype=torch.float16, device="cpu", pin_memory=True
                )
                mel_cpu_np = mel_cpu_buf.numpy()
                s_copy = torch.cuda.Stream()
                s_comp = torch.cuda.Stream()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                return dict(
                    host_wave=host_wave,
                    seg_cpu_pinned=seg_cpu_pinned,
                    seg_dev=seg_dev,
                    mel_cpu_buf=mel_cpu_buf,
                    mel_cpu_np=mel_cpu_np,
                    s_copy=s_copy,
                    s_comp=s_comp,
                    start_evt=start_evt,
                    end_evt=end_evt,
                    files_per_batch=files_per_batch,
                )

            def launch_batch(batch_files: List[int], buf: Dict):
                Bfiles = len(batch_files)
                if Bfiles == 0:
                    return None
                for j, fidx in enumerate(batch_files):
                    d_wave.read_direct(buf["host_wave"][j], np.s_[fidx, :, :, :])
                B = Bfiles * cfg.segments_per_file
                arr_view = buf["host_wave"][:Bfiles].reshape(B, cfg.short_window)
                buf["seg_cpu_pinned"][:B].copy_(torch.from_numpy(arr_view), non_blocking=False)

                s_copy, s_comp = buf["s_copy"], buf["s_comp"]
                MB = max(1, min(cfg.seg_microbatch_segments, B))
                buf["start_evt"].record(s_copy)

                for off in range(0, B, MB):
                    end = min(off + MB, B)
                    h2d_done = torch.cuda.Event()
                    with torch.cuda.stream(s_copy):
                        buf["seg_dev"][off:end].copy_(buf["seg_cpu_pinned"][off:end], non_blocking=True)
                        h2d_done.record(s_copy)

                    comp_done = torch.cuda.Event()
                    with torch.cuda.stream(s_comp):
                        s_comp.wait_event(h2d_done)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            m = self.mel_transform(buf["seg_dev"][off:end])
                        m = m.float().clamp_min_(1e-10).log10_().mul_(cfg.db_mult).to(torch.float16)
                        comp_done.record(s_comp)

                    with torch.cuda.stream(s_copy):
                        s_copy.wait_event(comp_done)
                        buf["mel_cpu_buf"][off:end].copy_(m, non_blocking=True)

                buf["end_evt"].record(s_copy)
                return dict(files=batch_files, buf=buf)

            def finish_batch(ctx: Dict | None):
                if ctx is None:
                    return 0
                files = ctx["files"]; buf = ctx["buf"]
                s_copy, s_comp = buf["s_copy"], buf["s_comp"]
                s_copy.synchronize(); s_comp.synchronize()
                self.gpu_times.append(buf["start_evt"].elapsed_time(buf["end_evt"]) / 1000.0)
                for j, fidx in enumerate(files):
                    start = j * cfg.segments_per_file
                    stop = start + cfg.segments_per_file
                    mnp = buf["mel_cpu_np"][start:stop].reshape(cfg.num_long, cfg.num_short, n_mels, t_frames)
                    d_mel[fidx, :, :, :, :] = mnp
                return len(files)

            # allocate ring buffers
            buffers = [_alloc_buffers(files_per_gpu_batch) for _ in range(cfg.num_mega_buffers)]
            mel_bar = tqdm(total=len(self.file_indices), desc=f"{CYAN}[{split_name} Mel MegaBatch]{RESET}", unit="file")

            work = list(sorted(self.file_indices))
            i = 0
            free_buf_ids = list(range(cfg.num_mega_buffers))
            inflight: List[Tuple[Dict, int]] = []

            while i < len(work) or inflight:
                while free_buf_ids and i < len(work):
                    bsz = files_per_gpu_batch
                    batch = work[i: i + bsz]
                    if not batch:
                        break
                    buf_id = free_buf_ids.pop(0)
                    buf = buffers[buf_id]
                    try:
                        ctx = launch_batch(batch, buf)
                        inflight.append((ctx, buf_id))
                        i += len(batch)
                    except torch.cuda.OutOfMemoryError:
                        # finish in-flight then shrink
                        for _ in range(len(inflight)):
                            ctx_k, bid_k = inflight.pop(0)
                            processed = finish_batch(ctx_k)
                            mel_bar.update(processed)
                            free_buf_ids.append(bid_k)
                        new_bsz = max(128, files_per_gpu_batch // 2)
                        # free & reallocate
                        for b in buffers: del b
                        gc.collect(); torch.cuda.empty_cache()
                        buffers = [_alloc_buffers(new_bsz) for _ in range(cfg.num_mega_buffers)]
                        files_per_gpu_batch = new_bsz
                        print(f"{YELLOW}[OOM-backoff]{RESET} Shrinking FILES_PER_GPU_BATCH → {new_bsz}")
                        free_buf_ids = list(range(cfg.num_mega_buffers))
                        inflight = []
                        break

                if not inflight:
                    continue
                ctx0, bid0 = inflight.pop(0)
                processed = finish_batch(ctx0)
                mel_bar.update(processed)
                free_buf_ids.append(bid0)

            mel_bar.close()
            for b in buffers: del b
            gc.collect(); torch.cuda.empty_cache()

            print(f"{YELLOW}Writing HDF5 to disk (single sequential flush on close)...{RESET}")

        print(f"{GREEN}{split_name} HDF5 written: {out_path}{RESET}")

        # per-split summaries
        if self.cpu_times:
            print(f"{YELLOW}{split_name} CPU segmentation avg: {np.mean(self.cpu_times):.4f}s, "
                  f"max: {np.max(self.cpu_times):.4f}s, min: {np.min(self.cpu_times):.4f}s{RESET}")
        if self.gpu_times:
            print(f"{CYAN}{split_name} GPU mel extraction avg: {np.mean(self.gpu_times):.4f}s, "
                  f"max: {np.max(self.gpu_times):.4f}s, min: {np.min(self.gpu_times):.4f}s{RESET}")
        if self.gpu_times and self.file_indices:
            total_gpu_s = sum(self.gpu_times)
            total_files = len(self.file_indices)
            files_per_s = total_files / total_gpu_s
            segs_per_s = (total_files * self.cfg.segments_per_file) / total_gpu_s
            print(f"{GREEN}{split_name} GPU throughput: {files_per_s:,.0f} files/s, "
                  f"{segs_per_s:,.0f} segments/s{RESET}")

    # --------- orchestrate all splits ---------
    def build_all(self):
        cfg = self.cfg
        cfg.stage_dir.mkdir(parents=True, exist_ok=True)

        self._start_profiling()

        # global label mapping across all splits
        _all, global_label2id = self.discover_global_labels(
            [cfg.training_dir, cfg.validation_dir, cfg.testing_dir]
        )
        global_label_list = [l for l, _id in sorted(global_label2id.items(), key=lambda kv: kv[1])]
        print(f"Global labels: {global_label2id}")

        # TRAIN
        self.build_split("TRAINING", cfg.training_dir, cfg.stage_dir / cfg.training_hdf5,
                         global_label2id, global_label_list)
        # VALIDATION
        self.build_split("VALIDATION", cfg.validation_dir, cfg.stage_dir / cfg.validation_hdf5,
                         global_label2id, global_label_list)
        # TESTING
        self.build_split("TESTING", cfg.testing_dir, cfg.stage_dir / cfg.testing_hdf5,
                         global_label2id, global_label_list)

        self._stop_profiling()

        # profiling summary
        if self.profile_stats:
            cpu_vals  = [x[1] for x in self.profile_stats if x[1] is not None]
            ram_vals  = [x[2] for x in self.profile_stats if x[2] is not None]
            gpu_vals  = [x[3] for x in self.profile_stats if x[3] is not None]
            vram_vals = [x[4] for x in self.profile_stats if x[4] is not None]
            print(f"{YELLOW}CPU usage avg: {np.mean(cpu_vals):.1f}%, max: {np.max(cpu_vals):.1f}%, min: {np.min(cpu_vals):.1f}%{RESET}")
            print(f"{CYAN}RAM usage avg: {np.mean(ram_vals):.1f}%, max: {np.max(ram_vals):.1f}%, min: {np.min(ram_vals):.1f}%{RESET}")
            if gpu_vals:
                print(f"{GREEN}GPU usage avg: {np.mean(gpu_vals):.1f}%, max: {np.max(gpu_vals):.1f}%, min: {np.min(gpu_vals):.1f}%{RESET}")
            if vram_vals:
                print(f"{CYAN}VRAM usage avg: {np.mean(vram_vals):.2f} GB, max: {np.max(vram_vals):.2f} GB, min: {np.min(vram_vals):.2f} GB{RESET}")


# ============================== MAIN =========================================

def main():
    cfg = Config()
    builder = MultiSplitBuilder(cfg)

    print(f"[I/O] threads={cfg.cpu_max_workers}, files_per_task={cfg.disk_files_per_task}, "
          f"inflight_depth={cfg.disk_max_inflight}, submit_window={cfg.disk_submit_window}")

    def _graceful_exit(signum, frame):
        print(f"\n{YELLOW}Signal {signum} received; stopping...{RESET}")
        builder.stop_evt.set()
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    builder.build_all()


if __name__ == "__main__":
    main()
