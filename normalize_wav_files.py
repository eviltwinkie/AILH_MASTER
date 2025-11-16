"""Per-file peak normalization script.

Scans all .wav files under DATASET_PATH and for each file:
  1. Loads audio (any channel count) as float32 in range [-1, 1] (using soundfile).
  2. Computes peak (max absolute sample across channels).
  3. If peak == 0 -> skip (silent/all zero).
  4. If peak >= ALREADY_NORMALIZED_THRESHOLD -> skip (already normalized enough).
  5. Otherwise scales entire waveform so new peak ~= TARGET_PEAK and writes back preserving
     original sample subtype (int16 or float formats handled by soundfile automatically).

Parallelized with ThreadPoolExecutor (MAX_WORKERS).
Outputs a summary of counts plus optional CSV log of actions.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

try:
    import soundfile as sf  # type: ignore
except ImportError:  # Fallback to wave (slower, int16 only)
    sf = None  # type: ignore
import wave
import numpy as np

# ================== CONFIG ==================
DATASET_PATH = Path("/mnt/d/DATASET_REFERENCE/TRAINING")
MAX_WORKERS = 32
TARGET_PEAK = 0.98                  # Desired peak ratio of full scale (for int16 it's 32767)
ALREADY_NORMALIZED_THRESHOLD = 0.90 # Skip if peak >= this
MIN_PEAK_FOR_ACTION = 1e-6          # Treat below as silent/skip
LOG_CSV = DATASET_PATH / "_analysis" / "normalization_actions.csv"
DRY_RUN = False                     # If True, don't overwrite files
SKIP_QUARANTINE_SUBDIR = True       # Skip files already in a quarantine directory

# ================== DATA STRUCTURES ==================
@dataclass
class NormResult:
    path: Path
    status: str   # normalized | skipped_zero | skipped_already | error | skipped_other
    peak_before: float
    peak_after: Optional[float]
    scale: Optional[float]
    message: str = ""


# ================== UTILITIES ==================
def list_wavs(dataset: Path) -> List[Path]:
    if not dataset.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset}", file=sys.stderr)
        return []
    wavs: List[Path] = []
    for root, _dirs, files in os.walk(dataset):
        for f in files:
            if f.lower().endswith('.wav'):
                p = Path(root) / f
                if SKIP_QUARANTINE_SUBDIR and any(part.startswith('_quarantine') for part in p.parts):
                    continue
                wavs.append(p)
    wavs.sort()
    return wavs


def load_audio(path: Path):
    """Return (audio ndarray float32 [-1,1], sample_rate, subtype, original_dtype_info)."""
    if sf is not None:
        data, sr = sf.read(str(path), always_2d=False)  # shape (n,) or (n, ch)
        orig_info = sf.info(str(path))
        # Convert to float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        # Ensure range -1..1 (soundfile already gives that for float; for int it converts)
        return data, sr, orig_info.subtype, orig_info
    # Fallback using wave (int16 only)
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError('Unsupported sample width without soundfile')
        n = wf.getnframes()
        raw = wf.readframes(n)
    arr = np.frombuffer(raw, dtype='<i2')
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1).round().astype('<i2')
    data = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return data, sr, 'PCM_16', {'channels': ch, 'samplerate': sr}


def write_audio(path: Path, data: np.ndarray, sr: int, subtype: str):
    if DRY_RUN:
        return
    if sf is not None:
        sf.write(str(path), data, sr, subtype=subtype)
    else:
        # Write int16 via wave
        int16 = np.clip((data * 32767.0).round(), -32768, 32767).astype('<i2')
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1 if data.ndim == 1 else data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int16.tobytes())


def peak_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.max(np.abs(x)))
    return float(np.max(np.abs(x), axis=0).max())


def process_file(path: Path) -> NormResult:
    try:
        data, sr, subtype, _info = load_audio(path)
        peak_before = peak_abs(data)
        if peak_before <= MIN_PEAK_FOR_ACTION:
            return NormResult(path, 'skipped_zero', peak_before, None, None, 'silent_or_zero')
        if peak_before >= ALREADY_NORMALIZED_THRESHOLD:
            return NormResult(path, 'skipped_already', peak_before, peak_before, 1.0, 'already_normalized')
        scale = TARGET_PEAK / peak_before
        data_scaled = np.clip(data * scale, -1.0, 1.0)
        peak_after = peak_abs(data_scaled)
        write_audio(path, data_scaled, sr, subtype)
        return NormResult(path, 'normalized', peak_before, peak_after, scale, '')
    except Exception as e:  # noqa: BLE001
        return NormResult(path, 'error', 0.0, None, None, f'{type(e).__name__}:{e}')


def write_log(results: List[NormResult]):
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(LOG_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path','status','peak_before','peak_after','scale','message'])
        for r in results:
            w.writerow([str(r.path), r.status, f'{r.peak_before:.6f}',
                        '' if r.peak_after is None else f'{r.peak_after:.6f}',
                        '' if r.scale is None else f'{r.scale:.6f}', r.message])
    print(f'[INFO] Wrote log: {LOG_CSV}')


def summarize(results: List[NormResult]):
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print('\n=== SUMMARY ===')
    for k in sorted(counts):
        print(f'{k}: {counts[k]}')
    print(f'total: {len(results)}')


def main() -> int:
    wavs = list_wavs(DATASET_PATH)
    print(f'[INFO] Found {len(wavs)} wav files for normalization')
    if not wavs:
        return 0
    results: List[NormResult] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(process_file, p): p for p in wavs}
        for i, fut in enumerate(as_completed(fut_map), 1):
            res = fut.result()
            results.append(res)
            if i % 1000 == 0 or i == len(wavs):
                print(f'[PROGRESS] {i}/{len(wavs)}')
    summarize(results)
    write_log(results)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())