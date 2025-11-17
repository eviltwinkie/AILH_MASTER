"""Utility script to validate WAV files in a dataset tree (structure & basic params).

Features:
 - Missing imports/constants fixed (HEADER_SIZE, NUM_SAMPLES, BYTES_PER_SAMPLE).
 - Defensive checks (dataset existence, empty set handling).
 - Deterministic ordering for reproducibility.
 - Lightweight integrity & parameter validation via Python's wave module.

Run by executing the file directly; configuration is via constants at top of file.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import wave
from typing import List

DATASET_PATH = Path("/mnt/d/DATASET_REFERENCE/TRAINING")
SAMPLE_RATE = 4096  # Expected sample rate; set to None to skip SR check
SAMPLE_LENGTH_SEC = 10
HEADER_SIZE = 44  # Standard PCM WAV header size
BYTES_PER_SAMPLE = 2  # int16
NUM_SAMPLES = SAMPLE_RATE * SAMPLE_LENGTH_SEC

# Parallelism configuration
VALIDATION_THREADS = 16  # Set to 1 to disable parallelism
WARN_SAMPLE_LIMIT = 10

 # Throughput/benchmark parameters removed (was not requested by user)


def list_wavs(dataset: Path) -> List[Path]:
    if not dataset.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset}", file=sys.stderr)
        return []
    wavs: List[Path] = []
    # os.walk is generally faster than Path.rglob for very large trees.
    for root, _dirs, files in os.walk(dataset):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(Path(root) / f)
    wavs.sort()  # deterministic order
    return wavs


def validate_wav(path: Path, expected_sr: int | None) -> tuple[bool, str]:
    """Validate one WAV file.

    Checks:
      - Can open & read params
      - (Optional) Sample rate matches expected
      - File size >= header size
    """
    try:
        stat_size = path.stat().st_size
        if stat_size < HEADER_SIZE:
            return False, "too_small_for_header"
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
        if expected_sr is not None and sr != expected_sr:
            return False, f"sample_rate_mismatch({sr})"
        if sampwidth != BYTES_PER_SAMPLE:
            return False, f"sample_width_{sampwidth}"
        if n_channels != 1:
            # Not necessarily an error; flag for awareness.
            return True, f"non_mono({n_channels})"
        # Optional length check if length is fixed
        if NUM_SAMPLES and n_frames not in (NUM_SAMPLES, NUM_SAMPLES - 1, NUM_SAMPLES + 1):
            # allow small +/- drift (rare)
            return True, f"frames={n_frames}"  # soft warn
        return True, "ok"
    except wave.Error as e:
        return False, f"wave_error:{e}"  # corrupt / unsupported
    except Exception as e:  # noqa: BLE001
        return False, f"exception:{type(e).__name__}:{e}"  # catch-all


def run_validation():  # noqa: C901
    wavs = list_wavs(DATASET_PATH)
    total = len(wavs)
    print(f"[INFO] Found {total} WAV files under {DATASET_PATH}")
    if not wavs:
        return 1

    bad = 0
    soft_warn = 0

    def iter_results():
        if VALIDATION_THREADS <= 1 or total < 2:
            for p in wavs:
                yield validate_wav(p, SAMPLE_RATE)
        else:
            with ThreadPoolExecutor(max_workers=VALIDATION_THREADS) as ex:
                for res in ex.map(lambda p: validate_wav(p, SAMPLE_RATE), wavs, chunksize=32):
                    yield res

    for i, (ok, msg) in enumerate(iter_results()):
        path = wavs[i]
        if not ok:
            bad += 1
            print(f"[BAD] {path} -> {msg}")
        elif msg != "ok":
            soft_warn += 1
            if soft_warn <= WARN_SAMPLE_LIMIT:
                print(f"[WARN] {path.name}: {msg}")
        if (i + 1) % 1000 == 0:
            print(f"[PROGRESS] {i+1}/{total}")

    print(f"[SUMMARY] total={total} bad={bad} soft_warn={soft_warn}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_validation())