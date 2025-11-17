"""Randomly split WAV files from SOURCES into TRAINING / VALIDATION / TESTING (70/20/10).

Features:
  - Recursively finds all .wav files under SOURCES_ROOT.
  - Shuffles deterministically with SEED for reproducibility.
  - Splits into 70% training, 20% validation, 10% testing (ratio adjustable).
  - Copies (does NOT move) files into destination roots preserving relative subdirectory structure.
  - Skips files that already exist at destination (configurable overwrite).
  - Provides a summary and optional CSV manifest of the split.

Usage: run directly. Adjust constants below as needed.
"""
from __future__ import annotations

import os
import sys
import shutil
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ========= CONFIG =========
SOURCES_ROOT = Path("/mnt/d/DATASET_REFERENCE/SOURCES")
DEST_TRAIN = Path("/mnt/d/DATASET_REFERENCE/TRAINING")
DEST_VALID = Path("/mnt/d/DATASET_REFERENCE/VALIDATION")
DEST_TEST = Path("/mnt/d/DATASET_REFERENCE/TESTING")

RATIOS: Tuple[float, float, float] = (0.7, 0.2, 0.1)  # Train, Val, Test must sum to 1.0
SEED = 1234
OVERWRITE = False                  # If True, overwrite existing destination files
VALIDATE_EXISTING = True           # If dest exists & not overwriting, still hash-validate
HASH_VALIDATE = True               # Enable hashing & validation
COPY_WORKERS = 32                  # Thread count for copy+hash
HASH_CHUNK_SIZE = 1 << 20          # 1 MiB chunk
MAX_HASH_RETRIES = 1               # Additional attempts on mismatch
MANIFEST_CSV = SOURCES_ROOT.parent / "_analysis" / "split_manifest.csv"
ALLOW_EMPTY = False                # If False, abort if no source wavs
INCLUDE_HASHES_IN_MANIFEST = True
PROGRESS_INTERVAL = 1000           # Files per progress update

# ========= DATA STRUCTURES =========
@dataclass
class Entry:
	src: Path
	rel: Path
	split: str  # train|val|test
	dest: Path
	copied: bool = False
	skipped_existing: bool = False
	src_hash: Optional[str] = None
	dest_hash: Optional[str] = None
	validated: Optional[bool] = None
	error: Optional[str] = None


# ========= HELPERS =========
def list_wavs(root: Path) -> List[Path]:
	out: List[Path] = []
	if not root.exists():
		return out
	for r, _d, files in os.walk(root):
		for f in files:
			if f.lower().endswith('.wav'):
				out.append(Path(r) / f)
	out.sort()
	return out


def choose_split(idx: int, n_train: int, n_val: int) -> str:
	if idx < n_train:
		return 'train'
	if idx < n_train + n_val:
		return 'val'
	return 'test'


def ensure_dir(path: Path):
	path.mkdir(parents=True, exist_ok=True)


def hash_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, 'rb') as f:
		while True:
			chunk = f.read(HASH_CHUNK_SIZE)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def copy_and_validate(entry: Entry) -> Entry:
	try:
		dest_exists = entry.dest.exists()
		if dest_exists and not OVERWRITE:
			entry.skipped_existing = True
			if HASH_VALIDATE and VALIDATE_EXISTING:
				entry.src_hash = hash_file(entry.src)
				entry.dest_hash = hash_file(entry.dest)
				entry.validated = (entry.src_hash == entry.dest_hash)
			return entry
		# Need to copy (either overwrite or missing)
		ensure_dir(entry.dest.parent)
		attempt = 0
		while True:
			shutil.copy2(entry.src, entry.dest)
			entry.copied = True
			if not HASH_VALIDATE:
				entry.validated = True
				break
			entry.src_hash = hash_file(entry.src)
			entry.dest_hash = hash_file(entry.dest)
			if entry.src_hash == entry.dest_hash:
				entry.validated = True
				break
			entry.validated = False
			attempt += 1
			if attempt > MAX_HASH_RETRIES:
				break
			# retry copy
		return entry
	except Exception as e:  # noqa: BLE001
		entry.error = f'{type(e).__name__}:{e}'
		entry.validated = False
		return entry


def write_manifest(entries: List[Entry]):
	if not entries:
		return
	MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
	with open(MANIFEST_CSV, 'w', newline='') as f:
		w = csv.writer(f)
		header = ['relative_path','split','dest_path','copied','skipped_existing','validated','error']
		if INCLUDE_HASHES_IN_MANIFEST and HASH_VALIDATE:
			header.extend(['src_hash','dest_hash'])
		w.writerow(header)
		for e in entries:
			row = [str(e.rel), e.split, str(e.dest), int(e.copied), int(e.skipped_existing),
				   '' if e.validated is None else int(e.validated), e.error or '']
			if INCLUDE_HASHES_IN_MANIFEST and HASH_VALIDATE:
				row.extend([e.src_hash or '', e.dest_hash or ''])
			w.writerow(row)
	print(f"[INFO] Wrote manifest: {MANIFEST_CSV} ({len(entries)} rows)")


def validate_ratios(r: Tuple[float, float, float]):
	total = sum(r)
	if abs(total - 1.0) > 1e-6:
		raise SystemExit(f"Ratios must sum to 1.0 (got {total})")
	if any(x < 0 for x in r):
		raise SystemExit("Ratios must be non-negative")


def main() -> int:
	validate_ratios(RATIOS)
	wavs = list_wavs(SOURCES_ROOT)
	n = len(wavs)
	if n == 0 and not ALLOW_EMPTY:
		print(f"[ERROR] No wav files under {SOURCES_ROOT}", file=sys.stderr)
		return 1
	print(f"[INFO] Found {n} source wav files")
	random.seed(SEED)
	random.shuffle(wavs)
	n_train = int(n * RATIOS[0])
	n_val = int(n * RATIOS[1])
	# Ensure all leftover go to test to keep total count
	n_test = n - n_train - n_val
	print(f"[INFO] Split counts -> train:{n_train} val:{n_val} test:{n_test}")

	entries: List[Entry] = []
	for idx, src in enumerate(wavs):
		rel = src.relative_to(SOURCES_ROOT)
		split = choose_split(idx, n_train, n_val)
		dest_root = DEST_TRAIN if split == 'train' else DEST_VALID if split == 'val' else DEST_TEST
		dest = dest_root / rel
		entries.append(Entry(src, rel, split, dest))

	# Multithreaded copy & hash validate
	start = time.time()
	completed = 0
	with ThreadPoolExecutor(max_workers=COPY_WORKERS) as ex:
		future_map = {ex.submit(copy_and_validate, e): e for e in entries}
		for fut in as_completed(future_map):
			_ = fut.result()  # Entry updated in place
			completed += 1
			if completed % PROGRESS_INTERVAL == 0 or completed == n:
				elapsed = time.time() - start
				rate = completed / elapsed if elapsed > 0 else 0
				print(f"[PROGRESS] {completed}/{n} ({rate:.1f} files/s)")

	copied = sum(1 for e in entries if e.copied)
	skipped_existing = sum(1 for e in entries if e.skipped_existing)
	validation_fail = sum(1 for e in entries if e.validated is False)
	errors = sum(1 for e in entries if e.error)

	print("\n=== SUMMARY ===")
	print(f"total_source: {n}")
	print(f"copied: {copied}")
	print(f"skipped_existing: {skipped_existing}")
	if HASH_VALIDATE:
		print(f"validation_fail: {validation_fail}")
	print(f"errors: {errors}")
	print(f"dest_train: {sum(1 for e in entries if e.split=='train')}")
	print(f"dest_val: {sum(1 for e in entries if e.split=='val')}")
	print(f"dest_test: {sum(1 for e in entries if e.split=='test')}")

	write_manifest(entries)
	return 0


if __name__ == '__main__':  # pragma: no cover
	raise SystemExit(main())