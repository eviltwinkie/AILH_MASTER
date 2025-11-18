#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-file disk read auto-tuner for Linux (ext4-friendly).
- Centralized Config including dataset selection knobs (root, pattern, min/max bytes, limit, shuffle).
- Storage context detection (mount, device, model/vendor/serial).
- O_NOATIME capability probe and auto-enable only if permitted.
- Coarse→fine auto-tuning to find best settings for ~80 KB files.
- Emits CSV of trial results, best.json, and a production code snippet.

Focus: disk READ path only (no writes).
"""

from __future__ import annotations
import argparse
import csv
import errno
import fnmatch
import itertools
import json
import logging
import mmap
import os
import random
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import quantiles
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- Status Symbols & Output Control ----------
SUCCESS_SYMBOL = "✅"
FAILURE_SYMBOL = "❌"
WARNING_SYMBOL = "⚠️"
INFO_SYMBOL = "ℹ️"

# Output redirection for dual console+file output
class TeeOutput:
    """Redirect stdout to both console and file."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file: Optional[Any] = None
        self.original_stdout: Optional[Any] = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.file = open(self.filepath, "w")
        return self
    
    def __exit__(self, *args):
        if self.file:
            self.file.close()
        if self.original_stdout:
            sys.stdout = self.original_stdout
    
    def write(self, msg: str) -> None:
        if self.original_stdout:
            self.original_stdout.write(msg)
        if self.file:
            self.file.write(msg)
            self.file.flush()
    
    def flush(self) -> None:
        if self.original_stdout:
            self.original_stdout.flush()
        if self.file:
            self.file.flush()

# Helper functions for status messages
def print_status(success: bool, message: str) -> None:
    """Print a status message with symbol."""
    sym = SUCCESS_SYMBOL if success else FAILURE_SYMBOL
    print(f"{sym} {message}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{WARNING_SYMBOL} {message}")

def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{INFO_SYMBOL} {message}")

# ---------- small helpers ----------
def _split_env_list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name, "")
    return default if not v.strip() else v.strip().split()

def _split_env_int_list(name: str, default: list[int]) -> list[int]:
    raw = _split_env_list(name, [])
    return default if not raw else [int(x) for x in raw]

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

# ---------- Config ----------
class Config:
    """
    Central place to configure ranges + dataset selection defaults.
    Override via environment variables if you like:
      DATA SELECTION:
        SFA_ROOT=/data/path
        SFA_PATTERN="*.wav"
        SFA_MIN_BYTES=60000
        SFA_MAX_BYTES=120000
        SFA_LIMIT=0
        SFA_SHUFFLE=1
      SEARCH SPACES (coarse):
        SFA_THREADS="1 2 4 8 16"
        SFA_BUFSIZES="32768 65536 131072"
        SFA_METHODS="osread readinto mmap"
        SFA_FADVISE="SEQUENTIAL WILLNEED NONE"
        SFA_MAX_INFLIGHT="32 64 128"
        SFA_FILES_PER_TASK="1 2 4"
      ROUNDS / MODES:
        SFA_ROUNDS=2
        SFA_COLD_ROUNDS=0
        SFA_TRUE_COLD=0
      MISC:
        SFA_RECORD_LATENCIES=0
        SFA_CSV="smallfile_autotune_results.csv"
        SFA_BEST_JSON="best_smallfiles.json"
        SFA_BEST_SNIPPET="best_smallfiles_snippet.py"
    """

    # ---- Dataset selection defaults ----
    ROOT        = os.environ.get("SFA_ROOT", "/DEVELOPMENT/ROOT_AILH/DATA_STORE/DATASET_DEV")  
    PATTERN     = os.environ.get("SFA_PATTERN", "*.wav")
    MIN_BYTES   = _env_int("SFA_MIN_BYTES", 60000)
    MAX_BYTES   = _env_int("SFA_MAX_BYTES", 120000)
    LIMIT       = _env_int("SFA_LIMIT", 0)                    # 0 = no limit
    SHUFFLE     = _env_bool("SFA_SHUFFLE", True)
    TRIALS      = 2
    
    # ---- Search spaces (coarse) ----
    THREADS_COARSE        = _split_env_int_list("SFA_THREADS",        [5])
    BUFSIZES_COARSE       = _split_env_int_list("SFA_BUFSIZES",       [131072])
    METHODS_COARSE        = _split_env_list    ("SFA_METHODS",        ["osread"])
    FADVISE_COARSE        = _split_env_list    ("SFA_FADVISE",        ["NONE"])
    MAX_INFLIGHT_COARSE   = _split_env_int_list("SFA_MAX_INFLIGHT",   [0])
    FILES_PER_TASK_COARSE = _split_env_int_list("SFA_FILES_PER_TASK", [192, 384, 512, 768])
    
    # ---- Search spaces (fine) - for neighbor search ----
    THREADS_FINE          = _split_env_int_list("SFA_THREADS_FINE",   [3, 4, 5, 6, 7, 8])
    BUFSIZES_FINE         = _split_env_int_list("SFA_BUFSIZES_FINE",  [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288])
    MAX_INFLIGHT_FINE     = _split_env_int_list("SFA_MAX_INFLIGHT_FINE", [0, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024])
    FILES_PER_TASK_FINE   = _split_env_int_list("SFA_FILES_PER_TASK_FINE", [128, 192, 256, 384, 512, 768, 1024])


    # ---- Rounds ----
    ROUNDS      = _env_int("SFA_ROUNDS", 3)       # warm-cache repeats per combo
    COLD_ROUNDS = _env_int("SFA_COLD_ROUNDS", 2)  # optional cold-ish rounds (per-file DONTNEED)
    TRUE_COLD   = _env_bool("SFA_TRUE_COLD", False)

    # ---- Latency recording ----
    RECORD_LATENCIES = _env_bool("SFA_RECORD_LATENCIES", False)

    # ---- Output artifacts ----
    BEST_JSON    = os.environ.get("SFA_BEST_JSON", "../DOCS/test_disk_tune_results.json")
    BEST_TEXT    = os.environ.get("SFA_BEST_TEXT", "../DOCS/test_disk_tune_results.txt")
    
    # ---- Output control ----
    WRITE_JSON = _env_bool("SFA_WRITE_JSON", True)
    WRITE_TEXT = _env_bool("SFA_WRITE_TEXT", True)
    VERBOSE = _env_bool("SFA_VERBOSE", False)

# ---------- Constants ----------
# fadvise constants (Linux)
POSIX_FADV_NORMAL     = 0
POSIX_FADV_RANDOM     = 1
POSIX_FADV_SEQUENTIAL = 2
POSIX_FADV_WILLNEED   = 3
POSIX_FADV_DONTNEED   = 4
POSIX_FADV_NOREUSE    = 5
HAVE_POSIX_FADVISE = hasattr(os, "posix_fadvise")

# Performance tuning defaults
DEFAULT_NEIGHBOR_WIDTH = 1
DEFAULT_FILE_LIMIT_MULTIPLIER = 2
WARM_ROUND_PREFIX = "warm"
COLD_ROUND_PREFIX = "cold"

# File system constants
CACHE_DROP_PATH = "/proc/sys/vm/drop_caches"
CACHE_DROP_VALUE = b"3"
MOUNTINFO_PATH = "/proc/self/mountinfo"
SYSFS_BLOCK_PATH = "/sys/block"

# ---------- File discovery ----------
def list_files(root: Path, pattern: str, min_bytes: int, max_bytes: int, 
               shuffle: bool, limit: int = 0) -> List[Path]:
    """
    Recursively find files matching pattern and size constraints.
    
    Args:
        root: Root directory to search
        pattern: Filename pattern (e.g., "*.wav")
        min_bytes: Minimum file size in bytes
        max_bytes: Maximum file size in bytes
        shuffle: Whether to shuffle the results
        limit: Maximum number of files to return (0 = no limit)
    
    Returns:
        List of Path objects matching criteria
    """
    if not root.exists():
        logger.error(f"Root directory does not exist: {root}")
        return []
    
    out = []
    try:
        for p, _, files in os.walk(root):
            for name in files:
                if pattern and not fnmatch.fnmatch(name, pattern):
                    continue
                fp = Path(p) / name
                try:
                    st = fp.stat()
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if min_bytes <= st.st_size <= max_bytes:
                        out.append(fp)
                        # Early exit if limit reached (before shuffle)
                        if limit > 0 and len(out) >= limit * 2:
                            break
                except (FileNotFoundError, PermissionError) as e:
                    logger.debug(f"Skipping file {fp}: {e}")
            if limit > 0 and len(out) >= limit * 2:
                break
    except Exception as e:
        logger.error(f"Error during file discovery: {e}")
        
    if shuffle:
        random.shuffle(out)
    
    if limit > 0 and len(out) > limit:
        out = out[:limit]
        
    logger.info(f"Found {len(out)} files matching criteria")
    return out

# ---------- Mount & device inspection ----------
def parse_mountinfo() -> list[dict]:
    out = []
    with open("/proc/self/mountinfo", "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            try:
                dash = parts.index("-")
            except ValueError:
                continue
            mount_point = parts[4]
            fstype = parts[dash+1]
            source = parts[dash+2] if len(parts) > dash+2 else ""
            mount_opts = parts[5]
            out.append({
                "mount_point": mount_point,
                "fstype": fstype,
                "source": source,
                "mount_opts": mount_opts,
            })
    return out

def find_mount_for_path(p: Path) -> dict|None:
    p = p.resolve()
    mounts = parse_mountinfo()
    best, best_len = None, -1
    for m in mounts:
        mp = Path(m["mount_point"])
        try:
            if str(p).startswith(str(mp.resolve())):
                l = len(str(mp))
                if l > best_len:
                    best, best_len = m, l
        except Exception:
            pass
    return best

def base_block_name(dev: str) -> str|None:
    if not dev.startswith("/dev/"):
        return None
    name = dev.split("/")[-1]
    if name.startswith("nvme") and "p" in name:
        while len(name) and name[-1].isdigit(): name = name[:-1]
        if name.endswith("p"): name = name[:-1]
        return name
    if name.startswith("sd") and name[-1].isdigit():
        while len(name) and name[-1].isdigit(): name = name[:-1]
        return name
    if name.startswith("mmcblk") and name[-1].isdigit():
        while len(name) and name[-1].isdigit(): name = name[:-1]
        if name.endswith("p"): name = name[:-1]
        return name
    return name

def read_sysfs_text(path: str) -> str|None:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None

def detect_device_info(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "mount_point": None,
        "fstype": None,
        "device": None,
        "noatime": None,
        "relatime": None,
        "strictatime": None,
        "model": None,
        "vendor": None,
        "serial": None,
        "block_name": None,
        "lsblk": None,
    }

    m = find_mount_for_path(root)
    if not m:
        return info

    info["mount_point"] = m.get("mount_point")
    info["fstype"] = m.get("fstype")
    info["device"] = m.get("source")

    # Parse mount options as a set of flags
    opts_raw = m.get("mount_opts") or ""
    opt_flags = set(o.strip() for o in opts_raw.split(",") if o.strip())

    info["noatime"] = "noatime" in opt_flags
    info["relatime"] = "relatime" in opt_flags
    info["strictatime"] = "strictatime" in opt_flags

    block = base_block_name(info["device"] or "")
    info["block_name"] = block

    # Read model/vendor/serial from sysfs if possible
    if block:
        if block.startswith("nvme"):
            dev_link = f"/sys/class/block/{block}/device"
            try:
                real = os.path.realpath(dev_link)
                parts = real.split("/")
                # Try to find something like "nvme0"
                nvme_ctrl = next(
                    (p for p in parts if p.startswith("nvme") and p[-1].isdigit()),
                    None,
                )
                ctrl = nvme_ctrl or "nvme0"
            except Exception:
                ctrl = "nvme0"

            info["model"] = read_sysfs_text(f"/sys/class/nvme/{ctrl}/model")
            info["serial"] = read_sysfs_text(f"/sys/class/nvme/{ctrl}/serial")
        else:
            info["model"] = read_sysfs_text(f"/sys/block/{block}/device/model")
            info["vendor"] = read_sysfs_text(f"/sys/block/{block}/device/vendor")

    # Fallback / enrichment via lsblk
    try:
        out = subprocess.check_output(
            ["lsblk", "-ndo", "NAME,MODEL,SERIAL,PKNAME,TYPE"],
            text=True,
        ).strip()

        if out:
            lines = out.splitlines()
        else:
            lines = []

        info["lsblk"] = lines

        # If we still don't have model or serial, try to infer from lsblk output
        if (info["model"] is None or info["serial"] is None) and block:
            for line in lines:
                if not line.strip():
                    continue

                # NAME MODEL SERIAL PKNAME TYPE
                cols = line.split(None, 4)
                if not cols:
                    continue

                name = cols[0]
                model = cols[1] if len(cols) > 1 else None
                serial = cols[2] if len(cols) > 2 else None
                pkname = cols[3] if len(cols) > 3 else None
                # type_field = cols[4] if len(cols) > 4 else None  # unused currently

                if name == block or pkname == block:
                    if info["model"] is None:
                        info["model"] = model
                    if info["serial"] is None:
                        info["serial"] = serial
                    break
    except Exception:
        # lsblk not available or failed – leave lsblk/model/serial as-is
        pass

    return info

# ---------- O_NOATIME test ----------
def can_use_onoatime(sample_file: Path) -> Tuple[bool, str]:
    """
    Test if O_NOATIME flag can be used on a file.
    
    Args:
        sample_file: Path to a test file
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    flags = os.O_RDONLY | getattr(os, "O_NOATIME", 0)
    try:
        fd = os.open(sample_file, flags)
        try:
            _ = os.read(fd, 1)
        finally:
            os.close(fd)
        return True, "success"
    except OSError as e:
        if e.errno in (errno.EPERM, errno.EACCES):
            return False, "permission denied (own the file or CAP_FOWNER required)"
        return False, f"open failed: {e.strerror}"
    except AttributeError:
        return False, "O_NOATIME not defined"

# ---------- Drop caches (root-only; use carefully) ----------
def drop_caches_if_root() -> bool:
    """
    Drop filesystem caches (requires root privileges).
    Uses /proc/sys/vm/drop_caches with value 3 to clear all caches.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if os.geteuid() != 0:
        logger.debug("Not running as root; skipping cache drop")
        return False
    try:
        os.sync()
        with open(CACHE_DROP_PATH, "w") as f:
            f.write(CACHE_DROP_VALUE.decode())
        logger.info("Successfully dropped caches")
        return True
    except PermissionError:
        logger.warning("Failed to write to drop_caches: permission denied")
        return False
    except Exception as e:
        logger.error(f"Failed to drop caches: {e}")
        return False

# ---------- Read methods ----------
def read_with_os_read(path: Path, bufsize: int, use_noatime: bool, 
                      fadvise: Optional[str], record_latencies: bool):
    """
    Read file using os.readv() with optional fadvise hints.
    Memory efficient with pre-allocated buffers.
    
    Args:
        path: File path to read
        bufsize: Buffer size in bytes
        use_noatime: Whether to use O_NOATIME flag
        fadvise: Fadvise hint (SEQUENTIAL, RANDOM, WILLNEED)
        record_latencies: Whether to record per-chunk latencies
        
    Returns:
        Dict with metrics: bytes_read, latencies, elapsed_time
    """
    flags = os.O_RDONLY
    if use_noatime:
        flags |= getattr(os, "O_NOATIME", 0)
    
    fd = os.open(path, flags)
    try:
        if HAVE_POSIX_FADVISE and fadvise:
            hint = {"SEQUENTIAL": 2, "RANDOM": 1, "WILLNEED": 3}.get(fadvise, 0)
            if hint:
                os.posix_fadvise(fd, 0, 0, hint)

        total = 0
        buf = bytearray(bufsize)
        mv = memoryview(buf)
        t0 = time.perf_counter()
        latencies = []
        t_chunk = 0  # Initialize to prevent unbound error

        # read loop: fill mv using readv
        while True:
            if record_latencies:
                t_chunk = time.perf_counter()
            # os.readv returns number of bytes read into the buffers (0 = EOF)
            n = os.readv(fd, [mv])
            if record_latencies and n > 0:
                latencies.append(time.perf_counter() - t_chunk)
            if n == 0:
                break
            total += n
            # (no need to shift the view; next read overwrites from start)

        if HAVE_POSIX_FADVISE:
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)

        lat = (time.perf_counter() - t0) if record_latencies else None
        return total, lat
    finally:
        os.close(fd)


def read_with_readinto(path: Path, bufsize: int, use_noatime: bool, 
                       fadvise: Optional[str], record_latencies: bool):
    """
    Read file using file.readinto() method.
    
    Args:
        path: File path to read
        bufsize: Buffer size in bytes
        use_noatime: Whether to use O_NOATIME flag
        fadvise: Fadvise hint (SEQUENTIAL, RANDOM, WILLNEED)
        record_latencies: Whether to record per-chunk latencies
        
    Returns:
        Dict with metrics: bytes_read, latencies, elapsed_time
    """
    flags = os.O_RDONLY
    if use_noatime:
        flags |= getattr(os, "O_NOATIME", 0)

    # open() + fdopen with closefd=True so f.close() closes the underlying fd
    fd = os.open(path, flags)
    f = os.fdopen(fd, "rb", closefd=True)
    try:
        if HAVE_POSIX_FADVISE and fadvise:
            hint = {"SEQUENTIAL":2, "RANDOM":1, "WILLNEED":3}.get(fadvise, 0)
            if hint:
                os.posix_fadvise(fd, 0, 0, hint)

        total = 0
        buf = bytearray(bufsize)
        mv = memoryview(buf)
        t0 = time.perf_counter()

        while True:
            n = f.readinto(mv)
            if not n:
                break
            total += n

        # advise drop while fd is still open
        if HAVE_POSIX_FADVISE:
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)

        lat = (time.perf_counter() - t0) if record_latencies else None
        return total, lat
    finally:
        # closes both file object and underlying fd (because closefd=True)
        f.close()


def read_with_mmap(path: Path, _bufsize: int, use_noatime: bool, 
                   fadvise: Optional[str], record_latencies: bool):
    """
    Read file using memory-mapped I/O (mmap).
    
    Args:
        path: File path to read
        _bufsize: Ignored for mmap (kept for interface compatibility)
        use_noatime: Whether to use O_NOATIME flag
        fadvise: Fadvise hint (SEQUENTIAL, RANDOM, WILLNEED)
        record_latencies: Whether to record per-chunk latencies
        
    Returns:
        Dict with metrics: bytes_read, latencies, elapsed_time
    """
    flags = os.O_RDONLY
    if use_noatime:
        flags |= getattr(os, "O_NOATIME", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if HAVE_POSIX_FADVISE and fadvise:
            hint = {"SEQUENTIAL":2,"RANDOM":1,"WILLNEED":3}.get(fadvise, 0)
            if hint:
                os.posix_fadvise(fd, 0, 0, hint)
        t0 = time.perf_counter()
        if st.st_size == 0:
            lat = time.perf_counter() - t0 if record_latencies else None
            return 0, lat
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        try:
            _ = mm[:]  # touch pages
            total = st.st_size
        finally:
            mm.close()
        if HAVE_POSIX_FADVISE:
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        lat = (time.perf_counter() - t0) if record_latencies else None
        return total, lat
    finally:
        os.close(fd)

def read_with_sendfile(path: Path, _bufsize: int, use_noatime: bool, 
                       fadvise: Optional[str], record_latencies: bool):
    """
    Read file using sendfile() syscall (Linux-specific).
    
    Args:
        path: File path to read
        _bufsize: Ignored for sendfile (kept for interface compatibility)
        use_noatime: Whether to use O_NOATIME flag
        fadvise: Fadvise hint (SEQUENTIAL, RANDOM, WILLNEED)
        record_latencies: Whether to record per-chunk latencies
        
    Returns:
        Dict with metrics: bytes_read, latencies, elapsed_time
    """
    flags = os.O_RDONLY
    if use_noatime:
        flags |= getattr(os, "O_NOATIME", 0)
    fd_in = os.open(path, flags)
    fd_out = os.open("/dev/null", os.O_WRONLY)
    try:
        if HAVE_POSIX_FADVISE and fadvise:
            hint = {"SEQUENTIAL":2,"RANDOM":1,"WILLNEED":3}.get(fadvise, 0)
            if hint:
                os.posix_fadvise(fd_in, 0, 0, hint)
        st = os.fstat(fd_in)
        remain, offset = st.st_size, 0
        t0 = time.perf_counter()
        while remain > 0:
            sent = os.sendfile(fd_out, fd_in, offset, remain)
            if sent == 0:
                break
            offset += sent
            remain -= sent
        if HAVE_POSIX_FADVISE:
            os.posix_fadvise(fd_in, 0, 0, POSIX_FADV_DONTNEED)
        lat = (time.perf_counter() - t0) if record_latencies else None
        return st.st_size, lat
    finally:
        os.close(fd_in); os.close(fd_out)

READ_METHODS = {
    "osread":   read_with_os_read,
    "readinto": read_with_readinto,
    "mmap":     read_with_mmap,
    "sendfile": read_with_sendfile,
}

# ---------- Runner with inflight cap & per-task batching ----------
def run_once(files: List[Path], threads: int, bufsize: int, method: str,
             fadvise: Optional[str], use_noatime: bool, record_latencies: bool,
             max_inflight: Optional[int], files_per_task: int) -> Tuple[int, Optional[List[float]]]:
    """
    Execute a single benchmark trial with specified parameters.
    
    Args:
        files: List of files to read
        threads: Number of worker threads
        bufsize: I/O buffer size
        method: Read method (osread, readinto, mmap, sendfile)
        fadvise: Fadvise hint or None
        use_noatime: Whether to use O_NOATIME flag
        record_latencies: Whether to record per-file latencies
        max_inflight: Max concurrent operations or None for unlimited
        files_per_task: Files to batch per task
        
    Returns:
        Tuple of (total_bytes_read, latencies_list or None)
    """
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        headroom = 64  # keep some space for other files/sockets
        per_task_fds = 2 if method == "sendfile" else 1
        if max_inflight is not None and max_inflight > 0:
            max_allowed = max(1, (soft - headroom) // per_task_fds)
            if max_inflight > max_allowed:
                logger.warning(f"Clamping max_inflight {max_inflight} -> {max_allowed} (soft limit {soft})")
                max_inflight = max_allowed
    except Exception as e:
        logger.debug(f"Failed to check file descriptor limits: {e}")

    fn = READ_METHODS[method]
    total_bytes = 0
    latencies: List[float] = []
    batches = [files[i:i+files_per_task] for i in range(0, len(files), files_per_task)]
    inflight_sem = threading.Semaphore(max_inflight if (max_inflight and max_inflight > 0) else len(batches))

    def task(batch: List[Path]):
        nonlocal total_bytes
        with inflight_sem:
            batch_lat = []
            for path in batch:
                try:
                    n, lat = fn(path, bufsize, use_noatime, fadvise, record_latencies)
                    total_bytes += n
                    if record_latencies and lat is not None:
                        batch_lat.append(lat)
                except Exception as e:
                    logger.error(f"Failed to read {path}: {e}")
            return batch_lat

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix=f"io{threads}") as ex:
        futs = [ex.submit(task, b) for b in batches]
        for fu in as_completed(futs):
            try:
                bl = fu.result()
                if record_latencies and bl:
                    latencies.extend(bl)
            except Exception as e:
                logger.error(f"Task failed: {e}")

    return total_bytes, latencies if record_latencies else None

def summarize(total_bytes: int, elapsed: float, nfiles: int, latencies: Optional[List[float]]):
    """
    Summarize benchmark metrics.
    
    Args:
        total_bytes: Total bytes read
        elapsed: Elapsed time in seconds
        nfiles: Number of files processed
        latencies: List of per-file latencies or None
        
    Returns:
        Dict with computed metrics
    """
    mbps = (total_bytes / 1_000_000) / elapsed if elapsed > 0 else float('inf')
    fps  = nfiles / elapsed if elapsed > 0 else float('inf')
    avg_lat = (sum(latencies)/len(latencies)) if latencies else (elapsed/nfiles if nfiles else 0.0)
    p95 = quantiles(latencies, n=20)[18] if latencies and len(latencies) >= 20 else None
    return {
        "mbps": mbps,
        "files_per_s": fps,
        "avg_ms": avg_lat * 1000.0,
        "p95_ms": (p95 * 1000.0) if p95 is not None else None,
    }

# ---------- Auto-tuning ----------
def autotune_best_config(root: Optional[str | Path] = None, pattern: Optional[str] = None,
                         min_bytes: Optional[int] = None, max_bytes: Optional[int] = None,
                         limit: Optional[int] = None, shuffle: Optional[bool] = None) -> Dict[str, Any]:
    """
    Auto-tune and find best configuration for disk read performance.
    
    Returns dict of best settings found for warm-cache throughput (files/sec primary, 
    avg latency tie-break).

    Parameter precedence:
      Function args (if not None) >
      Env vars (already read into Config) >
      Config defaults
      
    Returns:
        Dict with best configuration and performance metrics
    """
    cfg = Config

    # Resolve dataset selection with precedence
    root    = Path(root if root is not None else (cfg.ROOT or ""))
    pattern = pattern   if pattern   is not None else cfg.PATTERN
    min_b   = min_bytes if min_bytes is not None else cfg.MIN_BYTES
    max_b   = max_bytes if max_bytes is not None else cfg.MAX_BYTES
    limit   = limit     if limit     is not None else cfg.LIMIT
    shuffle = shuffle   if shuffle   is not None else cfg.SHUFFLE

    if not str(root):
        raise RuntimeError("Root path not provided. Set SFA_ROOT or pass root=... to autotune_best_config().")

    files = list_files(root, pattern, min_b, max_b, shuffle=shuffle, limit=limit)
    if not files:
        raise RuntimeError("No files matched criteria.")

    # Storage context
    ctx = detect_device_info(root)
    print("\n" + "=" * 80)
    print("STORAGE CONTEXT")
    print("=" * 80)
    print(f"Path:          {root.resolve()}")
    print(f"Mount point:   {ctx.get('mount_point')}")
    print(f"Filesystem:    {ctx.get('fstype')}")
    print(f"Device:        {ctx.get('device')} (block: {ctx.get('block_name')})")
    print(f"noatime:       {ctx.get('noatime')} | relatime: {ctx.get('relatime')} | strictatime: {ctx.get('strictatime')}")
    model_line = ((ctx.get('vendor') or '') + ' ' + (ctx.get('model') or '')).strip()
    if model_line or ctx.get('serial'):
        extra = f" (S/N {ctx.get('serial')})" if ctx.get('serial') else ""
        print(f"Drive model:   {model_line}{extra}")
    else:
        print_warning("Drive model:   (unknown)")

    # O_NOATIME probe
    test_file = files[0]
    ono_ok, ono_reason = can_use_onoatime(test_file)
    print_status(ono_ok, f"O_NOATIME usable ({ono_reason})")

    from statistics import mean, stdev

    def run_trials(space):
        """
        Run all combos in `space`, aggregate across repeated runs,
        and select winners by averaged warm metrics.

        Returns:
            Tuple of (best_mean_stats, best_meta, ranked) where:
            - best_mean_stats: Averaged performance metrics for best config
            - best_meta: Metadata dict for best config
            - ranked: List of entries sorted best→worst, each containing:
              - "meta": Combo knobs (threads, bufsize, method, etc.)
              - "means": Computed statistics with SD values
              - "sel_key": Selection key tuple for sorting
        """

        # Collect per-combo stats across rounds (warm only for selection)
        # key = (thr, buf, meth, fav, infl, fpt)
        agg = {}

        for (thr, buf, meth, fav, infl, fpt) in space:
            for mode, rounds in (("warm", Config.ROUNDS), ("coldish", Config.COLD_ROUNDS)):
                if rounds <= 0:
                    continue
                for r in range(1, rounds+cfg.TRIALS):
                    if shuffle:
                        random.shuffle(files)

                    t0 = time.perf_counter()
                    total, lats = run_once(
                        files=files,
                        threads=thr,
                        bufsize=buf,
                        method=meth,
                        fadvise=(None if fav == "NONE" else fav),
                        use_noatime=ono_ok,
                        record_latencies=Config.RECORD_LATENCIES,
                        max_inflight=(infl if infl > 0 else None),
                        files_per_task=max(1, fpt),
                    )
                    elapsed = time.perf_counter() - t0

                    stats = summarize(
                        total, elapsed, len(files),
                        lats if Config.RECORD_LATENCIES else None
                    )

                    meta = {
                        "threads": thr, "bufsize": buf, "method": meth,
                        "fadvise": fav, "noatime": ono_ok, "mode": mode, "round": r,
                        "nfiles": len(files), "bytes": total, "elapsed": elapsed,
                        "max_inflight": infl, "files_per_task": fpt
                    }

                    label = (f"thr={thr} buf={buf} method={meth} fadvise={fav} "
                            f"inflight={infl} fpt={fpt} noatime={int(ono_ok)}")
                    print(f"[{mode} r{r}] {label} | "
                        f"{stats['files_per_s']:.1f} files/s | "
                        f"{stats['mbps']:.1f} MB/s | avg {stats['avg_ms']:.2f} ms")

                    if mode == "warm":
                        key = (thr, buf, meth, fav, infl, fpt)
                        a = agg.get(key)
                        if a is None:
                            a = {
                                "fps": [], "avg": [], "mbps": [],
                                "meta": {
                                    "threads": thr, "bufsize": buf, "method": meth,
                                    "fadvise": fav, "noatime": ono_ok,
                                    "max_inflight": infl, "files_per_task": fpt
                                }
                            }
                            agg[key] = a
                        a["fps"].append(stats["files_per_s"])
                        a["avg"].append(stats["avg_ms"])
                        a["mbps"].append(stats["mbps"])

        # ---- compute means/stdevs and rank ----
        ranked = []
        for a in agg.values():
            cnt = len(a["fps"])
            if cnt == 0:
                continue
            fps_mean = mean(a["fps"])
            avg_ms_mean = mean(a["avg"])
            mbps_mean = mean(a["mbps"])
            fps_sd = stdev(a["fps"]) if cnt > 1 else None
            avg_sd = stdev(a["avg"]) if cnt > 1 else None
            mbps_sd = stdev(a["mbps"]) if cnt > 1 else None
            entry = {
                "meta": a["meta"],
                "means": {
                    "files_per_s": fps_mean,
                    "avg_ms": avg_ms_mean,
                    "mbps": mbps_mean,
                    "files_per_s_sd": fps_sd,
                    "avg_ms_sd": avg_sd,
                    "mbps_sd": mbps_sd,
                    "count": cnt,
                },
                # selection: higher fps, lower avg_ms, higher mbps
                "sel_key": (fps_mean, -avg_ms_mean, mbps_mean),
            }
            ranked.append(entry)

        if not ranked:
            raise RuntimeError("No warm results aggregated — check Config.ROUNDS > 0 and search space.")

        ranked.sort(key=lambda e: e["sel_key"], reverse=True)  # best first

        best_entry = ranked[0]
        best_mean_stats = best_entry["means"]
        best_meta = best_entry["meta"]
        return best_mean_stats, best_meta, ranked
    
    # ----- Coarse search -----
    print("\n=== COARSE SEARCH ===")
    coarse_space = list(itertools.product(
        cfg.THREADS_COARSE,
        cfg.BUFSIZES_COARSE,
        cfg.METHODS_COARSE,
        cfg.FADVISE_COARSE,
        cfg.MAX_INFLIGHT_COARSE,
        cfg.FILES_PER_TASK_COARSE
    ))

    if not coarse_space:
        raise RuntimeError("Empty search space; check THREADS/BUFSIZES/METHODS/FADVISE/MAX_INFLIGHT/FILES_PER_TASK.")

    best_stats, best, coarse_ranked = run_trials(coarse_space)

    # ----- Fine search around the coarse winner -----
    print("\n=== FINE SEARCH ===")
    def neigh_list(val, candidates, width=1):
        seen, uniq = set(), []
        for x in candidates:
            if x not in seen:
                seen.add(x); uniq.append(x)
        if not uniq: return [val]
        try:
            i = uniq.index(val)
        except ValueError:
            if isinstance(val, (int, float)) and all(isinstance(x,(int,float)) for x in uniq):
                i = min(range(len(uniq)), key=lambda k: abs(uniq[k]-val))
            else:
                return [val]
        lo = max(0, i-width); hi = min(len(uniq)-1, i+width)
        return uniq[lo:hi+1]

    # Use configuration values for fine search
    THREADS_ALL   = sorted(set(cfg.THREADS_FINE))
    BUFSIZES_ALL  = sorted(set(cfg.BUFSIZES_FINE))
    INFLIGHT_ALL  = sorted(set(cfg.MAX_INFLIGHT_FINE))
    FPT_ALL       = sorted(set(cfg.FILES_PER_TASK_FINE))

    thr_c  = neigh_list(best["threads"],           THREADS_ALL)
    buf_c  = neigh_list(best["bufsize"],           BUFSIZES_ALL)
    infl_c = neigh_list(best["max_inflight"] or 0, INFLIGHT_ALL)
    fpt_c  = neigh_list(best["files_per_task"],    FPT_ALL)
    meth_c = [best["method"]]
    fav_c  = [best["fadvise"]]

    fine_space = list(itertools.product(thr_c, buf_c, meth_c, fav_c, infl_c, fpt_c))

    best_stats_final, best_final, fine_ranked = run_trials(fine_space)

    # ---- Print AVERAGED WINNERS (warm) ----
    print("\n" + "=" * 80)
    print("AVERAGED WINNERS (warm-cache, ranked by files/sec)")
    print("=" * 80)
    
    def _fmt(entry):
        m, s = entry["meta"], entry["means"]
        sd_f = (f" ±{s['files_per_s_sd']:.3f}" if s["files_per_s_sd"] is not None else "")
        sd_a = (f" ±{s['avg_ms_sd']:.3f}" if s["avg_ms_sd"] is not None else "")
        sd_m = (f" ±{s['mbps_sd']:.3f}" if s["mbps_sd"] is not None else "")
        return (f"threads={m['threads']} bufsize={m['bufsize']} method={m['method']} "
                f"fadvise={m['fadvise']} inflight={m['max_inflight']} fpt={m['files_per_task']} noatime={int(m['noatime'])}\n"
                f"   files/s (mean){sd_f}: {s['files_per_s']:.3f} | "
                f"avg_ms (mean){sd_a}: {s['avg_ms']:.3f} | "
                f"MB/s (mean){sd_m}: {s['mbps']:.3f} | runs={s['count']}")

    print("\nBEST:\n" + _fmt(fine_ranked[0]))
    if len(fine_ranked) > 1:
        print("\nRUNNER-UP:\n" + _fmt(fine_ranked[1]))
    else:
        print_warning("RUNNER-UP: (only one combo evaluated)")

    chosen = {
        # Tuned knobs
        "threads": best_final["threads"],
        "bufsize": best_final["bufsize"],
        "method": best_final["method"],
        "fadvise": best_final["fadvise"],
        "use_noatime": bool(best_final["noatime"]),
        "max_inflight": best_final["max_inflight"],
        "files_per_task": best_final["files_per_task"],
        # Dataset selection that produced these results
        "root": str(root.resolve()),
        "pattern": pattern,
        "min_bytes": min_b,
        "max_bytes": max_b,
        "limit": limit,
        "shuffle": shuffle,
        # Metrics
        "files_per_s": round(best_stats_final["files_per_s"], 3),
        "mbps": round(best_stats_final["mbps"], 3),
        "avg_ms": round(best_stats_final["avg_ms"], 3),
        # Context
        "storage_context": {
            "mount_point": ctx.get("mount_point"), "fstype": ctx.get("fstype"),
            "device": ctx.get("device"), "block": ctx.get("block_name"),
            "noatime": ctx.get("noatime"), "relatime": ctx.get("relatime")
        }
    }

    # Write JSON output
    if Config.WRITE_JSON:
        try:
            with open(Config.BEST_JSON, "w") as f:
                json.dump(chosen, f, indent=2)
            print_status(True, f"Best config saved to: {Config.BEST_JSON}")
        except Exception as e:
            print_status(False, f"Failed to write JSON: {e}")

    # Write text output
    if Config.WRITE_TEXT:
        _write_best_text(chosen, ctx, fine_ranked)

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION (warm-cache, by files/sec)")
    print("=" * 80)
    print(json.dumps(chosen, indent=2))
    print("=" * 80)
    return chosen

# ---------- Results output functions ----------
def _write_best_text(best: Dict[str, Any], storage_context: Dict[str, Any], 
                     fine_ranked: List[Dict[str, Any]]) -> None:
    """
    Write formatted benchmark results to a text file.
    
    Args:
        best: Best configuration dictionary
        storage_context: Storage system information
        fine_ranked: Ranked list of configurations from fine search
    """
    try:
        with open(Config.BEST_TEXT, "w") as f:
            f.write("="*70 + "\n")
            f.write("SMALL-FILE DISK READ AUTO-TUNER RESULTS\n")
            f.write("="*70 + "\n\n")
            
            # Storage context
            f.write("STORAGE CONTEXT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mount Point:   {storage_context.get('mount_point', 'N/A')}\n")
            f.write(f"Filesystem:    {storage_context.get('fstype', 'N/A')}\n")
            f.write(f"Device:        {storage_context.get('device', 'N/A')}\n")
            f.write(f"Block Name:    {storage_context.get('block', 'N/A')}\n")
            f.write(f"Mount Options: noatime={storage_context.get('noatime', 'N/A')}, "
                   f"relatime={storage_context.get('relatime', 'N/A')}\n\n")
            
            # Dataset info
            f.write("DATASET SELECTION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Root Path:     {best['root']}\n")
            f.write(f"Pattern:       {best['pattern']}\n")
            f.write(f"Size Range:    {best['min_bytes']} - {best['max_bytes']} bytes\n")
            f.write(f"Limit:         {best['limit'] or 'None'}\n\n")
            
            # Best configuration
            f.write("BEST CONFIGURATION (by warm-cache throughput)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Threads:       {best['threads']}\n")
            f.write(f"Buffer Size:   {best['bufsize']} bytes\n")
            f.write(f"Read Method:   {best['method']}\n")
            f.write(f"Fadvise Hint:  {best['fadvise'] or 'NONE'}\n")
            f.write(f"Use O_NOATIME: {best['use_noatime']}\n")
            f.write(f"Max Inflight:  {best['max_inflight'] or 'Unlimited'}\n")
            f.write(f"Files/Task:    {best['files_per_task']}\n\n")
            
            # Performance metrics
            if fine_ranked:
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 70 + "\n")
                best_entry = fine_ranked[0]
                m, s = best_entry["meta"], best_entry["means"]
                
                sd_f = (f" ±{s['files_per_s_sd']:.3f}" if s.get('files_per_s_sd') is not None else "")
                sd_a = (f" ±{s['avg_ms_sd']:.3f}" if s.get('avg_ms_sd') is not None else "")
                sd_m = (f" ±{s['mbps_sd']:.3f}" if s.get('mbps_sd') is not None else "")
                
                f.write(f"Files/Second:  {s.get('files_per_s', 0):.1f}{sd_f} files/s\n")
                f.write(f"Latency:       {s.get('avg_ms', 0):.3f}{sd_a} ms/file\n")
                f.write(f"Throughput:    {s.get('mbps', 0):.1f}{sd_m} MB/s\n")
                f.write(f"Trials:        {s.get('count', 0)} runs\n\n")
                
                # Runner-up if available
                if len(fine_ranked) > 1:
                    f.write("RUNNER-UP CONFIGURATION\n")
                    f.write("-" * 70 + "\n")
                    runner = fine_ranked[1]
                    m2, s2 = runner["meta"], runner["means"]
                    f.write(f"Config: threads={m2['threads']} bufsize={m2['bufsize']} "
                           f"method={m2['method']} inflight={m2['max_inflight']}\n")
                    f.write(f"Performance: {s2.get('files_per_s', 0):.1f} files/s, "
                           f"{s2.get('mbps', 0):.1f} MB/s\n\n")
            
            f.write("="*70 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Results written to: {Config.BEST_TEXT}")
    except Exception as e:
        logger.error(f"Failed to write text results: {e}")


# ---------- CLI (args override env which override Config defaults) ----------
def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Small-file disk read auto-tuner for Linux (ext4-friendly)"
    )
    ap.add_argument("--root", type=str, default=None, 
                    help="Root directory to scan (overrides SFA_ROOT/Config)")
    ap.add_argument("--pattern", default=None, 
                    help="Glob pattern, e.g. '*.wav'")
    ap.add_argument("--min-bytes", type=int, default=None,
                    help="Minimum file size in bytes")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="Maximum file size in bytes")
    ap.add_argument("--limit", type=int, default=None, 
                    help="Limit number of files (0 = all)")
    ap.add_argument("--shuffle", type=str, default=None, 
                    help="true/false to override shuffling")
    ap.add_argument("--no-json", action="store_true", dest="no_json",
                    help="Skip JSON output")
    ap.add_argument("--no-text", action="store_true", dest="no_text",
                    help="Skip text output")
    ap.add_argument("--verbose", action="store_true", dest="verbose",
                    help="Enable verbose output")
    return ap.parse_args()

def main() -> None:
    """Main entry point for the disk tuning application."""
    args = _parse_cli()
    
    # Apply CLI overrides to Config
    if args.no_json:
        Config.WRITE_JSON = False
    if args.no_text:
        Config.WRITE_TEXT = False
    if args.verbose:
        Config.VERBOSE = True
        logger.setLevel(logging.DEBUG)
    
    # Convert shuffle str to bool if provided
    shuffle: Optional[bool] = None
    if args.shuffle is not None:
        s = args.shuffle.strip().lower()
        shuffle = s in ("1", "true", "yes", "y", "on")
    
    autotune_best_config(
        root=args.root if args.root is not None else None,
        pattern=args.pattern if args.pattern is not None else None,
        min_bytes=args.min_bytes if args.min_bytes is not None else None,
        max_bytes=args.max_bytes if args.max_bytes is not None else None,
        limit=args.limit if args.limit is not None else None,
        shuffle=shuffle
    )

if __name__ == "__main__":
    main()
