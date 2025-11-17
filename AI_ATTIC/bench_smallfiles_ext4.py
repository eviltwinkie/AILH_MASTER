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
import argparse, fnmatch, os, sys, stat, time, itertools, threading, mmap, errno, csv, random, subprocess, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import quantiles

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
    ROOT        = os.environ.get("SFA_ROOT", "DEVELOPMENT\ROOT_AILH\DATA_STORE/TRAINING")              # empty means "must be provided by CLI or function call"
    PATTERN     = os.environ.get("SFA_PATTERN", "*.wav")
    MIN_BYTES   = _env_int("SFA_MIN_BYTES", 60000)
    MAX_BYTES   = _env_int("SFA_MAX_BYTES", 120000)
    LIMIT       = _env_int("SFA_LIMIT", 0)                    # 0 = no limit
    SHUFFLE     = _env_bool("SFA_SHUFFLE", True)
    TRIALS      = 8
    # ---- Search spaces (coarse) ----
    #THREADS_COARSE        = _split_env_int_list("SFA_THREADS",        [2, 4, 8, 16, 32])
    THREADS_COARSE        = _split_env_int_list("SFA_THREADS",        [4])
    #BUFSIZES_COARSE       = _split_env_int_list("SFA_BUFSIZES",       [8192, 32768, 131072, 524288])
    BUFSIZES_COARSE       = _split_env_int_list("SFA_BUFSIZES",       [131072])
    #METHODS_COARSE        = _split_env_list    ("SFA_METHODS",        ["osread", "readinto", "mmap"])
    METHODS_COARSE        = _split_env_list    ("SFA_METHODS",        ["osread"])
    FADVISE_COARSE        = _split_env_list    ("SFA_FADVISE",        ["SEQUENTIAL", "WILLNEED", "NONE"])
    #MAX_INFLIGHT_COARSE   = _split_env_int_list("SFA_MAX_INFLIGHT",   [16, 32, 64, 128, 256, 1024])
    MAX_INFLIGHT_COARSE   = _split_env_int_list("SFA_MAX_INFLIGHT",   [64, 128, 256, 512, 1024])
    FILES_PER_TASK_COARSE = _split_env_int_list("SFA_FILES_PER_TASK", [256])

    # ---- Rounds ----
    ROUNDS      = _env_int("SFA_ROUNDS", 2)       # warm-cache repeats per combo
    COLD_ROUNDS = _env_int("SFA_COLD_ROUNDS", 2)  # optional cold-ish rounds (per-file DONTNEED)
    TRUE_COLD   = _env_bool("SFA_TRUE_COLD", False)

    # ---- Latency recording ----
    RECORD_LATENCIES = _env_bool("SFA_RECORD_LATENCIES", False)

    # ---- Output artifacts ----
    CSV_FILE     = os.environ.get("SFA_CSV", "smallfile_autotune_results.csv")
    BEST_JSON    = os.environ.get("SFA_BEST_JSON", "best_smallfiles.json")
    BEST_SNIPPET = os.environ.get("SFA_BEST_SNIPPET", "best_smallfiles_snippet.py")

# ---------- fadvise constants (Linux) ----------
POSIX_FADV_NORMAL     = 0
POSIX_FADV_RANDOM     = 1
POSIX_FADV_SEQUENTIAL = 2
POSIX_FADV_WILLNEED   = 3
POSIX_FADV_DONTNEED   = 4
POSIX_FADV_NOREUSE    = 5
HAVE_POSIX_FADVISE = hasattr(os, "posix_fadvise")

# ---------- File discovery ----------
def list_files(root: Path, pattern: str, min_bytes: int, max_bytes: int, shuffle: bool) -> list[Path]:
    out = []
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
            except FileNotFoundError:
                pass
    if shuffle:
        random.shuffle(out)
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

def detect_device_info(root: Path) -> dict:
    info = {
        "mount_point": None, "fstype": None, "device": None,
        "noatime": None, "relatime": None, "strictatime": None,
        "model": None, "vendor": None, "serial": None, "block_name": None,
        "lsblk": None,
    }
    m = find_mount_for_path(root)
    if not m:
        return info
    info["mount_point"] = m["mount_point"]
    info["fstype"] = m["fstype"]
    info["device"] = m["source"]
    opts = (m["mount_opts"] or "")
    info["noatime"] = "noatime" in opts
    info["relatime"] = "relatime" in opts
    info["strictatime"] = "strictatime" in opts
    block = base_block_name(info["device"] or ""); info["block_name"] = block

    if block:
        if block.startswith("nvme"):
            dev_link = f"/sys/class/block/{block}/device"
            try:
                real = os.path.realpath(dev_link)
                parts = real.split("/")
                nvme_ctrl = next((p for p in parts if p.startswith("nvme") and p[-1].isdigit()), None)
                ctrl = nvme_ctrl or "nvme0"
            except Exception:
                ctrl = "nvme0"
            info["model"]  = read_sysfs_text(f"/sys/class/nvme/{ctrl}/model")
            info["serial"] = read_sysfs_text(f"/sys/class/nvme/{ctrl}/serial")
        else:
            info["model"]  = read_sysfs_text(f"/sys/block/{block}/device/model")
            info["vendor"] = read_sysfs_text(f"/sys/block/{block}/device/vendor")

    try:
        out = subprocess.check_output(["lsblk", "-ndo", "NAME,MODEL,SERIAL,PKNAME,TYPE"], text=True).strip().splitlines()
        info["lsblk"] = out
        if (info["model"] is None or info["serial"] is None) and block:
            for line in out:
                cols = line.split(None, 5)
                if not cols:
                    continue
                name = cols[0]
                model = cols[1] if len(cols) > 1 else None
                serial = cols[2] if len(cols) > 2 else None
                pkname = cols[3] if len(cols) > 3 else None
                if name == block or pkname == block:
                    info["model"] = info["model"] or model
                    info["serial"] = info["serial"] or serial
                    break
    except Exception:
        pass
    return info

# ---------- O_NOATIME test ----------
def can_use_onoatime(sample_file: Path) -> tuple[bool, str]:
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
    if os.geteuid() != 0:
        return False
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception:
        return False

# ---------- Read methods ----------
def read_with_os_read(path: Path, bufsize: int, use_noatime: bool, fadvise: str|None, record_latencies: bool):
    # Linux-friendly, zero-temp-alloc per chunk via os.readv into a preallocated buffer
    flags = os.O_RDONLY
    if use_noatime:
        flags |= getattr(os, "O_NOATIME", 0)
    fd = os.open(path, flags)
    try:
        if HAVE_POSIX_FADVISE and fadvise:
            hint = {"SEQUENTIAL":2, "RANDOM":1, "WILLNEED":3}.get(fadvise, 0)
            if hint:
                os.posix_fadvise(fd, 0, 0, hint)

        total = 0
        buf = bytearray(bufsize)
        mv = memoryview(buf)
        t0 = time.perf_counter() if record_latencies else None

        # read loop: fill mv using readv
        while True:
            # os.readv returns number of bytes read into the buffers (0 = EOF)
            n = os.readv(fd, [mv])
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


def read_with_readinto(path: Path, bufsize: int, use_noatime: bool, fadvise: str|None, record_latencies: bool):
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
        t0 = time.perf_counter() if record_latencies else None

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


def read_with_mmap(path: Path, _bufsize: int, use_noatime: bool, fadvise: str|None, record_latencies: bool):
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
        t0 = time.perf_counter() if record_latencies else None
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

def read_with_sendfile(path: Path, _bufsize: int, use_noatime: bool, fadvise: str|None, record_latencies: bool):
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
        t0 = time.perf_counter() if record_latencies else None
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
def run_once(files: list[Path], threads: int, bufsize: int, method: str,
             fadvise: str|None, use_noatime: bool, record_latencies: bool,
             max_inflight: int|None, files_per_task: int) -> tuple[int, list[float]|None]:
    
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        headroom = 64  # keep some space for other files/sockets
        per_task_fds = 2 if method == "sendfile" else 1
        if max_inflight is not None and max_inflight > 0:
            max_allowed = max(1, (soft - headroom) // per_task_fds)
            if max_inflight > max_allowed:
                print(f"[FDGUARD] Clamping max_inflight {max_inflight} -> {max_allowed} (soft limit {soft})")
                max_inflight = max_allowed
    except Exception:
        pass

    fn = READ_METHODS[method]
    total_bytes = 0
    latencies: list[float] = []
    batches = [files[i:i+files_per_task] for i in range(0, len(files), files_per_task)]
    inflight_sem = threading.Semaphore(max_inflight if (max_inflight and max_inflight > 0) else len(batches))

    def task(batch: list[Path]):
        nonlocal total_bytes
        with inflight_sem:
            batch_lat = []
            for path in batch:
                n, lat = fn(path, bufsize, use_noatime, fadvise, record_latencies)
                total_bytes += n
                if record_latencies and lat is not None:
                    batch_lat.append(lat)
            return batch_lat

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix=f"io{threads}") as ex:
        futs = [ex.submit(task, b) for b in batches]
        for fu in as_completed(futs):
            bl = fu.result()
            if record_latencies and bl:
                latencies.extend(bl)

    return total_bytes, latencies if record_latencies else None

def summarize(total_bytes: int, elapsed: float, nfiles: int, latencies: list[float]|None):
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
def _trial_csv_header():
    return [
        "threads","bufsize","method","fadvise","noatime","mode","round",
        "nfiles","bytes","elapsed_s","mbps","files_per_s","avg_ms","p95_ms",
        "max_inflight","files_per_task"
    ]

def _emit_csv_row(writer, stats, meta):
    row = {
        "threads": meta["threads"],
        "bufsize": meta["bufsize"],
        "method": meta["method"],
        "fadvise": meta["fadvise"] or "NONE",
        "noatime": int(meta["noatime"]),
        "mode": meta["mode"],
        "round": meta["round"],
        "nfiles": meta["nfiles"],
        "bytes": meta["bytes"],
        "elapsed_s": round(meta["elapsed"], 6),
        "mbps": round(stats["mbps"], 3),
        "files_per_s": round(stats["files_per_s"], 3),
        "avg_ms": round(stats["avg_ms"], 3),
        "p95_ms": round(stats["p95_ms"], 3) if stats["p95_ms"] is not None else "",
        "max_inflight": meta["max_inflight"] or 0,
        "files_per_task": meta["files_per_task"],
    }
    writer.writerow(row)

def autotune_best_config(root: str|Path|None=None, pattern: str|None=None,
                         min_bytes: int|None=None, max_bytes: int|None=None,
                         limit: int|None=None, shuffle: bool|None=None) -> dict:
    """
    Returns dict of best settings found for warm-cache throughput (files/sec primary, avg latency tie-break).

    Parameter precedence:
      Function args (if not None) >
      Env vars (already read into Config) >
      Config defaults
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

    files = list_files(root, pattern, min_b, max_b, shuffle=shuffle)
    if limit and len(files) > limit:
        files = files[:limit]
    if not files:
        raise RuntimeError("No files matched criteria.")

    # Storage context
    ctx = detect_device_info(root)
    print("=== STORAGE CONTEXT ===")
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
        print("Drive model:   (unknown)")

    # O_NOATIME probe
    test_file = files[0]
    ono_ok, ono_reason = can_use_onoatime(test_file)
    print(f"O_NOATIME test on sample file: usable={ono_ok} ({ono_reason})")

    # CSV
    fcsv = open(cfg.CSV_FILE, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=_trial_csv_header()); writer.writeheader()    

    from statistics import mean, stdev

    def run_trials(space):
        """
        Run all combos in `space`, aggregate across repeated runs,
        and select winners by averaged warm metrics.

        Returns:
        best_mean_stats, best_meta, ranked  (ranked is a list of entries sorted best→worst)
        where each ranked item is:
        {
            "meta": {... combo knobs ...},
            "means": {"files_per_s": float, "avg_ms": float, "mbps": float,
                    "files_per_s_sd": float|None, "avg_ms_sd": float|None, "mbps_sd": float|None,
                    "count": int},
            "sel_key": (files_per_s_mean, -avg_ms_mean, mbps_mean)
        }
        """

        # ---- collect per-combo stats across rounds (warm only for selection) ----
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

                    _emit_csv_row(writer, stats, meta)
                    fcsv.flush()

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

#    THREADS_ALL   = sorted(set([1,2,3,4,6,8,12,16,24,32]))
    THREADS_ALL   = sorted(set([4]))
#    BUFSIZES_ALL  = sorted(set([4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]))
    BUFSIZES_ALL  = sorted(set([131072]))
    INFLIGHT_ALL  = sorted(set([0, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]))
#    FPT_ALL       = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024]))
    FPT_ALL       = sorted(set([256]))

    thr_c  = neigh_list(best["threads"],           THREADS_ALL)
    buf_c  = neigh_list(best["bufsize"],           BUFSIZES_ALL)
    infl_c = neigh_list(best["max_inflight"] or 0, INFLIGHT_ALL)
    fpt_c  = neigh_list(best["files_per_task"],    FPT_ALL)
    meth_c = [best["method"]]
    fav_c  = [best["fadvise"]]

    fine_space = list(itertools.product(thr_c, buf_c, meth_c, fav_c, infl_c, fpt_c))

    best_stats_final, best_final, fine_ranked = run_trials(fine_space)

    fcsv.close()

    # ---- Emit aggregated CSV (averages) from fine search ----
    agg_path = "smallfile_autotune_results_aggregated.csv"
    with open(agg_path, "w", newline="") as fagg:
        w = csv.DictWriter(fagg, fieldnames=[
            "threads","bufsize","method","fadvise","noatime",
            "max_inflight","files_per_task",
            "runs","files_per_s_mean","files_per_s_sd",
            "avg_ms_mean","avg_ms_sd","mbps_mean","mbps_sd"
        ])
        w.writeheader()
        for e in fine_ranked:
            m = e["meta"]; s = e["means"]
            w.writerow({
                "threads": m["threads"],
                "bufsize": m["bufsize"],
                "method": m["method"],
                "fadvise": m["fadvise"],
                "noatime": int(m["noatime"]),
                "max_inflight": m["max_inflight"],
                "files_per_task": m["files_per_task"],
                "runs": s["count"],
                "files_per_s_mean": round(s["files_per_s"], 3),
                "files_per_s_sd": (round(s["files_per_s_sd"], 3) if s["files_per_s_sd"] is not None else ""),
                "avg_ms_mean": round(s["avg_ms"], 3),
                "avg_ms_sd": (round(s["avg_ms_sd"], 3) if s["avg_ms_sd"] is not None else ""),
                "mbps_mean": round(s["mbps"], 3),
                "mbps_sd": (round(s["mbps_sd"], 3) if s["mbps_sd"] is not None else ""),
            })

    # ---- Print BEST and RUNNER-UP with averaged metrics ----
    print("\n=== AVERAGED WINNERS (warm) ===")
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

    print("BEST:\n" + _fmt(fine_ranked[0]))
    if len(fine_ranked) > 1:
        print("\nRUNNER-UP:\n" + _fmt(fine_ranked[1]))
    else:
        print("\nRUNNER-UP: (not available; only one combo evaluated)")
    print(f"\nAggregated CSV (means): {agg_path}")

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

    with open(Config.BEST_JSON, "w") as f:
        json.dump(chosen, f, indent=2)
    _write_best_snippet(chosen)

    print("\n=== BEST (warm, by files/sec) ===")
    print(json.dumps(chosen, indent=2))
    print(f"\nResults CSV: {Config.CSV_FILE}")
    print(f"Best JSON  : {Config.BEST_JSON}")
    print(f"Snippet    : {Config.BEST_SNIPPET}")
    return chosen

# ---------- Production snippet emitter ----------
def _write_best_snippet(best: dict):
    code = f'''# Auto-generated by smallfile_autotune.py
# Paste this into your project to replicate the fastest read path and dataset selection used.

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Tuned knobs
BEST_THREADS        = {best["threads"]}
BEST_BUFSIZE        = {best["bufsize"]}
BEST_METHOD         = "{best["method"]}"
BEST_FADVISE        = "{best["fadvise"]}"
BEST_USE_NOATIME    = {str(best["use_noatime"])}
BEST_MAX_INFLIGHT   = {best["max_inflight"] or 0}
BEST_FILES_PER_TASK = {best["files_per_task"]}

# Dataset selection (documenting what the tuner used)
BEST_ROOT    = r"{best["root"]}"
BEST_PATTERN = "{best["pattern"]}"
BEST_MIN_B   = {best["min_bytes"]}
BEST_MAX_B   = {best["max_bytes"]}
BEST_LIMIT   = {best["limit"]}
BEST_SHUFFLE = {str(best["shuffle"])}

POSIX_FADV = {{"SEQUENTIAL":2,"RANDOM":1,"WILLNEED":3,"DONTNEED":4}}

def _read_osread(path: Path):
    flags = os.O_RDONLY | (getattr(os, "O_NOATIME", 0) if BEST_USE_NOATIME else 0)
    fd = os.open(path, flags)
    try:
        if BEST_FADVISE != "NONE" and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, POSIX_FADV[BEST_FADVISE])
        out = bytearray()
        buf = bytearray(BEST_BUFSIZE); mv = memoryview(buf)
        while True:
            n = os.read(fd, mv)
            if not n: break
            out += mv[:n]
        return bytes(out)
    finally:
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, POSIX_FADV["DONTNEED"])
        except Exception:
            pass
        os.close(fd)

def _read_readinto(path: Path):
    flags = os.O_RDONLY | (getattr(os, "O_NOATIME", 0) if BEST_USE_NOATIME else 0)
    fd = os.open(path, flags); f = os.fdopen(fd, "rb", closefd=False)
    try:
        if BEST_FADVISE != "NONE" and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, POSIX_FADV[BEST_FADVISE])
        out = bytearray()
        buf = bytearray(BEST_BUFSIZE); mv = memoryview(buf)
        while True:
            n = f.readinto(mv)
            if not n: break
            out += mv[:n]
        return bytes(out)
    finally:
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, POSIX_FADV["DONTNEED"])
        except Exception:
            pass
        f.close()

def _read_mmap(path: Path):
    flags = os.O_RDONLY | (getattr(os, "O_NOATIME", 0) if BEST_USE_NOATIME else 0)
    fd = os.open(path, flags)
    try:
        if BEST_FADVISE != "NONE" and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, POSIX_FADV[BEST_FADVISE])
        st = os.fstat(fd)
        if st.st_size == 0: return b""
        import mmap
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        try:
            data = mm[:]
        finally:
            mm.close()
        return data
    finally:
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, POSIX_FADV["DONTNEED"])
        except Exception: pass
        os.close(fd)

READERS = {{
    "osread": _read_osread,
    "readinto": _read_readinto,
    "mmap": _read_mmap
}}

def fast_read_many(paths: list[Path]) -> list[bytes]:
    reader = READERS.get(BEST_METHOD, _read_osread)
    batches = [paths[i:i+BEST_FILES_PER_TASK] for i in range(0, len(paths), BEST_FILES_PER_TASK)]
    from threading import Semaphore
    sem = Semaphore(BEST_MAX_INFLIGHT if BEST_MAX_INFLIGHT>0 else len(batches))
    out = [None]*len(paths)

    def task(batch_idx, batch):
        with sem:
            result = []
            for p in batch:
                result.append(reader(Path(p)))
            return batch_idx, result

    with ThreadPoolExecutor(max_workers=BEST_THREADS) as ex:
        futs = [ex.submit(task, i, b) for i, b in enumerate(batches)]
        cursor = 0
        for fu in as_completed(futs):
            idx, res = fu.result()
            for data in res:
                out[cursor] = data; cursor += 1
    return out
'''
    with open(Config.BEST_SNIPPET, "w") as f:
        f.write(code)

# ---------- CLI (args override env which override Config defaults) ----------
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Root directory to scan (overrides SFA_ROOT/Config)")
    ap.add_argument("--pattern", default=None, help="Glob pattern, e.g. '*.wav'")
    ap.add_argument("--min-bytes", type=int, default=None)
    ap.add_argument("--max-bytes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Limit files (0 = all)")
    ap.add_argument("--shuffle", type=str, default=None, help="true/false to override shuffling")
    return ap.parse_args()

def main():
    args = _parse_cli()
    # convert shuffle str to bool if provided
    shuffle = None
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
