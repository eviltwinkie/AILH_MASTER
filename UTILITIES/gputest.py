#!/usr/bin/env python3
"""
gpu_diagnostic_suite.py

Unified GPU diagnostics, stress tests, and GEMM benchmarks.

Features:
- System & environment diagnostics (NVIDIA driver, CPU info, CUDA PATH)
- GPU inventory / capabilities (via nvidia-smi + PyTorch device props)
- TensorFlow build info + GPU visibility + compute capability analysis
- TensorFlow CPU+GPU functional tests (Conv2D, matmul, gradients) with timers
- PyTorch CPU+GPU functional tests (Conv2d, matmul, gradients) with timers
- CPU vs GPU timing comparisons & speedups
- Medium-stress GPU load (sizes derived from GPU memory, not max-out)
- TF stress Conv2D batch clamped to avoid int32 launch overflow
- TF stress matmul timings made accurate with explicit synchronization
- PTXAS diagnostics
- Resource suggestion (parallelism hints)
- GEMM benchmarks:
    * NumPy CPU fp32 (also used as CPU baseline for fp16)
    * PyTorch GPU fp32 + fp16
    * NVMath GPU fp32 + fp16 (nvmath.linalg.advanced.matmul)
  Run fp32 and fp16 back-to-back and produce combined TFLOP/s comparisons.

CLI:
  python gpu_diagnostic_suite.py
  python gpu_diagnostic_suite.py --gemm-size 12288 --gemm-iters 10 --gemm-warmup 3
  python gpu_diagnostic_suite.py --no-gemm
  python gpu_diagnostic_suite.py --no-numpy-gemm
"""

import os
import sys
import time
import textwrap
import subprocess
import shutil
import platform
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Environment: must be set BEFORE TensorFlow import to quiet logs & disable oneDNN
# -----------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide INFO & WARNING
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # disable oneDNN fusions

# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import torch
    torch.backends.cudnn.benchmark = True
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import nvmath
    from nvmath.linalg.advanced import matmul as nvm_matmul
except ImportError:
    nvmath = None
    nvm_matmul = None

# After importing TF, optionally enable memory growth to avoid grabbing all VRAM at once.
if tf is not None:
    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
        for _gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(_gpu, True)
            except Exception:
                pass
    except Exception:
        pass

# -----------------------------------------------------------------------------
# TensorFlow ↔ CUDA ↔ cuDNN compatibility & CC recommendations
# -----------------------------------------------------------------------------
TF_CUDA_CUDNN_TABLE = {
    "2.10": ("11.2", "8.1"),
    "2.11": ("11.2", "8.1"),
    "2.12": ("11.8", "8.6"),
    "2.14": ("11.8", "8.7"),
    "2.15": ("12.2", "8.9"),
    "2.16": ("12.3", "8.9"),
    "2.17": ("12.3", "8.9"),
    "2.18": ("12.5", "9.3"),
    "2.19": ("12.5", "9.3"),
    "2.20": ("12.5", "9.3"),
    "2.20.0": ("12.8.1", "9.8"),
}

TF_MAX_CC = {
    "2.10": 8.6,
    "2.11": 8.6,
    "2.12": 8.6,
    "2.13": 8.6,
    "2.14": 8.6,
    "2.15": 8.6,
    "2.16": 8.6,
    "2.17": 8.9,
    "2.18": 8.9,
    "2.19": 8.9,
    "2.20": 8.9,
    "2.20.0": 12.0,
}

CC_CUDA_TF_RECOMMEND = {
    8.6: ("12.3", "2.17"),       # Ampere
    8.9: ("12.5", "2.18"),       # Ada
    9.0: ("12.5", "2.18"),       # Hopper
    12.0: ("12.8.1", "2.20.0"),  # Blackwell
}

# Base stress iterations (we use shapes adapted to GPU) — per your request
STRESS_ITERS = 10

# GEMM defaults (for NVMath / PyTorch / NumPy benchmarks)
GEMM_SIZE_DEFAULT = 12288
GEMM_ITERS_DEFAULT = 10
GEMM_WARMUP_DEFAULT = 3

# =============================================================================
# UTILS
# =============================================================================

def hr(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return None


def human_tflops(flops_per_second: float) -> str:
    """Format FLOPs/s into TFLOP/s with 2 decimals."""
    return f"{flops_per_second / 1e12:0.2f} TFLOP/s"


def compute_gemm_flops(n: int, m: int, k: int) -> float:
    """
    FLOP count for GEMM: C[m, n] = A[m, k] @ B[k, n]
    ~ 2 * m * n * k (multiply + add)
    """
    return 2.0 * float(m) * float(n) * float(k)


# =============================================================================
# SYSTEM / DRIVER / CPU / CUDA PATH
# =============================================================================

def print_nvidia_driver() -> None:
    hr("NVIDIA Driver")
    out = run_cmd(["nvidia-smi"])
    if not out:
        print("nvidia-smi not found or driver unavailable.")
        return
    for line in out.splitlines():
        if "Driver Version" in line:
            print(line.strip())
            return
    print("Driver version not detected in nvidia-smi output.")


def print_cpu_info() -> None:
    hr("CPU Diagnostics")
    print(f"CPU cores: {os.cpu_count() or 1}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor string: {platform.processor()}")
    if cpuinfo:
        info = cpuinfo.get_cpu_info()
        print(f"CPU brand: {info.get('brand_raw', 'Unknown')}")
        print(f"CPU bits:  {info.get('bits', 'Unknown')}")
    else:
        print("Install 'py-cpuinfo' for detailed CPU info (pip install py-cpuinfo).")


def print_cuda_path() -> None:
    hr("CUDA PATH Check")
    found = False
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if "cuda" in p.lower():
            print("  ", p)
            found = True
    if not found:
        print("No CUDA directories found in PATH.")


# =============================================================================
# GPU CAPABILITIES / INVENTORY
# =============================================================================

def print_gpu_capabilities() -> List[Dict[str, Any]]:
    """
    Enumerate GPUs via nvidia-smi and PyTorch.
    """
    hr("GPU Capabilities / Inventory")

    gpu_inventory: List[Dict[str, Any]] = []

    query_fields = [
        "index",
        "name",
        "uuid",
        "compute_cap",
        "memory.total",
        "memory.used",
        "memory.free",
        "clocks.gr",
        "clocks.mem",
        "temperature.gpu",
        "pstate",
        "pci.bus_id",
        "driver_version",
        "display_active",
        "display_mode",
    ]

    cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(query_fields),
        "--format=csv,noheader,nounits",
    ]
    out = run_cmd(cmd)

    if not out:
        print("nvidia-smi not available or no NVIDIA GPU detected.")
    else:
        print("nvidia-smi GPU inventory:")
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != len(query_fields):
                continue
            data = dict(zip(query_fields, parts))

            def to_int(s: str) -> Optional[int]:
                try:
                    return int(s)
                except Exception:
                    return None

            def to_float(s: str) -> Optional[float]:
                try:
                    return float(s)
                except Exception:
                    return None

            rec: Dict[str, Any] = {
                "index": to_int(data["index"]),
                "name": data["name"],
                "uuid": data["uuid"],
                "compute_cap": to_float(data["compute_cap"]) if data["compute_cap"].lower() != "n/a" else None,
                "memory_total_mib": to_int(data["memory.total"]),
                "memory_used_mib": to_int(data["memory.used"]),
                "memory_free_mib": to_int(data["memory.free"]),
                "clocks_gr_mhz": to_int(data["clocks.gr"]),
                "clocks_mem_mhz": to_int(data["clocks.mem"]),
                "temperature_c": to_int(data["temperature.gpu"]),
                "pstate": data["pstate"],
                "pci_bus_id": data["pci.bus_id"],
                "driver_version": data["driver_version"],
                "display_active": data["display_active"],
                "display_mode": data["display_mode"],
            }
            gpu_inventory.append(rec)

            print(f"\nGPU[{data['index']}] {data['name']}")
            print("  UUID:         {}".format(data["uuid"]))
            print("  Compute Cap:  {}".format(data["compute_cap"]))
            print("  PCI Bus ID:   {}".format(data["pci.bus_id"]))
            print("  Driver Ver:   {}".format(data["driver_version"]))
            print("  Memory Total: {} MiB".format(data["memory.total"]))
            print("  Memory Used:  {} MiB".format(data["memory.used"]))
            print("  Memory Free:  {} MiB".format(data["memory.free"]))
            print("  GFX Clock:    {} MHz".format(data["clocks.gr"]))
            print("  Mem Clock:    {} MHz".format(data["clocks.mem"]))
            print("  Temperature:  {} C".format(data["temperature.gpu"]))
            print("  Power State:  {}".format(data["pstate"]))
            print("  Display:      active={} mode={}".format(
                data["display_active"], data["display_mode"]
            ))

    # PyTorch device properties
    if torch is not None and torch.cuda.is_available():
        print("\nPyTorch CUDA device properties:")
        num = torch.cuda.device_count()

        for idx in range(num):
            props = torch.cuda.get_device_properties(idx)

            def p_attr(attr_name: str, label: str) -> None:
                if hasattr(props, attr_name):
                    val = getattr(props, attr_name)
                    print(f"  {label}: {val}")

            total_gb: Optional[float] = None
            if hasattr(props, "total_memory"):
                total_gb = props.total_memory / (1024 ** 3)

            print(f"\nCUDA Device [{idx}] — {props.name}")
            if hasattr(props, "pci_bus_id"):
                print(f"  PCI Bus ID:               {props.pci_bus_id}")
            print(f"  Compute Capability:       {props.major}.{props.minor}")
            if total_gb is not None:
                print(f"  Total Memory:             {total_gb:0.2f} GiB")

            p_attr("multi_processor_count", "Multi-Processor Count")
            if hasattr(props, "max_threads_per_block"):
                p_attr("max_threads_per_block", "Max Threads per Block")
            if hasattr(props, "max_threads_per_multi_processor"):
                p_attr("max_threads_per_multi_processor", "Max Threads per SM")
            elif hasattr(props, "max_threads_per_multiprocessor"):
                p_attr("max_threads_per_multiprocessor", "Max Threads per SM")
            p_attr("warp_size", "Warp Size")
            p_attr("shared_memory_per_block", "Shared Mem per Block (bytes)")
            p_attr("regs_per_block", "Registers per Block")
            p_attr("max_grid_size", "Max Grid Size (threads_dim)")
            p_attr("max_threads_dim", "Max Block Size (threads_dim)")
    else:
        print("\nPyTorch CUDA device properties: PyTorch or CUDA not available from PyTorch.")

    return gpu_inventory


# =============================================================================
# PTXAS TESTS
# =============================================================================

def test_ptxas() -> None:
    hr("PTXAS Diagnostics")

    ptxas_path = which("ptxas")
    if not ptxas_path:
        print("❌ ptxas not found in PATH.")
        print("   - CUDA toolkit (compiler) is not installed or not on PATH.")
        print("   - Required only for custom CUDA kernels / source builds.")
        return

    print(f"✅ ptxas found at: {ptxas_path}")
    t0 = time.time()
    try:
        out = subprocess.check_output(
            ["ptxas", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        t1 = time.time()
        print("ptxas --version output:")
        print(textwrap.indent(out.strip(), "  "))
        print(f"✅ ptxas version query OK. Time: {(t1 - t0)*1000:.2f} ms")
    except subprocess.CalledProcessError as e:
        t1 = time.time()
        print("❌ ptxas --version returned non-zero exit status.")
        print("   Output:", e.output)
        print(f"Time until failure: {(t1 - t0)*1000:.2f} ms")
    except FileNotFoundError:
        t1 = time.time()
        print("❌ ptxas disappeared from PATH unexpectedly.")
        print(f"Time until failure: {(t1 - t0)*1000:.2f} ms")


# =============================================================================
# TF INFO / GPU ENUM / CC ANALYSIS
# =============================================================================

def print_tf_info() -> None:
    hr("TensorFlow Build Info")
    if tf is None:
        print("TensorFlow not installed.")
        return

    tf_version = tf.__version__
    print(f"TensorFlow version: {tf_version}")
    try:
        build = tf.sysconfig.get_build_info()
        print(f"Built with CUDA:  {tf.test.is_built_with_cuda()}")
        print(f"CUDA runtime:     {build.get('cuda_version', 'Unknown')}")
        print(f"cuDNN runtime:    {build.get('cudnn_version', 'Unknown')}")
    except Exception:
        print("Could not query TensorFlow build info.")


def print_tf_gpus():
    hr("TensorFlow GPU Detection")
    if tf is None:
        print("TensorFlow not installed — no GPU detection.")
        return []
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs detected by TensorFlow: {len(gpus)}")
        for idx, gpu in enumerate(gpus):
            print(f"  [{idx}] {gpu}")
    else:
        print("TensorFlow reports: NO GPU FOUND.")
    return gpus


def detect_cc_from_nvidia_smi():
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,compute_cap",
        "--format=csv,noheader"
    ])
    results = []
    if not out:
        return results
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, cc_str = parts
        if cc_str.lower() == "n/a":
            continue
        try:
            cc_val = float(cc_str)
            results.append((name, cc_val))
        except ValueError:
            continue
    return results


def analyze_gpu(gpus) -> None:
    hr("Compute Capability Analysis")
    if tf is None:
        print("TensorFlow not installed; skipping TF CC analysis.")
        return

    tf_version = tf.__version__
    tf_major_minor = ".".join(tf_version.split(".")[:2])
    if tf_version == "2.20.0-dev0+selfbuilt":
        tf_major_minor = "2.20.0"
    supported_cc = TF_MAX_CC.get(tf_major_minor, max(TF_MAX_CC.values()))
    print(f"TF Native CC Supported (approx): {supported_cc}")

    smi_cc_list = detect_cc_from_nvidia_smi()

    if not gpus and not smi_cc_list:
        print("No GPU visible to TF or nvidia-smi; skipping CC analysis.")
        return

    if smi_cc_list:
        print("\nCompute capabilities from nvidia-smi:")
        for idx, (name, cc) in enumerate(smi_cc_list):
            print(f"  [{idx}] {name} — Compute Capability {cc}")
    else:
        print("\nnvidia-smi did not report compute capabilities (or not available).")

    for idx, gpu in enumerate(gpus or []):
        details = tf.config.experimental.get_device_details(gpu)
        tf_cc = details.get("compute_capability", None)
        tf_name = details.get("device_name", "Unknown GPU")

        if smi_cc_list and idx < len(smi_cc_list):
            smi_name, smi_cc = smi_cc_list[idx]
            cc_float = smi_cc
            name = smi_name
            source = "nvidia-smi"
        elif tf_cc:
            cc_float = float(f"{tf_cc[0]}.{tf_cc[1]}")
            name = tf_name
            source = "TensorFlow"
        else:
            print(f"\nGPU[{idx}]: {tf_name} — no compute capability info.")
            continue

        print(f"\nGPU[{idx}] ({source}): {name} — Compute Capability {cc_float}")

        if cc_float > supported_cc:
            print(
                f"⚠️  WARNING: GPU CC {cc_float} > TensorFlow {tf_version} native support (≈CC {supported_cc}).\n"
                "   TensorFlow will rely on PTX JIT. Expect slower startup for first GPU ops."
            )

        best = None
        for cc_req, (cuda_ver, tf_ver) in sorted(CC_CUDA_TF_RECOMMEND.items()):
            if cc_float >= cc_req:
                best = (cuda_ver, tf_ver)

        if best:
            cuda_ver, tf_ver = best
            cuda_tbl, cudnn_tbl = TF_CUDA_CUDNN_TABLE.get(tf_ver, (cuda_ver, "?"))
            print(f"\nRecommended pairing for CC {cc_float}:")
            print(f"  - TensorFlow {tf_ver}")
            print(f"  - CUDA {cuda_tbl}")
            print(f"  - cuDNN {cudnn_tbl}")

            if tf_ver == "2.20.0":
                print(textwrap.dedent(
                    """
                    Suggested (destructive) clean reinstall sequence:

                        sudo apt remove --purge '*cuda' 'cuda*' '*cuda*' '*nvidia' 'nvidia*' '*nvidia*'
                        sudo apt autoremove --purge -y
                        sudo apt clean

                        deactivate
                        rm -rf ~/venvs/mlenv
                        rm -rf ~/.nv
                        rm -rf ~/.cache/cuda*
                        rm -rf ~/.cache/torch
                        rm -rf ~/.cache/cupy
                        rm -rf ~/.local/lib/python*/site-packages/nvidia*

                        python3 -m venv ~/venvs/mlenv
                        source ~/venvs/mlenv/bin/activate

                        pip install --upgrade pip setuptools wheel

                        wget https://github.com/mypapit/tensorflowRTX50/releases/download/2.20dev-ubuntu-24.04-avx-too/tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl
                        pip install --force-reinstall tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl seaborn pandas matplotlib opencv-python pillow imutils pydot graphviz librosa

                        wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
                        sudo sh cuda_12.8.1_570.124.06_linux.run

                        echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
                        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
                        echo "/usr/local/cuda-12.8/lib64" | sudo tee /etc/ld.so.conf.d/cuda-12-8.conf
                        sudo ldconfig

                        wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2404-9.8.0_1.0-1_amd64.deb
                        sudo dpkg -i cudnn-local-repo-ubuntu2404-9.8.0_1.0-1_amd64.deb 
                        sudo cp /var/cudnn-local-repo-ubuntu2404-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/

                        sudo apt update
                        sudo apt install cudnn torch
                        
                        pip install nvmath-python tensorrt

                    """
                ).rstrip())
        else:
            print("No recommendation entry for this GPU class.")


# =============================================================================
# ADAPTIVE STRESS SHAPES (MEDIUM-STRESS PROFILE)
# =============================================================================

def derive_stress_shapes(gpu_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Derive stress-test shapes based on GPU memory and compute capability.
    Medium-stress profile for ~8–10 GiB peak on a 24 GiB GPU.
    """
    conv_batch = 8
    conv_hw = 256
    conv_in_ch = 4
    conv_out_ch = 32
    mm_size = 4096

    if not gpu_info:
        return {
            "conv_batch": conv_batch,
            "conv_hw": conv_hw,
            "conv_in_ch": conv_in_ch,
            "conv_out_ch": conv_out_ch,
            "mm_size": mm_size,
        }

    free_mib = gpu_info.get("memory_free_mib") or gpu_info.get("memory_total_mib") or 8000
    cc = gpu_info.get("compute_cap") or 7.5

    if free_mib < 8000:
        conv_batch = 4
        conv_hw = 224
        conv_in_ch = 3
        conv_out_ch = 32
        mm_size = 3072
    elif free_mib < 16000:
        conv_batch = 8
        conv_hw = 256
        conv_in_ch = 4
        conv_out_ch = 48
        mm_size = 6144
    else:
        conv_batch = 16
        conv_hw = 384
        conv_in_ch = 4
        conv_out_ch = 64
        mm_size = 12288

    if cc >= 12.0 and free_mib >= 20000:
        # Blackwell-class with plenty of VRAM; current profile is "medium" already.
        pass

    return {
        "conv_batch": conv_batch,
        "conv_hw": conv_hw,
        "conv_in_ch": conv_in_ch,
        "conv_out_ch": conv_out_ch,
        "mm_size": mm_size,
    }


def clamp_tf_conv_batch_for_launch_limit(
    conv_batch: int,
    conv_hw: int,
    conv_out_ch: int,
    safety_limit: int = 2_000_000_000,
) -> Tuple[int, int, float]:
    per_batch_elems = conv_hw * conv_hw * conv_out_ch
    if per_batch_elems <= 0:
        work_elems = 0
        return max(1, conv_batch), work_elems, 0.0

    max_batch = safety_limit // per_batch_elems
    if max_batch < 1:
        clamped = 1
    else:
        clamped = min(conv_batch, max_batch)

    work_elems = clamped * per_batch_elems
    frac = float(work_elems) / float(safety_limit) if safety_limit > 0 else 0.0
    return clamped, work_elems, frac


# =============================================================================
# TF / TORCH TEST SUITES
# =============================================================================

def tf_sync() -> None:
    if tf is None:
        return
    if hasattr(tf.experimental, "async_wait"):
        try:
            tf.experimental.async_wait()  # best-effort
        except Exception:
            pass


def run_tensorflow_suite(gpus, gpu_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    hr("TensorFlow Diagnostics + Stress Suite")

    results: Dict[str, Any] = {
        "installed": False,
        "gpu_available": False,
        "gpu_conv_ok": False,
        "gpu_matmul_ok": False,
        "gpu_grad_ok": False,
        "gpu_stress_ok": False,
    }

    if tf is None:
        print("❌ TensorFlow is not installed.")
        return results

    results["installed"] = True

    cpus = tf.config.list_physical_devices("CPU")
    print(f"CPUs visible to TF: {len(cpus)}")

    if not gpus:
        print("No GPU detected by TensorFlow; running CPU-only tests.")
    else:
        print(f"GPUs visible to TF: {len(gpus)}")

    conv_shape = (16, 64, 64, 3)
    mm_shape = (1024, 1024)
    grad_shape = (8, 64, 64, 3)

    cpu_conv_ms = cpu_mm_ms = cpu_grad_ms = None
    gpu_conv_ms = gpu_mm_ms = gpu_grad_ms = None

    # ------------------------------ CPU smoke tests ----------------------------
    print("\n[TF] CPU smoke tests (Conv2D, matmul, gradients)")

    # CPU Conv2D
    try:
        with tf.device("/CPU:0"):
            x = tf.random.normal(conv_shape)
            conv = tf.keras.layers.Conv2D(32, 3, padding="same")
            _ = conv(x)  # warm-up
            t0 = time.time()
            y = conv(x)
            t1 = time.time()
        cpu_conv_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU Conv2D OK. Time: {cpu_conv_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU Conv2D FAILED:", repr(e))

    # CPU matmul
    try:
        with tf.device("/CPU:0"):
            a = tf.random.normal(mm_shape)
            b = tf.random.normal(mm_shape)
            t0 = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()
            t1 = time.time()
        cpu_mm_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU matmul OK. Time: {cpu_mm_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU matmul FAILED:", repr(e))

    # CPU gradient
    try:
        with tf.device("/CPU:0"):
            x = tf.random.normal(grad_shape)
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(2),
                tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(10),
            ])
            t0 = time.time()
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            grads = tape.gradient(loss, model.trainable_variables)
            t1 = time.time()
            if any(g is None for g in grads):
                raise RuntimeError("Some gradients are None")
        cpu_grad_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU gradient test OK. Time: {cpu_grad_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU gradient test FAILED:", repr(e))

    if not gpus:
        print("\n[TF] No GPU detected; skipping TF GPU tests and stress.")
        return results

    results["gpu_available"] = True
    tf_device = "/GPU:0"

    # ------------------------------ GPU smoke tests ----------------------------
    print("\n[TF] GPU smoke tests (Conv2D, matmul, gradients)")

    # GPU Conv2D
    try:
        with tf.device(tf_device):
            x = tf.random.normal(conv_shape)
            conv = tf.keras.layers.Conv2D(32, 3, padding="same")
            _ = conv(x)  # warm-up
            tf_sync()
            t0 = time.time()
            y = conv(x)
            tf_sync()
            _ = y.numpy()  # force sync
            t1 = time.time()

        gpu_conv_ms = (t1 - t0) * 1000.0
        device_str = getattr(y, "device", "") or ""
        print("Conv2D output device:", device_str)
        print(f"✅ GPU Conv2D OK. Time: {gpu_conv_ms:.2f} ms")
        results["gpu_conv_ok"] = True
    except Exception as e:
        print("❌ GPU Conv2D FAILED:", repr(e))

    # GPU matmul (with warmup + explicit sync)
    try:
        with tf.device(tf_device):
            a = tf.random.normal(mm_shape)
            b = tf.random.normal(mm_shape)

            # Warmup a few times so we don't benchmark first-use overhead only
            for _ in range(3):
                _ = tf.matmul(a, b)

            tf_sync()
            t0 = time.time()
            c = tf.matmul(a, b)
            tf_sync()
            _ = c.numpy()  # explicit sync
            t1 = time.time()

        gpu_mm_ms = (t1 - t0) * 1000.0
        print(f"✅ GPU matmul OK. Time: {gpu_mm_ms:.2f} ms")
        results["gpu_matmul_ok"] = True
    except Exception as e:
        print("❌ GPU matmul FAILED:", repr(e))

    # GPU gradient (with warmup-ish pattern)
    try:
        with tf.device(tf_device):
            x = tf.random.normal(grad_shape)
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(2),
                tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(10),
            ])

            # One warmup forward/backward
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            _ = tape.gradient(loss, model.trainable_variables)

            tf_sync()
            t0 = time.time()
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            grads = tape.gradient(loss, model.trainable_variables)
            tf_sync()
            t1 = time.time()
            if any(g is None for g in grads):
                raise RuntimeError("Some gradients are None")

        gpu_grad_ms = (t1 - t0) * 1000.0
        print(f"✅ GPU gradient test OK. Time: {gpu_grad_ms:.2f} ms")
        results["gpu_grad_ok"] = True
    except Exception as e:
        print("❌ GPU gradient test FAILED:", repr(e))

    # ---------------------------- CPU vs GPU compare ---------------------------
    print("\n[TF] CPU vs GPU comparison (same shapes):")
    def cmp(label: str, cpu_ms, gpu_ms) -> None:
        if cpu_ms is not None and gpu_ms is not None and gpu_ms > 0:
            speedup = cpu_ms / gpu_ms
            print(f"  {label}: CPU {cpu_ms:.2f} ms vs GPU {gpu_ms:.2f} ms → {speedup:.1f}x faster")
        else:
            print(f"  {label}: timing not available for both CPU and GPU")

    cmp("Conv2D", cpu_conv_ms, gpu_conv_ms)
    cmp("Matmul", cpu_mm_ms,  gpu_mm_ms)
    cmp("Gradient", cpu_grad_ms, gpu_grad_ms)

    # ----------------------------- TF GPU STRESS -------------------------------
    if results["gpu_conv_ok"] and results["gpu_matmul_ok"] and results["gpu_grad_ok"]:
        stress_cfg = derive_stress_shapes(gpu_info)
        conv_batch = stress_cfg["conv_batch"]
        conv_hw = stress_cfg["conv_hw"]
        conv_in_ch = stress_cfg["conv_in_ch"]
        conv_out_ch = stress_cfg["conv_out_ch"]
        mm_n = stress_cfg["mm_size"]

        safe_batch, work_elems, frac = clamp_tf_conv_batch_for_launch_limit(
            conv_batch=conv_batch,
            conv_hw=conv_hw,
            conv_out_ch=conv_out_ch,
            safety_limit=2_000_000_000,
        )

        if safe_batch != conv_batch:
            print(
                f"[TF] Adjusting Conv2D stress batch from {conv_batch} to {safe_batch} "
                f"to stay within TF int32 launch limits "
                f"(H=W={conv_hw}, C_out={conv_out_ch})."
            )
        conv_batch = safe_batch

        print(
            f"\n[TF] GPU stress test (adapted, medium) — "
            f"Conv2D: batch={conv_batch}, H=W={conv_hw}, C_in={conv_in_ch}, C_out={conv_out_ch}; "
            f"Matmul: {mm_n} x {mm_n}; Iters={STRESS_ITERS}"
        )
        print(
            f"[TF] Conv2D work elements: {work_elems} "
            f"({frac*100.0:.2f}% of 2,000,000,000 limit)"
        )

        try:
            with tf.device(tf_device):
                # Conv2D stress
                x = tf.random.normal([conv_batch, conv_hw, conv_hw, conv_in_ch])
                conv = tf.keras.layers.Conv2D(
                    filters=conv_out_ch,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    use_bias=True,
                )

                tf_sync()
                t0 = time.time()
                for _ in range(STRESS_ITERS):
                    y = conv(x)
                tf_sync()
                _ = y.numpy()  # force sync after loop
                t1 = time.time()

                conv_time_ms = (t1 - t0) * 1000.0
                print(
                    f"  Conv2D stress: {STRESS_ITERS} iters, total {conv_time_ms:.2f} ms, "
                    f"avg {conv_time_ms / STRESS_ITERS:.2f} ms/iter"
                )

                # Matmul stress (with proper sync)
                a = tf.random.normal([mm_n, mm_n])
                b = tf.random.normal([mm_n, mm_n])

                # One warmup
                _ = tf.matmul(a, b)

                tf_sync()
                t2 = time.time()
                for _ in range(STRESS_ITERS):
                    c = tf.matmul(a, b)
                tf_sync()
                _ = c.numpy()  # force sync after loop
                t3 = time.time()

                mm_time_ms = (t3 - t2) * 1000.0
                print(
                    f"  Matmul stress: {STRESS_ITERS} iters, total {mm_time_ms:.2f} ms, "
                    f"avg {mm_time_ms / STRESS_ITERS:.2f} ms/iter"
                )

            print("✅ TensorFlow GPU stress test (medium) completed without errors.")
            results["gpu_stress_ok"] = True

        except Exception as e:
            print("❌ TensorFlow GPU stress test FAILED:", repr(e))
    else:
        print("\n[TF] Skipping stress test because some GPU smoke tests failed.")

    return results


def run_pytorch_suite(gpu_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    hr("PyTorch Diagnostics + Stress Suite")

    results: Dict[str, Any] = {
        "installed": False,
        "gpu_available": False,
        "gpu_conv_ok": False,
        "gpu_matmul_ok": False,
        "gpu_grad_ok": False,
        "gpu_stress_ok": False,
    }

    if torch is None:
        print("❌ PyTorch is not installed.")
        return results

    results["installed"] = True

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: True" if torch.cuda.is_available() else "CUDA available: False")

    conv_shape = (16, 3, 64, 64)
    mm_shape = (1024, 1024)
    grad_shape = (8, 3, 64, 64)

    cpu_conv_ms = cpu_mm_ms = cpu_grad_ms = None
    gpu_conv_ms = gpu_mm_ms = gpu_grad_ms = None

    # CPU tests
    print("\n[PyTorch] CPU smoke tests (Conv2d, matmul, gradients)")
    try:
        x = torch.randn(*conv_shape)
        conv = torch.nn.Conv2d(3, 32, 3, padding=1)
        t0 = time.time()
        y = conv(x)
        t1 = time.time()
        cpu_conv_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU Conv2d OK. Time: {cpu_conv_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU Conv2d FAILED:", repr(e))

    try:
        a = torch.randn(*mm_shape)
        b = torch.randn(*mm_shape)
        t0 = time.time()
        c = a @ b
        _ = c.sum().item()
        t1 = time.time()
        cpu_mm_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU matmul OK. Time: {cpu_mm_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU matmul FAILED:", repr(e))

    try:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32 * 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )
        x = torch.randn(*grad_shape)
        t0 = time.time()
        out = model(x)
        loss = out.mean()
        loss.backward()
        t1 = time.time()
        if not all(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Some gradients are None")
        cpu_grad_ms = (t1 - t0) * 1000.0
        print(f"✅ CPU gradient test OK. Time: {cpu_grad_ms:.2f} ms")
    except Exception as e:
        print("❌ CPU gradient test FAILED:", repr(e))

    if not torch.cuda.is_available():
        print("\n[PyTorch] CUDA not available; skipping GPU tests and stress.")
        return results

    results["gpu_available"] = True
    device = torch.device("cuda:0")
    print(f"\n[PyTorch] Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(device)}")

    # GPU tests
    print("\n[PyTorch] GPU smoke tests (Conv2d, matmul, gradients)")
    try:
        x = torch.randn(*conv_shape, device=device)
        conv = torch.nn.Conv2d(3, 32, 3, padding=1).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        y = conv(x)
        torch.cuda.synchronize()
        t1 = time.time()
        gpu_conv_ms = (t1 - t0) * 1000.0
        print(f"✅ GPU Conv2d OK. Time: {gpu_conv_ms:.2f} ms")
        print("Conv2d output shape:", tuple(y.shape))
        results["gpu_conv_ok"] = True
    except Exception as e:
        print("❌ GPU Conv2d FAILED:", repr(e))

    try:
        a = torch.randn(*mm_shape, device=device)
        b = torch.randn(*mm_shape, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        t1 = time.time()
        _ = c.sum().item()
        gpu_mm_ms = (t1 - t0) * 1000.0
        print(f"✅ GPU matmul OK. Time: {gpu_mm_ms:.2f} ms")
        results["gpu_matmul_ok"] = True
    except Exception as e:
        print("❌ GPU matmul FAILED:", repr(e))

    try:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32 * 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        ).to(device)
        x = torch.randn(*grad_shape, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(x)
        loss = out.mean()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        if not all(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Some gradients are None")
        gpu_grad_ms = (t1 - t0) * 1000.0
        print(f"✅ GPU gradient test OK. Time: {gpu_grad_ms:.2f} ms")
        results["gpu_grad_ok"] = True
    except Exception as e:
        print("❌ GPU gradient test FAILED:", repr(e))

    print("\n[PyTorch] CPU vs GPU comparison (same shapes):")
    def cmp(label: str, cpu_ms, gpu_ms) -> None:
        if cpu_ms is not None and gpu_ms is not None and gpu_ms > 0:
            speedup = cpu_ms / gpu_ms
            print(f"  {label}: CPU {cpu_ms:.2f} ms vs GPU {gpu_ms:.2f} ms → {speedup:.1f}x faster")
        else:
            print(f"  {label}: timing not available for both CPU and GPU")
    cmp("Conv2d", cpu_conv_ms, gpu_conv_ms)
    cmp("Matmul", cpu_mm_ms, gpu_mm_ms)
    cmp("Gradient", cpu_grad_ms, gpu_grad_ms)

    # PyTorch stress
    if results["gpu_conv_ok"] and results["gpu_matmul_ok"] and results["gpu_grad_ok"]:
        stress_cfg = derive_stress_shapes(gpu_info)
        conv_batch = stress_cfg["conv_batch"]
        conv_hw = stress_cfg["conv_hw"]
        conv_in_ch = stress_cfg["conv_in_ch"]
        conv_out_ch = stress_cfg["conv_out_ch"]
        mm_n = stress_cfg["mm_size"]

        print(
            f"\n[PyTorch] GPU stress test (adapted, medium) — "
            f"Conv2d: N={conv_batch}, C_in={conv_in_ch}, H=W={conv_hw}, C_out={conv_out_ch}; "
            f"Matmul: {mm_n} x {mm_n}; Iters={STRESS_ITERS}"
        )
        try:
            x = torch.randn(conv_batch, conv_in_ch, conv_hw, conv_hw, device=device)
            conv = torch.nn.Conv2d(
                in_channels=conv_in_ch,
                out_channels=conv_out_ch,
                kernel_size=3,
                padding=1,
                bias=True,
            ).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(STRESS_ITERS):
                y = conv(x)
            torch.cuda.synchronize()
            t1 = time.time()
            conv_time_ms = (t1 - t0) * 1000.0
            print(
                f"  Conv2d stress: {STRESS_ITERS} iters, total {conv_time_ms:.2f} ms, "
                f"avg {conv_time_ms / STRESS_ITERS:.2f} ms/iter"
            )

            a = torch.randn(mm_n, mm_n, device=device)
            b = torch.randn(mm_n, mm_n, device=device)
            torch.cuda.synchronize()
            t2 = time.time()
            for _ in range(STRESS_ITERS):
                c = a @ b
            torch.cuda.synchronize()
            t3 = time.time()
            _ = c.sum().item()
            mm_time_ms = (t3 - t2) * 1000.0
            print(
                f"  Matmul stress: {STRESS_ITERS} iters, total {mm_time_ms:.2f} ms, "
                f"avg {mm_time_ms / STRESS_ITERS:.2f} ms/iter"
            )

            print("✅ PyTorch GPU stress test (medium) completed without errors.")
            results["gpu_stress_ok"] = True
        except Exception as e:
            print("❌ PyTorch GPU stress test FAILED:", repr(e))
    else:
        print("\n[PyTorch] Skipping stress test due to earlier GPU failures.")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# =============================================================================
# NVMath / GEMM BENCHMARK SUITE (NumPy, PyTorch, NVMath; fp32 + fp16)
# =============================================================================

def bench_numpy_gemm(n: int, dtype: str, iters: int, warmup: int) -> float:
    """
    CPU baseline using NumPy (BLAS).

    - For dtype == "fp32": true fp32.
    - For dtype == "fp16": we UPCAST to fp32 for compute, and label accordingly.
      This avoids extremely slow or poorly optimized CPU fp16 paths while still
      giving you a rough CPU baseline for the same problem size.
    """
    if dtype == "fp32":
        np_dtype = np.float32
        label = "fp32"
    elif dtype == "fp16":
        np_dtype = np.float32
        label = "fp32 (CPU baseline for fp16)"
    else:
        raise ValueError(f"Unsupported dtype for NumPy: {dtype}")

    print(f"\n[NumPy] GEMM benchmark (CPU) — size={n}x{n}, dtype={label}, "
          f"iters={iters}, warmup={warmup}")

    A = np.random.randn(n, n).astype(np_dtype)
    B = np.random.randn(n, n).astype(np_dtype)

    # Warmup
    for _ in range(warmup):
        C = A @ B

    flops_per_iter = compute_gemm_flops(n, n, n)

    t0 = time.time()
    for _ in range(iters):
        C = A @ B
    t1 = time.time()

    elapsed = t1 - t0
    if elapsed <= 0:
        return float("nan")

    total_flops = flops_per_iter * iters
    flops_per_sec = total_flops / elapsed

    print(f"  Total time: {elapsed * 1000.0:0.2f} ms")
    print(f"  Per-iter time: {elapsed * 1000.0 / iters:0.2f} ms")
    print(f"  Average rate: {human_tflops(flops_per_sec)}")

    return flops_per_sec


def bench_torch_gemm(n: int, dtype: str, iters: int, warmup: int,
                     device: str = "cuda") -> float:
    """
    Benchmark PyTorch GEMM C = A @ B with given size and dtype on CPU or GPU.
    Returns achieved FLOP/s.
    """
    if torch is None:
        print("[PyTorch] NOT installed; skipping PyTorch GEMM benchmark.")
        return float("nan")

    if device == "cuda" and not torch.cuda.is_available():
        print("[PyTorch] CUDA is not available; skipping GPU GEMM benchmark.")
        return float("nan")

    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype for PyTorch: {dtype}")

    dev = torch.device("cuda:0" if device == "cuda" else "cpu")

    print(f"\n[PyTorch] GEMM benchmark on {dev} — size={n}x{n}, "
          f"dtype={dtype}, iters={iters}, warmup={warmup}")

    A = torch.randn(n, n, device=dev, dtype=torch_dtype)
    B = torch.randn(n, n, device=dev, dtype=torch_dtype)

    # Warmup
    for _ in range(warmup):
        C = A @ B
        if device == "cuda":
            torch.cuda.synchronize()

    flops_per_iter = compute_gemm_flops(n, n, n)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        C = A @ B
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    if elapsed <= 0:
        return float("nan")

    total_flops = flops_per_iter * iters
    flops_per_sec = total_flops / elapsed

    print(f"  Total time: {elapsed * 1000.0:0.2f} ms")
    print(f"  Per-iter time: {elapsed * 1000.0 / iters:0.2f} ms")
    print(f"  Average rate: {human_tflops(flops_per_sec)}")

    return flops_per_sec


def bench_nvmath_gemm(n: int, dtype: str, iters: int, warmup: int) -> float:
    """
    Benchmark NVMath GEMM using nvmath.linalg.advanced.matmul on GPU.

    We allocate A, B as PyTorch CUDA tensors and call nvm_matmul(A, B)
    which operates on PyTorch tensors and returns a tensor of the same
    type/device.
    """
    if nvmath is None or nvm_matmul is None:
        print("[NVMath] nvmath-python not installed; skipping NVMath GEMM benchmark.")
        return float("nan")

    if torch is None or not torch.cuda.is_available():
        print("[NVMath] Requires PyTorch with CUDA for this benchmark; skipping.")
        return float("nan")

    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype for NVMath: {dtype}")

    dev = torch.device("cuda:0")
    print(f"\n[NVMath] GEMM benchmark via nvmath.linalg.advanced.matmul — "
          f"size={n}x{n}, dtype={dtype}, iters={iters}, warmup={warmup}")

    A = torch.randn(n, n, device=dev, dtype=torch_dtype)
    B = torch.randn(n, n, device=dev, dtype=torch_dtype)

    # Warmup
    for _ in range(warmup):
        C = nvm_matmul(A, B)
        torch.cuda.synchronize()

    flops_per_iter = compute_gemm_flops(n, n, n)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        C = nvm_matmul(A, B)
    torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    if elapsed <= 0:
        return float("nan")

    total_flops = flops_per_iter * iters
    flops_per_sec = total_flops / elapsed

    print(f"  Total time: {elapsed * 1000.0:0.2f} ms")
    print(f"  Per-iter time: {elapsed * 1000.0 / iters:0.2f} ms")
    print(f"  Average rate: {human_tflops(flops_per_sec)}")

    return flops_per_sec


def run_gemm_suite_for_dtype(
    n: int,
    iters: int,
    warmup: int,
    dtype: str,
    use_numpy: bool,
) -> Dict[str, Any]:
    """
    Run NumPy, PyTorch, and NVMath GEMM benchmarks for a single dtype.
    Returns:
        {
            "dtype": "fp32" or "fp16",
            "numpy": <FLOP/s (float) or NaN>,
            "torch": <FLOP/s>,
            "nvmath": <FLOP/s>,
        }
    """
    results = {
        "dtype": dtype,
        "numpy": float("nan"),
        "torch": float("nan"),
        "nvmath": float("nan"),
    }

    print("\n" + "=" * 80)
    print(f"Running {dtype} GEMM benchmarks")
    print("=" * 80)

    # NumPy CPU baseline
    if use_numpy:
        numpy_flops = bench_numpy_gemm(n, dtype, iters=iters, warmup=warmup)
        results["numpy"] = numpy_flops

    # PyTorch GPU GEMM
    torch_flops = bench_torch_gemm(n, dtype, iters=iters, warmup=warmup, device="cuda")
    results["torch"] = torch_flops

    # NVMath GPU GEMM
    nvm_flops = bench_nvmath_gemm(n, dtype, iters=iters, warmup=warmup)
    results["nvmath"] = nvm_flops

    print("\n" + "-" * 80)
    print(f"{dtype} GEMM Summary (TFLOP/s)")
    print("-" * 80)
    if not math.isnan(results["numpy"]):
        print(f"NumPy (CPU):    {human_tflops(results['numpy'])}")
    else:
        print("NumPy (CPU):    N/A")

    if not math.isnan(results["torch"]):
        print(f"PyTorch (GPU):  {human_tflops(results['torch'])}")
    else:
        print("PyTorch (GPU):  N/A")

    if not math.isnan(results["nvmath"]):
        print(f"NVMath (GPU):   {human_tflops(results['nvmath'])}")
    else:
        print("NVMath (GPU):   N/A")

    return results


def run_gemm_benchmarks(
    n: int,
    iters: int,
    warmup: int,
    use_numpy: bool,
) -> None:
    """
    Run fp32 and fp16 GEMM suites back-to-back and print combined summary.
    """
    hr("GEMM / NVMath Benchmarks (NumPy, PyTorch, NVMath; fp32 + fp16)")
    print(f"Matrix size: {n} x {n}")
    print(f"Timed iters: {iters}, Warmup: {warmup}")
    print(f"NumPy baseline in GEMM: {'enabled' if use_numpy else 'disabled'}")

    fp32_results = run_gemm_suite_for_dtype(n, iters, warmup, dtype="fp32", use_numpy=use_numpy)
    fp16_results = run_gemm_suite_for_dtype(n, iters, warmup, dtype="fp16", use_numpy=use_numpy)

    print("\n" + "=" * 80)
    print("Combined GEMM Summary (TFLOP/s)")
    print("=" * 80)
    print(f"{'Backend':<12} {'fp32':>12} {'fp16':>12} {'fp16/fp32':>12}")
    print("-" * 80)

    def row(name: str, key: str):
        v32 = fp32_results.get(key, float("nan"))
        v16 = fp16_results.get(key, float("nan"))
        if not math.isnan(v32):
            s32 = human_tflops(v32)
        else:
            s32 = "N/A"
        if not math.isnan(v16):
            s16 = human_tflops(v16)
        else:
            s16 = "N/A"
        if (not math.isnan(v32)) and (not math.isnan(v16)) and v32 > 0:
            speed = v16 / v32
            s_speed = f"{speed:0.2f}x"
        else:
            s_speed = "N/A"
        print(f"{name:<12} {s32:>12} {s16:>12} {s_speed:>12}")

    row("NumPy (CPU)", "numpy")
    row("PyTorch", "torch")
    row("NVMath", "nvmath")

    print("\nGEMM benchmarks done.")


# =============================================================================
# RESOURCE SUMMARY / MAIN
# =============================================================================

def print_resources(has_gpu: bool) -> None:
    hr("Resource / Parallelism Suggestion")
    cpu_count = os.cpu_count() or 1
    optuna_jobs = 1 if has_gpu else max(cpu_count // 3, 1)
    main_jobs = cpu_count if "--optuna" not in sys.argv else 1
    print(f"OPTUNA_JOBS={optuna_jobs}, MAIN_JOBS={main_jobs}, CPU_COUNT={cpu_count}, GPU={has_gpu}")
    print("=====================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU diagnostics + TF/PTXAS + PyTorch + NVMath GEMM benchmarks"
    )
    parser.add_argument(
        "--gemm-size",
        type=int,
        default=GEMM_SIZE_DEFAULT,
        help=f"GEMM size n for n x n matrices in GEMM benchmarks (default: {GEMM_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--gemm-iters",
        type=int,
        default=GEMM_ITERS_DEFAULT,
        help=f"Timed iterations per backend per dtype in GEMM benchmarks (default: {GEMM_ITERS_DEFAULT})",
    )
    parser.add_argument(
        "--gemm-warmup",
        type=int,
        default=GEMM_WARMUP_DEFAULT,
        help=f"Warmup iterations per backend per dtype in GEMM benchmarks (default: {GEMM_WARMUP_DEFAULT})",
    )
    parser.add_argument(
        "--no-gemm",
        action="store_true",
        help="Skip the NVMath/PyTorch/NumPy GEMM benchmarks section.",
    )
    parser.add_argument(
        "--no-numpy-gemm",
        action="store_true",
        help="Skip NumPy CPU baseline in GEMM benchmarks.",
    )
    args = parser.parse_args()

    gemm_size = args.gemm_size
    gemm_iters = args.gemm_iters
    gemm_warmup = args.gemm_warmup
    use_numpy_gemm = not args.no_numpy_gemm

    print_nvidia_driver()
    print_cpu_info()
    print_cuda_path()

    gpu_inventory = print_gpu_capabilities()
    gpu0_info = gpu_inventory[0] if gpu_inventory else None

    gpus = print_tf_gpus()
    print_tf_info()
    analyze_gpu(gpus)

    has_gpu = bool(gpus)

    tf_results = run_tensorflow_suite(gpus, gpu0_info)
    torch_results = run_pytorch_suite(gpu0_info)

    # GEMM benchmarks (NumPy / PyTorch / NVMath)
    if not args.no_gemm:
        run_gemm_benchmarks(
            n=gemm_size,
            iters=gemm_iters,
            warmup=gemm_warmup,
            use_numpy=use_numpy_gemm,
        )

    test_ptxas()
    print_resources(has_gpu)

    hr("Summary (Diagnostics + Medium Stress Tests)")
    print("TensorFlow:")
    print(f"  Installed:      {tf_results.get('installed')}")
    print(f"  GPU available:  {tf_results.get('gpu_available')}")
    print(f"  GPU Conv2D OK:  {tf_results.get('gpu_conv_ok')}")
    print(f"  GPU matmul OK:  {tf_results.get('gpu_matmul_ok')}")
    print(f"  GPU grad OK:    {tf_results.get('gpu_grad_ok')}")
    print(f"  GPU stress OK:  {tf_results.get('gpu_stress_ok')}")

    print("\nPyTorch:")
    print(f"  Installed:      {torch_results.get('installed')}")
    print(f"  GPU available:  {torch_results.get('gpu_available')}")
    print(f"  GPU Conv2d OK:  {torch_results.get('gpu_conv_ok')}")
    print(f"  GPU matmul OK:  {torch_results.get('gpu_matmul_ok')}")
    print(f"  GPU grad OK:    {torch_results.get('gpu_grad_ok')}")
    print(f"  GPU stress OK:  {torch_results.get('gpu_stress_ok')}")

    print("\nPTXAS:")
    print("  (see PTXAS Diagnostics section above for details)")
    if not args.no_gemm:
        print("\nGEMM Benchmarks:")
        print("  (see GEMM / NVMath Benchmarks section above for TFLOP/s results)")
    print("\nDone.")


if __name__ == "__main__":
    main()
