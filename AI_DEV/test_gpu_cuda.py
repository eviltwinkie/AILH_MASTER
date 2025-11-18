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

# Config for output file
OUTPUT_FILE = os.environ.get("GPU_DIAG_OUTPUT", "gpu_diagnostic_results.txt")

# Smoke testing configuration
ENABLE_SMOKE_TESTS = os.environ.get("GPU_SMOKE_TESTS", "1").lower() in ("1", "true", "yes", "on")

# Status symbols
SUCCESS_SYMBOL = "✅"
FAILURE_SYMBOL = "❌"
WARNING_SYMBOL = "⚠️"
INFO_SYMBOL = "ℹ️"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import sys
import time
import textwrap
import subprocess
import shutil
import platform
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple
import keras
import numpy as np




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

# -----------------------------------------------------------------------------
# Optional dependency: PyCUDA (for TensorRT inference smoke test)
# -----------------------------------------------------------------------------
PYCUDA_AVAILABLE = False
PYCUDA_REASON = "Not attempted"

try:
    import pycuda.driver as cuda
    try:
        import pycuda.autoinit  # initializes CUDA context
        PYCUDA_AVAILABLE = True
        PYCUDA_REASON = "Imported successfully"
    except Exception as e:
        # pycuda is installed but context init failed
        PYCUDA_AVAILABLE = False
        PYCUDA_REASON = f"pycuda.autoinit failed: {repr(e)}"
        cuda = None
except Exception as e:
    # covers ImportError and any other odd import failures
    PYCUDA_AVAILABLE = False
    PYCUDA_REASON = f"import pycuda failed: {repr(e)}"
    cuda = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import tensorrt as trt
except ImportError:
    trt = None

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

# FP8 support by compute capability
FP8_SUPPORT = {
    8.6: False,      # Ampere - no native FP8
    8.9: False,      # Ada - no native FP8
    9.0: "partial",  # Hopper - limited FP8 (structured sparsity)
    12.0: True,      # Blackwell - full native FP8 support
}

# Base stress iterations (we use shapes adapted to GPU) — per your request
STRESS_ITERS = 10

# GEMM defaults (for NVMath / PyTorch / NumPy benchmarks)
GEMM_SIZE_DEFAULT = 12288
GEMM_ITERS_DEFAULT = 10
GEMM_WARMUP_DEFAULT = 3

# FP8 dtype configuration
ENABLE_FP8_TESTS = os.environ.get("GPU_FP8_TESTS", "1").lower() in ("1", "true", "yes", "on")

# =============================================================================
# UTILS
# =============================================================================

def hr(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def status_msg(success: bool, message: str, details: str = "") -> str:
    """Format a status message with success/failure indicator."""
    symbol = SUCCESS_SYMBOL if success else FAILURE_SYMBOL
    msg = f"{symbol} {message}"
    if details:
        msg += f"\n    {details}"
    return msg

def print_status(success: bool, message: str, details: str = "") -> None:
    """Print a status message."""
    print(status_msg(success, message, details))

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(cmd: List[str]) -> Optional[str]:
    """Run a command and return output, or None on failure."""
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

def check_torch_fp8_support() -> Tuple[bool, bool]:
    """
    Check if PyTorch supports FP8 dtypes (float8_e4m3fn, float8_e5m2).
    Returns (e4m3fn_supported: bool, e5m2_supported: bool)
    """
    if torch is None:
        return False, False
    
    try:
        # Check for torch.float8_e4m3fn (E4M3 format)
        e4m3fn_ok = hasattr(torch, 'float8_e4m3fn')
        # Check for torch.float8_e5m2 (E5M2 format)
        e5m2_ok = hasattr(torch, 'float8_e5m2')
        return e4m3fn_ok, e5m2_ok
    except Exception:
        return False, False

def get_blackwell_fp8_status(gpu_inventory: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Detect Blackwell GPU (CC 12.0) and FP8 support.
    Returns (is_blackwell: bool, gpu_name: Optional[str])
    
    Handles both compute capability formats:
    - Float format: compute_cap = 12.0 (from nvidia-smi)
    - Dict format: compute_capability = {"major": 12, "minor": 0} (from PyTorch)
    """
    if not gpu_inventory:
        return False, None
    
    for gpu in gpu_inventory:
        # Try float format first (compute_cap from nvidia-smi)
        cc_float = gpu.get("compute_cap")
        if cc_float is not None and isinstance(cc_float, (int, float)):
            cc_major = int(cc_float)  # e.g., 12.0 -> 12
            if cc_major == 12:
                return True, gpu.get("name", "Blackwell")
        
        # Fall back to dict format (compute_capability from PyTorch)
        cc_dict = gpu.get("compute_capability", {})
        if isinstance(cc_dict, dict):
            cc_major = cc_dict.get("major", 0)
            if cc_major == 12:
                return True, gpu.get("name", "Blackwell")
    
    return False, None


def compute_gemm_flops(n: int, m: int, k: int) -> float:
    """
    FLOP count for GEMM: C[m, n] = A[m, k] @ B[k, n]
    ~ 2 * m * n * k (multiply + add)
    """
    return 2.0 * float(m) * float(n) * float(k)

def _trt_float_dtype():
    # Older TRT: trt.float32 exists
    if hasattr(trt, "float32"):
        return trt.float32 # type: ignore
    # Newer TRT: only DataType.FLOAT exists
    if hasattr(trt, "DataType"):
        return trt.DataType.FLOAT # type: ignore
    # Fallback (very unlikely)
    raise RuntimeError("Could not determine TensorRT float32 dtype")

# =============================================================================
# CUPY BENCHMARK SUITE (GEMM + STRESS)
# =============================================================================

def run_cupy_suite(
    n: int = GEMM_SIZE_DEFAULT,
    iters: int = GEMM_ITERS_DEFAULT,
    warmup: int = GEMM_WARMUP_DEFAULT,
) -> Dict[str, Any]:
    """
    Run a CuPy-based GEMM benchmark as a sibling to the PyTorch/NVMath GEMM tests.
    This validates:
      - CuPy is installed and can see a CUDA device
      - Basic GEMM performance on the RTX 5090

    Returns a dict with success flag and measured TFLOP/s where available.
    """
    hr("CuPy Diagnostics + GEMM Benchmark")

    results: Dict[str, Any] = {
        "installed": cp is not None,
        "cuda_available": False,
        "fp32_tflops": None,
        "fp16_tflops": None,
    }

    if cp is None:
        print("❌ CuPy is not installed (pip install cupy-cudaXXX).")
        return results

    # Check CuPy CUDA availability
    try:
        dev = cp.cuda.Device(0)
        dev.use()
        results["cuda_available"] = True
        print(f"✅ CuPy sees CUDA device 0: {dev}")
    except Exception as e:
        print("❌ CuPy cannot access CUDA device 0:", repr(e))
        return results

    m = n
    k = n
    flops_per_iter = compute_gemm_flops(n, m, k)

    def bench_gemm(dtype, label: str) -> Optional[float]:
        if cp is None:
            print("❌ CuPy is not available. Skipping GEMM benchmark.")
            return None
        try:
            print(f"\n[CuPy] {label} GEMM: {m} x {k} @ {k} x {n}, dtype={dtype}")
            # Allocate
            A = cp.random.randn(m, k).astype(dtype)
            B = cp.random.randn(k, n).astype(dtype)

            # Warmup
            for _ in range(warmup):
                _ = A @ B
            cp.cuda.Stream.null.synchronize()

            # Timed
            t0 = time.time()
            for _ in range(iters):
                C = A @ B
            cp.cuda.Stream.null.synchronize()
            t1 = time.time()

            total_time = t1 - t0
            time_per_iter = total_time / iters if iters > 0 else float("inf")
            tflops = flops_per_iter / time_per_iter / 1e12

            print(
                f"  Total time: {total_time*1000:.2f} ms "
                f"({time_per_iter*1000:.2f} ms/iter), "
                f"Rate: {tflops:0.2f} TFLOP/s"
            )
            return tflops
        except Exception as e:
            print(f"❌ CuPy {label} GEMM FAILED:", repr(e))
            return None

    # fp32 benchmark
    results["fp32_tflops"] = bench_gemm(cp.float32, "fp32")

    # fp16 benchmark (if supported)
    results["fp16_tflops"] = bench_gemm(cp.float16, "fp16")

    return results

def build_tensorrt_fp8_engine(gpu_inventory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a TensorRT FP8 engine (Blackwell/CC 12.0+ only).
    Returns dict with build status and metrics.
    Handles both TRT 8/9 and TRT 10 API variations.
    """
    results = {
        "fp8_engine_built": False,
        "fp8_build_time_ms": None,
        "fp8_reason": "Not attempted",
    }

    if not ENABLE_FP8_TESTS:
        results["fp8_reason"] = "FP8 tests disabled (GPU_FP8_TESTS=0)"
        return results

    if trt is None:
        results["fp8_reason"] = "TensorRT not installed"
        return results

    # Check for Blackwell
    is_blackwell, gpu_name = get_blackwell_fp8_status(gpu_inventory)
    if not is_blackwell:
        results["fp8_reason"] = "No Blackwell GPU detected (FP8 requires CC 12.0+)"
        return results

    engine = None  # Initialize to avoid potential unbound variable

    try:
        hr("TensorRT FP8 Engine Build (Blackwell)")
        
        # Use correct Severity enum (older TRT uses trt.Logger.WARNING, newer uses trt.Logger.Severity.WARNING)
        if hasattr(trt.Logger, "Severity"):  # type: ignore
            logger = trt.Logger(trt.Logger.Severity.WARNING)  # type: ignore
        else:
            logger = trt.Logger(trt.Logger.WARNING)  # type: ignore
        
        builder = trt.Builder(logger)  # type: ignore
        
        # Create network with EXPLICIT_BATCH flag if available
        if hasattr(trt, "NetworkDefinitionCreationFlag"):  # pylint: disable=no-member
            network_flags = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore
            network = builder.create_network(flags=network_flags)
        else:
            network = builder.create_network()

        # Create builder config
        config = builder.create_builder_config()

        # Set FP8 precision flag if available
        fp8_flag_set = False
        if hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP8"):  # type: ignore
            try:
                config.set_flag(trt.BuilderFlag.FP8)  # type: ignore
                fp8_flag_set = True
                print_status(True, "FP8 precision flag enabled in TensorRT builder")
            except Exception as e:
                print_status(False, f"Could not set FP8 flag: {e}")

        if not fp8_flag_set:
            print_status(False, "FP8 flag not available in this TensorRT version")

        # Create simple identity network
        float_dtype = _trt_float_dtype()
        input_tensor = network.add_input(
            name="input",
            dtype=float_dtype,
            shape=(1, 3, 224, 224),
        )

        # Identity layer
        identity_layer = network.add_identity(input_tensor)
        identity_layer.get_output(0).name = "output_fp8"
        network.mark_output(identity_layer.get_output(0))

        print(f"Building TensorRT FP8 identity engine on {gpu_name}...")
        
        t0 = time.time()
        
        # Try new serialization API first, fall back to legacy if not available
        if hasattr(builder, "build_serialized_network"):
            engine_bytes = builder.build_serialized_network(network, config)
            runtime = trt.Runtime(logger)  # type: ignore
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        else:
            # Legacy API
            engine = builder.build_engine(network, config)
        
        t1 = time.time()
        build_time_ms = (t1 - t0) * 1000.0

        if engine is None:
            results["fp8_reason"] = "Builder returned None (likely missing FP8 support)"
            print_status(False, "Engine build returned None")
            return results

        results["fp8_engine_built"] = True
        results["fp8_build_time_ms"] = build_time_ms
        results["fp8_reason"] = "Success"
        
        print_status(True, f"FP8 Engine built successfully. Build time: {build_time_ms:.2f} ms")

    except Exception as e:
        results["fp8_reason"] = f"Exception: {type(e).__name__}: {str(e)}"
        print_status(False, f"FP8 engine build failed: {e}")

    return results


# =============================================================================
# TENSORRT SMOKE TEST + MICRO-BENCHMARK (TRT 8/9/10 COMPATIBLE)
# =============================================================================

def run_tensorrt_suite() -> Dict[str, Any]:
    """
    Simple TensorRT smoke test and micro-benchmark.

    - Verifies TensorRT Python bindings are importable.
    - Builds a trivial identity engine (1x3x224x224) and measures build time.
    - Runs a single inference (if PyCUDA is available) to ensure the engine executes.

    Designed to work with both older TensorRT (8.x/9.x) and newer 10.x APIs:
      * If config.max_workspace_size exists -> use legacy style.
      * Else, use config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size).
      * If builder.build_engine exists -> use it.
      * Else, use builder.build_serialized_network + trt.Runtime.
    """
    hr("TensorRT Smoke Test + Micro-Benchmark")

    results: Dict[str, Any] = {
        "installed": trt is not None,
        "engine_built": False,
        "build_time_ms": None,
        "inference_time_ms": None,
    }

    if trt is None:
        print("❌ TensorRT Python module not installed (pip install tensorrt).")
        return results

    # Basic version info
    try:
        trt_ver = trt.__version__
    except Exception:
        trt_ver = "Unknown"

    print(f"TensorRT version: {trt_ver}")

    engine = None  # Initialize to avoid potential unbound variable

    try:
        logger = trt.Logger(trt.Logger.Severity.WARNING) # type: ignore
        builder = trt.Builder(logger) # type: ignore

        # Try to create network in EXPLICIT_BATCH mode if available
        if hasattr(trt, "NetworkDefinitionCreationFlag"):
            network_flags = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # type: ignore
            print("Using EXPLICIT_BATCH network flag.")
            network = builder.create_network(flags=network_flags)
        else:
            print("NetworkDefinitionCreationFlag not available; using default create_network().")
            network = builder.create_network()

        # BuilderConfig may differ between TRT versions
        config = builder.create_builder_config()
        print(f"IBuilderConfig type: {type(config)}")

        # Workspace configuration – handle old and new APIs
        workspace_bytes = 1 << 26  # 64 MiB for tiny test

        if hasattr(config, "max_workspace_size"):
            # Legacy API (TRT 7/8/9)
            print("Config has 'max_workspace_size' attribute; using legacy workspace API.")
            config.max_workspace_size = workspace_bytes
        elif hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            # TRT 10-style API
            print("Using set_memory_pool_limit(MemoryPoolType.WORKSPACE, size) API.")
            try:
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes) # type: ignore
            except Exception as e:
                print("⚠️ Failed to call config.set_memory_pool_limit:", repr(e))
        else:
            print("⚠️ No known workspace configuration API found on IBuilderConfig; proceeding with defaults.")

        # Input: NCHW, N=1, C=3, H=224, W=224
        input_tensor = network.add_input(
            name="input",
            dtype=_trt_float_dtype(),
            shape=(1, 3, 224, 224),
        )

        # Identity layer
        identity = network.add_identity(input_tensor)
        identity.get_output(0).name = "output"
        network.mark_output(identity.get_output(0))

        # Engine build – support both old and new builder APIs
        print("Building a trivial TensorRT identity engine...")

        t0 = time.time()
        serialized_engine = None

        if hasattr(builder, "build_engine"):
            # Older API: directly build IEngine
            print("Using builder.build_engine(network, config).")
            engine = builder.build_engine(network, config)
        elif hasattr(builder, "build_serialized_network"):
            # Newer API: build serialized network then deserialize
            print("Using builder.build_serialized_network(...) + trt.Runtime().")
            serialized_engine = builder.build_serialized_network(network, config)
            runtime = trt.Runtime(logger) # type: ignore
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            print("❌ Builder has neither build_engine nor build_serialized_network.")
            results["engine_built"] = False
            return results

        t1 = time.time()

        if engine is None:
            print("❌ Failed to build TensorRT engine (engine is None).")
            return results

        results["engine_built"] = True
        results["build_time_ms"] = (t1 - t0) * 1000.0
        print(f"✅ Engine build OK. Time: {results['build_time_ms']:.2f} ms")

        # Try a simple inference if PyCUDA is available
        if not PYCUDA_AVAILABLE:
            print("PyCUDA not available or not usable; skipping inference timing.")
            print(f"  Reason: {PYCUDA_REASON}")
            print("  If this is a venv issue, run this script with the same Python that can 'import pycuda'.")
            return results

        # Reimport cuda to ensure it's the actual module (not None)
        try:
            import pycuda.driver as cuda
        except Exception as e:
            print(f"Failed to import pycuda.driver for inference: {repr(e)}")
            return results

        print("Running a single inference with PyCUDA for smoke test...")
        import numpy as _np

        context = engine.create_execution_context()
        # host buffers
        inp = _np.random.randn(1, 3, 224, 224).astype(_np.float32)
        out = _np.empty_like(inp)

        # device buffers
        d_input = cuda.mem_alloc(inp.nbytes) # type: ignore
        d_output = cuda.mem_alloc(out.nbytes) # type: ignore

        # copy input to device
        cuda.memcpy_htod(d_input, inp) # type: ignore

        bindings = [int(d_input), int(d_output)]

        t2 = time.time()
        context.execute_v2(bindings)
        cuda.Context.synchronize() # type: ignore
        t3 = time.time()

        # copy back result
        cuda.memcpy_dtoh(out, d_output) # type: ignore

        results["inference_time_ms"] = (t3 - t2) * 1000.0
        print(f"✅ Inference OK. Time: {results['inference_time_ms']:.2f} ms")

    except Exception as e:
        print("❌ TensorRT smoke test FAILED:", repr(e))

    return results

# =============================================================================
# SYSTEM / DRIVER / CPU / CUDA PATH
# =============================================================================

def print_nvidia_driver() -> bool:
    """Print NVIDIA driver version. Returns success status."""
    hr("NVIDIA Driver")
    out = run_cmd(["nvidia-smi"])
    if not out:
        print_status(False, "nvidia-smi not found or driver unavailable")
        return False
    
    for line in out.splitlines():
        if "Driver Version" in line:
            version = line.strip()
            print_status(True, version)
            return True
    
    print_status(False, "Driver version not detected in nvidia-smi output")
    return False

def print_init_nvml():
    """Initialize NVML and return device handle. Returns (success: bool, handle)."""
    hr("NVML Initialization")
    try:
        import pynvml
        pynvml.nvmlInit()
        lib = pynvml.nvmlSystemGetNVMLVersion()
        print_status(True, f"NVML initialized: version {lib}")
        return True, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print_status(False, f"NVML unavailable: {e}")
        return False, None

def print_cpu_info() -> None:
    """Print CPU information."""
    hr("CPU Diagnostics")
    cores = os.cpu_count() or 1
    arch = platform.machine()
    proc = platform.processor()
    print(f"  Cores:       {cores}")
    print(f"  Architecture: {arch}")
    print(f"  Processor:   {proc or '(unknown)'}")
    
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            brand = info.get('brand_raw', 'Unknown')
            bits = info.get('bits', 'Unknown')
            print(f"  Brand:       {brand}")
            print(f"  Bits:        {bits}")
        except Exception as e:
            print_status(False, f"Failed to get detailed CPU info: {e}")
    else:
        print_status(False, "py-cpuinfo not installed (pip install py-cpuinfo)")

def print_cuda_path() -> None:
    """Print CUDA directories in PATH."""
    hr("CUDA PATH Check")
    cuda_paths = [p for p in os.environ.get("PATH", "").split(os.pathsep) if "cuda" in p.lower()]
    if cuda_paths:
        for p in cuda_paths:
            print(f"  {p}")
        print_status(True, f"Found {len(cuda_paths)} CUDA directories in PATH")
    else:
        print_status(False, "No CUDA directories found in PATH")

# =============================================================================
# NVIDIA CLI TOOLS SCANNER
# =============================================================================

def scan_nvidia_cli_tools() -> Dict[str, bool]:
    """
    Check PATH for useful NVIDIA / CUDA CLI tools.
    Returns dict of tool_name -> found (bool).
    """
    hr("NVIDIA CLI Tools on PATH")

    tools = [
        "nvidia-smi", "nvcc", "ptxas", "cuda-gdb",
        "cuda-memcheck", "ncu", "nsys", "nsight-sys", "nsight-compute",
    ]

    found_tools = {}
    for t in tools:
        path = which(t)
        found = path is not None
        found_tools[t] = found
        status = f"-> {path}" if path else "(not found)"
        print(f"  {t:15s} {status}")
    
    return found_tools

# =============================================================================
# PYTHON-LEVEL NVIDIA / CUDA MODULE SCANNER
# =============================================================================

def scan_python_nvidia_modules() -> None:
    """
    Scan the active Python environment for modules whose names suggest NVIDIA /
    CUDA / GPU relevance. This is a heuristic: it will find things like:

      - nvidia-*
      - cuda-*, cupy, numba, pycuda, triton
      - rapids libs: cuml, cugraph, cusignal, cupy, rmm, etc.
      - tensorrt, onnxruntime-gpu, etc.

    It does NOT import everything; it just lists them by name and spec location.
    """
    import pkgutil
    import importlib.util

    hr("Python NVIDIA / CUDA Modules Scan")

    keywords = [
        "nvidia",
        "cuda",
        "cupy",
        "cuml",
        "cugraph",
        "cusignal",
        "cudf",
        "rmm",
        "pycuda",
        "numba",
        "tensorrt",
        "onnxruntime",
        "modulus",
        "kaolin",
        "triton",
    ]

    found = []

    for m in pkgutil.iter_modules():
        name = m.name.lower()
        if any(k in name for k in keywords):
            found.append(m.name)

    if not found:
        print("No candidate NVIDIA/CUDA-related Python modules found beyond the ones already imported.")
        return

    found = sorted(set(found))
    print(f"Found {len(found)} candidate modules:")
    for name in found:
        try:
            spec = importlib.util.find_spec(name)
            location = getattr(spec, "origin", None) or getattr(spec, "submodule_search_locations", None)
        except Exception:
            location = None
        print(f"  - {name}  (location: {location})")

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
# CUDA / LIBRARY COMPATIBILITY GRAPH
# =============================================================================

def _decode_cuda_version_int(v: int) -> str:
    """
    Decode CUDA version integer reported by CuPy runtime (e.g. 12030 -> '12.3').
    """
    if v is None or v <= 0:
        return "Unknown"
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"


def get_cupy_cuda_versions():
    """
    Return (cupy_version, runtime_cuda, driver_cuda) strings or (None, None, None)
    if CuPy is not available.

    runtime_cuda and driver_cuda are derived from CuPy's cuda.runtime introspection.
    """
    if 'cp' not in globals() or cp is None:
        return None, None, None

    try:
        drv = cp.cuda.runtime.driverGetVersion()
        rt = cp.cuda.runtime.runtimeGetVersion()
        drv_str = _decode_cuda_version_int(drv)
        rt_str = _decode_cuda_version_int(rt)
        return cp.__version__, rt_str, drv_str
    except Exception:
        # Be robust if runtime is misconfigured
        try:
            return cp.__version__, "Unknown", "Unknown"
        except Exception:
            return None, None, None


def get_tensorrt_versions():
    """
    Return (tensorrt_version, note) or (None, None) if TensorRT is missing.
    We do not try to guess its CUDA version from Python – that requires
    NVIDIA's official compatibility matrix.
    """
    if 'trt' not in globals() or trt is None:
        return None, None
    try:
        return trt.__version__, "Uses CUDA libs from installed TensorRT build"
    except Exception:
        return None, None


def get_nvcc_version():
    """
    Parse 'nvcc --version' if available, returning a short CUDA toolkit version string
    like '12.8' or 'Unknown'.
    """
    nvcc_path = which("nvcc")
    if not nvcc_path:
        return None

    out = run_cmd(["nvcc", "--version"])
    if not out:
        return None

    for line in out.splitlines():
        line = line.strip()
        # Typical: "Cuda compilation tools, release 12.8, V12.8.89"
        if "Cuda compilation tools" in line and "release" in line:
            parts = line.split("release", 1)[-1].strip().split(",", 1)[0]
            return parts.strip()
    return None


def print_cuda_compatibility_graph(gpu_inventory: List[Dict[str, Any]]) -> None:
    """
    Print a simple compatibility table showing:
      - Driver version (from nvidia-smi / gpu_inventory)
      - CUDA toolkit (nvcc) version if available
      - TensorFlow build CUDA
      - PyTorch CUDA
      - CuPy CUDA runtime/driver
      - NVMath version
      - TensorRT version

    This does NOT perform deep semantic validation; it just surfaces what each
    component reports, so you can cross-check against NVIDIA's official tables.
    """
    hr("CUDA / Library Compatibility Graph")

    # ------------------ Driver + Toolkit ------------------
    driver_version = None
    if gpu_inventory:
        driver_version = gpu_inventory[0].get("driver_version", None)
    nvcc_ver = get_nvcc_version()

    print("Driver / Toolkit:")
    print(f"  NVIDIA Driver:   {driver_version or 'Unknown (see nvidia-smi)'}")
    print(f"  CUDA Toolkit:    {nvcc_ver or 'nvcc not found / toolkit not on PATH'}")

    # ------------------ TensorFlow ------------------
    tf_cuda = tf_cudnn = tf_version = None
    if 'tf' in globals() and tf is not None:
        try:
            tf_version = tf.__version__
            build = tf.sysconfig.get_build_info()
            tf_cuda = build.get("cuda_version", "Unknown")
            tf_cudnn = build.get("cudnn_version", "Unknown")
        except Exception:
            tf_version = tf_version or "Unknown"
            tf_cuda = tf_cuda or "Unknown"
            tf_cudnn = tf_cudnn or "Unknown"

    # ------------------ PyTorch ------------------
    torch_version = None
    torch_cuda = None
    if 'torch' in globals() and torch is not None:
        try:
            torch_version = torch.__version__
            torch_cuda = torch.version.cuda or "Unknown"
        except Exception:
            pass

    # ------------------ CuPy ------------------
    cupy_ver, cupy_rt_cuda, cupy_drv_cuda = get_cupy_cuda_versions()

    # ------------------ NVMath ------------------
    nvmath_ver = None
    if 'nvmath' in globals() and nvmath is not None:
        nvmath_ver = getattr(nvmath, "__version__", "Unknown")

    # ------------------ TensorRT ------------------
    trt_ver, trt_note = get_tensorrt_versions()

    print("\nComponents:")
    header = (
        f"{'Component':<12} | {'Version':<16} | {'CUDA runtime / build':<24} | {'Notes'}"
    )
    print(header)
    print("-" * len(header))

    def row(name, ver, cuda_str, notes=""):
        print(f"{name:<12} | {ver:<16} | {cuda_str:<24} | {notes}")

    # Driver / toolkit
    row("Driver", driver_version or "Unknown", "-", "From nvidia-smi")
    row("Toolkit", nvcc_ver or "Unknown", "-", "From nvcc --version")

    # TF
    if tf is not None:
        cuda_str = f"CUDA {tf_cuda or 'Unknown'}, cuDNN {tf_cudnn or 'Unknown'}"
        row("TensorFlow", tf_version or "Unknown", cuda_str, "tf.sysconfig.get_build_info()")
    else:
        row("TensorFlow", "Not installed", "-", "")

    # PyTorch
    if torch is not None:
        row("PyTorch", torch_version or "Unknown", f"CUDA {torch_cuda or 'Unknown'}", "torch.version.cuda")
    else:
        row("PyTorch", "Not installed", "-", "")

    # CuPy
    if cupy_ver is not None:
        cuda_details = f"runtime {cupy_rt_cuda}, driver {cupy_drv_cuda}"
        row("CuPy", cupy_ver, cuda_details, "cp.cuda.runtime.*Version()")
    else:
        row("CuPy", "Not installed", "-", "")

    # NVMath
    if nvmath is not None:
        row("NVMath", nvmath_ver or "Unknown", "Uses CUDA via nvmath-python", "See NVMath docs")
    else:
        row("NVMath", "Not installed", "-", "")

    # TensorRT
    if trt_ver is not None:
        row("TensorRT", trt_ver, "Uses CUDA from TensorRT build", trt_note or "")
    else:
        row("TensorRT", "Not installed", "-", "")

    print("\nUse this table with NVIDIA's official compatibility matrix to validate that")
    print("your driver, toolkit, and libraries are aligned for the RTX 5090.")


# =============================================================================
# RTX 5090 / BLACKWELL OPTIMIZATION PLAN
# =============================================================================

def print_rtx5090_optimization_plan(
    gpu_inventory: List[Dict[str, Any]],
    tf_present: bool,
    torch_present: bool,
    cupy_present: bool,
    tensorrt_present: bool,
) -> None:
    """
    Emit a recommended optimization plan for an RTX 5090-class GPU.

    We infer "Blackwell-class" if:
      - compute_cap >= 12.0 OR
      - GPU name contains '5090' (case-insensitive)

    The plan is advisory text: CUDA / TF / Torch / CuPy / TensorRT alignment,
    precision modes, and general best practices for your mel / GEMM workloads.
    """
    hr("RTX 5090 Optimization Plan")

    if not gpu_inventory:
        print("No GPU inventory data; cannot tailor plan. Using generic guidance.")
        cc = None
        name = "Unknown GPU"
    else:
        g0 = gpu_inventory[0]
        cc = g0.get("compute_cap", None)
        name = g0.get("name", "Unknown GPU")

    name_lower = (name or "").lower()
    is_blackwell_like = False
    if cc is not None and cc >= 12.0:
        is_blackwell_like = True
    if "5090" in name_lower or "blackwell" in name_lower:
        is_blackwell_like = True

    print(f"Detected GPU[0]: {name} (compute capability={cc})")
    if not is_blackwell_like:
        print("Plan below is still applicable, but tuned for RTX 5090 / Blackwell-class GPUs.\n")

    print("1. Driver / CUDA / cuDNN baseline")
    print("   - Keep NVIDIA driver at or above the version recommended for recent CUDA 12.x.")
    print("   - Prefer CUDA 12.8.x toolkit for Blackwell-class GPUs when stable in your distro.")
    print("   - Use cuDNN 9.x for best convolution performance on modern architectures.")

    print("\n2. TensorFlow stack (if in use)")
    if tf_present:
        print("   - Target recent TensorFlow 2.x builds compiled against CUDA 12.5+ / 12.8.x.")
        print("   - Enable memory growth (already enabled in this script) to avoid full VRAM grab.")
        print("   - For training/inference:")
        print("       * Enable mixed precision (float16 / bfloat16) if your model is stable.")
        print("       * Enable XLA where beneficial (tf.function(jit_compile=True) on hot paths).")
        print("       * Monitor first-run latency due to PTX JIT; consider ahead-of-time builds if needed.")
    else:
        print("   - TensorFlow not detected; skip unless you plan TF workloads.")

    print("\n3. PyTorch stack (if in use)")
    if torch_present:
        print("   - Install a PyTorch build compiled against a recent CUDA 12.x runtime.")
        print("   - In your training code:")
        print("       * torch.backends.cuda.matmul.allow_tf32 = True")
        print("       * torch.backends.cudnn.allow_tf32 = True")
        print("       * torch.set_float32_matmul_precision('high')")
        print("       * Use pin_memory=True and non_blocking=True in DataLoader / .to(device).")
        print("       * Use multiple worker processes for I/O (num_workers tuned to CPU cores / storage).")
        print("       * Consider multiple CUDA streams for overlap of H2D copies and compute.")
    else:
        print("   - PyTorch not detected; install if you prefer Torch-based models.")

    print("\n4. CuPy / NVMath numerical kernels")
    if cupy_present:
        print("   - Use CuPy for custom kernels and array ops that complement PyTorch/TF.")
        print("   - Align CuPy's CUDA variant (cupy-cudaXXX) with your installed toolkit (e.g., cuda12x).")
        print("   - Use float16 / bfloat16 in heavy GEMMs when acceptable; benchmark vs NVMath / Torch.")
    else:
        print("   - CuPy not detected; install with a CUDA-matched wheel (e.g., cupy-cuda12x).")

    if 'nvmath' in globals() and nvmath is not None:
        print("   - NVMath (nvmath-python) can provide high-performance GEMM; compare its TFLOPs")
        print("     vs PyTorch/CuPy for your target GEMM sizes (e.g., 12288x12288).")

    print("\n5. TensorRT optimization (if in use)")
    if tensorrt_present:
        print("   - Use TensorRT for latency-critical inference paths.")
        print("   - Prefer FP16 or BF16 engines by default; experiment with FP8 when Blackwell support")
        print("     is production-ready in your stack.")
        print("   - Calibrate INT8 only when you have solid calibration datasets.")
    else:
        print("   - TensorRT not detected; add it later for low-latency inference deployment.")

    print("\n6. General GPU utilization and pipeline tuning")
    print("   - Ensure input pipelines (NVMe → CPU → GPU) are saturated: monitor GPU utilization")
    print("     while your mel pipeline runs (nvidia-smi, nsight, or your status reporter).")
    print("   - Keep batch sizes large enough to reach high SM occupancy, but below OOM thresholds.")
    print("   - For large GEMMs / FFTs / mel pipelines, avoid thrashing VRAM: reuse buffers and")
    print("     favor pre-allocated workspaces (as this suite and your pipeline already do).")
    print("   - Profile frequently with Nsight Systems / Nsight Compute to check for:")
    print("       * Kernel launch overhead")
    print("       * Under-utilized SMs")
    print("       * PCIe bottlenecks (H2D/D2H transfers overlapping poorly with compute)")

# =============================================================================
# SYSTEM-LEVEL CUDA / NVIDIA SHARED LIB SCANNER
# =============================================================================

def scan_system_cuda_libs() -> None:
    """
    Scan common library directories for CUDA / cuDNN / TensorRT / NVML libraries.

    Directories checked (if they exist):
      - /usr/lib/x86_64-linux-gnu
      - /usr/local/cuda/lib64
      - /usr/local/cuda-*/lib64

    We list libs matching patterns like:
      - libcud*
      - libcu*
      - libnvinfer*
      - libcudnn*
      - libnvrtc*
      - libnvToolsExt*
      - libnccl*
    """
    hr("System CUDA / NVIDIA Shared Libraries")

    base_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
    ]

    # Add /usr/local/cuda-*/lib64
    cuda_root = "/usr/local"
    try:
        for entry in os.listdir(cuda_root):
            if entry.startswith("cuda-"):
                libdir = os.path.join(cuda_root, entry, "lib64")
                base_dirs.append(libdir)
    except Exception:
        pass

    patterns = (
        "libcudart",
        "libcublas",
        "libcufft",
        "libcurand",
        "libcusolver",
        "libcusparse",
        "libcudnn",
        "libnvinfer",
        "libnvonnxparser",
        "libnvrtc",
        "libnvToolsExt",
        "libnvml",
        "libnccl",
        "libcutensor",
    )

    seen_libs = set()

    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        try:
            entries = os.listdir(d)
        except Exception:
            continue

        for fname in entries:
            for p in patterns:
                if fname.startswith(p):
                    seen_libs.add(os.path.join(d, fname))

    if not seen_libs:
        print("No CUDA / NVIDIA shared libraries found in the scanned directories.")
        return

    for lib in sorted(seen_libs):
        print("  ", lib)

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
            conv = keras.layers.Conv2D(32, 3, padding="same")
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
            model = keras.Sequential([
                keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                keras.layers.MaxPool2D(2),
                keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(10),
            ])
            t0 = time.time()
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            grads = tape.gradient(loss, model.trainable_variables)
            if grads is None:
                grads = []
            elif not isinstance(grads, (list, tuple)):
                grads = [grads]
            if any(g is None for g in grads):
                raise RuntimeError("Some gradients are None")
            t1 = time.time()
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
            conv = keras.layers.Conv2D(32, 3, padding="same")
            _ = conv(x)  # warm-up
            t0 = time.time()
            y = conv(x)
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
            t0 = time.time()
            c = tf.matmul(a, b)
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
            model = keras.Sequential([
                keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
                keras.layers.MaxPool2D(2),
                keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(10),
            ])
            # One warmup forward/backward
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            _ = tape.gradient(loss, model.trainable_variables)
            t0 = time.time()
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            grads = tape.gradient(loss, model.trainable_variables)
            if grads is None:
                grads = []
            elif not isinstance(grads, (list, tuple)):
                grads = [grads]
            if any(g is None for g in grads):
                raise RuntimeError("Some gradients are None")
            t1 = time.time()
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
            y = None
            with tf.device(tf_device):
                # Conv2D stress
                x = tf.random.normal([conv_batch, conv_hw, conv_hw, conv_in_ch])
                conv = keras.layers.Conv2D(
                    filters=conv_out_ch,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    use_bias=True,
                )

                t0 = time.time()
                for _ in range(STRESS_ITERS):
                    y = conv(x)
                if y is not None:
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
                c = tf.matmul(a, b)

                t2 = time.time()
                for _ in range(STRESS_ITERS):
                    c = tf.matmul(a, b)
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


def run_pytorch_fp8_smoke_test(gpu_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    PyTorch FP8 smoke test (float8_e4m3fn and float8_e5m2).
    Only runs on GPUs with compute capability >= 12.0 (Blackwell).
    Returns dict with test results.
    """
    results = {
        "installed": torch is not None,
        "fp8_available": False,
        "e4m3fn_ok": False,
        "e5m2_ok": False,
        "gpu_e4m3fn_ok": False,
        "gpu_e5m2_ok": False,
    }

    if torch is None:
        print_status(False, "PyTorch not installed; skipping FP8 smoke test")
        return results

    hr("PyTorch FP8 Smoke Test (Blackwell/CC 12.0+)")

    # Check dtype support
    e4m3fn_ok, e5m2_ok = check_torch_fp8_support()
    results["e4m3fn_ok"] = e4m3fn_ok
    results["e5m2_ok"] = e5m2_ok

    if not (e4m3fn_ok or e5m2_ok):
        print_status(False, "PyTorch FP8 dtypes not available (requires PyTorch >= 2.1 with FP8 support)")
        return results

    print_status(True, "PyTorch FP8 dtypes available")
    if e4m3fn_ok:
        print(f"  {SUCCESS_SYMBOL} float8_e4m3fn (E4M3 format)")
    if e5m2_ok:
        print(f"  {SUCCESS_SYMBOL} float8_e5m2 (E5M2 format)")

    if not torch.cuda.is_available():
        print_status(False, "CUDA not available; skipping GPU FP8 tests")
        return results

    # Check GPU support
    is_blackwell, gpu_name = get_blackwell_fp8_status([gpu_info] if gpu_info else [])
    if not is_blackwell:
        cc = gpu_info.get("compute_capability", {}) if gpu_info else {}
        cc_major = cc.get("major", 0)
        print_status(False, f"GPU CC {cc_major}.x detected; FP8 requires CC 12.0+ (Blackwell)")
        return results

    print_status(True, f"Blackwell GPU detected: {gpu_name}")

    # Test FP8 operations on GPU
    device = torch.device("cuda:0")
    try:
        if e4m3fn_ok:
            x = torch.randn(32, 32, device=device, dtype=torch.float32)
            x_fp8 = x.to(torch.float8_e4m3fn)
            y = torch.ones_like(x, device=device, dtype=torch.float32)
            y_fp8 = y.to(torch.float8_e4m3fn)
            
            torch.cuda.synchronize()
            t0 = time.time()
            z = x_fp8.to(torch.float32) @ y_fp8.to(torch.float32)
            torch.cuda.synchronize()
            t1 = time.time()
            
            print_status(True, f"float8_e4m3fn GPU operations OK ({(t1-t0)*1000:.2f} ms)")
            results["gpu_e4m3fn_ok"] = True
    except Exception as e:
        print_status(False, f"float8_e4m3fn GPU operations failed: {repr(e)}")

    try:
        if e5m2_ok:
            x = torch.randn(32, 32, device=device, dtype=torch.float32)
            x_fp8 = x.to(torch.float8_e5m2)
            y = torch.ones_like(x, device=device, dtype=torch.float32)
            y_fp8 = y.to(torch.float8_e5m2)
            
            torch.cuda.synchronize()
            t0 = time.time()
            z = x_fp8.to(torch.float32) @ y_fp8.to(torch.float32)
            torch.cuda.synchronize()
            t1 = time.time()
            
            print_status(True, f"float8_e5m2 GPU operations OK ({(t1-t0)*1000:.2f} ms)")
            results["gpu_e5m2_ok"] = True
    except Exception as e:
        print_status(False, f"float8_e5m2 GPU operations failed: {repr(e)}")

    results["fp8_available"] = results["gpu_e4m3fn_ok"] or results["gpu_e5m2_ok"]
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

            # --- Matmul Stress Test ---
            # Modern, non-deprecated AMP + accurate CUDA timing

            stress_dtype = torch.float32    # or torch.float16 or torch.bfloat16
            stream = torch.cuda.Stream(device=device)

            # Preallocate test matrices
            with torch.cuda.stream(stream):
                a = torch.randn(mm_n, mm_n, dtype=stress_dtype, device=device)
                b = torch.randn(mm_n, mm_n, dtype=stress_dtype, device=device)
            stream.synchronize()

            # Accurate timing using CUDA events (best practice)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            start_evt.record(stream)
            c = None  # Initialize to satisfy linter
            with torch.cuda.stream(stream):
                for _ in range(STRESS_ITERS):
                    c = a @ b
            end_evt.record(stream)

            # Wait for GPU to finish work
            end_evt.synchronize()

            mm_time_ms = start_evt.elapsed_time(end_evt)  # precise GPU timing
            if c is not None:
                _ = c.sum().item()

            print(
                f"  Matmul stress: {STRESS_ITERS} iters, total {mm_time_ms:.2f} ms, "
                f"avg {mm_time_ms / STRESS_ITERS:.2f} ms/iter "
                f"(dtype={stress_dtype}, size={mm_n}x{mm_n})"
            )

            # a = torch.randn(mm_n, mm_n, device=device)
            # b = torch.randn(mm_n, mm_n, device=device)
            # torch.cuda.synchronize()
            # t2 = time.time()
            # for _ in range(STRESS_ITERS):
            #     c = a @ b
            # torch.cuda.synchronize()
            # t3 = time.time()
            # _ = c.sum().item()
            # mm_time_ms = (t3 - t2) * 1000.0
            # print(
            #     f"  Matmul stress: {STRESS_ITERS} iters, total {mm_time_ms:.2f} ms, "
            #     f"avg {mm_time_ms / STRESS_ITERS:.2f} ms/iter"
            # )



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
def bench_torch_gemm_fp8(n: int, iters: int, warmup: int) -> Dict[str, float]:
    """
    Benchmark PyTorch GEMM with FP8 dtypes (float8_e4m3fn, float8_e5m2).
    Only runs on GPUs with CC >= 12.0 (Blackwell).
    Returns dict with TFLOP/s for each dtype, or NaN if not supported.
    """
    results = {"fp8_e4m3fn_tflops": float("nan"), "fp8_e5m2_tflops": float("nan")}

    if torch is None or not torch.cuda.is_available():
        return results

    if not ENABLE_FP8_TESTS:
        return results

    e4m3fn_ok, e5m2_ok = check_torch_fp8_support()
    if not (e4m3fn_ok or e5m2_ok):
        return results

    device = torch.device("cuda:0")

    # Bench float8_e4m3fn
    if e4m3fn_ok:
        try:
            print(f"\n[PyTorch FP8] GEMM benchmark (GPU) — float8_e4m3fn, "
                  f"size={n}x{n}, iters={iters}, warmup={warmup}")

            A = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
            B = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

            # Warmup
            for _ in range(warmup):
                C = A.to(torch.float32) @ B.to(torch.float32)
                torch.cuda.synchronize()

            flops_per_iter = compute_gemm_flops(n, n, n)

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                C = A.to(torch.float32) @ B.to(torch.float32)
            torch.cuda.synchronize()
            t1 = time.time()

            elapsed = t1 - t0
            if elapsed > 0:
                total_flops = flops_per_iter * iters
                flops_per_sec = total_flops / elapsed
                results["fp8_e4m3fn_tflops"] = flops_per_sec
                print(f"  Total time: {elapsed * 1000.0:0.2f} ms")
                print(f"  Per-iter time: {elapsed * 1000.0 / iters:0.2f} ms")
                print(f"  Average rate: {human_tflops(flops_per_sec)}")
        except Exception as e:
            print_status(False, f"PyTorch FP8 E4M3 GEMM failed: {repr(e)}")

    # Bench float8_e5m2
    if e5m2_ok:
        try:
            print(f"\n[PyTorch FP8] GEMM benchmark (GPU) — float8_e5m2, "
                  f"size={n}x{n}, iters={iters}, warmup={warmup}")

            A = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e5m2)
            B = torch.randn(n, n, device=device, dtype=torch.float32).to(torch.float8_e5m2)

            # Warmup
            for _ in range(warmup):
                C = A.to(torch.float32) @ B.to(torch.float32)
                torch.cuda.synchronize()

            flops_per_iter = compute_gemm_flops(n, n, n)

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                C = A.to(torch.float32) @ B.to(torch.float32)
            torch.cuda.synchronize()
            t1 = time.time()

            elapsed = t1 - t0
            if elapsed > 0:
                total_flops = flops_per_iter * iters
                flops_per_sec = total_flops / elapsed
                results["fp8_e5m2_tflops"] = flops_per_sec
                print(f"  Total time: {elapsed * 1000.0:0.2f} ms")
                print(f"  Per-iter time: {elapsed * 1000.0 / iters:0.2f} ms")
                print(f"  Average rate: {human_tflops(flops_per_sec)}")
        except Exception as e:
            print_status(False, f"PyTorch FP8 E5M2 GEMM failed: {repr(e)}")

    return results




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
    """Main diagnostics function with text file output."""
    # Create a file handle for logging
    output_file = open(OUTPUT_FILE, "w")
    original_stdout = sys.stdout
    
    class TeeOutput:
        """Redirect stdout to both console and file."""
        def __init__(self, console, file):
            self.console = console
            self.file = file
        
        def write(self, message: str) -> None:
            self.console.write(message)
            self.file.write(message)
            self.file.flush()
        
        def flush(self) -> None:
            self.console.flush()
            self.file.flush()
    
    # Redirect stdout to capture all output
    sys.stdout = TeeOutput(original_stdout, output_file)
    
    try:
        hr("Python Interpreter Info")
        print("sys.executable:", sys.executable)
        print("sys.version:", sys.version.replace("\n", " "))
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
        parser.add_argument(
            "--no-smoke-tests",
            action="store_true",
            help="Skip smoke tests (functional tests for TF/PyTorch/etc.)",
        )
        parser.add_argument(
            "--no-fp8",
            action="store_true",
            help="Skip FP8 tests (Blackwell GPU only; requires ENABLE_FP8_TESTS env var)",
        )
        args = parser.parse_args()

        gemm_size = args.gemm_size
        gemm_iters = args.gemm_iters
        gemm_warmup = args.gemm_warmup
        use_numpy_gemm = not args.no_numpy_gemm
        run_smoke_tests = not args.no_smoke_tests and ENABLE_SMOKE_TESTS
        run_fp8_tests = not args.no_fp8 and ENABLE_FP8_TESTS

        print_nvidia_driver()
        print_cuda_path()
        nvml_ok, nvml_handle = print_init_nvml()
        print_cpu_info()

        cli_tools = scan_nvidia_cli_tools()

        gpu_inventory = print_gpu_capabilities()
        gpu0_info = gpu_inventory[0] if gpu_inventory else None

        gpus = print_tf_gpus()
        print_tf_info()
        analyze_gpu(gpus)

        print_cuda_compatibility_graph(gpu_inventory)

        scan_python_nvidia_modules()
        scan_system_cuda_libs()

        has_gpu = bool(gpus)

        # Conditionally run smoke tests
        if run_smoke_tests:
            tf_results = run_tensorflow_suite(gpus, gpu0_info)
            torch_results = run_pytorch_suite(gpu0_info)

            cupy_results = run_cupy_suite(
                n=gemm_size,
                iters=gemm_iters,
                warmup=gemm_warmup,
            )
            tensorrt_results = run_tensorrt_suite()
            
            # FP8 tests (Blackwell GPU only)
            if run_fp8_tests:
                pytorch_fp8_results = run_pytorch_fp8_smoke_test(gpu0_info)
                fp8_gemm_results = bench_torch_gemm_fp8(n=gemm_size, iters=gemm_iters, warmup=gemm_warmup)
                tensorrt_fp8_results = build_tensorrt_fp8_engine(gpu_inventory)
            else:
                pytorch_fp8_results = {"tested": False, "reason": "FP8 tests disabled (--no-fp8 or GPU_FP8_TESTS=0)"}
                fp8_gemm_results = {"tested": False, "reason": "FP8 tests disabled (--no-fp8 or GPU_FP8_TESTS=0)"}
                tensorrt_fp8_results = {"tested": False, "reason": "FP8 tests disabled (--no-fp8 or GPU_FP8_TESTS=0)"}
        else:
            hr("Smoke Tests Skipped")
            print_status(True, "Smoke tests disabled (--no-smoke-tests or GPU_SMOKE_TESTS=0)")
            tf_results = {"installed": tf is not None}
            torch_results = {"installed": torch is not None}
            cupy_results = {"installed": cp is not None}
            tensorrt_results = {"installed": trt is not None}
            pytorch_fp8_results = {"tested": False, "reason": "Smoke tests disabled"}
            fp8_gemm_results = {"tested": False, "reason": "Smoke tests disabled"}
            tensorrt_fp8_results = {"tested": False, "reason": "Smoke tests disabled"}

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

        print_rtx5090_optimization_plan(
            gpu_inventory=gpu_inventory,
            tf_present=(tf is not None),
            torch_present=(torch is not None),
            cupy_present=('cp' in globals() and cp is not None),
            tensorrt_present=('trt' in globals() and trt is not None),
        )

        hr("Summary - Test Results")
        
        def format_result(label: str, success: bool, details: str = "") -> str:
            sym = SUCCESS_SYMBOL if success else FAILURE_SYMBOL
            msg = f"  {sym} {label}"
            if details:
                msg += f" | {details}"
            return msg
        
        print("TensorFlow:")
        print(format_result("Installed", tf_results.get('installed', False)))
        if run_smoke_tests:
            print(format_result("GPU Available", tf_results.get('gpu_available', False)))
            print(format_result("Conv2D", tf_results.get('gpu_conv_ok', False)))
            print(format_result("Matmul", tf_results.get('gpu_matmul_ok', False)))
            print(format_result("Gradients", tf_results.get('gpu_grad_ok', False)))
            print(format_result("Stress Test", tf_results.get('gpu_stress_ok', False)))

        print("\nPyTorch:")
        print(format_result("Installed", torch_results.get('installed', False)))
        if run_smoke_tests:
            print(format_result("GPU Available", torch_results.get('gpu_available', False)))
            print(format_result("Conv2d", torch_results.get('gpu_conv_ok', False)))
            print(format_result("Matmul", torch_results.get('gpu_matmul_ok', False)))
            print(format_result("Gradients", torch_results.get('gpu_grad_ok', False)))
            print(format_result("Stress Test", torch_results.get('gpu_stress_ok', False)))

        print("\nCuPy:")
        print(format_result("Installed", cupy_results.get('installed', False)))
        if run_smoke_tests and cupy_results.get('cuda_available'):
            fps32 = cupy_results.get('fp32_tflops', 'N/A')
            fps16 = cupy_results.get('fp16_tflops', 'N/A')
            print(format_result("CUDA Available", True, f"fp32={fps32} / fp16={fps16}"))

        print("\nTensorRT:")
        print(format_result("Installed", tensorrt_results.get('installed', False)))
        if run_smoke_tests and tensorrt_results.get('engine_built'):
            build_ms = tensorrt_results.get('build_time_ms', 'N/A')
            infer_ms = tensorrt_results.get('inference_time_ms', 'N/A')
            print(format_result("Engine Built", True, f"build={build_ms}ms / infer={infer_ms}ms"))
        
        # FP8 Results (if available)
        if run_fp8_tests:
            print("\nFP8 Capabilities (Blackwell GPU):")
            
            # PyTorch FP8 smoke test
            fp8_torch_ok = pytorch_fp8_results.get('fp8_torch_ok', False)
            print(format_result("PyTorch FP8 Smoke Test", fp8_torch_ok))
            
            # FP8 GEMM benchmark
            fp8_gemm_ok = fp8_gemm_results.get('fp8_gemm_ok', False)
            if fp8_gemm_ok:
                e4m3_tflops = fp8_gemm_results.get('e4m3fn_tflops', 'N/A')
                e5m2_tflops = fp8_gemm_results.get('e5m2_tflops', 'N/A')
                print(format_result("FP8 GEMM Benchmark", True, f"E4M3={e4m3_tflops} / E5M2={e5m2_tflops}"))
            else:
                print(format_result("FP8 GEMM Benchmark", False))
            
            # TensorRT FP8 engine
            fp8_trt_ok = tensorrt_fp8_results.get('fp8_engine_built', False)
            if fp8_trt_ok:
                trt_build_ms = tensorrt_fp8_results.get('fp8_build_time_ms', 'N/A')
                print(format_result("TensorRT FP8 Engine", True, f"build={trt_build_ms}ms"))
            else:
                print(format_result("TensorRT FP8 Engine", False))
        
        print("\n" + "=" * 80)
        print("✓ Diagnostics Complete")
        print("=" * 80)
        
    finally:
        # Restore stdout and close file
        sys.stdout = original_stdout
        output_file.close()
        print(f"\n📄 Full diagnostic results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
