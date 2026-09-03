"""CUDA library preloading for ONNX Runtime.

ONNX Runtime's CUDA execution provider dlopen()s libcudnn / libcublas at session-creation
time. When those live inside pip's `nvidia-*` packages rather than on the system linker
path, the provider fails to load and **ORT silently falls back to CPU** - the session is
created, inference works, and you only notice because throughput is ~20x lower than it
should be.

Preloading the libraries with RTLD_GLOBAL puts them in the process's global symbol table
so the provider resolves them. Call `ensure_cuda_libs()` before creating any GPU session.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

_LOADED = False

# Order matters: cudnn depends on cublas, which depends on the cuda runtime.
_LIB_ORDER = (
    "cuda_runtime",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "cudnn",
)


def _nvidia_root() -> Path | None:
    for p in sys.path:
        cand = Path(p) / "nvidia"
        if cand.is_dir():
            return cand
    return None


def ensure_cuda_libs() -> list[str]:
    """Preload pip-installed CUDA shared objects. Idempotent. Returns what it loaded."""
    global _LOADED
    if _LOADED:
        return []
    root = _nvidia_root()
    if root is None:
        _LOADED = True
        return []

    loaded: list[str] = []
    for pkg in _LIB_ORDER:
        lib_dir = root / pkg / "lib"
        if not lib_dir.is_dir():
            continue
        for so in sorted(glob.glob(str(lib_dir / "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(os.path.basename(so))
            except OSError:
                pass  # optional/unsupported library - the provider will report if it matters
    _LOADED = True
    return loaded


def assert_gpu_session(session, strict: bool = True) -> str:
    """Verify an ORT session actually got a GPU provider.

    Without this check a CPU fallback is invisible. `strict=True` turns a silent 20x
    slowdown into a loud failure, which is what you want in a benchmark.
    """
    providers = session.get_providers()
    if "CUDAExecutionProvider" not in providers and "TensorrtExecutionProvider" not in providers:
        msg = f"ONNX Runtime fell back to CPU; active providers = {providers}"
        if strict:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
    return providers[0]
