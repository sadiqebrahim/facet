"""Where to run a model, when the GPU is somebody else's too.

A hosted deployment shares its hardware. The failure that motivated this file: an indexing
run on a box whose GPU was 94% occupied by unrelated processes died with

    torch.OutOfMemoryError: Tried to allocate 90.00 MiB ... 41.00 MiB is free

and took the whole import with it, after the faces had already been detected, encoded and
scored. Every stage after that point was lost to an allocation the size of a JPEG.

The policy here is simple and deliberately conservative: **ask whether there is room before
taking it, and fall back to the CPU rather than failing.** A slower answer is a far better
outcome than no answer, and for the small batches this app runs interactively (a handful of
reference faces, a few dozen uploads) the difference is seconds.
"""
from __future__ import annotations

import os

#: Headroom a stage wants before it will claim the GPU. MiVOLO at 384px with a batch of 32
#: is the largest single allocation in the pipeline; 1.5GB clears it with room for the
#: allocator's own fragmentation.
DEFAULT_MIN_FREE_MB = 1500


def gpu_free_mb() -> float | None:
    """Free VRAM in MiB, or None if there is no usable CUDA device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    except Exception:      # noqa: BLE001 - no torch, no driver, no device
        return None


def prefer_gpu(min_free_mb: float = DEFAULT_MIN_FREE_MB) -> bool:
    """Whether to use the GPU for a stage that needs `min_free_mb` of headroom.

    FACET_FORCE_CPU=1 overrides everything, which is what a deployment sharing a card with
    a training job should set.
    """
    if os.environ.get("FACET_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    free = gpu_free_mb()
    return free is not None and free >= min_free_mb


def resolve(device: str = "auto", min_free_mb: float = DEFAULT_MIN_FREE_MB) -> str:
    if device != "auto":
        return device
    return "cuda" if prefer_gpu(min_free_mb) else "cpu"


def is_oom(exc: BaseException) -> bool:
    """Whether an exception is an out-of-memory condition worth retrying on the CPU.

    Matched on the message as well as the type: onnxruntime raises its own allocation
    failures as plain RuntimeError, and they deserve the same fallback as torch's.
    """
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaError"):
        return True
    text = str(exc).lower()
    return any(s in text for s in
               ("out of memory", "cuda error", "cudnn_status_alloc_failed",
                "cublas_status_alloc_failed", "failed to allocate"))
