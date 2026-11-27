"""Hardware & kernel preflight checks and TF32 fast-math enablement."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List
import torch


def enable_fast_math() -> bool:
    """Enable TF32 matmul & cuDNN fast math when CUDA is available (safe no-op on CPU/MPS)."""
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def _is_module_installed(module_name: str) -> bool:
    """Safely check whether an optional Python package is importable without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return sys.modules.get(module_name) is not None


def preflight_gpu_check(require_causal_conv1d: bool = False) -> Dict[str, Any]:
    """Inspect CUDA, TF32, Flash SDP, and causal_conv1d availability.

    Returns:
        Structured report dict with keys:
        ``cuda_available``, ``allow_tf32``, ``flash_sdp``,
        ``causal_conv1d_installed``, and ``warnings``.

    Raises:
        RuntimeError: If ``require_causal_conv1d=True`` and ``causal_conv1d`` is not installed.
    """
    cuda_available = bool(torch.cuda.is_available())
    allow_tf32 = bool(
        getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        and getattr(torch.backends.cudnn, "allow_tf32", False)
    )
    flash_sdp = bool(
        cuda_available
        and hasattr(torch.backends.cuda, "flash_sdp_enabled")
        and torch.backends.cuda.flash_sdp_enabled()
    )
    causal_conv1d_installed = _is_module_installed("causal_conv1d")

    warnings: List[str] = []
    if not cuda_available:
        warnings.append("CUDA is not available; GPU-accelerated kernels are inactive.")
    else:
        if not allow_tf32:
            warnings.append("TF32 fast math is disabled; call enable_fast_math() before training or rollout.")
        if not flash_sdp:
            warnings.append("Flash SDP attention kernel is disabled on this CUDA backend.")
    if not causal_conv1d_installed:
        warnings.append("Optional package 'causal_conv1d' is not installed; short convolutions will use fallback kernels.")

    if require_causal_conv1d and not causal_conv1d_installed:
        raise RuntimeError(
            "Required kernel package 'causal_conv1d' is not installed. "
            "Install causal-conv1d or pass require_causal_conv1d=False."
        )

    return {
        "cuda_available": cuda_available,
        "allow_tf32": allow_tf32,
        "flash_sdp": flash_sdp,
        "causal_conv1d_installed": causal_conv1d_installed,
        "warnings": warnings,
    }
