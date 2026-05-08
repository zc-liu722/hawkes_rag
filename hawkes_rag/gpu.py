from __future__ import annotations

import os
from typing import Any


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "GPU acceleration requires PyTorch. Install hawkes-rag[torch] first."
        ) from exc
    return torch


def resolve_torch_device(device: str | None = None):
    """Resolve a PyTorch device, preferring CUDA then MPS when available."""
    torch = import_torch()
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    torch_device = torch.device(requested)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device, but torch.cuda.is_available() is false")
    if torch_device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("requested MPS device, but torch.backends.mps.is_available() is false")
    return torch_device


def best_float_dtype(torch: Any, device) -> Any:
    requested = os.environ.get("HAWKES_RAG_TORCH_DTYPE", "auto").lower()
    if requested in {"float64", "fp64", "double"}:
        return torch.float64
    if requested in {"float32", "fp32", "single"}:
        return torch.float32
    # GPU runs are memory-bound on the likelihood matrices; float32 keeps them
    # on GPU with much lower peak memory and is also the native fast path.
    return torch.float64 if device.type == "cpu" else torch.float32


def adaptive_cuda_chunk_size(
    torch: Any,
    device,
    dtype,
    n_columns: int,
    *,
    preferred: int,
    minimum: int = 1,
) -> int:
    """Shrink temporary matrix chunks only when CUDA free memory is tight."""
    preferred = max(minimum, int(preferred))
    if device.type != "cuda" or n_columns <= 0:
        return preferred
    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return preferred
    element_size = torch.empty((), dtype=dtype, device=device).element_size()
    bytes_per_row = max(1, int(n_columns)) * element_size * 4
    target_bytes = max(16 * 1024 * 1024, int(free_bytes * 0.50))
    memory_fit = max(minimum, target_bytes // max(1, bytes_per_row))
    return max(minimum, min(preferred, int(memory_fit)))


def torch_spectral_radius(torch: Any, matrix, *, iterations: int = 32):
    """Differentiable power-iteration estimate for non-negative Hawkes alpha."""
    if matrix.numel() == 0:
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    vector = torch.ones((matrix.shape[1],), dtype=matrix.dtype, device=matrix.device)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    for _ in range(iterations):
        vector = matrix @ vector
        norm = torch.linalg.vector_norm(vector).clamp_min(1e-12)
        vector = vector / norm
    rayleigh = torch.dot(vector, matrix @ vector) / torch.dot(vector, vector).clamp_min(1e-12)
    return torch.abs(rayleigh)
