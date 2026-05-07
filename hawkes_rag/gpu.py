from __future__ import annotations

from typing import Any


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
    # MPS has much better operator coverage and performance in float32.
    return torch.float32 if device.type == "mps" else torch.float64


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
