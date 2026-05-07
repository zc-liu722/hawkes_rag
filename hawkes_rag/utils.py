from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from hawkes_rag.gpu import resolve_torch_device


def as_1d_float_array(value: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D array, got shape {arr.shape}")
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = as_1d_float_array(a)
    b = as_1d_float_array(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(embeddings: np.ndarray, *, device: str | None = None) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {embeddings.shape}")
    if device is not None and device.lower() != "cpu":
        try:
            import torch
        except ImportError:
            pass
        else:
            torch_device = resolve_torch_device(device)
            tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=torch_device)
            normalized = torch.nn.functional.normalize(tensor, p=2, dim=1, eps=1e-12)
            return (normalized @ normalized.T).detach().cpu().numpy().astype(float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def spectral_radius(matrix: np.ndarray) -> float:
    if sparse.issparse(matrix):
        if matrix.size == 0 or matrix.nnz == 0:
            return 0.0
        if min(matrix.shape) <= 2:
            matrix = matrix.toarray()
        else:
            try:
                return float(np.max(np.abs(eigs(matrix, k=1, return_eigenvectors=False))))
            except Exception:
                matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


def project_spectral_radius(matrix: np.ndarray, max_radius: float = 0.95) -> np.ndarray:
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(float)
    else:
        matrix = np.asarray(matrix, dtype=float)
    radius = spectral_radius(matrix)
    if radius > max_radius and radius > 0:
        return matrix * (max_radius / radius)
    return matrix
