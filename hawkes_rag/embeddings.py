from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

import numpy as np

from hawkes_rag.gpu import resolve_torch_device


EmbeddingFn = Callable[[str], np.ndarray]


class HashingEmbedding:
    """Deterministic bag-of-words hashing embedding for reproducible fallbacks."""

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def __call__(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for token in tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


class SentenceTransformerEmbedding:
    """Local sentence-transformers embedding wrapper with normalized vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for this embedding; "
                "install hawkes-rag[embeddings] or use --embedding hashing"
            ) from exc
        self.model_name = model_name
        self.device = str(resolve_torch_device(device)) if device is not None else None
        self.batch_size = int(batch_size)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=str(self.cache_dir) if self.cache_dir is not None else None,
        )

    def __call__(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                text,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            ),
            dtype=float,
        )


def make_embedding_fn(
    name: str,
    *,
    fallback_to_hashing: bool = False,
    device: str | None = None,
    batch_size: int = 32,
    cache_dir: str | Path | None = None,
) -> EmbeddingFn:
    normalized = name.lower()
    if normalized == "hashing":
        return HashingEmbedding()
    model_names = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "bge": "BAAI/bge-small-en-v1.5",
    }
    if normalized not in model_names:
        raise ValueError(f"unknown embedding backend: {name}")
    try:
        return SentenceTransformerEmbedding(
            model_names[normalized],
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        if not fallback_to_hashing:
            raise
        if normalized in {"minilm", "bge"}:
            raise RuntimeError(
                f"{exc}; --embedding {normalized} requires sentence-transformers. "
                "Install hawkes-rag[embeddings] or choose --embedding hashing."
            ) from exc
        return HashingEmbedding()


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
