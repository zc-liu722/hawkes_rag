from __future__ import annotations

import hashlib
import re
import sys
from typing import Callable

import numpy as np


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

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for this embedding; "
                "install hawkes-rag[embeddings] or use --embedding hashing"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=float)


def make_embedding_fn(name: str, *, fallback_to_hashing: bool = True) -> EmbeddingFn:
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
        return SentenceTransformerEmbedding(model_names[normalized])
    except Exception as exc:
        if not fallback_to_hashing:
            raise
        print(f"{exc}; falling back to hashing embeddings", file=sys.stderr)
        return HashingEmbedding()


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
