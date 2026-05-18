from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np

# Repo root: hawkes_rag/embeddings.py -> hawkes_rag/ -> project root
REPO_ROOT = Path(__file__).resolve().parent.parent


def default_sentence_transformers_cache_dir() -> Path:
    """HF / sentence-transformers snapshot directory shared with benchmark sweeps."""
    return REPO_ROOT / "benchmarks" / "longmemeval" / "cache" / "models"


def default_embedding_vector_cache_dir() -> Path:
    """Persistent embedding-vector cache shared by repeated benchmark sweeps."""
    return REPO_ROOT / "benchmarks" / "longmemeval" / "cache" / "embeddings"


def _resolve_sentence_transformers_cache_dir(cache_dir: Path | str | None) -> Path:
    """Resolve cache path against ``REPO_ROOT`` when relative; default to sweep cache."""
    if cache_dir is None:
        return default_sentence_transformers_cache_dir()
    p = Path(cache_dir)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _resolve_embedding_vector_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        return default_embedding_vector_cache_dir()
    p = Path(cache_dir)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SQLiteEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (embedding_model, text_hash)
            )
            """
        )
        self.conn.commit()

    def get_many(self, embedding_model: str, texts: list[str]) -> dict[int, np.ndarray]:
        if not texts:
            return {}
        hashes = [_text_hash(text) for text in texts]
        unique_hashes = sorted(set(hashes))
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(unique_hashes), 900):
            chunk = unique_hashes[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT text_hash, dim, dtype, vector
                FROM embeddings
                WHERE embedding_model = ? AND text_hash IN ({placeholders})
                """,
                [embedding_model, *chunk],
            )
            for text_hash, dim, dtype, blob in rows:
                vec = np.frombuffer(blob, dtype=np.dtype(dtype)).copy()
                found[str(text_hash)] = vec.reshape(int(dim))
        return {i: found[h] for i, h in enumerate(hashes) if h in found}

    def set_many(self, embedding_model: str, texts: list[str], vectors: np.ndarray) -> None:
        if not texts:
            return
        rows = []
        for text, vector in zip(texts, np.asarray(vectors), strict=True):
            arr = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    embedding_model,
                    _text_hash(text),
                    int(arr.shape[0]),
                    str(arr.dtype),
                    arr.tobytes(),
                )
            )
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (embedding_model, text_hash, dim, dtype, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )


class CachedSentenceTransformerEmbedding:
    def __init__(self, model, *, model_key: str, vector_cache: SQLiteEmbeddingCache) -> None:
        self.model = model
        self.model_key = model_key
        self.vector_cache = vector_cache

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        normalized_texts = [str(text) for text in texts]
        cached = self.vector_cache.get_many(self.model_key, normalized_texts)
        missing_indices = [i for i in range(len(normalized_texts)) if i not in cached]
        if missing_indices:
            missing_texts = [normalized_texts[i] for i in missing_indices]
            encoded = self.model.encode(
                missing_texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoded = np.asarray(encoded, dtype=np.float32)
            self.vector_cache.set_many(self.model_key, missing_texts, encoded)
            for offset, index in enumerate(missing_indices):
                cached[index] = encoded[offset]
        return np.vstack([cached[i] for i in range(len(normalized_texts))]).astype(float, copy=False)

    def __call__(self, text: str) -> np.ndarray:
        return self.encode_texts([str(text)], batch_size=1)[0]


class HashingEmbedding:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for token in str(text).lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        return _normalize(vec)


def make_embedding_fn(
    name: str,
    *,
    device: str = "auto",
    cache_dir: Path | str | None = None,
    vector_cache_dir: Path | str | None = None,
) -> Callable[[str], np.ndarray]:
    name = name.lower()
    if name == "hashing":
        return HashingEmbedding()
    # Back-compat for older configs / CLI (qwen3 → qwen).
    if name == "qwen3":
        name = "qwen"

    from sentence_transformers import SentenceTransformer

    model_name = {
        # Qwen3-Embedding-0.6B: strong multilingual (incl. Chinese) retrieval; fits repeated sweeps.
        "qwen": "Qwen/Qwen3-Embedding-0.6B",
        "bge": "BAAI/bge-small-en-v1.5",
    }.get(name)
    if model_name is None:
        raise ValueError(f"Unknown embedding backend: {name}")

    resolved_cache = _resolve_sentence_transformers_cache_dir(cache_dir)
    kwargs = {
        "cache_folder": str(resolved_cache),
        "local_files_only": True,
    }
    if device != "auto":
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)
    vector_cache = SQLiteEmbeddingCache(
        _resolve_embedding_vector_cache_dir(vector_cache_dir) / "embeddings.sqlite3"
    )
    return CachedSentenceTransformerEmbedding(
        model,
        model_key=f"sentence-transformers:{model_name}",
        vector_cache=vector_cache,
    )
