from __future__ import annotations

import numpy as np

from hawkes_rag.core import HawkesParams, diagonal_only
from hawkes_rag.memory import HawkesMemoryStore, RetrievalResult
from hawkes_rag.utils import cosine_similarity


def naive_retrieve(
    store: HawkesMemoryStore,
    query_embedding: np.ndarray | list[float],
    *,
    top_k: int = 5,
) -> list[RetrievalResult]:
    query = np.asarray(query_embedding, dtype=float)
    results = []
    for item in store.memories:
        sim = cosine_similarity(query, item.embedding)
        results.append(RetrievalResult(memory=item, similarity=sim, intensity=1.0, score=sim))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def diagonal_hawkes_retrieve(
    store: HawkesMemoryStore,
    query_embedding: np.ndarray | list[float],
    *,
    top_k: int = 5,
    time: float | None = None,
) -> list[RetrievalResult]:
    original = store.params()
    try:
        store.set_params(diagonal_only(original))
        return store.retrieve(query_embedding, top_k=top_k, time=time, record_event=False)
    finally:
        store.set_params(HawkesParams(mu=original.mu, alpha=original.alpha, beta=original.beta))
