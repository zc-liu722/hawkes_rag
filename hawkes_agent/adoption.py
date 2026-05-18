from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hawkes_agent.memory import RetrievedSegment, normalize_vector


def token_overlap_score(answer: str, memory_text: str) -> float:
    answer_tokens = {t for t in answer.lower().split() if t}
    memory_tokens = {t for t in memory_text.lower().split() if t}
    if not answer_tokens or not memory_tokens:
        return 0.0
    return len(answer_tokens & memory_tokens) / max(1, len(memory_tokens))


def embedding_similarity_score(
    answer: str,
    memory_text: str,
    embed_fn: Callable[[str], np.ndarray],
) -> float:
    a = normalize_vector(np.asarray(embed_fn(answer), dtype=float))
    m = normalize_vector(np.asarray(embed_fn(memory_text), dtype=float))
    if a.size == 0 or m.size == 0:
        return 0.0
    return float(a @ m)


def adopted_ids(
    answer: str,
    segments: list[RetrievedSegment],
    *,
    method: str,
    theta_a: float,
    embed_fn: Callable[[str], np.ndarray] | None = None,
) -> tuple[list[str], dict[str, float]]:
    scores: dict[str, float] = {}
    out: list[str] = []
    for segment in segments:
        if method == "embedding":
            if embed_fn is None:
                raise ValueError("embed_fn is required for embedding adoption")
            score = embedding_similarity_score(answer, segment.text, embed_fn)
        elif method == "token_overlap":
            score = token_overlap_score(answer, segment.text)
        else:
            raise ValueError(f"Unknown adoption method: {method}")
        scores[segment.id] = score
        if score >= theta_a:
            out.append(segment.id)
    return out, scores
