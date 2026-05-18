from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from hawkes_agent.config import DEFAULT_QWEN_RERANKER_MODEL


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Reranker(Protocol):
    name: str

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        priors: list[float] | None = None,
    ) -> list[float]:
        ...


@dataclass
class HeuristicReranker:
    """Offline fallback for ablations when a Qwen reranker is unavailable.

    It keeps the same call shape as a cross-encoder reranker and uses a stable
    blend of normalized candidate prior plus token overlap. This is deliberately
    simple; production or final paper runs should set a frozen Qwen reranker.
    """

    name: str = "heuristic"

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        priors: list[float] | None = None,
    ) -> list[float]:
        query_tokens = set(tokenize(query))
        overlaps: list[float] = []
        for passage in passages:
            passage_tokens = set(tokenize(passage))
            if not query_tokens or not passage_tokens:
                overlaps.append(0.0)
                continue
            overlaps.append(len(query_tokens & passage_tokens) / len(query_tokens | passage_tokens))
        prior_scores = minmax([float(p) for p in priors]) if priors is not None else [0.0] * len(passages)
        overlap_scores = minmax(overlaps)
        return [
            0.65 * prior + 0.35 * overlap
            for prior, overlap in zip(prior_scores, overlap_scores, strict=True)
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder

        kwargs = {} if device == "auto" else {"device": device}
        self.model = CrossEncoder(model_name, **kwargs)
        self.name = model_name

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        priors: list[float] | None = None,
    ) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, passage) for passage in passages])
        return [float(s) for s in scores]


def make_reranker(
    backend: str = "heuristic",
    *,
    model_name: str | None = None,
    device: str = "auto",
) -> Reranker:
    if backend in {"heuristic", "none", ""}:
        return HeuristicReranker()
    if backend in {"cross-encoder", "qwen"}:
        if backend == "qwen" and not model_name:
            model_name = DEFAULT_QWEN_RERANKER_MODEL
        if not model_name:
            raise ValueError("--reranker-model is required for cross-encoder/qwen reranking")
        return CrossEncoderReranker(model_name, device=device)
    raise ValueError(f"Unknown reranker backend: {backend}")
