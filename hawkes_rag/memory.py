from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.utils import as_1d_float_array, cosine_similarity, pairwise_cosine, project_spectral_radius


RETRIEVAL_EVENT_WEIGHT = 1.0
MENTION_EVENT_WEIGHT = 0.3


@dataclass
class MemoryItem:
    id: int
    content: str
    embedding: np.ndarray
    created_at: float
    last_accessed: float | None = None
    base_intensity: float = 0.05
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.embedding = as_1d_float_array(self.embedding)


@dataclass
class RetrievalResult:
    memory: MemoryItem
    similarity: float
    intensity: float
    score: float


class HawkesMemoryStore:
    """In-memory atomic-fact store driven by Hawkes activation events."""

    def __init__(
        self,
        *,
        beta: float = 1.0,
        self_excitation: float = 0.7,
        similarity_threshold: float = 0.3,
        similarity_scale: float = 0.3,
        max_radius: float = 0.95,
    ):
        self.beta = float(beta)
        self.self_excitation = float(self_excitation)
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_scale = float(similarity_scale)
        self.max_radius = float(max_radius)
        self.memories: list[MemoryItem] = []
        self.events: list[Event] = []
        self.alpha = np.zeros((0, 0), dtype=float)

    @property
    def n_memories(self) -> int:
        return len(self.memories)

    def add(
        self,
        content: str,
        embedding: np.ndarray | list[float],
        *,
        created_at: float | None = None,
        base_intensity: float = 0.05,
        metadata: dict | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            id=len(self.memories),
            content=content,
            embedding=as_1d_float_array(embedding),
            created_at=self._now(created_at),
            base_intensity=base_intensity,
            metadata=metadata or {},
        )
        self.memories.append(item)
        self._rebuild_alpha_from_similarity()
        return item

    def retrieve(
        self,
        query_embedding: np.ndarray | list[float],
        *,
        top_k: int = 5,
        time: float | None = None,
        record_event: bool = True,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        t = self._now(time)
        query = as_1d_float_array(query_embedding)
        intensities = self.intensities(t)
        results: list[RetrievalResult] = []
        for item, lam in zip(self.memories, intensities):
            sim = cosine_similarity(query, item.embedding)
            results.append(
                RetrievalResult(
                    memory=item,
                    similarity=sim,
                    intensity=float(lam),
                    score=float(sim * lam),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        chosen = results[:top_k]
        if record_event:
            for result in chosen:
                self.record_access(
                    result.memory.id,
                    time=t,
                    weight=RETRIEVAL_EVENT_WEIGHT,
                )
        return chosen

    def record_access(
        self,
        memory_id: int,
        *,
        time: float | None = None,
        weight: float = RETRIEVAL_EVENT_WEIGHT,
    ) -> Event:
        self._check_memory_id(memory_id)
        t = self._now(time)
        event = Event(time=t, memory_id=memory_id, weight=float(weight))
        self.events.append(event)
        self.memories[memory_id].last_accessed = t
        return event

    def record_mentions(
        self,
        memory_ids: Iterable[int],
        *,
        time: float | None = None,
        weight: float = MENTION_EVENT_WEIGHT,
    ) -> list[Event]:
        t = self._now(time)
        return [self.record_access(memory_id, time=t, weight=weight) for memory_id in memory_ids]

    def intensities(self, time: float | None = None) -> np.ndarray:
        if not self.memories:
            return np.zeros(0, dtype=float)
        t = self._now(time)
        process = MultivariateHawkesProcess(self.params())
        return process.intensities(t, self.events)

    def params(self) -> HawkesParams:
        if not self.memories:
            raise ValueError("cannot build Hawkes parameters without memories")
        mu = np.array([m.base_intensity for m in self.memories], dtype=float)
        return HawkesParams(mu=mu, alpha=self.alpha.copy(), beta=self.beta)

    def set_params(self, params: HawkesParams) -> None:
        if params.n_memories != self.n_memories:
            raise ValueError("params size does not match memory store size")
        for item, mu in zip(self.memories, params.mu):
            item.base_intensity = float(mu)
        self.alpha = project_spectral_radius(params.alpha, self.max_radius)
        self.beta = float(params.beta)

    def trajectories(self) -> tuple[list[Event], float]:
        if not self.events:
            return [], 0.0
        start = min(event.time for event in self.events)
        shifted = [
            Event(time=event.time - start, memory_id=event.memory_id, weight=event.weight)
            for event in sorted(self.events, key=lambda e: e.time)
        ]
        horizon = max(event.time for event in shifted) + 1e-6
        return shifted, horizon

    def _rebuild_alpha_from_similarity(self) -> None:
        n = len(self.memories)
        if n == 0:
            self.alpha = np.zeros((0, 0), dtype=float)
            return
        embeddings = np.vstack([m.embedding for m in self.memories])
        sim = pairwise_cosine(embeddings)
        alpha = self.similarity_scale * np.maximum(0.0, sim - self.similarity_threshold)
        np.fill_diagonal(alpha, self.self_excitation)
        self.alpha = project_spectral_radius(alpha, self.max_radius)

    def _check_memory_id(self, memory_id: int) -> None:
        if not (0 <= memory_id < self.n_memories):
            raise IndexError(f"memory_id {memory_id} out of range")

    @staticmethod
    def _now(value: float | None) -> float:
        return float(time_module.time() if value is None else value)
