from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from hawkes_agent.config import DynamicsConfig
from hawkes_agent.dynamics import decayed_lambda, recall_scores


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return arr
    return arr / norm


@dataclass
class MemoryRecord:
    id: str
    text: str
    embedding: np.ndarray
    lambda_plus: float
    t_last_event: float
    t_created: float
    type_class: str
    namespace: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedSegment:
    id: str
    text: str
    score: float
    cos_at_recall: float
    lambda_minus_snapshot: float
    t_created: float
    t_last_event: float
    type_class: str
    namespace: str
    metadata: dict


class InMemoryVectorStore:
    """Single-pool vector + mutable payload store for reproducible evals."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def clear(self) -> None:
        self._records.clear()

    def upsert(self, record: MemoryRecord) -> None:
        record.embedding = normalize_vector(record.embedding)
        self._records[record.id] = record

    def add_memory(
        self,
        *,
        id: str,
        text: str,
        embedding: np.ndarray,
        now: float,
        namespace: str,
        type_class: str,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=id,
            text=text,
            embedding=normalize_vector(embedding),
            lambda_plus=1.0,
            t_last_event=float(now),
            t_created=float(now),
            type_class=type_class,
            namespace=namespace,
            metadata=dict(metadata or {}),
        )
        self.upsert(record)
        return record

    def get(self, id: str) -> MemoryRecord:
        return self._records[id]

    def records(self, namespace: str | None = None) -> list[MemoryRecord]:
        values = list(self._records.values())
        if namespace is None:
            return values
        return [r for r in values if r.namespace == namespace]

    def update_lambda(self, id: str, *, lambda_plus: float, now: float) -> None:
        record = self._records[id]
        record.lambda_plus = min(max(float(lambda_plus), 0.0), 1.0)
        record.t_last_event = float(now)

    def decayed_lambdas(
        self,
        records: Iterable[MemoryRecord],
        *,
        now: float,
        dynamics: DynamicsConfig,
    ) -> np.ndarray:
        return np.asarray(
            [
                decayed_lambda(
                    r.lambda_plus,
                    dynamics.beta_for(r.type_class),
                    now,
                    r.t_last_event,
                )
                for r in records
            ],
            dtype=float,
        )

    def recall(
        self,
        query_embedding: np.ndarray,
        *,
        now: float,
        namespace: str,
        dynamics: DynamicsConfig,
        top_k: int,
        use_lambda: bool = True,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float]:
        records = self.records(namespace)
        if not records:
            return [], 0.0
        q = normalize_vector(query_embedding)
        matrix = np.vstack([r.embedding for r in records])
        cosines = matrix @ q
        lambdas = self.decayed_lambdas(records, now=now, dynamics=dynamics)
        if use_lambda:
            scores, mu = recall_scores(
                cosines,
                lambdas,
                mu_base=dynamics.mu_base,
                cosine_floor=dynamics.cosine_floor,
            )
        else:
            scores = np.maximum(cosines, dynamics.cosine_floor)
            mu = 1.0
        order = np.argsort(-scores)
        cutoff = dynamics.tau if threshold is None else float(threshold)
        segments: list[RetrievedSegment] = []
        for idx in order:
            if len(segments) >= top_k:
                break
            score = float(scores[int(idx)])
            if score < cutoff:
                continue
            record = records[int(idx)]
            segments.append(
                RetrievedSegment(
                    id=record.id,
                    text=record.text,
                    score=score,
                    cos_at_recall=float(cosines[int(idx)]),
                    lambda_minus_snapshot=float(lambdas[int(idx)]),
                    t_created=record.t_created,
                    t_last_event=record.t_last_event,
                    type_class=record.type_class,
                    namespace=record.namespace,
                    metadata=dict(record.metadata),
                )
            )
        return segments, mu
