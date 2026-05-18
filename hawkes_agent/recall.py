from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hawkes_agent.adoption import adopted_ids
from hawkes_agent.config import AgentHarnessConfig
from hawkes_agent.dynamics import reinforce_lambda, suppress_lambda
from hawkes_agent.memory import InMemoryVectorStore, RetrievedSegment
from hawkes_agent.rerank import Reranker, make_reranker


class RecallMiddleware:
    """Deterministic memory middleware around a single main agent."""

    def __init__(
        self,
        store: InMemoryVectorStore,
        embed_fn: Callable[[str], np.ndarray],
        config: AgentHarnessConfig,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.config = config
        self.reranker: Reranker = make_reranker(
            config.reranker_backend,
            model_name=config.reranker_model,
        )

    def write_turn(
        self,
        *,
        id: str,
        text: str,
        now: float,
        namespace: str,
        type_class: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.store.add_memory(
            id=id,
            text=text,
            embedding=self.embed_fn(text),
            now=now,
            namespace=namespace,
            type_class=type_class or self.config.dynamics.default_type_class,
            metadata=metadata,
        )

    def recall(
        self,
        query: str,
        *,
        now: float,
        namespace: str,
        top_k: int | None = None,
        use_lambda: bool = True,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float]:
        return self.store.recall(
            self.embed_fn(query),
            now=now,
            namespace=namespace,
            dynamics=self.config.dynamics,
            top_k=top_k or self.config.dynamics.final_top_k,
            use_lambda=use_lambda,
            threshold=threshold,
        )

    def cold_recall(
        self,
        query: str,
        *,
        now: float,
        namespace: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedSegment], float]:
        return self.recall(
            query,
            now=now,
            namespace=namespace,
            top_k=top_k,
            use_lambda=False,
            threshold=None,
        )

    def hot_cold_recall(
        self,
        query: str,
        *,
        now: float,
        namespace: str,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float, dict[str, int]]:
        return self.store.recall_hot_cold(
            self.embed_fn(query),
            now=now,
            namespace=namespace,
            dynamics=self.config.dynamics,
            hot_top_k=self.config.dynamics.hot_top_k,
            cold_top_k=self.config.dynamics.cold_top_k,
            threshold=threshold,
        )

    def hot_cold_reranked_recall(
        self,
        query: str,
        *,
        now: float,
        namespace: str,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float, dict[str, int | float | str | None]]:
        return self.store.recall_hot_cold_reranked(
            query,
            self.embed_fn(query),
            now=now,
            namespace=namespace,
            dynamics=self.config.dynamics,
            reranker=self.reranker,
            threshold=threshold,
        )

    def score_adoption(
        self,
        answer: str,
        segments: list[RetrievedSegment],
    ) -> tuple[list[str], dict[str, float]]:
        return adopted_ids(
            answer,
            segments,
            method=self.config.adoption_method,
            theta_a=self.config.dynamics.theta_a,
            embed_fn=self.embed_fn,
        )

    def reinforce(self, segments: list[RetrievedSegment], adopted: list[str], *, now: float) -> None:
        adopted_set = set(adopted)
        for segment in segments:
            if segment.id not in adopted_set:
                continue
            activation_score = segment.hawkes_score or segment.score
            lam_plus = reinforce_lambda(segment.lambda_minus_snapshot, activation_score)
            self.store.update_lambda(segment.id, lambda_plus=lam_plus, now=now)

    def suppress(
        self,
        segments: list[RetrievedSegment],
        contradicted: list[str],
        *,
        now: float,
    ) -> None:
        contradicted_set = set(contradicted)
        for segment in segments:
            if segment.id not in contradicted_set:
                continue
            lam_plus = suppress_lambda(
                segment.lambda_minus_snapshot,
                max(0.0, segment.cos_at_recall),
            )
            self.store.update_lambda(segment.id, lambda_plus=lam_plus, now=now)

    def prescreen_contradiction_signal(
        self,
        segments: list[RetrievedSegment],
        adopted: list[str],
        adoption_scores: dict[str, float] | None = None,
    ) -> tuple[float, list[RetrievedSegment]]:
        adopted_set = set(adopted)
        adoption_scores = adoption_scores or {}
        suspicious = [
            s
            for s in segments
            if s.id not in adopted_set
            and s.cos_at_recall >= self.config.dynamics.theta_c
            and adoption_scores.get(s.id, 0.0) < self.config.dynamics.theta_a
        ]
        if not suspicious:
            return 0.0, []
        suspicious.sort(key=lambda s: s.cos_at_recall, reverse=True)
        signal = float(suspicious[0].cos_at_recall)
        return signal, suspicious[: self.config.dynamics.contradiction_top_k]
