from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from hawkes_agent.config import DynamicsConfig
from hawkes_agent.dynamics import decayed_lambda, recall_scores
from hawkes_agent.rerank import Reranker, tokenize


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
    retrieval_pool: str | None = None
    bm25_at_recall: float = 0.0
    hawkes_score: float = 0.0
    cold_candidate_score: float = 0.0
    rerank_score: float = 0.0


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

    def _bm25_scores(self, query: str, records: list[MemoryRecord]) -> np.ndarray:
        if not records:
            return np.asarray([], dtype=float)
        query_terms = tokenize(query)
        if not query_terms:
            return np.zeros(len(records), dtype=float)
        tokenized = [tokenize(r.metadata.get("bm25_text") or r.text) for r in records]
        doc_freq: Counter[str] = Counter()
        for terms in tokenized:
            doc_freq.update(set(terms))
        avgdl = sum(len(terms) for terms in tokenized) / max(1, len(tokenized))
        k1 = 1.5
        b = 0.75
        scores: list[float] = []
        n_docs = len(records)
        for terms in tokenized:
            tf = Counter(terms)
            dl = len(terms) or 1
            score = 0.0
            for term in query_terms:
                if tf[term] <= 0:
                    continue
                idf = math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                denom = tf[term] + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
                score += idf * (tf[term] * (k1 + 1.0)) / denom
            scores.append(float(score))
        return np.asarray(scores, dtype=float)

    def _normalize_scores(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        lo = float(np.min(values))
        hi = float(np.max(values))
        if math.isclose(lo, hi):
            return np.ones_like(values, dtype=float) if hi > 0.0 else np.zeros_like(values, dtype=float)
        return (values - lo) / (hi - lo)

    def _hybrid_scores_by_id(
        self,
        query: str,
        records: list[MemoryRecord],
        query_embedding: np.ndarray,
        dynamics: DynamicsConfig,
    ) -> dict[str, float]:
        """Cosine+BM25 blend (same formula as the cold hybrid path), over ``records``."""
        if not records:
            return {}
        q = normalize_vector(query_embedding)
        matrix = np.vstack([r.embedding for r in records])
        cosines = matrix @ q
        bm25 = self._bm25_scores(query, records)
        blended = (
            dynamics.alpha * self._normalize_scores(cosines)
            + (1.0 - dynamics.alpha) * self._normalize_scores(bm25)
        )
        return {r.id: float(blended[i]) for i, r in enumerate(records)}

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
        return self._recall_from_records(
            records,
            query_embedding,
            now=now,
            dynamics=dynamics,
            top_k=top_k,
            use_lambda=use_lambda,
            threshold=threshold,
        )

    def recall_hot_cold(
        self,
        query_embedding: np.ndarray,
        *,
        now: float,
        namespace: str,
        dynamics: DynamicsConfig,
        hot_top_k: int,
        cold_top_k: int,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float, dict[str, int]]:
        records = self.records(namespace)
        if not records:
            return [], 0.0, {"hot": 0, "cold": 0}
        lambdas = self.decayed_lambdas(records, now=now, dynamics=dynamics)
        hot_records = [
            record
            for record, lam in zip(records, lambdas, strict=True)
            if lam >= dynamics.hot_lambda_threshold
        ]
        cold_records = [
            record
            for record, lam in zip(records, lambdas, strict=True)
            if lam < dynamics.hot_lambda_threshold
        ]
        hot_segments, hot_mu = self._recall_from_records(
            hot_records,
            query_embedding,
            now=now,
            dynamics=dynamics,
            top_k=hot_top_k,
            use_lambda=True,
            threshold=threshold,
            retrieval_pool="hot",
        )
        cold_segments, cold_mu = self._recall_from_records(
            cold_records,
            query_embedding,
            now=now,
            dynamics=dynamics,
            top_k=cold_top_k,
            use_lambda=False,
            threshold=threshold,
            retrieval_pool="cold",
        )
        merged: dict[str, RetrievedSegment] = {}
        for segment in [*hot_segments, *cold_segments]:
            current = merged.get(segment.id)
            if current is None or segment.score > current.score:
                merged[segment.id] = segment
        segments = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        mu = hot_mu if hot_segments else cold_mu
        return segments, mu, {"hot": len(hot_segments), "cold": len(cold_segments)}

    def recall_hot_cold_reranked(
        self,
        query: str,
        query_embedding: np.ndarray,
        *,
        now: float,
        namespace: str,
        dynamics: DynamicsConfig,
        reranker: Reranker,
        threshold: float | None = None,
    ) -> tuple[list[RetrievedSegment], float, dict[str, int | float | str | None]]:
        records = self.records(namespace)
        if not records:
            return [], 0.0, {
                "hot": 0,
                "cold": 0,
                "cold_triggered": 0,
                "cold_trigger_reason": None,
                "hot_top1_score": 0.0,
                "hot_margin": 0.0,
                "hot_score_entropy": 0.0,
            }

        q = normalize_vector(query_embedding)
        lambdas = self.decayed_lambdas(records, now=now, dynamics=dynamics)
        hot_records = [
            record
            for record, lam in zip(records, lambdas, strict=True)
            if lam >= dynamics.hot_lambda_threshold
        ]
        cold_records = [
            record
            for record, lam in zip(records, lambdas, strict=True)
            if lam < dynamics.hot_lambda_threshold
        ]

        pool_k = max(0, dynamics.intermediate_top_k)
        hot_segments, mu = self._recall_from_records(
            hot_records,
            q,
            now=now,
            dynamics=dynamics,
            top_k=pool_k,
            use_lambda=True,
            threshold=-1.0,
            retrieval_pool="hot",
        )
        hot_segments.sort(key=lambda s: s.score, reverse=True)
        if dynamics.rerank_top_k > 0:
            hot_segments = hot_segments[: dynamics.rerank_top_k]
        self._apply_rerank(query, hot_segments, reranker)
        hot_ranked = sorted(hot_segments, key=lambda s: s.rerank_score, reverse=True)
        hot_top1_hawkes = max([s.hawkes_score for s in hot_ranked], default=0.0)
        hot_top1_rerank = hot_ranked[0].rerank_score if hot_ranked else 0.0
        hot_margin = (
            hot_ranked[0].rerank_score - hot_ranked[1].rerank_score
            if len(hot_ranked) >= 2
            else hot_top1_rerank
        )
        hot_entropy = self._score_entropy([s.rerank_score for s in hot_ranked])

        trigger_reasons: list[str] = []
        cutoff = dynamics.tau_r if threshold is None else float(threshold)
        if hot_top1_rerank < cutoff or hot_top1_hawkes < dynamics.tau_h:
            trigger_reasons.append("low_confidence")
        if len(hot_ranked) >= 2 and (
            hot_margin < dynamics.hot_margin_threshold
            or hot_entropy >= dynamics.hot_entropy_threshold
        ):
            trigger_reasons.append("flat_hot_distribution")
        if sum(1 for s in hot_ranked if s.rerank_score >= cutoff) < dynamics.min_hot_injected:
            trigger_reasons.append("insufficient_hot_coverage")
        if self._query_asks_old_or_exact(query):
            trigger_reasons.append("explicit_old_or_exact_query")

        meta_tail = {
            "hot_top1_score": float(hot_top1_rerank),
            "hot_top1_hawkes": float(hot_top1_hawkes),
            "hot_margin": float(hot_margin),
            "hot_score_entropy": float(hot_entropy),
        }

        if not trigger_reasons:
            final_segments = [s for s in hot_ranked if s.rerank_score >= cutoff]
            final_segments = final_segments[:pool_k]
            return final_segments, mu, {
                "hot": len(hot_ranked),
                "cold": 0,
                "cold_triggered": 0,
                "cold_trigger_reason": None,
                "hot_budget": pool_k,
                "cold_budget": 0,
                **meta_tail,
            }

        # Once cold retrieval is triggered, keep the total candidate budget
        # aligned to intermediate_top_k: 1/4 hot + 3/4 cold.
        hot_merge_k = max(1, pool_k // 4) if pool_k > 0 else 0
        cold_merge_k = max(0, pool_k - hot_merge_k)
        hot_top_coarse_for_merge = list(hot_segments[:hot_merge_k])
        hybrid_by_id = self._hybrid_scores_by_id(query, records, q, dynamics)
        cold_pool = cold_records if cold_records else records
        cold_sorted = sorted(
            cold_pool,
            key=lambda r: hybrid_by_id[r.id],
            reverse=True,
        )
        hot_merge_ids = {s.id for s in hot_top_coarse_for_merge}
        cold_chosen: list[MemoryRecord] = []
        for r in cold_sorted:
            if r.id in hot_merge_ids:
                continue
            cold_chosen.append(r)
            if len(cold_chosen) >= cold_merge_k:
                break

        idx = {r.id: i for i, r in enumerate(records)}
        matrix = np.vstack([r.embedding for r in records])
        cosines = matrix @ q
        bm25_arr = self._bm25_scores(query, records)
        lam_arr = self.decayed_lambdas(records, now=now, dynamics=dynamics)

        merged_segments: list[RetrievedSegment] = []
        for seg in hot_top_coarse_for_merge:
            i = idx[seg.id]
            h = hybrid_by_id[seg.id]
            merged_segments.append(
                RetrievedSegment(
                    id=seg.id,
                    text=seg.text,
                    score=h,
                    cos_at_recall=float(cosines[i]),
                    lambda_minus_snapshot=float(lam_arr[i]),
                    t_created=seg.t_created,
                    t_last_event=seg.t_last_event,
                    type_class=seg.type_class,
                    namespace=seg.namespace,
                    metadata=dict(seg.metadata),
                    retrieval_pool="hot",
                    bm25_at_recall=float(bm25_arr[i]),
                    hawkes_score=seg.hawkes_score,
                    cold_candidate_score=h,
                    rerank_score=h,
                )
            )
        for r in cold_chosen:
            i = idx[r.id]
            h = hybrid_by_id[r.id]
            merged_segments.append(
                RetrievedSegment(
                    id=r.id,
                    text=r.text,
                    score=h,
                    cos_at_recall=float(cosines[i]),
                    lambda_minus_snapshot=float(lam_arr[i]),
                    t_created=r.t_created,
                    t_last_event=r.t_last_event,
                    type_class=r.type_class,
                    namespace=r.namespace,
                    metadata=dict(r.metadata),
                    retrieval_pool="cold",
                    bm25_at_recall=float(bm25_arr[i]),
                    hawkes_score=0.0,
                    cold_candidate_score=h,
                    rerank_score=h,
                )
            )

        merged_segments.sort(key=lambda s: s.rerank_score, reverse=True)
        final_segments = [s for s in merged_segments if s.rerank_score >= cutoff]
        final_segments = final_segments[:pool_k]
        return final_segments, mu, {
            "hot": len(hot_top_coarse_for_merge),
            "cold": len(cold_chosen),
            "cold_triggered": 1,
            "cold_trigger_reason": ",".join(trigger_reasons),
            "hot_budget": hot_merge_k,
            "cold_budget": cold_merge_k,
            **meta_tail,
        }

    def _recall_from_records(
        self,
        records: list[MemoryRecord],
        query_embedding: np.ndarray,
        *,
        now: float,
        dynamics: DynamicsConfig,
        top_k: int,
        use_lambda: bool,
        threshold: float | None,
        retrieval_pool: str | None = None,
    ) -> tuple[list[RetrievedSegment], float]:
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
            metadata = dict(record.metadata)
            if retrieval_pool is not None:
                metadata["retrieval_pool"] = retrieval_pool
            hawkes_score = score if use_lambda else 0.0
            cold_candidate_score = 0.0 if use_lambda else score
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
                    metadata=metadata,
                    retrieval_pool=retrieval_pool,
                    bm25_at_recall=0.0,
                    hawkes_score=float(hawkes_score),
                    cold_candidate_score=float(cold_candidate_score),
                    rerank_score=float(score),
                )
            )
        return segments, mu

    def _cold_hybrid_candidates(
        self,
        *,
        query: str,
        records: list[MemoryRecord],
        query_embedding: np.ndarray,
        now: float,
        dynamics: DynamicsConfig,
        top_k: int,
    ) -> list[RetrievedSegment]:
        if not records:
            return []
        q = normalize_vector(query_embedding)
        matrix = np.vstack([r.embedding for r in records])
        cosines = matrix @ q
        bm25 = self._bm25_scores(query, records)
        cold_scores = (
            dynamics.alpha * self._normalize_scores(cosines)
            + (1.0 - dynamics.alpha) * self._normalize_scores(bm25)
        )
        lambdas = self.decayed_lambdas(records, now=now, dynamics=dynamics)
        order = np.argsort(-cold_scores)[: max(0, top_k)]
        segments: list[RetrievedSegment] = []
        for idx in order:
            record = records[int(idx)]
            metadata = dict(record.metadata)
            metadata["retrieval_pool"] = "cold"
            segments.append(
                RetrievedSegment(
                    id=record.id,
                    text=record.text,
                    score=float(cold_scores[int(idx)]),
                    cos_at_recall=float(cosines[int(idx)]),
                    lambda_minus_snapshot=float(lambdas[int(idx)]),
                    t_created=record.t_created,
                    t_last_event=record.t_last_event,
                    type_class=record.type_class,
                    namespace=record.namespace,
                    metadata=metadata,
                    retrieval_pool="cold",
                    bm25_at_recall=float(bm25[int(idx)]),
                    hawkes_score=0.0,
                    cold_candidate_score=float(cold_scores[int(idx)]),
                    rerank_score=float(cold_scores[int(idx)]),
                )
            )
        return segments

    def _apply_rerank(self, query: str, segments: list[RetrievedSegment], reranker: Reranker) -> None:
        if not segments:
            return
        priors = [
            s.hawkes_score if s.retrieval_pool == "hot" else s.cold_candidate_score
            for s in segments
        ]
        scores = reranker.score(query, [s.text for s in segments], priors=priors)
        for idx, score in enumerate(scores):
            segment = segments[idx]
            segments[idx] = RetrievedSegment(
                id=segment.id,
                text=segment.text,
                score=float(score),
                cos_at_recall=segment.cos_at_recall,
                lambda_minus_snapshot=segment.lambda_minus_snapshot,
                t_created=segment.t_created,
                t_last_event=segment.t_last_event,
                type_class=segment.type_class,
                namespace=segment.namespace,
                metadata=dict(segment.metadata),
                retrieval_pool=segment.retrieval_pool,
                bm25_at_recall=segment.bm25_at_recall,
                hawkes_score=segment.hawkes_score,
                cold_candidate_score=segment.cold_candidate_score,
                rerank_score=float(score),
            )

    def _score_entropy(self, scores: list[float]) -> float:
        if len(scores) <= 1:
            return 0.0
        arr = np.asarray(scores, dtype=float)
        arr = arr - float(np.min(arr))
        if float(np.sum(arr)) <= 1e-12:
            return 1.0
        probs = arr / float(np.sum(arr))
        entropy = -float(np.sum([p * math.log(p) for p in probs if p > 0.0]))
        return entropy / math.log(len(scores))

    def _query_asks_old_or_exact(self, query: str) -> bool:
        markers = (
            "以前",
            "上次",
            "最早",
            "之前",
            "说过",
            "具体编号",
            "哪一天",
            "哪个房间",
            "previous",
            "before",
            "earlier",
            "last time",
            "first",
            "old",
            "date",
            "which day",
            "room",
            "number",
            "id",
            "invoice",
            "model",
        )
        lower = query.lower()
        if any(marker in lower for marker in markers):
            return True
        return bool(__import__("re").search(r"\b[A-Z]{1,5}[-_]?\d{2,}\b|\b\d{3,}\b", query))
