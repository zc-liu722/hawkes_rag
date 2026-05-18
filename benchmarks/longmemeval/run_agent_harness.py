"""Paper-evaluation harness for the v3 agent memory architecture.

First-version scope:
  - turn-pair LongMemEval streaming path
  - InMemoryVectorStore backend
  - λ scoring with entropy-derived μ
  - embedding-similarity adoption during turn-pair replay
  - cosine and no-adoption ablations
  - no dreaming by default

The harness intentionally reports retrieval/session metrics and cost counters
separately from answer generation. This keeps the first paper loop focused on
whether the memory mechanism finds the right sessions before a backbone LLM is
introduced as another source of variance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
benchmarks_dir = ROOT / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

from hawkes_agent import AgentHarnessConfig, DynamicsConfig, InMemoryVectorStore, ModelRoutingConfig, RecallMiddleware
from hawkes_rag.embeddings import make_embedding_fn
from longmemeval.run_originidea_sessions import (
    Session,
    expand_sessions,
    session_metrics_at_k,
)
from longmemeval.run_originidea_turns import (
    embed_texts,
    evidence_session_ids,
    is_multi_session,
    is_single_session,
    normalize_rows,
    parse_date,
    session_recall_at_k,
)

if TYPE_CHECKING:
    from hawkes_agent.llm import LiteLLMRouter


@dataclass(frozen=True)
class HarnessResult:
    retrieved_session_indices: list[int]
    retrieved_session_ids: list[str]
    full_order_indices: list[int]
    session_recall_at_k: float
    session_metrics: dict[str, dict[str, float]]
    replay_events: dict[str, int]
    event_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayTurn:
    session_id: str
    session_index: int
    sorted_pos: int
    turn_index: int
    query_text: str
    memory_text: str
    time: float
    is_evidence: bool = False


def question_type(record: dict[str, Any]) -> str:
    raw = str(record.get("question_type") or record.get("category") or "unknown")
    return raw.strip().lower().replace(" ", "_") or "unknown"


def select_records(
    records: list[dict[str, Any]],
    *,
    n_questions: int,
    multi_only: bool,
    single_only: bool,
    question_types: set[str] | None,
    question_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if single_only:
        out = [r for r in records if is_single_session(r)]
    elif multi_only:
        out = [r for r in records if is_multi_session(r)]
    else:
        out = list(records)
    if question_types:
        out = [r for r in out if question_type(r) in question_types]
    if question_ids:
        out = [r for r in out if str(r.get("question_id") or "") in question_ids]
    return out[:n_questions]


def replay_type_class(_turn: ReplayTurn, default: str = "stable") -> str:
    # Keep benchmark replay deterministic; type classification can be ablated
    # separately without changing the turn-pair streaming contract.
    return default


def _role_content(turn: Any) -> tuple[str, str]:
    if isinstance(turn, dict):
        return str(turn.get("role") or "").lower(), str(turn.get("content") or "")
    return "", str(turn)


def _format_role_text(role: str, content: str) -> str:
    role = role or "turn"
    content = content.strip()
    return f"{role}: {content}" if content else ""


def expand_replay_turns(record: dict[str, Any], sessions: list[Session]) -> list[ReplayTurn]:
    """Build user-query / user+assistant-memory events for benchmark replay."""
    session_ids = [str(v) for v in record.get("haystack_session_ids") or []]
    raw_sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    evidence_ids = evidence_session_ids(record)
    sorted_pos_by_session_id = {session.session_id: pos for pos, session in enumerate(sessions)}
    out: list[ReplayTurn] = []

    for s_idx, raw_session in enumerate(raw_sessions):
        sid = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
        if sid not in sorted_pos_by_session_id:
            continue
        base_time = parse_date(dates[s_idx]) if s_idx < len(dates) else float(s_idx)
        turns = raw_session if isinstance(raw_session, list) else [raw_session]
        event_idx = 0
        i = 0
        while i < len(turns):
            role, content = _role_content(turns[i])
            if role != "user" or not content.strip():
                i += 1
                continue

            query_text = _format_role_text("user", content)
            memory_parts = [query_text]
            next_i = i + 1
            if next_i < len(turns):
                next_role, next_content = _role_content(turns[next_i])
                if next_role == "assistant" and next_content.strip():
                    memory_parts.append(_format_role_text("assistant", next_content))
                    next_i += 1

            out.append(
                ReplayTurn(
                    session_id=sid,
                    session_index=s_idx,
                    sorted_pos=sorted_pos_by_session_id[sid],
                    turn_index=event_idx,
                    query_text=query_text,
                    memory_text="\n".join(memory_parts),
                    time=base_time + (event_idx + 1) * (1.0 / 24.0 / 60.0),
                    is_evidence=sid in evidence_ids,
                )
            )
            event_idx += 1
            i = next_i

    out.sort(key=lambda x: (x.time, x.session_index, x.turn_index))
    return out


def make_recall_middleware(
    *,
    embed_fn,
    dynamics: DynamicsConfig,
    adoption_method: str,
    reranker_backend: str = "off",
    reranker_model: str | None = None,
    enable_contradiction_micro: bool = False,
) -> RecallMiddleware:
    config = AgentHarnessConfig(
        dynamics=dynamics,
        adoption_method=adoption_method,
        reranker_backend=reranker_backend,
        reranker_model=reranker_model,
        enable_contradiction_micro=enable_contradiction_micro,
        enable_dreaming=False,
    )
    return RecallMiddleware(InMemoryVectorStore(), embed_fn, config)


def final_rank_from_segments(
    middleware: RecallMiddleware,
    *,
    question: str,
    now: float,
    namespace: str,
    use_lambda: bool,
) -> list[int]:
    records = middleware.store.records(namespace)
    segments, _mu = middleware.recall(
        question,
        now=now,
        namespace=namespace,
        top_k=max(1, len(records)),
        use_lambda=use_lambda,
        threshold=-1.0,
    )
    return [int(s.metadata["sorted_pos"]) for s in segments]


def session_ids_from_rank(
    rank: list[int],
    sessions: list[Session],
    *,
    final_top_k: int,
) -> tuple[list[int], list[str]]:
    seen: set[str] = set()
    indices: list[int] = []
    ids: list[str] = []
    for idx in rank:
        sid = sessions[idx].session_id
        if sid in seen:
            continue
        seen.add(sid)
        indices.append(idx)
        ids.append(sid)
        if len(ids) >= final_top_k:
            break
    return indices, ids


def compute_result(
    *,
    rank: list[int],
    sessions: list[Session],
    evidence_ids: set[str],
    final_top_k: int,
    replay_events: dict[str, int],
    event_trace: list[dict[str, Any]] | None = None,
) -> HarnessResult:
    top_indices, top_ids = session_ids_from_rank(rank, sessions, final_top_k=final_top_k)
    gold_indices = {i for i, s in enumerate(sessions) if s.is_evidence}
    metrics = {
        f"k{k}": session_metrics_at_k(rank, gold_indices, k)
        for k in (1, 3, 5, 10)
    }
    return HarnessResult(
        retrieved_session_indices=top_indices,
        retrieved_session_ids=top_ids,
        full_order_indices=rank,
        session_recall_at_k=session_recall_at_k(top_ids, evidence_ids),
        session_metrics=metrics,
        replay_events=dict(replay_events),
        event_trace=list(event_trace or []),
    )


def run_lambda_harness(
    *,
    replay_turns: list[ReplayTurn],
    sessions: list[Session],
    question: str,
    question_time: float,
    evidence_ids: set[str],
    embed_fn,
    dynamics: DynamicsConfig,
    adoption_method: str,
    namespace: str,
    adopt: bool,
    llm: "LiteLLMRouter | None" = None,
    trace_events: bool = False,
) -> HarnessResult:
    middleware = make_recall_middleware(
        embed_fn=embed_fn,
        dynamics=dynamics,
        adoption_method=adoption_method,
    )
    replay_events = {
        "retrieved": 0,
        "adopted": 0,
        "contradicted": 0,
        "written": 0,
        "contradiction_micro_calls": 0,
        "dreaming_calls": 0,
        "llm_calls": 0,
    }
    event_trace: list[dict[str, Any]] = []
    for replay_turn in replay_turns:
        segments, _mu = middleware.recall(
            replay_turn.query_text,
            now=replay_turn.time,
            namespace=namespace,
            top_k=dynamics.intermediate_top_k,
            use_lambda=True,
            threshold=-1.0,
        )
        replay_events["retrieved"] += len(segments)
        if adopt and segments:
            adopted, _scores = middleware.score_adoption(replay_turn.memory_text, segments)
            signal, suspicious = middleware.prescreen_contradiction_signal(segments, adopted, _scores)
            contradicted: list[str] = []
            if llm is not None and signal > 0.0 and suspicious:
                result = llm.classify_contradictions(
                    user_turn=replay_turn.memory_text,
                    candidates=[
                        {
                            "id": s.id,
                            "text": s.text,
                            "cos_at_recall": f"{s.cos_at_recall:.6f}",
                        }
                        for s in suspicious
                    ],
                )
                contradicted = result.contradicted
                replay_events["contradiction_micro_calls"] += 1
                replay_events["llm_calls"] += 1
            middleware.reinforce(segments, adopted, now=replay_turn.time)
            middleware.suppress(segments, contradicted, now=replay_turn.time)
            replay_events["adopted"] += len(adopted)
            replay_events["contradicted"] += len(contradicted)
            retrieved_ids = [s.id for s in segments]
            trace = {
                "turn_id": f"session:{replay_turn.sorted_pos}:turn:{replay_turn.turn_index}",
                "session_id": replay_turn.session_id,
                "retrieved_ids": retrieved_ids,
                "retrieved_segments": [
                    {
                        "id": s.id,
                        "pool": s.retrieval_pool,
                        "cos": round(float(s.cos_at_recall), 6),
                        "hawkes_score": round(float(s.hawkes_score), 6),
                        "rerank_score": round(float(s.rerank_score), 6),
                        "lambda_minus": round(float(s.lambda_minus_snapshot), 6),
                        "adoption_score": round(float(_scores.get(s.id, 0.0)), 6),
                    }
                    for s in segments
                ],
                "adopted_ids": adopted,
                "not_adopted_ids": [sid for sid in retrieved_ids if sid not in set(adopted)],
                "adoption_scores": {sid: round(float(score), 6) for sid, score in _scores.items()},
                "suspicious_ids": [s.id for s in suspicious],
                "contradicted_ids": contradicted,
                "thresholds": {
                    "theta_a": dynamics.theta_a,
                    "theta_c": dynamics.theta_c,
                    "contradiction_top_k": dynamics.contradiction_top_k,
                },
            }
            event_trace.append(trace)
            if trace_events:
                print(
                    "    [trace:lambda] "
                    f"{trace['turn_id']} retrieved={len(retrieved_ids)} "
                    f"adopted={adopted} not_adopted={trace['not_adopted_ids']} "
                    f"suspicious={trace['suspicious_ids']} contradicted={contradicted}"
                )
        middleware.write_turn(
            id=f"{namespace}:session:{replay_turn.sorted_pos}:turn:{replay_turn.turn_index}",
            text=replay_turn.memory_text,
            now=replay_turn.time,
            namespace=namespace,
            type_class=replay_type_class(replay_turn, dynamics.default_type_class),
            metadata={
                "session_id": replay_turn.session_id,
                "session_index": replay_turn.session_index,
                "sorted_pos": replay_turn.sorted_pos,
                "turn_index": replay_turn.turn_index,
                "is_evidence": replay_turn.is_evidence,
                "query_text": replay_turn.query_text,
            },
        )
        replay_events["written"] += 1

    rank = final_rank_from_segments(
        middleware,
        question=question,
        now=question_time,
        namespace=namespace,
        use_lambda=True,
    )
    return compute_result(
        rank=rank,
        sessions=sessions,
        evidence_ids=evidence_ids,
        final_top_k=dynamics.final_top_k,
        replay_events=replay_events,
        event_trace=event_trace,
    )


def run_cosine_baseline(
    *,
    replay_turns: list[ReplayTurn],
    sessions: list[Session],
    question: str,
    question_time: float,
    evidence_ids: set[str],
    embed_fn,
    dynamics: DynamicsConfig,
    namespace: str,
) -> HarnessResult:
    middleware = make_recall_middleware(
        embed_fn=embed_fn,
        dynamics=dynamics,
        adoption_method="embedding",
    )
    for replay_turn in replay_turns:
        middleware.write_turn(
            id=f"{namespace}:session:{replay_turn.sorted_pos}:turn:{replay_turn.turn_index}",
            text=replay_turn.memory_text,
            now=replay_turn.time,
            namespace=namespace,
            type_class=replay_type_class(replay_turn, dynamics.default_type_class),
            metadata={
                "session_id": replay_turn.session_id,
                "session_index": replay_turn.session_index,
                "sorted_pos": replay_turn.sorted_pos,
                "turn_index": replay_turn.turn_index,
                "is_evidence": replay_turn.is_evidence,
                "query_text": replay_turn.query_text,
            },
        )
    rank = final_rank_from_segments(
        middleware,
        question=question,
        now=question_time,
        namespace=namespace,
        use_lambda=False,
    )
    return compute_result(
        rank=rank,
        sessions=sessions,
        evidence_ids=evidence_ids,
        final_top_k=dynamics.final_top_k,
        replay_events={
            "retrieved": 0,
            "adopted": 0,
            "written": len(replay_turns),
            "contradiction_micro_calls": 0,
            "dreaming_calls": 0,
            "llm_calls": 0,
        },
        event_trace=[],
    )


def final_hot_cold_rank(
    middleware: RecallMiddleware,
    *,
    question: str,
    now: float,
    namespace: str,
) -> tuple[list[int], dict[str, Any]]:
    records = middleware.store.records(namespace)
    if not records:
        return [], {"hot": 0, "cold": 0}
    segments, _mu, counts = middleware.store.recall_hot_cold_reranked(
        question,
        middleware.embed_fn(question),
        now=now,
        namespace=namespace,
        dynamics=replace(
            middleware.config.dynamics,
            intermediate_top_k=middleware.config.dynamics.final_top_k,
        ),
        reranker=middleware.reranker,
        threshold=None,
    )
    return [int(s.metadata["sorted_pos"]) for s in segments], counts


def run_full_agent_memory_harness(
    *,
    replay_turns: list[ReplayTurn],
    sessions: list[Session],
    question: str,
    question_time: float,
    evidence_ids: set[str],
    embed_fn,
    dynamics: DynamicsConfig,
    adoption_method: str,
    namespace: str,
    llm: "LiteLLMRouter",
    reranker_backend: str,
    reranker_model: str | None,
    trace_events: bool = False,
) -> HarnessResult:
    try:
        from hawkes_agent.graph import build_memory_replay_graph
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "full_agent_memory requires agent-loop dependencies. "
            "Install requirements-agent-harness.txt or run with --no-full-agent."
        ) from exc

    middleware = make_recall_middleware(
        embed_fn=embed_fn,
        dynamics=dynamics,
        adoption_method=adoption_method,
        reranker_backend=reranker_backend,
        reranker_model=reranker_model,
        enable_contradiction_micro=True,
    )
    graph = build_memory_replay_graph(middleware, llm)
    replay_events = {
        "retrieved": 0,
        "hot_retrieved": 0,
        "cold_retrieved": 0,
        "adopted": 0,
        "contradicted": 0,
        "written": 0,
        "contradiction_micro_calls": 0,
        "cold_triggered": 0,
        "dreaming_calls": 0,
        "llm_calls": 0,
    }
    event_trace: list[dict[str, Any]] = []
    for replay_turn in replay_turns:
        state = graph.invoke(
            {
                "turn_id": f"{namespace}:session:{replay_turn.sorted_pos}:turn:{replay_turn.turn_index}",
                "user_turn": replay_turn.query_text,
                "answer": replay_turn.memory_text,
                "now": replay_turn.time,
                "namespace": namespace,
                "type_class": replay_type_class(replay_turn, dynamics.default_type_class),
                "metadata": {
                    "session_id": replay_turn.session_id,
                    "session_index": replay_turn.session_index,
                    "sorted_pos": replay_turn.sorted_pos,
                    "turn_index": replay_turn.turn_index,
                    "is_evidence": replay_turn.is_evidence,
                    "query_text": replay_turn.query_text,
                },
                "replay_events": replay_events,
            }
        )
        replay_events.update(state.get("replay_events", {}))
        retrieved = state.get("retrieved_segments", [])
        retrieved_ids = [str(s.get("id")) for s in retrieved]
        adopted = [str(v) for v in state.get("adopted_ids", [])]
        adopted_set = set(adopted)
        adoption_scores = {
            str(k): round(float(v), 6)
            for k, v in (state.get("adoption_scores") or {}).items()
        }
        trace = {
            "turn_id": f"session:{replay_turn.sorted_pos}:turn:{replay_turn.turn_index}",
            "session_id": replay_turn.session_id,
            "retrieved_ids": retrieved_ids,
            "retrieved_segments": [
                {
                    "id": str(s.get("id")),
                    "pool": s.get("retrieval_pool"),
                    "cos": round(float(s.get("cos_at_recall", 0.0) or 0.0), 6),
                    "hawkes_score": round(float(s.get("hawkes_score", 0.0) or 0.0), 6),
                    "rerank_score": round(float(s.get("rerank_score", 0.0) or 0.0), 6),
                    "lambda_minus": round(float(s.get("lambda_minus_snapshot", 0.0) or 0.0), 6),
                    "adoption_score": adoption_scores.get(str(s.get("id")), 0.0),
                }
                for s in retrieved
            ],
            "adopted_ids": adopted,
            "not_adopted_ids": [sid for sid in retrieved_ids if sid not in adopted_set],
            "adoption_scores": adoption_scores,
            "suspicious_ids": [str(c.get("id")) for c in state.get("contradiction_candidates", [])],
            "contradicted_ids": [str(v) for v in state.get("contradicted_ids", [])],
            "cold_triggered": int((state.get("retrieval_counts") or {}).get("cold_triggered", 0) or 0),
            "cold_trigger_reason": state.get("cold_trigger_reason"),
            "retrieval_counts": state.get("retrieval_counts") or {},
            "thresholds": {
                "hot_lambda_threshold": dynamics.hot_lambda_threshold,
                "tau_h": dynamics.tau_h,
                "tau_r": dynamics.tau_r,
                "hot_margin_threshold": dynamics.hot_margin_threshold,
                "hot_entropy_threshold": dynamics.hot_entropy_threshold,
                "min_hot_injected": dynamics.min_hot_injected,
                "theta_a": dynamics.theta_a,
                "theta_c": dynamics.theta_c,
                "contradiction_top_k": dynamics.contradiction_top_k,
            },
        }
        event_trace.append(trace)
        if trace_events:
            print(
                "    [trace:full] "
                f"{trace['turn_id']} cold={trace['cold_triggered']} "
                f"reason={trace['cold_trigger_reason']} "
                f"retrieved={len(retrieved_ids)} adopted={adopted} "
                f"not_adopted={trace['not_adopted_ids']} "
                f"suspicious={trace['suspicious_ids']} "
                f"contradicted={trace['contradicted_ids']}"
            )

    rank, final_counts = final_hot_cold_rank(
        middleware,
        question=question,
        now=question_time,
        namespace=namespace,
    )
    replay_events["cold_triggered"] = int(replay_events.get("cold_triggered", 0)) + int(
        final_counts.get("cold_triggered", 0) or 0
    )
    replay_events["final_hot_candidates"] = final_counts.get("hot", 0)
    replay_events["final_cold_candidates"] = final_counts.get("cold", 0)
    replay_events["final_hot_budget"] = final_counts.get("hot_budget", 0)
    replay_events["final_cold_budget"] = final_counts.get("cold_budget", 0)
    return compute_result(
        rank=rank,
        sessions=sessions,
        evidence_ids=evidence_ids,
        final_top_k=dynamics.final_top_k,
        replay_events=replay_events,
        event_trace=event_trace,
    )


def mean_metric(per_question: list[dict[str, Any]], method: str, k: int, metric: str) -> float:
    if not per_question:
        return 0.0
    values = [q[method]["session_metrics"][f"k{k}"][metric] for q in per_question]
    return float(np.mean(values)) if values else 0.0


def aggregate_by_type(per_question: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        grouped[row["question_type"]].append(row)
    out: dict[str, Any] = {}
    for qtype, rows in sorted(grouped.items()):
        out[qtype] = {
            "n_questions": len(rows),
            "methods": {
                method: {
                    "session_recall_at_k": float(
                        np.mean([r[method]["session_recall_at_k"] for r in rows])
                    ),
                    "k10_mrr": float(
                        np.mean([r[method]["session_metrics"]["k10"]["mrr"] for r in rows])
                    ),
                    "k10_srr": float(
                        np.mean([r[method]["session_metrics"]["k10"]["srr"] for r in rows])
                    ),
                }
                for method in methods
            },
        }
    return out


def half_life_days(beta: float) -> float:
    return float("inf") if beta <= 0.0 else float(np.log(2.0) / beta)


def progress(message: str) -> None:
    print(message, flush=True)


def embed_texts_with_progress(embed_fn, texts: list[str], *, batch_size: int, label: str) -> np.ndarray:
    batches: list[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        started = time.perf_counter()
        progress(
            f"[agent-harness] {label} embedding batch {batch_idx}/{total_batches} "
            f"start size={len(batch)}"
        )
        batches.append(np.asarray(embed_texts(embed_fn, batch, batch_size=batch_size), dtype=float))
        progress(
            f"[agent-harness] {label} embedding batch {batch_idx}/{total_batches} "
            f"done elapsed={time.perf_counter() - started:.2f}s"
        )
    return np.vstack(batches) if batches else np.zeros((0, 0), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v3 agent memory paper harness.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/longmemeval_agent_harness"),
    )
    parser.add_argument("--embedding", choices=["minilm", "qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("benchmarks/longmemeval/cache/models"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("benchmarks/longmemeval/cache/embeddings"))
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--multi-only", action="store_true", default=True)
    parser.add_argument("--no-multi-only", dest="multi_only", action="store_false")
    parser.add_argument("--single-only", action="store_true", default=False)
    parser.add_argument(
        "--question-types",
        nargs="*",
        default=None,
        help="Optional normalized question_type filters, e.g. knowledge_update temporal_reasoning.",
    )
    parser.add_argument(
        "--question-ids",
        nargs="*",
        default=None,
        help="Optional exact LongMemEval question_id filters for small debug runs.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Debug cap on expanded sessions for a selected question. <=0 keeps all sessions.",
    )
    parser.add_argument(
        "--max-replay-turns",
        type=int,
        default=0,
        help="Debug cap on expanded replay turns for a selected question. <=0 keeps all turns.",
    )
    parser.add_argument("--beta-volatile", type=float, default=0.20)
    parser.add_argument("--beta-stable", type=float, default=0.05)
    parser.add_argument("--beta-identity", type=float, default=0.001)
    parser.add_argument("--mu-base", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--tau-h", type=float, default=0.05)
    parser.add_argument("--tau-r", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--theta-flat", type=float, default=0.85)
    parser.add_argument("--hot-margin-threshold", type=float, default=0.05)
    parser.add_argument("--hot-entropy-threshold", type=float, default=0.90)
    parser.add_argument("--intermediate-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--hot-top-k", type=int, default=3)
    parser.add_argument("--cold-top-k", type=int, default=3)
    parser.add_argument(
        "--hot-candidate-k",
        type=int,
        default=0,
        help="Deprecated compatibility flag; budgets are derived from --intermediate-top-k.",
    )
    parser.add_argument(
        "--cold-candidate-k",
        type=int,
        default=0,
        help="Deprecated compatibility flag; budgets are derived from --intermediate-top-k.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=0,
        help="Cap hot-path reranker batch by coarse score (after sorting). "
        "<=0 reranks the full hot coarse list (up to --intermediate-top-k).",
    )
    parser.add_argument("--min-hot-injected", type=int, default=3)
    parser.add_argument("--hot-lambda-threshold", type=float, default=0.10)
    parser.add_argument("--theta-a", type=float, default=0.55)
    parser.add_argument("--theta-c", type=float, default=0.65)
    parser.add_argument("--contradiction-top-k", type=int, default=8)
    parser.add_argument(
        "--adoption-method",
        choices=["embedding", "token_overlap"],
        default="embedding",
    )
    parser.add_argument("--no-full-agent", dest="full_agent", action="store_false", default=True)
    parser.add_argument("--main-llm-model", default="deepseek-v4-pro")
    parser.add_argument("--contradiction-model", default="deepseek-v4-pro")
    parser.add_argument(
        "--reranker-backend",
        choices=["off", "heuristic", "cross-encoder", "qwen"],
        default="off",
        help="off: no reranking (hot scores pass through); heuristic: token+prior blend; "
        "cross-encoder/qwen: neural reranker (requires --reranker-model unless qwen default path).",
    )
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--trace-events", dest="trace_events", action="store_true", default=True)
    parser.add_argument("--no-trace-events", dest="trace_events", action="store_false")
    args = parser.parse_args()
    if args.hot_candidate_k > 0 or args.cold_candidate_k > 0:
        progress(
            "[agent-harness] ignoring deprecated --hot-candidate-k/--cold-candidate-k; "
            "budgets use --intermediate-top-k "
            "(hot-only=k, cold-triggered=1/4 hot + 3/4 cold)"
        )

    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run benchmarks/longmemeval/download.py first.")
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else ROOT / args.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dynamics = DynamicsConfig(
        beta_by_type={
            "volatile": args.beta_volatile,
            "stable": args.beta_stable,
            "identity": args.beta_identity,
        },
        default_type_class="stable",
        mu_base=args.mu_base,
        tau=args.tau,
        tau_h=args.tau_h,
        tau_r=args.tau_r,
        alpha=args.alpha,
        theta_flat=args.theta_flat,
        hot_margin_threshold=args.hot_margin_threshold,
        hot_entropy_threshold=args.hot_entropy_threshold,
        intermediate_top_k=args.intermediate_top_k,
        final_top_k=args.final_top_k,
        hot_top_k=args.hot_top_k,
        cold_top_k=args.cold_top_k,
        rerank_top_k=args.rerank_top_k,
        min_hot_injected=args.min_hot_injected,
        hot_lambda_threshold=args.hot_lambda_threshold,
        theta_a=args.theta_a,
        theta_c=args.theta_c,
        contradiction_top_k=args.contradiction_top_k,
    )

    progress(f"[agent-harness] loading {data_path}")
    records = json.loads(data_path.read_text())
    qtypes = {q.lower().replace("-", "_") for q in args.question_types} if args.question_types else None
    qids = {str(q) for q in args.question_ids} if args.question_ids else None
    selected = select_records(
        records,
        n_questions=args.n_questions,
        multi_only=args.multi_only,
        single_only=args.single_only,
        question_types=qtypes,
        question_ids=qids,
    )
    progress(
        f"[agent-harness] selected {len(selected)} questions "
        f"(single_only={args.single_only}, multi_only={args.multi_only}, "
        f"qtypes={sorted(qtypes) if qtypes else 'all'}, qids={sorted(qids) if qids else 'all'})"
    )
    progress(f"[agent-harness] dynamics={asdict(dynamics)} adoption={args.adoption_method}")
    progress(
        "[agent-harness] half_life_days="
        f"volatile:{half_life_days(args.beta_volatile):.2f}, "
        f"stable:{half_life_days(args.beta_stable):.2f}, "
        f"identity:{half_life_days(args.beta_identity):.2f}"
    )
    progress(
        "[agent-harness] thresholds "
        f"hot_lambda>={args.hot_lambda_threshold} tau_h={args.tau_h} tau_r={args.tau_r} "
        f"hot_margin<{args.hot_margin_threshold} hot_entropy>={args.hot_entropy_threshold} "
        f"min_hot_injected={args.min_hot_injected} theta_a={args.theta_a} "
        f"theta_c={args.theta_c} contradiction_top_k={args.contradiction_top_k}"
    )

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
        vector_cache_dir=args.embedding_cache_dir if args.embedding != "hashing" else None,
    )

    methods = ["cosine", "lambda_embedding_adoption", "lambda_no_adoption"]
    if args.full_agent:
        methods.append("full_agent_memory")
    llm = None
    if args.full_agent:
        try:
            from hawkes_agent.llm import LiteLLMRouter
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "full_agent_memory requires LLM dependencies. "
                "Install requirements-agent-harness.txt or run with --no-full-agent."
            ) from exc

        llm = LiteLLMRouter(
            models=ModelRoutingConfig(
                main_llm=args.main_llm_model,
                contradiction_micro=args.contradiction_model,
                dreaming=args.main_llm_model,
            )
        )
    per_question: list[dict[str, Any]] = []
    started = time.perf_counter()

    for idx, record in enumerate(selected, start=1):
        qid = str(record.get("question_id") or f"q{idx}")
        question = str(record.get("question", ""))
        evidence_ids = evidence_session_ids(record)
        question_time = parse_date(record.get("question_date"))
        sessions = expand_sessions(record)
        if not sessions:
            continue
        replay_turns = expand_replay_turns(record, sessions)
        if not replay_turns:
            continue
        if question_time <= 0.0:
            question_time = max(s.time for s in sessions) + 1.0
        if args.max_sessions > 0:
            original_sessions = len(sessions)
            sessions = sessions[: args.max_sessions]
            allowed_session_ids = {s.session_id for s in sessions}
            evidence_ids = {sid for sid in evidence_ids if sid in allowed_session_ids}
            replay_turns = [turn for turn in replay_turns if turn.session_id in allowed_session_ids]
            progress(
                f"[agent-harness] Q{idx} debug cap sessions {original_sessions}->{len(sessions)} "
                f"replay_turns_now={len(replay_turns)}"
            )
        if args.max_replay_turns > 0:
            original_replay_turns = len(replay_turns)
            replay_turns = replay_turns[: args.max_replay_turns]
            progress(
                f"[agent-harness] Q{idx} debug cap replay_turns "
                f"{original_replay_turns}->{len(replay_turns)}"
            )
        namespace = f"{qid}:{idx}"
        progress(
            f"[agent-harness] Q{idx}/{len(selected)} id={qid} "
            f"type={question_type(record)} sessions={len(sessions)} "
            f"replay_turns={len(replay_turns)} evidence={sorted(evidence_ids)}"
        )

        # Warm the embedding cache in batches. The store still owns individual
        # writes, but sentence-transformers is much faster when encoded upfront.
        if hasattr(embed_fn, "model"):
            texts = (
                [s.text for s in sessions]
                + [turn.query_text for turn in replay_turns]
                + [turn.memory_text for turn in replay_turns]
                + [question]
            )
            warm_started = time.perf_counter()
            progress(
                f"[agent-harness] Q{idx} embedding warmup start texts={len(texts)} "
                f"batch_size={args.embed_batch_size}"
            )
            vectors = normalize_rows(
                embed_texts_with_progress(
                    embed_fn,
                    texts,
                    batch_size=args.embed_batch_size,
                    label=f"Q{idx}/warmup",
                )
            )
            lookup = {text: vectors[i] for i, text in enumerate(texts)}
            progress(
                f"[agent-harness] Q{idx} embedding warmup done "
                f"elapsed={time.perf_counter() - warm_started:.2f}s"
            )

            def cached_embed(text: str):
                return lookup.get(text) if text in lookup else np.asarray(embed_fn(text), dtype=float)

            active_embed_fn = cached_embed
        else:
            active_embed_fn = embed_fn

        results: dict[str, HarnessResult] = {}

        method_started = time.perf_counter()
        progress(f"[agent-harness] Q{idx} method=cosine start")
        results["cosine"] = run_cosine_baseline(
                replay_turns=replay_turns,
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                namespace=namespace + ":cosine",
        )
        progress(
            f"[agent-harness] Q{idx} method=cosine done "
            f"elapsed={time.perf_counter() - method_started:.2f}s"
        )

        method_started = time.perf_counter()
        progress(f"[agent-harness] Q{idx} method=lambda_embedding_adoption start")
        results["lambda_embedding_adoption"] = run_lambda_harness(
                replay_turns=replay_turns,
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                adoption_method=args.adoption_method,
                namespace=namespace + ":lambda_adopt",
                adopt=True,
                llm=llm,
                trace_events=args.trace_events,
        )
        progress(
            f"[agent-harness] Q{idx} method=lambda_embedding_adoption done "
            f"elapsed={time.perf_counter() - method_started:.2f}s "
            f"llm_calls={results['lambda_embedding_adoption'].replay_events.get('llm_calls', 0)}"
        )

        method_started = time.perf_counter()
        progress(f"[agent-harness] Q{idx} method=lambda_no_adoption start")
        results["lambda_no_adoption"] = run_lambda_harness(
                replay_turns=replay_turns,
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                adoption_method=args.adoption_method,
                namespace=namespace + ":lambda_no_adopt",
                adopt=False,
                llm=None,
                trace_events=args.trace_events,
        )
        progress(
            f"[agent-harness] Q{idx} method=lambda_no_adoption done "
            f"elapsed={time.perf_counter() - method_started:.2f}s"
        )

        if args.full_agent:
            method_started = time.perf_counter()
            progress(
                f"[agent-harness] Q{idx} method=full_agent_memory start "
                f"reranker_backend={args.reranker_backend}"
            )
            results["full_agent_memory"] = run_full_agent_memory_harness(
                replay_turns=replay_turns,
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                adoption_method=args.adoption_method,
                namespace=namespace + ":full_agent",
                llm=llm,
                reranker_backend=args.reranker_backend,
                reranker_model=args.reranker_model,
                trace_events=args.trace_events,
            )
            progress(
                f"[agent-harness] Q{idx} method=full_agent_memory done "
                f"elapsed={time.perf_counter() - method_started:.2f}s "
                f"llm_calls={results['full_agent_memory'].replay_events.get('llm_calls', 0)}"
            )

        row: dict[str, Any] = {
            "question_id": qid,
            "question_type": question_type(record),
            "question": question,
            "answer": record.get("answer"),
            "evidence_session_ids": sorted(evidence_ids),
            "n_sessions": len(sessions),
            "n_replay_turns": len(replay_turns),
        }
        for method in methods:
            row[method] = asdict(results[method])
            m = results[method].session_metrics["k10"]
            progress(
                f"  {method:25s} recall@{dynamics.final_top_k}="
                f"{results[method].session_recall_at_k:.3f} "
                f"k10_mrr={m['mrr']:.3f} k10_srr={m['srr']:.3f} "
                f"adopted={results[method].replay_events.get('adopted', 0)} "
                f"cold_triggered={results[method].replay_events.get('cold_triggered', 0)}"
            )
        per_question.append(row)

    summary = {
        "config": {
            "embedding": args.embedding,
            "adoption_method": args.adoption_method,
            "model_routing": {
                "main_llm": args.main_llm_model,
                "contradiction_micro": args.contradiction_model,
                "dreaming": args.main_llm_model,
                "reranker": args.reranker_model or args.reranker_backend,
            },
            "reranker_backend": args.reranker_backend,
            "reranker_model": args.reranker_model,
            "dreaming_enabled": False,
            "contradiction_micro_enabled": args.full_agent,
            "storage_backend": "InMemoryVectorStore hot/cold logical pools",
            "grain": "turn_pair",
            "dynamics": asdict(dynamics),
            "n_questions": len(per_question),
            "multi_only": args.multi_only,
            "single_only": args.single_only,
        },
        "aggregate": {
            method: {
                "session_recall_at_k": float(
                    np.mean([q[method]["session_recall_at_k"] for q in per_question])
                )
                if per_question
                else 0.0,
                "metrics": {
                    f"k{k}": {
                        metric: mean_metric(per_question, method, k, metric)
                        for metric in ("recall", "hit", "mrr", "srr")
                    }
                    for k in (1, 3, 5, 10)
                },
                "replay_events": {
                    key: int(sum(q[method]["replay_events"].get(key, 0) for q in per_question))
                    for key in (
                        "retrieved",
                        "hot_retrieved",
                        "cold_retrieved",
                        "cold_triggered",
                        "adopted",
                        "contradicted",
                        "written",
                        "contradiction_micro_calls",
                        "dreaming_calls",
                        "llm_calls",
                        "final_hot_candidates",
                        "final_cold_candidates",
                        "final_hot_budget",
                        "final_cold_budget",
                    )
                },
            }
            for method in methods
        },
        "by_question_type": aggregate_by_type(per_question, methods),
        "per_question": per_question,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    suffix = "single" if args.single_only else ("multi" if args.multi_only else "all")
    out_json = outputs_dir / f"agent_harness_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    progress(f"\n[agent-harness] wrote {out_json}")
    for method in methods:
        agg = summary["aggregate"][method]
        progress(
            f"[agent-harness] {method:25s} recall@{dynamics.final_top_k}="
            f"{agg['session_recall_at_k']:.3f} "
            f"k10_mrr={agg['metrics']['k10']['mrr']:.3f} "
            f"k10_srr={agg['metrics']['k10']['srr']:.3f} "
            f"cold_triggered={agg['replay_events'].get('cold_triggered', 0)} "
            f"llm_calls={agg['replay_events']['llm_calls']}"
        )


if __name__ == "__main__":
    main()
