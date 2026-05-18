"""Paper-evaluation harness for the v3 agent memory architecture.

First-version scope:
  - session-grain LongMemEval streaming path
  - InMemoryVectorStore backend
  - λ scoring with entropy-derived μ
  - embedding-similarity adoption during session replay
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
benchmarks_dir = ROOT / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

from hawkes_agent import AgentHarnessConfig, DynamicsConfig, InMemoryVectorStore, RecallMiddleware
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


@dataclass(frozen=True)
class HarnessResult:
    retrieved_session_indices: list[int]
    retrieved_session_ids: list[str]
    full_order_indices: list[int]
    session_recall_at_k: float
    session_metrics: dict[str, dict[str, float]]
    replay_events: dict[str, int]


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
) -> list[dict[str, Any]]:
    if single_only:
        out = [r for r in records if is_single_session(r)]
    elif multi_only:
        out = [r for r in records if is_multi_session(r)]
    else:
        out = list(records)
    if question_types:
        out = [r for r in out if question_type(r) in question_types]
    return out[:n_questions]


def session_type_class(_session: Session, default: str = "stable") -> str:
    # The first paper harness keeps type classification controlled rather than
    # inferred by an LLM. Datasets or ablations can override this later.
    return default


def make_recall_middleware(
    *,
    embed_fn,
    dynamics: DynamicsConfig,
    adoption_method: str,
) -> RecallMiddleware:
    config = AgentHarnessConfig(
        dynamics=dynamics,
        adoption_method=adoption_method,
        enable_contradiction_micro=False,
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
    )


def run_lambda_harness(
    *,
    sessions: list[Session],
    question: str,
    question_time: float,
    evidence_ids: set[str],
    embed_fn,
    dynamics: DynamicsConfig,
    adoption_method: str,
    namespace: str,
    adopt: bool,
) -> HarnessResult:
    middleware = make_recall_middleware(
        embed_fn=embed_fn,
        dynamics=dynamics,
        adoption_method=adoption_method,
    )
    replay_events = {
        "retrieved": 0,
        "adopted": 0,
        "written": 0,
        "contradiction_micro_calls": 0,
        "dreaming_calls": 0,
        "llm_calls": 0,
    }
    for sorted_pos, session in enumerate(sessions):
        segments, _mu = middleware.recall(
            session.text,
            now=session.time,
            namespace=namespace,
            top_k=dynamics.intermediate_top_k,
            use_lambda=True,
            threshold=-1.0,
        )
        replay_events["retrieved"] += len(segments)
        if adopt and segments:
            adopted, _scores = middleware.score_adoption(session.text, segments)
            middleware.reinforce(segments, adopted, now=session.time)
            replay_events["adopted"] += len(adopted)
        middleware.write_turn(
            id=f"{namespace}:session:{sorted_pos}",
            text=session.text,
            now=session.time,
            namespace=namespace,
            type_class=session_type_class(session, dynamics.default_type_class),
            metadata={
                "session_id": session.session_id,
                "session_index": session.session_index,
                "sorted_pos": sorted_pos,
                "is_evidence": session.is_evidence,
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
    )


def run_cosine_baseline(
    *,
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
    for sorted_pos, session in enumerate(sessions):
        middleware.write_turn(
            id=f"{namespace}:session:{sorted_pos}",
            text=session.text,
            now=session.time,
            namespace=namespace,
            type_class=session_type_class(session, dynamics.default_type_class),
            metadata={
                "session_id": session.session_id,
                "session_index": session.session_index,
                "sorted_pos": sorted_pos,
                "is_evidence": session.is_evidence,
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
            "written": len(sessions),
            "contradiction_micro_calls": 0,
            "dreaming_calls": 0,
            "llm_calls": 0,
        },
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
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
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
    parser.add_argument("--beta-volatile", type=float, default=0.20)
    parser.add_argument("--beta-stable", type=float, default=0.05)
    parser.add_argument("--beta-identity", type=float, default=0.001)
    parser.add_argument("--mu-base", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--intermediate-top-k", type=int, default=3)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--theta-a", type=float, default=0.55)
    parser.add_argument(
        "--adoption-method",
        choices=["embedding", "token_overlap"],
        default="embedding",
    )
    parser.add_argument("--embed-batch-size", type=int, default=64)
    args = parser.parse_args()

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
        intermediate_top_k=args.intermediate_top_k,
        final_top_k=args.final_top_k,
        theta_a=args.theta_a,
    )

    print(f"[agent-harness] loading {data_path}")
    records = json.loads(data_path.read_text())
    qtypes = {q.lower().replace("-", "_") for q in args.question_types} if args.question_types else None
    selected = select_records(
        records,
        n_questions=args.n_questions,
        multi_only=args.multi_only,
        single_only=args.single_only,
        question_types=qtypes,
    )
    print(
        f"[agent-harness] selected {len(selected)} questions "
        f"(single_only={args.single_only}, multi_only={args.multi_only}, qtypes={sorted(qtypes) if qtypes else 'all'})"
    )
    print(f"[agent-harness] dynamics={asdict(dynamics)} adoption={args.adoption_method}")

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
        vector_cache_dir=args.embedding_cache_dir if args.embedding != "hashing" else None,
    )

    methods = ["cosine", "lambda_embedding_adoption", "lambda_no_adoption"]
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
        if question_time <= 0.0:
            question_time = max(s.time for s in sessions) + 1.0
        namespace = f"{qid}:{idx}"
        print(
            f"[agent-harness] Q{idx}/{len(selected)} id={qid} "
            f"type={question_type(record)} sessions={len(sessions)} evidence={sorted(evidence_ids)}"
        )

        # Warm the embedding cache in batches. The store still owns individual
        # writes, but sentence-transformers is much faster when encoded upfront.
        if hasattr(embed_fn, "model"):
            texts = [s.text for s in sessions] + [question]
            vectors = normalize_rows(embed_texts(embed_fn, texts, batch_size=args.embed_batch_size))
            lookup = {text: vectors[i] for i, text in enumerate(texts)}

            def cached_embed(text: str):
                return lookup.get(text) if text in lookup else np.asarray(embed_fn(text), dtype=float)

            active_embed_fn = cached_embed
        else:
            active_embed_fn = embed_fn

        results = {
            "cosine": run_cosine_baseline(
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                namespace=namespace + ":cosine",
            ),
            "lambda_embedding_adoption": run_lambda_harness(
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                adoption_method=args.adoption_method,
                namespace=namespace + ":lambda_adopt",
                adopt=True,
            ),
            "lambda_no_adoption": run_lambda_harness(
                sessions=sessions,
                question=question,
                question_time=question_time,
                evidence_ids=evidence_ids,
                embed_fn=active_embed_fn,
                dynamics=dynamics,
                adoption_method=args.adoption_method,
                namespace=namespace + ":lambda_no_adopt",
                adopt=False,
            ),
        }

        row: dict[str, Any] = {
            "question_id": qid,
            "question_type": question_type(record),
            "question": question,
            "answer": record.get("answer"),
            "evidence_session_ids": sorted(evidence_ids),
            "n_sessions": len(sessions),
        }
        for method in methods:
            row[method] = asdict(results[method])
            m = results[method].session_metrics["k10"]
            print(
                f"  {method:25s} recall@{dynamics.final_top_k}="
                f"{results[method].session_recall_at_k:.3f} "
                f"k10_mrr={m['mrr']:.3f} k10_srr={m['srr']:.3f} "
                f"adopted={results[method].replay_events.get('adopted', 0)}"
            )
        per_question.append(row)

    summary = {
        "config": {
            "embedding": args.embedding,
            "adoption_method": args.adoption_method,
            "model_routing": {
                "main_llm": "deepseek-v4-pro",
                "contradiction_micro": "deepseek-v4-pro",
                "dreaming": "deepseek-v4-pro",
            },
            "dreaming_enabled": False,
            "contradiction_micro_enabled": False,
            "storage_backend": "InMemoryVectorStore",
            "grain": "session",
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
                        "adopted",
                        "written",
                        "contradiction_micro_calls",
                        "dreaming_calls",
                        "llm_calls",
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
    print(f"\n[agent-harness] wrote {out_json}")
    for method in methods:
        agg = summary["aggregate"][method]
        print(
            f"[agent-harness] {method:25s} recall@{dynamics.final_top_k}="
            f"{agg['session_recall_at_k']:.3f} "
            f"k10_mrr={agg['metrics']['k10']['mrr']:.3f} "
            f"k10_srr={agg['metrics']['k10']['srr']:.3f} "
            f"llm_calls={agg['replay_events']['llm_calls']}"
        )


if __name__ == "__main__":
    main()
