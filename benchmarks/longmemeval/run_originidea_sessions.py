"""按 originidea.md 在 LongMemEval 上做 **session 级** 动力学评测。

相对 [run_originidea_turns.py](./run_originidea_turns.py) 的关键差异：

1. 记忆单元 = 整个 session（拼接其内所有 turn 文本），而非单 turn。
2. 每个 session 的时间戳直接取 `haystack_dates[i]`，是 LongMemEval 提供的
   真实日期（精度到分钟，跨度可达数月），从根本上消除 turn 级别脚本里
   `(t_idx+1) * 1/24/60` 造成的"假时间网格"问题。
3. 评测 gold 也提升到 session 维度：
   - `session_recall@K`：retrieved_session_ids ∩ answer_session_ids
   - `session_metrics@k`（recall/hit/mrr/srr）：以 evidence session 在
     按时间衰减后的全量排序中的位置计算；srr 为 SRR@k（top-k 内各 gold 的 1/名次 之和）。

动力学回放与 turn 版本同构：按时间正序播放 sessions，
当前 session 既作为查询又作为新记忆，对历史 sessions 衰减打分、
取 top intermediate_top_k 视为成功调用并按
`lambda^+ = lambda^- + (1-lambda^-) * score_i` 激励。

不调用任何 LLM。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
benchmarks_dir = str(ROOT / "benchmarks")
if benchmarks_dir not in sys.path:
    sys.path.insert(0, benchmarks_dir)

from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402

from longmemeval.run_originidea_turns import (  # noqa: E402
    embed_texts,
    evidence_session_ids,
    is_multi_session,
    is_single_session,
    normalize_rows,
    parse_date,
    session_recall_at_k,
    turn_to_text,
)


@dataclass
class Session:
    session_id: str
    session_index: int
    text: str
    time: float
    is_evidence: bool = False


def expand_sessions(record: dict[str, Any]) -> list[Session]:
    """把每个 haystack session 折叠成一条 (text, real_time) 记忆。"""
    session_ids = [str(v) for v in record.get("haystack_session_ids") or []]
    sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    evidence_ids = evidence_session_ids(record)
    out: list[Session] = []
    for s_idx, raw_session in enumerate(sessions):
        sid = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
        t = parse_date(dates[s_idx]) if s_idx < len(dates) else float(s_idx)
        if not isinstance(raw_session, list):
            raw_session = [raw_session]
        parts: list[str] = []
        for turn in raw_session:
            txt = turn_to_text(turn)
            if txt.strip():
                parts.append(txt)
        merged = "\n".join(parts).strip()
        if not merged:
            continue
        out.append(
            Session(
                session_id=sid,
                session_index=s_idx,
                text=merged,
                time=t,
                is_evidence=sid in evidence_ids,
            )
        )
    out.sort(key=lambda s: (s.time, s.session_index))
    return out


def session_metrics_at_k(
    retrieved_session_indices: list[int],
    gold_session_indices: set[int],
    k: int,
) -> dict[str, float]:
    if not gold_session_indices:
        return {"recall": 0.0, "hit": 0.0, "mrr": 0.0, "srr": 0.0}
    topk = retrieved_session_indices[:k]
    hits = [i for i, sid in enumerate(topk) if sid in gold_session_indices]
    recall = len(set(topk).intersection(gold_session_indices)) / len(gold_session_indices)
    hit = 1.0 if hits else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    pos: dict[int, int] = {}
    for idx, sid in enumerate(retrieved_session_indices):
        if sid not in pos:
            pos[sid] = idx
    srr = 0.0
    for g in gold_session_indices:
        if g not in pos:
            continue
        rank = pos[g] + 1
        if rank <= k:
            srr += 1.0 / rank
    return {"recall": recall, "hit": hit, "mrr": mrr, "srr": srr}


def select_questions(
    records: list[dict[str, Any]],
    *,
    n_questions: int,
    multi_only: bool,
    single_only: bool = False,
) -> list[dict[str, Any]]:
    if single_only:
        candidates = [r for r in records if is_single_session(r)]
    elif multi_only:
        candidates = [r for r in records if is_multi_session(r)]
    else:
        candidates = list(records)
    return candidates[:n_questions]


def compute_mu(lambdas_active: np.ndarray, mu_base: float) -> float:
    """按 originidea.md §1：μ = μ_base + (1-μ_base)·√(1-Ĥ)。

    其中 p_m = λ_m² / Σ λ_j²，H = -Σ p_m·ln(p_m)，Ĥ = H/lnN。

    边界：
      - Ĥ=1（活跃度均匀分散，熵满载） ⇒ √(1-Ĥ)=0 ⇒ μ = μ_base（默认 0.1）
      - Ĥ=0（活跃度极度集中）         ⇒ √(1-Ĥ)=1 ⇒ μ = 1
      - N=1（池中只有一条记忆）        ⇒ 约定 Ĥ=0 ⇒ μ = 1
      - λ 全零（池中无活跃记忆）      ⇒ p 退化为零向量，约定 Ĥ=0 ⇒ μ = 1
    """
    lam2 = np.asarray(lambdas_active, dtype=float) ** 2
    total = float(lam2.sum())
    n = int(lam2.size)
    if n <= 1 or total <= 0.0:
        h_hat = 0.0
    else:
        p = lam2 / total
        # 仅对 p>0 的项求和；约定 0·ln(0)=0
        nz = p > 0.0
        h = -float(np.sum(p[nz] * np.log(p[nz])))
        h_hat = h / math.log(n)
    h_hat = min(max(h_hat, 0.0), 1.0)
    return float(mu_base + (1.0 - mu_base) * math.sqrt(1.0 - h_hat))


def run_originidea_sessions(
    session_vectors: np.ndarray,
    session_times: np.ndarray,
    query_vector: np.ndarray,
    query_time: float,
    *,
    beta: float,
    mu_base: float,
    intermediate_top_k: int,
    final_top_k: int,
    session_ids: list[str],
) -> tuple[list[int], list[str], list[int]]:
    """按 originidea.md 的动力学按时间回放 sessions。

    召回分数：score_i = cos(q_i, m) · [μ + (1-μ)·λ^-(t_i)]
    激励更新：λ^+ = λ^- + (1-λ^-) · score_i
    自然衰减：λ(t) = λ^+ · exp(-β·(t - t_i))
    μ = μ_base + (1-μ_base)·√(1-Ĥ)，由活跃度归一化熵决定。

    返回：(top-K session 的全局下标, top-K session_ids, 全量按分数降序的下标)
    """
    n = int(session_vectors.shape[0])
    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    last_call_time = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)

    for i in range(n):
        t_i = float(session_times[i])
        if created.any():
            idx_global = np.flatnonzero(created)
            # 衰减到 t_i
            decayed = lambdas[idx_global] * np.exp(-beta * (t_i - last_update[idx_global]))
            decayed = np.clip(decayed, 0.0, 1.0)
            # μ 由当前候选池的归一化熵决定
            mu = compute_mu(decayed, mu_base)
            # 召回分数：score = cos · [μ + (1-μ)·λ^-]
            cos_vec = session_vectors[idx_global] @ session_vectors[i]
            mix_vec = mu + (1.0 - mu) * decayed
            scores = cos_vec * mix_vec
            order = np.argsort(-scores)
            k = min(intermediate_top_k, len(order))
            called_local = order[:k]
            for j in called_local:
                score_i = float(scores[j])
                if score_i <= 0.0:
                    continue
                lam_minus = float(decayed[j])
                global_j = int(idx_global[j])
                delta = (1.0 - lam_minus) * score_i
                new_lam = min(1.0, lam_minus + delta)
                lambdas[global_j] = new_lam
                last_update[global_j] = t_i
                last_call_time[global_j] = t_i
        # 当前 session 作为新记忆创建：λ(t_0)=1
        lambdas[i] = 1.0
        last_update[i] = t_i
        last_call_time[i] = t_i
        created[i] = True

    # 最终查询时刻：先衰减，再用相同 μ 公式打分
    decayed_q = lambdas * np.exp(-beta * (query_time - last_update))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    mu_q = compute_mu(decayed_q, mu_base)
    cos_q = session_vectors @ query_vector
    final_scores = cos_q * (mu_q + (1.0 - mu_q) * decayed_q)
    order = np.argsort(-final_scores)
    full_order = [int(i) for i in order]

    chosen_sessions: list[str] = []
    chosen_indices: list[int] = []
    seen: set[str] = set()
    for idx in order:
        sid = session_ids[int(idx)]
        if sid in seen:
            continue
        seen.add(sid)
        chosen_sessions.append(sid)
        chosen_indices.append(int(idx))
        if len(chosen_sessions) >= final_top_k:
            break
    return chosen_indices, chosen_sessions, full_order


def run_cosine_sessions(
    session_vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    final_top_k: int,
    session_ids: list[str],
) -> tuple[list[int], list[str], list[int]]:
    cos_q = session_vectors @ query_vector
    order = np.argsort(-cos_q)
    full_order = [int(i) for i in order]
    chosen_sessions: list[str] = []
    chosen_indices: list[int] = []
    seen: set[str] = set()
    for idx in order:
        sid = session_ids[int(idx)]
        if sid in seen:
            continue
        seen.add(sid)
        chosen_sessions.append(sid)
        chosen_indices.append(int(idx))
        if len(chosen_sessions) >= final_top_k:
            break
    return chosen_indices, chosen_sessions, full_order


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-level originidea (real timestamps) vs cosine on LongMemEval."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/longmemeval_originidea_sessions"),
    )
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
    )
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--multi-only", action="store_true", default=True)
    parser.add_argument("--no-multi-only", dest="multi_only", action="store_false")
    parser.add_argument("--single-only", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument(
        "--mu-base",
        type=float,
        default=0.1,
        help="μ_base，熵满载时的噪声地板，默认 0.1",
    )
    parser.add_argument("--intermediate-top-k", type=int, default=3)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing {args.data}. Run `python3 benchmarks/longmemeval/download.py` first."
        )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[originidea-sessions] loading {args.data}")
    records = json.loads(args.data.read_text())
    questions = select_questions(
        records,
        n_questions=args.n_questions,
        multi_only=args.multi_only,
        single_only=args.single_only,
    )
    print(
        f"[originidea-sessions] selected {len(questions)} questions "
        f"(single_only={args.single_only}, multi_only={args.multi_only})"
    )
    print(
        f"[originidea-sessions] hyperparams beta={args.beta} "
        f"mu_base={args.mu_base} ik={args.intermediate_top_k} fk={args.final_top_k}"
    )

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    aggregate = {"originidea": [], "cosine": []}
    per_question: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for q_idx, record in enumerate(questions, start=1):
        q_start = time.perf_counter()
        question = str(record.get("question", ""))
        evidence_ids = evidence_session_ids(record)
        question_time = parse_date(record.get("question_date"))
        sessions = expand_sessions(record)
        if not sessions:
            continue
        if question_time <= 0:
            question_time = max(s.time for s in sessions) + 1.0

        print(
            f"[originidea-sessions] Q{q_idx}/{len(questions)} id={record.get('question_id')} "
            f"sessions={len(sessions)} evidence_sessions={sorted(evidence_ids)}"
        )
        print(f"  Q: {question[:160]}")

        texts = [s.text for s in sessions]
        embed_start = time.perf_counter()
        session_vectors = normalize_rows(
            embed_texts(embed_fn, texts, batch_size=args.embed_batch_size)
        )
        query_vector = embed_texts(embed_fn, [question], batch_size=args.embed_batch_size)[0]
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        embed_seconds = time.perf_counter() - embed_start
        print(f"  embed_seconds={embed_seconds:.2f}s")

        sids = [s.session_id for s in sessions]
        s_times = np.asarray([s.time for s in sessions], dtype=float)
        gold_session_indices = {i for i, s in enumerate(sessions) if s.is_evidence}

        h_indices, h_sessions, h_full = run_originidea_sessions(
            session_vectors,
            s_times,
            query_vector,
            query_time=question_time,
            beta=args.beta,
            mu_base=args.mu_base,
            intermediate_top_k=args.intermediate_top_k,
            final_top_k=args.final_top_k,
            session_ids=sids,
        )
        h_recall = session_recall_at_k(h_sessions, evidence_ids)
        h_metrics = {
            f"k{k}": session_metrics_at_k(h_full, gold_session_indices, k)
            for k in (1, 3, 5, 10)
        }

        c_indices, c_sessions, c_full = run_cosine_sessions(
            session_vectors,
            query_vector,
            final_top_k=args.final_top_k,
            session_ids=sids,
        )
        c_recall = session_recall_at_k(c_sessions, evidence_ids)
        c_metrics = {
            f"k{k}": session_metrics_at_k(c_full, gold_session_indices, k)
            for k in (1, 3, 5, 10)
        }

        aggregate["originidea"].append(h_recall)
        aggregate["cosine"].append(c_recall)
        per_question.append(
            {
                "question_id": record.get("question_id"),
                "question_type": record.get("question_type"),
                "question": question,
                "answer": record.get("answer"),
                "evidence_session_ids": sorted(evidence_ids),
                "gold_session_indices": sorted(gold_session_indices),
                "n_sessions": len(sessions),
                "originidea": {
                    "retrieved_session_ids": h_sessions,
                    "retrieved_session_indices": h_indices,
                    "top10_session_indices": h_full[:10],
                    "session_recall_at_k": h_recall,
                    "session_metrics": h_metrics,
                },
                "cosine": {
                    "retrieved_session_ids": c_sessions,
                    "retrieved_session_indices": c_indices,
                    "top10_session_indices": c_full[:10],
                    "session_recall_at_k": c_recall,
                    "session_metrics": c_metrics,
                },
                "elapsed_seconds": round(time.perf_counter() - q_start, 2),
            }
        )

        print(
            f"  cosine     sess_recall@{args.final_top_k}={c_recall:.3f} | "
            f"sess_recall@1={c_metrics['k1']['recall']:.3f} @5={c_metrics['k5']['recall']:.3f} "
            f"mrr@10={c_metrics['k10']['mrr']:.3f}"
        )
        print(
            f"  originidea sess_recall@{args.final_top_k}={h_recall:.3f} | "
            f"sess_recall@1={h_metrics['k1']['recall']:.3f} @5={h_metrics['k5']['recall']:.3f} "
            f"mrr@10={h_metrics['k10']['mrr']:.3f}"
        )

    summary = {
        "config": {
            "embedding": args.embedding,
            "n_questions": len(per_question),
            "multi_only": args.multi_only,
            "single_only": args.single_only,
            "beta": args.beta,
            "mu_base": args.mu_base,
            "intermediate_top_k": args.intermediate_top_k,
            "final_top_k": args.final_top_k,
        },
        "aggregate_recall_at_k": {
            "cosine": float(np.mean(aggregate["cosine"])) if aggregate["cosine"] else 0.0,
            "originidea": float(np.mean(aggregate["originidea"])) if aggregate["originidea"] else 0.0,
        },
        "aggregate_session_metrics": {
            method: {
                f"k{k}": {
                    metric: float(
                        np.mean([q[method]["session_metrics"][f"k{k}"][metric] for q in per_question])
                    )
                    if per_question
                    else 0.0
                    for metric in ("recall", "hit", "mrr", "srr")
                }
                for k in (1, 3, 5, 10)
            }
            for method in ("cosine", "originidea")
        },
        "per_question": per_question,
        "elapsed_seconds": round(time.perf_counter() - overall_start, 2),
    }

    suffix = "single" if args.single_only else ("multi" if args.multi_only else "all")
    out_json = args.outputs_dir / f"originidea_sessions_results_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[originidea-sessions] wrote {out_json}")
    print(
        "[originidea-sessions] aggregate session_recall@{k}: cosine={c:.3f} originidea={h:.3f}".format(
            k=args.final_top_k,
            c=summary["aggregate_recall_at_k"]["cosine"],
            h=summary["aggregate_recall_at_k"]["originidea"],
        )
    )
    for k in (1, 3, 5, 10):
        c = summary["aggregate_session_metrics"]["cosine"][f"k{k}"]
        h = summary["aggregate_session_metrics"]["originidea"][f"k{k}"]
        print(
            f"[originidea-sessions] sess @{k}: "
            f"cosine recall={c['recall']:.3f} hit={c['hit']:.3f} mrr={c['mrr']:.3f} | "
            f"originidea recall={h['recall']:.3f} hit={h['hit']:.3f} mrr={h['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
