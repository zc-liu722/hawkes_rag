"""按 originidea.md（新版）的公式在 LongMemEval 上做 turn 级动力学评测。

每个问题的执行流程：

1. 把整个对话按 turn 切碎并向量化（qwen，归一化到单位向量）。
2. 按 turn 时间从前往后回放：
   - 衰减：lambda^-(t_i) = lambda^+(t_{prev}) * exp(-beta * (t_i - t_{prev}))
   - 召回分数：score_i = cos(q_i, m) * lambda^-(t_i)^n
   - 取 top intermediate_top_k 视为"被成功调用"
   - 激励：lambda^+(t_i) = lambda^- + score_i * (1 - lambda^-)
     t_last_call 为该记忆上一次成功调用时刻；首次调用时取创建时刻。
   - 当前 turn 自身随后作为新记忆创建，lambda=1，t_last_call=t_i。
3. 在 question_date 时刻把所有记忆衰减到当前，按
   score = cos(q, m) * lambda^-^n 排序，按 session 去重取 top-K。
4. 与 LongMemEval 的 answer_session_ids 计算 session-level Recall@K。
5. 同时跑纯 cosine 基线，使用同一套向量。

不调用任何 LLM。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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


def parse_date(value: Any) -> float:
    text = str(value or "").strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp() / 86400.0
        except ValueError:
            continue
    return 0.0


def evidence_session_ids(record: dict[str, Any]) -> set[str]:
    for key in ("answer_session_ids", "evidence_session_ids", "evidence_sessions"):
        values = record.get(key)
        if not values:
            continue
        ids: list[str] = []
        for value in values:
            if isinstance(value, dict):
                raw = value.get("session_id") or value.get("id")
            else:
                raw = value
            if raw is not None:
                ids.append(str(raw))
        if ids:
            return set(ids)
    return set()


def is_multi_session(record: dict[str, Any]) -> bool:
    qtype = str(record.get("question_type", "")).lower().replace("_", "-")
    if qtype == "multi-session":
        return True
    return len(evidence_session_ids(record)) > 1


def is_single_session(record: dict[str, Any]) -> bool:
    qtype = str(record.get("question_type", "")).lower().replace("_", "-")
    if qtype.startswith("single-session"):
        return True
    return len(evidence_session_ids(record)) == 1


def turn_to_text(turn: Any) -> str:
    if isinstance(turn, dict):
        role = str(turn.get("role", ""))
        content = str(turn.get("content", ""))
        if role and content:
            return f"{role}: {content}"
        return content
    return str(turn)


@dataclass
class Turn:
    session_id: str
    session_index: int
    turn_index: int
    text: str
    time: float
    has_answer: bool = False


def expand_turns(record: dict[str, Any]) -> list[Turn]:
    session_ids = [str(v) for v in record.get("haystack_session_ids") or []]
    sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    turns: list[Turn] = []
    for s_idx, raw_session in enumerate(sessions):
        session_id = session_ids[s_idx] if s_idx < len(session_ids) else f"session_{s_idx}"
        base_time = parse_date(dates[s_idx]) if s_idx < len(dates) else float(s_idx)
        if not isinstance(raw_session, list):
            raw_session = [raw_session]
        for t_idx, turn in enumerate(raw_session):
            text = turn_to_text(turn)
            if not text.strip():
                continue
            has_answer = bool(turn.get("has_answer")) if isinstance(turn, dict) else False
            # 同一 session 内每个 turn 偏移 1 分钟，保持顺序单调且与 session 日期一致
            t = base_time + (t_idx + 1) * (1.0 / 24.0 / 60.0)
            turns.append(
                Turn(
                    session_id=session_id,
                    session_index=s_idx,
                    turn_index=t_idx,
                    text=text,
                    time=t,
                    has_answer=has_answer,
                )
            )
    turns.sort(key=lambda x: (x.time, x.session_index, x.turn_index))
    return turns


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms


def embed_texts(embed_fn, texts: list[str], batch_size: int = 64) -> np.ndarray:
    if hasattr(embed_fn, "encode_texts"):
        vectors = embed_fn.encode_texts(texts, batch_size=batch_size)
        return np.asarray(vectors, dtype=float)
    if hasattr(embed_fn, "model"):
        vectors = embed_fn.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)
    vectors = [np.asarray(embed_fn(text), dtype=float) for text in texts]
    return normalize_rows(np.vstack(vectors))


def session_recall_at_k(retrieved_session_ids: list[str], evidence_ids: set[str]) -> float:
    if not evidence_ids:
        return 0.0
    return len(set(retrieved_session_ids).intersection(evidence_ids)) / len(evidence_ids)


def turn_metrics_at_k(
    retrieved_turn_indices: list[int],
    gold_turn_indices: set[int],
    k: int,
) -> dict[str, float]:
    """gold 取 has_answer=True 的全局 turn 下标，top-k 未做 session 去重。"""
    if not gold_turn_indices:
        return {"recall": 0.0, "hit": 0.0, "mrr": 0.0}
    topk = retrieved_turn_indices[:k]
    hits = [i for i, tid in enumerate(topk) if tid in gold_turn_indices]
    recall = len(set(topk).intersection(gold_turn_indices)) / len(gold_turn_indices)
    hit = 1.0 if hits else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    return {"recall": recall, "hit": hit, "mrr": mrr}


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


def run_originidea(
    turn_vectors: np.ndarray,
    turn_times: np.ndarray,
    query_vector: np.ndarray,
    query_time: float,
    *,
    beta: float,
    n_exp: float,
    intermediate_top_k: int,
    final_top_k: int,
    sessions_for_turns: list[str],
) -> tuple[list[int], list[str]]:
    """按新版 originidea 的动力学回放整段对话。

    维护：
        lambdas[j]        : 上次成功调用后的 lambda^+
        last_update[j]    : 上次写回 lambda^+ 对应的时间（用于衰减）
        last_call_time[j] : 上一次成功调用（含创建）时刻
    """
    n = int(turn_vectors.shape[0])
    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    last_call_time = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)

    for i in range(n):
        t_i = float(turn_times[i])
        if created.any():
            mask = created
            idx_global = np.flatnonzero(mask)
            # (A) 衰减到 t_i —— §2/§4
            decayed = lambdas[idx_global] * np.exp(-beta * (t_i - last_update[idx_global]))
            decayed = np.clip(decayed, 0.0, 1.0)
            # (B) 当前 turn 作为 query 打分 —— §1
            cos_vec = turn_vectors[idx_global] @ turn_vectors[i]
            scores = cos_vec * np.power(decayed, n_exp)
            # (C) 成功调用 = top intermediate_top_k
            order = np.argsort(-scores)
            k = min(intermediate_top_k, len(order))
            called_local = order[:k]
            # (D) 激励 —— §3：lambda^+ = lambda^- + score_i * (1 - lambda^-)
            for j in called_local:
                cos_j = float(cos_vec[j])
                if cos_j <= 0:
                    continue
                lam_minus = float(decayed[j])
                global_j = int(idx_global[j])
                score_i = cos_j * (lam_minus ** n_exp)
                delta = score_i * (1.0 - lam_minus)
                new_lam = min(1.0, lam_minus + delta)
                lambdas[global_j] = new_lam
                last_update[global_j] = t_i
                last_call_time[global_j] = t_i
            # (E) 未被调用的：保持 (lambda^+, t_last_update) 不变即可
            #     因为 exp 衰减无记忆，下次再衰减时仍从原起点出发
        # (F) 当前 turn 作为新记忆创建：lambda=1
        lambdas[i] = 1.0
        last_update[i] = t_i
        last_call_time[i] = t_i
        created[i] = True

    # 最终 question 时刻打分
    decayed_q = lambdas * np.exp(-beta * (query_time - last_update))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    cos_q = turn_vectors @ query_vector
    final_scores = cos_q * np.power(decayed_q, n_exp)
    order = np.argsort(-final_scores)
    full_turn_order = [int(i) for i in order]

    seen_sessions: list[str] = []
    seen_set: set[str] = set()
    chosen_turn_indices: list[int] = []
    for idx in order:
        sid = sessions_for_turns[int(idx)]
        if sid in seen_set:
            continue
        seen_set.add(sid)
        seen_sessions.append(sid)
        chosen_turn_indices.append(int(idx))
        if len(seen_sessions) >= final_top_k:
            break
    return chosen_turn_indices, seen_sessions, full_turn_order


def run_cosine(
    turn_vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    final_top_k: int,
    sessions_for_turns: list[str],
) -> tuple[list[int], list[str], list[int]]:
    cos_q = turn_vectors @ query_vector
    order = np.argsort(-cos_q)
    full_turn_order = [int(i) for i in order]
    seen_sessions: list[str] = []
    seen_set: set[str] = set()
    chosen_turn_indices: list[int] = []
    for idx in order:
        sid = sessions_for_turns[int(idx)]
        if sid in seen_set:
            continue
        seen_set.add(sid)
        seen_sessions.append(sid)
        chosen_turn_indices.append(int(idx))
        if len(seen_sessions) >= final_top_k:
            break
    return chosen_turn_indices, seen_sessions, full_turn_order


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn-level originidea vs cosine on LongMemEval."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/longmemeval_originidea_turns"),
    )
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
    )
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--multi-only", action="store_true", default=True)
    parser.add_argument("--no-multi-only", dest="multi_only", action="store_false")
    parser.add_argument(
        "--single-only",
        action="store_true",
        default=False,
        help="只选 single-session* 题，与 turn 级激励语义一致",
    )
    parser.add_argument("--beta", type=float, default=0.05, help="自然衰减率（每天）")
    parser.add_argument("--n-exp", type=float, default=1.0, help="lambda^- 上的融合指数 n")
    parser.add_argument("--intermediate-top-k", type=int, default=3)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--max-turns-per-question", type=int, default=0, help="0 表示不限制")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing {args.data}. Run `python3 benchmarks/longmemeval/download.py` first."
        )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[originidea-turns] loading {args.data}")
    records = json.loads(args.data.read_text())
    questions = select_questions(
        records,
        n_questions=args.n_questions,
        multi_only=args.multi_only,
        single_only=args.single_only,
    )
    print(
        f"[originidea-turns] selected {len(questions)} questions "
        f"(single_only={args.single_only}, multi_only={args.multi_only})"
    )
    print(
        f"[originidea-turns] hyperparams beta={args.beta} "
        f"n_exp={args.n_exp} inter_top_k={args.intermediate_top_k} final_top_k={args.final_top_k}"
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
        turns = expand_turns(record)
        if args.max_turns_per_question > 0 and len(turns) > args.max_turns_per_question:
            turns = turns[: args.max_turns_per_question]
        if not turns:
            continue
        if question_time <= 0:
            question_time = max(t.time for t in turns) + 1.0

        print(
            f"[originidea-turns] Q{q_idx}/{len(questions)} id={record.get('question_id')} "
            f"turns={len(turns)} sessions={len({t.session_id for t in turns})} "
            f"evidence_sessions={sorted(evidence_ids)}"
        )
        print(f"  Q: {question[:160]}")

        texts = [t.text for t in turns]
        embed_start = time.perf_counter()
        turn_vectors = embed_texts(embed_fn, texts, batch_size=args.embed_batch_size)
        turn_vectors = normalize_rows(turn_vectors)
        query_vector = embed_texts(embed_fn, [question], batch_size=args.embed_batch_size)[0]
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        embed_seconds = time.perf_counter() - embed_start
        print(f"  embed_seconds={embed_seconds:.2f}s")

        sessions_for_turns = [t.session_id for t in turns]
        turn_times = np.asarray([t.time for t in turns], dtype=float)
        gold_turn_indices = {i for i, t in enumerate(turns) if t.has_answer}

        h_indices, h_sessions, h_full_order = run_originidea(
            turn_vectors,
            turn_times,
            query_vector,
            query_time=question_time,
            beta=args.beta,
            n_exp=args.n_exp,
            intermediate_top_k=args.intermediate_top_k,
            final_top_k=args.final_top_k,
            sessions_for_turns=sessions_for_turns,
        )
        h_recall = session_recall_at_k(h_sessions, evidence_ids)
        h_turn = {
            f"k{k}": turn_metrics_at_k(h_full_order, gold_turn_indices, k)
            for k in (1, 3, 5, 10)
        }

        c_indices, c_sessions, c_full_order = run_cosine(
            turn_vectors,
            query_vector,
            final_top_k=args.final_top_k,
            sessions_for_turns=sessions_for_turns,
        )
        c_recall = session_recall_at_k(c_sessions, evidence_ids)
        c_turn = {
            f"k{k}": turn_metrics_at_k(c_full_order, gold_turn_indices, k)
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
                "gold_turn_indices": sorted(gold_turn_indices),
                "n_turns": len(turns),
                "originidea": {
                    "retrieved_session_ids": h_sessions,
                    "retrieved_turn_indices": h_indices,
                    "top10_turn_indices": h_full_order[:10],
                    "session_recall_at_k": h_recall,
                    "turn_metrics": h_turn,
                },
                "cosine": {
                    "retrieved_session_ids": c_sessions,
                    "retrieved_turn_indices": c_indices,
                    "top10_turn_indices": c_full_order[:10],
                    "session_recall_at_k": c_recall,
                    "turn_metrics": c_turn,
                },
                "elapsed_seconds": round(time.perf_counter() - q_start, 2),
            }
        )

        print(f"  gold_turns={sorted(gold_turn_indices)}")
        print(
            f"  cosine     session_recall@{args.final_top_k}={c_recall:.3f} | "
            f"turn_recall@1={c_turn['k1']['recall']:.3f} @5={c_turn['k5']['recall']:.3f} "
            f"mrr@10={c_turn['k10']['mrr']:.3f}"
        )
        print(
            f"  originidea session_recall@{args.final_top_k}={h_recall:.3f} | "
            f"turn_recall@1={h_turn['k1']['recall']:.3f} @5={h_turn['k5']['recall']:.3f} "
            f"mrr@10={h_turn['k10']['mrr']:.3f}"
        )

    summary = {
        "config": {
            "embedding": args.embedding,
            "n_questions": len(per_question),
            "multi_only": args.multi_only,
            "single_only": args.single_only,
            "beta": args.beta,
            "n_exp": args.n_exp,
            "intermediate_top_k": args.intermediate_top_k,
            "final_top_k": args.final_top_k,
            "max_turns_per_question": args.max_turns_per_question,
        },
        "aggregate_recall_at_k": {
            "cosine": float(np.mean(aggregate["cosine"])) if aggregate["cosine"] else 0.0,
            "originidea": float(np.mean(aggregate["originidea"])) if aggregate["originidea"] else 0.0,
        },
        "aggregate_turn_metrics": {
            method: {
                f"k{k}": {
                    metric: float(
                        np.mean([q[method]["turn_metrics"][f"k{k}"][metric] for q in per_question])
                    )
                    if per_question
                    else 0.0
                    for metric in ("recall", "hit", "mrr")
                }
                for k in (1, 3, 5, 10)
            }
            for method in ("cosine", "originidea")
        },
        "per_question": per_question,
        "elapsed_seconds": round(time.perf_counter() - overall_start, 2),
    }

    suffix = "single" if args.single_only else ("multi" if args.multi_only else "all")
    out_json = args.outputs_dir / f"originidea_turns_results_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[originidea-turns] wrote {out_json}")
    print(
        "[originidea-turns] aggregate session_recall@{k}: cosine={c:.3f} originidea={h:.3f}".format(
            k=args.final_top_k,
            c=summary["aggregate_recall_at_k"]["cosine"],
            h=summary["aggregate_recall_at_k"]["originidea"],
        )
    )
    for k in (1, 3, 5, 10):
        c = summary["aggregate_turn_metrics"]["cosine"][f"k{k}"]
        h = summary["aggregate_turn_metrics"]["originidea"][f"k{k}"]
        print(
            f"[originidea-turns] turn @{k}: "
            f"cosine recall={c['recall']:.3f} hit={c['hit']:.3f} mrr={c['mrr']:.3f} | "
            f"originidea recall={h['recall']:.3f} hit={h['hit']:.3f} mrr={h['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
