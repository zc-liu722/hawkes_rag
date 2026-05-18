"""LongMemEval multi-session evaluation: cosine baseline vs incentive-decay (originidea.md).

Each haystack session is one memory unit (text = full session, embedding from
Qwen3-Embedding-0.6B).  Sessions are replayed in chronological order using the
session-level timestamp from haystack_dates; when a session is consumed it
fires exactly one retrieval event against the previously entered sessions
(query = the new session's embedding) and updates the top-K memories' lambda
according to the incentive rule in originidea.md (sec. 3).  After the whole
haystack has been streamed, the actual question is issued at question_date
and ranked.

No turn-level processing is performed: there is no synthetic intra-session
turn timestamp and no per-turn embedding.

This script focuses on a beta sweep; mu_min/mu_max/k stay at their defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hawkes_rag.embeddings import default_sentence_transformers_cache_dir  # noqa: E402

DATA_PATH = ROOT / "benchmarks" / "longmemeval" / "cache" / "longmemeval_s.json"
CACHE_DIR = default_sentence_transformers_cache_dir()

DATE_FMT = "%Y/%m/%d (%a) %H:%M"


class InvalidDateError(ValueError):
    """Raised when a haystack date or question_date cannot be parsed as DATE_FMT."""


def parse_ts(text: str) -> float:
    if not isinstance(text, str) or not text:
        raise InvalidDateError(f"empty or non-string date: {text!r}")
    try:
        return datetime.strptime(text, DATE_FMT).timestamp()
    except (ValueError, TypeError) as exc:
        raise InvalidDateError(
            f"cannot parse date {text!r} with format {DATE_FMT!r}: {exc}"
        ) from exc


def session_text(session: Sequence[dict]) -> str:
    parts = []
    for turn in session:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


@dataclass
class HawkesConfig:
    name: str
    beta_per_day: float
    mu_min: float = 0.1
    mu_max: float = 1.0
    k: float = 0.5
    top_k_update: int = 5
    cosine_floor: float = 0.0

    @property
    def beta_per_sec(self) -> float:
        return self.beta_per_day / 86400.0


@dataclass
class Memory:
    idx: int
    session_id: str
    created_at: float
    last_event_at: float
    lambda_after: float = 1.0


def normalised_entropy(lambdas: np.ndarray) -> float:
    if lambdas.size <= 1:
        return 0.0
    pos = lambdas[lambdas > 0]
    if pos.size == 0:
        return 0.0
    sq = pos ** 2
    p = sq / sq.sum()
    h = -float((p * np.log(p)).sum())
    return h / math.log(lambdas.size)


def mu_from_entropy(h_hat: float, cfg: HawkesConfig) -> float:
    return cfg.mu_min + (cfg.mu_max - cfg.mu_min) * (1.0 - h_hat) ** cfg.k


def current_lambdas(mems: List[Memory], now: float, beta_sec: float) -> np.ndarray:
    if not mems:
        return np.zeros(0, dtype=np.float64)
    arr = np.fromiter(
        (m.lambda_after * math.exp(-beta_sec * max(0.0, now - m.last_event_at)) for m in mems),
        dtype=np.float64,
        count=len(mems),
    )
    return arr


def hawkes_score(cosine: np.ndarray, lambdas: np.ndarray, cfg: HawkesConfig) -> Tuple[np.ndarray, float]:
    cos_clip = np.maximum(cosine, cfg.cosine_floor)
    h_hat = normalised_entropy(lambdas)
    mu = mu_from_entropy(h_hat, cfg)
    weight = mu + (1.0 - mu) * lambdas
    return cos_clip * weight, mu


def update_lambdas(
    mems: List[Memory],
    indices: np.ndarray,
    scores: np.ndarray,
    lambdas_now: np.ndarray,
    now: float,
) -> None:
    for pos in indices:
        m = mems[int(pos)]
        lam_minus = float(lambdas_now[int(pos)])
        s = float(scores[int(pos)])
        if s < 0.0:
            s = 0.0
        lam_plus = lam_minus + (1.0 - lam_minus) * s
        lam_plus = min(max(lam_plus, 0.0), 1.0)
        m.lambda_after = lam_plus
        m.last_event_at = now


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(query)):
        query = np.where(np.isfinite(query), query, 0.0)
    q_norm = float(np.linalg.norm(query))
    if not np.isfinite(q_norm) or q_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    qn = query / q_norm

    matrix_safe = np.where(np.isfinite(matrix), matrix, 0.0)
    m_norms = np.linalg.norm(matrix_safe, axis=1, keepdims=True)
    bad_rows = (~np.isfinite(m_norms.squeeze(-1))) | (m_norms.squeeze(-1) == 0.0)
    safe_norms = np.where(bad_rows[:, None], 1.0, m_norms)
    matrix_safe = np.where(bad_rows[:, None], 0.0, matrix_safe)
    mn = matrix_safe / safe_norms
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = mn @ qn
    out = np.where(np.isfinite(out), out, 0.0)
    if bad_rows.any():
        out = out.copy()
        out[bad_rows] = 0.0
    return out


def evaluate_question(
    question: dict,
    encoder,
    cfgs: List[HawkesConfig],
) -> Dict[str, Dict[str, float]]:
    sess_ids: List[str] = list(question["haystack_session_ids"])
    sess_dates: List[str] = list(question["haystack_dates"])
    sessions: List[Sequence[dict]] = list(question["haystack_sessions"])
    answer_ids = set(question["answer_session_ids"])

    try:
        parsed_ts = [parse_ts(d) for d in sess_dates]
        question_ts = parse_ts(question["question_date"])
    except InvalidDateError as exc:
        qid = question.get("question_id", "<unknown>")
        print(
            f"  [warn] skipping qid={qid}: invalid date ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return {}

    order = sorted(range(len(sess_ids)), key=lambda i: parsed_ts[i])
    ordered_ids = [sess_ids[i] for i in order]
    ordered_ts = [parsed_ts[i] for i in order]
    ordered_sessions = [sessions[i] for i in order]

    full_texts = [session_text(s) for s in ordered_sessions]

    mem_emb = encoder.encode(full_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    real_q_emb = encoder.encode([question["question"]], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)[0]

    relevant_indices = [pos for pos, sid in enumerate(ordered_ids) if sid in answer_ids]
    if not relevant_indices:
        return {}

    final_cos = cosine_matrix(real_q_emb, mem_emb)
    results: Dict[str, Dict[str, float]] = {}

    cosine_rank = np.argsort(-final_cos)
    results["cosine"] = compute_metrics(cosine_rank, relevant_indices)

    n_sessions = len(ordered_ids)

    for cfg in cfgs:
        mems: List[Memory] = [
            Memory(idx=p, session_id=ordered_ids[p], created_at=ordered_ts[p], last_event_at=ordered_ts[p], lambda_after=1.0)
            for p in range(n_sessions)
        ]
        beta_sec = cfg.beta_per_sec
        session_entered = [False] * n_sessions

        for owner in range(n_sessions):
            ts = ordered_ts[owner]

            existing = [j for j in range(n_sessions) if session_entered[j] and j != owner]
            if existing:
                emb_matrix = mem_emb[existing]
                cos = cosine_matrix(mem_emb[owner], emb_matrix)
                sub_mems = [mems[j] for j in existing]
                lambdas_now = current_lambdas(sub_mems, ts, beta_sec)
                scores, _mu = hawkes_score(cos, lambdas_now, cfg)
                top_local = np.argsort(-scores)[: cfg.top_k_update]
                for rel_pos in top_local:
                    mem_pos = existing[int(rel_pos)]
                    lam_minus = float(lambdas_now[int(rel_pos)])
                    s = float(scores[int(rel_pos)])
                    if s < 0.0:
                        s = 0.0
                    lam_plus = lam_minus + (1.0 - lam_minus) * s
                    lam_plus = min(max(lam_plus, 0.0), 1.0)
                    mems[mem_pos].lambda_after = lam_plus
                    mems[mem_pos].last_event_at = ts

            mems[owner].last_event_at = ts
            mems[owner].created_at = ts
            mems[owner].lambda_after = 1.0
            session_entered[owner] = True

        lambdas_q = current_lambdas(mems, question_ts, beta_sec)
        scores_q, mu_q = hawkes_score(final_cos, lambdas_q, cfg)
        rank = np.argsort(-scores_q)
        metrics = compute_metrics(rank, relevant_indices)
        metrics["mu_at_query"] = mu_q
        metrics["entropy_norm"] = normalised_entropy(lambdas_q)
        metrics["lambda_mean"] = float(np.mean(lambdas_q))
        results[cfg.name] = metrics

    return results


def compute_metrics(rank: np.ndarray, relevant: Sequence[int]) -> Dict[str, float]:
    rel_set = set(int(i) for i in relevant)
    k = len(rel_set)
    hits = 0
    first_hit = None
    for r, idx in enumerate(rank, start=1):
        if int(idx) in rel_set:
            if first_hit is None:
                first_hit = r
            if r <= k:
                hits += 1
    recall_at_k = hits / k if k else 0.0
    mrr = 1.0 / first_hit if first_hit else 0.0
    return {"recall_at_k": recall_at_k, "mrr": mrr, "k": float(k), "first_hit_rank": float(first_hit or 0)}


def aggregate(results_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    keys = set()
    for r in results_list:
        keys.update(r.keys())
    agg: Dict[str, Dict[str, float]] = {}
    for key in keys:
        recs, mrrs, mus, ents = [], [], [], []
        for r in results_list:
            if key in r:
                recs.append(r[key]["recall_at_k"])
                mrrs.append(r[key]["mrr"])
                if "mu_at_query" in r[key]:
                    mus.append(r[key]["mu_at_query"])
                    ents.append(r[key]["entropy_norm"])
        agg[key] = {
            "recall_at_k": float(np.mean(recs)) if recs else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "n": len(recs),
        }
        if mus:
            agg[key]["mu_at_query_mean"] = float(np.mean(mus))
            agg[key]["entropy_norm_mean"] = float(np.mean(ents))
    return agg


def build_configs() -> List[HawkesConfig]:
    half_life_days = 1.0
    beta_per_day = math.log(2) / half_life_days
    cfgs = []
    name = f"hawkes_T1d"
    cfgs.append(HawkesConfig(name=name, beta_per_day=beta_per_day))
    return cfgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "longmemeval" / "eval_report.json")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit(f"missing dataset: {DATA_PATH}")

    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.model, cache_folder=str(CACHE_DIR))

    data = json.loads(DATA_PATH.read_text())
    pool = [d for d in data if d["question_type"] == "multi-session"]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = pool[: args.num_questions]

    cfgs = build_configs()
    per_question: List[Dict[str, Dict[str, float]]] = []
    skipped: List[str] = []
    print(f"Evaluating {len(selected)} multi-session questions...", flush=True)
    for i, q in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] qid={q['question_id']} sessions={len(q['haystack_sessions'])}", flush=True)
        res = evaluate_question(q, encoder, cfgs)
        per_question.append(res)
        if not res:
            skipped.append(str(q.get("question_id", "<unknown>")))

    agg = aggregate(per_question)

    print("\n=== Aggregate over {} questions ===".format(len(per_question)))
    header = f"{'method':<28} {'recall@k':>10} {'mrr':>8} {'mu':>8} {'H_hat':>8}"
    print(header)
    print("-" * len(header))
    order = ["cosine"] + [c.name for c in cfgs]
    for key in order:
        if key not in agg:
            continue
        v = agg[key]
        mu_s = f"{v.get('mu_at_query_mean', float('nan')):.3f}" if "mu_at_query_mean" in v else "  -  "
        h_s = f"{v.get('entropy_norm_mean', float('nan')):.3f}" if "entropy_norm_mean" in v else "  -  "
        print(f"{key:<28} {v['recall_at_k']:>10.4f} {v['mrr']:>8.4f} {mu_s:>8} {h_s:>8}")

    if skipped:
        print(f"\n[warn] skipped {len(skipped)} questions due to invalid dates: {skipped}")

    meta = {
        "date_format": DATE_FMT,
        "date_format_description": (
            "strptime pattern applied to haystack_dates and question_date; "
            "parsed via datetime.strptime(...).timestamp() -> Unix seconds."
        ),
        "time_unit": "unix_seconds",
        "beta_unit": "per_day",
        "beta_internal_unit": "per_second (= beta_per_day / 86400)",
        "beta_semantics": "half_life_seconds = ln(2) / beta_per_sec; half_life_days = ln(2) / beta_per_day",
        "k_default": 0.5,
        "k_semantics": "mu = mu_min + (mu_max - mu_min) * (1 - H_hat)^k",
        "granularity": "session-level only (no turn-level events or embeddings)",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "meta": meta,
        "num_questions": len(per_question),
        "num_skipped": len(skipped),
        "skipped_question_ids": skipped,
        "configs": [c.__dict__ for c in cfgs],
        "aggregate": agg,
        "per_question": per_question,
    }, indent=2))
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
