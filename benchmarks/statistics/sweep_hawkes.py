"""Statistics benchmark: Hawkes hyperparameter sweep and mechanism comparison.

This script evaluates the turn-level temporal memory dataset under several
retrieval mechanisms:

- cosine: semantic similarity only.
- recency: exponential time decay only.
- cosine_recency: semantic score gated by exponential recency.
- hawkes: originidea-style replay with excitation and self-decay.

It does not call an LLM; evaluation is purely retrieval/ranking based on the
scenario JSON labels (`positive_turns`, `negative_turns`, and `pairs`).

Example:

    python3 benchmarks/statistics/sweep_hawkes.py --embedding hashing
    python3 benchmarks/statistics/sweep_hawkes.py --embedding qwen
    python3 benchmarks/statistics/sweep_hawkes.py --embedding qwen --n-per-category 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

STATISTICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STATISTICS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
benchmarks_dir = str(PROJECT_ROOT / "benchmarks")
if benchmarks_dir not in sys.path:
    sys.path.insert(0, benchmarks_dir)

from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402
from longmemeval.run_originidea_sessions import compute_mu  # noqa: E402
from longmemeval.run_originidea_turns import embed_texts, normalize_rows  # noqa: E402


CATEGORY_DIRS = ("A", "B", "C", "D", "E")
DEFAULT_TOP_KS = (1, 3, 5, 10)


@dataclass
class PreparedScenario:
    path: str
    scenario_id: str
    category: str
    category_dir: str
    turn_texts: list[str]
    turn_times: np.ndarray
    turn_vectors: np.ndarray


def parse_iso_days(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp() / 86400.0
    except ValueError:
        return 0.0


def turn_to_text(turn: dict[str, Any]) -> str:
    messages = turn.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            speaker = str(message.get("speaker") or "").strip()
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            parts.append(f"{speaker}: {text}" if speaker else text)
        if parts:
            return " / ".join(parts)

    speaker = str(turn.get("speaker") or "").strip()
    text = str(turn.get("text") or "").strip()
    return f"{speaker}: {text}" if speaker and text else text


def scenario_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return sys.maxsize, path.name


def iter_scenario_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dirname in CATEGORY_DIRS:
        category_dir = data_dir / dirname
        if category_dir.exists():
            files.extend(sorted(category_dir.glob("*.json"), key=scenario_sort_key))
    return files


def select_scenarios(records: list[dict[str, Any]], n_per_category: int) -> list[dict[str, Any]]:
    if n_per_category <= 0:
        return records
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get("category") or "unknown")].append(record)
    selected: list[dict[str, Any]] = []
    for category in sorted(buckets):
        selected.extend(buckets[category][:n_per_category])
    return selected


def load_records(data_dir: Path, n_per_category: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_scenario_files(data_dir):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        record["_category_dir"] = path.parent.name
        records.append(record)
    return select_scenarios(records, n_per_category)


def prepare_scenarios(records: list[dict[str, Any]], embed_fn, batch_size: int) -> list[PreparedScenario]:
    prepared: list[PreparedScenario] = []
    for idx, record in enumerate(records, start=1):
        turns = record.get("turns") or []
        texts = [turn_to_text(t) for t in turns]
        times = np.asarray([parse_iso_days(t.get("t")) for t in turns], dtype=float)
        vectors = normalize_rows(embed_texts(embed_fn, texts, batch_size=batch_size))
        prepared.append(
            PreparedScenario(
                path=str(record.get("_path") or ""),
                scenario_id=str(record.get("scenario_id") or f"scenario_{idx}"),
                category=str(record.get("category") or "unknown"),
                category_dir=str(record.get("_category_dir") or ""),
                turn_texts=texts,
                turn_times=times,
                turn_vectors=vectors,
            )
        )
        print(
            f"[prep] {idx}/{len(records)} {prepared[-1].scenario_id} "
            f"category={prepared[-1].category} turns={len(texts)}"
        )
    return prepared


def design_recipes() -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = [
        {
            "name": "R0_cosine",
            "mechanism": "cosine",
            "beta": 0.0,
            "mu_base": 0.0,
            "inter_top_k": 0,
            "target": {},
        }
    ]
    t_halfs = [1.0, 3.0, 7.0, 14.0, 30.0, 60.0]
    mu_bases = [0.0, 0.2, 0.4, 0.6, 0.8]
    inter_top_ks = [1, 3, 5]

    for t_half in t_halfs:
        beta = math.log(2.0) / t_half
        tag = f"{t_half:g}d"
        recipes.append(
            {
                "name": f"R1_recency_T{tag}",
                "mechanism": "recency",
                "beta": beta,
                "mu_base": 0.0,
                "inter_top_k": 0,
                "target": {"T_half_days": t_half},
            }
        )
        for mu_base in mu_bases:
            recipes.append(
                {
                    "name": f"R2_cosrec_T{tag}_mu{mu_base:g}",
                    "mechanism": "cosine_recency",
                    "beta": beta,
                    "mu_base": mu_base,
                    "inter_top_k": 0,
                    "target": {"T_half_days": t_half, "mu_base": mu_base},
                }
            )
        for mu_base in mu_bases:
            for inter_top_k in inter_top_ks:
                recipes.append(
                    {
                        "name": f"R3_hawkes_T{tag}_mu{mu_base:g}_k{inter_top_k}",
                        "mechanism": "hawkes",
                        "beta": beta,
                        "mu_base": mu_base,
                        "inter_top_k": inter_top_k,
                        "target": {
                            "T_half_days": t_half,
                            "mu_base": mu_base,
                            "intermediate_top_k": inter_top_k,
                        },
                    }
                )
    return recipes


def score_cosine(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    return vectors @ query_vector


def score_recency(times: np.ndarray, query_time: float, beta: float) -> np.ndarray:
    ages = np.maximum(query_time - times, 0.0)
    return np.exp(-beta * ages)


def score_cosine_recency(
    vectors: np.ndarray,
    times: np.ndarray,
    query_vector: np.ndarray,
    query_time: float,
    beta: float,
    mu_base: float,
) -> np.ndarray:
    cos = score_cosine(vectors, query_vector)
    recency = score_recency(times, query_time, beta)
    return cos * (mu_base + (1.0 - mu_base) * recency)


def score_hawkes(
    vectors: np.ndarray,
    times: np.ndarray,
    query_vector: np.ndarray,
    query_time: float,
    *,
    beta: float,
    mu_base: float,
    intermediate_top_k: int,
) -> np.ndarray:
    n = int(vectors.shape[0])
    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)

    for i in range(n):
        t_i = float(times[i])
        if created.any():
            idx_global = np.flatnonzero(created)
            decayed = lambdas[idx_global] * np.exp(-beta * (t_i - last_update[idx_global]))
            decayed = np.clip(decayed, 0.0, 1.0)
            mu = compute_mu(decayed, mu_base)
            cos_vec = vectors[idx_global] @ vectors[i]
            scores = cos_vec * (mu + (1.0 - mu) * decayed)
            order = np.argsort(-scores)
            for local_j in order[: min(intermediate_top_k, len(order))]:
                score_i = float(scores[local_j])
                if score_i <= 0.0:
                    continue
                global_j = int(idx_global[local_j])
                lam_minus = float(decayed[local_j])
                lambdas[global_j] = min(1.0, lam_minus + (1.0 - lam_minus) * score_i)
                last_update[global_j] = t_i
        lambdas[i] = 1.0
        last_update[i] = t_i
        created[i] = True

    decayed_q = lambdas * np.exp(-beta * np.maximum(query_time - last_update, 0.0))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    mu_q = compute_mu(decayed_q, mu_base)
    cos_q = vectors @ query_vector
    return cos_q * (mu_q + (1.0 - mu_q) * decayed_q)


def rank_scores(scores: np.ndarray) -> tuple[list[int], dict[int, int]]:
    order = [int(i) for i in np.argsort(-scores)]
    ranks = {turn_idx: rank + 1 for rank, turn_idx in enumerate(order)}
    return order, ranks


def reciprocal_rank(order: list[int], positive: set[int], k: int) -> float:
    for rank, turn_idx in enumerate(order[:k], start=1):
        if turn_idx in positive:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_in_topk(order: list[int], labeled: set[int], k: int) -> float:
    """Mean of 1/rank over labeled turns that fall inside the top-k cutoff."""
    if not labeled:
        return 0.0
    acc = 0.0
    for rank, turn_idx in enumerate(order[:k], start=1):
        if turn_idx in labeled:
            acc += 1.0 / rank
    return acc / len(labeled)


def eval_ranking(
    scores: np.ndarray,
    ev: dict[str, Any],
    *,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    order, ranks = rank_scores(scores)
    positives = {int(v) for v in ev.get("positive_turns") or []}
    negatives = {int(v) for v in ev.get("negative_turns") or []}
    pairs = [p for p in ev.get("pairs") or [] if isinstance(p, dict)]
    max_rank = len(order) + 1

    best_pos_rank = min((ranks.get(p, max_rank) for p in positives), default=max_rank)
    best_neg_rank = min((ranks.get(n, max_rank) for n in negatives), default=max_rank)
    pos_score = max((float(scores[p]) for p in positives if 0 <= p < len(scores)), default=0.0)
    neg_score = max((float(scores[n]) for n in negatives if 0 <= n < len(scores)), default=0.0)

    pair_wins = 0
    pair_margins: list[float] = []
    for pair in pairs:
        pos = pair.get("positive")
        neg = pair.get("negative")
        if not isinstance(pos, int) or not isinstance(neg, int):
            continue
        pos_rank = ranks.get(pos, max_rank)
        neg_rank = ranks.get(neg, max_rank)
        if pos_rank < neg_rank:
            pair_wins += 1
        pair_margins.append(float(neg_rank - pos_rank))

    by_k = {}
    for k in top_ks:
        topk = set(order[:k])
        pos_hits = len(topk & positives)
        neg_hits = len(topk & negatives)
        pos_rec = pos_hits / len(positives) if positives else 0.0
        neg_intr = neg_hits / len(negatives) if negatives else 0.0
        pos_rr = mean_reciprocal_rank_in_topk(order, positives, k)
        neg_rr = mean_reciprocal_rank_in_topk(order, negatives, k)
        by_k[f"k{k}"] = {
            "srr": float(pos_rr - neg_rr),
            "positive_hit": 1.0 if pos_hits > 0 else 0.0,
            "positive_recall": pos_rec,
            "negative_intrusion": neg_intr,
            "mrr_positive": reciprocal_rank(order, positives, k),
        }

    return {
        "top_indices": order[:10],
        "best_positive_rank": best_pos_rank,
        "best_negative_rank": best_neg_rank,
        "rank_margin": float(best_neg_rank - best_pos_rank),
        "score_ratio": float(pos_score / max(abs(neg_score), 1e-12)) if negatives else 0.0,
        "pair_win_rate": pair_wins / len(pairs) if pairs else 0.0,
        "pair_margin": float(np.mean(pair_margins)) if pair_margins else 0.0,
        "by_k": by_k,
    }


def evaluate_recipe(
    records: list[dict[str, Any]],
    prepared_by_id: dict[str, PreparedScenario],
    recipe: dict[str, Any],
    *,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    per_eval: list[dict[str, Any]] = []
    for record in records:
        scenario = prepared_by_id[str(record.get("scenario_id"))]
        for eval_idx, ev in enumerate(record.get("evals") or []):
            query_turn = int(ev.get("query_turn"))
            candidate_count = query_turn
            if candidate_count <= 0 or query_turn >= len(scenario.turn_texts):
                continue
            vectors = scenario.turn_vectors[:candidate_count]
            times = scenario.turn_times[:candidate_count]
            query_vector = scenario.turn_vectors[query_turn]
            query_time = float(scenario.turn_times[query_turn])
            mechanism = recipe["mechanism"]
            if mechanism == "cosine":
                scores = score_cosine(vectors, query_vector)
            elif mechanism == "recency":
                scores = score_recency(times, query_time, recipe["beta"])
            elif mechanism == "cosine_recency":
                scores = score_cosine_recency(
                    vectors,
                    times,
                    query_vector,
                    query_time,
                    recipe["beta"],
                    recipe["mu_base"],
                )
            elif mechanism == "hawkes":
                scores = score_hawkes(
                    vectors,
                    times,
                    query_vector,
                    query_time,
                    beta=recipe["beta"],
                    mu_base=recipe["mu_base"],
                    intermediate_top_k=recipe["inter_top_k"],
                )
            else:
                raise ValueError(f"unknown mechanism: {mechanism}")

            metrics = eval_ranking(scores, ev, top_ks=top_ks)
            per_eval.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "category": scenario.category,
                    "category_dir": scenario.category_dir,
                    "eval_index": eval_idx,
                    "query_turn": query_turn,
                    **metrics,
                }
            )
    return {
        "aggregate": aggregate_metrics(per_eval, top_ks),
        "category_results": bucket_metrics(per_eval, "category", top_ks),
        "per_eval": per_eval,
    }


def aggregate_metrics(per_eval: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    if not per_eval:
        return {
            "n": 0,
            "rank_margin": 0.0,
            "score_ratio": 0.0,
            "pair_win_rate": 0.0,
            "pair_margin": 0.0,
            "by_k": {
                f"k{k}": {
                    "srr": 0.0,
                    "positive_hit": 0.0,
                    "positive_recall": 0.0,
                    "negative_intrusion": 0.0,
                    "mrr_positive": 0.0,
                }
                for k in top_ks
            },
        }
    return {
        "n": len(per_eval),
        "rank_margin": float(np.mean([x["rank_margin"] for x in per_eval])),
        "score_ratio": float(np.mean([x["score_ratio"] for x in per_eval])),
        "pair_win_rate": float(np.mean([x["pair_win_rate"] for x in per_eval])),
        "pair_margin": float(np.mean([x["pair_margin"] for x in per_eval])),
        "by_k": {
            f"k{k}": {
                metric: float(np.mean([x["by_k"][f"k{k}"][metric] for x in per_eval]))
                for metric in (
                    "srr",
                    "positive_hit",
                    "positive_recall",
                    "negative_intrusion",
                    "mrr_positive",
                )
            }
            for k in top_ks
        },
    }


def bucket_metrics(
    per_eval: list[dict[str, Any]],
    bucket_key: str,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_eval:
        buckets[str(item.get(bucket_key) or "unknown")].append(item)
    return {name: aggregate_metrics(items, top_ks) for name, items in sorted(buckets.items())}


def recipe_sort_key(result: dict[str, Any], primary_k: int) -> tuple[float, float, float, float]:
    agg = result["aggregate"]
    k_metrics = agg["by_k"][f"k{primary_k}"]
    return (
        k_metrics["srr"],
        agg["pair_win_rate"],
        k_metrics["positive_recall"],
        agg["rank_margin"],
    )


def best_by_mechanism(results: list[dict[str, Any]], primary_k: int) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        buckets[result["recipe"]["mechanism"]].append(result)
    return {
        mechanism: max(items, key=lambda r: recipe_sort_key(r, primary_k))
        for mechanism, items in buckets.items()
    }


def category_sort_key(agg: dict[str, Any], primary_k: int) -> tuple[float, float, float, float]:
    k_metrics = agg["by_k"][f"k{primary_k}"]
    return (
        k_metrics["srr"],
        k_metrics["positive_recall"],
        -k_metrics["negative_intrusion"],
        agg["rank_margin"],
    )


def best_by_category(results: list[dict[str, Any]], primary_k: int) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for result in results:
        for category, agg in result["category_results"].items():
            candidates[category].append((result, agg))

    best: dict[str, dict[str, Any]] = {}
    for category, items in sorted(candidates.items()):
        result, agg = max(items, key=lambda item: category_sort_key(item[1], primary_k))
        best[category] = {
            "recipe": result["recipe"],
            "aggregate": agg,
        }
    return best


def best_by_category_and_mechanism(
    results: list[dict[str, Any]], primary_k: int
) -> dict[str, dict[str, dict[str, Any]]]:
    candidates: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        mechanism = str(result["recipe"]["mechanism"])
        for category, agg in result["category_results"].items():
            candidates[category][mechanism].append((result, agg))

    best: dict[str, dict[str, dict[str, Any]]] = {}
    for category, mechanism_items in sorted(candidates.items()):
        best[category] = {}
        for mechanism, items in sorted(mechanism_items.items()):
            result, agg = max(items, key=lambda item: category_sort_key(item[1], primary_k))
            best[category][mechanism] = {
                "recipe": result["recipe"],
                "aggregate": agg,
            }
    return best


def win_tie_loss(
    per_eval: list[dict[str, Any]],
    baseline_per_eval: dict[tuple[str, int], float],
    primary_k: int,
) -> dict[str, Any]:
    detail = {"win": 0, "tie": 0, "loss": 0, "n": 0, "win_ids": [], "loss_ids": []}
    for item in per_eval:
        key = (str(item["scenario_id"]), int(item["eval_index"]))
        base = baseline_per_eval.get(key)
        if base is None:
            continue
        score = float(item["by_k"][f"k{primary_k}"]["srr"])
        detail["n"] += 1
        label = f"{item['scenario_id']}#{item['eval_index']}"
        if score > base + 1e-9:
            detail["win"] += 1
            detail["win_ids"].append(label)
        elif score < base - 1e-9:
            detail["loss"] += 1
            detail["loss_ids"].append(label)
        else:
            detail["tie"] += 1
    return detail


def md_ids(values: list[str], limit: int = 12) -> str:
    if not values:
        return "-"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f"<br>...(+{len(values) - limit})"
    return "<br>".join(shown) + suffix


def write_markdown_report(summary: dict[str, Any], out_md: Path) -> None:
    primary_k = int(summary["config"]["primary_k"])
    top = summary["best_by_mechanism"]
    top_by_category = summary["best_by_category"]
    top_by_category_mechanism = summary["best_by_category_and_mechanism"]
    lines: list[str] = [
        "# Statistics Hawkes Sweep Summary",
        "",
        (
            "说明：SRR@K 基于倒数排名（reciprocal rank）："
            "对每个落在 top-K 的 positive（negative）累加 1/rank，再分别除以该 eval 的 "
            "|positive_turns|（|negative_turns|），两者相减得到 Side-effect aware reciprocal-rank 分数；"
            "W/T/L 按每个 eval 的 SRR@K 与 cosine baseline 逐项比较。"
        ),
        (
            f"数据选择：每类问题取前 {summary['config']['n_per_category']} 个 scenario；"
            f"共 {summary['config']['n_scenarios']} 个 scenario。"
            if summary["config"]["n_per_category"] > 0
            else f"数据选择：使用全部 {summary['config']['n_scenarios']} 个 scenario。"
        ),
        "",
        "## Best Mechanism Comparison",
        "",
        "| mechanism | best recipe | SRR@K | positive_recall@K | negative_intrusion@K | pair_win | rank_margin | vs cosine W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mechanism in ("cosine", "recency", "cosine_recency", "hawkes"):
        if mechanism not in top:
            continue
        result = top[mechanism]
        agg = result["aggregate"]
        km = agg["by_k"][f"k{primary_k}"]
        wtl = result["win_tie_loss_vs_cosine"]
        lines.append(
            "| {mechanism} | {name} | {srr:.4f} | {recall:.4f} | {intr:.4f} | "
            "{pair:.4f} | {margin:.3f} | {w}/{t}/{l} |".format(
                mechanism=mechanism,
                name=result["recipe"]["name"],
                srr=km["srr"],
                recall=km["positive_recall"],
                intr=km["negative_intrusion"],
                pair=agg["pair_win_rate"],
                margin=agg["rank_margin"],
                w=wtl["win"],
                t=wtl["tie"],
                l=wtl["loss"],
            )
        )

    lines.extend([
        "",
        "## Question Category Summary",
        "",
        (
            f"按每个问题类别单独选 SRR@{primary_k} 最高的 recipe；"
            f"tie 按 positive_recall@{primary_k} 更高、negative_intrusion@{primary_k} 更低排序。"
        ),
        "",
        f"| category | best recipe | mechanism | SRR@{primary_k} | positive_recall@{primary_k} | negative_intrusion@{primary_k} | pair_win | rank_margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for category, result in top_by_category.items():
        recipe = result["recipe"]
        agg = result["aggregate"]
        km = agg["by_k"][f"k{primary_k}"]
        lines.append(
            "| {category} | {name} | {mechanism} | {srr:.4f} | {recall:.4f} | "
            "{intr:.4f} | {pair:.4f} | {margin:.3f} |".format(
                category=category,
                name=recipe["name"],
                mechanism=recipe["mechanism"],
                srr=km["srr"],
                recall=km["positive_recall"],
                intr=km["negative_intrusion"],
                pair=agg["pair_win_rate"],
                margin=agg["rank_margin"],
            )
        )

    lines.extend([
        "",
        "## Per-Category Best By Mechanism",
        "",
        f"每个问题类别内，各机制各自取该类 SRR@{primary_k} 最高的配置，便于看机制偏科。",
        "",
    ])
    for category, mechanism_results in top_by_category_mechanism.items():
        lines.extend([
            f"### {category}",
            "",
            f"| mechanism | best recipe | SRR@{primary_k} | positive_recall@{primary_k} | negative_intrusion@{primary_k} | pair_win | rank_margin |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for mechanism in ("cosine", "recency", "cosine_recency", "hawkes"):
            if mechanism not in mechanism_results:
                continue
            result = mechanism_results[mechanism]
            agg = result["aggregate"]
            km = agg["by_k"][f"k{primary_k}"]
            lines.append(
                "| {mechanism} | {name} | {srr:.4f} | {recall:.4f} | {intr:.4f} | "
                "{pair:.4f} | {margin:.3f} |".format(
                    mechanism=mechanism,
                    name=result["recipe"]["name"],
                    srr=km["srr"],
                    recall=km["positive_recall"],
                    intr=km["negative_intrusion"],
                    pair=agg["pair_win_rate"],
                    margin=agg["rank_margin"],
                )
            )
        lines.append("")

    lines.extend(["", "## Category Breakdown", ""])
    for mechanism in ("cosine", "recency", "cosine_recency", "hawkes"):
        if mechanism not in top:
            continue
        result = top[mechanism]
        lines.extend([
            f"### {mechanism}: {result['recipe']['name']}",
            "",
            "| category | n | SRR@K | positive_recall@K | negative_intrusion@K | pair_win | rank_margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for category, agg in result["category_results"].items():
            km = agg["by_k"][f"k{primary_k}"]
            lines.append(
                "| {cat} | {n} | {srr:.4f} | {recall:.4f} | {intr:.4f} | {pair:.4f} | {margin:.3f} |".format(
                    cat=category,
                    n=agg["n"],
                    srr=km["srr"],
                    recall=km["positive_recall"],
                    intr=km["negative_intrusion"],
                    pair=agg["pair_win_rate"],
                    margin=agg["rank_margin"],
                )
            )
        lines.append("")

    lines.extend([
        "## Hawkes Grid",
        "",
        "| recipe | T_half | mu_base | inter_top_k | SRR@K | positive_recall@K | negative_intrusion@K | pair_win | rank_margin | W/T/L | win ids | loss ids |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    hawkes_results = [
        r for r in summary["results"] if r["recipe"]["mechanism"] == "hawkes"
    ]
    hawkes_results.sort(key=lambda r: recipe_sort_key(r, primary_k), reverse=True)
    for result in hawkes_results:
        recipe = result["recipe"]
        agg = result["aggregate"]
        km = agg["by_k"][f"k{primary_k}"]
        wtl = result["win_tie_loss_vs_cosine"]
        target = recipe.get("target", {})
        lines.append(
            "| {name} | {thalf:.1f} | {mu:.3f} | {ik} | {srr:.4f} | {recall:.4f} | "
            "{intr:.4f} | {pair:.4f} | {margin:.3f} | {w}/{t}/{l} | {wins} | {losses} |".format(
                name=recipe["name"],
                thalf=float(target.get("T_half_days", 0.0)),
                mu=float(recipe["mu_base"]),
                ik=int(recipe["inter_top_k"]),
                srr=km["srr"],
                recall=km["positive_recall"],
                intr=km["negative_intrusion"],
                pair=agg["pair_win_rate"],
                margin=agg["rank_margin"],
                w=wtl["win"],
                t=wtl["tie"],
                l=wtl["loss"],
                wins=md_ids(wtl["win_ids"]),
                losses=md_ids(wtl["loss_ids"]),
            )
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_top_ks(values: list[int]) -> tuple[int, ...]:
    cleaned = sorted({int(v) for v in values if int(v) > 0})
    return tuple(cleaned or DEFAULT_TOP_KS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hawkes hyperparameter sweep and mechanism comparison for benchmarks/statistics."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/statistics"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/statistics_hawkes_sweep"))
    parser.add_argument(
        "--embedding",
        choices=["qwen", "bge", "hashing"],
        default="qwen",
        help="Dense models use sentence-transformers (Qwen3-0.6B, BGE). "
        "See hawkes_rag.embeddings.make_embedding_fn for HF ids.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
        help="sentence-transformers cache; relative paths resolve from repo root.",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=5,
        help="Limit scenarios per question category; <=0 means all. Default: 5.",
    )
    parser.add_argument("--top-ks", nargs="+", type=int, default=list(DEFAULT_TOP_KS))
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-recency",
        action="store_true",
        help="Skip the pure recency baseline recipes while keeping cosine, cosine_recency, and Hawkes.",
    )
    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)
    if args.primary_k not in top_ks:
        top_ks = tuple(sorted(set(top_ks + (args.primary_k,))))

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[statistics-sweep] loading scenarios from {args.data_dir}")
    records = load_records(args.data_dir, args.n_per_category)
    if not records:
        raise SystemExit(f"No scenario JSON files found under {args.data_dir}")
    print(f"[statistics-sweep] selected {len(records)} scenario(s)")

    try:
        embed_fn = make_embedding_fn(
            args.embedding,
            device=args.device,
            cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
        )
    except ModuleNotFoundError as exc:
        if args.embedding in {"qwen", "bge"} and exc.name == "sentence_transformers":
            raise SystemExit(
                "Missing optional dependency `sentence_transformers`. "
                "Install it to run Qwen/BGE experiments, or use "
                "`--embedding hashing` for a dependency-light smoke test."
            ) from exc
        raise

    t0 = time.perf_counter()
    prepared = prepare_scenarios(records, embed_fn, args.embed_batch_size)
    prepared_by_id = {item.scenario_id: item for item in prepared}
    recipes = design_recipes()
    if args.skip_recency:
        recipes = [recipe for recipe in recipes if recipe["mechanism"] != "recency"]
    print(
        f"[statistics-sweep] evaluating {len(recipes)} recipes "
        f"(cosine + recency/cosine_recency baselines + Hawkes grid)"
    )

    results: list[dict[str, Any]] = []
    baseline_per_eval: dict[tuple[str, int], float] = {}
    for idx, recipe in enumerate(recipes, start=1):
        start = time.perf_counter()
        out = evaluate_recipe(records, prepared_by_id, recipe, top_ks=top_ks)
        if recipe["mechanism"] == "cosine":
            baseline_per_eval = {
                (str(item["scenario_id"]), int(item["eval_index"])): float(
                    item["by_k"][f"k{args.primary_k}"]["srr"]
                )
                for item in out["per_eval"]
            }
        wtl = win_tie_loss(out["per_eval"], baseline_per_eval, args.primary_k)
        result = {
            "recipe": recipe,
            "aggregate": out["aggregate"],
            "category_results": out["category_results"],
            "per_eval": out["per_eval"],
            "win_tie_loss_vs_cosine": wtl,
        }
        results.append(result)
        agg = result["aggregate"]
        km = agg["by_k"][f"k{args.primary_k}"]
        print(
            f"[statistics-sweep] {idx:03d}/{len(recipes)} {recipe['name']:30s} "
            f"SRR@{args.primary_k}={km['srr']:.3f} "
            f"pos_recall@{args.primary_k}={km['positive_recall']:.3f} "
            f"neg_intr@{args.primary_k}={km['negative_intrusion']:.3f} "
            f"pair_win={agg['pair_win_rate']:.3f} "
            f"W/T/L={wtl['win']}/{wtl['tie']}/{wtl['loss']} "
            f"({time.perf_counter() - start:.1f}s)"
        )

    best = best_by_mechanism(results, args.primary_k)
    summary = {
        "config": {
            "data_dir": str(args.data_dir),
            "embedding": args.embedding,
            "n_scenarios": len(records),
            "n_per_category": args.n_per_category,
            "selected_scenarios": [str(record.get("scenario_id")) for record in records],
            "top_ks": list(top_ks),
            "primary_k": args.primary_k,
            "n_recipes": len(recipes),
            "skip_recency": bool(args.skip_recency),
        },
        "best_by_mechanism": best,
        "best_by_category": best_by_category(results, args.primary_k),
        "best_by_category_and_mechanism": best_by_category_and_mechanism(results, args.primary_k),
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    subset_tag = "all" if args.n_per_category <= 0 else f"n{args.n_per_category}_per_category"
    out_json = args.outputs_dir / f"sweep_{subset_tag}_{args.embedding}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    write_markdown_report(summary, out_md)
    print(f"\n[statistics-sweep] wrote {out_json}")
    print(f"[statistics-sweep] wrote {out_md}")

    print("\n=== Best mechanism summary ===")
    header = (
        f"{'mechanism':16s}  {'best_recipe':32s}  "
        f"{'SRR@K':>9s}  {'pos_rec@K':>9s}  {'neg_intr@K':>10s}  {'pair_win':>8s}"
    )
    print(header)
    print("-" * len(header))
    for mechanism in ("cosine", "recency", "cosine_recency", "hawkes"):
        if mechanism not in best:
            continue
        result = best[mechanism]
        agg = result["aggregate"]
        km = agg["by_k"][f"k{args.primary_k}"]
        print(
            f"{mechanism:16s}  {result['recipe']['name'][:32]:32s}  "
            f"{km['srr']:9.3f}  {km['positive_recall']:9.3f}  "
            f"{km['negative_intrusion']:10.3f}  {agg['pair_win_rate']:8.3f}"
        )


if __name__ == "__main__":
    main()
