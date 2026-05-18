"""Run the LoCoMo session-vector R0 cosine vs R1-lite OriginIdea sweep.

Experiment scope:
  - memory unit: one full LoCoMo session
  - memory text: dialog by default
  - default mu_base=0.1; scan multiple via --mu-bases
  - intermediate_top_k configurable (--intermediate-top-k)
  - default: sweep T_half grid, plus R0 cosine (override via
    --t-half-days)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

LOCOMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LOCOMO_DIR.parents[1]
for path in (PROJECT_ROOT, LOCOMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402

from run_originidea_sessions import (  # noqa: E402
    answer_lag_days,
    embed_texts,
    evidence_session_ids,
    expand_sessions,
    gold_session_span_days,
    normalize_rows,
    run_cosine_sessions,
    run_originidea_sessions,
    session_metrics_at_k,
    session_recall_at_k,
)


K_VALUES = (1, 3, 5, 10)


def characterize_corpus(prepared: list[dict[str, Any]]) -> dict[str, float]:
    cos_samples: list[float] = []
    cos_low: list[float] = []
    cos_high: list[float] = []
    dt_samples: list[float] = []
    rng = np.random.default_rng(0)
    seen_samples: set[str] = set()
    for q in prepared:
        sample_id = str(q["sample_id"])
        if sample_id in seen_samples:
            continue
        seen_samples.add(sample_id)
        sv = q["session_vectors"]
        st = np.sort(q["session_times"])
        n = sv.shape[0]
        if n >= 2:
            m = min(500, n * n)
            i = rng.integers(0, n, size=m)
            j = rng.integers(0, n, size=m)
            cs = np.einsum("ij,ij->i", sv[i], sv[j])
            cos_samples.extend(cs.tolist())
            cos_low.append(float(np.quantile(cs, 0.25)))
            d = np.diff(st)
            d = d[d > 0]
            dt_samples.extend(d.tolist())

    for q in prepared:
        gold = q["gold_session_indices"]
        sv = q["session_vectors"]
        n = sv.shape[0]
        for g in gold:
            cos_to_gold = sv @ sv[g]
            order = np.argsort(-cos_to_gold)
            top_signal = cos_to_gold[order[1:min(11, n)]]
            if top_signal.size:
                cos_high.append(float(np.median(top_signal)))

    c_ref = float(np.median(np.abs(cos_samples))) if cos_samples else 0.3
    c_L = float(np.median(cos_low)) if cos_low else 0.0
    c_H = float(np.median(cos_high)) if cos_high else max(c_ref, 0.5)
    dt_ref = float(np.median(dt_samples)) if dt_samples else 1.0
    return {
        "c_ref": max(c_ref, 1e-3),
        "c_L": max(c_L, 0.0),
        "c_H": max(c_H, c_ref),
        "dt_ref": max(dt_ref, 1e-3),
    }


def design_recipes(
    stats: dict[str, float],
    *,
    t_half_days_list: tuple[float, ...] | None = None,
    intermediate_top_k: int = 3,
    mu_bases: tuple[float, ...] = (0.1,),
) -> list[dict[str, Any]]:
    """Build R0 + R1-lite grid. Defaults match the original LoCoMo half-life sweep."""
    t_halves = t_half_days_list if t_half_days_list is not None else (
        7.0, 14.0, 30.0, 60.0, 90.0, 120.0
    )
    mu_list = tuple(mu_bases) if mu_bases else (0.1,)
    multi_mu = len(mu_list) > 1
    recipes: list[dict[str, Any]] = [
        {
            "name": "R0_cosine",
            "group": "R0",
            "beta": 0.0,
            "mu_base": float(mu_list[0]),
            "intermediate_top_k": intermediate_top_k,
            "target": {},
        }
    ]
    for mu_base in mu_list:
        mu_base = float(mu_base)
        for t_half in t_halves:
            beta = math.log(2.0) / t_half
            suffix = f"_mu{mu_base:g}" if multi_mu else ""
            recipes.append(
                {
                    "name": f"R1_lite_T{t_half:g}d{suffix}",
                    "group": "R1-lite",
                    "beta": beta,
                    "mu_base": mu_base,
                    "intermediate_top_k": intermediate_top_k,
                    "target": {
                        "T_half_days": t_half,
                    },
                }
            )
    return recipes


def evidence_count_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def assign_quantile_buckets(values: list[float], labels: tuple[str, str, str]) -> list[str]:
    if not values:
        return []
    q1, q2 = np.quantile(np.asarray(values, dtype=float), [1 / 3, 2 / 3])
    out: list[str] = []
    for value in values:
        if value <= q1:
            out.append(labels[0])
        elif value <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return out


def prepare_corpus(samples: list[dict[str, Any]], embed_fn, *,
                   memory_text_mode: str, batch_size: int,
                   categories: set[str] | None,
                   n_questions: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id") or f"sample_{sample_idx}")
        qa_items = sample.get("qa") or []
        sample_sessions = None
        sample_vectors = None
        sample_times = None
        sample_session_ids = None
        sample_texts = None
        query_texts: list[str] = []
        selected_qas: list[tuple[int, dict[str, Any]]] = []
        for qa_idx, qa in enumerate(qa_items):
            category = str(qa.get("category") or "unknown")
            if categories is not None and category not in categories:
                continue
            if not evidence_session_ids(qa):
                continue
            selected_qas.append((qa_idx, qa))
            query_texts.append(str(qa.get("question") or ""))
        if not selected_qas:
            continue

        # Use the first selected QA only to mark evidence; text and timestamps are QA-independent.
        sample_sessions = expand_sessions(
            sample, selected_qas[0][1], memory_text_mode=memory_text_mode
        )
        if not sample_sessions:
            continue
        sample_texts = [s.text for s in sample_sessions]
        sample_vectors = normalize_rows(embed_texts(embed_fn, sample_texts, batch_size=batch_size))
        sample_times = np.asarray([s.time for s in sample_sessions], dtype=float)
        sample_session_ids = [s.session_id for s in sample_sessions]
        max_session_time = max(float(t) for t in sample_times)
        query_vectors = normalize_rows(embed_texts(embed_fn, query_texts, batch_size=batch_size))

        session_pos = {sid: i for i, sid in enumerate(sample_session_ids)}
        for q_local_idx, (qa_idx, qa) in enumerate(selected_qas):
            gold_ids = evidence_session_ids(qa)
            gold_indices = {session_pos[sid] for sid in gold_ids if sid in session_pos}
            if not gold_indices:
                continue
            sessions_for_qa = [
                type(s)(
                    session_id=s.session_id,
                    session_index=s.session_index,
                    text=s.text,
                    time=s.time,
                    is_evidence=s.session_id in gold_ids,
                )
                for s in sample_sessions
            ]
            question_time = max_session_time + (1.0 / 24.0 / 60.0)
            prepared.append(
                {
                    "question_id": f"{sample_id}::qa_{qa_idx}",
                    "sample_id": sample_id,
                    "category": str(qa.get("category") or "unknown"),
                    "question": str(qa.get("question") or ""),
                    "answer": qa.get("answer"),
                    "evidence_dialog_ids": [str(v) for v in qa.get("evidence") or []],
                    "evidence_ids": gold_ids,
                    "gold_session_indices": gold_indices,
                    "question_time": question_time,
                    "answer_lag_days": answer_lag_days(sessions_for_qa, question_time),
                    "session_span_days": gold_session_span_days(sessions_for_qa),
                    "evidence_session_count": len(gold_ids),
                    "evidence_count_bucket": evidence_count_bucket(len(gold_ids)),
                    "session_vectors": sample_vectors,
                    "query_vector": query_vectors[q_local_idx],
                    "session_times": sample_times,
                    "session_ids": sample_session_ids,
                    "n_sessions": len(sample_session_ids),
                }
            )
            if n_questions > 0 and len(prepared) >= n_questions:
                break
        print(
            f"[prep] sample {sample_idx}/{len(samples)} {sample_id}: "
            f"sessions={len(sample_session_ids)} selected_qas={len(selected_qas)} "
            f"prepared_total={len(prepared)}"
        )
        if n_questions > 0 and len(prepared) >= n_questions:
            break

    lag_buckets = assign_quantile_buckets(
        [float(q["answer_lag_days"]) for q in prepared],
        ("low", "mid", "high"),
    )
    span_buckets = assign_quantile_buckets(
        [float(q["session_span_days"]) for q in prepared],
        ("short", "mid", "long"),
    )
    for q, lag_bucket, span_bucket in zip(prepared, lag_buckets, span_buckets):
        q["answer_lag_bucket"] = lag_bucket
        q["session_span_bucket"] = span_bucket
    return prepared


def aggregate_per_question(per_q: list[dict[str, Any]]) -> dict[str, Any]:
    empty_metrics = {
        f"k{k}": {"recall": 0.0, "hit": 0.0, "mrr": 0.0, "srr": 0.0}
        for k in K_VALUES
    }
    if not per_q:
        return {"n": 0, "session_recall_at_k": 0.0, "session_metrics": empty_metrics}
    return {
        "n": len(per_q),
        "session_recall_at_k": float(np.mean([x["session_recall_at_k"] for x in per_q])),
        "session_metrics": {
            f"k{k}": {
                metric: float(np.mean([x["session_metrics"][f"k{k}"][metric] for x in per_q]))
                for metric in ("recall", "hit", "mrr", "srr")
            }
            for k in K_VALUES
        },
    }


def bucket_aggregates(per_q: list[dict[str, Any]], bucket_key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_q:
        buckets[str(item.get(bucket_key) or "unknown")].append(item)
    return {name: aggregate_per_question(items) for name, items in sorted(buckets.items())}


def win_tie_loss(per_q: list[dict[str, Any]], baseline: dict[str, float] | None,
                 bucket_key: str | None = None) -> dict[str, Any]:
    if baseline is None:
        return {"win": 0, "tie": 0, "loss": 0, "n": 0}
    if bucket_key is not None:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in per_q:
            buckets[str(item.get(bucket_key) or "unknown")].append(item)
        return {name: win_tie_loss(items, baseline) for name, items in sorted(buckets.items())}
    out: dict[str, Any] = {
        "win": 0,
        "tie": 0,
        "loss": 0,
        "n": 0,
        "win_question_ids": [],
        "loss_question_ids": [],
    }
    for item in per_q:
        qid = str(item["question_id"])
        if qid not in baseline:
            continue
        out["n"] += 1
        value = float(item["session_recall_at_k"])
        base = float(baseline[qid])
        if value > base + 1e-9:
            out["win"] += 1
            out["win_question_ids"].append(qid)
        elif value < base - 1e-9:
            out["loss"] += 1
            out["loss_question_ids"].append(qid)
        else:
            out["tie"] += 1
    return out


def evaluate_recipe(prepared: list[dict[str, Any]], recipe: dict[str, Any],
                    final_top_k: int) -> dict[str, Any]:
    per_q: list[dict[str, Any]] = []
    for q in prepared:
        if recipe["group"] == "R0":
            _, retrieved_sessions, full_order = run_cosine_sessions(
                q["session_vectors"],
                q["query_vector"],
                final_top_k=final_top_k,
                session_ids=q["session_ids"],
            )
        else:
            _, retrieved_sessions, full_order = run_originidea_sessions(
                q["session_vectors"],
                q["session_times"],
                q["query_vector"],
                query_time=q["question_time"],
                beta=recipe["beta"],
                mu_base=recipe["mu_base"],
                intermediate_top_k=recipe["intermediate_top_k"],
                final_top_k=final_top_k,
                session_ids=q["session_ids"],
            )
        metrics = {
            f"k{k}": session_metrics_at_k(full_order, q["gold_session_indices"], k)
            for k in K_VALUES
        }
        per_q.append(
            {
                "question_id": q["question_id"],
                "sample_id": q["sample_id"],
                "category": q["category"],
                "evidence_session_count": q["evidence_session_count"],
                "evidence_count_bucket": q["evidence_count_bucket"],
                "answer_lag_days": q["answer_lag_days"],
                "answer_lag_bucket": q["answer_lag_bucket"],
                "session_span_days": q["session_span_days"],
                "session_span_bucket": q["session_span_bucket"],
                "n_sessions": q["n_sessions"],
                "session_recall_at_k": session_recall_at_k(retrieved_sessions, q["evidence_ids"]),
                "session_metrics": metrics,
                "retrieved_session_ids": retrieved_sessions,
                "top10_session_ids": [q["session_ids"][i] for i in full_order[:10]],
            }
        )
    return {
        "aggregate": aggregate_per_question(per_q),
        "bucket_results": {
            key: bucket_aggregates(per_q, key)
            for key in (
                "category",
                "sample_id",
                "evidence_count_bucket",
                "answer_lag_bucket",
                "session_span_bucket",
            )
        },
        "per_question": per_q,
    }


def fmt_metric(aggregate: dict[str, Any], final_top_k: int) -> str:
    m1 = aggregate["session_metrics"]["k1"]
    m5 = aggregate["session_metrics"]["k5"]
    m10 = aggregate["session_metrics"]["k10"]
    return (
        f"recall@{final_top_k}={aggregate['session_recall_at_k']:.4f}, "
        f"hit@1={m1['hit']:.4f}, hit@5={m5['hit']:.4f}, "
        f"mrr@10={m10['mrr']:.4f}, srr@10={m10['srr']:.4f}"
    )


def write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    final_top_k = summary["config"]["final_top_k"]
    results = summary["recipes_results"]
    base = next(r for r in results if r["recipe"]["group"] == "R0")
    non_base = [r for r in results if r["recipe"]["group"] != "R0"]
    best = max(non_base, key=lambda r: r["aggregate"]["session_metrics"]["k10"]["mrr"])
    lines = [
        "# LoCoMo Session-Vector R0 vs R1-lite Sweep",
        "",
        f"- n_questions: {summary['config']['n_questions']}",
        f"- memory_text_mode: {summary['config']['memory_text_mode']}",
        f"- embedding: {summary['config']['embedding']}",
        f"- mu_bases: {summary['config']['mu_bases']}",
        f"- intermediate_top_k: {summary['config']['intermediate_top_k']}",
        f"- cosine baseline: {fmt_metric(base['aggregate'], final_top_k)}",
        f"- best by mrr@10: {best['recipe']['name']} ({fmt_metric(best['aggregate'], final_top_k)})",
        "",
        "## Recipe Summary",
        "",
        f"| recipe | T_half | beta | recall@{final_top_k} | hit@1 | hit@5 | mrr@10 | srr@10 | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        recipe = result["recipe"]
        target = recipe.get("target") or {}
        agg = result["aggregate"]
        m1 = agg["session_metrics"]["k1"]
        m5 = agg["session_metrics"]["k5"]
        m10 = agg["session_metrics"]["k10"]
        wtl = result["win_tie_loss_vs_baseline"]
        lines.append(
            "| {name} | {thalf} | {beta:.8f} | "
            "{rec:.4f} | {hit1:.4f} | {hit5:.4f} | {mrr:.4f} | {srr:.4f} | {w}/{t}/{l} |".format(
                name=recipe["name"],
                thalf=target.get("T_half_days", "-"),
                beta=recipe["beta"],
                rec=agg["session_recall_at_k"],
                hit1=m1["hit"],
                hit5=m5["hit"],
                mrr=m10["mrr"],
                srr=m10["srr"],
                w=wtl["win"],
                t=wtl["tie"],
                l=wtl["loss"],
            )
        )

    for bucket_key, title in (
        ("category", "Category"),
        ("answer_lag_bucket", "Answer Lag"),
        ("evidence_count_bucket", "Evidence Count"),
        ("sample_id", "Sample"),
    ):
        lines.extend(["", f"## Best Buckets: {title}", ""])
        lines.append("| bucket | cosine mrr@10 | best recipe | best mrr@10 | W/T/L |")
        lines.append("|---|---:|---|---:|---:|")
        base_buckets = base["bucket_results"][bucket_key]
        bucket_names = sorted(base_buckets)
        for bucket in bucket_names:
            best_bucket = None
            for result in non_base:
                b = result["bucket_results"][bucket_key].get(bucket)
                if b is None:
                    continue
                if best_bucket is None or (
                    b["session_metrics"]["k10"]["mrr"]
                    > best_bucket["bucket"]["session_metrics"]["k10"]["mrr"]
                ):
                    best_bucket = {"result": result, "bucket": b}
            if best_bucket is None:
                continue
            wtl = best_bucket["result"]["bucket_win_tie_loss_vs_baseline"][bucket_key].get(
                bucket, {"win": 0, "tie": 0, "loss": 0}
            )
            lines.append(
                f"| {bucket} | "
                f"{base_buckets[bucket]['session_metrics']['k10']['mrr']:.4f} | "
                f"{best_bucket['result']['recipe']['name']} | "
                f"{best_bucket['bucket']['session_metrics']['k10']['mrr']:.4f} | "
                f"{wtl['win']}/{wtl['tie']}/{wtl['loss']} |"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/locomo_session_vector_r0_vs_r1lite_v1"),
    )
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
    )
    parser.add_argument("--memory-text-mode", choices=["dialog", "observation", "session_summary"],
                        default="dialog")
    parser.add_argument("--question-categories", nargs="*", default=None)
    parser.add_argument("--n-questions", type=int, default=0,
                        help="<=0 means all mapped questions.")
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument(
        "--intermediate-top-k",
        type=int,
        default=3,
        metavar="K",
        help="Top-K histories treated as intermediate calls per session step "
        "(originidea). Default: 3.",
    )
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--t-half-days",
        nargs="*",
        type=float,
        default=None,
        metavar="DAYS",
        help="Half-life grid in days for R1-lite only. Omit for default "
        "(7,14,30,60,90,120). Example: --t-half-days 30",
    )
    parser.add_argument(
        "--mu-bases",
        nargs="*",
        type=float,
        default=None,
        metavar="MU",
        help="μ_base grid for R1-lite (and recorded on R0). Omit for default (0.1). "
        "Example: --mu-bases 0.02 0.05 0.1 0.2",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Missing {args.data}. Run benchmarks/locomo/download.py first.")

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    samples = json.loads(args.data.read_text())
    categories = {str(v) for v in args.question_categories} if args.question_categories else None
    print(f"[locomo-sweep] loading {args.data}")
    print(
        f"[locomo-sweep] embedding={args.embedding} memory_text_mode={args.memory_text_mode} "
        f"intermediate_top_k={args.intermediate_top_k} "
        f"categories={sorted(categories) if categories else 'all'}"
    )
    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    t0 = time.perf_counter()
    prepared = prepare_corpus(
        samples,
        embed_fn,
        memory_text_mode=args.memory_text_mode,
        batch_size=args.embed_batch_size,
        categories=categories,
        n_questions=args.n_questions,
    )
    stats = characterize_corpus(prepared)
    t_half_tuple = tuple(args.t_half_days) if args.t_half_days else None
    mu_bases_tuple = tuple(args.mu_bases) if args.mu_bases else (0.1,)
    recipes = design_recipes(
        stats,
        t_half_days_list=t_half_tuple,
        intermediate_top_k=args.intermediate_top_k,
        mu_bases=mu_bases_tuple,
    )
    print(
        f"[locomo-sweep] prepared={len(prepared)} recipes={len(recipes)} "
        f"dt_ref={stats['dt_ref']:.3f}d c_ref={stats['c_ref']:.3f}"
    )

    results: list[dict[str, Any]] = []
    baseline_per_q: dict[str, float] | None = None
    for recipe in recipes:
        rs = time.perf_counter()
        out = evaluate_recipe(prepared, recipe, args.final_top_k)
        per_q_recall = {str(x["question_id"]): float(x["session_recall_at_k"])
                        for x in out["per_question"]}
        if recipe["group"] == "R0":
            baseline_per_q = per_q_recall
        wtl = win_tie_loss(out["per_question"], baseline_per_q)
        bucket_wtl = {
            key: win_tie_loss(out["per_question"], baseline_per_q, key)
            for key in out["bucket_results"]
        }
        elapsed = time.perf_counter() - rs
        agg = out["aggregate"]
        m10 = agg["session_metrics"]["k10"]
        print(
            f"[locomo-sweep] {recipe['name']:22s} "
            f"recall@{args.final_top_k}={agg['session_recall_at_k']:.3f} "
            f"mrr@10={m10['mrr']:.3f} srr@10={m10['srr']:.3f} "
            f"W/T/L={wtl['win']}/{wtl['tie']}/{wtl['loss']} ({elapsed:.1f}s)"
        )
        results.append(
            {
                "recipe": recipe,
                "aggregate": agg,
                "bucket_results": out["bucket_results"],
                "per_question": out["per_question"],
                "win_tie_loss_vs_baseline": wtl,
                "bucket_win_tie_loss_vs_baseline": bucket_wtl,
            }
        )

    summary = {
        "config": {
            "embedding": args.embedding,
            "memory_text_mode": args.memory_text_mode,
            "n_questions": len(prepared),
            "requested_n_questions": args.n_questions,
            "question_categories": sorted(categories) if categories else "all",
            "final_top_k": args.final_top_k,
            "mu_bases": list(mu_bases_tuple),
            "intermediate_top_k": args.intermediate_top_k,
            "t_half_days": list(t_half_tuple) if t_half_tuple else "default_7_14_30_60_90_120",
            "global_stats": stats,
        },
        "recipes_results": results,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    suffix = "all" if args.n_questions <= 0 else f"n{args.n_questions}"
    out_json = args.outputs_dir / f"sweep_{suffix}_{args.memory_text_mode}_{args.embedding}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    write_markdown(summary, out_md)
    print(f"[locomo-sweep] wrote {out_json}")
    print(f"[locomo-sweep] wrote {out_md}")


if __name__ == "__main__":
    main()
