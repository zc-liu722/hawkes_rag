"""按 originidea.md 在 LongMemEval-S 全量三类问题上
做三维网格超参扫描（T_half × mu_base × intermediate_top_k），
并与纯 cosine 基线比较 session-level recall@k 与 MRR@k。

使用方式：

    python3 sweep_originidea_sessions.py

## 评测分组

- **R0**：纯 cosine 基线（β=0，所有 λ≡1，方括号恒为 1，score=cos）。
- **S 组三维网格扫描**：
  - T_{1/2} ∈ {1d, 5d, 15d, 30d, 50d}（5 个）
  - μ_base ∈ {0.0, 0.2, 0.4, 0.6, 0.8}（5 个）
  - intermediate_top_k ∈ {1, 3, 5}（3 个）
  共 5×5×3 = 75 个实验配置 + 1 个 cosine 基线 = 76 个 recipe。

不调用 LLM；评测只看检索阶段。
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

LONGMEMEVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LONGMEMEVAL_DIR.parent
for path in (PROJECT_ROOT, str(PROJECT_ROOT / "benchmarks"), LONGMEMEVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hawkes_rag.embeddings import make_embedding_fn  # noqa: E402

from run_originidea_turns import (  # noqa: E402
    embed_texts,
    evidence_session_ids,
    is_multi_session,
    normalize_rows,
    parse_date,
    session_recall_at_k,
)
from run_originidea_sessions import (  # noqa: E402
    expand_sessions,
    run_cosine_sessions,
    run_originidea_sessions,
    session_metrics_at_k,
)


# ---------- 数据画像（session 粒度，单位：天） ----------

def characterize_corpus(prepared: list[dict[str, Any]]) -> dict[str, float]:
    """对 selected questions 的 session 池做全局统计。

    产出 ideadiscuss.md §4 Q13 可分性不等式所需的 (c_L, c_H, Δt_L, Δt_H)
    以及 Q3 反推 θ 所需的 (c_ref, Δt_ref)：

    - **c_ref**：所有 session 间余弦相似度的中位数（取绝对值），代表"典型相关度"。
    - **c_L**：全体余弦的 25% 分位，代表"噪声档相似度"。
    - **c_H**：以 gold session 为锚，top-20 邻居相似度的中位数，代表"高价值档"。
    - **Δt_ref**：相邻 session 时间间隔的中位数（天），Q3 反推 θ 的时间基准。
    - **Δt_L**：间隔的 50% 分位（典型"噪声档间隔"）。
    - **Δt_H**：间隔的 90% 分位（典型"高价值档间隔"，偏稀疏）。
    """
    cos_samples: list[float] = []
    cos_L_samples: list[float] = []
    cos_H_samples: list[float] = []
    dt_samples: list[float] = []
    dt_L_samples: list[float] = []
    dt_H_samples: list[float] = []
    rng = np.random.default_rng(0)
    for q in prepared:
        sv = q["session_vectors"]
        st = q["session_times"]
        gold = q["gold_session_indices"]
        n = sv.shape[0]
        if n >= 2:
            m = min(500, n * n)
            i = rng.integers(0, n, size=m)
            j = rng.integers(0, n, size=m)
            cs = np.einsum("ij,ij->i", sv[i], sv[j])
            cos_samples.extend(cs.tolist())
            cos_L_samples.append(float(np.quantile(cs, 0.25)))
            # 以 gold session 为锚：top-20 邻居（排除自身）
            if gold:
                g = next(iter(gold))
                cos_to_gold = sv @ sv[g]
                order = np.argsort(-cos_to_gold)
                top_signal = cos_to_gold[order[1: min(21, n)]]
                if top_signal.size:
                    cos_H_samples.append(float(np.median(top_signal)))
            d = np.diff(np.sort(st))
            d = d[d > 0]
            if d.size:
                dt_samples.extend(d.tolist())
                dt_L_samples.append(float(np.quantile(d, 0.5)))
                dt_H_samples.append(float(np.quantile(d, 0.9)))
    c_ref = float(np.median(np.abs(cos_samples))) if cos_samples else 0.3
    c_L = float(np.median(cos_L_samples)) if cos_L_samples else 0.0
    c_H = float(np.median(cos_H_samples)) if cos_H_samples else 0.5
    dt_ref = float(np.median(dt_samples)) if dt_samples else 1.0
    dt_L = float(np.median(dt_L_samples)) if dt_L_samples else dt_ref
    dt_H = float(np.median(dt_H_samples)) if dt_H_samples else dt_ref
    return {
        "c_ref": max(c_ref, 1e-3),
        "c_L": max(c_L, 0.0),
        "c_H": max(c_H, c_ref),
        "dt_ref": max(dt_ref, 1e-3),
        "dt_L": max(dt_L, 1e-3),
        "dt_H": max(dt_H, dt_L),
    }


# ---------- Recipe 设计 ----------

def design_recipes(stats: dict[str, float]) -> list[dict[str, Any]]:
    """三维网格扫描：T_half × mu_base × intermediate_top_k。"""
    recipes: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, float, int]] = set()

    def add(name: str, T_half: float, mu_base: float,
            inter_top_k: int, rationale: str = "",
            cosine_baseline: bool = False) -> None:
        if cosine_baseline:
            beta = 0.0
            tgt: dict[str, Any] = {}
        else:
            beta = math.log(2.0) / T_half
            tgt = {
                "T_half_days": T_half,
                "mu_base": mu_base,
            }
        key = (round(beta, 8), round(mu_base, 6), int(inter_top_k))
        if key in seen_keys:
            return
        seen_keys.add(key)
        recipes.append({
            "name": name,
            "rationale": rationale,
            "beta": beta,
            "mu_base": mu_base,
            "inter_top_k": inter_top_k,
            "target": tgt,
        })

    add("R0_cosine_baseline",
        T_half=1.0, mu_base=0.1, inter_top_k=3,
        rationale="纯 cosine 退化：β=0 ⇒ λ≡1 ⇒ [μ+(1-μ)·1]=1 ⇒ score=cos",
        cosine_baseline=True)

    T_halfs = [1.0, 5.0, 15.0, 30.0, 50.0]
    mu_bases = [0.0, 0.2, 0.4, 0.6, 0.8]
    inter_top_ks = [1, 3, 5]

    for T_half in T_halfs:
        t_tag = f"{T_half:g}d"
        beta_val = math.log(2.0) / T_half
        for mu_base in mu_bases:
            mu_tag = f"{mu_base:g}"
            for ik in inter_top_ks:
                name = f"S_T{t_tag}_mu{mu_tag}_k{ik}"
                add(name,
                    T_half=T_half, mu_base=mu_base,
                    inter_top_k=ik,
                    rationale=f"T_{{1/2}}={t_tag}（β={beta_val:.3f}）, μ_base={mu_base}, k={ik}")

    return recipes


QUESTION_TYPE_ALIASES = {
    "multisession": "multi-session",
    "multi-session": "multi-session",
    "temporalreasoning": "temporal-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "knowledgeupdate": "knowledge-update",
    "knowledge-update": "knowledge-update",
}


def normalize_question_type_name(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    compact = key.replace("-", "")
    return QUESTION_TYPE_ALIASES.get(key) or QUESTION_TYPE_ALIASES.get(compact) or key


# ---------- 评测 ----------

def prepare_corpus(records, embed_fn, batch_size: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for q_idx, record in enumerate(records, start=1):
        question = str(record.get("question", ""))
        evidence_ids = evidence_session_ids(record)
        question_time = parse_date(record.get("question_date"))
        sessions = expand_sessions(record)
        if not sessions:
            continue
        if question_time <= 0:
            question_time = max(s.time for s in sessions) + 1.0
        texts = [s.text for s in sessions]
        sv = normalize_rows(embed_texts(embed_fn, texts, batch_size=batch_size))
        qv = embed_texts(embed_fn, [question], batch_size=batch_size)[0]
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        st = np.asarray([s.time for s in sessions], dtype=float)
        gold = {i for i, s in enumerate(sessions) if s.is_evidence}
        prepared.append({
            "record": record,
            "question_id": record.get("question_id"),
            "question_type": str(record.get("question_type") or "unknown"),
            "evidence_ids": evidence_ids,
            "question_time": question_time,
            "session_vectors": sv,
            "query_vector": qv,
            "session_times": st,
            "session_ids": [s.session_id for s in sessions],
            "gold_session_indices": gold,
            "n_sessions": len(sessions),
            "n_evidence_sessions": len(evidence_ids),
        })
        print(f"[prep] Q{q_idx}/{len(records)} sess={len(sessions)} "
              f"evidence={sorted(evidence_ids)}")
    return prepared


def aggregate_per_question(per_q: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_q:
        return {
            "n": 0,
            "session_recall_at_k": 0.0,
            "session_metrics": {
                f"k{k}": {"recall": 0.0, "hit": 0.0, "mrr": 0.0, "srr": 0.0}
                for k in (1, 3, 5, 10)
            },
        }
    return {
        "n": len(per_q),
        "session_recall_at_k": float(np.mean([x["session_recall_at_k"] for x in per_q])),
        "session_metrics": {
            f"k{k}": {m: float(np.mean(
                [x["session_metrics"][f"k{k}"][m] for x in per_q]))
                for m in ("recall", "hit", "mrr", "srr")}
            for k in (1, 3, 5, 10)
        },
    }


def bucket_aggregates(per_q: list[dict[str, Any]], bucket_key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_q:
        buckets[str(item.get(bucket_key) or "unknown")].append(item)
    return {name: aggregate_per_question(items)
            for name, items in sorted(buckets.items())}


def win_tie_loss_by_bucket(per_q: list[dict[str, Any]],
                           baseline_per_q: dict[str, float] | None,
                           bucket_key: str) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "tie": 0, "loss": 0, "n": 0}
    )
    if baseline_per_q is None:
        return {}
    for item in per_q:
        qid = str(item["question_id"])
        bv = baseline_per_q.get(qid)
        if bv is None:
            continue
        bucket = str(item.get(bucket_key) or "unknown")
        buckets[bucket]["n"] += 1
        v = float(item["session_recall_at_k"])
        if v > bv + 1e-9:
            buckets[bucket]["win"] += 1
        elif v < bv - 1e-9:
            buckets[bucket]["loss"] += 1
        else:
            buckets[bucket]["tie"] += 1
    return dict(sorted(buckets.items()))


def win_tie_loss_detail(per_q: list[dict[str, Any]],
                        baseline_per_q: dict[str, float] | None) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "win": 0,
        "tie": 0,
        "loss": 0,
        "n": 0,
        "win_question_ids": [],
        "tie_question_ids": [],
        "loss_question_ids": [],
    }
    if baseline_per_q is None:
        return detail
    for item in per_q:
        qid = str(item["question_id"])
        bv = baseline_per_q.get(qid)
        if bv is None:
            continue
        detail["n"] += 1
        v = float(item["session_recall_at_k"])
        if v > bv + 1e-9:
            detail["win"] += 1
            detail["win_question_ids"].append(qid)
        elif v < bv - 1e-9:
            detail["loss"] += 1
            detail["loss_question_ids"].append(qid)
        else:
            detail["tie"] += 1
            detail["tie_question_ids"].append(qid)
    return detail


def md_cell_ids(values: list[str]) -> str:
    if not values:
        return "-"
    return "<br>".join(str(v) for v in values)


def write_markdown_report(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = [
        "# OriginIdea Session Sweep Summary",
        "",
        "说明：每个表对应一个问题类别；每行是一个非 cosine 实验配置，输赢均按该类别内每题的 session_recall@final_top_k 与 cosine baseline 逐题比较。",
        "",
    ]
    for category_result in summary["category_results"]:
        category = category_result["question_type"]
        config = category_result["config"]
        baseline = next(
            r for r in category_result["recipes_results"]
            if r["recipe"]["name"] == "R0_cosine_baseline"
        )
        base_agg = baseline["aggregate"]
        base_m1 = base_agg["session_metrics"]["k1"]
        base_m5 = base_agg["session_metrics"]["k5"]
        base_m10 = base_agg["session_metrics"]["k10"]
        lines.extend([
            f"## {category}",
            "",
            (
                f"cosine baseline：n={config['n_questions']}，"
                f"recall@{config['final_top_k']}={base_agg['session_recall_at_k']:.4f}，"
                f"hit@1={base_m1['hit']:.4f}，hit@5={base_m5['hit']:.4f}，"
                f"mrr@10={base_m10['mrr']:.4f}"
            ),
            "",
            "| 实验代号 | beta | mu_base | intermediate_top_k | 半衰期(days) | 实验结果 | vs cos W/T/L | 赢的题号 | 输的题号 |",
            "|---|---:|---:|---:|---:|---|---:|---|---|",
        ])
        for result in category_result["recipes_results"]:
            recipe = result["recipe"]
            if recipe["name"] == "R0_cosine_baseline":
                continue
            target = recipe.get("target", {})
            agg = result["aggregate"]
            m1 = agg["session_metrics"]["k1"]
            m5 = agg["session_metrics"]["k5"]
            m10 = agg["session_metrics"]["k10"]
            wtl = result["win_tie_loss_vs_baseline"]
            exp_result = (
                f"recall@{config['final_top_k']}={agg['session_recall_at_k']:.4f}<br>"
                f"hit@1={m1['hit']:.4f}<br>"
                f"hit@5={m5['hit']:.4f}<br>"
                f"mrr@10={m10['mrr']:.4f}"
            )
            lines.append(
                "| {name} | {beta:.8f} | {mu:.3f} | {ik} | "
                "{thalf:.1f} | {exp_result} | {w}/{t}/{l} | {wins} | {losses} |".format(
                    name=recipe["name"],
                    beta=recipe["beta"],
                    mu=recipe["mu_base"],
                    ik=recipe["inter_top_k"],
                    thalf=float(target.get("T_half_days", 0.0)),
                    exp_result=exp_result,
                    w=wtl["win"],
                    t=wtl["tie"],
                    l=wtl["loss"],
                    wins=md_cell_ids(wtl["win_question_ids"]),
                    losses=md_cell_ids(wtl["loss_question_ids"]),
                )
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def evaluate_config(prepared: list[dict[str, Any]], cfg: dict[str, Any],
                    final_top_k: int) -> dict[str, Any]:
    per_q = []
    for q in prepared:
        if cfg["beta"] == 0.0:
            _, h_sessions, h_full = run_cosine_sessions(
                q["session_vectors"], q["query_vector"],
                final_top_k=final_top_k, session_ids=q["session_ids"],
            )
        else:
            _, h_sessions, h_full = run_originidea_sessions(
                q["session_vectors"], q["session_times"], q["query_vector"],
                query_time=q["question_time"],
                beta=cfg["beta"], mu_base=cfg["mu_base"],
                intermediate_top_k=cfg["inter_top_k"],
                final_top_k=final_top_k,
                session_ids=q["session_ids"],
            )
        rec = session_recall_at_k(h_sessions, q["evidence_ids"])
        sm = {f"k{k}": session_metrics_at_k(h_full, q["gold_session_indices"], k)
              for k in (1, 3, 5, 10)}
        per_q.append({
            "question_id": q["question_id"],
            "question_type": q["question_type"],
            "n_sessions": q["n_sessions"],
            "n_evidence_sessions": q["n_evidence_sessions"],
            "session_recall_at_k": rec,
            "session_metrics": sm,
        })
    return {
        "aggregate": aggregate_per_question(per_q),
        "bucket_results": {
            "question_type": bucket_aggregates(per_q, "question_type"),
        },
        "per_question": per_q,
    }


# ---------- 主流程 ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="originidea.md sweep over (β, mu_base, inter_top_k) on LongMemEval-S multi-session."
    )
    parser.add_argument("--data", type=Path,
                        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"))
    parser.add_argument("--outputs-dir", type=Path,
                        default=Path("outputs/longmemeval_originidea_sweep"))
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-cache-dir", type=Path,
                        default=Path("benchmarks/longmemeval/cache/models"))
    parser.add_argument("--n-questions", type=int, default=0,
                        help="Number of questions per selected category to run; <=0 means all.")
    parser.add_argument("--question-types", nargs="+",
                        default=["multisession", "temporalreasoning", "knowledgeupdate"],
                        help="Question categories to run. Defaults to multisession temporalreasoning knowledgeupdate.")
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing {args.data}. Run `python3 benchmarks/longmemeval/download.py` first."
        )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] loading {args.data}")
    records = json.loads(args.data.read_text())
    requested_types = [normalize_question_type_name(v) for v in args.question_types]
    print(f"[sweep] requested question_types={requested_types}")

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    t0 = time.perf_counter()
    category_results: list[dict[str, Any]] = []
    for qtype in requested_types:
        available = [r for r in records if normalize_question_type_name(r.get("question_type", "")) == qtype]
        selected = available if args.n_questions <= 0 else available[: args.n_questions]
        print(f"\n[sweep] selected {len(selected)}/{len(available)} {qtype} questions")
        prepared = prepare_corpus(selected, embed_fn, args.embed_batch_size)
        stats = characterize_corpus(prepared)
        print(f"[sweep] {qtype} stats: c_ref={stats['c_ref']:.3f}  c_L={stats['c_L']:.3f}  "
              f"c_H={stats['c_H']:.3f}  dt_ref={stats['dt_ref']:.3f}d  "
              f"dt_L={stats['dt_L']:.3f}d  dt_H={stats['dt_H']:.3f}d")

        recipes = design_recipes(stats)
        print(f"[sweep] {qtype}: {len(recipes)} recipes "
              f"(1 cosine baseline + {len(recipes) - 1} experiment configs)")
        results: list[dict[str, Any]] = []
        baseline_per_q: dict[str, float] | None = None
        for r in recipes:
            rs = time.perf_counter()
            out = evaluate_config(prepared, r, args.final_top_k)
            elapsed = time.perf_counter() - rs
            agg = out["aggregate"]
            m1 = agg["session_metrics"]["k1"]
            m5 = agg["session_metrics"]["k5"]
            m10 = agg["session_metrics"]["k10"]
            per_q_recall = {str(x["question_id"]): float(x["session_recall_at_k"])
                            for x in out["per_question"]}
            if r["name"] == "R0_cosine_baseline":
                baseline_per_q = per_q_recall
            wtl = win_tie_loss_detail(out["per_question"], baseline_per_q)
            bucket_wtl = {
                "question_type": win_tie_loss_by_bucket(
                    out["per_question"], baseline_per_q, "question_type"
                )
            }
            print(f"[sweep] {qtype:20s} {r['name']:24s} "
                  f"recall@{args.final_top_k}={agg['session_recall_at_k']:.3f}  "
                  f"hit@1={m1['hit']:.3f}  hit@5={m5['hit']:.3f}  "
                  f"mrr@10={m10['mrr']:.3f}  srr@10={m10['srr']:.3f}  "
                  f"W/T/L={wtl['win']}/{wtl['tie']}/{wtl['loss']}  "
                  f"({elapsed:.1f}s)")
            results.append({
                "recipe": r,
                "aggregate": agg,
                "bucket_results": out["bucket_results"],
                "per_question": out["per_question"],
                "win_tie_loss_vs_baseline": wtl,
                "bucket_win_tie_loss_vs_baseline": bucket_wtl,
            })
        category_results.append({
            "question_type": qtype,
            "config": {
                "embedding": args.embedding,
                "n_questions": len(prepared),
                "requested_n_questions": args.n_questions,
                "n_available": len(available),
                "final_top_k": args.final_top_k,
                "global_stats": stats,
            },
            "recipes_results": results,
        })

    summary = {
        "config": {
            "embedding": args.embedding,
            "question_types": requested_types,
            "requested_n_questions": args.n_questions,
            "final_top_k": args.final_top_k,
        },
        "category_results": category_results,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    type_slug = "_".join(q.replace("-", "") for q in requested_types)
    out_json = args.outputs_dir / f"sweep_n{args.n_questions}_{type_slug}.json"
    if args.n_questions <= 0:
        out_json = args.outputs_dir / f"sweep_all_{type_slug}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[sweep] wrote {out_json}")
    out_md = out_json.with_suffix(".md")
    write_markdown_report(summary, out_md)
    print(f"[sweep] wrote {out_md}")

    print("\n=== Category summary ===")
    header = (f"{'question_type':20s}  {'n':>4s}  {'configs':>7s}  "
              f"{'cos recall':>10s}  {'best config':26s}  {'best recall':>11s}  {'W/T/L':>9s}")
    print(header)
    print("-" * len(header))
    for category_result in category_results:
        results = category_result["recipes_results"]
        base = next(r for r in results if r["recipe"]["name"] == "R0_cosine_baseline")
        non_base = [r for r in results if r["recipe"]["name"] != "R0_cosine_baseline"]
        best = max(non_base, key=lambda x: x["aggregate"]["session_recall_at_k"])
        wtl = best["win_tie_loss_vs_baseline"]
        print(f"{category_result['question_type']:20s}  "
              f"{category_result['config']['n_questions']:4d}  "
              f"{len(non_base):7d}  "
              f"{base['aggregate']['session_recall_at_k']:10.3f}  "
              f"{best['recipe']['name']:26s}  "
              f"{best['aggregate']['session_recall_at_k']:11.3f}  "
              f"{wtl['win']}/{wtl['tie']}/{wtl['loss']:>3d}")


if __name__ == "__main__":
    main()
