"""Trace Hawkes dynamics for one Statistics benchmark scenario.

This is the turn-level counterpart of
benchmarks/longmemeval/trace_originidea_one_question.py, adapted to the
synthetic statistics dataset. It does not call an LLM: it replays all turns
before an eval's query_turn, prints lambda decay/excitation dynamics, then
compares the same retrieval mechanisms used by sweep_hawkes.py against
positive/negative labels.

In this dataset, one turn is one complete back-and-forth exchange containing
both participants' messages. Turn indices in query_turn, positive_turns, and
negative_turns therefore refer to exchanges rather than individual sentences.

Examples:
  python3 benchmarks/statistics/trace_originidea_one_question.py \
    --scenario-id decay_B_weekend_breakfast_000 --embedding qwen

  python3 benchmarks/statistics/trace_originidea_one_question.py \
    --category update_override --embedding hashing --top-k 3

  python3 benchmarks/statistics/trace_originidea_one_question.py \
    --all-categories --output-dir outputs/statistics_hawkes_trace --embedding qwen
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

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
from benchmarks.statistics.sweep_hawkes import (  # noqa: E402
    DEFAULT_TOP_KS,
    eval_ranking,
    iter_scenario_files,
    parse_iso_days,
    rank_scores,
    scenario_sort_key,
    score_cosine,
    score_cosine_recency,
    score_hawkes,
    score_recency,
    turn_to_text,
)


def fmt_ts(t_days: float) -> str:
    if t_days <= 0:
        return "unknown"
    return datetime.fromtimestamp(t_days * 86400.0).strftime("%Y-%m-%d %H:%M")


def short(text: str, width: int = 86) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    return text if len(text) <= width else text[: width - 1] + "..."


def h_hat(lambdas: np.ndarray) -> tuple[float, float, int]:
    n = int(lambdas.size)
    if n <= 1:
        return 0.0, 0.0, n
    lam2 = np.asarray(lambdas, dtype=float) ** 2
    total = float(lam2.sum())
    if total <= 0.0:
        return 0.0, 0.0, n
    p = lam2 / total
    nz = p > 0.0
    h = -float(np.sum(p[nz] * np.log(p[nz])))
    return float(h / math.log(n)), h, n


def quantile_str(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "(empty)"
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return (
        f"q10={qs[0]:.4f} q25={qs[1]:.4f} q50={qs[2]:.4f} "
        f"q75={qs[3]:.4f} q90={qs[4]:.4f} n={arr.size}"
    )


def label_for_turn(idx: int, positives: set[int], negatives: set[int]) -> str:
    if idx in positives:
        return "+"
    if idx in negatives:
        return "-"
    return " "


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_scenario_files(data_dir):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        record["_category_dir"] = path.parent.name
        records.append(record)
    return records


def select_records(
    records: list[dict[str, Any]],
    *,
    scenario_id: str | None,
    category: str | None,
    all_categories: bool,
) -> list[dict[str, Any]]:
    if scenario_id:
        rec = next((r for r in records if str(r.get("scenario_id")) == scenario_id), None)
        if rec is None:
            raise SystemExit(f"scenario-id not found: {scenario_id}")
        return [rec]

    if all_categories:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in records:
            cat = str(rec.get("category") or "unknown")
            if cat not in seen:
                selected.append(rec)
                seen.add(cat)
        return selected

    if category:
        rec = next((r for r in records if str(r.get("category")) == category), None)
        if rec is None:
            choices = sorted({str(r.get("category") or "unknown") for r in records})
            raise SystemExit(f"category not found: {category}; choices={choices}")
        return [rec]

    return [records[0]]


def choose_eval(record: dict[str, Any], eval_index: int) -> dict[str, Any]:
    evals = record.get("evals") or []
    if not evals:
        raise SystemExit(f"scenario has no evals: {record.get('scenario_id')}")
    if eval_index < 0 or eval_index >= len(evals):
        raise SystemExit(
            f"eval-index out of range for {record.get('scenario_id')}: "
            f"{eval_index} not in [0, {len(evals) - 1}]"
        )
    return evals[eval_index]


def trace_record(
    record: dict[str, Any],
    embed_fn,
    *,
    eval_index: int,
    beta: float,
    mu_base: float,
    inter_top_k: int,
    top_ks: tuple[int, ...],
    primary_k: int,
    top_k_show: int,
    out: TextIO,
) -> None:
    ev = choose_eval(record, eval_index)
    turns = record.get("turns") or []
    query_turn = int(ev.get("query_turn"))
    if query_turn <= 0 or query_turn >= len(turns):
        raise SystemExit(f"bad query_turn={query_turn} in {record.get('scenario_id')}")

    texts = [turn_to_text(t) for t in turns]
    times = np.asarray([parse_iso_days(t.get("t")) for t in turns], dtype=float)
    vectors = normalize_rows(embed_texts(embed_fn, texts, batch_size=64))
    positives = {int(v) for v in ev.get("positive_turns") or []}
    negatives = {int(v) for v in ev.get("negative_turns") or []}

    n = query_turn
    hist_vec = vectors[:n]
    query_vec = vectors[query_turn]
    hist_times = times[:n]
    query_time = float(times[query_turn])

    scenario_id = str(record.get("scenario_id"))
    category = str(record.get("category") or "unknown")
    t_half = math.log(2.0) / beta if beta > 0 else float("inf")

    print("=" * 110, file=out)
    print(f"scenario_id : {scenario_id}", file=out)
    print(f"category    : {category} ({record.get('_category_dir', '')})", file=out)
    print(f"description : {short(record.get('description', ''), 130)}", file=out)
    print("turn_unit   : one back-and-forth exchange (not one sentence)", file=out)
    print(f"eval_index  : {eval_index}", file=out)
    print(f"query_turn  : exchange {query_turn}  t={fmt_ts(query_time)}", file=out)
    print(f"query_text  : {short(texts[query_turn], 130)}", file=out)
    print(f"positives   : {sorted(positives)}", file=out)
    print(f"negatives   : {sorted(negatives)}", file=out)
    print(
        f"hyperparams : beta={beta:.6f}/day (T_half={t_half:.3f}d) "
        f"mu_base={mu_base} inter_top_k={inter_top_k} primary_k={primary_k}",
        file=out,
    )
    print("=" * 110, file=out)

    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)

    print("\n# Replay before query", file=out)
    print("# each event is one exchange-level turn; marker: + positive, - negative, blank unlabelled", file=out)
    for i in range(n):
        t_i = float(hist_times[i])
        marker = label_for_turn(i, positives, negatives)
        print(
            f"\n[event {i + 1:>2}/{n}] exchange_turn={marker}{i:<3} "
            f"t={fmt_ts(t_i)} text={short(texts[i])}",
            file=out,
        )

        if not created.any():
            print("           (no prior memories)", file=out)
        else:
            idx_global = np.flatnonzero(created)
            decayed = lambdas[idx_global] * np.exp(-beta * (t_i - last_update[idx_global]))
            decayed = np.clip(decayed, 0.0, 1.0)
            mu = compute_mu(decayed, mu_base)
            h_norm, h_nats, n_used = h_hat(decayed)
            cos_vec = hist_vec[idx_global] @ hist_vec[i]
            bracket = mu + (1.0 - mu) * decayed
            scores = cos_vec * bracket

            print(
                f"           pool={len(idx_global)} H_hat={h_norm:.4f} H={h_nats:.4f} "
                f"lnN={math.log(max(n_used, 1)):.4f} mu={mu:.4f}",
                file=out,
            )
            print(f"           lambda- stats: {quantile_str(decayed)}", file=out)

            k_show = min(top_k_show, len(idx_global))
            top_haw = np.argsort(-scores)[:k_show]
            print(f"           top-{k_show} replay scores:", file=out)
            for rel in top_haw:
                rel = int(rel)
                j = int(idx_global[rel])
                m = label_for_turn(j, positives, negatives)
                print(
                    f"             score={scores[rel]:+.4f} cos={cos_vec[rel]:+.4f} "
                    f"lambda-={decayed[rel]:.4f} bracket={bracket[rel]:.4f} "
                    f"turn={m}{j:<3} :: {short(texts[j], 76)}",
                    file=out,
                )

            order = np.argsort(-scores)
            excited: list[str] = []
            for rel in order[: min(inter_top_k, len(order))]:
                rel = int(rel)
                score_i = float(scores[rel])
                if score_i <= 0.0:
                    continue
                global_j = int(idx_global[rel])
                lam_minus = float(decayed[rel])
                delta = (1.0 - lam_minus) * score_i
                new_lam = min(1.0, lam_minus + delta)
                lambdas[global_j] = new_lam
                last_update[global_j] = t_i
                excited.append(
                    f"turn={global_j}: lambda-={lam_minus:.4f} "
                    f"+ delta={delta:.4f} -> lambda+={new_lam:.4f}"
                )
            if excited:
                print(f"           excited {len(excited)}:", file=out)
                for line in excited:
                    print(f"             {line}", file=out)
            else:
                print("           excited 0 (all selected scores <= 0)", file=out)

        lambdas[i] = 1.0
        last_update[i] = t_i
        created[i] = True

    print("\n" + "#" * 110, file=out)
    print(f"[FINAL QUERY] exchange_turn={query_turn} t={fmt_ts(query_time)}", file=out)
    print(f"{short(texts[query_turn], 130)}", file=out)
    print("#" * 110, file=out)

    decayed_q = lambdas * np.exp(-beta * np.maximum(query_time - last_update, 0.0))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    mu_q = compute_mu(decayed_q, mu_base)
    h_norm_q, h_nats_q, n_used_q = h_hat(decayed_q)
    cos_q = hist_vec @ query_vec
    bracket_q = mu_q + (1.0 - mu_q) * decayed_q
    final_scores = cos_q * bracket_q
    sweep_hawkes_scores = score_hawkes(
        hist_vec,
        hist_times,
        query_vec,
        query_time,
        beta=beta,
        mu_base=mu_base,
        intermediate_top_k=inter_top_k,
    )
    max_hawkes_diff = float(np.max(np.abs(final_scores - sweep_hawkes_scores))) if n else 0.0
    cosine_scores = score_cosine(hist_vec, query_vec)
    recency_scores = score_recency(hist_times, query_time, beta)
    cosrec_scores = score_cosine_recency(hist_vec, hist_times, query_vec, query_time, beta, mu_base)

    print(
        f"H_hat={h_norm_q:.4f} H={h_nats_q:.4f} lnN={math.log(max(n_used_q, 1)):.4f} "
        f"mu={mu_q:.4f}",
        file=out,
    )
    print(f"lambda at query : {quantile_str(decayed_q)}", file=out)
    print(f"bracket stats   : {quantile_str(bracket_q)}", file=out)
    print(f"sweep_hawkes.score_hawkes max_abs_diff={max_hawkes_diff:.3e}", file=out)
    if len(cos_q) > 1 and np.std(cos_q) > 0 and np.std(bracket_q) > 0:
        corr = float(np.corrcoef(cos_q, bracket_q)[0, 1])
        print(f"corr(cos, bracket)={corr:.4f}", file=out)

    k_show = min(top_k_show, n)
    top_cos = np.argsort(-cosine_scores)[:k_show]
    top_haw = np.argsort(-final_scores)[:k_show]
    print(f"\nTop-{k_show} by cosine:", file=out)
    for j in top_cos:
        j = int(j)
        m = label_for_turn(j, positives, negatives)
        print(
            f"  cos={cosine_scores[j]:+.4f} lambda={decayed_q[j]:.4f} "
            f"bracket={bracket_q[j]:.4f} score={final_scores[j]:+.4f} "
            f"turn={m}{j:<3} :: {short(texts[j], 88)}",
            file=out,
        )

    print(f"\nTop-{k_show} by hawkes:", file=out)
    for j in top_haw:
        j = int(j)
        m = label_for_turn(j, positives, negatives)
        print(
            f"  score={final_scores[j]:+.4f} cos={cosine_scores[j]:+.4f} "
            f"lambda={decayed_q[j]:.4f} bracket={bracket_q[j]:.4f} "
            f"turn={m}{j:<3} :: {short(texts[j], 88)}",
            file=out,
        )

    mechanism_scores = {
        "cosine": cosine_scores,
        "recency": recency_scores,
        "cosine_recency": cosrec_scores,
        "hawkes": final_scores,
    }
    mechanism_metrics = {
        name: eval_ranking(scores, ev, top_ks=top_ks)
        for name, scores in mechanism_scores.items()
    }
    mechanism_orders: dict[str, list[int]] = {}
    mechanism_ranks: dict[str, dict[int, int]] = {}
    for name, scores in mechanism_scores.items():
        order, ranks = rank_scores(scores)
        mechanism_orders[name] = order
        mechanism_ranks[name] = ranks
    cosine_ranks = mechanism_ranks["cosine"]
    hawkes_ranks = mechanism_ranks["hawkes"]

    print("\nRanks of labelled turns (1-indexed):", file=out)
    for label, ids in (("positive", sorted(positives)), ("negative", sorted(negatives))):
        print(f"  {label}:", file=out)
        for idx in ids:
            if 0 <= idx < n:
                print(
                    f"    turn={idx:<3} cosine_rank={cosine_ranks[idx]:<3} "
                    f"recency_rank={mechanism_ranks['recency'][idx]:<3} "
                    f"cosrec_rank={mechanism_ranks['cosine_recency'][idx]:<3} "
                    f"hawkes_rank={hawkes_ranks[idx]:<3} "
                    f"text={short(texts[idx], 88)}",
                    file=out,
                )

    print("\nMetrics:", file=out)
    for method in ("cosine", "recency", "cosine_recency", "hawkes"):
        metrics = mechanism_metrics[method]
        km = metrics["by_k"][f"k{primary_k}"]
        print(
            f"  {method:<10} SRR@{primary_k}={km['srr']:.3f} "
            f"pos_recall@{primary_k}={km['positive_recall']:.3f} "
            f"neg_intr@{primary_k}={km['negative_intrusion']:.3f} "
            f"mrr@{primary_k}={km['mrr_positive']:.3f} "
            f"pair_win={metrics['pair_win_rate']:.3f} "
            f"rank_margin={metrics['rank_margin']:.1f}",
            file=out,
        )

    print("\nCompact final orders:", file=out)
    for method in ("cosine", "recency", "cosine_recency", "hawkes"):
        print(f"  {method:<14}: {mechanism_orders[method][:primary_k]}", file=out)


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_top_ks(values: list[int]) -> tuple[int, ...]:
    cleaned = sorted({int(v) for v in values if int(v) > 0})
    return tuple(cleaned or DEFAULT_TOP_KS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace one Statistics scenario under Hawkes dynamics and sweep_hawkes baselines."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/statistics"))
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Trace the first scenario from each category.",
    )
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--T-half", type=float, default=1.0, help="T_{1/2} in days.")
    parser.add_argument("--mu-base", type=float, default=0.6)
    parser.add_argument("--inter-top-k", type=int, default=1)
    parser.add_argument("--top-ks", nargs="+", type=int, default=list(DEFAULT_TOP_KS))
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10, help="How many rows to print per top list.")
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
        help="sentence-transformers cache; relative paths resolve from repo root.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="When tracing multiple records, write one .txt file per scenario.",
    )
    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)
    if args.primary_k not in top_ks:
        top_ks = tuple(sorted(set(top_ks + (args.primary_k,))))

    records = load_records(args.data_dir)
    if not records:
        raise SystemExit(f"No scenario JSON files found under {args.data_dir}")
    records = sorted(records, key=lambda r: (str(r.get("category") or ""), scenario_sort_key(Path(str(r["_path"])))))
    selected = select_records(
        records,
        scenario_id=args.scenario_id,
        category=args.category,
        all_categories=args.all_categories,
    )

    embed_fn = make_embedding_fn(
        args.embedding,
        device=args.device,
        cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
    )

    beta = math.log(2.0) / args.T_half

    if args.output and len(selected) > 1:
        raise SystemExit("--output can only be used with a single selected scenario; use --output-dir")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    elif len(selected) > 1:
        raise SystemExit("Multiple scenarios selected; provide --output-dir")

    for rec in selected:
        scenario_id = str(rec.get("scenario_id"))
        out_file = args.output
        if args.output_dir is not None:
            out_file = args.output_dir / f"trace_{scenario_id}_{args.embedding}.txt"

        with (out_file.open("w", encoding="utf-8") if out_file else nullcontext(None)) as fh:
            stream: TextIO
            if fh is None:
                stream = sys.stdout
            else:
                stream = Tee(sys.stdout, fh)
            trace_record(
                rec,
                embed_fn,
                eval_index=args.eval_index,
                beta=beta,
                mu_base=args.mu_base,
                inter_top_k=args.inter_top_k,
                top_ks=top_ks,
                primary_k=args.primary_k,
                top_k_show=args.top_k,
                out=stream,
            )
        if out_file:
            print(f"[trace saved -> {out_file}]")


if __name__ == "__main__":
    main()
