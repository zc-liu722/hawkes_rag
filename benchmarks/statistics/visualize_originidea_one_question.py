"""Generate an HTML visualization for one Statistics Hawkes trace.

This script is intentionally separate from trace_originidea_one_question.py:
it recomputes the same replay dynamics, records structured intermediate state,
and writes a self-contained HTML file with SVG charts and sortable tables.
The Statistics dataset is exchange-level: one turn is a complete back-and-forth
with both participants' messages, not a single sentence.

Examples:
  python3 benchmarks/statistics/visualize_originidea_one_question.py \
    --scenario-id override_A_clinic_slot_000 --embedding hashing

  python3 benchmarks/statistics/visualize_originidea_one_question.py \
    --scenario-id decay_B_weekend_breakfast_000 --embedding qwen \
    --output outputs/statistics_hawkes_trace/decay_B_weekend_breakfast_000.html \
    --trace-json outputs/statistics_hawkes_trace/decay_B_weekend_breakfast_000_trace.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
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


GREEN = "#0f9f6e"
RED = "#d43f3a"
BLUE = "#276ef1"
INK = "#172033"
MUTED = "#667085"
GRID = "#d7dde8"
BG = "#f7f8fb"


@dataclass
class FinalRow:
    turn: int
    label: str
    time: str
    text: str
    cosine: float
    recency: float
    cosine_recency: float
    lambda_q: float
    bracket: float
    score: float
    cosine_rank: int
    recency_rank: int
    cosine_recency_rank: int
    hawkes_rank: int
    selected_cosine: bool
    selected_hawkes: bool


def fmt_ts(t_days: float) -> str:
    if t_days <= 0:
        return "unknown"
    return datetime.fromtimestamp(t_days * 86400.0).strftime("%Y-%m-%d %H:%M")


def short(text: str, width: int = 96) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    return text if len(text) <= width else text[: width - 1] + "..."


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def label_for_turn(idx: int, positives: set[int], negatives: set[int]) -> str:
    if idx in positives:
        return "positive"
    if idx in negatives:
        return "negative"
    return "neutral"


def label_badge(label: str) -> str:
    if label == "positive":
        return '<span class="badge positive">positive</span>'
    if label == "negative":
        return '<span class="badge negative">negative</span>'
    return '<span class="badge neutral">neutral</span>'


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_scenario_files(data_dir):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        record["_category_dir"] = path.parent.name
        records.append(record)
    return sorted(
        records,
        key=lambda r: (str(r.get("category") or ""), scenario_sort_key(Path(str(r["_path"])))),
    )


def select_record(
    records: list[dict[str, Any]],
    *,
    scenario_id: str | None,
    category: str | None,
) -> dict[str, Any]:
    if scenario_id:
        rec = next((r for r in records if str(r.get("scenario_id")) == scenario_id), None)
        if rec is None:
            raise SystemExit(f"scenario-id not found: {scenario_id}")
        return rec
    if category:
        rec = next((r for r in records if str(r.get("category")) == category), None)
        if rec is None:
            choices = sorted({str(r.get("category") or "unknown") for r in records})
            raise SystemExit(f"category not found: {category}; choices={choices}")
        return rec
    return records[0]


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


def finite_float(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(value)


def h_hat(lambdas: np.ndarray) -> tuple[float, float, int]:
    """Normalized entropy H_hat, entropy in nats H, and pool size (matches text trace)."""
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


def build_trace_data(
    record: dict[str, Any],
    embed_fn,
    *,
    eval_index: int,
    beta: float,
    mu_base: float,
    inter_top_k: int,
    top_ks: tuple[int, ...],
    primary_k: int,
) -> dict[str, Any]:
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

    lambdas = np.zeros(n, dtype=float)
    last_update = np.zeros(n, dtype=float)
    created = np.zeros(n, dtype=bool)
    lambda_after_events = np.full((n, n), np.nan, dtype=float)
    event_records: list[dict[str, Any]] = []
    excitations: list[dict[str, Any]] = []

    for i in range(n):
        t_i = float(hist_times[i])
        event: dict[str, Any] = {
            "turn": i,
            "time": fmt_ts(t_i),
            "text": texts[i],
            "label": label_for_turn(i, positives, negatives),
            "pool_size": int(created.sum()),
            "mu": None,
            "H": None,
            "H_hat": None,
            "lambda_stats": None,
            "top_replay": [],
            "excited": [],
        }

        if created.any():
            idx_global = np.flatnonzero(created)
            decayed = lambdas[idx_global] * np.exp(-beta * (t_i - last_update[idx_global]))
            decayed = np.clip(decayed, 0.0, 1.0)
            mu = compute_mu(decayed, mu_base)
            h_norm, h_nats, _n_pool = h_hat(decayed)
            cos_vec = hist_vec[idx_global] @ hist_vec[i]
            bracket = mu + (1.0 - mu) * decayed
            scores = cos_vec * bracket
            order = np.argsort(-scores)
            event["mu"] = finite_float(mu)
            event["H"] = finite_float(h_nats)
            event["H_hat"] = finite_float(h_norm)
            event["lambda_stats"] = {
                "min": finite_float(np.min(decayed)),
                "q25": finite_float(np.quantile(decayed, 0.25)),
                "median": finite_float(np.quantile(decayed, 0.5)),
                "q75": finite_float(np.quantile(decayed, 0.75)),
                "max": finite_float(np.max(decayed)),
            }
            for rel in order[: min(10, len(order))]:
                rel = int(rel)
                j = int(idx_global[rel])
                event["top_replay"].append(
                    {
                        "turn": j,
                        "label": label_for_turn(j, positives, negatives),
                        "cosine": finite_float(cos_vec[rel]),
                        "lambda_minus": finite_float(decayed[rel]),
                        "bracket": finite_float(bracket[rel]),
                        "score": finite_float(scores[rel]),
                        "text": texts[j],
                    }
                )

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
                item = {
                    "event_turn": i,
                    "target_turn": global_j,
                    "event_label": label_for_turn(i, positives, negatives),
                    "target_label": label_for_turn(global_j, positives, negatives),
                    "cosine": finite_float(cos_vec[rel]),
                    "score": finite_float(score_i),
                    "lambda_minus": finite_float(lam_minus),
                    "delta": finite_float(delta),
                    "lambda_plus": finite_float(new_lam),
                }
                event["excited"].append(item)
                excitations.append(item)

        lambdas[i] = 1.0
        last_update[i] = t_i
        created[i] = True
        visible = np.flatnonzero(created)
        lambda_after_events[i, visible] = lambdas[visible] * np.exp(
            -beta * np.maximum(t_i - last_update[visible], 0.0)
        )
        lambda_after_events[i, visible] = np.clip(lambda_after_events[i, visible], 0.0, 1.0)
        event_records.append(event)

    decayed_q = lambdas * np.exp(-beta * np.maximum(query_time - last_update, 0.0))
    decayed_q = np.clip(decayed_q, 0.0, 1.0)
    mu_q = compute_mu(decayed_q, mu_base)
    h_norm_q, h_nats_q, _ = h_hat(decayed_q)
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
    mechanism_scores = {
        "cosine": score_cosine(hist_vec, query_vec),
        "recency": score_recency(hist_times, query_time, beta),
        "cosine_recency": score_cosine_recency(
            hist_vec, hist_times, query_vec, query_time, beta, mu_base
        ),
        "hawkes": final_scores,
    }
    mechanism_orders: dict[str, list[int]] = {}
    mechanism_ranks: dict[str, dict[int, int]] = {}
    mechanism_metrics: dict[str, dict[str, Any]] = {}
    for name, scores in mechanism_scores.items():
        order, ranks = rank_scores(scores)
        mechanism_orders[name] = order
        mechanism_ranks[name] = ranks
        mechanism_metrics[name] = eval_ranking(scores, ev, top_ks=top_ks)
    selected_cos = set(mechanism_orders["cosine"][:primary_k])
    selected_hawkes = set(mechanism_orders["hawkes"][:primary_k])

    final_rows = []
    for j in range(n):
        final_rows.append(
            {
                "turn": j,
                "label": label_for_turn(j, positives, negatives),
                "time": fmt_ts(float(hist_times[j])),
                "text": texts[j],
                "cosine": finite_float(mechanism_scores["cosine"][j]),
                "recency": finite_float(mechanism_scores["recency"][j]),
                "cosine_recency": finite_float(mechanism_scores["cosine_recency"][j]),
                "lambda_q": finite_float(decayed_q[j]),
                "bracket": finite_float(bracket_q[j]),
                "score": finite_float(final_scores[j]),
                "cosine_rank": int(mechanism_ranks["cosine"][j]),
                "recency_rank": int(mechanism_ranks["recency"][j]),
                "cosine_recency_rank": int(mechanism_ranks["cosine_recency"][j]),
                "hawkes_rank": int(mechanism_ranks["hawkes"][j]),
                "selected_cosine": j in selected_cos,
                "selected_hawkes": j in selected_hawkes,
            }
        )

    return {
        "scenario": {
            "scenario_id": str(record.get("scenario_id")),
            "category": str(record.get("category") or "unknown"),
            "category_dir": str(record.get("_category_dir", "")),
            "description": str(record.get("description") or ""),
            "path": str(record.get("_path") or ""),
        },
        "hyperparams": {
            "beta": beta,
            "T_half_days": math.log(2.0) / beta if beta > 0 else None,
            "mu_base": mu_base,
            "inter_top_k": inter_top_k,
            "primary_k": primary_k,
            "score_hawkes_max_abs_diff": max_hawkes_diff,
        },
        "eval": {
            "eval_index": eval_index,
            "query_turn": query_turn,
            "query_time": fmt_ts(query_time),
            "query_text": texts[query_turn],
            "positive_turns": sorted(positives),
            "negative_turns": sorted(negatives),
        },
        "turns": [
            {
                "turn": i,
                "time": fmt_ts(float(times[i])),
                "text": texts[i],
                "label": label_for_turn(i, positives, negatives),
                "is_query": i == query_turn,
            }
            for i in range(len(turns))
        ],
        "events": event_records,
        "excitations": excitations,
        "lambda_after_events": [
            [None if np.isnan(v) else finite_float(v) for v in row] for row in lambda_after_events
        ],
        "final": {
            "mu": finite_float(mu_q),
            "H": finite_float(h_nats_q),
            "H_hat": finite_float(h_norm_q),
            "top_cosine": mechanism_orders["cosine"][:primary_k],
            "top_recency": mechanism_orders["recency"][:primary_k],
            "top_cosine_recency": mechanism_orders["cosine_recency"][:primary_k],
            "top_hawkes": mechanism_orders["hawkes"][:primary_k],
            "rows": final_rows,
            "metrics": mechanism_metrics,
        },
    }


def lambda_color(value: float | None) -> str:
    if value is None:
        return "#eef1f6"
    v = max(0.0, min(1.0, float(value)))
    # Light blue to saturated blue, with enough contrast for dense matrices.
    r = int(238 - 198 * v)
    g = int(244 - 118 * v)
    b = int(255 - 18 * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def label_color(label: str) -> str:
    if label == "positive":
        return GREEN
    if label == "negative":
        return RED
    return "#98a2b3"


def metric_summary(metrics: dict[str, Any], primary_k: int) -> str:
    km = metrics["by_k"][f"k{primary_k}"]
    return (
        f"SRR@{primary_k}={km['srr']:.3f}, "
        f"pos_recall@{primary_k}={km['positive_recall']:.3f}, "
        f"neg_intr@{primary_k}={km['negative_intrusion']:.3f}, "
        f"mrr@{primary_k}={km['mrr_positive']:.3f}, "
        f"pair_win={metrics['pair_win_rate']:.3f}, "
        f"rank_margin={metrics['rank_margin']:.1f}"
    )


def _excited_targets_by_event(excitations: list[dict[str, Any]]) -> dict[int, list[int]]:
    """event_turn -> ordered unique memory turns excited at that event."""
    raw: dict[int, list[int]] = {}
    for e in excitations:
        ei = int(e["event_turn"])
        tj = int(e["target_turn"])
        raw.setdefault(ei, []).append(tj)
    out: dict[int, list[int]] = {}
    for ei, tgts in raw.items():
        seen: set[int] = set()
        uniq: list[int] = []
        for t in tgts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        out[ei] = uniq
    return out


def render_lambda_heatmap(data: dict[str, Any]) -> str:
    matrix = data["lambda_after_events"]
    turns = data["turns"]
    n = len(matrix)
    cell = 16 if n <= 55 else 12
    left = 82
    top = 34
    exc_row_gap = 22
    caption_gap = 44
    width = left + n * cell + 28
    exc_y = top + n * cell + 14
    height = top + n * cell + exc_row_gap + caption_gap
    excitations = data.get("excitations") or []
    excited_pairs = {(int(e["event_turn"]), int(e["target_turn"])) for e in excitations}
    by_event = _excited_targets_by_event(excitations)
    query_turn = int(data["eval"]["query_turn"])
    primary_k = int(data["hyperparams"].get("primary_k") or len(data["final"].get("top_cosine") or ()))
    top_cos = [int(t) for t in (data["final"].get("top_cosine") or [])]
    top_hawkes = [int(t) for t in (data["final"].get("top_hawkes") or [])]
    top_cos_set = set(top_cos)
    top_hawkes_set = set(top_hawkes)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">',
        f'<text x="{left}" y="20" class="axis-label">event turn</text>',
        f'<text x="8" y="{top - 10}" class="axis-label">memory</text>',
        f'<text x="{left}" y="{exc_y - 4}" class="tick-micro" fill="{MUTED}">excited memory turn(s) per event column →</text>',
    ]
    for i in range(n):
        x = left + i * cell + cell / 2
        if i % 5 == 0 or i == n - 1:
            parts.append(f'<text x="{x:.1f}" y="{top - 10}" class="tick" text-anchor="middle">{i}</text>')
        y = top + i * cell + cell / 2 + 4
        label = turns[i]["label"]
        color = label_color(label)
        pick_cos = i in top_cos_set
        pick_h = i in top_hawkes_set
        if pick_cos and pick_h:
            pick_note = '<tspan dx="2" class="pick-c">C</tspan><tspan class="pick-h">H</tspan>'
        elif pick_cos:
            pick_note = '<tspan dx="2" class="pick-c">C</tspan>'
        elif pick_h:
            pick_note = '<tspan dx="2" class="pick-h">H</tspan>'
        else:
            pick_note = ""
        parts.append(
            f'<text x="{left - 10}" y="{y:.1f}" class="tick" text-anchor="end" fill="{color}">'
            f"{i}{pick_note}</text>"
        )
        tgt_list = by_event.get(i, [])
        exc_txt = ",".join(str(t) for t in tgt_list) if tgt_list else "—"
        exc_title = (
            f"event {i} excited memory turn(s): {exc_txt}"
            if tgt_list
            else f"event {i}: no replay excitations (empty pool or non-positive scores)"
        )
        tspan_exc = esc(exc_txt) if len(exc_txt) <= 12 or cell >= 14 else esc(exc_txt[:10] + "…")
        parts.append(
            f'<g><title>{esc(exc_title)}</title>'
            f'<text x="{x:.1f}" y="{exc_y}" class="tick-micro" text-anchor="middle">{tspan_exc}</text></g>'
        )
    for event_i, row in enumerate(matrix):
        for mem_j, value in enumerate(row):
            x = left + event_i * cell
            y = top + mem_j * cell
            label = turns[mem_j]["label"]
            is_excited = (event_i, mem_j) in excited_pairs
            stroke = label_color(label) if label != "neutral" else "#ffffff"
            sw = 1.7 if label != "neutral" else 0.35
            title = (
                f"event turn {event_i}, memory turn {mem_j}, "
                f"lambda={'not created' if value is None else f'{value:.4f}'}, label={label}"
            )
            if is_excited:
                title += "; excited at this event"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{lambda_color(value)}" stroke="{stroke}" stroke-width="{sw}">'
                f"<title>{esc(title)}</title></rect>"
            )
            if is_excited:
                inset = max(0.65, min(2.2, cell * 0.12))
                inner = cell - 2 * inset
                parts.append(
                    f'<rect x="{x + inset}" y="{y + inset}" width="{inner}" height="{inner}" '
                    f'fill="none" stroke="#f59e0b" stroke-width="1.75" pointer-events="none"/>'
                )
    q_col = min(query_turn, n - 1)
    qx = left + q_col * cell
    parts.append(
        f'<line x1="{qx}" y1="{top}" x2="{qx}" y2="{top + n * cell}" '
        f'stroke="{INK}" stroke-width="2" stroke-dasharray="4 3">'
        f"<title>{esc(f'last replay event column {q_col}; query at turn {query_turn}')}</title></line>"
    )
    cap1 = (
        "Darker cells = higher λ after each replay event. "
        "Amber outline = memory turn excited by that event column. "
        "Row suffix C/H = in pure-cosine / Hawkes top-"
        f"{primary_k} at query turn {query_turn}."
    )
    cos_str = ",".join(str(t) for t in top_cos)
    hawk_str = ",".join(str(t) for t in top_hawkes)
    cap2 = f"Query top-{primary_k}: cosine → [{cos_str}] ; Hawkes → [{hawk_str}]"
    parts.append(f'<text x="{left}" y="{height - 28}" class="caption">{esc(cap1)}</text>')
    parts.append(f'<text x="{left}" y="{height - 12}" class="caption">{esc(cap2)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_excitation_graph(data: dict[str, Any]) -> str:
    n = int(data["eval"]["query_turn"])
    excitations = data["excitations"]
    turns = data["turns"]
    width = max(980, 34 * n)
    height = 330
    left = 42
    right = 24
    axis_y = 245
    scale = (width - left - right) / max(n - 1, 1)

    def x_for(turn: int) -> float:
        return left + turn * scale

    max_delta = max((float(e["delta"]) for e in excitations), default=1.0)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" stroke="{GRID}" />')
    for e in excitations:
        src = int(e["event_turn"])
        tgt = int(e["target_turn"])
        x1 = x_for(src)
        x2 = x_for(tgt)
        span = abs(x1 - x2)
        arch = max(32, min(180, span * 0.36))
        y_mid = axis_y - arch
        stroke = label_color(str(e["target_label"]))
        sw = 1.2 + 5.0 * (float(e["delta"]) / max_delta)
        title = (
            f"event {src} excited turn {tgt}; score={e['score']:.4f}; "
            f"cos={e['cosine']:.4f}; lambda {e['lambda_minus']:.4f} "
            f"+ {e['delta']:.4f} -> {e['lambda_plus']:.4f}"
        )
        parts.append(
            f'<path d="M{x1:.1f},{axis_y:.1f} Q{(x1 + x2) / 2:.1f},{y_mid:.1f} {x2:.1f},{axis_y:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{sw:.2f}" stroke-opacity="0.42">'
            f"<title>{esc(title)}</title></path>"
        )
    for i in range(n):
        x = x_for(i)
        label = turns[i]["label"]
        fill = label_color(label) if label != "neutral" else "#ffffff"
        stroke = label_color(label) if label != "neutral" else "#7d889b"
        r = 6.5 if label != "neutral" else 4.5
        parts.append(
            f'<circle cx="{x:.1f}" cy="{axis_y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.8">'
            f"<title>{esc(f'turn {i} {label}: {short(turns[i]['text'], 130)}')}</title></circle>"
        )
        if i % 5 == 0 or label != "neutral" or i == n - 1:
            parts.append(f'<text x="{x:.1f}" y="{axis_y + 24}" class="tick" text-anchor="middle">{i}</text>')
    parts.append(
        f'<text x="{left}" y="24" class="axis-label">Excitation edges: event turn -> historical turn selected by replay top-k</text>'
    )
    parts.append(
        f'<text x="{left}" y="{height - 18}" class="caption">Line width follows lambda delta; edge color follows the target turn label.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_lambda_lines(data: dict[str, Any]) -> str:
    matrix = data["lambda_after_events"]
    final_rows = data["final"]["rows"]
    n = int(data["eval"]["query_turn"])
    interesting = []
    labelled = [r["turn"] for r in final_rows if r["label"] != "neutral"]
    top_hawkes = data["final"]["top_hawkes"]
    top_cos = data["final"]["top_cosine"]
    for idx in labelled + top_hawkes + top_cos:
        if idx not in interesting:
            interesting.append(idx)
    interesting = interesting[:12]
    width = 980
    height = 320
    left = 48
    top = 24
    plot_w = width - 82
    plot_h = height - 78

    def xy(event_i: int, value: float) -> tuple[float, float]:
        x = left + event_i * plot_w / max(n - 1, 1)
        y = top + (1.0 - value) * plot_h
        return x, y

    palette = [
        "#276ef1",
        "#0f9f6e",
        "#d43f3a",
        "#7a4dd8",
        "#d97706",
        "#0086a8",
        "#b42318",
        "#344054",
        "#2e90fa",
        "#12b76a",
        "#f04438",
        "#6938ef",
    ]
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="{GRID}" />')
    for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - yv) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#edf0f5" />')
        parts.append(f'<text x="{left - 9}" y="{y + 4:.1f}" class="tick" text-anchor="end">{yv:.2f}</text>')
    for k, turn in enumerate(interesting):
        pts = []
        for event_i, row in enumerate(matrix):
            value = row[turn] if turn < len(row) else None
            if value is None:
                continue
            pts.append(xy(event_i, float(value)))
        if len(pts) < 2:
            continue
        color = palette[k % len(palette)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        row = final_rows[turn]
        title = f"turn {turn} {row['label']}; final lambda={row['lambda_q']:.4f}; hawkes rank={row['hawkes_rank']}"
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-opacity="0.9">'
            f"<title>{esc(title)}</title></polyline>"
        )
        x_end, y_end = pts[-1]
        parts.append(f'<text x="{x_end + 5:.1f}" y="{y_end + 4:.1f}" class="tick" fill="{color}">t{turn}</text>')
    parts.append(f'<text x="{left}" y="{height - 18}" class="caption">Lambda trajectories for labelled turns plus final top-k candidates.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_mu_h_curves(data: dict[str, Any]) -> str:
    """Line chart: μ and normalized entropy Ĥ (H_hat, 0–1) vs turn; shared y-axis; includes query turn."""
    query_turn = int(data["eval"]["query_turn"])
    events = data["events"]
    final = data["final"]
    mu_f = float(final["mu"])
    h_hat_f = float(final.get("H_hat") or 0.0)

    xs: list[float] = []
    mu_series: list[float] = []
    h_hat_series: list[float] = []
    for ev in events:
        if ev["mu"] is None:
            continue
        xs.append(float(ev["turn"]))
        mu_series.append(float(ev["mu"]))
        h_hat_series.append(float(ev["H_hat"]) if ev.get("H_hat") is not None else 0.0)

    xs.append(float(query_turn))
    mu_series.append(mu_f)
    h_hat_series.append(h_hat_f)

    width = 980
    height = 310
    left = 52
    right_pad = 12
    top = 38
    bottom = 48
    plot_w = width - left - right_pad
    plot_h = height - top - bottom

    def x_pix(t: float) -> float:
        return left + (t / max(query_turn, 1)) * plot_w

    def y_01(v: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, float(v)))) * plot_h

    parts: list[str] = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="{GRID}" />')

    for yv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_01(yv)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#edf0f5" />'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end" fill="{MUTED}">{yv:.2f}</text>'
        )

    step = max(1, query_turn // 10) if query_turn >= 10 else 1
    tick_turns = list(range(0, query_turn + 1, step))
    if tick_turns[-1] != query_turn:
        tick_turns.append(query_turn)
    for t in tick_turns:
        x = x_pix(float(t))
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 22}" class="tick" text-anchor="middle">{t}</text>'
        )

    if len(xs) >= 2:
        d_mu = " ".join(f"{x_pix(xs[i]):.1f},{y_01(mu_series[i]):.1f}" for i in range(len(xs)))
        d_h = " ".join(f"{x_pix(xs[i]):.1f},{y_01(h_hat_series[i]):.1f}" for i in range(len(xs)))
        parts.append(
            f'<polyline points="{d_mu}" fill="none" stroke="{BLUE}" stroke-width="2.2">'
            f"<title>{esc('μ at replay step start and at query turn')}</title></polyline>"
        )
        parts.append(
            f'<polyline points="{d_h}" fill="none" stroke="#b45309" stroke-width="2.2">'
            f"<title>{esc('Ĥ = H / ln N (normalized λ² entropy, 0–1)')}</title></polyline>"
        )
    for i in range(len(xs)):
        turn_i = int(xs[i])
        parts.append(
            f'<circle cx="{x_pix(xs[i]):.1f}" cy="{y_01(mu_series[i]):.1f}" r="3.5" '
            f'fill="{BLUE}" fill-opacity="0.85">'
            f"<title>{esc(f'turn {turn_i}: μ={mu_series[i]:.4f}')}</title></circle>"
        )
        parts.append(
            f'<circle cx="{x_pix(xs[i]):.1f}" cy="{y_01(h_hat_series[i]):.1f}" r="3.5" '
            f'fill="#b45309" fill-opacity="0.85">'
            f"<title>{esc(f'turn {turn_i}: H_hat={h_hat_series[i]:.4f}')}</title></circle>"
        )

    parts.append(f'<text x="{left}" y="24" class="axis-label" fill="{BLUE}">μ</text>')
    parts.append(f'<text x="{left + 36}" y="24" class="axis-label" fill="#b45309">Ĥ</text>')
    parts.append(
        f'<text x="{left + 78}" y="24" class="axis-label" fill="{MUTED}">(shared axis 0–1)</text>'
    )
    parts.append(
        f'<text x="{left}" y="{height - 10}" class="caption">'
        f"μ and Ĥ from decayed λ over the memory pool before each replay event (Ĥ = H/ln N); "
        f"rightmost point is query turn {query_turn}.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_final_table(data: dict[str, Any]) -> str:
    rows = [FinalRow(**r) for r in data["final"]["rows"]]
    max_abs_cos = max((abs(r.cosine) for r in rows), default=1.0) or 1.0
    max_abs_score = max((abs(r.score) for r in rows), default=1.0) or 1.0
    body = []
    for r in rows:
        classes = ["final-row", r.label]
        if r.selected_hawkes:
            classes.append("picked-hawkes")
        if r.selected_cosine:
            classes.append("picked-cosine")
        cos_w = 100 * abs(r.cosine) / max_abs_cos
        score_w = 100 * abs(r.score) / max_abs_score
        body.append(
            "<tr "
            f'class="{" ".join(classes)}" '
            f'data-label="{esc(r.label)}" '
            f'data-picked-hawkes="{str(r.selected_hawkes).lower()}" '
            f'data-picked-cos="{str(r.selected_cosine).lower()}">'
            f'<td class="num">{r.turn}</td>'
            f"<td>{label_badge(r.label)}</td>"
            f'<td class="num">{r.cosine_rank}</td>'
            f'<td class="num">{r.hawkes_rank}</td>'
            f'<td><div class="bar"><span style="width:{cos_w:.1f}%;"></span><b>{r.cosine:+.4f}</b></div></td>'
            f'<td class="num">{r.lambda_q:.4f}</td>'
            f'<td class="num">{r.bracket:.4f}</td>'
            f'<td><div class="bar score"><span style="width:{score_w:.1f}%;"></span><b>{r.score:+.4f}</b></div></td>'
            f'<td class="pick">{"hawkes " if r.selected_hawkes else ""}{"cosine" if r.selected_cosine else ""}</td>'
            f'<td class="text-cell"><span class="time">{esc(r.time)}</span>{esc(r.text)}</td>'
            "</tr>"
        )
    return f"""
<div class="table-toolbar">
  <button type="button" data-filter="all">all</button>
  <button type="button" data-filter="positive">positive</button>
  <button type="button" data-filter="negative">negative</button>
  <button type="button" data-filter="picked-hawkes">hawkes top-k</button>
  <button type="button" data-filter="picked-cos">cosine top-k</button>
</div>
<div class="table-wrap">
<table id="finalTable">
  <thead>
    <tr>
      <th data-sort="num">turn</th>
      <th>label</th>
      <th data-sort="num">cos rank</th>
      <th data-sort="num">hawkes rank</th>
      <th data-sort="num">cosine</th>
      <th data-sort="num">lambda</th>
      <th data-sort="num">bracket</th>
      <th data-sort="num">score</th>
      <th>picked</th>
      <th>turn text</th>
    </tr>
  </thead>
  <tbody>
    {"".join(body)}
  </tbody>
</table>
</div>
"""


def render_top_lists(data: dict[str, Any]) -> str:
    rows = {int(r["turn"]): r for r in data["final"]["rows"]}

    def render_list(name: str, indices: list[int], metric: str) -> str:
        items = []
        for rank, idx in enumerate(indices, start=1):
            r = rows[int(idx)]
            items.append(
                f'<li class="{esc(r["label"])}">'
                f'<span class="rank">#{rank}</span> '
                f'<span class="turn">turn {idx}</span> '
                f'{label_badge(r["label"])} '
                f'<span class="metric">{metric}={r[metric]:+.4f}</span>'
                f'<p>{esc(short(r["text"], 150))}</p>'
                "</li>"
            )
        return f"<div class=\"top-list\"><h3>{esc(name)}</h3><ol>{''.join(items)}</ol></div>"

    return (
        '<div class="top-grid">'
        + render_list("Pure cosine top-k", data["final"]["top_cosine"], "cosine")
        + render_list("Hawkes score top-k", data["final"]["top_hawkes"], "score")
        + "</div>"
    )


def render_event_table(data: dict[str, Any]) -> str:
    rows = []
    for ev in data["events"]:
        mu_text = "" if ev["mu"] is None else f"{ev['mu']:.4f}"
        h_hat_text = "" if ev.get("H_hat") is None else f"{ev['H_hat']:.4f}"
        excited = ev["excited"]
        if excited:
            excited_html = "<br>".join(
                f't{e["target_turn"]} '
                f'<span class="{esc(e["target_label"])}">({esc(e["target_label"])})</span> '
                f'score={e["score"]:+.4f}, delta={e["delta"]:.4f}'
                for e in excited
            )
        else:
            excited_html = '<span class="muted">none</span>'
        top = ev["top_replay"][:3]
        top_html = "<br>".join(
            f't{r["turn"]} score={r["score"]:+.4f} cos={r["cosine"]:+.4f} '
            f'lambda={r["lambda_minus"]:.4f}'
            for r in top
        )
        rows.append(
            f'<tr class="{esc(ev["label"])}">'
            f'<td class="num">{ev["turn"]}</td>'
            f"<td>{label_badge(ev['label'])}</td>"
            f'<td class="num">{mu_text}</td>'
            f'<td class="num">{h_hat_text}</td>'
            f"<td>{excited_html}</td>"
            f"<td>{top_html}</td>"
            f'<td class="text-cell"><span class="time">{esc(ev["time"])}</span>{esc(short(ev["text"], 180))}</td>'
            "</tr>"
        )
    return f"""
<div class="table-wrap compact">
<table>
  <thead>
    <tr><th>event</th><th>label</th><th>mu</th><th>H_hat</th><th>excited turns</th><th>top replay candidates</th><th>event text</th></tr>
  </thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</div>
"""


def render_html(data: dict[str, Any]) -> str:
    scenario = data["scenario"]
    ev = data["eval"]
    hp = data["hyperparams"]
    primary_k = int(hp["primary_k"])
    json_blob = json.dumps(data, ensure_ascii=False)
    cosine_summary = metric_summary(data["final"]["metrics"]["cosine"], primary_k)
    hawkes_summary = metric_summary(data["final"]["metrics"]["hawkes"], primary_k)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hawkes Trace: {esc(scenario["scenario_id"])}</title>
<style>
:root {{
  --ink: {INK};
  --muted: {MUTED};
  --grid: {GRID};
  --bg: {BG};
  --green: {GREEN};
  --red: {RED};
  --blue: {BLUE};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  padding: 28px 32px 18px;
  background: #fff;
  border-bottom: 1px solid var(--grid);
}}
h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
h2 {{ margin: 0 0 14px; font-size: 18px; letter-spacing: 0; }}
h3 {{ margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }}
.sub {{ color: var(--muted); max-width: 1100px; }}
.meta {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
  margin-top: 18px;
}}
.meta div, .metric-card {{
  background: #fff;
  border: 1px solid var(--grid);
  border-radius: 8px;
  padding: 10px 12px;
}}
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 12px;
  margin-top: 14px;
}}
main {{ padding: 22px 32px 48px; }}
section {{
  margin: 0 0 24px;
  background: #fff;
  border: 1px solid var(--grid);
  border-radius: 8px;
  padding: 18px;
  overflow-x: auto;
}}
.chart {{
  display: block;
  width: 100%;
  min-width: 860px;
  height: auto;
}}
.axis-label {{ font-size: 12px; font-weight: 700; fill: var(--ink); }}
.tick {{ font-size: 10px; fill: var(--muted); }}
.tick-micro {{ font-size: 7px; fill: var(--muted); }}
.pick-c {{ fill: #276ef1; font-size: 7.5px; font-weight: 700; }}
.pick-h {{ fill: #b45309; font-size: 7.5px; font-weight: 700; }}
.caption {{ font-size: 12px; fill: var(--muted); }}
.badge {{
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid var(--grid);
  background: #f5f7fb;
}}
.badge.positive {{ color: var(--green); background: #ecfdf5; border-color: #a7f3d0; }}
.badge.negative {{ color: var(--red); background: #fff1f0; border-color: #ffc9c5; }}
.badge.neutral {{ color: #667085; }}
.top-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
}}
.top-list {{
  border: 1px solid var(--grid);
  border-radius: 8px;
  padding: 14px;
}}
.top-list ol {{ margin: 0; padding: 0; list-style: none; }}
.top-list li {{ padding: 10px 0; border-top: 1px solid #edf0f5; }}
.top-list li:first-child {{ border-top: 0; }}
.top-list p {{ margin: 6px 0 0; color: var(--muted); }}
.top-list .rank, .top-list .turn, .metric {{ font-weight: 700; }}
.positive .turn {{ color: var(--green); }}
.negative .turn {{ color: var(--red); }}
.table-toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
button {{
  border: 1px solid var(--grid);
  background: #fff;
  color: var(--ink);
  border-radius: 7px;
  padding: 7px 10px;
  cursor: pointer;
}}
button.active {{ border-color: var(--blue); color: var(--blue); background: #eef4ff; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--grid); border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; min-width: 1100px; background: #fff; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f5; vertical-align: top; }}
th {{ text-align: left; font-size: 12px; color: var(--muted); background: #fafbfe; position: sticky; top: 0; }}
th[data-sort] {{ cursor: pointer; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.text-cell {{ min-width: 360px; }}
.time {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }}
.pick {{ color: var(--blue); font-weight: 700; min-width: 92px; }}
tr.positive {{ background: #fbfffd; }}
tr.negative {{ background: #fffafa; }}
tr.picked-hawkes {{ box-shadow: inset 3px 0 0 var(--blue); }}
tr.picked-cosine td:first-child {{ text-decoration: underline; text-decoration-color: var(--blue); text-decoration-thickness: 2px; }}
.bar {{
  position: relative;
  min-width: 120px;
  height: 22px;
  background: #f2f4f7;
  border-radius: 5px;
  overflow: hidden;
}}
.bar span {{
  display: block;
  height: 100%;
  background: rgba(39, 110, 241, 0.22);
}}
.bar.score span {{ background: rgba(15, 159, 110, 0.22); }}
.bar b {{
  position: absolute;
  inset: 2px 6px;
  font-variant-numeric: tabular-nums;
}}
.muted {{ color: var(--muted); }}
.compact table {{ min-width: 900px; }}
code {{ background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>{esc(scenario["scenario_id"])}</h1>
  <div class="sub">{esc(scenario["description"])}</div>
  <div class="meta">
    <div><b>category</b><br>{esc(scenario["category"])} ({esc(scenario["category_dir"])})</div>
    <div><b>query</b><br>exchange-level turn {ev["query_turn"]} at {esc(ev["query_time"])}</div>
    <div><b>labels</b><br>positive={esc(ev["positive_turns"])} negative={esc(ev["negative_turns"])}</div>
    <div><b>hyperparams</b><br>T_half={hp["T_half_days"]:.4g}d, mu_base={hp["mu_base"]}, inter_top_k={hp["inter_top_k"]}, top_k={primary_k}</div>
  </div>
  <div class="metric-grid">
    <div class="metric-card"><b>pure cosine</b><br>{esc(cosine_summary)}</div>
    <div class="metric-card"><b>hawkes</b><br>{esc(hawkes_summary)}</div>
  </div>
</header>
<main>
  <section>
    <h2>Query</h2>
    <p>{esc(ev["query_text"])}</p>
  </section>
  <section>
    <h2>Final Top-K Comparison</h2>
    {render_top_lists(data)}
  </section>
  <section>
    <h2>Lambda Heatmap</h2>
    {render_lambda_heatmap(data)}
  </section>
  <section>
    <h2>Excitation Graph</h2>
    {render_excitation_graph(data)}
  </section>
  <section>
    <h2>Lambda Trajectories</h2>
    {render_lambda_lines(data)}
  </section>
  <section>
    <h2>μ and Ĥ vs turn</h2>
    {render_mu_h_curves(data)}
  </section>
  <section>
    <h2>Final Query Scores</h2>
    {render_final_table(data)}
  </section>
  <section>
    <h2>Replay Events</h2>
    {render_event_table(data)}
  </section>
</main>
<script id="trace-data" type="application/json">{esc(json_blob)}</script>
<script>
(function () {{
  const table = document.getElementById('finalTable');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const getCellValue = (row, idx) => row.children[idx].innerText.trim();
  table.querySelectorAll('th[data-sort]').forEach((th, idx) => {{
    th.addEventListener('click', () => {{
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const asc = th.dataset.asc !== 'true';
      th.dataset.asc = String(asc);
      rows.sort((a, b) => {{
        const av = parseFloat(getCellValue(a, idx).replace(/[^0-9.+-]/g, ''));
        const bv = parseFloat(getCellValue(b, idx).replace(/[^0-9.+-]/g, ''));
        return asc ? av - bv : bv - av;
      }});
      rows.forEach(row => tbody.appendChild(row));
    }});
  }});
  const buttons = document.querySelectorAll('.table-toolbar button');
  buttons.forEach(btn => {{
    btn.addEventListener('click', () => {{
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;
      tbody.querySelectorAll('tr').forEach(row => {{
        let show = filter === 'all';
        if (filter === 'positive' || filter === 'negative') show = row.dataset.label === filter;
        if (filter === 'picked-hawkes') show = row.dataset.pickedHawkes === 'true';
        if (filter === 'picked-cos') show = row.dataset.pickedCos === 'true';
        row.style.display = show ? '' : 'none';
      }});
    }});
  }});
  const first = document.querySelector('.table-toolbar button[data-filter="all"]');
  if (first) first.classList.add('active');
}})();
</script>
</body>
</html>
"""


def parse_top_ks(values: list[int]) -> tuple[int, ...]:
    cleaned = sorted({int(v) for v in values if int(v) > 0})
    return tuple(cleaned or DEFAULT_TOP_KS)


def default_output_path(output_dir: Path, scenario_id: str, embedding: str) -> Path:
    return output_dir / f"trace_{scenario_id}_{embedding}.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one Statistics scenario under Hawkes dynamics."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/statistics"))
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--T-half", type=float, default=1.0, help="T_{1/2} in days.")
    parser.add_argument("--mu-base", type=float, default=0.6)
    parser.add_argument("--inter-top-k", type=int, default=1)
    parser.add_argument("--top-ks", nargs="+", type=int, default=list(DEFAULT_TOP_KS))
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--embedding", choices=["qwen", "bge", "hashing"], default="qwen")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/models"),
        help="sentence-transformers cache; relative paths resolve from repo root "
        "(same default as sweep).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/statistics_hawkes_trace"))
    parser.add_argument("--trace-json", type=Path, default=None)
    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)
    if args.primary_k not in top_ks:
        top_ks = tuple(sorted(set(top_ks + (args.primary_k,))))
    if args.T_half <= 0:
        raise SystemExit("--T-half must be > 0")

    records = load_records(args.data_dir)
    if not records:
        raise SystemExit(f"No scenario JSON files found under {args.data_dir}")
    record = select_record(records, scenario_id=args.scenario_id, category=args.category)

    try:
        embed_fn = make_embedding_fn(
            args.embedding,
            device=args.device,
            cache_dir=args.model_cache_dir if args.embedding != "hashing" else None,
        )
    except ModuleNotFoundError as exc:
        if args.embedding in {"qwen", "bge"} and exc.name == "sentence_transformers":
            raise SystemExit(
                f"--embedding {args.embedding} requires sentence_transformers. "
                "Install that dependency or use --embedding hashing for a dependency-light trace."
            ) from exc
        raise
    beta = math.log(2.0) / args.T_half
    data = build_trace_data(
        record,
        embed_fn,
        eval_index=args.eval_index,
        beta=beta,
        mu_base=args.mu_base,
        inter_top_k=args.inter_top_k,
        top_ks=top_ks,
        primary_k=args.primary_k,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or default_output_path(
        args.output_dir,
        str(data["scenario"]["scenario_id"]),
        args.embedding,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    print(f"[html saved -> {output}]")

    if args.trace_json is not None:
        args.trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.trace_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[trace json saved -> {args.trace_json}]")


if __name__ == "__main__":
    main()
