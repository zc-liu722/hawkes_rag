#!/usr/bin/env python3
"""Parse statistics sweep Markdown summaries and render a dashboard figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    cells = [c.strip() for c in line.strip("|").split("|")]
    return cells


def parse_markdown_table(lines: list[str], start_idx: int) -> tuple[list[str], list[list[str]], int]:
    """Parse a pipe table starting at start_idx. Returns header, rows, next line index."""
    if start_idx >= len(lines):
        return [], [], start_idx
    header = _split_table_row(lines[start_idx])
    if not header:
        return [], [], start_idx + 1
    sep = _split_table_row(lines[start_idx + 1]) if start_idx + 1 < len(lines) else []
    if not sep or not all(re.match(r"^:?-{3,}:?$", c.replace(" ", "")) for c in sep if c):
        return [], [], start_idx + 1
    rows: list[list[str]] = []
    j = start_idx + 2
    while j < len(lines):
        row = _split_table_row(lines[j])
        if not row:
            break
        rows.append(row)
        j += 1
    return header, rows, j


def find_table_after_heading(lines: list[str], heading: str) -> tuple[list[str], list[list[str]]] | None:
    for i, ln in enumerate(lines):
        if ln.strip() == heading:
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1
            if j >= len(lines):
                return None
            h, r, _ = parse_markdown_table(lines, j)
            if h and r:
                return h, r
    return None


def parse_per_category_mechanism(lines: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """category -> mechanism -> metric -> value from '## Per-Category Best By Mechanism' section."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "## Per-Category Best By Mechanism":
            start = i
            break
    if start is None:
        return out

    i = start + 1
    current_cat: str | None = None
    while i < len(lines):
        if lines[i].startswith("## ") and i > start:
            break
        m = re.match(r"^###\s+(.+)$", lines[i].strip())
        if m:
            current_cat = m.group(1).strip()
            out.setdefault(current_cat, {})
            i += 1
            continue
        if current_cat and lines[i].strip().startswith("|"):
            h, rows, ni = parse_markdown_table(lines, i)
            i = ni
            if not h or not rows:
                continue
            idx = {name: h.index(name) for name in h if name in h}
            for row in rows:
                if len(row) < len(h):
                    continue
                mech = row[h.index("mechanism")].strip()
                rec: dict[str, float] = {}
                for col in (
                    "SRR@10",
                    "positive_recall@10",
                    "negative_intrusion@10",
                    "pair_win",
                    "rank_margin",
                ):
                    if col not in idx:
                        continue
                    try:
                        rec[col] = float(row[idx[col]].strip())
                    except ValueError:
                        pass
                out[current_cat][mech] = rec
            continue
        i += 1
    return out


def parse_hawkes_grid(lines: list[str]) -> list[dict[str, str | float]]:
    res = find_table_after_heading(lines, "## Hawkes Grid")
    if not res:
        return []
    h, rows = res
    idx = {name: h.index(name) for name in h}
    float_keys = (
        "T_half",
        "mu_base",
        "inter_top_k",
        "SRR@K",
        "positive_recall@K",
        "negative_intrusion@K",
        "pair_win",
        "rank_margin",
    )
    records: list[dict[str, str | float]] = []
    for row in rows:
        if len(row) < len(h):
            continue
        rec: dict[str, str | float] = {}
        if "recipe" in idx:
            rec["recipe"] = row[idx["recipe"]].strip()
        for key in float_keys:
            if key not in idx:
                continue
            raw = row[idx[key]].strip()
            try:
                rec[key] = float(raw)
            except ValueError:
                rec[key] = raw
        if "SRR@K" in rec and isinstance(rec["SRR@K"], float):
            records.append(rec)
    records.sort(key=lambda x: float(x["SRR@K"]), reverse=True)
    return records


def _ordered_categories(per_cat: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    preferred = [
        "decay_forget",
        "reactivation",
        "semantic_distractor",
        "stability_check",
        "update_override",
    ]
    return [c for c in preferred if c in per_cat] + sorted(c for c in per_cat.keys() if c not in preferred)


def _heatmap_panel(
    ax,
    categories: list[str],
    mechanisms: list[str],
    per_cat: dict[str, dict[str, dict[str, float]]],
    metric_key: str,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str,
    *,
    text_contrast: bool = False,
) -> None:
    mat = np.full((len(categories), len(mechanisms)), np.nan)
    for ci, cat in enumerate(categories):
        for mi, mech in enumerate(mechanisms):
            cell = per_cat.get(cat, {}).get(mech, {})
            if metric_key in cell:
                mat[ci, mi] = cell[metric_key]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=22, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ci in range(mat.shape[0]):
        for mi in range(mat.shape[1]):
            v = mat[ci, mi]
            if np.isnan(v):
                continue
            if text_contrast and vmax > vmin:
                t = (v - vmin) / (vmax - vmin)
                use_white = t < 0.25 or t > 0.75
                tc = "white" if use_white else "#111"
            else:
                tc = "#111"
            ax.text(mi, ci, format(v, fmt), ha="center", va="center", fontsize=7, color=tc)


def render_dashboard(
    md_path: Path,
    out_path: Path,
) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    best_cmp = find_table_after_heading(lines, "## Best Mechanism Comparison")
    qsum = find_table_after_heading(lines, "## Question Category Summary")
    per_cat = parse_per_category_mechanism(lines)
    hawkes = parse_hawkes_grid(lines)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#fafafa",
            "savefig.facecolor": "#fafafa",
        }
    )

    if per_cat:
        fig = plt.figure(figsize=(16, 22))
        gs = fig.add_gridspec(5, 2, height_ratios=[1.85, 1.15, 1.05, 0.7, 0.95])
        hawkes_row = 4
    else:
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0])
        hawkes_row = 1

    mechanism_order = ["cosine", "recency", "cosine_recency", "hawkes"]
    mech_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # --- Row 0 left: per-metric bars for global best mechanism comparison ---
    gs_me = gs[0, 0].subgridspec(5, 1, hspace=0.40)
    if best_cmp:
        h, rows = best_cmp
        by_mech: dict[str, dict[str, str]] = {}
        for row in rows:
            d = dict(zip(h, row))
            mech = d.get("mechanism", "").strip()
            by_mech[mech] = d

        def _f(d: dict[str, str], key: str) -> float:
            try:
                return float(d.get(key, "nan"))
            except (TypeError, ValueError):
                return float("nan")

        metric_specs: list[tuple[str, str]] = [
            ("SRR@K", "SRR@K (global best recipe)"),
            ("positive_recall@K", "positive_recall@K"),
            ("negative_intrusion@K", "negative_intrusion@K (lower is better)"),
            ("pair_win", "pair_win"),
            ("rank_margin", "rank_margin"),
        ]
        for row_i, (col_key, title) in enumerate(metric_specs):
            axm = fig.add_subplot(gs_me[row_i])
            vals = []
            for m in mechanism_order:
                if m in by_mech:
                    vals.append(_f(by_mech[m], col_key))
                else:
                    vals.append(float("nan"))
            x = np.arange(len(mechanism_order))
            bars = axm.bar(
                x,
                vals,
                color=mech_colors[: len(mechanism_order)],
                edgecolor="#333",
                linewidth=0.55,
            )
            axm.set_xticks(x)
            axm.set_xticklabels(mechanism_order, rotation=16, ha="right", fontsize=8)
            axm.set_title(title, fontsize=10)
            axm.axhline(0, color="#666", linewidth=0.75)
            if row_i == 0:
                for b, m in zip(bars, mechanism_order):
                    wtl = (by_mech.get(m, {}) or {}).get("vs cosine W/T/L", "")
                    if not wtl:
                        continue
                    hgt = b.get_height()
                    if np.isnan(hgt):
                        continue
                    axm.text(
                        b.get_x() + b.get_width() / 2,
                        hgt + 0.015 * (1 if hgt >= 0 else -1),
                        str(wtl),
                        ha="center",
                        va="bottom" if hgt >= 0 else "top",
                        fontsize=6,
                        color="#333",
                    )

    # --- Row 0 right: per-category winner, 4 metrics grouped ---
    ax_cat = fig.add_subplot(gs[0, 1])
    if qsum:
        h, rows = qsum
        cats: list[str] = []
        metrics_block: dict[str, list[float]] = {
            "SRR@10": [],
            "positive_recall@10": [],
            "negative_intrusion@10": [],
            "pair_win": [],
        }
        mech_lab: list[str] = []
        for row in rows:
            d = dict(zip(h, row))
            cats.append(d.get("category", "").strip())
            mech_lab.append(d.get("mechanism", "").strip())
            for k in metrics_block:
                try:
                    metrics_block[k].append(float(d.get(k, "nan")))
                except (TypeError, ValueError):
                    metrics_block[k].append(float("nan"))

        y = np.arange(len(cats))
        n_m = len(metrics_block)
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * 0.22
        bar_colors = ["#5975a4", "#9ecae1", "#fdae6b", "#66c2a5"]
        for o, (mname, bcol) in zip(offsets, zip(metrics_block.keys(), bar_colors)):
            ax_cat.barh(y + o, metrics_block[mname], height=0.2, label=mname, color=bcol, edgecolor="#222", linewidth=0.35)
        ax_cat.set_yticks(y)
        ax_cat.set_yticklabels([f"{c}\n({m})" for c, m in zip(cats, mech_lab)], fontsize=8)
        ax_cat.invert_yaxis()
        ax_cat.set_xlabel("value")
        ax_cat.set_title("Per-category best recipe — multiple metrics")
        ax_cat.axvline(0, color="#666", linewidth=0.75)
        ax_cat.legend(loc="lower right", fontsize=7, ncol=2)

    # --- Row 1–2: 2×2 heatmaps (category × mechanism) ---
    if per_cat:
        categories = _ordered_categories(per_cat)
        mechanisms = mechanism_order

        panels = [
            (gs[1, 0], "SRR@10", "SRR@10", -0.35, 0.95, "RdYlGn", ".2f"),
            (gs[1, 1], "positive_recall@10", "positive_recall@10", 0.0, 1.0, "YlGn", ".2f"),
            (gs[2, 0], "negative_intrusion@10", "negative_intrusion@10\n(lower is better)", 0.0, 1.0, "YlOrRd", ".2f"),
            (gs[2, 1], "pair_win", "pair_win", 0.0, 1.0, "Blues", ".2f"),
        ]
        for cell, key, title, vmin, vmax, cmap, fmt in panels:
            ax_h = fig.add_subplot(cell)
            _heatmap_panel(ax_h, categories, mechanisms, per_cat, key, title, vmin, vmax, cmap, fmt)

        ax_rm_h = fig.add_subplot(gs[3, :])
        _heatmap_panel(
            ax_rm_h,
            categories,
            mechanisms,
            per_cat,
            "rank_margin",
            "rank_margin@10 (category × mechanism)",
            -15.0,
            15.0,
            "coolwarm",
            ".1f",
            text_contrast=True,
        )

    # --- Hawkes sweep — SRR + rank_margin + companion metrics ---
    gs_h = gs[hawkes_row, :].subgridspec(1, 2, wspace=0.22)
    ax_h1 = fig.add_subplot(gs_h[0, 0])
    ax_h2 = fig.add_subplot(gs_h[0, 1])
    if hawkes:
        top_n = 14
        top = hawkes[:top_n]
        labels = [str(r["recipe"]).replace("R3_hawkes_", "") for r in top]
        x = np.arange(len(top))

        srr = [float(r["SRR@K"]) for r in top]
        ax_h1.bar(x, srr, color="#c44e52", edgecolor="#222", linewidth=0.45, label="SRR@K", zorder=2)
        ax_h1.set_xticks(x)
        ax_h1.set_xticklabels(labels, rotation=58, ha="right", fontsize=7)
        ax_h1.set_ylabel("SRR@K")
        ax_h1.set_title(f"Hawkes grid — top {top_n} by SRR@K (+ rank_margin)")
        ax_h1.axhline(0, color="#666", linewidth=0.65)
        ax_h1.grid(axis="y", linestyle=":", alpha=0.35)

        ax_rm = ax_h1.twinx()
        rms = []
        for r in top:
            v = r.get("rank_margin", float("nan"))
            rms.append(float(v) if isinstance(v, (int, float)) else float("nan"))
        ax_rm.plot(x, rms, color="#7b3294", marker="o", markersize=4, linewidth=1.1, label="rank_margin")
        ax_rm.set_ylabel("rank_margin", color="#7b3294")
        ax_rm.tick_params(axis="y", labelcolor="#7b3294")
        h1, l1 = ax_h1.get_legend_handles_labels()
        h2, l2 = ax_rm.get_legend_handles_labels()
        ax_h1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)

        # Grouped bars: same recipes
        pr = [float(r.get("positive_recall@K", np.nan)) for r in top]
        ni = [float(r.get("negative_intrusion@K", np.nan)) for r in top]
        pw = [float(r.get("pair_win", np.nan)) for r in top]
        w = 0.26
        ax_h2.bar(x - w, pr, w, label="positive_recall@K", color="#4c72b0", edgecolor="#222", linewidth=0.35)
        ax_h2.bar(x, ni, w, label="negative_intrusion@K", color="#dd8452", edgecolor="#222", linewidth=0.35)
        ax_h2.bar(x + w, pw, w, label="pair_win", color="#55a868", edgecolor="#222", linewidth=0.35)
        ax_h2.set_xticks(x)
        ax_h2.set_xticklabels(labels, rotation=58, ha="right", fontsize=7)
        ax_h2.set_ylim(0, 1.05)
        ax_h2.set_ylabel("rate / fraction")
        ax_h2.set_title("Same recipes — recall / intrusion / pair_win @K")
        ax_h2.legend(loc="upper right", fontsize=7)
        ax_h2.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(f"Sweep summary (multi-metric): {md_path.name}", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "markdown",
        type=Path,
        nargs="?",
        default=Path("outputs/statistics_hawkes_sweep/sweep_n5_per_category_qwen3.md"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: alongside markdown)",
    )
    args = ap.parse_args()
    md = args.markdown
    out = args.output
    if out is None:
        out = md.with_suffix(".dashboard.png")
    render_dashboard(md, out)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
