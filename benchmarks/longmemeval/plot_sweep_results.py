"""Plot sweep results: 3 rows (categories) × 3 cols (heatmap / T_half curves / bar chart)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = Path("outputs/longmemeval_originidea_sweep/sweep_all_multisession_temporalreasoning_knowledgeupdate.json")
OUT_PATH = Path("outputs/longmemeval_originidea_sweep/sweep_heatmap.png")

T_HALFS = [1.0, 5.0, 15.0, 30.0, 50.0]
MU_BASES = [0.0, 0.2, 0.4, 0.6, 0.8]
KS = [1, 3, 5]

CATEGORY_LABELS = {
    "multi-session": "multi-session\n(~8 day span)",
    "temporal-reasoning": "temporal-reasoning\n(~28 day span)",
    "knowledge-update": "knowledge-update",
}


def load_data():
    data = json.loads(JSON_PATH.read_text())
    return data["category_results"]


def build_matrix(category_result, k):
    matrix = np.full((len(MU_BASES), len(T_HALFS)), np.nan)
    for recipe_result in category_result["recipes_results"]:
        r = recipe_result["recipe"]
        if r["name"] == "R0_cosine_baseline":
            continue
        if r["inter_top_k"] != k:
            continue
        th = r["target"].get("T_half_days", 0)
        mu = r["mu_base"]
        if th in T_HALFS and mu in MU_BASES:
            j = T_HALFS.index(th)
            i = MU_BASES.index(mu)
            matrix[i, j] = recipe_result["aggregate"]["session_recall_at_k"]
    return matrix


def find_best(category_result):
    best = None
    best_recall = -1.0
    for recipe_result in category_result["recipes_results"]:
        r = recipe_result["recipe"]
        if r["name"] == "R0_cosine_baseline":
            continue
        rec = recipe_result["aggregate"]["session_recall_at_k"]
        if rec > best_recall:
            best_recall = rec
            best = recipe_result
    return best


def get_cosine(category_result):
    for recipe_result in category_result["recipes_results"]:
        if recipe_result["recipe"]["name"] == "R0_cosine_baseline":
            return recipe_result
    return None


def main():
    category_results = load_data()

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.dpi": 150,
    })

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "OriginIdea Hyperparameter Sweep on LongMemEval-S\n"
        "T_half ∈ {1, 5, 15, 30, 50}d  ×  μ_base ∈ {0.0, 0.2, 0.4, 0.6, 0.8}  ×  k ∈ {1, 3, 5}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    n_rows = len(category_results)
    n_cols = 3

    for row_idx, category_result in enumerate(category_results):
        qtype = category_result["question_type"]
        label = CATEGORY_LABELS.get(qtype, qtype)
        n_q = category_result["config"]["n_questions"]
        cosine = get_cosine(category_result)
        best = find_best(category_result)
        cos_recall = cosine["aggregate"]["session_recall_at_k"]
        best_recall = best["aggregate"]["session_recall_at_k"]
        best_name = best["recipe"]["name"]

        # --- Col 1: Heatmap (k=3 as representative) ---
        ax_hm = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 1)
        matrix = build_matrix(category_result, k=3)
        im = ax_hm.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.80, vmax=0.90)
        ax_hm.set_xticks(range(len(T_HALFS)))
        ax_hm.set_xticklabels([f"{t:g}d" for t in T_HALFS])
        ax_hm.set_yticks(range(len(MU_BASES)))
        ax_hm.set_yticklabels([f"{m:g}" for m in MU_BASES])
        ax_hm.set_xlabel("T_half")
        ax_hm.set_ylabel("μ_base")
        ax_hm.set_title(f"{label}  |  recall@5 (k=3)  |  n={n_q}")

        for i in range(len(MU_BASES)):
            for j in range(len(T_HALFS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.85 else "black"
                    ax_hm.text(j, i, f"{val:.3f}", ha="center", va="center",
                               fontsize=7, color=color, fontweight="bold")

        best_th = best["recipe"]["target"].get("T_half_days", 0)
        best_mu = best["recipe"]["mu_base"]
        if best_th in T_HALFS and best_mu in MU_BASES:
            bj = T_HALFS.index(best_th)
            bi = MU_BASES.index(best_mu)
            ax_hm.plot(bj, bi, marker="*", markersize=16, color="blue",
                       markeredgecolor="white", markeredgewidth=1.0)

        cbar = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("recall@5", fontsize=8)

        # --- Col 2: recall@5 vs T_half curves (one per mu_base, k=3) ---
        ax_line = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 2)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(MU_BASES)))
        for mi, mu in enumerate(MU_BASES):
            recalls = []
            for th in T_HALFS:
                for recipe_result in category_result["recipes_results"]:
                    r = recipe_result["recipe"]
                    if r["name"] == "R0_cosine_baseline":
                        continue
                    tgt = r["target"].get("T_half_days", 0)
                    if tgt == th and r["mu_base"] == mu and r["inter_top_k"] == 3:
                        recalls.append(recipe_result["aggregate"]["session_recall_at_k"])
                        break
                else:
                    recalls.append(np.nan)
            ax_line.plot(T_HALFS, recalls, "o-", color=colors[mi],
                         label=f"μ={mu:g}", markersize=5, linewidth=1.5)

        ax_line.axhline(y=cos_recall, color="gray", linestyle="--", linewidth=1.2,
                        label=f"cosine ({cos_recall:.3f})")
        ax_line.set_xlabel("T_half (days)")
        ax_line.set_ylabel("recall@5")
        ax_line.set_title(f"{label}  |  recall@5 vs T_half (k=3)")
        ax_line.legend(fontsize=7, loc="lower right", ncol=2)
        ax_line.set_xscale("log")
        ax_line.set_xticks(T_HALFS)
        ax_line.set_xticklabels([f"{t:g}d" for t in T_HALFS])
        ax_line.grid(True, alpha=0.3)

        # --- Col 3: Bar chart (cosine vs best originidea) ---
        ax_bar = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + 3)

        cos_m = cosine["aggregate"]["session_metrics"]
        best_m = best["aggregate"]["session_metrics"]

        metrics = ["recall@5", "hit@1", "MRR@10"]
        cos_vals = [cos_recall, cos_m["k1"]["hit"], cos_m["k10"]["mrr"]]
        best_vals = [best_recall, best_m["k1"]["hit"], best_m["k10"]["mrr"]]

        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax_bar.bar(x - width / 2, cos_vals, width, label="cosine",
                           color="#b0b0b0", edgecolor="white")
        bars2 = ax_bar.bar(x + width / 2, best_vals, width, label=f"best: {best_name}",
                           color="#2c7bb6", edgecolor="white")

        for bar, val in zip(bars1, cos_vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, best_vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metrics)
        ax_bar.set_ylabel("score")
        ax_bar.set_title(f"{label}  |  cosine vs best originidea")
        ax_bar.legend(fontsize=7, loc="lower right")
        ax_bar.set_ylim(0.75, 1.0)
        ax_bar.grid(True, alpha=0.3, axis="y")

        delta = best_recall - cos_recall
        ax_bar.text(0.98, 0.92, f"Δ recall = +{delta:.3f}",
                    transform=ax_bar.transAxes, fontsize=9,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[plot] saved to {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()