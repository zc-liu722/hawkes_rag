from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from hawkes_rag import HawkesMemoryStore
from hawkes_rag.retrieval import naive_retrieve
from hawkes_rag.viz import plot_alpha_heatmap, plot_lambda_curve


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    store = HawkesMemoryStore(beta=0.28)
    store.add("The user's dog is named Max.", [1.0, 0.0, 0.0])
    store.add("The user walks Max in Riverside Park.", [0.92, 0.08, 0.0])
    store.add("Max dislikes thunderstorms.", [0.9, 0.0, 0.1])
    store.add("The user mentioned Python packaging once.", [0.0, 1.0, 0.0])

    query = np.array([0.86, 0.1, 0.0])
    rows = []
    for turn in range(1, 51):
        if turn in {3, 8, 14, 27, 39}:
            store.record_mentions([0], time=float(turn), weight=0.3)
        if turn in {10, 28}:
            store.record_mentions([1], time=float(turn), weight=0.3)
        if turn == 21:
            store.record_mentions([3], time=float(turn), weight=0.3)
        hawkes = store.retrieve(query, top_k=4, time=float(turn) + 0.01, record_event=False)
        naive = naive_retrieve(store, query, top_k=4)
        rows.append(
            {
                "turn": turn,
                "naive_top": naive[0].memory.content,
                "hawkes_top": hawkes[0].memory.content,
                "max_lambda": store.intensities(float(turn) + 0.01)[0],
                "python_lambda": store.intensities(float(turn) + 0.01)[3],
            }
        )

    with (out / "naive_vs_hawkes_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_alpha_heatmap(
        store.alpha,
        labels=["Max", "Park", "Storm", "Python"],
        path=out / "demo_alpha_heatmap.png",
    )
    plot_lambda_curve(
        store.params(),
        store.events,
        memory_id=0,
        horizon=52.0,
        path=out / "demo_lambda_curve.png",
        title="Max memory strengthens when repeatedly mentioned",
    )
    print(f"wrote {out / 'naive_vs_hawkes_scores.csv'}")
    print(f"wrote {out / 'demo_alpha_heatmap.png'}")
    print(f"wrote {out / 'demo_lambda_curve.png'}")


if __name__ == "__main__":
    main()
