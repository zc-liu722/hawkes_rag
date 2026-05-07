from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from hawkes_rag import HawkesMemoryStore
from hawkes_rag.viz import plot_alpha_heatmap, plot_lambda_curve


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    store = HawkesMemoryStore(beta=0.35)
    store.add("The user's dog is named Max.", [1.0, 0.0, 0.0])
    store.add("The user takes Max to the park on Saturdays.", [0.88, 0.12, 0.0])
    store.add("The user likes Python packaging.", [0.0, 1.0, 0.0])
    store.add("The user is preparing a Hawkes-RAG paper.", [0.0, 0.85, 0.1])

    for t in [1, 2, 4, 7, 11]:
        store.record_access(0, time=float(t))
    for t in [3, 8]:
        store.record_mentions([1], time=float(t))
    store.record_access(2, time=5.0)

    plot_alpha_heatmap(
        store.alpha,
        labels=["Max", "Park", "Python", "Paper"],
        path=out / "alpha_heatmap.png",
    )
    plot_lambda_curve(
        store.params(),
        store.events,
        memory_id=1,
        horizon=16.0,
        path=out / "lambda_curve.png",
        title="Memory activation: Saturday park fact",
    )
    print(f"wrote {out / 'alpha_heatmap.png'}")
    print(f"wrote {out / 'lambda_curve.png'}")


if __name__ == "__main__":
    main()
