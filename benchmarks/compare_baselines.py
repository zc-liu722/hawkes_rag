from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawkes_rag.core import Event, HawkesParams, diagonal_only
from hawkes_rag.estimation import fit_full_hawkes
from hawkes_rag.evaluation import heldout_predictive_log_likelihood, temporal_train_test_split
from hawkes_rag.memory import HawkesMemoryStore
from hawkes_rag.retrieval import diagonal_hawkes_retrieve, naive_retrieve


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    mrr: float


PAIR_EDGES = [(0, 1), (2, 3), (4, 5)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare naive, diagonal Hawkes, and full-alpha Hawkes baselines."
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=160)
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    embeddings = synthetic_embeddings(seed=args.seed)
    trajectories, horizons = synthetic_cross_excitation_corpus()
    splits = [temporal_train_test_split(events, horizon, train_fraction=0.72) for events, horizon in zip(trajectories, horizons)]

    fit = fit_full_hawkes(
        [split.train_events for split in splits],
        [split.train_horizon for split in splits],
        n_memories=embeddings.shape[0],
        max_iter=args.max_iter,
        learn_beta=False,
    )
    full_params = fit.params
    diagonal_params = diagonal_only(full_params)
    naive_params = HawkesParams(
        mu=estimate_mu([split.train_events for split in splits], [split.train_horizon for split in splits], embeddings.shape[0]),
        alpha=np.zeros_like(full_params.alpha),
        beta=full_params.beta,
    )

    rows = [
        (
            "naive_retrieve",
            retrieval_metrics("naive", naive_params, embeddings, splits),
            heldout_predictive_log_likelihood(naive_params, splits),
        ),
        (
            "diagonal_hawkes_retrieve",
            retrieval_metrics("diagonal", full_params, embeddings, splits),
            heldout_predictive_log_likelihood(diagonal_params, splits),
        ),
        (
            "full_alpha_hawkes_retrieve",
            retrieval_metrics("full", full_params, embeddings, splits),
            heldout_predictive_log_likelihood(full_params, splits),
        ),
    ]

    payload = {
        "corpus": {
            "n_memories": int(embeddings.shape[0]),
            "n_trajectories": len(trajectories),
            "n_train_events": int(sum(len(split.train_events) for split in splits)),
            "n_heldout_events": int(sum(len(split.test_events) for split in splits)),
            "synthetic_pattern": "cue memory j is followed by associated memory i; full alpha can learn alpha[i,j]",
        },
        "fit": {
            "success": fit.success,
            "objective": fit.objective,
            "message": fit.message,
            "n_iter": fit.n_iter,
        },
        "rows": [
            {
                "retriever": name,
                "recall_at_1": metrics.recall_at_1,
                "recall_at_3": metrics.recall_at_3,
                "mrr": metrics.mrr,
                "heldout_pll_per_event": pll.per_event,
                "heldout_pll_total": pll.total,
                "heldout_events": pll.n_events,
            }
            for name, metrics, pll in rows
        ],
    }
    (args.outputs_dir / "baseline_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = format_markdown_table(rows)
    (args.outputs_dir / "baseline_comparison.md").write_text(markdown + "\n")
    print(markdown)


def synthetic_embeddings(*, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embeddings = []
    for _cue, _target in PAIR_EDGES:
        cue = rng.normal(size=24)
        cue /= np.linalg.norm(cue)
        target = cue * 0.91 + rng.normal(scale=0.08, size=24)
        target /= np.linalg.norm(target)
        embeddings.extend([cue, target])
    return np.vstack(embeddings)


def synthetic_cross_excitation_corpus() -> tuple[list[list[Event]], list[float]]:
    trajectories: list[list[Event]] = []
    horizons: list[float] = []
    for trajectory_id in range(5):
        events: list[Event] = []
        t = 0.25 + trajectory_id * 0.03
        for repeat in range(18):
            cue, target = PAIR_EDGES[(repeat + trajectory_id) % len(PAIR_EDGES)]
            events.append(Event(t, cue))
            events.append(Event(t + 0.08, target))
            t += 0.42
        trajectories.append(events)
        horizons.append(t + 0.5)
    return trajectories, horizons


def estimate_mu(trajectories: list[list[Event]], horizons: list[float], n_memories: int) -> np.ndarray:
    counts = np.full(n_memories, 0.5, dtype=float)
    for events in trajectories:
        for event in events:
            counts[event.memory_id] += event.weight
    horizon = max(float(sum(horizons)), 1e-6)
    return np.maximum(counts / horizon, 1e-4)


def retrieval_metrics(
    baseline: str,
    params: HawkesParams,
    embeddings: np.ndarray,
    splits,
) -> RetrievalMetrics:
    reciprocal_ranks = []
    recall_1 = 0
    recall_3 = 0
    n = 0
    for split_index, split in enumerate(splits):
        for pair_index, (cue, target) in enumerate(PAIR_EDGES):
            probe_time = split.train_horizon + 0.05 + pair_index * 0.02
            events = list(split.train_events) + [Event(probe_time - 0.01, cue)]
            store = build_store(params, embeddings, events)
            query = embeddings[cue]
            if baseline == "naive":
                results = naive_retrieve(store, query, top_k=embeddings.shape[0])
            elif baseline == "diagonal":
                results = diagonal_hawkes_retrieve(
                    store,
                    query,
                    top_k=embeddings.shape[0],
                    time=probe_time,
                )
            else:
                results = store.retrieve(
                    query,
                    top_k=embeddings.shape[0],
                    time=probe_time,
                    record_event=False,
                )
            ranked_ids = [result.memory.id for result in results]
            rank = ranked_ids.index(target) + 1 if target in ranked_ids else len(ranked_ids) + 1
            reciprocal_ranks.append(1.0 / rank)
            recall_1 += int(rank <= 1)
            recall_3 += int(rank <= 3)
            n += 1
    return RetrievalMetrics(
        recall_at_1=recall_1 / n,
        recall_at_3=recall_3 / n,
        mrr=float(np.mean(reciprocal_ranks)),
    )


def build_store(params: HawkesParams, embeddings: np.ndarray, events: list[Event]) -> HawkesMemoryStore:
    store = HawkesMemoryStore(beta=params.beta)
    for memory_id, embedding in enumerate(embeddings):
        store.add(
            f"synthetic memory {memory_id}",
            embedding,
            created_at=0.0,
            base_intensity=float(params.mu[memory_id]),
        )
    store.set_params(params)
    store.events = sorted(events, key=lambda e: e.time)
    return store


def format_markdown_table(rows) -> str:
    lines = [
        "| Retriever | Recall@1 | Recall@3 | MRR | Held-out PLL/event |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics, pll in rows:
        lines.append(
            f"| `{name}` | {metrics.recall_at_1:.3f} | {metrics.recall_at_3:.3f} | "
            f"{metrics.mrr:.3f} | {pll.per_event:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
