from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
from hawkes_rag import ConversationMessage, LoCoMoEventizer
from hawkes_rag.estimation import LowRankHawkesEstimator
from hawkes_rag.evaluation import heldout_predictive_log_likelihood


def main() -> None:
    conversations = [
        [
            ConversationMessage("c1", "0", "My dog is named Max.", 0, "user"),
            ConversationMessage("c1", "1", "Max loves Riverside Park on Saturdays.", 1, "user"),
            ConversationMessage("c1", "2", "Can you remember that Max likes the park?", 2, "user"),
            ConversationMessage("c1", "3", "Max should avoid thunderstorms.", 3, "user"),
        ],
        [
            ConversationMessage("c2", "0", "I am writing a Hawkes-RAG paper.", 0, "user"),
            ConversationMessage("c2", "1", "The Hawkes-RAG paper needs a memory demo.", 1, "user"),
            ConversationMessage("c2", "2", "The memory demo should show lambda curves.", 2, "user"),
        ],
    ]
    eventizer = LoCoMoEventizer()
    corpus = eventizer.eventize(conversations)

    estimator = LowRankHawkesEstimator.from_embeddings(
        corpus.embeddings(),
        rank=2,
        seed=0,
    )
    fit = estimator.fit(
        corpus.trajectories(),
        corpus.horizons(),
        active_memory_ids=corpus.active_memory_ids(),
        max_iter=100,
    )
    pll = heldout_predictive_log_likelihood(fit.params, corpus.heldout_splits(0.7))

    print(f"facts={corpus.n_memories}")
    print(f"trajectories={len(corpus.conversations)}")
    print(f"fit_success={fit.success} objective={fit.objective:.3f}")
    print(f"heldout_events={pll.n_events} heldout_pll_per_event={pll.per_event:.3f}")
    print(f"alpha_shape={fit.params.alpha.shape} beta={fit.params.beta:.3f}")
    print(f"alpha_radius={max(abs(np.linalg.eigvals(fit.params.alpha))):.3f}")


if __name__ == "__main__":
    main()
