from __future__ import annotations

import numpy as np

from hawkes_rag.estimation import LowRankHawkesEstimator
from hawkes_rag.evaluation import heldout_predictive_log_likelihood
from hawkes_rag.locomo import ConversationMessage, LoCoMoEventizer


def test_locomo_eventizer_builds_facts_events_and_active_masks() -> None:
    conversations = [
        [
            ConversationMessage("c1", "0", "My dog is named Max.", 0, "user"),
            ConversationMessage("c1", "1", "Max loves Riverside Park on Saturdays.", 1, "user"),
            ConversationMessage("c1", "2", "Please remember that Max loves the park.", 2, "user"),
        ],
        [
            ConversationMessage("c2", "0", "I am writing a Hawkes-RAG paper.", 0, "user"),
            ConversationMessage("c2", "1", "The Hawkes-RAG paper needs a demo.", 1, "user"),
        ],
    ]
    corpus = LoCoMoEventizer().eventize(conversations)
    assert corpus.n_memories >= 2
    assert len(corpus.trajectories()) == 2
    assert len(corpus.active_memory_ids()) == 2
    assert all(conversation.events for conversation in corpus.conversations)
    assert corpus.embeddings().shape[0] == corpus.n_memories


def test_locomo_pooled_likelihood_pipeline_smoke() -> None:
    conversations = [
        [
            ConversationMessage("c1", "0", "My dog is named Max.", 0, "user"),
            ConversationMessage("c1", "1", "Max loves Riverside Park on Saturdays.", 1, "user"),
            ConversationMessage("c1", "2", "Max likes the park.", 2, "user"),
            ConversationMessage("c1", "3", "Max dislikes storms.", 3, "user"),
        ],
        [
            ConversationMessage("c2", "0", "I am writing a Hawkes-RAG paper.", 0, "user"),
            ConversationMessage("c2", "1", "The Hawkes-RAG paper needs a demo.", 1, "user"),
            ConversationMessage("c2", "2", "The demo should show lambda curves.", 2, "user"),
        ],
    ]
    corpus = LoCoMoEventizer().eventize(conversations)
    estimator = LowRankHawkesEstimator.from_embeddings(corpus.embeddings(), rank=2, seed=0)
    fit = estimator.fit(
        corpus.trajectories(),
        corpus.horizons(),
        active_memory_ids=corpus.active_memory_ids(),
        max_iter=5,
    )
    pll = heldout_predictive_log_likelihood(fit.params, corpus.heldout_splits(0.7))
    assert np.isfinite(fit.objective)
    assert np.isfinite(pll.total)
