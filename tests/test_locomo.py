from __future__ import annotations

import numpy as np

from benchmarks.locomo.run_locomo import batch_intensities_at_times
from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.estimation import LowRankHawkesEstimator
from hawkes_rag.evaluation import heldout_predictive_log_likelihood
from hawkes_rag.locomo import (
    ConversationMessage,
    LoCoMoEventizer,
    load_official_locomo10_json,
    load_official_locomo10_samples,
)


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


def test_official_locomo10_loader_pins_sessions(tmp_path) -> None:
    path = tmp_path / "locomo10.json"
    path.write_text(
        """
        [
          {
            "sample_id": "sample_0",
            "conversation": {
              "speaker_a": "Alice",
              "speaker_b": "Bob",
              "session_1_date_time": "1 Jan, 2024",
              "session_1": [
                {"speaker": "Alice", "dia_id": "d0", "text": "Alice likes green tea."}
              ],
              "session_2_date_time": "2 Jan, 2024",
              "session_2": [
                {"speaker": "Bob", "dia_id": "d1", "text": "Bob remembers Alice likes tea."}
              ]
            },
            "qa": [
              {
                "question": "What does Alice like?",
                "answer": "green tea",
                "evidence": ["d0"],
                "category": 1
              }
            ]
          }
        ]
        """
    )
    conversations = load_official_locomo10_json(path)
    assert len(conversations) == 1
    assert [message.message_id for message in conversations[0]] == ["d0", "d1"]
    assert conversations[0][1].timestamp > conversations[0][0].timestamp
    assert conversations[0][1].timestamp == 1.0

    samples = load_official_locomo10_samples(path)
    corpus = LoCoMoEventizer().eventize(samples)
    assert corpus.conversations[0].qa_pairs
    assert corpus.conversations[0].qa_pairs[0].question == "What does Alice like?"
    assert corpus.conversations[0].qa_pairs[0].evidence_message_ids == ["d0"]


def test_batch_intensities_at_times_matches_process() -> None:
    params = HawkesParams(
        mu=np.array([0.1, 0.08, 0.06]),
        alpha=np.array(
            [
                [0.3, 0.05, 0.02],
                [0.04, 0.25, 0.03],
                [0.01, 0.06, 0.2],
            ]
        ),
        beta=1.2,
    )
    events = [
        Event(0.2, 0),
        Event(0.8, 1, 0.5),
        Event(1.4, 0),
        Event(1.8, 2),
    ]
    query_times = [0.1, 0.9, 2.0]
    process = MultivariateHawkesProcess(params)
    expected = np.vstack(
        [
            process.intensities(
                query_time,
                [event for event in events if event.time < query_time],
            )
            for query_time in query_times
        ]
    )
    actual = batch_intensities_at_times(params, events, query_times, device="cpu")
    assert np.allclose(actual, expected)
