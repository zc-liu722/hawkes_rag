from __future__ import annotations

import numpy as np
import pytest

from hawkes_rag.core import Event, HawkesParams
from hawkes_rag.evaluation import heldout_predictive_log_likelihood, temporal_train_test_split


def test_heldout_predictive_log_likelihood_uses_tail_events() -> None:
    params = HawkesParams(
        mu=np.array([0.1, 0.1]),
        alpha=np.array([[0.4, 0.1], [0.2, 0.4]]),
        beta=1.0,
    )
    events = [
        Event(0.2, 0),
        Event(0.8, 1),
        Event(1.4, 0),
        Event(1.8, 1),
    ]
    split = temporal_train_test_split(events, 2.0, active_memory_ids=[0, 1], train_fraction=0.5)
    pll = heldout_predictive_log_likelihood(params, [split])
    assert pll.n_events == 2
    assert np.isfinite(pll.total)
    assert np.isfinite(pll.per_event)


def test_torch_heldout_predictive_log_likelihood_matches_cpu(monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("HAWKES_RAG_TORCH_CHUNK_SIZE", "2")
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
        Event(0.8, 1),
        Event(1.4, 0),
        Event(1.8, 2),
        Event(2.2, 1),
    ]
    split = temporal_train_test_split(events, 2.5, active_memory_ids=[0, 1, 2], train_fraction=0.5)

    cpu_pll = heldout_predictive_log_likelihood(params, [split])
    torch_pll = heldout_predictive_log_likelihood(params, [split], device="cpu")

    assert torch_pll.n_events == cpu_pll.n_events
    assert torch_pll.total == pytest.approx(cpu_pll.total)
    assert torch_pll.per_event == pytest.approx(cpu_pll.per_event)
