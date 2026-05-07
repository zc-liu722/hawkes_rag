from __future__ import annotations

import numpy as np

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
