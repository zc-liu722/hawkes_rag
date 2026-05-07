from __future__ import annotations

import numpy as np

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess, simulate_ogata


def test_intensity_increases_after_related_event() -> None:
    params = HawkesParams(
        mu=np.array([0.1, 0.1]),
        alpha=np.array([[0.5, 0.4], [0.0, 0.5]]),
        beta=1.0,
    )
    process = MultivariateHawkesProcess(params)
    before = process.intensity(0, 1.0, [])
    after = process.intensity(0, 1.0, [Event(time=0.5, memory_id=1)])
    assert after > before


def test_intensity_decays_toward_baseline() -> None:
    params = HawkesParams(
        mu=np.array([0.1]),
        alpha=np.array([[0.8]]),
        beta=1.0,
    )
    process = MultivariateHawkesProcess(params)
    near = process.intensity(0, 1.0, [Event(time=0.9, memory_id=0)])
    far = process.intensity(0, 10.0, [Event(time=0.9, memory_id=0)])
    assert near > far
    assert abs(far - 0.1) < 1e-3


def test_simulation_generates_sorted_events() -> None:
    params = HawkesParams(
        mu=np.array([0.2, 0.15]),
        alpha=np.array([[0.2, 0.05], [0.04, 0.2]]),
        beta=1.0,
    )
    events = simulate_ogata(params, horizon=20.0, seed=7)
    assert events
    assert all(a.time <= b.time for a, b in zip(events, events[1:]))
    assert all(0 <= e.memory_id < 2 for e in events)
