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


def test_vectorized_log_likelihood_matches_naive_history_loop() -> None:
    params = HawkesParams(
        mu=np.array([0.12, 0.09, 0.07]),
        alpha=np.array(
            [
                [0.2, 0.05, 0.01],
                [0.03, 0.18, 0.04],
                [0.02, 0.06, 0.16],
            ]
        ),
        beta=0.8,
    )
    events = [
        Event(time=0.4, memory_id=0, weight=1.0),
        Event(time=0.9, memory_id=1, weight=0.7),
        Event(time=1.4, memory_id=0, weight=1.2),
        Event(time=2.1, memory_id=2, weight=0.8),
    ]
    process = MultivariateHawkesProcess(params)
    vectorized = process.log_likelihood(events, horizon=3.0)

    history: list[Event] = []
    log_terms = 0.0
    for event in events:
        lam = params.mu[event.memory_id]
        for past in history:
            if past.time < event.time:
                lam += (
                    params.alpha[event.memory_id, past.memory_id]
                    * past.weight
                    * np.exp(-params.beta * (event.time - past.time))
                )
        log_terms += np.log(max(lam, 1e-12))
        history.append(event)
    integral = process.integrated_intensity(3.0, events)
    assert np.isclose(vectorized, log_terms - integral)
