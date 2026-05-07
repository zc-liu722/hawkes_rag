from __future__ import annotations

from dataclasses import dataclass

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess


@dataclass(frozen=True)
class HeldoutSplit:
    train_events: list[Event]
    test_events: list[Event]
    train_horizon: float
    test_horizon: float
    full_horizon: float
    active_memory_ids: list[int] | None = None


@dataclass(frozen=True)
class PredictiveLogLikelihood:
    total: float
    per_event: float
    n_events: int
    n_trajectories: int


def temporal_train_test_split(
    events: list[Event],
    horizon: float,
    *,
    active_memory_ids: list[int] | None = None,
    train_fraction: float = 0.8,
) -> HeldoutSplit:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    cutoff = float(horizon * train_fraction)
    train = [event for event in events if event.time < cutoff]
    test = [event for event in events if cutoff <= event.time < horizon]
    return HeldoutSplit(
        train_events=train,
        test_events=test,
        train_horizon=cutoff,
        test_horizon=horizon - cutoff,
        full_horizon=horizon,
        active_memory_ids=active_memory_ids,
    )


def heldout_predictive_log_likelihood(
    params: HawkesParams,
    splits: list[HeldoutSplit],
) -> PredictiveLogLikelihood:
    process = MultivariateHawkesProcess(params)
    total = 0.0
    n_events = 0
    for split in splits:
        total += process.conditional_log_likelihood(
            split.test_events,
            start=split.train_horizon,
            end=split.full_horizon,
            initial_history=split.train_events,
            active_memory_ids=split.active_memory_ids,
        )
        n_events += len(split.test_events)
    per_event = total / max(n_events, 1)
    return PredictiveLogLikelihood(
        total=float(total),
        per_event=float(per_event),
        n_events=n_events,
        n_trajectories=len(splits),
    )
