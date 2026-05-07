from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hawkes_rag.utils import project_spectral_radius


@dataclass(frozen=True)
class Event:
    """A memory activation event for a marked point process."""

    time: float
    memory_id: int
    weight: float = 1.0


@dataclass
class HawkesParams:
    """Parameters for an exponential-kernel multivariate Hawkes process."""

    mu: np.ndarray
    alpha: np.ndarray
    beta: float

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.beta = float(self.beta)
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D vector")
        if self.alpha.shape != (self.mu.size, self.mu.size):
            raise ValueError("alpha must have shape (n_memories, n_memories)")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if np.any(self.mu <= 0):
            raise ValueError("all baseline intensities in mu must be positive")
        if np.any(self.alpha < 0):
            raise ValueError("alpha must be non-negative")

    @property
    def n_memories(self) -> int:
        return int(self.mu.size)

    def stable(self, max_radius: float = 0.95) -> "HawkesParams":
        return HawkesParams(
            mu=self.mu.copy(),
            alpha=project_spectral_radius(self.alpha, max_radius=max_radius),
            beta=self.beta,
        )


class MultivariateHawkesProcess:
    """Exponential-kernel MHP over memory activation events.

    The convention is alpha[i, j]: an event on memory j excites memory i.
    """

    def __init__(self, params: HawkesParams):
        self.params = params

    def intensity(self, memory_id: int, time: float, history: Iterable[Event]) -> float:
        return float(self.intensities(time, history)[memory_id])

    def intensities(self, time: float, history: Iterable[Event]) -> np.ndarray:
        lam = self.params.mu.astype(float).copy()
        for event in history:
            if event.time >= time:
                continue
            dt = time - event.time
            lam += self.params.alpha[:, event.memory_id] * event.weight * np.exp(
                -self.params.beta * dt
            )
        return np.maximum(lam, 1e-12)

    def integrated_intensity(self, horizon: float, history: Iterable[Event]) -> float:
        """Compute sum_i int_0^T lambda_i(s) ds for one trajectory."""
        return self.integrated_intensity_interval(0.0, horizon, history)

    def integrated_intensity_interval(
        self,
        start: float,
        end: float,
        history: Iterable[Event],
        active_memory_ids: Iterable[int] | None = None,
    ) -> float:
        """Compute sum_i int_start^end lambda_i(s) ds for one trajectory."""
        if end <= start:
            return 0.0
        if start < 0:
            raise ValueError("start must be non-negative")
        horizon = end
        if horizon <= 0:
            return 0.0
        active = self._active_mask(active_memory_ids)
        total = float(np.sum(self.params.mu[active]) * (end - start))
        alpha_col_sums = np.sum(self.params.alpha[active, :], axis=0)
        for event in history:
            if not (0.0 <= event.time < end):
                continue
            lower = max(start, event.time)
            upper = end
            if upper <= lower:
                continue
            tail = np.exp(-self.params.beta * (lower - event.time)) - np.exp(
                -self.params.beta * (upper - event.time)
            )
            total += float(
                event.weight * alpha_col_sums[event.memory_id] * tail / self.params.beta
            )
        return total

    def log_likelihood(
        self,
        events: Iterable[Event],
        horizon: float,
        *,
        active_memory_ids: Iterable[int] | None = None,
    ) -> float:
        events_sorted = sorted(events, key=lambda e: e.time)
        active = self._active_mask(active_memory_ids)
        self._validate_events_observed(events_sorted, active)
        history: list[Event] = []
        log_terms = 0.0
        for event in events_sorted:
            lam = self.intensity(event.memory_id, event.time, history)
            log_terms += np.log(max(lam, 1e-12))
            history.append(event)
        integral = self.integrated_intensity_interval(
            0.0,
            horizon,
            events_sorted,
            active_memory_ids=np.flatnonzero(active),
        )
        return float(log_terms - integral)

    def conditional_log_likelihood(
        self,
        events: Iterable[Event],
        *,
        start: float,
        end: float,
        initial_history: Iterable[Event] | None = None,
        active_memory_ids: Iterable[int] | None = None,
    ) -> float:
        """Log-likelihood for events in [start, end), conditioned on history.

        Held-out predictive log-likelihood uses this method with training events
        as `initial_history` and the held-out tail as `events`.
        """
        if end <= start:
            return 0.0
        history = sorted(initial_history or [], key=lambda e: e.time)
        test_events = sorted(
            [event for event in events if start <= event.time < end],
            key=lambda e: e.time,
        )
        active = self._active_mask(active_memory_ids)
        self._validate_events_observed(test_events, active)
        log_terms = 0.0
        running_history = list(history)
        for event in test_events:
            lam = self.intensity(event.memory_id, event.time, running_history)
            log_terms += np.log(max(lam, 1e-12))
            running_history.append(event)
        all_history = history + test_events
        integral = self.integrated_intensity_interval(
            start,
            end,
            all_history,
            active_memory_ids=np.flatnonzero(active),
        )
        return float(log_terms - integral)

    def _active_mask(self, active_memory_ids: Iterable[int] | None) -> np.ndarray:
        if active_memory_ids is None:
            return np.ones(self.params.n_memories, dtype=bool)
        mask = np.zeros(self.params.n_memories, dtype=bool)
        for memory_id in active_memory_ids:
            if not (0 <= memory_id < self.params.n_memories):
                raise IndexError(f"memory_id {memory_id} out of range")
            mask[int(memory_id)] = True
        return mask

    @staticmethod
    def _validate_events_observed(events: Iterable[Event], active: np.ndarray) -> None:
        for event in events:
            if not (0 <= event.memory_id < active.size) or not active[event.memory_id]:
                raise ValueError(
                    f"event on memory_id {event.memory_id} is outside active_memory_ids"
                )


def simulate_ogata(
    params: HawkesParams,
    horizon: float,
    *,
    seed: int | None = None,
    max_events: int = 100_000,
) -> list[Event]:
    """Sample a multivariate Hawkes trajectory with Ogata thinning."""
    rng = np.random.default_rng(seed)
    process = MultivariateHawkesProcess(params)
    events: list[Event] = []
    time = 0.0

    while time < horizon and len(events) < max_events:
        current = process.intensities(time + 1e-12, events)
        upper = float(np.sum(current))
        if upper <= 0:
            break
        time += float(rng.exponential(1.0 / upper))
        if time >= horizon:
            break
        proposed = process.intensities(time, events)
        accept_prob = min(1.0, float(np.sum(proposed) / upper))
        if rng.random() <= accept_prob:
            probs = proposed / np.sum(proposed)
            memory_id = int(rng.choice(params.n_memories, p=probs))
            events.append(Event(time=time, memory_id=memory_id))
    return events


def diagonal_only(params: HawkesParams) -> HawkesParams:
    return HawkesParams(
        mu=params.mu.copy(),
        alpha=np.diag(np.diag(params.alpha)),
        beta=params.beta,
    )
