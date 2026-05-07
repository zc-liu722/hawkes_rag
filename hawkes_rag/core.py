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
        if horizon <= 0:
            return 0.0
        total = float(np.sum(self.params.mu) * horizon)
        alpha_col_sums = np.sum(self.params.alpha, axis=0)
        for event in history:
            if not (0.0 <= event.time < horizon):
                continue
            tail = 1.0 - np.exp(-self.params.beta * (horizon - event.time))
            total += float(event.weight * alpha_col_sums[event.memory_id] * tail / self.params.beta)
        return total

    def log_likelihood(self, events: Iterable[Event], horizon: float) -> float:
        events_sorted = sorted(events, key=lambda e: e.time)
        history: list[Event] = []
        log_terms = 0.0
        for event in events_sorted:
            lam = self.intensity(event.memory_id, event.time, history)
            log_terms += np.log(max(lam, 1e-12))
            history.append(event)
        return float(log_terms - self.integrated_intensity(horizon, events_sorted))


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
