from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.utils import pairwise_cosine, project_spectral_radius


@dataclass
class FitResult:
    params: HawkesParams
    success: bool
    objective: float
    message: str
    n_iter: int


class LowRankHawkesEstimator:
    """MLE for an MHP with alpha = softplus(U V^T + gamma S + d I).

    This keeps the interaction matrix expressive without learning N^2 free
    parameters. The learned dense alpha is projected to a stable spectral
    radius before likelihood evaluation.
    """

    def __init__(
        self,
        n_memories: int,
        *,
        rank: int = 2,
        max_radius: float = 0.95,
        similarity_prior: np.ndarray | None = None,
        learn_beta: bool = True,
        seed: int | None = 0,
    ):
        if n_memories <= 0:
            raise ValueError("n_memories must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.n_memories = int(n_memories)
        self.rank = int(rank)
        self.max_radius = float(max_radius)
        self.learn_beta = bool(learn_beta)
        self.rng = np.random.default_rng(seed)
        if similarity_prior is None:
            similarity_prior = np.zeros((n_memories, n_memories), dtype=float)
        self.similarity_prior = np.asarray(similarity_prior, dtype=float)
        if self.similarity_prior.shape != (n_memories, n_memories):
            raise ValueError("similarity_prior has the wrong shape")

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        *,
        rank: int = 2,
        max_radius: float = 0.95,
        learn_beta: bool = True,
        seed: int | None = 0,
        threshold: float = 0.3,
    ) -> "LowRankHawkesEstimator":
        similarities = np.maximum(0.0, pairwise_cosine(embeddings) - threshold)
        np.fill_diagonal(similarities, 0.0)
        return cls(
            n_memories=similarities.shape[0],
            rank=rank,
            max_radius=max_radius,
            similarity_prior=similarities,
            learn_beta=learn_beta,
            seed=seed,
        )

    def fit(
        self,
        trajectories: list[list[Event]],
        horizons: list[float],
        *,
        active_memory_ids: list[list[int]] | None = None,
        max_iter: int = 200,
    ) -> FitResult:
        if len(trajectories) != len(horizons):
            raise ValueError("trajectories and horizons must have the same length")
        if active_memory_ids is None:
            active_memory_ids = [list(range(self.n_memories)) for _ in trajectories]
        if len(active_memory_ids) != len(trajectories):
            raise ValueError("active_memory_ids must match trajectories length")
        x0 = self._initial_vector()

        def objective(x: np.ndarray) -> float:
            params = self.unpack(x)
            value = 0.0
            for events, horizon, active in zip(trajectories, horizons, active_memory_ids):
                value -= MultivariateHawkesProcess(params).log_likelihood(
                    events,
                    horizon,
                    active_memory_ids=active,
                )
            if not np.isfinite(value):
                return 1e100
            return float(value)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "maxls": 30},
        )
        return FitResult(
            params=self.unpack(result.x),
            success=bool(result.success),
            objective=float(result.fun),
            message=str(result.message),
            n_iter=int(result.nit),
        )

    def unpack(self, x: np.ndarray) -> HawkesParams:
        n = self.n_memories
        r = self.rank
        idx = 0
        raw_mu = x[idx : idx + n]
        idx += n
        u = x[idx : idx + n * r].reshape(n, r)
        idx += n * r
        v = x[idx : idx + n * r].reshape(n, r)
        idx += n * r
        gamma = x[idx]
        idx += 1
        diagonal_bias = x[idx]
        idx += 1
        if self.learn_beta:
            raw_beta = x[idx]
        else:
            raw_beta = np.log(np.exp(1.0) - 1.0)

        mu = softplus(raw_mu) + 1e-5
        raw_alpha = u @ v.T + gamma * self.similarity_prior
        raw_alpha = raw_alpha + np.eye(n) * diagonal_bias
        alpha = softplus(raw_alpha)
        alpha = project_spectral_radius(alpha, max_radius=self.max_radius)
        beta = float(softplus(raw_beta) + 1e-5)
        return HawkesParams(mu=mu, alpha=alpha, beta=beta)

    def _initial_vector(self) -> np.ndarray:
        n = self.n_memories
        r = self.rank
        raw_mu = np.full(n, inverse_softplus(0.05), dtype=float)
        u = self.rng.normal(0.0, 0.05, size=(n, r))
        v = self.rng.normal(0.0, 0.05, size=(n, r))
        gamma = np.array([0.5])
        diagonal_bias = np.array([inverse_softplus(0.5)])
        if self.learn_beta:
            raw_beta = np.array([inverse_softplus(1.0)])
            return np.concatenate([raw_mu, u.ravel(), v.ravel(), gamma, diagonal_bias, raw_beta])
        return np.concatenate([raw_mu, u.ravel(), v.ravel(), gamma, diagonal_bias])


def fit_full_hawkes(
    trajectories: list[list[Event]],
    horizons: list[float],
    n_memories: int,
    *,
    active_memory_ids: list[list[int]] | None = None,
    max_radius: float = 0.95,
    learn_beta: bool = True,
    max_iter: int = 200,
) -> FitResult:
    """Small-N full alpha MLE, intended for synthetic recovery tests."""
    if active_memory_ids is None:
        active_memory_ids = [list(range(n_memories)) for _ in trajectories]
    if len(active_memory_ids) != len(trajectories):
        raise ValueError("active_memory_ids must match trajectories length")

    def unpack(x: np.ndarray) -> HawkesParams:
        idx = 0
        raw_mu = x[idx : idx + n_memories]
        idx += n_memories
        raw_alpha = x[idx : idx + n_memories * n_memories].reshape(n_memories, n_memories)
        idx += n_memories * n_memories
        if learn_beta:
            raw_beta = x[idx]
        else:
            raw_beta = inverse_softplus(1.0)
        mu = softplus(raw_mu) + 1e-5
        alpha = project_spectral_radius(softplus(raw_alpha), max_radius=max_radius)
        beta = float(softplus(raw_beta) + 1e-5)
        return HawkesParams(mu=mu, alpha=alpha, beta=beta)

    def objective(x: np.ndarray) -> float:
        params = unpack(x)
        value = 0.0
        for events, horizon, active in zip(trajectories, horizons, active_memory_ids):
            value -= MultivariateHawkesProcess(params).log_likelihood(
                events,
                horizon,
                active_memory_ids=active,
            )
        return float(value) if np.isfinite(value) else 1e100

    raw_mu = np.full(n_memories, inverse_softplus(0.05), dtype=float)
    raw_alpha = np.full((n_memories, n_memories), inverse_softplus(0.05), dtype=float)
    raw_beta = np.array([inverse_softplus(1.0)])
    x0 = np.concatenate([raw_mu, raw_alpha.ravel(), raw_beta if learn_beta else np.array([])])
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxls": 30},
    )
    return FitResult(
        params=unpack(result.x),
        success=bool(result.success),
        objective=float(result.fun),
        message=str(result.message),
        n_iter=int(result.nit),
    )


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0)
    if np.isscalar(x):
        return float(out)
    return out


def inverse_softplus(y: float) -> float:
    y = float(max(y, 1e-12))
    if y > 20:
        return y
    return float(np.log(np.expm1(y)))
