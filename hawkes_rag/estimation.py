from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
from scipy.optimize import minimize
from scipy import sparse

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.gpu import (
    adaptive_cuda_chunk_size,
    best_float_dtype,
    resolve_torch_device,
    torch_spectral_radius,
)
from hawkes_rag.utils import pairwise_cosine, project_spectral_radius


@dataclass
class FitResult:
    params: HawkesParams
    success: bool
    objective: float
    message: str
    n_iter: int


def _make_objective_callback(label: str, objective):
    state = {"iteration": 0}

    def callback(xk: np.ndarray) -> None:
        state["iteration"] += 1
        value = objective(xk)
        print(f"[{label}] iter={state['iteration']} objective={value:.6f}", flush=True)

    return callback


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        value = default
    return max(minimum, value)


def _adam_log_every() -> int:
    return _env_int("HAWKES_RAG_ADAM_LOG_EVERY", 1)


def _log_adam_progress(
    label: str,
    *,
    iteration: int,
    max_iter: int,
    objective: float,
    best_objective: float,
    started: float,
) -> None:
    elapsed = time.perf_counter() - started
    print(
        f"[{label}] iter={iteration}/{max_iter} "
        f"objective={objective:.6f} best={best_objective:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )


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
        dense_threshold: int = 4000,
        optimizer: str = "lbfgsb",
        learning_rate: float = 0.05,
        device: str | None = None,
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
        self.dense_threshold = int(dense_threshold)
        self.optimizer = optimizer.lower()
        self.learning_rate = float(learning_rate)
        self.device = device
        self.rng = np.random.default_rng(seed)
        if similarity_prior is None:
            similarity_prior = np.zeros((n_memories, n_memories), dtype=float)
        if sparse.issparse(similarity_prior):
            self.similarity_prior = similarity_prior.tocsr().astype(float)
        else:
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
        top_k: int = 32,
        dense_threshold: int = 4000,
        optimizer: str = "lbfgsb",
        learning_rate: float = 0.05,
        device: str | None = None,
    ) -> "LowRankHawkesEstimator":
        similarities = topk_similarity_prior(
            embeddings,
            threshold=threshold,
            top_k=top_k,
            dense_output=embeddings.shape[0] <= dense_threshold,
            device=device,
        )
        return cls(
            n_memories=similarities.shape[0],
            rank=rank,
            max_radius=max_radius,
            similarity_prior=similarities,
            learn_beta=learn_beta,
            dense_threshold=dense_threshold,
            optimizer=optimizer,
            learning_rate=learning_rate,
            device=device,
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
        if self.optimizer in {"adam", "torch", "pytorch"}:
            if self.n_memories > self.dense_threshold:
                return self._fit_local_active_sets(
                    trajectories,
                    horizons,
                    active_memory_ids=active_memory_ids,
                    max_iter=max_iter,
                )
            return self._fit_torch_adam(
                trajectories,
                horizons,
                active_memory_ids=active_memory_ids,
                max_iter=max_iter,
            )
        if self.optimizer not in {"lbfgsb", "l-bfgs-b", "scipy"}:
            raise ValueError("optimizer must be one of 'lbfgsb' or 'adam'")
        if self.n_memories > self.dense_threshold:
            return self._fit_local_active_sets(
                trajectories,
                horizons,
                active_memory_ids=active_memory_ids,
                max_iter=max_iter,
            )
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
            callback=_make_objective_callback("low_rank_mle", objective),
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
        similarity_prior = (
            self.similarity_prior.toarray()
            if sparse.issparse(self.similarity_prior)
            else self.similarity_prior
        )
        raw_alpha = u @ v.T + gamma * similarity_prior
        raw_alpha = raw_alpha + np.eye(n) * diagonal_bias
        alpha = softplus(raw_alpha)
        alpha = project_spectral_radius(alpha, max_radius=self.max_radius)
        beta = float(softplus(raw_beta) + 1e-5)
        return HawkesParams(mu=mu, alpha=alpha, beta=beta)

    def _fit_local_active_sets(
        self,
        trajectories: list[list[Event]],
        horizons: list[float],
        *,
        active_memory_ids: list[list[int]],
        max_iter: int,
    ) -> FitResult:
        mu = np.full(self.n_memories, 1e-5, dtype=float)
        alpha = sparse.lil_matrix((self.n_memories, self.n_memories), dtype=float)
        beta_values = []
        objectives = []
        successes = []
        messages = []
        n_iter = 0
        total = len(trajectories)
        for index, (events, horizon, active) in enumerate(
            zip(trajectories, horizons, active_memory_ids)
        ):
            active = sorted(set(int(memory_id) for memory_id in active))
            if not active:
                continue
            print(
                f"[local_active_mle] trajectory={index + 1}/{total} "
                f"events={len(events)} active_memories={len(active)} horizon={horizon:.3f}",
                flush=True,
            )
            if len(active) > self.dense_threshold:
                raise ValueError(
                    "conversation active set is too large for local dense MLE; "
                    "lower fact extraction density or raise dense_threshold"
                )
            remap = {memory_id: local_id for local_id, memory_id in enumerate(active)}
            local_events = [
                Event(time=event.time, memory_id=remap[event.memory_id], weight=event.weight)
                for event in events
                if event.memory_id in remap
            ]
            local_prior = self.similarity_prior[active, :][:, active]
            local_estimator = LowRankHawkesEstimator(
                len(active),
                rank=self.rank,
                max_radius=self.max_radius,
                similarity_prior=local_prior,
                learn_beta=self.learn_beta,
                dense_threshold=self.dense_threshold,
                optimizer=self.optimizer,
                learning_rate=self.learning_rate,
                device=self.device,
                seed=int(self.rng.integers(0, 2**32 - 1)),
            )
            fit = local_estimator.fit([local_events], [horizon], max_iter=max_iter)
            print(
                f"[local_active_mle] trajectory={index + 1}/{total} done "
                f"n_iter={fit.n_iter} objective={fit.objective:.6f}",
                flush=True,
            )
            local_alpha = (
                fit.params.alpha.toarray() if sparse.issparse(fit.params.alpha) else fit.params.alpha
            )
            mu[active] = fit.params.mu
            alpha[np.ix_(active, active)] = local_alpha
            beta_values.append(fit.params.beta)
            objectives.append(fit.objective)
            successes.append(fit.success)
            messages.append(f"trajectory_{index}: {fit.message}")
            n_iter += fit.n_iter
        beta = float(np.mean(beta_values)) if beta_values else 1.0
        params = HawkesParams(mu=mu, alpha=alpha.tocsr(), beta=beta)
        return FitResult(
            params=params,
            success=all(successes) if successes else True,
            objective=float(np.sum(objectives)) if objectives else 0.0,
            message="; ".join(messages) if messages else "no active trajectories",
            n_iter=int(n_iter),
        )

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

    def _fit_torch_adam(
        self,
        trajectories: list[list[Event]],
        horizons: list[float],
        *,
        active_memory_ids: list[list[int]],
        max_iter: int,
    ) -> FitResult:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "LowRankHawkesEstimator(..., optimizer='adam') requires PyTorch. "
                "Install hawkes-rag[torch] first."
            ) from exc

        torch_device = resolve_torch_device(self.device)
        dtype = best_float_dtype(torch, torch_device)
        n = self.n_memories
        r = self.rank
        raw_mu = torch.full(
            (n,),
            inverse_softplus(0.05),
            dtype=dtype,
            device=torch_device,
            requires_grad=True,
        )
        u = torch.as_tensor(
            self.rng.normal(0.0, 0.05, size=(n, r)),
            dtype=dtype,
            device=torch_device,
        ).requires_grad_()
        v = torch.as_tensor(
            self.rng.normal(0.0, 0.05, size=(n, r)),
            dtype=dtype,
            device=torch_device,
        ).requires_grad_()
        gamma = torch.tensor(0.5, dtype=dtype, device=torch_device, requires_grad=True)
        diagonal_bias = torch.tensor(
            inverse_softplus(0.5),
            dtype=dtype,
            device=torch_device,
            requires_grad=True,
        )
        parameters = [raw_mu, u, v, gamma, diagonal_bias]
        if self.learn_beta:
            raw_beta = torch.tensor(
                inverse_softplus(1.0),
                dtype=dtype,
                device=torch_device,
                requires_grad=True,
            )
            parameters.append(raw_beta)
        else:
            raw_beta = torch.tensor(inverse_softplus(1.0), dtype=dtype, device=torch_device)
        prior_np = (
            self.similarity_prior.toarray()
            if sparse.issparse(self.similarity_prior)
            else self.similarity_prior
        )
        similarity_prior = torch.as_tensor(prior_np, dtype=dtype, device=torch_device)
        eye = torch.eye(n, dtype=dtype, device=torch_device)
        prepared = [
            _prepare_torch_trajectory(torch, events, horizon, active, n, torch_device, dtype)
            for events, horizon, active in zip(trajectories, horizons, active_memory_ids)
        ]
        adam = torch.optim.Adam(parameters, lr=self.learning_rate)
        best_state: dict[str, torch.Tensor] | None = None
        best_objective = float("inf")
        completed_iters = 0
        log_every = _adam_log_every()
        started = time.perf_counter()
        for iteration in range(max_iter):
            adam.zero_grad()
            mu, alpha, beta = _unpack_low_rank_torch(
                torch,
                raw_mu,
                u,
                v,
                gamma,
                diagonal_bias,
                raw_beta,
                similarity_prior,
                eye,
                max_radius=self.max_radius,
            )
            objective = torch.zeros((), dtype=dtype, device=torch_device)
            for trajectory in prepared:
                objective = objective - _torch_log_likelihood(torch, mu, alpha, beta, trajectory)
            if not torch.isfinite(objective):
                break
            objective.backward()
            adam.step()
            completed_iters = iteration + 1
            objective_value = float(objective.detach().cpu())
            if objective_value < best_objective:
                best_objective = objective_value
                best_state = {
                    "raw_mu": raw_mu.detach().clone(),
                    "u": u.detach().clone(),
                    "v": v.detach().clone(),
                    "gamma": gamma.detach().clone(),
                    "diagonal_bias": diagonal_bias.detach().clone(),
                    "raw_beta": raw_beta.detach().clone(),
                }
            if completed_iters == 1 or completed_iters == max_iter or completed_iters % log_every == 0:
                _log_adam_progress(
                    "low_rank_adam",
                    iteration=completed_iters,
                    max_iter=max_iter,
                    objective=objective_value,
                    best_objective=best_objective,
                    started=started,
                )
        if best_state is not None:
            with torch.no_grad():
                raw_mu.copy_(best_state["raw_mu"])
                u.copy_(best_state["u"])
                v.copy_(best_state["v"])
                gamma.copy_(best_state["gamma"])
                diagonal_bias.copy_(best_state["diagonal_bias"])
                if self.learn_beta:
                    raw_beta.copy_(best_state["raw_beta"])
        mu_t, alpha_t, beta_t = _unpack_low_rank_torch(
            torch,
            raw_mu,
            u,
            v,
            gamma,
            diagonal_bias,
            raw_beta,
            similarity_prior,
            eye,
            max_radius=self.max_radius,
        )
        params = HawkesParams(
            mu=mu_t.detach().cpu().numpy(),
            alpha=alpha_t.detach().cpu().numpy(),
            beta=float(beta_t.detach().cpu()),
        )
        return FitResult(
            params=params,
            success=bool(np.isfinite(best_objective)),
            objective=float(best_objective),
            message=f"Adam low-rank MLE finished on {torch_device}",
            n_iter=int(completed_iters),
        )


def fit_full_hawkes(
    trajectories: list[list[Event]],
    horizons: list[float],
    n_memories: int,
    *,
    active_memory_ids: list[list[int]] | None = None,
    max_radius: float = 0.95,
    learn_beta: bool = True,
    max_iter: int = 200,
    optimizer: str = "lbfgsb",
    learning_rate: float = 0.05,
    device: str | None = None,
) -> FitResult:
    """Small-N full alpha MLE, intended for synthetic recovery tests."""
    if active_memory_ids is None:
        active_memory_ids = [list(range(n_memories)) for _ in trajectories]
    if len(active_memory_ids) != len(trajectories):
        raise ValueError("active_memory_ids must match trajectories length")
    optimizer = optimizer.lower()
    if optimizer in {"adam", "torch", "pytorch"}:
        return _fit_full_hawkes_torch(
            trajectories,
            horizons,
            n_memories,
            active_memory_ids=active_memory_ids,
            max_radius=max_radius,
            learn_beta=learn_beta,
            max_iter=max_iter,
            learning_rate=learning_rate,
            device=device,
        )
    if optimizer not in {"lbfgsb", "l-bfgs-b", "scipy"}:
        raise ValueError("optimizer must be one of 'lbfgsb' or 'adam'")

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
        callback=_make_objective_callback("full_hawkes_mle", objective),
        options={"maxiter": max_iter, "maxls": 30},
    )
    return FitResult(
        params=unpack(result.x),
        success=bool(result.success),
        objective=float(result.fun),
        message=str(result.message),
        n_iter=int(result.nit),
    )


def _fit_full_hawkes_torch(
    trajectories: list[list[Event]],
    horizons: list[float],
    n_memories: int,
    *,
    active_memory_ids: list[list[int]],
    max_radius: float,
    learn_beta: bool,
    max_iter: int,
    learning_rate: float,
    device: str | None,
) -> FitResult:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "fit_full_hawkes(..., optimizer='adam') requires PyTorch. "
            "Install the optional torch dependency first."
        ) from exc

    torch_device = resolve_torch_device(device)
    dtype = best_float_dtype(torch, torch_device)
    raw_mu = torch.full(
        (n_memories,),
        inverse_softplus(0.05),
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    raw_alpha = torch.full(
        (n_memories, n_memories),
        inverse_softplus(0.05),
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    parameters = [raw_mu, raw_alpha]
    if learn_beta:
        raw_beta = torch.tensor(
            inverse_softplus(1.0),
            dtype=dtype,
            device=torch_device,
            requires_grad=True,
        )
        parameters.append(raw_beta)
    else:
        raw_beta = torch.tensor(inverse_softplus(1.0), dtype=dtype, device=torch_device)

    prepared = [
        _prepare_torch_trajectory(torch, events, horizon, active, n_memories, torch_device, dtype)
        for events, horizon, active in zip(trajectories, horizons, active_memory_ids)
    ]
    adam = torch.optim.Adam(parameters, lr=learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    completed_iters = 0
    log_every = _adam_log_every()
    started = time.perf_counter()

    for iteration in range(max_iter):
        adam.zero_grad()
        mu, alpha, beta = _unpack_full_hawkes_torch(
            torch,
            raw_mu,
            raw_alpha,
            raw_beta,
            max_radius=max_radius,
        )
        objective = torch.zeros((), dtype=dtype, device=torch_device)
        for trajectory in prepared:
            objective = objective - _torch_log_likelihood(torch, mu, alpha, beta, trajectory)
        if not torch.isfinite(objective):
            break
        objective.backward()
        adam.step()
        completed_iters = iteration + 1
        objective_value = float(objective.detach().cpu())
        if objective_value < best_objective:
            best_objective = objective_value
            best_state = {
                "raw_mu": raw_mu.detach().clone(),
                "raw_alpha": raw_alpha.detach().clone(),
                "raw_beta": raw_beta.detach().clone(),
            }
        if completed_iters == 1 or completed_iters == max_iter or completed_iters % log_every == 0:
            _log_adam_progress(
                "full_hawkes_adam",
                iteration=completed_iters,
                max_iter=max_iter,
                objective=objective_value,
                best_objective=best_objective,
                started=started,
            )

    if best_state is not None:
        with torch.no_grad():
            raw_mu.copy_(best_state["raw_mu"])
            raw_alpha.copy_(best_state["raw_alpha"])
            if learn_beta:
                raw_beta.copy_(best_state["raw_beta"])
    mu_t, alpha_t, beta_t = _unpack_full_hawkes_torch(
        torch,
        raw_mu,
        raw_alpha,
        raw_beta,
        max_radius=max_radius,
    )
    params = HawkesParams(
        mu=mu_t.detach().cpu().numpy(),
        alpha=alpha_t.detach().cpu().numpy(),
        beta=float(beta_t.detach().cpu()),
    )
    return FitResult(
        params=params,
        success=bool(np.isfinite(best_objective)),
        objective=float(best_objective),
        message=f"Adam finished on {torch_device}",
        n_iter=int(completed_iters),
    )


def _prepare_torch_trajectory(
    torch,
    events: list[Event],
    horizon: float,
    active_memory_ids: list[int],
    n_memories: int,
    device,
    dtype,
) -> dict[str, object]:
    events_sorted = sorted(events, key=lambda event: event.time)
    active = np.zeros(n_memories, dtype=bool)
    active[np.asarray(active_memory_ids, dtype=int)] = True
    for event in events_sorted:
        if not (0 <= event.memory_id < n_memories) or not active[event.memory_id]:
            raise ValueError(f"event on memory_id {event.memory_id} is outside active_memory_ids")
    times_np = np.asarray([event.time for event in events_sorted], dtype=float)
    memory_ids_np = np.asarray([event.memory_id for event in events_sorted], dtype=int)
    weights_np = np.asarray([event.weight for event in events_sorted], dtype=float)
    return {
        "times": torch.as_tensor(times_np, dtype=dtype, device=device),
        "memory_ids": torch.as_tensor(memory_ids_np, dtype=torch.long, device=device),
        "weights": torch.as_tensor(weights_np, dtype=dtype, device=device),
        "horizon": torch.tensor(float(horizon), dtype=dtype, device=device),
        "active": torch.as_tensor(active, dtype=torch.bool, device=device),
    }


def _unpack_full_hawkes_torch(torch, raw_mu, raw_alpha, raw_beta, *, max_radius: float):
    mu = torch.nn.functional.softplus(raw_mu) + 1e-5
    alpha = torch.nn.functional.softplus(raw_alpha)
    if alpha.numel():
        radius = torch_spectral_radius(torch, alpha)
        max_radius_t = torch.as_tensor(max_radius, dtype=alpha.dtype, device=alpha.device)
        scale = torch.clamp(max_radius_t / radius.clamp_min(1e-12), max=1.0)
        alpha = alpha * scale
    beta = torch.nn.functional.softplus(raw_beta) + 1e-5
    return mu, alpha, beta


def _unpack_low_rank_torch(
    torch,
    raw_mu,
    u,
    v,
    gamma,
    diagonal_bias,
    raw_beta,
    similarity_prior,
    eye,
    *,
    max_radius: float,
):
    mu = torch.nn.functional.softplus(raw_mu) + 1e-5
    raw_alpha = u @ v.T + gamma * similarity_prior + diagonal_bias * eye
    alpha = torch.nn.functional.softplus(raw_alpha)
    if alpha.numel():
        radius = torch_spectral_radius(torch, alpha)
        max_radius_t = torch.as_tensor(max_radius, dtype=alpha.dtype, device=alpha.device)
        scale = torch.clamp(max_radius_t / radius.clamp_min(1e-12), max=1.0)
        alpha = alpha * scale
    beta = torch.nn.functional.softplus(raw_beta) + 1e-5
    return mu, alpha, beta


def _torch_likelihood_chunk_size(torch, device, dtype, n_events: int) -> int:
    chunk_size = _env_int("HAWKES_RAG_TORCH_CHUNK_SIZE", 1024)
    chunk_size = adaptive_cuda_chunk_size(
        torch,
        device,
        dtype,
        n_events,
        preferred=chunk_size,
        minimum=32,
    )
    return max(1, min(chunk_size, n_events))


def _torch_log_likelihood(torch, mu, alpha, beta, trajectory: dict[str, object]):
    times = trajectory["times"]
    memory_ids = trajectory["memory_ids"]
    weights = trajectory["weights"]
    horizon = trajectory["horizon"]
    active = trajectory["active"]
    if times.numel() == 0:
        log_terms = torch.zeros((), dtype=mu.dtype, device=mu.device)
    else:
        n_events = int(times.numel())
        chunk_size = _torch_likelihood_chunk_size(torch, mu.device, mu.dtype, n_events)
        log_terms = torch.zeros((), dtype=mu.dtype, device=mu.device)
        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            chunk_times = times[start:end]
            chunk_ids = memory_ids[start:end]
            dt = chunk_times[:, None] - times[None, :]
            past = dt > 0.0
            decay = torch.exp(-beta * torch.clamp(dt, min=0.0)) * past.to(mu.dtype)
            alpha_for_events = alpha[chunk_ids[:, None], memory_ids[None, :]]
            excitation = torch.sum(alpha_for_events * decay * weights[None, :], dim=1)
            lam = mu[chunk_ids] + excitation
            log_terms = log_terms + torch.sum(torch.log(torch.clamp(lam, min=1e-12)))
    integral = torch.sum(mu[active]) * horizon
    if times.numel():
        alpha_col_sums = torch.sum(alpha[active, :], dim=0)
        tail = 1.0 - torch.exp(-beta * (horizon - times))
        integral = integral + torch.sum(weights * alpha_col_sums[memory_ids] * tail) / beta
    return log_terms - integral


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


def topk_similarity_prior(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.3,
    top_k: int = 32,
    dense_output: bool = False,
    device: str | None = None,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if dense_output:
        similarities = np.maximum(0.0, pairwise_cosine(embeddings, device=device) - threshold)
        np.fill_diagonal(similarities, 0.0)
        if top_k > 0 and top_k < n - 1:
            for row in range(n):
                keep = np.argpartition(similarities[row], -top_k)[-top_k:]
                mask = np.ones(n, dtype=bool)
                mask[keep] = False
                similarities[row, mask] = 0.0
        return similarities

    top_k = max(0, min(int(top_k), max(n - 1, 0)))
    if device is not None and device.lower() != "cpu":
        try:
            return _topk_similarity_prior_torch(
                embeddings,
                threshold=threshold,
                top_k=top_k,
                device=device,
            )
        except ImportError:
            pass
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    rows = []
    cols = []
    values = []
    for row in range(n):
        sims = normalized @ normalized[row]
        sims[row] = -np.inf
        if top_k == 0:
            continue
        candidate_count = min(top_k, n - 1)
        candidate_ids = np.argpartition(sims, -candidate_count)[-candidate_count:]
        for col in candidate_ids:
            value = float(sims[col] - threshold)
            if value > 0.0:
                rows.append(row)
                cols.append(int(col))
                values.append(value)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=float)


def _topk_similarity_prior_torch(
    embeddings: np.ndarray,
    *,
    threshold: float,
    top_k: int,
    device: str | None,
) -> sparse.csr_matrix:
    import torch

    n = embeddings.shape[0]
    if top_k == 0:
        return sparse.csr_matrix((n, n), dtype=float)
    torch_device = resolve_torch_device(device)
    tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=torch_device)
    normalized = torch.nn.functional.normalize(tensor, p=2, dim=1, eps=1e-12)
    rows = []
    cols = []
    values = []
    candidate_count = min(top_k, n - 1)
    for row in range(n):
        sims = normalized @ normalized[row]
        sims[row] = -torch.inf
        candidate_values, candidate_ids = torch.topk(sims, k=candidate_count)
        candidate_values = candidate_values.detach().cpu().numpy()
        candidate_ids = candidate_ids.detach().cpu().numpy()
        for col, sim in zip(candidate_ids, candidate_values):
            value = float(sim - threshold)
            if value > 0.0:
                rows.append(row)
                cols.append(int(col))
                values.append(value)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=float)
