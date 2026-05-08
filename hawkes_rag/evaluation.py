from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
from scipy import sparse

from hawkes_rag.core import Event, HawkesParams, MultivariateHawkesProcess
from hawkes_rag.gpu import best_float_dtype, resolve_torch_device


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
    *,
    device: str | None = "auto",
    label: str = "heldout_pll",
) -> PredictiveLogLikelihood:
    if device is not None:
        try:
            return _heldout_predictive_log_likelihood_torch(
                params,
                splits,
                device=device,
                label=label,
            )
        except ImportError:
            pass
    process = MultivariateHawkesProcess(params)
    total = 0.0
    n_events = 0
    started = time.perf_counter()
    for index, split in enumerate(splits, start=1):
        total += process.conditional_log_likelihood(
            split.test_events,
            start=split.train_horizon,
            end=split.full_horizon,
            initial_history=split.train_events,
            active_memory_ids=split.active_memory_ids,
        )
        n_events += len(split.test_events)
        print(
            f"[{label}] split={index}/{len(splits)} events={len(split.test_events)} "
            f"cumulative_events={n_events} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    per_event = total / max(n_events, 1)
    return PredictiveLogLikelihood(
        total=float(total),
        per_event=float(per_event),
        n_events=n_events,
        n_trajectories=len(splits),
    )


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        value = default
    return max(minimum, value)


def _likelihood_chunk_size(n_events: int) -> int:
    chunk_size = _env_int("HAWKES_RAG_TORCH_CHUNK_SIZE", 1024)
    return max(1, min(chunk_size, n_events))


def _alpha_to_numpy(alpha) -> np.ndarray:
    return alpha.toarray() if sparse.issparse(alpha) else np.asarray(alpha, dtype=float)


def _heldout_predictive_log_likelihood_torch(
    params: HawkesParams,
    splits: list[HeldoutSplit],
    *,
    device: str,
    label: str,
) -> PredictiveLogLikelihood:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for GPU heldout PLL evaluation.") from exc

    torch_device = resolve_torch_device(device)
    dtype = best_float_dtype(torch, torch_device)
    mu = torch.as_tensor(params.mu, dtype=dtype, device=torch_device)
    alpha = torch.as_tensor(_alpha_to_numpy(params.alpha), dtype=dtype, device=torch_device)
    beta = torch.tensor(float(params.beta), dtype=dtype, device=torch_device)
    total = torch.zeros((), dtype=dtype, device=torch_device)
    n_events = 0
    started = time.perf_counter()
    for index, split in enumerate(splits, start=1):
        split_value = _conditional_log_likelihood_torch(
            torch,
            mu,
            alpha,
            beta,
            split,
            torch_device,
            dtype,
        )
        total = total + split_value
        n_events += len(split.test_events)
        print(
            f"[{label}] split={index}/{len(splits)} events={len(split.test_events)} "
            f"cumulative_events={n_events} split_total={float(split_value.detach().cpu()):.6f} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    total_value = float(total.detach().cpu())
    per_event = total_value / max(n_events, 1)
    return PredictiveLogLikelihood(
        total=total_value,
        per_event=float(per_event),
        n_events=n_events,
        n_trajectories=len(splits),
    )


def _conditional_log_likelihood_torch(
    torch,
    mu,
    alpha,
    beta,
    split: HeldoutSplit,
    device,
    dtype,
):
    if split.full_horizon <= split.train_horizon:
        return torch.zeros((), dtype=dtype, device=device)
    history = sorted(split.train_events, key=lambda event: event.time)
    test_events = sorted(
        [event for event in split.test_events if split.train_horizon <= event.time < split.full_horizon],
        key=lambda event: event.time,
    )
    all_events = history + test_events
    if all_events:
        times = torch.as_tensor([event.time for event in all_events], dtype=dtype, device=device)
        memory_ids = torch.as_tensor(
            [event.memory_id for event in all_events],
            dtype=torch.long,
            device=device,
        )
        weights = torch.as_tensor([event.weight for event in all_events], dtype=dtype, device=device)
    else:
        times = torch.empty((0,), dtype=dtype, device=device)
        memory_ids = torch.empty((0,), dtype=torch.long, device=device)
        weights = torch.empty((0,), dtype=dtype, device=device)
    n_history = len(history)
    n_test = len(test_events)
    if n_test:
        test_times = times[n_history:]
        test_ids = memory_ids[n_history:]
        chunk_size = _likelihood_chunk_size(n_test)
        log_terms = torch.zeros((), dtype=dtype, device=device)
        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            chunk_times = test_times[start:end]
            chunk_ids = test_ids[start:end]
            dt = chunk_times[:, None] - times[None, :]
            past = dt > 0.0
            decay = torch.exp(-beta * torch.clamp(dt, min=0.0)) * past.to(dtype)
            alpha_for_events = alpha[chunk_ids[:, None], memory_ids[None, :]]
            excitation = torch.sum(alpha_for_events * decay * weights[None, :], dim=1)
            lam = mu[chunk_ids] + excitation
            log_terms = log_terms + torch.sum(torch.log(torch.clamp(lam, min=1e-12)))
    else:
        log_terms = torch.zeros((), dtype=dtype, device=device)

    if split.active_memory_ids is None:
        active_ids = torch.arange(mu.shape[0], dtype=torch.long, device=device)
    else:
        active_ids = torch.as_tensor(split.active_memory_ids, dtype=torch.long, device=device)
    interval = float(split.full_horizon - split.train_horizon)
    integral = torch.sum(mu[active_ids]) * interval
    if times.numel():
        alpha_col_sums = torch.sum(alpha[active_ids, :], dim=0)
        start_t = torch.tensor(float(split.train_horizon), dtype=dtype, device=device)
        end_t = torch.tensor(float(split.full_horizon), dtype=dtype, device=device)
        lower = torch.maximum(start_t, times)
        valid = end_t > lower
        if torch.any(valid):
            tail = torch.exp(-beta * (lower[valid] - times[valid])) - torch.exp(
                -beta * (end_t - times[valid])
            )
            integral = integral + torch.sum(
                weights[valid] * alpha_col_sums[memory_ids[valid]] * tail
            ) / beta
    return log_terms - integral
