from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from hawkes_rag.core import HawkesParams, simulate_ogata
from hawkes_rag.estimation import LowRankHawkesEstimator, fit_full_hawkes, topk_similarity_prior


def test_full_mle_smoke_on_synthetic_events() -> None:
    true_params = HawkesParams(
        mu=np.array([0.08, 0.06]),
        alpha=np.array([[0.25, 0.05], [0.2, 0.22]]),
        beta=1.1,
    ).stable()
    trajectories = [simulate_ogata(true_params, 80.0, seed=i) for i in range(3)]
    result = fit_full_hawkes(trajectories, [80.0, 80.0, 80.0], 2, max_iter=20)
    assert np.isfinite(result.objective)
    assert result.params.alpha.shape == (2, 2)
    assert result.params.beta > 0


def test_full_mle_adam_smoke_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    true_params = HawkesParams(
        mu=np.array([0.08, 0.06]),
        alpha=np.array([[0.25, 0.05], [0.2, 0.22]]),
        beta=1.1,
    ).stable()
    trajectories = [simulate_ogata(true_params, 40.0, seed=i) for i in range(2)]
    result = fit_full_hawkes(
        trajectories,
        [40.0, 40.0],
        2,
        max_iter=5,
        optimizer="adam",
        learning_rate=0.02,
        device="cpu",
    )
    assert np.isfinite(result.objective)
    assert result.params.alpha.shape == (2, 2)
    assert result.params.beta > 0


def test_low_rank_estimator_returns_stable_alpha() -> None:
    true_params = HawkesParams(
        mu=np.array([0.08, 0.06, 0.05]),
        alpha=np.array(
            [
                [0.25, 0.08, 0.01],
                [0.20, 0.23, 0.01],
                [0.01, 0.01, 0.20],
            ]
        ),
        beta=1.0,
    ).stable()
    trajectories = [simulate_ogata(true_params, 60.0, seed=i) for i in range(3)]
    estimator = LowRankHawkesEstimator(n_memories=3, rank=2, seed=0)
    result = estimator.fit(trajectories, [60.0, 60.0, 60.0], max_iter=10)
    assert np.isfinite(result.objective)
    assert result.params.alpha.shape == (3, 3)


def test_topk_similarity_prior_can_stay_sparse() -> None:
    embeddings = np.eye(6, dtype=float)
    prior = topk_similarity_prior(embeddings, threshold=-0.1, top_k=2, dense_output=False)
    assert sparse.issparse(prior)
    assert prior.shape == (6, 6)
    assert max(prior.getnnz(axis=1)) <= 2
