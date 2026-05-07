from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
from hawkes_rag.core import HawkesParams, simulate_ogata
from hawkes_rag.estimation import fit_full_hawkes


def main() -> None:
    true_params = HawkesParams(
        mu=np.array([0.08, 0.06, 0.05]),
        alpha=np.array(
            [
                [0.35, 0.04, 0.02],
                [0.28, 0.30, 0.03],
                [0.02, 0.04, 0.32],
            ]
        ),
        beta=1.2,
    ).stable()

    trajectories = [
        simulate_ogata(true_params, horizon=250.0, seed=seed)
        for seed in range(8)
    ]
    horizons = [250.0] * len(trajectories)
    result = fit_full_hawkes(
        trajectories,
        horizons,
        n_memories=3,
        max_iter=250,
    )

    print(f"success={result.success} message={result.message}")
    print("true alpha:")
    print(np.round(true_params.alpha, 3))
    print("estimated alpha:")
    print(np.round(result.params.alpha, 3))
    print(f"true beta={true_params.beta:.3f} estimated beta={result.params.beta:.3f}")
    print(f"alpha frobenius error={np.linalg.norm(result.params.alpha - true_params.alpha):.3f}")


if __name__ == "__main__":
    main()
