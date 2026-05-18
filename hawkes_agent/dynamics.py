from __future__ import annotations

import math

import numpy as np


def compute_mu(lambdas: np.ndarray, mu_base: float) -> float:
    lam2 = np.asarray(lambdas, dtype=float) ** 2
    n = int(lam2.size)
    total = float(lam2.sum())
    if n <= 1 or total <= 0.0:
        h_hat = 0.0
    else:
        p = lam2 / total
        nz = p > 0.0
        h = -float(np.sum(p[nz] * np.log(p[nz])))
        h_hat = h / math.log(n)
    h_hat = min(max(h_hat, 0.0), 1.0)
    return float(mu_base + (1.0 - mu_base) * math.sqrt(1.0 - h_hat))


def decayed_lambda(lambda_plus: float, beta: float, now: float, t_last_event: float) -> float:
    delta_t = max(0.0, float(now) - float(t_last_event))
    value = float(lambda_plus) * math.exp(-float(beta) * delta_t)
    return min(max(value, 0.0), 1.0)


def recall_scores(
    cosines: np.ndarray,
    lambdas_minus: np.ndarray,
    *,
    mu_base: float,
    cosine_floor: float = 0.0,
) -> tuple[np.ndarray, float]:
    mu = compute_mu(lambdas_minus, mu_base)
    cos = np.maximum(np.asarray(cosines, dtype=float), float(cosine_floor))
    lam = np.asarray(lambdas_minus, dtype=float)
    scores = cos * (mu + (1.0 - mu) * lam)
    return scores, mu


def reinforce_lambda(lambda_minus: float, score: float) -> float:
    if score <= 0.0:
        return float(lambda_minus)
    value = float(lambda_minus) + (1.0 - float(lambda_minus)) * float(score)
    return min(max(value, 0.0), 1.0)


def suppress_lambda(lambda_minus: float, contradiction_similarity: float) -> float:
    sc = min(max(float(contradiction_similarity), 0.0), 1.0)
    # Keep the value in the practical closed interval. The formal design uses
    # (0, 1], but storage can tolerate exact 0 for total contradiction.
    return min(max(float(lambda_minus) * (1.0 - sc), 0.0), 1.0)
