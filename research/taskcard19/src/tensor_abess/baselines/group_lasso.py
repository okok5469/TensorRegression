"""Mode-1 group LASSO baseline (block soft-threshold ISTA)."""

from __future__ import annotations

import numpy as np

from tensor_abess.baselines.base import BaselineResult
from tensor_abess.tensor.ops import (
    flatten_predictors,
    mode1_group_indices,
    support_from_tensor,
    tensor_loss,
    unflatten_tensor,
)


def _group_soft_threshold(beta: np.ndarray, groups: np.ndarray, lam: float) -> np.ndarray:
    out = beta.copy()
    for g in np.unique(groups):
        cols = np.where(groups == g)[0]
        norm = np.linalg.norm(out[cols])
        if norm <= lam:
            out[cols] = 0.0
        else:
            out[cols] *= 1.0 - lam / norm
    return out


def group_lasso_mode1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_lambdas: int = 20,
    max_iter: int = 500,
    tol: float = 1e-6,
    random_state: int = 0,
) -> BaselineResult:
    """
    Group LASSO with groups = mode-1 slices (size p2*p3 each).

    Penalty: lam * sum_g ||beta_g||_2. Lambda chosen by hold-out grid on training data.
    """
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape
    Xf = flatten_predictors(X)
    groups = mode1_group_indices(p1, p2, p3)

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    n_val = max(n // 5, 5)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr = Xf[tr_idx], y[tr_idx]
    X_val, y_val = Xf[val_idx], y[val_idx]

    # Spectral step size for ISTA
    lipschitz = np.linalg.norm(X_tr, ord=2) ** 2 / len(tr_idx)
    step = 1.0 / lipschitz

    lam_max = max(
        np.linalg.norm(X_tr[:, np.where(groups == g)[0]].T @ y_tr, ord=2)
        for g in range(p1)
    )
    lambdas = np.logspace(np.log10(lam_max * 0.01), np.log10(lam_max), n_lambdas)

    best_beta = np.zeros(Xf.shape[1])
    best_val = np.inf
    best_lam = lambdas[0]

    for lam in lambdas:
        beta = np.zeros(Xf.shape[1])
        for _ in range(max_iter):
            grad = X_tr.T @ (X_tr @ beta - y_tr) / len(tr_idx)
            beta = _group_soft_threshold(beta - step * grad, groups, lam * step)
        val_loss = np.mean((y_val - X_val @ beta) ** 2) / 2.0
        if val_loss < best_val:
            best_val = val_loss
            best_beta = beta.copy()
            best_lam = lam

    B = unflatten_tensor(best_beta, p1, p2, p3)
    S1, S2, S3 = support_from_tensor(B, tol=1e-8)

    return BaselineResult(
        name=f"group_lasso_mode1(lam={best_lam:.3g})",
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        loss=tensor_loss(X, y, B),
    )
