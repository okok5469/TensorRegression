"""Simple group-Lasso solvers used by Tensor-Init-GLasso."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def group_soft_threshold(beta: np.ndarray, group_index: np.ndarray, lam: float) -> np.ndarray:
    """Apply block soft-thresholding group by group."""
    out = beta.copy()
    for g in np.unique(group_index):
        cols = np.where(group_index == g)[0]
        norm = np.linalg.norm(out[cols])
        if norm <= lam:
            out[cols] = 0.0
        else:
            out[cols] *= 1.0 - lam / norm
    return out


@dataclass
class GroupLassoFit:
    beta: np.ndarray
    rss: float
    active_groups: np.ndarray
    lam: float


def fit_group_lasso_ista(
    Z: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    lam: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    beta_init: np.ndarray | None = None,
) -> GroupLassoFit:
    """Fit group Lasso with a basic ISTA loop."""
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    group_index = np.asarray(group_index, dtype=int)

    n, p = Z.shape
    if p == 0:
        beta = np.zeros(0, dtype=float)
        residual = y.copy()
        return GroupLassoFit(beta=beta, rss=float(residual @ residual), active_groups=np.array([], dtype=int), lam=lam)

    lipschitz = np.linalg.norm(Z, ord=2) ** 2 / max(n, 1)
    step = 1.0 if lipschitz <= 0 else 1.0 / lipschitz
    beta = np.zeros(p, dtype=float) if beta_init is None else np.asarray(beta_init, dtype=float).copy()

    for _ in range(max_iter):
        grad = Z.T @ (Z @ beta - y) / n
        new_beta = group_soft_threshold(beta - step * grad, group_index, lam * step)
        if np.linalg.norm(new_beta - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = new_beta
            break
        beta = new_beta

    residual = y - Z @ beta
    active_groups = np.array(
        [g for g in np.unique(group_index) if np.linalg.norm(beta[group_index == g]) > 1e-8],
        dtype=int,
    )
    return GroupLassoFit(
        beta=beta,
        rss=float(residual @ residual),
        active_groups=np.sort(active_groups),
        lam=lam,
    )


def _lambda_grid(Z: np.ndarray, y: np.ndarray, group_index: np.ndarray, n_lambdas: int) -> np.ndarray:
    """Geometric lambda grid from lambda_max down to 1e-2 * lambda_max."""
    lam_max = 0.0
    for g in np.unique(group_index):
        cols = np.where(group_index == g)[0]
        lam_max = max(lam_max, float(np.linalg.norm(Z[:, cols].T @ y / max(len(y), 1))))
    if lam_max <= 0:
        return np.array([0.0])
    lam_min = lam_max * 1e-2
    return np.logspace(np.log10(lam_max), np.log10(lam_min), num=max(n_lambdas, 2))


def select_group_lasso_lambda_bic(
    Z: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    block_size: int,
    *,
    n_lambdas: int = 20,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> GroupLassoFit:
    """Select lambda by BIC = n log(RSS/n) + df log n with df = active_groups * block_size."""
    y = np.asarray(y, dtype=float).ravel()
    lambdas = _lambda_grid(Z, y, group_index, n_lambdas)

    best_fit: GroupLassoFit | None = None
    best_bic = np.inf
    beta_warm: np.ndarray | None = None
    n = max(len(y), 1)
    eps = np.finfo(float).tiny
    for lam in lambdas:
        fit = fit_group_lasso_ista(
            Z, y, group_index, float(lam), max_iter=max_iter, tol=tol, beta_init=beta_warm
        )
        beta_warm = fit.beta
        df = len(fit.active_groups) * block_size
        bic = n * np.log(max(fit.rss / n, eps)) + df * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_fit = fit
    assert best_fit is not None
    return best_fit


def select_group_lasso_lambda_cv(
    Z: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    *,
    n_lambdas: int = 20,
    folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> GroupLassoFit:
    """Select lambda by simple K-fold CV on predictive RSS."""
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    group_index = np.asarray(group_index, dtype=int)
    n = len(y)
    lambdas = _lambda_grid(Z, y, group_index, n_lambdas)
    if n == 0:
        return GroupLassoFit(beta=np.zeros(Z.shape[1]), rss=0.0, active_groups=np.array([], dtype=int), lam=0.0)

    folds = max(2, min(folds, n))
    indices = np.arange(n)
    split = np.array_split(indices, folds)

    best_lam = float(lambdas[0])
    best_score = np.inf
    for lam in lambdas:
        score = 0.0
        for val_idx in split:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            fit = fit_group_lasso_ista(
                Z[train_mask],
                y[train_mask],
                group_index,
                float(lam),
                max_iter=max_iter,
                tol=tol,
            )
            residual = y[val_idx] - Z[val_idx] @ fit.beta
            score += float(residual @ residual)
        if score < best_score:
            best_score = score
            best_lam = float(lam)

    return fit_group_lasso_ista(
        Z, y, group_index, best_lam, max_iter=max_iter, tol=tol
    )
