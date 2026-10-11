"""Group-level OLS utilities, sacrifices, and thresholds."""

from __future__ import annotations

import numpy as np

from tensor_abess.vector.sacrifices import squared_loss


def columns_for_groups(group_index: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Column indices belonging to the given groups."""
    mask = np.isin(group_index, groups)
    return np.where(mask)[0]


def _group_gram(X: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Return Gamma_g = Z_g^T Z_g / n."""
    return X[:, cols].T @ X[:, cols] / n


def _group_corr(X: np.ndarray, residual: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Return d_g = Z_g^T residual / n."""
    return X[:, cols].T @ residual / n


def _solve_psd(Gamma: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Gamma x = rhs with pseudoinverse fallback for singular blocks."""
    try:
        return np.linalg.solve(Gamma, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Gamma) @ rhs


def fit_ols_on_groups(
    y: np.ndarray,
    X: np.ndarray,
    group_index: np.ndarray,
    active_groups: np.ndarray,
) -> tuple[np.ndarray, float]:
    """OLS with entire groups active; returns (beta, loss)."""
    p = X.shape[1]
    beta = np.zeros(p, dtype=float)
    active_cols = columns_for_groups(group_index, active_groups)
    if active_cols.size == 0:
        return beta, squared_loss(y, X, beta)

    coef, _, _, _ = np.linalg.lstsq(X[:, active_cols], y, rcond=None)
    beta[active_cols] = coef
    return beta, squared_loss(y, X, beta)


def compute_group_sacrifices(
    y: np.ndarray,
    X: np.ndarray,
    group_index: np.ndarray,
    active_groups: np.ndarray,
    inactive_groups: np.ndarray,
    current_loss: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group backward xi_g and forward zeta_g sacrifices from PDF §4.

    xi_g = 1/2 * beta_g^T Gamma_g beta_g
    zeta_g = 1/2 * d_g^T Gamma_g^{-1} d_g
    """
    del current_loss  # kept for backward-compatible call sites

    n_groups = int(group_index.max()) + 1
    n = y.shape[0]
    xi = np.zeros(n_groups, dtype=float)
    zeta = np.zeros(n_groups, dtype=float)

    beta, _ = fit_ols_on_groups(y, X, group_index, active_groups)
    residual = y - X @ beta

    for g in active_groups:
        cols = np.where(group_index == g)[0]
        Gamma_g = _group_gram(X, cols, n)
        beta_g = beta[cols]
        xi[g] = 0.5 * float(beta_g @ Gamma_g @ beta_g)

    for g in inactive_groups:
        cols = np.where(group_index == g)[0]
        Gamma_g = _group_gram(X, cols, n)
        d_g = _group_corr(X, residual, cols, n)
        zeta[g] = 0.5 * float(d_g @ _solve_psd(Gamma_g, d_g))

    return xi, zeta


def initial_active_groups(
    X: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    s: int,
) -> np.ndarray:
    """Initialize active groups by the Gram-standardized score from PDF §4.6."""
    n_groups = int(group_index.max()) + 1
    n = y.shape[0]
    scores = np.zeros(n_groups, dtype=float)
    for g in range(n_groups):
        cols = np.where(group_index == g)[0]
        Gamma_g = _group_gram(X, cols, n)
        corr_g = _group_corr(X, y, cols, n)
        scores[g] = float(corr_g @ _solve_psd(Gamma_g, corr_g))
    return np.argsort(-scores)[:s]


def group_tau_threshold(
    s: int,
    n_groups: int,
    n: int,
    block_size: int = 1,
    c: float = 0.01,
) -> float:
    """Splicing threshold tau ≍ c * s * d * log(J) * log log(n) / n."""
    log_n = max(np.log(n), np.e)
    return c * s * block_size * np.log(max(n_groups, 2)) * np.log(log_n) / n
