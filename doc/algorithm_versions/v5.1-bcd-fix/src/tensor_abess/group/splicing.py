"""Group-level splicing: swap entire groups in and out together."""

from __future__ import annotations

import numpy as np

from tensor_abess.common.ranks import bottom_k_by_rank, top_k_by_rank
from tensor_abess.group.sacrifices import compute_group_sacrifices, fit_ols_on_groups


def group_splicing(
    y: np.ndarray,
    X: np.ndarray,
    group_index: np.ndarray,
    active_groups: np.ndarray,
    inactive_groups: np.ndarray,
    k_max: int,
    tau_s: float,
    *,
    beta_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One group-level splicing update."""
    n_groups = int(group_index.max()) + 1
    if beta_init is None:
        beta, loss0 = fit_ols_on_groups(y, X, group_index, active_groups)
    else:
        beta = np.asarray(beta_init, dtype=float).copy()
        residual = y - X @ beta
        loss0 = float(residual @ residual / (2.0 * y.shape[0]))

    best_groups = active_groups.copy()
    best_beta = beta.copy()
    best_loss = loss0

    xi, zeta = compute_group_sacrifices(
        y, X, group_index, active_groups, inactive_groups, loss0
    )

    max_k = min(int(k_max), active_groups.size, inactive_groups.size)
    for k in range(1, max_k + 1):
        out_g = bottom_k_by_rank(xi, active_groups, k)
        in_g = top_k_by_rank(zeta, inactive_groups, k)
        if out_g.size == 0 or in_g.size == 0:
            continue

        new_active = np.sort(
            np.union1d(np.setdiff1d(active_groups, out_g), in_g).astype(int)
        )
        new_beta, new_loss = fit_ols_on_groups(y, X, group_index, new_active)
        if new_loss < best_loss:
            best_groups = new_active
            best_beta = new_beta
            best_loss = new_loss

    if loss0 - best_loss < tau_s:
        return beta, active_groups, inactive_groups, loss0

    new_inactive = np.setdiff1d(np.arange(n_groups), best_groups)
    return best_beta, best_groups, new_inactive, best_loss
