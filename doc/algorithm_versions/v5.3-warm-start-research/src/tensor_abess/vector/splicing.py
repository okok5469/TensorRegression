"""
向量 splicing 单步（Zhu et al. 2020, Algorithm 2）。

核心思想：从活跃集中换出 xi 最小的 k 个变量，同时从非活跃集换入 zeta 最大的 k 个变量，
在保持支撑大小 s 不变的前提下尝试降低损失。
"""

from __future__ import annotations

import numpy as np

from tensor_abess.common.ranks import bottom_k_by_rank, top_k_by_rank
from tensor_abess.vector.sacrifices import (
    compute_sacrifices,
    fit_ols_on_support,
    squared_loss,
)


def splicing(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    d: np.ndarray,
    active: np.ndarray,
    inactive: np.ndarray,
    k_max: int,
    tau_s: float,
    col_sq: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    One splicing update. Returns (beta, d, active, inactive, loss).

    If improvement is below ``tau_s``, the input state is returned unchanged.
    """
    n, p = X.shape
    if col_sq is None:
        col_sq = np.sum(X * X, axis=0)

    loss0 = squared_loss(y, X, beta)
    best_beta, best_d = beta.copy(), d.copy()
    best_active, best_inactive = active.copy(), inactive.copy()
    best_loss = loss0

    xi, zeta = compute_sacrifices(beta, d, active, inactive, col_sq, n)

    # 尝试 splice 大小 k = 1, 2, ..., k_max，取损失最小的方案
    for k in range(1, k_max + 1):
        out_a = bottom_k_by_rank(xi, active, k)   # 最该换出的 k 个活跃变量
        in_a = top_k_by_rank(zeta, inactive, k)  # 最该换入的 k 个非活跃变量
        if out_a.size == 0 or in_a.size == 0:
            continue

        new_active = np.union1d(np.setdiff1d(active, out_a), in_a)
        new_active = np.sort(new_active.astype(int))
        new_inactive = np.setdiff1d(np.arange(p), new_active)

        new_beta, new_d, new_loss = fit_ols_on_support(y, X, new_active)
        if new_loss < best_loss:
            best_beta, best_d = new_beta, new_d
            best_active, best_inactive = new_active, new_inactive
            best_loss = new_loss

    if loss0 - best_loss < tau_s:
        return beta, d, active, inactive, loss0

    return best_beta, best_d, best_active, best_inactive, best_loss
