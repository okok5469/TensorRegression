"""
向量最优子集选择：BESS.Fix(s) 与自适应 ABESS。

Algorithm 1 (BESS.Fix): 固定稀疏度 s，迭代 splicing 直至活跃集稳定。
Algorithm 3 (ABESS): 对 s = 1..s_max 分别运行 BESS.Fix，用 SIC 选最优 s。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_abess.vector.sacrifices import (
    column_norms,
    fit_ols_on_support,
    initial_active_set,
    squared_loss,
    tau_threshold,
)
from tensor_abess.vector.sic import compute_sic, default_s_max
from tensor_abess.vector.splicing import splicing


@dataclass
class BessFixResult:
    """Result of best-subset selection with fixed support size s."""

    beta: np.ndarray
    support: np.ndarray
    active_set: np.ndarray
    loss: float
    s: int
    n_iter: int


@dataclass
class AbessResult:
    """Result of adaptive best-subset selection with SIC."""

    beta: np.ndarray
    support: np.ndarray
    active_set: np.ndarray
    loss: float
    s: int
    sic: float
    sic_path: dict[int, float]
    bess_path: dict[int, BessFixResult]


def bess_fix(
    X: np.ndarray,
    y: np.ndarray,
    s: int,
    *,
    k_max: int | None = None,
    m_max: int = 20,
    active_init: np.ndarray | None = None,
) -> BessFixResult:
    """
    BESS.Fix(s): best-subset selection with known support size.

    Implements Algorithm 1 (Zhu et al. 2020) using column-wise splicing.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if not (1 <= s <= p):
        raise ValueError(f"support size s must satisfy 1 <= s <= p, got s={s}, p={p}")

    if k_max is None:
        k_max = min(s, 5)

    col_sq = column_norms(X)
    tau_s = tau_threshold(s, p, n)

    if active_init is None:
        active = initial_active_set(X, y, s)
    else:
        active = np.asarray(active_init, dtype=int)
        if active.size != s:
            raise ValueError(f"active_init must have length s={s}")

    inactive = np.setdiff1d(np.arange(p), active)
    beta, d, _ = fit_ols_on_support(y, X, active)

    for m in range(m_max):
        prev_active = active.copy()
        beta, d, active, inactive, _ = splicing(
            y, X, beta, d, active, inactive, k_max, tau_s, col_sq
        )
        # 活跃集不再变化 → 达到 splicing 稳定点，停止
        if np.array_equal(active, prev_active):
            return BessFixResult(
                beta=beta,
                support=active.copy(),
                active_set=active.copy(),
                loss=squared_loss(y, X, beta),
                s=s,
                n_iter=m + 1,
            )

    return BessFixResult(
        beta=beta,
        support=active.copy(),
        active_set=active.copy(),
        loss=squared_loss(y, X, beta),
        s=s,
        n_iter=m_max,
    )


def abess(
    X: np.ndarray,
    y: np.ndarray,
    *,
    s_max: int | None = None,
    s_min: int = 1,
    k_max: int | None = None,
    m_max: int = 20,
) -> AbessResult:
    """
    Adaptive BESS: select support size by minimizing SIC over s = s_min, ..., s_max.

    Implements Algorithm 3 (Zhu et al. 2020).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if s_max is None:
        s_max = default_s_max(n, p)
    s_max = min(s_max, p)
    s_min = max(1, s_min)

    bess_path: dict[int, BessFixResult] = {}
    sic_path: dict[int, float] = {}

    best_s = s_min
    best_sic = np.inf
    best_result: BessFixResult | None = None

    for s in range(s_min, s_max + 1):
        result = bess_fix(X, y, s, k_max=k_max, m_max=m_max)
        bess_path[s] = result
        sic_val = compute_sic(result.loss, s, n, p)
        sic_path[s] = sic_val
        if sic_val < best_sic:
            best_sic = sic_val
            best_s = s
            best_result = result

    assert best_result is not None
    return AbessResult(
        beta=best_result.beta,
        support=best_result.support,
        active_set=best_result.active_set,
        loss=best_result.loss,
        s=best_s,
        sic=best_sic,
        sic_path=sic_path,
        bess_path=bess_path,
    )
