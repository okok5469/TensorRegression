"""
向量 ABESS 的损失函数、牺牲量与 OLS 工具。

对应 Zhu et al. (2020) 线性模型 splicing 方法中的符号：
  - 活跃集 A / 非活跃集 I
  - 后向牺牲量 xi_j：移除变量 j 导致的损失增量
  - 前向牺牲量 zeta_j：加入变量 j 导致的损失减量
"""

from __future__ import annotations

import numpy as np


def squared_loss(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
    """Half mean squared error: L_n(beta) = ||y - X beta||^2 / (2n)."""
    n = y.shape[0]
    residual = y - X @ beta
    return float(residual @ residual / (2.0 * n))


def column_norms(X: np.ndarray) -> np.ndarray:
    """Return diagonal entries X_j^T X_j for each column."""
    return np.sum(X * X, axis=0)


def fit_ols_on_support(
    y: np.ndarray, X: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit OLS with support ``active`` and return (beta, d, loss).

    beta is p-dimensional with zeros outside active; d_j = X_j^T (y - X beta) / n.
    """
    p = X.shape[1]
    n = y.shape[0]
    beta = np.zeros(p, dtype=float)
    if active.size == 0:
        d = X.T @ y / n
        return beta, d, squared_loss(y, X, beta)

    coef, _, _, _ = np.linalg.lstsq(X[:, active], y, rcond=None)
    beta[active] = coef
    residual = y - X @ beta
    # d_j 是残差在列 j 上的投影，用于计算前向牺牲量（不重新拟合时的边际贡献）
    d = X.T @ residual / n
    loss = float(residual @ residual / (2.0 * n))
    return beta, d, loss


def compute_sacrifices(
    beta: np.ndarray,
    d: np.ndarray,
    active: np.ndarray,
    inactive: np.ndarray,
    col_sq: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward sacrifices xi_j (j in active) and forward sacrifices zeta_j (j in inactive).

    xi_j = (X_j^T X_j / 2n) * beta_j^2
    zeta_j = (X_j^T X_j / 2n) * (d_j / (X_j^T X_j / n))^2
    """
    xi = np.zeros_like(beta)
    zeta = np.zeros_like(beta)

    for j in active:
        # xi 越大 → 该活跃变量对拟合贡献越小 → 优先被换出
        xi[j] = col_sq[j] * beta[j] ** 2 / (2.0 * n)

    for j in inactive:
        denom = col_sq[j] / n
        if denom > 0:
            # zeta 越大 → 加入该变量越能降损失 → 优先被换入
            zeta[j] = col_sq[j] * (d[j] / denom) ** 2 / (2.0 * n)

    return xi, zeta


def initial_active_set(X: np.ndarray, y: np.ndarray, s: int) -> np.ndarray:
    """Top-s variables by |X_j^T y| / sqrt(X_j^T X_j)."""
    col_sq = column_norms(X)
    scores = np.zeros(X.shape[1], dtype=float)
    mask = col_sq > 0
    scores[mask] = np.abs(X[:, mask].T @ y) / np.sqrt(col_sq[mask])
    return np.argsort(-scores)[:s]


def tau_threshold(s: int, p: int, n: int) -> float:
    """Splicing improvement threshold tau_s from the paper."""
    log_n = max(np.log(n), np.e)
    # 低于此阈值的损失改进视为噪声，不再接受 splice（防止过度迭代）
    return 0.01 * s * np.log(p) / n * np.log(log_n)
