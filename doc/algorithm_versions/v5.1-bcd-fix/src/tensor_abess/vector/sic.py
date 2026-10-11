"""
向量 SIC 准则（Zhu et al. 2020, Algorithm 3）。

SIC(A) = n·log(L_A) + |A|·log(p)·log(log n)
用于在 s = 1, ..., s_max 之间自动选择稀疏度。
"""

from __future__ import annotations

import numpy as np


def compute_sic(loss: float, support_size: int, n: int, p: int) -> float:
    """
    SIC(A) = n log(L_A) + |A| log(p) log(log n).

    ``loss`` is L_A = min_{beta_I=0} ||y - X beta||^2 / (2n).
    """
    eps = np.finfo(float).tiny
    log_n = max(np.log(n), np.e)
    return n * np.log(max(loss, eps)) + support_size * np.log(p) * np.log(log_n)


def default_s_max(n: int, p: int) -> int:
    """Suggested maximum support size: floor(n / (log(p) log log n))."""
    log_n = max(np.log(n), np.e)
    denom = np.log(max(p, 2)) * np.log(log_n)
    if denom <= 0:
        return min(n - 1, p)
    return max(1, min(p, int(n / denom)))
