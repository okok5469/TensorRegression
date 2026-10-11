"""
Splicing 中按牺牲量排名选取 top-k / bottom-k 的工具。

排名规则（与论文一致）：
  - bottom_k：xi_j 越大排名越靠前 → 取「最该换出」的 k 个
  - top_k：zeta_j 越大排名越靠前 → 取「最该换入」的 k 个
"""

from __future__ import annotations

import numpy as np


def bottom_k_by_rank(values: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    """Return k indices with smallest backward-sacrifice rank."""
    if k <= 0 or indices.size == 0:
        return np.array([], dtype=int)
    k = min(k, indices.size)
    subset_vals = values[indices]
    # rank_j = #{i : v_j >= v_i}，rank 小 → 牺牲量相对更小 → 优先换出
    ranks = np.sum(subset_vals[:, None] >= subset_vals[None, :], axis=1)
    order = np.argsort(ranks, kind="stable")[:k]
    return indices[order]


def top_k_by_rank(values: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    """Return k indices with largest forward-sacrifice rank."""
    if k <= 0 or indices.size == 0:
        return np.array([], dtype=int)
    k = min(k, indices.size)
    subset_vals = values[indices]
    # rank_j = #{i : v_j <= v_i}，rank 小 → 牺牲量相对更大 → 优先换入
    ranks = np.sum(subset_vals[:, None] <= subset_vals[None, :], axis=1)
    order = np.argsort(ranks, kind="stable")[:k]
    return indices[order]
