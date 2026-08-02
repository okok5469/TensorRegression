"""Adaptive group BESS with revised mode-wise SIC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_abess.group.bess_fix import GroupBessFixResult, group_bess_fix
from tensor_abess.group.sacrifices import initial_active_groups
from tensor_abess.tensor.sic import compute_mode_sic_from_rss


@dataclass
class GroupAbessResult:
    """Adaptive grouped best-subset result."""

    beta: np.ndarray
    active_groups: np.ndarray
    s: int
    sic: float
    rss: float
    loss: float
    sic_path: dict[int, float]
    bess_path: dict[int, GroupBessFixResult]


def group_bess_adaptive(
    y: np.ndarray,
    X: np.ndarray,
    group_index: np.ndarray,
    s_max: int,
    *,
    n_groups: int | None = None,
    k_max: int | None = None,
    m_max: int = 20,
    block_size: int = 1,
    groups_warm: np.ndarray | None = None,
    beta_warm: np.ndarray | None = None,
    tau: float | None = None,
) -> GroupAbessResult:
    """Adaptive grouped BESS using the revised conditional SIC from Algorithm 5.1."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    group_index = np.asarray(group_index, dtype=int)

    if n_groups is None:
        n_groups = int(group_index.max()) + 1
    s_max = min(int(s_max), n_groups)
    if s_max < 1:
        raise ValueError("s_max must be at least 1")

    sic_path: dict[int, float] = {}
    bess_path: dict[int, GroupBessFixResult] = {}

    best_result: GroupBessFixResult | None = None
    best_s = 1
    best_sic = np.inf

    for s in range(1, s_max + 1):
        if groups_warm is not None and len(groups_warm) == s:
            groups_init = np.sort(np.asarray(groups_warm, dtype=int))
            beta_init = None if beta_warm is None else np.asarray(beta_warm, dtype=float)
        else:
            groups_init = np.sort(initial_active_groups(X, y, group_index, s).astype(int))
            beta_init = None

        result = group_bess_fix(
            y,
            X,
            group_index,
            s,
            n_groups=n_groups,
            k_max=k_max,
            m_max=m_max,
            groups_init=groups_init,
            beta_init=beta_init,
            block_size=block_size,
            tau=tau,
        )
        bess_path[s] = result
        sic_val = compute_mode_sic_from_rss(
            result.rss, s=s, n=y.shape[0], J=n_groups, block_size=block_size
        )
        sic_path[s] = sic_val
        if sic_val < best_sic:
            best_sic = sic_val
            best_s = s
            best_result = result

    assert best_result is not None
    return GroupAbessResult(
        beta=best_result.beta,
        active_groups=best_result.active_groups,
        s=best_s,
        sic=best_sic,
        rss=best_result.rss,
        loss=best_result.loss,
        sic_path=sic_path,
        bess_path=bess_path,
    )
