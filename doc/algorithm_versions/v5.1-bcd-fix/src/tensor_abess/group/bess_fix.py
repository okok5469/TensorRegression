"""Group-level BESS.Fix(s): select a fixed number of active groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_abess.group.sacrifices import (
    fit_ols_on_groups,
    group_tau_threshold,
    initial_active_groups,
)
from tensor_abess.group.splicing import group_splicing
from tensor_abess.vector.sacrifices import squared_loss


@dataclass
class GroupBessFixResult:
    """Best-subset selection over groups with fixed group support size."""

    beta: np.ndarray
    active_groups: np.ndarray
    loss: float
    rss: float
    s: int
    n_iter: int


def group_bess_fix(
    y: np.ndarray,
    X: np.ndarray,
    group_index: np.ndarray,
    s: int,
    *,
    n_groups: int | None = None,
    k_max: int | None = None,
    m_max: int = 20,
    groups_init: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    block_size: int = 1,
    tau: float | None = None,
) -> GroupBessFixResult:
    """
    BESS.Fix(s) on grouped predictors: select s out of n_groups groups.

    ``group_index[j]`` gives the group id of column j; each group enters
    or leaves the model as a whole block.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    group_index = np.asarray(group_index, dtype=int)

    if n_groups is None:
        n_groups = int(group_index.max()) + 1

    if not (1 <= s <= n_groups):
        raise ValueError(f"need 1 <= s <= n_groups, got s={s}, n_groups={n_groups}")

    if k_max is None:
        k_max = min(s, 5)
    else:
        k_max = min(int(k_max), s)

    n = y.shape[0]
    tau_s = (
        group_tau_threshold(s, n_groups, n, block_size=block_size)
        if tau is None
        else float(tau)
    )

    if groups_init is None:
        active = np.sort(initial_active_groups(X, y, group_index, s).astype(int))
    else:
        active = np.sort(np.asarray(groups_init, dtype=int))
        if active.size != s:
            raise ValueError(f"groups_init must have length s={s}")

    inactive = np.setdiff1d(np.arange(n_groups), active)
    beta = np.asarray(beta_init, dtype=float) if beta_init is not None else None

    for m in range(m_max):
        prev = active.copy()
        beta, active, inactive, _ = group_splicing(
            y,
            X,
            group_index,
            active,
            inactive,
            k_max,
            tau_s,
            beta_init=beta,
        )
        if active.size != s:
            raise RuntimeError(
                f"group_splicing must preserve support size s={s}, got {active.size}"
            )
        if np.array_equal(active, prev):
            loss = squared_loss(y, X, beta)
            return GroupBessFixResult(
                beta=beta,
                active_groups=active.copy(),
                loss=loss,
                rss=2.0 * n * loss,
                s=s,
                n_iter=m + 1,
            )

    beta, _ = fit_ols_on_groups(y, X, group_index, active)
    if active.size != s:
        raise RuntimeError(f"final active group count must equal s={s}, got {active.size}")
    loss = squared_loss(y, X, beta)
    return GroupBessFixResult(
        beta=beta,
        active_groups=active.copy(),
        loss=loss,
        rss=2.0 * n * loss,
        s=s,
        n_iter=m_max,
    )
