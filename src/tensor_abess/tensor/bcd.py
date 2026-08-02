"""Tensor BCD solvers: legacy fixed-size and revised adaptive outer loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensor_abess.group.adaptive import group_bess_adaptive
from tensor_abess.group.bess_fix import group_bess_fix
from tensor_abess.tensor.ops import refit_tensor_coefficients, tensor_loss
from tensor_abess.tensor.reduction import build_mode_design
from tensor_abess.tensor.sic import compute_tensor_sic_from_rss
from tensor_abess.utils.init import InitStrategy, initial_support


@dataclass
class TensorBcdResult:
    """Output of fixed-size tensor BCD with sparsities (s1, s2, s3)."""

    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    loss: float
    loss_history: list[float] = field(default_factory=list)
    n_cycles: int = 0


@dataclass
class TensorBcdAdaptiveResult:
    """Output of the revised adaptive tensor BCD outer loop."""

    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    s1: int
    s2: int
    s3: int
    loss: float
    rss: float
    sic: float
    loss_history: list[float] = field(default_factory=list)
    sic_history: list[float] = field(default_factory=list)
    n_cycles: int = 0


def tensor_bcd(
    X: np.ndarray,
    y: np.ndarray,
    s1: int,
    s2: int,
    s3: int,
    *,
    max_cycles: int = 30,
    m_max: int = 15,
    k_max: int | None = None,
    init_strategy: InitStrategy = "marginal",
    random_state: int = 0,
    S1_init: np.ndarray | None = None,
    S2_init: np.ndarray | None = None,
    S3_init: np.ndarray | None = None,
) -> TensorBcdResult:
    """Legacy fixed-size BCD retained for backward compatibility."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if S1_init is None or S2_init is None or S3_init is None:
        S1, S2, S3 = initial_support(
            X, y, s1, s2, s3, strategy=init_strategy, random_state=random_state
        )
    else:
        S1 = np.sort(np.asarray(S1_init, dtype=int))
        S2 = np.sort(np.asarray(S2_init, dtype=int))
        S3 = np.sort(np.asarray(S3_init, dtype=int))

    B = refit_tensor_coefficients(X, y, S1, S2, S3)
    loss_history = [tensor_loss(X, y, B)]

    for cycle in range(max_cycles):
        changed = False

        Z, gi, n_g = build_mode_design(X, 1, S1, S2, S3)
        res1 = group_bess_fix(
            y,
            Z,
            gi,
            s1,
            n_groups=n_g,
            m_max=m_max,
            k_max=k_max,
            block_size=max(len(S2) * len(S3), 1),
        )
        new_S1 = np.sort(res1.active_groups)
        if not np.array_equal(new_S1, S1):
            S1 = new_S1
            changed = True

        Z, gi, n_g = build_mode_design(X, 2, S1, S2, S3)
        res2 = group_bess_fix(
            y,
            Z,
            gi,
            s2,
            n_groups=n_g,
            m_max=m_max,
            k_max=k_max,
            block_size=max(len(S1) * len(S3), 1),
        )
        new_S2 = np.sort(res2.active_groups)
        if not np.array_equal(new_S2, S2):
            S2 = new_S2
            changed = True

        Z, gi, n_g = build_mode_design(X, 3, S1, S2, S3)
        res3 = group_bess_fix(
            y,
            Z,
            gi,
            s3,
            n_groups=n_g,
            m_max=m_max,
            k_max=k_max,
            block_size=max(len(S1) * len(S2), 1),
        )
        new_S3 = np.sort(res3.active_groups)
        if not np.array_equal(new_S3, S3):
            S3 = new_S3
            changed = True

        B = refit_tensor_coefficients(X, y, S1, S2, S3)
        loss = tensor_loss(X, y, B)
        loss_history.append(loss)

        if not changed:
            return TensorBcdResult(
                B=B,
                S1=S1,
                S2=S2,
                S3=S3,
                loss=loss,
                loss_history=loss_history,
                n_cycles=cycle + 1,
            )

    loss = tensor_loss(X, y, B)
    return TensorBcdResult(
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        loss=loss,
        loss_history=loss_history,
        n_cycles=max_cycles,
    )


def tensor_bcd_adaptive(
    X: np.ndarray,
    y: np.ndarray,
    *,
    s1_max: int,
    s2_max: int,
    s3_max: int,
    S1_init: np.ndarray,
    S2_init: np.ndarray,
    S3_init: np.ndarray,
    max_cycles: int = 30,
    m_max: int = 15,
    k_max: int | None = None,
    tau: float | None = None,
    eps: float = 1e-8,
) -> TensorBcdAdaptiveResult:
    """Algorithm 5.2': revised adaptive BCD + GroupBESS.Adaptive outer loop."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape

    S1 = np.sort(np.asarray(S1_init, dtype=int))
    S2 = np.sort(np.asarray(S2_init, dtype=int))
    S3 = np.sort(np.asarray(S3_init, dtype=int))

    B = refit_tensor_coefficients(X, y, S1, S2, S3)
    loss = tensor_loss(X, y, B)
    rss = 2.0 * n * loss
    sic = compute_tensor_sic_from_rss(rss, len(S1), len(S2), len(S3), n, p1, p2, p3)
    loss_history = [loss]
    sic_history = [sic]

    for cycle in range(max_cycles):
        prev_supports = (S1.copy(), S2.copy(), S3.copy())
        prev_sic = sic_history[-1]

        d1 = max(len(S2) * len(S3), 1)
        Z, gi, n_g = build_mode_design(X, 1, S1, S2, S3)
        res1 = group_bess_adaptive(
            y,
            Z,
            gi,
            s_max=s1_max,
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d1,
            groups_warm=S1,
            tau=tau,
        )
        S1 = np.sort(res1.active_groups)

        d2 = max(len(S1) * len(S3), 1)
        Z, gi, n_g = build_mode_design(X, 2, S1, S2, S3)
        res2 = group_bess_adaptive(
            y,
            Z,
            gi,
            s_max=s2_max,
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d2,
            groups_warm=S2,
            tau=tau,
        )
        S2 = np.sort(res2.active_groups)

        d3 = max(len(S1) * len(S2), 1)
        Z, gi, n_g = build_mode_design(X, 3, S1, S2, S3)
        res3 = group_bess_adaptive(
            y,
            Z,
            gi,
            s_max=s3_max,
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d3,
            groups_warm=S3,
            tau=tau,
        )
        S3 = np.sort(res3.active_groups)

        B = refit_tensor_coefficients(X, y, S1, S2, S3)
        loss = tensor_loss(X, y, B)
        rss = 2.0 * n * loss
        sic = compute_tensor_sic_from_rss(rss, len(S1), len(S2), len(S3), n, p1, p2, p3)
        loss_history.append(loss)
        sic_history.append(sic)

        unchanged = (
            np.array_equal(S1, prev_supports[0])
            and np.array_equal(S2, prev_supports[1])
            and np.array_equal(S3, prev_supports[2])
        )
        if unchanged or abs(prev_sic - sic) < eps:
            return TensorBcdAdaptiveResult(
                B=B,
                S1=S1,
                S2=S2,
                S3=S3,
                s1=len(S1),
                s2=len(S2),
                s3=len(S3),
                loss=loss,
                rss=rss,
                sic=sic,
                loss_history=loss_history,
                sic_history=sic_history,
                n_cycles=cycle + 1,
            )

    return TensorBcdAdaptiveResult(
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        s1=len(S1),
        s2=len(S2),
        s3=len(S3),
        loss=loss_history[-1],
        rss=2.0 * n * loss_history[-1],
        sic=sic_history[-1],
        loss_history=loss_history,
        sic_history=sic_history,
        n_cycles=max_cycles,
    )
