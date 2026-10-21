"""Tensor BCD solvers: legacy fixed-size and revised adaptive outer loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from tensor_abess.group.adaptive import group_bess_adaptive
from tensor_abess.group.bess_fix import group_bess_fix
from tensor_abess.tensor.ops import refit_tensor_coefficients, tensor_loss
from tensor_abess.tensor.reduction import build_mode_design
from tensor_abess.tensor.sic import compute_tensor_msic_from_rss, compute_tensor_sic_from_rss
from tensor_abess.utils.init import InitStrategy, _mode_slice_scores, initial_support


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


def _candidate_size_triples(
    s1_max: int,
    s2_max: int,
    s3_max: int,
    max_params: int,
    *,
    window: int = 3,
) -> list[tuple[int, int, int]]:
    def values_for_cap(s_max: int) -> list[int]:
        cap = int(s_max)
        near_cap = range(max(1, cap - window + 1), cap + 1)
        anchors = {1, max(1, round(0.4 * cap)), max(1, round(0.6 * cap))}
        return sorted({*near_cap, *anchors})

    ranges = [values_for_cap(s_max) for s_max in (s1_max, s2_max, s3_max)]
    triples = [
        (s1, s2, s3)
        for s1 in ranges[0]
        for s2 in ranges[1]
        for s3 in ranges[2]
        if s1 * s2 * s3 <= max_params
    ]
    return triples or [(1, 1, 1)]


def _marginal_supports_for_size(
    X: np.ndarray,
    y: np.ndarray,
    sizes: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.sort(np.argsort(-_mode_slice_scores(X, y, mode))[:s])
        for mode, s in enumerate(sizes, start=1)
    )  # type: ignore[return-value]


def _support_size(result: TensorBcdAdaptiveResult) -> int:
    return int(result.s1 * result.s2 * result.s3)


def _msic_one_se_tolerance(n: int, q_best: int, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(scale * np.sqrt(max(q_best, 1) * max(np.log(max(n, 2)), 1.0)))


def _refine_msic_global_candidates(
    X: np.ndarray,
    y: np.ndarray,
    result: TensorBcdAdaptiveResult,
    *,
    s1_max: int,
    s2_max: int,
    s3_max: int,
    max_cycles: int,
    m_max: int,
    k_max: int | None,
    tensor_criterion_fn,
    one_se_scale: float,
) -> TensorBcdAdaptiveResult:
    n, p1, p2, p3 = X.shape
    current_score = tensor_criterion_fn(
        result.rss, result.s1, result.s2, result.s3, n, p1, p2, p3
    )
    candidates: list[tuple[TensorBcdAdaptiveResult, float]] = [(result, current_score)]
    max_params = max(1, int(np.floor(0.7 * n)))

    for sizes in _candidate_size_triples(s1_max, s2_max, s3_max, max_params):
        S1_init, S2_init, S3_init = _marginal_supports_for_size(X, y, sizes)
        fixed = tensor_bcd(
            X,
            y,
            *sizes,
            max_cycles=max_cycles,
            m_max=m_max,
            k_max=k_max,
            S1_init=S1_init,
            S2_init=S2_init,
            S3_init=S3_init,
        )
        rss = 2.0 * n * fixed.loss
        score = tensor_criterion_fn(rss, *sizes, n, p1, p2, p3)
        candidates.append(
            (
                TensorBcdAdaptiveResult(
                    B=fixed.B,
                    S1=fixed.S1,
                    S2=fixed.S2,
                    S3=fixed.S3,
                    s1=sizes[0],
                    s2=sizes[1],
                    s3=sizes[2],
                    loss=fixed.loss,
                    rss=rss,
                    sic=score,
                    loss_history=fixed.loss_history,
                    sic_history=[*result.sic_history, score],
                    n_cycles=result.n_cycles + fixed.n_cycles,
                ),
                score,
            )
        )

    _, min_score = min(candidates, key=lambda item: item[1])
    q_best = _support_size(min(candidates, key=lambda item: item[1])[0])
    tolerance = _msic_one_se_tolerance(n, q_best, one_se_scale)
    eligible = [
        (candidate, score)
        for candidate, score in candidates
        if score <= min_score + tolerance
    ]
    selected, selected_score = min(
        eligible,
        key=lambda item: (_support_size(item[0]), item[1], item[0].s1 + item[0].s2 + item[0].s3),
    )
    if selected_score != selected.sic:
        selected = TensorBcdAdaptiveResult(
            B=selected.B,
            S1=selected.S1,
            S2=selected.S2,
            S3=selected.S3,
            s1=selected.s1,
            s2=selected.s2,
            s3=selected.s3,
            loss=selected.loss,
            rss=selected.rss,
            sic=selected_score,
            loss_history=selected.loss_history,
            sic_history=selected.sic_history,
            n_cycles=selected.n_cycles,
        )
    return selected


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
            groups_init=S1,
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
            groups_init=S2,
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
            groups_init=S3,
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
    information_criterion: Literal["sic", "msic"] = "msic",
    msic_refine_alpha: float = 1.5,
    msic_refine_gamma: float = 1.0,
    msic_refine_one_se_scale: float = 1.0,
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
    if information_criterion == "msic":
        tensor_criterion_fn = compute_tensor_msic_from_rss
        refine_tensor_criterion_fn = lambda rss, s1, s2, s3, n, p1, p2, p3: compute_tensor_msic_from_rss(
            rss,
            s1,
            s2,
            s3,
            n,
            p1,
            p2,
            p3,
            alpha=msic_refine_alpha,
            gamma=msic_refine_gamma,
        )
    else:
        tensor_criterion_fn = compute_tensor_sic_from_rss
        refine_tensor_criterion_fn = compute_tensor_sic_from_rss
    sic = tensor_criterion_fn(rss, len(S1), len(S2), len(S3), n, p1, p2, p3)
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
            information_criterion=information_criterion,
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
            information_criterion=information_criterion,
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
            information_criterion=information_criterion,
        )
        S3 = np.sort(res3.active_groups)

        B = refit_tensor_coefficients(X, y, S1, S2, S3)
        loss = tensor_loss(X, y, B)
        rss = 2.0 * n * loss
        sic = tensor_criterion_fn(rss, len(S1), len(S2), len(S3), n, p1, p2, p3)
        loss_history.append(loss)
        sic_history.append(sic)

        unchanged = (
            np.array_equal(S1, prev_supports[0])
            and np.array_equal(S2, prev_supports[1])
            and np.array_equal(S3, prev_supports[2])
        )
        if unchanged or abs(prev_sic - sic) < eps:
            result = TensorBcdAdaptiveResult(
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
            if information_criterion == "msic":
                return _refine_msic_global_candidates(
                    X,
                    y,
                    result,
                    s1_max=s1_max,
                    s2_max=s2_max,
                    s3_max=s3_max,
                    max_cycles=max_cycles,
                    m_max=m_max,
                    k_max=k_max,
                    tensor_criterion_fn=refine_tensor_criterion_fn,
                    one_se_scale=msic_refine_one_se_scale,
                )
            return result

    result = TensorBcdAdaptiveResult(
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
    if information_criterion == "msic":
        return _refine_msic_global_candidates(
            X,
            y,
            result,
            s1_max=s1_max,
            s2_max=s2_max,
            s3_max=s3_max,
            max_cycles=max_cycles,
            m_max=m_max,
            k_max=k_max,
            tensor_criterion_fn=refine_tensor_criterion_fn,
            one_se_scale=msic_refine_one_se_scale,
        )
    return result
