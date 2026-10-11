"""Tensor BCD solvers: legacy fixed-size and revised adaptive outer loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from time import perf_counter
from tensor_abess.tensor.candidate_search import candidate_size_triples as _candidate_size_triples
from tensor_abess.tensor.candidate_search import refine_msic_global_candidates as _refine_msic_global_candidates
from tensor_abess.tensor.candidate_search import validate_candidate_state

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
    best_iteration: int = 0
    best_rss: float = 0.0
    stop_reason: str = ""
    visited_loss_history: list[float] = field(default_factory=list)
    accepted_support_history: list[tuple[tuple[int, ...], ...]] = field(default_factory=list)


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
    candidate_records: list[dict] = field(default_factory=list)
    search_metadata: dict = field(default_factory=dict)


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
    """Fixed-size search with explicit exploration and best-state acceptance.

    ``loss_history`` records initial and accepted mode states; ``best_iteration``
    indexes all complete OLS proposals in ``visited_loss_history``. The returned
    supports, coefficients and loss always belong to the best visited state.
    Cold Group-BESS proposals retain the legacy exploration opportunities; their
    nonmonotone states are never mistaken for accepted best-model updates.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 4 or len(y) != len(X) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fixed BCD requires aligned finite tensor samples")
    if S1_init is None or S2_init is None or S3_init is None:
        S1, S2, S3 = initial_support(
            X, y, s1, s2, s3, strategy=init_strategy, random_state=random_state
        )
    else:
        S1 = np.sort(np.asarray(S1_init, dtype=int))
        S2 = np.sort(np.asarray(S2_init, dtype=int))
        S3 = np.sort(np.asarray(S3_init, dtype=int))

    sizes = (s1, s2, s3)
    supports = [S1, S2, S3]
    for support, size, dimension in zip(supports, sizes, X.shape[1:]):
        if len(support) != size or len(np.unique(support)) != size or np.any((support < 0) | (support >= dimension)):
            raise ValueError("Initial supports must satisfy the fixed-size constraints")

    def state_key(state):
        return tuple(tuple(int(index) for index in support) for support in state)

    B = refit_tensor_coefficients(X, y, *supports)
    loss = tensor_loss(X, y, B)
    if not np.isfinite(loss) or not np.isfinite(B).all():
        raise FloatingPointError("Nonfinite initial fixed-BCD OLS state")
    best_supports = tuple(support.copy() for support in supports)
    best_B, best_loss, best_iteration = B.copy(), loss, 0
    loss_history = [loss]
    visited_loss_history = [loss]
    accepted_support_history = [state_key(supports)]
    seen_states = {state_key(supports)}
    accepted_loss = loss
    n_cycles, stop_reason = 0, "max_cycles"

    for cycle in range(max_cycles):
        n_cycles = cycle + 1
        sweep_start = state_key(supports)
        for mode in range(3):
            Z, gi, n_groups = build_mode_design(X, mode + 1, *supports)
            block_size = int(np.prod([len(supports[i]) for i in range(3) if i != mode]))
            proposal = group_bess_fix(
                y, Z, gi, sizes[mode], n_groups=n_groups, m_max=m_max,
                k_max=k_max, block_size=max(block_size, 1),
            )
            active = np.sort(np.asarray(proposal.active_groups, dtype=int))
            if len(active) != sizes[mode] or len(np.unique(active)) != sizes[mode] or np.any((active < 0) | (active >= n_groups)):
                raise RuntimeError("Group-BESS violated the fixed support size")
            candidate = [support.copy() for support in supports]
            candidate[mode] = active
            candidate_B = refit_tensor_coefficients(X, y, *candidate)
            candidate_loss = tensor_loss(X, y, candidate_B)
            if not np.isfinite(candidate_loss) or not np.isfinite(candidate_B).all():
                raise FloatingPointError("Nonfinite fixed-BCD OLS candidate")
            visited_loss_history.append(candidate_loss)
            if candidate_loss < best_loss:
                best_supports = tuple(support.copy() for support in candidate)
                best_B, best_loss = candidate_B.copy(), candidate_loss
                best_iteration = len(visited_loss_history) - 1
            key = state_key(candidate)
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(accepted_loss), abs(candidate_loss))
            if candidate_loss < accepted_loss - tolerance:
                accepted_loss = candidate_loss
                loss_history.append(candidate_loss)
                accepted_support_history.append(key)
            supports, B, loss = candidate, candidate_B, candidate_loss
        key = state_key(supports)
        if key in seen_states:
            stop_reason = "unchanged_support" if key == sweep_start else "exploration_cycle"
            break
        seen_states.add(key)

    return TensorBcdResult(
        B=best_B,
        S1=best_supports[0],
        S2=best_supports[1],
        S3=best_supports[2],
        loss=best_loss,
        loss_history=loss_history,
        n_cycles=n_cycles,
        best_iteration=best_iteration,
        best_rss=2.0 * len(y) * best_loss,
        stop_reason=stop_reason,
        visited_loss_history=visited_loss_history,
        accepted_support_history=accepted_support_history,
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
    cap_policy: Literal["legacy", "candidate"] = "legacy",
    grid_policy: Literal["legacy", "full", "coarse_to_fine"] = "legacy",
    candidate_budget: int = 2000,
    fixed_candidate_cache: dict | None = None,
) -> TensorBcdAdaptiveResult:
    """Algorithm 5.2': revised adaptive BCD + GroupBESS.Adaptive outer loop."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape

    S1 = np.sort(np.asarray(S1_init, dtype=int))
    S2 = np.sort(np.asarray(S2_init, dtype=int))
    S3 = np.sort(np.asarray(S3_init, dtype=int))
    if cap_policy not in ("legacy", "candidate") or grid_policy not in ("legacy", "full", "coarse_to_fine"):
        raise ValueError("Invalid cap_policy or grid_policy")
    caps = (s1_max, s2_max, s3_max)
    max_params = max(1, int(np.floor(0.7 * n)))
    if cap_policy == "candidate":
        validate_candidate_state((S1, S2, S3), caps, max_params, X.shape[1:], stage="initial")
    metadata = {"cap_policy": cap_policy, "grid_policy": grid_policy,
                "adaptive_init_supports": (S1.copy(), S2.copy(), S3.copy()),
                "dynamic_cap_history": []}
    adaptive_start = perf_counter()

    def mode_cap(cap: int, block_size: int, mode: int) -> int:
        bound = min(cap, max_params // block_size) if cap_policy == "candidate" else cap
        metadata["dynamic_cap_history"].append({"mode": mode, "cap": cap,
                                                "block_size": block_size, "s_max": bound})
        if bound < 1:
            raise ValueError(f"No legal adaptive mode-{mode} candidate: block_size={block_size}, Q={max_params}")
        return bound

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
            s_max=mode_cap(s1_max, d1, 1),
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d1,
            groups_warm=S1,
            tau=tau,
            information_criterion=information_criterion,
        )
        S1 = np.sort(res1.active_groups)
        if cap_policy == "candidate":
            validate_candidate_state((S1, S2, S3), caps, max_params, X.shape[1:], stage="adaptive-mode")

        d2 = max(len(S1) * len(S3), 1)
        Z, gi, n_g = build_mode_design(X, 2, S1, S2, S3)
        res2 = group_bess_adaptive(
            y,
            Z,
            gi,
            s_max=mode_cap(s2_max, d2, 2),
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d2,
            groups_warm=S2,
            tau=tau,
            information_criterion=information_criterion,
        )
        S2 = np.sort(res2.active_groups)
        if cap_policy == "candidate":
            validate_candidate_state((S1, S2, S3), caps, max_params, X.shape[1:], stage="adaptive-mode")

        d3 = max(len(S1) * len(S2), 1)
        Z, gi, n_g = build_mode_design(X, 3, S1, S2, S3)
        res3 = group_bess_adaptive(
            y,
            Z,
            gi,
            s_max=mode_cap(s3_max, d3, 3),
            n_groups=n_g,
            k_max=k_max,
            m_max=m_max,
            block_size=d3,
            groups_warm=S3,
            tau=tau,
            information_criterion=information_criterion,
        )
        S3 = np.sort(res3.active_groups)
        if cap_policy == "candidate":
            validate_candidate_state((S1, S2, S3), caps, max_params, X.shape[1:], stage="adaptive-mode")

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
                search_metadata={**metadata, "adaptive_fit_seconds": perf_counter() - adaptive_start},
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
                    cap_policy=cap_policy, grid_policy=grid_policy,
                    candidate_budget=candidate_budget, fixed_candidate_cache=fixed_candidate_cache,
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
        search_metadata={**metadata, "adaptive_fit_seconds": perf_counter() - adaptive_start},
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
            cap_policy=cap_policy, grid_policy=grid_policy,
            candidate_budget=candidate_budget, fixed_candidate_cache=fixed_candidate_cache,
        )
    return result
