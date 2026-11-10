"""Initialization utilities for tensor BCD and the revised Tensor-ABESS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tensor_abess.tensor.reduction import build_mode_design_subset
from tensor_abess.utils.glasso import (
    GroupLassoFit,
    select_group_lasso_lambda_bic,
    select_group_lasso_lambda_cv,
)

InitStrategy = Literal["marginal", "random", "dense"]
InitCriterion = Literal["cv", "bic"]


def _mode_slice_scores(X: np.ndarray, y: np.ndarray, mode: int) -> np.ndarray:
    """score_k^(m) = || X_(m)^(k)^T y ||^2 / n."""
    n = X.shape[0]
    if mode == 1:
        return np.array(
            [np.linalg.norm(X[:, k, :, :].reshape(n, -1).T @ y) ** 2 / n for k in range(X.shape[1])]
        )
    if mode == 2:
        return np.array(
            [np.linalg.norm(X[:, :, j, :].reshape(n, -1).T @ y) ** 2 / n for j in range(X.shape[2])]
        )
    if mode == 3:
        return np.array(
            [np.linalg.norm(X[:, :, :, l].reshape(n, -1).T @ y) ** 2 / n for l in range(X.shape[3])]
        )
    raise ValueError("mode must be 1, 2, or 3")


def initial_support(
    X: np.ndarray,
    y: np.ndarray,
    s1: int,
    s2: int,
    s3: int,
    strategy: InitStrategy = "marginal",
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy support initializer retained for fixed-size BCD compatibility."""
    n, p1, p2, p3 = X.shape
    rng = np.random.default_rng(random_state)

    if strategy == "random":
        return (
            np.sort(rng.choice(p1, size=s1, replace=False)),
            np.sort(rng.choice(p2, size=s2, replace=False)),
            np.sort(rng.choice(p3, size=s3, replace=False)),
        )

    if strategy == "dense":
        return (
            np.sort(np.arange(min(s1, p1))),
            np.sort(np.arange(min(s2, p2))),
            np.sort(np.arange(min(s3, p3))),
        )

    score1 = _mode_slice_scores(X, y, 1)
    score2 = _mode_slice_scores(X, y, 2)
    score3 = _mode_slice_scores(X, y, 3)
    return (
        np.sort(np.argsort(-score1)[:s1]),
        np.sort(np.argsort(-score2)[:s2]),
        np.sort(np.argsort(-score3)[:s3]),
    )


@dataclass
class TensorInitResult:
    S1_init: np.ndarray
    S2_init: np.ndarray
    S3_init: np.ndarray
    s1_max: int
    s2_max: int
    s3_max: int
    C1: np.ndarray
    C2: np.ndarray
    C3: np.ndarray
    glasso_fits: dict[int, GroupLassoFit]


def _candidate_set(scores: np.ndarray, n: int) -> np.ndarray:
    size = min(int(np.floor(n ** (2.0 / 3.0))), len(scores))
    size = max(1, size)
    return np.sort(np.argsort(-scores)[:size])


def _top_from_candidates(scores: np.ndarray, candidates: np.ndarray, n: int) -> np.ndarray:
    size = min(max(int(np.floor(np.sqrt(n))), 1), len(candidates))
    order = np.argsort(-scores[candidates])[:size]
    return np.sort(candidates[order])


def _fallback_from_candidates(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Use the highest-scoring candidate when group Lasso selects nothing."""
    candidates = np.asarray(candidates, dtype=int)
    best_local = int(np.argmax(scores[candidates]))
    return np.array([int(candidates[best_local])], dtype=int)


def _select_group_lasso_fit(
    Z: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    block_size: int,
    criterion: InitCriterion,
    *,
    n_lambdas: int,
    cv_folds: int,
    max_iter: int,
    tol: float,
) -> GroupLassoFit:
    if criterion == "cv":
        return select_group_lasso_lambda_cv(
            Z,
            y,
            group_index,
            n_lambdas=n_lambdas,
            folds=cv_folds,
            max_iter=max_iter,
            tol=tol,
        )
    return select_group_lasso_lambda_bic(
        Z,
        y,
        group_index,
        block_size=block_size,
        n_lambdas=n_lambdas,
        max_iter=max_iter,
        tol=tol,
    )


def tensor_init_glasso(
    X: np.ndarray,
    y: np.ndarray,
    c: float = 2.0,
    *,
    criterion: InitCriterion = "cv",
    n_lambdas: int = 20,
    cv_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> TensorInitResult:
    """Algorithm 5.0: marginal pre-screen + sequential group-Lasso warm start."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape

    score1 = _mode_slice_scores(X, y, 1)
    score2 = _mode_slice_scores(X, y, 2)
    score3 = _mode_slice_scores(X, y, 3)

    C1 = _candidate_set(score1, n)
    C2 = _candidate_set(score2, n)
    C3 = _candidate_set(score3, n)

    S1_tilde = _top_from_candidates(score1, C1, n)
    S2_tilde = _top_from_candidates(score2, C2, n)
    S3_tilde = _top_from_candidates(score3, C3, n)
    glasso_fits: dict[int, GroupLassoFit] = {}

    for mode in (1, 2, 3):
        if mode == 1:
            candidates = C1
            Z, group_index, original = build_mode_design_subset(
                X, 1, candidates, S1_tilde, S2_tilde, S3_tilde
            )
            block_size = len(S2_tilde) * len(S3_tilde)
        elif mode == 2:
            candidates = C2
            Z, group_index, original = build_mode_design_subset(
                X, 2, candidates, S1_tilde, S2_tilde, S3_tilde
            )
            block_size = len(S1_tilde) * len(S3_tilde)
        else:
            candidates = C3
            Z, group_index, original = build_mode_design_subset(
                X, 3, candidates, S1_tilde, S2_tilde, S3_tilde
            )
            block_size = len(S1_tilde) * len(S2_tilde)

        fit = _select_group_lasso_fit(
            Z,
            y,
            group_index,
            max(block_size, 1),
            criterion,
            n_lambdas=n_lambdas,
            cv_folds=cv_folds,
            max_iter=max_iter,
            tol=tol,
        )
        glasso_fits[mode] = fit
        scores = {1: score1, 2: score2, 3: score3}[mode]
        selected = (
            original[fit.active_groups]
            if fit.active_groups.size
            else _fallback_from_candidates(scores, original)
        )
        selected = np.sort(np.asarray(selected, dtype=int))
        if mode == 1:
            S1_tilde = selected
        elif mode == 2:
            S2_tilde = selected
        else:
            S3_tilde = selected

    s_tilde = np.array([len(S1_tilde), len(S2_tilde), len(S3_tilde)], dtype=int)
    marginal_floor = max(1, int(np.floor(np.sqrt(n) / 2.0)))

    s1_max = min(max(int(np.ceil(c * len(S1_tilde))), marginal_floor), p1)
    s2_max = min(max(int(np.ceil(c * len(S2_tilde))), marginal_floor), p2)
    s3_max = min(max(int(np.ceil(c * len(S3_tilde))), marginal_floor), p3)
    s1_max = max(1, s1_max)
    s2_max = max(1, s2_max)
    s3_max = max(1, s3_max)

    max_rectangular_params = max(1, int(np.floor(0.7 * n)))
    while s1_max * s2_max * s3_max > max_rectangular_params:
        largest = int(np.argmax([s1_max, s2_max, s3_max]))
        if largest == 0 and s1_max > 1:
            s1_max -= 1
        elif largest == 1 and s2_max > 1:
            s2_max -= 1
        elif largest == 2 and s3_max > 1:
            s3_max -= 1
        else:
            break

    return TensorInitResult(
        S1_init=np.sort(S1_tilde),
        S2_init=np.sort(S2_tilde),
        S3_init=np.sort(S3_tilde),
        s1_max=s1_max,
        s2_max=s2_max,
        s3_max=s3_max,
        C1=np.sort(C1),
        C2=np.sort(C2),
        C3=np.sort(C3),
        glasso_fits=glasso_fits,
    )
