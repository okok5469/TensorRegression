"""Method wrappers used in the numerical simulation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from tensor_abess.tensor.abess import tensor_abess as tensor_abess_core
from tensor_abess.tensor.ops import (
    flatten_predictors,
    refit_tensor_coefficients,
    support_from_tensor,
    tensor_loss,
)
from tensor_abess.tensor.reduction import build_mode_design
from tensor_abess.utils.glasso import (
    GroupLassoFit,
    select_group_lasso_lambda_bic,
    select_group_lasso_lambda_cv,
)
from tensor_abess.utils.init import tensor_init_glasso

from numerical_simulation.data_generation import TensorSimulationConfig, TensorSimulationDataset


@dataclass
class MethodEstimate:
    """Common container for one fitted method."""

    name: str
    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    s1: int
    s2: int
    s3: int
    loss: float
    rss: float
    runtime_sec: float
    n_cycles: int
    sic: float | None = None
    details: dict[str, float | int | str] = field(default_factory=dict)


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _fit_lasso_ista(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    max_iter: int,
    tol: float,
    beta_init: np.ndarray | None = None,
) -> np.ndarray:
    n, p = X.shape
    if p == 0:
        return np.zeros(0, dtype=float)
    lipschitz = np.linalg.norm(X, ord=2) ** 2 / max(n, 1)
    step = 1.0 if lipschitz <= 0 else 1.0 / lipschitz
    beta = np.zeros(p, dtype=float) if beta_init is None else np.asarray(beta_init, dtype=float).copy()

    for _ in range(max_iter):
        grad = X.T @ (X @ beta - y) / max(n, 1)
        new_beta = _soft_threshold(beta - step * grad, lam * step)
        if np.linalg.norm(new_beta - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = new_beta
            break
        beta = new_beta
    return beta


def _lasso_lambda_grid(X: np.ndarray, y: np.ndarray, n_lambdas: int) -> np.ndarray:
    lam_max = float(np.max(np.abs(X.T @ y / max(len(y), 1))))
    if lam_max <= 0:
        return np.array([0.0], dtype=float)
    lam_min = lam_max * 1e-2
    return np.logspace(np.log10(lam_max), np.log10(lam_min), num=max(n_lambdas, 2))


def _fit_vector_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_lambdas: int,
    cv_folds: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n = len(y)
    folds = max(2, min(cv_folds, n))
    indices = np.arange(n)
    split = np.array_split(indices, folds)
    lambdas = _lasso_lambda_grid(X, y, n_lambdas)

    best_lam = float(lambdas[0])
    best_score = np.inf
    for lam in lambdas:
        score = 0.0
        beta_warm: np.ndarray | None = None
        for val_idx in split:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            beta = _fit_lasso_ista(
                X[train_mask],
                y[train_mask],
                float(lam),
                max_iter=max_iter,
                tol=tol,
                beta_init=beta_warm,
            )
            beta_warm = beta
            residual = y[val_idx] - X[val_idx] @ beta
            score += float(residual @ residual)
        if score < best_score:
            best_score = score
            best_lam = float(lam)

    return _fit_lasso_ista(X, y, best_lam, max_iter=max_iter, tol=tol)


def _select_group_lasso(
    Z: np.ndarray,
    y: np.ndarray,
    group_index: np.ndarray,
    *,
    block_size: int,
    criterion: str,
    n_lambdas: int,
    cv_folds: int,
    max_iter: int,
    tol: float,
) -> GroupLassoFit:
    if criterion == "bic":
        return select_group_lasso_lambda_bic(
            Z,
            y,
            group_index,
            block_size=block_size,
            n_lambdas=n_lambdas,
            max_iter=max_iter,
            tol=tol,
        )
    return select_group_lasso_lambda_cv(
        Z,
        y,
        group_index,
        n_lambdas=n_lambdas,
        folds=cv_folds,
        max_iter=max_iter,
        tol=tol,
    )


def _ensure_nonempty_support(selected: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    selected = np.sort(np.asarray(selected, dtype=int))
    if selected.size:
        return selected
    fallback = np.sort(np.asarray(fallback, dtype=int))
    if fallback.size:
        return fallback
    return np.array([0], dtype=int)


def run_tensor_abess_method(
    X: np.ndarray,
    y: np.ndarray,
    config: TensorSimulationConfig,
    *,
    random_state: int,
) -> MethodEstimate:
    """Run the project's Tensor-ABESS core solver."""
    start = perf_counter()
    target_sparsity = config.sparsity if config.tensor_abess_mode == "fixed" else None
    result = tensor_abess_core(
        X,
        y,
        target_sparsity=target_sparsity,
        k_max=config.k_max,
        c=config.abess_init_c,
        criterion=config.abess_init_criterion,
        n_lambdas=config.n_lambdas,
        cv_folds=config.cv_folds,
        max_iter=config.max_iter,
        tol=config.tol,
        max_cycles=config.max_cycles,
        eps=1e-8,
        adaptive_criterion="msic",
    )
    runtime = perf_counter() - start
    n_cycles = 1 if result.bcd_result is None else int(result.bcd_result.n_cycles)
    return MethodEstimate(
        name="Tensor-ABESS",
        B=result.B,
        S1=np.sort(result.S1),
        S2=np.sort(result.S2),
        S3=np.sort(result.S3),
        s1=int(result.s1),
        s2=int(result.s2),
        s3=int(result.s3),
        loss=float(result.loss),
        rss=float(result.rss),
        runtime_sec=float(runtime),
        n_cycles=n_cycles,
        sic=float(result.sic),
        details={"random_state": int(random_state), "tensor_abess_mode": config.tensor_abess_mode},
    )


def tensor_glasso(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    criterion: str = "cv",
    c: float = 2.0,
    n_lambdas: int = 20,
    cv_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
    max_cycles: int = 10,
) -> MethodEstimate:
    """Tensor grouped-lasso baseline under the same BCD outer loop."""
    del random_state  # deterministic conditional on (X, y)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]

    start = perf_counter()
    init = tensor_init_glasso(
        X,
        y,
        c=c,
        criterion="cv" if criterion not in {"cv", "bic"} else criterion,  # keep init aligned
        n_lambdas=n_lambdas,
        cv_folds=cv_folds,
        max_iter=max_iter,
        tol=tol,
    )
    S1 = np.sort(init.S1_init.copy())
    S2 = np.sort(init.S2_init.copy())
    S3 = np.sort(init.S3_init.copy())

    B = refit_tensor_coefficients(X, y, S1, S2, S3)
    loss = tensor_loss(X, y, B)
    n_cycles = 0

    for cycle in range(max_cycles):
        prev_S1 = S1.copy()
        prev_S2 = S2.copy()
        prev_S3 = S3.copy()

        Z1, gi1, _ = build_mode_design(X, 1, S1, S2, S3)
        fit1 = _select_group_lasso(
            Z1,
            y,
            gi1,
            block_size=max(len(S2) * len(S3), 1),
            criterion=criterion,
            n_lambdas=n_lambdas,
            cv_folds=cv_folds,
            max_iter=max_iter,
            tol=tol,
        )
        S1 = _ensure_nonempty_support(fit1.active_groups, prev_S1)

        Z2, gi2, _ = build_mode_design(X, 2, S1, S2, S3)
        fit2 = _select_group_lasso(
            Z2,
            y,
            gi2,
            block_size=max(len(S1) * len(S3), 1),
            criterion=criterion,
            n_lambdas=n_lambdas,
            cv_folds=cv_folds,
            max_iter=max_iter,
            tol=tol,
        )
        S2 = _ensure_nonempty_support(fit2.active_groups, prev_S2)

        Z3, gi3, _ = build_mode_design(X, 3, S1, S2, S3)
        fit3 = _select_group_lasso(
            Z3,
            y,
            gi3,
            block_size=max(len(S1) * len(S2), 1),
            criterion=criterion,
            n_lambdas=n_lambdas,
            cv_folds=cv_folds,
            max_iter=max_iter,
            tol=tol,
        )
        S3 = _ensure_nonempty_support(fit3.active_groups, prev_S3)

        B = refit_tensor_coefficients(X, y, S1, S2, S3)
        loss = tensor_loss(X, y, B)
        n_cycles = cycle + 1

        if (
            np.array_equal(S1, prev_S1)
            and np.array_equal(S2, prev_S2)
            and np.array_equal(S3, prev_S3)
        ):
            break

    rss = float(2.0 * n * loss)
    runtime = perf_counter() - start
    return MethodEstimate(
        name="Tensor-GLasso",
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        s1=int(len(S1)),
        s2=int(len(S2)),
        s3=int(len(S3)),
        loss=float(loss),
        rss=rss,
        runtime_sec=float(runtime),
        n_cycles=max(n_cycles, 1),
        details={"criterion": criterion},
    )


def run_vector_lasso_method(
    X: np.ndarray,
    y: np.ndarray,
    config: TensorSimulationConfig,
) -> MethodEstimate:
    """Run elementwise lasso on the vectorized tensor design with a NumPy CV solver."""
    start = perf_counter()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    _, p1, p2, p3 = X.shape
    Xf = flatten_predictors(X)
    beta = _fit_vector_lasso_cv(
        Xf,
        y,
        n_lambdas=max(config.n_lambdas, 2),
        cv_folds=config.cv_folds,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    B = beta.reshape(p1, p2, p3)
    S1, S2, S3 = support_from_tensor(B, tol=1e-8)
    loss = tensor_loss(X, y, B)
    runtime = perf_counter() - start
    return MethodEstimate(
        name="Vector Lasso",
        B=B,
        S1=np.sort(S1),
        S2=np.sort(S2),
        S3=np.sort(S3),
        s1=int(len(S1)),
        s2=int(len(S2)),
        s3=int(len(S3)),
        loss=float(loss),
        rss=float(2.0 * len(y) * loss),
        runtime_sec=float(runtime),
        n_cycles=1,
    )


def run_oracle_ols_method(
    X: np.ndarray,
    y: np.ndarray,
    supports_true: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> MethodEstimate:
    """Run OLS on the true rectangular support."""
    start = perf_counter()
    S1_true, S2_true, S3_true = supports_true
    B = refit_tensor_coefficients(X, y, S1_true, S2_true, S3_true)
    loss = tensor_loss(X, y, B)
    runtime = perf_counter() - start
    n = len(y)
    return MethodEstimate(
        name="Oracle OLS",
        B=B,
        S1=np.sort(S1_true),
        S2=np.sort(S2_true),
        S3=np.sort(S3_true),
        s1=int(len(S1_true)),
        s2=int(len(S2_true)),
        s3=int(len(S3_true)),
        loss=float(loss),
        rss=float(2.0 * n * loss),
        runtime_sec=float(runtime),
        n_cycles=1,
    )


def run_all_methods(
    dataset: TensorSimulationDataset,
    *,
    random_state: int,
) -> list[MethodEstimate]:
    """Run the four main methods used in the simulation study."""
    X = dataset.X_train
    y = dataset.y_train
    return [
        run_tensor_abess_method(X, y, dataset.config, random_state=random_state),
        tensor_glasso(
            X,
            y,
            random_state=random_state,
            criterion="cv",
            n_lambdas=dataset.config.n_lambdas,
            cv_folds=dataset.config.cv_folds,
            max_iter=dataset.config.max_iter,
            tol=dataset.config.tol,
            max_cycles=dataset.config.max_cycles,
        ),
        run_vector_lasso_method(X, y, dataset.config),
        run_oracle_ols_method(X, y, dataset.supports_true),
    ]
