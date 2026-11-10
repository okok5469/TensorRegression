"""Certified replacements for the historical simulation's two baselines.

The frozen production files and historical run directories are never patched.
CV fitting state is scoped to one fold; lambda grids and local tensor designs
still use the complete current training data as in the original definitions.
"""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np

from tensor_abess.tensor.ops import flatten_predictors, refit_tensor_coefficients, support_from_tensor, tensor_loss
from tensor_abess.tensor.reduction import build_mode_design, build_mode_design_subset
from tensor_abess.utils.init import _candidate_set, _fallback_from_candidates, _mode_slice_scores, _top_from_candidates

from research.taskcard18.solvers import SolverFailure, prepare_problem, solve


def _array_hash(value: np.ndarray) -> str:
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def lambda_grid(Z, y, groups=None, n_lambdas=20) -> np.ndarray:
    n = len(y)
    cross = np.asarray(Z).T @ np.asarray(y) / n
    if groups is None:
        maximum = float(np.max(np.abs(cross), initial=0.0))
    else:
        maximum = max((float(np.linalg.norm(cross[np.asarray(groups) == group])) for group in np.unique(groups)), default=0.0)
    if maximum <= 0.0:
        return np.array([0.0])
    return np.logspace(np.log10(maximum), np.log10(maximum * 1e-2), num=max(int(n_lambdas), 2))


def select_lambda_cv(Z, y, groups=None, *, lambdas=None, n_lambdas=20, cv_folds=5,
                     budgets=(2000, 10000, 50000), solver=solve) -> dict:
    """Contiguous folds, pooled validation SSE, first/larger lambda on ties."""
    Z, y = np.asarray(Z, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    if n < 2:
        raise ValueError("At least two observations are required for CV")
    lambdas = lambda_grid(Z, y, groups, n_lambdas) if lambdas is None else np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1 or len(lambdas) == 0 or np.any(np.diff(lambdas) > 0.0) or np.any(lambdas < 0.0) or not np.isfinite(lambdas).all():
        raise ValueError("CV lambdas must be a finite nonnegative descending grid")
    split = np.array_split(np.arange(n), max(2, min(int(cv_folds), n)))
    scores = np.zeros(len(lambdas))
    diagnostics = []
    # The loop is deliberately fold-first: beta never outlives its fit subset.
    for fold_index, validation in enumerate(split):
        training = np.setdiff1d(np.arange(n), validation, assume_unique=True)
        problem = prepare_problem(Z[training], y[training], groups)
        beta_warm = None
        for lambda_index, lam in enumerate(lambdas):
            fitted = solver(problem.Z, problem.y, float(lam), groups=groups, beta_init=beta_warm,
                            budgets=budgets, problem=problem)
            diagnostic = dict(fitted["diagnostic"], fit_kind="cv", fold_index=fold_index,
                              lambda_index=lambda_index, train_indices_hash=_array_hash(training),
                              train_response_hash=_array_hash(problem.y), beta_hash=_array_hash(fitted["beta"]),
                              warm_start_from=None if beta_warm is None else lambda_index - 1)
            diagnostics.append(diagnostic)
            if not diagnostic["converged"]:
                raise SolverFailure(diagnostic, diagnostics=diagnostics)
            beta_warm = fitted["beta"].copy()
            residual = y[validation] - Z[validation] @ beta_warm
            sse = float(residual @ residual)
            diagnostic["validation_sse"] = sse
            scores[lambda_index] += sse
    selected = int(np.argmin(scores))
    fitted = solver(Z, y, float(lambdas[selected]), groups=groups, beta_init=None, budgets=budgets)
    final = dict(fitted["diagnostic"], fit_kind="final", fold_index=None, lambda_index=selected,
                 train_indices_hash=_array_hash(np.arange(n)), train_response_hash=_array_hash(y),
                 beta_hash=_array_hash(fitted["beta"]), warm_start_from=None)
    diagnostics.append(final)
    if not final["converged"]:
        raise SolverFailure(final, diagnostics=diagnostics)
    return {"beta": fitted["beta"], "diagnostics": diagnostics, "selected_lambda": float(lambdas[selected]),
            "selected_index": selected, "lambda_grid": lambdas.tolist(), "cv_scores": scores.tolist(),
            "boundary": "upper" if selected == 0 else ("lower" if selected == len(lambdas) - 1 else "interior")}


def vector_lasso_cv(X, y, config, *, budgets=(2000, 10000, 50000)) -> dict:
    """Elementwise L1 estimate; no post-selection OLS."""
    started = perf_counter()
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    selected = select_lambda_cv(flatten_predictors(X), y, n_lambdas=config.n_lambdas,
                                cv_folds=config.cv_folds, budgets=budgets)
    B = selected["beta"].reshape(X.shape[1:])
    supports = tuple(np.sort(S) for S in support_from_tensor(B, tol=1e-8))
    loss = tensor_loss(X, y, B)
    return {"method_id": "vector_lasso_cv_isolated_v2", "B": B, "supports": supports,
            "diagnostics": selected["diagnostics"], "selected_lambdas": [{key: selected[key] for key in
             ("selected_lambda", "selected_index", "lambda_grid", "cv_scores", "boundary")}],
            "loss": float(loss), "rss": float(2 * len(y) * loss), "n_cycles": 1,
            "runtime_sec": perf_counter() - started, "valid": True}


def _active_groups(beta: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.array([group for group in np.unique(groups) if np.linalg.norm(beta[groups == group]) > 1e-8], dtype=int)


def sequential_tensor_glasso(X, y, config, *, budgets=(2000, 10000, 50000), solver=solve) -> dict:
    """Original ordered local-CV/support procedure with certified convex fits."""
    started = perf_counter()
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    scores = [_mode_slice_scores(X, y, mode) for mode in (1, 2, 3)]
    candidates = [_candidate_set(score, n) for score in scores]
    supports = [_top_from_candidates(score, candidate, n) for score, candidate in zip(scores, candidates)]
    diagnostics, selected_lambdas, support_history = [], [], []

    def local_fit(Z, group_index, phase, cycle, mode):
        context = {"phase": phase, "cycle": cycle, "mode": mode}
        try:
            selected = select_lambda_cv(Z, y, group_index, n_lambdas=config.n_lambdas,
                                        cv_folds=config.cv_folds, budgets=budgets, solver=solver)
        except SolverFailure as error:
            current = [dict(item, **context) for item in error.diagnostic["completed_diagnostics"]]
            raise SolverFailure(dict(error.diagnostic, **context), diagnostics=[*diagnostics, *current]) from error
        diagnostics.extend(dict(diagnostic, **context) for diagnostic in selected["diagnostics"])
        selected_lambdas.append(dict({key: selected[key] for key in
            ("selected_lambda", "selected_index", "lambda_grid", "cv_scores", "boundary")}, **context))
        return _active_groups(selected["beta"], group_index)

    for axis in range(3):
        Z, group_index, original = build_mode_design_subset(X, axis + 1, candidates[axis], *supports)
        active = local_fit(Z, group_index, "initialization", 0, axis + 1)
        supports[axis] = np.sort(original[active] if active.size else _fallback_from_candidates(scores[axis], original))
        support_history.append({"phase": "initialization", "cycle": 0, "mode": axis + 1,
                                "supports": [S.tolist() for S in supports]})
    initialization_supports = [S.tolist() for S in supports]
    # Caps computed by the production initializer do not constrain this baseline.
    B = refit_tensor_coefficients(X, y, *supports)
    n_cycles = 0
    for cycle in range(int(config.max_cycles)):
        previous = [S.copy() for S in supports]
        for axis in range(3):
            Z, group_index, _ = build_mode_design(X, axis + 1, *supports)
            active = local_fit(Z, group_index, "outer", cycle + 1, axis + 1)
            fallback = previous[axis] if previous[axis].size else np.array([0], dtype=int)
            supports[axis] = np.sort(active if active.size else fallback)
            support_history.append({"phase": "outer", "cycle": cycle + 1, "mode": axis + 1,
                                    "supports": [S.tolist() for S in supports]})
        B = refit_tensor_coefficients(X, y, *supports)
        n_cycles = cycle + 1
        if all(np.array_equal(current, old) for current, old in zip(supports, previous)):
            break
    loss = tensor_loss(X, y, B)
    # v2 binds this run to the revised protocol; the group solver is unchanged.
    return {"method_id": "sequential_tensor_glasso_certified_v2", "B": B, "supports": tuple(supports),
            "diagnostics": diagnostics, "selected_lambdas": selected_lambdas,
            "initialization_supports": initialization_supports, "support_history": support_history,
            "loss": float(loss), "rss": float(2 * n * loss), "n_cycles": max(n_cycles, 1),
            "runtime_sec": perf_counter() - started, "valid": True}
