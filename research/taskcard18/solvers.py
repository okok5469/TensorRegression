"""Isolated convex solvers with independently computed optimality certificates.

All objectives are RSS/(2*n) + lambda * Omega(beta), without an intercept,
centering, feature standardization, or group weights.  A certificate always
uses the current fitting subset's n and actual nonzero coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import minimize


@dataclass(frozen=True)
class ConvexProblem:
    Z: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None
    inverse_groups: np.ndarray | None
    n_groups: int
    lipschitz: float | None
    gram: np.ndarray | None
    cross: np.ndarray | None


class SolverFailure(RuntimeError):
    """A required fit has exhausted the frozen numerical budget."""

    def __init__(self, diagnostic: dict, *, diagnostics=None):
        super().__init__(f"Uncertified convex fit: {diagnostic.get('stop_reason')}")
        summary = {key: value for key, value in diagnostic.items() if key != "completed_diagnostics"}
        history = [summary] if diagnostics is None else diagnostics
        self.diagnostic = dict(summary, completed_diagnostics=[
            {key: value for key, value in item.items() if key != "completed_diagnostics"} for item in history
        ])


def prepare_problem(Z, y, groups=None) -> ConvexProblem:
    """Reuse response/design calculations only within this explicit fit subset."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if Z.ndim != 2 or Z.shape[0] != len(y) or len(y) == 0:
        raise ValueError("A nonempty response and matching two-dimensional design are required")
    if not np.isfinite(Z).all() or not np.isfinite(y).all():
        raise ValueError("Design and response must be finite")
    if groups is None:
        return ConvexProblem(np.asfortranarray(Z), y, None, None, Z.shape[1], None, None, None)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != Z.shape[1]:
        raise ValueError("Every coefficient must belong to one disjoint group")
    unique, inverse = np.unique(groups, return_inverse=True)
    n, p = Z.shape
    small_gram = Z.T @ Z / n if p <= n else None
    spectral_gram = small_gram if small_gram is not None else Z @ Z.T / n
    lipschitz = float(eigvalsh(spectral_gram, subset_by_index=[len(spectral_gram) - 1] * 2, check_finite=False)[0]) if p else 0.0
    cross = Z.T @ y / n if small_gram is not None else None
    return ConvexProblem(Z, y, groups.copy(), inverse, len(unique), max(lipschitz, 0.0), small_gram, cross)


def _group_norms(vector: np.ndarray, inverse: np.ndarray, count: int) -> np.ndarray:
    return np.sqrt(np.bincount(inverse, weights=vector * vector, minlength=count))


def certify(Z, y, beta, lam, groups=None, *, kkt_tolerance=1e-4, gap_tolerance=1e-6) -> dict:
    """Compute primal/dual objectives, feasibility, and the actual KKT residual."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    n = len(y)
    if n == 0 or Z.shape != (n, len(beta)) or lam < 0 or not np.isfinite(lam):
        raise ValueError("Invalid certificate dimensions or penalty")
    residual = y - Z @ beta
    gradient = -Z.T @ residual / n
    base = max(float(y @ y / (2.0 * n)), 1e-12)
    if groups is None:
        penalty = float(np.abs(beta).sum())
        dual_norm = float(np.max(np.abs(gradient), initial=0.0))
        active = beta != 0.0
        violations = np.maximum(np.abs(gradient) - lam, 0.0)
        violations[active] = np.abs(gradient[active] + lam * np.sign(beta[active]))
    else:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        count = int(np.max(inverse, initial=-1)) + 1
        norms = _group_norms(beta, inverse, count)
        grad_norms = _group_norms(gradient, inverse, count)
        penalty = float(norms.sum())
        dual_norm = float(np.max(grad_norms, initial=0.0))
        active = norms > 0.0
        adjusted = gradient.copy()
        scaling = np.divide(lam, norms, out=np.zeros_like(norms), where=active)
        adjusted += beta * scaling[inverse]
        violations = np.maximum(grad_norms - lam, 0.0)
        violations[active] = _group_norms(adjusted, inverse, count)[active]
    kkt = float(np.max(violations, initial=0.0))
    objective = float(residual @ residual / (2.0 * n) + lam * penalty)
    if lam == 0.0:
        # The zero-penalty dual is not obtained by dividing by lambda.
        scale = max(float(np.linalg.norm(Z, ord="fro") * np.linalg.norm(y) / n), 1.0)
        gradient_normalized = float(np.max(np.abs(gradient), initial=0.0) / scale)
        return {"objective": objective, "dual_objective": None, "gap": None, "gap_normalized": None,
                "kkt": kkt, "kkt_ratio": None, "gradient_normalized": gradient_normalized,
                "dual_feasible": None, "negative_gap_error": False, "converged": bool(gradient_normalized <= 1e-8),
                "lambda": float(lam), "n_fit": n, "penalty_norm": penalty, "certificate": "least_squares_gradient"}
    scaling = max(1.0, dual_norm / lam)
    theta = residual / (n * scaling)
    dual = float(y @ theta - n * (theta @ theta) / 2.0)
    gap = objective - dual
    normalized_gap = gap / base
    dual_feasible = bool(dual_norm / scaling <= lam * (1.0 + 1e-10))
    negative_gap_error = bool(normalized_gap < -1e-10)
    ratio = kkt / lam
    converged = bool(np.isfinite([objective, dual, ratio, normalized_gap]).all() and dual_feasible
                     and not negative_gap_error and ratio <= kkt_tolerance and normalized_gap <= gap_tolerance)
    return {"objective": objective, "dual_objective": dual, "gap": gap, "gap_normalized": normalized_gap,
            "kkt": kkt, "kkt_ratio": ratio, "gradient_normalized": None,
            "dual_feasible": dual_feasible, "negative_gap_error": negative_gap_error,
            "converged": converged, "lambda": float(lam), "n_fit": n, "penalty_norm": penalty,
            "certificate": "kkt_and_primal_dual", "dual_norm": dual_norm / scaling}


def solve(Z, y, lam, groups=None, beta_init=None, *, budgets=(2000, 10000, 50000), problem=None) -> dict:
    """Solve one fixed objective; return failure evidence instead of hiding it.

    Budgets are cumulative iteration ceilings.  The caller must reject an
    uncertified result before using it for CV or test evaluation.
    """
    started = perf_counter()
    problem = prepare_problem(Z, y, groups) if problem is None else problem
    Z, y, groups = problem.Z, problem.y, problem.groups
    lam = float(lam)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda must be finite and nonnegative")
    budgets = tuple(int(value) for value in budgets)
    if not budgets or len(budgets) > 3 or budgets[0] <= 0 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Use one to three strictly increasing positive iteration ceilings")
    p = Z.shape[1]
    beta = np.zeros(p) if beta_init is None else np.asarray(beta_init, dtype=float).copy()
    if beta.shape != (p,) or not np.isfinite(beta).all():
        raise ValueError("Invalid initial coefficients")
    iterations = 0
    attempts = []
    lasso_checks = None
    solver_name = "analytic_zero"
    special = p == 0 or not np.any(Z) or not np.any(y)
    if special:
        beta = np.zeros(p)
    elif lam == 0.0:
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        solver_name = "numpy_lstsq"
    elif groups is None:
        from sklearn.linear_model import Lasso
        from sklearn.exceptions import ConvergenceWarning
        solver_name = "sklearn_coordinate_descent"
        diagnostic = certify(Z, y, beta, lam)
        lasso_checks = [0]
        estimator = Lasso(alpha=lam, fit_intercept=False, precompute=False, copy_X=False,
                          tol=0.0, warm_start=True, selection="cyclic")
        estimator.coef_ = beta
        estimator.intercept_ = 0.0
        for budget in budgets:
            if diagnostic["converged"]:
                break
            stage_start = iterations
            stage_warnings = []
            while iterations < budget and not diagnostic["converged"]:
                estimator.max_iter = min(50, budget - iterations)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(Z, y, check_input=False)
                beta = estimator.coef_.copy()
                iterations += int(estimator.n_iter_)
                diagnostic = certify(Z, y, beta, lam)
                lasso_checks.append(iterations)
                stage_warnings.extend(str(item.message) for item in caught)
            attempts.append({"budget": budget, "iterations": iterations - stage_start,
                             "converged": diagnostic["converged"], "warnings": stage_warnings})
    else:
        solver_name = "restarted_fista"
        inverse, count = problem.inverse_groups, problem.n_groups
        step = 1.0 / problem.lipschitz
        extrapolated = beta.copy()
        momentum = 1.0
        previous_budget = 0
        for budget in budgets:
            for iteration in range(previous_budget + 1, budget + 1):
                gradient = (problem.gram @ extrapolated - problem.cross) if problem.gram is not None else Z.T @ (Z @ extrapolated - y) / len(y)
                proposal = extrapolated - step * gradient
                norms = _group_norms(proposal, inverse, count)
                shrink = np.maximum(1.0 - np.divide(lam * step, norms, out=np.full_like(norms, np.inf), where=norms > 0.0), 0.0)
                new_beta = proposal * shrink[inverse]
                new_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
                restart = float((extrapolated - new_beta) @ (new_beta - beta)) > 0.0
                weight = 0.0 if restart else (momentum - 1.0) / new_momentum
                extrapolated = new_beta + weight * (new_beta - beta)
                beta = new_beta
                momentum = 1.0 if restart else new_momentum
                iterations = iteration
                if iteration == 1 or iteration % 10 == 0 or iteration == budget:
                    diagnostic = certify(Z, y, beta, lam, groups)
                    if diagnostic["converged"]:
                        break
            attempts.append({"budget": budget, "iterations": iterations - previous_budget,
                             "converged": diagnostic["converged"], "warnings": []})
            if diagnostic["converged"]:
                break
            previous_budget = budget
    diagnostic = certify(Z, y, beta, lam, groups)
    at_limit = bool(not diagnostic["converged"] and iterations >= budgets[-1])
    numerical_error = bool(diagnostic["negative_gap_error"] or not np.isfinite(beta).all())
    stop_reason = "iteration_limit" if at_limit else "solver_stopped_uncertified"
    stop_reason = "numerical_error" if numerical_error else stop_reason
    stop_reason = "certificate" if diagnostic["converged"] else stop_reason
    diagnostic.update({"iterations": iterations, "solver": solver_name, "attempts": attempts,
                       "at_limit": at_limit, "stop_reason": stop_reason,
                       "numerical_error": numerical_error, "runtime_sec": perf_counter() - started})
    if lasso_checks is not None:
        diagnostic.update({"certificate_check_iterations": lasso_checks, "check_interval": 50,
                           "library_tolerance": 0.0})
    return {"beta": beta, "diagnostic": diagnostic}


def solve_group_dual(Z, y, lam, groups, *, max_iter=2000) -> dict:
    """Independent smooth constrained dual, intended for bounded pilot audits.

    The variable is u=n*theta.  Every group constraint is
    ||Z_g.T @ u / n||^2 <= lambda^2.  No primal solver state is used.
    """
    Z, y = np.asarray(Z, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    groups = np.asarray(groups)
    n = len(y)
    if lam <= 0.0:
        raise ValueError("The constrained-dual audit requires a positive penalty")
    gram_groups = np.stack([Z[:, groups == group] @ Z[:, groups == group].T / (n * n) for group in np.unique(groups)])
    base = max(float(y @ y / (2 * n)), 1e-12)

    def objective(u):
        return float((u @ u / 2.0 - y @ u) / (n * base))

    def jacobian(u):
        return (u - y) / (n * base)

    def constraints(u):
        return 1.0 - np.einsum("i,gij,j->g", u, gram_groups, u) / (lam * lam)

    def constraint_jacobian(u):
        return -2.0 * np.einsum("gij,j->gi", gram_groups, u) / (lam * lam)

    result = minimize(objective, np.zeros(n), jac=jacobian, method="SLSQP",
                      constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                      options={"ftol": 1e-12, "maxiter": int(max_iter)})
    feasible = bool(np.min(constraints(result.x), initial=0.0) >= -1e-7)
    return {"theta": result.x / n, "fitted_values": y - result.x,
            "dual_objective": float(-result.fun * base), "success": bool(result.success and feasible),
            "feasible": feasible, "iterations": int(result.nit), "message": str(result.message)}


def solve_lasso_split(Z, y, lam, *, max_iter=50000) -> dict:
    """Independent L-BFGS-B audit of beta = positive - negative, both >= 0."""
    problem = prepare_problem(Z, y)
    Z, y = problem.Z, problem.y
    n, p = Z.shape
    if lam <= 0.0 or p == 0:
        raise ValueError("The split audit requires positive lambda and at least one column")

    def objective_and_gradient(split):
        beta = split[:p] - split[p:]
        residual = Z @ beta - y
        gradient = Z.T @ residual / n
        value = float(residual @ residual / (2.0 * n) + lam * split.sum())
        return value, np.concatenate((gradient + lam, -gradient + lam))

    result = minimize(objective_and_gradient, np.zeros(2 * p), jac=True, method="L-BFGS-B",
                      bounds=[(0.0, None)] * (2 * p),
                      options={"maxiter": int(max_iter), "maxfun": int(max_iter) * 3,
                               "ftol": 1e-15, "gtol": 1e-10, "maxls": 40})
    beta = result.x[:p] - result.x[p:]
    diagnostic = certify(Z, y, beta, float(lam))
    diagnostic.update({"solver": "scipy_nonnegative_split_lbfgsb", "iterations": int(result.nit),
                       "solver_success": bool(result.success), "message": str(result.message)})
    return {"beta": beta, "diagnostic": diagnostic}
