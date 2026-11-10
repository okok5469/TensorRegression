"""Focused numerical and data-boundary checks for the closeout adapters."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np


def api(module: str):
    name = f"research.taskcard18.{module}"
    assert importlib.util.find_spec(name) is not None, f"Missing isolated {module} implementation"
    return importlib.import_module(name)


class ConvexSolverTests(unittest.TestCase):
    def test_lasso_certifies_initial_state_and_checks_within_fifty_sweeps(self):
        solver = api("solvers")
        Z = 2.0 * np.eye(4)
        y = Z @ np.array([1.2, -0.7, 0.1, 0.0])
        initial = solver.solve(Z, y, 1.5)
        self.assertEqual(initial["diagnostic"]["iterations"], 0)
        self.assertEqual(initial["diagnostic"]["certificate_check_iterations"], [0])
        fitted = solver.solve(Z, y, 0.2)
        diagnostic = fitted["diagnostic"]
        self.assertTrue(diagnostic["converged"])
        self.assertLessEqual(diagnostic["iterations"], 50)
        self.assertEqual(diagnostic["library_tolerance"], 0.0)
        checks = diagnostic["certificate_check_iterations"]
        self.assertEqual(checks[0], 0)
        self.assertEqual(checks[-1], diagnostic["iterations"])
        self.assertTrue(np.all(np.diff(checks) <= 50))

    def test_lasso_matches_orthogonal_soft_threshold_solution(self):
        solver = api("solvers")
        Z = np.sqrt(4.0) * np.eye(4)
        target = np.array([1.2, -0.7, 0.1, 0.0])
        y = Z @ target
        result = solver.solve(Z, y, 0.2)
        np.testing.assert_allclose(result["beta"], [1.0, -0.5, 0.0, 0.0], atol=1e-10)
        self.assertTrue(result["diagnostic"]["converged"])
        self.assertLessEqual(result["diagnostic"]["kkt_ratio"], 1e-4)
        self.assertLessEqual(result["diagnostic"]["gap_normalized"], 1e-6)

    def test_group_lasso_matches_orthogonal_block_solution(self):
        solver = api("solvers")
        Z = 2.0 * np.eye(4)
        groups = np.array([4, 4, 9, 9])
        target = np.array([0.6, 0.8, 0.03, -0.04])
        result = solver.solve(Z, Z @ target, 0.2, groups=groups)
        np.testing.assert_allclose(result["beta"], [0.48, 0.64, 0.0, 0.0], atol=1e-9)
        self.assertTrue(result["diagnostic"]["converged"])

    def test_rank_deficient_group_fit_agrees_with_independent_dual(self):
        solver = api("solvers")
        rng = np.random.default_rng(73)
        Z = rng.normal(size=(9, 12))
        Z[:, -1] = Z[:, 0]
        y = rng.normal(size=9)
        groups = np.repeat(np.arange(4), 3)
        result = solver.solve(Z, y, 0.18, groups=groups)
        dual = solver.solve_group_dual(Z, y, 0.18, groups)
        self.assertTrue(result["diagnostic"]["converged"])
        self.assertTrue(dual["success"], dual["message"])
        self.assertLess(abs(result["diagnostic"]["objective"] - dual["dual_objective"]), 2e-6)
        np.testing.assert_allclose(Z @ result["beta"], dual["fitted_values"], atol=3e-3)

    def test_lasso_agrees_with_independent_nonnegative_split_solver(self):
        solver = api("solvers")
        self.assertTrue(hasattr(solver, "solve_lasso_split"), "Missing independent Lasso split solver")
        rng = np.random.default_rng(74)
        Z = rng.normal(size=(8, 13))
        Z[:, -1] = Z[:, 0]
        y = rng.normal(size=8)
        result = solver.solve(Z, y, 0.12)
        independent = solver.solve_lasso_split(Z, y, 0.12)
        self.assertTrue(independent["diagnostic"]["converged"])
        self.assertLess(abs(result["diagnostic"]["objective"] - independent["diagnostic"]["objective"]), 1e-7)
        np.testing.assert_allclose(Z @ result["beta"], Z @ independent["beta"], atol=2e-4)

    def test_zero_penalty_design_response_and_empty_columns_have_certificates(self):
        solver = api("solvers")
        Z = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]])
        y = np.array([1.0, -2.0, 0.5])
        for design, response, lam in [(Z, y, 0.0), (Z, np.zeros(3), 0.4), (np.zeros_like(Z), y, 0.4), (Z[:, :0], y, 0.4)]:
            with self.subTest(shape=design.shape, lam=lam):
                result = solver.solve(design, response, lam)
                self.assertTrue(result["diagnostic"]["converged"])
                self.assertTrue(np.isfinite(result["diagnostic"]["objective"]))
        least_squares = solver.solve(Z, y, 0.0)
        np.testing.assert_allclose(Z.T @ (Z @ least_squares["beta"] - y), 0.0, atol=1e-12)

    def test_kkt_uses_actual_nonzero_coefficients(self):
        solver = api("solvers")
        Z = np.eye(2)
        diagnostic = solver.certify(Z, np.array([0.0, 1.0]), np.array([1e-12, 0.0]), 0.25)
        self.assertGreaterEqual(diagnostic["kkt_ratio"], 1.0)
        self.assertFalse(diagnostic["converged"])


class BaselineBoundaryTests(unittest.TestCase):
    def test_budget_failure_preserves_the_entire_diagnostic_path(self):
        solvers = api("solvers")
        baselines = api("baselines")
        rng = np.random.default_rng(31)
        X = rng.normal(size=(12, 2, 2, 2))
        y = rng.normal(size=12)

        def fail_on_call(target):
            calls = []

            def limited_solver(design, response, lam, groups=None, beta_init=None, **kwargs):
                fitted = solvers.solve(design, response, lam, groups=groups, beta_init=beta_init, **kwargs)
                calls.append(None)
                if len(calls) == target:
                    fitted["diagnostic"].update(converged=False, at_limit=True, stop_reason="iteration_limit")
                return fitted

            return limited_solver

        with self.assertRaises(solvers.SolverFailure) as caught:
            baselines.select_lambda_cv(X.reshape(12, -1), y, lambdas=np.array([0.3, 0.1]),
                                       cv_folds=2, solver=fail_on_call(4))
        self.assertIn("completed_diagnostics", caught.exception.diagnostic)
        history = caught.exception.diagnostic["completed_diagnostics"]
        self.assertEqual(len(history), 4)
        self.assertTrue(all(item["converged"] for item in history[:-1]))
        self.assertFalse(history[-1]["converged"])
        self.assertEqual(history[-1]["fold_index"], 1)
        config = SimpleNamespace(n_lambdas=2, cv_folds=2, max_cycles=1)
        with self.assertRaises(solvers.SolverFailure) as caught:
            baselines.sequential_tensor_glasso(X, y, config, solver=fail_on_call(7))
        history = caught.exception.diagnostic["completed_diagnostics"]
        self.assertEqual(len(history), 7)
        self.assertEqual([item["mode"] for item in history], [1] * 5 + [2] * 2)
        self.assertTrue(all(item["phase"] == "initialization" for item in history))
        self.assertFalse(history[-1]["converged"])

    def test_cv_reuses_only_previous_lambda_from_the_same_fold(self):
        solvers = api("solvers")
        baselines = api("baselines")
        rng = np.random.default_rng(11)
        Z = rng.normal(size=(18, 5))
        y = rng.normal(size=18)
        calls = []

        def recording_solver(design, response, lam, groups=None, beta_init=None, **kwargs):
            result = solvers.solve(design, response, lam, groups=groups, beta_init=beta_init, **kwargs)
            calls.append((response.copy(), None if beta_init is None else beta_init.copy(), result["beta"].copy()))
            return result

        result = baselines.select_lambda_cv(Z, y, lambdas=np.array([0.3, 0.1]), cv_folds=3, solver=recording_solver)
        self.assertEqual(len(result["diagnostics"]), 7)
        self.assertEqual(len(calls), 7)
        fold_history = {}
        for response, beta_init, beta in calls[:-1]:
            key = response.tobytes()
            previous = fold_history.get(key)
            self.assertEqual(beta_init is None, previous is None)
            if previous is not None:
                np.testing.assert_array_equal(beta_init, previous)
            fold_history[key] = beta
        self.assertIsNone(calls[-1][1])

    def test_cv_uses_validation_sse_and_first_descending_lambda_on_ties(self):
        baselines = api("baselines")
        Z = np.zeros((11, 3))
        y = np.arange(11, dtype=float)
        result = baselines.select_lambda_cv(Z, y, lambdas=np.array([0.5, 0.1]), cv_folds=5)
        self.assertEqual(result["selected_lambda"], 0.5)
        np.testing.assert_allclose(result["cv_scores"], np.repeat(y @ y, 2))
        self.assertEqual(len(result["diagnostics"]), 11)

    def test_tensor_adapters_preserve_l1_and_final_rectangular_ols(self):
        baselines = api("baselines")
        from tensor_abess.tensor.ops import build_tensor_design
        rng = np.random.default_rng(22)
        X = rng.normal(size=(30, 3, 2, 2))
        y = 1.5 * X[:, 1, 0, 1] + 0.1 * rng.normal(size=30)
        config = SimpleNamespace(n_lambdas=3, cv_folds=3, max_cycles=2)
        vector = baselines.vector_lasso_cv(X, y, config)
        group = baselines.sequential_tensor_glasso(X, y, config)
        self.assertEqual(vector["method_id"], "vector_lasso_cv_isolated_v2")
        self.assertEqual(group["method_id"], "sequential_tensor_glasso_certified_v2")
        self.assertEqual(len(vector["diagnostics"]), 10)
        self.assertTrue(all(d["converged"] for d in group["diagnostics"]))
        self.assertTrue(all(len(S) > 0 for S in group["supports"]))
        design = build_tensor_design(X, *group["supports"])
        residual = y - np.einsum("nijk,ijk->n", X, group["B"])
        np.testing.assert_allclose(design.T @ residual, 0.0, atol=1e-9)
        self.assertGreater(np.linalg.norm(y - np.einsum("nijk,ijk->n", X, vector["B"])), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
