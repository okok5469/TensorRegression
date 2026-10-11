"""Fixed-size BCD regression guards; no claim of general global optimality."""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tensor_abess.tensor.bcd import tensor_bcd
from tensor_abess.tensor.ops import build_tensor_design, tensor_inner, tensor_loss


class FixedBcdRegressionTests(unittest.TestCase):
    def test_nonfinite_initial_ols_loss_is_rejected_even_without_cycles(self):
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.zeros(8)
        y[7] = 1e200
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(FloatingPointError):
                tensor_bcd(X, y, 1, 1, 1, max_cycles=0,
                           S1_init=np.array([0]), S2_init=np.array([0]),
                           S3_init=np.array([0]))

    @staticmethod
    def orthogonal_fixture():
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.zeros(8)
        y[[0, 1, 3, 7]] = [4.0, 3.0, 2.0, 1.0]
        return X, y

    @staticmethod
    def fake_solver(sequence, calls):
        """Inject support proposals; use real OLS so comparisons remain genuine."""
        def solve(y, Z, gi, s, **kwargs):
            calls.append(kwargs.get("groups_init"))
            active = np.array([sequence[(len(calls) - 1) % len(sequence)]])
            cols = np.flatnonzero(np.isin(gi, active))
            beta = np.zeros(Z.shape[1])
            beta[cols] = np.linalg.lstsq(Z[:, cols], y, rcond=None)[0]
            residual = y - Z @ beta
            rss = float(residual @ residual)
            return SimpleNamespace(active_groups=active, beta=beta,
                                   rss=rss, loss=rss / (2 * len(y)), s=s)
        return solve

    def fit_injected(self, X, y, sequence, initial, max_cycles=12):
        calls = []
        with patch("tensor_abess.tensor.bcd.group_bess_fix",
                   side_effect=self.fake_solver(sequence, calls)):
            result = tensor_bcd(
                X, y, 1, 1, 1, max_cycles=max_cycles,
                S1_init=np.array([initial]), S2_init=np.array([initial]),
                S3_init=np.array([initial]),
            )
        return result, calls

    def assert_consistent_ols(self, X, y, result, sizes):
        supports = (result.S1, result.S2, result.S3)
        self.assertEqual(result.B.shape, X.shape[1:])
        for support, size, dimension in zip(supports, sizes, X.shape[1:]):
            self.assertEqual(len(support), size)
            self.assertEqual(len(np.unique(support)), size)
            self.assertTrue(np.all((support >= 0) & (support < dimension)))
        Z = build_tensor_design(X, *supports)
        coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        np.testing.assert_allclose(tensor_inner(X, result.B), Z @ coef,
                                   rtol=1e-10, atol=1e-11)
        self.assertAlmostEqual(result.loss, tensor_loss(X, y, result.B), places=12)
        outside = np.ones(result.B.shape, dtype=bool)
        outside[np.ix_(*supports)] = False
        np.testing.assert_array_equal(result.B[outside], 0)

    def test_worse_mode_candidate_is_rejected_before_next_mode(self):
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.zeros(8)
        y[7] = 1.0
        result, _ = self.fit_injected(X, y, [0], initial=1)
        self.assertAlmostEqual(result.loss, 0.0, places=14)
        for support in (result.S1, result.S2, result.S3):
            np.testing.assert_array_equal(support, [1])
        self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-14))

    def test_nonmonotone_exploration_can_reach_a_later_better_state(self):
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.zeros(8)
        y[0], y[7] = 2.0, 4.0
        result, calls = self.fit_injected(X, y, [1, 0, 0, 1, 1, 1], initial=0)
        self.assertTrue(calls)
        self.assertAlmostEqual(result.loss, 4.0 / 16.0, places=14)
        for support in (result.S1, result.S2, result.S3):
            np.testing.assert_array_equal(support, [1])
        self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-14))
        self.assertTrue(np.any(np.diff(result.visited_loss_history) > 0))

    def test_returns_best_visited_state_after_later_worse_proposals(self):
        X, y = self.orthogonal_fixture()
        result, _ = self.fit_injected(X, y, [0, 0, 0, 1, 1, 1], initial=1)
        self.assertAlmostEqual(result.loss, 14.0 / 16.0, places=14)
        self.assertAlmostEqual(result.loss, min(result.visited_loss_history), places=14)
        self.assertAlmostEqual(result.loss,
                               result.visited_loss_history[result.best_iteration], places=14)
        self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-14))
        self.assert_consistent_ols(X, y, result, (1, 1, 1))

    def test_equal_rss_support_proposals_terminate_without_cycle(self):
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.ones(8)
        result, _ = self.fit_injected(X, y, [1, 1, 1, 0, 0, 0], initial=0)
        self.assertLess(result.n_cycles, 12)
        self.assertTrue(result.stop_reason)
        for support in (result.S1, result.S2, result.S3):
            np.testing.assert_array_equal(support, [0])
        states = [tuple(tuple(int(i) for i in mode) for mode in state)
                  for state in result.accepted_support_history]
        self.assertEqual(len(states), len(set(states)))

    def test_zero_cycle_result_is_initial_ols_with_stop_metadata(self):
        X, y = self.orthogonal_fixture()
        result = tensor_bcd(X, y, 1, 1, 1, max_cycles=0,
                            S1_init=np.array([1]), S2_init=np.array([1]),
                            S3_init=np.array([1]))
        self.assertEqual(result.n_cycles, 0)
        self.assertEqual(result.best_iteration, 0)
        self.assertTrue(result.stop_reason)
        self.assert_consistent_ols(X, y, result, (1, 1, 1))

    def test_real_small_noisy_tensors_are_monotone_and_ols_consistent(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                X = rng.normal(size=(35, 4, 3, 2))
                y = rng.normal(size=35)
                result = tensor_bcd(X, y, 2, 1, 1, max_cycles=10)
                self.assertTrue(np.all(np.diff(result.loss_history) <= 1e-12))
                self.assertAlmostEqual(result.loss, min(result.visited_loss_history), places=12)
                self.assert_consistent_ols(X, y, result, (2, 1, 1))

    def test_same_seed_reproduces_history_support_and_coefficients(self):
        rng = np.random.default_rng(39)
        X = rng.normal(size=(45, 4, 3, 2))
        y = rng.normal(size=45)
        kwargs = dict(init_strategy="random", random_state=72, max_cycles=10)
        first = tensor_bcd(X, y, 2, 1, 1, **kwargs)
        second = tensor_bcd(X, y, 2, 1, 1, **kwargs)
        for name in ("B", "S1", "S2", "S3"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(first.loss_history, second.loss_history)
        self.assertEqual(first.visited_loss_history, second.visited_loss_history)
        self.assertEqual(first.stop_reason, second.stop_reason)

    @staticmethod
    def exhaustive_min(X, y, sizes):
        supports = [list(itertools.combinations(range(p), s))
                    for p, s in zip(X.shape[1:], sizes)]
        losses = []
        for state in itertools.product(*supports):
            Z = build_tensor_design(X, *(np.array(mode) for mode in state))
            residual = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            losses.append(float(residual @ residual / (2 * len(y))))
        return min(losses)

    def test_exact_orthogonal_example_matches_exhaustive_minimum(self):
        X = np.eye(12).reshape(12, 3, 2, 2)
        y = np.zeros(12)
        y[[1, 9]] = [3.0, 2.0]
        result = tensor_bcd(X, y, 2, 1, 1, max_cycles=10)
        self.assertAlmostEqual(result.loss, self.exhaustive_min(X, y, (2, 1, 1)), places=13)
        self.assert_consistent_ols(X, y, result, (2, 1, 1))

    def test_general_toy_result_cannot_beat_exhaustive_reference(self):
        rng = np.random.default_rng(84)
        X = rng.normal(size=(25, 3, 2, 2))
        y = rng.normal(size=25)
        result = tensor_bcd(X, y, 2, 1, 1, max_cycles=10)
        self.assertGreaterEqual(result.loss + 1e-12,
                                self.exhaustive_min(X, y, (2, 1, 1)))


if __name__ == "__main__":
    unittest.main()
