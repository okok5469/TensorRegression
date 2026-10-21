"""Negative controls for independent saved-evidence checks, without solver fits."""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np


class IndependentAuditTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec("research.taskcard10.audit")
        self.assertIsNotNone(spec, "Independent audit behavior is not implemented")
        return __import__("research.taskcard10.audit", fromlist=["audit"])

    def fixture(self):
        rng = np.random.default_rng(839)
        X = rng.normal(size=(19, 3, 2, 2))
        support = ([0, 2], [1], [0, 1])
        Z = X[:, [0, 2]][:, :, [1]][:, :, :, [0, 1]].reshape(19, -1)
        y = rng.normal(size=19)
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        B = np.zeros((3, 2, 2))
        B[np.ix_(*support)] = beta.reshape(2, 1, 2)
        rss = float(np.sum((y - Z @ beta) ** 2))
        return X, y, B, support, rss

    def test_saved_fit_matches_independent_rectangular_ols_and_prediction(self):
        mod = self.module()
        X, y, B, support, rss = self.fixture()
        pred = np.einsum("nijk,ijk->n", X, B)
        report = mod.audit_saved_fit(X, y, B, support, rss, X, pred)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rank"], 4)

    def test_saved_fit_rejects_corrupt_coefficient_rss_and_prediction(self):
        mod = self.module()
        X, y, B, support, rss = self.fixture()
        with self.assertRaises(AssertionError):
            mod.audit_saved_fit(X, y, B + .1, support, rss)
        with self.assertRaises(AssertionError):
            mod.audit_saved_fit(X, y, B, support, rss + 1)
        with self.assertRaises(AssertionError):
            mod.audit_saved_fit(X, y, B, support, rss, X, np.ones(len(y)))

    def test_weighted_cv_uses_samples_and_rejects_unweighted_winner(self):
        mod = self.module()
        # Candidate0 wins an erroneous average of fold MSE, candidate1 total SSE.
        sse = np.array([[0., 1.8], [30., 27.]])
        curve, selected = mod.weighted_cv_choice(sse, [1, 9], [(1, 1, 1), (1, 1, 2)])
        np.testing.assert_allclose(curve, [3., 2.88])
        self.assertEqual(selected, 1)

    def test_cv_tie_prefers_smaller_q_then_stable_shape_id(self):
        mod = self.module()
        grid = [(2, 1, 2), (1, 2, 1), (1, 1, 2)]
        _, selected = mod.weighted_cv_choice(np.ones((3, 3)), [3, 3, 4], grid)
        self.assertEqual(selected, 2)


if __name__ == "__main__":
    unittest.main()
