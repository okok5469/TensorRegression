"""Selected-model OLS audit, without numerical rescue or coefficient replacement."""
import unittest
import numpy as np

from numerical_simulation.candidate_search_study.numerics import (
    audit_selected_model, rectangular_design,
)


class SelectedNumericsTests(unittest.TestCase):
    def test_rectangular_design_preserves_tensor_beta_order(self):
        X = np.arange(3*4*3*2, dtype=float).reshape(3, 4, 3, 2)
        supports = [np.array([1, 3]), np.array([0, 2]), np.array([0, 1])]
        expected = np.array([[X[n, i, j, k] for i in supports[0] for j in supports[1]
                              for k in supports[2]] for n in range(3)])
        np.testing.assert_array_equal(rectangular_design(X, supports), expected)

    def test_independent_ols_and_tensor_predictions_agree(self):
        rng = np.random.default_rng(4901)
        X = rng.normal(size=(30, 4, 3, 2)); y = rng.normal(size=30)
        supports = [np.array([1, 3]), np.array([0, 2]), np.array([1])]
        D = rectangular_design(X, supports)
        beta = np.linalg.lstsq(D, y, rcond=None)[0]
        B = np.zeros(X.shape[1:]); B[np.ix_(*supports)] = beta.reshape(2, 2, 1)
        rss = float(np.sum((y-D @ beta)**2))
        result = audit_selected_model(X, y, B, supports, expected_rss=rss)
        self.assertEqual(result['rank'], 4)
        self.assertFalse(result['rank_deficient'])
        self.assertTrue(result['checks_passed'])
        self.assertLess(result['normal_equation_relative_residual'], 1e-12)
        self.assertLess(result['prediction_max_abs_difference'], 1e-12)

    def test_corrupted_coefficient_cannot_pass_by_refitting(self):
        rng = np.random.default_rng(4902)
        X = rng.normal(size=(30, 2, 2, 2)); y = rng.normal(size=30)
        supports = [np.array([0]), np.array([1]), np.array([0])]
        D = rectangular_design(X, supports)
        beta = np.linalg.lstsq(D, y, rcond=None)[0]
        B = np.zeros(X.shape[1:]); B[0, 1, 0] = beta[0]
        rss = float(np.sum((y-D @ beta)**2)); B[0, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'RSS'):
            audit_selected_model(X, y, B, supports, expected_rss=rss)

    def test_rank_deficiency_is_reported_without_rescue(self):
        x = np.linspace(0.3, 2.0, 12)
        X = np.stack([x, x], axis=-1).reshape(12, 1, 1, 2)
        y = 2*x
        supports = [np.array([0]), np.array([0]), np.array([0, 1])]
        # An alternative OLS solution; nonunique coefficients are not rewritten.
        B = np.array([[[2.0, 0.0]]]); before = B.copy()
        result = audit_selected_model(X, y, B, supports, expected_rss=0.0)
        self.assertEqual(result['rank'], 1)
        self.assertTrue(result['rank_deficient'])
        self.assertTrue(result['checks_passed'])
        self.assertIsNone(result['condition_number'])
        np.testing.assert_array_equal(B, before)


if __name__ == '__main__':
    unittest.main()
