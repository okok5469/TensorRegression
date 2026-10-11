"""Independent numerical contracts for training-only EEG model adapters."""
from __future__ import annotations
import importlib
from pathlib import Path
import unittest
import numpy as np

class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name('model_adapters.py').is_file(),
                        'Independent model adapters are not implemented')
        self.api = importlib.import_module('empirical_analysis.lemon.incremental_value.model_adapters')
        rng = np.random.default_rng(184)
        self.X = rng.normal(size=(16, 3, 2, 2))
        self.y = rng.normal(size=16)

    def test_ridge_dual_matches_primal(self):
        fit = self.api.fit_eeg_model('Ridge', self.X, self.y, {'lambda': .7}, {})
        X = self.X.reshape(16, -1)
        beta = np.linalg.solve(X.T @ X + 16 * .7 * np.eye(X.shape[1]), X.T @ self.y)
        np.testing.assert_allclose(fit.B.ravel(), beta, atol=1e-10)
        np.testing.assert_allclose(fit.predict(self.X), X @ beta, atol=1e-10)

    def test_lasso_orthogonal_soft_threshold(self):
        X = (np.eye(4)*2).reshape(4, 2, 2, 1)
        y = np.array([2., -1., .2, -.1])
        fit = self.api.fit_eeg_model('Vector Lasso', X, y, {'lambda': .25}, {'tol': 1e-12})
        z = X.reshape(4, 4).T @ y / 4
        np.testing.assert_allclose(fit.B.ravel(), np.sign(z)*np.maximum(abs(z)-.25, 0), atol=1e-12)

    def test_ratio_one_zero_lasso_and_no_internal_cv(self):
        fit = self.api.fit_eeg_model('Vector Lasso', self.X, self.y, {'lambda_ratio': 1.}, {})
        np.testing.assert_allclose(fit.B, 0., atol=1e-12)
        self.assertFalse(fit.parameters['internal_cv'])

    def test_grid_independent_of_response(self):
        cfg = {'lambda_ratios': [1., .1, .01], 'ridge_lambdas': [.1, 1.]}
        for name in ('Vector Lasso', 'Tensor-GLasso'):
            expected = [{'lambda_ratio': r} for r in cfg['lambda_ratios']]
            self.assertEqual(self.api.make_parameter_grid(name, self.X, self.y, cfg), expected)
            self.assertEqual(self.api.make_parameter_grid(name, self.X, self.y*100, cfg), expected)

    def test_glasso_fixed_ratio(self):
        fit = self.api.fit_eeg_model('Tensor-GLasso', self.X, self.y, {'lambda_ratio': .2},
                                     {'max_iter': 50, 'max_cycles': 2})
        self.assertEqual(fit.B.shape, (3, 2, 2))
        self.assertTrue(np.isfinite(fit.predict(self.X)).all())
        self.assertFalse(fit.parameters['internal_cv'])
        self.assertGreaterEqual(len(fit.parameters['group_lambda_trace']), 6)

    def test_path_order(self):
        params = [{'lambda_ratio': 1.}, {'lambda_ratio': .1}]
        fits = self.api.fit_eeg_path('Vector Lasso', self.X, self.y, params, {})
        self.assertEqual([f.parameters['lambda_ratio'] for f in fits], [1., .1])

    def test_ista_matches_unchanged_core_solvers(self):
        from numerical_simulation.methods import _fit_lasso_ista
        from tensor_abess.utils.glasso import fit_group_lasso_ista
        flat = self.X.reshape(16, -1)
        options = {'max_iter': 100, 'tol': 1e-9}
        expected = _fit_lasso_ista(flat, self.y, .07, **options)
        result = self.api.fit_eeg_model('Vector Lasso', self.X, self.y,
                                        {'lambda': .07}, options)
        np.testing.assert_allclose(result.B.ravel(), expected, rtol=1e-10, atol=1e-12)
        groups = np.repeat(np.arange(3), 4)
        expected_group = fit_group_lasso_ista(flat, self.y, groups, .07, **options)
        actual = self.api._ista(flat, self.y, .07, options,
                                self.api._spectral_lipschitz(flat), groups)
        np.testing.assert_allclose(actual, expected_group.beta, rtol=1e-10, atol=1e-12)

    def test_abess_adaptive_core_and_bic_policy(self):
        options = {'criterion': 'bic', 'n_lambdas': 3, 'max_iter': 30,
                   'max_cycles': 1, 'm_max': 2, 'cv_folds': 3}
        fit = self.api.fit_eeg_model('Tensor-ABESS', self.X, self.y, None, options)
        self.assertEqual(fit.B.shape, self.X.shape[1:])
        self.assertEqual(fit.parameters['q'], np.prod([len(s) for s in fit.supports]))
        with self.assertRaises(ValueError):
            self.api.fit_eeg_model('Tensor-ABESS', self.X, self.y, None,
                                    dict(options, criterion='cv'))

    def test_invalid_lambda(self):
        for name, param in [('Ridge', {'lambda': 0.}), ('Vector Lasso', {'lambda_ratio': -1.})]:
            with self.assertRaises(ValueError):
                self.api.fit_eeg_model(name, self.X, self.y, param, {})

if __name__ == '__main__':
    unittest.main()
