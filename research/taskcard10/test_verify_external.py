"""Pure external-evidence arithmetic gates; generates no scientific dataset."""
import importlib.util
import itertools
import unittest

import numpy as np


class ExternalAuditTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('research.taskcard10.verify_external')
        self.assertIsNotNone(spec, 'Independent external replay checks are not implemented')
        return __import__('research.taskcard10.verify_external', fromlist=['verify_external'])

    def requests(self):
        rows = []
        for n, snr, shape, rho in itertools.product((100, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)), (0., .6)):
            for rep in (0, 1):
                rows.append({'cohort': 'development', 'task_id': len(rows), 'replicate': rep,
                             'config': {'n_train': n, 'snr': snr, 'sparsity': list(shape), 'rho': [rho] * 3}})
        return rows

    def test_scope_selects_exact_twelve_without_outcome_or_completion_inputs(self):
        mod = self.module()
        chosen = mod.selected_requests(self.requests(), 'development')
        self.assertEqual([r['task_id'] for r in chosen], list(range(0, 48, 4)))
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len({(r['config']['n_train'], r['config']['snr'], tuple(r['config']['sparsity'])) for r in chosen}), 12)

    def test_missing_predeclared_cell_is_rejected(self):
        mod = self.module()
        with self.assertRaises(AssertionError):
            mod.selected_requests(self.requests()[1:], 'development')

    def fixture(self):
        X = np.arange(36., dtype=float).reshape(3, 3, 2, 2) / 10.
        B = np.zeros((3, 2, 2)); B[0, 0, 1] = 2; B[2, 0, 1] = -.5
        zeros = np.zeros_like(B)
        coefficients = {'A0': B, 'A1': zeros, 'A2': B}
        supports = {'A0': [[0, 2], [0], [1]], 'A1': [[1], [1], [0]], 'A2': [[0, 2], [0], [1]]}
        y, signal = np.array([1., 2., 3.]), np.array([0., 1., 2.])
        prediction = np.column_stack([np.einsum('nijk,ijk->n', X, coefficients[a]) for a in coefficients])
        saved = {'y': y.copy(), 'signal': signal.copy(), 'prediction': prediction}
        return [(X[:2], y[:2], signal[:2]), (X[2:], y[2:], signal[2:])], coefficients, supports, saved

    def test_rectangular_prediction_and_pmse_match_saved_shared_stream(self):
        mod = self.module()
        batches, Bs, supports, saved = self.fixture()
        result = mod.verify_stream(batches, Bs, supports, saved)
        self.assertEqual(result['test_fingerprint']['n_observations'], 3)
        self.assertEqual(result['test_fingerprint']['n_batches'], 2)
        self.assertAlmostEqual(result['metrics']['A1']['pmse'], 14 / 3)
        self.assertLess(result['maximum_prediction_absolute_error'], 1e-12)

    def test_prediction_or_response_corruption_is_rejected(self):
        mod = self.module()
        batches, Bs, supports, saved = self.fixture()
        changed = {key: value.copy() for key, value in saved.items()}
        changed['prediction'][0, 1] += .1
        with self.assertRaises(AssertionError):
            mod.verify_stream(batches, Bs, supports, changed)
        changed = {key: value.copy() for key, value in saved.items()}
        changed['y'][1] += 1
        with self.assertRaises(AssertionError):
            mod.verify_stream(batches, Bs, supports, changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
