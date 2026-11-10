"""Behavior checks for identity pairing, support metrics, and cache validity."""
import importlib
import importlib.util
import unittest

import numpy as np


def api(name):
    qualified = 'research.taskcard18.' + name
    assert importlib.util.find_spec(qualified) is not None, 'Missing isolated ' + name
    return importlib.import_module(qualified)


class CloseoutTests(unittest.TestCase):
    def test_actual_elements_do_not_equal_rectangular_envelope(self):
        m = api('analysis')
        B = np.zeros((2, 2, 2)); B[0, 0, 0] = 1; B[1, 1, 1] = 1
        metrics = m.support_metrics(B, np.ones((2, 2, 2)), [np.arange(2)] * 3)
        self.assertEqual(metrics['element_nonzero'], 2)
        self.assertEqual(metrics['q_rect'], 8)
        self.assertEqual(metrics['element_recall'], 0.25)
        self.assertEqual(metrics['element_f1'], 0.4)
        self.assertEqual(metrics['legacy_element_recall'], 1)

    def test_same_input_requires_test_stream_and_all_training_fingerprints(self):
        m = api('workflow')
        base = {'input_fingerprints': dict(X_train='x', y_train='y', B_true='b', train_noise='e'), 'task': {'test_seed': 4}}
        copy = {'input_fingerprints': base['input_fingerprints'].copy(), 'task': {'test_seed': 4}}
        self.assertEqual(m.unique_id(base), m.unique_id(copy))
        copy['task']['test_seed'] = 5
        self.assertNotEqual(m.unique_id(base), m.unique_id(copy))

    def test_paired_statistics_refuse_missing_replicates(self):
        m = api('analysis')
        a = {'u0': 1., 'u1': 3., 'u2': 5.}
        b = {'u0': 0., 'u1': 1., 'u2': 3.}
        result = m.paired_statistics(a, b)
        self.assertAlmostEqual(result['mean_difference'], 5/3)
        self.assertEqual(result['n'], 3)
        with self.assertRaises(ValueError):
            m.paired_statistics(a, {'u0': 0.})

    def test_nonfinite_coefficients_and_pairs_are_rejected(self):
        m = api('analysis')
        with self.assertRaises(ValueError):
            m.paired_statistics({'u0': 0., 'u1': float('nan')}, {'u0': 1., 'u1': 1.})
        B = np.ones((1, 1, 1)); B[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            m.support_metrics(B, np.ones((1, 1, 1)), [[0]] * 3)

    def test_wilson_boundary_contains_all_failure_and_all_success_rates(self):
        m = api('analysis')
        self.assertEqual(m.wilson(0, 50)[0], 0.)
        self.assertEqual(m.wilson(50, 50)[1], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
