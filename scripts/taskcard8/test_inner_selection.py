"""Training isolation and prespecified selection rules for task card 8."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import os
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np


def implementation():
    path = Path(__file__).with_name('inner_selection.py')
    assert path.is_file(), 'Task 8 inner selection implementation is missing'
    spec = importlib.util.spec_from_file_location('task8_inner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InnerSelectionTests(unittest.TestCase):
    def test_shared_budget_stops_at_absolute_deadline(self):
        m = implementation()
        plan = {'shared_budget': {'deadline_utc': '2026-10-16T16:46:46+00:00'}}
        self.assertEqual(m.remaining_budget(plan, datetime(2026, 10, 16, 16, 46, 45,
                                                          tzinfo=timezone.utc)), 1.)
        with self.assertRaisesRegex(ValueError, 'budget'):
            m.remaining_budget(plan, datetime(2026, 10, 16, 16, 46, 46, tzinfo=timezone.utc))

    def test_common_grid_intersects_actual_ordered_grids_and_rejects_empty(self):
        m = implementation()
        self.assertEqual(m.common_grid([[[1, 1, 1], [2, 1, 1]],
                                       [[1, 1, 1], [1, 2, 1]], [[1, 1, 1]]]), [(1, 1, 1)])
        with self.assertRaisesRegex(ValueError, 'common'):
            m.common_grid([[[1, 1, 1]], [[2, 1, 1]]])

    def test_pooled_cv_weights_subjects_and_breaks_numerical_ties_by_q(self):
        m = implementation()
        rows = [dict(sizes=[2, 1, 1], inner_sse=[2., 20.], inner_n=[1, 10]),
                dict(sizes=[1, 1, 1], inner_sse=[4., 10.], inner_n=[1, 10])]
        self.assertEqual(m.select_cv(rows)['sizes'], [1, 1, 1])
        self.assertEqual(m.select_cv([dict(sizes=[2, 1, 1], inner_sse=[1.], inner_n=[1]),
                                     dict(sizes=[1, 1, 1], inner_sse=[1.+5e-13], inner_n=[1])])['sizes'], [1, 1, 1])

    def test_stability_is_pairwise_mode_jaccard_and_only_inside_msic_band(self):
        m = implementation()
        supports = [[[0], [1], [2]], [[0], [1], [3]], [[0], [1], [2]]]
        self.assertAlmostEqual(m.stability(supports), 7/9)
        rows = [dict(sizes=[1, 1, 1], q=1, msic=0., stability=.2),
                dict(sizes=[2, 1, 1], q=2, msic=.5, stability=.8),
                dict(sizes=[3, 1, 1], q=3, msic=100., stability=1.)]
        self.assertEqual(m.select_msic(rows, 78, stable=True)['sizes'], [2, 1, 1])
        self.assertEqual(m.select_msic(rows, 78)['sizes'], [1, 1, 1])

    def test_split_contract_rejects_outer_test_leakage_and_duplicate_validation(self):
        m = implementation()
        outer = np.arange(6)
        folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])),
                 (np.array([0, 1, 4, 5]), np.array([2, 3])),
                 (np.array([0, 1, 2, 3]), np.array([4, 5]))]
        m.validate_inner_partition(outer, folds)
        with self.assertRaisesRegex(ValueError, 'partition'):
            m.validate_inner_partition(outer, [(np.array([0, 1]), np.array([6]))])
        with self.assertRaisesRegex(ValueError, 'partition'):
            m.validate_inner_partition(outer, folds + [folds[0]])

    def test_candidate_map_excludes_adaptive_and_checks_fixed_grid_identity(self):
        m = implementation()
        record = dict(sizes=[1, 1, 1], q=1, support=[[0], [0], [0]])
        trace = dict(ordered_size_grid=[[1, 1, 1]], candidates=[record, record])
        mapped = m.fixed_map(trace)
        self.assertEqual(mapped[(1, 1, 1)][0], 1)
        bad = dict(trace, ordered_size_grid=[[2, 1, 1]])
        with self.assertRaisesRegex(ValueError, 'Fixed'):
            m.fixed_map(bad)

    def test_true_core_fit_captures_fresh_search_and_training_only_preprocessor(self):
        if os.environ.get('TASK8_INNER_TEST_ISOLATED') != '1':
            environment = dict(os.environ, TASK8_INNER_TEST_ISOLATED='1',
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            subprocess.run([sys.executable, '-B', '-m', 'unittest',
                            'scripts.taskcard8.test_inner_selection.InnerSelectionTests.'
                            'test_true_core_fit_captures_fresh_search_and_training_only_preprocessor'],
                           check=True, env=environment, capture_output=True, text=True)
            return
        m = implementation()
        m.configure_core('v5.2-search-final')
        rng = np.random.default_rng(20801)
        X = rng.normal(size=(36, 2, 2, 2))
        y = X[:, 0, 0, 0] + rng.normal(size=36)
        C = np.ones((36, 1))
        options = dict(c=3., criterion='bic', n_lambdas=3, max_iter=20,
                       max_cycles=2, m_max=2, k_max=1)
        with tempfile.TemporaryDirectory() as temporary:
            _, p, trace, coefficients = m.fit_dataset(X[:27], y[:27], C[:27], options,
                                                      Path(temporary) / 'fit')
            mapping = m.fixed_map(trace)
            self.assertEqual(trace['n_train'], 27)
            self.assertEqual(trace['capacity'], 18)
            self.assertEqual(len(coefficients), len(mapping) + 1)
            from empirical_analysis.lemon.incremental_value.preprocessing import fit_preprocessor
            expected = fit_preprocessor(X[:27], y[:27], C[:27])
            for key, value in expected.to_arrays().items():
                np.testing.assert_array_equal(value, p.to_arrays()[key])
            self.assertFalse(np.array_equal(p.x_scale, fit_preprocessor(X, y, C).x_scale))
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                m.fit_dataset(X[:27], y[:27], C[:27], options, Path(temporary) / 'fit')


if __name__ == '__main__':
    unittest.main()
