"""Behavioral gates for the isolated training-only selector."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.taskcard7.lemon import configure_core, observe_fit
configure_core('v5.2-search-final')

try:
    cv = importlib.import_module('research.taskcard10.cv')
except ModuleNotFoundError:
    cv = None


class SelectorGates(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cv, 'Task10 selector behavior is not implemented')

    def test_public_capacity_protects_true_n400_shape(self):
        a = cv.capacity_protocol(100, (32, 20, 25), 3)
        b = cv.capacity_protocol(400, (32, 20, 25), 3)
        self.assertEqual(a['common_caps'], [3, 3, 4])
        self.assertEqual(b['common_caps'], [5, 6, 6])
        self.assertEqual(len(a['common_grid']), 36)
        self.assertEqual(len(b['common_grid']), 180)
        self.assertIn([4, 3, 5], b['common_grid'])
        self.assertNotIn([4, 3, 5], a['common_grid'])

    def test_unproven_floor_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            cv.capacity_protocol(30, (3, 3, 3), 3)

    def test_partition_is_disjoint_and_exact_cover(self):
        folds = cv.make_folds(100, 90210)
        validation = []
        for train, valid in folds:
            self.assertFalse(set(train) & set(valid))
            self.assertEqual(set(train) | set(valid), set(range(100)))
            validation.extend(valid)
        self.assertEqual(sorted(validation), list(range(100)))
        self.assertEqual([len(v) for _, v in folds], [34, 33, 33])

    def test_weighted_cv_differs_from_unweighted_fold_mean(self):
        rows = [dict(sizes=[1, 1, 1], q=1, inner_sse=[0., 20.], inner_n=[10, 2]),
                dict(sizes=[1, 1, 2], q=2, inner_sse=[15., 0.], inner_n=[10, 2])]
        self.assertEqual(cv.select_cv(rows)['sizes'], [1, 1, 2])

    def test_cv_tie_uses_q_then_stable_configuration(self):
        rows = [dict(sizes=[2, 1, 1], q=2, inner_sse=[4.], inner_n=[2]),
                dict(sizes=[1, 1, 2], q=2, inner_sse=[4.], inner_n=[2]),
                dict(sizes=[1, 1, 1], q=1, inner_sse=[4.], inner_n=[2])]
        self.assertEqual(cv.select_cv(rows)['sizes'], [1, 1, 1])
        self.assertEqual(cv.select_cv(rows[:2])['sizes'], [1, 1, 2])

    def test_centering_uses_training_only_and_restores_response_units(self):
        X = np.arange(40., dtype=float).reshape(5, 2, 2, 2)
        y = np.arange(5.) + 10
        p = cv.fit_preprocessor(X[:3], y[:3], 'center', [0, 1, 2])
        np.testing.assert_array_equal(p['X_mean'], X[:3].mean(axis=0))
        B = np.ones((2, 2, 2))
        expected = (X[3:] - X[:3].mean(axis=0)).reshape(2, -1) @ B.ravel() + y[:3].mean()
        np.testing.assert_allclose(cv.predict(p, B, X[3:]), expected)

    def test_batch_sparse_predictions_preserve_dense_prediction_units_and_order(self):
        self.assertTrue(hasattr(cv, 'predict_pool'), 'Batch candidate prediction is not implemented')
        rng = np.random.default_rng(7001)
        X, y = rng.normal(size=(15, 4, 3, 5)), rng.normal(size=15)
        Bs = rng.normal(size=(4, 4, 3, 5))
        Bs[rng.uniform(size=Bs.shape) < .8] = 0.
        for kind in ['identity', 'center']:
            p = cv.fit_preprocessor(X[:10], y[:10], kind, list(range(10)))
            expected = np.column_stack([cv.predict(p, B, X[10:]) for B in Bs])
            np.testing.assert_allclose(cv.predict_pool(p, Bs, X[10:]), expected,
                                       rtol=1e-13, atol=1e-13)

    def test_cache_identity_rejects_fold_source_seed_and_training_changes(self):
        base = dict(data_identity={'X': 'a', 'y': 'b'}, train_indices=[0, 1],
                    validation_indices=[2], grid=[[1, 1, 1]], config={'max_cycles': 2},
                    source_identity={'core': 'old'}, preprocessing={'kind': 'identity'},
                    seed=1, program='formal-fixed')
        key = cv.cache_identity(**base)[0]
        for field, value in [('seed', 2), ('train_indices', [0, 2]),
                             ('source_identity', {'core': 'new'}),
                             ('data_identity', {'X': 'a', 'y': 'new'}),
                             ('preprocessing', {'kind': 'center'})]:
            changed = dict(base, **{field: value})
            self.assertNotEqual(key, cv.cache_identity(**changed)[0])

    def test_fixed_bridge_matches_actual_formal_global_candidate(self):
        from tensor_abess.tensor.abess import tensor_abess
        rng = np.random.default_rng(104729)
        X, y = rng.normal(size=(100, 6, 6, 6)), rng.normal(size=100)
        options = dict(k_max=1, c=3., criterion='bic', n_lambdas=3, cv_folds=3,
                       max_iter=30, tol=1e-6, max_cycles=2, eps=1e-8,
                       adaptive_criterion='msic', msic_refine_alpha=1.5,
                       msic_refine_gamma=1., msic_refine_one_se_scale=1.)
        _, trace, Bs = observe_fit(lambda: tensor_abess(X, y, **options))
        sizes = trace['ordered_size_grid'][13]
        record, B = cv.fit_fixed_candidate(X, y, sizes, {'k_max': 1, 'max_cycles': 2})
        np.testing.assert_array_equal(B, Bs[14])
        self.assertEqual(record['support'], trace['candidates'][14]['support'])
        self.assertEqual(record['visited_loss_history'], trace['fixed_searches'][13]['visited_loss_history'])

    def test_training_cache_recovers_without_refitting_and_detects_corruption(self):
        rng = np.random.default_rng(73)
        X, y = rng.normal(size=(100, 6, 6, 6)), rng.normal(size=100)
        config = {'k_max': 1, 'max_cycles': 2, 'abess_init_c': 3.,
                  'abess_init_criterion': 'bic', 'n_lambdas': 3,
                  'cv_folds': 3, 'max_iter': 30, 'tol': 1e-6}
        args = dict(train_indices=list(range(66)), validation_indices=list(range(66, 100)),
                    grid=[[1, 1, 1], [1, 2, 1]], config=config,
                    source_identity={'core': 'frozen'}, seed=73, preproc='identity')
        with tempfile.TemporaryDirectory() as temporary:
            first = cv.fit_fold_pool(X[:66], y[:66], cache_dir=Path(temporary), **args)
            second = cv.fit_fold_pool(X[:66], y[:66], cache_dir=Path(temporary), **args)
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            np.testing.assert_array_equal(first['B'], second['B'])
            path = Path(temporary) / first['cache_key'] / 'pool.json'
            saved = json.loads(path.read_text(encoding='utf-8'))
            saved['candidates'][0]['rss'] += 1
            path.write_text(json.dumps(saved), encoding='utf-8')
            with self.assertRaises(ValueError):
                cv.fit_fold_pool(X[:66], y[:66], cache_dir=Path(temporary), **args)

    def test_three_arms_native_is_exact_and_common_models_reuse_full_training_pool(self):
        from tensor_abess.tensor.abess import tensor_abess
        rng = np.random.default_rng(1009)
        X, y = rng.normal(size=(100, 6, 6, 6)), rng.normal(size=100)
        config = dict(k_max=1, max_cycles=2, abess_init_c=3., abess_init_criterion='bic',
                      n_lambdas=3, cv_folds=3, max_iter=30, tol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            run = cv.run_three_arms(X, y, config, 17, Path(temporary), {'core': 'frozen'})
        direct = tensor_abess(X, y, **cv.native_options(config))
        np.testing.assert_array_equal(run['arrays']['A0_B'], direct.B)
        self.assertEqual(run['summary']['counters']['extra_full_fixed_fits'], 0)
        for arm in ['A1', 'A2']:
            record = run['summary']['arms'][arm]
            np.testing.assert_array_equal(run['arrays'][arm + '_B'],
                                          run['arrays']['full_B'][record['full_candidate_index']])
        self.assertEqual(run['summary']['counters']['inner_fixed_computed'], 108)
        self.assertEqual(sum(f['n_validation'] for f in run['summary']['folds']), 100)

    def test_expired_budget_stops_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TimeoutError):
                cv.run_three_arms(np.zeros((100, 6, 6, 6)), np.zeros(100), {},
                                  1, Path(temporary), {}, deadline_monotonic=0.)

    def test_heldout_response_perturbation_cannot_change_its_training_fit(self):
        rng = np.random.default_rng(65537)
        X, y = rng.normal(size=(100, 6, 6, 6)), rng.normal(size=100)
        config = dict(k_max=1, max_cycles=1, abess_init_c=3., abess_init_criterion='bic',
                      n_lambdas=3, cv_folds=3, max_iter=30, tol=1e-6)
        valid = cv.make_folds(100, 31)[0][1]
        changed_y = y.copy()
        changed_y[valid] += 100.
        with tempfile.TemporaryDirectory() as temporary:
            first = cv.run_three_arms(X, y, config, 31, Path(temporary), {'core': 'frozen'})
            second = cv.run_three_arms(X, changed_y, config, 31, Path(temporary), {'core': 'frozen'})
        self.assertEqual(first['summary']['folds'][0]['cache_key'],
                         second['summary']['folds'][0]['cache_key'])
        self.assertTrue(second['summary']['folds'][0]['cache_hit'])
        np.testing.assert_array_equal(first['arrays']['inner0_B'], second['arrays']['inner0_B'])
        np.testing.assert_array_equal(first['arrays']['inner0_prediction'],
                                      second['arrays']['inner0_prediction'])
        self.assertNotEqual(first['summary']['folds'][0]['validation_sse'],
                            second['summary']['folds'][0]['validation_sse'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
