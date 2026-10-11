"""Independent candidate-search research core regressions (not production)."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import inspect
import json
import os
import subprocess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'doc' / 'algorithm_versions' / 'v5.2-candidate-search' / 'src'))
from tensor_abess.tensor.bcd import _candidate_size_triples
from tensor_abess.tensor.bcd import tensor_bcd_adaptive, tensor_bcd
from tensor_abess.tensor.abess import tensor_abess
from tensor_abess.tensor.ops import tensor_loss
from tensor_abess.utils.init import tensor_init_glasso


class CandidateGridTests(unittest.TestCase):
    def test_full_covers_all_legal_shapes(self):
        expected = [(a, b, c) for a in range(1, 5) for b in range(1, 4) for c in range(1, 6) if a*b*c <= 30]
        actual = _candidate_size_triples(4, 3, 5, 30, grid_policy='full')
        self.assertEqual(actual, expected)

    def test_full_budget_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            _candidate_size_triples(10, 10, 10, 1000, grid_policy='full', candidate_budget=20)

    def test_n400_old_grid_excludes_legal_true_shape(self):
        shape = (4, 3, 5)  # Regression fixture; never passed to the solver.
        self.assertNotIn(shape, _candidate_size_triples(7, 6, 6, 280))
        self.assertIn(shape, _candidate_size_triples(7, 6, 6, 280, grid_policy='full'))

    def test_full_sorted_unique_and_capacity(self):
        grid = _candidate_size_triples(10, 10, 15, 280, grid_policy='full')
        self.assertEqual(len(grid), 1044)
        self.assertEqual(grid, sorted(set(grid)))
        self.assertTrue(all(np.prod(shape) <= 280 for shape in grid))


class CandidateCapacityTests(unittest.TestCase):
    def test_n100_precompression_caps_retained(self):
        X = np.zeros((100, 8, 7, 8))
        y = np.ones(100)
        fit = SimpleNamespace(active_groups=np.array([0]))
        with patch('tensor_abess.utils.init._select_group_lasso_fit', return_value=fit):
            legacy = tensor_init_glasso(X, y)
            candidate = tensor_init_glasso(X, y, cap_policy='candidate')
        self.assertEqual((legacy.s1_max, legacy.s2_max, legacy.s3_max), (4, 4, 4))
        self.assertEqual((candidate.s1_max, candidate.s2_max, candidate.s3_max), (5, 5, 5))
        self.assertFalse(5 <= legacy.s3_max)
        self.assertIn((4, 3, 5), _candidate_size_triples(5, 5, 5, 70, grid_policy='full'))
        for name in ('S1_init', 'S2_init', 'S3_init'):
            np.testing.assert_array_equal(getattr(candidate, name), getattr(legacy, name))

    def test_overcapacity_initialization_fails_without_clipping(self):
        X = np.zeros((20, 4, 4, 4)); y = np.ones(20)
        with self.assertRaisesRegex(ValueError, 'Illegal initial.*Q=14'):
            tensor_bcd_adaptive(X, y, s1_max=4, s2_max=4, s3_max=4,
                S1_init=np.arange(3), S2_init=np.arange(3), S3_init=np.arange(2),
                cap_policy='candidate', max_cycles=0)

    def test_dynamic_caps_bound_each_conditional_parameter_count(self):
        rng = np.random.default_rng(171)
        X = rng.normal(size=(20, 4, 4, 4)); y = rng.normal(size=20)
        def proposal(*args, **kwargs):
            return SimpleNamespace(active_groups=np.arange(kwargs['s_max']))
        with patch('tensor_abess.tensor.bcd.group_bess_adaptive', side_effect=proposal):
            result = tensor_bcd_adaptive(X, y, s1_max=4, s2_max=4, s3_max=4,
                S1_init=np.arange(2), S2_init=np.arange(2), S3_init=np.arange(2),
                cap_policy='candidate', information_criterion='sic', max_cycles=1)
        caps = result.search_metadata['dynamic_cap_history']
        self.assertEqual([row['s_max'] for row in caps], [3, 2, 2])
        self.assertTrue(all(row['s_max']*row['block_size'] <= 14 for row in caps))
        self.assertLessEqual(result.s1*result.s2*result.s3, 14)

    def test_no_truth_or_test_arguments_in_research_search(self):
        for solver in (tensor_abess, tensor_bcd_adaptive, _candidate_size_triples):
            names = inspect.signature(solver).parameters
            self.assertFalse(any('truth' in name or 'test' in name for name in names))


class CandidateFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(708)
        cls.X = rng.normal(size=(40, 4, 3, 4))
        cls.y = 2*cls.X[:, 0, 0, 0] + rng.normal(size=40)
        cls.kw = dict(s1_max=4, s2_max=3, s3_max=4,
                      S1_init=np.array([0]), S2_init=np.array([0]), S3_init=np.array([0]),
                      max_cycles=1, m_max=2, k_max=1, cap_policy='candidate')
        cls.cache = {}
        cls.full = tensor_bcd_adaptive(cls.X, cls.y, **cls.kw, grid_policy='full',
                                      fixed_candidate_cache=cls.cache)

    def test_each_fixed_candidate_is_best_visited_OLS_state(self):
        for record in self.full.candidate_records:
            if record['kind'] != 'global-fixed':
                continue
            self.assertAlmostEqual(record['loss'], tensor_loss(self.X, self.y, record['B']), places=12)
            self.assertAlmostEqual(record['rss'], 80*record['loss'], places=12)
            visited = record['visited_loss_history']
            self.assertEqual(record['loss'], min(visited))
            self.assertEqual(record['loss'], visited[record['best_iteration']])
            self.assertTrue(all(b <= a for a, b in zip(record['loss_history'], record['loss_history'][1:])))
            self.assertEqual(tuple(map(len, record['supports'])), record['sizes'])
            fitted = record['B'][np.ix_(*record['supports'])].ravel()
            design = self.X[:, record['supports'][0]][:, :, record['supports'][1]][:, :, :, record['supports'][2]].reshape(40, -1)
            np.testing.assert_allclose(design.T @ (self.y-design @ fitted), 0, atol=1e-10)

    def test_full_grid_ledger_and_reproducible_cache(self):
        repeated = tensor_bcd_adaptive(self.X, self.y, **self.kw, grid_policy='full',
                                      fixed_candidate_cache=self.cache)
        np.testing.assert_array_equal(repeated.B, self.full.B)
        self.assertEqual(repeated.sic, self.full.sic)
        meta = repeated.search_metadata
        self.assertTrue(meta['grid_complete'])
        self.assertEqual(meta['excluded_legal_shapes'], [])
        self.assertEqual(meta['size_grid'], meta['legal_shapes'])
        self.assertTrue(all(rec['cache_hit'] for rec in repeated.candidate_records if rec['kind']=='global-fixed'))

    def test_training_change_cannot_reuse_cached_fit(self):
        changed = tensor_bcd_adaptive(self.X, self.y+0.123, **{**self.kw, 'max_cycles': 0},
                                     grid_policy='legacy', fixed_candidate_cache=self.cache)
        self.assertNotEqual(changed.search_metadata['training_fingerprint'], self.full.search_metadata['training_fingerprint'])
        self.assertTrue(all(not rec['cache_hit'] for rec in changed.candidate_records if rec['kind']=='global-fixed'))

    def test_coarse_to_fine_excluded_shapes_are_traceable(self):
        rng = np.random.default_rng(812)
        X = rng.normal(size=(60, 6, 6, 6)); y = X[:, 0, 0, 0] + rng.normal(size=60)
        result = tensor_bcd_adaptive(X, y, s1_max=6, s2_max=6, s3_max=6,
            S1_init=np.array([0]), S2_init=np.array([0]), S3_init=np.array([0]),
            max_cycles=0, cap_policy='candidate', grid_policy='coarse_to_fine')
        meta = result.search_metadata
        evaluated, excluded, legal = map(set, (meta['size_grid'], meta['excluded_legal_shapes'], meta['legal_shapes']))
        self.assertFalse(evaluated & excluded)
        self.assertEqual(evaluated | excluded, legal)
        self.assertEqual(len(meta['size_grid']), len(evaluated))
        self.assertTrue(excluded)
        self.assertFalse(meta['grid_complete'])
        self.assertTrue(set(meta['initial_size_grid']) <= evaluated)
        self.assertFalse(set(meta['initial_size_grid']) & set(meta['refinement_size_grid']))

    def test_A0_exactly_matches_production_v51(self):
        kwargs = dict(criterion='bic', n_lambdas=2, max_iter=3, max_cycles=1, m_max=2, k_max=1)
        current = tensor_abess(self.X, self.y, **kwargs)
        script = """import json, numpy as np
from tensor_abess.tensor.abess import tensor_abess
rng=np.random.default_rng(708); X=rng.normal(size=(40,4,3,4)); y=2*X[:,0,0,0]+rng.normal(size=40)
r=tensor_abess(X,y,criterion='bic',n_lambdas=2,max_iter=3,max_cycles=1,m_max=2,k_max=1)
print(json.dumps({'B':r.B.tolist(),'sizes':[r.s1,r.s2,r.s3],'sic':r.sic,'rss':r.rss}))
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
        expected = json.loads(subprocess.check_output([sys.executable, '-c', script], env=env, text=True))
        np.testing.assert_array_equal(current.B, expected['B'])
        self.assertEqual([current.s1,current.s2,current.s3], expected['sizes'])
        self.assertEqual(current.sic, expected['sic'])
        self.assertEqual(current.rss, expected['rss'])


if __name__ == '__main__':
    unittest.main()
