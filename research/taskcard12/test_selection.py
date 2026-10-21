"""Task12 wrapper gates: deterministic tiny fits and strict cache rejection."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    selection = importlib.import_module('research.taskcard12.selection')
except ModuleNotFoundError:
    selection = None


class SelectionGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = dict(criterion='bic', n_lambdas=3, max_cycles=2)
        rng = np.random.default_rng(12012)
        cls.X = rng.normal(size=(24, 2, 2, 2))
        cls.y = cls.X[:, 0, 0, 0] + rng.normal(size=24)
        cls.pair = None if selection is None else selection.fit_pair(cls.X, cls.y, **cls.options)

    def setUp(self):
        self.assertIsNotNone(selection, 'Task12 selection wrapper is not implemented')

    def test_pair_default_single_fs_and_repeat_are_identical(self):
        original = selection.fit(self.X, self.y, **self.options)
        fs = selection.fit(self.X, self.y, selection_rule='fs', **self.options)
        repeated = selection.fit_pair(self.X, self.y, **self.options)
        for rule, single in [('original', original), ('fs', fs)]:
            expected = self.pair[rule]
            np.testing.assert_array_equal(single['B'], expected['B'])
            self.assertEqual(single['selection'], expected['selection'])
            np.testing.assert_array_equal(repeated[rule]['B'], expected['B'])
        self.assertEqual(self.pair['pool']['metadata']['counters']['native_search_calls'], 1)
        self.assertEqual(self.pair['pool']['metadata']['counters']['extra_search_calls'], 0)

    def test_pool_rescoring_matches_task11_exactly_and_has_own_band(self):
        from research.taskcard11.rescore import select_candidates
        pool = self.pair['pool']
        for rule, fs in [('original', False), ('fs', True)]:
            actual = selection.select_from_pool(pool, selection_rule=rule)
            expected = select_candidates(pool['metadata']['trace']['candidates'], 24, (2, 2, 2), fs)
            self.assertEqual(actual['selection'], expected)
            self.assertEqual(actual['q'], int(np.prod(actual['sizes'])))
            np.testing.assert_array_equal(actual['B'], pool['coefficients'][expected['selected_index']])

    def test_cache_roundtrip_is_lossless_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool'
            selection.save_pool(path, self.pair['pool'])
            loaded = selection.load_pool(path, X=self.X, y=self.y, **self.options)
            self.assertEqual(loaded['digest'], self.pair['pool']['digest'])
            for rule in ['original', 'fs']:
                actual = selection.select_from_pool(loaded, selection_rule=rule)
                np.testing.assert_array_equal(actual['B'], self.pair[rule]['B'])
                self.assertEqual(actual['selection'], self.pair[rule]['selection'])
            with self.assertRaises(FileExistsError):
                selection.save_pool(path, self.pair['pool'])

    def test_wrong_training_or_native_options_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool'
            selection.save_pool(path, self.pair['pool'])
            with self.assertRaisesRegex(ValueError, 'Training'):
                selection.load_pool(path, X=self.X, y=self.y + .1, **self.options)
            with self.assertRaisesRegex(ValueError, 'options'):
                selection.load_pool(path, X=self.X, y=self.y, **dict(self.options, max_cycles=3))

    def test_corrupt_disk_and_memory_are_rejected(self):
        broken = copy.deepcopy(self.pair['pool'])
        broken['coefficients'][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'hash|digest'):
            selection.select_from_pool(broken)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool'
            selection.save_pool(path, self.pair['pool'])
            with (path / 'coefficients.npz').open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'hash'):
                selection.verify_pool_files(path)

    def test_unknown_rule_or_forbidden_scoring_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selection_rule'):
            selection.select_from_pool(self.pair['pool'], selection_rule='auto')
        for options in [{'msic_refine_alpha': 2}, {'target_sparsity': (1, 1, 1)},
                        {'adaptive_criterion': 'sic'}, {'msic_refine_one_se_scale': 0}]:
            with self.assertRaises(ValueError):
                selection.fit_pair(self.X, self.y, **options)

    def test_identity_uses_relative_source_and_exact_environment(self):
        identity = self.pair['pool']['metadata']['identity']
        self.assertEqual(len(identity['core_sha256']), 32)
        self.assertTrue(all(not Path(name).is_absolute() for name in identity['source_sha256']))
        broken = copy.deepcopy(self.pair['pool'])
        broken['metadata']['identity']['environment']['python'] = 'different'
        broken['digest'] = selection.pool_digest(broken['metadata'], broken['coefficients'])
        with self.assertRaisesRegex(ValueError, 'environment|identity'):
            selection.select_from_pool(broken)


class HistoricalRegressionGates(unittest.TestCase):
    def test_saved_pilot_case_cache_and_replay_identity(self):
        try:
            history = importlib.import_module('research.taskcard12.selection_history')
        except ModuleNotFoundError:
            history = None
        self.assertIsNotNone(history, 'Task12 historical cache regression is not implemented')
        repo = Path(__file__).resolve().parents[2]
        run = repo / 'doc/taskcard11/results/pilot'
        replay = json.loads((repo / 'doc/taskcard11/verification/pilot_full_replay.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            result = history.audit_case(repo, run, run / 'pilot/task_000', Path(directory) / 'pool', replay)
        self.assertTrue(result['passed'])
        self.assertEqual(result['native_search_calls'], 0)
        self.assertTrue(result['saved_prediction_identity_exact'])
        self.assertEqual(result['candidate_count'], 65)


class NativeDomainGates(unittest.TestCase):
    def check_native_domain(self, X, y, *, zero_rss=False, rank_deficient=False):
        options = dict(criterion='bic', n_lambdas=3, max_cycles=2)
        native = selection.configure_core()(X, y, **options)
        self.assertTrue(np.isfinite(native.B).all())
        try:
            pair = selection.fit_pair(X, y, **options)
        except ValueError as error:
            self.fail('Official native input is valid but wrapper rejected it: ' + str(error))
        np.testing.assert_array_equal(pair['original']['B'], native.B)
        self.assertEqual(pair['original']['rss'], native.rss)
        self.assertEqual(pair['original']['score'], native.sic)
        audit = pair['pool']['metadata']['audit']
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['all_candidate_designs_full_rank'], not rank_deficient)
        if zero_rss:
            self.assertEqual(pair['original']['rss'], 0)
        if rank_deficient:
            self.assertTrue(any(c['rank_deficient'] for c in audit['checks']))
            self.assertTrue(all(c['condition_number'] is None for c in audit['checks'] if c['rank_deficient']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool'
            selection.save_pool(path, pair['pool'])
            restored = selection.load_pool(path, X=X, y=y, **options)
            for rule in ('original', 'fs'):
                recovered = selection.select_from_pool(restored, selection_rule=rule)
                np.testing.assert_array_equal(recovered['B'], pair[rule]['B'])
                np.testing.assert_array_equal(np.einsum('nijk,ijk->n', X, recovered['B']),
                                              np.einsum('nijk,ijk->n', X, pair[rule]['B']))

    def test_zero_response_matches_native_including_exact_zero_rss(self):
        rng = np.random.default_rng(12012020)
        self.check_native_domain(rng.normal(size=(24, 2, 2, 2)), np.zeros(24), zero_rss=True)

    def test_duplicate_features_match_native_minimum_norm_ols(self):
        rng = np.random.default_rng(12012020)
        X = np.broadcast_to(rng.normal(size=(24, 1, 1, 1)), (24, 2, 2, 2)).copy()
        self.check_native_domain(X, rng.normal(size=24), rank_deficient=True)

    def test_one_observation_uses_native_minimum_capacity(self):
        self.check_native_domain(np.ones((1, 1, 1, 1)), np.array([2.]), zero_rss=True)

    def test_two_observations_keep_native_legal_q_one(self):
        self.check_native_domain(np.array([1., 2.]).reshape(2, 1, 1, 1), np.array([2., 3.]))


if __name__ == '__main__':
    unittest.main()
