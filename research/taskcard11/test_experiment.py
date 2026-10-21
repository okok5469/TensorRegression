"""Task11 runner gates use tiny deterministic fixtures, never new DGP fits."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    experiment = importlib.import_module('research.taskcard11.experiment')
except ModuleNotFoundError:
    experiment = None


class RunnerGates(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiment, 'Task11 runner is not implemented')

    def test_validation_is_exact_24_cells_four_independent_inputs(self):
        rows = experiment.build_requests({}, 'validation', 4, 1601810000)
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r['cell_id'] for r in rows}), 24)
        self.assertEqual(len({r[k] for r in rows for k in ('seed', 'test_seed')}), 192)
        self.assertTrue(all(r['test_seed'] == r['seed'] + 50_000_000 for r in rows))
        experiment.validate_requests(rows, 'formal')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            experiment.validate_requests(rows[:-1], 'formal')

    def test_pilot_covers_both_n_and_snr_and_expensive_shapes(self):
        rows = experiment.build_requests({}, 'pilot', 1, 1201810000)
        self.assertEqual(len(rows), 6)
        self.assertEqual({(r['config']['n_train'], r['config']['snr']) for r in rows},
                         {(100, .5), (100, 4.), (400, .5), (400, 4.)})
        large = [r for r in rows if r['config']['n_train'] == 400]
        self.assertEqual({(tuple(r['config']['sparsity']), r['config']['rho'][0]) for r in large},
                         {((4, 3, 5), 0.), ((4, 3, 5), .6), ((2, 3, 4), 0.), ((2, 3, 4), .6)})

    def test_duplicate_new_or_historical_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            old = repo / 'numerical_simulation/outputs_old/plan.json'
            old.parent.mkdir(parents=True)
            old.write_text(json.dumps({'nested': [{'test_seed': 77}]}))
            with self.assertRaisesRegex(ValueError, 'historical'):
                experiment.audit_seed_novelty(repo, [{'seed': 1, 'test_seed': 77}])
            with self.assertRaisesRegex(ValueError, 'collision'):
                experiment.audit_seed_novelty(repo, [{'seed': 1, 'test_seed': 1}])

    def test_input_identity_claim_prevents_same_run_race_and_history_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fp = {'X_train': 'x', 'y_train': 'y'}
            experiment.claim_input(root, fp, [], {'cohort': 'pilot', 'task_id': 0})
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                experiment.claim_input(root, fp, [], {'cohort': 'pilot', 'task_id': 1})
            with self.assertRaisesRegex(ValueError, 'historical'):
                experiment.claim_input(root, {'X_train': 'a', 'y_train': 'b'},
                    [{'X_train': 'a', 'y_train': 'b', 'path': 'old'}], {})

    def test_historical_input_scan_excludes_only_current_output_before_read(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            current = repo / 'doc/taskcard11/results/formal'
            partial = current / 'validation/task_001/input_identity.json'
            partial.parent.mkdir(parents=True)
            partial.write_text('{"input_fingerprints":', encoding='utf-8')
            pilot = repo / 'doc/taskcard11/results/pilot/pilot/task_000/input_identity.json'
            experiment.write_json(pilot, {'input_fingerprints': {'X_train': 'pilot_x', 'y_train': 'pilot_y'}})
            self.assertIn('exclude_output', __import__('inspect').signature(experiment.historical_inputs).parameters,
                          'Current-run identities must be excluded before reading partial files')
            known = experiment.historical_inputs(repo, exclude_output=current)
            self.assertEqual(len(known), 1)
            self.assertEqual(known[0]['X_train'], 'pilot_x')
            with self.assertRaisesRegex(ValueError, 'historical'):
                experiment.claim_input(current, {'X_train': 'pilot_x', 'y_train': 'pilot_y'}, known, {})
            with self.assertRaises(json.JSONDecodeError):
                experiment.historical_inputs(repo)

    def test_budget_never_exceeds_two_workers_one_hour_case_eight_hours_total(self):
        for workers, total, case in [(3, 10, 5), (2, 28801, 5), (2, 100, 3601)]:
            with self.assertRaisesRegex(ValueError, 'budget'):
                experiment.validate_budget({'workers': workers, 'wall_time_limit_seconds': total,
                                             'case_timeout_seconds': case})
        experiment.validate_budget({'workers': 2, 'wall_time_limit_seconds': 28800,
                                     'case_timeout_seconds': 3600})

    def test_shared_clock_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            first = experiment.shared_clock(temp, 28800, now=100.)
            second = experiment.shared_clock(temp, 28800, now=1000.)
            self.assertEqual(first, second)
            self.assertEqual(experiment.remaining_seconds(temp, now=110.), 28790.)
            with self.assertRaisesRegex(ValueError, 'budget'):
                experiment.shared_clock(temp, 30000, now=101.)

    def test_a0_must_match_actual_solver_B_support_and_score_exactly(self):
        B = np.ones((1, 1, 1))
        row = {'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
               'rss': 10., 'msic': 3., 'B_sha256': experiment.array_sha(B)}
        fit = SimpleNamespace(B=B.copy(), S1=np.array([0]), S2=np.array([0]),
                              S3=np.array([0]), sic=3., rss=10.)
        choice = {'selected_index': 0, 'selected_score': 3.}
        experiment.assert_native_identity(fit, {'candidates': [row], 'final': row}, [B], choice)
        for change in ('B', 'S1', 'sic'):
            broken = copy.deepcopy(fit)
            setattr(broken, change, {'B': B + 1e-15, 'S1': np.array([1]), 'sic': 3. + 1e-15}[change])
            with self.assertRaisesRegex(ValueError, 'A0'):
                experiment.assert_native_identity(broken, {'candidates': [row], 'final': row}, [B], choice)

    def test_all_candidate_OLS_audit_rejects_corrupt_coefficient(self):
        from scripts.taskcard8.selection import terms
        X = np.arange(1., 21.).reshape(20, 1, 1, 1)
        y = np.array([2. * i + (i % 3) for i in range(1, 21)])
        beta = np.linalg.lstsq(X.reshape(20, 1), y, rcond=None)[0]
        B = beta.reshape(1, 1, 1)
        residual = y - X.reshape(20, 1) @ beta
        row = {'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
               'rss': float(residual @ residual), 'B_sha256': experiment.array_sha(B)}
        row['msic'] = terms(row, 20, (1, 1, 1), 1.5)['score']
        trace = {'candidates': [row, row.copy()], 'ordered_size_grid': [[1, 1, 1]],
                 'caps': [1, 1, 1], 'fixed_searches': [{}]}
        audit = experiment.audit_candidates(X, y, trace, [B, B])
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['candidate_count'], 2)
        with self.assertRaises((AssertionError, ValueError)):
            experiment.audit_candidates(X, y, trace, [B, B + 1.])

    def test_illegal_candidate_q_stops_before_selection(self):
        X = np.ones((2, 1, 1, 1)); y = np.array([1., 2.]); B = np.ones((1, 1, 1))
        trace = {'candidates': [{'q': 1, 'sizes': [1, 1, 1], 'rss': 1.}],
                 'ordered_size_grid': [], 'caps': [1, 1, 1], 'fixed_searches': []}
        with self.assertRaisesRegex(ValueError, 'n-1'):
            experiment.audit_candidates(X, y, trace, [B])

    def test_complete_collection_requires_exact_plan_and_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            request = {'cohort': 'pilot', 'task_id': 0}
            experiment.write_json(output / 'plan.json', {'requests': [request]})
            with self.assertRaisesRegex(ValueError, 'complete'):
                experiment.collect(output, 'pilot')
            folder = output / 'pilot/task_000'
            experiment.write_json(folder / 'evidence.json', {'passed': True})
            experiment.write_json(folder / 'complete.json', {'request': request,
                'plan_sha256': experiment.file_sha(output / 'plan.json'),
                'artifact_sha256': {'evidence.json': experiment.file_sha(folder / 'evidence.json')}})
            self.assertEqual(len(experiment.collect(output, 'pilot')), 1)
            (folder / 'evidence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                experiment.collect(output, 'pilot')

    def test_run_two_arms_performs_one_native_search_and_reuses_all_coefficients(self):
        self.assertTrue(hasattr(experiment, 'run_two_arms'), 'Paired native runner missing')
        from scripts.taskcard8.selection import terms
        X = np.arange(1., 21.).reshape(20, 1, 1, 1)
        y = np.array([2. * i + (i % 3) for i in range(1, 21)])
        B = np.linalg.lstsq(X.reshape(20, 1), y, rcond=None)[0].reshape(1, 1, 1)
        residual = y - X.reshape(20, 1) @ B.ravel()
        row = {'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
               'rss': float(residual @ residual), 'B_sha256': experiment.array_sha(B)}
        row['msic'] = terms(row, 20, (1, 1, 1), 1.5)['score']
        trace = {'candidates': [row, row.copy()], 'ordered_size_grid': [[1, 1, 1]],
                 'caps': [1, 1, 1], 'fixed_searches': [{}], 'final': row,
                 'eligible_candidate_indices': [0, 1],
                 'one_se_tolerance': float(np.sqrt(np.log(20)))}
        fitted = SimpleNamespace(B=B, S1=np.array([0]), S2=np.array([0]),
                                 S3=np.array([0]), rss=row['rss'], sic=row['msic'])
        observed = []
        def capture(fit):
            observed.append(fit)
            return fitted, trace, [B, B.copy()]
        fit = experiment.run_two_arms(X, y, {}, native_fit=lambda: None, observer=capture)
        self.assertEqual(len(observed), 1)
        self.assertEqual(fit['summary']['counters']['native_search_calls'], 1)
        self.assertEqual(fit['summary']['counters']['extra_search_calls'], 0)
        for label in ('A0', 'A1'):
            index = fit['summary']['arms'][label]['full_candidate_index']
            np.testing.assert_array_equal(fit['arrays'][label + '_B'], fit['arrays']['full_B'][index])
        self.assertTrue(fit['audit']['passed'])

    def test_stage_refuses_missing_frozen_protocol_before_launch(self):
        self.assertTrue(hasattr(experiment, 'run_stage'), 'Bounded stage runner missing')
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            experiment.write_json(repo / 'plan.json', {'protocol': {'source_files': {}, 'core_sha256': {}},
                                                      'protocol_path': 'missing', 'protocol_sha256': 'bad'})
            with patch.object(experiment.previous, 'run_subprocess_batch') as launch:
                with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
                    experiment.run_stage(repo, repo, 'pilot')
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
