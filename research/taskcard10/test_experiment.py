"""Controller safety and independent prediction tests; no scientific fits."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    experiment = importlib.import_module('research.taskcard10.experiment')
except ModuleNotFoundError:
    experiment = None


class ExperimentGates(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiment, 'Task10 experiment controller is not implemented')

    def test_pilot_is_four_new_families_with_reserved_streams(self):
        rows = experiment.build_requests({'dims': [32, 20, 25]}, 'pilot', 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual([(r['config']['n_train'], r['config']['snr']) for r in rows],
                         [(100, .5), (100, 4.), (400, .5), (400, 4.)])
        self.assertEqual({tuple(r['config']['sparsity']) for r in rows}, {(4, 3, 5)})
        self.assertTrue(all(r['config']['rho'] == [.6, .6, .6] for r in rows))
        self.assertEqual(len({r[k] for r in rows for k in ('seed', 'test_seed', 'split_seed')}), 12)

    def test_development_and_confirmation_are_disjoint_balanced_24_cells(self):
        development = experiment.build_requests({}, 'development', 2)
        confirmation = experiment.build_requests({}, 'confirmation', 2)
        self.assertEqual(len(development), 48)
        self.assertEqual(len({r['cell_id'] for r in development}), 24)
        self.assertEqual(len({r[k] for r in development + confirmation
                              for k in ('seed', 'test_seed', 'split_seed')}), 288)
        self.assertTrue(all(r['replicate'] in (0, 1) for r in development))

    def test_historical_seed_audit_catches_nested_test_seed_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            old = repo / 'numerical_simulation/outputs_old'
            old.mkdir(parents=True)
            (old / 'plan.json').write_text(json.dumps({'nested': [{'test_seed': 77}]}))
            rows = [{'seed': 1, 'test_seed': 77, 'split_seed': 99}]
            with self.assertRaisesRegex(ValueError, 'historical'):
                experiment.audit_seed_novelty(repo, rows)

    def test_request_seeds_must_not_reuse_each_other(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'collision'):
                experiment.audit_seed_novelty(Path(temp), [
                    {'seed': 1, 'test_seed': 2, 'split_seed': 3},
                    {'seed': 3, 'test_seed': 4, 'split_seed': 5}])

    def test_json_and_npz_artifacts_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            experiment.write_json(folder / 'a.json', {'a': 1})
            with self.assertRaises(FileExistsError):
                experiment.write_json(folder / 'a.json', {'a': 2})
            experiment.write_npz(folder / 'a.npz', {'B': np.ones((2, 2))})
            with self.assertRaises(FileExistsError):
                experiment.write_npz(folder / 'a.npz', {'B': np.zeros((2, 2))})

    def test_source_fingerprint_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / 'solver.py').write_text('original')
            identity = {'solver.py': experiment.file_sha(repo / 'solver.py')}
            experiment.verify_sources(repo, identity)
            (repo / 'solver.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source'):
                experiment.verify_sources(repo, identity)

    def test_pool_recovery_distinguishes_shape_from_actual_support(self):
        self.assertTrue(hasattr(experiment, 'summarize_recovery'), 'Pool recovery reporting is missing')
        summary = {'public_capacity': {'full_caps': [2, 2, 2], 'common_caps': [1, 1, 1],
                                      'common_Q': 1},
                   'full_trace': {'candidates': [
                       {'sizes': [1, 1, 1], 'support': [[0], [0], [0]]},
                       {'sizes': [1, 1, 1], 'support': [[1], [0], [0]]}]},
                   'common_grid': [[1, 1, 1]], 'cv_curve': [{'full_candidate_index': 1}]}
        row = experiment.summarize_recovery(summary, [[0], [0], [0]], 100)
        self.assertEqual(row['q_true'], 1)
        self.assertTrue(row['truth_shape_common_representable'])
        self.assertTrue(row['truth_support_found_full_pool'])
        self.assertFalse(row['truth_support_found_common_pool'])

    def test_process_launch_failure_is_saved_as_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            try:
                result = experiment.run_subprocess_batch(
                    [{'task_id': 0, 'command': ['taskcard10_nonexistent_program']}], out,
                    workers=1, total_seconds=1, case_seconds=1)
            except OSError:
                self.fail('Launch failure escaped without its terminal evidence record')
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['failed'][0]['reason'], 'launch_failed')
            self.assertTrue((out / 'stopped.json').is_file())

    def test_prediction_stream_shared_and_no_model_selection_inside(self):
        X = np.arange(12., dtype=float).reshape(3, 2, 2, 1)
        B0 = np.zeros((2, 2, 1)); B1 = np.ones((2, 2, 1))
        y = np.array([1., 2., 3.]); signal = np.array([0., 1., 2.])
        batches = [(X[:2], y[:2], signal[:2]), (X[2:], y[2:], signal[2:])]
        rows, fingerprint = experiment.evaluate_test_stream(batches, {'A0': B0, 'A1': B1})
        self.assertAlmostEqual(rows['A0']['pmse'], float(y @ y) / 3)
        residual = y - X.reshape(3, -1).sum(axis=1)
        self.assertAlmostEqual(rows['A1']['pmse'], float(residual @ residual) / 3)
        self.assertEqual(fingerprint['n_observations'], 3)
        self.assertEqual(fingerprint['n_batches'], 2)
        self.assertEqual(len(fingerprint['sha256']), 64)

    def test_confirmation_requires_complete_artifact_lock_and_passing_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            experiment.write_json(folder / 'plan.json', {'requests': []})
            with self.assertRaisesRegex(ValueError, 'lock'):
                experiment.confirmation_lock(folder)
            experiment.write_json(folder / 'development_lock.json', {
                'plan_sha256': experiment.file_sha(folder / 'plan.json'),
                'entry_gate': {'passed': False}, 'development_artifacts': {}})
            with self.assertRaisesRegex(ValueError, 'entry'):
                experiment.confirmation_lock(folder)

    def test_failed_worker_preserves_failure_and_stops_dependents(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            commands = [{'task_id': 0, 'command': [sys.executable, '-c', 'raise SystemExit(3)']},
                        {'task_id': 1, 'command': [sys.executable, '-c', "print('must not launch')"]}]
            result = experiment.run_subprocess_batch(commands, out, workers=1,
                                                     total_seconds=10, case_seconds=5)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['completed'], [])
            self.assertEqual(result['not_started'], [1])
            self.assertTrue((out / 'stopped.json').is_file())
            self.assertIn('3', (out / 'task_000/stderr.log').read_text() +
                          json.dumps(result['failed']))

    def test_timeout_is_terminal_and_no_budget_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            commands = [{'task_id': 0, 'command': [sys.executable, '-c', 'import time; time.sleep(5)']}]
            result = experiment.run_subprocess_batch(commands, out, workers=1,
                                                     total_seconds=2, case_seconds=.15)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['failed'][0]['reason'], 'case_timeout')
            self.assertLess(result['elapsed_seconds'], 4)
            with self.assertRaises(FileExistsError):
                experiment.run_subprocess_batch(commands, out, workers=1,
                                                total_seconds=2, case_seconds=.15)

    @unittest.skipUnless(os.name == 'nt', 'Windows owned-job descendant contract')
    def test_windows_timeout_terminates_created_interpreter_grandchild(self):
        import ctypes
        from ctypes import wintypes
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            child_pid = base / 'owned_grandchild.pid'
            marker = base / 'orphan_computation.txt'
            child_code = ('import os,time;from pathlib import Path;'
                          f'Path({str(child_pid)!r}).write_text(str(os.getpid()));'
                          f'time.sleep(5);Path({str(marker)!r}).write_text("orphan")')
            parent_code = ('import subprocess,sys,time;'
                           f'subprocess.Popen([sys.executable,"-c",{child_code!r}]);time.sleep(10)')
            result = experiment.run_subprocess_batch([{'task_id': 0,
                'command': [sys.executable, '-c', parent_code]}], base / 'run',
                workers=1, total_seconds=3, case_seconds=.8)
            self.assertEqual(result['failed'][0]['reason'], 'case_timeout')
            self.assertTrue(child_pid.is_file(), 'Owned descendant must really execute before timeout')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            open_process = kernel.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            handle = open_process(0x00100000, False, int(child_pid.read_text()))  # SYNCHRONIZE only
            if handle:
                wait = kernel.WaitForSingleObject
                wait.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                wait.restype = wintypes.DWORD
                self.assertEqual(wait(handle, 1000), 0, 'Owned descendant remained alive after job close')
                close = kernel.CloseHandle
                close.argtypes = [wintypes.HANDLE]
                close(handle)
            else:
                self.assertEqual(ctypes.get_last_error(), 87, 'Unexpected denial opening owned descendant')
            self.assertFalse(marker.exists(), 'Timed-out descendant continued computation')


if __name__ == '__main__':
    unittest.main()
