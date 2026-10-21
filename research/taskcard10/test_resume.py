"""Continuation contracts, with no scientific fitting."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    resume = importlib.import_module('research.taskcard10.resume')
except ModuleNotFoundError:
    resume = None


class ResumeContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(resume, 'Continuation runner is not implemented')

    def test_missing_queue_exact_and_duplicate_rejected(self):
        requests = [{'cohort': 'development', 'task_id': i} for i in range(48)]
        records = [{'request': r} for r in requests[:30]]
        self.assertEqual(resume.missing_development(requests, records), list(range(30, 48)))
        with self.assertRaises(ValueError):
            resume.missing_development(requests, records + [records[0]])
        with self.assertRaises(ValueError):
            resume.missing_development(requests, records[:-1])

    def test_full_coverage_requires_every_original_request(self):
        requests = [{'cohort': 'development', 'task_id': i} for i in range(48)]
        records = [{'request': r} for r in requests]
        self.assertEqual(len(resume.exact_development(requests, records)), 48)
        with self.assertRaises(ValueError):
            resume.exact_development(requests, records[:-1])
        changed = copy.deepcopy(records)
        changed[1]['request']['task_id'] = 0
        with self.assertRaises(ValueError):
            resume.exact_development(requests, changed)

    def test_only_compute_and_provenance_protocol_changes_allowed(self):
        old = {'formal': {'workers': 4, 'wall_time_limit_seconds': 7200,
                         'case_timeout_seconds': 900, 'replicates': 2},
               'source_files': {'old.py': 'a'}, 'source_identity': {'old.py': 'a'},
               'promotion_gates': {'same': .98}, 'cv_tie': 'same'}
        new = copy.deepcopy(old)
        new['formal'].update(workers=2, wall_time_limit_seconds=28800, case_timeout_seconds=3600)
        new['continuation'] = {'from': 'old'}
        resume.assert_science_unchanged(old, new)
        changed = copy.deepcopy(new)
        changed['promotion_gates']['same'] = 1.
        with self.assertRaises(ValueError):
            resume.assert_science_unchanged(old, changed)
        changed = copy.deepcopy(new)
        changed['formal']['replicates'] = 1
        with self.assertRaises(ValueError):
            resume.assert_science_unchanged(old, changed)

    def test_extra_scientific_dependency_is_rejected(self):
        old = {'formal': {'workers': 4, 'wall_time_limit_seconds': 7200,
                         'case_timeout_seconds': 900, 'replicates': 2},
               'source_files': {'old.py': 'a'}, 'source_identity': {'old.py': 'a'}}
        new = copy.deepcopy(old)
        new['formal'].update(workers=2, wall_time_limit_seconds=28800, case_timeout_seconds=3600)
        new['continuation'] = {'from': 'old'}
        new['source_files']['research/taskcard10/resume.py'] = 'b'
        new['source_identity']['research/taskcard10/resume.py'] = 'b'
        with self.assertRaises(ValueError):
            resume.assert_science_unchanged(old, new)

    def test_partial_input_identity_is_exact(self):
        saved = {'input_fingerprints': {'X_train': 'x', 'y_train': 'y', 'B_true': 'b', 'supports_true': ['s']}}
        resume.assert_partial_identity(saved, copy.deepcopy(saved))
        changed = copy.deepcopy(saved)
        changed['input_fingerprints']['y_train'] = 'other'
        with self.assertRaises(ValueError):
            resume.assert_partial_identity(saved, changed)

    def test_byte_reuse_preserves_old_complete_identity_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root/'old', root/'new'
            case = old/'development/task_000'
            case.mkdir(parents=True)
            (case/'complete.json').write_text(json.dumps({'plan_sha256': 'old-plan'}), encoding='utf-8')
            (case/'raw.bin').write_bytes(b'exact original bytes')
            manifest = resume.copy_reused_cases(old, new, [0])
            resume.verify_reused_cases(old, new, manifest)
            self.assertEqual(json.loads((new/'development/task_000/complete.json').read_text())['plan_sha256'], 'old-plan')
            (new/'development/task_000/raw.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                resume.verify_reused_cases(old, new, manifest)

    def test_budget_clock_is_not_reset_and_remaining_is_shared(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = resume.shared_clock(output, now=100.)
            second = resume.shared_clock(output, now=900.)
            self.assertEqual(first, second)
            self.assertEqual(resume.remaining_seconds(output, now=900.), 28000.)

    def test_failed_or_missing_gate_cannot_enter_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            resume.shared_clock(output, now=100.)
            with self.assertRaises(ValueError):
                resume.require_confirmation(output, now=101.)
            (output/'confirmation_budget_gate.json').write_text('{"passed": false}', encoding='utf-8')
            with self.assertRaises(ValueError):
                resume.require_confirmation(output, now=101.)

    def test_confirmation_reserve_uses_all48_case_costs(self):
        records = [{'resources': {'wall_seconds': 100.}} for _ in range(48)]
        self.assertEqual(resume.confirmation_reserve(records), 3660.)
        with self.assertRaises(ValueError):
            resume.confirmation_reserve(records[:-1])

    def test_confirmation_budget_reserve_must_still_hold_at_launch(self):
        self.assertTrue(hasattr(resume, 'assert_remaining_reserve'), 'Launch-time shared reserve check is missing')
        resume.assert_remaining_reserve({'passed': True, 'required_seconds': 3600.}, 3600.)
        with self.assertRaises(ValueError):
            resume.assert_remaining_reserve({'passed': True, 'required_seconds': 3600.}, 3599.)
        with self.assertRaises(ValueError):
            resume.assert_remaining_reserve({'passed': False, 'required_seconds': 1.}, 5000.)

    def test_cli_stopped_science_returns_nonzero(self):
        with patch('sys.argv', ['resume.py', 'run-dev']), \
             patch.object(resume, 'run_development', return_value={'status': 'FAILED'}):
            with self.assertRaises(SystemExit) as stopped:
                resume.main()
        self.assertEqual(stopped.exception.code, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
