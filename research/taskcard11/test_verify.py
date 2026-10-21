"""Independent verification must reject corrupted saved scientific evidence."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    verify = importlib.import_module('research.taskcard11.verify')
except ModuleNotFoundError:
    verify = None


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verify, 'Independent Task11 evidence verifier is absent')

    def selection_fixture(self):
        from research.taskcard11.rescore import select_candidates
        from scripts.taskcard7.lemon import array_sha
        coefficients = np.asarray([[[[1.]]], [[[2.]]]])
        rows = [{'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
                 'rss': rss, 'B_sha256': array_sha(B)}
                for rss, B in zip((20., 10.), coefficients)]
        choices = {arm: select_candidates(rows, 20, (1, 1, 1), fs)
                   for arm, fs in (('A0', False), ('A1', True))}
        for i, row in enumerate(rows):
            row['msic'] = choices['A0']['candidate_scores'][i]
        arms = {arm: dict(rows[c['selected_index']], full_candidate_index=c['selected_index'],
                         selected_score=c['selected_score']) for arm, c in choices.items()}
        return rows, coefficients, choices, arms

    def test_wrong_saved_selection_is_rejected_even_with_valid_coefficients(self):
        rows, coefficients, choices, arms = self.selection_fixture()
        self.assertTrue(verify.audit_selection(rows, coefficients, choices, arms, 20, (1, 1, 1))['passed'])
        corrupted = copy.deepcopy(choices)
        corrupted['A1']['selected_index'] = 0
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_selection(rows, coefficients, corrupted, arms, 20, (1, 1, 1))

    def test_unselected_candidate_corruption_is_rejected(self):
        rows, coefficients, choices, arms = self.selection_fixture()
        coefficients[0, 0, 0, 0] += .01
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_selection(rows, coefficients, choices, arms, 20, (1, 1, 1))

    def test_score_decomposition_corruption_is_rejected(self):
        rows, coefficients, choices, arms = self.selection_fixture()
        choices['A1']['candidate_terms'][1]['fs_penalty'] += .1
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_selection(rows, coefficients, choices, arms, 20, (1, 1, 1))

    def test_wrong_pmse_is_rejected_from_saved_predictions(self):
        saved = {'y': np.array([1., 3.]), 'signal': np.array([1., 2.]),
                 'prediction': np.array([[1., 2.], [2., 4.]])}
        metrics = {'A0': {'pmse': .5, 'signal_mse': 0.},
                   'A1': {'pmse': 1., 'signal_mse': 2.5}}
        verify.audit_saved_predictions(saved, metrics, 2)
        metrics['A0']['pmse'] = .4
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_saved_predictions(saved, metrics, 2)

    def test_recovery_must_match_saved_truth_and_support(self):
        from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
        B = np.zeros((2, 1, 1)); B[0] = 2.
        truth = np.zeros_like(B); truth[1] = 2.
        estimated = [[0], [0], [0]]; true = [[1], [0], [0]]
        metrics = {**compute_coefficient_metrics(B, truth),
                   **compute_support_metrics(*map(np.asarray, estimated), *map(np.asarray, true)),
                   'q': 1, 'sizes': [1, 1, 1], 'support': estimated,
                   'coefficient_norm': 2.}
        verify.audit_recovery(B, truth, estimated, true, metrics)
        metrics['exact_recovery'] = 1.
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_recovery(B, truth, estimated, true, metrics)

    def test_hash_manifest_cannot_omit_required_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'data.json').write_text('{}', encoding='utf-8')
            hashes = {'data.json': verify.file_sha(folder / 'data.json')}
            verify.verify_artifact_hashes(folder, hashes, {'data.json'})
            with self.assertRaises((AssertionError, ValueError)):
                verify.verify_artifact_hashes(folder, {}, {'data.json'})
            (folder / 'data.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaises((AssertionError, ValueError)):
                verify.verify_artifact_hashes(folder, hashes, {'data.json'})

    def test_replayed_stream_rejects_saved_prediction_from_wrong_model(self):
        from research.taskcard10.experiment import evaluate_test_stream
        X = np.array([1., 2.]).reshape(2, 1, 1, 1)
        y = np.array([1., 3.]); signal = np.array([1., 2.])
        coefficients = {'A0': np.ones((1, 1, 1)), 'A1': np.full((1, 1, 1), 2.)}
        batches = []
        _, fingerprint = evaluate_test_stream(iter([(X, y, signal)]), coefficients, batches)
        saved = {key: batches[0][key] for key in ('y', 'signal', 'prediction')}
        self.assertTrue(verify.audit_test_replay(iter([(X, y, signal)]), coefficients, saved, fingerprint)['passed'])
        saved['prediction'] = saved['prediction'].copy()
        saved['prediction'][0, 0] += .1
        with self.assertRaises((AssertionError, ValueError)):
            verify.audit_test_replay(iter([(X, y, signal)]), coefficients, saved, fingerprint)

    def test_incomplete_frozen_cohort_never_claims_verification_complete(self):
        self.assertTrue(hasattr(verify, 'verify_run'), 'Whole-cohort verifier is absent')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'plan.json').write_text(json.dumps({'requests': [
                {'cohort': 'validation', 'task_id': i} for i in range(96)]}), encoding='utf-8')
            result = verify.verify_run(root, root, 'validation')
            self.assertEqual(result['status'], 'NOT_RUN')
            self.assertEqual(result['completed'], 0)
            self.assertEqual(result['planned'], 96)

    def test_frozen_protocol_hash_drift_stops_before_case_replay(self):
        self.assertTrue(hasattr(verify, 'verify_envelope'), 'Case envelope verifier is absent')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root / 'case'; folder.mkdir()
            (root / 'protocol.json').write_text('{}', encoding='utf-8')
            (root / 'plan.json').write_text(json.dumps({'protocol_path': 'protocol.json',
                'protocol_sha256': 'wrong'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protocol'):
                verify.verify_envelope(root, root, folder)

    def test_repeated_replay_core_configuration_preserves_isolation(self):
        self.assertTrue(hasattr(verify, 'configure_replay_core'), 'Repeated replay core lifecycle check is absent')
        code = '''
from pathlib import Path
from types import SimpleNamespace
import sys
from research.taskcard11.verify import configure_replay_core
source = configure_replay_core(Path.cwd())
assert configure_replay_core(Path.cwd()) == source
sys.modules['tensor_abess.false_import'] = SimpleNamespace(__file__=str(Path.cwd()/'outside.py'))
try:
    configure_replay_core(Path.cwd())
except ValueError as error:
    assert 'escaped' in str(error)
else:
    raise AssertionError('Escaped preloaded core was accepted')
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
