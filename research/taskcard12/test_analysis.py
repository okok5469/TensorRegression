"""Acceptance of frozen balanced/pressure inference, independent of cluster execution."""
import copy
import importlib
import unittest
from pathlib import Path
import json


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / 'doc/taskcard12'
        self.protocol = json.loads((source / 'REQUESTS.json').read_text(encoding='utf-8'))
        self.protocol['analysis'] = json.loads((source / 'ANALYSIS_SPEC.json').read_text(encoding='utf-8'))
        self.protocol['analysis']['bootstrap_draws'] = 64
        self.records = []
        for request in self.protocol['requests']:
            q = int(__import__('math').prod(request['config']['sparsity']))
            metrics = {'A0': {'pmse': 2., 'exact_recovery': 1., 'q': q},
                       'A1': {'pmse': 1.8, 'exact_recovery': 1., 'q': q}}
            self.records.append({'request': copy.deepcopy(request), 'metrics': metrics,
                'recovery_evidence': {'truth_shape_native_representable': True,
                                      'truth_support_found_full_pool': True}})

    def api(self):
        self.assertTrue((Path(__file__).parent / 'analysis.py').exists(), 'Frozen inference implementation missing')
        return importlib.import_module('research.taskcard12.analysis')

    def test_balanced_primary_does_not_reweight_extra_pressure(self):
        for row in self.records[360:]:
            row['metrics']['A1']['pmse'] = 100.
        result = self.api().analyze(self.records, self.protocol)
        self.assertAlmostEqual(result['groups']['balanced360']['pmse_mean_ratio'], .9)
        self.assertGreater(result['groups']['stress30']['pmse_mean_ratio'], 30.)
        self.assertEqual(result['groups']['n200_strong_representable']['count'], 60)

    def test_exact_loss_is_distinct_from_q_underselection(self):
        for row in self.records[360:]:
            row['metrics']['A1']['q'] -= 1
        result = self.api().analyze(self.records, self.protocol)
        self.assertEqual(result['groups']['stress30']['exact_losses'], 0)
        self.assertGreater(result['groups']['stress30']['q_under_delta'], .5)
        self.assertFalse(result['gates']['n200_strong_q_under_delta']['passed'])

    def test_missing_duplicate_and_request_drift_rejected(self):
        api = self.api()
        with self.assertRaises(ValueError): api.analyze(self.records[:-1], self.protocol)
        duplicate = copy.deepcopy(self.records); duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError): api.analyze(duplicate, self.protocol)
        drift = copy.deepcopy(self.records); drift[0]['request']['config']['snr'] = 7.
        with self.assertRaises(ValueError): api.analyze(drift, self.protocol)

    def test_nonfinite_metrics_rejected(self):
        self.records[0]['metrics']['A1']['pmse'] = float('nan')
        with self.assertRaises(ValueError): self.api().analyze(self.records, self.protocol)

    def test_exact_loss_upper_bound_and_empty_denominator(self):
        api = self.api()
        self.assertAlmostEqual(api.loss_upper_bound(0, 30), 1. - .05 ** (1. / 30))
        self.assertEqual(api.loss_upper_bound(30, 30), 1.)
        self.assertIsNone(api.loss_upper_bound(0, 0))

    def test_no_original_recovery_cannot_establish_loss_safety(self):
        for row in self.records:
            row['metrics']['A0']['exact_recovery'] = 0.
            row['metrics']['A1']['exact_recovery'] = 0.
        result = self.api().analyze(self.records, self.protocol)
        self.assertIsNone(result['groups']['stress30']['conditional_exact_loss_upper95'])
        self.assertFalse(result['gates']['stress_conditional_exact_loss_upper95']['passed'])
        self.assertFalse(result['eligible_for_version_review'])

    def test_frozen_bootstrap_reproducible_and_no_automatic_release(self):
        api = self.api()
        first = api.analyze(self.records, self.protocol)
        second = api.analyze(list(reversed(self.records)), self.protocol)
        self.assertEqual(first, second)
        self.assertTrue(first['eligible_for_version_review'])
        self.assertTrue(first['stable_paired_mean_gain'])
        self.assertFalse(first['release_approved'])
        self.assertEqual(first['groups']['stress30']['exact_loss_denominator'], 30)


if __name__ == '__main__':
    unittest.main()
