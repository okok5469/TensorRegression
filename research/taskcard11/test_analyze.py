"""Decision tests: complete pairing, denominator, boundaries and uncertainty."""
import copy
import importlib
import unittest

try:
    analysis = importlib.import_module('research.taskcard11.analyze')
except ModuleNotFoundError:
    analysis = None


def fixture():
    requests, records = [], []
    for cell in range(24):
        for rep in range(4):
            req = {'task_id': cell * 4 + rep, 'cell_id': cell, 'replicate': rep,
                   'config': {'n_train': 100 if cell < 12 else 400,
                              'snr': (.5, 2., 4.)[(cell % 12) // 4],
                              'sparsity': [4, 3, 5], 'rho': [0., 0., 0.]}}
            metric = {'pmse': 2., 'q': 60, 'exact_recovery': 1., 'ree': .1,
                      'element_f1': 1., 'sizes': [4, 3, 5]}
            requests.append(req)
            records.append({'request': copy.deepcopy(req),
                            'metrics': {'A0': dict(metric), 'A1': dict(metric, pmse=1.8)}})
    return requests, records


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(analysis, 'Task11 paired decision implementation absent')

    def test_incomplete_or_duplicate_case_cannot_pass_gates(self):
        req, rows = fixture()
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                analysis.evaluate_gates(bad, req)

    def test_changed_request_identity_is_rejected(self):
        req, rows = fixture()
        rows[0]['request']['config']['snr'] = 100.
        with self.assertRaises(ValueError):
            analysis.evaluate_gates(rows, req)

    def test_all_seven_statistical_gates_pass_clear_improvement(self):
        req, rows = fixture()
        result = analysis.evaluate_gates(rows, req)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['criteria']), 7)
        self.assertAlmostEqual(result['criteria']['overall_pmse_ratio']['value'], .9)

    def test_scenario_is_mean_ratio_not_max_case_ratio(self):
        req, rows = fixture()
        rows[0]['metrics']['A1']['pmse'] = 5.
        values = analysis.evaluate_gates(rows, req)['criteria']
        self.assertAlmostEqual(values['worst_cell_pmse_ratio']['value'], 1.3)
        self.assertFalse(values['worst_cell_pmse_ratio']['passed'])

    def test_exact_and_strong_underuse_have_separate_denominators(self):
        req, rows = fixture()
        for row in rows[48:51]:
            row['metrics']['A1']['exact_recovery'] = 0.
        strong = [r for r in rows if r['request']['config']['snr'] == 4.]
        for row in strong[:2]:
            row['metrics']['A1']['q'] = 59
        values = analysis.evaluate_gates(rows, req)['criteria']
        self.assertEqual(values['n400_exact_delta']['value'], -3/48)
        self.assertEqual(values['strong_under_delta']['value'], 2/32)
        self.assertFalse(values['n400_exact_delta']['passed'])
        self.assertFalse(values['strong_under_delta']['passed'])

    def test_invalid_pmse_never_silently_enters_mean(self):
        req, rows = fixture()
        for value in (0., float('nan'), float('inf'), -1.):
            rows[0]['metrics']['A0']['pmse'] = value
            with self.assertRaises(ValueError):
                analysis.evaluate_gates(rows, req)

    def test_paired_bootstrap_preserves_identical_arm_difference(self):
        _, rows = fixture()
        for row in rows:
            row['metrics']['A1']['pmse'] = row['metrics']['A0']['pmse']
        ci = analysis.paired_uncertainty(rows, draws=200, seed=11011011)
        self.assertEqual(ci['overall']['ratio_ci95'], [1., 1.])
        self.assertEqual(ci['n100']['delta_ci95'], [0., 0.])
        self.assertFalse(ci['stable_gain'])


if __name__ == '__main__':
    unittest.main()
