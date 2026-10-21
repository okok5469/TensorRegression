"""Report aggregation tests use deterministic records, never scientific inputs."""
from __future__ import annotations

import copy
import importlib
import importlib.util
from itertools import product
import math
import unittest

from research.taskcard11 import analyze


def fixture():
    requests, records, summaries = [], [], {}
    cells = product((100, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)), (0., .6))
    for cell, (n, snr, shape, rho) in enumerate(cells):
        for rep in range(4):
            task = cell * 4 + rep
            request = {'cohort': 'validation', 'task_id': task, 'cell_id': cell, 'replicate': rep,
                       'seed': 100000 + cell * 1000 + rep, 'test_seed': 50100000 + cell * 1000 + rep,
                       'config': {'n_train': n, 'snr': snr, 'sparsity': list(shape),
                                  'rho': [rho] * 3, 'dims': [32, 20, 25]}}
            sizes = [2, 3, 4]
            support = [list(range(s)) for s in sizes]
            candidate = {'sizes': sizes, 'q': 24, 'rss': 100., 'support': support,
                         'B_sha256': str(task), 'msic': 20.}
            term = {'fit_term': 1., 'df_penalty': 10., 'multiplicity_penalty': 9.,
                    'baseline_score': 20., 'fs_penalty': 0., 'score': 20.}
            choice = {'selected_index': 0, 'selected_score': 20., 'minimum_index': 0,
                      'minimum_q': 24, 'minimum_score': 20., 'tolerance': 2.,
                      'candidate_scores': [20.], 'candidate_terms': [term]}
            fs = copy.deepcopy(choice)
            fs['selected_score'] = fs['minimum_score'] = 21.
            fs['candidate_scores'] = [21.]
            fs['candidate_terms'][0].update(fs_penalty=1., score=21.)
            summary = {'config': copy.deepcopy(request['config']), 'full_trace': {'candidates': [candidate]},
                       'choices': {'A0': choice, 'A1': fs},
                       'arms': {a: dict(candidate, full_candidate_index=0, candidate_source='adaptive_terminal')
                                for a in ('A0', 'A1')}}
            truth = [list(range(s)) for s in shape]
            exact = int(truth == support)
            metric = dict(candidate, pmse=2., ree=.25, element_f1=.75, mode_f1_macro=.8,
                          exact_recovery=exact)
            record = {'request': copy.deepcopy(request), 'metrics': {'A0': metric, 'A1': dict(metric)},
                      'recovery_evidence': {'q_true': math.prod(shape), 'truth_sizes': list(shape),
                          'truth_support': truth, 'Q_native': math.floor(.7 * n),
                          'truth_shape_native_representable': n != 100 or shape == (2, 3, 4),
                          'truth_shape_found_full_pool': exact == 1, 'truth_support_found_full_pool': exact == 1}}
            requests.append(request)
            records.append(record)
            summaries[task] = summary
    return requests, records, summaries


class ReportTests(unittest.TestCase):
    def setUp(self):
        name = 'research.taskcard11.report'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Task11 report implementation absent')
        self.report = importlib.import_module(name)

    def test_only_complete_original_formal_requests_are_accepted(self):
        requests, records, summaries = fixture()
        for bad in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaises(ValueError):
                self.report.assemble(bad, summaries, requests)
        wrong = copy.deepcopy(records)
        wrong[0]['request']['seed'] += 1
        with self.assertRaises(ValueError):
            self.report.assemble(wrong, summaries, requests)

    def test_same_case_pairs_and_all_group_denominators_are_retained(self):
        requests, records, summaries = fixture()
        result = self.report.assemble(records, summaries, requests)
        tables = result['tables']
        self.assertEqual(len(tables['selected_metrics']), 192)
        self.assertEqual(len(tables['paired_changes']), 96)
        self.assertEqual(len(tables['cell_metrics']), 48)
        n100 = next(r for r in tables['group_metrics'] if r['group_kind'] == 'n' and
                    r['group_value'] == '100' and r['arm'] == 'A0')
        self.assertEqual(n100['datasets'], 48)
        self.assertEqual(n100['pmse_mean'], 2.)
        self.assertEqual(sum(r['count'] for r in tables['shape_q_distribution']
                             if r['group_kind'] == 'all'), 192)
        self.assertEqual(result['gates'], analyze.evaluate_gates(records, requests))
        self.assertEqual(result['summary']['overall'], analyze.paired_summary(records))

    def test_every_unfavorable_case_remains_in_pair_table_and_summary(self):
        requests, records, summaries = fixture()
        records[0]['metrics']['A1']['pmse'] = 5.
        records[1]['metrics']['A1']['pmse'] = 1.
        records[2]['metrics']['A1']['ree'] = .5
        result = self.report.assemble(records, summaries, requests)
        rows = result['tables']['paired_changes']
        self.assertEqual(rows[0]['prediction_change'], 'worsened')
        self.assertEqual(rows[1]['prediction_change'], 'improved')
        self.assertEqual(rows[0]['pmse_delta'], 3.)
        self.assertIn(0, result['summary']['adverse_task_ids']['pmse_worsened'])
        self.assertIn(2, result['summary']['adverse_task_ids']['ree_worsened'])
        self.assertEqual(result['summary']['overall']['top1_fraction_gross_improvement'], 1.)

    def test_change_classification_separates_q_shape_support_and_coefficients(self):
        original = {'q': 8, 'sizes': [2, 2, 2], 'support': [[0, 1]] * 3, 'B_sha256': 'a'}
        examples = [(dict(original, q=4, sizes=[1, 2, 2], B_sha256='b'), 'q_changed'),
                    (dict(original, sizes=[1, 2, 4], B_sha256='b'), 'same_q_shape_changed'),
                    (dict(original, support=[[2, 3]] * 3, B_sha256='b'), 'same_q_shape_support_changed'),
                    (dict(original, B_sha256='b'), 'same_support_coefficients_changed'),
                    (dict(original), 'identical_model')]
        for other, label in examples:
            with self.subTest(label=label):
                self.assertEqual(self.report.change_category(original, other), label)

    def test_changed_candidate_keeps_both_penalty_and_band_decompositions(self):
        requests, records, summaries = fixture()
        second = dict(summaries[0]['full_trace']['candidates'][0], sizes=[1, 2, 2], q=4,
                      support=[[0], [0, 1], [0, 1]], B_sha256='changed', rss=130.)
        summaries[0]['full_trace']['candidates'].append(second)
        summaries[0]['arms']['A1'] = dict(second, full_candidate_index=1, candidate_source='fixed_grid')
        choice = summaries[0]['choices']['A1']
        choice['selected_index'] = choice['minimum_index'] = 1
        choice['minimum_q'] = 4
        choice['tolerance'] = 1.
        choice['candidate_terms'].append(dict(choice['candidate_terms'][0]))
        choice['candidate_scores'].append(21.)
        metric = records[0]['metrics']['A1']
        metric.update({k: second[k] for k in ('q', 'sizes', 'rss', 'support', 'B_sha256')})
        result = self.report.assemble(records, summaries, requests)
        changed = result['tables']['selection_changes']
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]['A0_q'], 24)
        self.assertEqual(changed[0]['A1_q'], 4)
        self.assertEqual(changed[0]['A1_fs_penalty'], 1.)
        self.assertEqual(changed[0]['A1_minimum_q'], 4)
        self.assertEqual(changed[0]['A1_tolerance'], 1.)

    def test_case_metric_and_selected_candidate_identity_mismatch_is_rejected(self):
        requests, records, summaries = fixture()
        records[0]['metrics']['A1']['B_sha256'] = 'unrelated'
        with self.assertRaises(ValueError):
            self.report.assemble(records, summaries, requests)

    def test_identical_predictions_are_not_listed_as_extreme_improvement_or_harm(self):
        requests, records, summaries = fixture()
        result = self.report.assemble(records, summaries, requests)
        self.assertEqual(result['summary']['largest_improvement_tasks'], [])
        self.assertEqual(result['summary']['largest_deterioration_tasks'], [])


if __name__ == '__main__':
    unittest.main()
