"""Training-only regression tests for the single task 11 FS rule."""
from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from scripts.taskcard8.selection import select, terms


def candidate(sizes, rss=100.0):
    return {'sizes': list(sizes), 'support': [list(range(s)) for s in sizes],
            'q': int(np.prod(sizes)), 'rss': rss}


class RescoreTests(unittest.TestCase):
    def setUp(self):
        name = 'research.taskcard11.rescore'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Task 11 selector must be implemented')
        self.select = importlib.import_module(name).select_candidates

    def test_a0_is_bitwise_equal_to_frozen_order_and_scores(self):
        pool = [candidate((4, 3, 5), 47.134), candidate((2, 3, 4), 83.79),
                candidate((1, 2, 3), 125.25)]
        expected = select(pool, 100, (32, 20, 25), 1.5, 1.0)
        actual = self.select(pool, 100, (32, 20, 25))
        self.assertEqual(expected, {key: actual[key] for key in expected})

    def test_fs_recomputes_its_own_argmin_q_and_band(self):
        pool = [candidate((4, 4, 4)), candidate((1, 1, 1)), candidate((1, 2, 2))]
        for item, target in zip(pool, (0.0, 5.0, 2.0)):
            penalty = terms(item, 100, (8, 8, 8), 1.5)['score']
            item['rss'] = float(100 * np.exp((target - penalty) / 100))
        a0, a1 = [self.select(pool, 100, (8, 8, 8), fs) for fs in (False, True)]
        self.assertEqual(a0['minimum_q'], 64)
        self.assertEqual(a1['minimum_q'], 4)
        self.assertEqual(a1['minimum_index'], 2)
        self.assertEqual(a1['tolerance'], float(np.sqrt(4 * np.log(100))))
        self.assertEqual(a0['eligible_indices'], [0, 1, 2])
        self.assertEqual(a1['eligible_indices'], [1, 2])

    def test_same_q_gets_identical_extra_penalty(self):
        pool = [candidate((1, 2, 4), 51.2), candidate((2, 2, 2), 67.3)]
        answer = self.select(pool, 100, (8, 8, 8), True)
        expected = float(np.log(max(np.log(100), np.e)) * 8 ** 2 / (2 * 100))
        self.assertEqual([t['fs_penalty'] for t in answer['candidate_terms']], [expected, expected])
        self.assertEqual(answer['candidate_scores'], [t['baseline_score'] + expected for t in answer['candidate_terms']])

    def test_equal_scores_preserve_original_candidate_order(self):
        first = candidate((1, 2, 2))
        second = copy.deepcopy(first)
        second['support'] = [[1], [2, 3], [3, 4]]
        for pool in ([first, second], [second, first]):
            for fs in (False, True):
                actual = self.select(pool, 100, (8, 8, 8), fs)
                self.assertEqual(actual['minimum_index'], 0)
                self.assertEqual(actual['selected_index'], 0)
                self.assertEqual(actual['eligible_indices'], [0, 1])

    def test_no_mutation_and_zero_rss_uses_original_floor(self):
        pool = [candidate((1, 1, 1), 0.0)]
        original = copy.deepcopy(pool)
        actual = self.select(pool, 100, (8, 8, 8), True)
        self.assertEqual(pool, original)
        self.assertTrue(np.isfinite(actual['selected_score']))
        self.assertEqual(actual['candidate_terms'][0]['fit_term'],
                         float(100 * np.log(np.finfo(float).tiny)))

    def test_invalid_and_nonfinite_candidates_are_rejected(self):
        invalid = []
        for rss in (-1.0, float('nan'), float('inf')):
            invalid.append(candidate((1, 2, 2), rss))
        for update in ({'q': 5}, {'sizes': [0, 2, 2]}, {'sizes': [1.5, 2, 2]},
                       {'support': [[0], [0, 0], [0, 1]]},
                       {'support': [[8], [0, 1], [0, 1]]},
                       {'support': [[0], [1, 0], [0, 1]]},
                       {'B': np.full((8, 8, 8), np.nan)}):
            invalid.append(dict(candidate((1, 2, 2)), **update))
        for item in invalid:
            with self.subTest(item=str(item)[:90]), self.assertRaises(ValueError):
                self.select([item], 100, (8, 8, 8), True)

    def test_empty_pool_and_invalid_problem_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.select([], 100, (8, 8, 8))
        for n, p in [(0, (8, 8, 8)), (100.5, (8, 8, 8)), (100, (8, 8)),
                     (100, (8, 0, 8)), (100, (8, 8, float('inf')))]:
            with self.subTest(n=n, p=p), self.assertRaises(ValueError):
                self.select([candidate((1, 2, 2))], n, p)


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        name = 'research.taskcard11.history'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Historical evidence audit must be implemented')
        self.history = importlib.import_module(name)

    def test_missing_candidate_predictions_remain_na(self):
        record = {'request': {'cohort': 'development', 'task_id': 7},
                  'metrics': [{'pmse': 2.0, 'signal_mse': 1.0}]}
        pool = [{'B_sha256': 'old'}, {'B_sha256': 'new'}]
        saved = [{'selected_index': 0}]
        unrelated = {('confirmation', 7, 'new'): {'pmse': 0.1, 'signal_mse': 0.01}}
        result = self.history.saved_prediction(record, pool, saved, unrelated, 1)
        self.assertEqual(result, {'pmse': None, 'signal_mse': None, 'pmse_source': 'NOT_AVAILABLE'})

    def test_predictions_require_case_and_coefficient_identity(self):
        record = {'request': {'cohort': 'development', 'task_id': 7},
                  'metrics': [{'pmse': 2.0, 'signal_mse': 1.0}]}
        pool = [{'B_sha256': 'old'}, {'B_sha256': 'same-new-B'}]
        saved = [{'selected_index': 0}]
        replay = {('development', 7, 'same-new-B'): {'pmse': 1.1, 'signal_mse': 0.2}}
        self.assertEqual(self.history.saved_prediction(record, pool, saved, replay, 1)['pmse'], 1.1)
        self.assertEqual(self.history.saved_prediction(record, pool, saved, {}, 0)['pmse'], 2.0)

    def test_original_saved_case_restores_native_a0(self):
        repo = Path(__file__).resolve().parents[2]
        folder = repo / 'numerical_simulation/outputs_selection_research_task8/development/task_000'
        result = self.history.audit_case(folder)
        self.assertTrue(result['audit']['A0_native_exact'])
        self.assertEqual(result['audit']['candidate_count'], 65)
        self.assertEqual(result['choices'][0]['selected_index'], 6)


if __name__ == '__main__':
    unittest.main()
