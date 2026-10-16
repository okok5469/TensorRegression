"""Fail-closed tests for the isolated, 250-task v5.1 diagnostic runner."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class AttributionTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('numerical_simulation.version_attribution.run')
        except ModuleNotFoundError:
            self.fail('The isolated attribution runner is not implemented')

    def test_only_five_prespecified_scenarios_are_selected(self):
        api = self.api()
        plan = api.build_plan()
        self.assertEqual(len(plan['tasks']), 250)
        self.assertEqual({t['scenario_id'] for t in plan['tasks']}, {0, 1, 5, 7, 11})
        self.assertEqual(plan['gate_task_ids'], [0, 50, 250, 350, 550])
        self.assertEqual(plan['fit_version'], 'v5.1-bcd-fix')
        self.assertEqual(len({t['input_key'] for t in plan['tasks']}), 250)

    def test_full_three_version_manifests_are_verified(self):
        manifests = self.api().source_manifests()
        self.assertEqual(set(manifests), {'v5-original', 'v5.1-bcd-fix', 'v5.2-search-final'})
        self.assertTrue(all(len(x) == 32 for x in manifests.values()))
        self.assertNotEqual(manifests['v5-original'], manifests['v5.1-bcd-fix'])

    def test_a_different_input_hash_is_rejected_before_fit(self):
        with self.assertRaisesRegex(ValueError, 'input fingerprints differ'):
            self.api().require_inputs({'X_train': 'original'}, {'X_train': 'different'})

    def test_each_of_the_four_input_hashes_is_required(self):
        expected = dict.fromkeys(('X_train', 'y_train', 'B_true', 'train_noise'), 'same')
        self.api().require_inputs(expected, expected.copy())
        for key in expected:
            changed = expected.copy()
            changed.pop(key)
            with self.assertRaisesRegex(ValueError, 'input fingerprints differ'):
                self.api().require_inputs(expected, changed)

    def test_immutable_json_refuses_overwrite(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            api.write_json(path, {'status': 'first'})
            with self.assertRaises(FileExistsError):
                api.write_json(path, {'status': 'second'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'first'})

    def test_sealed_plan_detects_any_mutation(self):
        api = self.api()
        plan = api.seal({'tasks': [1], 'fit_version': 'v5.1-bcd-fix'})
        api.check_seal(plan)
        plan['tasks'].append(2)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            api.check_seal(plan)

    def test_bulk_execution_requires_each_scenario_gate(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            plan = {'gate_task_ids': [0, 50, 250, 350, 550]}
            with self.assertRaisesRegex(ValueError, 'gate'):
                api.require_gates(Path(directory), plan)

    def test_worker_rejects_already_imported_production_core(self):
        api = self.api()
        import tensor_abess
        with self.assertRaisesRegex(RuntimeError, 'already imported'):
            api.configure_v51()

    def test_candidate_grid_and_order_match_at_n100_caps(self):
        api = self.api()
        comparison = api.compare_grids((4, 4, 4), 100)
        self.assertTrue(comparison['same_ordered_grid'])
        self.assertEqual(comparison['v51_count'], 64)
        self.assertEqual(comparison['v52_count'], 64)

    def test_original_numpy_initialization_trace_is_serializable_without_mutation(self):
        import numpy as np
        trace = {'pipeline_init_supports': (np.array([1, 2]), np.array([0]), np.array([2])),
                 'pipeline_glasso_init_supports': (np.array([1]), np.array([0]), np.array([2])),
                 's_max': [4, 4, 4]}
        converted = self.api().serializable_trace(trace)
        json.dumps(converted, allow_nan=False)
        self.assertEqual(converted['pipeline_init_supports'], [[1, 2], [0], [2]])
        self.assertIsInstance(trace['pipeline_init_supports'][0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
