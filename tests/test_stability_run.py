"""Independent runner checks: training-selected shapes and immutable artifacts."""
import importlib
import unittest
import numpy as np


class StabilityRunnerTests(unittest.TestCase):
    def runner(self):
        name = 'numerical_simulation.candidate_search_study.stability_run'
        spec = importlib.util.find_spec(name)
        self.assertIsNotNone(spec, 'Independent stability runner is not implemented')
        return importlib.import_module(name)

    def job(self):
        candidates = [{'sizes': [2, 1, 1], 'supports': [[0, 1], [0], [0]]}]
        return {'status': 'complete', 'groups': {g: {'candidates': candidates} for g in ('A0', 'A3')},
                'rows': [{'group': g, 'rule': 'msic-alpha1.5-scale1',
                          'selection': {'selected_index': 0}} for g in ('A0', 'A3')],
                'config': {'sparsity': [4, 3, 5], 'dims': [5, 5, 5], 'n_train': 20}}

    def test_sizes_come_from_training_default_not_truth(self):
        self.assertEqual(self.runner().selected_sizes(self.job()), {'A0': (2, 1, 1), 'A3': (2, 1, 1)})

    def test_incomplete_pair_is_rejected(self):
        job = self.job()
        job['rows'] = job['rows'][:1]
        with self.assertRaisesRegex(ValueError, 'A3'):
            self.runner().selected_sizes(job)

    def test_resume_rejects_changed_candidate_identity(self):
        report = {'status': 'complete', 'identity': {'candidate_json_sha256': 'old'},
                  'records': [], 'array_fingerprints': {}}
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.runner().validate_artifact(report, {}, {'candidate_json_sha256': 'new'})

    def test_resume_rejects_changed_coefficient_cache(self):
        report = {'status': 'complete', 'identity': {'seed': 7}, 'records': [],
                  'array_fingerprints': {'B': 'incorrect'}}
        with self.assertRaisesRegex(ValueError, 'array'):
            self.runner().validate_artifact(report, {'B': np.zeros((2, 2, 2))}, {'seed': 7})

    def test_actual_multistart_reuses_shape_and_excludes_oracle(self):
        from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
        from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test
        from tensor_abess.tensor.bcd import tensor_bcd
        config = TensorSimulationConfig(n_train=24, n_test=9, dims=(3, 3, 3), sparsity=(2, 1, 1),
                                        max_cycles=2, k_max=1, test_batch_size=3)
        data = generate_simulation_dataset(config, seed=91)
        fixed = tensor_bcd(data.X_train, data.y_train, 1, 1, 1, max_cycles=2, k_max=1)
        candidate = {'sizes': [1, 1, 1], 'supports': [getattr(fixed, f'S{m}').tolist() for m in (1, 2, 3)],
                     'B_fingerprint': array_fingerprint(fixed.B), 'rss': 48*fixed.loss, 'loss': fixed.loss}
        from dataclasses import asdict
        job = self.job() | {'config': asdict(config), 'seed': 91, 'test_seed': 1000091,
                            'groups': {g: {'candidates': [candidate]} for g in ('A0', 'A3')}}
        _, test_hash = evaluate_shared_test(data, [fixed.B], job['test_seed'])
        job['fingerprints'] = {k: array_fingerprint(getattr(data, k)) for k in ('X_train', 'y_train', 'B_true')}
        job['fingerprints']['independent_test'] = test_hash
        arrays = {f'{g}_candidate_0_B': fixed.B for g in ('A0', 'A3')}
        arrays.update({f'{g}_candidate_0_S{m}': getattr(fixed, f'S{m}') for g in ('A0', 'A3') for m in (1, 2, 3)})
        report, cache = self.runner().evaluate_job(job, arrays)
        self.assertEqual(len(report['records']), 6)
        self.assertEqual(report['groups']['A3']['reused_from'], 'A0')
        self.assertEqual(report['groups']['A0']['start_records'], report['groups']['A3']['start_records'])
        for group in report['groups'].values():
            chosen = group['selected_record']
            self.assertNotEqual(chosen, report['oracle_record'])
            self.assertEqual(report['records'][chosen]['rss'], min(report['records'][i]['rss'] for i in group['start_records']))
        self.assertEqual(report['fingerprints']['independent_test'], test_hash)
        self.assertIn('record_5_B', cache)


if __name__ == '__main__':
    unittest.main()
