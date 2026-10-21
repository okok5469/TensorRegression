"""Train-only diagnosis contracts; synthetic data never use outer test subjects."""
import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import numpy as np


class TrainOnlyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('train_only.py')
        self.assertTrue(path.is_file(), 'Missing train-only diagnosis implementation')
        spec = importlib.util.spec_from_file_location('train_only', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_illegal_q_is_rejected_before_solver(self):
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.module.validate_sizes((4, 4, 5), 77, (61, 16, 5))
        self.assertEqual(self.module.validate_sizes((3, 3, 5), 77, (61, 16, 5)), 45)

    def test_bind_checks_complete_snapshot_without_importing_unused_baselines(self):
        code = ('from scripts.taskcard7.train_only import bind_core; from pathlib import Path; '
                'import sys; b=bind_core(Path.cwd()); '
                'assert len(b["source32"]) == 32; '
                'assert "tensor_abess.tensor.bcd" in b["actual_imports"]; '
                'assert "tensor_abess.baselines.lasso" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_representative_folds_are_frozen_by_index(self):
        C = np.column_stack((np.ones(145), np.arange(145) % 2))
        config = {'outer_folds': 5, 'outer_seeds': [20260908, 20261009, 20261010],
                  'inner_seed': 20261009, 'inner_folds': 3}
        plan = self.module.make_plan(C, np.arange(145).astype(str), config)
        self.assertEqual([(p['repeat'], p['outer_fold']) for p in plan], [(0, 1), (1, 1), (2, 1)])
        for item in plan:
            allowed = set(item['outer_train_ids'])
            self.assertEqual(len(allowed), 116)
            valid = []
            for inner in item['inner']:
                a, b = set(inner['train_ids']), set(inner['validation_ids'])
                self.assertFalse(a & b)
                self.assertEqual(a | b, allowed)
                self.assertIn(len(a), (77, 78))
                valid += list(b)
            self.assertEqual(len(valid), len(set(valid)))
            self.assertEqual(set(valid), allowed)

    def test_validation_changes_cannot_affect_preprocessing_or_support(self):
        rng = np.random.default_rng(72)
        X = rng.normal(size=(30, 4, 3, 2))
        C = np.ones((30, 1))
        y = X[:, 0, 0, 0] + rng.normal(scale=.2, size=30)
        options = {'max_cycles': 2, 'm_max': 3, 'k_max': 1,
                   'msic_refine_alpha': 1.5, 'msic_refine_gamma': 1.0}
        first = self.module.fit_inner(X, y, C, np.arange(20), np.arange(20, 30),
                                      options, [('Fixed small', (1, 1, 1))])
        X[20:] *= 100
        y[20:] += 500
        second = self.module.fit_inner(X, y, C, np.arange(20), np.arange(20, 30),
                                       options, [('Fixed small', (1, 1, 1))])
        self.assertEqual(first['preprocessing_hash'], second['preprocessing_hash'])
        self.assertEqual(first['fits'][0]['support'], second['fits'][0]['support'])
        np.testing.assert_array_equal(first['fits'][0]['B'], second['fits'][0]['B'])
        self.assertNotEqual(first['fits'][0]['validation_rmse'], second['fits'][0]['validation_rmse'])
        self.assertEqual(first['fits'][0]['q'], 1)
        self.assertEqual(first['fits'][0]['rank'], 1)


if __name__ == '__main__':
    unittest.main()
