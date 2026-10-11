"""Persistence and independent OOF completeness acceptance."""
import unittest
import tempfile
from pathlib import Path


class RunTests(unittest.TestCase):
    def test_prediction_completeness_rejects_missing_and_duplicate(self):
        from .run import validate_predictions
        rows = [{'subject_id': i, 'repeat': 0, 'model': m, 'y_true': 1., 'y_pred': 1.}
                for i in ('a', 'b') for m in ('Covariates Only', 'Ridge')]
        validate_predictions(rows, ['a', 'b'], ['Covariates Only', 'Ridge'], [0])
        with self.assertRaises(ValueError):
            validate_predictions(rows[:-1], ['a', 'b'], ['Covariates Only', 'Ridge'], [0])
        with self.assertRaises(ValueError):
            validate_predictions(rows + rows[:1], ['a', 'b'], ['Covariates Only', 'Ridge'], [0])

    def test_config_fingerprint_is_sensitive_to_seed_and_source(self):
        from .run import experiment_id
        a = experiment_id({'seed': 10}, {'X': 'a'})
        self.assertEqual(a, experiment_id({'seed': 10}, {'X': 'a'}))
        self.assertNotEqual(a, experiment_id({'seed': 11}, {'X': 'a'}))
        self.assertNotEqual(a, experiment_id({'seed': 10}, {'X': 'b'}))

    def test_atomic_json_disallows_nonfinite_statistics(self):
        from .run import save_json
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data.json'
            with self.assertRaises(ValueError):
                save_json(p, {'x': float('nan')})
            self.assertFalse(p.exists())


if __name__ == '__main__':
    unittest.main()
