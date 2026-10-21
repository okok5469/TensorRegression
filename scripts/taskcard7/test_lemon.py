"""Strict identity and passive-observation checks for the LEMON attribution."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import unittest


def implementation():
    path = Path(__file__).with_name('lemon.py')
    assert path.is_file(), 'LEMON attribution implementation is missing'
    spec = importlib.util.spec_from_file_location('taskcard7_lemon', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LemonTests(unittest.TestCase):
    def test_preprocessing_identity_rejects_even_single_bit_drift(self):
        check = implementation().validate_preprocessing
        saved = {'x': np.array([1., 2.]), 'y': np.asarray(.5)}
        check(saved, {k: v.copy() for k, v in saved.items()})
        changed = {**saved, 'x': np.array([1., np.nextafter(2., 3.)])}
        with self.assertRaisesRegex(ValueError, 'preprocessing'):
            check(saved, changed)


    def test_passive_trace_preserves_real_solver_and_records_actual_candidates(self):
        module = implementation()
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / 'doc/algorithm_versions/v5.1-bcd-fix/src'))
        from tensor_abess.tensor.abess import tensor_abess
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 3, 3, 3))
        y = X[:, 1, 1, 1] + rng.normal(size=40)
        options = dict(c=3., criterion='bic', n_lambdas=3, max_iter=20,
                       max_cycles=2, m_max=2, k_max=1)
        plain = tensor_abess(X, y, **options)
        observed, trace, coefficients = module.observe_fit(lambda: tensor_abess(X, y, **options))
        assert np.array_equal(plain.B, observed.B)
        assert plain.rss == observed.rss and plain.sic == observed.sic
        assert len(trace['candidates']) == len(trace['ordered_size_grid']) + 1
        assert len(coefficients) == len(trace['candidates'])
        assert trace['final']['B_sha256'] == module.array_sha(observed.B)
        assert trace['adaptive_terminal'] == trace['candidates'][0]
        assert all(candidate['B_sha256'] == module.array_sha(B)
                   for candidate, B in zip(trace['candidates'], coefficients))
        assert trace['one_se_tolerance'] >= 0
        assert trace['initial_support'] == [s.tolist() for s in observed.init_supports]
        assert trace['caps'] == list(observed.s_max)
        assert sys.getprofile() is None


    def test_repeat_oof_aggregation_uses_all_subject_predictions(self):
        aggregate = implementation().aggregate_oof
        rows = [dict(version='v', repeat=0, subject_id=str(i), y_true=y, y_pred=p)
                for i, (y, p) in enumerate([(0., 0.), (1., 1.), (2., 5.)])]
        result = aggregate(rows)
        assert result[0]['rmse'] == np.sqrt(3.)
        assert result[0]['r2'] == -3.5
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(rows + [rows[0]])
