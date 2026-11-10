"""Reject invalid cached model claims before evaluation or aggregation."""
import importlib
import importlib.util
import copy
import unittest

import numpy as np


class CacheAcceptanceTests(unittest.TestCase):
    def test_saved_cv_selection_rejects_wrong_lambda_and_nonfinite_scores(self):
        from research.taskcard18.baselines import select_lambda_cv
        module = importlib.import_module('research.taskcard18.accept')
        selected = select_lambda_cv(np.zeros((10, 2)), np.ones(10), lambdas=np.array([1., .5]))
        metadata = {'method_id': 'vector_lasso_cv_isolated_v2', 'valid': True,
                    'diagnostics': selected['diagnostics'], 'selected_lambdas': [
                        {k:v for k,v in selected.items() if k not in ('beta', 'diagnostics')}]}
        self.assertTrue(module.validate_metadata(metadata, 'lasso'))
        bad = copy.deepcopy(metadata); bad['selected_lambdas'][0]['selected_lambda'] = 999.
        with self.assertRaises(ValueError): module.validate_metadata(bad, 'lasso')
        bad = copy.deepcopy(metadata); bad['selected_lambdas'][0]['cv_scores'][0] = float('nan')
        with self.assertRaises(ValueError): module.validate_metadata(bad, 'lasso')

    def test_cache_certificate_cannot_be_replaced_by_a_valid_flag(self):
        assert importlib.util.find_spec('research.taskcard18.accept') is not None, 'Missing independent acceptance entry'
        module = importlib.import_module('research.taskcard18.accept')
        record = {'method_id': 'vector_lasso_cv_isolated_v2', 'valid': True,
                  'diagnostics': [{'converged': True, 'lambda': 1., 'certificate': 'kkt_and_primal_dual',
                                   'kkt_ratio': .01, 'gap_normalized': 0., 'dual_feasible': True, 'negative_gap_error': False}]}
        with self.assertRaises(ValueError):
            module.validate_metadata(record, 'lasso')

    def test_model_arrays_must_be_finite_and_match_declared_axes(self):
        assert importlib.util.find_spec('research.taskcard18.accept') is not None, 'Missing independent acceptance entry'
        module = importlib.import_module('research.taskcard18.accept')
        with self.assertRaises(ValueError):
            module.validate_arrays(np.full((2, 2, 2), np.nan), [np.array([0])] * 3, (2, 2, 2))
        B = np.zeros((2, 2, 2)); B[1, 1, 1] = 1
        with self.assertRaises(ValueError):
            module.validate_arrays(B, [np.array([0])] * 3, (2, 2, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
