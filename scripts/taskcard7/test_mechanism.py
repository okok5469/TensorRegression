"""Mechanism replay selection and exact candidate comparison contracts."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


class MechanismTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('mechanism.py')
        self.assertTrue(path.is_file(), 'Missing mechanism replay implementation')
        spec = importlib.util.spec_from_file_location('mechanism', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_representatives_use_only_training_q_and_first_control(self):
        rows = [{'model': 'Tensor-ABESS', 'repeat': str(r), 'outer_fold': str(f),
                 'q': str(q), 'test_rmse': str(e)}
                for r, f, q, e in [(0,1,4,999),(0,2,80,1),(0,3,3,0),(1,1,6,1)]]
        new = [dict(row) for row in rows]
        new[-1]['q'] = '80'
        expected = [(0,1),(0,2),(1,1)]
        self.assertEqual(self.m.select_representatives(rows, new), expected)
        for row in rows + new:
            row['test_rmse'] = '-99'
        self.assertEqual(self.m.select_representatives(rows, new), expected)

    def test_same_shape_score_change_has_zero_penalty_change(self):
        a={'rss': 10., 'sizes': [2,2,3], 'msic': 116*np.log(10/116)+30}
        b={'rss': 5., 'sizes': [2,2,3], 'msic': 116*np.log(5/116)+30}
        d=self.m.decompose(a,b,116)
        self.assertAlmostEqual(d['penalty_delta'],0.)
        self.assertAlmostEqual(d['msic_delta'],116*np.log(.5))
        self.assertLess(abs(d['decomposition_error']),1e-10)

    def test_exact_candidate_comparison_rejects_one_coefficient_change(self):
        trace={'candidates': [{'q':1,'sizes':[1,1,1],'support':[[0],[0],[0]],
                              'rss':1.,'msic':2.,'B_sha256':'a'}],
               'initial_support':[[0],[0],[0]],'caps':[1,1,1],
               'ordered_size_grid':[[1,1,1]],'adaptive_terminal':{},
               'one_se_tolerance':1.,'minimum_msic':2.,'minimum_msic_q':1,
               'eligible_candidate_indices':[0],'final':{},'fixed_searches':[]}
        a=np.zeros((1,2,2,2)); b=a.copy()
        self.assertTrue(self.m.compare_candidates(trace,trace,a,b)['all_equal'])
        b[0,0,0,0]=1
        self.assertFalse(self.m.compare_candidates(trace,trace,a,b)['all_equal'])


if __name__=='__main__':
    unittest.main()
