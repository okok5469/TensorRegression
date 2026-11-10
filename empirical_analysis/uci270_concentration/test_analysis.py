import unittest
import numpy as np
from empirical_analysis.uci270_concentration import analysis

class AnalysisTests(unittest.TestCase):
    def test_constant_r2_and_bias(self):
        m=analysis.metrics([2,2],[1,3]);self.assertIsNone(m['r2']);self.assertEqual(m['rmse'],1);self.assertEqual(m['bias'],0)
    def test_lasso_envelope_is_not_coefficient_count(self):
        beta=np.zeros((16,3,2));beta[0,0,0]=1;beta[1,1,1]=2
        a=dict(model_id='M3',coef_standardized=beta.ravel(),diagnostics={},parameters={})
        c=analysis.complexity(a)
        self.assertEqual(c['effective_nnz'],2);self.assertEqual(c['envelope_capacity'],8)
        self.assertEqual(c['sensors'],[1,2]);self.assertEqual(c['stages'],[0,1])
    def test_full_input_steady_dynamic(self):
        b=np.zeros(128);b[0]=1;b[10]=2
        a=dict(model_id='M7',coef_standardized=b,diagnostics={},parameters={})
        c=analysis.complexity(a)
        self.assertEqual(c['steady_nnz'],1);self.assertEqual(c['dynamic_nnz'],1);self.assertIsNone(c['stages'])

if __name__=='__main__':unittest.main()
