"""Independent numerical and selection-rule acceptance tests."""
import unittest
import numpy as np


class DiagnosticTests(unittest.TestCase):
    def test_one_se_prefers_smallest_eligible_not_smallest_overall(self):
        from .diagnostic import selection_audit
        rows=[{'q':80,'score':-310.},{'q':1,'score':-286.},{'q':8,'score':-295.}]
        d=selection_audit(rows,116,1.)
        self.assertAlmostEqual(d['tolerance'],np.sqrt(80*np.log(116)))
        self.assertEqual(d['minimum_index'],0)
        self.assertEqual(d['selected_index'],2)
        self.assertEqual(d['eligible_indices'],[0,2])

    def test_ols_rank_rss_and_solver_consistency(self):
        from .diagnostic import ols_audit
        rng=np.random.default_rng(22);Z=rng.normal(size=(30,4));y=Z@np.array([1.,2.,3.,4.])
        d=ols_audit(Z,y,np.ones((30,1)))
        self.assertEqual(d['rank'],4)
        self.assertEqual(d['joint_rank'],5)
        self.assertEqual(d['residual_df_joint'],25)
        self.assertLess(d['rss'],1e-24)
        self.assertLess(d['solver_prediction_max_difference'],1e-12)

    def test_rank_deficiency_is_visible_not_silently_fixed(self):
        from .diagnostic import ols_audit
        Z=np.column_stack((np.arange(10.),np.arange(10.)))
        d=ols_audit(Z,np.arange(10.),np.ones((10,1)))
        self.assertEqual(d['rank'],1)
        self.assertFalse(d['full_column_rank'])

    def test_trace_preserves_core_result_and_restores_functions(self):
        from .diagnostic import traced_abess
        import importlib
        module=importlib.import_module('tensor_abess.tensor.bcd')
        entry=importlib.import_module('tensor_abess.tensor.abess')
        rng=np.random.default_rng(23);X=rng.normal(size=(24,3,2,1));y=rng.normal(size=24)
        options={'criterion':'bic','max_cycles':1,'m_max':2,'n_lambdas':2,'max_iter':10}
        original=module.tensor_bcd
        expected=entry.tensor_abess(X,y,**options)
        actual,trace=traced_abess(X,y,options)
        self.assertIs(module.tensor_bcd,original)
        np.testing.assert_allclose(actual.B,expected.B,atol=0,rtol=0)
        self.assertTrue(trace['candidates'])


if __name__=='__main__':unittest.main()
