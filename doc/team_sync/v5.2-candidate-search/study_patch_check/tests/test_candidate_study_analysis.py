"""Guard balanced comparison and truthful shared-cache cost accounting."""
import importlib.util
from pathlib import Path
import unittest

class AnalysisTests(unittest.TestCase):
    def module(self):
        p=Path(__file__).resolve().parents[1]/'numerical_simulation/candidate_search_study/analyse.py'
        self.assertTrue(p.is_file(),'Cohort analysis is missing')
        spec=importlib.util.spec_from_file_location('candidate_analysis',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def fixture(self):
        row={'group':'A0','q_hat':60,'exact_recovery':0,'mode_f1_macro':.5,'element_f1':.4,'pmse':2.,'r2_test':.2,'rss_train':90.,'runtime_sec':1.}
        group={'s_max':[5,5,5],'true_size_in_pool':False,'true_support_in_pool':False,'candidate_count':2,'cache_hits':1,'new_fixed_fits':0,'search_metadata':{'legal_shapes':[[1,1,1],[4,3,5]],'size_grid':[[1,1,1]],'excluded_legal_shapes':[[4,3,5]]},'candidates':[{'kind':'adaptive-stage','fit_seconds':2,'cache_hit':False},{'kind':'global-fixed','fit_seconds':9,'cache_hit':True}]}
        return {'config':{'n_train':200,'snr':2,'sparsity':[4,3,5]},'rows':[row],'groups':{'A0':group}}
    def test_unbalanced_groups_are_rejected(self):
        m=self.module()
        with self.assertRaises(ValueError):m.build_overview([self.fixture()],['A0','A1'])
    def test_q60_is_not_mistaken_for_true_shape(self):
        m=self.module();r=m.build_overview([self.fixture()],['A0'])[0]
        self.assertEqual(r['q_over_mean'],0);self.assertEqual(r['q_under_mean'],0)
        self.assertEqual(r['true_size_in_pool_mean'],0);self.assertEqual(r['exact_recovery_mean'],0)
    def test_shared_cache_walltime_does_not_claim_standalone_cost(self):
        m=self.module();r=m.build_overview([self.fixture()],['A0'])[0]
        self.assertEqual(r['runtime_sec_mean'],1)
        self.assertEqual(r['cached_fixed_fit_cost_sec_mean'],9)
        self.assertEqual(r['standalone_cost_estimate_sec_mean'],10)
if __name__=='__main__':unittest.main()
