import importlib.util
import unittest
from pathlib import Path

import numpy as np


def analysis():
    path=Path('doc/mushroom_empirical/analyze_results.py')
    assert path.exists(),'Result analysis module missing'
    spec=importlib.util.spec_from_file_location('mushroom_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class AnalysisTests(unittest.TestCase):
    def test_single_pooled_block_statistic_has_no_repeat_sd(self):
        a=analysis()
        rows=[dict(protocol='block',repeat=1,model='M1',n=250,rmse=2.,mae=1.,r2=.5)]
        result=a.summarize(rows)[0]
        self.assertIsNone(result['rmse_sd'])
        self.assertIsNone(result['mae_sd'])
        self.assertIsNone(result['r2_sd'])
        repeated=[dict(row,protocol='main',repeat=i,rmse=float(i)) for i,row in enumerate(rows*3,1)]
        self.assertEqual(a.summarize(repeated)[0]['rmse_sd'],1.)
    def test_figure_qa_never_calls_missing_exports_pass_or_ignores_warning(self):
        path=Path('doc/mushroom_empirical/figure_qa.py')
        self.assertTrue(path.exists(),'Figure QA module missing')
        spec=importlib.util.spec_from_file_location('mushroom_qa',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.qa_status([0,0,2],0),'NOT AUDITABLE')
        self.assertEqual(module.qa_status([0,0,0],1),'REVIEW REQUIRED')
        self.assertEqual(module.qa_status([0,1,0],0),'FIX BEFORE DELIVERY')

    def test_bootstrap_retains_all_cluster_records_and_repeats_with_record_weights(self):
        a=analysis()
        # Duplicate cluster A has two records, each with three repeats.
        squared=np.array([[1.,4.,9.],[16.,25.,36.],[49.,64.,81.]])
        sampled=a.cluster_sample_indices(np.array(['A','A','B']),np.array([0,0]))
        np.testing.assert_array_equal(sampled,[0,1,0,1])
        self.assertEqual(a.record_weighted_mse(squared,sampled),float(squared[:2].mean()))
        self.assertNotEqual(a.record_weighted_mse(squared,[0,1,2]),float(np.mean([squared[:2].mean(),squared[2].mean()])))

    def test_repeat_metric_uses_pooled_oof_not_mean_fold_r2(self):
        a=analysis()
        rows=[dict(repeat=1,fold=f,model='M1',protocol='main',sample_id=f'M{i}',
                   y_true_pct=y,y_pred_pct=p) for i,(f,y,p) in enumerate([(1,0.,1.),(1,1.,2.),(2,10.,9.),(2,11.,10.)],1)]
        result=a.repeat_metrics(rows)[0]
        self.assertAlmostEqual(result['r2'],1-4/101)

    def test_block_small_variance_r2_is_retained(self):
        a=analysis()
        result=a.metrics([1.,1.0001],[2.,2.])
        self.assertLess(result['r2'],-1e7)
        self.assertEqual(result['n'],2)

    def test_support_table_has_explicit_training_rss_and_total_capacity(self):
        a=analysis()
        tasks=[dict(model='M1',protocol='block',repeat=1,fold=fold,n_train=200,y_train_std=2.,
                    diagnostics=dict(supports=[[0],[0],[0]],q=1,caps=[2,3,4],q_max=140,
                                     msic=5.,ols_rank=1,ols_condition=1.),
                    elapsed_seconds=1.,train_metrics=dict(rmse=3.),test_metrics=dict(rmse=4.))
               for fold in [1,2]]
        rows,_,_=a.support_tables(tasks,[dict(min_nm=400.,max_nm=420.,representative_nm=410.)]*24)
        self.assertIn('train_rss_pct2',rows[0])
        self.assertEqual(rows[0]['train_rss_pct2'],1800.)
        self.assertEqual(rows[0]['train_rss_standardized'],450.)
        self.assertEqual(rows[0]['Q'],24)
        self.assertIn('effective_capacity',rows[0])
        self.assertEqual(rows[0]['effective_capacity'],24)
        self.assertAlmostEqual(rows[0]['q_over_Q'],1/24)
        self.assertAlmostEqual(rows[0]['q_over_budget'],1/140)
        self.assertAlmostEqual(rows[0]['q_over_effective_capacity'],1/24)


if __name__=='__main__':unittest.main()
