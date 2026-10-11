"""Independent paired OOF estimand and subject-cluster tests."""
import unittest


def fixture():
    names=['Covariates Only','Tensor-ABESS','Vector Lasso','Tensor-GLasso','Ridge']
    return [{'subject_id':str(s),'repeat':r,'outer_fold':s%2,'y_true':float(s),
             'y_pred':s+(2. if m==names[0] else 1.),'model':m}
            for r in range(2) for s in range(4) for m in names]


class MetricTests(unittest.TestCase):
    def test_incremental_estimands_and_cluster_interval(self):
        from .metrics import summarize_oof
        stats=summarize_oof(fixture(),bootstrap_samples=100)
        self.assertEqual(len(stats['metrics_by_repeat']),10)
        for row in stats['incremental_metrics']:
            self.assertEqual(row['delta_mse'],3.)
            self.assertEqual(row['delta_rmse'],1.)
            self.assertEqual(row['incremental_r2'],.75)
            self.assertEqual(row['delta_mse_ci_low'],3.)
            self.assertEqual(row['delta_mse_ci_high'],3.)

    def test_missing_duplicate_nonfinite_and_unpaired_rejected(self):
        from .metrics import summarize_oof
        for rows in [fixture()+[fixture()[0]],fixture()[:-1]]:
            with self.assertRaises(ValueError): summarize_oof(rows,bootstrap_samples=10)
        rows=fixture(); rows[1]['outer_fold']=99
        with self.assertRaises(ValueError): summarize_oof(rows,bootstrap_samples=10)
        rows=fixture(); rows[1]['y_pred']=float('nan')
        with self.assertRaises(ValueError): summarize_oof(rows,bootstrap_samples=10)

    def test_cluster_reproducibility_and_subject_weight(self):
        from .metrics import summarize_oof
        rows=fixture()
        for row in rows:
            if row['model']!='Covariates Only': row['y_pred']=row['y_true']+int(row['subject_id'])
        a=summarize_oof(rows,123,100); self.assertEqual(a,summarize_oof(rows,123,100))
        p=[r['mean_improvement'] for r in a['per_subject_improvement'] if r['model']=='Ridge']
        self.assertEqual(p,[4.,3.,0.,-5.])
        agg=next(r for r in a['incremental_metrics'] if r['model']=='Ridge' and r['repeat']=='all')
        self.assertEqual(agg['improved_subject_fraction'],.5)
        self.assertEqual(agg['worsened_subject_fraction'],.25)
        self.assertEqual(agg['median_subject_improvement'],1.5)

    def test_single_repeat_sd_undefined(self):
        from .metrics import summarize_oof
        a=summarize_oof([r for r in fixture() if r['repeat']==0],bootstrap_samples=10)
        self.assertIsNone(a['model_summary'][0]['rmse_sd'])


if __name__=='__main__': unittest.main()
