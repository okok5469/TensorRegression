import importlib.util
import unittest
from pathlib import Path


def module():
    path=Path('doc/beijing_air_quality/analyze_results.py')
    assert path.exists(),'Beijing analysis module missing'
    spec=importlib.util.spec_from_file_location('beijing_analysis',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class AnalysisTests(unittest.TestCase):
    def test_pooled_r2_uses_all_dates_and_bias_is_prediction_minus_observed(self):
        a=module();m=a.metrics([0,1,10,11],[1,2,11,12])
        self.assertAlmostEqual(m['r2'],1-4/101)
        self.assertEqual(m['bias'],1)
        self.assertEqual(m['n'],4)

    def test_monthly_comparisons_use_same_dates_and_named_baselines(self):
        a=module();rows=[dict(protocol='development',fold=1,date=f'2014-03-0{i}',model=model,y=float(i),yhat=float(i)+err)
              for model,err in [('M1',3),('M2',2),('M6',1)] for i in [1,2]]
        table=a.monthly_metrics(rows);row=next(r for r in table if r['model']=='M6')
        self.assertEqual(row['rmse_minus_persistence'],-2)
        self.assertEqual(row['rmse_minus_single_ridge'],-1)
        with self.assertRaises(ValueError): a.monthly_metrics(rows[:-1])

    def test_future_does_not_enter_development_selection_frequency(self):
        a=module();records=[dict(protocol='development',fold=1,supports=[[0],[0],[0]]),
                          dict(protocol='development',fold=2,supports=[[1],[0],[0]]),
                          dict(protocol='future',fold=1,supports=[[0],[0],[0]])]
        freq=a.selection_frequency(records,[['A','B'],['PM2.5'],['00-03']])
        row=next(r for r in freq if r['protocol']=='development' and r['mode']==0 and r['index']==0)
        self.assertEqual(row['frequency'],.5)
        self.assertEqual(row['n_fits'],2)


if __name__=='__main__':unittest.main()
