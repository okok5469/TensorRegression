import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np


class MushroomScreenTests(unittest.TestCase):
    def api(self):
        try: return importlib.import_module('empirical_analysis.mushroom_screen')
        except ModuleNotFoundError: self.fail('Mushroom screen not implemented')

    def test_section_specific_header_and_padding(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv'
            p.write_text('\ufeffApple,Dry matter,430,434,,,\nA1,.16,9,8,,,\n\nMush No,Dry matter,400,402.95,,,\nM2,.08,.1,.2,,,\nM1,.09,.3,.4,,,\nBroccoli,Dry matter percentage,470,475,,,\nB1,.13,3,4,,,\n',encoding='utf-8')
            ids,y,w,x,lines=api.read_segment(p)
            self.assertEqual(ids,['M2','M1'])
            np.testing.assert_allclose(y,[.08,.09])
            np.testing.assert_allclose(w,[400,402.95])
            np.testing.assert_allclose(x,[[.1,.2],[.3,.4]])

    def test_exact_duplicate_spectra_group_despite_different_labels(self):
        api=self.api()
        X=np.array([[1.,2],[3,4],[1,2],[5,6],[7,8],[9,10]])
        groups=api.spectrum_groups(X)
        self.assertEqual(groups[0],groups[2])
        for tr,va in api.group_folds(groups,3,23001):
            self.assertFalse(set(groups[tr])&set(groups[va]))
        self.assertEqual(sorted(i for _,va in api.group_folds(groups,3,23001) for i in va),list(range(6)))

    def test_training_scaler_and_inverse_ignore_heldout_data(self):
        api=self.api()
        rng=np.random.default_rng(82)
        X=rng.normal(size=(30,5));X[:,0]=3
        y=15+X[:,2]+rng.normal(size=30)*.01
        model=api.fit_model('Ridge',X[:20],y[:20],.01)
        test=X[20:].copy();test[:,0]=1e8
        np.testing.assert_allclose(api.predict(model,test),api.predict(model,X[20:]),rtol=0,atol=1e-12)
        self.assertAlmostEqual(model['y_mean'],y[:20].mean())
        self.assertLess(np.sqrt(np.mean((api.predict(model,test)-y[20:])**2)),.05)
        mean=api.fit_model('Mean',X[:20],y[:20],None)
        np.testing.assert_array_equal(api.predict(mean,test),np.full(10,y[:20].mean()))

    def test_inner_validation_mse_independently_refits_scaling(self):
        api=self.api()
        rng=np.random.default_rng(83);X=rng.normal(size=(30,5));y=10+X[:,0]
        splits=api.group_folds(np.arange(30).astype(str),3,230101)
        parameter,records=api.tune('Ridge',X,y,splits,[.01,1])
        for record in records:
            total=0
            for tr,va in splits:
                model=api.fit_model('Ridge',X[tr],y[tr],record['parameter'])
                total+=np.sum((api.predict(model,X[va])-y[va])**2)
            self.assertAlmostEqual(record['mse'],total/30,places=12)
        self.assertEqual(parameter,min(records,key=lambda r:r['mse'])['parameter'])

    def test_complete_oof_metric_definition(self):
        api=self.api()
        y=np.array([0.,1,10,12]);pred=np.array([1.,2,8,10])
        result=api.metrics(y,pred)
        self.assertAlmostEqual(result['rmse'],np.sqrt(2.5))
        self.assertAlmostEqual(result['r2'],1-10/np.sum((y-y.mean())**2))


if __name__=='__main__': unittest.main()
