import importlib
import pickle
import unittest
import numpy as np


class BeijingModels(unittest.TestCase):
    def test_training_median_and_replay(self):
        api=importlib.import_module('empirical_analysis.beijing_air_quality.models')
        rng=np.random.default_rng(3);X=rng.normal(size=(18,2,2,3));y=np.arange(18.)+10
        X[0,0,0,0]=np.nan
        fit=api.train_model('M3',X[:12],y[:12],{'alpha':1.})
        self.assertEqual(fit['imputer'][0,0,0],np.nanmedian(X[:12,0,0,0]))
        test=X[12:].copy();test[:,0,0,0]=np.nan
        np.testing.assert_array_equal(api.predict_model(fit,test),api.predict_model(pickle.loads(pickle.dumps(fit)),test))
        changed=X.copy();changed[12:]=1e9
        second=api.train_model('M3',changed[:12],y[:12],{'alpha':1.})
        np.testing.assert_array_equal(fit['imputer'],second['imputer'])
        X[:12,1,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'all-missing'):
            api.train_model('M3',X[:12],y[:12])

    def test_means_persistence_and_inner_medians(self):
        api=importlib.import_module('empirical_analysis.beijing_air_quality.models')
        rng=np.random.default_rng(2);X=rng.normal(size=(18,2,2,3));y=np.arange(18.)+10
        X[::3,0,0,0]=np.nan
        fit=api.train_model('M0',X,y)
        np.testing.assert_array_equal(api.predict_model(fit,X[:4]),np.full(4,y.mean()))
        fit=api.train_model('M1',X,y)
        np.testing.assert_array_equal(api.predict_model(fit,X[:4],np.array([7.,8.,9.,10.])),[7.,8.,9.,10.])
        splits=[(np.arange(6),np.arange(6,10)),(np.arange(10),np.arange(10,14))]
        _,records=api.tune_model('M3',X,y,splits,config={'ridge_alphas':[1.]})
        total=0
        for a,b in splits:
            fitted=api.train_model('M3',X[a],y[a],{'alpha':1.})
            total+=np.sum((api.predict_model(fitted,X[b])-y[b])**2)
        self.assertAlmostEqual(records[0]['mse'],total/8)
