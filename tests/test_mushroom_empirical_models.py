import importlib
import pickle
import unittest
import numpy as np


class MushroomModels(unittest.TestCase):
    def test_screen_identity_and_train_only_scaling(self):
        api=importlib.import_module('empirical_analysis.spectrofood_mushroom.models')
        from empirical_analysis.mushroom_screen import fit_model,predict
        rng=np.random.default_rng(3);X=rng.normal(size=(18,5));y=14+X[:,1]*2
        for mid,name,param in [('M6','Ridge',1.),('M7','PLS',2)]:
            a=api.train_model(mid,X[:12],y[:12],param)
            b=fit_model(name,X[:12],y[:12],param)
            np.testing.assert_allclose(api.predict_model(pickle.loads(pickle.dumps(a)),X[12:]),predict(b,X[12:]),atol=1e-12)
            np.testing.assert_allclose(a['preprocessing']['x_mean'],X[:12].mean(0))
        self.assertEqual(a['native_artifact']['estimator'].max_iter,1000)
        self.assertEqual(a['native_artifact']['estimator'].tol,1e-7)

    def test_shared_grid_and_training_only_inner_fits(self):
        api=importlib.import_module('empirical_analysis.spectrofood_mushroom.models')
        rng=np.random.default_rng(3);X=rng.normal(size=(18,5));y=14+X[:,1]*2
        splits=[(np.arange(9),np.arange(9,18)),(np.arange(9,18),np.arange(9))]
        p,records=api.tune_model('M6',X,y,splits,config={'ridge_alphas':[.1,1.]})
        self.assertEqual(len(records),2)
        self.assertIn(p,[.1,1.])
