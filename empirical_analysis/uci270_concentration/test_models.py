import pickle
import unittest
import numpy as np
from empirical_analysis.uci270_concentration import models


class ModelContracts(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(17)
        self.X = r.normal(size=(90,16,3,2))
        self.raw = r.normal(size=(90,128))
        self.y = 30 + self.X[:,0,0,0]*5 + r.normal(size=90)
        self.train, self.test = np.arange(70), np.arange(70,90)
        self.inner = [(np.arange(35),np.arange(35,45)),(np.arange(45),np.arange(45,60))]

    def test_train_only_scaling_replay_and_coef(self):
        for method in ('M0','M3','M4','M5','M6','M7'):
            out = models.fit_task(self.X,self.raw,self.y,self.train,self.test,self.inner,method)
            artifact = pickle.loads(pickle.dumps(out['artifact']))
            np.testing.assert_allclose(models.predict_model(artifact,self.X[self.test],self.raw[self.test]),out['predictions'])
            selected = self.raw if method in ('M6','M7') else self.X
            np.testing.assert_allclose(artifact['preprocessing']['x_mean'],selected[self.train].mean(axis=0))
            Z = models.transform(selected[self.test],artifact['preprocessing']).reshape(20,-1)
            pred = (Z@out['coef_standardized']+artifact['intercept_standardized'])*artifact['preprocessing']['y_scale']+artifact['preprocessing']['y_mean']
            np.testing.assert_allclose(pred,out['predictions'],rtol=1e-10,atol=1e-10)
            if 'estimator' in artifact:
                native=np.asarray(artifact['estimator'].predict(Z)).reshape(-1)*artifact['preprocessing']['y_scale']+artifact['preprocessing']['y_mean']
                np.testing.assert_allclose(native,out['predictions'],rtol=1e-10,atol=1e-10)
            for candidate in out['cv_records']:
                self.assertAlmostEqual(candidate['mse'],sum(f['sse'] for f in candidate['folds'])/sum(f['n_validation'] for f in candidate['folds']))
                for fold in candidate['folds']:
                    a=np.array(fold['train_indices'])
                    b=np.array(fold['validation_indices'])
                    np.testing.assert_allclose(fold['sse'],np.sum((np.array(fold['predictions'])-self.y[b])**2))
                    innerZ=models.transform(selected[b],fold['preprocessing']).reshape(len(b),-1)
                    direct=(innerZ@fold['coef_standardized']+fold['intercept_standardized'])*fold['preprocessing']['y_scale']+fold['preprocessing']['y_mean']
                    np.testing.assert_allclose(direct,fold['predictions'])
                    np.testing.assert_allclose(fold['preprocessing']['x_mean'],selected[a].mean(axis=0))
                    if method in ('M3','M7'):
                        prep=fold['preprocessing']
                        innerX=models.transform(selected[a],prep).reshape(len(a),-1)
                        innerY=(self.y[a]-prep['y_mean'])/prep['y_scale']
                        self.assertAlmostEqual(fold['diagnostics']['alpha_max'],np.max(np.abs(innerX.T@innerY))/len(a))

    def test_future_labels_do_not_change_fit(self):
        first=models.fit_task(self.X,self.raw,self.y,self.train,self.test,self.inner,'M3')
        changed=self.y.copy(); changed[self.test]=1e12
        second=models.fit_task(self.X,self.raw,changed,self.train,self.test,self.inner,'M3')
        np.testing.assert_array_equal(first['predictions'],second['predictions'])

    def test_zero_and_near_zero_scale(self):
        X=np.array([[1.,1e-20],[1.,2e-20],[1.,3e-20]])
        prep=models.preprocess(X,np.arange(3.))
        self.assertEqual(prep['x_scale'][0],1.)
        self.assertLess(prep['x_scale'][1],1e-19)
        self.assertEqual(models.transform(np.array([[2.,4e-20]]),prep)[0,0],1.)

    def test_inner_membership_rejected(self):
        with self.assertRaises(ValueError):
            models.fit_task(self.X,self.raw,self.y,self.train,self.test,[(np.arange(35),np.array([80]))], 'M4')

    def test_ties_and_frozen_grids(self):
        self.assertEqual(models.select_parameter([{'parameter':{'alpha':1.},'mse':1.},{'parameter':{'alpha':10.},'mse':1.+1e-13}], 'M4'),{'alpha':10.})
        self.assertEqual(len(models.parameter_grid('M3',self.X,self.y,self.inner)),10)

    def test_tensor_native_identity_and_ols(self):
        from tensor_abess.tensor.abess import tensor_abess
        X=self.X[:40,:2]
        y=self.y[:40]
        for method,param in [('M1',None),('M2',{'lambda_ratio':.1})]:
            artifact=models.train_model(method,X,y,param)
            Z=models.transform(X,artifact['preprocessing'])
            ys=(y-y.mean())/y.std()
            support=artifact['diagnostics']['supports']
            design=Z[np.ix_(np.arange(len(X)),*support)].reshape(len(X),-1)
            beta=np.linalg.lstsq(design,ys,rcond=None)[0]
            np.testing.assert_allclose(design@beta,Z.reshape(len(X),-1)@artifact['coef_standardized'],atol=1e-8)
            if method=='M1':
                native=tensor_abess(Z,ys,**artifact['parameters'])
                np.testing.assert_array_equal(native.B,artifact['B'])
                self.assertAlmostEqual(native.sic,sum(artifact['diagnostics']['msic_components'].values()),places=8)

if __name__=='__main__':
    unittest.main()
