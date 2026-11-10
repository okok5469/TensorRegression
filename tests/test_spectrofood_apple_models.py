"""Contract checks for training-local Apple model adapters."""
import importlib
import pickle
import unittest
import json
from pathlib import Path
import numpy as np


class AppleModelTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('empirical_analysis.spectrofood_apple.models')
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(24, 2, 2, 3))
        self.X[:, 0, 0, 0] = 7
        self.y = 16 + 2*self.X[:, 1, 0, 2] + rng.normal(size=24)*.02

    def test_training_only_scaling_constant_coordinates_and_physical_units(self):
        fit = self.api.train_model('M4', self.X[:18], self.y[:18], {'alpha': .01})
        prep = fit['preprocessing']
        np.testing.assert_allclose(prep['x_mean'], self.X[:18].mean(axis=0))
        self.assertEqual(prep['y_mean'], self.y[:18].mean())
        self.assertTrue(prep['constant_mask'][0, 0, 0])
        changed = self.X[18:].copy()
        changed[:, 0, 0, 0] = 1e9
        np.testing.assert_allclose(self.api.predict_model(fit, changed),
                                   self.api.predict_model(fit, self.X[18:]), atol=1e-12)
        self.assertLess(np.mean((self.api.predict_model(fit, self.X[18:])-self.y[18:])**2), .1)

    def test_mean_and_constant_response(self):
        fit = self.api.train_model('M0', self.X, self.y)
        np.testing.assert_array_equal(self.api.predict_model(fit, self.X[:3]), np.full(3,self.y.mean()))
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            self.api.train_model('M4', self.X, np.ones(24))

    def test_c_order_and_pickle_independent_prediction(self):
        fit = self.api.train_model('M4', self.X, self.y, {'alpha': 1})
        prep = fit['preprocessing']
        z = (self.X-prep['x_mean'])/prep['x_scale']
        z[:,prep['constant_mask']] = 0
        expected = fit['estimator'].predict(z.reshape(24,-1,order='C'))*prep['y_scale']+prep['y_mean']
        restored = pickle.loads(pickle.dumps(fit))
        np.testing.assert_allclose(self.api.predict_model(restored,self.X),expected,atol=1e-12)

    def test_inner_scalers_refit_and_weighted_percentage_mse(self):
        splits = [(np.arange(8,24),np.arange(8)),(np.r_[0:8,16:24],np.arange(8,16)),(np.arange(16),np.arange(16,24))]
        param, records = self.api.tune_model('M4',self.X,self.y,splits,config={'ridge_alphas':[.01,1]})
        self.assertEqual(len(records),2)
        for record in records:
            sse=0
            for tr,va in splits:
                fit=self.api.train_model('M4',self.X[tr],self.y[tr],record['parameter'])
                sse+=np.sum((self.api.predict_model(fit,self.X[va])-self.y[va])**2)
            self.assertAlmostEqual(record['mse_pct2'],sse/24,places=12)
        self.assertEqual(param,min(records,key=lambda r:r['mse_pct2'])['parameter'])

    def test_lasso_training_local_alpha_max_and_pls_rank_grid(self):
        fit=self.api.train_model('M3',self.X,self.y,{'lambda_ratio':.1})
        p=fit['preprocessing']; z=(self.X-p['x_mean'])/p['x_scale']; z[:,p['constant_mask']]=0
        ys=(self.y-p['y_mean'])/p['y_scale']
        maximum=np.max(np.abs(z.reshape(24,-1).T@ys/24))
        self.assertAlmostEqual(fit['diagnostics']['alpha_max'],maximum)
        spectrum=np.column_stack([self.y,self.y*2])
        splits=[(np.arange(12),np.arange(12,24)),(np.arange(12,24),np.arange(12))]
        _, records=self.api.tune_model('M5',spectrum,self.y,splits)
        self.assertEqual([r['parameter']['n_components'] for r in records],[1])

    def test_m1_retains_formal_adaptive_original_msic(self):
        fit=self.api.train_model('M1',self.X,self.y)
        direct=importlib.import_module('tensor_abess.tensor.abess').tensor_abess
        p=fit['preprocessing']; z=(self.X-p['x_mean'])/p['x_scale']; z[:,p['constant_mask']]=0
        result=direct(z,(self.y-p['y_mean'])/p['y_scale'],**fit['parameters'])
        np.testing.assert_array_equal(fit['B'],result.B)
        self.assertIsNone(fit['parameters']['target_sparsity'])
        self.assertEqual(fit['parameters']['adaptive_criterion'],'msic')
        self.assertAlmostEqual(fit['diagnostics']['msic'],result.sic)

    def test_m1_msic_components_candidate_count_and_original_ols_reconstruction(self):
        from tensor_abess.tensor.bcd import _candidate_size_triples
        from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
        from tensor_abess.tensor.ops import build_tensor_design, refit_tensor_coefficients
        fit = self.api.train_model('M1',self.X,self.y)
        diag,prep = fit['diagnostics'],fit['preprocessing']
        z=(self.X-prep['x_mean'])/prep['x_scale']
        z[:,prep['constant_mask']]=0
        ys=(self.y-prep['y_mean'])/prep['y_scale']
        supports=[np.asarray(s,dtype=int) for s in diag['supports']]
        original_B=refit_tensor_coefficients(z,ys,*supports)
        np.testing.assert_allclose(original_B,fit['B'],rtol=1e-12,atol=1e-12)
        design=build_tensor_design(z,*supports)
        recovered,_,rank,_=np.linalg.lstsq(design,ys,rcond=None)
        np.testing.assert_allclose(design@recovered,np.einsum('nijk,ijk->n',z,fit['B']),rtol=1e-12,atol=1e-12)
        self.assertEqual(diag['ols_rank'],rank)
        residual=ys-design@recovered
        expected=compute_tensor_msic_from_rss(float(residual@residual),*[len(s) for s in supports],len(z),*z.shape[1:],alpha=1.5,gamma=1.)
        self.assertAlmostEqual(sum(diag['msic_components'].values()),expected,places=10)
        self.assertAlmostEqual(diag['msic'],expected,places=10)
        # The official refinement evaluates the warm incumbent in addition
        # to every size triple, even when a triple duplicates its shape.
        triples=_candidate_size_triples(*diag['caps'],int(np.floor(.7*len(z))))
        self.assertEqual(diag['candidate_count'],1+len(triples))
        self.assertEqual(diag['candidate_grid_count'],len(triples))
        np.testing.assert_allclose(self.api.predict_model(pickle.loads(pickle.dumps(fit)),self.X),
                                   (design@recovered)*prep['y_scale']+prep['y_mean'],rtol=1e-12,atol=1e-12)

    def test_m2_reuses_verified_training_local_tensor_glasso(self):
        fit=self.api.train_model('M2',self.X,self.y,{'lambda_ratio':.1})
        self.assertTrue(fit['diagnostics']['group_lambda_trace'])
        self.assertTrue(np.isfinite(self.api.predict_model(pickle.loads(pickle.dumps(fit)),self.X)).all())
        self.assertFalse(fit['diagnostics']['internal_cv'])

    @unittest.skipUnless(Path('doc/taskcard13/data/data.npz').is_file(), 'Requires frozen local Apple training cache')
    def test_lasso_converges_on_frozen_smoke_training_only(self):
        from empirical_analysis.spectrofood_apple.run import group_folds
        from threadpoolctl import threadpool_limits
        output = Path('empirical_analysis/spectrofood_apple/outputs/apple_v1')
        split = json.loads((output/'splits.json').read_text(encoding='utf-8'))[0]
        outer_train = np.asarray(split['train'],dtype=int)[:64]
        with np.load('doc/taskcard13/data/data.npz',allow_pickle=False) as data:
            inner_train,_ = group_folds(data['entity_ids'][outer_train],3,130101)[0]
            selected = outer_train[inner_train]
            # Do not inspect outer-test X/y, validation errors, or rankings.
            X,y = data['X'][selected],data['y_pct'][selected]
        self.assertEqual(len(selected),42)
        with threadpool_limits(limits=1):
            fit = self.api.train_model('M3',X,y,{'lambda_ratio':.01},seed=130101)
        diag = fit['diagnostics']
        self.assertTrue(diag['converged'],str(diag))
        self.assertFalse(diag['warnings'],str(diag))
        self.assertEqual(fit['estimator'].max_iter,100000)
        self.assertEqual(fit['estimator'].tol,1e-6)
        self.assertLessEqual(diag['dual_gap'],1e-6)

if __name__=='__main__':
    unittest.main()
