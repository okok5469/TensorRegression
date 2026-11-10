"""Adversarial acceptance checks: fabricated evidence must not pass."""
import unittest
import copy
import tempfile
import json
from pathlib import Path
from math import lgamma
import numpy as np



def fixture():
    rng=np.random.default_rng(17)
    raw=rng.normal(size=(245,128)); x=raw.reshape(-1,16,8)[:,:,2:].reshape(-1,16,2,3).transpose(0,1,3,2).copy()
    batch=np.repeat(np.arange(1,8),35); y=raw[:,0]+10
    data=dict(raw128=raw,X=x,y=y,batch=batch,gas=np.ones(245,int),group96=np.arange(245))
    train=np.arange(210); test=np.arange(210,245)
    prep=dict(x_mean=x[train].mean(0),x_scale=x[train].std(0),constant_mask=np.zeros((16,3,2),bool),y_mean=y[train].mean(),y_scale=y[train].std())
    artifact=dict(model_id='M0',train_indices=train,test_indices=test,inner_pairs=[(np.arange(105),np.arange(105,140)),(np.arange(140),np.arange(140,175)),(np.arange(175),np.arange(175,210))],preprocessing=prep,coef_standardized=np.zeros(96),intercept_standardized=0.,parameters={},cv_records=[],diagnostics={'training_rank':96})
    return artifact,data


def call(a,d,p=None):
    from empirical_analysis.uci270_concentration.audit import audit_artifact
    return audit_artifact(a,d,p)


class AuditTests(unittest.TestCase):
    def test_training_rank_corruption_rejected(self):
        a,d=fixture();a['diagnostics']['training_rank']=95
        with self.assertRaisesRegex(ValueError,'training rank'):call(a,d)

    def test_effective_rank_of_collinear_lasso_coordinates(self):
        from empirical_analysis.uci270_concentration.audit import _design_complexity
        a,d=fixture();a['model_id']='M3';a['coef_standardized'][:2]=[1,2]
        d['X'][:,:,:,1]=d['X'][:,:,:,0]
        a['preprocessing']['x_mean']=d['X'][:210].mean(0)
        a['preprocessing']['x_scale']=d['X'][:210].std(0)
        result=_design_complexity(a,d)
        self.assertEqual(result['effective_design_rank'],1)
        self.assertEqual(result['native_active_coordinates'],[0,1])
        self.assertEqual(result['support_capacity'],2)

    def test_final_requires_development_lock_before_reading(self):
        from empirical_analysis.uci270_concentration.audit import verify_all
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'contract.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'development acceptance'):verify_all(root,'final')

    def test_independent_metrics_constant_and_empty(self):
        from empirical_analysis.uci270_concentration.audit import _expected_metrics
        value=_expected_metrics([2,2],[1,3])
        self.assertIsNone(value['r2']);self.assertEqual(value['rmse'],1);self.assertEqual(value['bias'],0)
        self.assertIsNone(_expected_metrics([],[])['rmse'])

    def test_tolerance_prefers_smallest_eligible_rectangle(self):
        from empirical_analysis.uci270_concentration.audit import _select_candidates
        result=_select_candidates([([2,2,2],0.),([1,1,1],1.),([1,1,2],50.)],100,1.)
        self.assertEqual(result['selected_index'],1)
        self.assertAlmostEqual(result['tolerance'],np.sqrt(8*np.log(100)))
    def test_complete_seal_rejects_missing_files(self):
        from empirical_analysis.uci270_concentration.audit import _sealed_task
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name); (folder/'COMPLETE.json').write_text('{"status":"COMPLETE","task_id":"x","contract_sha256":"c","files":{}}')
            with self.assertRaisesRegex(ValueError,'required files'):_sealed_task(folder,'x','c')

    def test_correct_mean_replays(self):
        a,d=fixture(); r=call(a,d,np.repeat(a['preprocessing']['y_mean'],35))
        self.assertTrue(r['passed']); self.assertEqual(r['prediction_max_abs'],0)

    def test_corrupt_evidence_fails(self):
        for damage in ['mapping','scaler','time','duplicates','prediction','inner']:
            with self.subTest(damage=damage):
                a,d=fixture(); p=np.repeat(a['preprocessing']['y_mean'],35)
                if damage=='mapping':d['X'][0,0,0,0]+=1
                if damage=='scaler':a['preprocessing']['y_mean']+=1
                if damage=='time':d['batch'][0]=10
                if damage=='duplicates':d['group96'][0]=d['group96'][210]
                if damage=='prediction':p[0]+=1
                if damage=='inner':a['inner_pairs'][0]=(np.arange(141),np.arange(140,175))
                with self.assertRaises(ValueError):call(a,d,p)

    def test_cv_aggregation_rejected(self):
        a,d=fixture();a['model_id']='M4'
        a['cv_records']=[dict(parameter={'alpha':1.},mse=99.,folds=[dict(train_indices=t,validation_indices=v,n_validation=len(v),sse=1.,preprocessing=dict(x_mean=d['X'][t].mean(0),x_scale=d['X'][t].std(0),constant_mask=np.zeros((16,3,2),bool),y_mean=d['y'][t].mean(),y_scale=d['y'][t].std())) for t,v in a['inner_pairs']])]
        a['parameters']={'alpha':1.}
        with self.assertRaisesRegex(ValueError,'mse'):call(a,d)

    def test_older_inner_boundary_rejected(self):
        a,d=fixture();a['inner_pairs'][0]=(np.arange(70),np.arange(70,105))
        with self.assertRaisesRegex(ValueError,'recent'):call(a,d)

    def test_abess_ols_and_msic(self):
        a,d=fixture();a['model_id']='M1'; p=a['preprocessing'];t=a['train_indices']
        z=((d['X'][t]-p['x_mean'])/p['x_scale']).reshape(len(t),96)
        target=(d['y'][t]-p['y_mean'])/p['y_scale'];c=np.linalg.lstsq(z[:,:1],target,rcond=None)[0]
        a['coef_standardized'][0]=c[0];a['B']=a['coef_standardized'].reshape(16,3,2)
        rss=float(np.sum((target-z[:,0]*c[0])**2)); ell=np.log(max(np.log(len(t)),np.e))
        parts=dict(log_rss=len(t)*np.log(rss/len(t)),df=1.5*ell,combinatorial=ell*np.log(16*3*2))
        a['diagnostics']=dict(q=1,supports=[[0],[0],[0]],rss=rss,msic_components=parts,msic=sum(parts.values()),training_rank=96)
        self.assertEqual(call(a,d)['abess']['rank'],1)
        broken=copy.deepcopy(a);broken['diagnostics']['msic']+=1
        with self.assertRaisesRegex(ValueError,'MSIC'):call(broken,d)
