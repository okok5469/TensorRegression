"""Trace unchanged solvers, inspect OLS, and validate sizes inside outer train."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from scipy.linalg import lstsq
from tensor_abess.tensor.ops import build_tensor_design, refit_tensor_coefficients
from tensor_abess.tensor.sic import log_binomial
from ..confounds import FoldPreprocessor, fit_preprocessor
from ..incremental_value.nested_cv import make_splits


def selection_audit(rows, n, scale):
    minimum=min(range(len(rows)),key=lambda i:rows[i]['score'])
    tolerance=float(scale*np.sqrt(max(rows[minimum]['q'],1)*max(np.log(max(n,2)),1))) if scale>0 else 0.
    threshold=rows[minimum]['score']+tolerance
    eligible=[i for i,r in enumerate(rows) if r['score']<=threshold]
    selected=min(eligible,key=lambda i:(rows[i]['q'],rows[i]['score'],rows[i].get('s_sum',0)))
    return {'minimum_index':minimum,'minimum_score':rows[minimum]['score'],
            'q_best':rows[minimum]['q'],'tolerance':tolerance,'threshold':threshold,
            'eligible_indices':eligible,'selected_index':selected}


def ols_audit(Z, y, C):
    coef, _, rank, singular=np.linalg.lstsq(Z,y,rcond=None)
    predicted=Z@coef;residual=y-predicted
    alternative=[lstsq(Z,y,lapack_driver=driver)[0] for driver in ('gelsd','gelsy','gelss')]
    delta=max(float(np.max(np.abs(Z@b-predicted))) for b in alternative)
    joint_rank=int(np.linalg.matrix_rank(np.column_stack((C,Z))))
    condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else None
    return {'n':len(y),'q':Z.shape[1],'rank':int(rank),'full_column_rank':int(rank)==Z.shape[1],
            'singular_max':float(singular[0]),'singular_min':float(singular[-1]),
            'condition_number':condition,'joint_rank':joint_rank,
            'residual_df_joint':len(y)-joint_rank,'rss':float(residual@residual),
            'coefficient_l2':float(np.linalg.norm(coef)),
            'normal_equation_residual_l2':float(np.linalg.norm(Z.T@residual)),
            'solver_prediction_max_difference':delta,
            'solver_rss_max_difference':max(abs(float((y-Z@b)@(y-Z@b))-float(residual@residual)) for b in alternative),
            'finite':bool(np.isfinite(coef).all() and np.isfinite(predicted).all())}


def _support(result):
    return [np.asarray(getattr(result,f'S{i}'),dtype=int).tolist() for i in (1,2,3)]


def _result(result, n):
    supports=_support(result);sizes=[len(s) for s in supports]
    return {'sizes':sizes,'q':int(np.prod(sizes)),'supports':supports,
            'rss':float(2*n*result.loss),'loss_history':list(result.loss_history),
            'rss_history':(2*n*np.asarray(result.loss_history)).tolist(),
            'sic':float(result.sic) if hasattr(result,'sic') else None}


def traced_abess(X,y,options):
    entry=importlib.import_module('tensor_abess.tensor.abess')
    bcd=importlib.import_module('tensor_abess.tensor.bcd')
    original_init=entry.tensor_init_glasso
    original_mode=bcd.group_bess_adaptive
    original_refine=bcd._refine_msic_global_candidates
    original_fixed=bcd.tensor_bcd
    trace={'initialization':{},'mode_updates':[],'candidates':[]}
    n,p1,p2,p3=X.shape

    def init_wrapper(*args,**kwargs):
        result=original_init(*args,**kwargs)
        trace['initialization']={'supports':_support_init(result),
            'sizes':[len(getattr(result,f'S{i}_init')) for i in (1,2,3)],
            'caps':[getattr(result,f's{i}_max') for i in (1,2,3)],
            'candidate_set_sizes':[len(getattr(result,f'C{i}')) for i in (1,2,3)],
            'marginal_floor':max(1,int(np.floor(np.sqrt(n)/2))),
            'max_rectangular_params':int(np.floor(.7*n)),
            'glasso_fits':{str(mode):{'lambda':float(f.lam),'rss':float(f.rss),
                                    'active_group_count':len(f.active_groups)}
                          for mode,f in result.glasso_fits.items()}}
        return result

    def mode_wrapper(*args,**kwargs):
        result=original_mode(*args,**kwargs)
        number=len(trace['mode_updates'])
        trace['mode_updates'].append({'cycle':number//3+1,'mode':number%3+1,
            'previous_support':np.asarray(kwargs['groups_warm']).tolist(),
            'selected_support':result.active_groups.tolist(),'selected_s':int(result.s),
            'selected_rss':float(result.rss),'selected_msic':float(result.sic),
            'block_size':int(kwargs['block_size']),
            'candidates':[{'s':int(s),'rss':float(f.rss),'msic':float(result.sic_path[s])}
                          for s,f in result.bess_path.items()]})
        return result

    def candidate_record(result,sizes,score,label,initial_supports):
        q=int(np.prod(sizes));rss=float(2*n*result.loss);loglog=float(np.log(np.log(n)))
        terms={'fit_term':float(n*np.log(rss/n)),
               'df_penalty':float(options.get('msic_refine_alpha',1.5)*q*loglog),
               'support_penalty':float(options.get('msic_refine_gamma',1.)*loglog*
                 sum(log_binomial(p,s) for p,s in zip((p1,p2,p3),sizes)))}
        Z=build_tensor_design(X,*[np.asarray(s) for s in _support(result)])
        trace['candidates'].append({'source':label,'sizes':list(sizes),'q':q,'s_sum':sum(sizes),
            'score':float(score),**terms,**_result(result,n),
            'initial_supports':initial_supports,
            'ols':ols_audit(Z,y,np.empty((n,0)))})

    def refine_wrapper(*args,**kwargs):
        before=args[2];trace['before_refinement']=_result(before,n)
        criterion=kwargs['tensor_criterion_fn']
        sizes=tuple(len(s) for s in _support(before))
        score=criterion(2*n*before.loss,*sizes,n,p1,p2,p3)
        candidate_record(before,sizes,score,'adaptive_before_refinement',None)
        def fixed_wrapper(*fixed_args,**fixed_kwargs):
            result=original_fixed(*fixed_args,**fixed_kwargs)
            candidate_sizes=tuple(int(s) for s in fixed_args[2:5])
            candidate_score=criterion(2*n*result.loss,*candidate_sizes,n,p1,p2,p3)
            candidate_record(result,candidate_sizes,candidate_score,'global_fixed_candidate',
                             [np.asarray(fixed_kwargs[f'S{i}_init']).tolist() for i in (1,2,3)])
            return result
        with patch.object(bcd,'tensor_bcd',fixed_wrapper):
            result=original_refine(*args,**kwargs)
        trace['one_se']=selection_audit(trace['candidates'],n,kwargs['one_se_scale'])
        trace['after_refinement']=_result(result,n)
        expected=trace['candidates'][trace['one_se']['selected_index']]
        if expected['supports']!=_support(result):
            raise AssertionError('Trace selection disagrees with unchanged core selection')
        return result

    with patch.object(entry,'tensor_init_glasso',init_wrapper), \
         patch.object(bcd,'group_bess_adaptive',mode_wrapper), \
         patch.object(bcd,'_refine_msic_global_candidates',refine_wrapper):
        result=entry.tensor_abess(X,y,**options)
    return result,trace


def _support_init(result):
    return [getattr(result,f'S{i}_init').tolist() for i in (1,2,3)]


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _hash_files(roots):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def run(formal,source,output):
    output=Path(output).resolve();formal=Path(formal).resolve();source=Path(source).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refuse to overwrite an existing diagnostic output')
    output.mkdir(parents=True,exist_ok=True)
    protected=_hash_files([formal,source,formal.parents[2]/'src/tensor_abess'])
    _write(output/'protected_fingerprints.json',protected)
    start=perf_counter()
    try:
        with np.load(source/'qc/analysis_dataset.npz') as f:
            rawX,rawy,rawC,ids=f['X'],f['y'],f['C'],f['subject_ids']
        index={str(s):i for i,s in enumerate(ids)}
        rows=[];target=None
        for fold_dir in sorted((formal/'cv/folds').iterdir()):
            info=json.loads((fold_dir/'split.json').read_text(encoding='utf-8'))
            train=np.array([index[s] for s in info['train_subject_ids']])
            saved=json.loads((fold_dir/'Tensor-ABESS/complete.json').read_text(encoding='utf-8'))
            parameters=json.loads(saved['fold_metrics']['selected_parameters'])
            entry=importlib.import_module('tensor_abess.tensor.abess')
            allowed=set(inspect.signature(entry.tensor_abess).parameters)-{'X','y'}
            options={k:v for k,v in parameters.items() if k in allowed}
            with np.load(fold_dir/'preprocessing.npz') as pz:
                p=FoldPreprocessor(**{k:float(pz[k]) if k=='y_mean' else pz[k] for k in pz.files})
            X=p.apply(rawX[train],rawC[train]);y=p.transform_response(rawy[train],rawC[train])
            result,trace=traced_abess(X,y,options)
            saved_B=np.load(fold_dir/'Tensor-ABESS/B.npy')
            difference=float(np.max(np.abs(result.B-saved_B)))
            if difference>1e-12:
                raise AssertionError(f'Replay mismatch: {fold_dir.name}, {difference}')
            trace['saved_B_max_difference']=difference
            _write(output/'replays'/(fold_dir.name+'.json'),trace)
            one=trace['one_se'];chosen=trace['candidates'][one['selected_index']]
            rows.append({'repeat':info['repeat'],'outer_fold':info['outer_fold'],
                         'init_sizes':str(trace['initialization']['sizes']),
                         'caps':str(trace['initialization']['caps']),
                         'adaptive_q_before_refinement':trace['before_refinement']['q'],
                         'selected_q':int(np.prod([result.s1,result.s2,result.s3])),
                         'minimum_q':one['q_best'],'minimum_score':one['minimum_score'],
                         'one_se_tolerance':one['tolerance'],'threshold':one['threshold'],
                         'selected_source':chosen['source'],'saved_B_max_difference':difference})
            print('REPLAY',fold_dir.name,rows[-1],flush=True)
            if info['repeat']==2 and info['outer_fold']==5:
                target=(info,train,options,trace,result,X,y)
        _csv(output/'fold_stage_comparison.csv',rows)
        if target is None:
            raise ValueError('Target record repeat=2 outer_fold=5 missing')
        info,train,options,trace,result,X,y=target
        targetC=rawC[train]
        selected_supports=[np.asarray(s) for s in _support(result)]
        audit=ols_audit(build_tensor_design(X,*selected_supports),y,targetC)
        audit['saved_coefficient_rss']=float(np.sum((y-np.einsum('nijk,ijk->n',X,result.B))**2))
        _write(output/'target_ols_audit.json',audit)
        _csv(output/'target_candidate_scores.csv',[{
            'candidate_index':i,'source':r['source'],'s1':r['sizes'][0],'s2':r['sizes'][1],
            's3':r['sizes'][2],'q':r['q'],'rss':r['rss'],'score':r['score'],
            'fit_term':r['fit_term'],'df_penalty':r['df_penalty'],'support_penalty':r['support_penalty'],
            'gap_to_minimum':r['score']-trace['one_se']['minimum_score'],
            'one_se_eligible':i in trace['one_se']['eligible_indices'],
            'selected':i==trace['one_se']['selected_index'],
            'rank':r['ols']['rank'],'condition_number':r['ols']['condition_number'],
            'rss_increasing_steps':int(np.sum(np.diff(r['rss_history'])>1e-8))}
            for i,r in enumerate(trace['candidates'])])
        sizes=[(1,1,1),(3,1,1),(3,2,1),(2,3,2),(3,4,2),(4,4,3),(4,4,5)]
        seeds=[20261214,20261215,20261216]
        internal_splits=make_splits(targetC,5,seeds)
        _write(output/'internal_validation_config.json',{
            'seeds':seeds,'folds':5,'repeats':3,'sizes':[list(s) for s in sizes],
            'outer_train_subject_ids':info['train_subject_ids'],
            'excluded_outer_test_subject_ids':info['test_subject_ids'],
            'solver':'unchanged fixed BCD, max_cycles=6, marginal start; nonmonotonicity logged',
            'policy':'exploratory training-only size comparison; not a new independent formal test'})
        oof=[];fits=[];assign=[]
        bcd=importlib.import_module('tensor_abess.tensor.bcd')
        for split in internal_splits:
            itr,iva=split['train'],split['test']; repeat,fold=split['repeat'],split['outer_fold']
            global_train=train[itr];global_valid=train[iva]
            p=fit_preprocessor(rawX[global_train],rawy[global_train],rawC[global_train])
            tx=p.apply(rawX[global_train],rawC[global_train]);ty=p.transform_response(rawy[global_train],rawC[global_train])
            assign.extend({'subject_id':str(ids[i]),'repeat':repeat,'inner_fold':fold} for i in global_valid)
            for support_size in [None,*sizes]:
                if support_size is None:
                    name='Covariates Only';B=np.zeros(rawX.shape[1:]);history=[];rank=0;cond=None
                else:
                    name='q'+str(int(np.prod(support_size)))
                    initial=bcd._marginal_supports_for_size(tx,ty,support_size)
                    fitted=bcd.tensor_bcd(tx,ty,*support_size,max_cycles=options['max_cycles'],
                        m_max=options['m_max'],k_max=options['k_max'],
                        S1_init=initial[0],S2_init=initial[1],S3_init=initial[2])
                    B=fitted.B;history=(2*len(ty)*np.asarray(fitted.loss_history)).tolist()
                    Z=build_tensor_design(tx,*[np.asarray(s) for s in _support(fitted)])
                    singular=np.linalg.svd(Z,compute_uv=False);rank=int(np.linalg.matrix_rank(Z));cond=float(singular[0]/singular[-1])
                    if rank!=int(np.prod(support_size)):
                        raise ValueError(f'Nonidentifiable internal OLS: {repeat}/{fold}/{name}')
                val_pred=p.predict(B,rawX[global_valid],rawC[global_valid])
                train_pred=p.predict(B,rawX[global_train],rawC[global_train])
                if not np.isfinite(val_pred).all():raise ValueError('Nonfinite internal prediction')
                fits.append({'repeat':repeat,'inner_fold':fold,'model':name,
                             'q':0 if support_size is None else int(np.prod(support_size)),
                             'n_train':len(itr),'n_validation':len(iva),'rank':rank,'condition_number':cond,
                             'train_rmse':float(np.sqrt(np.mean((rawy[global_train]-train_pred)**2))),
                             'validation_rmse':float(np.sqrt(np.mean((rawy[global_valid]-val_pred)**2))),
                             'rss_increasing_steps':int(np.sum(np.diff(history)>1e-8))})
                oof.extend({'subject_id':str(ids[i]),'repeat':repeat,'inner_fold':fold,
                            'model':name,'y_true':float(rawy[i]),'y_pred':float(prediction)}
                           for i,prediction in zip(global_valid,val_pred))
            _csv(output/'internal_fold_metrics.csv',fits)
            _csv(output/'internal_oof_predictions.csv',oof)
            print('INTERNAL_VALIDATION',repeat,fold,'completed',flush=True)
        _csv(output/'internal_splits.csv',assign)
        summary=[];by_repeat=[]
        names=['Covariates Only']+['q'+str(int(np.prod(s))) for s in sizes]
        for name in names:
            errors=[]
            for repeat in range(3):
                selected=[r for r in oof if r['repeat']==repeat and r['model']==name]
                if len(selected)!=len(train) or len({r['subject_id'] for r in selected})!=len(train):
                    raise ValueError('Incomplete internal OOF coverage')
                error=np.array([r['y_true']-r['y_pred'] for r in selected]);ys=np.array([r['y_true'] for r in selected])
                mse=float(np.mean(error**2));errors.append(mse)
                by_repeat.append({'model':name,'repeat':repeat,'n_subjects':len(train),
                                  'mse':mse,'rmse':float(np.sqrt(mse)),
                                  'r2':float(1-np.sum(error**2)/np.sum((ys-ys.mean())**2))})
            rmses=np.sqrt(errors)
            relevant=[r for r in fits if r['model']==name]
            summary.append({'model':name,'q':relevant[0]['q'],'rmse_mean':float(rmses.mean()),
                            'rmse_repeat_sd':float(rmses.std(ddof=1)),
                            'train_rmse_mean':float(np.mean([r['train_rmse'] for r in relevant])),
                            'max_condition_number':max([r['condition_number'] for r in relevant if r['condition_number'] is not None],default=None),
                            'nonmonotone_fits':sum(r['rss_increasing_steps']>0 for r in relevant)})
        _csv(output/'internal_metrics_by_repeat.csv',by_repeat)
        _csv(output/'internal_model_summary.csv',summary)
        changes=[path for path,h in protected.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h]
        if changes:raise AssertionError('Protected formal result or core source changed')
        _write(output/'run_state.json',{'complete':True,'replayed_folds':15,
             'internal_splits':15,'internal_oof_predictions':len(oof),
             'protected_files_verified':len(protected),'protected_files_changed':changes,
             'runtime_sec':perf_counter()-start})
    except Exception as error:
        _write(output/'failure.json',{'message':str(error),'type':type(error).__name__})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formal',default='empirical_analysis/lemon/outputs_eeg_incremental')
    p.add_argument('--source',default='empirical_analysis/lemon/outputs_lemon_adjusted')
    p.add_argument('--output',default='empirical_analysis/lemon/outputs_fold80_diagnostic')
    a=p.parse_args();run(a.formal,a.source,a.output)


if __name__=='__main__':main()
