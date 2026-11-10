"""Task 17: fold-local preprocessing and frozen eight-method fitting.

Indices passed to fit_task and inner_pairs are global row indices. No test
response is accessed. Temporal and exact-duplicate split construction belongs
to protocol.py; membership and disjointness are checked here as well.
"""
from time import perf_counter
import warnings
import numpy as np
from empirical_analysis.spectrofood_apple.models import _abess_options, _tensor_diagnostics


def preprocess(X,y):
    X,y=np.asarray(X,float),np.asarray(y,float)
    if not np.isfinite(X).all() or not np.isfinite(y).all() or y.shape!=(len(X),):
        raise ValueError('Nonfinite or misaligned training data')
    scale=X.std(axis=0); ys=float(y.std())
    if ys==0:
        raise ValueError('Constant training response')
    return dict(x_mean=X.mean(axis=0),x_scale=np.where(scale==0,1.,scale),
                constant_mask=scale==0,y_mean=float(y.mean()),y_scale=ys)


def transform(X,prep):
    X=np.asarray(X,float)
    if X.shape[1:]!=prep['x_mean'].shape or not np.isfinite(X).all():
        raise ValueError('Prediction input invalid')
    return (X-prep['x_mean'])/prep['x_scale']


def parameter_grid(method,X,y,inner_pairs):
    if method in ('M0','M1'):
        return [None]
    if method=='M5':
        limits=[]
        for train,_ in inner_pairs:
            Z=transform(X[train],preprocess(X[train],y[train])).reshape(len(train),-1)
            limits.append(min(len(train)-1,Z.shape[1],np.linalg.matrix_rank(Z)))
        return [dict(n_components=k) for k in (1,2,3,5,8,12,16) if k<=min(limits)]
    key,values={'M2':('lambda_ratio',np.geomspace(1,.01,12)),
                'M3':('lambda_ratio',[1,.5,.2,.1,.05,.02,.01,.005,.002,.001]),
                'M7':('lambda_ratio',[1,.5,.2,.1,.05,.02,.01,.005,.002,.001]),
                'M4':('alpha',np.logspace(-4,4,17)),
                'M6':('alpha',np.logspace(-4,4,17))}[method]
    return [{key:float(v)} for v in values]


def select_parameter(records,method):
    best=min(r['mse'] for r in records)
    eligible=[r for r in records if r['mse']<=best+1e-12*max(1.,abs(best))]
    direction=1 if method=='M5' else -1
    return min(eligible,key=lambda r:direction*next(iter(r['parameter'].values())))['parameter']


def train_model(method,X,y,parameter=None,config=None):
    from sklearn.exceptions import ConvergenceWarning
    start=perf_counter(); prep=preprocess(X,y); Z=transform(X,prep)
    flat=Z.reshape(len(X),-1); target=(y-prep['y_mean'])/prep['y_scale']
    parameter={} if parameter is None else parameter
    artifact=dict(model_id=method,preprocessing=prep,parameters=parameter,intercept_standardized=0.)
    diag=dict(converged=None,convergence_status='unknown',warnings=[],internal_cv=False,fit_count=1,cache_hits=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if method=='M0':
            coef=np.zeros(flat.shape[1]); diag.update(converged=True,convergence_status='analytic')
        elif method=='M1':
            from tensor_abess.tensor.abess import tensor_abess
            from tensor_abess.tensor.bcd import _candidate_size_triples
            from tensor_abess.tensor.sic import log_binomial
            frozen=_abess_options(config)
            result=tensor_abess(Z,target,**frozen)
            artifact.update(B=result.B,parameters=frozen,native_result=result)
            coef=result.B.ravel(order='C')
            diag.update(_tensor_diagnostics(Z,result.B,[result.S1,result.S2,result.S3]))
            ell=float(np.log(max(np.log(len(y)),np.e)))
            parts=dict(log_rss=float(len(y)*np.log(max(result.rss/len(y),np.finfo(float).tiny))),
                       df=1.5*diag['q']*ell,
                       combinatorial=ell*sum(log_binomial(p,len(s)) for p,s in zip(Z.shape[1:],diag['supports'])))
            qmax=int(np.floor(.7*len(y)))
            diag.update(caps=list(result.s_max),q_max=qmax,msic=float(result.sic),rss=float(result.rss),msic_components=parts,
                        candidate_grid_count=len(_candidate_size_triples(*result.s_max,qmax)),
                        iterations=int(result.bcd_result.n_cycles),loss_history=list(result.loss_history),
                        sic_history=list(result.sic_history),convergence_status='not_exposed_by_formal_solver')
        elif method=='M2':
            from empirical_analysis.lemon.incremental_value.model_adapters import fit_eeg_model
            result=fit_eeg_model('Tensor-GLasso',Z,target,parameter,dict(max_iter=500,tol=1e-6,max_cycles=6))
            artifact.update(B=result.B,native_result=result)
            coef=result.B.ravel(order='C')
            diag.update(_tensor_diagnostics(Z,result.B,result.supports)); diag.update(result.parameters)
            diag.update(convergence_status='group_ISTA_stopping_not_exposed_by_original_adapter',rectangular_ols=True)
        elif method in ('M3','M7'):
            from sklearn.linear_model import Lasso
            alpha_max=float(np.max(np.abs(flat.T@target))/len(y)); alpha=alpha_max*parameter['lambda_ratio']
            if alpha<=0:
                raise ValueError('Zero Lasso alpha_max; blocked without solver fallback')
            attempts=[]
            for budget in (10000,50000):
                with warnings.catch_warnings(record=True) as attempt_warnings:
                    warnings.simplefilter('always')
                    estimator=Lasso(alpha=alpha,fit_intercept=False,max_iter=budget,tol=1e-6,selection='cyclic').fit(flat,target)
                converged=not any(issubclass(w.category,ConvergenceWarning) for w in attempt_warnings)
                attempts.append(dict(max_iter=budget,iterations=int(estimator.n_iter_),dual_gap=float(estimator.dual_gap_),
                                     warnings=[str(w.message) for w in attempt_warnings],converged=converged))
                if converged:
                    break
            artifact['estimator']=estimator; coef=estimator.coef_.ravel()
            diag.update(alpha_max=alpha_max,alpha_actual=alpha,attempts=attempts,converged=converged,
                        convergence_status='converged' if converged else 'not_converged',fit_count=len(attempts),
                        dual_gap=float(estimator.dual_gap_),iterations=int(estimator.n_iter_),
                        dual_gap_threshold=float(1e-6*(target@target)/len(y)))
            diag['warnings'].extend(w for a in attempts for w in a['warnings'])
        elif method in ('M4','M6'):
            from sklearn.linear_model import Ridge
            estimator=Ridge(alpha=parameter['alpha'],fit_intercept=False,solver='svd').fit(flat,target)
            artifact['estimator']=estimator;coef=estimator.coef_.ravel()
            diag.update(converged=True,convergence_status='direct_svd')
        elif method=='M5':
            from sklearn.cross_decomposition import PLSRegression
            estimator=PLSRegression(n_components=parameter['n_components'],scale=False,max_iter=500,tol=1e-6).fit(flat,target)
            artifact['estimator']=estimator;coef=estimator.coef_.ravel()
            # sklearn internally centers even with scale=False; preserve that intercept exactly.
            artifact['intercept_standardized']=float(np.asarray(estimator.predict(np.zeros((1,flat.shape[1])))).ravel()[0])
            diag.update(iterations=list(estimator.n_iter_),converged=all(i<500 for i in estimator.n_iter_),convergence_status='nipals_iteration_check')
        else:
            raise ValueError('Unknown model')
        diag['warnings'].extend(str(w.message) for w in caught)
    if not np.isfinite(coef).all():
        raise ValueError('Nonfinite coefficient')
    artifact['coef_standardized']=coef
    diag.update(fit_seconds=perf_counter()-start,training_rank=int(np.linalg.matrix_rank(flat)))
    artifact['diagnostics']=diag
    return artifact


def predict_model(artifact,X96tensor,raw128):
    selected=raw128 if artifact['model_id'] in ('M6','M7') else X96tensor
    Z=transform(selected,artifact['preprocessing']).reshape(len(selected),-1)
    pred=Z@artifact['coef_standardized']+artifact['intercept_standardized']
    pred=pred*artifact['preprocessing']['y_scale']+artifact['preprocessing']['y_mean']
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction')
    return pred


def fit_task(X96tensor,raw128,y,train_indices,test_indices,inner_pairs,method_id,config=None):
    started=perf_counter()
    X96tensor,raw128,y=np.asarray(X96tensor),np.asarray(raw128),np.asarray(y)
    train,test=np.asarray(train_indices,int),np.asarray(test_indices,int)
    if X96tensor.shape!=(len(y),16,3,2) or raw128.shape!=(len(y),128):
        raise ValueError('Expected aligned 16x3x2 and 128 representations')
    if len(np.intersect1d(train,test)) or not len(train) or not len(test):
        raise ValueError('Invalid outer split')
    pairs=[(np.asarray(a,int),np.asarray(b,int)) for a,b in inner_pairs]
    if len(pairs)<2:
        raise ValueError('At least two valid inner folds required')
    for a,b in pairs:
        if not len(a) or not len(b) or len(np.intersect1d(a,b)) or not np.isin(np.r_[a,b],train).all():
            raise ValueError('Invalid inner split membership')
    X=raw128 if method_id in ('M6','M7') else X96tensor
    records=[]
    if method_id not in ('M0','M1'):
        candidates=parameter_grid(method_id,X,y,pairs)
        if not candidates:
            raise ValueError('No common legal candidate')
        for param in candidates:
            folds=[]
            for i,(a,b) in enumerate(pairs):
                artifact=train_model(method_id,X[a],y[a],param,config)
                predictions=predict_model(artifact,X96tensor[b],raw128[b])
                residual=predictions-y[b]
                folds.append(dict(fold=i,train_indices=a.tolist(),validation_indices=b.tolist(),
                                  n_validation=len(b),sse=float(residual@residual),diagnostics=artifact['diagnostics'],
                                  preprocessing=artifact['preprocessing'],predictions=predictions,
                                  coef_standardized=artifact['coef_standardized'],
                                  intercept_standardized=artifact['intercept_standardized']))
            records.append(dict(parameter=param,mse=sum(f['sse'] for f in folds)/sum(f['n_validation'] for f in folds),folds=folds))
    chosen=select_parameter(records,method_id) if records else None
    artifact=train_model(method_id,X[train],y[train],chosen,config)
    artifact.update(cv_records=records,train_indices=train,test_indices=test,inner_pairs=pairs)
    prediction_start=perf_counter(); pred=predict_model(artifact,X96tensor[test],raw128[test]); prediction_seconds=perf_counter()-prediction_start
    train_pred=predict_model(artifact,X96tensor[train],raw128[train])
    diagnostics=dict(artifact['diagnostics'],total_seconds=perf_counter()-started,prediction_seconds=prediction_seconds,
                     total_fit_count=artifact['diagnostics']['fit_count']+sum(f['diagnostics']['fit_count'] for r in records for f in r['folds']),
                     inner_fit_count=sum(f['diagnostics']['fit_count'] for r in records for f in r['folds']),
                     all_fits_converged=all(f['diagnostics']['converged'] is True for r in records for f in r['folds']) and artifact['diagnostics']['converged'] is True)
    return dict(artifact=artifact,predictions=pred,train_predictions=train_pred,coef_standardized=artifact['coef_standardized'],cv_records=records,diagnostics=diagnostics)
