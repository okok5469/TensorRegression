"""Training-local preprocessing and frozen model adapters for task card 13.

Lasso: RSS/(2n) + alpha*||beta||_1, alpha=training alpha_max*ratio.
Ridge (sklearn): RSS + alpha*||beta||_2^2. Tensor-GLasso reuses the
verified LEMON external-tuning adapter, including its original grouped grid
normalization and rectangular OLS; it never runs internal cross-validation.
"""
from __future__ import annotations

from time import perf_counter
import warnings
import numpy as np


def _validate(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim not in (2, 4) or not len(X) or not np.isfinite(X).all():
        raise ValueError('Expected a finite sample-by-feature matrix or sample-by-3D tensor')
    if y is None:
        return X
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError('Expected a finite response vector aligned with X')
    return X, y


def _preprocess(X, y):
    scale = X.std(axis=0, ddof=0)
    constant = scale <= 1e-12*max(1.,float(scale.max()))
    ys = float(y.std(ddof=0))
    if ys <= 1e-12*max(1.,float(np.max(np.abs(y)))):
        raise ValueError('Training response is degenerate (approximately constant)')
    return {'x_mean':X.mean(axis=0), 'x_scale':np.where(constant,1.,scale),
            'constant_mask':constant,'y_mean':float(y.mean()),'y_scale':ys}


def _transform(X, prep):
    X = _validate(X)
    if X.shape[1:] != prep['x_mean'].shape:
        raise ValueError('Prediction input shape differs from training representation')
    return np.where(prep['constant_mask'],0.,(X-prep['x_mean'])/prep['x_scale'])


def _options(config):
    config = {} if config is None else config
    return {k: config.get(k,v) for k,v in [('max_iter',500),('tol',1e-6),('max_cycles',6)]}


def _abess_options(config):
    config = {} if config is None else config
    # These are the frozen official adaptive defaults, never tuned by inner CV.
    frozen = dict(target_sparsity=None,c=3.,criterion='bic',k_max=1,m_max=15,
                  eps=1e-8,adaptive_criterion='msic',msic_refine_alpha=1.5,
                  msic_refine_gamma=1.,msic_refine_one_se_scale=1.,
                  n_lambdas=20,cv_folds=5,max_iter=500,tol=1e-6,max_cycles=6)
    supplied = config.get('abess',{})
    if any(k not in frozen or frozen[k] != v for k,v in supplied.items()):
        raise ValueError('M1 settings are frozen to official v5.2 original MSIC')
    return frozen


def _tensor_diagnostics(X, B, supports):
    supports = [np.asarray(s,dtype=int) for s in supports]
    q = int(np.prod([len(s) for s in supports]))
    design = X[np.ix_(np.arange(len(X)),*supports)].reshape(len(X),q,order='C')
    singular = np.linalg.svd(design,compute_uv=False)
    condition = None if not len(singular) or singular[-1] == 0 else float(singular[0]/singular[-1])
    return dict(supports=[s.tolist() for s in supports],q=q,
                ols_rank=int(np.linalg.matrix_rank(design)),ols_condition=condition)


def train_model(model_id, X, y_pct, param=None, seed=0, config=None):
    """Fit one current training set; artifact is self-contained and pickleable."""
    X, y = _validate(X,y_pct)
    if model_id not in ('M0','M1','M2','M3','M4','M5','M6'):
        raise ValueError(f'Unknown model {model_id}')
    if model_id in ('M1','M2','M3','M4') and X.ndim != 4:
        raise ValueError('Tensor and vectorized tensor methods require sample-by-3D tensor')
    if model_id in ('M5','M6') and X.ndim != 2:
        raise ValueError('PLS requires a spectrum matrix')
    started = perf_counter()
    prep = _preprocess(X,y)
    Z = _transform(X,prep)
    ys = (y-prep['y_mean'])/prep['y_scale']
    flat = Z.reshape(len(X),-1,order='C')
    options = _options(config)
    defaults = {'M0':{},'M1':{},'M2':{'lambda_ratio':.1},'M3':{'lambda_ratio':.01},
                'M4':{'alpha':1.},'M5':{'n_components':2},'M6':{'n_components':2}}
    parameter = dict(defaults[model_id] if param is None else param)
    artifact = {'model_id':model_id,'preprocessing':prep,'parameters':parameter,'seed':int(seed)}
    diagnostic = {'converged':None,'iterations':None,'warnings':[],'internal_cv':False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model_id == 'M0':
            artifact['estimator'] = None
            diagnostic.update(converged=True,iterations=0)
        elif model_id == 'M1':
            from tensor_abess.tensor.abess import tensor_abess
            from tensor_abess.tensor.bcd import _candidate_size_triples
            from tensor_abess.tensor.sic import log_binomial
            if parameter:
                raise ValueError('M1 does not accept tuned parameters')
            frozen = _abess_options(config)
            result = tensor_abess(Z,ys,**frozen)
            artifact.update(B=np.asarray(result.B),parameters=frozen)
            diagnostic.update(_tensor_diagnostics(Z,result.B,[result.S1,result.S2,result.S3]))
            q_max = int(np.floor(.7*len(X)))
            loglog = float(np.log(max(np.log(len(X)),np.e)))
            components = dict(log_rss=float(len(X)*np.log(max(result.rss/len(X),np.finfo(float).tiny))),
                              df=float(1.5*diagnostic['q']*loglog),
                              combinatorial=float(loglog*sum(log_binomial(p,len(s)) for p,s in zip(Z.shape[1:],diagnostic['supports']))))
            diagnostic.update(caps=list(result.s_max),q_max=q_max,
                              candidate_count=1+len(_candidate_size_triples(*result.s_max,q_max)),
                              candidate_grid_count=len(_candidate_size_triples(*result.s_max,q_max)),
                              msic=float(result.sic),msic_components=components,
                              iterations=int(result.bcd_result.n_cycles),
                              convergence_status='not_exposed_by_formal_solver',
                              loss_history=list(result.loss_history),sic_history=list(result.sic_history))
        elif model_id == 'M2':
            from empirical_analysis.lemon.incremental_value.model_adapters import fit_eeg_model
            if set(parameter) != {'lambda_ratio'} or not 0 < parameter['lambda_ratio'] <= 1:
                raise ValueError('M2 requires lambda_ratio in (0, 1]')
            result = fit_eeg_model('Tensor-GLasso',Z,ys,parameter,options)
            artifact['B'] = result.B
            diagnostic.update(_tensor_diagnostics(Z,result.B,result.supports))
            diagnostic.update(result.parameters)
            diagnostic.update(iterations=result.parameters['n_cycles'],
                              convergence_status='group_ISTA_stopping_not_exposed_by_original_adapter')
        elif model_id == 'M3':
            from sklearn.linear_model import Lasso
            if set(parameter) != {'lambda_ratio'} or not 0 < parameter['lambda_ratio'] <= 1:
                raise ValueError('M3 requires lambda_ratio in (0, 1]')
            maximum = float(np.max(np.abs(flat.T@ys/len(X))))
            alpha = maximum*float(parameter['lambda_ratio'])
            if alpha <= 0:
                raise ValueError('Lasso alpha_max is zero; no nonzero predictive training correlation')
            estimator = Lasso(alpha=alpha,fit_intercept=False,max_iter=100000,tol=1e-6,
                              selection='cyclic',random_state=seed).fit(flat,ys)
            artifact['estimator'] = estimator
            diagnostic.update(alpha_max=maximum,alpha_actual=alpha,iterations=int(estimator.n_iter_),
                              dual_gap=float(estimator.dual_gap_),converged=bool(estimator.n_iter_<100000),
                              max_iter=100000,tol=1e-6,
                              dual_gap_threshold=float(1e-6*(ys@ys)/len(ys)))
        elif model_id == 'M4':
            from sklearn.linear_model import Ridge
            if set(parameter) != {'alpha'} or parameter['alpha'] <= 0:
                raise ValueError('M4 requires positive alpha')
            artifact['estimator'] = Ridge(alpha=float(parameter['alpha']),fit_intercept=False,
                                           solver='svd').fit(flat,ys)
            diagnostic.update(converged=True,iterations=1)
        else:
            from sklearn.cross_decomposition import PLSRegression
            if set(parameter) != {'n_components'}:
                raise ValueError('PLS requires n_components')
            components = int(parameter['n_components'])
            if components != parameter['n_components'] or not 1 <= components <= min(len(X)-1,flat.shape[1],np.linalg.matrix_rank(flat)):
                raise ValueError('PLS component count exceeds training rank')
            estimator = PLSRegression(n_components=components,scale=False,max_iter=500,tol=1e-6).fit(flat,ys)
            artifact['estimator'] = estimator
            diagnostic.update(iterations=list(estimator.n_iter_),converged=all(i<500 for i in estimator.n_iter_))
        diagnostic['warnings'] = [str(w.message) for w in caught]
    if 'B' in artifact and (artifact['B'].shape != Z.shape[1:] or not np.isfinite(artifact['B']).all()):
        raise ValueError('Tensor model returned invalid coefficients')
    diagnostic['fit_seconds'] = float(perf_counter()-started)
    artifact['diagnostics'] = diagnostic
    return artifact


def predict_model(artifact, X):
    Z = _transform(X,artifact['preprocessing'])
    if artifact['model_id'] == 'M0':
        pred = np.zeros(len(Z))
    elif 'B' in artifact:
        pred = np.einsum('nijk,ijk->n',Z,artifact['B'])
    else:
        pred = np.asarray(artifact['estimator'].predict(Z.reshape(len(Z),-1,order='C'))).reshape(-1)
    pred = pred*artifact['preprocessing']['y_scale']+artifact['preprocessing']['y_mean']
    if not np.isfinite(pred).all():
        raise ValueError('Model prediction is not finite')
    return pred


def parameter_grid(model_id,X,y,inner_splits,config=None):
    """Freeze comparable candidates before inspecting any validation error."""
    config = {} if config is None else config
    if model_id in ('M0','M1'):
        return [None]
    if model_id in ('M5','M6'):
        limits = []
        for train,_ in inner_splits:
            prep = _preprocess(X[train],y[train])
            flat = _transform(X[train],prep).reshape(len(train),-1,order='C')
            limits.append(min(len(train)-1,flat.shape[1],np.linalg.matrix_rank(flat)))
        candidates = [{'n_components':int(v)} for v in config.get('pls_components',[1,2,3,5,8,12,16]) if 1 <= v <= min(limits)]
    else:
        key,parameter,default = {'M2':('glasso_ratios','lambda_ratio',np.geomspace(1,.01,12)),
                                 'M3':('lasso_ratios','lambda_ratio',np.logspace(0,-3,13)),
                                 'M4':('ridge_alphas','alpha',np.logspace(-4,4,17))}[model_id]
        values = np.asarray(config.get(key,default),dtype=float)
        if not values.ndim == 1 or not len(values) or not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('Parameter grid must be nonempty, finite and positive')
        if model_id == 'M2' and len(values)>12:
            raise ValueError('Tensor-GLasso grid may contain at most 12 frozen strengths')
        candidates = [{parameter:float(v)} for v in values]
    if not candidates:
        raise ValueError('No valid candidates for the training folds')
    return candidates


def tune_model(model_id,X,y_pct,inner_splits,seed=0,config=None):
    X,y = _validate(X,y_pct)
    splits = [(np.asarray(t,dtype=int),np.asarray(v,dtype=int)) for t,v in inner_splits]
    if not splits:
        raise ValueError('Inner splits are required')
    for train,validation in splits:
        if not len(train) or not len(validation) or len(np.intersect1d(train,validation)):
            raise ValueError('Invalid or overlapping inner split')
    parameters = parameter_grid(model_id,X,y,splits,config)
    if model_id in ('M0','M1'):
        return None,[]
    records = []
    for parameter in parameters:
        errors,total,folds = 0.,0,[]
        for fold,(train,validation) in enumerate(splits):
            fit = train_model(model_id,X[train],y[train],parameter,seed+fold,config)
            residual = predict_model(fit,X[validation])-y[validation]
            error = float(residual@residual)
            errors += error
            total += len(validation)
            folds.append({'fold':fold,'n_validation':len(validation),'sse_pct2':error,
                          'diagnostics':fit['diagnostics']})
        records.append({'parameter':parameter,'mse_pct2':errors/total,'folds':folds})
    # Stable first-in-frozen-grid tie breaking, with no tolerance retuning.
    winner = min(range(len(records)),key=lambda i:records[i]['mse_pct2'])
    return parameters[winner],records


def fit_model(model_id,X_train,y_train_pct,X_test,inner_splits,seed=0,config=None):
    parameter,records = tune_model(model_id,X_train,y_train_pct,inner_splits,seed,config)
    artifact = train_model(model_id,X_train,y_train_pct,parameter,seed,config)
    artifact['cv_records'] = records
    return predict_model(artifact,X_test),artifact
