"""Training-local median imputation around unchanged released estimators."""
from __future__ import annotations
import numpy as np
from empirical_analysis.spectrofood_apple import models as apple


def _input(model_id,X,config):
    X=np.asarray(X,float)
    if X.ndim!=4 or np.isinf(X).any():
        raise ValueError('Expected tensor with finite values or NaNs')
    if model_id=='M2':
        index=int((config or {}).get('target_station_index',0))
        return X[:,index:index+1]
    return X


def _median(X):
    if np.isnan(X).all(axis=0).any():
        raise ValueError('Current training tensor contains an all-missing coordinate')
    return np.nanmedian(X,axis=0)


def train_model(model_id,X,y,param=None,seed=0,config=None):
    y=np.asarray(y,float)
    if y.ndim!=1 or len(y)!=len(X) or not np.isfinite(y).all():
        raise ValueError('Invalid training response')
    if model_id in ('M0','M1'):
        return dict(model_id=model_id,y_mean=float(y.mean()),diagnostics=dict(converged=True,iterations=0))
    mapping={'M2':'M4','M3':'M4','M4':'M3','M5':'M2','M6':'M1'}
    selected=_input(model_id,X,config)
    median=_median(selected)
    filled=np.where(np.isnan(selected),median,selected)
    native=apple.train_model(mapping[model_id],filled,y,param,seed,config)
    return dict(model_id=model_id,native_artifact=native,imputer=median,
                target_station_index=int((config or {}).get('target_station_index',0)),
                preprocessing=native['preprocessing'],parameters=native['parameters'],diagnostics=native['diagnostics'])


def predict_model(artifact,X,persistence=None):
    model=artifact['model_id']
    if model=='M0':
        return np.full(len(X),artifact['y_mean'])
    if model=='M1':
        values=np.asarray(persistence,float)
        if values.shape!=(len(X),) or not np.isfinite(values).all():
            raise ValueError('Persistence requires the original finite previous-day PM2.5 mean')
        return values.copy()
    selected=_input(model,X,artifact)
    filled=np.where(np.isnan(selected),artifact['imputer'],selected)
    return apple.predict_model(artifact['native_artifact'],filled)


def tune_model(model_id,X,y,inner_splits,seed=0,config=None):
    if model_id in ('M0','M1','M6'):
        return None,[]
    config={} if config is None else config
    key,param,default={
        'M2':('ridge_alphas','alpha',np.logspace(-4,4,17)),
        'M3':('ridge_alphas','alpha',np.logspace(-4,4,17)),
        'M4':('lasso_ratios','lambda_ratio',np.logspace(0,-3,13)),
        'M5':('glasso_ratios','lambda_ratio',np.geomspace(1,.01,12))}[model_id]
    values=np.asarray(config.get(key,default),float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Invalid frozen parameter grid')
    X=np.asarray(X,float);y=np.asarray(y,float);records=[]
    for value in values:
        parameter={param:float(value)};sse=0.;count=0;folds=[]
        for i,(train,valid) in enumerate(inner_splits):
            a,b=np.asarray(train,int),np.asarray(valid,int)
            if not len(a) or not len(b) or max(a)>=min(b):
                raise ValueError('Inner indices must follow chronological expanding order')
            fit=train_model(model_id,X[a],y[a],parameter,seed+i,config)
            error=predict_model(fit,X[b])-y[b]
            sse+=float(error@error);count+=len(b)
            folds.append(dict(validation_n=len(b),diagnostics=fit['diagnostics']))
        if not count:
            raise ValueError('No chronological inner folds')
        records.append(dict(parameter=parameter,mse=sse/count,folds=folds))
    return min(records,key=lambda row:row['mse'])['parameter'],records
