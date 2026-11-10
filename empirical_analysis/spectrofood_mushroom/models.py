"""Explicit reuse of frozen Apple and native-spectrum screening estimators."""
from __future__ import annotations
import numpy as np
from empirical_analysis.spectrofood_apple import models as apple
from empirical_analysis import mushroom_screen as screen


def select_input(data,model):
    if model in ('M6','M7'):
        return data['X_native']
    if model=='M5':
        return data['X_tensor'].mean(axis=(1,2))
    return data['X_tensor']


def train_model(model_id,X,y_pct,param=None,seed=0,config=None):
    if model_id not in ('M6','M7'):
        return apple.train_model(model_id,X,y_pct,param,seed,config)
    name={'M6':'Ridge','M7':'PLS'}[model_id]
    parameter=param if param is not None else (1. if model_id=='M6' else 2)
    native=screen.fit_model(name,X,y_pct,parameter)
    prep={key:native[key] for key in ('x_mean','x_scale','constant_mask','y_mean','y_scale')}
    return dict(model_id=model_id,backend='screen',native_artifact=native,preprocessing=prep,
                parameters=parameter,diagnostics={key:native[key] for key in ('iterations','converged','warnings')})


def predict_model(artifact,X):
    if artifact.get('backend')=='screen':
        result=screen.predict(artifact['native_artifact'],X)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite screen prediction')
        return result
    return apple.predict_model(artifact,X)


def tune_model(model_id,X,y_pct,inner_splits,seed=0,config=None):
    if model_id not in ('M6','M7'):
        return apple.tune_model(model_id,X,y_pct,inner_splits,seed,config)
    config={} if config is None else config
    X,y=apple._validate(X,y_pct)
    if model_id=='M6':
        grid=config.get('ridge_alphas',np.logspace(-4,4,17).tolist())
    else:
        limits=[]
        for train,_ in inner_splits:
            prep=apple._preprocess(X[train],y[train])
            Z=apple._transform(X[train],prep)
            limits.append(min(len(train)-1,X.shape[1],np.linalg.matrix_rank(Z)))
        grid=[v for v in config.get('pls_components',[1,2,3,5,8,12,16]) if v<=min(limits)]
    if not len(grid):
        raise ValueError('Empty screening grid')
    return screen.tune({'M6':'Ridge','M7':'PLS'}[model_id],X,y,inner_splits,grid)
