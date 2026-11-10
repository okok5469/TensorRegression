"""Fixed calendar expanding windows for genuine next-day, updated-information forecasts."""
from __future__ import annotations
import numpy as np


def shift_months(start,count):
    year,month,_=map(int,start.split('-'))
    index=year*12+month-1+count
    return f'{index//12:04d}-{index%12+1:02d}-01'


def make_splits(data):
    dates=np.asarray(data['dates']).astype('datetime64[D]')
    previous=np.asarray(data['input_dates']).astype('datetime64[D]')
    if not np.all(previous==dates-np.timedelta64(1,'D')):
        raise ValueError('Input must come exclusively from the previous calendar day')
    if len(dates)!=len(np.unique(dates)) or not np.all(dates[1:]>dates[:-1]):
        raise ValueError('Dates must be unique and strictly increasing')
    policies=[('development',1,12,'2014-03-01','2014-09-01'),
              ('development',2,18,'2014-09-01','2015-03-01'),
              ('development',3,24,'2015-03-01','2015-09-01'),
              ('development',4,30,'2015-09-01','2016-03-01'),
              ('future',1,36,'2016-03-01','2017-03-01')]
    result=[]
    for protocol,fold,months,start,end in policies:
        train=np.flatnonzero((dates>=np.datetime64('2013-03-01'))&(dates<np.datetime64(start)))
        test=np.flatnonzero((dates>=np.datetime64(start))&(dates<np.datetime64(end)))
        if len(train)<60 or not len(test):
            raise ValueError('Insufficient valid dates for fixed calendar evaluation')
        if dates[train].max()>=dates[test].min():
            raise ValueError('Training target labels unavailable at forecast origin')
        inner=[]
        initial=months//2;step=(months-initial)//3
        for k in range(3):
            valid_start=shift_months('2013-03-01',initial+k*step)
            valid_end=shift_months('2013-03-01',initial+(k+1)*step)
            a=np.flatnonzero(dates[train]<np.datetime64(valid_start))
            b=np.flatnonzero((dates[train]>=np.datetime64(valid_start))&(dates[train]<np.datetime64(valid_end)))
            if len(a)<60 or not len(b) or dates[train[a]].max()>=dates[train[b]].min():
                raise ValueError('Insufficient history or unavailable labels in temporal inner fold')
            inner.append(dict(train=a.tolist(),validation=b.tolist(),
                train_dates=dates[train[a]].astype(str).tolist(),validation_dates=dates[train[b]].astype(str).tolist(),
                validation_start=valid_start,validation_end_exclusive=valid_end))
        result.append(dict(protocol=protocol,fold=fold,train_start='2013-03-01',
            train_end_exclusive=start,test_start=start,test_end_exclusive=end,
            train=train.tolist(),test=test.tolist(),train_dates=dates[train].astype(str).tolist(),
            test_dates=dates[test].astype(str).tolist(),inner_seed=260000+fold,
            inner=inner))
    return result


def check_training_missingness(data,splits):
    """No imputer can invent a median for an entirely unobserved training coordinate."""
    X=np.asarray(data['X'],float)
    if np.isinf(X).any():
        raise ValueError('Infinite historical inputs are invalid')
    rows=[]
    for split in splits:
        train=np.asarray(split['train'])
        periods=[('outer',train)]+[(f'inner{k+1}',train[np.asarray(v['train'])]) for k,v in enumerate(split['inner'])]
        for name,index in periods:
            missing=np.isnan(X[index]).all(axis=0)
            if missing.any():
                raise ValueError(f'All-missing training coordinate: {split["protocol"]} fold{split["fold"]} {name}')
            rows.append(dict(protocol=split['protocol'],fold=split['fold'],scope=name,n_train=len(index),
                             missing_fraction=float(np.isnan(X[index]).mean()),all_missing_coordinates=0))
    return rows
