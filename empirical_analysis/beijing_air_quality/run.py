"""Frozen chronological single-task execution and independent artifact replay."""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import pickle
from pathlib import Path
import time
import traceback
import numpy as np
from empirical_analysis.spectrofood_apple.run import fingerprint,file_hash,write_json,read_json,write_csv,runtime_identity
from .models import train_model,predict_model,tune_model


def metrics(y,pred):
    residual=np.asarray(pred)-np.asarray(y)
    denominator=float(np.sum((y-np.mean(y))**2))
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),mae=float(np.mean(abs(residual))),
                r2=None if denominator==0 else float(1-np.sum(residual**2)/denominator),bias=float(np.mean(residual)))


def source_identity(config):
    if 'source_files' in config:
        # Explicit dependency lists exist only for isolated synthetic fixtures.
        if config.get('synthetic_fixture') is not True:
            raise ValueError('Production source_files override is forbidden')
        return {str(p):file_hash(p) for p in config['source_files']}
    from empirical_analysis.spectrofood_apple.run import source_identity as original
    result=original(config)
    for path in Path('empirical_analysis/beijing_air_quality').glob('*.py'):
        result[path.as_posix()]=file_hash(path)
    module=importlib.import_module('tensor_abess.tensor.abess')
    expected=Path('src/tensor_abess/tensor/abess.py').resolve()
    if Path(module.__file__).resolve()!=expected:
        raise ValueError('Formal core imported from unexpected path')
    return result


def frozen_contract(config):
    contract=read_json(Path(config.get('contract_path',str(Path(config['output_dir'])/'contract.json'))))
    body=dict(config=config,source=source_identity(config),data_sha256=file_hash(config['data_path']),splits=contract['splits'])
    if fingerprint(body)!=contract['fingerprint']:
        raise ValueError('Frozen data/config/source/split contract changed')
    return contract


def _temporal(data,split):
    train,test=np.asarray(split['train'],int),np.asarray(split['test'],int)
    dates=np.asarray(data['dates']).astype('datetime64[D]')
    def chronological(a,b):
        if not len(a) or not len(b) or len(set(a))!=len(a) or len(set(b))!=len(b) or np.any(a<0) or np.any(b<0) or np.any(a>=len(dates)) or np.any(b>=len(dates)):
            raise ValueError('Invalid temporal indices')
        if np.any(dates[a][1:]<=dates[a][:-1]) or np.any(dates[b][1:]<=dates[b][:-1]) or dates[a].max()>=dates[b].min():
            raise ValueError('Future labels cross chronological validation boundary')
    chronological(train,test)
    if 'input_dates' in data:
        inputs=np.asarray(data['input_dates']).astype('datetime64[D]')
        if np.any(inputs!=dates-np.timedelta64(1,'D')):
            raise ValueError('Forecast input date is not exactly the previous day')
    for inner in split['inner']:
        a,b=np.asarray(inner['train'],int),np.asarray(inner['validation'],int)
        if np.any(a<0) or np.any(b<0) or np.any(a>=len(train)) or np.any(b>=len(train)):
            raise ValueError('Invalid local inner temporal index')
        chronological(train[a],train[b])
    return train,test


def task_info(config,index):
    contract=frozen_contract(config)
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(contract['splits'])*len(config['models']):
        raise ValueError('Invalid task index')
    split=contract['splits'][index//len(config['models'])];model=config['models'][index%len(config['models'])]
    directory=Path(config['output_dir'])/'tasks'/f"{split['protocol']}_f{split['fold']}_{model}"
    with np.load(config['data_path'],allow_pickle=False) as data:
        _temporal(data,split)
    return contract,split,model,directory


def verify_task(config,index):
    contract,split,model,directory=task_info(config,index)
    marker=read_json(directory/'COMPLETE.json')
    key=fingerprint(dict(contract=contract['fingerprint'],split=split,model=model,environment=marker['environment']))
    if marker['contract']!=contract['fingerprint'] or marker['cache_key']!=key:
        raise ValueError('Completed task identity mismatch')
    if set(marker['artifacts'])!={'model.pkl','predictions.csv','diagnostics.json','cv_records.json'}:
        raise ValueError('Incomplete saved artifact identity')
    for name,digest in marker['artifacts'].items():
        if file_hash(directory/name)!=digest:
            raise ValueError('Corrupt artifact hash: '+name)
    with (directory/'model.pkl').open('rb') as stream:
        artifact=pickle.load(stream)
    with np.load(config['data_path'],allow_pickle=False) as data:
        train,test=_temporal(data,split)
        X,y,persistence=data['X'],data['y'],data['persistence']
        pred=predict_model(artifact,X[test],persistence[test])
        if 'imputer' in artifact:
            from .models import _input,_median
            selected=_input(model,X[train],artifact)
            median=_median(selected)
            if not np.allclose(median,artifact['imputer'],rtol=1e-12,atol=1e-12):
                raise ValueError('Training imputer mismatch')
            filled=np.where(np.isnan(selected),median,selected)
            prep=artifact['preprocessing']
            if not np.allclose(prep['x_mean'],filled.mean(0),rtol=1e-12,atol=1e-12) or not np.allclose([prep['y_mean'],prep['y_scale']],[y[train].mean(),y[train].std()],rtol=1e-12,atol=1e-12):
                raise ValueError('Training standardization mismatch')
        with (directory/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        if [r['date'] for r in rows]!=data['dates'][test].tolist():
            raise ValueError('Saved held-out dates mismatch')
        saved=np.array([float(r['yhat']) for r in rows])
        if not np.allclose(saved,pred,rtol=1e-9,atol=1e-10):
            raise ValueError('Independent prediction replay mismatch')
        if not np.allclose([float(r['y']) for r in rows],y[test],rtol=0,atol=1e-12):
            raise ValueError('Saved response mismatch')
        for r,i in zip(rows,test):
            previous=str(np.datetime64(data['dates'][i])-np.timedelta64(1,'D'))
            if r['model']!=model or r['protocol']!=split['protocol'] or int(r['fold'])!=split['fold'] or r['input_date']!=previous or r['forecast_origin']!=r['date']+'T00:00:00' or float(r['persist'])!=float(persistence[i]):
                raise ValueError('Saved forecast identity mismatch')
    return dict(prediction_count=len(rows),max_abs_replay_error=float(np.max(abs(saved-pred))))


def run_one(config,task_index):
    started=time.perf_counter();contract,split,model,directory=task_info(config,task_index)
    environment=runtime_identity()
    key=fingerprint(dict(contract=contract['fingerprint'],split=split,model=model,environment=environment))
    marker=directory/'COMPLETE.json'
    if marker.exists():
        if read_json(marker)['cache_key']!=key:
            raise ValueError('Completed task native environment mismatch')
        verify_task(config,task_index);return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete task requires explicit failure review')
    directory.mkdir(parents=True,exist_ok=True)
    try:
        with np.load(config['data_path'],allow_pickle=False) as data:
            train,test=_temporal(data,split);X,y,persistence=data['X'],data['y'],data['persistence']
            options=dict(config.get('model_config',{}))
            station_index=list(data['stations']).index('Aotizhongxin')
            if 'target_station_index' in config and config['target_station_index']!=station_index:
                raise ValueError('Frozen target station index mismatch')
            options['target_station_index']=station_index
            inner=[(np.asarray(s['train'],int),np.asarray(s['validation'],int)) for s in split['inner']]
            parameter,records=tune_model(model,X[train],y[train],inner,0,options)
            artifact=train_model(model,X[train],y[train],parameter,0,options)
            pred=predict_model(artifact,X[test],persistence[test]);training=predict_model(artifact,X[train],persistence[train])
            rows=[]
            for i,value in zip(test,pred):
                date=str(data['dates'][i]);residual=float(value-y[i])
                rows.append(dict(date=date,input_date=str(np.datetime64(date)-np.timedelta64(1,'D')),forecast_origin=date+'T00:00:00',
                    model=model,protocol=split['protocol'],fold=split['fold'],y=float(y[i]),yhat=float(value),persist=float(persistence[i]),
                    residual=residual,squared_error=residual**2,model_artifact=str(directory/'model.pkl')))
            diagnostic=dict(model=model,protocol=split['protocol'],fold=split['fold'],parameters=artifact.get('parameters'),
                train_metrics=metrics(y[train],training),test_metrics=metrics(y[test],pred),diagnostics=artifact['diagnostics'],elapsed_seconds=time.perf_counter()-started)
        with (directory/'model.pkl').open('wb') as stream:
            pickle.dump(artifact,stream,protocol=4)
        write_json(directory/'diagnostics.json',diagnostic);write_json(directory/'cv_records.json',records)
        write_csv(directory/'predictions.csv',rows)
        names=['model.pkl','predictions.csv','diagnostics.json','cv_records.json']
        write_json(marker,dict(contract=contract['fingerprint'],cache_key=key,environment=environment,artifacts={n:file_hash(directory/n) for n in names}))
        verify_task(config,task_index)
    except Exception:
        if marker.exists():
            marker.unlink()
        write_json(directory/'FAILED.json',dict(traceback=traceback.format_exc(),cache_key=key))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['one','verify'],required=True);parser.add_argument('--task-index',type=int,required=True)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        config=read_json(args.config)
        if 'source_files' in config or config.get('synthetic_fixture'):
            raise ValueError('Synthetic source overrides cannot run through the production CLI')
        if args.stage=='one':
            run_one(config,args.task_index)
        else:
            print(json.dumps(verify_task(config,args.task_index)))


if __name__=='__main__':
    main()
