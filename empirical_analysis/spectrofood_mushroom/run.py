"""Frozen single-task Mushroom worker; no scientific split generation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
from empirical_analysis.spectrofood_apple.run import (
    fingerprint,file_hash,write_json,read_json,write_csv,runtime_identity,metrics)
from .models import select_input,train_model,tune_model,predict_model


def source_identity(config):
    if 'source_files' in config:
        files=config['source_files']
    else:
        from empirical_analysis.spectrofood_apple.run import source_identity as original
        result=original(config)
        files=list(Path('empirical_analysis/spectrofood_mushroom').glob('*.py'))
        files.append(Path('empirical_analysis/mushroom_screen.py'))
        result.update({str(p).replace('\\','/'):file_hash(p) for p in files})
        return result
    return {str(p):file_hash(p) for p in files}


def frozen_contract(config):
    path=Path(config.get('contract_path',str(Path(config['output_dir'])/'contract.json')))
    contract=read_json(path)
    body=dict(config=config,source=source_identity(config),data_sha256=file_hash(config['data_path']),splits=contract['splits'])
    if fingerprint(body)!=contract['fingerprint']:
        raise ValueError('Frozen data/source/config/split contract changed')
    return contract


def task_info(config,index):
    contract=frozen_contract(config)
    count=len(contract['splits'])*len(config['models'])
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<count:
        raise ValueError('Invalid task index')
    split=contract['splits'][index//len(config['models'])]
    model=config['models'][index%len(config['models'])]
    directory=Path(config['output_dir'])/'tasks'/f"{split['protocol']}_r{split['repeat']}_f{split['fold']}_{model}"
    return contract,split,model,directory


def verify_task(config,index):
    contract,split,model,directory=task_info(config,index)
    complete=read_json(directory/'COMPLETE.json')
    if complete['contract']!=contract['fingerprint']:
        raise ValueError('Completed contract mismatch')
    key=fingerprint(dict(contract=contract['fingerprint'],split=split,model=model,environment=complete['environment']))
    if complete['cache_key']!=key:
        raise ValueError('Completion identity mismatch')
    expected={'model.pkl','predictions.csv','diagnostics.json','cv_records.json'}
    if set(complete['artifacts'])!=expected:
        raise ValueError('Missing completed artifact identity')
    for name,digest in complete['artifacts'].items():
        if file_hash(directory/name)!=digest:
            raise ValueError('Corrupt artifact hash: '+name)
    with np.load(config['data_path'],allow_pickle=False) as data:
        train,test=np.array(split['train']),np.array(split['test'])
        X=select_input(data,model);y=data['y_pct'];ids=data['ids']
        with (directory/'model.pkl').open('rb') as stream:
            artifact=pickle.load(stream)
        prep=artifact['preprocessing']
        if not np.allclose(prep['x_mean'],X[train].mean(0),rtol=1e-12,atol=1e-12):
            raise ValueError('Saved training scaler mismatch')
        if not np.allclose([prep['y_mean'],prep['y_scale']],[y[train].mean(),y[train].std()],rtol=1e-12,atol=1e-12):
            raise ValueError('Saved response scaler mismatch')
        with (directory/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        by_id={r['sample_id']:r for r in rows}
        if len(rows)!=len(test) or set(by_id)!=set(ids[test]):
            raise ValueError('Held-out prediction ID mismatch')
        saved=np.array([float(by_id[str(ids[i])]['y_pred_pct']) for i in test])
        replay=predict_model(artifact,X[test])
        if not np.allclose(saved,replay,rtol=1e-9,atol=1e-10):
            raise ValueError('Independent prediction replay mismatch')
        for i in test:
            row=by_id[str(ids[i])]
            if row['model']!=model or row['protocol']!=split['protocol'] or int(row['repeat'])!=split['repeat'] or int(row['fold'])!=split['fold']:
                raise ValueError('Saved prediction task identity mismatch')
            if not np.isclose(float(row['y_true_pct']),y[i],rtol=0,atol=1e-12):
                raise ValueError('Response alignment mismatch')
    return dict(prediction_count=len(rows),max_abs_replay_error=float(np.max(abs(saved-replay))))


def run_one(config,task_index):
    started=time.perf_counter()
    contract,split,model,directory=task_info(config,task_index)
    environment=runtime_identity()
    key=fingerprint(dict(contract=contract['fingerprint'],split=split,model=model,environment=environment))
    marker=directory/'COMPLETE.json'
    if marker.exists():
        complete=read_json(marker)
        if complete['cache_key']!=key:
            raise ValueError('Completed task environment/identity mismatch')
        verify_task(config,task_index)
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete prior task requires explicit failure review: '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    try:
        with np.load(config['data_path'],allow_pickle=False) as data:
            X=select_input(data,model);y=data['y_pct']
            train,test=np.array(split['train']),np.array(split['test'])
            if set(data['groups'][train]) & set(data['groups'][test]):
                raise ValueError('Group crosses outer split')
            inner=[(np.array(s['train']),np.array(s['validation'])) for s in split['inner']]
            for a,b in inner:
                if set(data['groups'][train[a]]) & set(data['groups'][train[b]]):
                    raise ValueError('Group crosses inner split')
            options=config.get('model_config',{})
            parameter,records=tune_model(model,X[train],y[train],inner,split['inner_seed'],options)
            artifact=train_model(model,X[train],y[train],parameter,split['inner_seed'],options)
            pred=predict_model(artifact,X[test]);training=predict_model(artifact,X[train])
            rows=[]
            for i,value in zip(test,pred):
                residual=float(y[i]-value)
                rows.append(dict(sample_id=str(data['ids'][i]),entity_id=str(data['groups'][i]),group_id=str(data['groups'][i]),
                    block=int(data['blocks'][i]),protocol=split['protocol'],repeat=split['repeat'],fold=split['fold'],model=model,
                    y_true_pct=float(y[i]),y_pred_pct=float(value),residual_pct=residual,squared_error=residual**2,
                    model_artifact=str(directory/'model.pkl')))
            diagnostic=dict(model=model,protocol=split['protocol'],repeat=split['repeat'],fold=split['fold'],
                train_metrics=metrics(y[train],training),test_metrics=metrics(y[test],pred),diagnostics=artifact['diagnostics'],
                elapsed_seconds=time.perf_counter()-started)
        with (directory/'model.pkl').open('wb') as stream:
            pickle.dump(artifact,stream,protocol=4)
        write_csv(directory/'predictions.csv',rows)
        write_json(directory/'diagnostics.json',diagnostic)
        write_json(directory/'cv_records.json',records)
        names=['model.pkl','predictions.csv','diagnostics.json','cv_records.json']
        write_json(marker,dict(contract=contract['fingerprint'],cache_key=key,environment=environment,
                              artifacts={name:file_hash(directory/name) for name in names}))
        verify_task(config,task_index)
    except Exception:
        if marker.exists():
            marker.unlink()
        write_json(directory/'FAILED.json',dict(traceback=traceback.format_exc(),cache_key=key))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['one','verify'],required=True)
    parser.add_argument('--task-index',type=int,required=True)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        config=read_json(args.config)
        if args.stage=='one':
            run_one(config,args.task_index)
        else:
            print(json.dumps(verify_task(config,args.task_index)))


if __name__=='__main__':
    main()
