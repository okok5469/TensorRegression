"""Prediction-derived metrics and common-coordinate complexity; no fitting."""
import csv
import json
import pickle
import hashlib
from pathlib import Path
from itertools import combinations
import numpy as np


def metrics(y,pred):
    y=np.asarray(y,float);pred=np.asarray(pred,float);e=pred-y
    tss=float(np.sum((y-y.mean())**2)) if len(y) else 0.
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))) if len(y) else None,
                mae=float(np.mean(abs(e))) if len(y) else None,
                r2=float(1-np.sum(e*e)/tss) if len(y)>=2 and tss>0 else None,
                bias=float(e.mean()) if len(y) else None)


def complexity(artifact):
    beta=np.asarray(artifact['coef_standardized']);method=artifact['model_id'];diag=artifact['diagnostics']
    threshold=1e-8*max(1.,float(np.max(abs(beta))))
    active=abs(beta)>threshold;full=method in ('M6','M7')
    shape=(16,8) if full else (16,3,2)
    support=active.reshape(shape)
    axes=[np.flatnonzero(support.any(axis=tuple(j for j in range(len(shape)) if j!=i))).tolist() for i in range(len(shape))]
    native=diag.get('supports')
    native_coordinates=np.flatnonzero(beta!=0).tolist() if method in ('M3','M7') else None
    return dict(input_dim=0 if method=='M0' else len(beta),available_dim=len(beta),native_active_coordinates=native_coordinates,
                native_active_count=len(native_coordinates) if native_coordinates is not None else None,strict_nnz=int(np.count_nonzero(beta)),
                effective_nnz=int(active.sum()),threshold=threshold,sensors=[i+1 for i in axes[0]],
                n_sensors=len(axes[0]),scales=None if full else axes[1],stages=None if full else axes[2],
                n_scales=None if full else len(axes[1]),n_stages=None if full else len(axes[2]),
                envelope_capacity=None if full else int(np.prod([len(s) for s in axes])),
                native_supports=native,native_group_counts=[len(s) for s in native] if native is not None else None,
                support_capacity=diag.get('q',int(np.count_nonzero(beta))),
                q=diag.get('q'),training_rank=diag.get('training_rank'),selected_rank=diag.get('ols_rank'),
                pls_components=artifact['parameters'].get('n_components'),
                steady_nnz=int(support[:,:2].sum()) if full else 0,
                dynamic_nnz=int(support[:,2:].sum()) if full else int(active.sum()))


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def collect(root):
    root=Path(root);summary=[];prediction=[];comp=[];range_rows=[];support_rows=[]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8'));digest=sha(root/'contract.json')
    data=np.load(contract['data_path'],allow_pickle=False)
    for lockname in ('DEVELOPMENT_ACCEPTED.json','FINAL_ACCEPTED.json'):
        lock=json.loads((root/lockname).read_text(encoding='utf-8'))
        if lock['status']!='PASS' or lock['contract_sha256']!=digest or sha(root/lock['report'])!=lock['report_sha256']:
            raise ValueError('Independent acceptance absent or changed: '+lockname)
    statuses=[]
    for split in contract['splits']:
        for method in ['M'+str(i) for i in range(8)]:
            name=f"gas{split['gas_id']}_{split['split']}_{method}";folder=root/'models'/name
            status='not_evaluable' if split['status']!='ready' else 'complete' if (folder/'COMPLETE.json').exists() else 'failed' if (folder/'FAILED.json').exists() else 'not_run'
            statuses.append(dict(task_id=name,gas_id=split['gas_id'],split=split['split'],method=method,status=status,reasons=split['reasons']))
    write_csv(root/'task_status.csv',statuses)
    for marker in sorted((root/'models').glob('*/COMPLETE.json')):
        folder=marker.parent
        seal=json.loads(marker.read_text(encoding='utf-8'))
        if seal['contract_sha256']!=digest:raise ValueError('Changed model contract')
        for name,expected in seal['files'].items():
            if sha(folder/name)!=expected:raise ValueError('Changed model artifact: '+str(folder/name))
        with (folder/'model.pkl').open('rb') as f:a=pickle.load(f)
        with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        diag=json.loads((folder/'diagnostics.json').read_text(encoding='utf-8'))
        identity=dict(gas_id=a['split']['gas_id'],split=a['split']['split'],method=a['model_id'])
        parts={key:[r for r in rows if r['subset']==key] for key in ('train','test')}
        train,test=parts['train'],parts['test']
        ym=np.array([float(r['y']) for r in train]);yp=np.array([float(r['prediction']) for r in train])
        y=np.array([float(r['y']) for r in test]);pred=np.array([float(r['prediction']) for r in test])
        m=metrics(y,pred);tm=metrics(ym,yp)
        inner_diagnostics=[f['diagnostics'] for r in a['cv_records'] for f in r['folds']]
        selected_diagnostics=[f['diagnostics'] for r in a['cv_records'] if r['parameter']==a['parameters'] for f in r['folds']]
        summary.append(dict(**identity,n_train=len(train),n_test=len(test),**m,train_rmse=tm['rmse'],
                            train_r2=tm['r2'],train_y_sd=float(ym.std()),normalized_rmse=m['rmse']/ym.std(),
                            train_y_min=float(ym.min()),train_y_max=float(ym.max()),test_y_min=float(y.min()),
                            test_y_max=float(y.max()),outside_n=sum(r['range']!='inside' for r in test),
                            total_seconds=diag['total_seconds'],fit_count=diag['total_fit_count'],
                            convergence=diag['convergence_status'],all_fits_converged=diag['all_fits_converged'],
                            inner_unconverged=sum(d['converged'] is False for d in inner_diagnostics),
                            selected_inner_unconverged=sum(d['converged'] is False for d in selected_diagnostics),
                            inner_unknown=sum(d['converged'] is None for d in inner_diagnostics)))
        for group in ('inside','below','above'):
            selected=[r for r in test if r['range']==group]
            range_rows.append(dict(**identity,range=group,**metrics([float(r['y']) for r in selected],[float(r['prediction']) for r in selected])))
        prediction.extend(test)
        c=complexity(a);c.update(total_seconds=diag['total_seconds'],prediction_seconds=diag['prediction_seconds'])
        selected=data['raw128'] if a['model_id'] in ('M6','M7') else data['X']
        prep=a['preprocessing'];train_indices=np.asarray(a['train_indices'])
        design=((selected[train_indices]-prep['x_mean'])/prep['x_scale']).reshape(len(train_indices),-1)
        active=np.abs(a['coef_standardized'])>c['threshold']
        c['effective_design_rank']=int(np.linalg.matrix_rank(design[:,active])) if active.any() else 0
        comp.append(dict(**identity,**c))
        support_rows.append(dict(**identity,**c))
    if not summary:raise ValueError('No completed empirical tasks')
    write_csv(root/'summary_metrics.csv',summary);write_csv(root/'predictions.csv',prediction)
    write_csv(root/'model_complexity.csv',comp);write_csv(root/'range_metrics.csv',range_rows)
    stability=[];frequencies=[]
    for gas in range(1,7):
        for method in ['M'+str(i) for i in range(8)]:
            dev=[r for r in support_rows if r['gas_id']==gas and r['method']==method and r['split']!='final']
            for axis,size in [('sensors',16),('scales',3),('stages',2)]:
                eligible=[r for r in dev if r[axis] is not None]
                for a,b in combinations(eligible,2):
                    A,B=set(a[axis]),set(b[axis]);union=A|B
                    stability.append(dict(gas_id=gas,method=method,axis=axis,split_a=a['split'],split_b=b['split'],
                                          jaccard=len(A&B)/len(union) if union else 1.))
                for i in range(size):
                    coordinate=i+1 if axis=='sensors' else i
                    frequencies.append(dict(gas_id=gas,method=method,axis=axis,coordinate=coordinate,
                        selected_count=sum(coordinate in r[axis] for r in eligible),n_development_models=len(eligible),
                        frequency=sum(coordinate in r[axis] for r in eligible)/len(eligible) if eligible else None))
    if stability:write_csv(root/'support_stability.csv',stability)
    write_csv(root/'support_frequency.csv',frequencies)
    data.close()
    return dict(summary=summary,complexity=comp,ranges=range_rows,stability=stability,frequency=frequencies)


if __name__=='__main__':
    import sys
    sys.path[:0]=[str(Path('doc/taskcard13/runtime').resolve()),str(Path('src').resolve())]
    collect('doc/uci270_concentration')
