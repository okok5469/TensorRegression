"""Immutable per-task artifacts; serial bounded workers and explicit final seal."""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_files(contract):
    for path,expected in contract['source'].items():
        if sha(path)!=expected:raise ValueError('Frozen source changed: '+path)
    for key in ('data_path','config_path'):
        if key in contract and sha(contract[key])!=contract[key+'_sha256']:
            raise ValueError('Frozen file changed: '+contract[key])


def require_lock(root,name,contract_sha):
    path=Path(root)/name
    if not path.exists():raise ValueError('Required independent audit lock absent: '+name)
    lock=read_json(path)
    if lock.get('status')!='PASS' or lock.get('contract_sha256')!=contract_sha:
        raise ValueError('Independent audit lock mismatches contract: '+name)


def source_identity():
    manifest=read_json('doc/releases/v5.2-search-final/RELEASE_MANIFEST.json')
    names=[]
    for name,expected in manifest['source_canonical_lf_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise ValueError('Release mismatch: '+name)
        names.append(Path(name))
    names+=list(Path('numerical_simulation').glob('*.py'))
    names+=list(Path('empirical_analysis/lemon').glob('*.py'))
    names+=list(Path('empirical_analysis/lemon/incremental_value').glob('*.py'))
    names += [Path(p) for p in ('empirical_analysis/__init__.py','empirical_analysis/spectrofood_apple/models.py')]
    names += [Path('empirical_analysis/uci270_concentration')/p for p in ('data.py','models.py','protocol.py','runner.py')]
    names += [Path('doc/releases/v5.2-search-final')/p for p in ('RELEASE_MANIFEST.json','SOURCE_SHA256.txt')]
    return {p.as_posix():sha(p) for p in sorted(set(names))}


def environment():
    result={'python':sys.version,'executable':sys.executable,'packages':{},'imports':{}}
    for name in ('numpy','scipy','scikit-learn','threadpoolctl'):
        result['packages'][name]=importlib.metadata.version(name)
    for name in ('numpy','scipy','sklearn','tensor_abess.tensor.abess'):
        result['imports'][name]=str(Path(importlib.import_module(name).__file__).resolve())
    if Path(result['imports']['tensor_abess.tensor.abess'])!=Path('src/tensor_abess/tensor/abess.py').resolve():
        raise ValueError('Unexpected native ABESS import')
    return result


def freeze(root):
    from .protocol import make_splits
    from .models import _abess_options
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/'contract.json').exists():raise ValueError('Contract already exists; immutable')
    data_path=root/'data/dataset.npz'
    with np.load(data_path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
    config=dict(methods=['M'+str(i) for i in range(8)],abess=_abess_options(None),
                ridge_alphas=np.logspace(-4,4,17),lasso_ratios=[1,.5,.2,.1,.05,.02,.01,.005,.002,.001],
                glasso_ratios=np.geomspace(1,.01,12),glasso_options=dict(max_iter=500,tol=1e-6,max_cycles=6),
                glasso_solver='LEMON external-tuning grouped ISTA + rectangular OLS; convergence unknown',
                pls_components=[1,2,3,5,8,12,16],lasso_budgets=[10000,50000],lasso_tol=1e-6,
                tie_tolerance='1e-12*max(1,abs(best pooled inner MSE)); stronger penalty/fewer PLS components',
                coefficient_threshold='1e-8*max(1,max(abs(beta_standardized)))',
                workers=1,threads_per_worker=1,shared_fit_seconds=14400,task_timeout_seconds=1800,
                standardization='train-only mean, population SD; exactly zero X SD -> 1; y standardized; no clipping',
                runtime_dir='doc/taskcard13/runtime',final_gate='DEVELOPMENT_ACCEPTED.json')
    config_path=root/'config.json';write_json(config_path,config)
    contract=dict(source=source_identity(),environment=environment(),data_path=data_path.as_posix(),
                  data_path_sha256=sha(data_path),config_path=config_path.as_posix(),config_path_sha256=sha(config_path),
                  splits=make_splits(data),release='v5.2-search-final',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(root/'contract.json',contract)
    (root/'FROZEN_PROTOCOL.md').write_text('# UCI270 frozen protocol\n\n'
        'Six separate gas regressions; dev1 batches 1-6 -> 7, dev2 1-7 -> 8, dev3 1-8 -> 9; final 1-9 -> 10. '
        'The most recent up to three eligible inner batch boundaries are used; fewer than two blocks evaluation. '
        'Exact dynamic96 duplicates are removed from training against validation/test, using X only. '
        'Eight methods use identical cleaned IDs. Input is sensor x smoothing alpha x rising/decaying dynamic features (16 x 3 x 2); '
        'M6/M7 alone use all128. Every fold scales its own training X and y; no missing-value imputation or prediction clipping.\n\n'
        'M0 mean; M1 frozen native ABESS with original MSIC; M2 validated LEMON grouped-ISTA adapter with rectangular OLS; '
        'M3 Lasso96; M4 Ridge96; M5 PLS96; M6 Ridge128; M7 Lasso128. '
        'All grids, solver parameters, precision retry and tie tolerance are recorded in config.json. '
        'Lasso alpha_max is recomputed per training fold; inner selection minimizes pooled SSE/validation count. '
        'PLS uses the intersection of training rank-legal candidates and scale=False. '
        'ABESS and GLasso do not expose full convergence certification.\n\n'
        'One worker, BLAS/OMP/MKL=1; shared cumulative model subprocess wall time 14400 seconds; each task at most 1800 seconds. '
        'Pilot must pass an independent lock before development continuation; final remains sealed until a matching independent development lock. '
        'Existing results are never overwritten. No remote submission is authorized by this runner.\n\n'
        'Contract SHA256: `'+sha(root/'contract.json')+'`. See contract.json for raw hashes, full source dependency list, actual imports, environment and exact row IDs.\n',encoding='utf-8')
    return contract


def task_id(split,method):
    return f"gas{split['gas_id']}_{split['split']}_{method}"


def replay(path,X,raw,expected):
    from .models import predict_model
    with Path(path).open('rb') as f:artifact=pickle.load(f)
    actual=predict_model(artifact,X,raw)
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
    return float(np.max(np.abs(actual-expected)))


def verify_complete(root,name,contract_sha):
    folder=Path(root)/'models'/name; marker=folder/'COMPLETE.json'
    if not marker.exists():return False
    record=read_json(marker)
    if record['contract_sha256']!=contract_sha:raise ValueError('Task contract identity mismatch')
    for filename,digest in record['files'].items():
        if sha(folder/filename)!=digest:raise ValueError('Saved task changed: '+name+'/'+filename)
    return True


def run_one(root,name):
    from .models import fit_task
    root=Path(root);contract=read_json(root/'contract.json');verify_files(contract)
    digest=sha(root/'contract.json');config=read_json(root/'config.json')
    if environment()!=contract['environment']:raise ValueError('Worker runtime environment changed')
    running=root/'RUNNING.json'
    if not running.exists() or read_json(running).get('task_id')!=name or read_json(running).get('contract_sha256')!=digest:
        raise ValueError('Single-task fits must be launched by the budgeted controller')
    candidates=[(s,m) for s in contract['splits'] for m in config['methods'] if task_id(s,m)==name]
    if len(candidates)!=1:raise ValueError('Unknown task')
    split,method=candidates[0]
    if split['status']!='ready':raise ValueError('Task not evaluable')
    if split['split']=='final':require_lock(root,'DEVELOPMENT_ACCEPTED.json',digest)
    if verify_complete(root,name,digest):return
    folder=root/'models'/name;folder.mkdir(parents=True,exist_ok=False)
    try:
        with np.load(contract['data_path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
        train,test=np.asarray(split['train']),np.asarray(split['test'])
        out=fit_task(d['X'],d['raw128'],d['y'],train,test,[(i['train'],i['validation']) for i in split['inner']],method,config)
        artifact=out['artifact'];artifact.update(contract_sha256=digest,task_id=name,split=split)
        (folder/'model.pkl').write_bytes(pickle.dumps(artifact,protocol=pickle.HIGHEST_PROTOCOL))
        max_error=replay(folder/'model.pkl',d['X'][test],d['raw128'][test],out['predictions'])
        rows=[];low=float(d['y'][train].min());high=float(d['y'][train].max())
        for subset,indices,pred in [('train',train,out['train_predictions']),('test',test,out['predictions'])]:
            for idx,value in zip(indices,pred):
                y=float(d['y'][idx]);rows.append(dict(record_id=str(d['record_id'][idx]),row_index=int(idx),gas_id=int(d['gas'][idx]),
                    batch=int(d['batch'][idx]),split=split['split'],method=method,subset=subset,y=y,prediction=float(value),
                    range='below' if y<low else 'above' if y>high else 'inside'))
        with (folder/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(folder/'diagnostics.json',out['diagnostics']);write_json(folder/'cv_records.json',out['cv_records'])
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        write_json(folder/'COMPLETE.json',dict(status='COMPLETE',task_id=name,contract_sha256=digest,files=files,replay_max_abs=max_error))
    except BaseException:
        write_json(folder/'FAILED.json',dict(status='FAILED',traceback=traceback.format_exc(),contract_sha256=digest))
        raise


def controller(root,stage):
    root=Path(root); contract=read_json(root/'contract.json');verify_files(contract)
    if environment()!=contract['environment']:raise ValueError('Runtime environment changed')
    config=read_json(root/'config.json');digest=sha(root/'contract.json')
    ready=[s for s in contract['splits'] if s['status']=='ready']
    pilot=next(s for s in ready if s['split']=='dev1')
    selected={'pilot':[pilot],'dev':[s for s in ready if s['split']!='final'],'final':[s for s in ready if s['split']=='final']}[stage]
    if stage=='dev':require_lock(root,'PILOT_ACCEPTED.json',digest)
    if stage=='final':
        require_lock(root,'DEVELOPMENT_ACCEPTED.json',digest)
        for s in ready:
            if s['split']!='final':
                for m in config['methods']:
                    if not verify_complete(root,task_id(s,m),digest):raise ValueError('Development task incomplete')
    ledger_path=root/'fit_budget.json';ledger=read_json(ledger_path) if ledger_path.exists() else dict(spent_seconds=0.,attempts=[])
    running=root/'RUNNING.json'
    if running.exists():raise ValueError('Interrupted execution requires explicit audit of RUNNING.json')
    for split in selected:
        for method in config['methods']:
            name=task_id(split,method)
            if verify_complete(root,name,digest):continue
            remaining=config['shared_fit_seconds']-ledger['spent_seconds']
            if remaining<=0:
                write_json(root/'BUDGET_EXHAUSTED.json',dict(status='PARTIAL',spent_seconds=ledger['spent_seconds']));return
            if (root/'models'/name).exists():raise ValueError('Prior incomplete task preserved: '+name)
            write_json(running,dict(task_id=name,started_unix=time.time(),contract_sha256=digest))
            started=time.perf_counter();status='FAILED'
            logdir=root/'logs';logdir.mkdir(exist_ok=True)
            try:
                with (logdir/(name+'.log')).open('xb') as log:
                    completed=subprocess.run([sys.executable,'-m','empirical_analysis.uci270_concentration.runner','--stage','one',
                                              '--root',str(root),'--task',name],stdout=log,stderr=subprocess.STDOUT,
                                             timeout=min(config['task_timeout_seconds'],remaining),check=False)
                status='COMPLETE' if completed.returncode==0 and verify_complete(root,name,digest) else 'FAILED'
            except subprocess.TimeoutExpired:
                status='TIMEOUT';write_json(root/(name+'_TIMEOUT.json'),dict(status=status,task_id=name))
            finally:
                elapsed=time.perf_counter()-started;ledger['spent_seconds']+=elapsed
                ledger['attempts'].append(dict(task_id=name,status=status,seconds=elapsed));write_json(ledger_path,ledger);running.unlink()
            print(name,status,round(elapsed,3),flush=True)
            if status!='COMPLETE':raise RuntimeError('Task stopped: '+name+' '+status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','pilot','dev','final','one'],required=True)
    parser.add_argument('--root',default='doc/uci270_concentration');parser.add_argument('--task');args=parser.parse_args()
    runtime=Path('doc/taskcard13/runtime').resolve();src=Path('src').resolve()
    sys.path[:0]=[str(runtime),str(src)]
    os.environ['PYTHONPATH']=os.pathsep.join([str(runtime),str(src),str(Path.cwd())])
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.stage=='freeze':freeze(args.root)
        elif args.stage=='one':run_one(args.root,args.task)
        else:controller(args.root,args.stage)


if __name__=='__main__':main()
