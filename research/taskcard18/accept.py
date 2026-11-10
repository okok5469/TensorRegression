"""Independent cache acceptance, frozen-pilot gate and explicit formal execution."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .workflow import paths, repository_root, source_identity, unique_id, pilot_ids, restore, write_json
from numerical_simulation.known_s_cluster.run import load_plan, read_task, file_digest, array_digest, digest


def validate_metadata(metadata, method):
    expected = {'lasso': 'vector_lasso_cv_isolated_v2', 'glasso': 'sequential_tensor_glasso_certified_v2'}[method]
    if metadata.get('method_id') != expected or metadata.get('valid') is not True or not metadata.get('diagnostics'):
        raise ValueError('Wrong method, invalid flag or missing convex certificates')
    for d in metadata['diagnostics']:
        if not d['converged'] or d.get('negative_gap_error') or d.get('numerical_error') or d.get('at_limit'):
            raise ValueError('A cached convex fit is uncertified')
        if d['lambda'] > 0:
            numbers = [d['kkt_ratio'], d['gap_normalized']]
            if not np.isfinite(numbers).all() or d['kkt_ratio'] > 1e-4 or not -1e-10 <= d['gap_normalized'] <= 1e-6 or not d['dual_feasible']:
                raise ValueError('Cached positive-penalty fit fails hard dual criteria')
        elif not np.isfinite(d['gradient_normalized']) or d['gradient_normalized'] > 1e-8:
            raise ValueError('Zero-penalty fit fails least-squares certificate')
    groups = defaultdict(list)
    for d in metadata['diagnostics']:
        groups[(d.get('phase'), d.get('cycle'), d.get('mode'))].append(d)
    if len(groups) != len(metadata['selected_lambdas']):
        raise ValueError('Missing local-CV stage diagnostics')
    for diagnostics, selection in zip(groups.values(), metadata['selected_lambdas']):
        grid=np.asarray(selection['lambda_grid'],dtype=float)
        scores=np.asarray(selection['cv_scores'],dtype=float)
        if grid.ndim!=1 or not len(grid) or scores.shape!=grid.shape or not np.isfinite(grid).all() or not np.isfinite(scores).all() or np.any(grid<0) or np.any(np.diff(grid)>0):
            raise ValueError('Finite descending lambda grid and matching finite CV scores required')
        count = len(grid)
        final = [d for d in diagnostics if d['fit_kind'] == 'final']
        cv = [d for d in diagnostics if d['fit_kind'] == 'cv']
        keys = {(d['fold_index'], d['lambda_index']) for d in cv}
        if len(cv) != 5*count or keys != {(f, l) for f in range(5) for l in range(count)} or len(final) != 1:
            raise ValueError('Every CV candidate and final fit must have one certificate')
        selected = int(np.argmin(scores))
        if selected != selection['selected_index'] or final[0]['lambda_index'] != selected or selection['selected_lambda']!=grid[selected]:
            raise ValueError('Saved CV selection disagrees with the frozen tie/score rule')
        if any(d['lambda']!=grid[d['lambda_index']] for d in diagnostics):
            raise ValueError('Certificate penalty differs from its saved lambda-grid entry')
        reconstructed_scores=np.array([sum(d['validation_sse'] for d in cv if d['lambda_index']==index) for index in range(count)])
        if not np.isfinite(reconstructed_scores).all() or not np.allclose(scores,reconstructed_scores,rtol=1e-12,atol=1e-12):
            raise ValueError('Saved CV scores disagree with all recorded validation SSEs')
        for fold in range(5):
            ds = sorted((d for d in cv if d['fold_index']==fold), key=lambda d: d['lambda_index'])
            if len({d['train_indices_hash'] for d in ds}) != 1 or len({d['train_response_hash'] for d in ds}) != 1:
                raise ValueError('CV path changes fitting-data identity within a fold')
            if [d['warm_start_from'] for d in ds] != [None]+list(range(count-1)):
                raise ValueError('Invalid within-fold warm-state provenance')
    return True


def validate_arrays(B, supports, shape):
    if B.shape != tuple(shape) or not np.isfinite(B).all() or len(supports) != 3:
        raise ValueError('Invalid coefficient shape or nonfinite model')
    mask = np.ones(tuple(shape), dtype=bool)
    clean = []
    for support, p in zip(supports, shape):
        S = np.asarray(support)
        if S.ndim != 1 or S.dtype.kind not in 'iu' or len(np.unique(S)) != len(S) or np.any((S<0)|(S>=p)):
            raise ValueError('Invalid declared axis support')
        clean.append(S)
    mask[np.ix_(*clean)] = False
    if np.any(np.abs(B[mask]) > 1e-8):
        raise ValueError('Nonzero coefficients outside declared rectangular support')
    return True


def validate_fit(task, record, method):
    _, output = paths(); uid = unique_id(record); dest = output/'fits'/f'{uid}_{method}'
    metadata = json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('record_sha256') != digest({k:v for k,v in metadata.items() if k!='record_sha256'}):
        raise ValueError('Fit metadata seal mismatch')
    if metadata['identity'] != source_identity() or metadata['unique_input_id'] != uid or metadata['npz_sha256'] != file_digest(dest.with_suffix('.npz')):
        raise ValueError('Cached execution, input or coefficient identity mismatch')
    validate_metadata(metadata, method)
    n=task['config']['n_train']; folds=np.array_split(np.arange(n),5)
    for d in metadata['diagnostics']:
        expected_n=n if d['fit_kind']=='final' else n-len(folds[d['fold_index']])
        if d['n_fit']!=expected_n: raise ValueError('Certificate uses wrong fitting-subset sample count')
    with np.load(dest.with_suffix('.npz'), allow_pickle=False) as saved:
        B = saved['B'].copy(); supports = [saved[f'S{i}'].copy() for i in (1,2,3)]
    validate_arrays(B, supports, task['config']['dims'])
    return metadata, B, supports


def pilot_gate():
    historical, output = paths(); plan = load_plan(historical); ids = pilot_ids(plan); records=[]
    for task_id in ids:
        record = read_task(historical, plan, task_id)
        for method in ('lasso','glasso'):
            metadata, _, _ = validate_fit(record['task'], record, method)
            records.append({'task_id':task_id,'method':method,'runtime_sec':metadata['runtime_sec'],
                            'n_certificates':len(metadata['diagnostics']),
                            'max_kkt_ratio':max(d['kkt_ratio'] or 0 for d in metadata['diagnostics']),
                            'max_normalized_gap':max(d['gap_normalized'] or 0 for d in metadata['diagnostics'])})
    estimates={}
    for method in ('lasso','glasso'):
        durations=np.array([r['runtime_sec'] for r in records if r['method']==method])
        estimates[method]={'pilot_aggregate_method_elapsed_seconds':float(durations.sum()),'median_method_elapsed_seconds':float(np.median(durations)),
                           'full600_estimated_aggregate_method_elapsed_seconds':float(durations.mean()*600),
                           'cpu_time_measured':False,
                           'scope':'perf_counter method durations in the current execution environment and scheduling; not measured CPU time or parallel wall time; training-pilot extrapolation only'}
    write_json(output/'PILOT_GATE.json',{'status':'PASS','fits':40,'test_evaluated':False,'records':records,'resource_estimate':estimates,'identity':source_identity()})
    print('PILOT_GATE_PASS', json.dumps(estimates), flush=True)


def formal(workers):
    historical, output=paths(); plan=load_plan(historical)
    gate=json.loads((output/'PILOT_GATE.json').read_text(encoding='utf-8'))
    if gate['status']!='PASS' or gate['identity']!=source_identity():
        raise ValueError('Pilot numerical gate and frozen identity required')
    seen=set(); jobs=[]
    for task in plan['tasks']:
        record=read_task(historical,plan,task['task_id']);uid=unique_id(record)
        if uid in seen: continue
        seen.add(uid)
        for method in ('lasso','glasso'):
            dest=output/'fits'/f'{uid}_{method}.json'
            if dest.exists(): validate_fit(task,record,method)
            else: jobs.append((task['task_id'],method))
    if len(seen)!=600: raise ValueError('Expected600 unique complete input identities')
    logdir=output/'logs';logdir.mkdir(exist_ok=True,parents=True)
    def launch(job):
        task_id,method=job
        attempt=len(list(logdir.glob(f'formal_{task_id}_{method}_attempt*.txt')))
        with (logdir/f'formal_{task_id}_{method}_attempt{attempt}.txt').open('xb') as stream:
            done=subprocess.run([sys.executable,'-B','-m','research.taskcard18.workflow','fit','--task-id',str(task_id),'--method',method],cwd=repository_root(),stdout=stream,stderr=subprocess.STDOUT)
        return job,done.returncode
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending=iter(jobs); active={executor.submit(launch,job) for job in [next(pending,None) for _ in range(workers)] if job is not None}
        while active:
            future=next(as_completed(active));active.remove(future);job,code=future.result()
            print('FORMAL',*job,'EXIT',code,flush=True)
            if code: raise RuntimeError('Required certified fit failed; preserve evidence and stop')
            next_job=next(pending,None)
            if next_job is not None: active.add(executor.submit(launch,next_job))
    print('FORMAL_FITS_COMPLETE_1200',flush=True)


def evaluate_all():
    from numerical_simulation.metrics import compute_prediction_metrics_from_batches
    from numerical_simulation.data_generation import iter_test_batches
    from .solvers import certify
    historical, output=paths();plan=load_plan(historical)
    gate=json.loads((output/'PILOT_GATE.json').read_text(encoding='utf-8'))
    if gate['status']!='PASS' or gate['identity']!=source_identity(): raise ValueError('Unfrozen pilot')
    # Check completeness before exposing any new-method test performance.
    seen=set();tasks=[]
    for task in plan['tasks']:
        record=read_task(historical,plan,task['task_id']);uid=unique_id(record)
        if uid in seen: continue
        seen.add(uid)
        for method in ('lasso','glasso'): validate_fit(task,record,method)
        tasks.append(task)
    if len(tasks)!=600: raise ValueError('Incomplete comparison input list')
    for task in tasks:
        dataset,record,correspondence=restore(task['task_id']);uid=unique_id(record)
        for method in ('lasso','glasso'):
            metadata,B,S=validate_fit(task,record,method)
            dest=output/'evaluations'/f'{uid}_{method}.json'
            if dest.exists():
                prior=json.loads(dest.read_text(encoding='utf-8'))
                if prior['fit_npz_sha256']!=metadata['npz_sha256'] or prior['identity']!=source_identity(): raise ValueError('Evaluation cache identity mismatch')
                continue
            if method=='lasso':
                cert=certify(dataset.X_train.reshape(len(dataset.y_train),-1),dataset.y_train,B.ravel(),metadata['selected_lambdas'][0]['selected_lambda'])
                if not cert['converged']: raise ValueError('Saved final Lasso fails independent certificate replay')
            stream_hashes=[]
            def batches():
                for X,y,signal in iter_test_batches(dataset,seed=task['test_seed']):
                    stream_hashes.append({'X':array_digest(X),'y':array_digest(y),'signal':array_digest(signal)})
                    yield X,y,signal
            metrics=compute_prediction_metrics_from_batches(batches(),dataset.B_true,B)
            write_json(dest,{'unique_input_id':uid,'method_id':metadata['method_id'],'identity':source_identity(),
                             'fit_npz_sha256':metadata['npz_sha256'],'metrics':metrics,'test_stream_hashes':stream_hashes,'input_correspondence':correspondence})
        print('EVALUATED',task['task_id'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['pilot','formal','evaluate']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.command=='pilot':pilot_gate()
    elif args.command=='formal':formal(args.workers)
    else:evaluate_all()
