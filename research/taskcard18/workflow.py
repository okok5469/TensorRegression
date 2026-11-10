"""Explicit, immutable input reconstruction and certified comparison entry points."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from numerical_simulation.data_generation import generate_simulation_dataset, iter_test_batches
from numerical_simulation.known_s_cluster.run import (
    array_digest, commit_exclusive, config_from_dict, digest, file_digest, json_bytes, load_plan, read_task,
)
from numerical_simulation.metrics import compute_prediction_metrics_from_batches
from tensor_abess.tensor.ops import tensor_inner

def repository_root():
    return Path(__file__).resolve().parents[2]


def unique_id(record):
    return 'input_' + digest({'input_fingerprints': record['input_fingerprints'], 'test_seed': record['task']['test_seed']})


def paths():
    ROOT = repository_root()
    return ROOT / 'numerical_simulation/outputs_v52_search_final_full', ROOT / 'numerical_simulation/outputs_taskcard18_closeout'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Refusing to overwrite immutable evidence: ' + str(path))
        return
    commit_exclusive(path, content)


def restore(task_id, *, check_test=True):
    historical, _ = paths()
    plan = load_plan(historical)
    record = read_task(historical, plan, task_id)
    task = record['task']
    dataset = generate_simulation_dataset(config_from_dict(task['config']), seed=task['seed'])
    noise = dataset.y_train - tensor_inner(dataset.X_train, dataset.B_true)
    arrays = {'X_train': dataset.X_train, 'y_train': dataset.y_train, 'B_true': dataset.B_true, 'train_noise': noise}
    hashes = {k: array_digest(v) for k, v in arrays.items()}
    checks = {'old_fingerprints': record['input_fingerprints'], 'new_fingerprints': hashes,
              'exact_hash_match': {k: v == record['input_fingerprints'][k] for k, v in hashes.items()},
              'scope': 'frozen generator, seeds and saved arrays/Oracle predictions; no old X/y/test array available'}
    with np.load(historical / 'tasks' / record['arrays_file'], allow_pickle=False) as saved:
        for name in ('B_true', 'train_noise'):
            np.testing.assert_allclose(arrays[name], saved[name], rtol=1e-10, atol=1e-10)
        for axis, support in enumerate(dataset.supports_true, 1):
            np.testing.assert_array_equal(support, saved[f'S{axis}_true'])
        oracle = next(r for r in record['results'] if r['method'] == 'oracle')['metrics']
        residual = dataset.y_train - tensor_inner(dataset.X_train, saved['B_oracle'])
        rss = float(residual @ residual)
        np.testing.assert_allclose(rss, oracle['rss_train'], rtol=1e-10, atol=1e-10)
        checks['oracle_training_rss_abs_difference'] = abs(rss - oracle['rss_train'])
        if check_test:
            prediction = compute_prediction_metrics_from_batches(
                iter_test_batches(dataset, seed=task['test_seed']), dataset.B_true, saved['B_oracle'])
            for metric in ('pmse', 'signal_mse'):
                np.testing.assert_allclose(prediction[metric], oracle[metric], rtol=1e-10, atol=1e-10)
            checks['oracle_prediction_abs_differences'] = {m: abs(prediction[m] - oracle[m]) for m in prediction}
    checks.update({'task_id': task_id, 'unique_input_id': unique_id(record), 'valid': True,
                   'identity_level': 'BYTE_IDENTICAL' if all(checks['exact_hash_match'].values()) else 'NUMERICAL_CORRESPONDENCE_NOT_BYTE_IDENTITY'})
    return dataset, record, checks


def source_identity():
    import sklearn
    return {'adapter_hashes': {name: file_digest(Path(__file__).parent / name) for name in ('solvers.py', 'baselines.py', 'workflow.py')},
            'protocol_sha256': file_digest(repository_root() / 'doc/taskcard18/CLOSEOUT_PROTOCOL.json'),
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'platform': platform.system(), 'machine': platform.machine(),
            'numpy_blas_build': np.show_config(mode='dicts').get('Build Dependencies', {}),
            'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def pilot_ids(plan):
    protocol = json.loads((repository_root() / 'doc/taskcard18/CLOSEOUT_PROTOCOL.json').read_text(encoding='utf-8'))
    return [t['task_id'] for t in plan['tasks'] if t['scenario_id'] in protocol['pilot_scenarios'] and t['replicate'] in protocol['pilot_replicates']]


def reconstruct_pilot():
    historical, output = paths(); plan = load_plan(historical)
    ids = pilot_ids(plan)
    configs = {digest({k: v for k, v in t['config'].items() if k != 'random_state'}) for t in plan['tasks'] if t['task_id'] in ids}
    if len(ids) != 20 or len(configs) != 10:
        raise ValueError('Fixed pilot does not cover exactly10 configurations/20 inputs')
    checks = []
    for task_id in ids:
        _, _, check = restore(task_id)
        checks.append(check)
    write_json(output / 'pilot_input_audit_v2.json', {'valid': True, 'checks': checks, 'pilot_task_ids': ids})
    print('INPUT_PASS', len(checks), flush=True)


def fit_one(task_id, method):
    from .baselines import vector_lasso_cv, sequential_tensor_glasso
    historical, output = paths(); plan = load_plan(historical)
    check = json.loads((output / 'pilot_input_audit_v2.json').read_text(encoding='utf-8'))
    if not check['valid']:
        raise ValueError('Input pilot gate missing')
    dataset, record, correspondence = restore(task_id)
    uid = unique_id(record)
    dest = output / 'fits' / f'{uid}_{method}'
    identity = source_identity()
    if dest.with_suffix('.json').exists():
        existing = json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))
        if existing['identity'] != identity or existing['unique_input_id'] != uid:
            raise ValueError('Cached fit identity conflict')
        if file_digest(dest.with_suffix('.npz')) != existing['npz_sha256']:
            raise ValueError('Cached fit coefficient hash conflict')
        print('VERIFIED', task_id, method, flush=True); return
    function = {'lasso': vector_lasso_cv, 'glasso': sequential_tensor_glasso}[method]
    started = perf_counter()
    try:
        result = function(dataset.X_train, dataset.y_train, dataset.config)
    except Exception as error:
        failure = {'task_id': task_id, 'unique_input_id': uid, 'method': method, 'identity': identity,
                   'error_type': type(error).__name__, 'error': str(error), 'diagnostic': getattr(error, 'diagnostic', None),
                   'elapsed_sec': perf_counter() - started}
        write_json(output / 'failures' / f'{uid}_{method}.json', failure)
        raise
    if not result['valid'] or not all(d['converged'] for d in result['diagnostics']):
        raise ValueError('Uncertified result must not enter CV/test evaluation')
    dest.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, B=result['B'], **{f'S{i}': S for i, S in enumerate(result['supports'], 1)})
    commit_exclusive(dest.with_suffix('.npz'), buffer.getvalue())
    metadata = {k: v for k, v in result.items() if k not in ('B', 'supports')}
    metadata.update({'task_id': task_id, 'unique_input_id': uid, 'identity': identity, 'input_correspondence': correspondence,
                     'npz_sha256': file_digest(dest.with_suffix('.npz')), 'test_evaluated': False})
    metadata['record_sha256'] = digest(metadata)
    write_json(dest.with_suffix('.json'), metadata)
    print('FIT_PASS', task_id, method, result['runtime_sec'], len(result['diagnostics']), flush=True)


def batch(phase, workers):
    historical, output = paths(); plan = load_plan(historical)
    ids = pilot_ids(plan)
    if phase != 'pilot':
        raise ValueError('Formal comparison requires a completed frozen pilot gate')
    jobs = [(t, m) for t in ids for m in ('lasso', 'glasso')]
    logdir = output / 'logs'; logdir.mkdir(parents=True, exist_ok=True)
    def launch(job):
        task_id, method = job
        with (logdir / f'{task_id}_{method}.txt').open('wb') as stream:
            completed = subprocess.run([sys.executable, '-B', '-m', 'research.taskcard18.workflow', 'fit', '--task-id', str(task_id), '--method', method], cwd=repository_root(), stdout=stream, stderr=subprocess.STDOUT)
        return job, completed.returncode
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            job, code = future.result()
            print('JOB', *job, 'EXIT', code, flush=True)
            if code:
                for pending in futures: pending.cancel()
                raise RuntimeError('Pilot fit failure; see immutable failure record and log')
    print('PILOT_TRAINING_FITS_COMPLETE', len(jobs), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['inputs', 'fit', 'pilot'])
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--method', choices=['lasso', 'glasso'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'inputs': reconstruct_pilot()
    elif args.command == 'fit': fit_one(args.task_id, args.method)
    else: batch(args.command, args.workers)


if __name__ == '__main__':
    main()
