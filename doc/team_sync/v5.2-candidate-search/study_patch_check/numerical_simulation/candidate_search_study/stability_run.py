"""Immutable, training-selected-size multistart experiment; Oracle is evaluation only."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from numerical_simulation.candidate_search_study import run as candidate_run
from numerical_simulation.candidate_search_study.stability import fit_multistart
from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test


GROUPS = ('A0', 'A3')
CORE = 'doc/algorithm_versions/v5.1-bcd-fix/src'


def selected_sizes(job):
    candidate_run.require(job['status'] == 'complete', 'Incomplete candidate job')
    result = {}
    for group in GROUPS:
        rows = [r for r in job['rows'] if r['group'] == group and r['rule'] == 'msic-alpha1.5-scale1']
        candidate_run.require(len(rows) == 1, f'Missing/duplicate training default {group}')
        candidate = job['groups'][group]['candidates'][rows[0]['selection']['selected_index']]
        sizes = tuple(int(s) for s in candidate['sizes'])
        candidate_run.require(len(sizes) == 3 and all(1 <= s <= p for s, p in zip(sizes, job['config']['dims']))
                              and np.prod(sizes) <= np.floor(.7 * job['config']['n_train']), 'Illegal selected sizes')
        result[group] = sizes
    return result


def validate_artifact(report, arrays, identity):
    candidate_run.require(report['status'] == 'complete' and report['identity'] == identity, 'Result identity mismatch')
    candidate_run.require(set(arrays) == set(report['array_fingerprints']), 'Result array keys mismatch')
    for key, expected in report['array_fingerprints'].items():
        candidate_run.require(array_fingerprint(arrays[key]) == expected, f'Result array fingerprint mismatch: {key}')
    for record in report['records']:
        index = record['record_index']
        candidate_run.require(array_fingerprint(arrays[f'record_{index}_B']) == record['B_fingerprint'], 'Record B differs')
        supports = [arrays[f'record_{index}_S{m}'].tolist() for m in (1, 2, 3)]
        candidate_run.require(supports == record['supports'], 'Record supports differ')


def sources():
    root = candidate_run.root()
    source = root / CORE
    files = sorted((source / 'tensor_abess').rglob('*.py'))
    candidate_run.require(bool(files), 'Missing frozen v5.1 core')
    experiment = [Path(__file__), Path(__file__).with_name('stability.py'),
                  root/'numerical_simulation/data_generation.py', root/'numerical_simulation/metrics.py',
                  root/'numerical_simulation/version_validation/worker.py', Path(candidate_run.__file__)]
    return {'core': {p.relative_to(source).as_posix(): candidate_run.file_hash(p) for p in files},
            'experiment': {p.relative_to(root).as_posix(): candidate_run.file_hash(p) for p in experiment}}


def configure_core():
    source = (candidate_run.root()/CORE).resolve()
    loaded = sys.modules.get('tensor_abess')
    candidate_run.require(loaded is None or Path(loaded.__file__).resolve().parent == source/'tensor_abess',
                          'Different core already loaded; use an isolated worker')
    sys.path.insert(0, str(source))


def create_plan(candidate_dir, output):
    candidate_dir, output = Path(candidate_dir).resolve(), Path(output)
    plan = candidate_run.load_plan(candidate_dir)
    candidate_run.require(plan['cohort'] == 'historical' and len(plan['tasks']) == 27, 'Expected historical 27-task plan')
    candidate_run.require(set(GROUPS) <= set(plan['groups']), 'Plan lacks A0/A3')
    output.mkdir(parents=True, exist_ok=True)
    candidate_run.require(not any(output.iterdir()), 'Stability plan requires empty output directory')
    body = {'schema_version': 1, 'candidate_dir': str(candidate_dir), 'candidate_plan_sha256': candidate_run.file_hash(candidate_dir/'plan.json'),
            'candidate_plan_id': plan['plan_id'], 'tasks': plan['tasks'], 'sources': sources(),
            'environment': candidate_run.environment(), 'core': CORE,
            'selection': 'A0/A3 original MSIC-selected sizes, marginal/random(seed+30000000)/random(seed+31000000); minimum training RSS; Oracle evaluation only',
            'fit_options': {'m_max': 15}, 'duplicate_shapes': 'Reuse identical fixed-size multistart fits explicitly within each dataset'}
    value = {**body, 'plan_id': candidate_run.digest(body)}
    for name in ('jobs', 'logs', 'failures'):
        (output/name).mkdir()
    candidate_run.write_json(output/'plan.json', value)
    return value


def load_plan(output):
    plan = candidate_run.read_json(Path(output)/'plan.json')
    candidate_run.require(plan['plan_id'] == candidate_run.digest({k: v for k, v in plan.items() if k != 'plan_id'}), 'Stability plan checksum mismatch')
    candidate_run.require(plan['sources'] == sources(), 'Frozen stability source changed')
    candidate_run.require(plan['environment'] == candidate_run.environment(), 'Stability environment changed')
    candidate_dir = Path(plan['candidate_dir'])
    candidate_run.require(candidate_run.file_hash(candidate_dir/'plan.json') == plan['candidate_plan_sha256'], 'Candidate plan changed')
    original = candidate_run.load_plan(candidate_dir)
    candidate_run.require(original['plan_id'] == plan['candidate_plan_id'] and original['tasks'] == plan['tasks'], 'Candidate task identity changed')
    return plan, original


def identity_for(plan, task, candidate_path):
    return {'plan_id': plan['plan_id'], 'task_id': task['task_id'], 'request_id': task['request_id'],
            'candidate_json_sha256': candidate_run.file_hash(candidate_path),
            'candidate_npz_sha256': candidate_run.file_hash(candidate_path.with_suffix('.npz'))}


def evaluate_job(job, original_arrays):
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_support_metrics, compute_coefficient_metrics
    from tensor_abess.tensor.ops import refit_tensor_coefficients, tensor_loss
    config = TensorSimulationConfig(**job['config'])
    dataset = generate_simulation_dataset(config, seed=int(job['seed']))
    fingerprints = {name: array_fingerprint(getattr(dataset, name)) for name in ('X_train', 'y_train', 'B_true')}
    candidate_run.require(all(fingerprints[k] == job['fingerprints'][k] for k in fingerprints), 'Recreated training inputs differ')
    sizes = selected_sizes(job)
    records, groups, cache = [], {}, {}
    for group in GROUPS:
        oldrow = next(r for r in job['rows'] if r['group'] == group and r['rule'] == 'msic-alpha1.5-scale1')
        ci = oldrow['selection']['selected_index']
        candidate = job['groups'][group]['candidates'][ci]
        B = original_arrays[f'{group}_candidate_{ci}_B'].copy()
        S = tuple(original_arrays[f'{group}_candidate_{ci}_S{m}'].copy() for m in (1, 2, 3))
        candidate_run.require(array_fingerprint(B) == candidate['B_fingerprint'] and [s.tolist() for s in S] == candidate['supports'], 'Original training selection cache differs')
        original_index = len(records)
        records.append({'strategy': 'candidate_selected', 'group_owner': group, 'sizes': sizes[group], 'B': B,
                        'supports': S, 'initial_supports': None, 'rss': candidate['rss'], 'loss': candidate['loss'],
                        'runtime_sec': 0.0, 'runtime_scope': 'Imported completed candidate fit; no new fitting cost'})
        if sizes[group] not in cache:
            selected, starts = fit_multistart(dataset.X_train, dataset.y_train, sizes[group], training_seed=int(job['seed']),
                max_cycles=config.max_cycles, k_max=config.k_max, m_max=15)
            indices = list(range(len(records), len(records)+len(starts)))
            records.extend(dict(start, group_owner=group) for start in starts)
            cache[sizes[group]] = (group, indices, indices[selected['selected_start']])
        owner, indices, winner = cache[sizes[group]]
        groups[group] = {'sizes': list(sizes[group]), 'original_record': original_index, 'start_records': indices,
                         'marginal_record': indices[0], 'selected_record': winner,
                         'reused_from': owner if owner != group else None,
                         'selection_scope': 'minimum training RSS among three predeclared starts; deterministic earliest tie',
                         'incremental_fit_sec': sum(records[i]['runtime_sec'] for i in indices) if owner == group else 0.0}
    # Oracle is deliberately outside training candidate construction and selection.
    oracle_started = perf_counter()
    oracle_B = refit_tensor_coefficients(dataset.X_train, dataset.y_train, *dataset.supports_true)
    oracle_loss = tensor_loss(dataset.X_train, dataset.y_train, oracle_B)
    oracle_index = len(records)
    records.append({'strategy': 'oracle_ols', 'group_owner': 'evaluation_only', 'sizes': tuple(config.sparsity),
                    'B': oracle_B, 'supports': dataset.supports_true, 'initial_supports': None,
                    'rss': float(2*config.n_train*oracle_loss), 'loss': float(oracle_loss), 'runtime_sec': perf_counter()-oracle_started,
                    'runtime_scope': 'Oracle reference, excluded from all training selection'})
    predictions, test_hash = evaluate_shared_test(dataset, [r['B'] for r in records], int(job['test_seed']))
    candidate_run.require(test_hash == job['fingerprints']['independent_test'], 'Independent test stream differs')
    arrays, metadata = {'B_true': dataset.B_true}, []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        B, S = record['B'], record['supports']
        rss = float(np.sum((dataset.y_train - np.einsum('nijk,ijk->n', dataset.X_train, B))**2))
        candidate_run.require(np.isclose(rss, record['rss'], rtol=1e-10, atol=1e-9), 'Stability RSS inconsistent')
        arrays[f'record_{index}_B'] = B
        for mode, support in enumerate(S, 1):
            arrays[f'record_{index}_S{mode}'] = np.asarray(support)
        metadata.append({k: v for k, v in record.items() if k != 'B'} | {'record_index': index,
            'B_fingerprint': array_fingerprint(B), **prediction, **compute_support_metrics(*S, *dataset.supports_true),
            **compute_coefficient_metrics(B, dataset.B_true)})
    return {'status': 'complete', 'config': job['config'], 'seed': job['seed'], 'test_seed': job['test_seed'],
            'groups': groups, 'oracle_record': oracle_index, 'records': metadata,
            'fingerprints': {**fingerprints, 'independent_test': test_hash},
            'array_fingerprints': {key: array_fingerprint(value) for key, value in arrays.items()}}, arrays


def job_path(output, task):
    return Path(output)/'jobs'/f"task_{task['task_id']:03d}.json"


def validate_result(output, plan, original_plan, task):
    oldpath = candidate_run.result_path(plan['candidate_dir'], task)
    candidate_run.validate_result(plan['candidate_dir'], original_plan, task)
    path = job_path(output, task)
    report = candidate_run.read_json(path)
    candidate_run.require(report['report_id'] == candidate_run.digest({k: v for k, v in report.items() if k != 'report_id'}), 'Stability report checksum differs')
    candidate_run.require(report['array_cache_sha256'] == candidate_run.file_hash(path.with_suffix('.npz')), 'Stability NPZ checksum differs')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
        validate_artifact(report, arrays, identity_for(plan, task, oldpath))
    return report


def worker(output, task_id):
    plan, original_plan = load_plan(output)
    task = next(t for t in plan['tasks'] if t['task_id'] == task_id)
    job = candidate_run.validate_result(plan['candidate_dir'], original_plan, task)
    oldpath = candidate_run.result_path(plan['candidate_dir'], task)
    path = job_path(output, task)
    candidate_run.require(not path.exists() and not path.with_suffix('.npz').exists(), 'Immutable stability output exists')
    configure_core()
    with np.load(oldpath.with_suffix('.npz'), allow_pickle=False) as original_arrays:
        report, arrays = evaluate_job(job, original_arrays)
    from numerical_simulation.candidate_search_study.worker import jsonable
    report = jsonable(report | {'identity': identity_for(plan, task, oldpath), 'task_id': task_id,
                               'source_fingerprints': plan['sources'], 'environment': plan['environment']})
    with path.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    report.update(array_cache=path.with_suffix('.npz').name, array_cache_sha256=candidate_run.file_hash(path.with_suffix('.npz')))
    report['report_id'] = candidate_run.digest(report)
    candidate_run.write_json(path, report)
    validate_result(output, plan, original_plan, task)


def execute(output, task_id):
    env = dict(os.environ)
    env.update({key: '1' for key in candidate_run.THREADS})
    command = [sys.executable, '-m', 'numerical_simulation.candidate_search_study.stability_run',
               'worker', '--output-dir', str(Path(output).resolve()), '--task-id', str(task_id)]
    completed = subprocess.run(command, cwd=candidate_run.root(), env=env, capture_output=True, text=True, encoding='utf-8')
    for name, value in (('stdout', completed.stdout), ('stderr', completed.stderr)):
        with (Path(output)/'logs'/f'task_{task_id:03d}.{name}.txt').open('x', encoding='utf-8') as handle:
            handle.write(value)
    if completed.returncode:
        candidate_run.write_json(Path(output)/'failures'/f'task_{task_id:03d}.json',
                                 {'task_id': task_id, 'error': completed.stderr, 'returncode': completed.returncode})
        raise RuntimeError(f'Stability task {task_id} failed: {completed.stderr}')
    print(f'Completed stability task {task_id}', flush=True)


def run(output, workers, task_ids=None):
    plan, original = load_plan(output)
    candidate_run.require(1 <= workers <= 64, 'workers must be in 1..64')
    candidate_run.require(not any((Path(output)/'failures').iterdir()), 'Recorded failure prevents execution')
    selected = plan['tasks'] if task_ids is None else [t for t in plan['tasks'] if t['task_id'] in set(task_ids)]
    candidate_run.require(task_ids is None or len(selected) == len(set(task_ids)), 'Unknown task-id')
    pending = []
    for task in selected:
        # Missing candidate inputs fail before any dependent stability work starts.
        candidate_run.validate_result(plan['candidate_dir'], original, task)
        path = job_path(output, task)
        if path.exists() or path.with_suffix('.npz').exists():
            validate_result(output, plan, original, task)
        else:
            candidate_run.require(not any((Path(output)/'logs').glob(f"task_{task['task_id']:03d}.*")),
                                  'Orphaned immutable worker logs require investigation')
            pending.append(task['task_id'])
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {executor.submit(execute, output, next(iterator)) for _ in range(min(workers, len(pending)))}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            errors = [future.exception() for future in done if future.exception() is not None]
            if errors:
                for future in active:
                    future.cancel()
                raise errors[0]
            for _ in done:
                task_id = next(iterator, None)
                if task_id is not None:
                    active.add(executor.submit(execute, output, task_id))


def summarize(output):
    plan, original = load_plan(output)
    reports = [validate_result(output, plan, original, task) for task in plan['tasks']]
    rows, paired = [], []
    for report in reports:
        for group, info in report['groups'].items():
            start, winner, baseline = [report['records'][info[k]] for k in ('marginal_record', 'selected_record', 'original_record')]
            row = {'task_id': report['task_id'], 'group': group, 'n_train': report['config']['n_train'],
                   'snr': report['config']['snr'], 'seed': report['seed'], 'sizes': json.dumps(info['sizes']),
                   'reused_from': info['reused_from'], 'selected_strategy': winner['strategy'],
                   'incremental_fit_sec': info['incremental_fit_sec']}
            for key in ('rss', 'pmse', 'r2_test', 'exact_recovery', 'mode_f1_macro', 'element_f1'):
                row.update({f'{prefix}_{key}': value[key] for prefix, value in [('candidate', baseline), ('marginal', start), ('multistart', winner)]})
                row[f'multistart_minus_marginal_{key}'] = winner[key]-start[key]
            paired.append(row)
        rows.extend({'task_id': report['task_id'], 'n_train': report['config']['n_train'], 'snr': report['config']['snr'],
                     **{k: v for k, v in record.items() if k not in ('loss_history', 'visited_loss_history')}} for record in report['records'])
    for name, values in [('all_starts.csv', rows), ('paired_multistart.csv', paired)]:
        path = Path(output)/name
        candidate_run.require(not path.exists(), 'Immutable summary exists')
        keys = list(dict.fromkeys(k for row in values for k in row))
        with path.open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in values)
    return paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'run', 'worker', 'summarize'))
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--task-id', type=int, action='append')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.action == 'create':
        candidate_run.require(args.candidate_dir is not None, 'create requires --candidate-dir')
        create_plan(args.candidate_dir, args.output_dir)
    elif args.action == 'worker':
        candidate_run.require(args.task_id is not None and len(args.task_id) == 1, 'worker requires one task-id')
        worker(args.output_dir, args.task_id[0])
    elif args.action == 'summarize':
        summarize(args.output_dir)
    else:
        run(args.output_dir, args.workers, args.task_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
