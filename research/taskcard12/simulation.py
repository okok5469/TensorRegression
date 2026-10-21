"""Frozen paired cluster simulation: exclusive cases, strict resume and merge."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from scripts.taskcard7.lemon import array_sha, require
from research.taskcard10.experiment import (file_sha, read_json, write_json, write_npz,
                                           artifact_manifest, evaluate_test_stream, verify_sources)


def validate_requests(requests):
    require(len(requests) == 380, 'Exactly 380 frozen formal requests required')
    require([r['task_id'] for r in requests] == list(range(380)), 'Task coverage or duplicate IDs')
    cells = list(itertools.product((100, 200, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)), (0., .6)))
    identities = []
    for index, row in enumerate(requests):
        config = row['config']
        cell, replicate = (index // 10, index % 10) if index < 360 else (21, index - 350)
        expected = cells[cell]
        actual = (config['n_train'], config['snr'], tuple(config['sparsity']), config['rho'][0])
        require(actual == expected and config['rho'] == [expected[3]] * 3 and
                row['cell_id'] == cell and row['replicate'] == replicate,
                'Scenario/pressure coverage differs from frozen 360+20 design')
        require(bool(row['pressure']) == (index >= 360), 'Pressure stratum differs')
        require(row['test_seed'] == row['seed'] + 50_000_000, 'Frozen test seed offset differs')
        identities.extend([row['seed'], row['test_seed']])
    require(len(identities) == len(set(identities)), 'Formal seed collision')
    require(all(isinstance(s, int) and 0 <= s < 2 ** 32 for s in identities), 'Seed range/type differs')


def environment():
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'system': platform.system(), 'machine': platform.machine(), 'executable': sys.executable,
            'numpy_config': np.show_config(mode='dicts'),
            'threads': {key: os.environ.get(key) for key in
                        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def validate_environment(actual, protocol):
    expected = {**protocol['expected_versions'], **protocol['expected_platform']}
    require(all(actual.get(k) == v for k, v in expected.items()), 'Cluster numerical environment differs')
    require(actual.get('threads') == dict.fromkeys(
        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'),
        'Cluster thread environment differs')
    blas = actual['numpy_config']['Build Dependencies']['blas']
    require('openblas' in blas['name'].lower() and
            blas['version'] == protocol['expected_blas']['openblas_version'], 'Cluster BLAS environment differs')


def validate_request_artifacts(repo, protocol):
    repo = Path(repo).resolve()
    request_path = (repo / protocol['requests_path']).resolve()
    require(request_path.is_relative_to(repo) and file_sha(request_path) == protocol['requests_sha256'],
            'Frozen request bytes differ')
    manifest = read_json(request_path)
    require(all(protocol[key] == manifest[key] for key in ('requests', 'smoke_requests')), 'Frozen request content differs')
    for row in protocol['requests'] + protocol['smoke_requests']:
        config = row['config']
        expected = dict(protocol['template_config'], **{k: config[k] for k in ('n_train', 'snr', 'sparsity', 'rho')})
        require(config == expected, 'DGP/solver template changed outside frozen scene factors')
    expected_smoke = [(100, .5, [4, 3, 5], [.6] * 3, 1),
                      (200, 4., [4, 3, 5], [.6] * 3, 21),
                      (400, 4., [2, 3, 4], [.6] * 3, 35)]
    actual = [(r['config']['n_train'], r['config']['snr'], r['config']['sparsity'], r['config']['rho'], r['cell_id'])
              for r in protocol['smoke_requests']]
    require(actual == expected_smoke and all(r['replicate'] == 0 and not r['pressure'] and r['cohort'] == 'smoke'
            for r in protocol['smoke_requests']), 'Smoke request coverage differs')
    registry = protocol['history_input_registry']
    history_path = (repo / registry['path']).resolve()
    require(history_path.is_relative_to(repo) and file_sha(history_path) == registry['sha256'],
            'Historical input registry bytes differ')
    rows = read_json(history_path)['inputs']
    require(all(isinstance(r['X_train'], str) and isinstance(r['y_train'], str) for r in rows),
            'Historical input registry schema differs')
    return rows


def verify_protocol(repo, protocol_path):
    protocol = read_json(protocol_path)
    verify_sources(repo, protocol['source_files'])
    validate_requests(protocol['requests'])
    validate_request_artifacts(repo, protocol)
    smoke = protocol['smoke_requests']
    require([r['task_id'] for r in smoke] == list(range(3)), 'Exactly three smoke request IDs required')
    all_seeds = [r[k] for r in protocol['requests'] + smoke for k in ('seed', 'test_seed')]
    require(len(all_seeds) == len(set(all_seeds)), 'Smoke/formal seed collision')
    require(protocol['max_parallel'] == 64 and protocol['case_timeout_seconds'] == 3600,
            'Frozen cluster resource configuration differs')
    validate_environment(environment(), protocol)
    return protocol


def prepare(repo, output, protocol_path):
    repo, output, protocol_path = Path(repo).resolve(), Path(output).resolve(), Path(protocol_path).resolve()
    protocol = verify_protocol(repo, protocol_path)
    plan = {'schema_version': 1, 'protocol': protocol,
            'protocol_path': protocol_path.relative_to(repo).as_posix(),
            'protocol_sha256': file_sha(protocol_path), 'environment': environment(),
            'requests': protocol['requests'], 'smoke_requests': protocol['smoke_requests'],
            'created_utc': datetime.now(timezone.utc).isoformat(), 'research_only': True,
            'release_approved': False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    return plan


def verify_plan(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = read_json(output / 'plan.json')
    path = repo / plan['protocol_path']
    require(file_sha(path) == plan['protocol_sha256'], 'Frozen cluster protocol bytes differ')
    require(verify_protocol(repo, path) == plan['protocol'], 'Frozen protocol differs')
    require(environment() == plan['environment'], 'Bound full numerical environment differs')
    require(plan['requests'] == plan['protocol']['requests'] and
            plan['smoke_requests'] == plan['protocol']['smoke_requests'], 'Frozen request identity differs')
    return plan


def require_gate(repo, output):
    output = Path(output)
    plan = verify_plan(repo, output)
    path = output / 'smoke_gate.json'
    require(path.is_file(), 'Complete real cluster smoke gate required before formal work')
    gate = read_json(path)
    require(gate['passed'] and gate['plan_sha256'] == file_sha(output / 'plan.json'), 'Frozen smoke gate differs')
    records = collect(output, 'smoke')
    require(gate['smoke_complete_sha256'] == {
        str(r['request']['task_id']): file_sha(output / 'smoke' / f"task_{r['request']['task_id']:03d}" / 'complete.json')
        for r in records}, 'Completed smoke artifacts differ from gate')
    require(gate['environment'] == plan['environment'], 'Smoke numerical environment differs')
    require(gate['cost'] == smoke_cost(records, plan['protocol']['max_parallel']) and gate['cost']['passed'],
            'Frozen smoke cost gate differs or failed')
    return gate


def smoke_cost(records, parallel):
    by_n = {r['request']['config']['n_train']: r['resources']['wall_seconds'] for r in records}
    require(set(by_n) == {100, 200, 400} and len(records) == 3 and
            all(np.isfinite(t) and t > 0 for t in by_n.values()), 'Complete three-size smoke costs required')
    total = 1.5 * sum(count * by_n[n] for n, count in ((100, 120), (200, 140), (400, 120)))
    return {'passed': 1.5 * max(by_n.values()) <= 3600,
            'maximum_case_seconds_with_margin': 1.5 * max(by_n.values()),
            'case_limit_seconds': 3600, 'projected_cpu_seconds_with_margin': total,
            'ideal_no_queue_wall_lower_bound_seconds': total / parallel,
            'assumption': 'One observed smoke per n, factor1.5; /64 assumes all slots available, not a site quota or runtime promise'}


def create_gate(repo, output):
    output = Path(output)
    plan = verify_plan(repo, output)
    records = collect(output, 'smoke')
    for record in records:
        require(record['slurm']['array_task_id'] == record['request']['task_id'] and
                record['slurm']['array_task_count'] == 3, 'Smoke requires real three-item SLURM array evidence')
    cost = smoke_cost(records, plan['protocol']['max_parallel'])
    gate = {'passed': cost['passed'], 'cost': cost,
            'plan_sha256': file_sha(output / 'plan.json'), 'environment': plan['environment'],
            'smoke_complete_sha256': {str(r['request']['task_id']): file_sha(output / 'smoke' /
                f"task_{r['request']['task_id']:03d}" / 'complete.json') for r in records},
            'case_wall_seconds': [r['resources']['wall_seconds'] for r in records],
            'interpretation': 'Correctness and complete real-job evidence; formal science remains NOT_RUN',
            'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'smoke_gate.json', gate)
    require(cost['passed'], 'Frozen smoke cost gate failed; preserve evidence and stop formal submission')
    return gate


def slurm_identity(env, task_id):
    names = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_ARRAY_TASK_COUNT')
    require(all(str(env.get(k, '')).isdigit() for k in names), 'Real SLURM array metadata required')
    require(int(env['SLURM_ARRAY_TASK_ID']) == task_id and int(env['SLURM_ARRAY_TASK_COUNT']) > 0,
            'SLURM task identity differs')
    return dict(zip(('job_id', 'array_job_id', 'array_task_id', 'array_task_count'),
                    [int(env[k]) for k in names]))


def claim_input(output, fingerprints, request, historical, *, alternate_pairs=()):
    pair = (fingerprints['X_train'], fingerprints['y_train'])
    known = {(r['X_train'], r['y_train']) for r in historical}
    require(not known.intersection((pair, *alternate_pairs)), 'New input duplicates historical input')
    key = hashlib.sha256(json.dumps(pair, separators=(',', ':')).encode()).hexdigest()
    path = Path(output) / 'input_claims' / (key + '.json')
    try:
        write_json(path, {'input_fingerprints': fingerprints, 'request': request})
    except FileExistsError as error:
        raise ValueError('New input duplicate within smoke/formal run') from error
    return path


def run_owned_child(command, folder, timeout_seconds, cwd=None):
    """Bound only our subprocess/group; preserve timeout and failure logs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
        from research.taskcard10.experiment import _launch_owned, _close_job, _terminate_process_tree
        process = (_launch_owned(command, cwd, dict(os.environ), stdout, stderr) if os.name == 'nt' else
                   subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True))
        timed_out = False
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'nt':
                _terminate_process_tree(process)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException:
            if os.name == 'nt':
                _terminate_process_tree(process)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            if os.name == 'nt':
                _close_job(process)
    return {'passed': code == 0 and not timed_out, 'returncode': 124 if timed_out else code,
            'process_returncode': code, 'timed_out': timed_out,
            'wall_seconds': time.monotonic() - started}


def check_artifact_inventory(folder, expected):
    actual = artifact_manifest(folder)
    actual.pop('complete.json', None)
    require(actual == expected, 'Completed artifact inventory or checksum differs')


def audit_saved_metrics(folder, request, record, pool, summary, coefficients):
    """Recompute reported metrics from saved arrays, without test generation."""
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    dims = tuple(request['config']['dims'])
    with np.load(folder / 'truth.npz', allow_pickle=False) as truth:
        B_true = truth['B_true']
        supports = [truth[f'S{i}'] for i in (1, 2, 3)]
    require(B_true.shape == dims and np.isfinite(B_true).all() and
            array_sha(B_true) == record['input_fingerprints']['B_true'], 'Saved truth coefficient identity differs')
    require([array_sha(s) for s in supports] == record['input_fingerprints']['supports_true'],
            'Saved truth support identity differs')
    require(all(s.ndim == 1 and np.issubdtype(s.dtype, np.integer) and
                np.array_equal(s, np.unique(s)) and (s >= 0).all() and (s < d).all()
                for s, d in zip(supports, dims)), 'Saved truth support domain differs')
    true_sizes = list(map(len, supports))
    true_support = [s.astype(int).tolist() for s in supports]
    true_q = int(np.prod(true_sizes))
    require(true_sizes == request['config']['sparsity'], 'Saved truth sparsity differs')
    trace = pool['metadata']['trace']
    Q = int(np.floor(.7 * pool['metadata']['n']))
    recovery = {'q_true': true_q, 'truth_sizes': true_sizes, 'truth_support': true_support,
                'Q_native': Q, 'truth_shape_native_representable': bool(
                    true_q <= Q and all(s <= c for s, c in zip(true_sizes, trace['caps']))),
                'truth_shape_found_full_pool': any(r['sizes'] == true_sizes for r in trace['candidates']),
                'truth_support_found_full_pool': any(r['support'] == true_support for r in trace['candidates'])}
    require(record['recovery_evidence'] == recovery, 'Saved recovery evidence differs from truth/pool')
    n_test = request['config']['n_test']
    with np.load(folder / 'test_predictions.npz', allow_pickle=False) as saved:
        y, signal, prediction = saved['y'], saved['signal'], saved['prediction']
        require(saved['arm_order'].tolist() == ['A0', 'A1'] and saved['model_sha256'].tolist() ==
                [summary[k]['B_sha256'] for k in ('A0', 'A1')], 'Saved prediction column identity differs')
    require(n_test > 0 and y.shape == signal.shape == (n_test,) and prediction.shape == (n_test, 2) and
            all(np.isfinite(a).all() for a in (y, signal, prediction)) and
            record['test_fingerprint']['n_observations'] == n_test, 'Saved prediction shape/finite count differs')
    pmse = np.mean((y[:, None] - prediction) ** 2, axis=0)
    signal_mse = np.mean((signal[:, None] - prediction) ** 2, axis=0)
    require(np.isfinite(pmse).all() and (pmse > 0).all() and np.isfinite(signal_mse).all(),
            'Saved prediction metric domain differs')
    for column, label in enumerate(('A0', 'A1')):
        B, model = coefficients[label], summary[label]
        estimates = [np.asarray(s, dtype=int) for s in model['support']]
        expected = {**compute_coefficient_metrics(B, B_true),
                    **compute_support_metrics(*estimates, *supports),
                    'pmse': float(pmse[column]), 'signal_mse': float(signal_mse[column]),
                    'coefficient_norm': float(np.linalg.norm(B)),
                    'q_under': model['q'] < true_q, 'q_over': model['q'] > true_q}
        require(all(np.isfinite(record['metrics'][label][k]) and
                    np.isclose(record['metrics'][label][k], v, rtol=1e-12, atol=1e-10)
                    for k, v in expected.items()), 'Saved numerical metric differs: ' + label)


def completed_result(folder, request, plan, plan_sha):
    folder = Path(folder)
    if not folder.exists():
        return None
    require((folder / 'complete.json').is_file() and not (folder / 'failed.json').exists(),
            'Partial/failed task preserved; no automatic overwrite or retry: ' + str(folder))
    record = read_json(folder / 'complete.json')
    require(record['request'] == request and record['plan_sha256'] == plan_sha, 'Complete request/plan identity differs')
    require(record['environment'] == plan['environment'] and record['source_files'] == plan['protocol']['source_files'],
            'Complete source/environment identity differs')
    for name, digest in record['artifact_sha256'].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and path.is_file() and file_sha(path) == digest,
                'Completed artifact checksum differs: ' + name)
    check_artifact_inventory(folder, record['artifact_sha256'])
    from research.taskcard12.selection import verify_pool_files
    pool = verify_pool_files(folder / 'pool')
    pool_sources = dict(pool['metadata']['identity']['source_sha256'])
    pool_sources.update({'doc/algorithm_versions/v5.2-search-final/src/' + name: digest
                         for name, digest in pool['metadata']['identity']['core_sha256'].items()})
    require(all(plan['protocol']['source_files'].get(name) == digest for name, digest in pool_sources.items()),
            'Saved pool source identity differs from current frozen plan')
    from research.taskcard10.cv import native_options
    from research.taskcard12.selection import normalized_options
    require(pool['metadata']['native_options'] == normalized_options(native_options(request['config'])),
            'Saved pool options differ from frozen request configuration')
    identity = read_json(folder / 'input_identity.json')
    require(identity['input_fingerprints'] == record['input_fingerprints'], 'Saved input identity differs')
    require(pool['metadata']['input_identity'] == {'X': record['input_fingerprints']['X_train'],
                                                  'y': record['input_fingerprints']['y_train']},
            'Cached candidate input identity differs')
    summary = read_json(folder / 'selection.json')
    from research.taskcard11.rescore import select_candidates
    coefficients = {}
    with np.load(folder / 'selected_models.npz', allow_pickle=False) as arrays:
        for label, rule in (('A0', 'original'), ('A1', 'fs')):
            row = summary[label]
            recovered = select_candidates(pool['metadata']['trace']['candidates'], pool['metadata']['n'],
                                          pool['metadata']['dimensions'], finite_sample=(rule == 'fs'))
            require(row['selection'] == recovered and row['score'] == recovered['selected_score'] and
                    row['full_candidate_index'] == recovered['selected_index'], 'Saved selection differs from independent rescore')
            B = arrays[label + '_B']
            coefficients[label] = B
            index = row['selection']['selected_index']
            candidate = pool['metadata']['trace']['candidates'][index]
            require(all(row[k] == candidate[k] == record['metrics'][label][k]
                        for k in ('q', 'sizes', 'rss', 'support', 'B_sha256')), 'Selected metric/pool identity differs')
            require(array_sha(B) == row['B_sha256'] == record['metrics'][label]['B_sha256'], 'Selected coefficient identity differs')
            require(np.array_equal(B, pool['coefficients'][index]), 'Selected coefficient differs from shared pool')
            require(row['support'] == pool['metadata']['trace']['candidates'][index]['support'], 'Selected support differs from pool')
            require(row['q'] == int(np.prod(row['sizes'])) and row['support'] == record['metrics'][label]['support'],
                    'Selected model support/q differs')
    audit_saved_metrics(folder, request, record, pool, summary, coefficients)
    return record


def worker(repo, output, phase, task_id):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = verify_plan(repo, output)
    if phase == 'formal':
        require_gate(repo, output)
    request = next(r for r in plan['smoke_requests' if phase == 'smoke' else 'requests'] if r['task_id'] == task_id)
    job = slurm_identity(os.environ, task_id)
    folder = output / phase / f'task_{task_id:03d}'
    plan_sha = file_sha(output / 'plan.json')
    existing = completed_result(folder, request, plan, plan_sha)
    if existing is not None:
        print('VERIFIED', phase, task_id, 'skipped', flush=True)
        return existing
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'request.json', request)
    write_json(folder / 'started.json', {'slurm': job, 'plan_sha256': plan_sha,
               'started_utc': datetime.now(timezone.utc).isoformat()})
    print('START', phase, task_id, 'job', job['job_id'], flush=True)
    command = [sys.executable, '-u', '-B', '-m', 'research.taskcard12.simulation', 'case',
               '--output', str(output), '--phase', phase, '--task-id', str(task_id)]
    try:
        outcome = run_owned_child(command, folder, plan['protocol']['case_timeout_seconds'], cwd=repo)
        write_json(folder / 'execution.json', outcome)
        require(outcome['passed'], 'Scientific case failed or exceeded frozen timeout')
        scientific = read_json(folder / 'case_result.json')
        verify_plan(repo, output)
        record = {**scientific, 'request': request, 'slurm': job, 'plan_sha256': plan_sha,
                  'source_files': plan['protocol']['source_files'], 'environment': plan['environment'],
                  'resources': {**scientific['resources'], 'controller_wall_seconds': outcome['wall_seconds']},
                  'artifact_sha256': artifact_manifest(folder),
                  'completed_utc': datetime.now(timezone.utc).isoformat()}
        write_json(folder / 'complete.json', record)
        completed_result(folder, request, plan, plan_sha)
        print('COMPLETE', phase, task_id, flush=True)
        return record
    except BaseException as error:
        write_json(folder / 'failed.json', {'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'failed_utc': datetime.now(timezone.utc).isoformat()})
        raise


def scientific_case(repo, output, phase, task_id):
    """Invoked only by the bounded owning worker; test generation follows freeze."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = verify_plan(repo, output)
    if phase == 'formal':
        require_gate(repo, output)
    folder = output / phase / f'task_{task_id:03d}'
    request = next(r for r in plan['smoke_requests' if phase == 'smoke' else 'requests'] if r['task_id'] == task_id)
    require(read_json(folder / 'request.json') == request, 'Owned case request differs')
    started_record = read_json(folder / 'started.json')
    require(started_record['plan_sha256'] == file_sha(output / 'plan.json') and
            started_record['slurm'] == slurm_identity(os.environ, task_id), 'Owned case identity differs')
    require(not (folder / 'case_result.json').exists(), 'Scientific case already executed')
    import inspect
    from research.taskcard12.selection import configure_core, fit_pair, save_pool
    source = Path(inspect.getfile(configure_core())).resolve().parents[2]
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset, iter_test_batches
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from research.taskcard10.cv import native_options
    from research.taskcard10.experiment import _actual_imports, _stream_digest, peak_memory_bytes
    values = dict(request['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    started, cpu_started = time.monotonic(), time.process_time()
    data = generate_simulation_dataset(TensorSimulationConfig(**values), seed=request['seed'])
    fingerprints = {name: array_sha(getattr(data, name)) for name in ('X_train', 'y_train', 'B_true')}
    fingerprints['supports_true'] = [array_sha(s) for s in data.supports_true]
    known_inputs = validate_request_artifacts(repo, plan['protocol'])
    claim = claim_input(output, fingerprints, request, known_inputs,
                        alternate_pairs=[(_stream_digest(data.X_train), _stream_digest(data.y_train))])
    write_json(folder / 'input_identity.json', {'input_fingerprints': fingerprints, 'claim_sha256': file_sha(claim)})
    pair = fit_pair(data.X_train, data.y_train, **native_options(request['config']))
    save_pool(folder / 'pool', pair['pool'])
    selected = {label: pair[rule] for label, rule in (('A0', 'original'), ('A1', 'fs'))}
    summary = {label: {k: v for k, v in model.items() if k != 'B'} for label, model in selected.items()}
    write_json(folder / 'selection.json', summary)
    write_npz(folder / 'selected_models.npz', {label + '_B': model['B'] for label, model in selected.items()})
    write_json(folder / 'independent_audit.json', pair['pool']['metadata']['audit'])
    predictions = []
    metrics, test_identity = evaluate_test_stream(iter_test_batches(data, seed=request['test_seed']),
                                                 {label: model['B'] for label, model in selected.items()}, predictions)
    write_npz(folder / 'test_predictions.npz', {
        **{key: np.concatenate([p[key] for p in predictions], axis=0) for key in ('y', 'signal', 'prediction')},
        'arm_order': np.asarray(list(selected)),
        'model_sha256': np.asarray([model['B_sha256'] for model in selected.values()])})
    true_support = [s.astype(int).tolist() for s in data.supports_true]
    true_sizes = list(map(len, true_support))
    true_q = int(np.prod(true_sizes))
    for label, model in selected.items():
        supports = [np.asarray(s, dtype=int) for s in model['support']]
        metrics[label].update(compute_coefficient_metrics(model['B'], data.B_true))
        metrics[label].update(compute_support_metrics(*supports, *data.supports_true))
        metrics[label].update({k: model[k] for k in ('support', 'sizes', 'q', 'rss', 'B_sha256')})
        metrics[label].update(q_under=bool(model['q'] < true_q), q_over=bool(model['q'] > true_q),
                              coefficient_norm=float(np.linalg.norm(model['B'])))
    write_npz(folder / 'truth.npz', {'B_true': data.B_true, **{f'S{i+1}': s for i, s in enumerate(data.supports_true)}})
    trace = pair['pool']['metadata']['trace']
    recovery = {'q_true': true_q, 'truth_sizes': true_sizes, 'truth_support': true_support,
                'Q_native': int(np.floor(.7 * len(data.y_train))),
                'truth_shape_native_representable': bool(true_q <= np.floor(.7 * len(data.y_train)) and
                    all(s <= c for s, c in zip(true_sizes, trace['caps']))),
                'truth_shape_found_full_pool': any(r['sizes'] == true_sizes for r in trace['candidates']),
                'truth_support_found_full_pool': any(r['support'] == true_support for r in trace['candidates'])}
    record = {'metrics': metrics, 'input_fingerprints': fingerprints, 'test_fingerprint': test_identity,
              'recovery_evidence': recovery, 'actual_core_imports': _actual_imports(source),
              'resources': {'wall_seconds': time.monotonic() - started,
                            'cpu_seconds': time.process_time() - cpu_started,
                            'peak_working_set_bytes': peak_memory_bytes()},
              'search_calls': 1, 'release_approved': False}
    verify_plan(repo, output)
    write_json(folder / 'case_result.json', record)


def collect(output, phase):
    output = Path(output)
    plan = read_json(output / 'plan.json')
    requests = plan['smoke_requests' if phase == 'smoke' else 'requests']
    paths = sorted((output / phase).glob('task_*'))
    expected = [output / phase / f"task_{r['task_id']:03d}" for r in requests]
    require(paths == expected and bool(requests), 'Missing/duplicate/unexpected complete task coverage')
    digest = file_sha(output / 'plan.json')
    return [completed_result(folder, request, plan, digest) for folder, request in zip(paths, requests)]


def _group(records):
    result = {'count': len(records)}
    for label in ('A0', 'A1'):
        pmse = np.asarray([r['metrics'][label]['pmse'] for r in records])
        require(len(pmse) > 0 and np.isfinite(pmse).all() and (pmse > 0).all(), 'Invalid paired PMSE')
        result[label] = {'pmse_mean': float(pmse.mean()), 'pmse_median': float(np.median(pmse)),
                         'pmse_p95': float(np.quantile(pmse, .95)),
                         **{key + '_mean': float(np.mean([r['metrics'][label][key] for r in records]))
                            for key in ('ree', 'element_f1', 'exact_recovery', 'q')},
                         'exact_numerator': sum(int(r['metrics'][label]['exact_recovery']) for r in records),
                         'q_under_numerator': sum(r['metrics'][label]['q'] < np.prod(r['request']['config']['sparsity']) for r in records),
                         'q_over_numerator': sum(r['metrics'][label]['q'] > np.prod(r['request']['config']['sparsity']) for r in records)}
        result[label]['q_under_numerator'] = int(result[label]['q_under_numerator'])
        result[label]['q_over_numerator'] = int(result[label]['q_over_numerator'])
    a = np.asarray([r['metrics']['A0']['pmse'] for r in records])
    b = np.asarray([r['metrics']['A1']['pmse'] for r in records])
    tolerance = 1e-12 * np.maximum(1., np.maximum(np.abs(a), np.abs(b)))
    result.update(pmse_mean_ratio=float(b.mean() / a.mean()), paired_pmse_difference=float(np.mean(b - a)),
                  improved=int(np.sum(b - a < -tolerance)), unchanged=int(np.sum(abs(b - a) <= tolerance)),
                  worsened=int(np.sum(b - a > tolerance)), worst_case_pmse_ratio=float(np.max(b / a)),
                  original_exact_losses=sum(r['metrics']['A0']['exact_recovery'] == 1 and
                      r['metrics']['A1']['exact_recovery'] == 0 for r in records),
                  original_exact_gains=sum(r['metrics']['A0']['exact_recovery'] == 0 and
                      r['metrics']['A1']['exact_recovery'] == 1 for r in records))
    return result


def summarize(records):
    validate_requests([r['request'] for r in records])
    balanced = [r for r in records if not r['request']['pressure']]
    pressure = [r for r in records if r['request']['pressure']]
    result = {'balanced360': _group(balanced), 'pressure_extra20': _group(pressure),
              'pressure_all30': _group([r for r in records if r['request']['cell_id'] == 21]),
              'strata': {}, 'cells': {}}
    for n in (100, 200, 400):
        result['strata']['n' + str(n)] = _group([r for r in balanced if r['request']['config']['n_train'] == n])
    for snr in (.5, 2., 4.):
        result['strata']['snr' + str(snr)] = _group([r for r in balanced if r['request']['config']['snr'] == snr])
    for rho in (0., .6):
        result['strata']['rho' + str(rho)] = _group([r for r in balanced if r['request']['config']['rho'][0] == rho])
    for cell in range(36):
        result['cells'][str(cell)] = _group([r for r in balanced if r['request']['cell_id'] == cell])
    for key in ('truth_shape_native_representable', 'truth_support_found_full_pool'):
        for value in (False, True):
            subset = [r for r in balanced if bool(r['recovery_evidence'][key]) == value]
            result['strata'][key + '=' + str(value)] = _group(subset) if subset else {'count': 0}
    result['worst_complete_cell_pmse_ratio'] = max(v['pmse_mean_ratio'] for v in result['cells'].values())
    result['interpretation'] = 'Descriptive summaries of frozen 360 balanced cases; extra20 pressure cases never reweight the primary mean'
    return result


def _write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def merge(repo, output):
    output = Path(output)
    plan = verify_plan(repo, output)
    require_gate(repo, output)
    records = collect(output, 'formal')
    summary = summarize(records)
    from research.taskcard12.analysis import analyze
    inference = analyze(records, plan['protocol'])
    result_dir = output / 'merged'
    result_dir.mkdir(exist_ok=False)
    raw, changes = [], []
    for record in records:
        request = record['request']
        identity = {k: request[k] for k in ('task_id', 'cell_id', 'replicate', 'pressure', 'seed', 'test_seed')}
        identity.update({k: request['config'][k] for k in ('n_train', 'snr', 'sparsity', 'rho')})
        choice = read_json(output / 'formal' / f"task_{request['task_id']:03d}" / 'selection.json')
        for label in ('A0', 'A1'):
            raw.append({**identity, 'arm': label, **record['metrics'][label], **record['recovery_evidence']})
        changes.append({**identity, 'A0_q': choice['A0']['q'], 'A1_q': choice['A1']['q'],
                        'A0_choice': choice['A0'], 'A1_choice': choice['A1'],
                        'pmse_difference': record['metrics']['A1']['pmse'] - record['metrics']['A0']['pmse'],
                        'original_exact_lost': record['metrics']['A0']['exact_recovery'] == 1 and
                                               record['metrics']['A1']['exact_recovery'] == 0})
    _write_csv(result_dir / 'paired_metrics.csv', raw)
    _write_csv(result_dir / 'selection_changes.csv', changes)
    write_json(result_dir / 'summary.json', summary)
    write_json(result_dir / 'statistical_analysis.json', inference)
    write_json(result_dir / 'integrity.json', {'passed': True, 'formal_cases': 380, 'paired_metric_rows': 760,
        'balanced_cases': 360, 'pressure_extra_cases': 20, 'plan_sha256': file_sha(output / 'plan.json'),
        'protocol_sha256': plan['protocol_sha256'], 'complete_sha256': {
            str(r['request']['task_id']): file_sha(output / 'formal' / f"task_{r['request']['task_id']:03d}" / 'complete.json')
            for r in records}, 'merged_artifact_sha256': artifact_manifest(result_dir), 'release_approved': False})
    return summary


def status(output):
    output = Path(output)
    plan = read_json(output / 'plan.json')
    result = {}
    for phase, key in (('smoke', 'smoke_requests'), ('formal', 'requests')):
        absent, complete, partial = [], [], []
        for row in plan[key]:
            folder = output / phase / f"task_{row['task_id']:03d}"
            target = absent if not folder.exists() else complete if (folder / 'complete.json').is_file() else partial
            target.append(row['task_id'])
        result[phase] = {'complete_markers': complete, 'missing_ids': absent, 'partial_or_running': partial,
                         'note': 'Markers are not integrity verification; only collect/merge checks every saved identity'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'prepare', 'worker', 'case', 'gate', 'check-gate', 'status', 'merge'))
    parser.add_argument('--output', type=Path, default=Path('numerical_simulation/outputs_msic_pair_task12'))
    parser.add_argument('--protocol', type=Path, default=Path('doc/taskcard12/CLUSTER_PROTOCOL.json'))
    parser.add_argument('--phase', choices=('smoke', 'formal'))
    parser.add_argument('--task-id', type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.command in ('worker', 'case'):
        require(args.phase is not None and args.task_id is not None, 'Explicit phase and task ID required')
        {'worker': worker, 'case': scientific_case}[args.command](repo, args.output, args.phase, args.task_id)
    elif args.command == 'verify':
        verify_protocol(repo, args.protocol)
        print('SOURCE_AND_ENVIRONMENT_VERIFIED', flush=True)
    elif args.command == 'prepare':
        prepare(repo, args.output, args.protocol)
        print('PREPARED; scientific inputs NOT_RUN', flush=True)
    elif args.command == 'status':
        print(json.dumps(status(args.output), ensure_ascii=False, indent=2))
    else:
        {'gate': create_gate, 'check-gate': require_gate, 'merge': merge}[args.command](repo, args.output)
        print(args.command.upper(), 'PASSED', flush=True)


if __name__ == '__main__':
    main()
