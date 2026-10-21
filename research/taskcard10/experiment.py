"""Bounded Task10 controller and isolated workers; research artifacts only.

Prepare requires a root-frozen protocol. No science is executed at import time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def write_npz(path, arrays):
    import numpy as np
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(repo, identity):
    repo = Path(repo).resolve()
    for name, digest in identity.items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or file_sha(path) != digest:
            raise ValueError('Frozen source fingerprint mismatch: ' + name)


def build_requests(template, cohort, replicates, seed_base=None):
    bases = {'pilot': 3101810000, 'development': 3501810000, 'confirmation': 3901810000}
    base = bases[cohort] if seed_base is None else int(seed_base)
    if cohort == 'pilot':
        cells = [(n, snr, (4, 3, 5), .6) for n in (100, 400) for snr in (.5, 4.)]
    else:
        cells = [(n, snr, shape, rho) for n in (100, 400) for snr in (.5, 2., 4.)
                 for shape in ((4, 3, 5), (2, 3, 4)) for rho in (0., .6)]
    result = []
    for cell, (n, snr, shape, rho) in enumerate(cells):
        for replicate in range(int(replicates)):
            config = dict(template, n_train=n, snr=snr, sparsity=list(shape), rho=[rho] * 3)
            seed = base + cell * 1000 + replicate
            result.append({'cohort': cohort, 'task_id': len(result), 'cell_id': cell,
                           'replicate': replicate, 'config': config, 'seed': seed,
                           'test_seed': seed + 50_000_000, 'split_seed': seed + 100_000_000})
    return result


def _seed_values(body):
    if isinstance(body, dict):
        for key, value in body.items():
            if key in ('seed', 'test_seed', 'split_seed') and isinstance(value, int):
                yield value
            yield from _seed_values(value)
    elif isinstance(body, list):
        for value in body:
            yield from _seed_values(value)


def audit_seed_novelty(repo, requests, extra_plans=()):
    repo = Path(repo)
    new = [r[k] for r in requests for k in ('seed', 'test_seed', 'split_seed')]
    if len(new) != len(set(new)):
        raise ValueError('New request seed collision')
    paths = set((repo / 'numerical_simulation').glob('outputs*/**/plan.json'))
    paths.update(Path(p) for p in extra_plans)
    historical = set()
    manifests = {}
    for path in sorted(paths):
        historical.update(_seed_values(read_json(path)))
        manifests[str(path.relative_to(repo))] = file_sha(path)
    overlap = sorted(historical.intersection(new))
    if overlap:
        raise ValueError('New request overlaps historical materialized seed: ' + str(overlap))
    return {'historical_plan_sha256': manifests, 'historical_unique_seeds': len(historical),
            'new_unique_seeds': len(new), 'overlap': overlap,
            'scope': 'Materialized seed/test_seed/split_seed fields, including prior Task10 plans; external unpublished seeds unknown'}


def environment():
    import numpy as np
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'executable': sys.executable, 'platform': platform.platform(),
            'threads': {key: os.environ.get(key) for key in
                        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def source_identity(repo, protocol):
    names = dict(protocol['source_files'])
    prefix = 'doc/algorithm_versions/v5.2-search-final/src/'
    names.update({prefix + name: digest for name, digest in protocol['core_sha256'].items()})
    verify_sources(repo, names)
    return names


def prepare(repo, output, protocol_path, stage):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    protocol = read_json(protocol_path)
    if protocol['stage'] != stage:
        raise ValueError('Frozen protocol stage mismatch')
    source_identity(repo, protocol)
    if environment() != protocol['environment']:
        raise ValueError('Frozen environment mismatch')
    template_path = repo / 'numerical_simulation/outputs_selection_research_task8/plan.json'
    template = read_json(template_path)['requests'][0]['config']
    cohorts = ('pilot',) if stage == 'pilot' else ('development', 'confirmation')
    budget = protocol[stage]
    requests = [r for cohort in cohorts for r in
                build_requests(template, cohort, budget['replicates'], protocol['seed_bases'][cohort])]
    if protocol['seed_offsets'] != {'test': 50_000_000, 'split': 100_000_000}:
        raise ValueError('Seed offset protocol differs from frozen request generator')
    prior = [p for p in (repo / 'doc/taskcard10/results').glob('*/plan.json') if p.parent != output]
    novelty = audit_seed_novelty(repo, requests, prior)
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema_version': 1, 'created_utc': utc_now(), 'stage': stage,
            'research_only': True, 'release_approved': False,
            'protocol': protocol, 'protocol_sha256': file_sha(protocol_path),
            'protocol_path': str(Path(protocol_path).resolve().relative_to(repo)),
            'template_sha256': file_sha(template_path), 'requests': requests,
            'historical_seed_audit': novelty,
            'historical_input_index': historical_input_index(repo)}
    write_json(output / 'plan.json', plan)
    return plan


def artifact_manifest(folder):
    return {p.relative_to(folder).as_posix(): file_sha(p)
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def confirmation_lock(output):
    output = Path(output)
    path = output / 'development_lock.json'
    if not path.is_file():
        raise ValueError('Missing complete development lock')
    lock = read_json(path)
    if lock['plan_sha256'] != file_sha(output / 'plan.json'):
        raise ValueError('Development lock plan fingerprint mismatch')
    if not lock.get('entry_gate', {}).get('passed', False):
        raise ValueError('Development entry gate failed; confirmation must remain NOT_RUN')
    expected = lock['development_artifacts']
    if artifact_manifest(output / 'development') != expected:
        raise ValueError('Development lock artifact fingerprint mismatch')
    plan = read_json(output / 'plan.json')
    planned = len([r for r in plan['requests'] if r['cohort'] == 'development'])
    completed = len(list((output / 'development').glob('task_*/complete.json')))
    if not planned or completed != planned or lock.get('completed_development_tasks') != planned:
        raise ValueError('Development lock is incomplete')
    return file_sha(path)


def _windows_job():
    """Create one owned kill-on-close job, never touching unrelated processes."""
    import ctypes
    from ctypes import wintypes
    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                    ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount',
                    'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateJobObjectW
    create.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    create.restype = wintypes.HANDLE
    job = create(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    set_limits = kernel.SetInformationJobObject
    set_limits.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    set_limits.restype = wintypes.BOOL
    if not set_limits(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(wintypes.HANDLE(job))
        raise error
    return job


def _close_job(process):
    job = getattr(process, '_taskcard10_job', None)
    if job is not None:
        import ctypes
        from ctypes import wintypes
        close = ctypes.WinDLL('kernel32', use_last_error=True).CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        if not close(job):
            raise ctypes.WinError(ctypes.get_last_error())
        process._taskcard10_job = None


def _launch_owned(command, cwd, env, stdout, stderr):
    if os.name != 'nt':
        return subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    import ctypes
    from ctypes import wintypes
    job = _windows_job()
    process = None
    try:
        # Suspended creation closes the race in which a venv launcher spawns
        # its real interpreter before membership in our job is established.
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   creationflags=0x00000004)  # CREATE_SUSPENDED
        process._taskcard10_job = job
        assign = ctypes.WinDLL('kernel32', use_last_error=True).AssignProcessToJobObject
        assign.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        assign.restype = wintypes.BOOL
        if not assign(job, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        resume = ctypes.WinDLL('ntdll').NtResumeProcess
        resume.argtypes = [wintypes.HANDLE]
        resume.restype = ctypes.c_long
        status = resume(wintypes.HANDLE(int(process._handle)))
        if status != 0:
            raise OSError('NtResumeProcess failed for owned suspended worker: ' + hex(status & 0xffffffff))
        return process
    except BaseException:
        if process is not None:
            process.kill(); process.wait()
            _close_job(process)
        else:
            close = ctypes.WinDLL('kernel32', use_last_error=True).CloseHandle
            close.argtypes = [wintypes.HANDLE]
            close(job)
        raise


def _terminate_process_tree(process):
    if os.name == 'nt':
        if getattr(process, '_taskcard10_job', None) is None:
            raise RuntimeError('Worker lacks its required owned process job')
        _close_job(process)  # Closing the sole handle terminates all descendants.
    else:
        process.kill()
    return process.wait()


def run_subprocess_batch(commands, output, workers, total_seconds, case_seconds, cwd=None):
    """One exclusive attempt; first failure/timeout stops dependent launches."""
    output = Path(output)
    if not 1 <= workers <= 4 or min(total_seconds, case_seconds) <= 0:
        raise ValueError('Invalid frozen worker/budget bounds')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / 'started.json', {'started_utc': utc_now(), 'workers': workers,
               'total_seconds': total_seconds, 'case_seconds': case_seconds,
               'task_ids': [c['task_id'] for c in commands]})
    pending = list(commands)
    running = []
    completed, failed = [], []
    child_env = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        child_env[key] = '1'
    child_env['PYTHONIOENCODING'] = 'utf-8'
    while pending or running:
        while pending and len(running) < workers and not failed and time.monotonic() - started < total_seconds:
            task = pending.pop(0)
            folder = output / f"task_{task['task_id']:03d}"
            folder.mkdir()
            write_json(folder / 'command.json', task)
            stdout = (folder / 'stdout.log').open('xb')
            stderr = (folder / 'stderr.log').open('xb')
            try:
                process = _launch_owned(task['command'], cwd, child_env, stdout, stderr)
            except OSError as error:
                stdout.close(); stderr.close()
                record = {'task_id': task['task_id'], 'reason': 'launch_failed',
                          'error_type': type(error).__name__, 'error': str(error)}
                failed.append(record)
                write_json(folder / 'failed.json', record)
                break
            running.append({'task': task, 'process': process, 'started': time.monotonic(),
                            'stdout': stdout, 'stderr': stderr, 'folder': folder})
        now = time.monotonic()
        for item in list(running):
            code = item['process'].poll()
            reason = None
            if code is not None:
                reason = 'nonzero_exit' if code != 0 else None
            elif now - started >= total_seconds:
                reason = 'total_timeout'
            elif now - item['started'] >= case_seconds:
                reason = 'case_timeout'
            else:
                continue
            if code is None:
                code = _terminate_process_tree(item['process'])
            else:
                _close_job(item['process'])
            item['stdout'].close(); item['stderr'].close()
            running.remove(item)
            record = {'task_id': item['task']['task_id'], 'returncode': code,
                      'elapsed_seconds': time.monotonic() - item['started']}
            if reason:
                record['reason'] = reason
                failed.append(record)
                write_json(item['folder'] / 'failed.json', record)
            else:
                completed.append(item['task']['task_id'])
                write_json(item['folder'] / 'complete.json', record)
        if failed or (pending and not running and now - started >= total_seconds):
            for item in running:
                _terminate_process_tree(item['process'])
                item['stdout'].close(); item['stderr'].close()
                failed.append({'task_id': item['task']['task_id'], 'reason': 'aborted_after_peer_failure'})
            running.clear()
            break
        if running:
            time.sleep(.05)
    result = {'status': 'FAILED' if failed or pending else 'COMPLETE', 'completed': sorted(completed),
              'failed': failed, 'not_started': [t['task_id'] for t in pending],
              'elapsed_seconds': time.monotonic() - started, 'finished_utc': utc_now()}
    write_json(output / ('completion.json' if result['status'] == 'COMPLETE' else 'stopped.json'), result)
    return result


def _stream_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(value.shape).encode('ascii'))
    digest.update(value.dtype.str.encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate_test_stream(batches, coefficients, saved_predictions=None):
    """After selection only; every frozen selected B sees the same test batches."""
    import numpy as np
    labels = list(coefficients)
    matrix = np.stack([coefficients[label].ravel() for label in labels], axis=1)
    sse, signal_sse = np.zeros(len(labels)), np.zeros(len(labels))
    digest = hashlib.sha256()
    n = n_batches = 0
    for X, y, signal in batches:
        for value in (X, y, signal):
            digest.update(_stream_digest(value).encode('ascii'))
        prediction = X.reshape(len(y), -1) @ matrix
        if saved_predictions is not None:
            saved_predictions.append({'y': y.copy(), 'signal': signal.copy(), 'prediction': prediction.copy()})
        sse += np.sum((y[:, None] - prediction) ** 2, axis=0)
        signal_sse += np.sum((signal[:, None] - prediction) ** 2, axis=0)
        n += len(y); n_batches += 1
    if n == 0 or not np.isfinite(sse).all():
        raise ValueError('Empty/nonfinite shared test evaluation')
    rows = {label: {'pmse': float(sse[j] / n), 'signal_mse': float(signal_sse[j] / n)}
            for j, label in enumerate(labels)}
    return rows, {'sha256': digest.hexdigest(), 'n_observations': n, 'n_batches': n_batches}


def summarize_recovery(summary, supports_true, n):
    import numpy as np
    sizes = list(map(len, supports_true))
    q = int(np.prod(sizes))
    capacity = summary['public_capacity']
    full = summary['full_trace']['candidates']
    common = [full[row['full_candidate_index']] for row in summary['cv_curve']]
    truth = [list(map(int, support)) for support in supports_true]
    return {'q_true': q, 'Q_native': int(np.floor(.7 * n)), 'Q_common': capacity['common_Q'],
            'truth_sizes': sizes, 'truth_support': truth,
            'truth_shape_native_representable': bool(q <= int(np.floor(.7 * n)) and
                 all(s <= cap for s, cap in zip(sizes, capacity['full_caps']))),
            'truth_shape_common_representable': sizes in summary['common_grid'],
            'truth_shape_found_full_pool': any(row['sizes'] == sizes for row in full),
            'truth_shape_found_common_pool': any(row['sizes'] == sizes for row in common),
            'truth_support_found_full_pool': any(row['support'] == truth for row in full),
            'truth_support_found_common_pool': any(row['support'] == truth for row in common),
            'candidate_counts': {'full_including_adaptive': len(full), 'common_fixed': len(common)}}


def peak_memory_bytes():
    if os.name != 'nt':
        import resource
        scale = 1 if sys.platform == 'darwin' else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)
    import ctypes
    from ctypes import wintypes
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    function.restype = wintypes.BOOL
    if not function(current(), ctypes.byref(counters), counters.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return int(counters.PeakWorkingSetSize)


def _actual_imports(source):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Actual core import escaped frozen source: ' + name)
            result[name] = {'path': str(path), 'sha256': file_sha(path)}
    return result


def historical_input_index(repo):
    """Known comparable saved X/y digests; never substitutes for seed novelty."""
    repo = Path(repo)
    paths = list((repo / 'numerical_simulation').glob('outputs*/tasks/task_*.json'))
    paths += list((repo / 'numerical_simulation/outputs_selection_research_task8').glob('*/*/complete.json'))
    paths += list((repo / 'numerical_simulation/outputs_selection_inner_task8').rglob('complete.json'))
    paths += list((repo / 'doc/taskcard10/results').glob('*/*/task_*/complete.json'))
    known = {}
    for path in paths:
        body = read_json(path)
        fp = body.get('input_fingerprints', {})
        if 'X_train' in fp and 'y_train' in fp:
            known[(fp['X_train'], fp['y_train'])] = str(path.relative_to(repo))
    return [{'X_train': key[0], 'y_train': key[1], 'path': name} for key, name in sorted(known.items())]


def worker(repo, output, cohort, task_id):
    import numpy as np
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = read_json(output / 'plan.json')
    protocol = plan['protocol']
    source_identity(repo, protocol)
    if environment() != protocol['environment']:
        raise ValueError('Worker environment differs from frozen protocol')
    if file_sha(repo / plan['protocol_path']) != plan['protocol_sha256']:
        raise ValueError('Protocol artifact drift')
    lock_sha = confirmation_lock(output) if cohort == 'confirmation' else None
    request = next(r for r in plan['requests'] if r['cohort'] == cohort and r['task_id'] == task_id)
    folder = output / cohort / f'task_{task_id:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'request.json', request)
    write_json(folder / 'started.json', {'started_utc': utc_now(), 'plan_sha256': file_sha(output / 'plan.json'),
               'development_lock_sha256': lock_sha})
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        from scripts.taskcard7.lemon import configure_core, array_sha
        source = configure_core('v5.2-search-final')
        # Preload the complete required solver graph before numerical_simulation's
        # historical package init can insert the production src path.
        import tensor_abess.tensor.abess
        from numerical_simulation.data_generation import (TensorSimulationConfig,
            generate_simulation_dataset, iter_test_batches)
        from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
        from research.taskcard10.cv import run_three_arms
        _actual_imports(source)
        values = dict(request['config'])
        for key in ('dims', 'sparsity', 'rho'):
            values[key] = tuple(values[key])
        data = generate_simulation_dataset(TensorSimulationConfig(**values), seed=request['seed'])
        fingerprints = {name: array_sha(getattr(data, name)) for name in ('X_train', 'y_train', 'B_true')}
        fingerprints['supports_true'] = [array_sha(s) for s in data.supports_true]
        known_rows = plan['historical_input_index'] + historical_input_index(repo)
        known = {(r['X_train'], r['y_train']): r['path'] for r in known_rows}
        pairs = [(fingerprints['X_train'], fingerprints['y_train']),
                 (_stream_digest(data.X_train), _stream_digest(data.y_train))]
        for pair in pairs:
            if pair in known:
                raise ValueError('Full training input duplicates historical/Pilot/other Task10 input: ' + known[pair])
        write_json(folder / 'input_identity.json', {'input_fingerprints': fingerprints,
                   'known_comparable_inputs_checked': len(known), 'duplicate': False})
        budget = protocol['pilot' if cohort == 'pilot' else 'formal']
        fit = run_three_arms(data.X_train, data.y_train, request['config'], request['split_seed'],
                             folder / 'cache', protocol['source_identity'], preproc='identity',
                             deadline_monotonic=started + budget['case_timeout_seconds'] - 2.)
        summary, arrays = fit['summary'], fit['arrays']
        from research.taskcard10.audit import audit_run
        audit_started = time.monotonic()
        independent_audit = audit_run(data.X_train, data.y_train, summary, arrays,
                                      repository=repo, cache_dir=folder / 'cache')
        audit_seconds = time.monotonic() - audit_started
        write_json(folder / 'independent_audit.json', independent_audit)
        # Persist the complete selection before generating the external test stream.
        write_json(folder / 'selection_frozen.json', summary)
        coefficients = {label: arrays[summary['arms'][label]['B_key']] for label in ('A0', 'A1', 'A2')}
        prediction_batches = []
        metrics, test_fingerprint = evaluate_test_stream(
            iter_test_batches(data, seed=request['test_seed']), coefficients, prediction_batches)
        write_npz(folder / 'test_predictions.npz', {
            name: np.concatenate([batch[name] for batch in prediction_batches], axis=0)
            for name in ('y', 'signal', 'prediction')})
        for label, B in coefficients.items():
            support = tuple(np.asarray(s, dtype=int) for s in summary['arms'][label]['support'])
            metrics[label].update(compute_coefficient_metrics(B, data.B_true))
            metrics[label].update(compute_support_metrics(*support, *data.supports_true))
            metrics[label]['sizes'] = list(map(len, support))
            metrics[label]['q'] = int(np.prod(metrics[label]['sizes']))
            metrics[label]['B_sha256'] = array_sha(B)
            metrics[label]['rss'] = summary['arms'][label]['rss']
            metrics[label]['support'] = summary['arms'][label]['support']
            metrics[label]['coefficient_norm'] = float(np.linalg.norm(B))
        arrays = dict(arrays)
        write_npz(folder / 'arrays.npz', arrays)
        write_npz(folder / 'truth.npz', {'B_true': data.B_true, **{
            f'S{i + 1}_true': s for i, s in enumerate(data.supports_true)}})
        write_json(folder / 'summary.json', summary)
        elapsed, cpu = time.monotonic() - started, time.process_time() - cpu_started
        timings = summary['timings']
        a0 = timings['A0_seconds']
        program_cost = {'A0': a0, 'A1': a0 + timings['A1_selection_seconds'],
                        'A2': a0 + timings['inner_seconds'] + timings['A2_selection_seconds']}
        complete = {'request': request, 'plan_sha256': file_sha(output / 'plan.json'),
                    'protocol_sha256': plan['protocol_sha256'], 'development_lock_sha256': lock_sha,
                    'input_fingerprints': fingerprints, 'test_fingerprint': test_fingerprint,
                    'metrics': metrics, 'actual_core_imports': _actual_imports(source),
                    'source_identity': protocol['source_identity'], 'environment': environment(),
                    'resources': {'wall_seconds': elapsed, 'cpu_seconds': cpu,
                                  'peak_working_set_bytes': peak_memory_bytes(),
                                  'independent_audit_seconds': audit_seconds},
                    'recovery_evidence': summarize_recovery(summary, data.supports_true, len(data.y_train)),
                    'measured_program_wall_seconds': program_cost,
                    'cost_definition': 'Conservative A1/A2 include complete native full library, plus respective selection and A2 inner fits/predictions; shared generation/evaluation excluded',
                    'artifact_sha256': {name: file_sha(folder / name) for name in
                        ('request.json', 'input_identity.json', 'selection_frozen.json',
                         'summary.json', 'arrays.npz', 'truth.npz', 'independent_audit.json', 'test_predictions.npz')},
                    'completed_utc': utc_now()}
        verify_sources(repo, source_identity(repo, protocol))
        write_json(folder / 'complete.json', complete)
        print('COMPLETE', cohort, task_id, round(elapsed, 3), flush=True)
    except BaseException as error:
        write_json(folder / 'failed.json', {'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'failed_utc': utc_now(),
                   'wall_seconds': time.monotonic() - started})
        raise


def evaluate_gates(records, thresholds):
    import numpy as np
    def mean(label, metric, rows):
        return float(np.mean([r['metrics'][label][metric] for r in rows]))
    def ratio(rows, denominator='A0'):
        return mean('A2', 'pmse', rows) / mean(denominator, 'pmse', rows)
    small = [r for r in records if r['request']['config']['n_train'] == 100]
    large = [r for r in records if r['request']['config']['n_train'] == 400]
    strong = [r for r in records if r['request']['config']['snr'] == 4.]
    cells = sorted({r['request']['cell_id'] for r in records})
    if not small or not large or not strong or len(cells) != 24:
        raise ValueError('All planned 24 cells are required before numeric entry/promotion gates')
    tail = lambda label: float(np.quantile([r['metrics'][label]['pmse'] for r in small], .95))
    under = lambda label: float(np.mean([r['metrics'][label]['q'] <
                                np.prod(r['request']['config']['sparsity']) for r in strong]))
    values = {'n100_pmse_ratio_A2_A0_max': ratio(small),
              'n100_pmse_p95_ratio_A2_A0_max': tail('A2') / tail('A0'),
              'overall_pmse_ratio_A2_A0_max': ratio(records),
              'overall_pmse_ratio_A2_A1_max': ratio(records, 'A1'),
              'worst_complete_cell_pmse_ratio_A2_A0_max': max(
                  ratio([r for r in records if r['request']['cell_id'] == cell]) for cell in cells),
              'n400_exact_rate_delta_A2_A0_min': mean('A2', 'exact_recovery', large) - mean('A0', 'exact_recovery', large),
              'n400_pmse_ratio_A2_A0_max': ratio(large),
              'strong_snr4_q_under_delta_A2_A0_max': under('A2') - under('A0'),
              'measured_program_cost_ratio_A2_A0_max': sum(r['measured_program_wall_seconds']['A2'] for r in records) /
                  sum(r['measured_program_wall_seconds']['A0'] for r in records)}
    if set(values) != set(thresholds):
        raise ValueError('Frozen gate dimensions differ from implementation')
    detail = {key: {'value': value, 'threshold': thresholds[key],
               'passed': bool(value >= thresholds[key] if key.endswith('_min') else value <= thresholds[key])}
               for key, value in values.items()}
    return {'passed': all(row['passed'] for row in detail.values()), 'criteria': detail,
            'interpretation': 'Predeclared descriptive thresholds, limited replication; no confidence-interval noninferiority claim'}


def run_stage(repo, output, cohort):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = read_json(output / 'plan.json')
    protocol = plan['protocol']
    source_identity(repo, protocol)
    budget = protocol['pilot' if cohort == 'pilot' else 'formal']
    if cohort == 'confirmation':
        confirmation_lock(output)
    if cohort == 'development':
        write_json(output / 'formal_budget.json', {'started_utc': utc_now(),
                   'started_epoch': time.time(), 'limit_seconds': budget['wall_time_limit_seconds']})
    remaining = budget['wall_time_limit_seconds']
    if cohort != 'pilot':
        clock = read_json(output / 'formal_budget.json')
        remaining -= time.time() - clock['started_epoch']
    requests = [r for r in plan['requests'] if r['cohort'] == cohort]
    if not requests or remaining <= 0:
        raise ValueError('No requests or frozen global budget exhausted')
    commands = [{'task_id': r['task_id'], 'command': [sys.executable, '-B', '-m',
                 'research.taskcard10.experiment', 'worker', '--output', str(output),
                 '--cohort', cohort, '--task-id', str(r['task_id'])]} for r in requests]
    execution = output / ('execution_' + cohort)
    result = run_subprocess_batch(commands, execution, workers=budget['workers'],
                                  total_seconds=remaining, case_seconds=budget['case_timeout_seconds'], cwd=repo)
    if result['status'] != 'COMPLETE':
        write_json(output / ('stopped_' + cohort + '.json'), result)
        return result
    records = [read_json(output / cohort / f"task_{r['task_id']:03d}" / 'complete.json') for r in requests]
    if cohort == 'pilot':
        write_json(output / 'completion.json', {**result, 'plan_sha256': file_sha(output / 'plan.json'),
                   'completed_cases': len(records), 'case_wall_seconds': [r['resources']['wall_seconds'] for r in records],
                   'maximum_peak_memory_bytes': max(r['resources']['peak_working_set_bytes'] for r in records)})
    elif cohort == 'development':
        gate = evaluate_gates(records, protocol['development_entry_gate'])
        write_json(output / 'development_lock.json', {'plan_sha256': file_sha(output / 'plan.json'),
                   'completed_development_tasks': len(records), 'entry_gate': gate,
                   'development_artifacts': artifact_manifest(output / 'development'), 'locked_utc': utc_now()})
        if not gate['passed']:
            write_json(output / 'confirmation_NOT_RUN.json', {'status': 'NOT_RUN',
                       'reason': 'Frozen development numeric entry gate failed', 'entry_gate': gate})
        else:
            # Budget rule fixed before formal execution: reserve 1.5 times the
            # measured development wall plus 60s for the identical-size cohort.
            required = 1.5 * result['elapsed_seconds'] + 60.
            left = budget['wall_time_limit_seconds'] - (time.time() - clock['started_epoch'])
            write_json(output / 'confirmation_budget_gate.json', {'passed': left >= required,
                       'required_seconds': required, 'remaining_seconds': left,
                       'rule': '1.5*measured development wall+60 seconds'})
            if left < required:
                write_json(output / 'confirmation_NOT_RUN.json', {'status': 'NOT_RUN',
                           'reason': 'Frozen remaining budget cannot cover complete confirmation'})
    else:
        gate = evaluate_gates(records, protocol['promotion_gates'])
        write_json(output / 'confirmation_gate.json', gate)
        write_json(output / 'completion.json', {**result, 'plan_sha256': file_sha(output / 'plan.json'),
                   'completed_confirmation_cases': len(records), 'promotion_gate': gate,
                   'release_approved': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare-pilot', 'prepare-formal', 'run-pilot', 'run-dev', 'run-confirm', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--cohort', choices=('pilot', 'development', 'confirmation'))
    parser.add_argument('--task-id', type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.mode.startswith('prepare-'):
        if args.protocol is None:
            parser.error('A frozen --protocol is required')
        body = prepare(repo, args.output, args.protocol, args.mode.removeprefix('prepare-'))
        print('PREPARED', body['stage'], len(body['requests']), flush=True)
    elif args.mode == 'worker':
        worker(repo, args.output, args.cohort, args.task_id)
    else:
        cohort = {'run-pilot': 'pilot', 'run-dev': 'development', 'run-confirm': 'confirmation'}[args.mode]
        if cohort == 'confirmation':
            if (args.output / 'confirmation_NOT_RUN.json').exists():
                raise ValueError('Confirmation has a terminal NOT_RUN decision')
            gate = read_json(args.output / 'confirmation_budget_gate.json')
            if not gate['passed']:
                raise ValueError('Frozen confirmation budget gate failed')
        body = run_stage(repo, args.output, cohort)
        print('STAGE', cohort, body['status'], flush=True)
        if body['status'] != 'COMPLETE':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
