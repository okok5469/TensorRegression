"""Payload-local Slurm dependency chain; scheduling uses the standard library.

Numerical imports occur only inside Linux job phases. Submission/dry-run,
status and cancellation never import the bundled Linux numerical runtime.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import csv
import hashlib
from itertools import count
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile


def root_path(): return Path(__file__).resolve().parents[2]


def output_path(root): return root / 'numerical_simulation/outputs_taskcard18_closeout'


def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
        handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
    os.replace(temporary, path)


def phase_task_ids(plan, mapping, phase, protocol):
    by_id = {int(row['task_id']): row for row in mapping}
    selected = sorted(plan['tasks'], key=lambda task: task['task_id'])
    if phase == 'pilot':
        selected = [t for t in selected if t['scenario_id'] in protocol['pilot_scenarios'] and t['replicate'] in protocol['pilot_replicates']]
    elif phase != 'formal':
        raise ValueError('Unknown array phase')
    seen, result = set(), []
    for task in selected:
        uid = by_id[task['task_id']]['unique_input_id']
        if uid not in seen:
            seen.add(uid); result.append(task['task_id'])
    return result


def task_inventory(root):
    plan = load(root / 'numerical_simulation/outputs_v52_search_final_full/plan.json')
    protocol = load(root / 'doc/taskcard18/CLOSEOUT_PROTOCOL.json')
    with (root / 'doc/taskcard18/input_identity_map.csv').open(encoding='utf-8-sig', newline='') as stream:
        mapping = list(csv.DictReader(stream))
    pilot = phase_task_ids(plan, mapping, 'pilot', protocol)
    formal = phase_task_ids(plan, mapping, 'formal', protocol)
    if len(pilot) != 20 or len(formal) != 600:
        raise ValueError('Expected20 pilot inputs and600 unique formal inputs')
    return plan, mapping, protocol, pilot, formal


def parse_job_id(value):
    if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', value.strip()):
        raise ValueError('Unexpected sbatch job ID')
    return value.strip().split(';')[0]


def check_submission_state(exists, active, *, resume):
    if exists and not resume:
        raise ValueError('Submission already recorded; use status or resume, never duplicate submit')
    if resume and not exists:
        raise ValueError('No recorded submission to resume')
    if resume and active:
        raise ValueError('Recorded jobs are still active; refusing overlapping resume')


def submission_commands(script, logs):
    phases = [('prepare', None, None), ('pilot', '0-19%64', 'prepare'),
              ('pilotgate', None, 'pilot'), ('formal', '0-599%64', 'pilotgate'), ('collect', None, 'formal')]
    commands = []
    for phase, array, dependency in phases:
        command = ['sbatch', '--parsable', '--job-name=tc18-' + phase,
                   '--chdir=' + str(script.parent),
                   '--output=' + str(logs / (phase + '-%A_%a.out')),
                   '--error=' + str(logs / (phase + '-%A_%a.err'))]
        if array: command.append('--array=' + array)
        if dependency: command.append('--dependency=afterok:{' + dependency + '}')
        commands.append({'phase': phase, 'command': command + [str(script), phase]})
    return commands


def recorded_ids(state):
    return sorted({parse_job_id(job['job_id']) for attempt in state.get('attempts', []) for job in attempt.get('jobs', [])}, key=int)


def active_ids(state):
    ids = recorded_ids(state)
    if not ids: return []
    result = subprocess.run(['squeue', '--noheader', '--jobs=' + ','.join(ids), '--format=%F'], capture_output=True, text=True, check=True)
    returned = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not returned <= set(ids): raise ValueError('Scheduler returned an unowned job ID')
    return sorted(returned, key=int)


def verify_payload(root):
    amendment = load(root / 'doc/taskcard18/EXECUTION_AMENDMENT.json')
    if amendment.get('execution') != 'existing_cluster' or amendment.get('user_authorized') is not True or amendment.get('statistics_unchanged') is not True or amendment.get('original_protocol_sha256') != sha(root / 'doc/taskcard18/CLOSEOUT_PROTOCOL.json'):
        raise ValueError('Explicit cluster execution amendment/protocol binding required')
    manifest = root / 'PAYLOAD_MANIFEST.json'
    if manifest.exists():
        from .bootstrap_runtime import safe_member
        for name, expected in load(manifest)['files_sha256'].items():
            path = root.joinpath(*safe_member(name).parts)
            if root not in path.resolve().parents or sha(path) != expected:
                raise ValueError('Payload file hash/path conflict: ' + name)
    release = load(root / 'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json')
    plan = load(root / 'numerical_simulation/outputs_v52_search_final_full/plan.json')
    expected = dict(release['source_sha256']); expected.update(plan['source_fingerprints'])
    if len(release['source_sha256']) != 32 or len(plan['source_fingerprints']) != 38 or plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Frozen release/plan inventory or seal conflict')
    for name, expected_hash in expected.items():
        if sha(root / name) != expected_hash:
            raise ValueError('Frozen source conflict: ' + name)


def check_fit_pair(metadata_path, failure_path):
    complete = metadata_path.exists()
    if complete != metadata_path.with_suffix('.npz').exists():
        raise ValueError('Incomplete fit artifact pair; preserve and inspect before resume: ' + str(metadata_path))
    if not complete and failure_path.exists():
        raise ValueError('Unresolved immutable fit failure; inspect it before resume: ' + str(failure_path))
    return complete


def next_test_log(logs):
    logs.mkdir(parents=True, exist_ok=True)
    for attempt in count():
        path = logs / f'focused_tests_attempt{attempt:04d}.txt'
        if not path.exists():
            return path


def validate_cache(root):
    from .accept import validate_fit
    from numerical_simulation.known_s_cluster.run import read_task
    plan, mapping, _, _, _ = task_inventory(root)
    by_id = {int(row['task_id']): row for row in mapping}
    directory = output_path(root) / 'fits'
    failures = output_path(root) / 'failures'
    for path in sorted(directory.glob('*.npz')):
        check_fit_pair(path.with_suffix('.json'), failures / path.with_suffix('.json').name)
    count = 0
    for path in sorted(directory.glob('*.json')):
        check_fit_pair(path, failures / path.name)
        metadata = load(path); task_id = metadata['task_id']; method = path.stem.rsplit('_', 1)[-1]
        if metadata['unique_input_id'] != by_id[task_id]['unique_input_id'] or path.stem != metadata['unique_input_id'] + '_' + method:
            raise ValueError('Cached complete-input identity conflict')
        record = read_task(root / 'numerical_simulation/outputs_v52_search_final_full', plan, task_id)
        validate_fit(record['task'], record, method); count += 1
    for failure_path in (output_path(root) / 'failures').glob('*.json'):
        failure = load(failure_path)
        success = directory / (failure['unique_input_id'] + '_' + failure['method'] + '.json')
        if not success.exists():
            raise ValueError('Unresolved immutable fit failure; inspect it before resume: ' + str(failure_path))
    return count


def prepare(root):
    from .bootstrap_runtime import bootstrap
    verify_payload(root)
    bootstrap(root)
    from .workflow import source_identity, write_json, reconstruct_pilot
    plan, mapping, _, _, _ = task_inventory(root)
    by_id = {int(row['task_id']): row for row in mapping}
    output = output_path(root)
    marker = output / 'CLUSTER_PREPARED.json'
    identity = source_identity()
    run = {'plan_id': plan['plan_id'], 'plan_file_sha256': sha(root / 'numerical_simulation/outputs_v52_search_final_full/plan.json'),
           'identity': identity, 'input_map_sha256': sha(root / 'doc/taskcard18/input_identity_map.csv'),
           'namespace': 'input_ + SHA256(canonical JSON of four recorded input fingerprints and test_seed)',
           'tasks': [{'task_id': t['task_id'], 'unique_input_id': by_id[t['task_id']]['unique_input_id'],
                      'config_sha256': digest(t['config']), 'seed': t['seed'], 'test_seed': t['test_seed']} for t in plan['tasks']]}
    run['record_sha256'] = digest(run)
    if marker.exists():
        preserved = load(marker)
        if not re.fullmatch(r'logs/focused_tests_attempt[0-9]+\.txt', preserved['focused_tests_file']):
            raise ValueError('Prepared test evidence path is invalid')
        if preserved['identity'] != identity or preserved['run_sha256'] != sha(output / 'RUN_IDENTITY.json') or load(output / 'RUN_IDENTITY.json') != run or preserved['input_audit_sha256'] != sha(output / 'pilot_input_audit_v2.json') or preserved['focused_tests_sha256'] != sha(output / preserved['focused_tests_file']):
            raise ValueError('Prepared execution identity changed; refusing cache repair')
        print('PREPARED_CACHE_VALID', validate_cache(root), flush=True)
        return
    if (output / 'RUN_IDENTITY.json').exists() or list((output / 'fits').glob('*')):
        raise ValueError('Unprepared output contains old/local numerical state; use a fresh payload output')
    modules = ['test_baselines', 'test_closeout', 'test_accept', 'test_identity', 'test_report', 'test_cluster']
    test_log = next_test_log(output / 'logs')
    with test_log.open('xb') as stream:
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', *['research.taskcard18.' + name for name in modules], '-v'], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Cluster focused tests failed; inspect ' + str(test_log))
    reconstruct_pilot()
    write_json(output / 'RUN_IDENTITY.json', run)
    record = {'identity': identity, 'run_sha256': sha(output / 'RUN_IDENTITY.json'),
              'input_audit_sha256': sha(output / 'pilot_input_audit_v2.json'), 'focused_tests_sha256': sha(test_log),
              'focused_tests_file': test_log.relative_to(output).as_posix(),
              'payload_manifest_sha256': sha(root / 'PAYLOAD_MANIFEST.json') if (root / 'PAYLOAD_MANIFEST.json').exists() else None}
    write_json(marker, record)
    print('CLUSTER_PREPARE_PASS', flush=True)


def require_prepared(root):
    from .workflow import source_identity
    marker = load(output_path(root) / 'CLUSTER_PREPARED.json')
    if marker['identity'] != source_identity() or marker['run_sha256'] != sha(output_path(root) / 'RUN_IDENTITY.json'):
        raise ValueError('Cluster prepare identity gate is stale')


def array_worker(root, phase, index):
    from .workflow import fit_one
    from .accept import validate_fit
    from numerical_simulation.known_s_cluster.run import read_task
    require_prepared(root)
    plan, _, _, pilot, formal = task_inventory(root)
    tasks = pilot if phase == 'pilot' else formal
    if index < 0 or index >= len(tasks): raise ValueError('Array index outside fixed mapping')
    if phase == 'formal':
        gate = load(output_path(root) / 'PILOT_GATE.json')
        from .workflow import source_identity
        if gate.get('status') != 'PASS' or gate.get('identity') != source_identity():
            raise ValueError('Formal work requires the40-certificate-pilot-fit gate')
    task_id = tasks[index]
    for method in ('lasso', 'glasso'):
        record = read_task(root / 'numerical_simulation/outputs_v52_search_final_full', plan, task_id)
        from .workflow import unique_id
        dest = output_path(root) / 'fits' / (unique_id(record) + '_' + method + '.json')
        if check_fit_pair(dest, output_path(root) / 'failures' / dest.name):
            validate_fit(record['task'], record, method)
            print('VERIFIED', task_id, method, flush=True)
        else:
            fit_one(task_id, method)
            validate_fit(record['task'], record, method)


def bundle(root):
    output = output_path(root)
    target_dir = output / 'return'; target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = target_dir / ('taskcard18_cluster_results_' + stamp + '.zip')
    files = [p for p in output.rglob('*') if p.is_file() and target_dir not in p.parents]
    for relative in ('doc/taskcard18', 'paper', 'research/taskcard18'):
        files.extend(p for p in (root / relative).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in (root / 'wheelhouse/manifest.json', root / 'PAYLOAD_MANIFEST.json', root / 'run_taskcard18_cluster.sh') if p.exists())
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root).as_posix())
    (target.with_suffix('.zip.sha256')).write_text(sha(target) + '  ' + target.name + '\n', encoding='ascii')
    print('RETURN_ZIP', target, flush=True)
    return target


def collect(root):
    from .accept import evaluate_all
    from .analysis import build
    from .report import main as report_main
    require_prepared(root)
    try:
        if validate_cache(root) != 1200:
            raise ValueError('Full1200 accepted fits required before any new test evaluation')
        evaluate_all()
        build(False)
        report_main(['--write'])
    finally:
        bundle(root)
    print('FULL_NUMERICAL_COLLECTION_COMPLETE; chapter/figure independent final review may remain pending', flush=True)


def submit(root, *, resume=False, dry_run=False):
    state_path = output_path(root) / 'cluster_submission.json'
    prior = load(state_path) if state_path.exists() else None
    if dry_run:
        check_submission_state(prior is not None, [], resume=resume)
        for item in submission_commands(root / 'run_taskcard18_cluster.sh', output_path(root) / 'logs/slurm'):
            print(shlex.join(item['command']))
        return
    active = active_ids(prior) if prior else []
    check_submission_state(prior is not None, active, resume=resume)
    verify_payload(root)
    # Slurm opens logs before executing a job, so create parents before sbatch.
    logs = output_path(root) / 'logs/slurm'; logs.mkdir(parents=True, exist_ok=True)
    lock = state_path.with_suffix('.submit.lock')
    with lock.open('x') as handle: handle.write(str(os.getpid()))
    try:
        state = prior or {'schema_version': 1, 'payload_root': str(root), 'attempts': []}
        attempt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'resume': resume, 'jobs': []}
        state['attempts'].append(attempt); atomic_json(state_path, state)
        ids = {}
        for item in submission_commands(root / 'run_taskcard18_cluster.sh', logs):
            command = [part.format(**ids) if '{' in part else part for part in item['command']]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
            job_id = parse_job_id(result.stdout)
            ids[item['phase']] = job_id
            attempt['jobs'].append({'phase': item['phase'], 'job_id': job_id, 'command': command})
            atomic_json(state_path, state)
            print('SUBMITTED', item['phase'], job_id, flush=True)
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['submit', 'resume', 'status', 'cancel', 'prepare', 'pilot', 'pilotgate', 'formal', 'collect', 'bundle'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv); root = root_path()
    if args.command in {'submit', 'resume'}:
        submit(root, resume=args.command == 'resume', dry_run=args.dry_run)
    elif args.command in {'status', 'cancel'}:
        state_path = output_path(root) / 'cluster_submission.json'
        if not state_path.exists(): raise ValueError('No submission IDs recorded in this payload')
        state = load(state_path); active = active_ids(state)
        print('RECORDED_JOB_IDS', ','.join(recorded_ids(state)), 'ACTIVE', ','.join(active), flush=True)
        if args.command == 'cancel' and active:
            subprocess.run(['scancel', *active], check=True)
        elif args.command == 'status':
            print('An empty active list does not prove success; inspect completed caches and Slurm logs.', flush=True)
    elif args.command == 'prepare': prepare(root)
    elif args.command in {'pilot', 'formal'}:
        index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        array_worker(root, args.command, index)
    elif args.command == 'pilotgate':
        require_prepared(root)
        from .accept import pilot_gate
        pilot_gate()
    elif args.command == 'collect': collect(root)
    elif args.command == 'bundle': bundle(root)
    return 0


if __name__ == '__main__': raise SystemExit(main())
