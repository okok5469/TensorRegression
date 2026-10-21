"""Bounded continuation of the sealed Task10 queue; original science unchanged."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

from research.taskcard10 import experiment as original

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return original.read_json(path)


def write(path, value):
    return original.write_json(path, value)


def sha(path):
    return original.file_sha(Path(path))


def exact_development(requests, records, ids=None):
    planned_rows = [r for r in requests if r['cohort'] == 'development']
    planned = {r['task_id']: r for r in planned_rows}
    require(len(planned_rows) == len(planned) == 48 and set(planned) == set(range(48)),
            'Original development request coverage differs')
    actual = {r['request']['task_id']: r for r in records}
    expected = set(range(48)) if ids is None else set(ids)
    require(len(actual) == len(records) and set(actual) == expected,
            'Missing, duplicate or unexpected development records')
    require(all(actual[i]['request'] == planned[i] for i in actual), 'Completed request drift')
    return [actual[i] for i in sorted(actual)]


def missing_development(requests, old_records):
    exact_development(requests, old_records, range(30))
    return list(range(30, 48))


def assert_science_unchanged(old, new):
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    expected_compute = dict(before['formal'], workers=2, wall_time_limit_seconds=28800,
                            case_timeout_seconds=3600)
    require(after['formal'] == expected_compute, 'Unexpected continuation compute protocol')
    after['formal'] = before['formal']
    for field in ('source_files', 'source_identity'):
        previous, current = before[field], after[field]
        require(current == previous, 'Scientific dependencies must remain exactly unchanged')
    require('continuation' not in before and isinstance(after.pop('continuation', None), dict),
            'Continuation provenance is missing or recursive')
    require(before == after, 'Scientific protocol changed beyond compute/provenance amendment')


def assert_partial_identity(saved, current):
    require(saved['input_fingerprints'] == current['input_fingerprints'],
            'Retry input differs from original partial training identity')


def copy_reused_cases(old, new, ids):
    old, new = Path(old), Path(new)
    result = {}
    for task_id in ids:
        relative = Path('development') / f'task_{task_id:03d}'
        source, destination = old / relative, new / relative
        require((source / 'complete.json').is_file(), 'Incomplete source cannot be reused')
        manifest = original.artifact_manifest(source)
        shutil.copytree(source, destination, dirs_exist_ok=False)
        require(original.artifact_manifest(destination) == manifest, 'Copy byte verification failed')
        result[str(task_id)] = {'source': str(source), 'relative_directory': relative.as_posix(),
                                'files': manifest, 'complete_sha256': sha(source / 'complete.json')}
    return result


def verify_reused_cases(old, new, manifest):
    old, new = Path(old), Path(new)
    for text_id, row in manifest.items():
        relative = Path('development') / f'task_{int(text_id):03d}'
        require(row['relative_directory'] == relative.as_posix(), 'Reuse directory identity differs')
        for base in (old, new):
            require(original.artifact_manifest(base / relative) == row['files'],
                    'Reused source or copy bytes changed')
            require(sha(base / relative / 'complete.json') == row['complete_sha256'],
                    'Reused complete provenance changed')


def protected_identity(repo):
    repo = Path(repo)
    baseline_path, index_path = repo/'doc/taskcard10/PROTECTED_BASELINE.json', repo/'doc/taskcard10/DELIVERY_INDEX.json'
    baseline, delivery = read(baseline_path), read(index_path)
    require(len(baseline['files']) == 5922 and len(delivery['files']) == 897,
            'Original protection inventory differs')
    combined = dict(baseline['files'])
    for name, record in delivery['files'].items():
        require(name not in combined or combined[name] == record, 'Protection inventories conflict')
        combined[name] = record
    combined[index_path.relative_to(repo).as_posix()] = {'size': index_path.stat().st_size, 'sha256': sha(index_path)}
    require(len(combined) == 6820, 'Expected 6820 original protected files')
    for name, record in combined.items():
        path = repo/name
        require(path.is_file() and path.stat().st_size == record['size'] and sha(path) == record['sha256'],
                'Protected original byte drift: ' + name)
    return {'file_count': len(combined), 'files': combined,
            'baseline_sha256': sha(baseline_path), 'delivery_index_sha256': sha(index_path)}


def _verify_case_artifacts(folder, complete):
    for name, digest in complete['artifact_sha256'].items():
        require(sha(Path(folder)/name) == digest, 'Completed artifact drift: ' + name)


def prepare(repo, old='doc/taskcard10/results/formal',
            output='doc/taskcard10/results/continuation_01', report='doc/taskcard10/continuation_01'):
    repo = Path(repo).resolve()
    old, output, report = [repo/p for p in (old, output, report)]
    require(not output.exists(), 'Continuation output already exists; refusing overwrite')
    require((report/'DESIGN_AMENDMENT.md').is_file(), 'Missing approved continuation amendment')
    protection = protected_identity(repo)
    old_plan = read(old/'plan.json')
    old_protocol = old_plan['protocol']
    require(sha(repo/old_plan['protocol_path']) == old_plan['protocol_sha256'], 'Original protocol drift')
    original.source_identity(repo, old_protocol)
    require(original.environment() == old_protocol['environment'], 'Original environment drift')
    old_records = [read(p) for p in sorted((old/'development').glob('task_*/complete.json'))]
    retry_ids = missing_development(old_plan['requests'], old_records)
    require(not (old/'development_lock.json').exists() and not (old/'confirmation').exists(),
            'Original run is not the expected partial, unconfirmed state')
    for record in old_records:
        require(record['plan_sha256'] == sha(old/'plan.json') and
                record['protocol_sha256'] == old_plan['protocol_sha256'], 'Original complete provenance differs')
        _verify_case_artifacts(old/'development'/f"task_{record['request']['task_id']:03d}", record)
    protocol = copy.deepcopy(old_protocol)
    protocol['formal'].update(workers=2, wall_time_limit_seconds=28800, case_timeout_seconds=3600)
    protocol['continuation'] = {'amendment_sha256': sha(report/'DESIGN_AMENDMENT.md'),
            'original_plan_sha256': sha(old/'plan.json'), 'original_protocol_sha256': old_plan['protocol_sha256'],
            'original_raw_directory': old.relative_to(repo).as_posix(), 'reused_ids': list(range(30)),
            'retry_ids': retry_ids, 'global_clock_reset_allowed': False,
            'confirmation_reserve': '1.5*(sum of all48 development case walls)/2+60 seconds'}
    assert_science_unchanged(old_protocol, protocol)
    output.mkdir(parents=True, exist_ok=False)
    write(report/'PROTOCOL.json', protocol)
    reuse = copy_reused_cases(old, output, range(30))
    write(output/'REUSE_MANIFEST.json', {'old_plan_sha256': sha(old/'plan.json'), 'cases': reuse})
    write(output/'PROTECTED_ORIGINALS.json', protection)
    plan = copy.deepcopy(old_plan)
    plan.update(protocol=protocol, protocol_sha256=sha(report/'PROTOCOL.json'),
                protocol_path=(report/'PROTOCOL.json').relative_to(repo).as_posix())
    plan['continuation'] = dict(protocol['continuation'], original_plan_path=(old/'plan.json').relative_to(repo).as_posix())
    require(plan['requests'] == old_plan['requests'], 'Original96 requests changed')
    write(output/'plan.json', plan)
    names = ['research/taskcard10/resume.py', 'research/taskcard10/test_resume.py',
             (report/'DESIGN_AMENDMENT.md').relative_to(repo).as_posix(),
             (report/'PROTOCOL.json').relative_to(repo).as_posix(),
             *[(output/name).relative_to(repo).as_posix() for name in
               ('plan.json', 'REUSE_MANIFEST.json', 'PROTECTED_ORIGINALS.json')]]
    write(output/'CONTINUATION_SEAL.json', {'files': {name: sha(repo/name) for name in names},
            'reused_cases': 30, 'retry_ids': retry_ids, 'original_protected_files': 6820})
    return {'prepared': True, 'reused': 30, 'retry_ids': retry_ids}


def verify_preparation(repo, output):
    repo, output = Path(repo), Path(output)
    for name, digest in read(output/'CONTINUATION_SEAL.json')['files'].items():
        require(sha(repo/name) == digest, 'Continuation preparation/source drift: ' + name)
    plan = read(output/'plan.json')
    old = repo/plan['protocol']['continuation']['original_raw_directory']
    old_plan = read(old/'plan.json')
    require(sha(old/'plan.json') == plan['continuation']['original_plan_sha256'], 'Original plan changed')
    require(old_plan['requests'] == plan['requests'], 'Original request identities changed')
    assert_science_unchanged(old_plan['protocol'], plan['protocol'])
    original.source_identity(repo, plan['protocol'])
    require(original.environment() == plan['protocol']['environment'], 'Continuation environment differs')
    for name, record in read(output/'PROTECTED_ORIGINALS.json')['files'].items():
        path = repo/name
        require(path.stat().st_size == record['size'] and sha(path) == record['sha256'], 'Original protected bytes changed')
    verify_reused_cases(old, output, read(output/'REUSE_MANIFEST.json')['cases'])
    return plan, old


def shared_clock(output, now=None):
    path = Path(output)/'formal_budget.json'
    if path.exists():
        body = read(path)
        require(body['limit_seconds'] == 28800, 'Shared continuation budget drift')
        return body
    body = {'started_epoch': time.time() if now is None else float(now),
            'started_utc': original.utc_now(), 'limit_seconds': 28800}
    write(path, body)
    return body


def remaining_seconds(output, now=None):
    clock = read(Path(output)/'formal_budget.json')
    require(clock['limit_seconds'] == 28800, 'Shared continuation budget drift')
    return clock['limit_seconds'] - ((time.time() if now is None else now) - clock['started_epoch'])


def confirmation_reserve(records):
    require(len(records) == 48, 'Full48 development required for confirmation cost reserve')
    return 1.5 * sum(r['resources']['wall_seconds'] for r in records) / 2. + 60.


def assert_remaining_reserve(gate, remaining):
    require(gate['passed'] and remaining >= gate['required_seconds'],
            'Frozen confirmation reserve is no longer available')


def require_confirmation(output, now=None):
    output = Path(output)
    require(not (output/'confirmation_NOT_RUN.json').exists(), 'Confirmation has terminal NOT_RUN')
    path = output/'confirmation_budget_gate.json'
    require(path.is_file() and read(path)['passed'], 'Confirmation compute entry gate missing or failed')
    original.confirmation_lock(output)
    assert_remaining_reserve(read(path), remaining_seconds(output, now))


def precheck_partial(repo, output):
    """One fresh process, training generation only; no search or test generation."""
    repo, output = Path(repo), Path(output)
    plan = read(output/'plan.json')
    old = repo/plan['continuation']['original_raw_directory']
    from scripts.taskcard7.lemon import configure_core, array_sha
    configure_core('v5.2-search-final')
    import tensor_abess.tensor.abess
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    rows = []
    for request in plan['requests']:
        task_id = request['task_id']
        path = old/'development'/f'task_{task_id:03d}'/'input_identity.json'
        if request['cohort'] != 'development' or task_id < 30 or not path.exists():
            continue
        config = dict(request['config'])
        for key in ('dims', 'sparsity', 'rho'):
            config[key] = tuple(config[key])
        data = generate_simulation_dataset(TensorSimulationConfig(**config), seed=request['seed'])
        fingerprint = {name: array_sha(getattr(data, name)) for name in ('X_train', 'y_train', 'B_true')}
        fingerprint['supports_true'] = [array_sha(s) for s in data.supports_true]
        current = {'input_fingerprints': fingerprint}
        assert_partial_identity(read(path), current)
        rows.append({'task_id': task_id, 'old_input_identity_sha256': sha(path), **current})
    write(output/'RETRY_INPUT_PRECHECK.json', {'passed': True, 'training_only': True,
            'search_fits': 0, 'test_generated': False, 'cases': rows})


def _collect_development(output, old, plan):
    records = []
    old_hash, new_hash = sha(Path(old)/'plan.json'), sha(Path(output)/'plan.json')
    old_protocol_hash = read(Path(old)/'plan.json')['protocol_sha256']
    for path in sorted((Path(output)/'development').glob('task_*/complete.json')):
        record = read(path)
        task_id = record['request']['task_id']
        require(record['plan_sha256'] == (old_hash if task_id < 30 else new_hash), 'Complete plan origin mismatch')
        require(record['protocol_sha256'] == (old_protocol_hash if task_id < 30 else plan['protocol_sha256']),
                'Complete protocol origin mismatch')
        _verify_case_artifacts(path.parent, record)
        partial = Path(old)/'development'/f'task_{task_id:03d}'/'input_identity.json'
        if task_id >= 30 and partial.exists():
            assert_partial_identity(read(partial), {'input_fingerprints': record['input_fingerprints']})
        records.append(record)
    return exact_development(plan['requests'], records)


def run_development(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    shared_clock(output)
    plan, old = verify_preparation(repo, output)
    require(not (output/'execution_development').exists(), 'Continuation was already attempted; no automatic retry')
    require(not (output/'stopped_development.json').exists(), 'Continuation has a terminal stop')
    retry = missing_development(plan['requests'], [read(p) for p in sorted((output/'development').glob('task_*/complete.json'))])
    command = [sys.executable, '-B', '-m', 'research.taskcard10.resume', 'precheck', '--output', str(output)]
    try:
        remaining = remaining_seconds(output)
        require(remaining > 0, 'Shared continuation budget exhausted')
        precheck = original.run_subprocess_batch([{'task_id': 0, 'command': command}],
                output/'execution_precheck', workers=1, total_seconds=remaining,
                case_seconds=3600, cwd=repo)
        require(precheck['status'] == 'COMPLETE', 'Training-only partial identity precheck stopped')
        commands = [{'task_id': task_id, 'command': [sys.executable, '-B', '-m',
            'research.taskcard10.experiment', 'worker', '--output', str(output),
            '--cohort', 'development', '--task-id', str(task_id)]} for task_id in retry]
        result = original.run_subprocess_batch(commands, output/'execution_development', workers=2,
                total_seconds=remaining_seconds(output), case_seconds=3600, cwd=repo)
        if result['status'] != 'COMPLETE':
            write(output/'stopped_development.json', result)
            write(output/'confirmation_NOT_RUN.json', {'status': 'NOT_RUN', 'reason': 'Continuation development stopped'})
            return result
        verify_preparation(repo, output)
        records = _collect_development(output, old, plan)
        from research.taskcard10.verify import recompute_gates, compare_gates
        gate = original.evaluate_gates(records, plan['protocol']['development_entry_gate'])
        independent = recompute_gates(records, plan['protocol']['development_entry_gate'])
        compare_gates(gate, independent)
        write(output/'development_gate_independent.json', independent)
        write(output/'development_lock.json', {'plan_sha256': sha(output/'plan.json'),
                'completed_development_tasks': 48, 'entry_gate': gate,
                'development_artifacts': original.artifact_manifest(output/'development'),
                'locked_utc': original.utc_now(), 'reused_cases': 30, 'new_complete_cases': 18,
                'reuse_manifest_sha256': sha(output/'REUSE_MANIFEST.json')})
        required, left = confirmation_reserve(records), remaining_seconds(output)
        passed = bool(gate['passed'] and left >= required)
        write(output/'confirmation_budget_gate.json', {'passed': passed, 'required_seconds': required,
                'remaining_seconds': left, 'rule': '1.5*all48 development case wall sum/2+60 seconds',
                'scientific_gate_passed': gate['passed']})
        if not passed:
            write(output/'confirmation_NOT_RUN.json', {'status': 'NOT_RUN',
                'reason': 'Original nine development gates failed' if not gate['passed'] else 'Insufficient shared remaining budget'})
        write(output/'development_completion.json', {'status': 'COMPLETE', 'completed_development_tasks': 48,
                'reused': 30, 'newly_completed': 18, 'entry_gate': gate, 'confirmation_entry': passed})
        return {'status': 'COMPLETE', 'confirmation_entry': passed}
    except BaseException as error:
        if not (output/'stopped_development.json').exists():
            write(output/'stopped_development.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error)})
        if not (output/'confirmation_NOT_RUN.json').exists():
            write(output/'confirmation_NOT_RUN.json', {'status': 'NOT_RUN', 'reason': 'Continuation dependency failed'})
        raise


def run_confirmation(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan, old = verify_preparation(repo, output)
    require_confirmation(output)
    _collect_development(output, old, plan)
    return original.run_stage(repo, output, 'confirmation')


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run-dev', 'run-confirm', 'precheck'))
    parser.add_argument('--output', type=Path, default=Path('doc/taskcard10/results/continuation_01'))
    parser.add_argument('--old', type=Path, default=Path('doc/taskcard10/results/formal'))
    parser.add_argument('--report', type=Path, default=Path('doc/taskcard10/continuation_01'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(repo, args.old, args.output, args.report)
    elif args.mode == 'run-dev':
        result = run_development(repo, args.output)
    elif args.mode == 'run-confirm':
        result = run_confirmation(repo, args.output)
    else:
        precheck_partial(repo, repo/args.output)
        result = {'precheck': 'PASS'}
    print(json.dumps(result, ensure_ascii=False))
    if result.get('status') == 'FAILED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
