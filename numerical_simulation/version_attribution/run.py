"""Immutable 250-label v5.1 diagnostics; run workers only with exact input/environment identity."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def root():
    return Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def seal(body):
    return {**body, 'sha256': digest(body)}


def check_seal(record):
    require(record['sha256'] == digest({k: v for k, v in record.items() if k != 'sha256'}), 'Record checksum differs')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def source_manifests():
    result = {}
    for version in ('v5-original', 'v5.1-bcd-fix', 'v5.2-search-final'):
        folder = root() / 'doc/algorithm_versions' / version
        expected = read(folder / 'VERSION.json')['source_sha256']
        actual = {p.relative_to(folder / 'src').as_posix(): sha(p) for p in (folder / 'src').rglob('*.py')}
        require(actual == expected and len(actual) == 32, 'Frozen core checksum differs: ' + version)
        result[version] = actual
    return result


def compare_grids(caps, n):
    grids = []
    for version in ('v5.1-bcd-fix', 'v5.2-search-final'):
        path = root() / 'doc/algorithm_versions' / version / 'src/tensor_abess/tensor/bcd.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_candidate_size_triples')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        grids.append(namespace['_candidate_size_triples'](*caps, max(1, int(.7 * n))))
    return {'caps': list(caps), 'n': n, 'v51_count': len(grids[0]), 'v52_count': len(grids[1]),
            'same_ordered_grid': grids[0] == grids[1],
            'truth_shape_in_v51': (4, 3, 5) in grids[0], 'truth_shape_in_v52': (4, 3, 5) in grids[1]}


def build_plan():
    old_dir = root() / 'numerical_simulation/outputs_known_s_full_50'
    new_dir = root() / 'numerical_simulation/outputs_v52_search_final_full'
    old, new = read(old_dir / 'plan.json'), read(new_dir / 'plan.json')
    require(old['tasks'] == new['tasks'] and old['environment'] == new['environment'], 'Formal designs differ')
    manifests = source_manifests()
    for plan, version in ((old, 'v5-original'), (new, 'v5.2-search-final')):
        require(plan['plan_id'] == digest({k: v for k, v in plan.items() if k != 'plan_id'}), 'Original plan checksum differs')
        require(all(plan['source_fingerprints']['src/' + k] == v for k, v in manifests[version].items()), 'Original core differs')
    tasks = []
    for task in new['tasks']:
        if task['scenario_id'] not in {0, 1, 5, 7, 11}:
            continue
        name = f"task_{task['task_id']:06d}.json"
        first, second = read(old_dir / 'tasks' / name), read(new_dir / 'tasks' / name)
        for record, plan in ((first, old), (second, new)):
            require(record['record_sha256'] == digest({k: v for k, v in record.items() if k != 'record_sha256'}), 'Original record checksum differs')
            require(record['task'] == task and record['plan_id'] == plan['plan_id']
                    and record['source_fingerprints'] == plan['source_fingerprints']
                    and record['environment'] == plan['environment'], 'Original record identity differs')
        require(first['input_fingerprints'] == second['input_fingerprints'], 'Original input pair differs')
        tasks.append({**task, 'expected_input_fingerprints': second['input_fingerprints'],
                      'input_key': digest({'inputs': second['input_fingerprints'], 'config': task['config'],
                                           'test_seed': task['test_seed']}),
                      'old_record_sha256': first['record_sha256'], 'new_record_sha256': second['record_sha256']})
    require(len(tasks) == 250, 'Expected exactly 250 prescribed labels')
    dependencies = {k: v for k, v in new['source_fingerprints'].items() if not k.startswith('src/')}
    require(all(sha(root() / k) == v for k, v in dependencies.items()), 'Generator/evaluator differs')
    return seal({'schema_version': 1, 'scope': 'Five scenes x50; not full700', 'fit_version': 'v5.1-bcd-fix',
                 'source_manifests': manifests, 'dependencies': dependencies, 'worker_sha256': sha(__file__),
                 'expected_environment': new['environment'], 'tasks': tasks,
                 'gate_task_ids': [0, 50, 250, 350, 550], 'old_plan_id': old['plan_id'], 'new_plan_id': new['plan_id']})


def load_plan(output):
    plan = read(Path(output) / 'plan.json')
    check_seal(plan)
    require(plan == build_plan(), 'Frozen plan/source metadata changed')
    return plan


def configure_v51():
    if any(n == 'tensor_abess' or n.startswith('tensor_abess.') for n in sys.modules):
        raise RuntimeError('A core is already imported; use an isolated process')
    source = root() / 'doc/algorithm_versions/v5.1-bcd-fix/src'
    sys.path.insert(0, str(source))
    import tensor_abess
    require(Path(tensor_abess.__file__).resolve() == source / 'tensor_abess/__init__.py', 'Wrong actual import path')
    return source


def require_inputs(expected, actual):
    require(set(expected) == {'X_train', 'y_train', 'B_true', 'train_noise'} and expected == actual,
            'Strict input fingerprints differ; no fitting or approximate substitution allowed')


def serializable_trace(trace):
    result = dict(trace)
    for key in ('pipeline_init_supports', 'pipeline_glasso_init_supports'):
        result[key] = [support.tolist() for support in trace[key]]
    return result


def regenerated(task):
    from numerical_simulation.known_s_cluster import run as original
    from numerical_simulation.data_generation import generate_simulation_dataset
    from tensor_abess.tensor.ops import tensor_inner
    dataset = generate_simulation_dataset(original.config_from_dict(task['config']), seed=task['seed'])
    arrays = {'X_train': dataset.X_train, 'y_train': dataset.y_train, 'B_true': dataset.B_true,
              'train_noise': dataset.y_train - tensor_inner(dataset.X_train, dataset.B_true)}
    return original, dataset, {k: original.array_digest(v) for k, v in arrays.items()}


def require_gates(output, plan):
    for task_id in plan['gate_task_ids']:
        path = Path(output) / 'gates' / f'task_{task_id:06d}.json'
        require(path.is_file(), 'Five scenario pilot gates required before bulk execution')
        record = read(path)
        check_seal(record)
        require(record['plan_id'] == plan['sha256'] and record['task_id'] == task_id, 'Pilot gate identity differs')
        result = read(Path(output) / 'tasks' / path.name)
        check_seal(result)
        require(record['result_sha256'] == result['sha256'], 'Pilot gate result differs')


def execute(output, task_id, preflight=False, pilot=False):
    plan = load_plan(output)
    task = next(t for t in plan['tasks'] if t['task_id'] == task_id)
    require(not pilot or task_id in plan['gate_task_ids'], 'Unknown pilot task')
    if not preflight and not pilot:
        require_gates(output, plan)
    configure_v51()
    original, dataset, actual = regenerated(task)
    environment = original.environment()
    if preflight:
        result = seal({'plan_id': plan['sha256'], 'task_id': task_id, 'fit_executed': False,
                       'expected': task['expected_input_fingerprints'], 'actual': actual,
                       'matches': {k: actual[k] == v for k, v in task['expected_input_fingerprints'].items()},
                       'actual_environment': environment, 'expected_environment': plan['expected_environment'],
                       'environment_matches': environment == plan['expected_environment'],
                       'passed': actual == task['expected_input_fingerprints'] and environment == plan['expected_environment']})
        write_json(Path(output) / f'preflight_task_{task_id:06d}.json', result)
        return result
    require_inputs(task['expected_input_fingerprints'], actual)
    require(environment == plan['expected_environment'], 'Worker environment differs')
    path = Path(output) / 'tasks' / f'task_{task_id:06d}.json'
    require(not path.exists() and not path.with_suffix('.npz').exists(), 'Existing/partial result; refusing overwrite')
    estimate, trace = original.fit_tensor(dataset.X_train, dataset.y_train, dataset.config, mode='adaptive', seed=task['seed'])
    metrics = original.evaluate(dataset, estimate, task['test_seed'])
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, B=estimate.B, S1=estimate.S1, S2=estimate.S2, S3=estimate.S3)
    record = seal({'plan_id': plan['sha256'], 'task_id': task_id, 'fit_version': plan['fit_version'],
                   'core': plan['source_manifests']['v5.1-bcd-fix'], 'environment': environment,
                   'input_fingerprints': actual, 'metrics': metrics, 'sic': estimate.sic,
                   'supports': [estimate.S1.tolist(), estimate.S2.tolist(), estimate.S3.tolist()],
                   'trace': serializable_trace(trace), 'grids': compare_grids(trace['s_max'], task['config']['n_train']),
                   'npz_sha256': sha(path.with_suffix('.npz'))})
    write_json(path, record)
    if pilot:
        write_json(Path(output) / 'gates' / path.name,
                   seal({'plan_id': plan['sha256'], 'task_id': task_id, 'result_sha256': record['sha256']}))
    return record


def merge(output):
    plan = load_plan(output)
    require_gates(output, plan)
    rows = []
    for task in plan['tasks']:
        name = f"task_{task['task_id']:06d}.json"
        path = Path(output) / 'tasks' / name
        require(path.is_file(), 'Incomplete 250-label diagnostic: ' + name)
        record = read(path)
        check_seal(record)
        require(record['plan_id'] == plan['sha256'] and record['input_fingerprints'] == task['expected_input_fingerprints']
                and record['npz_sha256'] == sha(path.with_suffix('.npz')), 'Result identity differs')
        row = {'task_id': task['task_id'], 'scenario_id': task['scenario_id'], 'replicate': task['replicate']}
        for label, folder in [('v5', 'outputs_known_s_full_50'), ('v52', 'outputs_v52_search_final_full')]:
            previous = read(root() / 'numerical_simulation' / folder / 'tasks' / name)
            selected = next(r for r in previous['results'] if r['method'] == 'adaptive')
            row.update({label + '_' + k: v for k, v in selected['metrics'].items()})
            row[label + '_sic'] = selected['sic']
            import numpy as np
            with np.load(root() / 'numerical_simulation' / folder / 'tasks' / name.replace('.json', '.npz'), allow_pickle=False) as saved:
                row[label + '_supports'] = json.dumps([saved[f'S{i}_adaptive'].tolist() for i in (1, 2, 3)])
            caps = selected['trace']['s_max']
            comparison = compare_grids(caps, task['config']['n_train'])
            row[label + '_candidate_count'] = comparison['v51_count' if label == 'v5' else 'v52_count'] + 1
        row.update({'v51_' + k: v for k, v in record['metrics'].items()})
        row['v51_sic'] = record['sic']
        row['v51_supports'] = json.dumps(record['supports'])
        row['v51_candidate_count'] = record['grids']['v51_count'] + 1
        row.update(delta1_pmse=row['v51_pmse'] - row['v5_pmse'], delta2_pmse=row['v52_pmse'] - row['v51_pmse'],
                   total_pmse=row['v52_pmse'] - row['v5_pmse'])
        require(abs(row['delta1_pmse'] + row['delta2_pmse'] - row['total_pmse']) < 1e-12, 'Decomposition identity failed')
        rows.append(row)
    with (Path(output) / 'paired_three_versions.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(Path(output) / 'integrity.json', seal({'completed': 250, 'plan_id': plan['sha256'], 'status': 'PASSED'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'pilot', 'run', 'worker', 'merge'])
    parser.add_argument('--output-dir', type=Path, default=root() / 'numerical_simulation/outputs_version_attribution/cluster_run')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = args.output_dir
    if args.command == 'prepare':
        require(not output.exists(), 'Prepare requires a new directory')
        write_json(output / 'plan.json', build_plan())
        print('Frozen 250-label v5.1-only plan; no fit executed')
    elif args.command == 'preflight':
        result = execute(output, args.task_id, preflight=True)
        print('Strict preflight passed:', result['passed'])
        return 0 if result['passed'] else 2
    elif args.command in {'pilot', 'run'}:
        plan = load_plan(output)
        if args.command == 'run':
            require_gates(output, plan)
        task_ids = plan['gate_task_ids'] if args.command == 'pilot' else [t['task_id'] for t in plan['tasks']
                                                                       if t['task_id'] not in plan['gate_task_ids']]
        for task_id in task_ids:
            command = [sys.executable, '-B', '-m', 'numerical_simulation.version_attribution.run', 'worker',
                       '--output-dir', str(output), '--task-id', str(task_id)]
            subprocess.run(command + (['--pilot'] if args.command == 'pilot' else []), check=True)
    elif args.command == 'worker':
        execute(output, args.task_id, pilot=args.pilot)
    else:
        merge(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
