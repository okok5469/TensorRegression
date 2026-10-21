"""Isolated Task11 single-search paired selector experiment and bounded runner."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from scripts.taskcard7.lemon import array_sha, observe_fit, require
from research.taskcard10 import experiment as previous
from research.taskcard10.experiment import (artifact_manifest, environment,
    evaluate_test_stream, file_sha, peak_memory_bytes, read_json, source_identity,
    utc_now, write_json, write_npz)


def build_requests(template, cohort, replicates, seed_base, cells=None):
    require(cohort in ('pilot', 'validation'), 'Unknown Task11 cohort')
    require(int(replicates) == replicates and replicates > 0, 'Invalid replicate count')
    default = {
        'pilot': [(100, .5, (4, 3, 5), .6), (100, 4., (2, 3, 4), 0.),
                  (400, .5, (4, 3, 5), 0.), (400, 4., (4, 3, 5), .6),
                  (400, 4., (2, 3, 4), 0.), (400, .5, (2, 3, 4), .6)],
        'validation': list(itertools.product((100, 400), (.5, 2., 4.),
                                             ((4, 3, 5), (2, 3, 4)), (0., .6)))}
    rows = []
    for cell, (n, snr, shape, rho) in enumerate(default[cohort] if cells is None else cells):
        for replicate in range(int(replicates)):
            seed = int(seed_base) + cell * 1000 + replicate
            config = dict(template, n_train=n, snr=snr, sparsity=list(shape), rho=[rho] * 3)
            rows.append({'cohort': cohort, 'task_id': len(rows), 'cell_id': cell,
                         'replicate': replicate, 'config': config, 'seed': seed,
                         'test_seed': seed + 50_000_000})
    return rows


def validate_requests(requests, stage):
    cohort = {'pilot': 'pilot', 'formal': 'validation'}[stage]
    require(bool(requests) and all(r['cohort'] == cohort for r in requests), 'Request cohort differs')
    require([r['task_id'] for r in requests] == list(range(len(requests))), 'Task coverage differs')
    seeds = [r[k] for r in requests for k in ('seed', 'test_seed')]
    require(len(seeds) == len(set(seeds)) and all(0 <= s < 2 ** 32 for s in seeds), 'Request seed collision or range')
    require(all(r['test_seed'] == r['seed'] + 50_000_000 for r in requests), 'Test seed offset differs')
    cells = [(r['config']['n_train'], r['config']['snr'], tuple(r['config']['sparsity']),
              tuple(r['config']['rho']), r['replicate']) for r in requests]
    require(len(cells) == len(set(cells)), 'Duplicate scenario/replicate coverage')
    if stage == 'formal':
        expected = set(itertools.product((100, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)),
                                         ((0.,) * 3, (.6,) * 3), range(4)))
        require(set(cells) == expected and len(requests) == 96, 'Formal coverage must be all 24 cells with four replicates')
    else:
        require(4 <= len(requests) <= 6, 'Pilot coverage must contain four to six cases')
        require({(r['config']['n_train'], r['config']['snr']) for r in requests} >=
                {(100, .5), (100, 4.), (400, .5), (400, 4.)}, 'Pilot coverage lacks sample size or SNR extreme')


def audit_seed_novelty(repo, requests, extra_plans=()):
    repo = Path(repo)
    proposed = [r[k] for r in requests for k in ('seed', 'test_seed')]
    require(len(proposed) == len(set(proposed)), 'New request seed collision')
    paths = set((repo / 'numerical_simulation').glob('outputs*/**/plan.json'))
    paths.update((repo / 'doc/taskcard10/results').glob('*/plan.json'))
    paths.update((repo / 'doc/taskcard11/results').glob('*/plan.json'))
    paths.update(Path(p) for p in extra_plans)
    old, manifests = set(), {}
    for path in sorted(paths):
        old.update(previous._seed_values(read_json(path)))
        manifests[path.relative_to(repo).as_posix()] = file_sha(path)
    overlap = sorted(old.intersection(proposed))
    require(not overlap, 'New seed overlaps historical plan: ' + str(overlap))
    return {'historical_plan_sha256': manifests, 'historical_unique_seeds': len(old),
            'new_unique_seeds': len(proposed), 'overlap': overlap,
            'scope': 'Materialized historical train/test/split seed fields; external unpublished inputs unknown'}


def historical_inputs(repo, *, exclude_output=None):
    repo = Path(repo)
    rows = previous.historical_input_index(repo)
    excluded = None if exclude_output is None else Path(exclude_output).resolve()
    for path in sorted((repo / 'doc/taskcard11/results').glob('*/*/task_*/input_identity.json')):
        if excluded is not None and path.resolve().is_relative_to(excluded):
            continue
        fp = read_json(path)['input_fingerprints']
        rows.append({'X_train': fp['X_train'], 'y_train': fp['y_train'],
                     'path': path.relative_to(repo).as_posix()})
    return rows


def claim_input(output, fingerprints, known, request, alternate_pairs=()):
    pair = (fingerprints['X_train'], fingerprints['y_train'])
    lookup = {(r['X_train'], r['y_train']): r['path'] for r in known}
    for identity in (pair, *alternate_pairs):
        require(identity not in lookup, 'Training input duplicates historical/Pilot input: ' + lookup.get(identity, ''))
    key = hashlib.sha256(json.dumps(pair, separators=(',', ':')).encode()).hexdigest()
    path = Path(output) / 'input_claims' / (key + '.json')
    try:
        write_json(path, {'input_fingerprints': fingerprints, 'request': request})
    except FileExistsError as error:
        raise ValueError('Training input duplicate inside current run') from error
    return {'input_fingerprints': fingerprints, 'known_comparable_inputs_checked': len(lookup),
            'duplicate': False, 'atomic_claim_sha256': file_sha(path)}


def validate_budget(budget):
    require(1 <= budget['workers'] <= 2 and
            0 < budget['wall_time_limit_seconds'] <= 28800 and
            0 < budget['case_timeout_seconds'] <= 3600, 'Task11 frozen budget bounds exceeded')


def shared_clock(output, limit, now=None):
    require(0 < limit <= 28800, 'Shared budget exceeds eight hours')
    path = Path(output) / 'shared_budget.json'
    if path.exists():
        clock = read_json(path)
        require(clock['limit_seconds'] == limit, 'Shared budget cannot change or reset')
        return clock
    clock = {'started_epoch': time.time() if now is None else float(now),
             'started_utc': utc_now(), 'limit_seconds': limit}
    write_json(path, clock)
    return clock


def remaining_seconds(output, now=None):
    clock = read_json(Path(output) / 'shared_budget.json')
    return clock['limit_seconds'] - ((time.time() if now is None else now) - clock['started_epoch'])


def assert_native_identity(fitted, trace, coefficients, choice):
    selected = trace['candidates'][choice['selected_index']]
    support = [getattr(fitted, 'S' + str(i)).astype(int).tolist() for i in (1, 2, 3)]
    require(np.array_equal(coefficients[choice['selected_index']], fitted.B), 'A0 coefficient differs from native result')
    require(selected['support'] == support == trace['final']['support'], 'A0 support differs from native result')
    require(selected['B_sha256'] == array_sha(fitted.B) == trace['final']['B_sha256'], 'A0 coefficient hash differs')
    require(choice['selected_score'] == fitted.sic == trace['final']['msic'], 'A0 score differs from native result')
    require(selected['rss'] == fitted.rss == trace['final']['rss'], 'A0 RSS differs from native result')


def audit_candidates(X, y, trace, coefficients):
    from research.taskcard10.audit import _check_rows
    rows = trace['candidates']
    require(len(rows) == len(coefficients) == len(trace['ordered_size_grid']) + 1,
            'Incomplete native candidate capture')
    require(all(0 < row['q'] < len(y) - 1 for row in rows), 'Candidate q must satisfy 0 < q < n-1')
    require(all(np.isfinite(row['rss']) and row['rss'] > 0 for row in rows), 'Nonfinite/nonpositive candidate RSS')
    require([row['sizes'] for row in rows[1:]] == trace['ordered_size_grid'], 'Native candidate order differs')
    grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in trace['caps']])
            if np.prod(s) <= np.floor(.7 * len(y))]
    require(grid == trace['ordered_size_grid'], 'Incomplete native fixed candidate grid')
    require(len(trace['fixed_searches']) == len(grid), 'Missing native fixed search evidence')
    checks = _check_rows(np.asarray(X), np.asarray(y), rows, coefficients,
                         extra_paths=[{}, *trace['fixed_searches']])
    require(all(c['rank'] == c['q'] and np.isfinite(c['condition_number']) for c in checks),
            'Singular/nonfinite candidate OLS design')
    return {'passed': True, 'audit_search_fits': 0, 'candidate_count': len(checks),
            'all_candidate_designs_full_rank': True,
            'maximum_OLS_coefficient_absolute_error': max(c['coefficient_max_absolute_error'] for c in checks),
            'maximum_RSS_absolute_error': max(c['rss_absolute_error'] for c in checks),
            'maximum_normalized_OLS_gradient': max(c['normalized_OLS_gradient'] for c in checks),
            'maximum_candidate_condition_number': max(c['condition_number'] for c in checks),
            'checks': checks, 'scope': 'Every unchanged native candidate independently refitted by OLS; no search refit or test access'}


def run_two_arms(X, y, config, *, native_fit=None, observer=observe_fit):
    """Training-only API: one native capture, independent audit, two selectors."""
    from research.taskcard11.rescore import select_candidates
    X, y = np.asarray(X), np.asarray(y)
    require(X.ndim == 4 and y.shape == (len(X),) and np.isfinite(X).all() and np.isfinite(y).all(),
            'Aligned finite training inputs required')
    if native_fit is None:
        from tensor_abess.tensor.abess import tensor_abess
        from research.taskcard10.cv import native_options
        native_fit = lambda: tensor_abess(X, y, **native_options(config))
    started = time.perf_counter()
    fitted, trace, coefficients = observer(native_fit)
    native_seconds = time.perf_counter() - started
    audit_started = time.perf_counter()
    audit = audit_candidates(X, y, trace, coefficients)
    audit_seconds = time.perf_counter() - audit_started
    choices, elapsed = {}, {}
    for label, finite_sample in (('A0', False), ('A1', True)):
        tick = time.perf_counter()
        choices[label] = select_candidates(trace['candidates'], len(y), X.shape[1:], finite_sample=finite_sample)
        elapsed[label] = time.perf_counter() - tick
    assert_native_identity(fitted, trace, coefficients, choices['A0'])
    require(choices['A0']['eligible_indices'] == trace['eligible_candidate_indices'] and
            choices['A0']['tolerance'] == trace['one_se_tolerance'], 'A0 native selection band differs')
    arrays, arms = {'full_B': np.stack(coefficients)}, {}
    for label, choice in choices.items():
        index = choice['selected_index']
        row = trace['candidates'][index]
        arrays[label + '_B'] = coefficients[index].copy()
        arms[label] = dict(row, full_candidate_index=index, B_key=label + '_B',
                           selected_score=choice['selected_score'],
                           candidate_source='adaptive_terminal' if index == 0 else 'fixed_grid')
    summary = {'schema_version': 1, 'arms': arms, 'choices': choices, 'full_trace': trace,
               'candidate_sources': ['adaptive_terminal'] + ['fixed_grid'] * (len(coefficients) - 1),
               'input_identity': {'X': array_sha(X), 'y': array_sha(y)}, 'config': config,
               'counters': {'native_search_calls': 1, 'extra_search_calls': 0,
                            'native_adaptive_calls': 1, 'native_fixed_computed': len(trace['ordered_size_grid'])},
               'timings': {'native_seconds': native_seconds, 'A0_selection_seconds': elapsed['A0'],
                           'A1_selection_seconds': elapsed['A1'], 'independent_audit_seconds': audit_seconds},
               'interpretation': 'A0/A1 share every captured native B/RSS/support and candidate order; only final score differs'}
    audit.update(A0_native_B_support_RSS_score_exact=True, A0_native_tolerance_eligible_exact=True,
                 A0_A1_same_candidate_library=True)
    return {'summary': summary, 'arrays': arrays, 'audit': audit}


def verify_plan(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = read_json(output / 'plan.json')
    require(file_sha(repo / plan['protocol_path']) == plan['protocol_sha256'], 'Frozen protocol artifact drift')
    require(read_json(repo / plan['protocol_path']) == plan['protocol'], 'Frozen protocol content differs')
    source_identity(repo, plan['protocol'])
    require(environment() == plan['protocol']['environment'], 'Frozen environment differs')
    validate_requests(plan['requests'], plan['stage'])
    validate_budget(plan['protocol']['pilot' if plan['stage'] == 'pilot' else 'formal'])
    return plan


def prepare(repo, output, protocol_path, stage, requests=None):
    repo, output, protocol_path = Path(repo).resolve(), Path(output).resolve(), Path(protocol_path).resolve()
    protocol = read_json(protocol_path)
    require(protocol['stage'] == stage, 'Frozen stage differs')
    source_identity(repo, protocol)
    require(environment() == protocol['environment'], 'Frozen environment differs')
    budget = protocol['pilot' if stage == 'pilot' else 'formal']
    validate_budget(budget)
    require(protocol['seed_offsets'] == {'test': 50_000_000}, 'Seed offset protocol differs')
    template_path = repo / 'numerical_simulation/outputs_selection_research_task8/plan.json'
    cohort = 'pilot' if stage == 'pilot' else 'validation'
    rows = requests if requests is not None else build_requests(
        read_json(template_path)['requests'][0]['config'], cohort, budget['replicates'], protocol['seed_bases'][cohort])
    validate_requests(rows, stage)
    novelty = audit_seed_novelty(repo, rows)
    plan = {'schema_version': 1, 'stage': stage, 'created_utc': utc_now(),
            'research_only': True, 'release_approved': False,
            'protocol': protocol, 'protocol_sha256': file_sha(protocol_path),
            'protocol_path': protocol_path.relative_to(repo).as_posix(),
            'template_sha256': file_sha(template_path), 'requests': rows,
            'historical_seed_audit': novelty, 'historical_input_index': historical_inputs(repo)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    return plan


def worker(repo, output, cohort, task_id):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = verify_plan(repo, output)
    request = next(r for r in plan['requests'] if r['cohort'] == cohort and r['task_id'] == task_id)
    folder = output / cohort / f'task_{task_id:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'request.json', request)
    write_json(folder / 'started.json', {'started_utc': utc_now(), 'plan_sha256': file_sha(output / 'plan.json')})
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        from scripts.taskcard7.lemon import configure_core
        source = configure_core('v5.2-search-final')
        import tensor_abess.tensor.abess
        from numerical_simulation.data_generation import (TensorSimulationConfig,
            generate_simulation_dataset, iter_test_batches)
        from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
        previous._actual_imports(source)
        values = dict(request['config'])
        for key in ('dims', 'sparsity', 'rho'):
            values[key] = tuple(values[key])
        data = generate_simulation_dataset(TensorSimulationConfig(**values), seed=request['seed'])
        fingerprints = {name: array_sha(getattr(data, name)) for name in ('X_train', 'y_train', 'B_true')}
        fingerprints['supports_true'] = [array_sha(s) for s in data.supports_true]
        known = plan['historical_input_index'] + historical_inputs(repo, exclude_output=output)
        identity = claim_input(output, fingerprints, known, request,
            alternate_pairs=[(previous._stream_digest(data.X_train), previous._stream_digest(data.y_train))])
        write_json(folder / 'input_identity.json', identity)
        require(remaining_seconds(output) > 0, 'Shared budget exhausted before search')
        fit = run_two_arms(data.X_train, data.y_train, request['config'])
        summary, arrays = fit['summary'], fit['arrays']
        write_json(folder / 'independent_audit.json', fit['audit'])
        write_json(folder / 'selection_frozen.json', summary)
        write_json(folder / 'trace.json', summary['full_trace'])
        write_npz(folder / 'candidate_coefficients.npz', {'B': arrays['full_B']})
        write_npz(folder / 'arrays.npz', arrays)
        # No test input exists until all candidate audits and both selections are persisted.
        require(remaining_seconds(output) > 0, 'Shared budget exhausted before external test')
        coefficients = {label: arrays[label + '_B'] for label in ('A0', 'A1')}
        prediction_batches = []
        metrics, test_fingerprint = evaluate_test_stream(
            iter_test_batches(data, seed=request['test_seed']), coefficients, prediction_batches)
        write_npz(folder / 'test_predictions.npz', {
            name: np.concatenate([batch[name] for batch in prediction_batches], axis=0)
            for name in ('y', 'signal', 'prediction')})
        for label, B in coefficients.items():
            row = summary['arms'][label]
            supports = tuple(np.asarray(s, dtype=int) for s in row['support'])
            metrics[label].update(compute_coefficient_metrics(B, data.B_true))
            metrics[label].update(compute_support_metrics(*supports, *data.supports_true))
            metrics[label].update({key: row[key] for key in ('sizes', 'q', 'rss', 'support', 'B_sha256')})
            metrics[label]['coefficient_norm'] = float(np.linalg.norm(B))
        write_npz(folder / 'truth.npz', {'B_true': data.B_true, **{
            f'S{i + 1}_true': s for i, s in enumerate(data.supports_true)}})
        write_json(folder / 'summary.json', summary)
        wall, cpu = time.monotonic() - started, time.process_time() - cpu_started
        timings = summary['timings']
        true_support = [s.astype(int).tolist() for s in data.supports_true]
        true_sizes = list(map(len, true_support))
        complete = {'request': request, 'plan_sha256': file_sha(output / 'plan.json'),
                    'protocol_sha256': plan['protocol_sha256'], 'input_fingerprints': fingerprints,
                    'test_fingerprint': test_fingerprint, 'metrics': metrics,
                    'actual_core_imports': previous._actual_imports(source),
                    'source_identity': plan['protocol']['source_identity'], 'environment': environment(),
                    'resources': {'wall_seconds': wall, 'cpu_seconds': cpu,
                                  'peak_working_set_bytes': peak_memory_bytes(),
                                  'independent_audit_seconds': timings['independent_audit_seconds']},
                    'measured_program_wall_seconds': {
                        label: timings['native_seconds'] + timings[label + '_selection_seconds'] for label in ('A0', 'A1')},
                    'recovery_evidence': {'q_true': int(np.prod(true_sizes)), 'truth_sizes': true_sizes,
                        'truth_support': true_support, 'Q_native': int(np.floor(.7 * len(data.y_train))),
                        'truth_shape_native_representable': all(s <= c for s, c in zip(true_sizes, summary['full_trace']['caps']))
                            and np.prod(true_sizes) <= np.floor(.7 * len(data.y_train)),
                        'truth_shape_found_full_pool': any(r['sizes'] == true_sizes for r in summary['full_trace']['candidates']),
                        'truth_support_found_full_pool': any(r['support'] == true_support for r in summary['full_trace']['candidates'])},
                    'artifact_sha256': artifact_manifest(folder), 'completed_utc': utc_now()}
        complete['recovery_evidence']['truth_shape_native_representable'] = bool(
            complete['recovery_evidence']['truth_shape_native_representable'])
        verify_plan(repo, output)
        require(remaining_seconds(output) > 0, 'Shared budget exhausted before completion')
        write_json(folder / 'complete.json', complete)
        print('COMPLETE', cohort, task_id, round(wall, 3), flush=True)
    except BaseException as error:
        write_json(folder / 'failed.json', {'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'failed_utc': utc_now(),
                   'wall_seconds': time.monotonic() - started})
        raise


def collect(output, cohort):
    output = Path(output)
    plan = read_json(output / 'plan.json')
    requests = [r for r in plan['requests'] if r['cohort'] == cohort]
    paths = sorted((output / cohort).glob('task_*/complete.json'))
    require(len(paths) == len(requests) > 0, 'Missing or unexpected complete records')
    records = [read_json(path) for path in paths]
    require([r['request'] for r in records] == requests, 'Completed request coverage drift')
    for path, record in zip(paths, records):
        require(record['plan_sha256'] == file_sha(output / 'plan.json'), 'Complete plan identity drift')
        for name, digest in record['artifact_sha256'].items():
            target = (path.parent / name).resolve()
            require(target.is_relative_to(path.parent.resolve()) and file_sha(target) == digest,
                    'Completed artifact fingerprint drift: ' + name)
    return records


def run_stage(repo, output, cohort):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = verify_plan(repo, output)
    require(cohort == ('pilot' if plan['stage'] == 'pilot' else 'validation'), 'Cohort stage mismatch')
    budget = plan['protocol']['pilot' if cohort == 'pilot' else 'formal']
    shared_clock(output, budget['wall_time_limit_seconds'])
    require(remaining_seconds(output) > 0, 'Shared compute budget exhausted')
    requests = [r for r in plan['requests'] if r['cohort'] == cohort]
    commands = [{'task_id': r['task_id'], 'command': [sys.executable, '-B', '-m',
        'research.taskcard11.experiment', 'worker', '--output', str(output), '--cohort', cohort,
        '--task-id', str(r['task_id'])]} for r in requests]
    result = previous.run_subprocess_batch(commands, output / ('execution_' + cohort),
        workers=budget['workers'], total_seconds=remaining_seconds(output),
        case_seconds=budget['case_timeout_seconds'], cwd=repo)
    if result['status'] != 'COMPLETE':
        write_json(output / ('stopped_' + cohort + '.json'), result)
        return result
    records = collect(output, cohort)
    verify_plan(repo, output)
    write_json(output / 'completion.json', {**result, 'plan_sha256': file_sha(output / 'plan.json'),
        'completed_cases': len(records), 'case_wall_seconds': [r['resources']['wall_seconds'] for r in records],
        'maximum_peak_memory_bytes': max(r['resources']['peak_working_set_bytes'] for r in records),
        'release_approved': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare-pilot', 'prepare-formal', 'run-pilot', 'run-validation', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--cohort', choices=('pilot', 'validation'))
    parser.add_argument('--task-id', type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.mode.startswith('prepare-'):
        require(args.protocol is not None, 'Frozen protocol is required')
        plan = prepare(repo, args.output, args.protocol, args.mode.removeprefix('prepare-'))
        print('PREPARED', plan['stage'], len(plan['requests']), flush=True)
    elif args.mode == 'worker':
        worker(repo, args.output, args.cohort, args.task_id)
    else:
        result = run_stage(repo, args.output, args.mode.removeprefix('run-'))
        print('STAGE', result['status'], flush=True)
        if result['status'] != 'COMPLETE':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
