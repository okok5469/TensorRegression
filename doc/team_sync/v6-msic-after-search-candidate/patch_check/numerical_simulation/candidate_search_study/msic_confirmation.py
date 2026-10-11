"""Evaluate original default and one prior development lock on untouched A3 pools.

No support search, eleven-rule sweep, rule selection, or coefficient replacement
is performed. Confirmation must already contain the original ninety five-group
searches; their A0/A2 outputs remain independent comparators in the parent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from numerical_simulation.candidate_search_study import msic_after_search as scoring
from numerical_simulation.candidate_search_study import run
from numerical_simulation.candidate_search_study.msic_refit import validate_lock
from numerical_simulation.candidate_search_study.worker import configure_core, pool_fingerprint
from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test, select_candidate


def two_choices(metadata, n, dims, locked_rule):
    declared = [{'id': rule['rule'], **{k: rule[k] for k in ('criterion', 'alpha', 'gamma', 'scale')}}
                for rule in scoring.declared_rules()]
    run.require(locked_rule in declared, 'Locked rule is not predeclared')
    default = next(rule for rule in declared if rule['id'] == scoring.DEFAULT_RULE)
    before = run.canonical(metadata)
    choices = []
    for role, rule in (('default', default), ('locked', locked_rule)):
        _, info = select_candidate(metadata, n, dims, criterion=rule['criterion'],
                                   alpha=rule['alpha'] or 1.5, scale=rule['scale'])
        label = 'locked:'+rule['id'] if role == 'locked' and rule['id'] == default['id'] else rule['id']
        choices.append(({**rule, 'role': role, 'label': label}, int(info['selected_index']), info))
    run.require(before == run.canonical(metadata), 'Two-rule selection mutated frozen metadata')
    return choices


def validate_confirmation_plan(plan, lock):
    groups = ['A0', 'A1', 'A2', 'A3', 'A4']
    run.require(plan['plan_id'] == run.digest({k: v for k, v in plan.items() if k != 'plan_id'}),
                'Confirmation plan checksum differs')
    run.require(lock.get('confirmation_search_plan_id') == plan['plan_id'], 'Locked confirmation plan differs')
    run.require(plan.get('cohort') == 'confirmation' and plan.get('n_reps') == 10,
                'Original ninety-dataset confirmation protocol required')
    run.require(plan['groups'] == groups and lock.get('confirmation_groups') == groups,
                'Original five confirmation groups must remain intact')
    run.require(plan['tasks'] == run.build_tasks('confirmation', 10, groups),
                'Confirmation seeds/configs/requests differ from predeclared protocol')


def validate_lock_timing(lock, input_dir):
    stamp = lock.get('locked_utc')
    run.require(isinstance(stamp, str), 'Locked UTC timestamp is required')
    try:
        when = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid locked UTC timestamp') from error
    run.require(when.tzinfo is not None and when.utcoffset() is not None, 'Naive lock time is forbidden')
    logs = [path for path in (Path(input_dir)/'logs').rglob('*') if path.is_file()]
    run.require(bool(logs), 'Completed confirmation worker logs are required for lock timing')
    earliest = min(path.stat().st_mtime for path in logs)
    run.require(when.timestamp() < earliest, 'Late lock: confirmation logs precede the locked rule')
    return {'locked_utc': stamp, 'lock_epoch': when.timestamp(),
            'earliest_confirmation_log_mtime': earliest, 'confirmation_log_count': len(logs),
            'scope': 'Lock precedes earliest saved worker-log mtime; controller separately enforces lock before launching any confirmation fit'}


def validate_development_lock(search_plan, manifests, msic_dir, lock):
    """Bind a lock to an already independently validated development search."""
    msic_dir = Path(msic_dir)
    plan = run.read_json(msic_dir/'plan.json')
    run.require(plan['plan_id'] == run.digest({k: v for k, v in plan.items() if k != 'plan_id'}), 'MSIC plan checksum differs')
    summary_sha = run.file_hash(msic_dir/'rule_summary.csv')
    rule = validate_lock(lock, search_plan, plan, summary_sha)
    integrity = run.read_json(msic_dir/'integrity_report.json')
    count = len(search_plan['tasks'])
    run.require(len(manifests) == count and [m['task_id'] for m in manifests]
                == [task['task_id'] for task in search_plan['tasks']], 'Development manifests are missing, duplicated, or reordered')
    run.require(integrity['status'] == 'complete' and integrity['checks_passed']
                and integrity['plan_id'] == plan['plan_id'] and integrity['dataset_count'] == count
                and integrity['rule_rows'] == 11*count and integrity['default_exact_replays'] == count
                and integrity['pool_fingerprints_unchanged'], 'Incomplete MSIC development parent')
    run.require(plan['inputs'] == manifests and plan['frozen_input_sources'] == search_plan['sources'], 'Development MSIC/search manifests differ')
    run.require(plan['rules'] == scoring.declared_rules(), 'Development rule declaration differs')
    run.require(plan['environment'] == run.environment(), 'Development MSIC environment differs')
    for relative, expected in plan['research_sources'].items():
        run.require(run.file_hash(run.root()/relative) == expected, 'Frozen development scoring source changed')
    run.require(lock.get('source_core_sha256') == search_plan['sources']['core'], 'Locked source core differs')
    run.require(lock.get('scoring_source_sha256') == plan['research_sources'], 'Locked scoring source differs')
    jobs = []
    for task, manifest in zip(search_plan['tasks'], manifests):
        run.require(task['task_id'] == manifest['task_id'], 'Development manifest order differs')
        path = msic_dir/'jobs'/f"task_{task['task_id']:03d}.json"
        report = run.read_json(path); request = task['request']
        run.require(report['status'] == 'complete' and report['plan_id'] == plan['plan_id']
                    and report['input_result_manifest'] == manifest, 'Development scoring job parent differs')
        run.require(all(run.canonical(report[k]) == run.canonical(request[k]) for k in ('config', 'seed', 'test_seed')),
                    'Development scoring job request differs')
        run.require(report['environment'] == search_plan['environment']['libraries']
                    and report['fingerprints']['core_sources'] == search_plan['sources']['core'], 'Development scoring job source/environment differs')
        run.require(report['default_exact_replay'] and report['pool_fingerprint_before'] == report['pool_fingerprint_after']
                    == manifest['A3_pool_fingerprint'], 'Development pool/default replay differs')
        rows = report['rows']; expected = {r['rule'] for r in scoring.declared_rules()}
        run.require(len(rows) == 11 and {r['rule'] for r in rows} == expected, 'Development scoring job rules differ')
        cache = path.with_suffix('.npz')
        run.require(report['array_cache'] == cache.name and report['array_cache_sha256'] == run.file_hash(cache), 'Development MSIC cache checksum differs')
        with np.load(cache, allow_pickle=False) as arrays:
            for row in rows:
                index = row['selected_index']
                run.require(array_fingerprint(arrays[f'candidate_{index}_B']) == row['B_fingerprint'], 'Development selected B differs')
                run.require([arrays[f'candidate_{index}_S{m}'].tolist() for m in (1, 2, 3)] == row['supports'], 'Development selected supports differ')
        jobs.append({'task_id': task['task_id'], 'json_sha256': run.file_hash(path), 'npz_sha256': report['array_cache_sha256']})
    return rule, {'plan_id': plan['plan_id'], 'plan_sha256': run.file_hash(msic_dir/'plan.json'),
                  'summary_sha256': summary_sha, 'integrity_sha256': run.file_hash(msic_dir/'integrity_report.json'),
                  'scoring_jobs': jobs, 'research_sources': plan['research_sources']}


def load_context(input_dir, development_search_dir, development_msic_dir, lock_file):
    input_dir, development_search_dir, development_msic_dir, lock_file = map(Path,
        (input_dir, development_search_dir, development_msic_dir, lock_file))
    lock = run.read_json(lock_file)
    development, development_inputs = scoring.validate_complete_cohort(development_search_dir)
    run.require(development['cohort'] == 'development' and len(development['tasks']) == 90,
                'Complete original ninety-dataset development parent required')
    run.require(run.load_plan(development_search_dir) == development, 'Development search protocol differs')
    rule, msic_parent = validate_development_lock(development, development_inputs, development_msic_dir, lock)
    run.require(not (input_dir/'STOPPED_FOR_AUDIT.json').exists(), 'Stopped confirmation is excluded')
    plan = run.load_plan(input_dir)
    validate_confirmation_plan(plan, lock)
    timing = validate_lock_timing(lock, input_dir)
    integrity = run.read_json(input_dir/'integrity_report.json')
    run.require(integrity['status'] == 'complete' and integrity['integrity_checks_passed']
                and integrity['plan_id'] == plan['plan_id'] and integrity['completed_tasks'] == 90
                and integrity['expected_tasks'] == 90 and integrity['paired_group_rows'] == 450,
                'Complete ninety-dataset/five-group confirmation parent required')
    run.require(not any((input_dir/'failures').iterdir()), 'Failed confirmation is excluded')
    manifests = []
    for task in plan['tasks']:
        report = run.validate_result(input_dir, plan, task)
        source = run.result_path(input_dir, task)
        manifests.append({'task_id': task['task_id'], 'json_sha256': run.file_hash(source),
                          'npz_sha256': report['array_cache_sha256'],
                          'request_sha256': run.file_hash(input_dir/task['request_file']),
                          'A3_pool_fingerprint': report['groups']['A3']['pool_fingerprint']})
    parents = {'confirmation_search_plan_id': plan['plan_id'], 'confirmation_plan_sha256': run.file_hash(input_dir/'plan.json'),
               'confirmation_integrity_sha256': run.file_hash(input_dir/'integrity_report.json'),
               'development_search_plan_id': development['plan_id'],
               'development_search_plan_sha256': run.file_hash(development_search_dir/'plan.json'),
               'development_integrity_sha256': run.file_hash(development_search_dir/'integrity_report.json'),
               'development_inputs': development_inputs, 'development_msic': msic_parent,
               'lock_file_sha256': run.file_hash(lock_file), 'lock': lock, 'lock_timing': timing}
    return plan, manifests, rule, parents


def evaluate_task(report, npz_path, locked_rule):
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from numerical_simulation.candidate_search_study.numerics import audit_selected_model
    from tensor_abess.tensor.ops import tensor_loss
    started = perf_counter()
    values = dict(report['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    data = generate_simulation_dataset(config, seed=int(report['seed']))
    for key, value in (('X_train', data.X_train), ('y_train', data.y_train), ('B_true', data.B_true)):
        run.require(array_fingerprint(value) == report['fingerprints'][key], f'Regenerated {key} fingerprint differs')
    group = report['groups']['A3']; metadata = group['candidates']
    original = next(row for row in report['rows'] if row['group'] == 'A3')
    choices = two_choices(metadata, config.n_train, config.dims, locked_rule)
    with np.load(npz_path, allow_pickle=False) as arrays:
        before = pool_fingerprint(scoring.stream_pool(group, arrays))
        run.require(before == group['pool_fingerprint'], 'Frozen confirmation pool fingerprint differs')
        selected = []
        for rule, index, info in choices:
            prefix = f'A3_candidate_{index}_'
            B = arrays[prefix+'B']; supports = [arrays[prefix+f'S{mode}'] for mode in (1, 2, 3)]
            loss = tensor_loss(data.X_train, data.y_train, B); rss = 2*config.n_train*loss
            run.require(np.isclose(rss, metadata[index]['rss'], rtol=1e-10, atol=1e-10)
                        and np.isclose(loss, metadata[index]['loss'], rtol=1e-10, atol=1e-10), 'Training RSS/loss differs')
            selected.append({'B': B, 'supports': supports, 'rss_recomputed': rss, 'loss_recomputed': loss})
        predictions, test_fingerprint = evaluate_shared_test(data, [value['B'] for value in selected], int(report['test_seed']))
        run.require(test_fingerprint == report['fingerprints']['independent_test'], 'Confirmation shared test stream differs')
        locked_numerics = audit_selected_model(data.X_train, data.y_train, selected[1]['B'], selected[1]['supports'],
                            expected_rss=metadata[choices[1][1]]['rss'], expected_loss=metadata[choices[1][1]]['loss'])
        after = pool_fingerprint(scoring.stream_pool(group, arrays))
        run.require(after == before, 'Confirmation evaluation changed frozen pool')
    rows = []; selected_arrays = {}
    for (rule, index, info), fitted, prediction in zip(choices, selected, predictions):
        candidate = metadata[index]; supports = fitted['supports']
        row = {**original, 'rule': rule['label'], 'method': f"A3/{rule['label']}",
               **compute_support_metrics(*supports, *data.supports_true),
               **compute_coefficient_metrics(fitted['B'], data.B_true), **prediction,
               'rss_train': candidate['rss'], 'loss_train': candidate['loss'], 'selection': info,
               'supports': [S.tolist() for S in supports], 'B_fingerprint': array_fingerprint(fitted['B'])}
        if rule['role'] == 'default': scoring.default_replay(original, row)
        row.pop('runtime_sec', None)
        q, q_true = int(row['q_hat']), int(row['q_true'])
        row.update(evaluation_role=rule['role'], locked_rule_id=locked_rule['id'], rule_id=rule['id'],
                   criterion=rule['criterion'], alpha=rule['alpha'], gamma=rule['gamma'], one_se_scale=rule['scale'],
                   selected_index=index, candidate_kind=candidate['kind'], source_A3_runtime_sec=original['runtime_sec'],
                   rss_train_recomputed=fitted['rss_recomputed'], loss_train_recomputed=fitted['loss_recomputed'],
                   q_over=int(q > q_true), q_under=int(q < q_true), q_small=int(q == 1),
                   q_near_cap=int(q >= .9*np.floor(.7*config.n_train)))
        rows.append(row); selected_arrays[rule['role']+'_B'] = fitted['B']
    return {'status': 'complete', 'rows': rows, 'seed': report['seed'], 'test_seed': report['test_seed'],
            'config': report['config'], 'fingerprints': report['fingerprints'], 'environment': report['environment'],
            'default_exact_replay': True, 'pool_fingerprint_before': before, 'pool_fingerprint_after': after,
            'locked_numerics': locked_numerics, 'new_model_fitting': False, 'rule_selection_performed': False,
            'elapsed_sec': perf_counter()-started}, selected_arrays


def run_cohort(input_dir, development_search_dir, development_msic_dir, lock_file, output_dir):
    output = Path(output_dir).resolve()
    if output.exists(): raise FileExistsError('Confirmation scoring requires a new immutable directory')
    input_dir, development_search_dir, development_msic_dir, lock_file = map(lambda p: Path(p).resolve(),
        (input_dir, development_search_dir, development_msic_dir, lock_file))
    configure_core()
    plan, manifests, rule, parents = load_context(input_dir, development_search_dir, development_msic_dir, lock_file)
    paths = [Path(__file__), Path(scoring.__file__), run.root()/'numerical_simulation/candidate_search_study/msic_refit.py',
             run.root()/'numerical_simulation/candidate_search_study/numerics.py',
             run.root()/'numerical_simulation/version_validation/worker.py']
    body = {'stage': 'P4-independent-locked-confirmation', 'research_only': True, 'rule_selection_performed': False,
            'input_dir': str(input_dir), 'development_search_dir': str(development_search_dir),
            'development_msic_dir': str(development_msic_dir), 'lock_file': str(lock_file),
            'parents': parents, 'inputs': manifests, 'locked_rule': rule, 'default_rule': scoring.DEFAULT_RULE,
            'evaluation_roles': ['default', 'locked'], 'rule_count': 2, 'tasks': plan['tasks'],
            'frozen_input_sources': plan['sources'], 'environment': run.environment(),
            'research_sources': {p.relative_to(run.root()).as_posix(): run.file_hash(p) for p in paths},
            'created_utc': datetime.now(timezone.utc).isoformat()}
    own_plan = {**body, 'plan_id': run.digest(body)}
    output.mkdir(); (output/'jobs').mkdir(); run.write_json(output/'plan.json', own_plan)
    rows, numeric = [], []
    for task, manifest in zip(plan['tasks'], manifests):
        run.require(run.file_hash(lock_file) == parents['lock_file_sha256'], 'Immutable lock changed during confirmation evaluation')
        source = run.result_path(input_dir, task)
        run.require(run.file_hash(source) == manifest['json_sha256'] and run.file_hash(source.with_suffix('.npz')) == manifest['npz_sha256'],
                    'Confirmation source changed after validation')
        result, arrays = evaluate_task(run.read_json(source), source.with_suffix('.npz'), rule)
        result.update(task_id=task['task_id'], plan_id=own_plan['plan_id'], lock_sha256=parents['lock_file_sha256'],
                      input_result_manifest=manifest, locked_rule=rule)
        for row in result['rows']: row['task_id'] = task['task_id']
        target = output/'jobs'/f"task_{task['task_id']:03d}.json"
        with target.with_suffix('.npz').open('xb') as handle: np.savez_compressed(handle, **arrays)
        result.update(array_cache=target.with_suffix('.npz').name, array_cache_sha256=run.file_hash(target.with_suffix('.npz')))
        run.write_json(target, result); rows.extend(result['rows'])
        numeric.append({'task_id': task['task_id'], 'seed': result['seed'], 'snr': result['config']['snr'],
                        'rule': rule['id'], **result['locked_numerics']})
        print(f"Locked confirmation scoring: {len(rows)//2}/90; task {task['task_id']}", flush=True)
    run.require(len(rows) == 180 and len(numeric) == 90, 'Incomplete two-rule confirmation output')
    _, current_inputs, current_rule, current_parents = load_context(input_dir, development_search_dir, development_msic_dir, lock_file)
    run.require(current_inputs == manifests and current_rule == rule and current_parents == parents,
                'Confirmation/development/lock parents changed during evaluation')
    compact = [{k: v for k, v in row.items() if k not in ('selection', 'supports')} for row in rows]
    scoring.write_csv(output/'raw_results.csv', compact)
    scoring.write_csv(output/'rule_summary.csv', scoring.summary_rows(rows, scenarios=False))
    scoring.write_csv(output/'scenario_summary.csv', scoring.summary_rows(rows, scenarios=True))
    scoring.write_csv(output/'paired_vs_default.csv', scoring.paired_rows(rows))
    run.write_csv(output/'locked_model_numerics.csv', numeric)
    conditions = [r['condition_number'] for r in numeric if r['condition_number'] is not None]
    numeric_summary = {'status': 'complete', 'checks_passed': True, 'locked_model_count': 90,
        'rank_deficient_count': sum(r['rank_deficient'] for r in numeric), 'condition_number_max': max(conditions) if conditions else None,
        'max_rss_relative_difference': max(r['rss_relative_difference'] for r in numeric),
        'max_ols_rss_relative_difference': max(r['ols_rss_relative_difference'] for r in numeric),
        'max_normal_equation_relative_residual': max(r['normal_equation_relative_residual'] for r in numeric),
        'scope': 'Only ninety locked-rule selected models; default five-group 450-model numerics remain in parent. No coefficients or supports replaced.',
        'tolerances': {'relative_scaled_rss_and_prediction': 1e-9, 'normal_equation_relative_residual': 1e-8}}
    run.write_json(output/'locked_numerics_diagnostics.json', numeric_summary)
    integrity = {'status': 'complete', 'checks_passed': True, 'plan_id': own_plan['plan_id'],
                 'confirmation_search_plan_id': plan['plan_id'], 'dataset_count': 90, 'rule_rows': 180, 'rule_count': 2,
                 'default_exact_replays': 90, 'pool_fingerprints_unchanged': True, 'locked_model_numerics_checked': 90,
                 'lock_sha256': parents['lock_file_sha256'], 'research_only': True,
                 'rule_selection_performed': False, 'new_model_fitting': False,
                 'interpretation': 'Independent confirmation of prior lock; parent A0/A2 comparators retained; no confirmation-based rule tuning or formal adoption.'}
    run.write_json(output/'integrity_report.json', integrity)
    return integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir', 'development-search-dir', 'development-msic-dir', 'lock-file', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_cohort(args.input_dir, args.development_search_dir, args.development_msic_dir, args.lock_file, args.output_dir)
    print(f"Confirmation complete: {summary['dataset_count']} datasets, exactly {summary['rule_rows']} default/locked rows")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
