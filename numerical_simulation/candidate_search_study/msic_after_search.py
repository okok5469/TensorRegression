"""Global-only scoring on immutable A3 libraries after search-recovery validation.

This research stage does not fit any BCD/Group-BESS model, change the core,
choose a winning rule, or inspect a confirmation cohort. Test outcomes are
descriptive development evidence, not independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from numerical_simulation.candidate_search_study import run
from numerical_simulation.candidate_search_study.worker import configure_core, pool_fingerprint
from numerical_simulation.version_validation.worker import (
    array_fingerprint, evaluate_shared_test, select_candidate,
)

DEFAULT_RULE = 'msic-alpha1.5-scale1'
PRIORITY_RULE = 'msic-alpha2-scale0'
METRICS = ('q_hat', 'exact_recovery', 'mode_f1_macro', 'element_f1', 'ree',
           'pmse', 'signal_mse', 'r2_test', 'rss_train', 'loss_train',
           'q_over', 'q_under', 'q_small', 'q_near_cap')


def declared_rules():
    msic = [dict(rule=f'msic-alpha{alpha:g}-scale{scale:g}', criterion='msic', alpha=alpha,
                 gamma=1.0, scale=scale)
            for alpha in (1.5, 2.0, 3.0) for scale in (0.0, 0.5, 1.0)]
    return [*msic, dict(rule='tensor-sic', criterion='tensor-sic', alpha=None, gamma=1.0, scale=0.0),
            dict(rule='pure-bic', criterion='bic', alpha=None, gamma=None, scale=0.0)]


def rescore_metadata(pool, n, dims):
    before = run.canonical(pool)
    choices = []
    for rule in declared_rules():
        _, info = select_candidate(pool, n, dims, criterion=rule['criterion'],
                                  alpha=rule['alpha'] or 1.5, scale=rule['scale'])
        choices.append((rule, int(info['selected_index']), info))
    run.require(run.canonical(pool) == before, 'Scoring mutated frozen candidate metadata')
    return choices


def validate_complete_cohort(input_dir):
    input_dir = Path(input_dir).resolve()
    run.require(not (input_dir/'STOPPED_FOR_AUDIT.json').exists(), 'Stopped cohort is excluded')
    plan = run.read_json(input_dir/'plan.json')
    run.require(plan.get('cohort') in ('historical', 'development'),
                'confirmation requires an immutable LOCK and default+locked rules; eleven-rule scoring is forbidden')
    run.require(plan['plan_id'] == run.digest({k: v for k, v in plan.items() if k != 'plan_id'}),
                'Input plan checksum mismatch')
    integrity = run.read_json(input_dir/'integrity_report.json')
    run.require(integrity['status'] == 'complete' and integrity['integrity_checks_passed']
                and integrity['plan_id'] == plan['plan_id']
                and integrity['completed_tasks'] == integrity['expected_tasks'] == len(plan['tasks']),
                'Input cohort is incomplete')
    run.require(plan['sources'] == run.source_manifests(), 'Frozen input sources changed')
    run.require(plan['environment']['libraries'] == run.environment()['libraries'], 'Input environment differs')
    run.require('A3' in plan['groups'], 'Frozen A3 group is missing')
    scene_counts = Counter((task['request']['config']['n_train'], task['request']['config']['snr'])
                           for task in plan['tasks'])
    run.require(len(scene_counts) == 9 and len(set(scene_counts.values())) == 1,
                'Complete balanced nine-scenario cohort required')
    run.require(len({task['task_id'] for task in plan['tasks']}) == len(plan['tasks']), 'Duplicate task IDs')
    inputs = []
    for task in plan['tasks']:
        request_path = input_dir/task['request_file']
        request = run.read_json(request_path)
        run.require(request == task['request'] and run.digest(request) == task['request_id'], 'Frozen request differs')
        report = run.validate_result(input_dir, plan, task)
        path = run.result_path(input_dir, task)
        inputs.append({'task_id': task['task_id'], 'json_sha256': run.file_hash(path),
                       'npz_sha256': report['array_cache_sha256'],
                       'request_sha256': run.file_hash(request_path),
                       'A3_pool_fingerprint': report['groups']['A3']['pool_fingerprint']})
    return plan, inputs


def stream_pool(group, arrays):
    for index, candidate in enumerate(group['candidates']):
        run.require(candidate['candidate_index'] == index, 'Frozen candidate ordering differs')
        prefix = f'A3_candidate_{index}_'
        B = arrays[prefix+'B']; supports = [arrays[prefix+f'S{mode}'] for mode in (1, 2, 3)]
        run.require(array_fingerprint(B) == candidate['B_fingerprint'], 'Candidate B fingerprint differs')
        run.require([S.tolist() for S in supports] == candidate['supports'], 'Candidate supports differ')
        yield {'B': B, 'supports': supports, **{key: candidate[key] for key in ('sizes', 'rss', 'loss')}}


def default_replay(old, actual):
    for key, expected in old.items():
        if key == 'runtime_sec':
            continue
        run.require(key in actual and run.canonical(actual[key]) == run.canonical(expected),
                    f'Default replay differs for {key}')


def rescore_task(report, npz_path):
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from tensor_abess.tensor.ops import tensor_loss

    started = perf_counter()
    values = dict(report['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    dataset = generate_simulation_dataset(config, seed=int(report['seed']))
    for key, value in (('X_train', dataset.X_train), ('y_train', dataset.y_train), ('B_true', dataset.B_true)):
        run.require(array_fingerprint(value) == report['fingerprints'][key], f'Regenerated {key} fingerprint differs')
    group = report['groups']['A3']
    metadata = group['candidates']
    original = next(row for row in report['rows'] if row['group'] == 'A3')
    choices = rescore_metadata(metadata, config.n_train, config.dims)
    with np.load(npz_path, allow_pickle=False) as arrays:
        before = pool_fingerprint(stream_pool(group, arrays))
        run.require(before == group['pool_fingerprint'], 'Frozen A3 pool fingerprint differs')
        selected = {}
        for _, index, _ in choices:
            if index in selected:
                continue
            prefix = f'A3_candidate_{index}_'
            selected[index] = {'B': arrays[prefix+'B'],
                'supports': [arrays[prefix+f'S{mode}'] for mode in (1, 2, 3)]}
            recomputed_loss = tensor_loss(dataset.X_train, dataset.y_train, selected[index]['B'])
            recomputed_rss = 2*config.n_train*recomputed_loss
            run.require(np.isclose(recomputed_rss, metadata[index]['rss'], rtol=1e-10, atol=1e-10)
                        and np.isclose(recomputed_loss, metadata[index]['loss'], rtol=1e-10, atol=1e-10),
                        'Training RSS/loss differs from the frozen selected coefficients')
            selected[index].update(rss_recomputed=recomputed_rss, loss_recomputed=recomputed_loss)
        predictions, test_fingerprint = evaluate_shared_test(dataset,
            [selected[index]['B'] for _, index, _ in choices], int(report['test_seed']))
        run.require(test_fingerprint == report['fingerprints']['independent_test'], 'Shared independent test fingerprint differs')
        after = pool_fingerprint(stream_pool(group, arrays))
        run.require(after == before, 'Scoring/test evaluation changed frozen training pool')
    rows = []; selected_arrays = {}
    for (rule, index, info), prediction in zip(choices, predictions):
        candidate = metadata[index]; fitted = selected[index]; supports = fitted['supports']
        row = {**original, 'rule': rule['rule'], 'method': f"A3/{rule['rule']}",
            **compute_support_metrics(*supports, *dataset.supports_true),
            **compute_coefficient_metrics(fitted['B'], dataset.B_true), **prediction,
            'rss_train': candidate['rss'], 'loss_train': candidate['loss'], 'selection': info,
            'supports': [S.tolist() for S in supports], 'B_fingerprint': array_fingerprint(fitted['B'])}
        if rule['rule'] == DEFAULT_RULE:
            default_replay(original, row)
        row.pop('runtime_sec', None)
        q, q_true = int(row['q_hat']), int(row['q_true'])
        row.update(criterion=rule['criterion'], alpha=rule['alpha'], gamma=rule['gamma'],
                   one_se_scale=rule['scale'], selected_index=index,
                   candidate_kind=candidate['kind'], source_A3_runtime_sec=original['runtime_sec'],
                   rss_train_recomputed=fitted['rss_recomputed'], loss_train_recomputed=fitted['loss_recomputed'],
                   q_over=int(q > q_true), q_under=int(q < q_true), q_small=int(q == 1),
                   q_near_cap=int(q >= .9*np.floor(.7*config.n_train)))
        rows.append(row)
    for index, fitted in selected.items():
        selected_arrays[f'candidate_{index}_B'] = fitted['B']
        for mode, S in enumerate(fitted['supports'], 1): selected_arrays[f'candidate_{index}_S{mode}'] = S
    return {'status': 'complete', 'seed': report['seed'], 'test_seed': report['test_seed'],
            'config': report['config'], 'rows': rows, 'default_exact_replay': True,
            'pool_fingerprint_before': before, 'pool_fingerprint_after': after,
            'fingerprints': report['fingerprints'], 'environment': report['environment'],
            'elapsed_sec': perf_counter()-started,
            'memory_policy': 'Candidate metadata scored; each B streamed for hashing, only selected B tensors retained'}, selected_arrays


def summary_rows(rows, *, scenarios):
    keys = (lambda row: (row['n_train'], row['snr'], row['rule'])) if scenarios else (lambda row: (row['rule'],))
    groups = {}
    for row in rows: groups.setdefault(keys(row), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        base = {'rule': key[-1], 'n_datasets': len(values)}
        if scenarios: base.update(n_train=key[0], snr=key[1])
        for metric in METRICS:
            array = np.array([row[metric] for row in values], dtype=float)
            base[metric+'_mean'] = float(np.mean(array))
            base[metric+'_std'] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
            base[metric+'_se'] = base[metric+'_std']/np.sqrt(len(array))
        output.append(base)
    return output


def paired_rows(rows):
    defaults = {row['task_id']: row for row in rows if row['rule'] == DEFAULT_RULE}
    output = []
    for row in rows:
        before = defaults[row['task_id']]
        output.append({'task_id': row['task_id'], 'n_train': row['n_train'], 'snr': row['snr'],
            'seed': row['seed'], 'rule': row['rule'], 'selected_index': row['selected_index'],
            'default_selected_index': before['selected_index'],
            'B_changed': int(row['B_fingerprint'] != before['B_fingerprint']),
            **{metric+'_diff': float(row[metric]-before[metric]) for metric in METRICS}})
    return output


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({key: run.canonical(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def run_cohort(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Rescoring requires a new immutable output directory')
    configure_core()
    plan, inputs = validate_complete_cohort(input_dir)
    own_sources = [Path(__file__), run.root()/'numerical_simulation/version_validation/worker.py',
                   run.root()/'numerical_simulation/candidate_search_study/worker.py']
    body = {'stage': 'P4-frozen-A3-global-scoring', 'research_only': True,
        'input_dir': str(input_dir), 'input_plan_id': plan['plan_id'], 'source_plan_id': plan['plan_id'], 'input_cohort': plan['cohort'],
        'input_plan_sha256': run.file_hash(input_dir/'plan.json'),
        'input_integrity_sha256': run.file_hash(input_dir/'integrity_report.json'), 'inputs': inputs,
        'frozen_input_sources': plan['sources'], 'environment': run.environment(),
        'research_sources': {p.relative_to(run.root()).as_posix(): run.file_hash(p) for p in own_sources},
        'rules': declared_rules(), 'default_rule': DEFAULT_RULE, 'predeclared_priority_candidate': PRIORITY_RULE,
        'rule_selection_performed': False, 'confirmation_performed': False, 'new_model_fitting': False,
        'purpose': 'Descriptive balanced development/historical rescoring only; lock a development rule before untouched confirmation.',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    research_plan = {**body, 'plan_id': run.digest(body)}
    output_dir.mkdir(parents=True); (output_dir/'jobs').mkdir()
    run.write_json(output_dir/'plan.json', research_plan)
    rows = []; started = perf_counter()
    for task, manifest in zip(plan['tasks'], inputs):
        source = run.result_path(input_dir, task)
        run.require(run.file_hash(source) == manifest['json_sha256'], 'Input result changed after plan validation')
        run.require(run.file_hash(source.with_suffix('.npz')) == manifest['npz_sha256'], 'Input cache changed after plan validation')
        result, selected = rescore_task(run.read_json(source), source.with_suffix('.npz'))
        result.update(task_id=task['task_id'], input_result_manifest=manifest, plan_id=research_plan['plan_id'])
        for row in result['rows']: row['task_id'] = task['task_id']
        target = output_dir/'jobs'/f"task_{task['task_id']:03d}.json"
        cache = target.with_suffix('.npz')
        with cache.open('xb') as handle: np.savez_compressed(handle, **selected)
        result.update(array_cache=cache.name, array_cache_sha256=run.file_hash(cache))
        run.write_json(target, result); rows.extend(result['rows'])
        print(f"Frozen-pool scoring completed {len(rows)//11}/{len(plan['tasks'])}: task {task['task_id']}", flush=True)
    run.require(len(rows) == 11*len(plan['tasks']), 'Incomplete eleven-rule output')
    compact = [{k: v for k, v in row.items() if k not in ('selection', 'supports')} for row in rows]
    write_csv(output_dir/'raw_results.csv', compact)
    write_csv(output_dir/'scenario_summary.csv', summary_rows(rows, scenarios=True))
    write_csv(output_dir/'rule_summary.csv', summary_rows(rows, scenarios=False))
    write_csv(output_dir/'paired_vs_default.csv', paired_rows(rows))
    summary = {'status': 'complete', 'checks_passed': True, 'plan_id': research_plan['plan_id'],
        'input_plan_id': plan['plan_id'], 'source_plan_id': plan['plan_id'], 'dataset_count': len(plan['tasks']), 'rule_rows': len(rows),
        'default_exact_replays': len(plan['tasks']), 'pool_fingerprints_unchanged': True,
        'research_only': True, 'rule_selection_performed': False, 'confirmation_performed': False,
        'new_model_fitting': False, 'runtime_sec': perf_counter()-started,
        'interpretation': 'Nine scenarios and pooled results are descriptive development/historical evidence; pooled SE does not assume one homogeneous population. Source A3 runtime shares prior caches and is not a new cold-fit benchmark.'}
    run.write_json(output_dir/'integrity_report.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_cohort(args.input_dir, args.output_dir)
    print(f"Complete: {summary['dataset_count']} frozen A3 datasets, {summary['rule_rows']} rule evaluations", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
