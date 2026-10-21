"""Prespecified training-only LEMON selection feasibility; frozen v5.2 core.

No outer-test predictions or metrics are computed. Independent workers use
fresh inner preprocessing and search, then select a full-training refit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from scripts.taskcard7.lemon import (array_sha, configure_core, observe_fit,
                                    read, require, root, sha, write, write_csv)


def common_grid(grids):
    common = set(map(tuple, grids[0])).intersection(*(set(map(tuple, g)) for g in grids[1:]))
    require(bool(common), 'No legal common size grid')
    return sorted(common)


def fixed_map(trace):
    require(len(trace['candidates']) == len(trace['ordered_size_grid']) + 1,
            'Fixed candidate count differs from actual grid')
    result = {}
    for index, sizes in enumerate(trace['ordered_size_grid'], start=1):
        record = trace['candidates'][index]
        require(record['sizes'] == sizes and tuple(sizes) not in result,
                'Fixed candidate identity differs from actual grid')
        result[tuple(sizes)] = (index, record)
    return result


def shape_key(row):
    sizes = tuple(row['sizes'])
    return int(np.prod(sizes)), sum(sizes), sizes


def select_cv(rows):
    scores = [sum(r['inner_sse']) / sum(r['inner_n']) for r in rows]
    best = min(scores)
    eligible = [r for r, score in zip(rows, scores) if abs(score - best) <= 1e-12]
    return min(eligible, key=shape_key)


def stability(supports):
    values = []
    for left, right in combinations(supports, 2):
        for a, b in zip(left, right):
            a, b = set(a), set(b)
            require(bool(a | b), 'Empty support cannot define Jaccard')
            values.append(len(a & b) / len(a | b))
    require(len(values) == 9, 'Stability requires three folds and three modes')
    return float(np.mean(values))


def select_msic(rows, n, stable=False):
    best = min(rows, key=lambda r: r['msic'])
    tolerance = float(np.sqrt(max(best['q'], 1) * max(np.log(max(n, 2)), 1.)))
    eligible = [r for r in rows if r['msic'] <= best['msic'] + tolerance]
    key = lambda r: (r['q'], r['msic'], sum(r['sizes']), tuple(r['sizes']))
    stable_key = lambda r: (-r['stability'], *key(r))
    return min(eligible, key=stable_key if stable else key)


def validate_inner_partition(outer_train, folds):
    universe = set(map(int, outer_train))
    validation = []
    for train, valid in folds:
        a, b = set(map(int, train)), set(map(int, valid))
        require(not a & b and a | b == universe and len(a) == len(train) and len(b) == len(valid),
                'Inner partition leaks, duplicates, or omits a training subject')
        validation.extend(map(int, valid))
    require(len(validation) == len(universe) and set(validation) == universe,
            'Inner partition validation is not an exact cover')


def source_identity():
    folder = root() / 'doc/algorithm_versions/v5.2-search-final'
    manifest = read(folder / 'VERSION.json')
    sources = manifest['source_sha256']
    require(len(sources) == 32, 'Frozen v5.2 must have exactly 32 declared sources')
    require({p.relative_to(folder / 'src').as_posix() for p in (folder / 'src').rglob('*.py')} == set(sources),
            'Frozen v5.2 source inventory differs')
    for name, digest in sources.items():
        require(sha(folder / 'src' / name) == digest, 'Frozen v5.2 SHA differs: ' + name)
    return {'version': manifest['version'], 'source_sha256': sources,
            'version_json_sha256': sha(folder / 'VERSION.json')}


def remaining_budget(plan, now=None):
    current = datetime.now(timezone.utc) if now is None else now
    remaining = (datetime.fromisoformat(plan['shared_budget']['deadline_utc']) - current).total_seconds()
    require(remaining > 0, 'Shared D compute budget exhausted')
    return remaining


def inputs():
    from empirical_analysis.lemon.incremental_value.data_loader import load_audited_dataset
    from empirical_analysis.lemon.incremental_value.nested_cv import make_splits
    from empirical_analysis.lemon.incremental_value.run import _versions
    config = read(root() / 'empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final/config_used.json')
    require(_versions() == config['environment'], 'Historical LEMON environment identity differs')
    require(config['model']['msic_refine_alpha'] == 1.5 and
            config['model']['msic_refine_gamma'] == 1. and
            config['model']['msic_refine_one_se_scale'] == 1., 'Original MSIC options changed')
    data = load_audited_dataset(root() / 'empirical_analysis/lemon/outputs_lemon_adjusted',
                               root() / 'empirical_analysis/TMT.csv',
                               root() / 'empirical_analysis/lemon/outputs_demographics/participants.tsv')
    historical_plan = read(root() / 'numerical_simulation/outputs_version_attribution/lemon/plan.json')
    require(data.fingerprints == historical_plan['input_fingerprints'],
            'LEMON input identity differs from historical attribution')
    splits = [s for s in make_splits(data.C, config['outer_folds'], config['outer_seeds'])
              if s['outer_fold'] == 1]
    require([s['repeat'] for s in splits] == [0, 1, 2], 'Prespecified repeats differ')
    require(all(len(s['train']) == 116 for s in splits), 'Outer training count differs from 116')
    for split in splits:
        saved = read(root() / 'empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final/cv/folds' /
                     f'repeat_{split["repeat"]}_fold_1/split.json')
        require(saved['train_subject_ids'] == data.subject_ids[split['train']].tolist(),
                'Outer training split differs from saved v5.2 identity')
    return config, data, splits


def make_protocol():
    config, data, splits = inputs()
    budget_path = root() / 'doc/taskcard8/COMPUTE_BUDGET.json'
    shared_budget = read(budget_path)
    require(shared_budget['user_approved'] and shared_budget['wall_time_limit_seconds'] == 7200,
            'Shared D budget must be explicitly approved for 7200 seconds')
    return {'schema_version': 1, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'shared_budget': shared_budget, 'shared_budget_sha256': sha(budget_path),
            'shared_budget_path': 'doc/taskcard8/COMPUTE_BUDGET.json',
            'research_only': True, 'task': 'taskcard8_D2_training_only_selection_feasibility',
            'source_identity': source_identity(), 'implementation_sha256': sha(__file__),
            'test_sha256': sha(Path(__file__).with_name('test_inner_selection.py')),
            'passive_observer_sha256': sha(root() / 'scripts/taskcard7/lemon.py'),
            'config': config, 'input_fingerprints': data.fingerprints,
            'outer_training_subject_ids': {str(s['repeat']): data.subject_ids[s['train']].tolist() for s in splits},
            'fold_indices': [[0, 1], [1, 1], [2, 1]],
            'inner_seeds': [config['inner_seed'] + repeat * 100 + 1 for repeat in range(3)],
            'budget_seconds': 7200,
            'search': 'Full true v5.2 adaptive plus complete fixed search separately on each 77/78 inner training and 116 outer training; unchanged caps, alpha=1/global alpha=1.5/gamma=1/one_se_scale=1',
            'candidate_domain': 'Only fixed-grid candidates, ordered intersection of all three actual inner grids and full-training grid; adaptive duplicate excluded',
            'rules': ['native_full_training_original_msic', 'common_grid_original_msic',
                      'inner_cv_size', 'stability_auxiliary_msic'],
            'cv_rule': 'Pooled validation SSE over all 116 subjects; absolute 1e-12 MSE tie then q, sum(sizes), lexicographic sizes',
            'cv_interpretation': 'Every reported CV MSE is for a fixed candidate size with fresh inner support searches, not CV of the complete MSIC selector; native size has no CV MSE outside common grid',
            'msic_rule': 'On common fixed grid original one-SE tolerance sqrt(q_best*log(n116)); then q, score, sum(sizes), lexicographic sizes',
            'stability_rule': 'Same common-grid MSIC one-SE eligible band; maximize mean of 9 pairwise mode support Jaccards, then q, score, sum(sizes), lexicographic sizes; no tuning',
            'refit': 'Each chosen size uses corresponding independent full-training fixed candidate, including its freshly searched support and B; native rule uses native selected B',
            'forbidden': ['outer_test_predictions', 'outer_test_metrics', 'outer_test_selection',
                          'full_training_support_reuse_for_inner_validation', 'production_changes'],
            'limitations': ['Only three correlated training subsets; exploratory feasibility, no new method release',
                            'CV MSE used for choosing size is selection-optimistic, not unbiased generalization or independent confirmation',
                            'Stability can favor small supports and does not establish prediction benefit',
                            'Native full-grid model can be outside common capacities; its common-grid CV score then unavailable']}


def verify_protocol(plan):
    require(plan['shared_budget_sha256'] == sha(root() / plan['shared_budget_path']),
            'Shared D budget changed after freeze')
    remaining_budget(plan)
    require(plan['implementation_sha256'] == sha(__file__), 'Implementation changed after freeze')
    require(plan['test_sha256'] == sha(Path(__file__).with_name('test_inner_selection.py')),
            'Tests changed after freeze')
    require(plan['source_identity'] == source_identity(), 'Source identity changed after freeze')
    require(plan['passive_observer_sha256'] == sha(root() / 'scripts/taskcard7/lemon.py'), 'Observer changed after freeze')
    config, data, splits = inputs()
    require(config == plan['config'] and data.fingerprints == plan['input_fingerprints'], 'Inputs changed after freeze')
    return config, data, splits


def verify_import(source):
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            require(Path(module.__file__).resolve().is_relative_to(source), 'Actual import escaped frozen v5.2')


def fit_dataset(X, y, C, options, folder):
    from empirical_analysis.lemon.incremental_value.preprocessing import fit_preprocessor
    from empirical_analysis.lemon.incremental_value.model_adapters import fit_eeg_model
    require(not folder.exists(), 'Fit directory already exists; refusing overwrite')
    folder.mkdir(parents=True)
    preprocessor = fit_preprocessor(X, y, C)
    started = perf_counter()
    fitted, trace, coefficients = observe_fit(lambda: fit_eeg_model(
        'Tensor-ABESS', preprocessor.apply(X, C), preprocessor.transform_response(y, C), None, options))
    trace.update(n_train=len(y), capacity=int(np.floor(.7 * len(y))), elapsed_seconds=perf_counter()-started)
    require(array_sha(fitted.B) == trace['final']['B_sha256'], 'Native fit differs from captured selection')
    for record, B in zip(trace['candidates'], coefficients):
        require(record['B_sha256'] == array_sha(B), 'Captured coefficient identity differs')
    np.savez_compressed(folder / 'preprocessing.npz', **preprocessor.to_arrays())
    np.savez_compressed(folder / 'candidate_coefficients.npz', **{f'B_{i}': B for i, B in enumerate(coefficients)})
    write(folder / 'trace.json', trace)
    return fitted, preprocessor, trace, coefficients


def worker(output, repeat):
    plan = read(output / 'protocol.json')
    source = configure_core('v5.2-search-final')
    config, data, splits = verify_protocol(plan)
    from empirical_analysis.lemon.incremental_value.nested_cv import make_splits
    split = splits[repeat]
    outer = split['train']
    X, y, C, ids = data.X[outer], data.y[outer], data.C[outer], data.subject_ids[outer]
    inner_splits = make_splits(C, 3, [plan['inner_seeds'][repeat]])
    validate_inner_partition(outer, [(outer[s['train']], outer[s['test']]) for s in inner_splits])
    folder = output / f'repeat_{repeat}_fold_1'
    require(not folder.exists(), 'Worker output exists; refusing overwrite')
    folder.mkdir()
    write(folder / 'split.json', {'repeat': repeat, 'outer_fold': 1,
          'outer_train_subject_ids': ids.tolist(), 'inner_seed': plan['inner_seeds'][repeat],
          'inner_splits': [{'inner_fold': s['outer_fold'], 'train_subject_ids': ids[s['train']].tolist(),
                            'validation_subject_ids': ids[s['test']].tolist()} for s in inner_splits]})
    inner = []
    for split_inner in inner_splits:
        remaining_budget(plan)
        train, valid = split_inner['train'], split_inner['test']
        _, p, trace, coefficients = fit_dataset(X[train], y[train], C[train], config['model'],
                                                folder / f'inner_{split_inner["outer_fold"]}')
        mapping = fixed_map(trace)
        scores = {}
        for sizes, (index, record) in mapping.items():
            prediction = p.predict(coefficients[index], X[valid], C[valid])
            residual = y[valid] - prediction
            scores[sizes] = {'sse': float(residual @ residual), 'n': len(valid),
                             'support': record['support'], 'candidate_index': index}
        write(folder / f'inner_{split_inner["outer_fold"]}' / 'validation.json',
              {'validation_subject_ids': ids[valid].tolist(),
               'scores': [{'sizes': list(sizes), **record} for sizes, record in sorted(scores.items())]})
        inner.append({'trace': trace, 'scores': scores})
    remaining_budget(plan)
    fitted, p, full, coefficients = fit_dataset(X, y, C, config['model'], folder / 'full_training')
    fixed = fixed_map(full)
    common = common_grid([full['ordered_size_grid'], *(part['trace']['ordered_size_grid'] for part in inner)])
    curve = []
    for sizes in common:
        index, record = fixed[sizes]
        scores = [part['scores'][sizes] for part in inner]
        error = y - p.predict(coefficients[index], X, C)
        row = dict(record, full_candidate_index=index, inner_sse=[s['sse'] for s in scores],
                   inner_n=[s['n'] for s in scores], inner_supports=[s['support'] for s in scores],
                   stability=stability([s['support'] for s in scores]),
                   training_rmse=float(np.sqrt(np.mean(error**2))))
        row['pooled_inner_cv_mse'] = sum(row['inner_sse']) / sum(row['inner_n'])
        curve.append(row)
    selected = {'common_grid_original_msic': select_msic(curve, len(y)),
                'inner_cv_size': select_cv(curve),
                'stability_auxiliary_msic': select_msic(curve, len(y), stable=True)}
    baseline = full['final']
    baseline_index = next(i for i, r in enumerate(full['candidates'])
                          if r['B_sha256'] == baseline['B_sha256'] and r['support'] == baseline['support'])
    baseline_curve = next((r for r in curve if r['sizes'] == baseline['sizes']), None)
    selected['native_full_training_original_msic'] = dict(baseline, full_candidate_index=baseline_index,
        pooled_inner_cv_mse=None if baseline_curve is None else baseline_curve['pooled_inner_cv_mse'],
        stability=None if baseline_curve is None else baseline_curve['stability'])
    records, final_B = [], {}
    for rule, row in selected.items():
        B = coefficients[row['full_candidate_index']]
        require(array_sha(B) == row['B_sha256'], 'Final refit coefficient identity differs')
        error = y - p.predict(B, X, C)
        records.append({**row, 'rule': rule, 'training_rmse': float(np.sqrt(np.mean(error**2)))})
        final_B[rule] = B
    verify_import(source)
    require(source_identity() == plan['source_identity'], 'Frozen sources changed during fit')
    np.savez_compressed(folder / 'selected_coefficients.npz', **final_B)
    best = min(curve, key=lambda r: r['msic'])
    tolerance = float(np.sqrt(best['q'] * np.log(len(y))))
    for row in curve:
        row['common_grid_one_se_eligible'] = row['msic'] <= best['msic'] + tolerance
    write(folder / 'common_candidate_curve.json', curve)
    write_csv(folder / 'common_candidate_curve.csv', [
        {k: row[k] for k in ('sizes', 'q', 'rss', 'msic', 'training_rmse', 'stability',
                             'pooled_inner_cv_mse', 'common_grid_one_se_eligible', 'B_sha256')} for row in curve])
    summary = {'repeat': repeat, 'outer_fold': 1, 'n_train': len(y), 'actual_import': str(source),
               'full_caps': full['caps'], 'full_Q': full['capacity'], 'full_fixed_grid_count': len(fixed),
               'inner_caps': [part['trace']['caps'] for part in inner],
               'inner_Q': [part['trace']['capacity'] for part in inner],
               'inner_n_train': [part['trace']['n_train'] for part in inner],
               'inner_grid_counts': [len(part['trace']['ordered_size_grid']) for part in inner],
               'common_grid_count': len(common), 'common_grid': [list(s) for s in common],
               'common_min_msic': best['msic'], 'common_one_se_tolerance': tolerance,
               'selected': records, 'outer_test_evaluations': 0,
               'limitations': plan['limitations']}
    write(folder / 'summary.json', summary)
    print(f'Completed training-only repeat {repeat}: common grid {len(common)}', flush=True)


def run(output):
    require(not output.exists(), 'Output exists; refusing overwrite')
    plan = make_protocol()
    output.mkdir(parents=True)
    write(output / 'protocol.json', plan)
    started = perf_counter()
    for repeat in range(3):
        remaining = remaining_budget(plan)
        command = [sys.executable, '-B', '-m', 'scripts.taskcard8.inner_selection',
                   '--output-dir', str(output), '--worker', str(repeat)]
        with (output / f'worker_{repeat}.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, check=True, timeout=remaining, cwd=root(), stdout=log,
                           stderr=subprocess.STDOUT, env=os.environ.copy())
    summaries = [read(output / f'repeat_{repeat}_fold_1/summary.json') for repeat in range(3)]
    write(output / 'summary.json', {'complete': True, 'elapsed_seconds': perf_counter()-started,
          'protocol_sha256': sha(output / 'protocol.json'), 'outer_test_evaluations': 0,
          'summaries': summaries, 'limitations': plan['limitations']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=root() / 'numerical_simulation/outputs_selection_inner_task8')
    parser.add_argument('--worker', type=int, choices=range(3))
    args = parser.parse_args()
    require(all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'All four thread limits must equal 1')
    (run(args.output_dir) if args.worker is None else worker(args.output_dir, args.worker))


if __name__ == '__main__':
    main()
