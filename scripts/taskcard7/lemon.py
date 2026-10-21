"""Strict, isolated 15-fold v5.1 LEMON attribution; frozen files stay read-only.

Run with the historical .venv interpreter, four BLAS thread variables set to 1:
python -B -m scripts.taskcard7.lemon --output-dir numerical_simulation/outputs_version_attribution/lemon
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def root():
    return Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(array.shape), 'dtype': array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_preprocessing(saved, current):
    require(set(saved) == set(current), 'Saved/current preprocessing keys differ')
    for key in saved:
        require(np.array_equal(saved[key], current[key]), 'Saved/current preprocessing differs: ' + key)


def model_record(result, score=None):
    supports = [getattr(result, key).astype(int).tolist() for key in ('S1', 'S2', 'S3')]
    return {'support': supports, 'sizes': list(map(len, supports)),
            'q': int(np.prod(list(map(len, supports)))), 'rss': float(result.rss),
            'msic': float(result.sic if score is None else score),
            'B_sha256': array_sha(result.B)}


def observe_fit(fit):
    """Observe real return-frame locals without replacing solver functions."""
    require(sys.getprofile() is None, 'An existing profiler prevents passive trace isolation')
    trace, coefficients = {}, []

    def profile(frame, event, value):
        if event != 'return' or not frame.f_code.co_filename.replace('\\', '/').endswith('/tensor_abess/tensor/bcd.py'):
            return
        name, values = frame.f_code.co_name, frame.f_locals
        if name == 'tensor_bcd_adaptive':
            trace['initial_support'] = [values[key].astype(int).tolist() for key in ('S1_init', 'S2_init', 'S3_init')]
            trace['caps'] = [int(values[key]) for key in ('s1_max', 's2_max', 's3_max')]
        if name == '_candidate_size_triples':
            trace['ordered_size_grid'] = [list(sizes) for sizes in value]
        if name == '_refine_msic_global_candidates':
            trace['candidates'] = [model_record(candidate, score) for candidate, score in values['candidates']]
            coefficients.extend(candidate.B.copy() for candidate, _ in values['candidates'])
            trace['adaptive_terminal'] = trace['candidates'][0]
            trace['one_se_tolerance'] = float(values['tolerance'])
            trace['minimum_msic'] = float(values['min_score'])
            trace['minimum_msic_q'] = int(values['q_best'])
            trace['eligible_candidate_indices'] = [i for i, (_, score) in enumerate(values['candidates'])
                                                   if score <= values['min_score'] + values['tolerance']]
            trace['final'] = model_record(value)
        if name == 'tensor_bcd':
            trace.setdefault('fixed_searches', []).append({
                'sizes': [int(values[key]) for key in ('s1', 's2', 's3')],
                'initial_support': [values[key].astype(int).tolist() for key in ('S1_init', 'S2_init', 'S3_init')],
                'loss_history': list(map(float, value.loss_history)),
                'visited_loss_history': list(map(float, getattr(value, 'visited_loss_history', []))),
                'best_iteration': int(getattr(value, 'best_iteration', -1)),
                'stop_reason': str(getattr(value, 'stop_reason', 'legacy')),
                'B_sha256': array_sha(value.B)})

    sys.setprofile(profile)
    try:
        result = fit()
    finally:
        sys.setprofile(None)
    require('candidates' in trace and 'ordered_size_grid' in trace, 'Missing actual global-selection trace')
    require(len(trace['candidates']) == len(trace['ordered_size_grid']) + 1, 'Incomplete candidate trace')
    return result, trace, coefficients


def aggregate_oof(rows):
    groups, seen = {}, set()
    for row in rows:
        key = (row['version'], int(row['repeat']), row['subject_id'])
        require(key not in seen, 'Duplicate OOF subject/repeat/version')
        seen.add(key)
        groups.setdefault(key[:2], []).append(row)
    output = []
    for (version, repeat), group in sorted(groups.items()):
        y = np.asarray([float(r['y_true']) for r in group])
        pred = np.asarray([float(r['y_pred']) for r in group])
        rss = float(np.sum((y - pred)**2))
        tss = float(np.sum((y - y.mean())**2))
        require(tss > 0, 'OOF response has no variation')
        output.append({'version': version, 'repeat': repeat, 'n_subjects': len(y),
                       'rmse': float(np.sqrt(rss / len(y))), 'r2': 1 - rss / tss,
                       'mse': rss / len(y), 'mae': float(np.mean(np.abs(y - pred)))})
    return output


def saved_runs():
    from scripts.taskcard6.lemon_analysis import load_run, validate_pair, verify_core
    runs = {}
    for version, directory in (('v5-original', 'outputs_eeg_incremental'),
                               ('v5.2-search-final', 'outputs_eeg_incremental_v52_search_final')):
        run = load_run(root() / 'empirical_analysis/lemon' / directory)
        verify_core(run, root() / 'doc/algorithm_versions' / version / 'VERSION.json')
        runs[version] = run
    validate_pair(*runs.values())
    return runs


def audit():
    from empirical_analysis.lemon.incremental_value.data_loader import load_audited_dataset
    from empirical_analysis.lemon.incremental_value.nested_cv import make_splits
    from empirical_analysis.lemon.incremental_value.preprocessing import fit_preprocessor
    from empirical_analysis.lemon.incremental_value.run import _versions
    from numerical_simulation.version_attribution.run import source_manifests, compare_grids
    runs = saved_runs()
    old = runs['v5-original']
    require(_versions() == old['config']['environment'], 'Historical LEMON environment identity differs')
    manifests = source_manifests()
    data = load_audited_dataset(root() / 'empirical_analysis/lemon/outputs_lemon_adjusted',
                               root() / 'empirical_analysis/TMT.csv',
                               root() / 'empirical_analysis/lemon/outputs_demographics/participants.tsv')
    for key, value in data.fingerprints.items():
        require(value['sha256'] == old['fingerprints'][key]['sha256'], 'Current input identity differs: ' + key)
    dependencies = {key: value for key, value in old['fingerprints'].items()
                    if isinstance(value, str) and not key.startswith('src/')}
    for key, value in dependencies.items():
        require(sha(root() / key) == value, 'Historical dependency fingerprint differs: ' + key)
    splits = make_splits(data.C, old['config']['outer_folds'], old['config']['outer_seeds'])
    prepared, identities = {}, []
    for split in splits:
        repeat, fold = split['repeat'], split['outer_fold']
        train = split['train']
        p = fit_preprocessor(data.X[train], data.y[train], data.C[train])
        expected = {'repeat': repeat, 'outer_fold': fold, 'seed': split['seed'],
                    'inner_seed': old['config']['inner_seed'] + repeat * 100 + fold,
                    'train_subject_ids': data.subject_ids[train].tolist(),
                    'test_subject_ids': data.subject_ids[split['test']].tolist()}
        for version, run in runs.items():
            folder = run['directory'] / 'cv/folds' / f'repeat_{repeat}_fold_{fold}'
            require(read(folder / 'split.json') == expected, 'Historical split identity differs: ' + version)
            with np.load(folder / 'preprocessing.npz', allow_pickle=False) as saved:
                validate_preprocessing(dict(saved), p.to_arrays())
        prepared[(repeat, fold)] = p
        identities.append({**expected, 'preprocessing_array_sha256': {key: array_sha(value) for key, value in p.to_arrays().items()}})
    body = {'schema_version': 1, 'fit_version': 'v5.1-bcd-fix', 'config': old['config'],
            'environment': _versions(), 'source_manifests': manifests, 'dependencies': dependencies,
            'input_fingerprints': data.fingerprints, 'worker_sha256': sha(__file__),
            'splits': identities, 'ordered_grid_audit': compare_grids((4, 4, 5), 116),
            'historical_results': {version: {'experiment_id': run['state']['experiment_id'],
                                            'oof_sha256': sha(run['directory'] / 'cv/oof_predictions.csv'),
                                            'fold_metrics_sha256': sha(run['directory'] / 'cv/fold_metrics.csv')}
                                   for version, run in runs.items()}}
    return body, data, splits, prepared, runs


def configure_core(version):
    require(not any(name == 'tensor_abess' or name.startswith('tensor_abess.') for name in sys.modules),
            'A core is already imported; use a fresh isolated process')
    source = root() / 'doc/algorithm_versions' / version / 'src'
    sys.path.insert(0, str(source))
    import tensor_abess
    require(Path(tensor_abess.__file__).resolve() == source / 'tensor_abess/__init__.py', 'Wrong actual core import')
    return source


def metrics(y, prediction):
    rss = float(np.sum((y - prediction)**2))
    return {'rmse': float(np.sqrt(rss / len(y))),
            'r2': float(1 - rss / np.sum((y - y.mean())**2)), 'rss': rss}


def worker(output, repeat, fold):
    source = configure_core('v5.1-bcd-fix')
    plan, data, splits, prepared, runs = audit()
    require(plan == read(output / 'plan.json'), 'Worker identity differs from frozen attribution plan')
    split = next(s for s in splits if (s['repeat'], s['outer_fold']) == (repeat, fold))
    p = prepared[(repeat, fold)]
    train, test = split['train'], split['test']
    train_x = p.apply(data.X[train], data.C[train])
    train_y = p.transform_response(data.y[train], data.C[train])
    folder = output / 'folds' / f'repeat_{repeat}_fold_{fold}'
    require(not folder.exists(), 'Fold output exists; refusing overwrite')
    from empirical_analysis.lemon.incremental_value.model_adapters import fit_eeg_model
    started = perf_counter()
    fitted, trace, candidates = observe_fit(lambda: fit_eeg_model('Tensor-ABESS', train_x, train_y, None, plan['config']['model']))
    elapsed = perf_counter() - started
    for module_name, module in tuple(sys.modules.items()):
        if module_name == 'tensor_abess' or module_name.startswith('tensor_abess.'):
            require(Path(module.__file__).resolve().is_relative_to(source), 'Imported core escaped frozen snapshot')
    train_prediction = p.predict(fitted.B, data.X[train], data.C[train])
    test_prediction = p.predict(fitted.B, data.X[test], data.C[test])
    new_run = runs['v5.2-search-final']
    new_folder = new_run['directory'] / 'cv/folds' / f'repeat_{repeat}_fold_{fold}' / 'Tensor-ABESS'
    new_B = np.load(new_folder / 'B.npy', allow_pickle=False)
    new_rows = read(new_folder / 'complete.json')['predictions']
    require([row['subject_id'] for row in new_rows] == data.subject_ids[test].tolist(), 'Historical prediction order differs')
    new_prediction = np.asarray([row['y_pred'] for row in new_rows])
    support = [indices.astype(int).tolist() for indices in fitted.supports]
    record = {'version': 'v5.1-bcd-fix', 'repeat': repeat, 'outer_fold': fold,
              'n_train': len(train), 'n_test': len(test), 'q': int(np.prod(list(map(len, support)))),
              'Q': int(np.floor(.7 * len(train))), 'sizes': list(map(len, support)), 'support': support,
              'msic': fitted.parameters['sic'], 'residual_rss': trace['final']['rss'],
              'train_rss': metrics(data.y[train], train_prediction)['rss'],
              'train_rmse': metrics(data.y[train], train_prediction)['rmse'],
              'test_rmse': metrics(data.y[test], test_prediction)['rmse'],
              'test_r2': metrics(data.y[test], test_prediction)['r2'], 'runtime_sec': elapsed,
              'q_before_refinement': trace['adaptive_terminal']['q'], 'q_after_refinement': trace['final']['q'],
              'candidate_grid_count': len(trace['ordered_size_grid']), 'candidate_search_count': len(trace['candidates']),
              'v52_B_exact_equal': bool(np.array_equal(fitted.B, new_B)),
              'v52_support_exact_equal': support == read(new_folder / 'support.json'),
              'v52_prediction_exact_equal': bool(np.array_equal(test_prediction, new_prediction)),
              'v52_B_max_abs_difference': float(np.max(np.abs(fitted.B - new_B))),
              'v52_prediction_max_abs_difference': float(np.max(np.abs(test_prediction - new_prediction))),
              'B_sha256': array_sha(fitted.B), 'actual_core_path': str(source)}
    record['q_over_Q'] = record['q'] / record['Q']
    rows = [{'version': 'v5.1-bcd-fix', 'repeat': repeat, 'outer_fold': fold,
             'subject_id': str(data.subject_ids[i]), 'y_true': float(data.y[i]), 'y_pred': float(pred)}
            for i, pred in zip(test, test_prediction)]
    folder.mkdir(parents=True)
    np.save(folder / 'B.npy', fitted.B, allow_pickle=False)
    np.savez_compressed(folder / 'candidate_coefficients.npz', B=np.stack(candidates))
    write(folder / 'trace.json', trace)
    write(folder / 'support.json', support)
    write(folder / 'complete.json', {'plan_sha256': sha(output / 'plan.json'), 'metrics': record, 'predictions': rows})
    print('FOLD_COMPLETE', repeat, fold, 'q', record['q'], 'v52_exact', record['v52_B_exact_equal'], flush=True)


def summarize(output, runs, data, splits, prepared):
    records, predictions, traces = [], [], []
    for split in splits:
        repeat, fold = split['repeat'], split['outer_fold']
        folder = output / 'folds' / f'repeat_{repeat}_fold_{fold}'
        saved = read(folder / 'complete.json')
        require(saved['plan_sha256'] == sha(output / 'plan.json'), 'Fold completion identity differs')
        records.append(saved['metrics'])
        predictions.extend(saved['predictions'])
        traces.append(read(folder / 'trace.json'))
        p = prepared[(repeat, fold)]
        train, test = split['train'], split['test']
        for version, run in runs.items():
            original = next(row for row in run['folds'] if row['model'] == 'Tensor-ABESS'
                            and int(row['repeat']) == repeat and int(row['outer_fold']) == fold)
            fit_dir = run['directory'] / 'cv/folds' / f'repeat_{repeat}_fold_{fold}' / 'Tensor-ABESS'
            B = np.load(fit_dir / 'B.npy', allow_pickle=False)
            support = read(fit_dir / 'support.json')
            parameters = json.loads(original['selected_parameters'])
            train_metrics = metrics(data.y[train], p.predict(B, data.X[train], data.C[train]))
            test_metrics = metrics(data.y[test], p.predict(B, data.X[test], data.C[test]))
            require(abs(train_metrics['rmse'] - float(original['train_rmse'])) <= 1e-14,
                    'Historical saved coefficient train-metric reconstruction differs')
            require(abs(test_metrics['rmse'] - float(original['test_rmse'])) <= 1e-14,
                    'Historical saved coefficient test-metric reconstruction differs')
            records.append({'version': version, 'repeat': repeat, 'outer_fold': fold,
                            'n_train': 116, 'n_test': 29, 'q': int(original['q']), 'Q': 81,
                            'q_over_Q': int(original['q']) / 81, 'sizes': list(map(len, support)),
                            'support': support, 'msic': parameters['sic'], 'train_rss': train_metrics['rss'],
                            'residual_rss': train_metrics['rss'], 'train_rmse': train_metrics['rmse'],
                            'test_rmse': test_metrics['rmse'], 'test_r2': test_metrics['r2'],
                            'runtime_sec': float(original['runtime_sec']), 'B_sha256': array_sha(B),
                            'q_before_refinement': None, 'q_after_refinement': int(original['q']),
                            'candidate_trace_available': False})
    for version, run in runs.items():
        predictions.extend({'version': version, **{key: row[key] for key in
                            ('repeat', 'outer_fold', 'subject_id', 'y_true', 'y_pred')}}
                           for row in run['oof'] if row['model'] == 'Tensor-ABESS')
    require(len(predictions) == 1305, 'Incomplete three-version OOF coverage')
    by_repeat = aggregate_oof(predictions)
    require(all(row['n_subjects'] == 145 for row in by_repeat), 'Incomplete repeated OOF')
    summary = []
    for version in ('v5-original', 'v5.1-bcd-fix', 'v5.2-search-final'):
        group = [row for row in by_repeat if row['version'] == version]
        folds = [row for row in records if row['version'] == version]
        summary.append({'version': version, 'rmse_mean': float(np.mean([row['rmse'] for row in group])),
                        'r2_mean': float(np.mean([row['r2'] for row in group])),
                        'q80_folds': sum(row['q'] == 80 for row in folds),
                        'train_rss_mean': float(np.mean([row['train_rss'] for row in folds]))})
    write(output / 'comparison.json', {'complete': True, 'v51_completed_fits': 15,
          'summary': summary, 'metrics_by_repeat': by_repeat, 'folds': records,
          'v51_v52_all_B_exact_equal': all(row.get('v52_B_exact_equal', True) for row in records),
          'v51_v52_all_prediction_exact_equal': all(row.get('v52_prediction_exact_equal', True) for row in records),
          'candidate_comparison_limit': 'Historical v5/v5.2 full candidate traces were not saved. Actual final B/support/prediction equality is checked; candidate-library identity additionally requires same-code/ordered-grid proof or independent trace replay.',
          'historical_fit_reuse': 'v5 and v5.2 reuse complete frozen OOF, supports and B; no baseline is retrained.'})
    write_csv(output / 'oof_predictions.csv', predictions)
    write_csv(output / 'metrics_by_repeat.csv', by_repeat)
    keys = list(dict.fromkeys(key for record in records for key in record))
    write_csv(output / 'three_version_folds.csv', [
        {key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in keys}
        for row in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', nargs=2, type=int, metavar=('REPEAT', 'FOLD'))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output, *args.worker)
        return
    plan, data, splits, prepared, runs = audit()
    if args.audit_only:
        print('STRICT_LEMON_GATE_PASS', data.X.shape, plan['ordered_grid_audit'], flush=True)
        return
    require(not output.exists(), 'Attribution output exists; refusing overwrite')
    output.mkdir(parents=True)
    write(output / 'plan.json', plan)
    (output / 'logs').mkdir()
    for split in splits:
        repeat, fold = split['repeat'], split['outer_fold']
        log = output / 'logs' / f'repeat_{repeat}_fold_{fold}.txt'
        command = [sys.executable, '-B', '-m', 'scripts.taskcard7.lemon', '--output-dir', str(output),
                   '--worker', str(repeat), str(fold)]
        with log.open('x', encoding='utf-8') as handle:
            completed = subprocess.run(command, cwd=root(), stdout=handle, stderr=subprocess.STDOUT, check=False)
        require(completed.returncode == 0, 'Isolated fold failed; see ' + str(log))
        print('COMPLETE', repeat, fold, flush=True)
    summarize(output, runs, data, splits, prepared)
    print('COMPLETE_15_FITS', output, flush=True)


if __name__ == '__main__':
    main()
