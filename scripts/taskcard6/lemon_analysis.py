"""Compare complete frozen LEMON runs without fitting or changing their files.

Inputs: old/new incremental_value output directories and their core manifests.
Outputs: an explicit independent analysis directory, created only after validation.
Run from repository root with PYTHONPATH=src plus repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from empirical_analysis.lemon.incremental_value.metrics import summarize_oof


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f'Missing required result: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    require(path.is_file(), f'Missing required result: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oof_key(row):
    return str(row['subject_id']), int(row['repeat']), row['model']


def split_key(row):
    return str(row['subject_id']), int(row['repeat'])


def validate_oof(rows, splits, models):
    assignments = {split_key(row): int(row['outer_fold']) for row in splits}
    expected = {(sid, repeat, model) for sid, repeat in assignments for model in models}
    keys = [oof_key(row) for row in rows]
    require(len(keys) == 2175 and len(set(keys)) == len(keys) and set(keys) == expected,
            'Incomplete or duplicated OOF coverage')
    truth = {}
    for row in rows:
        require(int(row['outer_fold']) == assignments[split_key(row)], 'OOF fold drift')
        y, prediction = float(row['y_true']), float(row['y_pred'])
        require(bool(np.isfinite([y, prediction]).all()), 'Nonfinite OOF prediction')
        sid = str(row['subject_id'])
        require(sid not in truth or truth[sid] == y, 'OOF response changes across models/repeats')
        truth[sid] = y


def load_run(directory):
    directory = Path(directory).resolve()
    state = read_json(directory / 'run_state.json')
    require(state.get('status') == 'complete' and state.get('complete') is True
            and state.get('smoke') is False and state.get('completed_fits') == 75
            and state.get('expected_fits') == 75 and state.get('n_oof_predictions') == 2175,
            'Run is not a complete 75-fit/2175-OOF experiment')
    config = read_json(directory / 'config_used.json')
    models = config['models']
    require(models == ['Covariates Only', 'Tensor-ABESS', 'Vector Lasso', 'Tensor-GLasso', 'Ridge'],
            'Model set/order differs from frozen five-model design')
    require(config['outer_folds'] == 5 and config['outer_repeats'] == 3,
            'Outer CV differs from frozen design')
    splits = read_csv(directory / 'cv/outer_splits.csv')
    require(len(splits) == 435 and len({split_key(row) for row in splits}) == 435,
            'Invalid split coverage')
    ids = {row['subject_id'] for row in splits}
    require(len(ids) == 145, 'Subject count differs from 145')
    for repeat, seed in enumerate(config['outer_seeds']):
        for fold in range(1, 6):
            group = [r for r in splits if int(r['repeat']) == repeat and int(r['outer_fold']) == fold]
            require(len(group) == 29 and all(int(r['outer_seed']) == seed for r in group),
                    'Fold size/seed differs from frozen split')
    oof = read_csv(directory / 'cv/oof_predictions.csv')
    validate_oof(oof, splits, models)
    folds = read_csv(directory / 'cv/fold_metrics.csv')
    keys = [(r['model'], int(r['repeat']), int(r['outer_fold'])) for r in folds]
    expected = set(itertools.product(models, range(3), range(1, 6)))
    require(len(keys) == 75 and len(set(keys)) == 75 and set(keys) == expected,
            'Missing or duplicated model-fold result')
    supports = {}
    for row in folds:
        name, repeat, fold = row['model'], int(row['repeat']), int(row['outer_fold'])
        require(int(row['n_train']) == 116 and int(row['n_test']) == 29, 'Fold sample size drift')
        require(bool(np.isfinite([float(row[k]) for k in ('train_rmse', 'test_rmse', 'runtime_sec')]).all()),
                'Nonfinite fold metrics')
        fold_dir = directory / 'cv/folds' / f'repeat_{repeat}_fold_{fold}'
        split = read_json(fold_dir / 'split.json')
        expected_test = {r['subject_id'] for r in splits
                         if int(r['repeat']) == repeat and int(r['outer_fold']) == fold}
        require(set(split['test_subject_ids']) == expected_test
                and set(split['train_subject_ids']) == ids - expected_test,
                'Fold train/test IDs disagree with split table')
        fit_dir = fold_dir / name.replace(' ', '_')
        complete = read_json(fit_dir / 'complete.json')
        require(complete['experiment_id'] == state['experiment_id'], 'Completion identity mismatch')
        saved_rows = complete['predictions']
        expected_rows = [r for r in oof if r['model'] == name
                         and int(r['repeat']) == repeat and int(r['outer_fold']) == fold]
        require({oof_key(r): (float(r['y_true']), float(r['y_pred'])) for r in saved_rows}
                == {oof_key(r): (float(r['y_true']), float(r['y_pred'])) for r in expected_rows},
                'Completion predictions disagree with OOF table')
        support = read_json(fit_dir / 'support.json')
        require(len(support) == 3, 'Invalid mode support count')
        for indices, dimension in zip(support, (61, 16, 5)):
            require(indices == sorted(set(indices)) and all(isinstance(i, int) and 0 <= i < dimension for i in indices),
                    'Invalid support indices')
        q = int(np.prod([len(s) for s in support]))
        require(q == int(row['q']), 'Saved support sizes disagree with q')
        B = np.load(fit_dir / 'B.npy', allow_pickle=False)
        require(B.shape == (61, 16, 5) and bool(np.isfinite(B).all()), 'Invalid saved coefficient tensor')
        supports[(name, repeat, fold)] = support
    return {'directory': directory, 'state': state, 'config': config, 'splits': splits,
            'oof': oof, 'folds': folds, 'supports': supports,
            'fingerprints': read_json(directory / 'diagnostics/input_fingerprint.json')}


def validate_pair(old, new):
    require(old['splits'] == new['splits'], 'Old/new outer splits differ')
    require({k: v for k, v in old['config'].items() if k != 'environment'}
            == {k: v for k, v in new['config'].items() if k != 'environment'},
            'Old/new configuration differs')
    require(old['config']['environment'] == new['config']['environment'], 'Old/new environment differs')
    for key, value in old['fingerprints'].items():
        if isinstance(value, dict) and 'sha256' in value:
            require(value['sha256'] == new['fingerprints'].get(key, {}).get('sha256'),
                    f'Old/new input fingerprint differs: {key}')
    previous = {oof_key(row): row for row in old['oof']}
    for row in new['oof']:
        saved = previous[oof_key(row)]
        require(float(saved['y_true']) == float(row['y_true']), 'Old/new response differs')
    for key in old['supports']:
        name, repeat, fold = key
        folder = f'repeat_{repeat}_fold_{fold}'
        require(read_json(old['directory'] / 'cv/folds' / folder / 'split.json')
                == read_json(new['directory'] / 'cv/folds' / folder / 'split.json'),
                'Old/new fold ID order or seed differs')


def jaccard(first, second):
    a, b = set(first), set(second)
    return len(a & b) / len(a | b) if a | b else 1.0


def support_summary(run):
    selected = [run['supports'][('Tensor-ABESS', r, f)] for r in range(3) for f in range(1, 6)]
    stability = [float(np.mean([jaccard(a[mode], b[mode])
                               for a, b in itertools.combinations(selected, 2)])) for mode in range(3)]
    return {'pairwise_mode_jaccard_mean': stability,
            'mode_selection_counts': [[sum(index in support[mode] for support in selected)
                                      for index in range(dimension)] for mode, dimension in enumerate((61, 16, 5))],
            'frequency_denominator': 15,
            'gamma_index': 4, 'gamma_selected_folds': sum(4 in s[2] for s in selected),
            'gamma_only_folds': sum(s[2] == [4] for s in selected)}


def compare_runs(old, new):
    validate_pair(old, new)
    results = {}
    for label, run in (('old', old), ('new', new)):
        summary = summarize_oof(run['oof'], run['config']['bootstrap_seed'], run['config']['bootstrap_samples'])
        results[label] = {'oof_tables': summary, 'support': support_summary(run)}
    paired_folds = []
    for repeat, fold in itertools.product(range(3), range(1, 6)):
        pair = {'repeat': repeat, 'outer_fold': fold}
        for label, run in (('old', old), ('new', new)):
            row = next(r for r in run['folds'] if r['model'] == 'Tensor-ABESS'
                       and int(r['repeat']) == repeat and int(r['outer_fold']) == fold)
            Q = int(np.floor(.7 * int(row['n_train'])))
            pair.update({label + '_' + key: float(row[key])
                         for key in ('q', 's1', 's2', 's3', 'train_rmse', 'test_rmse', 'runtime_sec')})
            pair[label + '_Q'] = Q
            pair[label + '_q_over_Q'] = float(row['q']) / Q
            pair[label + '_test_minus_train_rmse'] = float(row['test_rmse']) - float(row['train_rmse'])
        pair['support_jaccard_modes'] = [
            jaccard(a, b) for a, b in zip(old['supports'][('Tensor-ABESS', repeat, fold)],
                                         new['supports'][('Tensor-ABESS', repeat, fold)])]
        paired_folds.append(pair)
    results['paired_folds'] = paired_folds
    subjects = sorted({r['subject_id'] for r in old['splits']})
    errors = {}
    for label, run in (('old', old), ('new', new)):
        lookup = {oof_key(row): row for row in run['oof']}
        errors[label] = np.array([[(float(lookup[(sid, repeat, 'Tensor-ABESS')]['y_true'])
                                  - float(lookup[(sid, repeat, 'Tensor-ABESS')]['y_pred'])) ** 2
                                 for sid in subjects] for repeat in range(3)])
    repeat_delta = np.sqrt(errors['new'].mean(axis=1)) - np.sqrt(errors['old'].mean(axis=1))
    rng = np.random.default_rng(new['config']['bootstrap_seed'])
    indices = rng.integers(0, 145, size=(new['config']['bootstrap_samples'], 145))
    draws = (np.sqrt(errors['new'][:, indices].mean(axis=2))
             - np.sqrt(errors['old'][:, indices].mean(axis=2))).mean(axis=0)
    results['paired_abess'] = {'rmse_new_minus_old_by_repeat': repeat_delta.tolist(),
                              'mean_repeat_rmse_new_minus_old': float(repeat_delta.mean()),
                              'subject_cluster_ci_95': np.quantile(draws, [.025, .975]).tolist(),
                              'interval_scope': 'Conditional on saved OOF; 145 subject clusters retain all repeats.'}
    return results


def verify_core(run, manifest_path):
    expected = {k if k.startswith('src/') else 'src/' + k: v
                for k, v in read_json(manifest_path)['source_sha256'].items()}
    require(len(expected) == 32 and all(run['fingerprints'].get(k) == v for k, v in expected.items()),
            f'Saved core does not match complete manifest: {manifest_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--old-manifest', type=Path, required=True)
    parser.add_argument('--new-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    old, new = load_run(args.old), load_run(args.new)
    verify_core(old, args.old_manifest)
    verify_core(new, args.new_manifest)
    result = compare_runs(old, new)
    target = args.output_dir.resolve()
    require(all(target != run['directory'] and target not in run['directory'].parents
                and run['directory'] not in target.parents for run in (old, new)),
            'Analysis output must be separate from experiment outputs')
    require(not target.exists(), 'Analysis output already exists; refusing overwrite')
    result['provenance'] = {label: {'experiment_id': run['state']['experiment_id'],
                                  'oof_sha256': digest(run['directory'] / 'cv/oof_predictions.csv'),
                                  'fold_metrics_sha256': digest(run['directory'] / 'cv/fold_metrics.csv')}
                            for label, run in (('old', old), ('new', new))}
    target.mkdir(parents=True)
    (target / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (target / 'paired_abess_folds.csv').open('w', encoding='utf-8', newline='') as handle:
        records = [{**r, 'support_jaccard_modes': json.dumps(r['support_jaccard_modes'])}
                   for r in result['paired_folds']]
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    old_rows = {oof_key(r): r for r in old['oof']}
    paired = []
    for row in new['oof']:
        previous = old_rows[oof_key(row)]
        paired.append({**{k: row[k] for k in ('subject_id', 'repeat', 'outer_fold', 'model', 'y_true')},
                       'old_prediction': previous['y_pred'], 'new_prediction': row['y_pred'],
                       'squared_error_new_minus_old': (float(row['y_true']) - float(row['y_pred']))**2
                       - (float(previous['y_true']) - float(previous['y_pred']))**2})
    with (target / 'paired_oof.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print('Complete 75-fit/2175-OOF paired analysis saved:', target)


if __name__ == '__main__':
    main()
