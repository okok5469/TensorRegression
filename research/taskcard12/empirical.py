"""Strict paired empirical evaluation; all fitted state is training-only."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.taskcard7.lemon import array_sha, require
from research.taskcard10.experiment import file_sha, read_json, write_json, write_npz
from research.taskcard12 import selection


def root():
    return Path(__file__).resolve().parents[2]


def numeric(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in 'fiu' and np.isfinite(array).all(),
            name + ' must be finite numeric data without coercion')
    return array


def validate_dataset(data):
    """Validate natural third-order samples without reshaping or imputation."""
    X, y, C = (numeric(data[key], key) for key in ('X', 'y', 'C'))
    n = len(X)
    require(X.ndim == 4 and min(X.shape) > 0, 'Natural three-mode X required')
    require(y.ndim == 1 and y.shape == (n,) and n > 3 and np.unique(y).size > 2,
            'Aligned nonconstant continuous y required')
    require(C.ndim == 2 and C.shape[0] == n, 'Aligned explicit covariate design required')
    ids = np.asarray(data['subject_ids'])
    require(ids.shape == (n,) and ids.dtype.kind in 'US' and
            len(set(ids.tolist())) == n and all(str(s).strip() for s in ids),
            'Unique nonempty string subject IDs required')
    if 'groups' in data:
        groups = np.asarray(data['groups'])
        require(groups.shape == (n,) and groups.dtype.kind in 'USiu' and
                all(str(s).strip() for s in groups), 'Aligned nonempty group IDs required')


def validate_splits(data, splits, *, group_isolation=False):
    validate_dataset(data)
    require(bool(splits), 'Prespecified splits required')
    n, seen, coverage = len(data['y']), set(), {}
    for split in splits:
        repeat, fold = split['repeat'], split['outer_fold']
        require(type(repeat) is int and repeat >= 0 and type(fold) is int and fold >= 1,
                'Integer repeat/fold identity required')
        key = (repeat, fold)
        require(key not in seen, 'Duplicate fold')
        seen.add(key)
        train, test = (np.asarray(split[key]) for key in ('train', 'test'))
        require(train.ndim == test.ndim == 1 and train.dtype.kind in 'iu' and
                test.dtype.kind in 'iu' and len(train) > 2 and len(test) > 0,
                'Nonempty integer train/test indices required')
        require(len(set(train.tolist())) == len(train) and len(set(test.tolist())) == len(test),
                'Duplicate subject within split')
        require(set(train).isdisjoint(test) and set(train) | set(test) == set(range(n)),
                'Train/test must be disjoint and partition the audited dataset')
        coverage.setdefault(repeat, []).extend(test.tolist())
        if group_isolation:
            require('groups' in data, 'Declared group isolation requires group IDs')
            groups = np.asarray(data['groups'])
            require(set(groups[train]).isdisjoint(groups[test]), 'Group leaked across train/test')
    require(all(sorted(indices) == list(range(n)) for indices in coverage.values()),
            'Every subject must occur exactly once in OOF per repeat')


def fit_fold(X_train, y_train, C_train, X_test, C_test, native_options,
             fit_preprocessor, *, pair_solver=None, on_selected=None):
    """The signature deliberately has no held-out response argument."""
    preprocessor = fit_preprocessor(X_train, y_train, C_train)
    X = preprocessor.apply(X_train, C_train)
    y = preprocessor.transform_response(y_train, C_train)
    solver = selection.fit_pair if pair_solver is None else pair_solver
    pair = solver(X, y, **native_options)
    require(set(pair) == {'pool', 'original', 'fs'}, 'Incomplete paired solver result')
    if on_selected is not None:
        on_selected(pair, preprocessor)
    predictions, train_predictions = {}, {}
    for rule in ('original', 'fs'):
        model = pair[rule]
        predictions[rule] = preprocessor.predict(model['B'], X_test, C_test)
        train_predictions[rule] = preprocessor.predict(model['B'], X_train, C_train)
        require(np.asarray(predictions[rule]).shape == (len(X_test),) and
                np.isfinite(predictions[rule]).all(), 'Invalid held-out predictions')
    return {'pair': pair, 'preprocessor': preprocessor, 'predictions': predictions,
            'train_predictions': train_predictions}


def metric(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    require(y.ndim == prediction.ndim == 1 and y.shape == prediction.shape and
            len(y) > 0 and np.isfinite(y).all() and np.isfinite(prediction).all(),
            'Finite aligned prediction vectors required')
    rss, denominator = float(np.sum((y - prediction)**2)), float(np.sum((y - y.mean())**2))
    return {'rmse': float(np.sqrt(rss / len(y))), 'r2': None if denominator == 0 else 1 - rss / denominator,
            'rss': rss, 'n': len(y)}


def validate_predictions(rows, data, splits, models):
    validate_splits(data, splits)
    require(len(models) == len(set(models)), 'Duplicate expected model')
    expected = {(s['repeat'], s['outer_fold'], str(data['subject_ids'][i]), model)
                for s in splits for i in s['test'] for model in models}
    by_subject = {str(key): float(y) for key, y in zip(data['subject_ids'], data['y'])}
    seen, group, fold_group = set(), {}, {}
    for row in rows:
        key = (int(row['repeat']), int(row['outer_fold']), str(row['subject_id']), row['model'])
        require(key in expected and key not in seen, 'Unexpected or duplicate OOF identity')
        require(float(row['y_true']) == by_subject[key[2]], 'OOF response differs from audited input')
        require(np.isfinite(float(row['y_pred'])), 'Nonfinite OOF prediction')
        seen.add(key)
        group.setdefault((key[3], key[0]), []).append(row)
        fold_group.setdefault((key[3], key[0], key[1]), []).append(row)
    require(seen == expected, 'Incomplete OOF predictions')
    return group, fold_group


def summarize_predictions(rows, data, splits, models):
    require({'original', 'fs'} <= set(models), 'Both selection rules required')
    group, fold_group = validate_predictions(rows, data, splits, models)
    repeat_metrics = [{'model': model, 'repeat': repeat,
                       **metric([r['y_true'] for r in values], [r['y_pred'] for r in values])}
                      for (model, repeat), values in sorted(group.items())]
    fold_metrics = [{'model': model, 'repeat': repeat, 'outer_fold': fold,
                     **metric([r['y_true'] for r in values], [r['y_pred'] for r in values])}
                    for (model, repeat, fold), values in sorted(fold_group.items())]
    summary = {}
    for model in models:
        selected = [row for row in repeat_metrics if row['model'] == model]
        require(all(row['r2'] is not None for row in selected), 'OOF response has zero variance')
        summary[model] = {'mean_oof_rmse': float(np.mean([r['rmse'] for r in selected])),
                          'mean_oof_r2': float(np.mean([r['r2'] for r in selected])),
                          'repeat_count': len(selected)}
    lookup = {(r['model'], r['repeat'], r['outer_fold']): r for r in fold_metrics}
    changes = []
    for split in splits:
        repeat, fold = split['repeat'], split['outer_fold']
        a, b = (lookup[rule, repeat, fold] for rule in ('original', 'fs'))
        changes.append({'repeat': repeat, 'outer_fold': fold, 'original_rmse': a['rmse'],
                        'fs_rmse': b['rmse'], 'rmse_delta': b['rmse'] - a['rmse']})
    paired = {'rmse_delta': summary['fs']['mean_oof_rmse'] - summary['original']['mean_oof_rmse'],
              'r2_delta': summary['fs']['mean_oof_r2'] - summary['original']['mean_oof_r2'],
              'improved_folds': sum(r['rmse_delta'] < 0 for r in changes),
              'unchanged_folds': sum(r['rmse_delta'] == 0 for r in changes),
              'worsened_folds': sum(r['rmse_delta'] > 0 for r in changes),
              'adverse_folds': [r for r in changes if r['rmse_delta'] > 0],
              'fold_changes': changes,
              'fs_beats_covariates_only': (summary['fs']['mean_oof_rmse'] <
                                          summary['Covariates Only']['mean_oof_rmse'])
                                         if 'Covariates Only' in summary else None}
    return {'models': summary, 'paired': paired, 'repeat_metrics': repeat_metrics,
            'fold_metrics': fold_metrics, 'prediction_count': len(rows),
            'interpretation': 'Descriptive paired ablation; repeats reuse subjects and are not independent trials.'}


def verify_authorization(protocol, current_plan, gate_path, repository):
    repository, gate_path = Path(repository).resolve(), Path(gate_path)
    gate = read_json(gate_path)
    require(gate.get('passed') is True and bool(gate.get('source_files')), 'P0 correctness gate not passed')
    require(protocol.get('status') == 'FROZEN' and protocol.get('plan') == current_plan,
            'Empirical protocol is not frozen or current plan differs')
    require(protocol.get('p0_gate_sha256') == file_sha(gate_path), 'P0 gate identity differs')
    require(bool(protocol.get('source_files')), 'Empirical source manifest missing')
    for manifest in (gate['source_files'], protocol['source_files']):
        for relative, digest in manifest.items():
            path = (repository / relative).resolve()
            require(path.is_relative_to(repository) and file_sha(path) == digest,
                    'Frozen empirical/P0 source identity differs: ' + relative)


def verify_original(model, expected, predictions, saved_predictions):
    require(np.array_equal(model['B'], expected['B']) and model['support'] == expected['support'] and
            model['q'] == expected['q'] and model['score'] == expected['score'],
            'Original selection differs from historical official v5.2')
    require(np.array_equal(predictions, saved_predictions), 'Original held-out prediction differs from official v5.2')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def rows_for_output(rows):
    return [{key: (int(row[key]) if key in ('repeat', 'outer_fold') else
                   float(row[key]) if key in ('y_true', 'y_pred') else row[key])
             for key in ('subject_id', 'repeat', 'outer_fold', 'model', 'y_true', 'y_pred')}
            for row in rows]


def serial_splits(splits):
    return [{key: np.asarray(value).tolist() if key in ('train', 'test') else int(value)
             for key, value in split.items()} for split in splits]


def array_identities(data):
    return {key: array_sha(np.asarray(value)) for key, value in sorted(data.items())}


def preprocessing_identities(prepared):
    return [{'repeat': repeat, 'outer_fold': fold,
             'arrays': {name: array_sha(value) for name, value in preprocessor.to_arrays().items()}}
            for (repeat, fold), preprocessor in sorted(prepared.items())]


def source_manifest(config_path):
    """Include all repository code in each runtime dependency package."""
    repository = root()
    files = {Path(__file__).resolve(), Path(config_path).resolve(),
             repository / 'research/taskcard12/test_empirical.py',
             repository / 'research/taskcard12/selection.py',
             repository / 'empirical_analysis/__init__.py',
             repository / 'numerical_simulation/__init__.py',
             repository / 'scripts/taskcard7/lemon.py'}
    for package in ('empirical_analysis/lemon', 'scripts/taskcard6',
                    'numerical_simulation/version_attribution'):
        files.update((repository / package).rglob('*.py'))
    for relative in selection.identity()['source_sha256']:
        files.add(repository / relative)
    for version in ('v5-original', 'v5.1-bcd-fix', 'v5.2-search-final'):
        directory = repository / 'doc/algorithm_versions' / version
        files.add(directory / 'VERSION.json')
        files.update(directory / 'src' / relative
                     for relative in read_json(directory / 'VERSION.json')['source_sha256'])
    for relative in ('research/taskcard12/run_empirical.ps1', 'research/taskcard12/run_empirical.sh'):
        path = repository / relative
        require(path.is_file(), 'Empirical launch script missing: ' + relative)
        files.add(path)
    require(all(path.is_relative_to(repository) for path in files), 'Source/config must be inside repository')
    return {path.relative_to(repository).as_posix(): file_sha(path) for path in sorted(files)}


def prepare_lemon(config):
    from scripts.taskcard7.lemon import audit
    from empirical_analysis.lemon.confounds import fit_preprocessor
    selection.configure_core()
    legacy, audited, splits, prepared, runs = audit()
    data = {key: np.asarray(getattr(audited, key)) for key in ('X', 'y', 'C', 'subject_ids')}
    validate_splits(data, serial_splits(splits))
    require(len(data['y']) == config['expected_subjects'] == 145 and len(splits) == 15 and
            list(data['X'].shape[1:]) == config['expected_dimensions'], 'LEMON contract differs')
    official = runs['v5.2-search-final']
    require(config['baseline_models'] == [name for name in official['config']['models']
                                         if name != 'Tensor-ABESS'], 'Historical baseline list differs')
    baseline_rows = rows_for_output([row for row in official['oof'] if row['model'] != 'Tensor-ABESS'])
    evidence = {}
    for run in runs.values():
        for path in sorted(run['directory'].rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.csv', '.npy', '.npz'):
                evidence[path.relative_to(root()).as_posix()] = file_sha(path)
    plan = {'dataset': 'LEMON', 'response': config['response'], 'n': len(data['y']),
            'dimensions': list(data['X'].shape[1:]), 'covariates': audited.audit['covariate_columns'],
            'native_options': selection.normalized_options(official['config']['model']),
            'data_array_sha256': array_identities(data), 'input_fingerprints': legacy['input_fingerprints'],
            'splits': serial_splits(splits), 'preprocessing': preprocessing_identities(prepared),
            'baseline_models': config['baseline_models'], 'baseline_files': evidence,
            'historical_results': legacy['historical_results'],
            'historical_configuration': official['config'], 'group_isolation': False,
            'raw_eeg_regenerated': False, 'same_subject_repeated_cv': True,
            'selection_identity': selection.identity()}
    return {'plan': plan, 'data': data, 'splits': serial_splits(splits), 'prepared': prepared,
            'baseline_rows': baseline_rows, 'fit_preprocessor': fit_preprocessor, 'official': official}


def inside_repository(relative):
    require(isinstance(relative, str) and bool(relative), 'Required input path not supplied')
    path = (root() / relative).resolve()
    require(path.is_relative_to(root()) and path.is_file(), 'Missing repository-local input: ' + relative)
    return path


def prepare_generic(config):
    """Use a dataset's documented adapter and historical baseline identities."""
    require(config['response']['continuous'] is True and config['response']['name'] and
            config['response']['scale'] and len(config['tensor_axes']) == 3 and
            all(config['tensor_axes']), 'Actual response and natural tensor axes must be declared')
    data_path, split_path = (inside_repository(config[key]) for key in ('dataset_npz', 'splits_json'))
    with np.load(data_path, allow_pickle=False) as arrays:
        require(set(arrays.files) in ({'X', 'y', 'C', 'subject_ids'},
                                     {'X', 'y', 'C', 'subject_ids', 'groups'}), 'NPZ keys differ from contract')
        data = {key: arrays[key].copy() for key in arrays.files}
    splits = read_json(split_path)
    validate_splits(data, splits, group_isolation=config['group_isolation'])
    require(len(config['covariate_names']) == data['C'].shape[1], 'Covariate names/design differ')
    adapter_path = inside_repository(config['preprocessor_file'])
    require(file_sha(adapter_path) == config['preprocessor_sha256'], 'Preprocessor source hash differs')
    require(bool(config['preprocessing_dependencies']), 'Complete preprocessor dependency manifest required')
    for path, digest in config['preprocessing_dependencies'].items():
        require(file_sha(inside_repository(path)) == digest, 'Preprocessor dependency changed: ' + path)
    module_parts = adapter_path.relative_to(root()).with_suffix('').parts
    require(all(part.isidentifier() for part in module_parts), 'Adapter path must be an importable Python module')
    module = importlib.import_module('.'.join(module_parts))
    require(Path(module.__file__).resolve() == adapter_path, 'Actual adapter import path differs')
    require(callable(getattr(module, 'fit_preprocessor', None)), 'Adapter must define fit_preprocessor')
    prepared = {}
    for split in splits:
        indices = split['train']
        prepared[split['repeat'], split['outer_fold']] = module.fit_preprocessor(
            data['X'][indices], data['y'][indices], data['C'][indices])
    preprocessing = preprocessing_identities(prepared)
    baseline_path = inside_repository(config['baseline_predictions_csv'])
    baseline_manifest_path = inside_repository(config['baseline_manifest_json'])
    baseline_manifest = read_json(baseline_manifest_path)
    require(baseline_manifest['data_array_sha256'] == array_identities(data) and
            baseline_manifest['splits'] == splits and baseline_manifest['preprocessing'] == preprocessing,
            'Historical baselines were not made from identical inputs/splits/preprocessing')
    require(baseline_manifest['preprocessor_sha256'] == config['preprocessor_sha256'] and
            baseline_manifest['preprocessing_dependencies'] == config['preprocessing_dependencies'],
            'Historical preprocessing source identity differs')
    require(config['baseline_predictions_csv'] in baseline_manifest['files'] and
            bool(baseline_manifest['configuration']) and bool(baseline_manifest['source_files']),
            'Historical baseline configuration/source identity missing')
    for relative, digest in {**baseline_manifest['files'], **baseline_manifest['source_files']}.items():
        require(file_sha(inside_repository(relative)) == digest, 'Historical baseline file changed: ' + relative)
    require('Covariates Only' in config['baseline_models'] and
            len(config['baseline_models']) == len(set(config['baseline_models'])),
            'Prespecified historical baselines must include covariates-only')
    rows = rows_for_output(read_csv(baseline_path))
    require(set(r['model'] for r in rows) == set(config['baseline_models']), 'Baseline model coverage differs')
    inputs = {name: file_sha(path) for name, path in
              ((config['dataset_npz'], data_path), (config['splits_json'], split_path),
               (config['baseline_manifest_json'], baseline_manifest_path))}
    plan = {'dataset': config['dataset'], 'response': config['response'], 'tensor_axes': config['tensor_axes'],
            'n': len(data['y']), 'dimensions': list(data['X'].shape[1:]),
            'covariates': config['covariate_names'], 'native_options': selection.normalized_options(config['native_options']),
            'data_array_sha256': array_identities(data), 'input_fingerprints': inputs, 'splits': splits,
            'preprocessing': preprocessing, 'baseline_models': config['baseline_models'],
            'baseline_files': baseline_manifest['files'], 'baseline_manifest': baseline_manifest,
            'group_isolation': config['group_isolation'], 'selection_identity': selection.identity()}
    return {'plan': plan, 'data': data, 'splits': splits, 'prepared': prepared, 'baseline_rows': rows,
            'fit_preprocessor': module.fit_preprocessor, 'official': None}


def prepare(config_path):
    config = read_json(config_path)
    require(config['status'] == 'READY', 'NOT_RUN: actual data/response/covariates/splits/adapter/baselines unavailable')
    selection.configure_core()
    builders = {'LEMON': prepare_lemon, 'HCP-YA': prepare_generic, 'NKI-RS': prepare_generic}
    require(config['dataset'] in builders, 'Unknown empirical dataset')
    bundle = builders[config['dataset']](config)
    bundle['config'] = config
    bundle['plan']['config_sha256'] = file_sha(config_path)
    bundle['source_files'] = source_manifest(config_path)
    bundle['source_files'].update(config.get('preprocessing_dependencies', {}))
    if config.get('preprocessor_file'):
        bundle['source_files'][config['preprocessor_file']] = config['preprocessor_sha256']
    validate_predictions(bundle['baseline_rows'], bundle['data'], bundle['splits'], config['baseline_models'])
    return bundle


def build_protocol(bundle):
    return {'schema_version': 1, 'status': 'DRAFT', 'plan': bundle['plan'],
            'source_files': bundle['source_files'], 'p0_gate_sha256': None,
            'research_only': True, 'final_recommendation': 'PENDING_CLUSTER_RESULTS',
            'selection_rules': ['original', 'fs'], 'native_searches_per_fold': 1,
            'test_prediction_calls_per_rule_fold': 1,
            'baseline_policy': 'Reuse only exactly audited original inputs, splits and training preprocessing.',
            'error_policy': 'Stop immediately; no retries, tolerance changes, excluded folds or imputation.'}


def write_csv(path, rows):
    require(bool(rows), 'Refusing an unspecified empty CSV')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_record(model):
    return {key: value for key, value in model.items() if key != 'B'}


def check_historical_fold(bundle, split, result):
    official = bundle['official']
    if official is None:
        return {'historical_original_available': False}
    directory = official['directory'] / 'cv/folds' / f"repeat_{split['repeat']}_fold_{split['outer_fold']}" / 'Tensor-ABESS'
    saved = read_json(directory / 'complete.json')
    parameters = json.loads(saved['fold_metrics']['selected_parameters'])
    expected = {'B': np.load(directory / 'B.npy', allow_pickle=False),
                'support': read_json(directory / 'support.json'), 'q': saved['fold_metrics']['q'],
                'score': parameters['sic']}
    ids = bundle['data']['subject_ids'][split['test']].tolist()
    require([row['subject_id'] for row in saved['predictions']] == ids, 'Historical held-out subject order differs')
    verify_original(result['pair']['original'], expected, result['predictions']['original'],
                    np.asarray([row['y_pred'] for row in saved['predictions']]))
    train_rmse = metric(bundle['data']['y'][split['train']], result['train_predictions']['original'])['rmse']
    require(train_rmse == saved['fold_metrics']['train_rmse'], 'Historical training prediction error differs')
    return {'historical_original_available': True, 'exact_B_support_q_score_test_prediction': True,
            'exact_training_rmse': True, 'historical_complete_sha256': file_sha(directory / 'complete.json')}


def selection_change(split, result):
    pair = result['pair']
    record = {'repeat': split['repeat'], 'outer_fold': split['outer_fold'],
              'changed': pair['original']['B_sha256'] != pair['fs']['B_sha256'] or
                         pair['original']['support'] != pair['fs']['support']}
    for rule in ('original', 'fs'):
        model, choice = pair[rule], pair[rule]['selection']
        record.update({rule + '_q': model['q'], rule + '_shape': json.dumps(model['sizes']),
                       rule + '_support': json.dumps(model['support']), rule + '_B_sha256': model['B_sha256'],
                       rule + '_rss': model['rss'], rule + '_score': model['score'],
                       rule + '_argmin_q': choice['minimum_q'], rule + '_minimum_score': choice['minimum_score'],
                       rule + '_band': choice['tolerance']})
    return record


def render_report(summary):
    lines = [f"# {summary['dataset']} 双规则实证验证", "",
             "本报告是已见数据上的固定规则消融。全局建议仍为 PENDING_CLUSTER_RESULTS。",
             "OOF 指标先在每次重复的完整受试者上计算，再对重复取平均；重复使用同一批受试者，不作为独立试验。", "",
             "| 模型 | OOF RMSE | OOF R² |", "|---|---:|---:|"]
    for name, values in summary['models'].items():
        lines.append(f"| {name} | {values['mean_oof_rmse']:.6f} | {values['mean_oof_r2']:.6f} |")
    paired = summary['paired']
    lines += ["", f"FS−original：RMSE {paired['rmse_delta']:+.6f}，R² {paired['r2_delta']:+.6f}；"
              f"外折改善/相同/恶化={paired['improved_folds']}/{paired['unchanged_folds']}/{paired['worsened_folds']}。",
              f"同池原规则均值 q={summary['q']['original']['mean']:.3f}，FS={summary['q']['fs']['mean']:.3f}；"
              f"两臂每折一次外测预测，合计 {summary['native_search_calls']} 次原始搜索。",
              "仅协变量与既有 baseline 均保留，并逐项核验历史输入、划分、训练预处理与产物身份；未重新调参或重跑 baseline。"]
    if paired['fs_beats_covariates_only'] is False:
        lines += ["FS 的平均 OOF RMSE 仍未优于仅协变量；不能据相对原 MSIC 的变化宣称新增张量预测价值。"]
    lines += ["", "全部不利外折（无删选）："]
    adverse = paired['adverse_folds']
    lines.extend(f"- repeat={row['repeat']}，fold={row['outer_fold']}，RMSE "
                 f"{row['original_rmse']:.6f}→{row['fs_rmse']:.6f}（{row['rmse_delta']:+.6f}）。"
                 for row in adverse)
    if not adverse:
        lines.append("- 无 RMSE 恶化外折。")
    lines += ["", "完整逐受试者预测、每折训练/测试误差、q/形状/支撑、候选池与选择记录均随结果保存。",
              "本次不将外测响应传入选阶或预处理拟合，不根据结果调整公式、候选池、阈值或基线；未做事后数据集合并。", ""]
    return '\n'.join(lines)


def run(config_path):
    bundle = prepare(config_path)
    config = bundle['config']
    protocol = read_json(root() / config['protocol'])
    gate_path = root() / config['p0_gate']
    verify_authorization(protocol, bundle['plan'], gate_path, root())
    require(protocol['source_files'] == bundle['source_files'], 'Incomplete or changed runtime source manifest')
    output = (root() / config['output_dir']).resolve()
    require(output.is_relative_to(root()) and not output.exists(), 'Output exists or escapes repository')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    data, predictions, fits, changes, checks = bundle['data'], list(bundle['baseline_rows']), [], [], []
    native_calls, extra_calls, candidates = 0, 0, 0
    started = perf_counter()
    for split in bundle['splits']:
        verify_authorization(protocol, bundle['plan'], gate_path, root())
        repeat, fold = split['repeat'], split['outer_fold']
        train, test = split['train'], split['test']
        folder = output / 'folds' / f'repeat_{repeat}_fold_{fold}'
        folder.mkdir(parents=True, exist_ok=False)
        prepared = bundle['prepared'][repeat, fold]
        before = {key: array_sha(value) for key, value in prepared.to_arrays().items()}

        def persist_selected(pair, preprocessor):
            require(before == {key: array_sha(value) for key, value in preprocessor.to_arrays().items()},
                    'Training preprocessing mutated')
            selection.save_pool(folder / 'pool', pair['pool'])
            write_npz(folder / 'preprocessing.npz', preprocessor.to_arrays())
            write_json(folder / 'selection.json', {rule: model_record(pair[rule]) for rule in ('original', 'fs')})
            write_npz(folder / 'models.npz', {rule: pair[rule]['B'] for rule in ('original', 'fs')})
            write_json(folder / 'split.json', {**split, 'train_subject_ids': data['subject_ids'][train].tolist(),
                                               'test_subject_ids': data['subject_ids'][test].tolist()})

        fold_started = perf_counter()
        result = fit_fold(data['X'][train], data['y'][train], data['C'][train], data['X'][test],
                          data['C'][test], bundle['plan']['native_options'], lambda *args: prepared,
                          on_selected=persist_selected)
        elapsed = perf_counter() - fold_started
        checks.append({'repeat': repeat, 'outer_fold': fold, **check_historical_fold(bundle, split, result)})
        metadata = result['pair']['pool']['metadata']
        require(metadata['audit']['passed'] is True, 'Candidate audit did not pass')
        count = metadata['counters']
        require(count['native_search_calls'] == 1 and count['extra_search_calls'] == 0,
                'Empirical fold did not use exactly one native search')
        native_calls += count['native_search_calls']
        extra_calls += count['extra_search_calls']
        candidates += len(metadata['trace']['candidates'])
        arrays = {'y_train': data['y'][train], 'y_test': data['y'][test]}
        for rule in ('original', 'fs'):
            model = result['pair'][rule]
            train_metric = metric(data['y'][train], result['train_predictions'][rule])
            test_metric = metric(data['y'][test], result['predictions'][rule])
            fits.append({'model': rule, 'repeat': repeat, 'outer_fold': fold, 'n_train': len(train),
                         'n_test': len(test), 'q': model['q'], 'shape': json.dumps(model['sizes']),
                         'support': json.dumps(model['support']), 'B_sha256': model['B_sha256'],
                         'residual_rss': model['rss'], 'score': model['score'],
                         'train_rmse': train_metric['rmse'], 'train_rss': train_metric['rss'],
                         'test_rmse': test_metric['rmse'], 'test_r2': test_metric['r2'],
                         'test_rss': test_metric['rss'], 'shared_fold_seconds': elapsed})
            arrays[rule + '_train_prediction'] = result['train_predictions'][rule]
            arrays[rule + '_test_prediction'] = result['predictions'][rule]
            predictions.extend({'subject_id': str(data['subject_ids'][index]), 'repeat': repeat, 'outer_fold': fold,
                                'model': rule, 'y_true': float(data['y'][index]), 'y_pred': float(prediction)}
                               for index, prediction in zip(test, result['predictions'][rule]))
        write_npz(folder / 'predictions.npz', arrays)
        changes.append(selection_change(split, result))
        write_json(folder / 'complete.json', {'repeat': repeat, 'outer_fold': fold,
                    'pool_digest': result['pair']['pool']['digest'], 'audit': checks[-1],
                    'test_prediction_calls': {'original': 1, 'fs': 1}, 'elapsed_seconds': elapsed})
    current = prepare(config_path)
    require(current['plan'] == bundle['plan'] and current['source_files'] == bundle['source_files'],
            'Inputs, historical baselines, source, environment or preprocessing changed during run')
    models = ['original', 'fs', *sorted(set(row['model'] for row in bundle['baseline_rows']))]
    summary = summarize_predictions(predictions, data, bundle['splits'], models)
    summary.update({'dataset': bundle['plan']['dataset'], 'status': 'COMPLETE', 'research_only': True,
                    'final_recommendation': 'PENDING_CLUSTER_RESULTS', 'n_subjects': len(data['y']),
                    'fold_count': len(bundle['splits']), 'native_search_calls': native_calls,
                    'extra_search_calls': extra_calls, 'candidate_count': candidates,
                    'original_historical_checks': checks, 'elapsed_seconds': perf_counter() - started,
                    'selection_changed_folds': sum(row['changed'] for row in changes)})
    summary['q'] = {rule: {'mean': float(np.mean([r['q'] for r in fits if r['model'] == rule])),
                           'median': float(np.median([r['q'] for r in fits if r['model'] == rule])),
                           'minimum': min(r['q'] for r in fits if r['model'] == rule),
                           'maximum': max(r['q'] for r in fits if r['model'] == rule),
                           'above_half_training_n': sum(r['q'] > r['n_train'] / 2 for r in fits if r['model'] == rule),
                           'mean_train_rmse': float(np.mean([r['train_rmse'] for r in fits if r['model'] == rule]))}
                    for rule in ('original', 'fs')}
    write_csv(output / 'oof_predictions.csv', predictions)
    write_csv(output / 'fold_metrics.csv', fits)
    write_csv(output / 'repeat_metrics.csv', summary['repeat_metrics'])
    write_csv(output / 'selection_changes.csv', changes)
    write_csv(output / 'all_model_fold_metrics.csv', summary['fold_metrics'])
    write_json(output / 'summary.json', summary)
    (output / 'report.md').write_text(render_report(summary), encoding='utf-8')
    write_json(output / 'seal.json', {'schema_version': 1, 'complete': True,
               'protocol_sha256': file_sha(root() / config['protocol']),
               'files': {path.relative_to(output).as_posix(): file_sha(path)
                         for path in sorted(output.rglob('*')) if path.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        summary = run(args.config)
        print(summary['dataset'], summary['status'], summary['fold_count'], 'paired folds')
        return
    bundle = prepare(args.config)
    path = root() / 'doc/taskcard12/verification' / ('empirical_' + bundle['config']['dataset'].lower().replace('-', '_') + '_protocol_draft.json')
    write_json(path, build_protocol(bundle))
    print('Read-only audit passed; draft written:', path)


if __name__ == '__main__':
    main()
