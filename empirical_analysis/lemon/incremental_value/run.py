"""Audited, resumable repeated nested CV; old artifacts remain read-only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import traceback
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import yaml
from .nested_cv import make_splits, prepare_inner_folds, select_parameters
from .preprocessing import fit_preprocessor


def save_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload, encoding='utf-8')
    os.replace(temporary, path)


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def experiment_id(config, fingerprints):
    return hashlib.sha256(json.dumps({'config': config, 'fingerprints': fingerprints},
                          sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_predictions(rows, subject_ids, models, repeats):
    keys = [(str(r['subject_id']), int(r['repeat']), r['model']) for r in rows]
    expected = {(str(i), int(r), m) for i in subject_ids for r in repeats for m in models}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('OOF subject/repeat/model coverage missing or duplicated')
    values = np.array([[r['y_true'], r['y_pred']] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite OOF prediction')


def _versions():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(),
            'threads': {key: os.environ.get(key) for key in
                        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS')}}


def _source_fingerprints(repo):
    paths = list((repo / 'src/tensor_abess').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += [repo / 'empirical_analysis/lemon/confounds.py',
              repo / 'empirical_analysis/lemon/models.py',
              repo / 'numerical_simulation/methods.py',
              repo / 'numerical_simulation/data_generation.py']
    return {str(p.relative_to(repo)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if not p.name.startswith('test_') and p.name != 'reporting.py'}


def _scores(y, pred):
    e = np.asarray(y) - pred
    return {'rmse': float(np.sqrt(np.mean(e**2))), 'mae': float(np.mean(np.abs(e)))}


def execute(config_file, output_override=None, smoke=False, audit_only=False):
    from .data_loader import load_audited_dataset
    from .model_adapters import fit_eeg_model, make_parameter_grid
    cfg_path = Path(config_file).resolve()
    config = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if config['outer_repeats'] != len(config['outer_seeds']):
        raise ValueError('outer_repeats must equal the predefined seed count')
    anchor = cfg_path.parent
    source = (anchor / config['source_dir']).resolve()
    output = Path(output_override).resolve() if output_override else (anchor / config['output_dir']).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('New output must be physically separate from old results')
    output.mkdir(parents=True, exist_ok=True)
    try:
        data = load_audited_dataset(source, (anchor / config['tmt_csv']).resolve(),
                                   (anchor / config['participants_file']).resolve())
        repo = anchor.parents[2]
        fingerprints = {**data.fingerprints, **_source_fingerprints(repo)}
        effective = {**config, 'smoke': bool(smoke), 'environment': _versions()}
        run_id = experiment_id(effective, fingerprints)
        state_file = output / 'run_state.json'
        if state_file.exists():
            old = json.loads(state_file.read_text(encoding='utf-8'))
            if old.get('experiment_id') != run_id:
                raise ValueError('Existing output belongs to different inputs/config/code/environment')
        elif any(output.iterdir()):
            raise FileExistsError('Nonempty output without a valid experiment identity')
        save_json(output / 'config_used.json', effective)
        save_json(output / 'diagnostics/dataset_audit.json', data.audit)
        save_json(output / 'diagnostics/input_fingerprint.json', fingerprints)
        state = {'experiment_id': run_id, 'status': 'audited', 'smoke': smoke,
                 'completed_fits': 0, 'expected_fits': len(config['models']) *
                 config['outer_folds'] * config['outer_repeats'], 'complete': False}
        save_json(state_file, state)
        save_json(output / 'diagnostics/preprocessing_audit.json', {
            'old_outer_preprocessing': 'training-only nuisance/scaling',
            'old_baseline_inner_preprocessing': 'outer preprocessed data passed into internal CV; not refit per inner fold',
            'new_baseline_inner_preprocessing': 'independent inner-training nuisance/scaling/response centering',
            'abess_policy': 'unchanged adaptive MSIC, BIC initialization, no externally tuned ABESS hyperparameters',
            'prediction_formula': 'C alpha + y_residual_mean + <(X-C Gamma-X_mean)/X_scale,B>',
            'response_scale': 'natural log(TMT_5)', 'flatten_order': 'C',
            'inner_objective': 'pooled validation MSE on original log response',
            'tie_break': 'largest regularization strength among scores tied within 1e-12'})
        if audit_only:
            print('DATA_AUDIT_PASSED', data.X.shape, flush=True)
            return state
        splits = make_splits(data.C, config['outer_folds'], config['outer_seeds'])
        split_rows = [{'subject_id': str(data.subject_ids[i]), 'repeat': s['repeat'],
                       'outer_fold': s['outer_fold'], 'outer_seed': s['seed']}
                      for s in splits for i in s['test']]
        save_csv(output / 'cv/outer_splits.csv', split_rows)
        predictions, fold_rows, inner_rows, all_traces = [], [], [], []
        active_splits = splits[:1] if smoke else splits
        started = perf_counter()
        for split in active_splits:
            repeat, fold = split['repeat'], split['outer_fold']
            train, test = split['train'], split['test']
            X, y, C = data.X[train], data.y[train], data.C[train]
            inner_seed = config['inner_seed'] + repeat * 100 + fold
            fold_dir = output / 'cv/folds' / f'repeat_{repeat}_fold_{fold}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            p = fit_preprocessor(X, y, C)
            np.savez_compressed(fold_dir / 'preprocessing.npz', **p.to_arrays())
            train_x, train_y = p.apply(X, C), p.transform_response(y, C)
            prepared = None
            save_json(fold_dir / 'split.json', {
                'repeat': repeat, 'outer_fold': fold, 'seed': split['seed'],
                'inner_seed': inner_seed, 'train_subject_ids': data.subject_ids[train].tolist(),
                'test_subject_ids': data.subject_ids[test].tolist()})
            for name in config['models']:
                fit_dir = fold_dir / name.replace(' ', '_')
                marker = fit_dir / 'complete.json'
                if marker.exists():
                    saved = json.loads(marker.read_text(encoding='utf-8'))
                    if saved['experiment_id'] != run_id:
                        raise ValueError('Fold completion identity mismatch')
                    predictions.extend(saved['predictions'])
                    fold_rows.append(saved['fold_metrics'])
                    inner_rows.extend(saved['inner_selection'])
                    all_traces.extend(saved['inner_traces'])
                    print('RESUME', repeat, fold, name, flush=True)
                    continue
                print('FIT_START', repeat, fold, name, flush=True)
                fit_dir.mkdir(parents=True, exist_ok=True)
                t0 = perf_counter()
                rows, traces, parameter = [], [], {}
                if name == 'Covariates Only':
                    B = np.zeros(data.X.shape[1:])
                    supports = ([], [], [])
                else:
                    parameter = None
                    if name != 'Tensor-ABESS':
                        prepared = prepare_inner_folds(X, y, C, config['inner_folds'], inner_seed) if prepared is None else prepared
                        candidates = make_parameter_grid(name, train_x, train_y, config)
                        parameter, rows, traces = select_parameters(
                            name, X, y, C, candidates, config['inner_folds'], inner_seed,
                            config['model'], prepared=prepared)
                        rows = [{**r, 'repeat': repeat, 'outer_fold': fold,
                                 'parameters': json.dumps(r['parameters'], sort_keys=True)} for r in rows]
                        traces = [{**t, 'repeat': repeat, 'outer_fold': fold, 'model': name,
                                   'train_subject_ids': data.subject_ids[train[t['train']]].tolist(),
                                   'validation_subject_ids': data.subject_ids[train[t['validation']]].tolist()}
                                  for t in traces]
                    fitted = fit_eeg_model(name, train_x, train_y, parameter, config['model'])
                    B = fitted.B
                    supports = fitted.supports
                    parameter = fitted.parameters
                test_pred = p.predict(B, data.X[test], data.C[test])
                train_pred = p.predict(B, X, C)
                sizes = [len(s) for s in supports]
                model_rows = [{'subject_id': str(data.subject_ids[i]), 'repeat': repeat,
                               'outer_fold': fold, 'y_true': float(data.y[i]),
                               'y_pred': float(pred), 'model': name,
                               'selected_parameters': json.dumps(parameter, sort_keys=True),
                               's1': sizes[0], 's2': sizes[1], 's3': sizes[2],
                               'q': int(np.prod(sizes))}
                              for i, pred in zip(test, test_pred)]
                fm = {'model': name, 'repeat': repeat, 'outer_fold': fold,
                      'n_train': len(train), 'n_test': len(test),
                      'train_rmse': _scores(y, train_pred)['rmse'],
                      'test_rmse': _scores(data.y[test], test_pred)['rmse'],
                      'runtime_sec': perf_counter() - t0, 'selected_parameters': json.dumps(parameter, sort_keys=True),
                      's1': sizes[0], 's2': sizes[1], 's3': sizes[2], 'q': int(np.prod(sizes))}
                np.save(fit_dir / 'B.npy', B, allow_pickle=False)
                save_json(fit_dir / 'support.json', [np.asarray(s, dtype=int).tolist() for s in supports])
                save_json(marker, {'experiment_id': run_id, 'predictions': model_rows,
                                  'fold_metrics': fm, 'inner_selection': rows, 'inner_traces': traces})
                predictions.extend(model_rows)
                fold_rows.append(fm)
                inner_rows.extend(rows)
                all_traces.extend(traces)
                state.update(status='running', completed_fits=len(fold_rows),
                             elapsed_sec=perf_counter() - started)
                save_json(state_file, state)
                save_csv(output / 'cv/oof_predictions.csv', predictions)
                print('FIT_DONE', repeat, fold, name, round(fm['runtime_sec'], 2), flush=True)
            del prepared
        save_csv(output / 'cv/oof_predictions.csv', predictions)
        save_csv(output / 'cv/fold_metrics.csv', fold_rows)
        save_csv(output / 'cv/inner_selection.csv', inner_rows)
        save_json(output / 'diagnostics/leakage_checks.json', {
            'outer_splits_validated': True, 'same_splits_for_all_models': True,
            'preprocessing_fit_subjects': 'outer train or current inner train only',
            'inner_traces': all_traces, 'outer_test_used_for_selection': False,
            'all_subjects_once_per_repeat': not smoke})
        if smoke:
            state.update(status='smoke_complete', complete=False, completed_fits=len(fold_rows))
        else:
            validate_predictions(predictions, data.subject_ids, config['models'],
                                 range(config['outer_repeats']))
            from .metrics import summarize_oof
            from .reporting import generate_reports
            stats = summarize_oof(predictions, config['bootstrap_seed'], config['bootstrap_samples'])
            generate_reports(output, effective, data.audit, stats, fold_rows)
            save_json(output / 'diagnostics/reporting_fingerprint.json', {
                'reporting_sha256': hashlib.sha256(Path(__file__).with_name('reporting.py').read_bytes()).hexdigest(),
                'policy': 'figure layout is tracked separately from frozen model-fitting code'})
            state.update(status='complete', complete=True, completed_fits=len(fold_rows),
                         n_oof_predictions=len(predictions), elapsed_sec=perf_counter() - started)
        save_json(state_file, state)
        return state
    except Exception as error:
        save_json(output / 'diagnostics/failure.json', {
            'error_type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'real_run_complete': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--output-dir')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(json.dumps(execute(args.config, args.output_dir, args.smoke, args.audit_only), ensure_ascii=False))


if __name__ == '__main__':
    main()
