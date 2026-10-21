"""Frozen, isolated v5.1 training-internal complexity diagnosis (task card 7 C3).

Prepare protocol before running. All fitted supports and nuisance parameters use
only inner-training subjects. No outer-test predictions or scores are computed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_sizes(sizes, n, shape):
    require(len(sizes) == 3 and all(isinstance(s, int) and 1 <= s <= p
                                   for s, p in zip(sizes, shape)), 'Invalid fixed sizes')
    q = int(np.prod(sizes))
    require(q <= int(np.floor(.7 * n)) and q <= n, 'Fixed size exceeds training capacity')
    return q


def make_plan(C, ids, config):
    from empirical_analysis.lemon.incremental_value.nested_cv import make_splits
    plan = []
    for split in make_splits(C, config['outer_folds'], config['outer_seeds']):
        if split['outer_fold'] != 1:
            continue
        outer = split['train']
        seed = config['inner_seed'] + split['repeat'] * 100 + split['outer_fold']
        inner = [{'inner_fold': s['outer_fold'], 'train': s['train'].tolist(),
                  'validation': s['test'].tolist(),
                  'train_ids': ids[outer[s['train']]].tolist(),
                  'validation_ids': ids[outer[s['test']]].tolist()}
                 for s in make_splits(C[outer], config['inner_folds'], [seed])]
        plan.append({'repeat': split['repeat'], 'outer_fold': split['outer_fold'],
                     'outer_seed': split['seed'], 'inner_seed': seed,
                     'outer_train': outer.tolist(), 'outer_train_ids': ids[outer].tolist(),
                     'inner': inner})
    return plan


def fit_inner(X, y, C, train, validation, options, models):
    from empirical_analysis.lemon.incremental_value.preprocessing import fit_preprocessor
    from empirical_analysis.lemon.models import fit_model
    from tensor_abess.tensor.bcd import tensor_bcd
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
    train, validation = np.asarray(train), np.asarray(validation)
    require(not set(train) & set(validation), 'Inner train and validation overlap')
    p = fit_preprocessor(X[train], y[train], C[train])
    Xt = p.apply(X[train], C[train])
    yt = p.transform_response(y[train], C[train])
    fitted = []
    for name, sizes in models:
        started = perf_counter()
        if name == 'Covariates Only':
            B = np.zeros(X.shape[1:])
            support = [np.array([], dtype=int)] * 3
        elif sizes is None:
            result = fit_model('Tensor-ABESS', Xt, yt, options)
            B, support = result.B, [result.S1, result.S2, result.S3]
        else:
            validate_sizes(sizes, len(train), X.shape[1:])
            # Same marginal initialization and unchanged fixed BCD options as
            # the global MSIC candidate search; no full-outer support is reused.
            result = tensor_bcd(Xt, yt, *sizes,
                                **{k: options[k] for k in ('max_cycles', 'm_max', 'k_max')})
            B, support = result.B, [result.S1, result.S2, result.S3]
        lengths = tuple(len(s) for s in support)
        q = int(np.prod(lengths))
        require(q <= int(np.floor(.7 * len(train))), 'Actual q exceeds training capacity')
        design = Xt[np.ix_(np.arange(len(train)), *support)].reshape(len(train), q)
        rank = int(np.linalg.matrix_rank(design)) if q else 0
        require(rank == q, 'Selected tensor design is rank deficient')
        rss = float(np.sum((yt - np.einsum('nijk,ijk->n', Xt, B)) ** 2))
        msic = (float(compute_tensor_msic_from_rss(
            rss, *lengths, len(train), *X.shape[1:],
            alpha=options['msic_refine_alpha'], gamma=options['msic_refine_gamma'])) if q else None)
        train_pred = p.predict(B, X[train], C[train])
        valid_pred = p.predict(B, X[validation], C[validation])
        fitted.append({'model': name, 'B': B, 'support': [s.tolist() for s in support],
                       'q': q, 'sizes': lengths, 'rank': rank, 'rss': rss, 'msic': msic,
                       'condition_number': float(np.linalg.cond(design)) if q else None,
                       'train_rmse': float(np.sqrt(np.mean((y[train] - train_pred) ** 2))),
                       'validation_rmse': float(np.sqrt(np.mean((y[validation] - valid_pred) ** 2))),
                       'train_prediction': train_pred, 'validation_prediction': valid_pred,
                       'runtime_sec': perf_counter() - started})
    return {'preprocessing': p.to_arrays(),
            'preprocessing_hash': array_hash(p.to_arrays()), 'fits': fitted}


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bind_core(repo):
    require(not any(k == 'tensor_abess' or k.startswith('tensor_abess.') for k in sys.modules),
            'Requires a fresh process without any previously imported tensor core')
    core = repo / 'doc/algorithm_versions/v5.1-bcd-fix/src'
    files = sorted((core / 'tensor_abess').rglob('*.py'))
    require(len(files) == 32, 'Frozen v5.1 must contain exactly 32 Python sources')
    sys.path.insert(0, str(core))
    manifest = {p.relative_to(core).as_posix(): digest(p) for p in files}
    expected = json.loads((core.parent / 'VERSION.json').read_text(encoding='utf-8-sig'))
    require(manifest == expected['source_sha256'], 'Frozen v5.1 source32 differs from VERSION manifest')
    # Import exactly the methods exercised by C3. Unused optional baselines are
    # still covered by the complete 32-file manifest, never executed by C3.
    for name in ('tensor_abess.tensor.abess', 'tensor_abess.tensor.bcd',
                 'tensor_abess.tensor.sic'):
        importlib.import_module(name)
    imported = {}
    for name, module in list(sys.modules.items()):
        if not (name == 'tensor_abess' or name.startswith('tensor_abess.')):
            continue
        actual = Path(module.__file__).resolve()
        require(core.resolve() in actual.parents, f'Actual import outside frozen v5.1: {name}')
        relative = actual.relative_to(core).as_posix()
        require(relative in manifest and digest(actual) == manifest[relative],
                f'Actual import differs from frozen v5.1: {name}')
        imported[name] = {'path': str(actual), 'sha256': manifest[relative]}
    return {'version': 'v5.1-bcd-fix', 'source32': manifest, 'actual_imports': imported,
            'reason': 'BCD-corrected production version isolates model complexity from the A2 grid change'}


def execute(mode, repo, output):
    sys.path.insert(0, str(repo))
    core = bind_core(repo)
    from empirical_analysis.lemon.incremental_value.data_loader import load_audited_dataset
    cfg_path = repo / 'empirical_analysis/lemon/incremental_value/config.yaml'
    config = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    anchor = cfg_path.parent
    data = load_audited_dataset((anchor / config['source_dir']).resolve(),
                               (anchor / config['tmt_csv']).resolve(),
                               (anchor / config['participants_file']).resolve())
    require(data.X.shape == (145, 61, 16, 5) and data.C.shape == (145, 8),
            'LEMON sample/tensor/covariate shape drift')
    models = [('Covariates Only', None), ('Adaptive MSIC', None),
              ('Fixed small', (1, 1, 1)), ('Fixed medium', (2, 2, 3)),
              ('Fixed large', (3, 3, 5))]
    plan = make_plan(data.C, data.subject_ids, config)
    for outer in plan:
        require(len(outer['outer_train']) == 116, 'Unexpected outer training size')
        for inner in outer['inner']:
            for _, sizes in models[2:]:
                validate_sizes(sizes, len(inner['train']), data.X.shape[1:])
    helpers = ['data_loader.py', 'nested_cv.py', 'preprocessing.py']
    helpers = [anchor / h for h in helpers] + [anchor.parent / 'confounds.py', anchor.parent / 'models.py', Path(__file__).resolve()]
    protocol = {'phase': 'C3', 'research_only': True, 'outer_test_used': False,
                'representative_rule': 'first fold of every predeclared repeat, before looking at test outcomes',
                'models': models, 'rejected_size': {'sizes': [4, 4, 5], 'q': 80,
                    'reason': 'exceeds floor(0.7*n_inner_train) for n=77/78; never fitted'},
                'config': config, 'config_sha256': digest(cfg_path), 'core': core,
                'input_fingerprints': data.fingerprints,
                'helpers_sha256': {p.relative_to(repo).as_posix(): digest(p) for p in helpers},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'platform': platform.platform(),
                    'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}},
                'plan': plan, 'response': 'original log(TMT_5) response; nuisance restored',
                'aggregation': 'pooled inner validation SSE/n separately per outer training set; descriptive only, no selection'}
    canonical = json.loads(json.dumps(protocol))
    if mode == 'prepare':
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'protocol.json', canonical)
        print('PROTOCOL_FROZEN', digest(output / 'protocol.json'), flush=True)
        return
    require(json.loads((output / 'protocol.json').read_text(encoding='utf-8')) == canonical,
            'Frozen protocol input/config/source/environment drift')
    result_dir = output / 'results'
    result_dir.mkdir(exist_ok=False)
    metrics, predictions = [], []
    started = perf_counter()
    for outer in plan:
        indices = np.asarray(outer['outer_train'])
        # This is the only data handed to the fitting routine: 116 outer-training people.
        X, y, C, ids = data.X[indices], data.y[indices], data.C[indices], data.subject_ids[indices]
        for inner in outer['inner']:
            tag = f"repeat_{outer['repeat']}_fold_1_inner_{inner['inner_fold']}"
            fold_dir = result_dir / tag
            fold_dir.mkdir()
            fitted = fit_inner(X, y, C, inner['train'], inner['validation'], config['model'], models)
            np.savez(fold_dir / 'preprocessing.npz', **fitted['preprocessing'])
            save_json(fold_dir / 'split.json', dict(inner, preprocessing_hash=fitted['preprocessing_hash']))
            for fit in fitted['fits']:
                model_dir = fold_dir / fit['model'].replace(' ', '_')
                model_dir.mkdir()
                np.save(model_dir / 'B.npy', fit['B'], allow_pickle=False)
                save_json(model_dir / 'support.json', fit['support'])
                row = {k: v for k, v in fit.items() if k not in ('B', 'train_prediction', 'validation_prediction', 'support')}
                row.update(repeat=outer['repeat'], outer_fold=1, inner_fold=inner['inner_fold'],
                           n_train=len(inner['train']), n_validation=len(inner['validation']),
                           preprocessing_hash=fitted['preprocessing_hash'])
                save_json(model_dir / 'metrics.json', row)
                metrics.append(row)
                for partition, key in [('train', 'train_prediction'), ('validation', 'validation_prediction')]:
                    for index, prediction in zip(inner[partition], fit[key]):
                        predictions.append({'repeat': outer['repeat'], 'outer_fold': 1,
                            'inner_fold': inner['inner_fold'], 'partition': partition,
                            'subject_id': str(ids[index]), 'model': fit['model'],
                            'y_true': float(y[index]), 'y_pred': float(prediction)})
            save_csv(result_dir / 'fit_metrics.csv', metrics)
            save_csv(result_dir / 'predictions.csv', predictions)
            print('INNER_COMPLETE', tag, flush=True)
    aggregates = []
    for repeat in range(3):
        for name, _ in models:
            group = [r for r in predictions if r['repeat'] == repeat and r['model'] == name and r['partition'] == 'validation']
            require(len(group) == 116 and len({r['subject_id'] for r in group}) == 116, 'Incomplete or duplicate inner OOF')
            error = np.array([r['y_true'] - r['y_pred'] for r in group])
            fits = [r for r in metrics if r['repeat'] == repeat and r['model'] == name]
            aggregates.append({'repeat': repeat, 'outer_fold': 1, 'model': name,
                'n_inner_oof': len(group), 'inner_oof_rmse': float(np.sqrt(np.mean(error ** 2))),
                'inner_oof_mae': float(np.mean(np.abs(error))),
                'pooled_train_rmse': float(np.sqrt(sum(r['train_rmse']**2*r['n_train'] for r in fits)/sum(r['n_train'] for r in fits))),
                'q_by_inner_fold': [r['q'] for r in fits]})
    save_csv(result_dir / 'inner_oof_summary.csv', aggregates)
    artifacts = {p.relative_to(result_dir).as_posix(): digest(p) for p in sorted(result_dir.rglob('*')) if p.is_file()}
    save_json(output / 'completion.json', {'status': 'complete', 'fits': len(metrics),
        'n_validation_predictions': sum(r['partition'] == 'validation' for r in predictions),
        'protocol_sha256': digest(output / 'protocol.json'), 'artifact_sha256': artifacts,
        'runtime_sec': perf_counter() - started, 'outer_test_used': False})
    print('COMPLETE', len(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    execute(args.mode, root, root / 'numerical_simulation/outputs_version_attribution/train_only')
