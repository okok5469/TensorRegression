"""Frozen, resumable SpectroFood Apple experiment; no optimizer modifications."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import importlib.metadata

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial,
                               allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot publish an empty result table')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_folds(groups, count, seed):
    groups = np.asarray(groups).astype(str)
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < count:
        raise ValueError('Insufficient independent entities for frozen folds')
    order = np.random.default_rng(seed).permutation(len(unique))
    order = order[np.argsort(-sizes[order], kind='stable')]
    loads = np.zeros(count, dtype=int)
    assigned = np.empty(len(unique), dtype=int)
    for group in order:
        fold = int(np.argmin(loads))
        assigned[group] = fold
        loads[fold] += sizes[group]
    return [(np.flatnonzero(assigned[inverse] != k), np.flatnonzero(assigned[inverse] == k))
            for k in range(count)]


def make_splits(sample_ids, entity_ids):
    ids, groups = np.asarray(sample_ids).astype(str), np.asarray(entity_ids).astype(str)
    result = []
    for repeat, seed in enumerate((13001, 13002, 13003), 1):
        for fold, (train, test) in enumerate(group_folds(groups, 5, seed), 1):
            inner_seed = 130000 + repeat * 100 + fold
            inner = [dict(train=a.tolist(), validation=b.tolist(),
                          train_ids=ids[train[a]].tolist(), validation_ids=ids[train[b]].tolist())
                     for a, b in group_folds(groups[train], 3, inner_seed)]
            result.append(dict(repeat=repeat, fold=fold, outer_seed=seed, inner_seed=inner_seed,
                               train=train.tolist(), test=test.tolist(),
                               train_ids=ids[train].tolist(), test_ids=ids[test].tolist(), inner=inner))
    return result


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid metric inputs')
    residual = y - prediction
    denominator = np.sum((y - y.mean()) ** 2)
    if denominator <= 0:
        raise ValueError('Degenerate response in metric calculation')
    return dict(rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(abs(residual))),
                r2=float(1 - np.sum(residual**2) / denominator))


def validate_oof(rows, sample_ids, model_ids, repeats=3):
    expected = {(str(i), r, m) for i in sample_ids for r in range(1, repeats+1) for m in model_ids}
    keys = [(row['sample_id'], int(row['repeat']), row['model']) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate OOF sample/repeat/model coverage')


def source_identity(config):
    manifest = read_json(config['release_manifest'])
    actual = {}
    for name, expected in manifest['source_canonical_lf_sha256'].items():
        raw = Path(name).read_bytes()
        canonical = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
        if canonical != expected:
            raise ValueError('Released source identity mismatch: ' + name)
        actual[name] = file_hash(name)
    names = list(Path('numerical_simulation').glob('*.py'))
    names += list(Path('empirical_analysis/spectrofood_apple').glob('*.py'))
    names += [Path(p) for p in (
        'empirical_analysis/__init__.py', 'empirical_analysis/lemon/__init__.py',
        'empirical_analysis/lemon/models.py', 'empirical_analysis/lemon/confounds.py',
        'empirical_analysis/lemon/incremental_value/__init__.py',
        'empirical_analysis/lemon/incremental_value/model_adapters.py')]
    for name in sorted(names):
        actual[name.as_posix()] = file_hash(name)
    return actual


def runtime_identity():
    import platform
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name:importlib.metadata.version(name) for name in
                          ('numpy','scipy','scikit-learn','joblib','threadpoolctl')})


def load_data(config):
    with np.load(Path(config['data_dir']) / 'data.npz', allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def select_input(data, model):
    key = {'M5': 'spectrum_binned', 'M6': 'spectrum_native'}.get(model, 'X')
    return data[key]


def frozen_contract(config):
    output = Path(config['output_dir'])
    contract = read_json(output / 'contract.json')
    now = dict(config=config, source=source_identity(config),
               data_sha256=file_hash(Path(config['data_dir']) / 'data.npz'),
               splits=read_json(output / 'splits.json'))
    if fingerprint(now) != contract['fingerprint']:
        raise ValueError('Frozen data/config/source/split contract changed')
    return contract


def freeze(config):
    data = load_data(config)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    splits = make_splits(data['sample_ids'], data['entity_ids'])
    body = dict(config=config, source=source_identity(config),
                data_sha256=file_hash(Path(config['data_dir']) / 'data.npz'), splits=splits)
    contract = dict(fingerprint=fingerprint(body), **body)
    if (output / 'contract.json').exists():
        frozen_contract(config)
        return
    write_json(output / 'splits.json', splits)
    write_json(output / 'contract.json', contract)
    text = ('# Frozen Apple empirical protocol\n\n'
            'Design authority: taskcard13. Frozen before any formal OOF prediction.\n\n'
            '- All QC-passing IDs; no outcome-based exclusions. Entity grouping is audited in DATA_AUDIT.md.\n'
            '- Original rectangular ROI, area-weighted 8 x 8 spatial aggregation, 24 contiguous spectral means.\n'
            '- y in dry-matter percentage points; train-fold X and y z-scores (ddof=0); inverse response transform.\n'
            '- Constant coordinates: threshold 1e-12*max(1,max_train_std); zero under train mask.\n'
            '- Seven models M0..M6; original released v5.2 adaptive MSIC, no CV selection for M1.\n'
            '- Three repeats x five entity-disjoint folds, seeds13001/13002/13003; shared inner3 folds.\n'
            '- Baselines refit X/y scaling in each inner training fold; select weighted percentage-point MSE.\n'
            '- Exact grids, objectives, solver limits and tie rule: config.yaml and models.py frozen hashes.\n'
            '- Primary: complete-repeat entity-equal OOF RMSE, MAE, R2; repeat mean and sample SD.\n'
            '- Bootstrap:2000 independent entities, all3 predictions together, seed13999; descriptive conditional intervals.\n'
            '- Replay tolerance: rtol1e-9, atol1e-10 for percentage-point predictions; identity requires exact hashes.\n'
            '- No automatic scientific retry or altered seeds/parameters; only identical completed tasks resume.\n'
            '- Smoke uses first64 outer-training records, no outer-test response; at most2 workers, BLAS1, formal7200s.\n'
            '- Cost gate estimates15 outer training runs and all tuning; insufficient budget prepares cluster handoff.\n'
            '- M6 retains native spectral resolution; selection frequencies are descriptive, no tissue ground truth.\n\n'
            f'Contract fingerprint: `{contract["fingerprint"]}`\n')
    (output / 'FROZEN_PROTOCOL.md').write_text(text, encoding='utf-8')
    write_json(output / 'config.yaml', config)


def run_one(config, task_index):
    from .models import train_model, predict_model, tune_model
    start = time.perf_counter()
    contract = frozen_contract(config)
    model_ids = config['models']
    split = contract['splits'][task_index // len(model_ids)]
    model = model_ids[task_index % len(model_ids)]
    directory = Path(config['output_dir']) / 'tasks' / f'r{split["repeat"]}_f{split["fold"]}_{model}'
    directory.mkdir(parents=True, exist_ok=True)
    environment = runtime_identity()
    cache_key = fingerprint(dict(contract=contract['fingerprint'], split=split, model=model,
                                environment=environment))
    marker = directory / 'COMPLETE.json'
    if marker.exists():
        complete = read_json(marker)
        if complete['cache_key'] != cache_key:
            raise ValueError('Stale completion marker')
        for name, digest in complete['artifacts'].items():
            if file_hash(directory / name) != digest:
                raise ValueError('Corrupt completed artifact: ' + name)
        return
    if (directory / 'FAILED.json').exists():
        raise ValueError('Prior failed task requires explicit engineering review: ' + str(directory))
    try:
        data = load_data(config)
        X, y = select_input(data, model), data['y_pct']
        train, test = np.array(split['train']), np.array(split['test'])
        inner = [(np.array(s['train']), np.array(s['validation'])) for s in split['inner']]
        param, tuning = tune_model(model, X[train], y[train], inner,
                                   seed=split['inner_seed'], config=config['model_config'])
        artifact = train_model(model, X[train], y[train], param, seed=split['inner_seed'],
                               config=config['model_config'])
        prediction = predict_model(artifact, X[test])
        train_prediction = predict_model(artifact, X[train])
        model_path = directory / 'model.pkl'
        with model_path.open('wb') as stream:
            pickle.dump(artifact, stream, protocol=5)
        with model_path.open('rb') as stream:
            restored = pickle.load(stream)
        replay = predict_model(restored, X[test])
        if not np.allclose(replay, prediction, rtol=1e-9, atol=1e-10):
            raise ValueError('Saved model prediction replay failed')
        rows = [dict(sample_id=str(data['sample_ids'][i]), entity_id=str(data['entity_ids'][i]),
                     repeat=split['repeat'], fold=split['fold'], model=model,
                     y_true_pct=float(y[i]), y_pred_pct=float(p), residual_pct=float(y[i]-p),
                     squared_error=float((y[i]-p)**2), model_artifact=str(model_path),
                     preprocessing_artifact=str(model_path), run_id=config['run_id'])
                for i, p in zip(test, prediction)]
        write_csv(directory / 'predictions.csv', rows)
        write_json(directory / 'tuning.json', dict(parameter=param, records=tuning))
        write_json(directory / 'diagnostics.json', dict(
            model=model, repeat=split['repeat'], fold=split['fold'],
            train_metrics=metrics(y[train], train_prediction), test_metrics=metrics(y[test], prediction),
            diagnostics=artifact['diagnostics'], elapsed_seconds=time.perf_counter()-start,
            prediction_replay_max_abs=float(np.max(abs(replay-prediction)))))
        write_json(marker, dict(cache_key=cache_key, contract=contract['fingerprint'],
                                environment=environment,
                                artifacts={p.name:file_hash(p) for p in directory.iterdir()
                                           if p.name in ('model.pkl','predictions.csv','tuning.json','diagnostics.json')}))
    except Exception:
        write_json(directory / 'FAILED.json', dict(traceback=traceback.format_exc(), cache_key=cache_key))
        raise


def smoke(config):
    import psutil
    from .models import train_model, predict_model
    contract = frozen_contract(config)
    data = load_data(config)
    train = np.array(contract['splits'][0]['train'])[:64]
    subtrain, valid = group_folds(data['entity_ids'][train], 3, 130101)[0]
    records = []
    for model in config['models']:
        X = select_input(data, model)
        start = time.perf_counter()
        artifact = train_model(model, X[train[subtrain]], data['y_pct'][train[subtrain]],
                               seed=130101, config=config['model_config'])
        prediction = predict_model(artifact, X[train[valid]])
        replay = predict_model(pickle.loads(pickle.dumps(artifact, protocol=5)), X[train[valid]])
        if not np.isfinite(prediction).all() or not np.allclose(prediction, replay, rtol=1e-9, atol=1e-10):
            raise ValueError('Smoke round-trip failure')
        records.append(dict(model=model, seconds=time.perf_counter()-start,
                            train_n=len(subtrain), prediction_n=len(valid),actual_fits=1,
                            peak_process_working_set_bytes=int(psutil.Process().memory_info().peak_wset),
                            memory_scope='Cumulative Windows process peak including imports, not isolated per-fit peak',
                            diagnostics=artifact['diagnostics']))
        print('SMOKE PASS', model, round(records[-1]['seconds'], 3), flush=True)
    # Cubic training-size projection, includes inner tuning and two-worker efficiency overhead.
    n = len(contract['splits'][0]['train'])
    candidates = {'M0':0,'M1':0,'M2':12,'M3':13,'M4':17,'M5':7,'M6':7}
    total = sum(r['seconds'] * (n/r['train_n'])**3 *
                (1 + 3*candidates[r['model']]*(2/3)**3) for r in records)*15/1.5
    write_json(Path(config['output_dir']) / 'SMOKE.json', dict(
        status='PASS', contract=contract['fingerprint'], records=records,
        projected_formal_seconds=total, budget_seconds=config['budget_seconds'],
        measured_fits=7,maximum_projected_formal_fits=2625,
        gate='LOCAL' if total <= config['budget_seconds'] else 'PENDING_CLUSTER_RESULTS',
        projection='cubic n scaling; tuning count upper bounds; effective1.5 of2 workers; no test responses accessed'))


def full(config, config_path):
    contract = frozen_contract(config)
    smoke_result = read_json(Path(config['output_dir']) / 'SMOKE.json')
    if smoke_result['contract'] != contract['fingerprint']:
        raise ValueError('Smoke source identity changed')
    if smoke_result['gate'] != 'LOCAL':
        raise RuntimeError('PENDING_CLUSTER_RESULTS: frozen cost projection exceeds local budget')
    queue = list(range(len(contract['splits']) * len(config['models'])))
    active = []
    start = time.monotonic()
    try:
        while queue or active:
            if time.monotonic() - start >= config['budget_seconds']:
                write_json(Path(config['output_dir']) / 'BUDGET_EXHAUSTED.json', dict(status='PENDING_CLUSTER_RESULTS'))
                raise TimeoutError('Formal local wall-clock budget exhausted')
            while queue and len(active) < config['workers']:
                index = queue.pop(0)
                log = (Path(config['output_dir']) / f'worker_{index:03d}.log').open('ab')
                process = subprocess.Popen([sys.executable, '-B', '-m', __package__+'.run',
                    '--config', str(config_path), '--stage','one','--task-index',str(index)], stdout=log, stderr=log)
                active.append((process, log, index))
            for process, log, index in list(active):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                active.remove((process, log, index))
                if code:
                    raise RuntimeError(f'Task {index} failed; inspect worker log')
                print('FORMAL PASS', index, flush=True)
            time.sleep(0.25)
    finally:
        for process, log, _ in active:
            process.terminate()
            process.wait()
            log.close()


def collect(config):
    from .models import predict_model
    contract = frozen_contract(config)
    output = Path(config['output_dir'])
    rows, fold_rows, diagnostics, replay_records = [], [], [], []
    for split in contract['splits']:
        for model in config['models']:
            directory = output/'tasks'/f'r{split["repeat"]}_f{split["fold"]}_{model}'
            complete = read_json(directory/'COMPLETE.json')
            if replay_records and complete['environment'] != replay_records[0]['native_environment']:
                raise ValueError('Completed models come from different native numerical environments')
            if complete['contract'] != contract['fingerprint']:
                raise ValueError('Mixed OOF source identity')
            key = fingerprint(dict(contract=contract['fingerprint'],split=split,model=model,
                                   environment=complete['environment']))
            if complete['cache_key'] != key:
                raise ValueError('Completed task key differs from its native environment')
            for name, digest in complete['artifacts'].items():
                if file_hash(directory/name) != digest:
                    raise ValueError('Artifact hash mismatch')
            with (directory/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                task_rows = list(csv.DictReader(stream))
            data = load_data(config)
            X, y = select_input(data,model), data['y_pct']
            train,test = np.array(split['train']),np.array(split['test'])
            with (directory/'model.pkl').open('rb') as stream:
                artifact = pickle.load(stream)
            prep = artifact['preprocessing']
            if not np.allclose(prep['x_mean'],X[train].mean(axis=0),rtol=1e-12,atol=1e-12):
                raise ValueError('Saved training mean does not match current fold')
            if not np.allclose([prep['y_mean'],prep['y_scale']],
                               [y[train].mean(),y[train].std(ddof=0)],rtol=1e-12,atol=1e-12):
                raise ValueError('Saved response scaler does not match current fold')
            saved_by_id = {r['sample_id']:r for r in task_rows}
            if len(saved_by_id)!=len(test) or set(saved_by_id)!=set(data['sample_ids'][test]):
                raise ValueError('Task predictions do not match held-out IDs')
            saved = np.array([float(saved_by_id[str(data['sample_ids'][i])]['y_pred_pct']) for i in test])
            replay = predict_model(artifact,X[test])
            if not np.allclose(saved,replay,rtol=1e-9,atol=1e-10):
                raise ValueError('Independent saved-model OOF reconstruction failed')
            for i in test:
                row = saved_by_id[str(data['sample_ids'][i])]
                if not np.isclose(float(row['y_true_pct']),y[i],rtol=0,atol=1e-12):
                    raise ValueError('Saved response is not aligned to current data')
            replay_records.append(dict(model=model,repeat=split['repeat'],fold=split['fold'],
                max_abs_error=float(np.max(abs(saved-replay))),native_environment=complete['environment']))
            rows.extend(task_rows)
            diag = read_json(directory/'diagnostics.json')
            fold_rows.append(dict(model=model, repeat=split['repeat'], fold=split['fold'],
                                  **diag['test_metrics'], seconds=diag['elapsed_seconds']))
            if model == 'M1':
                diagnostics.append(dict(repeat=split['repeat'], fold=split['fold'],
                    train_rmse=diag['train_metrics']['rmse'], test_rmse=diag['test_metrics']['rmse'],
                    **{k:json.dumps(v, default=serial) if isinstance(v,(dict,list)) else v
                       for k,v in diag['diagnostics'].items()}))
    data = load_data(config)
    validate_oof(rows, data['sample_ids'], config['models'])
    write_csv(output/'oof_predictions.csv', rows)
    write_csv(output/'fold_metrics.csv', fold_rows)
    write_csv(output/'abess_fold_diagnostics.csv', diagnostics)
    repeats = []
    # Aggregate repeated scans before metrics, one independent entity has one vote.
    for model in config['models']:
        for repeat in range(1,4):
            selected = [r for r in rows if r['model']==model and int(r['repeat'])==repeat]
            grouped = {}
            for row in selected:
                grouped.setdefault(row['entity_id'], []).append(row)
            truth = [np.mean([float(r['y_true_pct']) for r in group]) for group in grouped.values()]
            pred = [np.mean([float(r['y_pred_pct']) for r in group]) for group in grouped.values()]
            repeats.append(dict(model=model, repeat=repeat, **metrics(truth,pred)))
    write_csv(output/'repeat_metrics.csv', repeats)
    summary = [dict(model=m, **{metric+'_'+stat:float(fn([r[metric] for r in repeats if r['model']==m]))
                 for metric in ('rmse','mae','r2') for stat,fn in
                 [('mean',np.mean),('sd',lambda a:np.std(a,ddof=1))]}) for m in config['models']]
    write_csv(output/'summary_metrics.csv', summary)
    write_json(output/'OOF_REPLAY_VALIDATION.json',dict(status='PASS',records=replay_records,
        task_count=len(replay_records),oof_rows=len(rows),runtime=runtime_identity()))
    return rows, repeats, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='empirical_analysis/spectrofood_apple/config.yaml')
    parser.add_argument('--stage', choices=['audit','prepare','freeze','smoke','full','one','collect','report'], required=True)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    runtime = str(Path(config['runtime_dir']).resolve())
    sys.path.insert(0, runtime)
    sys.path.insert(0, str(Path('src').resolve()))
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    env_paths = [runtime, str(Path('src').resolve()), str(Path.cwd())]
    os.environ['PYTHONPATH'] = os.pathsep.join(env_paths)
    from threadpoolctl import threadpool_limits
    thread_limiter = threadpool_limits(limits=1)
    if args.stage in ('audit','prepare'):
        from .data import audit_and_prepare
        audit_and_prepare(config['mat_path'],config['csv_path'],config['data_dir'])
        return
    if args.stage == 'freeze':
        freeze(config)
    elif args.stage == 'smoke':
        smoke(config)
    elif args.stage == 'one':
        run_one(config,args.task_index)
    elif args.stage == 'full':
        full(config,Path(args.config).resolve())
    else:
        collect(config)
        if args.stage == 'collect':
            return
        from .report import make_report
        make_report(config)


if __name__ == '__main__':
    main()
