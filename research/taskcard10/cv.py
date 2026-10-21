"""Isolated, training-only size-program selection on the frozen v5.2 solver.

No truth or test interface is provided. The native adaptive source is retained
only in A0; A1/A2 compare exactly the same fixed-size programs and coefficients.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import monotonic, perf_counter, process_time

import numpy as np

from scripts.taskcard7.lemon import array_sha, observe_fit, require


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _check_budget(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError('Frozen selector compute deadline exhausted')


def _floor_caps(n, dims):
    floor = max(1, int(np.floor(np.sqrt(n) / 2.)))
    Q = max(1, int(np.floor(.7 * n)))
    require(min(dims) >= floor and floor ** 3 > Q,
            'Response-independent formal caps are not proved for this n/dimension regime')
    caps = [min(int(d), floor) for d in dims]
    while int(np.prod(caps)) > Q:
        caps[int(np.argmax(caps))] -= 1
    return caps


def capacity_protocol(n, dims, n_folds=3):
    require(n_folds == 3, 'Only predeclared three-fold protocol is implemented')
    valid_sizes = [len(part) for part in np.array_split(np.arange(n), n_folds)]
    inner_n = [n - size for size in valid_sizes]
    full_caps = _floor_caps(n, dims)
    inner_caps = [_floor_caps(size, dims) for size in inner_n]
    common_caps = np.min(np.asarray([full_caps, *inner_caps]), axis=0).astype(int).tolist()
    Q = min(max(1, int(np.floor(.7 * size))) for size in [n, *inner_n])
    grid = [[a, b, c] for a in range(1, common_caps[0] + 1)
            for b in range(1, common_caps[1] + 1)
            for c in range(1, common_caps[2] + 1) if a * b * c <= Q]
    return {'n_full': int(n), 'inner_n': inner_n, 'full_caps': full_caps,
            'inner_caps': inner_caps, 'common_caps': common_caps, 'common_Q': Q,
            'common_grid': grid,
            'rule': 'formal largest-axis shrink from response-independent marginal floor; fixed programs only'}


def make_folds(n, seed):
    valid = np.array_split(np.random.default_rng(int(seed)).permutation(n), 3)
    universe = np.arange(n)
    return [(np.setdiff1d(universe, part).astype(int).tolist(),
             np.sort(part).astype(int).tolist()) for part in valid]


def fit_preprocessor(X, y, kind, train_indices):
    require(kind in {'identity', 'center'}, 'Unknown preprocessor')
    require(len(X) == len(y) == len(train_indices), 'Preprocessor training identity differs')
    means = {'identity': lambda: (np.zeros(X.shape[1:]), 0.),
             'center': lambda: (np.mean(X, axis=0), float(np.mean(y)))}
    X_mean, y_mean = means[kind]()
    return {'kind': kind, 'train_indices': list(map(int, train_indices)),
            'X_mean': X_mean, 'y_mean': y_mean}


def predict(preprocessor, B, X):
    Xp = np.asarray(X, dtype=float) - preprocessor['X_mean']
    return Xp.reshape(len(Xp), -1) @ np.asarray(B).ravel() + preprocessor['y_mean']


def predict_pool(preprocessor, Bs, X):
    from scipy.sparse import csr_matrix
    Xp = np.asarray(X, dtype=float) - preprocessor['X_mean']
    sparse_B = csr_matrix(np.asarray(Bs).reshape(len(Bs), -1).T)
    return np.asarray(Xp.reshape(len(Xp), -1) @ sparse_B) + preprocessor['y_mean']


def _preprocessing_record(p):
    return {'kind': p['kind'], 'train_indices': p['train_indices'],
            'X_mean_sha256': array_sha(p['X_mean']), 'y_mean': float(p['y_mean'])}


def native_options(config):
    return dict(k_max=config.get('k_max', 1), c=config.get('abess_init_c', 3.),
                criterion=config.get('abess_init_criterion', 'bic'),
                n_lambdas=config.get('n_lambdas', 20), cv_folds=config.get('cv_folds', 5),
                max_iter=config.get('max_iter', 500), tol=config.get('tol', 1e-6),
                max_cycles=config.get('max_cycles', 6), eps=1e-8,
                adaptive_criterion='msic', msic_refine_alpha=1.5,
                msic_refine_gamma=1., msic_refine_one_se_scale=1.)


def cache_identity(*, data_identity, train_indices, validation_indices, grid,
                   config, source_identity, preprocessing, seed, program):
    payload = {'schema_version': 1, 'data_identity': data_identity,
               'train_indices': list(map(int, train_indices)),
               'validation_indices': list(map(int, validation_indices)),
               'grid': grid, 'config': config, 'source_identity': source_identity,
               'preprocessing': preprocessing, 'seed': int(seed), 'program': program,
               'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return _digest(payload), payload


def fit_fixed_candidate(X, y, sizes, config):
    from tensor_abess.tensor.bcd import _marginal_supports_for_size, tensor_bcd
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
    sizes = tuple(map(int, sizes))
    require(int(np.prod(sizes)) <= int(np.floor(.7 * len(y))), 'Fixed candidate exceeds q capacity')
    initial = _marginal_supports_for_size(X, y, sizes)
    started = perf_counter()
    result = tensor_bcd(X, y, *sizes, max_cycles=config.get('max_cycles', 6),
                       m_max=15, k_max=config.get('k_max', 1),
                       S1_init=initial[0], S2_init=initial[1], S3_init=initial[2])
    elapsed = perf_counter() - started
    support = [getattr(result, 'S' + str(k)).astype(int).tolist() for k in [1, 2, 3]]
    rss = float(2 * len(y) * result.loss)
    score = compute_tensor_msic_from_rss(rss, *sizes, len(y), *X.shape[1:], alpha=1.5, gamma=1.)
    record = {'sizes': list(sizes), 'q': int(np.prod(sizes)), 'support': support,
              'rss': rss, 'msic': float(score), 'B_sha256': array_sha(result.B),
              'initial_support': [part.astype(int).tolist() for part in initial],
              'loss_history': list(map(float, result.loss_history)),
              'visited_loss_history': list(map(float, result.visited_loss_history)),
              'best_iteration': int(result.best_iteration), 'stop_reason': result.stop_reason,
              'n_cycles': int(result.n_cycles), 'fit_seconds': elapsed}
    _validate_candidate(X, y, record, result.B)
    return record, result.B.copy()


def _validate_candidate(X, y, record, B):
    require(np.isfinite(B).all() and array_sha(B) == record['B_sha256'], 'Coefficient identity differs')
    support = [np.asarray(s, dtype=int) for s in record['support']]
    require([len(s) for s in support] == record['sizes'], 'Returned candidate shape differs')
    for s, d in zip(support, X.shape[1:]):
        require(len(np.unique(s)) == len(s) and np.all((s >= 0) & (s < d)), 'Invalid support')
    Z = X[:, support[0], :, :][:, :, support[1], :][:, :, :, support[2]].reshape(len(y), -1)
    beta = B[np.ix_(*support)].ravel()
    require(np.count_nonzero(B) == np.count_nonzero(beta), 'Coefficient outside support')
    residual = y - Z @ beta
    rss = float(residual @ residual)
    require(np.isclose(rss, record['rss'], rtol=1e-10, atol=1e-9), 'Saved candidate RSS differs')
    bound = 1e-7 * max(1., np.linalg.norm(Z) * np.linalg.norm(y))
    require(np.linalg.norm(Z.T @ residual) <= bound, 'Saved coefficient is not a support OLS solution')
    history = record.get('visited_loss_history', [])
    require(not history or np.isclose(rss / (2 * len(y)), min(history), rtol=1e-10, atol=1e-10),
            'Fixed solver did not retain best visited OLS')


def fit_fold_pool(X, y, *, train_indices, validation_indices, grid, config,
                  source_identity, seed, preproc, cache_dir, deadline_monotonic=None):
    from tensor_abess.utils.init import tensor_init_glasso
    _check_budget(deadline_monotonic)
    p = fit_preprocessor(X, y, preproc, train_indices)
    Xp, yp = X - p['X_mean'], y - p['y_mean']
    key, payload = cache_identity(data_identity={'X': array_sha(X), 'y': array_sha(y)},
            train_indices=train_indices, validation_indices=validation_indices, grid=grid,
            config=config, source_identity=source_identity, preprocessing=_preprocessing_record(p),
            seed=seed, program='v5.2-formal-global-fixed-marginal-init')
    folder = Path(cache_dir) / key
    if folder.exists():
        metadata_path, arrays_path = folder / 'pool.json', folder / 'pool.npz'
        seal = json.loads((folder / 'seal.json').read_text(encoding='utf-8'))
        require(seal == {'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                         'arrays_sha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest()},
                'Cached pool seal differs')
        record = json.loads(metadata_path.read_text(encoding='utf-8'))
        require(record['cache_payload'] == payload, 'Cached training identity differs')
        with np.load(arrays_path, allow_pickle=False) as arrays:
            Bs = arrays['B'].copy()
        for candidate, B in zip(record['candidates'], Bs):
            _validate_candidate(Xp, yp, candidate, B)
        require(len(Bs) == len(grid) == len(record['candidates']), 'Incomplete cached pool')
        return dict(record, B=Bs, preprocessor=p, cache_hit=True)
    options = native_options(config)
    init_started = perf_counter()
    init = tensor_init_glasso(Xp, yp, c=options['c'], criterion=options['criterion'],
                n_lambdas=options['n_lambdas'], cv_folds=options['cv_folds'],
                max_iter=options['max_iter'], tol=options['tol'])
    init_seconds = perf_counter() - init_started
    caps = [int(init.s1_max), int(init.s2_max), int(init.s3_max)]
    require(caps == _floor_caps(len(y), X.shape[1:]), 'Actual fold caps differ from frozen capacity proof')
    require(all(all(s <= cap for s, cap in zip(shape, caps)) for shape in grid),
            'Public grid is not legal for this fold')
    records, Bs = [], []
    for sizes in grid:
        _check_budget(deadline_monotonic)
        candidate, B = fit_fixed_candidate(Xp, yp, sizes, config)
        records.append(candidate)
        Bs.append(B)
    record = {'cache_key': key, 'cache_payload': payload, 'candidates': records,
              'init_caps': caps, 'init_supports': [getattr(init, 'S' + str(i) + '_init').astype(int).tolist()
                                                 for i in [1, 2, 3]],
              'initialization_seconds': init_seconds, 'preprocessing': _preprocessing_record(p)}
    folder.mkdir(parents=True, exist_ok=False)
    metadata_path, arrays_path = folder / 'pool.json', folder / 'pool.npz'
    metadata_path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    Bs = np.stack(Bs)
    np.savez_compressed(arrays_path, B=Bs)
    (folder / 'seal.json').write_text(json.dumps({
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'arrays_sha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest()}, indent=2) + '\n', encoding='utf-8')
    return dict(record, B=Bs, preprocessor=p, cache_hit=False)


def select_cv(rows):
    scores = [sum(row['inner_sse']) / sum(row['inner_n']) for row in rows]
    minimum = min(scores)
    eligible = [row for row, score in zip(rows, scores) if abs(score - minimum) <= 1e-12]
    return min(eligible, key=lambda row: (row['q'], tuple(row['sizes'])))


def select_msic(rows, n):
    best = min(rows, key=lambda row: row['msic'])
    tolerance = float(np.sqrt(max(best['q'], 1) * max(np.log(max(n, 2)), 1.)))
    eligible = [row for row in rows if row['msic'] <= best['msic'] + tolerance]
    return min(eligible, key=lambda row: (row['q'], row['msic'], sum(row['sizes'])))


def run_three_arms(X, y, config_dict, split_seed, cache_dir, source_identity,
                   preproc='identity', *, deadline_monotonic=None):
    from tensor_abess.tensor.abess import tensor_abess
    _check_budget(deadline_monotonic)
    started, cpu_started = perf_counter(), process_time()
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    require(X.ndim == 4 and len(X) == len(y) and np.isfinite(X).all() and np.isfinite(y).all(),
            'Selector requires aligned finite training input')
    capacity = capacity_protocol(len(y), X.shape[1:])
    common_grid = capacity['common_grid']
    full_p = fit_preprocessor(X, y, preproc, list(range(len(y))))
    Xp, yp = X - full_p['X_mean'], y - full_p['y_mean']
    native_started = perf_counter()
    fitted, trace, full_B = observe_fit(lambda: tensor_abess(Xp, yp, **native_options(config_dict)))
    native_seconds = perf_counter() - native_started
    require(trace['caps'] == capacity['full_caps'], 'Actual full caps differ from capacity proof')
    mapping = {tuple(row['sizes']): (i, row) for i, row in enumerate(trace['candidates'][1:], start=1)}
    require(set(map(tuple, common_grid)).issubset(mapping), 'Common grid absent from actual full native pool')
    arrays = {'full_B': np.stack(full_B), 'full_preprocessing_X_mean': full_p['X_mean']}
    full_p_record = dict(_preprocessing_record(full_p), X_mean_key='full_preprocessing_X_mean')
    native_index = next(i for i, row in enumerate(trace['candidates'])
                        if row['B_sha256'] == array_sha(fitted.B)
                        and row['support'] == trace['final']['support'])
    folds, inner_sse = [], [[] for _ in common_grid]
    inner_started = perf_counter()
    computed, hits, init_calls = 0, 0, 0
    for fold_id, (train, valid) in enumerate(make_folds(len(y), split_seed)):
        _check_budget(deadline_monotonic)
        pool_started = perf_counter()
        pool = fit_fold_pool(X[train], y[train], train_indices=train, validation_indices=valid,
                    grid=common_grid, config=config_dict, source_identity=source_identity,
                    seed=split_seed, preproc=preproc, cache_dir=cache_dir,
                    deadline_monotonic=deadline_monotonic)
        pool_seconds = perf_counter() - pool_started
        prefix = 'inner' + str(fold_id)
        predictions = predict_pool(pool['preprocessor'], pool['B'], X[valid])
        residual = y[valid, None] - predictions
        sses = np.sum(residual ** 2, axis=0)
        for index, sse in enumerate(sses):
            inner_sse[index].append(float(sse))
        arrays[prefix + '_B'] = pool['B']
        arrays[prefix + '_prediction'] = predictions
        arrays[prefix + '_y_validation'] = y[valid].copy()
        arrays[prefix + '_preprocessing_X_mean'] = pool['preprocessor']['X_mean']
        folds.append({key: pool[key] for key in ['cache_key', 'cache_payload', 'candidates',
                     'init_caps', 'init_supports', 'initialization_seconds', 'preprocessing', 'cache_hit']})
        folds[-1].update(fold_id=fold_id, train_indices=train, validation_indices=valid,
            n_validation=len(valid), B_key=prefix + '_B', prediction_key=prefix + '_prediction',
            y_validation_key=prefix + '_y_validation', pool_seconds=pool_seconds,
            validation_sse=sses.astype(float).tolist())
        folds[-1]['preprocessing']['X_mean_key'] = prefix + '_preprocessing_X_mean'
        computed += 0 if pool['cache_hit'] else len(common_grid)
        hits += len(common_grid) if pool['cache_hit'] else 0
        init_calls += int(not pool['cache_hit'])
    inner_seconds = perf_counter() - inner_started
    curve = []
    for sizes, sses in zip(common_grid, inner_sse):
        index, row = mapping[tuple(sizes)]
        ns = [fold['n_validation'] for fold in folds]
        curve.append(dict(row, full_candidate_index=index, inner_sse=sses,
                          inner_n=ns, cv_mse=sum(sses) / sum(ns)))
    selection_started = perf_counter()
    selected_msic = dict(select_msic(curve, len(y)))
    A1_selection_seconds = perf_counter() - selection_started
    selection_started = perf_counter()
    selected_cv = dict(select_cv(curve))
    A2_selection_seconds = perf_counter() - selection_started
    chosen = {'A0': dict(trace['final'], full_candidate_index=native_index),
              'A1': selected_msic, 'A2': selected_cv}
    for arm, row in chosen.items():
        B = arrays['full_B'][row['full_candidate_index']]
        require(array_sha(B) == row['B_sha256'], 'Final full-training coefficient identity differs')
        arrays[arm + '_B'] = B.copy()
        row['B_key'] = arm + '_B'
    return {'summary': {'schema_version': 1, 'arms': chosen, 'full_trace': trace,
            'public_capacity': capacity, 'common_grid': common_grid,
            'full_preprocessing': full_p_record, 'folds': folds, 'cv_curve': curve,
            'source_identity': source_identity, 'config': config_dict, 'split_seed': int(split_seed),
            'input_identity': {'X': array_sha(X), 'y': array_sha(y)},
            'selection_rules': {'A0': 'unchanged native formal MSIC',
                'A1': 'original MSIC/tolerance on common fixed programs only',
                'A2': 'pooled validation SSE/n; abs1e-12 tie then q, lexicographic size'},
            'interpretation': 'A1-A0 includes capacity restriction and deletion of adaptive source; A2-A1 shares fixed programs',
            'timings': {'native_seconds': native_seconds, 'A0_seconds': native_seconds,
                        'inner_seconds': inner_seconds, 'full_fixed_reuse_seconds': 0.,
                        'A1_selection_seconds': A1_selection_seconds,
                        'A2_selection_seconds': A2_selection_seconds,
                        'total_seconds': perf_counter()-started, 'cpu_seconds': process_time()-cpu_started},
            'counters': {'native_fixed_computed': len(trace['ordered_size_grid']),
                         'native_adaptive_calls': 1, 'inner_fixed_computed': computed,
                         'inner_fixed_cache_hits': hits, 'inner_initialization_calls': init_calls,
                         'full_common_reused': len(common_grid), 'extra_full_fixed_fits': 0,
                         'validation_predictions': int(len(y) * len(common_grid))}},
            'arrays': arrays}
