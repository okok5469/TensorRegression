"""Independent numerical checks of Task10 evidence; never calls a search solver."""
from __future__ import annotations

import itertools
import math
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_saved_fit(X, y, B, supports, expected_rss, prediction_X=None, saved_prediction=None):
    """Reconstruct a saved rectangular OLS fit independently of tensor ops."""
    X, y, B = np.asarray(X), np.asarray(y), np.asarray(B)
    assert X.ndim == 4 and y.shape == (len(X),) and B.shape == X.shape[1:]
    assert np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(B).all()
    S = [np.asarray(s) for s in supports]
    for s, p in zip(S, X.shape[1:]):
        assert s.ndim == 1 and len(s) > 0 and np.issubdtype(s.dtype, np.integer)
        assert len(np.unique(s)) == len(s) and np.all((s >= 0) & (s < p))
        assert np.array_equal(s, np.sort(s))
    q = math.prod(map(len, S))
    Z = np.column_stack([X[:, i, j, k] for i, j, k in itertools.product(*S)])
    beta, _, rank, singular = np.linalg.lstsq(Z, y, rcond=None)
    expected_B = np.zeros(B.shape)
    expected_B[np.ix_(*S)] = beta.reshape(tuple(map(len, S)))
    np.testing.assert_allclose(B, expected_B, rtol=1e-8, atol=1e-9)
    pred = Z @ B[np.ix_(*S)].ravel()
    residual = y - pred
    rss = float(residual @ residual)
    np.testing.assert_allclose(rss, expected_rss, rtol=1e-9, atol=1e-9)
    gradient = float(np.linalg.norm(Z.T @ residual) /
                     max(1., np.linalg.norm(Z) * np.linalg.norm(y)))
    assert gradient < 1e-8
    prediction_error = None
    if prediction_X is not None:
        XP = np.asarray(prediction_X)
        prediction = np.column_stack([XP[:, i, j, k] for i, j, k in itertools.product(*S)]) @ beta
        np.testing.assert_allclose(prediction, saved_prediction, rtol=1e-8, atol=1e-9)
        prediction_error = float(np.max(np.abs(prediction - saved_prediction)))
    return {
        "passed": True, "q": q, "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "rss": rss, "rss_absolute_error": abs(rss - expected_rss),
        "coefficient_max_absolute_error": float(np.max(np.abs(B - expected_B))),
        "normalized_OLS_gradient": gradient,
        "prediction_max_absolute_error": prediction_error,
    }


def weighted_cv_choice(sse_matrix, fold_sizes, grid, tie_atol=1e-12):
    """Aggregate independent SSE and choose smaller q then lexicographic shape."""
    sse, sizes = np.asarray(sse_matrix), np.asarray(fold_sizes)
    assert sse.shape == (len(sizes), len(grid))
    assert np.isfinite(sse).all() and np.all(sse >= 0) and np.all(sizes > 0)
    curve = np.sum(sse, axis=0) / float(np.sum(sizes))
    eligible = np.flatnonzero(np.abs(curve - float(np.min(curve))) <= tie_atol)
    chosen = min(eligible, key=lambda i: (math.prod(grid[i]), tuple(grid[i])))
    return curve, int(chosen)


def array_identity(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def _json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _caps(n, dimensions):
    m, budget = max(1, math.floor(math.sqrt(n) / 2)), max(1, math.floor(.7 * n))
    assert min(dimensions) >= m and m ** 3 > budget
    caps = [m, m, m]
    while math.prod(caps) > budget:
        caps[max(range(3), key=caps.__getitem__)] -= 1
    return caps


def _score(rss, shape, n, dimensions):
    loglog = math.log(max(math.log(n), math.e))
    logcount = sum(math.lgamma(p + 1) - math.lgamma(s + 1) - math.lgamma(p - s + 1)
                   for p, s in zip(dimensions, shape))
    return n * math.log(max(rss / n, np.finfo(float).tiny)) + loglog * (1.5 * math.prod(shape) + logcount)


def _msic_index(rows, n):
    minimum_index = min(range(len(rows)), key=lambda i: rows[i]['msic'])
    minimum = rows[minimum_index]['msic']
    tolerance = math.sqrt(max(rows[minimum_index]['q'], 1) * max(math.log(max(n, 2)), 1.))
    eligible = [i for i, r in enumerate(rows) if r['msic'] <= minimum + tolerance]
    chosen = min(eligible, key=lambda i: (rows[i]['q'], rows[i]['msic'], sum(rows[i]['sizes'])))
    return chosen, tolerance, eligible


def _preprocessor(X, y, indices, record, arrays):
    assert record['train_indices'] == indices
    assert record['kind'] in ('identity', 'center')
    mean = np.zeros(X.shape[1:]) if record['kind'] == 'identity' else X.mean(axis=0)
    ymean = 0. if record['kind'] == 'identity' else float(y.mean())
    np.testing.assert_array_equal(arrays[record['X_mean_key']], mean)
    assert record['X_mean_sha256'] == array_identity(mean)
    assert record['y_mean'] == ymean
    return X - mean, y - ymean, mean, ymean


def _check_rows(X, y, records, Bs, prediction_X=None, predictions=None, extra_paths=None):
    assert len(records) == len(Bs)
    checks = []
    # Compute training-only marginal ranks once per data identity.
    ranks = []
    for mode in range(3):
        slices = np.moveaxis(X, mode + 1, 1)
        scores = [np.linalg.norm(slices[:, k].reshape(len(y), -1).T @ y) ** 2 / len(y)
                  for k in range(slices.shape[1])]
        ranks.append(np.argsort(-np.asarray(scores)))
    for i, (row, B) in enumerate(zip(records, Bs)):
        assert array_identity(B) == row['B_sha256']
        assert list(map(len, row['support'])) == row['sizes']
        assert row['q'] == math.prod(row['sizes'])
        # The adaptive terminal may exceed its fixed-grid caps; fixed paths cannot.
        check = audit_saved_fit(X, y, B, row['support'], row['rss'], prediction_X,
                                None if predictions is None else predictions[:, i])
        score = _score(row['rss'], row['sizes'], len(y), X.shape[1:])
        np.testing.assert_allclose(score, row['msic'], rtol=1e-10, atol=1e-9)
        path = row if extra_paths is None else extra_paths[i]
        if 'initial_support' in path:
            expected = [np.sort(rank[:s]).tolist() for rank, s in zip(ranks, row['sizes'])]
            assert path['initial_support'] == expected
            history = path['visited_loss_history']
            assert len(history) > 0 and np.isfinite(history).all()
            np.testing.assert_allclose(min(history), row['rss'] / (2 * len(y)), rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(history[path['best_iteration']], min(history), rtol=1e-9, atol=1e-10)
            assert path['stop_reason'] in ('max_cycles', 'unchanged_support', 'exploration_cycle')
        checks.append(check)
    return checks


def audit_run(X, y, summary, arrays, *, repository=None, cache_dir=None):
    """Audit all saved full/inner models and decisions; no BCD or test interface.

    ``X/y`` must be the exact outer-training input, never the test stream.
    Supplying ``cache_dir`` also checks on-disk seals and recovered coefficients.
    The caller preserves this return value as a separate evidence artifact.
    """
    X, y = np.asarray(X), np.asarray(y)
    n, dimensions = len(y), X.shape[1:]
    assert summary['input_identity'] == {'X': array_identity(X), 'y': array_identity(y)}
    if repository is not None:
        for relative, expected in summary['source_identity'].items():
            assert hashlib.sha256((Path(repository) / relative).read_bytes()).hexdigest() == expected
    capacity = summary['public_capacity']
    split = np.array_split(np.random.default_rng(summary['split_seed']).permutation(n), 3)
    expected_folds = [(np.setdiff1d(np.arange(n), v).tolist(), np.sort(v).tolist()) for v in split]
    full_caps, inner_caps = _caps(n, dimensions), [_caps(len(t), dimensions) for t, _ in expected_folds]
    common_caps = [min(x) for x in zip(full_caps, *inner_caps)]
    budget = min(math.floor(.7 * k) for k in [n, *map(lambda pair: len(pair[0]), expected_folds)])
    grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in common_caps])
            if math.prod(s) <= budget]
    assert summary['common_grid'] == grid == capacity['common_grid']
    assert capacity['full_caps'] == full_caps and capacity['inner_caps'] == inner_caps
    assert capacity['common_caps'] == common_caps and capacity['common_Q'] == budget
    Xf, yf, _, _ = _preprocessor(X, y, list(range(n)), summary['full_preprocessing'], arrays)
    trace, full_B = summary['full_trace'], arrays['full_B']
    assert trace['caps'] == full_caps
    full_grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in full_caps])
                 if math.prod(s) <= math.floor(.7 * n)]
    assert trace['ordered_size_grid'] == full_grid
    assert [r['sizes'] for r in trace['candidates'][1:]] == full_grid
    full_checks = _check_rows(Xf, yf, trace['candidates'], full_B,
                              extra_paths=[{}, *trace['fixed_searches']])
    selected_native, tolerance, eligible = _msic_index(trace['candidates'], n)
    np.testing.assert_allclose(tolerance, trace['one_se_tolerance'], rtol=1e-12)
    assert eligible == trace['eligible_candidate_indices']
    assert trace['final']['B_sha256'] == trace['candidates'][selected_native]['B_sha256']
    assert trace['final']['support'] == trace['candidates'][selected_native]['support']
    all_validation, keys, fold_checks, sse_rows = [], [], [], []
    impl_path = Path(__file__).with_name('cv.py')
    for fold, (train, valid), actual_caps in zip(summary['folds'], expected_folds, inner_caps):
        assert fold['train_indices'] == train and fold['validation_indices'] == valid
        assert set(train).isdisjoint(valid) and sorted(train + valid) == list(range(n))
        assert fold['n_validation'] == len(valid) and fold['init_caps'] == actual_caps
        assert all(all(s <= c for s, c in zip(shape, actual_caps)) for shape in grid)
        all_validation.extend(valid)
        Xt, yt, mean, ymean = _preprocessor(X[train], y[train], train, fold['preprocessing'], arrays)
        payload = fold['cache_payload']
        assert payload['data_identity'] == {'X': array_identity(X[train]), 'y': array_identity(y[train])}
        assert payload['train_indices'] == train and payload['validation_indices'] == valid
        assert payload['grid'] == grid and payload['config'] == summary['config']
        assert payload['source_identity'] == summary['source_identity']
        assert payload['seed'] == summary['split_seed']
        assert payload['program'] == 'v5.2-formal-global-fixed-marginal-init'
        assert payload['implementation_sha256'] == hashlib.sha256(impl_path.read_bytes()).hexdigest()
        assert payload['preprocessing'] == {k: v for k, v in fold['preprocessing'].items() if k != 'X_mean_key'}
        assert _json_digest(payload) == fold['cache_key']
        keys.append(fold['cache_key'])
        Bs, predictions = arrays[fold['B_key']], arrays[fold['prediction_key']]
        np.testing.assert_array_equal(arrays[fold['y_validation_key']], y[valid])
        assert [r['sizes'] for r in fold['candidates']] == grid
        checks = _check_rows(Xt, yt, fold['candidates'], Bs, X[valid] - mean, predictions - ymean)
        sses = np.sum((y[valid, None] - predictions) ** 2, axis=0)
        np.testing.assert_allclose(sses, fold['validation_sse'], rtol=1e-12, atol=1e-10)
        sse_rows.append(sses)
        if cache_dir is not None:
            folder = Path(cache_dir) / fold['cache_key']
            seal = json.loads((folder / 'seal.json').read_text(encoding='utf-8'))
            for name, field in [('pool.json', 'metadata_sha256'), ('pool.npz', 'arrays_sha256')]:
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == seal[field]
            metadata = json.loads((folder / 'pool.json').read_text(encoding='utf-8'))
            assert metadata['cache_payload'] == payload
            assert metadata['candidates'] == fold['candidates']
            with np.load(folder / 'pool.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['B'], Bs)
        fold_checks.extend(checks)
    assert len(summary['folds']) == 3 and sorted(all_validation) == list(range(n))
    assert len(keys) == len(set(keys)) == 3
    curve, cv_index = weighted_cv_choice(sse_rows, [len(v) for _, v in expected_folds], grid)
    full_by_shape = {tuple(r['sizes']): i for i, r in enumerate(trace['candidates'][1:], 1)}
    full_indices = [full_by_shape[tuple(s)] for s in grid]
    rows = [trace['candidates'][i] for i in full_indices]
    msic_index, _, _ = _msic_index(rows, n)
    assert len(summary['cv_curve']) == len(grid)
    for i, row in enumerate(summary['cv_curve']):
        assert row['sizes'] == grid[i] and row['full_candidate_index'] == full_indices[i]
        np.testing.assert_allclose(row['inner_sse'], np.asarray(sse_rows)[:, i], rtol=1e-12, atol=1e-10)
        assert row['inner_n'] == [len(v) for _, v in expected_folds]
        np.testing.assert_allclose(row['cv_mse'], curve[i], rtol=1e-12, atol=1e-10)
    for arm, index in [('A0', selected_native), ('A1', full_indices[msic_index]), ('A2', full_indices[cv_index])]:
        row = summary['arms'][arm]
        assert row['B_sha256'] == trace['candidates'][index]['B_sha256']
        assert row['support'] == trace['candidates'][index]['support']
        np.testing.assert_array_equal(arrays[row['B_key']], full_B[index])
    checks = full_checks + fold_checks
    return {'passed': True, 'audit_search_fits': 0, 'outer_training_n': n,
            'full_candidates': len(full_checks), 'inner_candidates': len(fold_checks),
            'common_candidates': len(grid), 'folds': 3,
            'distinct_training_cache_keys': len(set(keys)),
            'all_validation_samples_seen_once': True, 'all_candidate_shapes_match_frozen_grid': True,
            'A0_native_trace_selection_reconstructed': True,
            'A1_A2_same_full_candidate_program_and_coefficients': True,
            'CV_weighted_SSE_and_ties_reconstructed': True,
            'cache_recovery_files_checked': cache_dir is not None,
            'maximum_OLS_coefficient_absolute_error': max(c['coefficient_max_absolute_error'] for c in checks),
            'maximum_RSS_absolute_error': max(c['rss_absolute_error'] for c in checks),
            'maximum_normalized_OLS_gradient': max(c['normalized_OLS_gradient'] for c in checks),
            'maximum_validation_prediction_absolute_error': max(c['prediction_max_absolute_error'] for c in fold_checks),
            'minimum_candidate_rank': min(c['rank'] for c in checks),
            'all_candidate_designs_full_rank': all(c['rank'] == c['q'] for c in checks),
            'maximum_candidate_condition_number': max(c['condition_number'] for c in checks),
            'scope': 'All saved training candidates, independent OLS/score/prediction/cache reconstruction; no search refit or test access',
            'same_shape_different_full_support_selector_effect': 'NOT_IDENTIFIABLE'}
