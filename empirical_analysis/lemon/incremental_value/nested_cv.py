"""Shared subject splits and external, fully nested preprocessing."""
from __future__ import annotations
import numpy as np
from .preprocessing import fit_preprocessor, validate_covariates


def make_splits(C, n_folds, seeds):
    C = np.asarray(C, dtype=float)
    if not 2 <= n_folds <= len(C) // 2:
        raise ValueError('Invalid fold count')
    # Age dummy columns define prespecified strata; sex is never used as age.
    age_columns = C[:, 1:-1] if C.shape[1] > 2 else np.zeros((len(C), 1))
    _, strata = np.unique(age_columns, axis=0, return_inverse=True)
    splits = []
    for repeat, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        groups = [[] for _ in range(n_folds)]
        cursor = 0
        for stratum in rng.permutation(np.unique(strata)):
            indices = rng.permutation(np.flatnonzero(strata == stratum))
            for offset, index in enumerate(indices):
                groups[(cursor + offset) % n_folds].append(int(index))
            cursor = (cursor + len(indices)) % n_folds
        for fold, indices in enumerate(groups):
            test = np.asarray(sorted(indices), dtype=int)
            train = np.setdiff1d(np.arange(len(C)), test)
            validate_covariates(C[train], len(train), require_full_rank=True)
            splits.append({'repeat': repeat, 'outer_fold': fold + 1,
                           'seed': int(seed), 'train': train, 'test': test})
    return splits


def prepare_inner_folds(X, y, C, n_folds, seed):
    prepared = []
    for split in make_splits(C, n_folds, [seed]):
        train, valid = split['train'], split['test']
        p = fit_preprocessor(X[train], y[train], C[train])
        prepared.append({'train': train, 'validation': valid,
                         'preprocessor': p, 'X': p.apply(X[train], C[train]),
                         'y': p.transform_response(y[train], C[train]),
                         'X_validation': X[valid], 'C_validation': C[valid],
                         'y_validation': y[valid]})
    return prepared


def select_parameters(name, X, y, C, candidates, n_folds, seed, options,
                      solver=None, prepared=None):
    path_solver = None
    if solver is None:
        from .model_adapters import fit_eeg_path
        path_solver = fit_eeg_path
    prepared = prepare_inner_folds(X, y, C, n_folds, seed) if prepared is None else prepared
    totals = np.zeros(len(candidates))
    counts = np.zeros(len(candidates))
    rows, traces = [], []
    for fold, data in enumerate(prepared, start=1):
        traces.append({'inner_fold': fold, 'train': data['train'].tolist(),
                       'validation': data['validation'].tolist()})
        results = (path_solver(name, data['X'], data['y'], candidates, options)
                   if path_solver else [solver(name, data['X'], data['y'], parameter, options)
                                        for parameter in candidates])
        if len(results) != len(candidates):
            raise ValueError('Model path did not return every prespecified candidate')
        for index, (parameter, result) in enumerate(zip(candidates, results)):
            prediction = data['preprocessor'].predict(result.B, data['X_validation'],
                                                       data['C_validation'])
            error = data['y_validation'] - prediction
            sse = float(error @ error)
            totals[index] += sse
            counts[index] += len(error)
            rows.append({'model': name, 'inner_fold': fold, 'candidate_index': index,
                         'parameters': parameter, 'mse': sse / len(error),
                         'n_train': len(data['train']), 'n_validation': len(error)})
    scores = totals / counts
    best_score = float(scores.min())
    tied = np.flatnonzero(np.isclose(scores, best_score, atol=1e-12, rtol=1e-12))
    # Prespecified tie-break: choose the largest actual penalty/ratio.
    strength = lambda index: float(next(iter(candidates[int(index)].values())))
    best = max(tied, key=strength)
    return candidates[int(best)], rows, traces
