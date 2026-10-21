"""Training-only v5.2 dual selector; the official solver remains read-only."""
from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from scripts.taskcard7.lemon import array_sha, model_record, observe_fit, require
from research.taskcard10.experiment import environment, file_sha, read_json, write_json, write_npz
from research.taskcard11.rescore import select_candidates


def root():
    return Path(__file__).resolve().parents[2]


def configure_core():
    """Verify all 32 files and every actual import, including repeated calls."""
    repo = root()
    version = repo / 'doc/algorithm_versions/v5.2-search-final'
    source = (version / 'src').resolve()
    manifest = read_json(version / 'VERSION.json')['source_sha256']
    require(len(manifest) == 32, 'Frozen core file count differs')
    for name, digest in manifest.items():
        path = (source / name).resolve()
        require(path.is_relative_to(source) and file_sha(path) == digest, 'Frozen core hash differs: ' + name)
    loaded = {name: module for name, module in tuple(sys.modules.items())
              if name == 'tensor_abess' or name.startswith('tensor_abess.')}
    for name, module in loaded.items():
        path = Path(getattr(module, '__file__', '')).resolve()
        require(path.is_relative_to(source), 'Actual core import escaped frozen source: ' + name)
        relative = path.relative_to(source).as_posix()
        require(relative in manifest and file_sha(path) == manifest[relative], 'Actual core import hash differs')
    if not loaded:
        sys.path.insert(0, str(source))
    from tensor_abess.tensor.abess import tensor_abess
    require(Path(inspect.getfile(tensor_abess)).resolve() == source / 'tensor_abess/tensor/abess.py',
            'Wrong actual native solver')
    return tensor_abess


def identity():
    configure_core()
    repo = root()
    dependencies = ['research/taskcard12/selection.py', 'research/taskcard11/rescore.py',
                    'research/taskcard10/experiment.py', 'scripts/taskcard7/lemon.py',
                    'doc/algorithm_versions/v5.2-search-final/VERSION.json']
    return {'core_sha256': read_json(repo / dependencies[-1])['source_sha256'],
            'source_sha256': {name: file_sha(repo / name) for name in dependencies},
            'environment': environment()}


def normalized_options(options):
    signature = inspect.signature(configure_core())
    signature.bind(None, None, **options)
    result = {name: parameter.default for name, parameter in signature.parameters.items()
              if name not in ('X', 'y')}
    result.update(options)
    frozen = {'target_sparsity': None, 'adaptive_criterion': 'msic', 'msic_refine_alpha': 1.5,
              'msic_refine_gamma': 1.0, 'msic_refine_one_se_scale': 1.0}
    require(all(result[key] == value for key, value in frozen.items()), 'Frozen global selection options changed')
    return json.loads(json.dumps(result, allow_nan=False))


def training_inputs(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    require(X.ndim == 4 and y.shape == (len(X),) and len(X) > 0 and
            np.isfinite(X).all() and np.isfinite(y).all(), 'Aligned finite training inputs required')
    return X, y


def pool_digest(metadata, coefficients):
    body = {'metadata': metadata, 'coefficients_sha256': array_sha(coefficients)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def audit_candidates(X, y, trace, coefficients):
    """Verify the native minimum-norm OLS, including zero RSS and rank loss."""
    rows = trace['candidates']
    grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in trace['caps']])
            if np.prod(s) <= max(1, np.floor(.7 * len(y)))]
    require(len(rows) == len(coefficients) == len(grid) + 1 and
            trace['ordered_size_grid'] == [r['sizes'] for r in rows[1:]] == grid and
            len(trace['fixed_searches']) == len(grid), 'Incomplete native candidate capture/grid')
    choices = select_candidates([dict(row, B=B) for row, B in zip(rows, coefficients)], len(y), X.shape[1:])
    require(choices['candidate_scores'] == [r['msic'] for r in rows], 'Native candidate score differs')
    ranks = []
    for mode in range(3):
        slices = np.moveaxis(X, mode + 1, 1)
        scores = [np.linalg.norm(slices[:, k].reshape(len(y), -1).T @ y) ** 2 / len(y)
                  for k in range(slices.shape[1])]
        ranks.append(np.argsort(-np.asarray(scores)))
    checks = []
    for row, B, path in zip(rows, coefficients, [{}, *trace['fixed_searches']]):
        require(array_sha(B) == row['B_sha256'], 'Candidate coefficient hash differs')
        support = row['support']
        Z = np.column_stack([X[:, i, j, k] for i, j, k in itertools.product(*support)])
        beta, _, rank, singular = np.linalg.lstsq(Z, y, rcond=None)
        expected = np.zeros(B.shape)
        expected[np.ix_(*support)] = beta.reshape(tuple(row['sizes']))
        np.testing.assert_allclose(B, expected, rtol=1e-8, atol=1e-9)
        residual = y - Z @ B[np.ix_(*support)].ravel()
        rss = float(residual @ residual)
        np.testing.assert_allclose(rss, row['rss'], rtol=1e-9, atol=1e-9)
        gradient = float(np.linalg.norm(Z.T @ residual) / max(1., np.linalg.norm(Z) * np.linalg.norm(y)))
        require(np.isfinite(rss) and np.isfinite(gradient) and gradient < 1e-8,
                'Invalid OLS residual/normal equations')
        if path:
            initial = [np.sort(order[:size]).tolist() for order, size in zip(ranks, row['sizes'])]
            require(path['initial_support'] == initial, 'Native marginal start differs')
            history = path['visited_loss_history']
            require(bool(history) and np.isfinite(history).all(), 'Invalid native visited losses')
            np.testing.assert_allclose(min(history), row['rss'] / (2 * len(y)), rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(history[path['best_iteration']], min(history), rtol=1e-9, atol=1e-10)
            require(path['stop_reason'] in ('max_cycles', 'unchanged_support', 'exploration_cycle'),
                    'Invalid native stopping evidence')
        deficient = int(rank) < row['q']
        condition = None if deficient else float(singular[0] / singular[-1])
        condition = condition if condition is None or np.isfinite(condition) else None
        checks.append({'passed': True, 'q': row['q'], 'rank': int(rank), 'rank_deficient': deficient,
                       'minimum_norm_OLS_verified': True, 'condition_number': condition,
                       'condition_number_status': 'rank_deficient' if deficient else ('finite' if condition is not None else 'nonfinite'),
                       'rss': rss, 'rss_absolute_error': abs(rss - row['rss']),
                       'coefficient_max_absolute_error': float(np.max(np.abs(B - expected))),
                       'normalized_OLS_gradient': gradient})
    return {'passed': True, 'audit_search_fits': 0, 'candidate_count': len(checks),
            'all_candidate_designs_full_rank': all(not c['rank_deficient'] for c in checks),
            'all_candidates_minimum_norm_OLS_verified': True,
            'maximum_OLS_coefficient_absolute_error': max(c['coefficient_max_absolute_error'] for c in checks),
            'maximum_RSS_absolute_error': max(c['rss_absolute_error'] for c in checks),
            'maximum_normalized_OLS_gradient': max(c['normalized_OLS_gradient'] for c in checks),
            'maximum_candidate_condition_number': None if any(c['condition_number'] is None for c in checks)
                else max(c['condition_number'] for c in checks),
            'checks': checks, 'scope': 'Every original native candidate independently verified against minimum-norm OLS; no search/test access or coefficient/RSS replacement'}


def _pack(X, y, trace, coefficients, options, audit, counters, timings):
    metadata = {'schema_version': 1, 'version': 'v5.2-fs-research', 'research_only': True,
                'n': len(y), 'dimensions': list(X.shape[1:]),
                'input_identity': {'X': array_sha(X), 'y': array_sha(y)},
                'native_options': options, 'identity': identity(), 'trace': copy.deepcopy(trace),
                'audit': copy.deepcopy(audit), 'counters': counters, 'timings': timings}
    coefficients = np.stack(coefficients).copy()
    pool = {'metadata': metadata, 'coefficients': coefficients,
            'digest': pool_digest(metadata, coefficients)}
    _validate_pool(pool, runtime=True)
    return pool


def _validate_pool(pool, *, runtime):
    metadata, coefficients = pool['metadata'], np.asarray(pool['coefficients'])
    require(pool['digest'] == pool_digest(metadata, coefficients), 'Pool digest/hash differs')
    require(metadata['schema_version'] == 1 and metadata['version'] == 'v5.2-fs-research' and
            metadata['research_only'] is True, 'Pool schema/version differs')
    if runtime:
        require(metadata['identity'] == identity(), 'Current source/environment identity differs')
        require(metadata['native_options'] == normalized_options(metadata['native_options']), 'Native options differ')
    n, dims, trace = metadata['n'], metadata['dimensions'], metadata['trace']
    rows = trace['candidates']
    require(coefficients.shape == (len(rows), *dims), 'Pool coefficient shape differs')
    grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in trace['caps']])
            if np.prod(s) <= max(1, np.floor(.7 * n))]
    require(trace['ordered_size_grid'] == grid and [r['sizes'] for r in rows[1:]] == grid and
            len(trace['fixed_searches']) == len(grid), 'Incomplete candidate order/grid')
    require(metadata['audit']['passed'] is True and metadata['audit']['candidate_count'] == len(rows),
            'Missing full candidate audit')
    enriched = []
    for row, B in zip(rows, coefficients):
        require(row['B_sha256'] == array_sha(B), 'Candidate coefficient hash differs')
        require(row['q'] > 0 and row['rss'] >= 0, 'Invalid candidate q/RSS')
        enriched.append(dict(row, B=B))
    original = select_candidates(enriched, n, dims)
    require(original['candidate_scores'] == [r['msic'] for r in rows], 'Native candidate score differs')
    require(original['tolerance'] == trace['one_se_tolerance'] and
            original['eligible_indices'] == trace['eligible_candidate_indices'] and
            original['minimum_score'] == trace['minimum_msic'] and
            original['minimum_q'] == trace['minimum_msic_q'], 'Native original band differs')
    require(rows[original['selected_index']] == trace['final'], 'Native original selected model differs')
    return original


def select_from_pool(pool, *, selection_rule='original'):
    require(selection_rule in ('original', 'fs'), 'Unknown selection_rule')
    original = _validate_pool(pool, runtime=True)
    metadata = pool['metadata']
    choice = original if selection_rule == 'original' else select_candidates(
        metadata['trace']['candidates'], metadata['n'], metadata['dimensions'], finite_sample=True)
    index = choice['selected_index']
    row = metadata['trace']['candidates'][index]
    return {**copy.deepcopy(row), 'B': pool['coefficients'][index].copy(),
            'score': choice['selected_score'], 'selection_rule': selection_rule,
            'selection': choice, 'full_candidate_index': index}


def fit_pair(X, y, **native_options):
    """Exactly one complete native Cold search, then original and fixed FS."""
    X, y = training_inputs(X, y)
    options = normalized_options(native_options)
    native = configure_core()
    before = identity()
    started = time.perf_counter()
    fitted, trace, coefficients = observe_fit(lambda: native(X, y, **options))
    native_seconds = time.perf_counter() - started
    started = time.perf_counter()
    audit = audit_candidates(X, y, trace, coefficients)
    audit_seconds = time.perf_counter() - started
    require(before == identity(), 'Source/environment changed during fit')
    require(trace['final'] == model_record(fitted), 'Original native B/support/RSS/score differs')
    pool = _pack(X, y, trace, coefficients, options, audit,
                 {'native_search_calls': 1, 'extra_search_calls': 0, 'native_adaptive_calls': 1,
                  'native_fixed_computed': len(trace['ordered_size_grid'])},
                 {'native_seconds': native_seconds, 'independent_audit_seconds': audit_seconds})
    original = select_from_pool(pool)
    require(np.array_equal(original['B'], fitted.B), 'Original selected coefficients differ from native')
    return {'pool': pool, 'original': original, 'fs': select_from_pool(pool, selection_rule='fs')}


def fit(X, y, *, selection_rule='original', **native_options):
    require(selection_rule in ('original', 'fs'), 'Unknown selection_rule')
    return fit_pair(X, y, **native_options)[selection_rule]


def pool_from_saved(X, y, trace, coefficients, **native_options):
    """Audit an existing full native pool without any search or new candidates."""
    X, y = training_inputs(X, y)
    options = normalized_options(native_options)
    audit = audit_candidates(X, y, trace, coefficients)
    return _pack(X, y, trace, coefficients, options, audit,
                 {'native_search_calls': 0, 'extra_search_calls': 0, 'saved_pool_import': True}, {})


def save_pool(path, pool):
    _validate_pool(pool, runtime=True)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'metadata.json', pool['metadata'])
    write_npz(path / 'coefficients.npz', {'full_B': pool['coefficients']})
    write_json(path / 'seal.json', {'schema_version': 1, 'pool_digest': pool['digest'],
               'files': {name: file_sha(path / name) for name in ('metadata.json', 'coefficients.npz')}})


def verify_pool_files(path):
    """Verify stored identities only; this is not permission to reuse a cache."""
    path = Path(path)
    seal = read_json(path / 'seal.json')
    require(seal['schema_version'] == 1 and set(seal['files']) == {'metadata.json', 'coefficients.npz'},
            'Cache seal schema/file coverage differs')
    for name, digest in seal['files'].items():
        require(file_sha(path / name) == digest, 'Cache file hash differs: ' + name)
    with np.load(path / 'coefficients.npz', allow_pickle=False) as arrays:
        require(arrays.files == ['full_B'], 'Cache array keys differ')
        coefficients = arrays['full_B'].copy()
    pool = {'metadata': read_json(path / 'metadata.json'), 'coefficients': coefficients,
            'digest': seal['pool_digest']}
    _validate_pool(pool, runtime=False)
    return pool


def load_pool(path, *, X, y, **native_options):
    """Strict same-input/config/source/environment cache load; zero searches."""
    pool = verify_pool_files(path)
    _validate_pool(pool, runtime=True)
    X, y = training_inputs(X, y)
    metadata = pool['metadata']
    require(metadata['input_identity'] == {'X': array_sha(X), 'y': array_sha(y)}, 'Training identity differs')
    require(metadata['native_options'] == normalized_options(native_options), 'Native options differ')
    audit_candidates(X, y, metadata['trace'], pool['coefficients'])
    return pool
