"""Bounded independent replay of selected frozen coefficients; no model fits.

Scope: four Pilot cases, and rho0/replicate0 in each n x SNR x truth cell
of development and (only if actually run) confirmation: at most 28 cases.
This file is deliberately separate from every frozen fitting module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, body):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _array_sha(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def _stream_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode('ascii') + a.dtype.str.encode('ascii') + a.tobytes()).hexdigest()


def selected_requests(requests, cohort):
    """Outcome-independent, immutable selection; incomplete cell grids fail."""
    rows = [r for r in requests if r['cohort'] == cohort]
    if cohort == 'pilot':
        expected = set(itertools.product((100, 400), (.5, 4.)))
        assert len(rows) == 4
        assert {(r['config']['n_train'], r['config']['snr']) for r in rows} == expected
        assert all(r['replicate'] == 0 and r['config']['sparsity'] == [4, 3, 5]
                   and r['config']['rho'] == [.6] * 3 for r in rows)
        return sorted(rows, key=lambda r: r['task_id'])
    assert cohort in ('development', 'confirmation')
    chosen = [r for r in rows if r['replicate'] == 0 and r['config']['rho'] == [0.] * 3]
    expected = set(itertools.product((100, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4))))
    actual = {(r['config']['n_train'], r['config']['snr'], tuple(r['config']['sparsity'])) for r in chosen}
    assert len(chosen) == 12 and actual == expected
    return sorted(chosen, key=lambda r: r['task_id'])


def verify_stream(batches, coefficients, supports, saved):
    """Rebuild predictions via rectangular columns, independently of dense ops."""
    labels = ['A0', 'A1', 'A2']
    assert set(coefficients) == set(supports) == set(labels)
    assert saved['prediction'].shape == (len(saved['y']), 3)
    assert saved['signal'].shape == saved['y'].shape
    sse, signal_sse = np.zeros(3), np.zeros(3)
    digest, offset, count, maximum = hashlib.sha256(), 0, 0, 0.
    for X, y, signal in batches:
        assert np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(signal).all()
        stop = offset + len(y)
        np.testing.assert_array_equal(y, saved['y'][offset:stop])
        np.testing.assert_array_equal(signal, saved['signal'][offset:stop])
        for value in (X, y, signal):
            digest.update(_stream_sha(value).encode('ascii'))
        predictions = []
        for arm in labels:
            B, S = np.asarray(coefficients[arm]), [np.asarray(s, dtype=int) for s in supports[arm]]
            assert B.shape == X.shape[1:] and np.isfinite(B).all()
            permitted = np.zeros(B.shape, dtype=bool)
            permitted[np.ix_(*S)] = True
            assert np.count_nonzero(B[~permitted]) == 0
            Z = np.column_stack([X[:, i, j, k] for i, j, k in itertools.product(*S)])
            predictions.append(Z @ B[np.ix_(*S)].ravel())
        prediction = np.column_stack(predictions)
        np.testing.assert_allclose(prediction, saved['prediction'][offset:stop], rtol=1e-10, atol=1e-10)
        maximum = max(maximum, float(np.max(np.abs(prediction - saved['prediction'][offset:stop]))))
        sse += np.sum((y[:, None] - prediction) ** 2, axis=0)
        signal_sse += np.sum((signal[:, None] - prediction) ** 2, axis=0)
        offset, count = stop, count + 1
    assert offset == len(saved['y']) > 0
    return {'metrics': {a: {'pmse': float(sse[j] / offset), 'signal_mse': float(signal_sse[j] / offset)}
                        for j, a in enumerate(labels)},
            'test_fingerprint': {'sha256': digest.hexdigest(), 'n_observations': offset, 'n_batches': count},
            'maximum_prediction_absolute_error': maximum}


def prepare_scope(root, output):
    output.mkdir(parents=True, exist_ok=True)
    cases, plans = [], {}
    for cohort in ('pilot', 'development', 'confirmation'):
        plan_path = root / 'doc/taskcard10/results' / ('pilot' if cohort == 'pilot' else 'formal') / 'plan.json'
        plan = _read(plan_path)
        relative = plan_path.relative_to(root).as_posix()
        plans[relative] = _sha(plan_path)
        for request in selected_requests(plan['requests'], cohort):
            folder = plan_path.parent / cohort / f"task_{request['task_id']:03d}"
            cases.append({'cohort': cohort, 'task_id': request['task_id'],
                          'folder': folder.relative_to(root).as_posix(), 'request': request,
                          'plan': relative, 'plan_sha256': plans[relative]})
    assert len(cases) == 28 and len({c['folder'] for c in cases}) == 28
    _write(output / 'SCOPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rule': 'all4Pilot; formal first replicate0/rho0 for each n x SNR x truth; never outcome based',
        'case_limit': 28, 'cases': cases, 'plan_sha256': plans,
        'script_sha256': _sha(Path(__file__)),
        'test_sha256': _sha(Path(__file__).with_name('test_verify_external.py')),
        'no_search_fits': True, 'no_selection_changes': True,
        'test_role': 'Independent numerical replay of existing selected models, not new confirmation or tuning'})


def verify_case(root, case):
    folder = root / case['folder']
    assert (folder / 'complete.json').is_file(), 'Refusing partial scientific case'
    complete, summary = _read(folder / 'complete.json'), _read(folder / 'summary.json')
    assert complete['request'] == case['request'] == _read(folder / 'request.json')
    assert complete['plan_sha256'] == case['plan_sha256'] == _sha(root / case['plan'])
    plan = _read(root / case['plan'])
    assert plan['protocol_sha256'] == complete['protocol_sha256'] == _sha(root / plan['protocol_path'])
    assert complete['source_identity'] == plan['protocol']['source_identity']
    for relative, expected in complete['source_identity'].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root) and _sha(path) == expected
    for relative, expected in complete['artifact_sha256'].items():
        path = (folder / relative).resolve()
        assert path.is_relative_to(folder) and _sha(path) == expected
    assert _sha(folder / 'selection_frozen.json') == _sha(folder / 'summary.json')
    import scipy
    environment = complete['environment']
    assert (platform.python_version(), np.__version__, scipy.__version__) == (
        environment['python'], environment['numpy'], environment['scipy'])
    assert str(Path(sys.executable).resolve()) == str(Path(environment['executable']).resolve())
    # This imports frozen ops only; no fitting/search function is invoked.
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset, iter_test_batches
    options = dict(case['request']['config'])
    for key in ('dims', 'sparsity', 'rho'):
        options[key] = tuple(options[key])
    dataset = generate_simulation_dataset(TensorSimulationConfig(**options), seed=case['request']['seed'])
    identities = {name: _array_sha(getattr(dataset, name)) for name in ('X_train', 'y_train', 'B_true')}
    identities['supports_true'] = [_array_sha(s) for s in dataset.supports_true]
    assert identities == complete['input_fingerprints']
    assert summary['input_identity'] == {'X': identities['X_train'], 'y': identities['y_train']}
    with np.load(folder / 'truth.npz', allow_pickle=False) as truth:
        np.testing.assert_array_equal(truth['B_true'], dataset.B_true)
        for i, support in enumerate(dataset.supports_true, 1):
            np.testing.assert_array_equal(truth[f'S{i}_true'], support)
    with np.load(folder / 'arrays.npz', allow_pickle=False) as saved:
        coefficients = {arm: saved[summary['arms'][arm]['B_key']].copy() for arm in ('A0', 'A1', 'A2')}
    supports = {arm: summary['arms'][arm]['support'] for arm in coefficients}
    for arm, B in coefficients.items():
        assert _array_sha(B) == summary['arms'][arm]['B_sha256'] == complete['metrics'][arm]['B_sha256']
    with np.load(folder / 'test_predictions.npz', allow_pickle=False) as archive:
        saved_predictions = {name: archive[name].copy() for name in ('y', 'signal', 'prediction')}
    replay = verify_stream(iter_test_batches(dataset, seed=case['request']['test_seed']),
                           coefficients, supports, saved_predictions)
    assert replay['test_fingerprint'] == complete['test_fingerprint']
    maximum_metric_error = 0.
    for arm, metrics in replay['metrics'].items():
        for field, value in metrics.items():
            np.testing.assert_allclose(value, complete['metrics'][arm][field], rtol=1e-10, atol=1e-10)
            maximum_metric_error = max(maximum_metric_error, abs(value - complete['metrics'][arm][field]))
    actual_imports = {}
    source_root = root / 'doc/algorithm_versions/v5.2-search-final/src'
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(source_root)
            relative = path.relative_to(root).as_posix()
            assert _sha(path) == complete['source_identity'][relative]
            actual_imports[name] = {'path': relative, 'sha256': _sha(path)}
    return {'passed': True, 'cohort': case['cohort'], 'task_id': case['task_id'],
            'request': case['request'], 'complete_sha256': _sha(folder / 'complete.json'),
            'source_and_artifact_hashes_exact': True, 'training_fingerprints_exact': True,
            'truth_arrays_exact': True, 'saved_external_y_signal_exact': True,
            'test_fingerprint_exact': True, 'selected_B_identities_exact': True,
            'maximum_PMSE_signal_MSE_absolute_error': maximum_metric_error,
            'actual_core_imports': actual_imports, 'search_fits': 0,
            'selection_changes': 0, **replay}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    parser.add_argument('--cohort', choices=('pilot', 'development', 'confirmation'))
    parser.add_argument('--output', type=Path, default=Path('doc/taskcard10/external_audit'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    assert output.is_relative_to(root / 'doc/taskcard10'), 'Research output must stay in Task10 documents'
    if args.stage == 'prepare':
        prepare_scope(root, output)
        print('EXTERNAL_SCOPE_FROZEN', 28)
        return
    assert args.cohort is not None
    scope = _read(output / 'SCOPE.json')
    assert scope['script_sha256'] == _sha(Path(__file__))
    assert scope['test_sha256'] == _sha(Path(__file__).with_name('test_verify_external.py'))
    cases = [c for c in scope['cases'] if c['cohort'] == args.cohort]
    assert len(cases) == (4 if args.cohort == 'pilot' else 12)
    assert all((root / case['folder'] / 'complete.json').is_file() for case in cases), 'Predeclared scientific cases not all complete; no partial reads or regeneration'
    target = output / args.cohort
    target.mkdir(exist_ok=False)
    started = perf_counter()
    try:
        from scripts.taskcard7.lemon import configure_core
        configure_core('v5.2-search-final')
        # Preload before numerical_simulation's historical path insertion.
        import tensor_abess.tensor.abess
        results = []
        for case in cases:
            result = verify_case(root, case)
            _write(target / f"task_{case['task_id']:03d}.json", result)
            results.append(result)
        _write(target / 'SUMMARY.json', {
            'passed': True, 'cohort': args.cohort, 'cases_checked': len(results),
            'task_ids': [r['task_id'] for r in results], 'selected_models_checked': len(results) * 3,
            'maximum_prediction_absolute_error': max(r['maximum_prediction_absolute_error'] for r in results),
            'maximum_PMSE_signal_MSE_absolute_error': max(r['maximum_PMSE_signal_MSE_absolute_error'] for r in results),
            'source_fingerprint': _sha(Path(__file__)), 'scope_sha256': _sha(output / 'SCOPE.json'),
            'elapsed_seconds': perf_counter() - started, 'search_fits': 0, 'selection_changes': 0,
            'limit': 'Prespecified bounded numerical audit; not every dataset or new scientific confirmation'})
        print('EXTERNAL_AUDIT_COMPLETE', args.cohort, len(results))
    except BaseException as error:
        _write(target / 'FAILURE.json', {'type': type(error).__name__, 'error': str(error),
                                       'traceback': traceback.format_exc(), 'elapsed_seconds': perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
