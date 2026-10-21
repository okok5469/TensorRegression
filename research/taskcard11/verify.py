"""Independent saved-evidence checks; no optimizer or new random input families."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def configure_replay_core(repo):
    """Configure once, then reject any preloaded module outside the frozen tree."""
    source = (Path(repo).resolve() / 'doc/algorithm_versions/v5.2-search-final/src').resolve()
    loaded = {name: module for name, module in sys.modules.items()
              if name == 'tensor_abess' or name.startswith('tensor_abess.')}
    for name, module in loaded.items():
        require(getattr(module, '__file__', None) and Path(module.__file__).resolve().is_relative_to(source),
                'Preloaded replay core escaped frozen source: ' + name)
    if not loaded:
        from scripts.taskcard7.lemon import configure_core
        require(configure_core('v5.2-search-final') == source, 'Replay core source differs')
    import tensor_abess.tensor.abess
    require(Path(tensor_abess.tensor.abess.__file__).resolve().is_relative_to(source), 'Replay core escaped frozen source')
    return source


def verify_artifact_hashes(folder, hashes, required):
    folder = Path(folder).resolve()
    require(set(required) <= set(hashes), 'Required saved artifact omitted from manifest')
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder) and path.is_file(), 'Artifact path escaped or missing')
        require(file_sha(path) == expected, 'Saved artifact hash mismatch: ' + name)
    return len(hashes)


def audit_selection(rows, coefficients, choices, arms, n, dimensions):
    require(len(rows) == len(coefficients) > 0, 'Incomplete saved candidate coefficients')
    ell = np.log(max(np.log(n), np.e))
    scores, components = [], []
    for row, B in zip(rows, coefficients):
        support, sizes = row['support'], row['sizes']
        require(len(sizes) == len(support) == 3 and row['q'] == math.prod(sizes), 'Candidate shape/q mismatch')
        require(list(map(len, support)) == sizes, 'Candidate support/shape mismatch')
        for s, size, p in zip(support, sizes, dimensions):
            require(size > 0 and s == sorted(set(s)) and all(isinstance(i, int) and 0 <= i < p for i in s),
                    'Invalid candidate support')
        require(B.shape == tuple(dimensions) and np.isfinite(B).all(), 'Invalid candidate coefficients')
        require(array_sha(B) == row['B_sha256'], 'Candidate coefficient hash mismatch')
        require(np.count_nonzero(B[np.ix_(*support)]) == np.count_nonzero(B), 'Coefficient outside support')
        require(np.isfinite(row['rss']) and row['rss'] >= 0, 'Invalid candidate RSS')
        s1, s2, s3 = sizes
        fit = n * np.log(max(row['rss'] / n, np.finfo(float).tiny))
        df = 1.5 * s1 * s2 * s3 * ell
        count = sum(float(gammaln(p + 1) - gammaln(s + 1) - gammaln(p - s + 1))
                    for p, s in zip(dimensions, sizes))
        score = float(fit + df + ell * count)
        require(score == row['msic'], 'Saved baseline score differs from independent formula')
        scores.append(score)
        components.append({'fit_term': float(fit), 'df_penalty': float(df),
                           'multiplicity_penalty': float(ell * count), 'baseline_score': score})
    recovered = {}
    for label, corrected in (('A0', False), ('A1', True)):
        require(all(choices[label][k] == v for k, v in {'alpha': 1.5, 'gamma': 1., 'scale': 1.,
                'finite_sample': corrected}.items()), 'Saved selector parameters differ')
        values = [score + (float(ell * row['q'] ** 2 / (2 * n)) if corrected else 0.)
                  for score, row in zip(scores, rows)]
        terms = [{**component, 'fs_penalty': float(ell * row['q'] ** 2 / (2 * n)) if corrected else 0.,
                  'score': value} for component, row, value in zip(components, rows, values)]
        require(choices[label]['candidate_terms'] == terms, 'Saved score decomposition differs')
        minimum = min(range(len(rows)), key=values.__getitem__)
        q_min = rows[minimum]['q']
        tolerance = float(np.sqrt(max(q_min, 1) * max(np.log(max(n, 2)), 1.)))
        eligible = [i for i, value in enumerate(values) if value <= values[minimum] + tolerance]
        selected = min(eligible, key=lambda i: (rows[i]['q'], values[i], sum(rows[i]['sizes'])))
        expected = {'minimum_index': minimum, 'minimum_q': q_min, 'minimum_score': values[minimum],
                    'tolerance': tolerance, 'eligible_indices': eligible, 'selected_index': selected,
                    'selected_score': values[selected], 'candidate_scores': values}
        for name, value in expected.items():
            require(choices[label][name] == value, 'Saved selection differs: ' + label + '/' + name)
        require(arms[label]['full_candidate_index'] == selected, 'Selected arm index differs')
        require(arms[label]['selected_score'] == values[selected], 'Selected arm score differs')
        for name in ('support', 'sizes', 'q', 'rss', 'B_sha256'):
            require(arms[label][name] == rows[selected][name], 'Selected arm candidate identity differs')
        recovered[label] = selected
    return {'passed': True, 'candidate_count': len(rows), 'selected_indices': recovered}


def audit_saved_predictions(saved, metrics, n_test):
    y, signal, predicted = (np.asarray(saved[key]) for key in ('y', 'signal', 'prediction'))
    require(y.shape == signal.shape == (n_test,) and predicted.shape == (n_test, 2), 'Saved prediction shape differs')
    require(all(np.isfinite(a).all() for a in (y, signal, predicted)), 'Nonfinite saved prediction')
    results = {}
    for j, label in enumerate(('A0', 'A1')):
        actual = {'pmse': float(np.mean((y - predicted[:, j]) ** 2)),
                  'signal_mse': float(np.mean((signal - predicted[:, j]) ** 2))}
        for key, value in actual.items():
            np.testing.assert_allclose(value, metrics[label][key], rtol=1e-12, atol=1e-12)
        results[label] = actual
    return results


def audit_recovery(B, truth, support, true_support, metrics):
    require(B.shape == truth.shape and np.isfinite(truth).all(), 'Saved truth coefficient shape differs')
    sizes, true_sizes = list(map(len, support)), list(map(len, true_support))
    require(metrics['q'] == math.prod(sizes) and metrics['sizes'] == sizes and metrics['support'] == support,
            'Recovery metadata differs')
    overlap = [len(set(a).intersection(b)) for a, b in zip(support, true_support)]
    precision = [k / len(a) for k, a in zip(overlap, support)]
    recall = [k / len(b) for k, b in zip(overlap, true_support)]
    f1 = [2 * p * r / (p + r) if p + r else 0. for p, r in zip(precision, recall)]
    hit, q, tq = math.prod(overlap), math.prod(sizes), math.prod(true_sizes)
    ep, er = hit / q, hit / tq
    expected = {'ree': float(np.linalg.norm(B - truth) / max(float(np.linalg.norm(truth)), np.finfo(float).tiny)),
                'exact_recovery': float(all(a == b for a, b in zip(support, true_support))),
                'element_precision': ep, 'element_recall': er,
                'element_f1': 2 * ep * er / (ep + er) if ep + er else 0., 'element_fdr': 1. - ep,
                'coefficient_norm': float(np.linalg.norm(B)), 'q_hat': float(q),
                'mode_precision_macro': sum(precision) / 3., 'mode_tpr_macro': sum(recall) / 3.,
                'mode_f1_macro': sum(f1) / 3.}
    for i in range(3):
        expected.update({f'mode_precision_{i+1}': precision[i], f'mode_tpr_{i+1}': recall[i],
                         f'mode_f1_{i+1}': f1[i], f's{i+1}_hat': float(sizes[i])})
    for name, value in expected.items():
        np.testing.assert_allclose(metrics[name], value, rtol=1e-12, atol=1e-12)
    return expected


def audit_test_replay(batches, coefficients, saved, expected_fingerprint):
    digest = hashlib.sha256()
    offset = batch_count = 0
    maximum_error = 0.
    for X, y, signal in batches:
        for a in (X, y, signal):
            a = np.ascontiguousarray(a)
            raw = hashlib.sha256(str(a.shape).encode('ascii') + a.dtype.str.encode('ascii') + a.tobytes()).hexdigest()
            digest.update(raw.encode('ascii'))
        endpoint = offset + len(y)
        np.testing.assert_array_equal(y, saved['y'][offset:endpoint])
        np.testing.assert_array_equal(signal, saved['signal'][offset:endpoint])
        for j, label in enumerate(('A0', 'A1')):
            # Independent tensor contraction, not the worker's stacked flattened matmul.
            predicted = np.einsum('nijk,ijk->n', X, coefficients[label])
            expected = saved['prediction'][offset:endpoint, j]
            np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)
            maximum_error = max(maximum_error, float(np.max(np.abs(predicted - expected))))
        offset, batch_count = endpoint, batch_count + 1
    require(offset == len(saved['y']) > 0, 'Replayed test stream length differs')
    actual = {'sha256': digest.hexdigest(), 'n_observations': offset, 'n_batches': batch_count}
    require(actual == expected_fingerprint, 'Independent test stream fingerprint differs')
    return {'passed': True, 'test_fingerprint': actual, 'maximum_prediction_absolute_error': maximum_error}


def verify_envelope(repo, run, folder):
    repo, run, folder = Path(repo).resolve(), Path(run).resolve(), Path(folder).resolve()
    plan = read_json(run / 'plan.json')
    protocol_path = (repo / plan['protocol_path']).resolve()
    require(protocol_path.is_relative_to(repo) and file_sha(protocol_path) == plan['protocol_sha256'],
            'Frozen protocol hash differs')
    protocol = read_json(protocol_path)
    require(protocol == plan['protocol'], 'Frozen protocol contents differ')
    record = read_json(folder / 'complete.json')
    request = record['request']
    expected = [r for r in plan['requests'] if (r['cohort'], r['task_id']) == (request['cohort'], request['task_id'])]
    require(expected == [request], 'Case differs from the exact frozen request')
    require(record['plan_sha256'] == file_sha(run / 'plan.json') and
            record['protocol_sha256'] == plan['protocol_sha256'], 'Case plan/protocol identity differs')
    required = {'arrays.npz', 'candidate_coefficients.npz', 'independent_audit.json', 'input_identity.json',
                'request.json', 'selection_frozen.json', 'started.json', 'summary.json',
                'test_predictions.npz', 'trace.json', 'truth.npz'}
    count = verify_artifact_hashes(folder, record['artifact_sha256'], required)
    require(read_json(folder / 'request.json') == request, 'Saved request file differs')
    names = dict(protocol['source_files'])
    prefix = 'doc/algorithm_versions/v5.2-search-final/src/'
    names.update({prefix + name: digest for name, digest in protocol['core_sha256'].items()})
    require(record['source_identity'] == protocol['source_identity'] == names, 'Recorded source identity differs')
    for name, digest in names.items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and file_sha(path) == digest, 'Frozen source changed: ' + name)
    # Native options was protected globally in Pilot and explicitly frozen in formal.
    extra = 'research/taskcard10/cv.py'
    baseline = read_json(repo / 'doc/taskcard11/PROTECTED_BASELINE.json')
    require(file_sha(repo / extra) == baseline['files'][extra]['sha256'], 'Native options changed from pre-input baseline')
    require(record['environment'] == protocol['environment'], 'Saved environment differs from frozen environment')
    from research.taskcard10.experiment import environment
    require(environment() == protocol['environment'], 'Verification environment differs from frozen environment')
    source = (repo / prefix).resolve()
    imports = record['actual_core_imports']
    require('tensor_abess.tensor.abess' in imports and 'tensor_abess.tensor.bcd' in imports, 'Required actual import identity missing')
    for item in imports.values():
        path = Path(item['path']).resolve()
        require(path.is_relative_to(source), 'Actual core import escaped frozen source')
        require(file_sha(path) == item['sha256'] == protocol['core_sha256'][path.relative_to(source).as_posix()],
                'Actual imported core source identity differs')
    return plan, record, count


def verify_case(repo, run, folder, *, replay=False):
    repo, run, folder = Path(repo).resolve(), Path(run).resolve(), Path(folder).resolve()
    plan, record, hash_count = verify_envelope(repo, run, folder)
    request, config = record['request'], record['request']['config']
    summary = read_json(folder / 'summary.json')
    require(summary == read_json(folder / 'selection_frozen.json'), 'Selection changed after external testing')
    trace = read_json(folder / 'trace.json')
    require(trace == summary['full_trace'] and summary['config'] == config, 'Saved trace/config differs')
    arrays = dict(np.load(folder / 'arrays.npz', allow_pickle=False))
    candidate_B = np.load(folder / 'candidate_coefficients.npz', allow_pickle=False)['B']
    np.testing.assert_array_equal(arrays['full_B'], candidate_B)
    rows = trace['candidates']
    grid = [list(s) for s in itertools.product(*[range(1, c + 1) for c in trace['caps']])
            if math.prod(s) <= math.floor(.7 * config['n_train'])]
    require(grid == trace['ordered_size_grid'] == [r['sizes'] for r in rows[1:]], 'Saved native full grid differs')
    require(len(rows) == len(grid) + 1 and len(trace['fixed_searches']) == len(grid), 'Candidate/search count differs')
    selection = audit_selection(rows, candidate_B, summary['choices'], summary['arms'], config['n_train'], config['dims'])
    native = rows[selection['selected_indices']['A0']]
    for key in ('support', 'sizes', 'q', 'rss', 'msic', 'B_sha256'):
        require(native[key] == trace['final'][key], 'A0 differs from native final trace')
    require(summary['choices']['A0']['eligible_indices'] == trace['eligible_candidate_indices'] and
            summary['choices']['A0']['tolerance'] == trace['one_se_tolerance'], 'A0 native band differs')
    require(summary['counters']['native_search_calls'] == 1 and summary['counters']['extra_search_calls'] == 0,
            'Unexpected extra search in saved counters')
    audit = read_json(folder / 'independent_audit.json')
    require(audit['passed'] and audit['candidate_count'] == len(rows) and audit['all_candidate_designs_full_rank'],
            'Saved full-pool OLS gate is not complete')
    truth = dict(np.load(folder / 'truth.npz', allow_pickle=False))
    true_support = [truth[f'S{i}_true'].astype(int).tolist() for i in (1, 2, 3)]
    fp = record['input_fingerprints']
    require(array_sha(truth['B_true']) == fp['B_true'] and
            [array_sha(truth[f'S{i}_true']) for i in (1, 2, 3)] == fp['supports_true'], 'Saved truth identity differs')
    require(summary['input_identity'] == {'X': fp['X_train'], 'y': fp['y_train']}, 'Training input identity differs')
    input_record = read_json(folder / 'input_identity.json')
    require(input_record['input_fingerprints'] == fp and input_record['duplicate'] is False, 'Saved input duplicate gate differs')
    claim_key = hashlib.sha256(json.dumps((fp['X_train'], fp['y_train']), separators=(',', ':')).encode()).hexdigest()
    require(file_sha(run / 'input_claims' / (claim_key + '.json')) == input_record['atomic_claim_sha256'],
            'Atomic input claim differs')
    coefficients, recovery = {}, {}
    for label in ('A0', 'A1'):
        index = selection['selected_indices'][label]
        B = arrays[label + '_B']
        np.testing.assert_array_equal(B, candidate_B[index])
        require(record['metrics'][label]['B_sha256'] == array_sha(B), 'Reported selected B identity differs')
        coefficients[label] = B
        recovery[label] = audit_recovery(B, truth['B_true'], rows[index]['support'], true_support, record['metrics'][label])
    sizes = list(map(len, true_support))
    expected_recovery = {'q_true': math.prod(sizes), 'truth_sizes': sizes, 'truth_support': true_support,
        'Q_native': math.floor(.7 * config['n_train']),
        'truth_shape_native_representable': all(s <= c for s, c in zip(sizes, trace['caps'])) and math.prod(sizes) <= math.floor(.7 * config['n_train']),
        'truth_shape_found_full_pool': any(r['sizes'] == sizes for r in rows),
        'truth_support_found_full_pool': any(r['support'] == true_support for r in rows)}
    require(record['recovery_evidence'] == expected_recovery, 'Recovery representability/coverage evidence differs')
    saved = dict(np.load(folder / 'test_predictions.npz', allow_pickle=False))
    predictions = audit_saved_predictions(saved, record['metrics'], config['n_test'])
    replay_result = {'status': 'NOT_RUN', 'reason': 'Saved-evidence-only verification requested'}
    if replay:
        configure_replay_core(repo)
        from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset, iter_test_batches
        values = dict(config)
        for key in ('dims', 'sparsity', 'rho'):
            values[key] = tuple(values[key])
        data = generate_simulation_dataset(TensorSimulationConfig(**values), seed=request['seed'])
        regenerated = {name: array_sha(getattr(data, name)) for name in ('X_train', 'y_train', 'B_true')}
        regenerated['supports_true'] = [array_sha(s) for s in data.supports_true]
        require(regenerated == fp, 'Regenerated original training/truth input differs')
        worst_rss, worst_gradient = 0., 0.
        for row, B in zip(rows, candidate_B):
            support = row['support']
            Z = np.column_stack([data.X_train[:, i, j, k] for i, j, k in itertools.product(*support)])
            beta = B[np.ix_(*support)].ravel()
            residual = data.y_train - Z @ beta
            rss = float(residual @ residual)
            gradient = float(np.linalg.norm(Z.T @ residual) / max(1., np.linalg.norm(Z) * np.linalg.norm(data.y_train)))
            np.testing.assert_allclose(rss, row['rss'], rtol=1e-9, atol=1e-9)
            require(gradient < 1e-8, 'Replayed candidate violates OLS normal equations')
            worst_rss, worst_gradient = max(worst_rss, abs(rss-row['rss'])), max(worst_gradient, gradient)
        replay_result = audit_test_replay(iter_test_batches(data, seed=request['test_seed']), coefficients, saved, record['test_fingerprint'])
        replay_result.update(status='PASS', training_identity_exact=True, candidate_RSS_normal_equations_checked=len(rows),
                             maximum_candidate_RSS_absolute_error=worst_rss, maximum_normalized_OLS_gradient=worst_gradient)
    return {'passed': True, 'cohort': request['cohort'], 'task_id': request['task_id'],
            'complete_sha256': file_sha(folder / 'complete.json'), 'artifact_hashes_checked': hash_count,
            'selection': selection, 'recovery': recovery, 'recomputed_prediction_metrics': predictions,
            'replay': replay_result, 'search_calls': 0, 'new_scientific_repetitions': 0}


def verify_run(repo, run, cohort, *, replay=False):
    repo, run = Path(repo).resolve(), Path(run).resolve()
    plan = read_json(run / 'plan.json')
    requests = [r for r in plan['requests'] if r['cohort'] == cohort]
    folders = sorted((run / cohort).glob('task_*/complete.json'))
    require(bool(requests), 'No frozen requests for cohort')
    if len(folders) != len(requests):
        return {'status': 'NOT_RUN', 'cohort': cohort, 'planned': len(requests), 'completed': len(folders),
                'reason': 'Full frozen cohort is incomplete; no missing case is regenerated or replaced'}
    require([read_json(p)['request'] for p in folders] == requests, 'Frozen cohort request coverage differs')
    inputs = [read_json(p)['input_fingerprints'] for p in folders]
    require(len({(r['X_train'], r['y_train']) for r in inputs}) == len(inputs), 'Duplicate completed training input')
    started = time.perf_counter()
    checks = [verify_case(repo, run, p.parent, replay=replay) for p in folders]
    return {'status': 'PASS_FULL_REPLAY' if replay else 'PASS_SAVED_EVIDENCE_ONLY', 'cohort': cohort,
            'planned': len(requests), 'completed': len(checks), 'checks': checks,
            'elapsed_seconds': time.perf_counter()-started, 'search_calls': 0, 'new_scientific_repetitions': 0,
            'verifier_sha256': file_sha(Path(__file__)), 'plan_sha256': file_sha(run / 'plan.json')}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cohort', choices=('pilot', 'validation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    result = verify_run(repo, args.run_dir, args.cohort, replay=args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(result['status'], result['completed'], '/', result['planned'], flush=True)


if __name__ == '__main__':
    main()
