"""Read-only integrity and metadata audit of the completed Task 8 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path


PLAN_SHA = 'f4e84f877af3eedea4c4a3877526054d87c37165bc494b311d644f5a6fba1db0'
BUDGET_SHA = 'bf2221ea0fb22b788c00bb6f7cba4d9d3def4c93179d39aa3c94bfac6ec5ec23'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), message)


def manifest_check(directory, expected):
    for name, digest in expected.items():
        require(sha(directory / name) == digest, 'Artifact drift: ' + str(directory / name))
    return len(expected)


def gate_from_rows(rows, thresholds):
    by_rule = {rule: [r for r in rows if r['rule'] == rule]
               for rule in ('baseline', 'alpha2_scale0')}
    avg = lambda rule, key, pred=lambda r: True: sum(
        r[key] for r in by_rule[rule] if pred(r)) / sum(pred(r) for r in by_rule[rule])
    ratio = avg('alpha2_scale0', 'pmse') / avg('baseline', 'pmse')
    cells = {str(cell): avg('alpha2_scale0', 'pmse', lambda r, c=cell: r['cell_id'] == c) /
             avg('baseline', 'pmse', lambda r, c=cell: r['cell_id'] == c) for cell in range(36)}
    f1 = avg('alpha2_scale0', 'mode_f1_macro', lambda r: r['snr'] == 4) - avg(
        'baseline', 'mode_f1_macro', lambda r: r['snr'] == 4)
    under = avg('alpha2_scale0', 'q_under', lambda r: r['snr'] == 4) - avg(
        'baseline', 'q_under', lambda r: r['snr'] == 4)
    near = avg('alpha2_scale0', 'near_capacity', lambda r: r['n_train'] == 100) - avg(
        'baseline', 'near_capacity', lambda r: r['n_train'] == 100)
    checks = {'overall_pmse': ratio <= thresholds['overall_pmse_ratio_max'],
              'worst_cell_pmse': max(cells.values()) <= thresholds['worst_cell_pmse_ratio_max'],
              'strong_snr4_mode_f1': f1 >= thresholds['strong_snr4_mode_f1_delta_min'],
              'strong_snr4_underselection': under <= thresholds['strong_snr4_q_under_delta_max'],
              'n100_near_capacity': near <= thresholds['n100_near_capacity_delta_max']}
    return {'passed': all(checks.values()), 'checks': checks,
            'values': {'overall_pmse_ratio': ratio, 'worst_cell_pmse_ratio': max(cells.values()),
                       'cell_pmse_ratios': cells, 'strong_mode_f1_delta': f1,
                       'strong_underselection_delta': under, 'n100_near_capacity_delta': near}}


def verify_gate(actual, expected):
    require(actual['passed'] == expected['passed'] and actual['checks'] == expected['checks'],
            'Independent gate decision disagreement')
    for key, value in expected['values'].items():
        if isinstance(value, dict):
            for cell, ratio in value.items():
                close(actual['values'][key][cell], ratio, 'Cell gate ratio drift')
        else:
            close(actual['values'][key], value, 'Gate value drift: ' + key)


def verify_trace(trace, selection, record, rules):
    request = record['request']
    n, dims = request['config']['n_train'], request['config']['dims']
    Q = math.floor(.7 * n)
    caps = trace['caps']
    expected_grid = [list(s) for s in itertools.product(*(range(1, cap + 1) for cap in caps))
                     if math.prod(s) <= Q]
    require(trace['ordered_size_grid'] == expected_grid, 'A2 ordered full-grid drift')
    candidates, fixed = trace['candidates'], trace['fixed_searches']
    require(len(candidates) == 1 + len(expected_grid) and len(fixed) == len(expected_grid),
            'Incomplete candidate library')
    require(candidates[0] == trace['adaptive_terminal'], 'Adaptive terminal mismatch')
    for candidate in candidates:
        supports = candidate['support']
        sizes = [len(s) for s in supports]
        require(sizes == candidate['sizes'] and math.prod(sizes) == candidate['q'] <= Q,
                'Candidate support/size/Q mismatch')
        require(all(s == sorted(set(s)) and all(0 <= k < p for k in s)
                    for s, p in zip(supports, dims)), 'Illegal support index')
        require(all(s <= cap for s, cap in zip(sizes, caps)), 'Candidate exceeds original mode caps')
        require(math.isfinite(candidate['rss']) and candidate['rss'] >= 0, 'Nonfinite or negative RSS')
    max_loss_error = 0.
    for size, search, candidate in zip(expected_grid, fixed, candidates[1:]):
        require(size == search['sizes'] == candidate['sizes'], 'Fixed candidate order drift')
        require(search['B_sha256'] == candidate['B_sha256'], 'Best-visited B metadata mismatch')
        accepted, visited = search['loss_history'], search['visited_loss_history']
        require(accepted and visited and all(math.isfinite(x) for x in accepted + visited),
                'Nonfinite or empty BCD history')
        require(all(b <= a + 1e-10 * max(1., abs(a)) for a, b in zip(accepted, accepted[1:])),
                'Accepted BCD loss increased')
        best = search['best_iteration']
        require(isinstance(best, int) and 0 <= best < len(visited), 'Invalid best-visited iteration')
        loss = candidate['rss'] / (2 * n)
        for expected in (min(visited), visited[best], accepted[-1]):
            close(loss, expected, 'Saved full-OLS RSS does not match best visited loss')
            max_loss_error = max(max_loss_error, abs(loss - expected))
    require(selection['rules'] == rules, 'Saved rule identity drift')
    for rule, choice, metrics in zip(rules, selection['choices'], record['metrics']):
        ell = math.log(max(math.log(n), math.e))
        scores = [n * math.log(c['rss'] / n) + rule['alpha'] * c['q'] * ell + ell * sum(
            math.lgamma(p + 1) - math.lgamma(s + 1) - math.lgamma(p - s + 1)
            for p, s in zip(dims, c['sizes'])) for c in candidates]
        for score, saved in zip(scores, choice['candidate_scores']):
            close(score, saved, 'Independent MSIC decomposition disagreement')
        # Use saved exact scores for stable ties; the independently decomposed scores
        # above may differ by machine epsilon because gammaln is implemented differently.
        exact_scores = choice['candidate_scores']
        imin = min(range(len(candidates)), key=lambda i: exact_scores[i])
        tolerance = rule['scale'] * math.sqrt(candidates[imin]['q'] * math.log(n))
        eligible = [i for i, score in enumerate(exact_scores) if score <= exact_scores[imin] + tolerance]
        selected = min(eligible, key=lambda i: (candidates[i]['q'], exact_scores[i], sum(candidates[i]['sizes'])))
        require(imin == choice['minimum_index'] and selected == choice['selected_index'],
                'Independent selection replay disagreement')
        require(metrics['rule'] == rule['rule'] and metrics['selected_index'] == selected,
                'Metric row selection identity mismatch')
        require(metrics['q'] == candidates[selected]['q'] and metrics['Q'] == Q,
                'Metric q/Q mismatch')
        close(metrics['train_rss'], candidates[selected]['rss'], 'Metric training RSS drift')
        close(metrics['msic'], exact_scores[selected], 'Metric selected score drift')
    require(trace['final'] == candidates[selection['choices'][0]['selected_index']],
            'Captured original fitted result differs from baseline replay')
    return len(candidates), len(fixed), max_loss_error


def verify_coefficients(folder, trace, dims):
    import numpy as np
    with np.load(folder / 'candidate_coefficients.npz', allow_pickle=False) as archive:
        coefficients = archive['B']
    require(coefficients.shape == (len(trace['candidates']), *dims), 'Saved coefficient shape mismatch')
    require(np.isfinite(coefficients).all(), 'Nonfinite saved coefficient')
    for B, candidate in zip(coefficients, trace['candidates']):
        header = json.dumps({'shape': list(B.shape), 'dtype': B.dtype.str}, sort_keys=True).encode()
        digest = hashlib.sha256(header + np.ascontiguousarray(B).tobytes()).hexdigest()
        require(digest == candidate['B_sha256'], 'Saved candidate coefficient fingerprint drift')
        beta = B[np.ix_(*candidate['support'])]
        require(np.count_nonzero(B) == np.count_nonzero(beta), 'Saved coefficient outside legal support')


def audit(repo, allow_active=False):
    base = repo / 'numerical_simulation/outputs_selection_research_task8'
    terminal = next((base / name for name in ('completion.json', 'stopped.json')
                     if (base / name).is_file()), None)
    require(allow_active or terminal is not None, 'Controller active; final integrity must wait')
    require(sha(base / 'plan.json') == PLAN_SHA, 'Frozen primary plan drift')
    require(sha(repo / 'doc/taskcard8/COMPUTE_BUDGET.json') == BUDGET_SHA, 'Frozen budget drift')
    plan = read(base / 'plan.json')
    baseline = read(repo / 'doc/taskcard8/protected_baseline.json')['files']
    for name, entry in baseline.items():
        p = repo / name
        require(p.stat().st_size == entry['size'] and sha(p) == entry['sha256'],
                'Protected pre-existing file drift: ' + name)
    source = repo / 'doc/algorithm_versions/v5.2-search-final/src'
    manifest_check(source, plan['core_sha256'])
    manifest_check(repo, plan['dependencies'])
    require(sha(repo / plan['formal_config_source']) == plan['formal_plan_sha256'], 'Formal configuration drift')
    deadline = datetime.fromisoformat(plan['budget']['deadline_utc'])
    requests = {(r['cohort'], r['task_id']): r for r in plan['requests']}
    seeds = [r[k] for r in plan['requests'] for k in ('seed', 'test_seed')]
    require(len(seeds) == len(set(seeds)) == 432, 'Training/test seeds overlap')
    lock_path = base / 'development_lock.json'
    lock = read(lock_path) if lock_path.is_file() else None
    lock_sha = sha(lock_path) if lock else None
    if lock:
        require(lock['plan_sha256'] == PLAN_SHA and lock['rules'] == plan['rules'], 'Development lock identity drift')
        manifest_check(base / 'development', lock['development_artifacts'])
        require(sha(base / 'development_summary.json') == lock['development_summary_sha256'], 'Locked development summary drift')
    rows = {'development': [], 'confirmation': []}
    counts, candidate_count, fixed_count, max_loss_error, artifacts, imports = {}, 0, 0, 0., 0, 0
    for cohort in rows:
        records = sorted((base / cohort).glob('*/complete.json'))
        counts[cohort] = len(records)
        for path in records:
            record = read(path)
            request = record['request']
            require(request == requests[(cohort, request['task_id'])], 'Dataset differs from frozen request')
            require(path.parent.name == f"task_{request['task_id']:03d}", 'Dataset path identity mismatch')
            require(record['plan_sha256'] == PLAN_SHA, 'Per-dataset protocol SHA drift')
            started = read(path.parent / 'started.json')
            require(started['request'] == request and started['plan_sha256'] == PLAN_SHA, 'Started identity drift')
            require(datetime.fromisoformat(started['started_utc']) <=
                    datetime.fromisoformat(record['completed_utc']) <= deadline, 'Time order or frozen deadline violated')
            expected_lock = lock_sha if cohort == 'confirmation' else None
            require(record['development_lock_sha256'] == started['development_lock_sha256'] == expected_lock,
                    'Confirmation did not bind to exact development lock')
            if cohort == 'confirmation':
                require(lock and datetime.fromisoformat(started['started_utc']) >=
                        datetime.fromisoformat(lock['locked_utc']), 'Confirmation began before development lock')
            artifacts += manifest_check(path.parent, record['artifact_sha256'])
            for name, entry in record['actual_imports'].items():
                p = Path(entry['path']).resolve()
                require(p.is_relative_to(source.resolve()), 'Actual tensor import escaped frozen source')
                require(sha(p) == entry['sha256'] == plan['core_sha256'][p.relative_to(source.resolve()).as_posix()],
                        'Actual imported source SHA drift: ' + name)
                imports += 1
            require(len(record['metrics']) == 2, 'Missing matched metric row')
            require(all(math.isfinite(value) for row in record['metrics'] for value in row.values()
                        if isinstance(value, (int, float))), 'Nonfinite metric value')
            rows[cohort].extend(record['metrics'])
            trace = read(path.parent / 'trace.json')
            a, b, error = verify_trace(trace, read(path.parent / 'selection.json'), record, plan['rules'])
            verify_coefficients(path.parent, trace, request['config']['dims'])
            candidate_count += a
            fixed_count += b
            max_loss_error = max(error, max_loss_error)
        if counts[cohort] == 108:
            actual_gate = gate_from_rows(rows[cohort], plan['development_gate'])
            verify_gate(read(base / f'{cohort}_summary.json')['gate'], actual_gate)
            if cohort == 'development':
                require(lock is not None, 'Complete development lacks lock')
                verify_gate(lock['development_gate'], actual_gate)
    if terminal:
        term = read(terminal)
        require(term['plan_sha256'] == PLAN_SHA, 'Terminal protocol drift')
        if terminal.name == 'completion.json':
            require(counts == {'development': 108, 'confirmation': 108}, 'False complete coverage')
            require(term['development_lock_sha256'] == lock_sha and term['release_approved'] is False,
                    'Terminal lock/release identity mismatch')
            artifacts += manifest_check(base, term['artifact_sha256'])
        else:
            require(counts == {'development': term['completed_development'],
                               'confirmation': term['completed_confirmation']}, 'Stopped count mismatch')
    warm = repo / 'doc/algorithm_versions/v5.3-warm-start-research'
    version = read(warm / 'VERSION.json')
    manifest_check(warm / 'src', version['sha256'])
    manifest_check(warm / 'incoming', version['incoming_sha256'])
    require(version['research_only'] is True and version['release_approved'] is False, 'Warm research identity drift')
    require([k for k in version['sha256'] if version['sha256'][k] != version['baseline_sha256'][k]] ==
            ['tensor_abess/tensor/bcd.py'], 'Unexpected Warm source changes')
    inner = repo / 'numerical_simulation/outputs_selection_inner_task8'
    protocol = read(inner / 'protocol.json')
    require(sha(inner / 'protocol.json') == '34e7f5a4d074a16b57f8a4570c2614761bde9ea14c1a27305af4b71c0794c4be',
            'Inner protocol drift')
    require(sha(repo / 'scripts/taskcard8/inner_selection.py') == protocol['implementation_sha256'], 'Inner implementation drift')
    require(sha(repo / 'scripts/taskcard8/test_inner_selection.py') == protocol['test_sha256'], 'Inner test drift')
    require(read(inner / 'independent_numerical_verification.json')['passed'], 'Inner independent audit failed')
    require(read(inner / 'summary.json')['outer_test_evaluations'] == 0, 'Inner study evaluated outer test')
    return {'passed': True, 'final': terminal is not None, 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'protected_preexisting_files': len(baseline), 'frozen_core_files': len(plan['core_sha256']),
            'completed_datasets': counts, 'candidate_metadata_records': candidate_count,
            'candidate_coefficient_arrays_checked_finite_support_and_digest': candidate_count,
            'fixed_bcd_histories': fixed_count, 'artifact_digest_checks_including_nested_manifests': artifacts,
            'actual_import_digest_checks': imports, 'maximum_best_visited_loss_disagreement': max_loss_error,
            'plan_sha256': PLAN_SHA, 'budget_sha256': BUDGET_SHA,
            'development_lock_sha256': lock_sha, 'warm_only_bcd_changed': True,
            'scope': 'All completed metadata/hashes/selection/gates; full numeric regeneration is separately bounded to the explicitly listed cases in INDEPENDENT_NUMERICAL_AUDIT.json.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-active', action='store_true', help='Read-only interim audit; never a final acceptance')
    parser.add_argument('--output', default='doc/taskcard8/FINAL_VALIDATION.json')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = repo / args.output
    require(not output.exists(), 'Refusing to overwrite previous audit')
    result = audit(repo, args.allow_active)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Integrity passed; datasets:', result['completed_datasets'], 'protected files:', result['protected_preexisting_files'])


if __name__ == '__main__':
    main()
