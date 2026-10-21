"""Rescore the 216 saved task 8 pools; never generate inputs, refit, or predict."""
from __future__ import annotations

from collections import Counter
from itertools import product
import math
from pathlib import Path

import numpy as np

from research.taskcard9.pool_msic import (
    array_sha, identity_row, load_task, read, require, sha, support_metrics,
    task_records, write, write_csv,
)
from research.taskcard11.rescore import select_candidates


def saved_prediction(record, pool, old_choices, replay, index):
    """Join existing test metrics only on exact case and coefficient identity."""
    known = {pool[choice['selected_index']]['B_sha256']: metric
             for choice, metric in zip(old_choices, record['metrics'])}
    digest = pool[index]['B_sha256']
    request = record['request']
    key = (request['cohort'], request['task_id'], digest)
    sources = [(replay.get(key), 'Task9_identity_gated_saved_replay'),
               (known.get(digest), 'Task8_identical_B_saved_prediction')]
    available = next(((value, label) for value, label in sources if value is not None), None)
    if available is None:
        return {'pmse': None, 'signal_mse': None, 'pmse_source': 'NOT_AVAILABLE'}
    metric, label = available
    require(all(np.isfinite(metric[k]) and metric[k] >= 0 for k in ('pmse', 'signal_mse')),
            'Invalid saved prediction metric')
    return {'pmse': metric['pmse'], 'signal_mse': metric['signal_mse'], 'pmse_source': label}


def audit_case(folder):
    """Validate sealed arrays and recover all native score/selection decisions."""
    record, trace, coefficients, truth = load_task(folder)
    config, pool = record['request']['config'], trace['candidates']
    n, dims = config['n_train'], config['dims']
    old = read(folder / 'selection.json')['choices']
    choices = [select_candidates(pool, n, dims, fs) for fs in (False, True)]
    a0 = choices[0]
    require(all(a0[key] == value for key, value in old[0].items()), 'A0 differs from saved task 8 selection')
    require(a0['candidate_scores'] == [item['msic'] for item in pool], 'A0 differs from native candidate scores')
    selected = pool[a0['selected_index']]
    require(selected == trace['final'], 'A0 differs from native B/support/RSS/final score')
    require(a0['minimum_score'] == trace['minimum_msic'] and a0['minimum_q'] == trace['minimum_msic_q'],
            'Native minimum changed')
    require(a0['tolerance'] == trace['one_se_tolerance'] and a0['eligible_indices'] == trace['eligible_candidate_indices'],
            'Native tolerance or eligible order changed')
    caps, budget = trace['caps'], max(1, math.floor(.7 * n))
    grid = [list(s) for s in product(*(range(1, cap + 1) for cap in caps)) if math.prod(s) <= budget]
    require(trace['ordered_size_grid'] == grid == [item['sizes'] for item in pool[1:]], 'Cold grid incomplete or reordered')
    require(trace['adaptive_terminal'] == pool[0], 'Adaptive terminal identity changed')
    for item in pool:
        require(item['q'] <= budget and all(s <= cap for s, cap in zip(item['sizes'], caps)), 'Original capacity violated')
    for B in coefficients:
        require(B.shape == tuple(dims), 'Saved B mode dimensions changed')
    true_support = record['supports_true']
    require([array_sha(np.asarray(s, dtype=int)) for s in true_support] == record['input_fingerprints']['supports_true'],
            'True support fingerprints changed')
    audit = {**identity_row(record), 'candidate_count': len(pool), 'A0_native_exact': True,
             'candidate_B_sha_finite_support_checks': len(pool), 'native_candidate_score_checks': len(pool),
             'trace_sha256': sha(folder / 'trace.json'), 'coefficients_sha256': sha(folder / 'candidate_coefficients.npz'),
             'truth_sha256': sha(folder / 'truth.npz'), 'complete_sha256': sha(folder / 'complete.json'),
             'selection_sha256': sha(folder / 'selection.json'), 'caps': caps, 'Q': budget,
             'test_fingerprint': record['test_fingerprint']['sha256']}
    return {'record': record, 'trace': trace, 'coefficients': coefficients, 'truth': truth,
            'old_choices': old, 'choices': choices, 'audit': audit}


def paired_summary(cases):
    paired = [case for case in cases if all(arm['pmse'] is not None for arm in case['arms'])]
    metrics = ('q', 'q_under', 'q_over', 'ree', 'mode_f1_macro', 'element_f1', 'exact_recovery')
    result = {'datasets': len(cases), 'paired_pmse_available': len(paired),
              'changed_B': sum(case['changed_B'] for case in cases),
              'truth_representable': sum(case['truth_representable'] for case in cases),
              'truth_in_pool': sum(case['truth_in_pool'] for case in cases),
              'arms': [{key: float(np.mean([case['arms'][j][key] for case in cases])) for key in metrics}
                       for j in (0, 1)]}
    for j in (0, 1):
        result['arms'][j]['pmse_available'] = sum(case['arms'][j]['pmse'] is not None for case in cases)
        result['arms'][j]['exact_count'] = sum(case['arms'][j]['exact_recovery'] for case in cases)
        result['arms'][j]['under_count'] = sum(case['arms'][j]['q_under'] for case in cases)
    result['paired_prediction'] = None
    if paired:
        values = np.asarray([[arm['pmse'] for arm in case['arms']] for case in paired])
        result['paired_prediction'] = {
            'A0_mean': float(values[:, 0].mean()), 'A1_mean': float(values[:, 1].mean()),
            'mean_ratio': float(values[:, 1].mean() / values[:, 0].mean()),
            'mean_paired_difference': float(np.mean(values[:, 1] - values[:, 0])),
            'A0_median': float(np.median(values[:, 0])), 'A1_median': float(np.median(values[:, 1])),
            'A0_p95': float(np.quantile(values[:, 0], .95)), 'A1_p95': float(np.quantile(values[:, 1], .95)),
            'improved': int(np.sum(values[:, 1] < values[:, 0])),
            'same': int(np.sum(values[:, 1] == values[:, 0])),
            'worsened': int(np.sum(values[:, 1] > values[:, 0])),
            'scope': 'Identical available case pairs only; coverage-selected historical diagnostic, not promotion evidence'}
    return result


def run(repo, output):
    require(not output.exists(), 'Historical audit output already exists; refusing overwrite')
    base = repo / 'numerical_simulation/outputs_selection_research_task8'
    protocol9 = read(repo / 'doc/taskcard9/PROTOCOL.json')
    plan_sha = sha(base / 'plan.json')
    require(plan_sha == protocol9['historical_plan_sha256'], 'Historical plan changed')
    for relative, digest in protocol9['source_sha256'].items():
        require(sha(repo / 'src' / relative) == digest, 'Production source changed: ' + relative)
        require(sha(repo / 'doc/algorithm_versions/v5.2-search-final/src' / relative) == digest,
                'Frozen v5.2 source changed: ' + relative)
    plan = read(base / 'plan.json')
    expected = {(r['cohort'], r['task_id']): r for r in plan['requests']}
    completion = read(base / 'completion.json')
    require(completion['plan_sha256'] == plan_sha and completion['development'] == completion['confirmation'] == 108,
            'Historical completion binding failed')
    for relative, digest in completion['artifact_sha256'].items():
        require(sha(base / relative) == digest, 'Historical completion artifact drift: ' + relative)
    delivery = read(repo / 'doc/taskcard9/DELIVERY_INDEX.json')['files']
    replay, replay_files = {}, {}
    for path in sorted((repo / 'doc/taskcard9/pool_msic').glob('replay_development_*.json')):
        relative = path.relative_to(repo).as_posix()
        require(sha(path) == delivery[relative]['sha256'], 'Task 9 prediction replay seal changed')
        body = read(path)
        summary = body['summary']
        require(summary['gates_passed'], 'Historical replay identity gate failed')
        record = read(base / summary['cohort'] / f"task_{summary['task_id']:03d}" / 'complete.json')
        require(summary['test_fingerprint'] == record['test_fingerprint'], 'Historical replay test identity mismatch')
        for row in body['candidates']:
            replay[(row['cohort'], row['task_id'], row['B_sha256'])] = row
        replay_files[relative] = sha(path)
    cases, audits, decisions = [], [], []
    lock_sha = sha(base / 'development_lock.json')
    for folder in task_records(base):
        case = audit_case(folder)
        record, trace, choices = case['record'], case['trace'], case['choices']
        request, config = record['request'], record['request']['config']
        require(request == expected[(request['cohort'], request['task_id'])], 'Historical request changed')
        require(record['plan_sha256'] == plan_sha, 'Historical case plan binding failed')
        require(record['development_lock_sha256'] == (lock_sha if request['cohort'] == 'confirmation' else None),
                'Historical confirmation lock binding failed')
        for item in record['actual_imports'].values():
            path = Path(item['path'])
            source = repo / 'doc/algorithm_versions/v5.2-search-final/src'
            require(path.is_relative_to(source), 'Historical import escaped frozen v5.2')
            require(item['sha256'] == protocol9['source_sha256'][path.relative_to(source).as_posix()] == sha(path),
                    'Historical actual import identity changed')
        pool, qt = trace['candidates'], math.prod(config['sparsity'])
        identity = identity_row(record)
        exact_indices = [i for i, c in enumerate(pool) if c['support'] == record['supports_true']]
        arms = []
        for j, choice in enumerate(choices):
            index = choice['selected_index']
            candidate = pool[index]
            arms.append({'arm': f'A{j}', 'selected_index': index, 'q': candidate['q'],
                         'shape': candidate['sizes'], 'support': candidate['support'],
                         'B_sha256': candidate['B_sha256'], 'rss': candidate['rss'],
                         'q_under': int(candidate['q'] < qt), 'q_over': int(candidate['q'] > qt),
                         'ree': float(np.linalg.norm(case['coefficients'][index] - case['truth']) / np.linalg.norm(case['truth'])),
                         **support_metrics(candidate['support'], record['supports_true']),
                         **saved_prediction(record, pool, case['old_choices'], replay, index),
                         'minimum_index': choice['minimum_index'], 'minimum_q': choice['minimum_q'],
                         'tolerance': choice['tolerance'], 'score': choice['selected_score'],
                         'selected_terms': choice['candidate_terms'][index]})
        changed = arms[0]['B_sha256'] != arms[1]['B_sha256']
        category = ('q_changed' if arms[0]['q'] != arms[1]['q'] else 'same_q_shape_changed'
                    if arms[0]['shape'] != arms[1]['shape'] else 'same_q_shape_support_changed'
                    if arms[0]['support'] != arms[1]['support'] else 'identical_model')
        cases.append({**identity, 'truth_representable': qt <= case['audit']['Q'] and
                      all(s <= cap for s, cap in zip(config['sparsity'], trace['caps'])),
                      'truth_in_pool': bool(exact_indices), 'truth_candidate_indices': exact_indices,
                      'changed_B': changed, 'change_category': category, 'arms': arms})
        audits.append(case['audit'])
        decisions.append({**identity, 'choices': choices})
    groups = []
    for cohort in ('development', 'confirmation', 'both'):
        for label, n, snr, representable in [('all', None, None, None), ('n100', 100, None, None),
                ('n200', 200, None, None), ('n400', 400, None, None), ('SNR4', None, 4, None),
                ('SNR4_representable', None, 4, True), ('SNR4_nonrepresentable', None, 4, False)]:
            subset = [c for c in cases if (cohort == 'both' or c['cohort'] == cohort) and
                      (n is None or c['n_train'] == n) and (snr is None or c['snr'] == snr) and
                      (representable is None or c['truth_representable'] == representable)]
            groups.append({'cohort': cohort, 'group': label, **paired_summary(subset)})
    cells = []
    for cell_id in sorted({c['cell_id'] for c in cases}):
        subset = [c for c in cases if c['cell_id'] == cell_id]
        cells.append({'cell_id': cell_id, **{k: subset[0][k] for k in ('n_train', 'snr', 'truth_shape', 'rho')},
                      **paired_summary(subset)})
    new_under = [c for c in cases if c['snr'] == 4 and c['arms'][1]['q_under'] > c['arms'][0]['q_under']]
    lost_exact = [c for c in cases if c['arms'][0]['exact_recovery'] > c['arms'][1]['exact_recovery']]
    gained_exact = [c for c in cases if c['arms'][1]['exact_recovery'] > c['arms'][0]['exact_recovery']]
    summary = {'passed': True, 'datasets': len(cases), 'candidate_count': sum(a['candidate_count'] for a in audits),
               'A0_native_exact_cases': len(audits), 'groups': groups, 'cells': cells,
               'change_categories': dict(Counter(c['change_category'] for c in cases)),
               'new_strong_underselection': new_under, 'lost_exact_cases': lost_exact,
               'gained_exact_cases': gained_exact, 'replay_file_sha256': replay_files,
               'historical_completion_sha256': sha(base / 'completion.json'),
               'historical_plan_sha256': plan_sha, 'completion_artifact_checks': len(completion['artifact_sha256']),
               'source_sha256': protocol9['source_sha256'], 'research_only': True,
               'no_new_inputs_search_fits_or_predictions': True, 'promotion_evidence': False,
               'OLS_scope': 'Archived full-support OLS path audited in source and saved B/RSS identity checked; historical X/y absent, no fresh OLS recomputation'}
    write(output / 'summary.json', summary)
    write(output / 'paired_cases.json', cases)
    write(output / 'choices.json', decisions)
    write(output / 'identity_audit.json', audits)
    write_csv(output / 'selected.csv', [{**{k: v for k, v in c.items() if k != 'arms'}, **arm}
                                      for c in cases for arm in c['arms']])
    print(f'HISTORY_PASS datasets={len(cases)} candidates={summary["candidate_count"]} changed={sum(c["changed_B"] for c in cases)}', flush=True)


if __name__ == '__main__':
    repository = Path(__file__).resolve().parents[2]
    run(repository, repository / 'doc/taskcard11/history')
