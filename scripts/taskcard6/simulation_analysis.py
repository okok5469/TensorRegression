"""Paired descriptive analysis of existing full results; no fitting or tuning."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import numpy as np

from numerical_simulation.version_attribution.run import digest, read, require, sha, seal, write_json


def classify(old_exact, new_exact, delta):
    transition = 'gained_exact' if new_exact > old_exact else ('lost_exact' if new_exact < old_exact else 'same_exact')
    direction = 'equal_pmse' if abs(delta) <= 1e-12 else ('better_pmse' if delta < 0 else 'worse_pmse')
    return transition + '_' + direction


def tail_summary(values):
    values = np.asarray(values, dtype=float)
    positive = np.sort(values[values > 1e-12])[::-1]
    negative = values[values < -1e-12]
    plus, minus, net = float(positive.sum()), float(negative.sum()), float(values.sum())
    result = {'n': len(values), 'improved': int((values < -1e-12).sum()),
              'equal': int((np.abs(values) <= 1e-12).sum()), 'worsened': len(positive),
              'mean': float(values.mean()), 'median': float(np.median(values)),
              'quantiles': {str(q): float(np.quantile(values, q)) for q in (0, .01, .05, .25, .5, .75, .9, .95, .99, 1)},
              'positive_sum': plus, 'negative_sum': minus, 'net_sum': net}
    for count in (1, 5, 10, 20):
        contribution = float(positive[:count].sum())
        result[f'top_{count}_sum'] = contribution
        result[f'top_{count}_positive_share'] = contribution / plus if plus else None
        result[f'top_{count}_net_share'] = contribution / net if abs(net) > 1e-12 else None
    return result


def save_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def population_components(B, truth, rho):
    from numerical_simulation.data_generation import make_ar1_covariance, tensor_signal_variance
    covariance = [make_ar1_covariance(p, r) for p, r in zip(B.shape, rho)]
    inside = (B - truth) * (truth != 0)
    outside = B * (truth == 0)
    total = tensor_signal_variance(B - truth, *covariance)
    first, second = [tensor_signal_variance(E, *covariance) for E in (inside, outside)]
    return {'population_signal_risk': total, 'true_block_coefficient_error_risk': first,
            'false_positive_coefficient_risk': second, 'cross_covariance_term': total - first - second,
            'coefficient_squared_norm': float(np.sum(B**2)),
            'scope': 'Conditional population prediction risk of saved B under declared covariance; not repeated-sampling bias/variance.'}


def analyze(repository, output):
    repository, output = Path(repository), Path(output)
    require(not output.exists(), 'Refusing to overwrite an existing analysis')
    folders = [repository / 'numerical_simulation' / name for name in ('outputs_known_s_full_50', 'outputs_v52_search_final_full')]
    plans = [read(p / 'plan.json') for p in folders]
    require(plans[0]['tasks'] == plans[1]['tasks'] and len(plans[0]['tasks']) == 700, 'Formal task grids differ')
    records, rows = {}, []
    for task in plans[0]['tasks']:
        task_id = task['task_id']
        pair = [read(folder / 'tasks' / f'task_{task_id:06d}.json') for folder in folders]
        for record, plan in zip(pair, plans):
            require(record['record_sha256'] == digest({k: v for k, v in record.items() if k != 'record_sha256'}), 'Original record checksum differs')
            require(record['plan_id'] == plan['plan_id'] and record['task'] == task
                    and record['environment'] == plan['environment']
                    and record['source_fingerprints'] == plan['source_fingerprints'], 'Original identity differs')
        require(pair[0]['input_fingerprints'] == pair[1]['input_fingerprints'], 'Paired input fingerprints differ')
        records[task_id] = pair
        row = {key: task[key] for key in ('task_id', 'scenario_id', 'experiment', 'scenario_value', 'replicate', 'seed', 'test_seed')}
        row.update(n_train=task['config']['n_train'], Q=int(.7 * task['config']['n_train']),
                   input_key=digest(pair[0]['input_fingerprints']))
        for label, record in zip(('old', 'new'), pair):
            result = next(r for r in record['results'] if r['method'] == 'adaptive')
            row.update({label + '_' + key: value for key, value in result['metrics'].items()})
            row[label + '_sic'] = result['sic']
            row[label + '_q_over_Q'] = result['metrics']['q_hat'] / row['Q']
            row[label + '_train_mse'] = result['metrics']['rss_train'] / row['n_train']
            row[label + '_test_minus_train_mse'] = result['metrics']['pmse'] - row[label + '_train_mse']
            row[label + '_caps'] = json.dumps(result['trace']['s_max'])
        for metric in ('pmse', 'signal_mse', 'ree', 'q_hat', 'rss_train', 'exact_recovery', 'mode_f1_macro', 'element_f1'):
            row['delta_' + metric] = row['new_' + metric] - row['old_' + metric]
        row['transition'] = classify(row['old_exact_recovery'], row['new_exact_recovery'], row['delta_pmse'])
        rows.append(row)
    unique = list({r['input_key']: r for r in reversed(rows)}.values())
    unique.sort(key=lambda r: r['task_id'])
    worst = sorted(rows, key=lambda r: (-r['delta_pmse'], r['task_id']))
    best = sorted(rows, key=lambda r: (r['delta_pmse'], r['task_id']))
    candidates = worst[:3] + best[:3]
    gained_worse = [r for r in rows if r['transition'] == 'gained_exact_worse_pmse']
    lost = [r for r in worst if r['transition'].startswith('lost_exact')]
    candidates += gained_worse[:1] + lost[:1]
    case_ids = sorted({r['task_id'] for r in candidates})
    output.mkdir(parents=True)
    write_json(output / 'CASE_SELECTION.json', seal({'rule': 'Before loading model arrays: largest 3 increases, largest 3 decreases, first gained-exact/worse-PMSE, worst lost-exact; all by saved metrics, no tuning.',
                                                   'task_ids': case_ids, 'refitting_allowed': False,
                                                   'reason_no_reconstruction': 'Local training fingerprints do not match formal worker environment.'}))
    cases = []
    for task_id in case_ids:
        pair = records[task_id]
        case = {'task_id': task_id, 'config': pair[0]['task']['config'],
                'paired_metrics': next(r for r in rows if r['task_id'] == task_id)}
        for label, folder, record in zip(('old', 'new'), folders, pair):
            path = folder / 'tasks' / f'task_{task_id:06d}.npz'
            require(sha(path) == record['npz_sha256'], 'Selected NPZ checksum differs')
            with np.load(path, allow_pickle=False) as saved:
                B, truth = saved['B_adaptive'], saved['B_true']
                case[label] = {'npz_sha256': record['npz_sha256'],
                               'supports': [saved[f'S{i}_adaptive'].tolist() for i in (1, 2, 3)],
                               **population_components(B, truth, case['config']['rho'])}
        cases.append(case)
    scenarios = []
    for sid in range(14):
        group = [r for r in rows if r['scenario_id'] == sid]
        scenarios.append({'scenario_id': sid, 'experiment': group[0]['experiment'], 'value': group[0]['scenario_value'],
                          **tail_summary([r['delta_pmse'] for r in group]),
                          'old_exact': sum(r['old_exact_recovery'] for r in group),
                          'new_exact': sum(r['new_exact_recovery'] for r in group)})
    transitions = []
    for key, count in sorted(Counter(r['transition'] for r in rows).items()):
        group = [r for r in rows if r['transition'] == key]
        transitions.append({'transition': key, 'n': count,
                            'mean_delta_pmse': float(np.mean([r['delta_pmse'] for r in group])),
                            'sum_delta_pmse': float(sum(r['delta_pmse'] for r in group)),
                            'mean_delta_signal_mse': float(np.mean([r['delta_signal_mse'] for r in group])),
                            'mean_delta_ree': float(np.mean([r['delta_ree'] for r in group])),
                            'scenario_counts': dict(Counter(str(r['scenario_id']) for r in group))})
    worsening = [r for r in rows if r['delta_pmse'] > 1e-12]
    complexity = []
    for label, group in [('all_worsened', worsening),
                         ('worsened_q_increased', [r for r in worsening if r['delta_q_hat'] > 0]),
                         ('worsened_q_equal', [r for r in worsening if r['delta_q_hat'] == 0]),
                         ('worsened_q_decreased', [r for r in worsening if r['delta_q_hat'] < 0]),
                         ('worsened_near_capacity', [r for r in worsening if r['new_q_over_Q'] >= .9])]:
        complexity.append({'group': label, 'n': len(group),
                           'mean_delta_pmse': float(np.mean([r['delta_pmse'] for r in group])) if group else None,
                           'lower_train_rss_n': sum(r['delta_rss_train'] < -1e-10 for r in group)})
    report = {'interpretation': 'Descriptive paired effects;700 labels include repeated inputs,600 is also not asserted IID.',
              'old_plan_id': plans[0]['plan_id'], 'new_plan_id': plans[1]['plan_id'],
              'all': tail_summary([r['delta_pmse'] for r in rows]),
              'unique_input': tail_summary([r['delta_pmse'] for r in unique]),
              'scenarios': scenarios, 'transitions': transitions, 'complexity': complexity,
              'top_worsened_task_ids': [r['task_id'] for r in worst[:20]],
              'p0_v51_status': 'BLOCKED_NO_VALID_FITS; do not infer numerical delta1/delta2',
              'training_rss_recomputed': False, 'ols_rank_or_condition_recomputed': False,
              'source_sha256': sha(__file__)}
    write_json(output / 'summary.json', report)
    write_json(output / 'selected_cases.json', {'cases': cases})
    save_csv(output / 'paired_old_new_700.csv', rows)
    save_csv(output / 'paired_unique_input_600.csv', unique)
    save_csv(output / 'largest_20_increases.csv', worst[:20])
    save_csv(output / 'recovery_prediction_transitions.csv', transitions)
    print('Saved paired700/unique600 and', len(cases), 'preselected no-refit model diagnostics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.repository, args.output_dir)
