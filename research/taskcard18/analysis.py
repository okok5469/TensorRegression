"""Saved-coefficient metrics and strict matched-input summaries; no fitting."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def set_metrics(estimated, truth, total):
    overlap = len(estimated & truth); found = len(estimated); positives = len(truth)
    return {'precision': overlap/found if found else float('nan'),
            'recall': overlap/positives if positives else float('nan'),
            'fdr': (found-overlap)/found if found else 0.,
            'f1': 2*overlap/(found+positives) if found+positives else 0.,
            'fpr': len(estimated-truth)/(total-positives) if total > positives else float('nan')}


def support_metrics(B, B_true, declared, threshold=1e-8):
    B = np.asarray(B); B_true = np.asarray(B_true)
    if B.ndim != 3 or B.shape != B_true.shape or not np.isfinite(B).all() or not np.isfinite(B_true).all():
        raise ValueError('Finite matching three-dimensional coefficient arrays required')
    if not np.linalg.norm(B_true) > 0 or len(declared) != 3:
        raise ValueError('Nonzero true coefficient and three declared axis supports required')
    actual = np.abs(B) > threshold; truth = np.abs(B_true) > threshold
    E = set(np.flatnonzero(actual)); T = set(np.flatnonzero(truth))
    metrics = {f'element_{k}': v for k, v in set_metrics(E, T, B.size).items()}
    projected = [set(np.flatnonzero(np.any(actual, axis=tuple(j for j in range(3) if j != i)))) for i in range(3)]
    true_modes = [set(np.flatnonzero(np.any(truth, axis=tuple(j for j in range(3) if j != i)))) for i in range(3)]
    D = [set(map(int, S)) for S in declared]
    q_rect = int(np.prod([len(S) for S in D]))
    rectangle_truth_overlap = int(np.prod([len(S & Tm) for S, Tm in zip(D, true_modes)]))
    metrics.update({'element_nonzero': len(E), 'q_rect': q_rect,
                    'exact_recovery': int(D == true_modes), 'declared_equals_projection': int(D == projected),
                    'legacy_element_precision': rectangle_truth_overlap/q_rect if q_rect else float('nan'),
                    'legacy_element_recall': rectangle_truth_overlap/np.prod([len(S) for S in true_modes])})
    for i, (S, Tm, p) in enumerate(zip(D, true_modes, B.shape), 1):
        for metric, value in set_metrics(S, Tm, p).items():
            metrics[f'mode_{metric}_{i}'] = value
        metrics[f's{i}_hat'] = len(S)
    for metric in ('precision', 'recall', 'fdr', 'f1', 'fpr'):
        metrics[f'mode_{metric}_macro'] = float(np.mean([metrics[f'mode_{metric}_{i}'] for i in (1, 2, 3)]))
    metrics['ree'] = float(np.linalg.norm(B-B_true)/np.linalg.norm(B_true))
    return metrics


def paired_statistics(a, b):
    if set(a) != set(b) or len(a) < 2:
        raise ValueError('Complete identical unique-input keys and at least2 replicates required')
    differences = np.array([a[key]-b[key] for key in sorted(a)])
    if not np.isfinite(list(a.values())).all() or not np.isfinite(list(b.values())).all():
        raise ValueError('Every declared paired replicate must be finite')
    n = len(differences); mean = float(differences.mean()); sd = float(differences.std(ddof=1)); mcse = sd/np.sqrt(n)
    width = float(t.ppf(0.975, n-1))*mcse
    return {'n': n, 'mean_difference': mean, 'sd_difference': sd, 'mcse': mcse,
            'ci_low': mean-width, 'ci_high': mean+width, 'df': n-1}


def wilson(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError('Wilson interval requires a valid success count and positive denominator')
    z = 1.959963984540054; p = successes/n; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return (0.0 if successes == 0 else center-width,
            1.0 if successes == n else center+width)


def save_csv(path, rows):
    if not rows:
        raise ValueError('No rows to export')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def collect(history):
    from .workflow import repository_root, paths, source_identity
    from numerical_simulation.known_s_cluster.run import load_plan, file_digest
    root = repository_root(); historical, output = paths(); plan = load_plan(historical)
    with (root/'doc/taskcard18/input_identity_map.csv').open(encoding='utf-8', newline='') as stream:
        mapping = {int(row['task_id']): row for row in csv.DictReader(stream)}
    rows = []
    for task in plan['tasks']:
        task_id = task['task_id']; identity = mapping[task_id]
        record = json.loads((historical/f'tasks/task_{task_id:06d}.json').read_text(encoding='utf-8'))
        with np.load(historical/'tasks'/record['arrays_file'], allow_pickle=False) as saved:
            truth = saved['B_true']
            for method in ('adaptive', 'oracle', 'lasso', 'glasso'):
                old = next(r for r in record['results'] if r['method'] == method)
                if history or method in ('adaptive', 'oracle'):
                    B = saved['B_'+method]; S = [saved[f'S{i}_{method}'] for i in (1, 2, 3)]
                    metric = old['metrics']; method_id = ('v52_search_final_original_msic' if method == 'adaptive' else 'oracle_ols_frozen') if method in ('adaptive', 'oracle') else 'legacy_'+method
                    stage = 'frozen_verified' if method in ('adaptive', 'oracle') else 'legacy_uncertified'
                else:
                    from .accept import validate_fit
                    uid = identity['unique_input_id']
                    dest = output/'fits'/f'{uid}_{method}'
                    metadata, B, S = validate_fit(task, record, method)
                    evaluation = json.loads((output/'evaluations'/f'{uid}_{method}.json').read_text(encoding='utf-8'))
                    if evaluation['identity'] != metadata['identity'] or evaluation['fit_npz_sha256'] != metadata['npz_sha256'] or evaluation['unique_input_id'] != uid or evaluation['method_id'] != metadata['method_id']:
                        raise ValueError('Evaluation does not belong to the certified input/method/model')
                    metric = evaluation['metrics']; method_id = metadata['method_id']; stage = 'certified_new'
                common = {k: task[k] for k in ('task_id', 'scenario_id', 'replicate', 'experiment', 'scenario_value')}
                common.update({k: identity[k] for k in ('unique_input_id', 'config_id', 'canonical_task_id')})
                common.update({'method': method, 'method_id': method_id, 'evidence_stage': stage,
                               **{k: metric[k] for k in ('pmse', 'signal_mse')}, **support_metrics(B, truth, S)})
                for threshold, suffix in ((1e-10, '1e10'), (1e-6, '1e6')):
                    sensitivity = support_metrics(B, truth, S, threshold)
                    common.update({k+'_'+suffix: sensitivity[k] for k in ('element_nonzero', 'element_recall', 'element_fdr', 'element_f1')})
                rows.append(common)
    return rows


def summarize(rows, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[(row[grouping], row['method'])].append(row)
    metrics = [k for k in rows[0] if k not in ('task_id', 'scenario_id', 'replicate', 'experiment', 'scenario_value', 'unique_input_id', 'config_id', 'canonical_task_id', 'method', 'method_id', 'evidence_stage')]
    result = []
    for (key, method), values in sorted(groups.items()):
        item = {grouping: key, 'method': method, 'method_id': values[0]['method_id'], 'evidence_stage': values[0]['evidence_stage'], 'n': len(values)}
        for metric in metrics:
            array = np.array([r[metric] for r in values]); valid = array[np.isfinite(array)]
            item.update({metric+'_mean': float(valid.mean()) if len(valid) else '', metric+'_sd': float(valid.std(ddof=1)) if len(valid)>1 else '', metric+'_valid_n': len(valid)})
            if metric in ('pmse', 'ree'):
                item.update({metric+'_median': float(np.median(valid)), metric+'_q1': float(np.quantile(valid, .25)), metric+'_q3': float(np.quantile(valid, .75))})
        successes = sum(v['exact_recovery'] for v in values)
        lo, hi = wilson(successes, len(values))
        item.update({'exact_successes': successes, 'exact_wilson_low': lo, 'exact_wilson_high': hi})
        result.append(item)
    return result


def build(history):
    from .workflow import repository_root
    root = repository_root(); rows = collect(history)
    # Every label remains in the appendix; primary units use complete input identity.
    unique = {}; paired = []
    for row in rows:
        key = (row['unique_input_id'], row['method'])
        if key in unique:
            for metric in ('pmse', 'ree', 'element_f1', 'exact_recovery'):
                if unique[key][metric] != row[metric]:
                    raise ValueError('Alias result disagreement')
        else: unique[key] = row
    dedup = list(unique.values())
    if len(dedup) != 2400 or len({r['config_id'] for r in dedup}) != 10:
        raise ValueError('Expected600 complete inputs and10 configurations')
    for config in sorted({r['config_id'] for r in dedup}):
        subset = [r for r in dedup if r['config_id'] == config]
        a = {r['unique_input_id']: r['pmse'] for r in subset if r['method']=='adaptive'}
        for method in ('lasso', 'glasso', 'oracle'):
            b = {r['unique_input_id']: r['pmse'] for r in subset if r['method']==method}
            paired.append({'config_id': config, 'comparison': 'adaptive_minus_'+method, 'evidence_stage': 'historical_unrepaired_comparison' if history and method!='oracle' else 'verified', **paired_statistics(a, b)})
    label = 'historical' if history else 'certified'
    out = root/'doc/taskcard18/results'; out.mkdir(parents=True, exist_ok=True)
    save_csv(out/f'{label}_label_metrics.csv', rows)
    save_csv(out/f'{label}_unique_metrics.csv', dedup)
    save_csv(out/f'{label}_config_summary.csv', summarize(dedup, 'config_id'))
    save_csv(out/f'{label}_14_label_summary.csv', summarize(rows, 'scenario_id'))
    save_csv(out/f'{label}_paired_pmse.csv', paired)
    print(label, 'labels', len(rows), 'unique_method_rows', len(dedup), 'config_method_cells', 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--history', action='store_true', help='Explicit old-comparator appendix; never a certified main comparison')
    args = parser.parse_args(); build(args.history)
