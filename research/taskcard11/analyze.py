"""Paired Task11 statistics; no fitting, candidate selection or data generation."""
from collections import defaultdict
import numpy as np


def validate_complete(records, requests):
    expected = {r['task_id']: r for r in requests}
    actual = {r['request']['task_id']: r['request'] for r in records}
    if len(expected) != 96 or len(records) != 96 or len(actual) != 96 or actual != expected:
        raise ValueError('Exact coverage of the frozen 96 requests is required')
    counts = defaultdict(int)
    for row in records:
        counts[row['request']['cell_id']] += 1
        for label in ('A0', 'A1'):
            m = row['metrics'][label]
            vals = [m[k] for k in ('pmse', 'q', 'exact_recovery', 'ree', 'element_f1')]
            if not np.all(np.isfinite(vals)) or m['pmse'] <= 0:
                raise ValueError('Invalid evaluation metric; no dropping or imputation')
    if len(counts) != 24 or set(counts.values()) != {4}:
        raise ValueError('All 24 complete four-replicate cells are required')


def _values(records, label, key):
    return np.asarray([r['metrics'][label][key] for r in records], dtype=float)


def evaluate_gates(records, requests):
    validate_complete(records, requests)
    small = [r for r in records if r['request']['config']['n_train'] == 100]
    large = [r for r in records if r['request']['config']['n_train'] == 400]
    strong = [r for r in records if r['request']['config']['snr'] == 4.]
    if (len(small), len(large), len(strong)) != (48, 48, 32):
        raise ValueError('Frozen n/SNR denominators do not match')
    ratio = lambda rows: float(_values(rows, 'A1', 'pmse').mean() / _values(rows, 'A0', 'pmse').mean())
    under = lambda label: np.mean([r['metrics'][label]['q'] < np.prod(r['request']['config']['sparsity']) for r in strong])
    cells = defaultdict(list)
    for row in records:
        cells[row['request']['cell_id']].append(row)
    specifications = {
        'n100_pmse_ratio': (ratio(small), .95, 'max', 48),
        'overall_pmse_ratio': (ratio(records), .98, 'max', 96),
        'n100_p95_ratio': (float(np.quantile(_values(small, 'A1', 'pmse'), .95) /
                                  np.quantile(_values(small, 'A0', 'pmse'), .95)), 1.05, 'max', 48),
        'worst_cell_pmse_ratio': (max(map(ratio, cells.values())), 1.15, 'max', 24),
        'n400_pmse_ratio': (ratio(large), 1.03, 'max', 48),
        'n400_exact_delta': (float((_values(large, 'A1', 'exact_recovery') -
                                  _values(large, 'A0', 'exact_recovery')).mean()), -.05, 'min', 48),
        'strong_under_delta': (float(under('A1') - under('A0')), .05, 'max', 32),
    }
    criteria = {name: {'value': v, 'threshold': t, 'direction': direction, 'denominator': denominator,
                      'passed': bool(v <= t if direction == 'max' else v >= t)}
                for name, (v, t, direction, denominator) in specifications.items()}
    return {'passed': all(c['passed'] for c in criteria.values()), 'criteria': criteria,
            'engineering_correctness': 'Evaluated separately; statistical passage cannot override it'}


def paired_summary(records):
    a, b = (_values(records, label, 'pmse') for label in ('A0', 'A1'))
    delta = b - a
    tolerance = 1e-12 * np.maximum(1., np.maximum(abs(a), abs(b)))
    arms = {}
    for label in ('A0', 'A1'):
        pmse = _values(records, label, 'pmse')
        q = _values(records, label, 'q')
        true_q = np.array([np.prod(r['request']['config']['sparsity']) for r in records])
        arms[label] = {'pmse_mean': float(pmse.mean()), 'pmse_median': float(np.median(pmse)),
                       'pmse_p95': float(np.quantile(pmse, .95)),
                       'exact_count': int(_values(records, label, 'exact_recovery').sum()),
                       'ree_mean': float(_values(records, label, 'ree').mean()),
                       'element_f1_mean': float(_values(records, label, 'element_f1').mean()),
                       'q_mean': float(q.mean()), 'under_count': int((q < true_q).sum()),
                       'over_count': int((q > true_q).sum())}
    improvement = np.sort(np.maximum(-delta, 0.))[::-1]
    total = float(improvement.sum())
    return {'n': len(records), 'arms': arms, 'ratio': float(b.mean()/a.mean()),
            'mean_delta': float(delta.mean()), 'median_delta': float(np.median(delta)),
            'improved': int((delta < -tolerance).sum()), 'same': int((abs(delta) <= tolerance).sum()),
            'worsened': int((delta > tolerance).sum()), 'max_case_ratio': float(np.max(b/a)),
            'top1_fraction_gross_improvement': float(improvement[:1].sum()/total) if total else None,
            'top5_fraction_gross_improvement': float(improvement[:5].sum()/total) if total else None}


def paired_uncertainty(records, draws=20000, seed=11011011):
    """Stratified paired bootstrap: fixed cells, resample whole independent cases."""
    rng = np.random.default_rng(seed)
    groups = {'overall': records,
              'n100': [r for r in records if r['request']['config']['n_train'] == 100],
              'n400': [r for r in records if r['request']['config']['n_train'] == 400]}
    result = {'draws': draws, 'seed': seed, 'method': 'paired case bootstrap within each fixed scenario'}
    for name, rows in groups.items():
        cells = defaultdict(list)
        for i, row in enumerate(rows):
            cells[row['request']['cell_id']].append(i)
        indices = np.concatenate([rng.choice(ids, size=(draws, len(ids)), replace=True)
                                  for _, ids in sorted(cells.items())], axis=1)
        a, b = (_values(rows, label, 'pmse')[indices].mean(axis=1) for label in ('A0', 'A1'))
        result[name] = {'ratio_ci95': np.quantile(b/a, [.025, .975]).tolist(),
                        'delta_ci95': np.quantile(b-a, [.025, .975]).tolist()}
    result['stable_gain'] = all(result[g]['ratio_ci95'][1] < 1. for g in ('overall', 'n100'))
    result['limitation'] = 'Four repetitions per cell; conditional descriptive uncertainty, not uniform noninferiority proof'
    return result
