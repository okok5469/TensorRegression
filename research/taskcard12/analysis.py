"""Frozen paired inference; pressure repetitions never change primary weights."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import beta


def require(condition, message):
    if not condition:
        raise ValueError(message)


def loss_upper_bound(losses, denominator):
    require(0 <= losses <= denominator, 'Invalid binomial counts')
    if denominator == 0:
        return None
    return 1. if losses == denominator else float(beta.ppf(.95, losses + 1, denominator - losses))


def _binomial_interval(count, n, alpha):
    return (0. if count == 0 else float(beta.ppf(alpha / 2, count, n - count + 1)),
            1. if count == n else float(beta.ppf(1 - alpha / 2, count + 1, n - count)))


def _group(records):
    n = len(records)
    if not n:
        return {'count': 0, 'exact_delta': None, 'q_under_delta': None,
                'conditional_exact_loss_upper95': None}
    pmse = np.asarray([[r['metrics'][a]['pmse'] for a in ('A0', 'A1')] for r in records])
    exact = np.asarray([[r['metrics'][a]['exact_recovery'] for a in ('A0', 'A1')] for r in records])
    q = np.asarray([[r['metrics'][a]['q'] for a in ('A0', 'A1')] for r in records])
    truth_q = np.asarray([math.prod(r['request']['config']['sparsity']) for r in records])
    losses = int(np.sum((exact[:, 0] == 1) & (exact[:, 1] == 0)))
    gains = int(np.sum((exact[:, 0] == 0) & (exact[:, 1] == 1)))
    denominator = int(exact[:, 0].sum())
    gain_ci, loss_ci = _binomial_interval(gains, n, .025), _binomial_interval(losses, n, .025)
    return {'count': n, 'pmse_mean_ratio': float(pmse[:, 1].mean() / pmse[:, 0].mean()),
            'pmse_p95_ratio': float(np.quantile(pmse[:, 1], .95) / np.quantile(pmse[:, 0], .95)),
            'exact_A0': denominator, 'exact_A1': int(exact[:, 1].sum()),
            'exact_delta': float(np.mean(exact[:, 1] - exact[:, 0])),
            'exact_losses': losses, 'exact_gains': gains,
            'exact_delta_conservative_ci95': [gain_ci[0] - loss_ci[1], gain_ci[1] - loss_ci[0]],
            'exact_loss_denominator': denominator,
            'conditional_exact_loss_rate': losses / denominator if denominator else None,
            'conditional_exact_loss_upper95': loss_upper_bound(losses, denominator),
            'q_under_A0': int(np.sum(q[:, 0] < truth_q)),
            'q_under_A1': int(np.sum(q[:, 1] < truth_q)),
            'q_under_delta': float(np.mean((q[:, 1] < truth_q).astype(float) - (q[:, 0] < truth_q)))}


def _paired_bootstrap(records, draws, rng):
    totals = np.zeros((draws, 2))
    for cell in sorted({r['request']['cell_id'] for r in records}):
        values = np.asarray([[r['metrics'][a]['pmse'] for a in ('A0', 'A1')]
                             for r in records if r['request']['cell_id'] == cell])
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        totals += values[indices].sum(axis=1)
    ratios = totals[:, 1] / totals[:, 0]
    differences = (totals[:, 1] - totals[:, 0]) / len(records)
    return {'mean_ratio_ci95': np.quantile(ratios, [.025, .975]).tolist(),
            'mean_difference_ci95': np.quantile(differences, [.025, .975]).tolist()}


def analyze(records, protocol):
    records = sorted(records, key=lambda r: r['request']['task_id'])
    expected = protocol['requests']
    require(len(records) == 380 and [r['request'] for r in records] == expected,
            'Complete exact frozen380 requests required before inference')
    require(len({r['request']['task_id'] for r in records}) == 380, 'Duplicate requests')
    for record in records:
        for arm in ('A0', 'A1'):
            m = record['metrics'][arm]
            require(math.isfinite(m['pmse']) and m['pmse'] > 0, 'Invalid PMSE')
            require(m['exact_recovery'] in (0, 1) and isinstance(m['q'], (int, np.integer)) and m['q'] > 0,
                    'Invalid Exact or q')
    balanced = [r for r in records if not r['request']['pressure']]
    require(len(balanced) == 360, 'Balanced primary count differs')
    require(all(sum(r['request']['cell_id'] == c for r in balanced) == 10 for c in range(36)),
            'Balanced cell coverage differs')
    groups = {'balanced360': balanced,
              **{'n' + str(n): [r for r in balanced if r['request']['config']['n_train'] == n]
                 for n in (100, 200, 400)},
              'stress30': [r for r in records if r['request']['cell_id'] == 21],
              'n200_strong_representable': [r for r in records if r['request']['config']['n_train'] == 200
                  and r['request']['config']['snr'] == 4 and r['recovery_evidence']['truth_shape_native_representable']],
              'legacy_all_strong120': [r for r in balanced if r['request']['config']['snr'] == 4]}
    require(len(groups['stress30']) == 30, 'Stress cell coverage differs')
    summaries = {key: _group(rows) for key, rows in groups.items()}
    worst = max(_group([r for r in balanced if r['request']['cell_id'] == c])['pmse_mean_ratio'] for c in range(36))
    values = {'n100_pmse_ratio': summaries['n100']['pmse_mean_ratio'],
              'balanced_pmse_ratio': summaries['balanced360']['pmse_mean_ratio'],
              'n100_p95_ratio': summaries['n100']['pmse_p95_ratio'],
              'worst_balanced_cell_pmse_ratio': worst,
              'n200_pmse_ratio': summaries['n200']['pmse_mean_ratio'],
              'n400_pmse_ratio': summaries['n400']['pmse_mean_ratio'],
              'n400_exact_delta': summaries['n400']['exact_delta'],
              'n200_strong_exact_delta': summaries['n200_strong_representable']['exact_delta'],
              'n200_strong_q_under_delta': summaries['n200_strong_representable']['q_under_delta'],
              'stress_exact_delta': summaries['stress30']['exact_delta'],
              'stress_conditional_exact_loss_upper95': summaries['stress30']['conditional_exact_loss_upper95']}
    settings = protocol['analysis']
    require(set(settings['thresholds']) == set(values), 'Frozen inference thresholds differ')
    comparisons = {'max': lambda value, threshold: value <= threshold,
                   'min': lambda value, threshold: value >= threshold}
    gates = {name: {'value': values[name], 'threshold': threshold['value'], 'direction': threshold['direction'],
                   'passed': values[name] is not None and comparisons[threshold['direction']](values[name], threshold['value'])}
             for name, threshold in settings['thresholds'].items()}
    rng = np.random.default_rng(settings['bootstrap_seed'])
    intervals = {key: _paired_bootstrap(groups[key], settings['bootstrap_draws'], rng)
                 for key in ('balanced360', 'n100', 'n200', 'n400', 'stress30')}
    stable = all(intervals[key]['mean_ratio_ci95'][1] < 1 for key in ('balanced360', 'n100'))
    legacy_names = ['n100_pmse_ratio', 'balanced_pmse_ratio', 'n100_p95_ratio',
                    'worst_balanced_cell_pmse_ratio', 'n400_pmse_ratio', 'n400_exact_delta']
    legacy = {name: gates[name] for name in legacy_names}
    legacy['strong_under_delta'] = {'value': summaries['legacy_all_strong120']['q_under_delta'],
                                   'threshold': .05, 'direction': 'max',
                                   'passed': summaries['legacy_all_strong120']['q_under_delta'] <= .05}
    return {'status': 'COMPLETE380_PAIRED_INFERENCE', 'groups': summaries, 'gates': gates,
            'all_new_gates_passed': all(g['passed'] for g in gates.values()),
            'paired_bootstrap': intervals, 'bootstrap_draws': settings['bootstrap_draws'],
            'bootstrap_seed': settings['bootstrap_seed'], 'stable_paired_mean_gain': stable,
            'eligible_for_version_review': stable and all(g['passed'] for g in gates.values()),
            'task11_thresholds_on_new_balanced_data': legacy,
            'task11_historical_failed_strong_under_delta': .125,
            'release_approved': False,
            'decision_scope': 'Scientific review input only; engineering/empirical evidence and explicit user approval remain required.',
            'uncertainty_scope': 'Fixed-design paired resampling; conservative Exact difference intervals and one-sided conditional-loss bound; no broad noninferiority theorem.'}
