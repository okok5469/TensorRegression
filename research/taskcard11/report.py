"""Complete 96-case Task11 reports; no search, fitting, prediction or selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from research.taskcard11 import analyze
from scripts.taskcard7.lemon import require


def change_category(a0, a1):
    checks = [('q_changed', a0['q'] != a1['q']),
              ('same_q_shape_changed', a0['sizes'] != a1['sizes']),
              ('same_q_shape_support_changed', a0['support'] != a1['support']),
              ('same_support_coefficients_changed', a0['B_sha256'] != a1['B_sha256'])]
    return next((name for name, changed in checks if changed), 'identical_model')


def _identity(record):
    request, evidence = record['request'], record['recovery_evidence']
    config = request['config']
    return {'task_id': request['task_id'], 'cell_id': request['cell_id'], 'replicate': request['replicate'],
            'n_train': config['n_train'], 'snr': config['snr'], 'rho': config['rho'][0],
            'truth_shape': config['sparsity'], 'q_true': evidence['q_true'],
            'truth_representable': evidence['truth_shape_native_representable'],
            'truth_support_in_pool': evidence['truth_support_found_full_pool'],
            'seed': request['seed'], 'test_seed': request['test_seed']}


def _validate_summary(record, summary):
    require(summary['config'] == record['request']['config'], 'Summary configuration differs from request')
    pool = summary['full_trace']['candidates']
    evidence = record['recovery_evidence']
    require(evidence['truth_sizes'] == record['request']['config']['sparsity'] and
            evidence['q_true'] == math.prod(evidence['truth_sizes']), 'Truth shape/q differs')
    require(evidence['truth_support_found_full_pool'] ==
            any(c['support'] == evidence['truth_support'] for c in pool), 'Truth pool evidence differs')
    for label in ('A0', 'A1'):
        choice, metric = summary['choices'][label], record['metrics'][label]
        index = choice['selected_index']
        require(isinstance(index, int) and 0 <= index < len(pool), 'Selected candidate index invalid')
        candidate, arm = pool[index], summary['arms'][label]
        require(arm['full_candidate_index'] == index, 'Arm candidate index differs')
        for field in ('q', 'sizes', 'rss', 'support', 'B_sha256'):
            require(candidate[field] == arm[field] == metric[field], 'Selected candidate identity differs: ' + field)
        require(candidate['q'] == math.prod(candidate['sizes']), 'Selected q differs from shape')
        terms = choice['candidate_terms'][index]
        require(terms['score'] == choice['candidate_scores'][index] == choice['selected_score'],
                'Selected score/decomposition differs')
        require(all(np.isfinite(metric[k]) for k in ('mode_f1_macro', 'rss')) and
                all(np.isfinite(v) for v in terms.values()), 'Nonfinite selected evidence')


def _group_specs(records):
    dimensions = {
        'all': lambda r: 'all',
        'n': lambda r: str(r['request']['config']['n_train']),
        'snr': lambda r: str(r['request']['config']['snr']),
        'rho': lambda r: str(r['request']['config']['rho'][0]),
        'truth_shape': lambda r: 'x'.join(map(str, r['request']['config']['sparsity'])),
        'representable': lambda r: str(r['recovery_evidence']['truth_shape_native_representable']),
        'truth_in_pool': lambda r: str(r['recovery_evidence']['truth_support_found_full_pool']),
        'n_snr': lambda r: f"{r['request']['config']['n_train']}|{r['request']['config']['snr']}",
        'n_rho': lambda r: f"{r['request']['config']['n_train']}|{r['request']['config']['rho'][0]}",
        'n_truth_shape': lambda r: f"{r['request']['config']['n_train']}|" + 'x'.join(map(str, r['request']['config']['sparsity'])),
        'n_representable': lambda r: f"{r['request']['config']['n_train']}|{r['recovery_evidence']['truth_shape_native_representable']}",
        'snr_representable': lambda r: f"{r['request']['config']['snr']}|{r['recovery_evidence']['truth_shape_native_representable']}",
    }
    groups = []
    for kind, key in dimensions.items():
        buckets = defaultdict(list)
        for record in records:
            buckets[key(record)].append(record)
        groups.extend((kind, value, subset) for value, subset in sorted(buckets.items()))
    return groups


def _group_rows(kind, value, records):
    stats = analyze.paired_summary(records)
    common = {k: v for k, v in stats.items() if k not in ('arms', 'n')}
    rows = []
    for label in ('A0', 'A1'):
        arm = stats['arms'][label]
        rows.append({'group_kind': kind, 'group_value': value, 'datasets': len(records), 'arm': label,
                     **arm, 'exact_rate': arm['exact_count'] / len(records),
                     'q_under_rate': arm['under_count'] / len(records), 'q_over_rate': arm['over_count'] / len(records),
                     'mode_f1_macro_mean': float(np.mean([r['metrics'][label]['mode_f1_macro'] for r in records])),
                     'q_median': float(np.median([r['metrics'][label]['q'] for r in records])),
                     'q_p95': float(np.quantile([r['metrics'][label]['q'] for r in records], .95)), **common})
    return rows


def assemble(records, summaries, requests):
    """Pure aggregation; reject partial data and every identity mismatch."""
    analyze.validate_complete(records, requests)
    require(set(summaries) == {r['task_id'] for r in requests}, 'Summary case coverage differs')
    require(all(r['request']['cohort'] == 'validation' for r in records), 'Only formal validation is reportable')
    records = sorted(records, key=lambda r: r['request']['task_id'])
    selected, pairs, truths = [], [], []
    for record in records:
        ident = _identity(record)
        summary = summaries[ident['task_id']]
        _validate_summary(record, summary)
        a0, a1 = (record['metrics'][label] for label in ('A0', 'A1'))
        category = change_category(a0, a1)
        delta = a1['pmse'] - a0['pmse']
        tolerance = 1e-12 * max(1., abs(a0['pmse']), abs(a1['pmse']))
        prediction = 'improved' if delta < -tolerance else 'worsened' if delta > tolerance else 'same'
        pair = {**ident, 'change_category': category, 'B_changed': a0['B_sha256'] != a1['B_sha256'],
                'selected_index_changed': summary['choices']['A0']['selected_index'] != summary['choices']['A1']['selected_index'],
                'prediction_change': prediction, 'pmse_delta': delta, 'pmse_ratio': a1['pmse'] / a0['pmse'],
                'q_delta': a1['q'] - a0['q'], 'exact_delta': a1['exact_recovery'] - a0['exact_recovery'],
                'ree_delta': a1['ree'] - a0['ree'], 'mode_f1_delta': a1['mode_f1_macro'] - a0['mode_f1_macro'],
                'element_f1_delta': a1['element_f1'] - a0['element_f1'],
                'new_q_under': a0['q'] >= ident['q_true'] and a1['q'] < ident['q_true']}
        for label, metric in (('A0', a0), ('A1', a1)):
            choice = summary['choices'][label]
            index = choice['selected_index']
            terms = choice['candidate_terms'][index]
            row = {**ident, 'arm': label, 'selected_index': index,
                   'candidate_source': summary['arms'][label]['candidate_source'],
                   **{k: metric[k] for k in ('q', 'sizes', 'support', 'B_sha256', 'rss', 'pmse', 'ree',
                                           'exact_recovery', 'mode_f1_macro', 'element_f1')},
                   'q_under': int(metric['q'] < ident['q_true']), 'q_over': int(metric['q'] > ident['q_true']),
                   'q_over_Q': metric['q'] / record['recovery_evidence']['Q_native'],
                   **terms, **{k: choice[k] for k in ('minimum_index', 'minimum_q', 'minimum_score', 'tolerance')},
                   'band_changes_argmin': index != choice['minimum_index']}
            selected.append(row)
            pair.update({label + '_' + k: v for k, v in row.items() if k not in ident and k != 'arm'})
        pairs.append(pair)
        evidence = record['recovery_evidence']
        exact_indices = [i for i, c in enumerate(summary['full_trace']['candidates'])
                         if c['support'] == evidence['truth_support']]
        truths.append({**ident, 'truth_support': evidence['truth_support'],
                       'truth_shape_in_pool': evidence['truth_shape_found_full_pool'],
                       'truth_candidate_indices': exact_indices,
                       'A0_selected_index': summary['choices']['A0']['selected_index'],
                       'A1_selected_index': summary['choices']['A1']['selected_index'],
                       'A0_exact': a0['exact_recovery'], 'A1_exact': a1['exact_recovery'],
                       'truth_selected_A0': summary['choices']['A0']['selected_index'] in exact_indices,
                       'truth_selected_A1': summary['choices']['A1']['selected_index'] in exact_indices,
                       'truth_abandoned_by_FS': bool(exact_indices) and a0['exact_recovery'] > a1['exact_recovery'],
                       'truth_gained_by_FS': bool(exact_indices) and a1['exact_recovery'] > a0['exact_recovery']})
    groups, distributions = [], []
    for kind, value, subset in _group_specs(records):
        groups.extend(_group_rows(kind, value, subset))
        for label in ('A0', 'A1'):
            counts = Counter((tuple(r['metrics'][label]['sizes']), r['metrics'][label]['q']) for r in subset)
            distributions.extend({'group_kind': kind, 'group_value': value, 'datasets': len(subset),
                'arm': label, 'shape': list(shape), 'q': q, 'count': count, 'fraction': count / len(subset)}
                for (shape, q), count in sorted(counts.items()))
    cells = []
    for cell in sorted({r['request']['cell_id'] for r in records}):
        subset = [r for r in records if r['request']['cell_id'] == cell]
        config = subset[0]['request']['config']
        cells.extend({'cell_id': cell, 'n_train': config['n_train'], 'snr': config['snr'],
                      'rho': config['rho'][0], 'truth_shape': config['sparsity'], **row}
                     for row in _group_rows('cell', str(cell), subset))
    adverse_predicates = {'pmse_worsened': lambda r: r['prediction_change'] == 'worsened',
                          'exact_lost': lambda r: r['exact_delta'] < 0,
                          'ree_worsened': lambda r: r['ree_delta'] > 0,
                          'mode_f1_worsened': lambda r: r['mode_f1_delta'] < 0,
                          'element_f1_worsened': lambda r: r['element_f1_delta'] < 0,
                          'new_strong_underselection': lambda r: r['snr'] == 4. and r['new_q_under']}
    summary = {'status': 'COMPLETE_96_PAIRED_CASES', 'datasets': len(records), 'cells': 24,
               'overall': analyze.paired_summary(records),
               'by_n': {str(n): analyze.paired_summary([r for r in records if r['request']['config']['n_train'] == n])
                        for n in (100, 400)},
               'change_categories': dict(Counter(r['change_category'] for r in pairs)),
               'changed_B_cases': sum(r['B_changed'] for r in pairs),
               'truth_in_pool': sum(r['truth_support_in_pool'] for r in truths),
               'truth_abandoned_cases': [r['task_id'] for r in truths if r['truth_abandoned_by_FS']],
               'truth_gained_cases': [r['task_id'] for r in truths if r['truth_gained_by_FS']],
               'adverse_task_ids': {name: [r['task_id'] for r in pairs if predicate(r)]
                                    for name, predicate in adverse_predicates.items()},
               'largest_improvement_tasks': [r['task_id'] for r in sorted(
                   (r for r in pairs if r['prediction_change'] == 'improved'), key=lambda r: r['pmse_delta'])[:5]],
               'largest_deterioration_tasks': [r['task_id'] for r in sorted(
                   (r for r in pairs if r['prediction_change'] == 'worsened'), key=lambda r: -r['pmse_delta'])[:5]],
               'worst_cell': max((r for r in cells if r['arm'] == 'A1'), key=lambda r: r['ratio']),
               'scope': 'All 96 independent formal pairs; Pilot and historical data excluded; no outcome-based exclusions',
               'research_only': True, 'release_approved': False}
    return {'tables': {'selected_metrics': selected, 'paired_changes': pairs, 'group_metrics': groups,
                       'cell_metrics': cells, 'selection_changes': [r for r in pairs if r['change_category'] != 'identical_model'
                                                                 or r['selected_index_changed']],
                       'shape_q_distribution': distributions, 'truth_pool_selection': truths},
            'summary': summary, 'gates': analyze.evaluate_gates(records, requests)}


def _write_csv(path, rows, *, columns=None):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def _markdown(result, uncertainty):
    summary, gates = result['summary'], result['gates']
    comparison = ['# MSIC-FS 完整独立验证比较', '',
                  '完整纳入 24 场景 × 4 重复 = 96 个新输入；每个输入只搜索一次，两规则共享候选和测试流。历史数据与 Pilot 未并入。', '',
                  '| 分组 | N | A0 PMSE | A1 PMSE | 比值 | A0/A1 Exact | 改善/相同/恶化 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
    for label, stats in [('全部', summary['overall']), ('n100', summary['by_n']['100']), ('n400', summary['by_n']['400'])]:
        a, b = stats['arms']['A0'], stats['arms']['A1']
        comparison.append(f"| {label} | {stats['n']} | {a['pmse_mean']:.6f} | {b['pmse_mean']:.6f} | {stats['ratio']:.6f} | "
                          f"{a['exact_count']}/{b['exact_count']} | {stats['improved']}/{stats['same']}/{stats['worsened']} |")
    comparison.extend(['', '| 预声明门槛 | 实际值 | 阈值 | 通过 |', '|---|---:|---|---|'])
    for name, criterion in gates['criteria'].items():
        comparison.append(f"| {name} | {criterion['value']:.8f} | {criterion['direction']} {criterion['threshold']} | {criterion['passed']} |")
    comparison.extend(['', f"七项统计门槛全部通过：{gates['passed']}；总体与 n100 的配对区间均支持稳定改善：{uncertainty['stable_gain']}。",
                       '这两个条件均不能替代独立工程验收，也不自动授权替换正式版本。', '',
                       '| 分组 | PMSE 均值比 95% 区间 | 配对差 95% 区间 |', '|---|---|---|'])
    for label in ('overall', 'n100', 'n400'):
        comparison.append(f"| {label} | {uncertainty[label]['ratio_ci95']} | {uncertainty[label]['delta_ci95']} |")
    overall = summary['overall']
    comparison.extend(['', f"最大 1/5 个改善例占全部正改善量的比例：{overall['top1_fraction_gross_improvement']} / {overall['top5_fraction_gross_improvement']}；"
                       f"最坏单例 PMSE 比 {overall['max_case_ratio']:.6f}，最坏完整场景比 {summary['worst_cell']['ratio']:.6f}。",
                       '比例的分母为所有正改善的总量，不是可能发生抵消的净改善。所有分组中位数、P95、REE、F1、欠选和过选见 analysis/group_metrics.csv；全部 24 场景见 analysis/cell_metrics.csv。', '',
                       '不确定性严格复用事前固定的 20,000 次场景内配对 bootstrap、seed=11011011；每场景仅四重复，区间不能证明广泛适用性或所有恢复指标非劣。'])
    mechanisms = ['# MSIC-FS 选择变化与不利结果', '',
                  f"96 例中 B 改变 {summary['changed_B_cases']} 例；变化分类为 `{json.dumps(summary['change_categories'], ensure_ascii=False)}`。",
                  f"真支撑进入池的有 {summary['truth_in_pool']} 例；FS 放弃 A0 已精确恢复真支撑的 case 为 {summary['truth_abandoned_cases']}，新增精确恢复为 {summary['truth_gained_cases']}。", '',
                  '所有选择变化的原/新支撑、q、RSS、拟合项、线性与组合惩罚、FS 项、argmin、带宽和 PMSE 均完整列于 analysis/selection_changes.csv。',
                  '附加项只依赖 q/n；同 q 支撑的相对惩罚不变，因此这次实验不能解决同形状支撑之间的训练 RSS 与外部预测排序冲突。', '',
                  '下表保留任一预测、Exact、REE、F1 或强信号欠选恶化的全部例，不按有利与否删减；各指标变化为 A1−A0。', '',
                  '| case | n/SNR | q A0→A1 | PMSE差 | Exact差 | REE差 | 模式F1差 | 元素F1差 | 新欠选 |',
                  '|---|---|---|---:|---:|---:|---:|---:|---|']
    adverse = {task for ids in summary['adverse_task_ids'].values() for task in ids}
    for row in result['tables']['paired_changes']:
        if row['task_id'] in adverse:
            mechanisms.append(f"| {row['task_id']} | {row['n_train']}/{row['snr']} | {row['A0_q']}→{row['A1_q']} | "
                f"{row['pmse_delta']:.6f} | {row['exact_delta']:.0f} | {row['ree_delta']:.6f} | {row['mode_f1_delta']:.6f} | "
                f"{row['element_f1_delta']:.6f} | {row['new_q_under']} |")
    mechanisms.extend(['', '真形状可表示性与真支撑入池的逐例证据见 analysis/truth_pool_selection.csv；不可表示真形状的 Exact=0 不能单独归因于选阶。',
                       '全部 96 配对记录见 analysis/paired_changes.csv，192 个选中模型见 analysis/selected_metrics.csv，最终形状/q 分布见 analysis/shape_q_distribution.csv。'])
    return '\n'.join(comparison) + '\n', '\n'.join(mechanisms) + '\n'


def run(repo, source, output):
    from research.taskcard11.experiment import collect, verify_plan
    from research.taskcard10.experiment import read_json, write_json, file_sha
    repo, source, output = Path(repo).resolve(), Path(source).resolve(), Path(output).resolve()
    require(output.is_relative_to(repo / 'doc/taskcard11'), 'Report output must stay in doc/taskcard11')
    require(not output.exists(), 'Report output already exists; refusing overwrite')
    doc_paths = [output.parent / name for name in ('MSIC_FINAL_COMPARISON.md', 'SELECTION_CHANGE_ANALYSIS.md')]
    require(not any(path.exists() for path in doc_paths), 'Final comparison documents already exist')
    plan = verify_plan(repo, source)
    require(plan['stage'] == 'formal', 'Pilot outcomes cannot enter formal reports')
    completion = read_json(source / 'completion.json')
    require(completion['status'] == 'COMPLETE' and completion['completed_cases'] == 96 and
            completion['plan_sha256'] == file_sha(source / 'plan.json'), 'Formal execution is incomplete or unbound')
    records = collect(source, 'validation')
    summaries, audits = {}, []
    for record in records:
        task = record['request']['task_id']
        folder = source / 'validation' / f'task_{task:03d}'
        summaries[task] = read_json(folder / 'summary.json')
        audit = read_json(folder / 'independent_audit.json')
        require(all(audit[key] for key in ('passed', 'A0_native_B_support_RSS_score_exact',
                    'A0_native_tolerance_eligible_exact', 'A0_A1_same_candidate_library')),
                'Required independent per-case engineering audit failed')
        audits.append(audit)
    result = assemble(records, summaries, plan['requests'])
    uncertainty = analyze.paired_uncertainty(records)
    comparison, mechanisms = _markdown(result, uncertainty)
    result['summary'].update({'input_plan_sha256': file_sha(source / 'plan.json'),
        'input_completion_sha256': file_sha(source / 'completion.json'), 'input_directory': source.relative_to(repo).as_posix(),
        'report_source_sha256': file_sha(Path(__file__)),
        'analysis_source_sha256': file_sha(repo / 'research/taskcard11/analyze.py'),
        'statistical_gates_passed': result['gates']['passed'], 'stable_gain': uncertainty['stable_gain'],
        'per_case_engineering_passed': True, 'candidate_audit_count': sum(a['candidate_count'] for a in audits),
        'statistical_recommendation_eligible': result['gates']['passed'] and uncertainty['stable_gain'],
        'release_decision': 'Separate final decision and all historical protection checks required; never auto-publish'})
    # No artifact is written until complete coverage, integrity and aggregation pass.
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in result['tables'].items():
        _write_csv(output / (name + '.csv'), rows, columns=list(result['tables']['paired_changes'][0]))
    write_json(output / 'summary.json', result['summary'])
    write_json(output / 'gates.json', result['gates'])
    write_json(output / 'uncertainty.json', uncertainty)
    for path, text in zip(doc_paths, (comparison, mechanisms)):
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
    print('REPORT_COMPLETE cases=96 cells=24 statistical_gates=' + str(result['gates']['passed']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    run(repository, args.input, args.output or repository / 'doc/taskcard11/analysis')
