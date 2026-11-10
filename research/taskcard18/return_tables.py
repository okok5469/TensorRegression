"""Generate supplementary tables solely from accepted saved-result CSVs."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import paired_statistics, save_csv
from .return_report import configuration_labels, read_csv, verified_return_evidence


def build(root: Path):
    verified_return_evidence(root, root / 'doc/taskcard18/cluster_review/RETURN_VALIDATION.json')
    directory = root / 'doc/taskcard18/results'
    new = read_csv(directory / 'certified_unique_metrics.csv')
    old = read_csv(directory / 'historical_unique_metrics.csv')
    labels = configuration_labels(root)
    old_by_key = {(r['unique_input_id'], r['method']): r for r in old}
    updates, distributions, structures = [], [], []
    grouped = defaultdict(list)
    for row in new:
        grouped[(row['config_id'], row['method'])].append(row)
    for (config, method), rows in sorted(grouped.items()):
        required = 150 if len(rows) == 150 else 50
        if len(rows) != required or len({r['unique_input_id'] for r in rows}) != required:
            raise ValueError('Supplement table has incomplete configuration/input denominator')
        for axis in (1, 2, 3):
            counts = Counter(int(float(r[f's{axis}_hat'])) for r in rows)
            distributions.extend({'config_id': config, 'configuration': labels[config], 'method': method,
                                  'axis': axis, 'selected_mode_size': size, 'count': count,
                                  'n': required, 'proportion': count / required}
                                 for size, count in sorted(counts.items()))
        item = {'config_id': config, 'configuration': labels[config], 'method': method, 'n': required}
        for metric in ('element_nonzero', 'q_rect', 'declared_equals_projection', 'element_recall', 'element_fdr', 'element_f1',
                       'element_nonzero_1e10', 'element_nonzero_1e6', 'element_recall_1e10', 'element_recall_1e6',
                       'element_fdr_1e10', 'element_fdr_1e6', 'element_f1_1e10', 'element_f1_1e6'):
            values = np.array([float(r[metric]) for r in rows])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite support structure metric')
            item.update({metric + '_mean': float(values.mean()), metric + '_sd': float(values.std(ddof=1)),
                         metric + '_median': float(np.median(values)), metric + '_q1': float(np.quantile(values, .25)),
                         metric + '_q3': float(np.quantile(values, .75))})
        structures.append(item)
        if method not in ('lasso', 'glasso'):
            continue
        a = {r['unique_input_id']: float(r['pmse']) for r in rows}
        b = {r['unique_input_id']: float(old_by_key[(r['unique_input_id'], method)]['pmse']) for r in rows}
        updates.append({'config_id': config, 'configuration': labels[config], 'method': method,
                        'old_method_id': 'legacy_' + method, 'new_method_id': rows[0]['method_id'],
                        'old_pmse_mean': float(np.mean(list(b.values()))), 'new_pmse_mean': float(np.mean(list(a.values()))),
                        'difference_definition': 'new_minus_legacy', **paired_statistics(a, b)})
    out = root / 'paper/tables'
    save_csv(out / 'baseline_update.csv', updates)
    save_csv(out / 'support_scale_distribution.csv', distributions)
    save_csv(out / 'support_structure_summary.csv', structures)
    pairs = read_csv(directory / 'certified_paired_pmse.csv')
    text = ['# 配置内预测配对差', '', 'PMSE(ABESS)−PMSE(对照)；基准150输入、其余50输入。负值表示ABESS较低。', '',
            '| 配置 | 对照 | R | 均值差 | MCSE | 95%配对t区间 |', '|---|---|---:|---:|---:|---|']
    for row in pairs:
        text.append(f"| {labels[row['config_id']]} | {row['comparison'].removeprefix('adaptive_minus_')} | {row['n']} | {float(row['mean_difference']):.5f} | {float(row['mcse']):.5f} | [{float(row['ci_low']):.5f}, {float(row['ci_high']):.5f}] |")
    (out / 'certified_paired_pmse.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    old_new = ['# 对照数值更新的配对变化', '', '同完整输入的新方法−旧未认证实现。该表描述实现更新的影响，不把旧方法当作可靠主对照。', '',
               '| 配置 | 方法 | R | 旧PMSE | 新PMSE | 新−旧 | 95%配对t区间 |', '|---|---|---:|---:|---:|---:|---|']
    for row in updates:
        old_new.append(f"| {row['configuration']} | {row['method']} | {row['n']} | {row['old_pmse_mean']:.5f} | {row['new_pmse_mean']:.5f} | {row['mean_difference']:.5f} | [{row['ci_low']:.5f}, {row['ci_high']:.5f}] |")
    (out / 'baseline_update.md').write_text('\n'.join(old_new) + '\n', encoding='utf-8')
    appendix = read_csv(directory / 'certified_14_label_summary.csv')
    lines = ['# 14历史标签完整附表', '', '保留14×50标签口径；不将复用输入计为新的独立重复。全精度字段见certified_14_label_summary.csv。', '',
             '| scenario_id | 方法 | R | PMSE均值±样本SD | REE均值 | 元素F1均值 | 模式F1均值 | Exact成功数 |',
             '|---:|---|---:|---|---:|---:|---:|---:|']
    for row in appendix:
        lines.append(f"| {row['scenario_id']} | {row['method']} | {row['n']} | {float(row['pmse_mean']):.5f} ± {float(row['pmse_sd']):.5f} | {float(row['ree_mean']):.5f} | {float(row['element_f1_mean']):.5f} | {float(row['mode_f1_macro_mean']):.5f} | {row['exact_successes']} |")
    (out / 'certified_14_label_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SUPPLEMENT_TABLES_COMPLETE', len(updates), len(distributions), len(structures), len(appendix), flush=True)


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[2])
