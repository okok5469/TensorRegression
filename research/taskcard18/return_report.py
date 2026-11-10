"""Read sealed cluster-return acceptance without changing the local fitting identity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def verified_return_evidence(root: Path, path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence.get('record_sha256') != digest({k: v for k, v in evidence.items() if k != 'record_sha256'}):
        raise ValueError('Return validation record seal mismatch')
    if evidence.get('status') != 'PASS':
        raise ValueError('A complete PASS return validation is required')
    archive = root / evidence['archive_repo_relative_path']
    if sha(archive) != evidence['archive_sha256']:
        raise ValueError('Returned archive hash changed after acceptance')
    hashes = evidence['certified_csv_sha256']
    if not hashes:
        raise ValueError('Accepted derived table hashes are required')
    for relative, expected in hashes.items():
        if sha(root / relative) != expected:
            raise ValueError('Accepted derived table hash changed: ' + relative)
    return evidence


def require_complete_counts(counts: dict) -> bool:
    expected = {'historical_records': 700, 'historical_npz_verified': 700,
                'label_method_rows': 2800, 'unique_inputs': 600, 'configurations': 10,
                'valid_fits': 1200, 'evaluations': 1200, 'failures': 0,
                'unique_method_rows': 2400, 'config_method_cells': 40,
                'label_method_cells': 56, 'paired_config_comparisons': 30}
    if any(counts.get(key) != value for key, value in expected.items()):
        raise ValueError('Returned comparison does not contain every required statistical unit')
    return True


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def collect_return_snapshot(root: Path, validation_path: Path) -> dict:
    from .report import collect_snapshot, derive_status
    evidence = verified_return_evidence(root, validation_path)
    require_complete_counts(evidence['counts'])
    if evidence.get('all_csv_verified') is not True or evidence.get('blocking_issues'):
        raise ValueError('Complete saved-result and statistical verification is required')
    facts = collect_snapshot(root)
    facts['local_training_fit_records'] = facts.pop('fit_records')
    if facts['gates']['G0']['status'] == 'PASS_CURRENT':
        facts['gates']['G0'].update(status='PASS', remaining='开始与结束的32/38指纹及正式导入身份一致，已复验')
    identity = evidence['recorded_run_identity']
    for name, expected in identity['adapter_hashes'].items():
        if sha(root / 'research/taskcard18' / name) != expected:
            raise ValueError('Frozen returned numerical adapter changed: ' + name)
    if sha(root / 'doc/taskcard18/CLOSEOUT_PROTOCOL.json') != identity['protocol_sha256']:
        raise ValueError('Frozen returned protocol changed')
    facts.update({'execution': 'existing_cluster_returned', 'return_validation': validation_path.relative_to(root).as_posix(),
                  'return_validation_sha256': sha(validation_path), 'returned_archive_sha256': evidence['archive_sha256'],
                  'recorded_cluster_identity': identity, 'valid_fits': 1200, 'evaluations': 1200,
                  'fits_by_method': {'lasso': 600, 'glasso': 600}, 'fitted_unique_inputs': 600,
                  'inputs_with_both_fits': 600, 'verified_input_correspondences': 600,
                  'pilot_valid_fits': 40, 'pilot_gate_status': 'PASS', 'failure_records': [],
                  'certificate_counts': evidence['certificate_counts'], 'certificate_extrema': evidence['certificate_extrema'],
                  'selected_lambda_boundaries': evidence['selected_lambda_boundaries'], 'cluster_runtime': evidence['runtime'],
                  'input_identity_levels': evidence['input_identity_levels'],
                  'execution_note': '已回传并独立验收集群600套完整输入×2方法；本机旧训练诊断保持独立。',
                  'analysis_row_counts': {'label_metrics': 2800, 'unique_metrics': 2400, 'config_summary': 40,
                                          '14_label_summary': 56, 'paired_pmse': 30},
                  'paired_pmse': read_csv(root / 'doc/taskcard18/results/certified_paired_pmse.csv')})
    for name in ('G1', 'G2', 'G3', 'G4', 'G5'):
        facts['gates'][name] = {'status': 'PASS', 'evidence': [facts['return_validation']], 'remaining': '已验收'}
    qa_path = root / 'doc/taskcard18/figure_qa/FIGURE_VALIDATION.json'
    qa = json.loads(qa_path.read_text(encoding='utf-8'))
    figures_pass = qa.get('status') == 'PASS' and qa.get('not_complete_revised_baseline_comparison') is False
    if figures_pass:
        if sha(qa_path.parent / qa['visible_review']) != qa.get('visible_review_sha256'):
            raise ValueError('Final visible figure review changed after QA acceptance')
        if sha(root / 'research/taskcard18/figures.py') != qa.get('plot_source_sha256') or not qa.get('output_sha256'):
            raise ValueError('Final figure source or output inventory is not bound to QA')
        for relative, expected in {**qa['source_sha256'], **qa['output_sha256']}.items():
            if sha(root / relative) != expected:
                raise ValueError('Final figure/table source or artifact changed: ' + relative)
    review_path = root / 'doc/taskcard18/INDEPENDENT_REVIEW.md'
    review = review_path.read_text(encoding='utf-8') if review_path.exists() else ''
    review_pass = bool(re.search(r'(?im)^\s*(?:status|状态)\s*[:：]\s*(?:\*\*)?PASS(?:\*\*)?\s*$', review))
    reviewed = ('paper/sections/02_method.md', 'paper/sections/04_simulation.md',
                'paper/appendix/simulation_protocol.md', 'doc/taskcard18/figure_qa/FIGURE_VALIDATION.json',
                'doc/taskcard18/results/certified_config_summary.csv', 'doc/taskcard18/results/certified_paired_pmse.csv',
                'paper/tables/baseline_update.csv', 'paper/tables/support_scale_distribution.csv',
                'paper/tables/support_structure_summary.csv', 'research/taskcard18/return_report.py',
                'research/taskcard18/return_tables.py', 'research/taskcard18/return_replay.py',
                'doc/taskcard18/cluster_review/PREDICTION_REPLAY.json')
    review_bound = all((root / name).exists() and sha(root / name) in review for name in reviewed)
    for name in ('G6', 'G7'):
        passed = figures_pass and review_bound and (review_pass or name == 'G6')
        facts['gates'][name] = {'status': 'PASS' if passed else 'PENDING',
                                'evidence': [facts['return_validation'], 'figure_qa/FIGURE_VALIDATION.json', 'INDEPENDENT_REVIEW.md'],
                                'remaining': '已验收' if passed else '四方法图表及绑定最终文件的独立审查待完成'}
    facts['figure_status'] = qa.get('status', 'PENDING')
    facts['status'] = derive_status(facts)
    facts['chapter_status'] = 'FINAL' if facts['status'] == 'FINAL' else 'DRAFT'
    return facts


def configuration_labels(root: Path) -> dict:
    plan = json.loads((root / 'numerical_simulation/outputs_v52_search_final_full/plan.json').read_text(encoding='utf-8'))
    labels = {}
    for task in plan['tasks']:
        config = {k: v for k, v in task['config'].items() if k != 'random_state'}
        key = 'cfg_' + digest(config)
        labels[key] = (f"n={config['n_train']}, SNR={config['snr']}, "
                       f"p={'×'.join(map(str, config['dims']))}, ρ={config['rho'][0]}")
    return labels


def render_return_report(root: Path, facts: dict) -> str:
    labels = configuration_labels(root)
    paired = []
    for row in facts['paired_pmse']:
        if row['comparison'] == 'adaptive_minus_oracle':
            continue
        method = {'adaptive_minus_lasso': 'Lasso v2', 'adaptive_minus_glasso': '顺序Tensor-GLasso v2'}[row['comparison']]
        lo, hi = float(row['ci_low']), float(row['ci_high'])
        conclusion = 'ABESS较低' if hi < 0 else ('ABESS较高' if lo > 0 else '优势尚不明确')
        paired.append(f"| {labels[row['config_id']]} | {method} | {row['n']} | {float(row['mean_difference']):.5f} | [{lo:.5f}, {hi:.5f}] | {conclusion} |")
    gate_rows = '\n'.join(f"| {name} | {item['status']} | {item['remaining']} |" for name, item in facts['gates'].items())
    runtime_rows = '\n'.join(f"| {method} | {item['fits']} | {item['aggregate_method_elapsed_seconds']:.3f} | {item['median_method_elapsed_seconds']:.3f} | {item['max_method_elapsed_seconds']:.3f} |"
                             for method, item in facts['cluster_runtime'].items())
    identity = facts['recorded_cluster_identity']
    total_certificates = sum(facts['certificate_counts'].values())
    issues = '\n'.join('- ' + item for item in facts['blocking_issues']) or '无未解除的关键有效性问题。'
    return f"""# 任务卡18执行与最终验收报告

北京时间：{facts['checked_beijing']}。**任务状态：{facts['status']}；方法章与模拟章：{facts['chapter_status']}。**

1. **定稿范围。** 本卡仅验收方法章、数值模拟章、附录、主表主图及证据链；章节联合定稿以G0–G7全部通过为准，当前见下表。
2. **正式算法。** 32份正式生产Python源、38项模拟源码/配置及原计划封存身份保持一致；正式v5.2-search-final、原MSIC、caps及搜索规则均未改动，未重新拟合ABESS/Oracle。
3. **基线修复。** 原Vector Lasso在同lambda内跨CV折传递beta，旧两对照缺必要逐候选最优性证据；隔离v2保留目标、20λ网格、5折SSE、并列规则、无截距/标准化及原Tensor-GLasso顺序/回退/矩形OLS。每折状态隔离，最终全训练另拟合；Lasso保留L1估计。所有{total_certificates:,}个必要凸拟合同时通过KKT与gap标准，失败0；不是仅验证最终模型。
4. **实际完成量。** 集群600套唯一完整输入×2对照，1200拟合及1200测试评价完整；复用正式ABESS/Oracle各600个唯一结果。固定pilot40/40；700标签→600完整输入→10配置，基准R=150，其余R=50。保存原回传ZIP和本机旧8项训练诊断，不混合运行身份；本轮仅从缓存审计、重算与写作。
5. **更新后的比较。** 以本轮验收的逐配置配对差及95% t均值区间替代旧“11/14”表述，完整结果如下；不统计跨配置显著获胜数，不按成绩删除场景。新旧对照变化见[更新差异表](../../paper/tables/baseline_update.csv)，主结果见[四方法主表](../../paper/tables/certified_main_results.md)。
6. **支撑与统计。** 实际元素支撑、声明轴支撑/非零投影和矩形容量分开计算，阈值1e-8且附1e-10/1e-6敏感性；REE不平方。逐配置同输入配对，SD与MCSE分列，Exact给Wilson区间。n100的caps(4,4,4)排除真形状(4,3,5)，困难与负面重复完整保留。
7. **交付。** [方法章](../../paper/sections/02_method.md)、[模拟章](../../paper/sections/04_simulation.md)、[附录](../../paper/appendix/simulation_protocol.md)、[主图与图注](../../paper/figures/CAPTIONS.md)、[证据映射](CLAIM_EVIDENCE_MAP.md)、[独立审查](INDEPENDENT_REVIEW.md)、[复现入口](../../research/taskcard18/README.md)。全精度40主单元、56附录单元、30配置配对和原始映射位于results/；三轴规模分布及新旧差异由只读脚本生成。
8. **科学边界。** 本批为单一可分离高斯DGP、固定真形状(4,3,5)；有限近似位置搜索、caps及局部CV边界保留。训练拟合改善不保证泛化改善；Oracle不是可部署方法或任意估计器的有限样本风险下界。新理论、实证拟合、新颖性与整篇论文投稿条件不属本卡定稿范围。

## 同完整输入的预测差

差值为PMSE(ABESS)−PMSE(对照)，负值表示ABESS较低。区间是配置内模拟均值的不确定性，不是500测试点作为算法重复；Oracle配对见全精度CSV。

| 参数配置 | 对照 | R | 均值差 | 95%配对t区间 | 有限结论 |
|---|---|---:|---:|---|---|
{chr(10).join(paired)}

## 运行及数值证据

回传SHA256：`{facts['returned_archive_sha256']}`。独立审计：[RETURN_VALIDATION.json](cluster_review/RETURN_VALIDATION.json)；原归档保持完整。记录环境Python{identity['python']}、NumPy{identity['numpy']}、SciPy{identity['scipy']}、sklearn{identity['sklearn']}，Linux x86_64，四项数值线程均1。输入身份层级逐记录保留，不将种子重放冒称未保存测试数组的字节复原。

证书总数{total_certificates:,}；全路径最大KKT/λ={facts['certificate_extrema']['max_kkt_ratio']:.12g}，最大归一化gap={facts['certificate_extrema']['max_normalized_gap']:.12g}。各阶段触边及完整分母保留于返回验收JSON；触边不是自动失败，也没有根据测试排名扩展网格。

| 方法 | 拟合数 | 累计方法elapsed秒 | 中位elapsed秒 | 最大elapsed秒 |
|---|---:|---:|---:|---:|
{runtime_rows}

以上是perf_counter方法耗时的累计及分布，既不是实测CPU时间也不是并行墙钟；不与历史ABESS作跨环境通用加速比。

## Gate

| Gate | 状态 | 验收结果 |
|---|---|---|
{gate_rows}

{issues}
"""


def write_reports(root: Path, facts: dict) -> None:
    doc = root / 'doc/taskcard18'
    (doc / 'gates.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (doc / 'TASKCARD18_FINAL_REPORT.md').write_text(render_return_report(root, facts), encoding='utf-8')
    text = f"""# 方法章与模拟章冻结状态

北京时间：{facts['checked_beijing']}。**任务状态：{facts['status']}；两章状态：{facts['chapter_status']}。**

正式32源及38项模拟身份保持冻结。集群600套完整输入×2参考对照、1200拟合/评价、40项pilot、全部证书及40主单元/56附录单元/30配置配对已经独立审计；本机旧诊断和回传归档分别保留。

图表状态：{facts['figure_status']}；最终独立审查绑定章节、附录、图表QA及统计源的SHA256。当前Gate状态以[最终报告](../doc/taskcard18/TASKCARD18_FINAL_REPORT.md)和[机器验收](../doc/taskcard18/gates.json)为准，任一有效性Gate未通过时不授予联合定稿。

冻结范围限方法与模拟两章；单一DGP、固定真形状、有限搜索、小样本caps及非严格嵌套局部CV等边界已保留，理论、新实证、新颖性和整篇论文投稿条件另属后续工作。
"""
    (root / 'paper/CHAPTER_FREEZE.md').write_text(text, encoding='utf-8')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--validation', type=Path, default=Path('doc/taskcard18/cluster_review/RETURN_VALIDATION.json'))
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    path = args.validation if args.validation.is_absolute() else root / args.validation
    facts = collect_return_snapshot(root, path)
    if args.write:
        write_reports(root, facts)
    print(json.dumps({k: facts[k] for k in ('status', 'chapter_status', 'valid_fits', 'evaluations', 'blocking_issues')}, ensure_ascii=False))
    return int(facts['status'] == 'BLOCKED')


if __name__ == '__main__':
    raise SystemExit(main())
