"""Seal task17 after independent numerical and figure acceptance."""
import csv
import hashlib
import json
from pathlib import Path


def seal(root):
    from .audit import _validated_lock
    root=Path(root)
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    contract=load('contract.json');contract_sha=digest(root/'contract.json')
    for name in ('PILOT_ACCEPTED.json','DEVELOPMENT_ACCEPTED.json','FINAL_ACCEPTED.json'):
        lock=load(name)
        assert lock['status']=='PASS' and lock['contract_sha256']==contract_sha
        assert digest(root/lock['report'])==lock['report_sha256']
        _validated_lock(root,name,contract_sha)
    for key in ('data_path','config_path'):assert digest(contract[key])==contract[key+'_sha256']
    metrics=load('audit_metrics.json');assert metrics['status']=='PASS' and metrics['tasks']==192 and metrics['contract_sha256']==contract_sha
    for path,expected in metrics['files'].items():assert digest(root/path)==expected
    for path,expected in contract['source'].items():assert digest(path)==expected
    figure=load('figures/qa/FIGURE_VALIDATION.json');assert figure['visual_inspection']=='PASS'
    for path,expected in figure['input_sha256'].items():assert digest(root/path)==expected
    assert digest('empirical_analysis/uci270_concentration/plot_results.py')==figure['source_sha256']
    for item in figure['automatic_audits'].values():assert all(v['returncode']==0 for v in item.values())
    expected_figures={'final_rmse','temporal_error','final_predictions','range_error','error_complexity','abess_mode_frequency'}
    assert set(figure['automatic_audits'])==expected_figures
    expected_exports={f'{name}.{suffix}' for name in expected_figures for suffix in ('pdf','svg','png')}
    assert set(figure['output_sha256'])==expected_exports
    for name,expected in figure['output_sha256'].items():assert digest(root/'figures'/name)==expected
    with (root/'task_status.csv').open(encoding='utf-8-sig') as f:tasks=list(csv.DictReader(f))
    with (root/'summary_metrics.csv').open(encoding='utf-8-sig') as f:summary=list(csv.DictReader(f))
    convergence=load('convergence_audit.json')
    budget=load('fit_budget.json')
    expected_tasks={f"gas{s['gas_id']}_{s['split']}_{m}" for s in contract['splits'] for m in load('config.json')['methods']}
    assert len(tasks)==len(expected_tasks)==192 and {r['task_id'] for r in tasks}==expected_tasks
    assert all(r['status']=='complete' for r in tasks)
    metadata=load('data/metadata.json');assert metadata['gate_d']=='PASS' and metadata['semantic_gate']=='PASS'
    assert metadata['independent_parse_records']==metadata['n']==13910
    assert metadata['mapping_checked_values']==metadata['n']*96
    unit=load('UNIT_VALIDATION.json');assert unit['status']=='PASS'
    for name,expected in unit['source_sha256'].items():assert digest(name)==expected
    native=load('audit_native_pilot.json');assert native['status']=='PASS' and native['contract_sha256']==contract_sha
    auditdev=load('audit_dev.json');auditfinal=load('audit_final.json')
    replay=[x['prediction_max_abs'] for audit in (auditdev,auditfinal) for x in audit['tasks'].values()]
    selectedbad=[r for r in convergence if r['converged'] is False and r['selected']]
    validation=dict(status='COMPLETE',planned_outer_models=192,successful_outer_models=sum(r['status']=='complete' for r in tasks),
        failed_outer_models=sum(r['status']=='failed' for r in tasks),not_evaluable_models=sum(r['status']=='not_evaluable' for r in tasks),
        not_run_models=sum(r['status']=='not_run' for r in tasks),data_gate=metadata['gate_d'],mapping_gate='PASS',correctness_gate=metrics['status'],
        performance_acceptance='LIMITED; numerical candidate nonconvergence and single-system single-final-batch limits; no unconditional main empirical recommendation',
        contract_sha256=contract_sha,complete_core_source_files=32,source_identity_unchanged=True,
        replay_rtol=1e-9,replay_atol=1e-10,max_abs_prediction_replay=max(replay),
        test_predictions=metrics['test_predictions'],outer_and_inner_fit_attempts=sum(int(r['fit_count']) for r in summary),
        additional_native_audit_fits=native['extra_fit_count'],shared_fitting_seconds=budget['spent_seconds'],shared_budget_seconds=14400,
        failed_convergence_fits=sum(r['converged'] is False for r in convergence),selected_failed_convergence_fits=len(selectedbad),
        unknown_convergence_fits=sum(r['converged'] is None for r in convergence),
        unit_tests=unit['tests'],unit_test_status=unit['status'],figure_count=len(expected_figures),figure_qa=figure['visual_inspection'],remote_submission=False)
    assert validation['successful_outer_models']==192 and budget['spent_seconds']<=14400
    (root/'FINAL_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('DELIVERY_INDEX.json','DELIVERY_INDEX.md') and '__pycache__' not in p.parts]
    files+=list(Path('empirical_analysis/uci270_concentration').glob('*.py'))
    entries={p.as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
    (root/'DELIVERY_INDEX.json').write_text(json.dumps(dict(status='COMPLETE',files=entries),ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'DELIVERY_INDEX.md').write_text('''# 任务卡17交付索引

- [正式报告](UCI270_EMPIRICAL_FINAL_REPORT.md)：六类完整结论和全部方法结果。
- [预测与漂移](PREDICTION_AND_DRIFT_ANALYSIS.md)、[复杂度与稳定性](SPARSITY_AND_STABILITY_REPORT.md)。
- [数据审计](DATA_AUDIT.md)、[张量映射](TENSOR_MAPPING_AUDIT.md)、[官方来源](SOURCE_AUDIT.md)。
- [冻结协议](FROZEN_PROTOCOL.md)、[Pilot与正确性](PILOT_AND_CORRECTNESS.md)、[复现说明](README.md)。
- [机器验收](FINAL_VALIDATION.json)、[完整字节数及SHA256清单](DELIVERY_INDEX.json)。
- summary_metrics.csv、predictions.csv、model_complexity.csv、range_metrics.csv、support_stability.csv、support_frequency.csv、task_status.csv为可复算表格。
- models/保存192个独立模型、全部训练缩放器、训练/测试预测、内折选择记录、警告和完成密封；logs/保留每个任务日志。
- figures/提供六张PDF/SVG/PNG及自动/人工质量检查。原始文件位置、大小与哈希见raw_manifest.json；本地哈希不冒称官方校验和。

拟合源合同：`'''+contract_sha+'`。本研究未提交远程集群。\n',encoding='utf-8')
    for path,entry in entries.items():assert digest(path)==entry['sha256']
    return validation


if __name__=='__main__':print(json.dumps(seal('doc/uci270_concentration'),ensure_ascii=False))
