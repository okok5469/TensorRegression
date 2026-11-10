"""Check protected inputs and seal the final task19 delivery/report evidence."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def seal(root,pilot,installation):
    root=Path(root).resolve()
    docs=root/'doc/taskcard19'
    baseline=read(docs/'BASELINE.json')
    protected={**baseline['protected_sha256'],**read(docs/'BASELINE_SIMULATION38.json')}
    conflicts=[name for name,digest in protected.items() if sha(root/name)!=digest]
    if conflicts:
        raise RuntimeError(f'Protected inputs changed: {conflicts}')
    if any((root/name).exists() for name in ('task_plan.md','progress.md','findings.md','task_issue.md')):
        raise RuntimeError('Root workflow file present')
    tracked=subprocess.run(['git','status','--porcelain=v1','--untracked-files=no'],cwd=root,capture_output=True,check=True).stdout.decode('utf-8')
    allowed={'doc/task_plan.md','doc/progress.md','doc/findings.md'}
    old={line for line in baseline['git_tracked_status'].splitlines() if line[3:] not in allowed}
    new={line for line in tracked.splitlines() if line[3:] not in allowed}
    if old!=new:
        raise RuntimeError('Preexisting tracked workspace status changed outside task19 workflow files')
    install=read(installation/'CLEAN_INSTALLATION.json')
    archive=root/'deliverables/taskcard19/TensorABESS_OvRLS_v0.1.0.zip'
    if install['status']!='PASS' or install['archive_sha256']!=sha(archive):
        raise RuntimeError('Final ZIP does not match clean-install acceptance')
    log=(docs/'SCOPED_TESTS.log').read_text(encoding='utf-8')
    tests=re.findall(r'Ran (\d+) tests',log)
    if not tests or not log.rstrip().endswith('OK'):
        raise RuntimeError('Scoped tests are not accepted')
    run=read(pilot/'manifest.json')
    if run['evaluation_stage']!='dev' or run['formal_test_evaluated']:
        raise RuntimeError('Official test boundary violated')
    with (pilot/'metrics.csv').open(encoding='utf-8',newline='') as stream:
        metrics=list(csv.DictReader(stream))
    diagnostics=[json.loads(line) for line in (pilot/'fit_diagnostics.jsonl').read_text(encoding='utf-8').splitlines() if line]
    attempts=[row for row in diagnostics if (row.get('solver') and row.get('phase') in ('inner','final')) or row.get('stage')=='core_fit']
    rule_records=[row for row in diagnostics if row.get('method')=='majority' and row.get('phase')=='final']
    states={row['status'] for row in run['tasks']}
    unresolved=[row for row in diagnostics if row.get('status')=='NUMERICAL_UNRESOLVED' and row.get('max_iter')==20000]
    failure_details='\n'.join(f"- {row['method']} / {row['fold']}: {row['status']}; {row.get('error','required candidate diagnostics remain unresolved')}" for row in run['tasks'] if row['status']!='COMPLETE') or '无未解失败。'
    numerical_details='\n'.join(f"- {row['method']}: phase={row.get('phase')}, inner_fold={row.get('inner_fold')}, parameter={row.get('parameter')}, max_iter=20000, n_iter={row.get('n_iter')}" for row in unresolved) or '没有20000步后仍未解决的必要拟合。'
    next_action='先解决LOCAL_PILOT_REPORT.md中具体资源或数值阻塞，保留当前失败身份，再另存完整正式协议'
    if any(row['method']=='multinomial_l1' for row in unresolved):
        next_action='先解决L1必要候选及C=10最终拟合在20000步仍未收敛的问题，另存数值处理、训练量与资源协议后再开展正式实证；保持本卡结果不变，测试继续封存'
    local='DEV_PILOT_COMPLETE' if run['status']=='COMPLETE' else ('RESOURCE_BLOCKED' if 'RESOURCE_BLOCKED' in states else 'NUMERICAL_BLOCKED')
    package_status=read(docs/'PACKAGE_STATUS.json')
    package_status['local_stage']=local
    (docs/'PACKAGE_STATUS.json').write_text(json.dumps(package_status,indent=2),encoding='utf-8')
    tensor=next((row for row in metrics if row['method']=='tensor_abess_ovr_ls'),None)
    comparators=[row for row in metrics if row['method'] not in ('tensor_abess_ovr_ls','majority') and row['status']=='COMPLETE']
    finding='数值或资源状态不足，完整比较尚未验收'
    if tensor and tensor['status']=='COMPLETE' and comparators:
        best=max(float(row['macro_f1']) for row in comparators)
        finding='相对已验收对照有初步开发信号，尚需正式独立验证' if float(tensor['macro_f1'])>best else '有限训练样本的开发筛查尚无超越已验收最强对照的增量'
        if local!='DEV_PILOT_COMPLETE':
            finding+='；完整五方法比较仍受数值/资源状态阻塞'
    rows='\n'.join(f"| {m['method']} | {m['status']} | {float(m['macro_f1']):.6f} | {float(m['accuracy']):.6f} | {float(m['balanced_accuracy']):.6f} | {float(m['training_seconds']):.3f} |" for m in metrics)
    relative=pilot.relative_to(root).as_posix()
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    report=f'''# 任务卡19最终报告

北京时间：{now}。

- PACKAGE_READY: YES
- LOCAL_STAGE: {local}
- FORMAL_TEST_EVALUATED: NO
- SCIENTIFIC_FINDING: {finding}。

| Gate | 状态 | 证据 |
|---|---|---|
| G0 核心身份 | PASS | 32生产源与release原字节一致；38项源码/配置身份（含32核心与6模拟文件）及任务18定稿文件保持 |
| G1 A路线实现 | PASS | 指示响应仅中心化、截距恢复、独立核心完整状态/分数核对、并列/标签/保存与失败检查 |
| G2 数据与隔离 | PASS | 数据合同、group/time、训练内z-score、HAR6912原始坐标独立映射核对 |
| G3 对照与指标 | PASS | 软件协议检查：全五方法、完整九网格、诊断/OOF/重算；实际开发比较状态另见G6 |
| G4 便携交付 | PASS | 最终ZIP在原仓库外全新venv安装wheel，实际导入与依赖/线程检查 |
| G5 回传复核 | PASS | 完整/部分结果、预测缺行/损坏检测、安装版重算和pack-return |
| G6 本机开发 | {'PASS' if local=='DEV_PILOT_COMPLETE' else 'BLOCKED'} | HAR {run['completed_tasks']}/{run['expected_tasks']}方法任务完成；状态{local} |

## 给队友的文件

主文件：[TensorABESS_OvRLS_v0.1.0.zip](../../deliverables/taskcard19/TensorABESS_OvRLS_v0.1.0.zip)，SHA256 `{sha(archive)}`；旁置同名.sha256。ZIP含wheel、未修改32源、锁定依赖Windows wheels、唯一数据合同、固定fixture、测试/验收记录和两平台入口。无需复制历史仓库。

队友解压按README_ZH.md建立CPython3.12干净环境，安装wheel，执行verify-release和smoke，再validate-data/run。最后validate-return、pack-return并回传统一ZIP；原始X、venv和凭据不回传。本卡未替用户发送消息或发布公共包。

实际验证平台Windows；Linux UNVERIFIED。精确日志见IMPLEMENTATION_AND_PACKAGE_CHECKS.md及clean_install_final/。同环境开发/安装版分数、标签、系数、声明支撑和并列优先序均比较；无跨平台逐位一致声明。限定测试共{tests[-1]}项通过；不宣称全仓库测试通过。

## 本机HAR开发筛查

原官方train7352窗口/21受试者；开发训练400窗口/15受试者，验证2208窗口/6受试者，六类保留。固定3×3×17均值/窗频带摘要，subject seed20261031；官方test2947窗口/9受试者未评分。精确ID、原始下载身份与表示见DATASET_SCREENING.md和PROTOCOL_FREEZE.json。

运行 `{run['run_id']}`：5方法任务中{run['completed_tasks']}完成、{run['failed_tasks']}未验收；留有完整诊断的拟合调用记录{len(attempts)}（包括调参、同参数重试及6类核心），多数类规则构建{len(rule_records)}条另记。该计数不包含资源中断时尚未返回诊断的调用；训练内功能切片6次核心调用另列RESOURCE_BUDGET.json，不混入性能分母。数据集实际墙钟{run['wall_clock_seconds']:.3f}秒。全部正负与失败证据位于 `{relative}`；验证结果见LOCAL_RETURN_VALIDATION.json。

| 方法 | 状态 | Macro-F1 | Accuracy | Balanced Accuracy | 训练秒（全部调参+最终拟合） |
|---|---|---:|---:|---:|---:|
{rows}

任务失败/未验收状态：

{failure_details}

关键数值诊断：

{numerical_details}

主指标Macro-F1；不计算A路线概率Log-loss/Brier，不把原始分数当概率，不以方法获胜作为软件验收条件。窗口存在受试者内依赖，本次只有一次主体留出和有限训练样本，不作正式测试/统计显著性或分类一致性推断。MHEALTH未启动，本机仅尝试一个真实候选。独立重算见[LOCAL_PILOT_INDEPENDENT_REVIEW.md](LOCAL_PILOT_INDEPENDENT_REVIEW.md)；完整带失败状态的回传为[HAR_DEV_RETURN_20261031.zip](../../empirical_analysis/outputs_taskcard19/HAR_DEV_RETURN_20261031.zip)。

## 已暴露的问题与边界

最初软件fixture在L1高C20000步后仍不收敛，运行器正确标PARTIAL；原5输入与失败输出保留在runner_checks/fixture_v1_input和fixture_run_v1。后续安装fixture固定为各相同输入模式含全三类的非可分输入，仅用于软件验收；未更改真实HAR、网格、tol、搜索或预算，准确率不作通过门槛。

独立审查发现并修复安装路径/签名/ZIP库存、诊断完整性/内折身份/并列/OOF及超时子进程清理等缺口，均保留回归证据。首次广域检索遇到原历史测试工具访问受限；其已有Git状态原样保留，本卡不依赖或修复这些目录。独立安装没有受此影响。

下一步：{'冻结正式实证样本量、评价协议和资源预算后，开展队友/独立数据的正式实证；本开发结果不足以定稿分类实证章' if local=='DEV_PILOT_COMPLETE' else next_action}。原MSIC仍是平方损失工作准则，既有回归理论/模拟不证明本封装分类一致性；原初始化返回成功也不是全部内层凸问题的最优性证书。
'''
    (docs/'TASKCARD19_FINAL_REPORT.md').write_text(report,encoding='utf-8')
    (docs/'LOCAL_PILOT_REPORT.md').write_text('# 本机开发试跑\n\n'+report.split('## 本机HAR开发筛查\n\n')[1].split('## 已暴露的问题与边界')[0]+'\nLOCAL_STAGE: '+local+'\n\nFORMAL_TEST_EVALUATED: NO\n\nSCIENTIFIC_FINDING: '+finding+'。\n',encoding='utf-8')
    paths=[p for p in docs.rglob('*') if p.is_file() and p.name!='FINAL_VALIDATION.json']
    paths+=[archive,archive.with_suffix('.sha256')]
    run_hashes=read(pilot/'SHA256SUMS.json')
    for name,expected in run_hashes.items():
        path=(pilot/name).resolve()
        if not path.is_relative_to(pilot) or sha(path)!=expected:
            raise RuntimeError(f'Pilot evidence changed: {name}')
        paths.append(path)
    paths += [pilot/'SHA256SUMS.json',pilot.parent/'HAR_DEV_RETURN_20261031.zip',pilot.parent/'HAR_DEV_RETURN_20261031.zip.sha256']
    record={'status':'FINAL','package_ready':True,'local_stage':local,'formal_test_evaluated':False,'protected_files_checked':len(protected),'protected_conflicts':conflicts,'preexisting_tracked_status_preserved':True,'scoped_test_count':int(tests[-1]),'pilot_run_id':run['run_id'],'pilot_fit_attempts':len(attempts),'pilot_fit_attempts_definition':'calls with returned diagnostics, including retries; excludes interrupted calls without diagnostics and majority rule construction','majority_rule_records':len(rule_records),'package_sha256':sha(archive),'artifacts_sha256':{p.relative_to(root).as_posix():sha(p) for p in paths}}
    record['seal_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    (docs/'FINAL_VALIDATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='artifacts_sha256'},ensure_ascii=False,indent=2))


def reseal(root):
    docs=root/'doc/taskcard19'
    record=read(docs/'FINAL_VALIDATION.json')
    for name,expected in record['artifacts_sha256'].items():
        if sha(root/name)!=expected:
            raise RuntimeError(f'Previously sealed evidence changed: {name}')
    paths=[root/name for name in record['artifacts_sha256']]
    paths += [p for p in docs.rglob('*') if p.is_file() and p.name!='FINAL_VALIDATION.json']
    paths += [p for p in (root/'deliverables/taskcard19').iterdir() if p.is_file()]
    record['artifacts_sha256']={p.relative_to(root).as_posix():sha(p) for p in paths}
    record.pop('seal_sha256',None)
    record['seal_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    (docs/'FINAL_VALIDATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Final audit refreshed after independent review; reports remain unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot')
    parser.add_argument('--installation')
    parser.add_argument('--reseal',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.reseal:
        reseal(root)
    else:
        seal(root,Path(args.pilot).resolve(),Path(args.installation).resolve())
