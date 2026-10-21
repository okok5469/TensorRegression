"""Describe Task10 continuation results without fits or changes to sealed records."""
import argparse
from pathlib import Path
import numpy as np

from research.taskcard10.analyze import read, summarize, table, write_csv, write_json


def make_reports(root, recommendation, decision_reason):
    out = root/'doc/taskcard10/continuation_01'
    raw = root/'doc/taskcard10/results/continuation_01'
    plan = read(raw/'plan.json')
    complete_dev = (raw/'development_completion.json').is_file()
    assert complete_dev or (raw/'stopped_development.json').is_file(), 'Await terminal development state'
    records = {}
    outputs = {name:[] for name in ('selected_metrics','group_metrics','paired_changes','cell_metrics',
                                   'candidate_coverage','fit_resources')}
    for cohort in ('development','confirmation'):
        records[cohort] = [dict(read(p),_summary=read(p.parent/'summary.json'))
                           for p in sorted((raw/cohort).glob('task_*/complete.json'))]
        if not records[cohort]:
            continue
        result = summarize(records[cohort],cohort)
        for name,rows in zip(('selected_metrics','group_metrics','paired_changes','cell_metrics'),result):
            outputs[name].extend(rows)
        for record in records[cohort]:
            req,ev = record['request'],record['recovery_evidence']
            outputs['candidate_coverage'].append({'cohort':cohort,'task_id':req['task_id'],
                'n':req['config']['n_train'],'snr':req['config']['snr'],'truth_shape':ev['truth_sizes'],
                **{k:ev[k] for k in ('truth_shape_native_representable','truth_shape_common_representable',
                    'truth_shape_found_full_pool','truth_shape_found_common_pool',
                    'truth_support_found_full_pool','truth_support_found_common_pool')},
                **{arm+'_exact':int(record['metrics'][arm]['exact_recovery']) for arm in ('A0','A1','A2')}})
            outputs['fit_resources'].append({'cohort':cohort,'task_id':req['task_id'],
                'n':req['config']['n_train'],**record['_summary']['counters'],**record['resources']})
    counts = {k:len(v) for k,v in records.items()}
    assert not complete_dev or counts['development']==48
    confirmed = counts['confirmation']==48 and (raw/'confirmation_gate.json').is_file()
    confirmation_stopped = (raw/'stopped_confirmation.json').is_file()
    assert counts['confirmation']==0 or confirmed or confirmation_stopped
    assert counts['confirmation'] or confirmation_stopped or (raw/'confirmation_NOT_RUN.json').is_file()
    devgate = read(raw/'development_lock.json')['entry_gate'] if complete_dev else {
        'status':'NOT_EVALUABLE_INCOMPLETE','passed':None}
    confgate = read(raw/'confirmation_gate.json') if confirmed else None
    assert recommendation in ('A','B','C') and decision_reason.strip()
    assert recommendation!='A' or (confirmed and confgate['passed']), 'A requires full independent confirmation'
    label = recommendation if confirmed else '暂定'+recommendation
    target = out/'analysis'
    target.mkdir(exist_ok=False)
    for name,rows in outputs.items():
        write_csv(target/(name+'.csv'),rows)
    write_json(target/'summary.json',{'counts':counts,'development_gate':devgate,
        'confirmation_gate':confgate,'statistics':outputs['group_metrics'],
        'original_reused_development':30,'newly_completed_development':counts['development']-30,
        'original_pilot_cases':4,'new_pilot_cases':0,
        'status':'COMPLETE_DEVELOPMENT' if complete_dev else 'STOPPED_PARTIAL_DEVELOPMENT'})

    def report(name,body):
        with (out/name).open('x',encoding='utf-8') as stream:
            stream.write(body.rstrip()+'\n')

    def gate_table(gate):
        if not gate or not gate.get('criteria'):
            return '完整联合门槛未评价。'
        return table(['原冻结门槛','实际值','阈值','通过'],[
            [k,f"{v['value']:.6f}",v['threshold'],v['passed']] for k,v in gate['criteria'].items()])

    def stats_table(cohort):
        return table(['组','臂','N','PMSE均值','中位数','P90','P95','最大','Exact','平均q','欠选率'],[
            [r['group'],r['arm'],r['n_cases'],*[f"{r[k]:.6f}" for k in ('pmse_mean','pmse_median','pmse_p90','pmse_p95','pmse_max')],
             r['exact_count'],f"{r['q']:.3f}",f"{r['q_under']:.4f}"]
            for r in outputs['group_metrics'] if r['cohort']==cohort])

    def paired_table(cohort):
        rows = []
        for group, n in (('all',None),('n100',100),('n400',400)):
            for left,right in (('A0','A1'),('A1','A2'),('A0','A2')):
                subset = [r for r in outputs['paired_changes'] if r['cohort']==cohort
                          and (n is None or r['n']==n) and r['left']==left and r['right']==right]
                if subset:
                    rows.append([group,right+'−'+left,len(subset),
                        f"{np.mean([r['delta_pmse'] for r in subset]):+.6f}",
                        *[sum(r['outcome']==outcome for r in subset) for outcome in ('improved','equal','worse')]])
        return table(['组','比较','N','平均PMSE差','改善','相同','恶化'],rows)

    def recovery_table(cohort):
        return table(['组','臂','N','REE','模式F1','元素F1','过选率','近容量上限率'],[
            [r['group'],r['arm'],r['n_cases'],*[f"{r[k]:.6f}" for k in
                ('ree','mode_f1_macro','element_f1','q_over','near_selection_limit')]]
            for r in outputs['group_metrics'] if r['cohort']==cohort])

    context = ('原900秒/7200秒阶段及其NOT_RUN、报告、FINAL原样保留；本次依据用户继续授权追加计算协议，'
               '2worker、单case3600秒、开发与确认共享28800秒时钟。科学方法、原96请求、九门槛、候选库及选择规则不变。'
               '原30例结果按字节复用，重算原18例未完成请求，不将复制或重试计作新科学重复。\n\n')
    status = '完整48例开发已完成。' if complete_dev else f"开发仍只完成{counts['development']}/48例，属于部分描述，不能推广。"
    report('DEVELOPMENT_RESULTS.md','# 任务卡10续跑：开发结果\n\n'+context+status+'\n\n'+
        stats_table('development')+'\n\n'+paired_table('development')+'\n\n'+
        recovery_table('development')+'\n\n'+gate_table(devgate)+'\n\n'
        'P95为各臂自己的PMSE分位数，最坏场景为固定两重复cell均值比；不删除极端值，不以配对比值分位数替代。'
        '完整设计下每n24例、每cell2重复，n400 Exact步长4.17个百分点；强SNR16例、比例步长6.25个百分点，均为描述性门槛而非高功效非劣证明；实际完成分母见表。\n\n'
        '[逐例](analysis/selected_metrics.csv)、[完整分组](analysis/group_metrics.csv)、[场景与重复稳定性](analysis/cell_metrics.csv)、'
        '[配对变化](analysis/paired_changes.csv)、[候选覆盖](analysis/candidate_coverage.csv)、[拟合/缓存/资源](analysis/fit_resources.csv)。')
    if confirmed:
        confirmation_text = ('完整48个原预注册未见输入已运行，启动前完整开发及预算门槛通过，确认选择均在外测试生成前冻结。\n\n'+
                             stats_table('confirmation')+'\n\n'+paired_table('confirmation')+'\n\n'+
                             recovery_table('confirmation')+'\n\n'+gate_table(confgate))
    else:
        reason = read(raw/'stopped_confirmation.json') if confirmation_stopped else read(raw/'confirmation_NOT_RUN.json')['reason']
        confirmation_text = (f"**{'STOPPED_PARTIAL' if confirmation_stopped else 'NOT_RUN'}**，完成{counts['confirmation']}/48例；终态理由：{reason}。\n\n"
                             '不使用历史、Pilot或开发数据替代未见确认；没有确认通过证据时不能推荐方案A。')
        if records['confirmation']:
            confirmation_text += ('\n\n下表仅为不完整、不平衡确认子集的描述，不计算完整联合门槛、不用于推广判断。\n\n'+
                                  stats_table('confirmation')+'\n\n'+paired_table('confirmation')+'\n\n'+recovery_table('confirmation'))
        if complete_dev and not confirmation_stopped:
            reserve = read(raw/'confirmation_budget_gate.json')
            confirmation_text += (f"\n\n开发结束时剩余预算{reserve['remaining_seconds']:.3f}秒，预设确认储备要求{reserve['required_seconds']:.3f}秒；"
                                  f"科学进入门槛通过={reserve['scientific_gate_passed']}。本轮须同时满足科学门槛和预算条件。")
    report('INDEPENDENT_CONFIRMATION.md','# 任务卡10续跑：独立确认\n\n'+confirmation_text+
           '\n\n[续跑协议](DESIGN_AMENDMENT.md)、[原门槛](../FROZEN_EXPERIMENT_PLAN.md)。')

    cohort = 'confirmation' if confirmed else 'development'
    selected = [r for r in outputs['selected_metrics'] if r['cohort']==cohort]
    groups = {(r['group'],r['arm']):r for r in outputs['group_metrics'] if r['cohort']==cohort}
    pairs = [r for r in outputs['paired_changes'] if r['cohort']==cohort]
    cover = [r for r in outputs['candidate_coverage'] if r['cohort']==cohort]
    low_rss_worse = [r for r in pairs if r['delta_rss'] < -1e-10 and r['delta_pmse'] > 1e-12]
    same_shape_new_support = [r for r in pairs if r['same_shape'] and not r['same_support']]
    coverage_rows = []
    for n in (100,400):
        for shape in ([4,3,5],[2,3,4]):
            subset = [r for r in cover if r['n']==n and r['truth_shape']==shape]
            coverage_rows.append([n,str(shape),len(subset),*[sum(r[k] for r in subset) for k in
                ('truth_shape_native_representable','truth_shape_common_representable','truth_shape_found_full_pool',
                 'truth_shape_found_common_pool','truth_support_found_full_pool','truth_support_found_common_pool')]])
    failures = [r for r in cover if r['n']==400 and r['A0_exact'] and not r['A2_exact']]
    no_candidate = sum(not r['truth_support_found_common_pool'] for r in failures)
    missed_selection = len(failures)-no_candidate
    loss_ids = {r['task_id'] for r in failures}
    loss_rows = []
    for record in records[cohort]:
        req,m = record['request'],record['metrics']
        if req['task_id'] in loss_ids:
            loss_rows.append([req['task_id'],req['config']['snr'],str(req['config']['sparsity']),
                str(m['A0']['sizes']),str(m['A2']['sizes']),m['A0']['q'],m['A2']['q'],
                f"{m['A2']['rss']-m['A0']['rss']:+.6f}",f"{m['A2']['pmse']-m['A0']['pmse']:+.6f}"])
    strong_rows = []
    for n in (100,400):
        for shape in ([4,3,5],[2,3,4]):
            subset = [r for r in selected if r['n']==n and r['snr']==4 and r['truth_shape']==shape]
            strong_rows.append([n,str(shape),sum(r['arm']=='A0' for r in subset),
                                *[sum(r['q_under'] for r in subset if r['arm']==arm) for arm in ('A0','A1','A2')]])
    cell_rows = []
    for cell in sorted({r['cell_id'] for r in selected}):
        part = [r for r in selected if r['cell_id']==cell]
        first = part[0]
        means = {arm:float(np.mean([r['pmse'] for r in part if r['arm']==arm])) for arm in ('A0','A1','A2')}
        cell_rows.append([cell,first['n'],first['snr'],str(first['truth_shape']),first['rho'],
            f"{means['A1']-means['A0']:+.6f}",f"{means['A2']-means['A1']:+.6f}",
            f"{means['A2']/means['A0']:.6f}",f"{means['A2']/means['A1']:.6f}"])
    risk_rows = []
    for arm in ('A0','A1','A2'):
        part = [r for r in selected if r['n']==100 and r['arm']==arm]
        risk_rows.append([arm,len(part),f"{np.mean([r['q'] for r in part]):.3f}",
            sum(r['near_selection_limit'] for r in part),f"{max(r['coefficient_norm'] for r in part):.6f}"])
    risk_pairs = []
    for left,right in (('A0','A1'),('A1','A2'),('A0','A2')):
        subset = [r for r in pairs if r['n']==100 and r['left']==left and r['right']==right
                  and r['delta_rss'] < -1e-10 and r['delta_pmse'] > 1e-12]
        risk_pairs.append([right+'−'+left,len(subset),sum(r['delta_q']>0 for r in subset),
                           sum(r['same_shape'] for r in subset),sum(r['same_support'] for r in subset)])
    report('SELECTION_MECHANISM_ANALYSIS.md','# 任务卡10续跑：选择机制与边界\n\n'+
        f'本节使用{cohort}，按真实完成分母报告。A1−A0同时改变候选范围并删除adaptive来源，不能唯一归因容量；'
        'A2−A1是同容量、同完整训练fixed候选的选择器比较。\n\n'+
        table(['n','真形状','N','原生可表示','公共可表示','原生真形状入池','公共真形状入池','原生真支撑入池','公共真支撑入池'],coverage_rows)+
        f'\n\nn400中A0恢复而A2未恢复共{len(failures)}例，其中公共库没有真支撑{no_candidate}例、已有真支撑但未被选{missed_selection}例。'
        '可表示、找到真形状、找到真支撑、最终选中分别核对，不能相互替代。\n\n'+
        table(['task','SNR','真形状','A0形状','A2形状','A0 q','A2 q','A2−A0 RSS','A2−A0 PMSE'],loss_rows)+'\n\n'+
        table(['强SNR n','真形状','每臂N','A0欠选数','A1欠选数','A2欠选数'],strong_rows)+
        '\n\n欠选按q小于真q定义，保留原联合门槛。n100真形状(4,3,5)在两类库均不可表示，但原生最大q64与公共最大q36分列真q60两侧；'
        '该组欠选差同时反映容量上界，不能单独归因CV。同库A2−A1差异仍是选择准则差异。\n\n'+
        table(['cell','n','SNR','真形状','rho','A1−A0 PMSE','A2−A1 PMSE','A2/A0','A2/A1'],cell_rows)+
        '\n\n该表描述有限测试流上观察到的得失；后验PMSE差不能单独证明原MSIC在总体上选错。\n\n'+
        f'三种比较共{len(pairs)}条配对描述，其中低RSS但高PMSE{len(low_rss_worse)}条，同形状异支撑{len(same_shape_new_support)}条；'
        '记录数不是独立样本数。A1/A2每形状只有一个全训练B，同形状必同支撑；Q5仍为**NOT_IDENTIFIABLE，实证未解决**。'
        '跨内折支撑变化不是同一个固定B被交叉验证，A0若同形状换支撑还可能来自adaptive来源被排除。\n\n'+
        table(['n100臂','N','平均q','q≥各库最大q的90%','系数范数最大值'],risk_rows)+
        '\n\n'+table(['n100比较','RSS降而PMSE升','其中q增','同形状','同支撑'],risk_pairs)+
        '\n\n近上界分别按A0最大q64、A1/A2最大q36定义，只用于描述。全部完整候选OLS数值验收通过不等于估计方差小；'
        '范数最大值也不是方差估计，未做同支撑OLS/Ridge对照，不能证明收缩必要。P1/P2继续保持[独立方案](../SECONDARY_RESEARCH_PLAN.md)，未额外实施。\n\n'
        '[所有配对](analysis/paired_changes.csv)、[逐例候选覆盖](analysis/candidate_coverage.csv)。')

    small = [groups[('n100',a)] for a in ('A0','A1','A2')]
    large = [groups[('n400',a)] for a in ('A0','A1','A2')]
    allrows = [groups[('all',a)] for a in ('A0','A1','A2')]
    cost = allrows[2]['program_wall_seconds']/allrows[0]['program_wall_seconds']
    failed_gates = [name for name,value in (confgate if confirmed else devgate).get('criteria',{}).items() if not value['passed']]
    answers = [
        f"**n100：** A0/A1/A2平均PMSE={small[0]['pmse_mean']:.6f}/{small[1]['pmse_mean']:.6f}/{small[2]['pmse_mean']:.6f}，A2/A0={small[2]['pmse_mean']/small[0]['pmse_mean']:.6f}；P95={small[0]['pmse_p95']:.6f}/{small[1]['pmse_p95']:.6f}/{small[2]['pmse_p95']:.6f}，每臂{small[0]['n_cases']}例。",
        f"**容量：** n100的A1−A0={small[1]['pmse_mean']-small[0]['pmse_mean']:+.6f}，同容量A2−A1={small[2]['pmse_mean']-small[1]['pmse_mean']:+.6f}；前者含范围/来源混合作用，后者才是CV增量。",
        f"**n400：** A0/A1/A2 Exact={large[0]['exact_count']}/{large[1]['exact_count']}/{large[2]['exact_count']}，分母各{large[0]['n_cases']}；PMSE={large[0]['pmse_mean']:.6f}/{large[1]['pmse_mean']:.6f}/{large[2]['pmse_mean']:.6f}；A0恢复但A2丢失{len(failures)}例，其中无公共真支撑{no_candidate}、有真支撑未选{missed_selection}。",
        f"**MSIC：** 全组同库A2−A1平均PMSE={allrows[2]['pmse_mean']-allrows[1]['pmse_mean']:+.6f}；逐场景变化在机制表完整列出，不以总均值掩盖尾部或把后验测试最优当作选择准则。",
        '**同规模风险：** 当前设计每形状只有一个full fixed候选，Q5不可识别、尚未实证解决；该限制不由补齐样本消失，也不据已见结果新增来源来挽救门槛。',
        f'**成本：** 成功完成程序的A2/A0实测比{cost:.3f}；旧4worker与续跑2worker混合，失败/重试成本另列，不能解释为统一调度条件下的生产速度基准。',
        f"**后续决策：{label}。** {decision_reason} "
        f"{ {'A':'全部确认门槛通过，可进入候选集成研究，本卡不发布。','B':'保留研究实现，不替换正式算法。','C':'停止当前候选方向，优先考虑支撑稳定性或估计风险的独立研究。'}[recommendation]} "
        f"未通过的完整门槛：{', '.join(failed_gates) or '无（若确认未运行，仍不可判方案A）'}。"]
    report('TASKCARD10_FINAL_DECISION.md','# 任务卡10追加执行最终报告\n\n'+context+
        f"开发{counts['development']}/48（原30复用、本次新增完成{counts['development']-30}），确认{counts['confirmation']}/48；下列结论来自{cohort}。\n\n"+
        '\n\n'.join(f'{i}. {answer}' for i,answer in enumerate(answers,1))+
        '\n\n## 完成范围\n\n三臂实现、原46项门禁、Pilot4例及P1/P2设计沿用原验收；本次追加计算修订和完整来源核验。'
        '原停止事实永久保留；最终科学判断由本次真实完成数量及门槛决定，不能用报告文件存在代替实验完成。'
        'Q5范围判定已报告，实证问题仍未解决；独立确认若因预声明开发门槛失败而未执行，属于条件性停止，不能写成确认通过。\n\n'+
        gate_table(devgate)+'\n\n[开发结果](DEVELOPMENT_RESULTS.md)、[确认状态](INDEPENDENT_CONFIRMATION.md)、'
        '[机制分析](SELECTION_MECHANISM_ANALYSIS.md)、[追加协议](DESIGN_AMENDMENT.md)、[复现与成本](REPRODUCIBILITY.md)、'
        '[最终验收](FINAL_VALIDATION.json)、[原阶段报告](../TASKCARD10_FINAL_DECISION.md)。')
    old_stop = read(root/'doc/taskcard10/results/formal/stopped_development.json')
    resource_rows = [['原开发整批（30完成+4中止）',f"{old_stop['elapsed_seconds']:.3f}",'包含失败开销，4worker'],
                     ['原task030超时进程',f"{old_stop['failed'][0]['elapsed_seconds']:.3f}",'原控制器实测；不得与整批墙钟相加'],
                     ['原task031–033中止进程','未单独记录','CPU/峰值及各进程终止墙钟不可推造']]
    for cohort in ('development','confirmation'):
        execution = raw/('execution_'+cohort)
        terminal = [p for p in (execution/'completion.json',execution/'stopped.json') if p.is_file()]
        if terminal:
            result = read(terminal[0])
            resource_rows.append(['续跑'+cohort+'整批',f"{result['elapsed_seconds']:.3f}",result['status']+'；2worker'])
        new_records = [r for r in records[cohort] if cohort=='confirmation' or r['request']['task_id']>=30]
        if new_records:
            resource_rows.extend([
                ['续跑'+cohort+'成功case墙钟累计',f"{sum(r['resources']['wall_seconds'] for r in new_records):.3f}",'并发任务累计，不是批墙钟'],
                ['续跑'+cohort+'成功case CPU累计',f"{sum(r['resources']['cpu_seconds'] for r in new_records):.3f}",'失败CPU不在此项'],
                ['续跑'+cohort+'成功case最大峰值内存（MiB）',f"{max(r['resources']['peak_working_set_bytes'] for r in new_records)/2**20:.3f}",'单进程峰值，不是同时总内存']])
    report('REPRODUCIBILITY.md','# 任务卡10追加执行复现与成本\n\n'+context+
        '使用原.venv/Scripts/python.exe与原NumPy/SciPy，四个线程变量均为1；未安装依赖。'
        '新调度器resume.py复用原experiment.worker和Job Object控制器，CLI依次prepare、run-dev及满足条件后的run-confirm。'
        '已有目录拒覆写；不得把当前命令直接再次运行作为新科学重复。\n\n'
        '[新协议](PROTOCOL.json)、[新计划](../results/continuation_01/plan.json)、[30例复用来源](../results/continuation_01/REUSE_MANIFEST.json)、'
        '[原失败记录](../results/formal/stopped_development.json)、[追加共享时钟](../results/continuation_01/formal_budget.json)、'
        '[逐例实际资源](analysis/fit_resources.csv)是成本与身份依据。'
        '原阶段整段1938.703秒，含task030超时900.016秒及3个同伴中止的时间；这些失败进程未保存完整CPU峰值，不能伪造累计CPU。'
        '成功case wall之和是并发任务时间累计，不等于整批墙钟；报告分别保留原批墙钟、追加批执行日志及成功case CPU/wall/峰值。\n\n'+
        table(['成本项','秒（内存行除外）','口径'],resource_rows)+'\n\n'
        '外部测试流重放沿用原预选12个开发/12个确认ID，并在新SCOPE仅映射原30及新18来源。Pilot4外部重放不重复执行；'
        '新结果不能扩充独立科学样本数。全部独立矩形预测、输入/测试流指纹及最大误差以外审记录为准。\n\n'
        '[外审范围](external_audit/SCOPE.json)、[开发外审汇总](external_audit/development/SUMMARY.json)、'
        '[原Pilot外审](../external_audit/pilot/SUMMARY.json)。\n\n'
        '原897项交付、索引本身及5922历史文件共6820文件受保护；新科学源码与旧源码哈希仍须匹配。'
        '最终[数值与身份验收](FINAL_VALIDATION.json)及[追加文件索引](DELIVERY_INDEX.json)封存本次结果；旧FINAL和索引不改。')
    print({'development':counts['development'],'confirmation':counts['confirmation'],'decision':label})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recommendation',choices=('A','B','C'),required=True)
    parser.add_argument('--decision-reason',required=True)
    args = parser.parse_args()
    make_reports(Path(__file__).resolve().parents[2],args.recommendation,args.decision_reason)
