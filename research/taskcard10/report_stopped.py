"""Report an explicitly stopped Task10 study without evaluating incomplete gates."""
from pathlib import Path
import numpy as np

from research.taskcard10.analyze import read, summarize, table, write_csv, write_json


def write_stopped_report(root):
    out = root/'doc/taskcard10'
    formal_dir = out/'results/formal'
    stopped = read(formal_dir/'stopped_development.json')
    plan = read(formal_dir/'plan.json')
    requests = [r for r in plan['requests'] if r['cohort']=='development']
    completed = {int(p.parent.name.split('_')[1]) for p in (formal_dir/'development').glob('task_*/complete.json')}
    failed = {r['task_id'] for r in stopped['failed']}
    pending = set(stopped['not_started'])
    assert completed == set(stopped['completed'])
    assert not (completed & failed or completed & pending or failed & pending)
    assert completed | failed | pending == {r['task_id'] for r in requests}
    assert not (formal_dir/'confirmation').exists()
    assert not (formal_dir/'development_lock.json').exists()
    cohorts = {}
    for cohort,folder in [('pilot','pilot'),('development','formal')]:
        cohorts[cohort] = [dict(read(p),_summary=read(p.parent/'summary.json'))
                          for p in sorted((out/'results'/folder/cohort).glob('task_*/complete.json'))]
    assert len(cohorts['pilot']) == 4
    destination = out/'analysis'
    assert not destination.exists()
    counts = {'pilot':4,'development':len(completed),'confirmation':0}
    all_rows,all_stats,all_pairs,all_cells = [],[],[],[]
    coverage,resources = [],[]
    for cohort,records in cohorts.items():
        rows,stats,pairs,cells = summarize(records,cohort)
        all_rows.extend(rows); all_stats.extend(stats); all_pairs.extend(pairs); all_cells.extend(cells)
        for record in records:
            req,ev = record['request'],record['recovery_evidence']
            coverage.append({'cohort':cohort,'task_id':req['task_id'],'n':req['config']['n_train'],
                             'snr':req['config']['snr'],'truth_shape':ev['truth_sizes'],
                             **{k:ev[k] for k in ('truth_shape_native_representable','truth_shape_common_representable',
                                  'truth_shape_found_full_pool','truth_shape_found_common_pool',
                                  'truth_support_found_full_pool','truth_support_found_common_pool')},
                             **{arm+'_exact':int(record['metrics'][arm]['exact_recovery']) for arm in ('A0','A1','A2')}})
            resources.append({'cohort':cohort,'task_id':req['task_id'],'n':req['config']['n_train'],
                              **record['_summary']['counters'],**record['resources']})
    # No files are written until the terminal state and all completed metadata parse.
    destination.mkdir()
    for name,rows in [('selected_metrics',all_rows),('group_metrics',all_stats),('paired_changes',all_pairs),
                      ('cell_metrics',all_cells),('candidate_coverage',coverage),('fit_resources',resources)]:
        write_csv(destination/(name+'.csv'),rows)
    gate = {'status':'NOT_EVALUABLE_INCOMPLETE','passed':None,
            'reason':'Development stopped at the frozen per-case deadline; planned 48, completed 30'}
    write_json(destination/'summary.json',{'counts':counts,'status':'STOPPED_PARTIAL_DEVELOPMENT',
               'planned_development':len(requests),'development_gate':gate,'confirmation_gate':None,
               'statistics':all_stats,'stopped_development':stopped})
    write_json(formal_dir/'confirmation_NOT_RUN.json',{'status':'NOT_RUN',
               'reason':'Development stopped at frozen per-case timeout; incomplete development cannot enter confirmation',
               'development_completed':len(completed),'development_planned':len(requests),
               'numeric_gates_evaluated':False,'stop_evidence':'stopped_development.json',
               'budget_extended':False,'rerun_started':False})
    write_json(out/'external_audit/development_NOT_RUN.json',{'status':'NOT_RUN',
               'reason':'Frozen 12-case scope is incomplete after development stop; no replacement or partial replay',
               'planned_task_ids':[0,4,8,12,16,20,24,28,32,36,40,44],
               'completed_selected_task_ids':[0,4,8,12,16,20,24,28],
               'missing_completed_task_ids':[32,36,40,44],'regenerated_cases':0})
    write_json(out/'external_audit/confirmation_NOT_RUN.json',{'status':'NOT_RUN',
               'reason':'Scientific confirmation was not run','regenerated_cases':0})

    def report(name,body):
        with (out/name).open('x',encoding='utf-8') as handle:
            handle.write(body.rstrip()+'\n')

    warning = ('**开发中止，以下只描述已完成样本，不是完整开发结果或独立确认。** '
               '冻结计划48例，完成30例：n100为24/24，n400为6/24且仅SNR0.5；n400尚有2个弱信号例未完成，SNR2/4均未完成。'
               '按运行顺序和耗时留下的样本不具代表性，整体均值及最坏已见场景不能代替冻结24场景门槛。\n\n')
    pilot = [r for r in all_rows if r['cohort']=='pilot' and r['arm']=='A0']
    report('PILOT_FEASIBILITY_REPORT.md','# Pilot可行性与预算\n\n'
           '4个新随机族，n100/n400×SNR0.5/4，真435、rho0.6；三臂完整执行，逐例外测前独立审计通过。Pilot只用于实现与成本，不进入开发/确认，不据外测调规则。\n\n'+
           table(['Task','n','Wall秒','CPU秒','峰值MiB'],[[r['task_id'],r['n'],round(r['task_wall_seconds'],3),round(r['task_cpu_seconds'],3),round(r['task_peak_memory_bytes']/2**20,2)] for r in pilot])+
           '\n\n根据2worker Pilot成本估计，开发/确认各48例、4worker的开发预算估计2778.942秒，按1.5倍确认储备及60秒余量，总7007.355秒，低于冻结7200秒。900秒单case上限未改。实际开发已完成的6个n400任务耗时536.578–841.328秒，高于Pilot的382.469–417.594秒；task030在900.016秒被控制器判超时。场景与并发条件均有变化，不能唯一归因为场景难度或并发；Pilot成本不能保证所有条件下均可行。完整开发及确认因此未完成。\n\n'
           '种子草案冲突、Windows taskkill拒绝及测试加载冲突均在科学生成前解决，失败记录保留；46项联合门禁通过后才启动Pilot。标准Job Object在超时关闭自有进程树，不修改权限。\n\n'
           '[原完成记录](results/pilot/completion.json)、[预算决策](BUDGET_DECISION.json)、[开发停止证据](results/formal/stopped_development.json)。')
    devstats = [r for r in all_stats if r['cohort']=='development']
    devpairs = [r for r in all_pairs if r['cohort']=='development']
    devrows = [r for r in all_rows if r['cohort']=='development']
    stats_table = table(['组','臂','N','PMSE均值','中位数','P95','Exact','平均q','欠选率'],
                       [[r['group'],r['arm'],r['n_cases'],f"{r['pmse_mean']:.6f}",f"{r['pmse_median']:.6f}",f"{r['pmse_p95']:.6f}",r['exact_count'],f"{r['q']:.3f}",f"{r['q_under']:.4f}"] for r in devstats])
    report('DEVELOPMENT_RESULTS.md','# 开发结果：中止后的部分诊断\n\n'+warning+
           'task030达到900秒单case期限，task031/032/033因同伴失败终止，task034–047未启动。整段1938.703秒；没有超出7200秒才停止，也不能用剩余总预算覆盖单case期限。各未完成目录原样保留，未补算、未重启、未裁剪困难case。9项联合门槛为NOT_EVALUABLE_INCOMPLETE，没有development_lock；完整确认进入条件不满足。\n\n'+
           stats_table+'\n\n'+table(['比较（右−左）','改善/同/恶化','平均ΔPMSE'],
                  [[b+'−'+a,'/'.join(str(sum(r['outcome']==v for r in devpairs if r['left']==a and r['right']==b)) for v in ('improved','equal','worse')),
                    f"{np.mean([r['delta_pmse'] for r in devpairs if r['left']==a and r['right']==b]):.6f}"] for a,b in [('A0','A1'),('A1','A2'),('A0','A2')]])+
           '\n\nn100组24例覆盖全部预声明n100场景，可描述该组平均与尾部；每cell仅2重复，不作高功效推断。n400仅6例以及不平衡总体禁止用于推广判断。全部PMSE/P90/P95/max、REE/F1、q、支撑、系数范数和配对差保留，不删除恶化极值。'
           '\n\n[逐场景](analysis/cell_metrics.csv)、[逐数据配对](analysis/paired_changes.csv)、[完整所选指标](analysis/selected_metrics.csv)、[实际计算/缓存计数及资源](analysis/fit_resources.csv)、[终态](results/formal/stopped_development.json)。')
    report('INDEPENDENT_CONFIRMATION.md','# 独立确认：NOT_RUN\n\n'
           '**未执行。** 开发因900秒单case期限中止，只完成30/48例，未形成完整开发锁和9项门槛验收；不能进入确认。确认请求已预注册，确认训练输入/测试流均未生成，未用Pilot或历史数据替代。\n\n'
           '冻结7200秒总预算未延长，900秒上限未改变。不能用剩余预算补跑超时任务或在部分结果上调参后重新确认。任何后续实验须在独立新协议中先解决计算可行性，使用真正新种子族，并重新执行完整开发/确认；本卡不启动。\n\n'
           '[NOT_RUN记录](results/formal/confirmation_NOT_RUN.json)、[停止证据](results/formal/stopped_development.json)、[冻结计划](FROZEN_EXPERIMENT_PLAN.md)。')
    coverage_rows = []
    keys = ('truth_shape_native_representable','truth_shape_common_representable','truth_shape_found_full_pool',
            'truth_shape_found_common_pool','truth_support_found_full_pool','truth_support_found_common_pool')
    for n in (100,400):
        for shape in ([4,3,5],[2,3,4]):
            rows = [r for r in coverage if r['cohort']=='development' and r['n']==n and r['truth_shape']==shape]
            coverage_rows.append([n,str(shape),len(rows),*[sum(r[k] for r in rows) for k in keys]])
    lowerworse = [r for r in devpairs if r['delta_rss'] < -1e-10 and r['delta_pmse'] > 1e-12]
    same = [r for r in devpairs if r['same_shape'] and not r['same_support']]
    largecov = [r for r in coverage if r['cohort']=='development' and r['n']==400]
    loss = [r for r in largecov if r['A0_exact'] and not r['A2_exact']]
    cause = [sum(not r['truth_support_found_common_pool'] for r in loss),
             sum(r['truth_support_found_common_pool'] for r in loss)]
    smallrisk = []
    for arm in ('A0','A1','A2'):
        selected = [r for r in devrows if r['n']==100 and r['arm']==arm]
        smallrisk.append([arm,len(selected),f"{np.mean([r['q'] for r in selected]):.3f}",
                          sum(r['near_selection_limit'] for r in selected),
                          f"{max(r['coefficient_norm'] for r in selected):.6f}",
                          f"{max(r['pmse'] for r in selected):.6f}"])
    commoncells = []
    for cell in sorted({r['cell_id'] for r in devrows}):
        a = [r for r in devrows if r['cell_id']==cell and r['arm']=='A1']
        b = [r for r in devrows if r['cell_id']==cell and r['arm']=='A2']
        commoncells.append([cell,a[0]['n'],a[0]['snr'],str(a[0]['truth_shape']),a[0]['rho'],len(a),
                            f"{np.mean([r['pmse'] for r in b])-np.mean([r['pmse'] for r in a]):+.6f}",
                            f"{np.mean([r['pmse'] for r in b])/np.mean([r['pmse'] for r in a]):.6f}"])
    report('SELECTION_MECHANISM_ANALYSIS.md','# 选择机制与估计风险：部分诊断\n\n'+warning+
           '## 候选可达与位置命中\n\n'+table(['n','真形状','已完成N','原生可表示','公共可表示','原生真形状入池','公共真形状入池','原生真支撑入池','公共真支撑入池'],coverage_rows)+
           '\n\n所有表内数量均为已完成分母。可表示不等于找到真位置；真形状入池也不等于真支撑入池。n400两种真形状仍在公共网格，n100真435在原网格已不可达。'
           f' 已完成n400中A0恢复而A2未恢复共{len(loss)}例；其中公共库已无真支撑{cause[0]}例、公共库有真支撑但CV未选{cause[1]}例。前者属于候选范围/来源排除，后者是已有真支撑未被最终选择，均不能由这6例概括全部n400。\n\n'
           '## 同容量CV与原MSIC\n\n下面只描述有限测试流上的观测差异，负Δ表示CV预测更好，不将后验测试优劣称为原MSIC在总体意义上选错。\n\n'+
           table(['cell','n','SNR','真形状','rho','完成重复','A2−A1 PMSE','A2/A1'],commoncells)+
           '\n\nA1−A0混合公共范围缩小和删除adaptive来源，不能唯一归因容量。A2−A1才是同一fixed程序库及相同全训练候选上的选阶差。\n\n'
           f'## 同形状风险与OLS\n\n开发三种配对合计{len(devpairs)}条记录（不是独立样本数），训练RSS下降但PMSE升{len(lowerworse)}条，同形状异支撑{len(same)}条。'
           'A1/A2每形状只有一个全训练B，同形状必同支撑，因而同形状多来源风险排序为NOT_IDENTIFIABLE；A0的adaptive来源可能造成同形状异支撑，不能独立归因CV。跨折支撑变化是训练样本变化，不是同一个固定B被交叉验证。\n\n'
           '所有完整保存候选的OLS、RSS、秩、条件数及预测已独立核验；fixed候选另验边际起点和best-visited。数值正确不等于有限样本估计方差小；没有发现OLS实现错误，也未实施Ridge。\n\n'+
           table(['n100臂','N','平均q','q≥各库最大q的90%','系数范数最大值','PMSE最大值'],smallrisk)+
           '\n\n近上界按A0最大q64、A1/A2最大q36分别定义，只用于描述；范数最大值不是方差估计。没有同支撑不同估计器对照，不能证明Ridge必要。q/n、系数范数及同支撑差异仅是诊断关联，不能把选模与估计作用唯一分离。不同重复的真位置变化，不用跨重复原始支撑Jaccard解释恢复稳定性。\n\n'
           '[逐例候选覆盖](analysis/candidate_coverage.csv)、[全部配对](analysis/paired_changes.csv)、[次级独立研究方案](SECONDARY_RESEARCH_PLAN.md)。')
    index = {(r['group'],r['arm']):r for r in devstats}
    small = [index[('n100',a)] for a in ('A0','A1','A2')]
    large = [index[('n400',a)] for a in ('A0','A1','A2')]
    small_ratio = small[2]['pmse_mean']/small[0]['pmse_mean']
    cost = index[('all','A2')]['program_wall_seconds']/index[('all','A0')]['program_wall_seconds']
    answers = [
        f"**n100：** 已完成24/24。A0/A1/A2平均PMSE={small[0]['pmse_mean']:.6f}/{small[1]['pmse_mean']:.6f}/{small[2]['pmse_mean']:.6f}，A2/A0={small_ratio:.6f}；各自P95={small[0]['pmse_p95']:.6f}/{small[1]['pmse_p95']:.6f}/{small[2]['pmse_p95']:.6f}。仅为开发描述，不能代替确认。",
        f"**容量：** n100的A1−A0平均PMSE={small[1]['pmse_mean']-small[0]['pmse_mean']:+.6f}，A2−A1={small[2]['pmse_mean']-small[1]['pmse_mean']:+.6f}。前者混合范围与adaptive来源排除，不能证明容量单独贡献；后者才是同库CV增量。",
        f"**n400：** 只完成6/24且均SNR0.5，A0/A1/A2 Exact={large[0]['exact_count']}/{large[1]['exact_count']}/{large[2]['exact_count']}，分母各6，无法判断是否保护完整恢复优势。A0恢复而A2未恢复{len(loss)}例：公共库失去真支撑{cause[0]}例、有真支撑未选{cause[1]}例；两种真形状在公共网格均可表示。",
        '**MSIC：** 同库A2−A1的逐场景PMSE差见机制表；它能描述有限样本上选择得失，无法将部分结果概括为原MSIC普遍选错，也不足以验证完整程序推广。',
        '**同规模风险：NOT_IDENTIFIABLE。** 每形状仅一个全训练fixed候选，无法直接检验CV对同形状异支撑多来源的风险排序；本卡没有解决该问题。',
        f'**成本：** 已完成30例的完整程序A2/A0实测时间比约{cost:.3f}；这是含native全库和额外CV的实现成本，且遗漏超时及peerabort成本，不能当整个计划成本比。n400出现900秒超时，当前冻结预算下完整程序可行性未获验证，收益是否值得成本无法确认。',
        '**是否继续修改：暂定方案B，仅保留研究实现。** n100存在同容量CV的描述性改善，但n400恢复、完整尾部及计算可行性尚未验证；本轮按超时停止规则终止依赖确认，不替换正式算法。A/B/C的完整科学判定仍未完成，超时不等于关键正确性失败或CV已被证明无效；后续重启需要新协议和预算。']
    report('TASKCARD10_FINAL_DECISION.md','# 任务卡10最终研究决策\n\n2026-10-18。**研究执行中止，诊断交付完成；正式算法仍为v5.2-search-final。**\n\n'+
           '\n\n'.join(f'{i}. {answer}' for i,answer in enumerate(answers,1))+'\n\n## 执行结果与停止边界\n\n'+warning+
           '46项联合正确性门禁通过（原Cold16、CV13、审计4、控制器13）；外部审计工具另4项，不混加。Pilot4/4完整通过，1932候选OLS及4例独立测试流重放通过。开发30例完成并逐例在外测前审计，task030达到900秒，031–033终止、034–047未启动。没有开发锁、没有完整9门槛结果，确认0例且NOT_RUN。总预算未延长，不补跑、调参或更换困难样本。\n\n'
           'Pilot未覆盖24场景及4worker下的尾部耗时，原7007.355秒预测未保证单case可行性。超时记录不证明OLS或BCD数学实现故障；本卡没有出现核心正确性或已保存模型数值断言失败。需要在独立新协议中先解决计算可行性，不能把本卡部分结果当完整推广证据。\n\n'
           'A1−A0不是容量唯一因果效应，A2−A1才是同库选阶比较；Q5多来源排序不可识别。2重复/cell精度有限；CV选择分数不当独立泛化成绩。P1支撑内收缩与P2无张量基线仅设计，既有LEMON受试者换划分不能成为全新独立确认。未实施Warm、Ridge替代、LEMON OOF或版本发布。\n\n'
           '[设计](DESIGN_AND_PROTOCOL.md)、[正确性](CV_LEAKAGE_AND_CORRECTNESS.md)、[Pilot](PILOT_FEASIBILITY_REPORT.md)、[冻结计划](FROZEN_EXPERIMENT_PLAN.md)、[开发](DEVELOPMENT_RESULTS.md)、[确认](INDEPENDENT_CONFIRMATION.md)、[机制](SELECTION_MECHANISM_ANALYSIS.md)、[后续方案](SECONDARY_RESEARCH_PLAN.md)、[复现](REPRODUCIBILITY.md)、[最终验收](FINAL_VALIDATION.json)。')
    print({'counts':counts,'scientific_status':'STOPPED_PARTIAL_DEVELOPMENT','confirmation':'NOT_RUN'})


if __name__ == '__main__':
    write_stopped_report(Path(__file__).resolve().parents[2])
