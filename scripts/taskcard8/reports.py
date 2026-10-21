"""Finalize research reports only after the frozen controller has terminated."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, body):
    Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in records])


def save_csv(path, records):
    if not records:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def collect(base):
    rows, counts = [], {}
    for cohort in ('development', 'confirmation'):
        paths = sorted((base / cohort).glob('*/complete.json'))
        counts[cohort] = len(paths)
        for path in paths:
            record = read(path)
            pair = record['metrics']
            if len(pair) != 2 or {r['rule'] for r in pair} != {'baseline', 'alpha2_scale0'}:
                raise ValueError('Incomplete matched rule pair: ' + str(path))
            for key in ('cohort', 'task_id', 'cell_id', 'seed', 'n_train', 'snr', 'sparsity', 'rho'):
                if pair[0][key] != pair[1][key]:
                    raise ValueError('Mismatched pair field: ' + key)
            rows.extend(pair)
    return rows, counts


def paired(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['cohort'], row['task_id']), {})[row['rule']] = row
    return [(data['baseline'], data['alpha2_scale0']) for _, data in sorted(grouped.items())]


def mean(records, key):
    return sum(float(r[key]) for r in records) / len(records) if records else None


def performance_table(rows):
    data = []
    for cohort in ('development', 'confirmation'):
        for rule in ('baseline', 'alpha2_scale0'):
            subset = [r for r in rows if r['cohort'] == cohort and r['rule'] == rule]
            if subset:
                data.append([cohort, rule, len(subset), f"{mean(subset, 'pmse'):.6f}",
                             f"{int(sum(r['exact_recovery'] for r in subset))}/{len(subset)} ({mean(subset, 'exact_recovery'):.4f})", f"{mean(subset, 'mode_f1_macro'):.4f}",
                             f"{mean(subset, 'q_under'):.4f}", f"{mean(subset, 'q_over'):.4f}",
                             f"{mean(subset, 'near_capacity'):.4f}"])
    return table(['阶段', '规则', '完成数据', 'PMSE', 'Exact计数（率）', 'Mode-F1', 'q欠选率', 'q过选率', '近容量率'], data)


def main():
    repo = Path(__file__).resolve().parents[2]
    base = repo / 'numerical_simulation/outputs_selection_research_task8'
    target = repo / 'doc/taskcard8/analysis'
    terminal = base / 'completion.json' if (base / 'completion.json').is_file() else base / 'stopped.json'
    if not terminal.is_file():
        raise ValueError('Controller still active: refuse final reports before terminal evidence')
    termination = read(terminal)
    plan = read(base / 'plan.json')
    warm = read(repo / 'doc/taskcard8/warm_audit/status.json')
    if warm['correctness_passed'] or warm['warm_performance_allowed']:
        raise ValueError('These failure-gated reports require the actual failed Warm audit')
    inner_path = repo / 'numerical_simulation/outputs_selection_inner_task8'
    inner = read(inner_path / 'summary.json')
    verification = read(inner_path / 'independent_numerical_verification.json')
    if not inner['complete'] or not verification['passed'] or inner['outer_test_evaluations'] != 0:
        raise ValueError('Training-only feasibility evidence incomplete')
    rows, counts = collect(base)
    complete = terminal.name == 'completion.json' and counts == {'development': 108, 'confirmation': 108}
    if terminal.name == 'completion.json' and not complete:
        raise ValueError('Completion record disagrees with observed coverage')
    lock = read(base / 'development_lock.json') if (base / 'development_lock.json').is_file() else None
    qualified = bool(lock and lock['development_gate']['passed'])
    confirmation_gate = read(base / 'confirmation_summary.json')['gate'] if complete else None
    all_gates = bool(qualified and confirmation_gate and confirmation_gate['passed'])
    outcomes=[]
    for label, gate in [('开发',lock['development_gate'] if lock else None),('确认',confirmation_gate)]:
        if gate:
            values=gate['values']
            outcomes.append(f"{label}新/原平均PMSE比{values['overall_pmse_ratio']:.6f}，最坏cell比{values['worst_cell_pmse_ratio']:.6f}，门槛{'通过' if gate['passed'] else '未通过'}")
    outcome_sentence='；'.join(outcomes)+'。' if outcomes else '完整阶段门槛尚不可判定。'
    gate_rows = []
    for label, gate in [('development', lock['development_gate'] if lock else None),
                        ('confirmation', confirmation_gate)]:
        if gate:
            v = gate['values']
            gate_rows.append([label, gate['passed'], f"{v['overall_pmse_ratio']:.6f}",
                f"{v['worst_cell_pmse_ratio']:.6f}", f"{v['strong_mode_f1_delta']:.6f}",
                f"{v['strong_underselection_delta']:.6f}", f"{v['n100_near_capacity_delta']:.6f}"])
        else:
            gate_rows.append([label, '未完整', '—', '—', '—', '—', '—'])
    gate_table = table(['阶段','门槛通过','总体PMSE比','最坏cell比','SNR4 ΔMode-F1',
                        'SNR4 Δq欠选率','n100 Δ近容量率'],gate_rows)
    status = ('216数据完整，含108开发及108冻结后独立确认；均为固定候选库条件下的研究评价。'
              if complete else f"预算/门禁停止，开发{counts['development']}/108、确认{counts['confirmation']}/108；未形成完整独立确认。")
    target.mkdir(parents=True, exist_ok=True)
    serialized = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows]
    save_csv(target / 'observed_metrics.csv', serialized)
    pairs = paired(rows)
    bounds=[]
    for a,b in pairs:
        trace=read(base/a['cohort']/f"task_{a['task_id']:03d}"/'trace.json')
        maximum=max(c['q'] for c in trace['candidates'])
        bounds.append({'cohort':a['cohort'],'task_id':a['task_id'],'n_train':a['n_train'],
            'caps':json.dumps(trace['caps']),'global_Q':a['Q'],'actual_library_max_q':maximum,
            'baseline_q':a['q'],'alternative_q':b['q'],
            'baseline_at_library_max':a['q']==maximum,'alternative_at_library_max':b['q']==maximum,
            'baseline_q_over_library_max':a['q']/maximum,'alternative_q_over_library_max':b['q']/maximum})
    save_csv(target/'actual_library_bounds.csv',bounds)
    differences = [{'cohort': a['cohort'], 'task_id': a['task_id'], 'cell_id': a['cell_id'],
                    'n_train': a['n_train'], 'snr': a['snr'], 'sparsity': json.dumps(a['sparsity']),
                    'rho': json.dumps(a['rho']), 'delta_pmse': b['pmse']-a['pmse'],
                    'delta_mode_f1': b['mode_f1_macro']-a['mode_f1_macro'],
                    'delta_exact': float(b['exact_recovery'])-float(a['exact_recovery']),
                    'q_baseline': a['q'], 'q_alpha2_scale0': b['q'],
                    'q_over_Q_baseline': a['q_over_Q'], 'q_over_Q_alternative': b['q_over_Q'],
                    'caps_representable': a['caps_representable']} for a,b in pairs]
    save_csv(target / 'paired_differences.csv', differences)
    scene_rows = []
    for cohort in ('development', 'confirmation'):
        for cell in range(36):
            group = [(a,b) for a,b in pairs if a['cohort']==cohort and a['cell_id']==cell]
            if group:
                a, b = [x[0] for x in group], [x[1] for x in group]
                scene_rows.append({'cohort':cohort,'cell_id':cell,'n':len(group),'complete_3_replicates':len(group)==3,
                    'n_train':a[0]['n_train'],'snr':a[0]['snr'],'sparsity':json.dumps(a[0]['sparsity']),
                    'rho':json.dumps(a[0]['rho']),'pmse_baseline':mean(a,'pmse'),'pmse_alternative':mean(b,'pmse'),
                    'exact_baseline':mean(a,'exact_recovery'),'exact_alternative':mean(b,'exact_recovery'),
                    'under_baseline':mean(a,'q_under'),'under_alternative':mean(b,'q_under'),
                    'over_baseline':mean(a,'q_over'),'over_alternative':mean(b,'q_over')})
    save_csv(target / 'scenario_comparison.csv', scene_rows)
    worst_rows=[]
    for cohort in ('development','confirmation'):
        group=[r for r in scene_rows if r['cohort']==cohort and r['complete_3_replicates']]
        if group:
            worst=max(group,key=lambda r:r['pmse_alternative']/r['pmse_baseline'])
            worst_rows.append([cohort,worst['cell_id'],worst['n_train'],worst['snr'],
                worst['sparsity'],worst['rho'],f"{worst['pmse_alternative']/worst['pmse_baseline']:.6f}",
                f"{worst['pmse_baseline']:.6f}/{worst['pmse_alternative']:.6f}"])
    worst_table=table(['阶段','最坏完整cell','n','SNR','真形状','rho','新/原PMSE','原/新PMSE'],worst_rows)
    conditions=[]
    for cohort in ('development','confirmation'):
        for n in (100,200,400):
            for snr in (.5,2.,4.):
                group=[(a,b) for a,b in pairs if a['cohort']==cohort and a['n_train']==n and a['snr']==snr]
                if group:
                    a,b=[x[0] for x in group],[x[1] for x in group]
                    conditions.append({'cohort':cohort,'n_train':n,'snr':snr,'n_pairs':len(group),
                        **{f'{name}_{key}':mean(subset,key) for name,subset in [('baseline',a),('alternative',b)]
                           for key in ('pmse','exact_recovery','mode_f1_macro','q_over','q_under','near_capacity')}})
    save_csv(target/'condition_comparison.csv',conditions)
    mechanism_cohort='confirmation' if complete else 'development'
    condition_table=table(['阶段','n','SNR','配对数','原过选率','新过选率','原欠选率','新欠选率','原/新PMSE'],
        [[r['cohort'],r['n_train'],r['snr'],r['n_pairs'],f"{r['baseline_q_over']:.3f}",f"{r['alternative_q_over']:.3f}",
          f"{r['baseline_q_under']:.3f}",f"{r['alternative_q_under']:.3f}",
          f"{r['baseline_pmse']:.4f}/{r['alternative_pmse']:.4f}"] for r in conditions if r['cohort']==mechanism_cohort])
    if any(b['minimum_msic_q']>a['minimum_msic_q'] for a,b in pairs):
        raise ValueError('Greater linear q penalty unexpectedly increased same-library argmin q')
    minimum_vs_final=sum(a['q']!=a['minimum_msic_q'] for a,b in pairs)
    final_q_up=sum(b['q']>a['q'] for a,b in pairs)
    summary = {'research_only':True,'release_approved':False,'controller_terminal_file':str(terminal.relative_to(repo)),
               'planned_per_cohort':108,'completed':counts,'full_comparison_complete':complete,
               'development_qualified':qualified,'development_lock':lock,
               'confirmation_gate':confirmation_gate,'both_prespecified_gates_passed':all_gates,
               'pair_q_increased':sum(b['q']>a['q'] for a,b in pairs),
               'pair_q_decreased':sum(b['q']<a['q'] for a,b in pairs),
               'pair_q_same':sum(b['q']==a['q'] for a,b in pairs),
               'baseline_argmin_vs_final_q_changed':minimum_vs_final,
               'alternative_argmin_q_never_increased':True,
               'warm_performance_fits':0,'warm_combination_fits':0,
               'D2_actual_cold_fits':12,'D2_outer_test_evaluations':0,
               'interpretation':'Observed partial metrics are descriptive only if a cohort is incomplete; no selection or formal approval from a partial set.'}
    write_json(target / 'analysis.json', summary)
    perf = performance_table(rows)
    coverage_notice = ('两个阶段各36cell、每cell3次重复均齐全；完整逐cell结果见' if complete else
                       '阶段未完整时只能描述已完成配对，按场景顺序执行可能使构成失衡，不能代替108阶段总结果；逐cell重复数及完整性见')
    worst_notice = ('最差场景按预声明的cell均值比计算，不以单个最差重复替代；各阶段最差cell如下。' if complete else
                    '下表仅为当前已完成3次重复的cell中的最差，不能当作未完整阶段的门槛结果。')
    strong_rows=[]
    for cohort in ('development','confirmation'):
        for label,representable in [('全部SNR4',None),('SNR4可表示',True),('SNR4不可表示',False)]:
            group=[(a,b) for a,b in pairs if a['cohort']==cohort and a['snr']==4 and
                   (representable is None or a['caps_representable']==representable)]
            if group:
                a,b=[x[0] for x in group],[x[1] for x in group]
                strong_rows.append([cohort,label,len(group),
                    f"{mean(a,'pmse'):.6f}/{mean(b,'pmse'):.6f}",
                    f"{mean(a,'mode_f1_macro'):.6f}/{mean(b,'mode_f1_macro'):.6f}",
                    f"{mean(a,'q_under'):.6f}/{mean(b,'q_under'):.6f}",
                    f"{int(sum(r['exact_recovery'] for r in a))}/{int(sum(r['exact_recovery'] for r in b))}"])
    strong_table=table(['阶段','强信号范围','配对数','原/新PMSE','原/新Mode-F1','原/新q欠选率','原/新Exact计数'],strong_rows)
    def size_display(records,name):
        r=next(r for r in records if r['rule']==name)
        return f"{r['q']} ({','.join(map(str,r['sizes']))})"
    selected_table = table(['重复（0起）','原完整MSIC q(三模)','共同库MSIC q(三模)','内CV q(三模)','稳定性辅助 q(三模)','共同库规模数'],
       [[s['repeat']] + [size_display(s['selected'],name) for name in
          ('native_full_training_original_msic','common_grid_original_msic','inner_cv_size','stability_auxiliary_msic')]
         + [s['common_grid_count']] for s in inner['summaries']])
    cv_table = table(['重复','共同MSIC内CV MSE','CV所选内CV MSE','稳定性所选内CV MSE'],
       [[s['repeat']] + [f"{next(r['pooled_inner_cv_mse'] for r in s['selected'] if r['rule']==name):.6f}" for name in
          ('common_grid_original_msic','inner_cv_size','stability_auxiliary_msic')] for s in inner['summaries']])
    date = '任务发起日期：2026-10-16（北京时间）；实际UTC起止以冻结协议及控制器记录为准。'
    evidence = '[复现记录](taskcard8/REPRODUCIBILITY.md)；[交付索引](taskcard8/DELIVERY_INDEX.json)。'

    def doc(name, content):
        (repo / 'doc' / name).write_text(content.strip()+'\n', encoding='utf-8')

    doc('LEMON_WARM_START_COMPARISON.md', f'''# LEMON Warm/Cold比较状态

{date}

**Warm 15折未运行：NOT_RUN_CORRECTNESS_GATE_FAILED。** incoming原样方案虽通过上传6项toy测试，但未通过历史seed2026恢复、最佳已访问状态及其他正式契约。任务卡要求先过正确性，本轮因此没有Warm RMSE、R²、q80、支撑稳定性或计时结果，不以队友反馈填表，也没有重算对照。

| 指标 | 既有v5.2 Cold | Warm研究版 |
| --- | --- | --- |
| 完整OOF RMSE（3次完整OOF均值） | 0.504581 | 未运行 |
| 完整OOF R²均值 | -0.597008 | 未运行 |
| 相对协变量RMSE差 | +0.187096（协变量0.317485） | 未运行 |
| q80 | 5/15 | 未运行 |

这些Cold数字仅引用任务卡7已验收的历史OOF；没有新的Warm/Cold配对性能证据。没有创建虚假的outputs_warm_start_research完整标记。即使将来修正候选通过门禁，LEMON已经反复查看，也只能用于探索性历史比较，不可用来反复调参或宣布独立验证。

见[代码审计](WARM_START_CODE_AUDIT.md)与[正确性证据](WARM_START_CORRECTNESS_REPORT.md)。{evidence}
''')
    doc('WARM_START_GENERALIZATION_ASSESSMENT.md', f'''# Warm泛化能力评估

{date}

当前不能判断Warm是否改善LEMON或新模拟泛化，因为正确性门禁失败，性能比较按门禁停止。队友的小幅RMSE/R²改善尚不能替代同数据、同划分、同初始化/MSIC、同版本指纹的独立复核。

收到的bcd.py同时引入当前支撑groups_init、删除最佳访问状态保护/循环检测并将完整A2网格改成旧近上限网格，所以即使有整体性能变化，也不是单一暖启动因素。LEMON n116/caps(4,4,5)的两种网格本来同为80，网格改动不能自动解释该条件的预测；n400等更大caps的网格覆盖则确实可能不同。

历史无注入seed2026实例中Warm与旧warm-only-rejected的错误支撑及RSS252.229556相同，正式Cold恢复真支撑、RSS154.471958。该恢复回归证明本次候选没有克服旧否决问题；它不是新种子预测比较，也不能外推所有数据的全局最优性。

后续若研究纯Warm，应保持A2、MSIC、最佳访问状态与探索契约，先通过原回归，再冻结新种子与独立确认数据；不能根据已看LEMON继续调Warm参数。本轮不自行修成另一个算法，不发布或替换工程版本。{evidence}
''')
    doc('WARM_START_SIMULATION_REPORT.md', f'''# Warm模拟验证状态

{date}

**Phase C为NOT_RUN_CORRECTNESS_GATE_FAILED，Warm新模拟拟合0。** 原计划n100/200/400、SNR.5/2/4及至少两支撑的配对性能必须以正确性通过为前提；不能绕过门禁运行后用预测收益抵消回归失败。

独立D使用同样重要的n/SNR覆盖研究正式Cold固定候选库，并不是Warm/Cold模拟，不能借其结果声称Warm的Exact、PMSE或选阶稳定性改善。本轮历史seed2026只作既有正确性回归，既有700与任务卡7的250均未重跑。

Warm与新选阶四组E0–E3也未执行：E1/E3没有资格，E2是否值得进一步研究仅能由独立D证据判断，不是正式新算法。所有失败原样保留，详见[正确性报告](WARM_START_CORRECTNESS_REPORT.md)。{evidence}
''')
    doc('MSIC_COMPLEXITY_DIAGNOSIS.md', f'''# MSIC与模型复杂度机制诊断

{date}

## 固定研究边界

本轮只用正式冻结v5.2 Cold、原caps及完整A2整数规模网格。每个新数据只进行一次原1.5/scale1拟合，被动记录adaptive终态及全部fixed候选，再比较同库的原规则与预声明alpha2/scale0。mode alpha=1、gamma=1不变；不改BCD、不与Warm混用。后一规则属于固定库条件重评分，不是独立alpha2自适应搜索。

{status}

## 数学机制

原全局精修score为 `n log(RSS/n)+alpha*q*ell_n+ell_n*sum(log choose(p_m,s_m))`，`ell_n=log(max(log n,e))`。同一候选提高alpha1.5→2只增加`0.5*q*ell_n`，拟合项与形状组合项不变。两个不同规模/形状比较时，RSS降低的拟合收益需要超过df与组合惩罚增量，最低score才会转向较大模型；训练RSS好不等于未见数据风险好。

本轮还同时取消one-SE风格带（scale1→0），所以性能效应只能归于联合规则。原规则在最低score加`sqrt(q_min*log n)`带内优先小q；这个容忍量是确定性公式，不是验证集估计的标准误。更强alpha的score最小q通常减小或不变，但取消带内优先小模型后，最终q仍可能比原最终q更大，不能简单称为“加强惩罚必然更稀疏”。

每task的mechanism.json保存全部候选RSS、拟合项、df/组合惩罚与两套score；selection.json分开保存minimum与最终selected的q、score、容忍带及资格集合，因此可区分argmin变化和小模型优先规则变化，不需要增加事后参数臂。

实际已完成{len(pairs)}对中，原规则argmin q与最终q不同{minimum_vs_final}对，联合新规则最终q比原最终q更大{final_q_up}对；较强惩罚下的argmin q均未增大。这将“惩罚变强”与“取消带内小模型优先”的作用区分开，不能把最终q的方向仅归于alpha。

按n/SNR分层的q过选/欠选及PMSE如下；每完整层12对，仅用于解释，不能据此事后改规则：

{condition_table}

## 容量、欠选与训练内可行性

两真形状为(4,3,5)与(2,3,4)，rho0/.6，n100/200/400及SNR.5/2/4。原caps在n100可能排除第一真形状的第3模规模5；即使q≤Q，逐模不可表示仍使Exact不可达。报告单列caps_representable和强信号可表示子集，不能把这类固有欠选全归因于MSIC。

{strong_table}

全部SNR4是拟合前冻结的门槛范围；可表示/不可表示拆分仅解释机制，不替换门槛。矩形q欠选和逐模不可表示不同：不可表示形状仍可能选q64而被计为q过选，因此新规则从q64转向更小q可以提高q欠选率，却不是新产生了逐模不可表示问题。

预声明近容量率按全局Q=floor(.7n)计算；Q也不一定等于实际库的最大q。例如n200/caps(5,5,5)时最大125，小于.9Q=126，因此该频率为0不表示没有大模型或过选风险。[实际库边界表](taskcard8/analysis/actual_library_bounds.csv)另列q/库最大值及是否达到库上界，仅为描述，不回改原门槛。

仅训练内实验固定三个重复第一外折，每次77/78人重新预处理与支撑搜索。full Q81、80fixed规模，inner Q53/53/54与caps(3,4,4)，共同fixed规模48：

{selected_table}

{cv_table}

前两组native q80不在共同库，不能给它填造同尺度CV分数；共同库MSIC已转成q1/q2，说明容量库限制本身就改变结论。CV与共同库MSIC才是在同库比较，稳定性优先未显示一致好处，例如repeat0选q32而CV MSE0.299771高于CV所选0.120019。内CV分数参与选择，存在选择乐观性，不能代替外层或独立数据泛化评价。

相同q也可能对应不同三模形状、支撑和内验证误差；表中同时列出三模规模，不能把q相同误写成同一模型。

12次真Cold拟合、432个内验证候选及全部保存预处理/系数已独立数值验收，外测试评估0。详[训练内summary](../numerical_simulation/outputs_selection_inner_task8/summary.json)与[全144候选图](taskcard8/figures/inner_selection_feasibility.pdf)。{evidence}
''')
    doc('MSIC_SELECTION_COMPARISON.md', f'''# 预声明MSIC候选的开发与独立确认比较

{date}

{status}

## 设计及选择门禁

用户批准36场景×每阶段3重复，共108开发+108独立确认、全线7200秒截止。新seed分属1806101600与2806101600两段，按cell×1000+replicate，test另加50000000；432种训练/测试seed不重复，与已审计11份旧顶层计划的916个材料化seed无交集，不对未公开外部seed作无依据保证。

沿正式700的实际k_max1/max_cycles6等配置，而非dataclass通用默认；dims32×20×25、n_test500、两连续矩形支撑、rho0/.6。4个独立worker各单线程，完整32源码、真实import、数据/库/测试流指纹和预算均锁定；生产源码无修改。

预声明开发候选仅alpha2/scale0。门槛同时要求整体PMSE比≤.98、最坏cell比≤1.15、SNR4 Mode-F1下降≤.02、SNR4 q欠选率增加≤.05、n100近容量率不增，不能只按平均PMSE挑规则。只有完整108开发后冻结lock及其artifact SHA，才执行未见确认；开发未过时的双规则确认仅为固定规则研究复核，不能反称开发合格。

## 实际结果

{perf}

{gate_table}

{worst_notice}

{worst_table}

{coverage_notice}[场景表](taskcard8/analysis/scenario_comparison.csv)，完整所选指标及[逐任务配对](taskcard8/analysis/paired_differences.csv)均保留。

主指标之外同时保存signal MSE、REE、Element-F1、q/Q、RSS、Exact和成本。欠选/过选按矩形q与真q比较，不等同于元素FN/FP；近容量定义q≥.9 floor(.7n)。两规则共享一次候选库拟合，fit_sec_shared不能重复相加或当作两个独立算法的速度比。

每cell只有3个seed，门槛是预声明研究判据，不是已完成普适性或功效证明；图中展示全配对数据，不把跨情景离散度当确认置信区间。fit_sec_shared含被动记录/系数捕获开销，且四worker并行，不可当未经观测器的生产原生速度或与旧Linux集群直接比较。

[全配对预测差异图](taskcard8/figures/paired_msic_prediction.pdf)分开显示开发与确认，全部实际观测点均保留；PDF/SVG/TIFF/PNG及字号、对齐、碰撞检查见[图形验收](taskcard8/figures/QA_REPORT.json)。

开发合格状态为 `{qualified}`；本轮始终 `research_only=true, release_approved=false`。历史alpha2/scale0的A3研究、旧A2预算停止确认及已经看过的LEMON都不计作本轮独立确认。{evidence}
''')
    decision = ('固定双规则在本轮开发与独立确认均达到预声明门槛，值得进入完整fresh-refit等价核验和独立实证；这仍不等于可以自动发布。'
                if all_gates else '虽完成固定双规则独立确认，但至少一个预声明门槛未通过，不能声称已同时解决过选与欠选，也不能仅凭平均PMSE改善批准。'
                if complete else '预定开发/确认未完整，当前没有足够完整证据批准新选阶规则；不得放宽门槛或使用已完成子集批准。')
    doc('MSIC_NEW_METHOD_DECISION.md', f'''# 新选阶方法决策

{date}

{status} {outcome_sentence} {decision}

原alpha1.5/scale1仍为正式默认；alpha2/scale0只作已冻结联合候选，不把惩罚效应与one-SE效应混为一谈，不称固定库重评分为完整新自适应算法。开发是否合格与独立确认结果分别报告，任何未完成/失败证据均不允许“差不多通过”。

训练内CV可行但其训练样本容量小于full，当前48规模共同库与full80规模不可直接混同。稳定性辅助在三组训练内没有一致优势；较大支撑本身更容易重叠，最大化Jaccard不是已证明的泛化控制。CV选出较低的同一批验证误差也有选择乐观性，仍需外层或新的独立数据。

现阶段建议继续用v5.2作为冻结工程基线；可以设计新的预声明选阶研究，但不据LEMON设q_max=6、不挑外测最优alpha、不回退已知缺陷BCD，不发布v6。若本轮预算停止，原计划保留且不得事后扩预算续成“原协议完整”；新研究须独立解释、冻结来源/种子/预算及确认门槛。{evidence}
''')
    doc('WARM_START_FINAL_DECISION.md', f'''# 暖启动最终阶段决策

{date}

**不采用收到的原样Warm方案，也不启动它的性能比较。** 它同时改变暖启动、候选网格及正确性保护，尚不能分离纯暖启动贡献；历史seed2026真实恢复失败与旧否决草案一致。

每版本31顶层检查（27算法、4工具）：Cold31/31通过；Warm20方法全过、11方法失败/错误，9 failure和11 error为含subtest的事件数，不是20个独立失败方法。上传6/6虽过，不能代替历史恢复、最优访问状态、循环、有限性及兼容性门禁。

因此LEMON Warm RMSE/R²/q80及新模拟Exact/PMSE/稳定性没有结果；E0–E3交互没有执行，更不能凭队友LEMON改善忽略正确性。正式v5.2及队友版本不变，incoming原字节、研究副本和失败证据保留。详[代码审计](WARM_START_CODE_AUDIT.md)、[正确性报告](WARM_START_CORRECTNESS_REPORT.md)。{evidence}
''')
    doc('MSIC_RESEARCH_STATUS.md', f'''# MSIC研究状态

{date}

主D：{status} 完整开发资格=`{qualified}`，全部结果为冻结Cold固定库上的两规则比较，未改正式算法。

训练内可行性D2已完成12次真实v5.2拟合、7项测试及独立数值验收；每组共同网格48，原完整q80/80/24、共同MSIC q1/2/24、CV q2/2/4、稳定性q32/2/24。没有外测评价或新准则独立泛化验证。

主模拟按固定36cells全覆盖、开发与确认分离，若预算停止保留全部已完成/未完成/日志，未改配置、减重复、换seed或忽略输入Gate。主协议/lock/终态见[实验目录](../numerical_simulation/outputs_selection_research_task8/plan.json)，派生状态见[analysis](taskcard8/analysis/analysis.json)。任务卡7的250集群任务未纳入或启动，历史700亦未重跑。{evidence}
''')
    doc('TENSOR_ABESS_NEXT_VERSION_DECISION.md', f'''# Tensor-ABESS下一版本研究决策

{date}

## 一页阶段结论

**Warm：** 收到真实bcd.py并原样隔离到32文件研究版，但它删除best-visited保护、改回旧候选网格；历史seed2026恢复失败，Cold同测通过。上传toy测试通过不足以接受该候选，Warm LEMON/模拟/组合性能线全部NOT_RUN。

**选阶：** {status} {outcome_sentence} 两规则为原alpha1.5/scale1与预声明alpha2/scale0，正式Cold搜索不变；不能把联合规则效应归于alpha单项，也不能把固定库结果叫独立新算法。

**训练内：** 12次重拟合支持CV/稳定性机制可执行，但容量变化已使common MSIC的q80/80降至q1/2，稳定性无一致优势。所有CV分数参与选择，不能据其较低宣布外泛化提高。

**正式版本：** 保留v5.2-search-final，不替换生产或HCP/NKI；v5.3-warm-start-research与v6-selection-research均research_only、release_approved=false。当前没有足够证据自动提出下一正式版。

## 八项回答

1. Warm具体把当前mode支撑传给Group-BESS初始化，还附带移除最佳访问状态/校验/循环及缩减候选网格，不能视为纯Warm。
2. 上传6测试通过，完整31项中11方法失败/错误，原历史恢复未通过，Cold31全过。
3. Warm未进入LEMON，RMSE/R²/q80/稳定性影响均未知；不补造队友数字。
4. Warm新模拟未运行，Exact与PMSE稳定性未知。
5. 既有LEMON大q及弱信号/小样本风险有机制证据，本轮新模拟须以实际完整性和逐cell数据限定结论，逐模caps不可达须分开解释。
6. {decision} 见[选择比较](MSIC_SELECTION_COMPARISON.md)与[方法决策](MSIC_NEW_METHOD_DECISION.md)。
7. Warm与新MSIC组合没有资格执行，交互未知；不能从两个单独现象外推相互促进。
8. 继续用v5.2，保留负面/停止记录；若研究下一候选，另立预声明正确性、开发与独立确认方案，不根据已看LEMON调参。

## 交付

- [Warm代码审计](WARM_START_CODE_AUDIT.md)及[正确性门禁](WARM_START_CORRECTNESS_REPORT.md)
- [LEMON Warm状态](LEMON_WARM_START_COMPARISON.md)、[泛化边界](WARM_START_GENERALIZATION_ASSESSMENT.md)、[模拟状态](WARM_START_SIMULATION_REPORT.md)
- [MSIC机制](MSIC_COMPLEXITY_DIAGNOSIS.md)、[选择比较](MSIC_SELECTION_COMPARISON.md)、[方法决策](MSIC_NEW_METHOD_DECISION.md)
- [团队说明](TEAM_RESEARCH_UPDATE.md)、{evidence}

保护原源码/旧模拟/旧实证/incoming及既有用户launcher改动；本轮不提交、不发消息、不生成新算法ZIP。
''')
    doc('TEAM_RESEARCH_UPDATE.md', f'''# 团队研究进度说明（待转发，未发送）

{date}

正式v5.2继续冻结，HCP/NKI按原确认设计继续，不需要更换代码、重启正确运行的实验或为了匹配LEMON调整参数/筛选数据。

本项目已原样审查队友Warm：上传toy测试通过，但历史恢复与正式正确性门禁失败，且混入候选网格变化，因此没有启动Warm性能/组合研究，也未否认队友在其数据上的描述性反馈。

独立MSIC研究：{status} 训练内CV/稳定性可行性已完成，但不能把选择后的CV或未完整确认视为已发布改进。新研究身份均research_only=true/release_approved=false。

请继续完整保留各折OOF、q及支撑、训练/测试误差、输入/划分/源码/配置/种子指纹和失败记录；后续提供Warm/Cold结果时优先检查上述身份、原回归和测试独立性。HCP/NKI完整结果尚未提供，本轮不推断其表现，也未发送本说明。{evidence}
''')
    print('Finalized ten research reports; coverage:', counts, 'complete:', complete, 'qualified:', qualified)


if __name__ == '__main__':
    main()
