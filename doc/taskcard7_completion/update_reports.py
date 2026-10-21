"""Update only the snapshotted Task 7 reports after complete read-only acceptance."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    out=Path(__file__).resolve().parent
    repo=out.parents[1]
    analysis=read(out/'analysis.json')
    assert read(out/'cluster_audit.json')['status']==read(out/'estimate_audit.json')['status']=='PASS'
    assert analysis['complete'] and analysis['task_labels']==250
    scenes=analysis['scenarios']
    perf=table(['场景（各50配对）','v5 PMSE','v5.1 PMSE','v5.2 PMSE','ΔBCD','ΔGrid','Exact：v5→v5.1→v5.2'],[
        [s['name']]+[f"{s['versions'][v]['pmse']:.6f}" for v in ('v5','v51','v52')]+
        [f"{s['bcd_delta_pmse']:+.6f}",f"{s['grid_delta_pmse']:+.6f}",
         '→'.join(str(s['versions'][v]['exact_count']) for v in ('v5','v51','v52'))] for s in scenes])
    cost=table(['场景','v5秒','v5.1秒','v5.2秒'],[
        [s['name']]+[f"{s['versions'][v]['runtime_sec']:.3f}" for v in ('v5','v51','v52')] for s in scenes])
    with (out/'effect_summary.csv').open(encoding='utf-8',newline='') as handle:
        effects=[r for r in csv.DictReader(handle) if r['metric']=='pmse']
    effects_table=table(['场景','阶段','下降/相同/上升','中位ΔPMSE'],[
        [r['scenario_name'],r['phase'],f"{r['decreased']}/{r['tied']}/{r['increased']}",
         f"{float(r['median_delta']):+.6f}"] for r in effects])
    evidence='[本轮复现与边界](taskcard7_completion/REPRODUCIBILITY.md)；[本轮索引](taskcard7_completion/DELIVERY_INDEX.json)。'
    scope='2026-10-17（北京时间）。指定五场景各50，250/250已回传并通过只读验收；本轮新增拟合0、未重新merge，原计划、失败记录及原始结果保留。'
    boundary='本轮没有归档的原X/y或逐测试流数组，未在Windows独立重算RSS、PMSE、signal MSE或R²；这些指标来自通过封存身份核验的原记录。测试配对依据原生成器/配置/test_seed和原Linux环境记录，不能称逐数组测试流指纹已重建。'
    data='完整[250任务配对表](taskcard7_completion/paired_effects.csv)、[15个版本/场景全指标](taskcard7_completion/version_metrics.csv)、[逐指标配对变化](taskcard7_completion/effect_summary.csv)与[极值个案](taskcard7_completion/extreme_cases.csv)均保留。'
    updates={}
    updates['doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md']=f'''# 三版本效果拆分：完整250任务归因

{scope}

## 两段版本效应

`ΔBCD=PMSE(v5.1)−PMSE(v5)`；`ΔGrid=PMSE(v5.2)−PMSE(v5.1)`，逐任务两者之和精确对应总差（容差1e−12）。平均差为场景内50配对的描述统计。

{perf}

n100的平均恶化全部发生在BCD修复后，28/50恶化、19相同、3改善；v5.1/v5.2系数和支撑50/50精确相同。n400则BCD单独几乎不改变平均PMSE和Exact；网格补全后49改善、1相同、0恶化，Exact增加48，说明该场景的恢复收益主要来自A2。baseline、弱信号及rho0的网格效应也为零；这不能外推未纳入的DGP。

{effects_table}

## 机制与边界

本批200个非n400任务的v5.1/v5.2 B、支撑精确相同；n400只有1/50相同。原caps下n100两版fixed网格同为64，所选n200场景125，n400由150扩至252；计数均不包括额外adaptive候选。n400真规模(4,3,5)进入fixed网格，但旧adaptive候选仍可能恢复，不能把旧整个候选池称为绝无真模型。

n100的逐模caps本身排除真形状第3模规模5，三版本Exact均0；其预测退化可归因于本批BCD版本干预，不能仅凭这两段最终输出证明每次退化都由q增大或单一MSIC参数造成。LEMON已有候选轨迹和训练内证据另见[机制报告](BCD_MSIC_INTERACTION_REPORT.md)。

250个完整输入唯一，跨场景只含150组seed/test_seed族，不作250 IID的池化显著性推断。每场景的50配对用于描述；旧700中600唯一组合的138→182是历史标签加权结果。

{boundary}

{data}

[全配对效果与恢复图](taskcard7_completion/figures/version_decomposition.pdf)保留两段各250观测点及三版本恢复计数；成本见[接收报告](V51_FULL_ATTRIBUTION_REPORT.md)。{evidence}
'''
    updates['doc/V51_FULL_ATTRIBUTION_REPORT.md']=f'''# v5.1完整250任务归因接收报告

{scope}

## 完整性验收

| 项目 | 结果 |
| --- | --- |
| 指定任务JSON / NPZ | 250 / 250，恰好覆盖原任务清单 |
| Pilot Gate | 5，绑定对应结果seal |
| 原生plan重建 | 精确一致；seal `{analysis['plan_seal']}` |
| 三版源码 / 共享依赖 | 96源码及6共享依赖、冻结worker匹配 |
| 环境 / 四项训练指纹 | 所有250封存记录与原Linux计划一致 |
| 三版本NPZ / CSV | 750 NPZ原字节SHA通过；已有CSV250行每个字段与原merge逻辑一致 |
| 独立模型指标 | 750所选模型合法支撑、REE/F1/Exact/MSIC及loss恒等式通过 |
| 本轮重新拟合 / merge | 0 / 0 |

独立数值误差：支撑指标≤2.23e−16、REE≤4.45e−16、MSIC≤5.69e−14，RSS/(2n)与保存loss恒等误差0。v5.1/v5.2非n400的200个系数/支撑精确相同，n4001/50相同。详[身份审计](taskcard7_completion/cluster_audit.json)与[系数指标审计](taskcard7_completion/estimate_audit.json)。

## 指标与成本

{perf}

{data} 表中包含PMSE、signal MSE、REE、Exact、Mode-F1、Element-F1、q、q/Q、RSS、MSIC和运行秒数；极值个案列出具体task/重复、三版本误差/q/RSS，不删除恶化记录。

{cost}

保存计时来自不同运行/作业，未控制同硬件或随机化测时，仅作描述；相同网格和B的场景仍有计时差，不能将其直接解释为网格因果开销。候选数150→252是n400的结构计算量变化，非严格速度基准。

## 原环境恢复与证据限制

原Windows失败preflight继续保存在旧目录：当时X/y/noise不匹配、B_true匹配，fit_executed=false。回传新目录的成功preflight及250任务记录均匹配原Linux CPython3.9.25/NumPy2.0.2/SciPy1.13.1/OpenBLAS0.3.27和四线程变量1，说明原身份门禁已满足；没有放宽Gate或改历史plan。

{boundary} 源码与seal建立内部一致性，不是外部签名；原worker检查包入口路径，未保存每模块实际import日志。Gate绑定与文件传输时间也不独立证明作业先后，本轮没有SLURM作业日志审计。

已完成目录不再次启动或merge；原单文件提交说明见[目录README](../numerical_simulation/outputs_v51_attribution/README.md)，旧说明原字节已归档。{evidence}
'''
    updates['doc/FINAL_VERSION_DIAGNOSIS.md']=f'''# 任务卡7：最终版本诊断

{scope}

## 一页中文结论

**第一，版本归因：** LEMON恶化发生在v5→v5.1。15外折的v5.1/v5.2系数、支撑及435条OOF预测精确一致；完整OOF RMSE均值0.371667→0.504581，R²0.121430→−0.597008，q80从1/15→5/15。A2在该条件没有改变实际结果。

**第二，模拟归因：** 已完成250/250。n100 PMSE5.196560→6.442026→6.442026，恶化来自BCD版本变更；n400 PMSE1.297588→1.299304→1.189516，Exact1→1→49/50，主要恢复及预测收益来自网格补全。其余所选场景网格效应为零；各50配对的具体分布不能由一个平均数替代。

**第三，MSIC诊断：** 既有六代表折480固定候选中295 RSS下降、185相同、0变差，同规模惩罚不变；9内折中大q训练误差9/9下降而验证7/9上升，MSIC8/9仍较偏大模型。结果支持搜索与有限样本选阶失配，不能把工程搜索正确性、恢复与泛化混为一谈，也不能据此断言EEG无信号。

**第四，算法建议：** 保留v5.2及原MSIC，保留负面结果，不回退已知缺陷、不让HCP/NKI换算法。本任务不改算法；后续任务卡8已经独立研究Warm/选阶，但均未通过相应推广门槛，不能由本次归因补全反向批准新规则。

## 八项回答

1. LEMON变化在BCD修复，v5.1/v5.2在既有15折实际等价。
2. 修复使已有大模型拟合更充分，在该实证中暴露过拟合风险；不是所有DGP必然退化。
3. A2对LEMON无效果；n400的48次新增Exact及平均PMSE下降主要来自网格。
4. 原MSIC的有限样本复杂度控制值得研究，现有证据不保证任何替代规则成功。
5. v5.2继续作为工程基线，不能宣传为全面预测最优。
6. 独立选阶研究已经在任务卡8完成一轮；当前不发布其未过门槛的联合规则。
7. 新候选如继续研究，需固定正确BCD、预声明新开发与确认数据/预算/指标，不能再用已看LEMON外测调参。
8. 论文同时报告工程修复、n400恢复收益和n100/LEMON预测代价；旧700的138→182是700标签、600唯一输入的描述，当前250也不能按跨场景IID推断。

## 完整记录

{perf}

{boundary}

[LEMON三版本](LEMON_THREE_VERSION_COMPARISON.md)、[候选机制](BCD_MSIC_INTERACTION_REPORT.md)、[训练内诊断](LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md)、[完整模拟拆分](VERSION_EFFECT_DECOMPOSITION_FINAL.md)、[接收验收](V51_FULL_ATTRIBUTION_REPORT.md)、[下一步](NEXT_ALGORITHM_RESEARCH_DECISION.md)、[任务卡8独立结果](TENSOR_ABESS_NEXT_VERSION_DECISION.md)。{evidence}

A/B/C/D均完成既定工作；HCP/NKI结果尚未提供，未推断其表现、发送消息或重建ZIP。
'''
    updates['doc/NEXT_ALGORITHM_RESEARCH_DECISION.md']=f'''# 下一步算法研究决策

{scope}

**保留v5.2工程基线及原MSIC；当前没有新的正式算法发布决定。** 250配对已将n100的预测退化定位至BCD变更，将n400主要恢复收益定位至网格补全；LEMON实际没有A2结果变化。工程最优访问状态修复仍应保留，不能因旧版偶然较低外测误差恢复缺陷。

任务卡8已另行完成108开发+108独立复核的固定Cold候选库研究：alpha2/scale0确认平均PMSE下降约7.8%，但最差场景及强信号欠选门槛未过；原样Warm正确性也未过。因此并无已确认可替代当前默认的规则，详[任务卡8决策](TENSOR_ABESS_NEXT_VERSION_DECISION.md)。本次补全不改变其原协议、结果或资格。

若继续新研究，应把修复BCD和A2固定，开发/确认采用未见种子和独立实证，预声明指标、候选和停止预算；不要在旧LEMON外测上挑alpha或q上限。q=0是未来预测框架的研究候选，不混入当前正式Tensor-ABESS。原[研究提案](MODEL_SELECTION_RESEARCH_PROPOSAL.md)按此边界保留。

论文应分别描述工程可靠性、统计恢复和预测泛化：历史700标签Exact138→182与PMSE2.2368→2.3813并列，明确只有600唯一输入；增加本轮五场景各50的两段配对表和LEMON负面结果，不宣称全面优越或HCP/NKI有效。

本轮无需用户决定新参数、重新运行250或让队友换算法；HCP/NKI结果仍待提供。{evidence}
'''
    updates['doc/TEAM_RESEARCH_STATUS_UPDATE.md']=f'''# 团队研究状态说明（待转发，未发送）

2026-10-17（北京时间）。v5.2继续冻结，HCP/NKI按原确认设计运行，不需更换算法或重启正确实验。

LEMON三版本已确认恶化发生在BCD修复后：OOF RMSE0.371667→0.504581，v5.1/v5.2系数、支撑及预测精确一致。原定250模拟也已验收完成：n100退化发生在BCD变更，n400恢复1→1→49/50及预测收益主要来自网格补全。

搜索正确、恢复正确和未见数据预测好是不同目标；独立任务卡8的新规则尚未过推广门槛，当前默认不变。请完整保留q、支撑、OOF、训练/测试误差和源码/输入/划分指纹，不为匹配LEMON调整参数或筛选数据。

HCP/NKI结果尚未提供，本轮不推断其表现，也没有发送本说明或重新生成ZIP。{evidence}
'''
    phase_header='2026-10-17补记：250/250模拟回传已通过验收，任务卡7本地A/C证据保持原样；新增两段归因见[完整模拟报告](VERSION_EFFECT_DECOMPOSITION_FINAL.md)及[本轮复现](taskcard7_completion/REPRODUCIBILITY.md)。'
    for name in ('LEMON_THREE_VERSION_COMPARISON.md','LEMON_BCD_OVERFITTING_DIAGNOSIS.md',
                 'BCD_MSIC_INTERACTION_REPORT.md','LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md'):
        old=(out/'pre_update_snapshots/doc'/name).read_text(encoding='utf-8').splitlines()
        old[2]=phase_header
        updates['doc/'+name]='\n'.join(old)
    name='doc/MODEL_SELECTION_RESEARCH_PROPOSAL.md'
    old=(out/'pre_update_snapshots'/name).read_text(encoding='utf-8').splitlines()
    old[2]='2026-10-17补记：250归因前置已经完成；本提案保留为历史研究设计，任务卡8的独立执行及未过推广门槛结果另列。'
    text='\n'.join(old)
    text=text.replace('先在原Linux环境完成指定250严格配对任务，分离BCD与网格的数值作用；这不阻止预备新协议，但不能把未完成250称为完整既有证据。',
        '指定250严格配对任务已完成并通过回传验收，BCD与网格数值作用已分离，详见[模拟归因](VERSION_EFFECT_DECOMPOSITION_FINAL.md)。')
    updates[name]=text+'\n\n任务卡8已完成一轮独立Warm/选阶研究，但候选未通过相应门槛，详[实际研究决策](TENSOR_ABESS_NEXT_VERSION_DECISION.md)；不能把本历史提案当成未执行的同一计划继续调参。\n'
    updates['numerical_simulation/outputs_v51_attribution/README.md']='''# 完整250任务归因结果

2026-10-17接收验收：cluster_run包含250 JSON/250 NPZ、5 Gate、成功preflight、冻结plan、原生配对CSV与PASSED250 integrity；全部身份及750三版本模型的可复算指标通过。本机未拟合或再次merge。

[最终报告](../../doc/FINAL_VERSION_DIAGNOSIS.md)、[两段配对拆分](../../doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md)、[验收/边界](../../doc/V51_FULL_ATTRIBUTION_REPORT.md)、[复现与新索引](../../doc/taskcard7_completion/REPRODUCIBILITY.md)。原始结果及旧Windows失败记录均保留；指标与原始数组可用范围见验收报告。

原run_v51_attribution.sh是这批任务的单文件集群入口，已有完整结果无需再次提交或merge。独立副本复现时仍必须保留原Linux环境、完整冻结材料和严格Gate；其历史部署说明原字节位于本轮pre_update_snapshots，不修改既有plan或复制覆盖结果。
'''
    status_path='numerical_simulation/outputs_v51_attribution/STATUS.json'
    status=read(repo/status_path)
    status.update(date='2026-10-17',status='COMPLETE_250_CLUSTER_RESULTS_ACCEPTED',
        completed_v51_task_count=250,completed_pilot_gate_count=5,full_attribution_complete=True,
        new_preflight_executed=False,remote_execution_performed=False,delivered_execution_plan_present=True,
        cluster_execution_reported_in_delivered_records=True,acceptance_only_no_new_fits=True,
        action_flags_scope='Local acceptance actions only; prior cluster execution is represented by delivered sealed records',
        required_next_conditions=[],completion_evidence='doc/taskcard7_completion/DELIVERY_INDEX.json')
    status.pop('new_execution_plan_copied',None)
    updates[status_path]=json.dumps(status,ensure_ascii=False,indent=2)
    # Validate every original target before the first write.
    for name in updates:
        assert (repo/name).read_bytes()==(out/'pre_update_snapshots'/name).read_bytes(),name
    for name,text in updates.items():
        (repo/name).write_text(text.strip()+'\n',encoding='utf-8')
    print('Updated',len(updates),'snapshotted Task7 report/status targets')


if __name__=='__main__':
    main()
