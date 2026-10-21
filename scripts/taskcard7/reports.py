"""Build task card 7 reports from complete local evidence; never fit or tune."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def table(headers, data):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in data])


def document(repo, name, text):
    (repo / 'doc' / name).write_text(text.strip() + '\n', encoding='utf-8')


def main():
    repo = Path(__file__).resolve().parents[2]
    base = repo / 'numerical_simulation/outputs_version_attribution'
    comparison = read(base / 'lemon/comparison.json')
    if not (comparison['complete'] and comparison['v51_completed_fits'] == 15
            and comparison['v51_v52_all_B_exact_equal'] and comparison['v51_v52_all_prediction_exact_equal']):
        raise ValueError('Complete exact Phase A evidence is required')
    candidate_compare = read(base / 'mechanism/candidate_comparisons.json')
    if len(candidate_compare) != 6 or not all(r['all_equal'] for r in candidate_compare):
        raise ValueError('Six complete candidate replays are required')
    selection = read(base / 'mechanism/selection_summary.json')
    decomposition = rows(base / 'mechanism/candidate_msic_decomposition.csv')
    inner = rows(base / 'train_only/results/inner_oof_summary.csv')
    validation = read(base / 'train_only/verification.json')
    if len(inner) != 15 or not validation['all_45_fit_artifacts_verified']:
        raise ValueError('Complete training-only evidence is required')
    versions = ['v5-original', 'v5.1-bcd-fix', 'v5.2-search-final']
    lookup = {(r['version'], r['repeat'], r['outer_fold']): r for r in comparison['folds']}
    keys = [(r, f) for r in range(3) for f in range(1, 6)]
    old = [lookup[versions[0], *key] for key in keys]
    new = [lookup[versions[1], *key] for key in keys]
    summary_table = table(['版本', '完整OOF RMSE均值', '完整OOF R²均值', 'q80折数', '平均训练RSS'],
        [[r['version'], f"{r['rmse_mean']:.6f}", f"{r['r2_mean']:.6f}", f"{r['q80_folds']}/15", f"{r['train_rss_mean']:.6f}"]
         for r in comparison['summary']])
    repeat_table = table(['重复（0起）', 'v5 OOF RMSE', 'v5.1 OOF RMSE', 'v5.2 OOF RMSE'],
        [[repeat] + [f"{next(r['rmse'] for r in comparison['metrics_by_repeat'] if r['version']==v and r['repeat']==repeat):.6f}"
                      for v in versions] for repeat in range(3)])
    fold_table = table(['重复/折（重复0起）', 'q：v5→v5.1=v5.2', '训练RSS：旧→新', '测试RMSE：旧→新', 'v5.1精修前→后q'],
        [[f"{a['repeat']}/{a['outer_fold']}", f"{a['q']}→{b['q']}",
          f"{a['train_rss']:.6f}→{b['train_rss']:.6f}", f"{a['test_rmse']:.6f}→{b['test_rmse']:.6f}",
          f"{b['q_before_refinement']}→{b['q_after_refinement']}"] for a,b in zip(old,new)])
    inner_table = table(['模型', '重复0内OOF RMSE', '重复1内OOF RMSE', '重复2内OOF RMSE'],
        [[model] + [f"{float(next(r['inner_oof_rmse'] for r in inner if r['model']==model and int(r['repeat'])==repeat)):.6f}"
                    for repeat in range(3)] for model in ['Covariates Only','Adaptive MSIC','Fixed small','Fixed medium','Fixed large']])
    sel = {(r['version'], r['repeat'], r['outer_fold']):r for r in selection}
    representatives = read(base / 'mechanism/protocol.json')['representatives']
    mechanism_table = table(['重复/折', 'q：旧→新', 'q80 RSS：旧→新', 'q80 ΔMSIC', 'q80排名：旧→新', '容忍带：旧→新'],
        [[f'{r}/{f}', f"{sel[versions[0],r,f]['final_q']}→{sel[versions[1],r,f]['final_q']}",
          f"{sel[versions[0],r,f]['q80_candidates'][0]['rss']:.6f}→{sel[versions[1],r,f]['q80_candidates'][0]['rss']:.6f}",
          f"{sel[versions[1],r,f]['q80_candidates'][0]['msic']-sel[versions[0],r,f]['q80_candidates'][0]['msic']:.6f}",
          f"{sel[versions[0],r,f]['q80_candidates'][0]['rank']}→{sel[versions[1],r,f]['q80_candidates'][0]['rank']}",
          f"{sel[versions[0],r,f]['one_se_tolerance']:.6f}→{sel[versions[1],r,f]['one_se_tolerance']:.6f}"]
         for r,f in representatives])
    fixed = [r for r in decomposition if r['old_version']=='v5-original' and r['new_version']=='v5.1-bcd-fix'
             and r['candidate_type']=='fixed_grid']
    if len(fixed) != 480:
        raise ValueError('Requires 480 paired fixed-size candidates')
    improved = sum(float(r['rss_new']) < float(r['rss_old'])-1e-12 for r in fixed)
    worsened = sum(float(r['rss_new']) > float(r['rss_old'])+1e-12 for r in fixed)
    if (improved,worsened)!=(295,0):
        raise ValueError('Candidate mechanism count differs from verified evidence')
    stamp = '2026-10-16（北京时间）。本轮按用户选择完成本地诊断；250任务留待原Linux集群执行。'
    links = '[复现与身份记录](taskcard7/REPRODUCIBILITY.md)；[完整配对表](../numerical_simulation/outputs_version_attribution/lemon/three_version_folds.csv)；[实验索引](taskcard7/DELIVERY_INDEX.json)。'
    document(repo, 'LEMON_THREE_VERSION_COMPARISON.md', f'''# LEMON三版本严格配对比较

{stamp}

**LEMON恶化发生在v5→v5.1；v5.1与v5.2的15折完整B、支撑及435条外OOF预测逐值完全相同。** 本轮仅新增15个v5.1外折拟合，v5/v5.2及其他对照复用既有完整结果；C2另外12个训练重放只为候选机制验收，不替换外OOF。

{summary_table}

RMSE与R²先按每次重复全部145人的OOF计算，再对3次重复取均值；不是15折RMSE的简单平均。仅协变量参考结果为RMSE0.317485、R²0.367900，来自既有同划分完整OOF。

{repeat_table}

## 严格身份与等价证据

三版各32份源码精确匹配其VERSION SHA256，worker独立进程绑定目标历史import路径；未覆盖生产src。实际既有.venv环境为Python3.12.8/NumPy1.26.4/SciPy1.16.1/Windows11，四线程变量均1。145名ID及顺序、X=(145,61,16,5)、log(TMT_5)、8列协变量、原种子、435条外折分配、全部15折折内预处理数组与旧/新保存值精确一致。配置、依赖、环境及输出指纹绑定在[冻结计划](../numerical_simulation/outputs_version_attribution/lemon/plan.json)。

15折caps均(4,4,5)、容量Q=floor(.7×116)=81；有序fixed规模网格均80个，加adaptive候选共81。v5.1保存全部1215候选B、支撑、RSS/MSIC、初始化、adaptive终态、one-SE容忍带、搜索历史和精修前后q。两版其余31文件SHA及移除唯一网格函数后的BCD AST一致；不仅核对集合，也核对顺序和实际最终结果。

进一步按训练q预定的6个代表折独立重放v5/v5.2：两版历史最终B/支撑/MSIC精确复现，v5.1/v5.2的486候选B、支撑、RSS/MSIC、初始化、搜索路径及并列选择逐值一致。其余9折的历史候选全轨迹未保存，候选等价性以相同执行AST/有序网格/输入身份推论支撑，并有实际最终输出精确一致佐证；不冒充这9折已重放。详见[路径核验](../numerical_simulation/outputs_version_attribution/mechanism/candidate_comparisons.json)。

## 全部逐折配对

{fold_table}

完整CSV另含三模大小、逐模支撑位置、最终MSIC、训练/测试RMSE、测试R²、q/Q、精修前后q及候选数。旧版未保存的精修前q标为空，代表折在C2新重放记录中补充，不回写历史。

![全部逐折配对](taskcard7/figures/lemon_three_version_paired.png)

图中各折内连线；横轴显示1起重复编号，表格和文件重复编号0起；15折相关，不作独立样本显著性推断。PDF/SVG/TIFF同目录。

结论：q80由旧(2,5)一个折增至(0,1)/(1,1)/(1,2)/(2,3)/(2,5)五折；新增四折全在v5→v5.1发生。此LEMON条件下A2既未增加候选，也未改变实际选择和预测。{links}
''')
    document(repo, 'LEMON_BCD_OVERFITTING_DIAGNOSIS.md', f'''# LEMON的BCD修复与过拟合诊断

{stamp}

版本归因已经确定：v5.1=v5.2的15折B/支撑/预测完全一致，恶化在v5→v5.1。测试结果较差不能成为恢复已知fixed BCD缺陷的理由；搜索正确性、支撑恢复及预测泛化是三件不同的事。

## 观察到的外层变化

{summary_table}

最终所选模型训练RSS在15折中8折下降、4折相同、3折升高；q在7折增大、5折相同、3折减小；测试RMSE在9折恶化、4折相同、2折改善。7折同时出现训练RSS下降而测试RMSE上升。最终模型并非每折训练RSS都下降，因为模型选择也可能转向较小q；不能将fixed同规模搜索的单调改善混同于最终自适应所选模型的单调改善。

典型repeat0/fold1：q3→80，训练RMSE0.271718→0.092233，测试RMSE0.303006→0.656305。旧q80折repeat2/fold5测试0.670771→0.619982改善，但另四个q80新增折使整体完整OOF明显恶化；三次重复RMSE均升高。

## q80增加的直接机制

原caps、80个有序网格、GLasso初始化及adaptive终态未改变；fixed BCD修复使已有q80候选得到更低RSS，原MSIC排序/容忍带内可选集合随之改变，最终四折从小模型切换到q80。六个代表折完整候选重放显示480个同规模fixed对中295 RSS下降、185相同、0变差；同形状惩罚变化仅浮点舍入。该证据来自训练候选，不根据外测挑参数。

{mechanism_table}

## 泛化与信号的边界

预声明三个外训练集的训练内验证另有9个内折：规模q1→12→45时训练RMSE9/9下降，验证RMSE7/9严格上升，q45的MSIC在8/9比q1更小；自适应MSIC的三组完整内OOF均劣于协变量基线。联合证据支持“修复后的充分搜索暴露原MSIC在本有限样本条件下的复杂度控制不足”，而非证明公式在所有DGP必然失效。

现有设计未发现稳定EEG增量预测收益，但不能据此断言EEG完全无信号：特征表达、响应、样本量与支撑结构可能共同限制泛化，需独立确认数据。没有改alpha/caps、筛受试者、选择外测最优规则或发布v6。{links}
''')
    document(repo, 'BCD_MSIC_INTERACTION_REPORT.md', f'''# Fixed BCD与原MSIC的交互机制

{stamp}

已观察到：fixed BCD修复降低了相同规模候选的训练RSS，原不变MSIC因此更偏好部分接近容量的大模型；该机制由真实候选重放及独立训练内验证共同支持。

## 预声明代表与真实路径

在重放前冻结代表规则：历史v5或v5.2训练所选q80折的并集，另加字典序第一个两版q均小于80的折；未按测试RMSE选代表。实际6折为(0,1),(0,2),(1,1),(1,2),(2,3),(2,5)。仅新增v5/v5.2各6训练拟合，v5.1复用Phase A真实轨迹。

各版每折初始化三模大小、完整支撑、adaptive终态、80个有序fixed候选、RSS/MSIC/完整B、one-SE带与最终选择都已保存；历史最终B、支撑、score均精确重现。v5.1/v5.2的486候选连同所有固定搜索状态完全一致，不存在候选顺序、初始化、缓存或浮点并列差异的证据。

{mechanism_table}

## 数学分解

正式全局精修MSIC为 n log(RSS/n) + 1.5 q log(log n) + log(log n) Σ_m log choose(p_m,s_m)，n=116，原gamma=1；mode级自适应选阶与全局精修参数分开，未混用或修改。对于同一三模规模而不同支撑位置：

`ΔMSIC = n log(RSS_new/RSS_old) + ΔPenalty`。

同形状、相同n/p/alpha/gamma时惩罚严格相同；支撑位置不进入该惩罚。480个fixed对中295 RSS降低、185相同，无变差；由score相减反算的惩罚差不超过8.53e−14。例(0,1) q80 RSS1.745571→0.986812，MSIC−267.501784→−333.663313，下降66.161529；原q80位列4且在带外，新q80位列1且是唯一带内候选。

one-SE风格带是代码确定性容忍量 `sqrt(q_best log n)`，不是由独立验证估计的标准误。它可改变可选集合，但不能单独解释全部变化：(2,3)旧版q80已是最低MSIC却仍选q1，修复后同一容忍量下改选80；控制(0,2)修复后q80进入带内，最终仍选q1。不能把所有大模型归因于容忍带扩大。

## 验收与解释边界

三版本六折合计1458候选独立由保存B和训练预处理重算RSS/MSIC并核验满秩、q合法；候选与源码全hash保留。972行两段配对分解均为相同三模形状，逐对单列惩罚差，差异仅浮点舍入，见[CSV](../numerical_simulation/outputs_version_attribution/mechanism/candidate_msic_decomposition.csv)、[候选选择摘要](../numerical_simulation/outputs_version_attribution/mechanism/selection_summary.json)和[验收](../numerical_simulation/outputs_version_attribution/mechanism/verification.json)。

公式解释的是训练拟合改善如何转成更低MSIC，并不证明总体风险最优，也不证明普适的统计失效；训练内验证提供本样本的独立于外测选参的补充证据。没有为取得较小外测误差重评分选alpha。{links}
''')
    document(repo, 'LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md', f'''# LEMON仅训练数据的模型复杂度诊断

{stamp}

## 拟合前冻结的设计

固定选择三个重复的第一外折，未按外测表现挑选；每个外训练集仅116人，再按原inner_seed+repeat×100+fold做三折。每个内训练77/78人重新拟合协变量残差化、X标准化、响应中心化与候选支撑，恢复原log(TMT_5)响应评价。外测试者的X/y/C不传入任何内拟合、预测或误差计算。

研究版为独立进程加载冻结v5.1-bcd-fix；完整32源码及所有实际import、输入、配置、辅助代码、环境和种子写入[protocol](../numerical_simulation/outputs_version_attribution/train_only/protocol.json)，prepare先落盘后run严格校验，拒绝覆盖。原算法参数不变。

仅协变量、原自适应MSIC和固定q=1(1,1,1)、12(2,2,3)、45(3,3,5)同时比较。q80超过内训练容量floor(.7n)=53/54，已在协议中排除且没有拟合；45是预先定义的合法“大”诊断规模，不用于修改外OOF或正式caps。

## 完整内OOF结果

{inner_table}

每个数值由同一外训练集116人的完整内OOF SSE计算，未平均内折RMSE；三个外训练集高度重叠，因此这些不是三组独立确认样本。自适应各内折q为[2,6,32]、[48,48,48]、[8,2,48]。

固定规模q1→12→45时训练RMSE9/9严格下降、验证RMSE7/9严格上升；q45在全部9折验证误差均高于q1，MSIC却在8/9更低。三组自适应结果均劣于协变量基线。该现象支持当前有限样本下训练准则与泛化目标偏离，不足以认定所有EEG特征无稳定信息。

![全部内折复杂度诊断](taskcard7/figures/train_only_complexity.png)

## 完整性与边界

45模型、1740验证预测、5220含训练的总预测行全部保存，设计均满秩且q合法；独立按B/预处理/ID重构最大预测误差1.60e−14，最大tensor设计条件数212.765。没有使用完整116人先选支撑再交叉验证，没有更新正式外OOF，没有按内验证结果发布新模型。逐内折数据及[核验](../numerical_simulation/outputs_version_attribution/train_only/verification.json)均独立留存。{links}
''')
    document(repo, 'MODEL_SELECTION_RESEARCH_PROPOSAL.md', f'''# 下一轮模型选择研究提案（本轮不执行）

{stamp}

研究问题：在保持修复BCD及冻结A2搜索的前提下，原MSIC能否对有限样本下的充分位置/规模搜索进行足够的复杂度控制；训练内部预测准则与更强统计惩罚是否改善泛化，同时维持支撑恢复和计算可行性。当前LEMON证据支持启动独立研究，但不能据旧15折测试误差确定新alpha或caps。

## 先完成归因，再冻结新研究

先在原Linux环境完成指定250严格配对任务，分离BCD与网格的数值作用；这不阻止预备新协议，但不能把未完成250称为完整既有证据。新研究将修复后的BCD和A2网格固定，只允许模型选择机制作为干预，所有候选/预算/选参/停止规则在新数据生成前声明。

研究臂可包括：原MSIC基线、理论或训练内准则驱动的预声明惩罚族、完全内嵌交叉验证或样本拆分的复杂度选择。有限候选数量及参数范围必须在独立开发模拟上确定并锁定，随后独立新种子确认；当前任务没有计算或推荐任何LEMON外测最优alpha。每臂沿用同一拟合库比较时须报告“条件于固定候选库”，不能把未重新搜索的重评分等同于新算法完整训练。

## 验证方案

冻结匹配的n、SNR、相关、维度与矩形支撑设计，保留n100/n400及弱信号重点，还加入无增量张量信号情景；开发与确认使用不重叠的新seed/输入指纹，完整记录去重关系，不能将重复场景标签当独立样本。基于预定主指标PMSE/原响应RMSE、Exact、Mode/Element-F1、q/Q与成本统一判断，以完整配对结果报告效应和不确定性，并控制多重比较/选择乐观性。

每次选择都在外训练内部重拟合预处理和支撑；候选库、计算预算、并列规则、失败处置及确认门槛在任何确认结果出现前冻结。具体重复数须通过独立开发阶段的方差/功效及计算预算确定，再冻结，而非看到确认趋势后增删样本。确认数据仅一次验收；失败完整披露，不能重新调规则继续称独立确认。

## 独立实证与q=0

已看过的LEMON外测只作发现/机制资料，不能继续当新算法的独立确认集。优先使用尚未查看结果且未参与选参的独立队列、站点或预留受试者；HCP/NKI仅在标签、特征、划分与适用性匹配且其结果未被用于选择规则时，才可预声明为外部确认，不能事后假设它们会改善。

“不加入EEG”的q=0可作为未来实证预测框架的预声明候选，与协变量基线完整比较；不直接加入本轮正式Tensor-ABESS、改变其统计定义或替换冻结v5.2。需另立协议、测试与发布评审。

建议启动该研究的设计准备，当前不发布v6、不回退v5、不修改工程基线。证据见[机制报告](BCD_MSIC_INTERACTION_REPORT.md)与[训练内诊断](LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md)。
''')
    document(repo, 'TEAM_RESEARCH_STATUS_UPDATE.md', f'''# 团队研究状态简报

2026-10-16（北京时间）。本文仅准备可转发说明，未发送消息、重新打包算法或要求重启实验。

当前正式v5.2-search-final继续冻结，源码、原MSIC和caps不变；HCP/NKI继续按已确认设计运行，不需要为了匹配LEMON结果调整参数、筛选数据或重启正确实验。

LEMON发现明显过拟合：完整OOF RMSE由v5的0.371667升至v5.1/v5.2的0.504581。新增15折归因已确认v5.1/v5.2支撑、系数、预测精确一致；恶化发生在BCD修复后，训练候选机制及训练内验证提示有限样本模型复杂度控制值得独立研究。250模拟分解按用户安排待原集群执行，当前未完成。

请完整保留各折q/三模支撑、OOF预测、训练与测试误差、输入与划分身份、配置/种子及完整版本源码指纹，失败记录也保留。当前尚未获取HCP/NKI完整结果，不能推断其表现；后续按预声明标准评价，完整保留负面结果。
''')
    document(repo, 'FINAL_VERSION_DIAGNOSIS.md', f'''# 任务卡7：最终版本诊断（本地阶段）

{stamp}

## 一页中文结论

**第一，版本归因：** LEMON恶化明确发生在v5→v5.1。15个新增v5.1外折与v5.2的B、支撑和435条OOF预测精确一致；完整OOF RMSE均值0.371667→0.504581、R²0.121430→−0.597008、q80从1/15→5/15。A2在这些样本与caps下既未增加候选，也未改变实际结果。

**第二，模拟归因：** 按用户选择，250任务留待原Linux集群，当前0/250，不能给出完整两段数值归因。结构上n100的旧新有序网格相同，A2没有扩充候选；n400补全使真规模从fixed网格外进入网格内。n100恶化及n400改善的完整统计拆分仍须严格v5.1模拟，不能把结构解释当已完成250结果。

**第三，MSIC诊断：** 6个训练q预选代表折的480固定候选对中295 RSS下降、185相同、0变差；同规模惩罚不变，q80的MSIC因此下降并引发4个新增q80选择。仅训练内的9折另显示q增大时训练误差9/9下降、验证误差7/9严格上升，大模型MSIC8/9却低于小模型，支持有限样本复杂度失配；不证明所有DGP失效或EEG必然无信号。

**第四，算法建议：** 保留正确性修复后的冻结v5.2作为工程基线，保留全部负面结果，启动单独的新模型选择研究设计。先完成250归因，再以新预声明模拟及独立未见数据验证；不据旧15折外测选alpha、不改正式caps、不发布v6，不干扰HCP/NKI原设计。

## 三种评价不能混同

| 维度 | 当前证据 | 可作判断 |
| --- | --- | --- |
| 工程正确性 | fixed BCD保留已访问最佳OLS状态；历史正确性测试及本次候选/OLS核验 | 保留修复，不能因旧版测试更好恢复已知缺陷 |
| 统计恢复性 | 已有700组合Exact138→182；n400结构覆盖改善 | 有恢复收益，BCD/网格完整两段收益仍待250 |
| 预测泛化性 | 已有700组合平均PMSE2.2368→2.3813；LEMON更差 | 无全面预测改善证据；应报告收益与代价 |

## 八项决策回答

1. LEMON变差在v5→v5.1，v5.1→v5.2为实际零变化。
2. BCD修复使已有大模型训练搜索更充分；在本LEMON中增强了选中大模型及过拟合的风险，但不能断言所有条件均如此。
3. A2对LEMON结果无作用；对n400扩充了真规模的fixed候选覆盖，完整模拟恢复收益尚未分离BCD贡献。
4. 原MSIC确有值得研究的有限样本复杂度控制问题，已有训练候选与内验证证据，尚无独立新数据确认新准则更好。
5. 继续保留v5.2作为冻结工程基线，不能宣传为全面预测优越版本。
6. 建议单独启动模型选择研究设计；本任务没有实现或发布新准则。
7. 固定修复BCD/A2，开发与确认用新seed/输入，预声明规则、预算与主指标，独立未见实证确认；详见[研究提案](MODEL_SELECTION_RESEARCH_PROPOSAL.md)。
8. 论文应同时报告工程修复、恢复改善与PMSE/LEMON负面结果，明确选择准则的有限样本局限及未完成归因；不把支撑恢复当成预测泛化，不由单一实证断言无生物信号。

## 证据与完整交付

{summary_table}

- [三版本15折配对、身份与图](LEMON_THREE_VERSION_COMPARISON.md)
- [BCD过拟合诊断](LEMON_BCD_OVERFITTING_DIAGNOSIS.md)
- [候选RSS/MSIC分解](BCD_MSIC_INTERACTION_REPORT.md)
- [训练内重新搜索诊断](LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md)
- [250任务状态与恢复](V51_FULL_ATTRIBUTION_REPORT.md)
- [正式工程基线及下一步](NEXT_ALGORITHM_RESEARCH_DECISION.md)
- [团队可转发简报](TEAM_RESEARCH_STATUS_UPDATE.md)
- {links}

本地A/C/D已完成，B未运行；没有修改生产/历史源码、旧计划/结果、MSIC或caps，没有联系队友或重建ZIP。HCP/NKI结果尚未提供，不作推断。
''')
    document(repo, 'NEXT_ALGORITHM_RESEARCH_DECISION.md', f'''# 下一步算法研究决策

{stamp}

**决定建议：冻结v5.2工程基线，开展独立统计选阶研究；不回退v5、不直接调正式MSIC/caps、不发布v6。** 本轮只给研究建议，未启动下一轮实验或更改架构。

理由是工程正确性与统计泛化已经区分：充分fixed搜索确实降低已有候选RSS，不能因旧缺陷偶然产生较低外测误差而保留缺陷；但当前MSIC在LEMON中会将这种拟合收益转成接近容量的大模型选择，训练内验证也看到泛化损失。现有EEG增量表现弱，不能把复杂度控制视为全部原因，也不能保证换准则即可改善。

下一步先按[250恢复说明](V51_FULL_ATTRIBUTION_REPORT.md)完成模拟两段严格拆分，同时准备[新研究协议](MODEL_SELECTION_RESEARCH_PROPOSAL.md)。确认数据必须独立于已查看LEMON，预声明主指标和选择规则后再生成/查看结果；q=0仅列为未来框架研究，不写入当前正式Tensor-ABESS。

论文表述建议：报告修复使固定规模搜索可靠保留最佳访问状态、完整700组合Exact138/700→182/700；同时报告平均PMSE2.2368→2.3813及LEMON完整OOF RMSE0.371667→0.504581的负面结果，明确恢复和预测目标的差异。可解释为当前搜索/选阶/有限样本数据结构存在相互作用，而不能单独称A2导致LEMON过拟合，不能声称v5.2全面更优或HCP/NKI已确认有效。

当前无需决定新alpha或让队友换算法；唯一尚待执行事项是用户已安排稍后的原集群250任务，以及后续独立研究的正式协议与资源批准。
''')
    reproducibility = f'''# 任务卡7完整复现与产物记录

{stamp}

## 运行环境与输入

本轮本地使用既有`.venv/Scripts/python.exe`，不安装/升级依赖；Python3.12.8、NumPy1.26.4、SciPy1.16.1及Windows平台与LEMON历史配置一致。先设置OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS均为1。正式源码和三冻结版本、原模拟及两个LEMON完整结果只读；初始/最终Git状态与2880个文件的大小/SHA快照在本目录。

输入指纹、完整源码及实际import、配置、435外折分配、训练/测试ID顺序、预处理逐数组指纹分别冻结在lemon/plan.json、train_only/protocol.json、mechanism/protocol.json；仅使用既有145名受试者数据，无下载、重生成或筛选。

## 执行入口

仓库根目录PowerShell：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONPATH='.'
.venv/Scripts/python.exe -B -m scripts.taskcard7.lemon --output-dir numerical_simulation/outputs_version_attribution/lemon --audit-only
# 正式A命令（现有目录已完整；不得重跑覆盖）：
.venv/Scripts/python.exe -B -m scripts.taskcard7.lemon --output-dir numerical_simulation/outputs_version_attribution/lemon
.venv/Scripts/python.exe -B scripts/taskcard7/train_only.py prepare
.venv/Scripts/python.exe -B scripts/taskcard7/train_only.py run
```

A/C3/C2均拒绝覆盖已有正式实验；以上记录用于核查本轮命令，复现实验须在不包含本轮新输出的独立副本中执行，保留原实验字节；不要删除现有产物后重跑。C2实际命令及协议、受保护源/环境校验见[脚本](../../scripts/taskcard7/mechanism.py)，只训练代表折不生成新外OOF；各worker独立进程，v5.1真实轨迹直接复用A。

```powershell
.venv/Scripts/python.exe -B -m scripts.taskcard7.mechanism prepare
.venv/Scripts/python.exe -B -m scripts.taskcard7.mechanism run
```

```powershell
.venv/Scripts/python.exe -B -m unittest scripts.taskcard7.test_lemon scripts.taskcard7.test_train_only scripts.taskcard7.test_mechanism -v
.venv/Scripts/python.exe -B scripts/taskcard7/figures.py --qa-scripts 'C:/Users/xmx/.codex/skills/nature-figure/scripts'
.venv/Scripts/python.exe -B scripts/taskcard7/reports.py
```

绘图使用Python已保存偏好，内部报告宽183mm；原CSV全量输入，PDF/SVG/TIFF/PNG、对齐JSON、PDF字形与碰撞检查及目视记录在figures。科研图没有按结果删折，不对相关外/内折作IID显著性推断。

## 实际拟合与保存口径

| 实验 | 新拟合 | 评价/验收 |
| --- | ---: | --- |
| A：v5.1完整外折 | 15 | 435条v5.1OOF；三版共1305条，45行配对指标；1215候选B/trace |
| C2：v5/v5.2代表训练重放 | 12 | 三版6折1458候选验收；v5.1/v5.2的486候选逐值一致 |
| C3：训练内三组×3折×5模型 | 45 | 1740验证预测，含训练共5220行；45份B/支撑/指标 |
| B：原250模拟标签 | 0 | 严格Gate历史失败，本轮不运行，待原集群 |

C3的45含9个无EEG拟合的协变量基线评价；不能称45个张量搜索或把各内训练者重复计为独立受试者。C2依据训练q选择代表，不用外测选择参数，重放结果不替换原外OOF。所有运行采用原种子/原折内处理，B的Linux环境与本机不同不能冒充。

## 全部交付

上级目录包含任务卡要求10份报告；主入口[FINAL_VERSION_DIAGNOSIS.md](../FINAL_VERSION_DIAGNOSIS.md)先给一页中文总结。[DELIVERY_INDEX.json](DELIVERY_INDEX.json)列出脚本、报告、协议、关键结果和SHA；逐产物SHA分别存于A artifact_validation、C2 completion/verification、C3 completion/verification。完整支撑/B/候选score、Q及训练/测试R²等见各CSV/trace，不将历史缺失轨迹字段补成实测。

Phase B按用户选择推迟，并非放宽Gate或任务完成250；两份对应报告及outputs_v51_attribution/STATUS.json明确记录0/250与原字节恢复步骤。未发送团队消息、生成算法ZIP、修改HCP/NKI设计或启动下一轮研究。
'''
    (repo / 'doc/taskcard7/REPRODUCIBILITY.md').write_text(reproducibility.strip() + '\n', encoding='utf-8')
    print('Generated 8 task-card reports and reproducibility record from complete local evidence')


if __name__ == '__main__':
    main()
