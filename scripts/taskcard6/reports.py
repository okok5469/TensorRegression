"""Render task-card-6 reports from validated, saved results; never fit a model."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'doc'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def write(name, text):
    path = DOC / name
    if path.exists() and name != 'LEMON_V52_EMPIRICAL_REPORT.md':
        raise FileExistsError(path)
    path.write_text('# ' + text.strip() + '\n', encoding='utf-8')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    lemon = read('doc/taskcard6/lemon_comparison/comparison.json')
    sim = read('numerical_simulation/outputs_version_attribution/saved_result_analysis/summary.json')
    cases = read('numerical_simulation/outputs_version_attribution/saved_result_analysis/selected_cases.json')['cases']
    pilot = read('numerical_simulation/outputs_version_attribution/cluster_run/preflight_task_000000.json')
    plan = read('numerical_simulation/outputs_version_attribution/cluster_run/plan.json')
    grid = read('doc/taskcard6/structural_grid_audit.json')
    delivery = read('doc/team_sync/v5.2-search-final/full_package/DELIVERY_VALIDATION.json')
    ols = read('doc/taskcard6/lemon_comparison/ols_audit.json')
    assert not pilot['passed'] and not pilot['fit_executed'] and len(plan['tasks']) == 250
    assert delivery['status'] == 'PASSED' and ols['passed'] and ols['n'] == 30
    versions = {k: {r['model']: r for r in lemon[k]['oof_tables']['model_summary']} for k in ('old','new')}
    old, new = versions['old']['Tensor-ABESS'], versions['new']['Tensor-ABESS']
    inc = next(r for r in lemon['new']['oof_tables']['incremental_metrics']
               if r['model'] == 'Tensor-ABESS' and r['repeat'] == 'all')
    stamp = '2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。'
    block = '''P0的250标签方案已冻结，但**严格配对拟合0/250，未完成**。本机Python3.12.8/NumPy1.26.4/SciPy1.16.1重生正式task0时，仅B_true指纹相同，X_train、y_train、train_noise均不同；保存的原环境为Linux/Python3.9.25/NumPy2.0.2/SciPy1.13.1/OpenBLAS0.3.27。没有用相同seed、近似数值、改dtype或替换数据冒充严格配对，也未安装或升级依赖。

事实证据：[预检原记录](../numerical_simulation/outputs_version_attribution/cluster_run/preflight_task_000000.json)、[冻结250计划](../numerical_simulation/outputs_version_attribution/cluster_run/plan.json)。计划指明task0–49、50–99、250–299、350–399、550–599，各50标签，共250种完整输入；没有新提交700任务。早期根目录plan/preflight是工具定稿前的失败预检留存，实际后续协议使用cluster_run子目录；不覆盖先前证据。

因此不能填造本批v5→v5.1的Δ1和v5.1→v5.2的Δ2数值、胜平负或成本。以下结构检查与已有独立研究可以限定解释，但不能计为这250次已运行。'''
    grid_table = table(['训练n / caps','v5.1网格数','v5.2网格数','顺序相同','真规模在全局fixed网格：旧/新'],
                       [[f"{r['n']} / {r['caps']}",r['v51_count'],r['v52_count'],r['same_ordered_grid'],
                         f"{r['truth_shape_in_v51']} / {r['truth_shape_in_v52']}"] for r in grid['ordered_grid_checks']])
    structural = '''完整32源码比较中v5.1/v5.2仅bcd.py不同，移除_candidate_size_triples后两文件AST完全一致。直接执行两份冻结源码的网格函数，核对**候选集合和顺序**，结果如下：

''' + grid_table + '''

候选表还包含原adaptive结果，故所选候选池数量为网格数+1；表中是整数规模网格数，不是支持位置枚举数。

n=100的50次实际caps均(4,4,4)，真第3模规模5在两版都不可达；此处A2没有新增规模、次序也相同，不能把旧新预测恶化归因于“新增候选”。n=200被选诊断场景均caps(5,5,5)，同样已有完整网格。n=400实际caps(6,6,7)，原网格缺少第2模规模3，A2把真规模纳入全局fixed规模网格；原池还含adaptive结果，因此不能断言所有旧候选池均排除真模型。位置搜索与评分选择仍是独立问题。

LEMON n_train=116时，直接提取并执行原初始化代码中的caps生成AST，在全部6324种可能初始化支撑长度组合下均得(4,4,5)，两版有序网格均为80。这里也没有A2新增规模。结合31份其他core一致与BCD其余AST一致，可排除“网格增加”作为这些条件下版本差异的解释；这是源码等价性推断，**不是已实际完成v5.1的250次拟合或LEMON v5.1重跑**。

详见[结构审计](taskcard6/structural_grid_audit.json)。'''
    recovery = '''在原计算环境恢复P0：同步新增version_attribution模块、三版冻结源码/身份及原两批计划与所需任务JSON，保持正式源码及原结果不变，使用原已验证环境并在计算资源内执行。复制已冻结cluster_run/plan.json，不复制Windows失败preflight到新的集群诊断目录；源文件需保留原字节，不能改计划身份。

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B -m numerical_simulation.version_attribution.run preflight --task-id 0
# 只有四项数据指纹与完整环境均通过，才继续：
python -B -m numerical_simulation.version_attribution.run pilot
# 五场景各首个任务的真实拟合Gate全部通过后，补跑其余245任务：
python -B -m numerical_simulation.version_attribution.run run
python -B -m numerical_simulation.version_attribution.run merge
```

该工具在每个独立进程先绑定v5.1完整源码与实际import，再检查四项输入指纹/原环境；错配即在拟合前失败。结果采用新文件且拒绝覆盖/残缺断点；五场景Gate缺少任一个则拒绝批量运行；merge要求完整250。生产src没有临时切换，v5/v5.2不重新拟合。当前只验证了工具契约与真实失败预检，不能将其称为原Linux集群运行已验证。'''
    write('VERSION_EFFECT_DECOMPOSITION.md', f'三版本效果拆分：结构证据与未完成数值归因\n\n{stamp}\n\n{block}\n\n{structural}\n\n{recovery}')
    write('BCD_FIX_STATISTICAL_IMPACT.md', f'BCD正确性修复的统计影响\n\n{stamp}\n\n{block}\n\nfixed BCD修复具有已有最佳已访问状态保存、接受轨迹非增及回归验证证据；它修正搜索状态管理，不承诺降低样本外风险。不能因旧缺陷偶然产生更小测试误差而恢复旧BCD。\n\n在n=100、所选n=200和LEMON条件下，两版网格及顺序一致，其他core及BCD剩余AST相同，结构上不支持把变化归因A2新增规模；这指向BCD实现变化与原训练选择相互作用的解释。具体Δ1的配对分布仍需完成250诊断，不能将结构推断改写成测量结果。\n\nLEMON新q80从1折增为5折，训练误差降低而OOF误差明显增加，说明优化训练目标与泛化性能不同；其具体实证证据见[LEMON报告](LEMON_V52_EMPIRICAL_REPORT.md)，不是本次模拟Δ1的替代。\n\n完整恢复步骤见[拆分报告](VERSION_EFFECT_DECOMPOSITION.md)。')
    write('A2_GRID_STATISTICAL_IMPACT.md', f'A2网格补全：覆盖收益及适用边界\n\n{stamp}\n\n{structural}\n\n本次250配对尚未完成，不能给新的Δ2统计。既有任务卡4独立A0/A2研究已报告100组新增配对与冷启动验证，主90组Exact15→36、PMSE下降约1.03%，属于其既定队列；它不是当前五场景250次的替代。\n\n原full中n400的Exact1→49/50与真规模由全局fixed网格外进入网格内一致，但v5到v5.2还包含BCD修复，不能把该48次净增加全部当作已实测的A2单独效应。更多规模不必然改善原MSIC最终选择，也不保证位置搜索全局最优；未做新MSIC/caps筛选。')

    model_table = table(['模型','旧RMSE均值','新RMSE均值','旧R²均值','新R²均值'],
                        [[m, f"{versions['old'][m]['rmse_mean']:.6f}",f"{versions['new'][m]['rmse_mean']:.6f}",
                          f"{versions['old'][m]['r2_mean']:.6f}",f"{versions['new'][m]['r2_mean']:.6f}"] for m in versions['new']])
    fold_table = table(['repeat/fold','旧q','新q','新q/Q','新train RMSE','新test RMSE'],
                       [[f"{r['repeat']}/{r['outer_fold']}",int(r['old_q']),int(r['new_q']),f"{r['new_q_over_Q']:.4f}",
                         f"{r['new_train_rmse']:.6f}",f"{r['new_test_rmse']:.6f}"] for r in lemon['paired_folds']])
    repeat_table = table(['repeat','模型','完整145人OOF RMSE','OOF R²'],
                         [[r['repeat'],r['model'],f"{r['rmse']:.6f}",f"{r['r2']:.6f}"] for r in lemon['new']['oof_tables']['metrics_by_repeat']])
    paired = lemon['paired_abess']
    native = '../empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final'
    write('LEMON_V52_EMPIRICAL_REPORT.md', f'''LEMON v5.2完整实证：未获得EEG增量预测收益

{stamp}

**实际完整运行成功，但预测表现更差。** 原入口audit通过后使用相同配置重跑五模型，run_state记录complete=true、75 fits、2175 OOF、1253.02秒；实验ID为405a67add110fcbbe179c5640c6f04c388111842bb76dc3c4977bdd87702d64d。没有smoke或旧模型缓存冒充正式结果。四个对照重新拟合，其1740条OOF预测与旧结果逐条相同；改变集中于ABESS。

145人、张量(145,61,16,5)、log(TMT_5)、原8列年龄/性别协变量、3×5外折、原内层与折内预处理、固定超参数机制均保留。8份输入文件SHA、被试顺序与435条外折分配一致；15份训练/测试ID及顺序一致。完整32 core匹配发布manifest，43份pipeline指纹只改BCD。Python3.12.8/NumPy1.26.4/SciPy1.16.1及四线程=1与旧LEMON环境一致。

下表为各重复完整OOF指标的算术均值；不是15折RMSE的简单平均：

{model_table}

ABESS平均RMSE {old['rmse_mean']:.6f}→{new['rmse_mean']:.6f}（+{(new['rmse_mean']/old['rmse_mean']-1)*100:.2f}%），平均R² {old['r2_mean']:.6f}→{new['r2_mean']:.6f}。新旧ABESS按被试/重复/折严格配对，三重复RMSE差分别为{', '.join(f'{x:+.6f}' for x in paired['rmse_new_minus_old_by_repeat'])}；平均差{paired['mean_repeat_rmse_new_minus_old']:+.6f}，145被试簇保留全部重复的2000次bootstrap描述性95%区间[{paired['subject_cluster_ci_95'][0]:.6f},{paired['subject_cluster_ci_95'][1]:.6f}]，条件于保存OOF，不是外部验证或独立435样本推断。

相对Covariates Only，新ABESS的pooled equal-subject/repeat MSE口径ΔRMSE为{inc['delta_rmse']:.6f}，描述性区间[{inc['delta_rmse_ci_low']:.6f},{inc['delta_rmse_ci_high']:.6f}]；ΔMSE为{inc['delta_mse']:.6f}，incremental R²为{inc['incremental_r2']:.6f}。这里ΔRMSE=RMSE_cov−RMSE_model，负值表示EEG模型更差。该pooled口径与上表“先逐重复开方再平均”略有差别，不能混用。按跨重复平均平方误差，29/145人改善、116/145人恶化；没有删人或改参数。

完整逐重复OOF结果：

{repeat_table}

## 容量、训练测试差与稳定性

各折n_train=116、Q=81。q=80及q/Q≥.9均从旧1/15增至新5/15，另有q64与q60；平均q7.33→37.47。新9折test RMSE变差、2折改善、4折相同。15折平均train RMSE0.264405→0.192871，平均test RMSE0.363746→0.482232（这两项仅诊断训练测试差，不替代完整OOF评价）。

{fold_table}

跨15折、105对支持的平均Jaccard：通道{lemon['old']['support']['pairwise_mode_jaccard_mean'][0]:.4f}→{lemon['new']['support']['pairwise_mode_jaccard_mean'][0]:.4f}，窗口{lemon['old']['support']['pairwise_mode_jaccard_mean'][1]:.4f}→{lemon['new']['support']['pairwise_mode_jaccard_mean'][1]:.4f}，频带{lemon['old']['support']['pairwise_mode_jaccard_mean'][2]:.4f}→{lemon['new']['support']['pairwise_mode_jaccard_mean'][2]:.4f}。较大支持本身也可能增加重叠，不能称为真实脑支持恢复改善。经原tensor metadata确认gamma为第5个频带；gamma被选折数6→11，仅gamma的折数4→2，不能由此提出可靠gamma生物标志物结论。

**独立数值核验：** 从相同已审计LEMON输入、保存的预处理/B重算旧新各15个ABESS模型的设计矩阵与预测。30个设计均满列秩，最大条件数{max(r['condition_number'] for r in ols['models']):.4f}，正规方程相对残差最大{max(r['normal_equation_relative_residual'] for r in ols['models']):.3g}；训练RMSE匹配保存值，测试预测最大绝对差0。没有重训、改B或救补预测；此检查支持“表现差确实来自已保存模型”，不能证明泛化最优或把所有失败归因数值病态。

LEMON的caps经原代码穷举初始化长度均为(4,4,5)，v5.1/v5.2网格集合及顺序相同；A2在这里没有扩大规模候选。源码结构支持将变化定位到BCD实现更新及原选择准则的交互，不应说是A2新增规模致差；未实际运行LEMON v5.1版本，亦不能代替缺失250模拟的数值归因。

![LEMON完整OOF与容量](taskcard6/figures/lemon_prediction_capacity.png)

图a为每重复145人的完整OOF RMSE；三重复共享被试，不作独立重复显著性检验。图b为全部15折的配对q/Q，顺序为repeat0的fold1–5、repeat1的fold1–5、repeat2的fold1–5，虚线为Q。数值表保留五模型，图只突出所问对照。

证据：[原生新结果]({native}/run_state.json)、[运行日志](taskcard6/lemon_full_run.log)、[配对分析](taskcard6/lemon_comparison/comparison.json)、[2175行新旧OOF配对](taskcard6/lemon_comparison/paired_oof.csv)、[15折诊断](taskcard6/lemon_comparison/paired_abess_folds.csv)、[独立OLS/预测核验](taskcard6/lemon_comparison/ols_audit.json)。结论限于当前数据、响应和固定验证设计，不能据此断言EEG没有任何预测信息或其他脑数据必然失败。
''')
    write('LEMON_V5_V52_COMPARISON.md', f'''LEMON v5-original与v5.2严格配对比较

{stamp}

旧结果完整32 core匹配v5-original，不是v5.1；新结果完整32 core匹配D2。数据、顺序、配置、原划分、实际环境一致；所有75模型折及2175条OOF按键唯一匹配。四对照的1740条新旧OOF预测差均为0；ABESS仍全部重新训练，没有复用旧支撑/缓存。

{model_table}

每重复ABESS RMSE变化{paired['rmse_new_minus_old_by_repeat']}，平均+{paired['mean_repeat_rmse_new_minus_old']:.6f}，三个重复均恶化。q80从1→5折，旧异常repeat2/fold5的q80未消失，新版还出现更多接近容量模型；该旧折测试RMSE0.670771→0.619982虽改善，不能代表整体改善。

LEMON训练折caps(4,4,5)两版规模网格相同，不能把全部差异归于A2。完整折级结果、被试簇区间、gamma及数值核验见[完整报告](LEMON_V52_EMPIRICAL_REPORT.md)。所有预测比较为条件于既有OOF的描述性证据；不据此调整MSIC、caps、被试或划分。
''')

    zip_table = table(['文件','实际绝对路径','大小bytes','SHA256'],
                      [[r['zip'],str((DOC/'team_sync/v5.2-search-final/full_package'/r['zip']).resolve()),r['bytes'],r['sha256']]
                       for r in delivery['deliveries']])
    write('TEAM_V52_DELIVERY.md', f'''HCP/NKI完整v5.2算法交付

{stamp}

两份ZIP已经实际生成：先创建一个标准包再逐字节复制，SHA256完全相同；完整32源码不是单文件补丁。均独立解压到新目录，以python -I -B且移除PYTHONPATH执行版本/import检查、fixed/adaptive smoke及2项兼容测试，所有退出码0。核心字节与正式manifest一致，包内SHA256SUMS覆盖其他载荷；ZIP CRC、缺件、篡改和误导入检查通过。

{zip_table}

下载：[HCP ZIP](team_sync/v5.2-search-final/full_package/Tensor-ABESS-v5.2-HCP.zip)、[NKI ZIP](team_sync/v5.2-search-final/full_package/Tensor-ABESS-v5.2-NKI.zip)、[自检原记录](team_sync/v5.2-search-final/full_package/DELIVERY_VALIDATION.json)。未发送消息或上传他人项目。

内容：src/tensor_abess全部32文件、tests、examples/smoke_test.py、scripts/check_version.py、README、ALGORITHM_CHANGELOG、INSTALL_AND_UPGRADE、依赖列表、公开派生RELEASE_MANIFEST及SHA256SUMS。公开manifest保留完整原始core raw/canonical指纹与上游manifest哈希，明确是派生公开视图；未带私人路径、数据、模拟结果、venv、缓存或旧研究求解器。实际自检环境为本机已有Python3.12.8/NumPy1.26.4/SciPy1.16.1；不冒称已在队友环境通过。

v5→v5.1修复fixed BCD的最佳状态与接受轨迹；v5.1→v5.2补全旧caps内合法整数规模。原MSIC、caps、初始化、Group-BESS及公共接口不变。模拟完全恢复提高，但平均预测误差没有全面改善；本次LEMON还明显恶化，不能保证HCP/NKI改善。

队友按包内INSTALL_AND_UPGRADE：先记录现有import及源码、备份旧core、检查本地修改，再显式更新整个core，不覆盖项目其他文件；使用--source对实际项目目录核验32文件及import。独立项目可直接指向解压src，核心依赖NumPy/SciPy；不复制本机环境或擅自升级依赖。smoke只证明公共fixed/adaptive入口能运行，不是实证或统计正确性的替代。

**保持各自HCP/NKI原样本、响应、预处理、MRI模态顺序、外内验证划分和参数，只替换算法实现并重新拟合ABESS。** 所有训练折重新预处理、生成候选、搜索支持、OLS和预测，不复用旧系数、q、支持或缓存；原输出保留，使用独立v5.2目录。回交完整版本/输入/配置/划分/环境身份、完整OOF、各折支持/q、训练测试误差及失败记录；本仓库没有队友真实入口，不拼造CLI或结果。
''')

    tail = sim['all']
    case_table = table(['task / 场景','q旧→新','RSS旧→新','PMSE旧→新','ΔPMSE','条件总体signal risk旧→新'],
                       [[f"{c['task_id']} / {c['paired_metrics']['experiment']}={c['paired_metrics']['scenario_value']}",
                         f"{c['paired_metrics']['old_q_hat']:g}→{c['paired_metrics']['new_q_hat']:g}",
                         f"{c['paired_metrics']['old_rss_train']:.3f}→{c['paired_metrics']['new_rss_train']:.3f}",
                         f"{c['paired_metrics']['old_pmse']:.4f}→{c['paired_metrics']['new_pmse']:.4f}",
                         f"{c['paired_metrics']['delta_pmse']:+.4f}",
                         f"{c['old']['population_signal_risk']:.4f}→{c['new']['population_signal_risk']:.4f}"] for c in cases])
    trans_table = table(['Exact/PMSE状态','任务数','平均ΔPMSE','Δsignal MSE','ΔREE'],
                        [[r['transition'],r['n'],f"{r['mean_delta_pmse']:+.6f}",
                          f"{r['mean_delta_signal_mse']:+.6f}",f"{r['mean_delta_ree']:+.6f}"] for r in sim['transitions']])
    data_links = '''[700任务配对CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_old_new_700.csv)、[600唯一输入CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_unique_input_600.csv)、[尾部/分场景/分组统计](../numerical_simulation/outputs_version_attribution/saved_result_analysis/summary.json)、[事先冻结个案规则](../numerical_simulation/outputs_version_attribution/saved_result_analysis/CASE_SELECTION.json)、[个案完整诊断](../numerical_simulation/outputs_version_attribution/saved_result_analysis/selected_cases.json)。'''
    write('V52_FAILURE_CASE_ANALYSIS.md', f'''预测恶化个案：训练改进不能替代泛化验证

{stamp}

按已保存PMSE先冻结最大3恶化、最大3改善、最严重丢失Exact及首个“新增Exact却恶化”规则，再读取NPZ；最后一类本批为空，最终7例。仅审原模型，不重训或换规则；每份被读NPZ均核验原record的哈希。当前无法重建严格匹配的正式训练X/y，故未虚构模拟OLS秩/条件数或重新计算训练RSS。

{case_table}

task299：q4→64、RSS217.11→32.84，而PMSE3.76→9.60；task259相似，训练更好却测试更差。task562由真支持q60变成非真q125、RSS降低而PMSE上升；零相关下false-positive系数对条件总体risk的贡献从0升至2.3404。task127却在q15→80后PMSE改善；task277也在q增加时改善，反证“所有大模型变化都一定更差”。

条件总体signal risk直接由保存的E=B_hat−B_true与声明的可分离AR1协方差计算Var(<X,E>)，未生成新测试数据；拆为真块内系数误差、块外误选系数和交叉协方差项，完整项见JSON。它是**给定已估B的总体预测风险**，不是重复采样估计方差/偏差分解，不把块外风险一律叫作方差；相关设计交叉项不能省略。该风险与有限500测试样本signal MSE不同，不能替换原正式指标。

{data_links}
''')
    write('SUPPORT_RECOVERY_VS_PREDICTION.md', f'''恢复提高与平均预测恶化的真正关系

{stamp}

**本批没有“新增Exact成功但PMSE恶化”的观察例子。** 新增57次完全恢复均伴随PMSE改善；丢失13次完全恢复均恶化。整体Exact+44但平均PMSE更差，主要是不同任务上的收益与尾部损失相加，不能描述为恢复本身造成预测恶化。

{trans_table}

未出现的交叉组数量为0，不省略或造例。新增成功57中48来自n400；排除n400后其余650标签Exact137→133，整体恢复增益高度集中。81个Exact不变且PMSE恶化任务的变化同样是重要损失来源，不能只看恢复率。signal MSE和REE的方向亦见表，各指标是不同评价目标。

一般统计意义上真支持恢复并不保证有限样本预测误差最小，但当前这组新旧配对没有直接展示“由失败变成功同时预测更差”；应把一般原理与本批实际观察分开。q=60只代表规模，不证明位置正确；EEG实证没有真支持，Jaccard不等于真实支持恢复。

{data_links}
''')
    complexity_table = table(['恶化任务分组','n','平均ΔPMSE','训练RSS降低次数'],
                             [[r['group'],r['n'],f"{r['mean_delta_pmse']:.6f}" if r['mean_delta_pmse'] is not None else 'NA',r['lower_train_rss_n']]
                              for r in sim['complexity']])
    write('SIMULATION_ERROR_ATTRIBUTION.md', f'''模拟误差恶化：尾部集中与版本归因边界

{stamp}

700标签改善/持平/恶化为99/507/94（绝对差≤1e−12视作持平）。ΔPMSE均值{tail['mean']:.6f}，中位数0，95/99分位数{tail['quantiles']['0.95']:.6f}/{tail['quantiles']['0.99']:.6f}；最小/最大{tail['quantiles']['0']:.6f}/{tail['quantiles']['1']:.6f}。正差累计{tail['positive_sum']:.6f}，改善抵消{tail['negative_sum']:.6f}，净损失{tail['net_sum']:.6f}。94次恶化平均约1.3200，大于99次改善的平均幅度约0.2315，次数接近不意味着幅度抵消。

最大1/5/10/20例占全部正差的{tail['top_1_positive_share']:.2%}/{tail['top_5_positive_share']:.2%}/{tail['top_10_positive_share']:.2%}/{tail['top_20_positive_share']:.2%}；占净恶化{tail['top_1_net_share']:.2%}/{tail['top_5_net_share']:.2%}/{tail['top_10_net_share']:.2%}/{tail['top_20_net_share']:.2%}。不能说由单一离群任务决定；前20例显著集中尾部，前5例不足以解释多数净恶化。n100与rho0分别贡献净损失约61.56%及18.50%，合计约80.06%；不得据此删场景。

{complexity_table}

94次恶化中73次q增加且RSS降低，8次q不变但RSS降低，13次q减少且RSS没有降低。接近容量(q/Q≥.9)的恶化仅12次，说明复杂度增大是重要机制，但不能解释所有恶化；同规模也可能换位置，缩小模型也可能欠拟合。原MSIC在不同训练拟合候选上选择，训练RSS改进不蕴含测试risk改进。

![配对误差全分布与场景效应](taskcard6/figures/simulation_tail.png)

图a包含全部700标签，无删点；图b每标签50配对差，均值及描述性t区间，未校正多重比较。重复输入标签仍保留展示，但不当作700独立样本；去重600种输入仍是89/427/84、均值差0.164224，中位数0，也不声称600种输入IID。

P0尚无250次v5.1数值，故不能填Δ1/Δ2分布。源码/网格证据能排除n100和所选n200的A2新增规模解释；n400候选覆盖收益有明确机制但尚未完成当前协议的两段数值拆分。个案条件总体risk支持误差恶化来自已保存系数，而不是把测试噪声当成唯一解释；它并不建立重复采样的完整偏差方差分解。

{data_links}
''')

    write('FINAL_SIMULATION_ASSESSMENT.md', f'''模拟综合评价（阶段性，250归因待完成）

{stamp}

原v5及v5.2正式700批有完整身份与同输入比较，v5.2工程验收通过；14标签对应10配置、600种完整输入，不能作700独立样本总体推断。Exact138→182，但mean PMSE2.236761→2.381274，Mode-F1仅0.751917→0.753593；n400收益集中，其他650标签净丢失4次Exact。小样本、弱信号及部分rho0任务贡献明显尾部风险。

fixed BCD最佳状态保存修复具有正确性证据，不建议恢复已知缺陷来追求偶然低测试误差。A2只扩大旧caps内规模；n100真规模仍库外，n400真规模由全局fixed网格外进入网格内（旧池仍含可能命中真规模的adaptive候选），n116和选定n200的网格没有扩大。源代码等价性与覆盖证据不能代替本卡尚未完成的250次v5.1实际拟合。

当前支持“特定场景下结构恢复受益，训练优化与有限样本泛化存在取舍”，不支持“全场景预测提高、完整位置穷举、理论全局最优或普遍优于全部稀疏回归”。有条件进入论文证据整理，不能按已完成三版本归因或真实脑数据普适优势定稿。

参见[恢复与预测](SUPPORT_RECOVERY_VS_PREDICTION.md)、[尾部分析](SIMULATION_ERROR_ATTRIBUTION.md)、[未完成归因及恢复步骤](VERSION_EFFECT_DECOMPOSITION.md)。HCP/NKI尚未返回，不虚构其结果。
''')
    write('LEMON_EMPIRICAL_CONCLUSIONS.md', f'''LEMON实证结论：当前配置不支持EEG增量

{stamp}

v5.2完整75 fits/2175 OOF已验收；相同145人、输入、响应/协变量、划分与环境，新ABESS平均OOF RMSE0.504581、R²−0.597008；仅协变量0.317485、0.367900。新ABESS三个重复均比旧版更差，pooled相对协变量ΔRMSE−0.187159，不能称EEG增加预测价值。

q80从1/15增到5/15，训练误差降低、测试误差增加；通道Jaccard下降，频带重叠升高部分可能因支持变大。30个旧新ABESS设计满秩及保存预测独立重算一致，未发现需要救补的数值/身份错误。保持该负面结果，不删被试/折、不修改MSIC/caps。

结论限于现有LEMON特征、TMT目标、年龄/性别调整及内部验证；不证明EEG无信息，也不决定HCP/NKI表现。HCP/NKI仍应保持原设计交回真实结果。优先检查矩形稀疏结构与数据特征匹配、有限样本复杂度和稳定性；不自动试参数获取更好排名。

完整数值与所有重复/折见[实证报告](LEMON_V52_EMPIRICAL_REPORT.md)。
''')
    write('TENSOR_ABESS_RESEARCH_STATUS.md', f'''Tensor-ABESS当前研究状态（阶段性）

{stamp}

| 证据 | 状态 | 可支持结论 |
|---|---|---|
| fixed BCD修复及已冻结发布 | 已有验证；正式代码未变 | 最佳已访问状态管理正确性，不等于预测优越 |
| A2独立任务卡4验证 | 已有独立队列证据 | 特定场景规模覆盖/恢复收益 |
| 新版正式700模拟 | 完整验收 | 恢复均值改善但预测均值恶化、明显适用边界 |
| 本卡250个v5.1标签 | 环境/输入Gate阻塞，实际0/250 | 只有方案、失败预检及结构证据；不能给完成统计 |
| LEMON v5.2 | 完整75/2175并严格旧新核对 | 当前配置无EEG增量、q80风险增加 |
| 两份完整ZIP | 独立解压/import/fixed/adaptive自检通过 | 可交队友使用；不证明队友实证已运行 |
| HCP/NKI | 未返回、未核查其工程 | 不能给具体数据结论 |

论文可组织为三阶矩形支撑稀疏回归、Group-BESS/BCD搜索、原MSIC选阶、实现正确性修复及候选规模覆盖与泛化取舍；区分既有算法骨架、项目实现和本轮改动，不夸大为新统计理论。未新查外部文献，方法贡献/理论新颖性仍需独立原始文献核实。

当前可以进入诚实的论文框架与结果材料整理；完整归因及最终普适实证主张仍缺证据。见[下一步决策](NEXT_RESEARCH_DECISION.md)。
''')
    write('NEXT_RESEARCH_DECISION.md', f'''下一阶段决策：保持冻结，补归因并研究结构匹配

{stamp}

**建议保持正式v5.2代码作为可复核基线，先完成原环境250归因，保留LEMON负面结果并等待队友验证；不自动再改算法。** 不把“维持版本身份”当作推荐其在全部实际数据上使用，更不能以总体Exact提升掩盖预测风险。

优先级1：在原Linux环境通过严格数据/环境Gate，完成五场景250个v5.1标签和Δ1/Δ2；已有脚本与计划可用，不能用近似匹配解阻。

优先级2：两名队友用完整包在原HCP/NKI设计重跑ABESS，返回完整OOF、容量和稳定性；LEMON已经完成，无需为排名重复划分或改参数。

优先级3：从特征/真实支持与矩形结构是否匹配、训练样本容量和不稳定选择角度提出下一轮独立研究。新的caps/MSIC/正则化如要研究，应重新立项、预注册划分/规则与验证预算，并使用新的未见数据或独立确认；当前测试集不能用来选择最终参数。

可以开始论文写作与正式实证材料汇总，前提是明确阶段性、完整负面结果和缺失归因。暂不支持“全面优于旧算法”“EEG增量已成立”或“普遍优于所有稀疏方法”的对外结论。HCP/NKI未返回不阻碍这些已有证据下的判断。
''')
    write('TASKCARD6_EXECUTION_REPORT.md', f'''任务卡6执行交付索引

{stamp}

| 阶段 | 实际结果 |
|---|---|
| P0 | 250标签/种子/配置/输入与三版源码已冻结；严格本机预检失败，v5.1实际拟合0/250；已给原集群恢复入口与结构审计 |
| P1a | LEMON完整运行成功，75 fits/2175 OOF；32core/输入/划分/环境及旧新配对通过；独立30个OLS/预测核验通过 |
| P1b | HCP/NKI两份同字节ZIP均完整32源码，独立解压、import、fixed/adaptive及兼容测试通过 |
| P2 | 全700配对/600唯一输入、尾部/恢复预测交叉组及7个预选无重拟合个案已完成；模拟训练OLS无法重建处明确限界 |
| P3 | 综合评价及后续决策为阶段性，未虚构250归因或HCP/NKI结果 |

主要结论：LEMON ABESS平均OOF RMSE0.371667→0.504581，R²0.121430→−0.597008，q80折数1→5，未提供协变量之外增量。模拟新增57个Exact成功均伴随PMSE改善；总体平均变差由其他任务的尾部损失超过改善，前20个恶化贡献约72.20%净损失，n100+rho0贡献约80.06%。n100/n200/LEMON网格未增加，不能将这些条件下变化归因A2新增规模。

交付文档：

1. [三版本归因/恢复步骤](VERSION_EFFECT_DECOMPOSITION.md)、[BCD统计影响](BCD_FIX_STATISTICAL_IMPACT.md)、[A2影响](A2_GRID_STATISTICAL_IMPACT.md)。
2. [LEMON完整报告](LEMON_V52_EMPIRICAL_REPORT.md)、[旧新比较](LEMON_V5_V52_COMPARISON.md)。
3. [队友ZIP与哈希/使用说明](TEAM_V52_DELIVERY.md)。
4. [失败个案](V52_FAILURE_CASE_ANALYSIS.md)、[恢复—预测关系](SUPPORT_RECOVERY_VS_PREDICTION.md)、[模拟误差归因](SIMULATION_ERROR_ATTRIBUTION.md)。
5. [模拟最终评价](FINAL_SIMULATION_ASSESSMENT.md)、[LEMON结论](LEMON_EMPIRICAL_CONCLUSIONS.md)、[研究状态](TENSOR_ABESS_RESEARCH_STATUS.md)、[下一阶段决策](NEXT_RESEARCH_DECISION.md)。

P0恢复需原集群或经严格指纹验证的等价环境；本轮未访问/提交新集群作业，未发送队友消息，未更改正式算法、MSIC/caps、原始数据或旧结果。新的LEMON结果不是性能优越证据，必须与负面发现一起交付。
''')
    print('Rendered 14 task-card-6 report/index files from validated saved results.')


if __name__ == '__main__':
    main()

