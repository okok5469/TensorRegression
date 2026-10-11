# v5.1-bcd-fix：fixed BCD 正确性修复

## 范围与版本

仅修改 `src/tensor_abess/tensor/bcd.py`，原Group-BESS目标函数、交换定义、MSIC公式/默认参数、初始化策略、数据生成及对照算法均未改动。公共tensor_abess/tensor_bcd参数保持兼容；fixed结果新增诊断字段，详见交接文档。

原v5完整32源码快照与指纹在 `doc/algorithm_versions/v5-original/`；修复版指纹在 `doc/algorithm_versions/v5.1-bcd-fix/VERSION.json`。当前Git HEAD为d4e28969ac480ba6efc01bcc985a53d179e7ed31，未创建分支/提交/推送，也未改动用户其他未提交文件。已有1738个结果文件先取SHA256，最终逐文件复核均未改变，核验产物在doc/algorithm_versions/final_integrity_report.json。

## 缺陷根源

旧fixed BCD每mode调用group_bess_fix未传当前groups_init，子问题重新初始化；仅依据支持不同直接接受，没有统一完整矩形OLS损失比较。group_splicing相对它自己的输入状态只接受改善，但不能保证相对外层BCD当前状态改善。旧流程也只返回最后状态，未保存已访问最优状态，因此可以RSS回升并循环。

## 修复

1. 保留原v5确定性cold Group-BESS探索机会；探索支持与正式接受最优支持分别维护，保持fixed group数量。探索可以暂时变差，但不覆盖最优模型。
2. 每候选重新拟合完整矩形OLS，使用同一tensor_loss；只有相对最优接受值的改善超过64×float64 eps×max(1,|accepted_loss|,|candidate_loss|)时进入正式接受历史。
3. 全部有效完整OLS候选都进入visited记录，best supports/B/loss/RSS/索引属于同次拟合，最终返回真实最优已访问状态。
4. 在完整sweep边界识别探索支持状态重复，确定性未来路径重复时停止；不能因一个sweep没有改善best就截断后续探索。初始和候选OLS结果必须有限。
5. 用三个mode的统一循环替代三段重复代码；不改Group-BESS内部数学步骤，也不加入人工q上限。

这保证接受轨迹非增及返回已访问最优，但不声称非凸矩形搜索达到所有支持中的全局最优。容差内更低损失候选可被best记录而不作为正式接受更新；诊断字段明确区分两者。

## 自动验收

最终新10项unittest及原5项pytest共15项通过（1.77秒），原实证/数值调用层26项全部通过（74.825秒），研究模块12项通过（7.563秒），共53项。涵盖变差提案不覆盖best、暂时变差探索后恢复更优状态、best visited、RSS/OLS一致、循环停止、有限性、数量/形状、复现及穷举参照；原seed2026真实支撑恢复断言保持原样并通过。

首个“warm-only+逐mode拒绝”草案虽通过15项局部正确性测试，却失败原真实支撑恢复回归；已立即停止其研究，保留v5.1-warm-only-rejected快照及ABORTED记录，未作为最终版本。最终采用任务卡3.2B明确允许的探索/接受分离；旧探索候选得以保留，避免通过改测试或缩小q制造成功。

原组交换和Group-BESS源码未改。新32源码Python3.9语法检查通过。MSIC缓存重评分测试在独立模块进行，不属于P0修改。

本机原环境没有pytest，测试工具仅安装到 `doc/algorithm_versions/test_tools/`，测试进程临时PYTHONPATH指向该目录；现有venv、全局Python与集群均没有升级。这些可再生工具不在团队patch内。

## LEMON异常折修复前后重放

固定相同外层训练数据和预处理，原版独立进程从原v5快照加载，修复版从当前src加载；所有产物另存 `numerical_simulation/outputs_bcd_fix_validation/`。

|指标|v5-original|v5.1-bcd-fix|
|---|---:|---:|
|完整adaptive最终q|80|80|
|最终训练RSS|1.234092|1.067482|
|诊断性旧外层测试RMSE|0.670771|0.619982|
|单独q80返回RSS|1.234092|1.067482|
|q80正式接受轨迹单调|否|是|

原q80周期轨迹完整保留：2.217202→1.067482→1.366732→1.234092→1.067482→1.366732→1.234092。最终修复的正式接受RSS为2.217202→1.638195→1.067482，4个sweep后识别探索循环，返回best_iteration=2（visited索引）的RSS1.067482；探索历史仍允许回升并单独保存，不冒称探索本身单调。

最终修复保留q80且RSS更低，MSIC从−307.724348降低到−324.548155，过选问题没有因正确性修复消失。旧外测RMSE只是诊断，不作为选参或替换正式结果依据。原草案曾q1/RSS8.956318/RMSE0.357583，但其原回归失败，不能因这折预测更好而采用；正确性优先。

## 后续

MSIC默认保持alpha1.5/gamma1/one-SE1，研究单独记录在MSIC_ABLATION_REPORT.md。27数据集/54任务配对研究已完成，3051个global-fixed候选接受轨迹非增且返回best；默认最终B有6/27改变，PMSE平均增加0.302069（2胜/21平/4负），不能把搜索正确性当预测提升承诺。原历史seed2026指标复现最大差2.8422e−14，最终修复默认仍q60/Exact1/PMSE1.3250717。

由于模型输出改变，采用v5.1需完整重跑五组full模拟和实证ABESS；700任务正式full尚未完成，根run_bcd_fix_full.sh提供prepare→sbatch→任务全部结束后merge三阶段，输出单独outputs_v51_bcd_fix_full。原正式结果保留为v5-original，不能直接换版本标签。本轮MSIC研究不足以批准新v6；候选池覆盖、强罚欠选、CV支持搜索不稳定详见独立报告。

## 最终本机full入口预检

最终outputs_v51_final_full_preflight计划700任务，已完成task0全部adaptive/oracle/lasso/glasso四方法，JSON/NPZ、源码/环境/数据/支持与指标校验通过；再次--resume仅验证并跳过，不重复训练。详细首任务耗时和历史指标比较见doc/algorithm_versions/full_preflight_report.json。本机仅1/700完成，余699未运行，尚未提交集群，不能称正式full已完成。
