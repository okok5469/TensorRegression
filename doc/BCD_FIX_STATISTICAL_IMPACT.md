# BCD正确性修复的统计影响

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

P0的250标签方案已冻结，但**严格配对拟合0/250，未完成**。本机Python3.12.8/NumPy1.26.4/SciPy1.16.1重生正式task0时，仅B_true指纹相同，X_train、y_train、train_noise均不同；保存的原环境为Linux/Python3.9.25/NumPy2.0.2/SciPy1.13.1/OpenBLAS0.3.27。没有用相同seed、近似数值、改dtype或替换数据冒充严格配对，也未安装或升级依赖。

事实证据：[预检原记录](../numerical_simulation/outputs_version_attribution/cluster_run/preflight_task_000000.json)、[冻结250计划](../numerical_simulation/outputs_version_attribution/cluster_run/plan.json)。计划指明task0–49、50–99、250–299、350–399、550–599，各50标签，共250种完整输入；没有新提交700任务。早期根目录plan/preflight是工具定稿前的失败预检留存，实际后续协议使用cluster_run子目录；不覆盖先前证据。

因此不能填造本批v5→v5.1的Δ1和v5.1→v5.2的Δ2数值、胜平负或成本。以下结构检查与已有独立研究可以限定解释，但不能计为这250次已运行。

fixed BCD修复具有已有最佳已访问状态保存、接受轨迹非增及回归验证证据；它修正搜索状态管理，不承诺降低样本外风险。不能因旧缺陷偶然产生更小测试误差而恢复旧BCD。

在n=100、所选n=200和LEMON条件下，两版网格及顺序一致，其他core及BCD剩余AST相同，结构上不支持把变化归因A2新增规模；这指向BCD实现变化与原训练选择相互作用的解释。具体Δ1的配对分布仍需完成250诊断，不能将结构推断改写成测量结果。

LEMON新q80从1折增为5折，训练误差降低而OOF误差明显增加，说明优化训练目标与泛化性能不同；其具体实证证据见[LEMON报告](LEMON_V52_EMPIRICAL_REPORT.md)，不是本次模拟Δ1的替代。

完整恢复步骤见[拆分报告](VERSION_EFFECT_DECOMPOSITION.md)。
