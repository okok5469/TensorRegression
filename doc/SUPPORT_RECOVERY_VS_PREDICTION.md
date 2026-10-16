# 恢复提高与平均预测恶化的真正关系

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

**本批没有“新增Exact成功但PMSE恶化”的观察例子。** 新增57次完全恢复均伴随PMSE改善；丢失13次完全恢复均恶化。整体Exact+44但平均PMSE更差，主要是不同任务上的收益与尾部损失相加，不能描述为恢复本身造成预测恶化。

| Exact/PMSE状态 | 任务数 | 平均ΔPMSE | Δsignal MSE | ΔREE |
| --- | --- | --- | --- | --- |
| gained_exact_better_pmse | 57 | -0.166321 | -0.159396 | -0.104331 |
| lost_exact_worse_pmse | 13 | +1.561689 | +1.604295 | +0.615720 |
| same_exact_better_pmse | 42 | -0.320029 | -0.325766 | -0.105929 |
| same_exact_equal_pmse | 507 | +0.000000 | +0.000000 | +0.000000 |
| same_exact_worse_pmse | 81 | +1.281215 | +1.255161 | +0.288366 |

未出现的交叉组数量为0，不省略或造例。新增成功57中48来自n400；排除n400后其余650标签Exact137→133，整体恢复增益高度集中。81个Exact不变且PMSE恶化任务的变化同样是重要损失来源，不能只看恢复率。signal MSE和REE的方向亦见表，各指标是不同评价目标。

一般统计意义上真支持恢复并不保证有限样本预测误差最小，但当前这组新旧配对没有直接展示“由失败变成功同时预测更差”；应把一般原理与本批实际观察分开。q=60只代表规模，不证明位置正确；EEG实证没有真支持，Jaccard不等于真实支持恢复。

[700任务配对CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_old_new_700.csv)、[600唯一输入CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_unique_input_600.csv)、[尾部/分场景/分组统计](../numerical_simulation/outputs_version_attribution/saved_result_analysis/summary.json)、[事先冻结个案规则](../numerical_simulation/outputs_version_attribution/saved_result_analysis/CASE_SELECTION.json)、[个案完整诊断](../numerical_simulation/outputs_version_attribution/saved_result_analysis/selected_cases.json)。
