# 模拟误差恶化：尾部集中与版本归因边界

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

700标签改善/持平/恶化为99/507/94（绝对差≤1e−12视作持平）。ΔPMSE均值0.144513，中位数0，95/99分位数1.196549/4.063740；最小/最大-1.817478/5.832278。正差累计124.080355，改善抵消-22.921523，净损失101.158833。94次恶化平均约1.3200，大于99次改善的平均幅度约0.2315，次数接近不意味着幅度抵消。

最大1/5/10/20例占全部正差的4.70%/21.04%/37.27%/58.86%；占净恶化5.77%/25.81%/45.71%/72.20%。不能说由单一离群任务决定；前20例显著集中尾部，前5例不足以解释多数净恶化。n100与rho0分别贡献净损失约61.56%及18.50%，合计约80.06%；不得据此删场景。

| 恶化任务分组 | n | 平均ΔPMSE | 训练RSS降低次数 |
| --- | --- | --- | --- |
| all_worsened | 94 | 1.320004 | 81 |
| worsened_q_increased | 73 | 1.554733 | 73 |
| worsened_q_equal | 8 | 0.175910 | 8 |
| worsened_q_decreased | 13 | 0.705968 | 0 |
| worsened_near_capacity | 12 | 3.495733 | 12 |

94次恶化中73次q增加且RSS降低，8次q不变但RSS降低，13次q减少且RSS没有降低。接近容量(q/Q≥.9)的恶化仅12次，说明复杂度增大是重要机制，但不能解释所有恶化；同规模也可能换位置，缩小模型也可能欠拟合。原MSIC在不同训练拟合候选上选择，训练RSS改进不蕴含测试risk改进。

![配对误差全分布与场景效应](taskcard6/figures/simulation_tail.png)

图a包含全部700标签，无删点；图b每标签50配对差，均值及描述性t区间，未校正多重比较。重复输入标签仍保留展示，但不当作700独立样本；去重600种输入仍是89/427/84、均值差0.164224，中位数0，也不声称600种输入IID。

P0尚无250次v5.1数值，故不能填Δ1/Δ2分布。源码/网格证据能排除n100和所选n200的A2新增规模解释；n400候选覆盖收益有明确机制但尚未完成当前协议的两段数值拆分。个案条件总体risk支持误差恶化来自已保存系数，而不是把测试噪声当成唯一解释；它并不建立重复采样的完整偏差方差分解。

[700任务配对CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_old_new_700.csv)、[600唯一输入CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_unique_input_600.csv)、[尾部/分场景/分组统计](../numerical_simulation/outputs_version_attribution/saved_result_analysis/summary.json)、[事先冻结个案规则](../numerical_simulation/outputs_version_attribution/saved_result_analysis/CASE_SELECTION.json)、[个案完整诊断](../numerical_simulation/outputs_version_attribution/saved_result_analysis/selected_cases.json)。
