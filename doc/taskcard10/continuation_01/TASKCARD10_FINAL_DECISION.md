# 任务卡10追加执行最终报告

原900秒/7200秒阶段及其NOT_RUN、报告、FINAL原样保留；本次依据用户继续授权追加计算协议，2worker、单case3600秒、开发与确认共享28800秒时钟。科学方法、原96请求、九门槛、候选库及选择规则不变。原30例结果按字节复用，重算原18例未完成请求，不将复制或重试计作新科学重复。

开发48/48（原30复用、本次新增完成18），确认0/48；下列结论来自development。

1. **n100：** A0/A1/A2平均PMSE=4.679961/3.231389/3.128412，A2/A0=0.668470；P95=13.278015/6.708884/5.892376，每臂24例。

2. **容量：** n100的A1−A0=-1.448572，同容量A2−A1=-0.102977；前者含范围/来源混合作用，后者才是CV增量。

3. **n400：** A0/A1/A2 Exact=13/13/11，分母各24；PMSE=1.202004/1.202004/1.267451；A0恢复但A2丢失2例，其中无公共真支撑0、有真支撑未选2。

4. **MSIC：** 全组同库A2−A1平均PMSE=-0.018765；逐场景变化在机制表完整列出，不以总均值掩盖尾部或把后验测试最优当作选择准则。

5. **同规模风险：** 当前设计每形状只有一个full fixed候选，Q5不可识别、尚未实证解决；该限制不由补齐样本消失，也不据已见结果新增来源来挽救门槛。

6. **成本：** 成功完成程序的A2/A0实测比2.393；旧4worker与续跑2worker混合，失败/重试成本另列，不能解释为统一调度条件下的生产速度基准。

7. **后续决策：暂定B。** n100及弱信号存在同库改善，但总体同库平均PMSE仅降低0.85%，未达2%门槛；n400 Exact从13/24降至11/24，最坏场景相对原生基线恶化至2.836倍，且强信号欠选率相对A0增加18.75个百分点，收益不足以抵消预定恢复和稳健性代价；独立确认因开发门槛失败未启动，建议仅为暂定。 保留研究实现，不替换正式算法。 未通过的完整门槛：overall_pmse_ratio_A2_A1_max, worst_complete_cell_pmse_ratio_A2_A0_max, n400_exact_rate_delta_A2_A0_min, n400_pmse_ratio_A2_A0_max, strong_snr4_q_under_delta_A2_A0_max。

## 完成范围

三臂实现、原46项门禁、Pilot4例及P1/P2设计沿用原验收；本次追加计算修订和完整来源核验。原停止事实永久保留；最终科学判断由本次真实完成数量及门槛决定，不能用报告文件存在代替实验完成。Q5范围判定已报告，实证问题仍未解决；独立确认若因预声明开发门槛失败而未执行，属于条件性停止，不能写成确认通过。

| 原冻结门槛 | 实际值 | 阈值 | 通过 |
|---|---|---|---|
| n100_pmse_ratio_A2_A0_max | 0.668470 | 0.95 | True |
| n100_pmse_p95_ratio_A2_A0_max | 0.443769 | 1.1 | True |
| overall_pmse_ratio_A2_A0_max | 0.747346 | 0.98 | True |
| overall_pmse_ratio_A2_A1_max | 0.991535 | 0.98 | False |
| worst_complete_cell_pmse_ratio_A2_A0_max | 2.836125 | 1.15 | False |
| n400_exact_rate_delta_A2_A0_min | -0.083333 | -0.05 | False |
| n400_pmse_ratio_A2_A0_max | 1.054448 | 1.05 | False |
| strong_snr4_q_under_delta_A2_A0_max | 0.187500 | 0.05 | False |
| measured_program_cost_ratio_A2_A0_max | 2.393474 | 6.0 | True |

[开发结果](DEVELOPMENT_RESULTS.md)、[确认状态](INDEPENDENT_CONFIRMATION.md)、[机制分析](SELECTION_MECHANISM_ANALYSIS.md)、[追加协议](DESIGN_AMENDMENT.md)、[复现与成本](REPRODUCIBILITY.md)、[最终验收](FINAL_VALIDATION.json)、[原阶段报告](../TASKCARD10_FINAL_DECISION.md)。
