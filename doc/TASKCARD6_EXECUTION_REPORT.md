# 任务卡6执行交付索引

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

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
