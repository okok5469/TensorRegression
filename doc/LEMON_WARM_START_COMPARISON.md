# LEMON Warm/Cold比较状态

任务发起日期：2026-10-16（北京时间）；实际UTC起止以冻结协议及控制器记录为准。

**Warm 15折未运行：NOT_RUN_CORRECTNESS_GATE_FAILED。** incoming原样方案虽通过上传6项toy测试，但未通过历史seed2026恢复、最佳已访问状态及其他正式契约。任务卡要求先过正确性，本轮因此没有Warm RMSE、R²、q80、支撑稳定性或计时结果，不以队友反馈填表，也没有重算对照。

| 指标 | 既有v5.2 Cold | Warm研究版 |
| --- | --- | --- |
| 完整OOF RMSE（3次完整OOF均值） | 0.504581 | 未运行 |
| 完整OOF R²均值 | -0.597008 | 未运行 |
| 相对协变量RMSE差 | +0.187096（协变量0.317485） | 未运行 |
| q80 | 5/15 | 未运行 |

这些Cold数字仅引用任务卡7已验收的历史OOF；没有新的Warm/Cold配对性能证据。没有创建虚假的outputs_warm_start_research完整标记。即使将来修正候选通过门禁，LEMON已经反复查看，也只能用于探索性历史比较，不可用来反复调参或宣布独立验证。

见[代码审计](WARM_START_CODE_AUDIT.md)与[正确性证据](WARM_START_CORRECTNESS_REPORT.md)。[复现记录](taskcard8/REPRODUCIBILITY.md)；[交付索引](taskcard8/DELIVERY_INDEX.json)。
