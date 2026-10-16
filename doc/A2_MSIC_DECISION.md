# A2上的MSIC最终决策（任务卡4）

## 结论

保留原MSIC，推荐D2：旧caps内完整合法整数规模grid，mode alpha=1，global alpha=1.5、gamma=1、one_se_scale=1。开发筛选支持进一步确认，但独立80数据确认触发事前3600秒预算停止，未形成完整确认及fresh-refit证据；不能批准alpha=2/scale=0或D3。此次为证据不足时按协议保留原规则，不是证明候选规则无效。

## 开发评价：完整90数据、两个固定规则

只重评分任务卡3已有development的A2训练候选库；不重新拟合，不搜索更多alpha/scale，不把开发结果称为新的独立确认。90个原规则模型逐值重放，180评价行完整，候选池指纹保持，完整性记录见[开发integrity](../numerical_simulation/outputs_a2_msic_decision/integrity_report.json)。

| 指标 | 原1.5/scale1 | 候选2/scale0 |
|---|---:|---:|
| Exact | 29/90 | 28/90 |
| Mode-F1 | 0.617443 | 0.598313 |
| Element-F1 | 0.449405 | 0.447195 |
| PMSE | 2.996586 | 2.779899 |
| 平均q | 43.077778 | 35.188889 |
| 过选率 | 0.155556 | 0.122222 |
| 欠选率 | 0.511111 | 0.566667 |
| q=1率 | 0.033333 | 0.022222 |
| q≥0.9Q率 | 0.066667 | 0.066667 |

候选改变44/90个B；PMSE胜/平/负为23/46/21。整体预测改善伴随恢复与欠选权衡，不能只凭整体PMSE批准新规则。强信号且caps可表示子集的Exact增益0、Mode-F1增益0.002315、欠选率增量0；pool PMSE比0.927689，最坏场景比1.032822，因此事前开发筛选通过。判据及完整场景见[锁定规则](../numerical_simulation/outputs_a2_msic_decision/MSIC_DECISION_RULES.json)、[screen](../numerical_simulation/outputs_a2_msic_decision/DEVELOPMENT_SCREEN_RESULT.json)及[场景汇总](../numerical_simulation/outputs_a2_msic_decision/scenario_summary.csv)。这些是同一开发证据的描述，不提供独立确认保证。

## 独立确认：预算停止，禁止续跑或部分验收

事前冻结16场景×5重复=80新数据：两形状(2,3,4)/(3,4,5)，主70及rho=.6压力10，仅原规则与已锁定候选；seed_base=30601011，8 worker、每worker单BLAS线程，候选预算2000，全部阶段共用3600秒。protocol SHA256为`c70b62b854c041127dbd4bb45c77856e53ed9bb29b656e2c033b9baae4d29be3`，plan_id为`1857d78e504ede09a0e305d6cffa338007bea87b0a1f84ac80a2ee9a3843c828`。

控制器于北京时间2026-10-11 16:25:25启动，17:25:25以`Shared confirmation budget exceeded`停止，`resume_allowed=false`；停止时核验68/80评价完整。控制器finally等待既有worker期间仍落盘产物，随后仅结束已核实的本任务控制器及子进程；最终目录保留71个jobs JSON和71个evaluations JSON，但新增3项不能被计作预算内完成证据。未完成/迟到产物均保留，未删除、补改、重跑或按部分结果进行统计批准。

fresh refit为0/3，完整analysis.json与integrity_report.json均未生成；不报告80数据、160模型或独立统计门禁已通过。预算停止记录见[STOPPED.json](../numerical_simulation/outputs_a2_msic_confirmation/STOPPED.json)（SHA256 `bbb1c7cddd46a7bc0ce46e9ef5c11e3a9f1065c344347a75b1c59564c41a03bb`），冻结[protocol](../numerical_simulation/outputs_a2_msic_confirmation/protocol.json)不变。

冻结规则明确“failed, incomplete or ambiguous confirmation retains original MSIC”；本次符合不完整确认条件，因此保留原MSIC，禁止在本计划上续跑、扩大预算、补齐或继续调参。未调用停止计划的summarize/refit，不将部分测试指标用于改规则，也未利用LEMON外层测试或A3规则直接批准A2。

## 完成边界

MSIC决策已按冻结规则完成：原规则保留；独立确认实验本身未完成。A2搜索的100配对、16冷启动及200所选OLS完整证据仍有效，见[A2搜索决策](A2_VS_A0_DECISION.md)。最终推荐、源码与157项限定测试见[FINAL_ALGORITHM_VERSION.md](FINAL_ALGORITHM_VERSION.md)；正式发布仍待用户批准，未替换生产或启动700模拟/三个实证重训。
