# 任务卡12：终版建议 B

**最终建议B：原MSIC / v5.2-search-final继续作为正式默认，FS仅保留冻结研究对照。** 380组正式结果与3组smoke的保存链核验通过；用户回传原Linux环境严格只读核验成功，重算summary和analysis与归档完全相等。主360组PMSE下降19.65%、n100下降28.70%，但11项新门槛中4项恢复风险门槛失败，不支持FS替代正式默认。

任务卡11的事实继续有效：新96对中总体PMSE下降26.92%、n100下降33.78%、n400 Exact保持30/48；强信号q欠选增量0.125超过当时0.05门槛，故联合推广失败。任务卡12不修改该门槛、分母或历史决策，而是按用户明确授权对同一FS公式作更充分的n200恢复风险和预测研究。

完整门槛、区间、压力场景和失败案例见[最终审阅报告](cluster_review/FINAL_VERSION_RECOMMENDATION.md)，验收来源及跨环境历史见[验收状态](cluster_review/REVIEW_STATUS.md)。压力30例Exact由12降至8，原规则正确恢复的12例中有5例被FS丢失；条件丢失率上界0.68476超过0.20门槛。未改阈值、候选搜索或正式版本。

停止继续调整FS公式和阈值，不按测试表现逐数据集切换规则；后续重心转向原MSIC的数值模拟、实证及论文。FS预测收益可作为研究结果报告，但不能掩盖恢复代价。

实证状态与实际分数见[EMPIRICAL_DUAL_MSIC_REPORT.md](EMPIRICAL_DUAL_MSIC_REPORT.md)，实际已完成数量和所有未运行阶段见[FINAL_VALIDATION.json](FINAL_VALIDATION.json)。HCP-YA/NKI-RS输入尚未提供时只能交付运行契约，不把团队反馈当作实测成绩；LEMON属于已见数据上的配对消融，不能称为新的盲法外部确认。

本轮集群回传分析已完成，无需重跑模拟。原待回传交付快照及旧索引保存在[local_delivery_snapshot](cluster_review/local_delivery_snapshot/FINAL_VALIDATION.json)，新验收记录见[FINAL_VALIDATION.json](FINAL_VALIDATION.json)。HCP-YA/NKI-RS仍依赖尚未提供的输入，不计为已完成实测；没有批准FS发布。
