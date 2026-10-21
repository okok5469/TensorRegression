# Tensor-ABESS下一版本研究决策

任务发起日期：2026-10-16（北京时间）；实际UTC起止以冻结协议及控制器记录为准。

## 一页阶段结论

**Warm：** 收到真实bcd.py并原样隔离到32文件研究版，但它删除best-visited保护、改回旧候选网格；历史seed2026恢复失败，Cold同测通过。上传toy测试通过不足以接受该候选，Warm LEMON/模拟/组合性能线全部NOT_RUN。

**选阶：** 216数据完整，含108开发及108冻结后独立确认；均为固定候选库条件下的研究评价。 开发新/原平均PMSE比0.951085，最坏cell比1.566972，门槛未通过；确认新/原平均PMSE比0.921915，最坏cell比1.156737，门槛未通过。 两规则为原alpha1.5/scale1与预声明alpha2/scale0，正式Cold搜索不变；不能把联合规则效应归于alpha单项，也不能把固定库结果叫独立新算法。

**训练内：** 12次重拟合支持CV/稳定性机制可执行，但容量变化已使common MSIC的q80/80降至q1/2，稳定性无一致优势。所有CV分数参与选择，不能据其较低宣布外泛化提高。

**正式版本：** 保留v5.2-search-final，不替换生产或HCP/NKI；v5.3-warm-start-research与v6-selection-research均research_only、release_approved=false。当前没有足够证据自动提出下一正式版。

## 八项回答

1. Warm具体把当前mode支撑传给Group-BESS初始化，还附带移除最佳访问状态/校验/循环及缩减候选网格，不能视为纯Warm。
2. 上传6测试通过，完整31项中11方法失败/错误，原历史恢复未通过，Cold31全过。
3. Warm未进入LEMON，RMSE/R²/q80/稳定性影响均未知；不补造队友数字。
4. Warm新模拟未运行，Exact与PMSE稳定性未知。
5. 既有LEMON大q及弱信号/小样本风险有机制证据，本轮新模拟须以实际完整性和逐cell数据限定结论，逐模caps不可达须分开解释。
6. 虽完成固定双规则独立确认，但至少一个预声明门槛未通过，不能声称已同时解决过选与欠选，也不能仅凭平均PMSE改善批准。 见[选择比较](MSIC_SELECTION_COMPARISON.md)与[方法决策](MSIC_NEW_METHOD_DECISION.md)。
7. Warm与新MSIC组合没有资格执行，交互未知；不能从两个单独现象外推相互促进。
8. 继续用v5.2，保留负面/停止记录；若研究下一候选，另立预声明正确性、开发与独立确认方案，不根据已看LEMON调参。

## 交付

- [Warm代码审计](WARM_START_CODE_AUDIT.md)及[正确性门禁](WARM_START_CORRECTNESS_REPORT.md)
- [LEMON Warm状态](LEMON_WARM_START_COMPARISON.md)、[泛化边界](WARM_START_GENERALIZATION_ASSESSMENT.md)、[模拟状态](WARM_START_SIMULATION_REPORT.md)
- [MSIC机制](MSIC_COMPLEXITY_DIAGNOSIS.md)、[选择比较](MSIC_SELECTION_COMPARISON.md)、[方法决策](MSIC_NEW_METHOD_DECISION.md)
- [团队说明](TEAM_RESEARCH_UPDATE.md)、[复现记录](taskcard8/REPRODUCIBILITY.md)；[交付索引](taskcard8/DELIVERY_INDEX.json)。

保护原源码/旧模拟/旧实证/incoming及既有用户launcher改动；本轮不提交、不发消息、不生成新算法ZIP。
