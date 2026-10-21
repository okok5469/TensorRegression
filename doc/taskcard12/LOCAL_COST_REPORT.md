# Task12 本地成本与可行性检查

预先保留的 n200/SNR4/(4,3,5)/rho0.6 单例已完成：126 个原生候选、一次 Cold 搜索、零额外搜索；两种规则共用输入、候选和独立测试流，全部候选最小范数 OLS 审计通过。case 墙钟为 45.679 秒，包含生成、搜索、审计、保存和测试流；控制器总墙钟 47.046 秒，未触及预定 600 秒上限。

确切请求与执行前源码、环境在 [LOCAL_COST_PROTOCOL.json](LOCAL_COST_PROTOCOL.json)，实际产物在 [complete.json](verification/local_cost/case/complete.json)，控制器记录在 [LOCAL_COST_EXECUTION.json](verification/LOCAL_COST_EXECUTION.json)。双规则模型、指标、预测和指纹全部保留，未按本例预测表现修改公式或判据。

本例仅为 Windows 本地成本和正确性检查，不计入 380 例正式验证，不替代 Linux 集群的三例真实 smoke、环境及成本 Gate，不能据此估计 n200 恢复率或形成 FS 推广结论。集群用时须按操作指南在实际 myenv 中测量。
