# Pure Warm：固定规模并集原型设计与门禁

身份：研究候选；正式版本仍为 `v5.2-search-final`。P0/P1 稳定后执行轻量门禁，**固定规模原型 25/25 通过；全 adaptive 正确性与性能均为 `NOT_RUN`**。

## 历史失败与适用范围

任务卡8的 incoming 实现不是单因素 Warm：它在固定 BCD 中传入当前 mode 支撑，同时删除 best-visited、有限性和循环保护，并缩减部分完整整数网格。其原样正确性为 31 个顶层方法中 20 个完全通过、11 个失败或错误；这不是 20 个独立失败。原 seed2026 真实恢复案例中，正式 Cold 的 RSS 为 154.4719575096973 且三模式真支撑全部恢复，incoming 的 RSS 为 252.22955568355502 且未恢复。原报告及日志保留，不修改断言、数据或失败记录。

可核验的基线为 `doc/algorithm_versions/v5.2-search-final/src/`。其中 `tensor/bcd.py` 与生产文件字节相同，SHA256 为 `db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`。

## 三个实现选择与本轮边界

| 路径 | 特征 | 本轮决策 |
|---|---|---|
| 逐 mode 用当前支撑替换 Cold 初始化 | 会丢失原 Cold 提案路径，历史恢复已失败 | 不采用 |
| 每 mode 同时计算 Cold/Warm 提案 | 能保留单步 Cold 候选，但下一步输入仍会分叉；不能保证完整原 Cold 路线保留 | 暂不实现 |
| 完整 Cold 路线 + 额外初始支撑的完整路线 | 两条路线均使用未经修改的正式固定 BCD，原 Cold 完整轨迹明确保留 | 采用固定规模研究原型 |

额外来源仅改变第二条路线的初始支撑。两条路线中的 Group-BESS/Splicing、完整矩形 OLS、best-visited、有限性和循环保护全部调用冻结正式实现。它属于“额外 Warm 起点并集”原型，不是 incoming 的逐 mode warm-only。当前不接入 adaptive 外层、候选网格或 MSIC；因此本轮不声称全流程 Pure Warm 已通过门禁。

## 固定规模接口与确定性

接口接收训练 X/y、目标三元规模、显式额外支撑来源及原 Cold 参数。原 Cold 原样执行，其结果与额外路线结果分开保存，不用额外路线覆盖 Cold。

额外支撑必须是三模式合法、唯一的整数位置；非法来源直接报错。目标规模不一致时，仅使用训练边际分数：过长时按分数递减、位置递增裁剪；过短时保留合法原项，并按同一排序从剩余位置补齐。最终每模式按位置递增排列。相同训练数据和参数产生相同补齐、裁剪结果；不读取真支撑或测试标签。

对两个返回状态独立构造完整矩形 OLS 设计，校验支撑、系数、矩形外零值、预测、loss/RSS 与 best-visited 一致性后，按重新计算的训练 RSS 选择；完全并列优先 Cold。非法或非有限候选不静默退回 Cold。固定规模最小训练 RSS 不劣于被保留的 Cold 是并集性质，不能推出最终 MSIC 或测试 PMSE 不劣。

## 预声明门禁与成本

1. 原 `tests/test_tensor_bcd.py` 全部原函数、`tests/test_bcd_fix_regression.py` 全部原方法，以及 seed2026 原历史方法，以原断言和原数据执行；仅在内存移除未使用的 `import pytest`。
2. 新接口测试固定规模合法性、独立 OLS 一致性、全量 Cold 结果/访问轨迹保留、best-visited、循环停止、固定种子重复性、额外支撑确定性补齐/裁剪、非法输入和非有限状态拒绝。
3. 新行为先记录失败测试，再实现并记录通过结果。原测试文件不改。
4. 分别记录 Cold、额外支撑准备与额外路线/校验、合并总耗时。时间仅为本机门禁执行成本，不作速度或泛化优越性声明。
5. 若任何正确性失败，保存失败并停止性能研究，不运行 Warm 与新 MSIC 组合。

## 执行状态

实现位于 [pure_warm.py](../../research/taskcard9/pure_warm.py)，测试与运行入口位于 [test_pure_warm.py](../../research/taskcard9/test_pure_warm.py)。没有复制或改动正式算法；两条固定路线直接调用冻结正式 `tensor_bcd`。当前容器保留两条路线的完整原始结果对象、`visited_loss_history`、接受历史、best 索引和停止原因；它不承诺调用者返回后无法主动修改 Python 对象。

| 检查 | 结果 |
|---|---|
| TDD red | 新行为尚不存在时，9 个测试全部按预期断言失败，0 error；见 [red.log](warm/red.log) |
| 原 Cold 历史 | 原 toy/group 文件 5 个、固定 BCD 回归 10 个、原 seed2026 方法 1 个，16/16 通过；原断言与数据不变 |
| 新并集契约 | 9/9 通过；覆盖更好/更差/并列额外起点、训练分数裁剪补齐、非法/非有限输入、独立 OLS、重复性、循环/best 和 seed2026 保留 |
| green 总计 | 25/25 顶层检查，0 failure、0 error，用时 1.788 秒；见 [green.log](warm/green.log) |
| 新输出路径复现入口 | 25/25 再通过，1.712 秒；为核对新增 `--output` 和身份记录而重复，不能算成 50 个独立正确性检查 |
| 身份 | 运行前后冻结 32 核心哈希不变，已加载 `tensor_abess` 模块均来自冻结路径；版本、线程、实际 imports 与研究/原测试哈希见 [复现检查元数据](warm/repro_entry_check/green.json) |

### seed2026：保留 Cold 的实际意义

输入 X/y 指纹分别为 `b223ece8f266fe0259404f3bc0981056f3b798ed2734236c67f73770878c6479` 和 `db06202dfb5992cc0d33e0f96168354c3876e882090377ac601ef446949d42f3`，与任务卡8历史证据一致。额外起点完全由各 mode 的训练边际分数第 2 至 s+1 名产生，不用真支撑构造起点。

原 Cold 仍返回真支撑、RSS=154.4719575096973；额外路线的 RSS=277.86940011713943，且未恢复真支撑。并集因此保留并选择原 Cold。这个案例说明完整保留 Cold 能避免较差额外起点替换原恢复结果；它没有提供 Warm 搜索或预测优越性的证据。额外路线存在暂时变差的探索状态，但仍返回自己最优已访问状态，并在重复 sweep 时停止。

本机该次门禁的 Cold 时间为 0.192 秒，Warm 额外成本为 0.312 秒，合并总时间为 0.504 秒；额外成本包括准备、额外路线、独立 OLS 校验和少量调度。单次计时随运行波动，只报告门禁开销，不是公平性能基准；[完整轨迹和时间分项](warm/seed2026.json)全部保留。

### 通过范围与未运行项

机器决策见 [gate_summary.json](warm/gate_summary.json)：`fixed_scope_correctness_passed=true`、`full_adaptive_correctness=NOT_RUN`、`full_pipeline_performance_allowed=false`。通用旧适配器在原 `green.json` 写出的 `performance_allowed=true` 仅限这个固定规模原型已通过正确性，不能解释为全 adaptive 或整条算法获准做性能研究。

完整 adaptive 接入、全整数网格逐规模 Cold 保留验证、全局 MSIC 选择、Warm 泛化性能、LEMON OOF、Warm × 新 MSIC 和大规模模拟均为 `NOT_RUN`。正式网格、caps 和 MSIC 源码没有改变；尚不能从“源码未改”推出研究原型的全流程集成已经验证。固定规模并集最小训练 RSS 不高于 Cold，也不能推出全局 MSIC 选择或测试 PMSE 不恶化。

建议保留此固定规模原型作为后续独立研究工具，不替换正式版本，也不将它列为当前预测退化的已验证修复。

## 可复现入口

使用项目既有解释器和单线程配置，输出必须是新目录，已有日志或 seed 证据会在拟合前被拒绝覆盖：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -B research/taskcard9/test_pure_warm.py --phase green --output doc/taskcard9/warm/new_reproduction
```

原 red 在实现前运行，原始失败日志只作历史证据保留；实现完成后再执行 red 不应期望重现“缺模块”状态。本轮仅使用既有 Python/NumPy/SciPy，无新依赖、集群提交或测试预测计算。

## 历史依据

- [任务卡8 Warm 正确性](../WARM_START_CORRECTNESS_REPORT.md)
- [任务卡8原样代码审计](../WARM_START_CODE_AUDIT.md)
- [seed2026历史证据](../taskcard8/warm_audit/historical_recovery_evidence.json)
