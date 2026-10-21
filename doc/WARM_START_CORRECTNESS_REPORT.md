# Warm BCD 正确性验证

日期：2026-10-16。决策：`correctness_passed=false`，`warm_performance_allowed=false`。保留失败和无效导入顺序取证，未修队友代码或改原断言／数据。

## 环境、测试方式和覆盖

运行解释器为项目既有 `.venv/Scripts/python.exe`，Python 3.12.8、NumPy 1.26.4、SciPy 1.16.1；单进程测试显式设置 OpenBLAS/OMP/MKL 线程数为 1，禁写 bytecode。本环境没有 pytest 和 sklearn；本轮算法及所选历史正确性测试不需要 sklearn，未安装／升级任何依赖。

采用标准库 unittest。原和上传两个 pytest 风格文件均无 fixture、mark、参数化或 pytest API 使用，仅有未使用的 `import pytest`；适配器仅在内存 AST 中去掉该未使用导入，逐个调用原测试函数，保留所有函数体、断言和数据。实际使用 pytest API 的文件会被适配器明确拒绝。本报告不声称运行过完整 pytest、全仓库测试或集群测试。

每版本 31 个顶层测试包括：原 fixed toy 5 个、上传 6 个、未修改的 `tests/test_bcd_fix_regression.py` 10 个、原 `tests/test_numerical_simulation.py` seed=2026 恢复 1 个、补充正确性 5 个，以及适配器自身 4 个。补充检查为非法支撑、fixed/adaptive 签名、adaptive SIC OLS/RSS、A2 网格和同种子数值输出。四个工具测试在实现前出现缺模块／缺汇总功能失败，再通过；未把待审算法改成可通过的版本。

| 版本 | 顶层测试 | 完全通过的方法 | 失败／错误的方法 | assertion failure 事件 | error 事件 |
|---|---:|---:|---:|---:|---:|
| 正式 v5.2 | 31 | 31 | 0 | 0 | 0 |
| incoming Warm | 31 | 20 | 11 | 9 | 11 |
| 旧 warm-only-rejected | 31 | 28 | 3 | 3 | 0 |

事件数包含 unittest subtest：incoming 的 11 个 error 中，8 个来自同一个 noisy tensor 测试的 8 个种子缺 `visited_loss_history`；非法支撑一个方法内还产生两个 failure 和一个 error。不能将 9+11 相加当作 20 个独立测试失败。31 中算法检查为 27、工具检查为 4。

## 门禁失败

| 契约 | 原样 Warm 观察 |
|---|---|
| 正式接受损失非增 | 原保留的非单调探索注入案例中 `loss_history` 上升，原断言失败；上传与原 toy 单调测试均通过，不能替代该契约。 |
| 返回最优有效访问状态 | 原正交 OLS fixture 中应返回 loss=0.875，却返回 1.8125。 |
| 变差候选不覆盖 best | 原 fixture 初始 loss=0，Warm 最后返回 0.0625。 |
| 重复状态／循环 | 等 RSS 支撑在两状态间循环，跑满 12 sweep，原 `n_cycles<12` 断言失败。 |
| 有限性 | 原 overflow fixture、`max_cycles=0` 未抛 `FloatingPointError`，失败。 |
| 合法唯一支撑 | `[-1]` 和 `[0,0]` 未拒绝；上界越界 `[2]` 泄漏 `IndexError`，而正式接口提前抛 `ValueError`。 |
| 返回诊断字段兼容 | `best_iteration`、`visited_loss_history` 等被删，原消费者直接 AttributeError。 |
| A2 候选网格 | 92 个合法候选只留 68 个，故不能看作保持原选阶机制。 |
| 历史真支撑恢复 | seed=2026 第一维真支撑断言失败，且错误支撑与旧被否决 Warm 草案相同。 |

原固定规模恢复 toy、支撑大小正常案例、退化向量案例、真实穷举参考、adaptive SIC OLS/RSS、调用签名和同种子数值输出通过。原 noisy 8 个种子的后续 OLS 一致性断言被缺字段错误阻断，不能把该部分写成通过。原 best 访问 fixture 用明确合法支撑提案及真实完整 OLS 验证外层契约，不宣称这些注入的提案均会由真实 Group-BESS 自动产生；历史 seed=2026 是实际无注入求解器的回归失败。

## 历史 seed=2026 实际恢复

原数据及测试保持不变：X.shape=(200,32,20,25)，s=(4,3,5)，原 full profile `k_max=1`、`max_cycles=6`。同一 X/y 在三个独立进程的数组字节 SHA256 完全相同。

初始边际支撑为 `[3,26,29,30] / [3,5,15] / [0,1,8,15,19]`；真支撑为 `[24,25,26,27] / [3,4,5] / [0,1,2,3,4]`。

| 版本 | 最终支撑 | RSS | sweep | Exact |
|---|---|---:|---:|---|
| 正式 v5.2 | `[24,25,26,27] / [3,4,5] / [0,1,2,3,4]` | 154.4719575096973 | 4 | true |
| incoming Warm | `[24,26,29,30] / [3,5,15] / [0,1,8,10,15]` | 252.22955568355502 | 2 | false |
| 旧 rejected Warm | 与 incoming 完全相同 | 252.22955568355502 | 2 | false |

这只是预先存在的正确性恢复 fixture，不是新种子预测性能研究，也不是对全部非凸支撑的全局最优保证。

## 复现和身份检查

在仓库根目录 PowerShell 设置线程数后，分别执行下列命令。Warm 和旧版退出码为 1，正式版为 0；失败是研究结论，不能为通过而调整输入或算法。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
./.venv/Scripts/python.exe -B scripts/taskcard8/warm_audit.py --source doc/algorithm_versions/v5.3-warm-start-research/src --output doc/taskcard8/warm_audit/incoming
./.venv/Scripts/python.exe -B scripts/taskcard8/warm_audit.py --source doc/algorithm_versions/v5.2-search-final/src --output doc/taskcard8/warm_audit/baseline
./.venv/Scripts/python.exe -B scripts/taskcard8/warm_audit.py --source doc/algorithm_versions/v5.1-warm-only-rejected/src --output doc/taskcard8/warm_audit/old_rejected
```

各 `unittest.log` 为完整原断言 traceback，`correctness.json` 保存事件、解释器、全部 32 核心哈希、实际已加载模块路径及原测试／incoming 哈希。研究 `VERSION.json` 与运行指纹一致，三版本原输入哈希一致；正式生产源码没有修改。

必须先导入并断言目标 `tensor_abess.tensor.bcd`，然后才加载历史模拟测试：`numerical_simulation/__init__.py` 会把生产 src 插入 `sys.path` 首位。主审计程序从第一轮起即先锁定正确 BCD，测试前后均验证模块路径。另一次额外数值取证因先导入 numerical_simulation 而全加载了生产 BCD，已保留为 `historical_recovery_INVALID_IMPORT_ORDER.json`，明确 `valid_for_comparison=false`，禁止作为版本比较证据；正确顺序重跑结果在 `historical_recovery_evidence.json`。该问题未通过修改历史模块或动态更换已加载算法补救。

## 阶段决策

Phase A 已完成并否决当前原样方案进入性能比较。Phase B 的 LEMON 15 折、Phase C 新配对模拟和 Phase E 的 Warm 交互均为 `NOT_RUN_CORRECTNESS_GATE_FAILED`，无 Warm RMSE/R²、q80 或模拟 PMSE 改善结论。独立 Phase D 允许按任务卡继续；正式版本保持 v5.2。本轮不联系队友、不替换队友代码、不创建分支或发布版本。
