# 队友 Warm BCD 原样代码审计

日期：2026-10-16。状态：Phase A 已完成；正确性门禁失败，停止该方案 Phase B、C 及 E 性能实验；Phase D 独立进行。研究身份为 `research_only=true`、`release_approved=false`。

## 身份和隔离

真正的算法文件已收到，不能将测试文件当作算法。以正式 `v5.2-search-final/src` 的 32 个 Python 核心文件逐字节复制到 `doc/algorithm_versions/v5.3-warm-start-research/src`，只以 incoming 原字节替换 `tensor_abess/tensor/bcd.py`，没有修补队友算法、生产源码、原测试或历史结果。

| 文件 | SHA256 |
|---|---|
| incoming/bcd.py | `fdc61b4eb62dd1e4efc2f4d76b609da94f401085ffa3fae6c805395a9ee02fb2` |
| incoming/test_tensor_bcd.py | `84fa8af10b906cfd5d2df5ffbe67434c3a51cf155dc93f07168f16c5a5ffaeba` |
| 正式 v5.2 bcd.py／当前生产 bcd.py | `db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7` |
| 旧 warm-only-rejected bcd.py | `77c359823f956c275f0b4a54599c3878cb21e1f7f62473b6671debd4c0b4a054` |

完整 32 文件基线／研究指纹见研究目录 `VERSION.json`；每个独立测试进程记录全部 32 文件哈希，并在测试前后检查已加载的 `tensor_abess` 模块均来自指定研究或基线目录。算法与原测试输入指纹在三次运行间相同。

## 逐函数差异

以下为与正式 v5.2 的 AST 级对比；原始行号与结果另存 `doc/taskcard8/warm_audit/static_audit.json`。

| 类／函数 | 差异和影响 |
|---|---|
| `TensorBcdResult` | 删除 `best_iteration`、`best_rss`、`stop_reason`、`visited_loss_history`、`accepted_support_history` 五个已有字段；fixed 返回结构不再兼容现有诊断消费者。 |
| `TensorBcdAdaptiveResult` | 完全相同。 |
| `_candidate_size_triples` | 完整整数网格改成每维 near-cap `window=3` 加 `{1, round(.4 cap), round(.6 cap)}` 锚点；新增空集时返回 `(1,1,1)` 的逻辑。不是纯暖启动：候选覆盖也被改变。 |
| `_marginal_supports_for_size` | 完全相同，初始边际评分及支持未改。 |
| `_support_size` | 完全相同。 |
| `_msic_one_se_tolerance` | 完全相同，one-SE 公式未改。 |
| `_refine_msic_global_candidates` | 函数主体完全相同；但调用的网格和 fixed BCD 都改变，因此实际精修候选及其 RSS 会改变。 |
| `tensor_bcd` | 签名相同；每个 mode 新传 `groups_init=S1/S2/S3`，以当前 mode 支撑初始化 Group-BESS。删除所有外层支撑／有限性校验、每 mode 完整 OLS 候选评估、独立 best-visited 记录及重复 sweep 检测，改为每 sweep 末重拟合并无条件返回最后状态。 |
| `tensor_bcd_adaptive` | 签名与函数主体完全相同；其原有 `groups_warm=S1/S2/S3` 并非本次新增。通过 fixed 精修阶段间接受影响。 |

例如 caps=(8,4,3)、q≤70 时，正式 A2 有 92 个合法规模，incoming 只留 68 个，漏掉 `(2,1,1)` 等 24 个；完整列表见 static_audit.json。这改变候选选择机制，即使 MSIC 参数完全相同，也不能将整体预测差异归因于暖启动。

上下文限定：精确提取两份原网格函数执行，LEMON 的 n116/caps(4,4,5)/Q81 下有序规模均为80且相同；n100/caps(4,4,4)的64及n200/caps(5,5,5)的125也相同，而n400/caps(6,6,7)为正式252、incoming150。因此不能把网格混杂自动当作LEMON变化的解释；它对更大caps的模拟会影响覆盖。该检查不拟合模型，见[grid_context.json](taskcard8/warm_audit/grid_context.json)。

## 搜索、接口和终止机制

暖启动只发生在 fixed BCD 三个 mode 的 `group_bess_fix(groups_init=当前支撑)`；不传系数暖启动，也没有保留单独的 cold 候选或确定性探索重启。Group-BESS 内部交换、损失函数、OLS、边际初始化、MSIC 公式、one-SE 公式、随机种子接口、`max_cycles/m_max/k_max` 默认预算均来自原基线，未新增随机性；实际轨迹和停止时机改变。

正式 v5.2 将探索状态和正式接受最优状态分开：完整矩形 OLS 每候选检查，允许确定性探索暂时变差，返回全部有效访问状态中的最优模型，并在完整 sweep 重复时结束。incoming 没有这种分离，只有一个支撑状态；每 sweep 中有任意 mode 变化即继续，到完全无变化或预算耗尽才停。普通真实 warm 子问题可能仍产生单调轨迹，但外层已不具备正式 best-visited、验证和循环契约；不能从某个 toy 单调测试通过推出这些契约仍成立。

fixed/adaptive 调用签名保持一致，adaptive 返回类也一致；fixed 的五个字段删除属于真实返回接口破坏。新增上传测试相比原 `tests/test_tensor_bcd.py`，只增加 seed=16 的单调性案例；没有检查旧 seed=2026 恢复、探索后返最优、有限性、循环或网格覆盖。

## 与旧否决草案复核

旧 `v5.1-warm-only-rejected/STATUS.json` 明确记载“original historical marginal recovery regression failed”，且原 MSIC 研究已经中止。其做法是 warm-only 加每 mode 单调拒绝，减少冷初始化可到达的支撑路径；它保存 best 及诊断字段，而 incoming 连这些保护也移除了。二者都没有恢复原 cold 探索机会。

本轮严格重跑同一历史 seed=2026（n=200、dims=(32,20,25)、s=(4,3,5)、原 `max_cycles=6/k_max=1`），二者返回完全相同的错误支撑和 RSS=252.22955568355502；正式 v5.2 恢复全部真支撑，RSS=154.4719575096973。当前 incoming 没有克服旧否决问题，不能因队友其他数据的小幅 RMSE 改善接受它。

本轮只验证收到的混合方案并揭示 confounds，不擅自拆出新算法后评性能。纯暖启动因素若后续研究，必须明确保持 A2 网格和正式最佳状态／验证契约，并单独授权预声明新候选；当前身份不能用来表示已经完成纯因素验证。

## 结论与证据

已开始并完成原样正确性验证，失败后按任务卡停止 Warm 性能研究；没有运行 LEMON 15 折、新配对 Warm 模拟或 E 交互实验，也没有发布新版本。测试细节见 [WARM_START_CORRECTNESS_REPORT.md](WARM_START_CORRECTNESS_REPORT.md)，可机器读取的决策见 [status.json](taskcard8/warm_audit/status.json)。独立 MSIC 研究可继续，正式 v5.2 身份不变。
