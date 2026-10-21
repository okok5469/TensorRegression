# Tensor-ABESS 论文终版算法冻结说明

**任务卡11决策：继续采用已经正式发布的 `v5.2-search-final` 与原 MSIC。** 本卡不发布新算法，不修改现有正式源码；唯一研究候选 MSIC-FS 因强信号 q 欠选率增量未通过预声明门槛而不采用。算法调整阶段结束，进入真实数据适配性审查与论文撰写。

## 当前正式身份和历史记录

任务卡5已经批准并完成本地正式发布，当前身份以 [RELEASE_MANIFEST.json](../releases/v5.2-search-final/RELEASE_MANIFEST.json) 的 `approved_by_user=true`、`release_status=released` 和 [V52_RELEASE_REPORT.md](../V52_RELEASE_REPORT.md) 为准。正式源码位于 `src/tensor_abess/`，共 32 个 Python 文件。

[历史 VERSION.json](../algorithm_versions/v5.2-search-final/VERSION.json) 和 [任务卡4冻结建议](../FINAL_ALGORITHM_VERSION.md) 保留的是正式发布前的候选状态，其中 `research_only=true`、`release_approved=false` 不应替代后续任务卡5正式发布记录。本卡结果和研究实现使用 `release_approved=false` 表示 **MSIC-FS 没有新发布**，不表示 v5.2 尚未获批或需要再次批准。

## v5→v5.1→v5.2 必要改动及依据

| 阶段 | 必要变更/保留 | 工程或统计依据 | 解释边界 |
|---|---|---|---|
| v5-original | 原算法及结果保留为历史 | 发现固定规模 BCD 可能返回非最佳已访问完整状态 | 某些旧版预测较好不能作为恢复已知正确性缺陷的理由 |
| v5.1-bcd-fix | 保留 Cold 确定性探索、完整状态 best-visited、一致的 B/support/RSS，以及有限性和循环保护 | 修复最佳状态保存和返回契约，历史恢复/回归验证保留 | 更低训练目标不保证更好泛化；历史 n100 在更充分搜索后存在过选与预测恶化 |
| v5.2-search-final | 在原 caps 与 q 容量内补全最终精修的合法整数规模网格，保留 v5.1 修复和原 MSIC | 关键历史 n400/(4,3,5) 场景 Exact 1/50→49/50，平均 ΔPMSE=−0.109788；后续任务卡5正式发布 | 完整规模网格不是支撑位置穷举，也不扩张模式 caps；不能保证所有 DGP 或实证表现 |
| 任务卡11 MSIC-FS | 仅离线/隔离研究增加 `ell_n*q²/(2n)`，正式版不采用 | 新 96 对 n100 PMSE 降低33.78%、总体降低26.92%、n400 Exact 30/48不变；强信号欠选增量0.125超过0.05 | 有预测收益但联合门槛失败；不将失败误写为预测无效或普遍恢复破坏，不事后换分母 |

历史因果和分母说明见 [任务卡9最终报告](../taskcard9/TASKCARD9_FINAL_DECISION.md)；正式发布证据见上述任务卡5文件。旧研究推荐继续 CV/Warm/调参的段落仅保留其历史语境，不作为本卡结束后自动启动新算法路线的授权。

## 冻结的完整算法组成

正式流水线继续为：边际预筛、Group Lasso 初始化、Group-BESS/Splicing、BCD、原 caps 内完整整数规模 Cold 搜索、原最终 MSIC、小模型优先容忍带、矩形支撑内 OLS。具体冻结边界如下：

- 初始化规则、预筛、Group Lasso、BCD 更新顺序、停止机制及 Cold 路线保持。
- 保留 best-visited 完整状态、有限性、循环与一致性保护；不回退至已知缺陷版本。
- 保留逐模 caps 和 `q<=max(1,floor(0.7*n_train))` 容量；不因真形状不可表示而重分配或扩大。
- 保留完整合法整数规模网格及原候选顺序，包括原 adaptive 终态；不改为旧启发式网格，不增加 Warm 路线。
- 保留完整矩形支撑 OLS；不以 Ridge、收缩估计或后处理替换。
- 模式内 MSIC 保持原实现；最终全局 `alpha=1.5`、`gamma=1`，原容忍带 `scale=1`。
- 原最终评分为 `n*log(max(RSS/n,tiny)) + 1.5*q*ell_n + ell_n*sum(log_choose(p_m,s_m))`，其中 `ell_n=log(max(log(n),e))`。
- 使用首个分数 argmin 的 q_best 构造原带宽，按 `(q,score,s1+s2+s3)` 选带内最小候选，完全并列保留输入顺序；不添加新数值容差或以测试指标参与选择。
- 公共接口、默认参数、估计器、数据流和既有调用保持；研究重评分器不导入正式算法路径。

本卡没有对正式 `src/`、旧版本、旧模拟、旧实证、原发布与交接材料做覆盖或重标记。没有创建分支/worktree、提交混合更改或自动对外发布。

## 完整源码指纹与证据入口

完整 32 文件的 raw SHA256 与跨系统 canonical-LF SHA256 已存在正式 [RELEASE_MANIFEST.json](../releases/v5.2-search-final/RELEASE_MANIFEST.json) 和 [SOURCE_SHA256.txt](../releases/v5.2-search-final/SOURCE_SHA256.txt)，不以版本名或 Git HEAD 代替逐文件身份。历史 D2 快照在 `doc/algorithm_versions/v5.2-search-final/src/`。

本卡事前 [PROTECTED_BASELINE.json](PROTECTED_BASELINE.json) 保存旧文件完整清单及字节指纹；[FORMAL_PROTOCOL.json](FORMAL_PROTOCOL.json) 的 `core_sha256` 保存本次实际采用的全部 32 核心源码指纹，`source_identity` 绑定科学实现、评价与报告代码。最终7,874个旧文件逐一SHA均不变，tracked Git状态与事前相同；精确范围见 [FINAL_VALIDATION.json](FINAL_VALIDATION.json)。

| 关键身份 | SHA256 |
|---|---|
| 正式 `tensor/bcd.py` raw | `db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7` |
| 正式 `tensor/sic.py` raw | `84637e426a8ad8d9c471d8f81787135052f10f954475127403d5a9b2b4f51d9f` |
| 任务卡11事前保护清单 | `af5e2cb3cf3deb63623db5a870fda199dc9c65edb38e26574ef13734d2f7d2e2` |
| 任务卡11正式验证协议 | `6b898c6584718c83edefef5bc76a9c6cb516ae802541addd4a713a82571aeeb1` |
| 正式96请求 plan | `dd08228e7f2db3b99d48deeecb5fc3adf2dcffa1c2a6d541a35023cb76b09d7b` |

算法冻结不依赖一次新的 Git commit；旧未提交改动及历史发布记录继续受保护。不能将候选的不同换行指纹误认为已发生算法变更，也不能仅凭规范化哈希掩盖实际内容差异。

## 本卡证据如何保存与引用

历史 216 库与 31,968 候选的 A0 精确恢复在 [MSIC_RESCORING_AUDIT.md](MSIC_RESCORING_AUDIT.md)；它属于已见历史诊断，不能混入独立正式验证。新正式 96 对、15,264 候选、全部选择与外测数据独立保存于 `results/formal/`；Pilot 6 例位于独立 Pilot 目录，不计作正式重复。

[FINAL_ALGORITHM_DECISION.md](FINAL_ALGORITHM_DECISION.md) 给出 Q1—Q7、七项门槛及强信号失败分层；[MSIC_FINAL_COMPARISON.md](MSIC_FINAL_COMPARISON.md) 和 [SELECTION_CHANGE_ANALYSIS.md](SELECTION_CHANGE_ANALYSIS.md) 对应完整机器表格。`analysis/` 下保留192选中行、96配对、全部完整场景、变化分解、形状/q分布、真支撑入池证据及bootstrap区间；不删除5个预测坏例、4个强信号新增欠选或历史n200真支撑丢失。

每个正式 case 的外测前独立候选/OLS审计已经保存；最终47项新增测试与16项原Cold历史检查共63项通过。正式96/96和Pilot6/6的原seed独立重放均通过，共16,406候选、1,122产物哈希；最大预测误差6.66×10⁻¹⁵、RSS误差6.82×10⁻¹³、归一化OLS梯度4.45×10⁻¹⁵，完整记录由 [FINAL_VALIDATION.json](FINAL_VALIDATION.json) 汇总。重放仅核验既有科学输入身份和保存预测，零搜索、零新增独立模拟重复；限定通过不能表述为全仓库、远端集群或HCP/NKI实证已验证。

## 终止条件与后续工作

本卡按预声明联合门槛作出结果 B，不因预测收益明显而更改阈值，不缩小强信号分母，不再次选择 FS 系数、alpha、gamma 或带宽，也不自动发起任务卡12继续 MSIC 修复。正式算法保留的是有明确结构假设与适用边界的 v5.2，不承诺在所有数据集排名第一。

下一阶段以 [DATASET_FIT_AUDIT_PLAN.md](DATASET_FIT_AUDIT_PLAN.md) 为入口，先审查连续响应的科学意义、天然三阶结构、近似矩形稀疏依据、有效独立样本、超越简单协变量的预测增量、泄漏与批次风险、共同外部验证和负面结果可报告性，再决定数据集是否适配。该转向不自动授权下载、联系队友、运行新实证或修改算法；论文如描述 FS，应明确为单一预声明、未获最终采用的经验候选，完整报告收益与失败门槛。
