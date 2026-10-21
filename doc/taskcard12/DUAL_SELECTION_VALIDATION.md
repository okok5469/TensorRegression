# Task12 双规则正确性与历史回归

2026-10-20，**P0 selection 工程门禁通过**。最终研究 wrapper 为 `research/taskcard12/selection.py`，SHA256 `761062933e2a52e407298d7806c5bd31c52aabbbf743bd0b10a745dbf9ede4b3`。原正式 v5.2 与全部旧任务证据只读；此结论不等于 FS 的科学推广门槛通过。

汇总入口：[selection_final_validation.json](verification/selection_final_validation.json)。本范围12项新增 wrapper/历史回归测试、7项原冻结评分测试、16项原 Cold 保护均通过，共35项不同测试；不把此前重复运行或任务卡11的63项累加为新独立测试。

## 十项门禁的实际证据

| 门禁 | 实际验收 |
|---|---|
| original 与正式 v5.2 一致 | 单次真实 native 返回与 original 的 B/support/RSS/score 精确相等；正常、零响应、重复特征、n1/n2真实 tiny通过；102历史原选择全部恢复 |
| fs 与任务卡11一致 | 直接复用冻结 `rescore.select_candidates`；102历史 FS完整选择字典及选中 B均一致 |
| 原 Cold 保护 | 本轮最终源码环境实际运行原5+10+1=16项，全部通过，未运行 Warm9 |
| 完整同池系数/RSS | 一次 native被动捕获原 adaptive终态及完整整数网格；双臂只读取同一B/RSS，16,406候选身份全部匹配 |
| 固定FS数值 | 原7项评分回归覆盖附加项、同q相同增量、非法/非有限候选、零RSS tiny截断 |
| 独立容忍带/精确并列 | 原7项含两规则 argmin q改变后分别重算band的反例，以及全键相同保持原顺序；102全字典一致 |
| q/支撑/预测/保存模型一致 | 204选中模型逐B比较，1,122产物哈希复核；已保存测试预测重算PMSE，完整旧测试重放按complete hash引用 |
| 双/单/缓存一致 | 真实tiny实际调用pair、默认单选original、单选fs、save/load；缓存全文件/数组/身份检查并重核训练OLS |
| 固定输入重复确定性 | 同固定tiny重复pair输出B/完整选择字典精确一致 |
| 独立源码/环境身份 | 全部32冻结核心文件及实际imports核验；相对源码SHA、真实运行环境及缓存身份绑定，错源/错环境/错输入/错配置/损坏拒绝 |

## 三个源码阶段的证据必须分开

| 阶段 | 源码SHA前缀 | 已完成范围 | 耗时 |
|---|---|---|---:|
| 初版包装器 | `e3845917` | Pilot6+formal96，共102已存池；原seed训练流重建、独立全池OLS、严格缓存、选择/预测身份回归 | 471.945秒 |
| 修复零RSS/秩亏 | `6d96c0c2` | 独立新目录再次完整102池、16,406候选；原seed训练流、全池OLS与严格cache load | 550.457秒 |
| 最终四处n1/n2域修订 | `76106293` | 102原封归档池的当前结构校验、纯双重评分、204模型及1,122产物哈希/预测身份链 | 15.871秒 |

前两阶段均为**零搜索、零新增科学重复**；训练流只按原请求和原seed重建。第二阶段完整在原600秒工程窗内退出0，没有续窗或补足缺失case。第二阶段独立重建OLS的最大系数绝对差为0，最大RSS绝对差为 `6.821210263296962e-13`。

最终阶段**没有重生训练/测试流、没有OLS refit、没有搜索**。它调用当前 `verify_pool_files`/`_validate_pool(runtime=False)`，然后直接对归档池重评分；未调用当前 `load_pool`去接纳旧source。旧cache metadata/seal/source均未重写，最终runtime确实拒绝旧source缓存。精确diff证明最终四处修改在全部历史n100/400池上只移除原已满足的限制，native、OLS和评分没有改动。最终源码的真正严格load/pair/single/repeat及边界行为由12项真实tiny测试验证。

主要记录：

- [初版102完整窗口](verification/selection_history/summary.json)。
- [前一源码102完整OLS/cache窗口](verification/selection_history_native_domain/summary.json)。
- [最终源码102归档验证](verification/final_source_saved_pool_regression.json)。
- [最终四处diff](verification/selection_small_n_fix.patch)、[前后hash与12项结果](verification/selection_small_n_fix.json)、[修订前源码原字节副本](verification/selection_before_small_n.py.txt)。
- [最终12项GREEN日志](verification/selection_small_n_green.log)、[原评分7项日志](verification/selection_frozen_rescore_native_domain.log)、[最终源码环境下Cold16结果](verification/original_cold16_final_source.json)。

## 保留失败与修复原因

首版历史工具曾误把旧 `candidate_coefficients.npz` 的键写成 `full_B`；实际旧键为 `B`。真实单例测试失败后局部修复，未启动错误批次、未改旧数组。`selection_history_green.log` 是当时失败尝试的原日志，名字不能替代其中实际ERROR；正确后续为 `selection_history_green_corrected.log`。

首版复用Task11实验审计会拒绝官方合法的零响应和重复特征。真实tiny确认官方返回有限B，wrapper分别因RSS=0及rank不足拒绝；见 [初始事实](verification/selection_native_domain_initial.json)、[RED](verification/selection_native_domain_red.log) 和 [修订GREEN](verification/selection_native_domain_green.log)。新Task12审计接受原最小范数解，报告真实rank、rank_deficient和null条件数，不补正RSS/B、不改变q、不把数值非有限放行。

随后发现n1/n2官方有效而旧研究门禁额外限制n>2、q<n−1。见 [真实原生事实](verification/selection_small_n_domain_initial.json) 与 [RED](verification/selection_small_n_red.log)。最终仅四处域约束与官方对齐；fixed候选仍遵从官方完整网格预算，非法支撑/q、NaN/Inf、错误哈希等继续拒绝。所有失败及先前source证据保留，没有把修复前结果改写为最终source运行。

## 预测证据与科学边界

本轮102历史只回归原保存模型与测试产物。任务卡11既存完整测试流重放按文件hash及逐case complete hash链接；其最大预测误差 `6.661338147750939e-15` 属于**先前独立重放**，不是本轮重新生成测试预测的误差。本轮重算保存预测的PMSE，确认同case、同模型B及相同test身份，不用已见表现修改FS或门槛。

Pilot6与formal96仍分别保存，不合并成102个新正式实验。原Task11强信号欠选门槛失败、n200历史局部恢复风险都保留。Task12新集群及真实数据结果由各自独立程序/协议记录，本报告不宣布其已运行；最终科学版本结论仍为 `PENDING_CLUSTER_RESULTS`，无FS自动发布。
