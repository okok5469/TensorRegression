# 历史交接审计：任务卡1—12与算法版本

整理日期：2026-10-26。范围为 `work_progress/` 全部14份 Markdown、任务卡1—12的任务定义、对应最终报告/验收和根算法说明；这里只整理既有证据，没有重新拟合、运行模拟、修改源码或覆盖历史结果。数字沿用各自最终报告的口径，不将不同任务、复用数据、候选重评分和测试次数相加。

**当前结论：正式算法仍是 `v5.2-search-final + 原 MSIC`。** 任务卡5完成正式发布，卡11、12再次决定保留；Warm、训练内CV、更强MSIC和FS均未进入正式默认。历史中“待运行”“待批准”“进行中”必须放回当时的时间点阅读，不能据此恢复工作。

## 一、最容易交接错误的四个状态

| 易误读的旧记录 | 截至本次交接的准确状态 | 当前依据 |
|---|---|---|
| 卡6的250归因任务在Windows身份核验失败、尚待集群 | 后续卡7已完成250/250原集群归因、750个三版本模型及5个Gate验收；不再待补跑 | [最终归因](VERSION_EFFECT_DECOMPOSITION_FINAL.md)、[验收](taskcard7_completion/FINAL_VALIDATION.json) |
| 卡10根报告及根算法说明写开发30/48、900秒超时 | 这是首阶段快照；用户授权追加后，**开发48/48完成**（原30按字节复用＋18补齐），5项开发门槛失败，独立确认0/48按条件未启动 | [追加最终报告](taskcard10/continuation_01/TASKCARD10_FINAL_DECISION.md)、[追加验收](taskcard10/continuation_01/FINAL_VALIDATION.json) |
| 卡12实证子报告仍有 `PENDING_CLUSTER_RESULTS` | 该文字指当时尚待模拟回传的**全局算法建议**；380正式模拟＋3 smoke后来已验收、最终建议B。**LEMON15折已完成；HCP-YA/NKI-RS仍NOT_RUN、输入NOT_PROVIDED**，不能随模拟验收一并改为完成 | [当前根建议](taskcard12/FINAL_VERSION_RECOMMENDATION.md)、[回传审阅](taskcard12/cluster_review/FINAL_VERSION_RECOMMENDATION.md)、[实证子报告](taskcard12/EMPIRICAL_DUAL_MSIC_REPORT.md) |
| 根算法说明将北京列为“有限定的主实证候选” | 算法与数值说明仍可用；这一应用定位已被用户后续指令及修订汇报撤回，现有实证不能宣布已证明方法优势的主实验 | [修订近期工作报告](../Presentation/recent_work_20261024/近期工作报告.md) |

上述差别是历史状态与后续证据的关系，不改写旧报告以消除曾经的失败、中止或未运行事实。

## 二、任务卡1—12实际总账

“已完成”指明确的实验阶段或按门禁结束的决策，不表示原任务卡提出的所有分支都执行成功。

| 卡 | 要解决的问题与实际工作 | 当前完成范围及主要结果 | 负结果、未运行部分与正式落点 | 成果入口 |
|---|---|---|---|---|
| [1](../任务卡/任务卡1.md) | 检查EEG是否超出年龄/性别提供预测信息；内折独立拟合预处理，加入仅协变量、Ridge | 145人，3次5折，5方法，共75个外层拟合、2175条OOF。原v5 ABESS RMSE 0.371667，协变量0.317485；四个EEG方法均未稳定改善协变量 | 无可靠EEG增量；异常折q80推动后续诊断。此卡评价层修正，没有改核心；旧单次5折较好排名不能替代此重复检验 | [完整报告](../empirical_analysis/lemon/outputs_eeg_incremental/reports/EEG_INCREMENTAL_REPORT.md)、[执行验收](../empirical_analysis/lemon/outputs_eeg_incremental/reports/EXECUTION_SUMMARY.md) |
| [2](../任务卡/任务卡2.md) | fixed BCD途中RSS更好却返回较差末态；修复完整最佳状态保存，初步研究MSIC | 形成v5.1；27配对中Exact均3/27，PMSE 3.061981→3.364051；27库选阶探索与限定回归完成 | Warm-only草案未过历史恢复回归，被保留为失败版本；正确性修复不保证预测改善。旧“v5.1全700仅预检1”不等于后来做过全700 v5.1，后续三版本归因为250 | [BCD修复](BCD_FIX_REPORT.md)、[统计影响](BCD_FIX_STATISTICAL_IMPACT.md)、[MSIC消融](MSIC_ABLATION_REPORT.md) |
| [3](../任务卡/任务卡3.md) | 分离caps、整数规模覆盖、具体位置搜索及最终评分 | 历史27＋开发90＋独立确认90完整；A0/A2确认Exact 6→26/90，PMSE 3.118404→3.082733；A3评分开发/确认亦完成 | 扩caps增加过选和成本，多起点无稳健恢复收益；研究版A0—A4未直接发布。旧进度“确认尚未执行”已过时；错误验证器的STOPPED结果不混入最终样本 | [比较](CANDIDATE_SEARCH_COMPARISON.md)、[版本决策](ALGORITHM_VERSION_DECISION.md)、[位置稳定性](SUPPORT_SEARCH_STABILITY.md) |
| [4](../任务卡/任务卡4.md) | 对最小改动A2增加独立形状/场景验证，再决定是否改MSIC | A2新增100配对：主90 Exact 15→36，PMSE 2.500011→2.474219；压力10结果相同；200所选模型OLS及16冷启动检查完成。推荐D2 | 新MSIC确认80计划在3600秒预算停止：当时68完整，最终保留71落盘，但迟到3不计预算内完成；fresh refit 0/3。**确认不完整，不是证明候选无效**；按协议保留原MSIC | [A2独立验证](A2_INDEPENDENT_VALIDATION.md)、[A2决策](A2_VS_A0_DECISION.md)、[MSIC停止决策](A2_MSIC_DECISION.md) |
| [5](../任务卡/任务卡5.md) | 用户批准D2，建立正式发布、清理与集群/实证交付身份 | 发布v5.2-search-final，32个核心Python文件；相对v5.1有31个文件未变，仅bcd网格逻辑变化。全700后来已回传验收 | 当时本机单任务预检不等于700正式完成；正式身份由release清单确立，不以旧研究候选VERSION或Git HEAD替代 | [发布报告](V52_RELEASE_REPORT.md)、[发布清单](releases/v5.2-search-final/RELEASE_MANIFEST.json)、[700验收](V52_FULL_SIMULATION_REVIEW.md) |
| [6](../任务卡/任务卡6.md) | 分析700模拟、重跑LEMON、准备两名队友算法包 | 700任务标签、2800方法行；v5→v5.2 Exact 138→182，PMSE 2.236761→2.381274。LEMON75拟合完成，ABESS RMSE 0.371667→0.504581 | 模拟总体预测恶化、LEMON恶化均保留。250归因当时身份阻塞，卡7补齐；交付包字节一致不等于已经向队友发送、也不等于HCP/NKI已验收 | [执行报告](TASKCARD6_EXECUTION_REPORT.md)、[模拟诊断](SIMULATION_ERROR_ATTRIBUTION.md)、[LEMON比较](LEMON_V5_V52_COMPARISON.md)、[团队交付](TEAM_V52_DELIVERY.md) |
| [7](../任务卡/任务卡7.md) | 严格拆分BCD修复与补网格的影响 | 250集群归因完成；LEMON补15个v5.1外折，与v5.2系数/支撑及435预测全部相同；480同形候选RSS中295降低、185相同、0升高 | n100退化发生在BCD段；n400补网格Exact 1→49/50。训练内45拟合诊断保留真实折容量，q80在该较小内训练容量不合法，不能照搬早期q80诊断规则。不是全700补齐v5.1 | [最终归因](VERSION_EFFECT_DECOMPOSITION_FINAL.md)、[250验收报告](V51_FULL_ATTRIBUTION_REPORT.md)、[LEMON三版本](LEMON_THREE_VERSION_COMPARISON.md)、[训练内诊断](LEMON_TRAIN_ONLY_SELECTION_DIAGNOSIS.md) |
| [8](../任务卡/任务卡8.md) | 独立审查收到的Warm；比较新选阶 | 原样Warm正确性失败；新MSIC共享Cold候选，开发108＋独立复核108完整；另12次训练内选阶可行性拟合 | Warm性能线及组合未运行，不能说“实测没有性能收益”。新alpha2/scale0复核PMSE降约7.81%，但最坏场景比及强信号欠选门槛失败；不发布v6 | [Warm最终决定](WARM_START_FINAL_DECISION.md)、[MSIC比较](MSIC_SELECTION_COMPARISON.md)、[阶段决定](TENSOR_ABESS_NEXT_VERSION_DECISION.md) |
| [9](../任务卡/任务卡9.md) | 复用历史池，解耦惩罚与容忍带，分析同规模支撑风险 | 250归因细分、216库四规则864行重评分、预定12例测试流重放；Pure Warm固定规模原型25/25门禁通过 | A1/A2全216的PMSE不齐，保留NA，不用不同子集排序；同形状异支撑不能靠q惩罚分辨。Pure Warm保留Cold、仅固定规模原型，全adaptive及泛化未验；不发布 | [最终决定](taskcard9/TASKCARD9_FINAL_DECISION.md)、[Pure Warm](taskcard9/PURE_WARM_DESIGN_AND_GATE.md)、[候选差距](taskcard9/CANDIDATE_SELECTION_GAP.md) |
| [10](../任务卡/任务卡10.md) | 原生MSIC、匹配容量MSIC、同容量CV三臂，避免把容量变化全算成CV收益 | 首阶段30/48超时；用户授权追加后**开发48/48完成**（30复用＋18补齐）。n100三臂PMSE 4.679961/3.231389/3.128412；n400 Exact 13/13/11（各24） | 全体同库CV均值仅降0.85%，最坏cell比2.836、n400恢复和强信号欠选等共5/9门槛失败；确认0/48按门禁未启动。暂定B、仅研究；不是仍卡在预算，也不是CV普遍无效 | [追加最终报告](taskcard10/continuation_01/TASKCARD10_FINAL_DECISION.md)、[追加验收](taskcard10/continuation_01/FINAL_VALIDATION.json)、[保留的旧停止报告](taskcard10/TASKCARD10_FINAL_DECISION.md) |
| [11](../任务卡/任务卡11.md) | 仅验证一个有限样本修正FS，决定收官 | 96全新配对完整，PMSE 2.978516→2.176566（−26.92%），Exact 31→33；n400 Exact保持30/48 | 强信号q欠选增量4/32=0.125超0.05门槛，联合采用失败。FS是经验公式，未有普适最优性证明；原MSIC冻结，不继续调参数 | [最终决定](taskcard11/FINAL_ALGORITHM_DECISION.md)、[正式冻结](taskcard11/FINAL_ALGORITHM_FREEZE.md)、[验收](taskcard11/FINAL_VALIDATION.json) |
| [12](../任务卡/任务卡12.md) | 按新授权扩展同一FS公式，重点核查n200恢复风险及实证 | 集群380正式＋3 smoke核验完整：主360＋额外压力20；压力分组30含主样本10。主PMSE 2.571710→2.066497（−19.65%）。LEMON15折双规则完成 | 11门槛中4项失败；压力30 Exact 12→8，原成功12有5丢失。LEMON FS RMSE 0.359627仍差于协变量0.317485。HCP/NKI未提供输入，NOT_RUN。最终B：原MSIC默认、FS仅冻结研究对照 | [当前最终建议](taskcard12/FINAL_VERSION_RECOMMENDATION.md)、[集群详审](taskcard12/cluster_review/FINAL_VERSION_RECOMMENDATION.md)、[双规则实证](taskcard12/EMPIRICAL_DUAL_MSIC_REPORT.md)、[当前验收](taskcard12/FINAL_VALIDATION.json) |

## 三、算法版本到底差在哪里

| 版本 | 实际变化 | 收益与边界 |
|---|---|---|
| v5-original | 历史Cold固定规模搜索、旧稀疏锚点网格、原MSIC | 作为历史对照保留；固定规模可能返回非最佳末态，不能因其某些测试误差较低而恢复已知缺陷 |
| v5.1-bcd-fix | 分开Cold探索状态和最佳完整返回状态，保存同一次拟合的B/支撑/RSS，加有限性及循环检查 | 修复返回一致性与漏掉已访问好模型；探索轨迹可暂时变差，不是每一步强制下降，不保证全局最优或泛化提高 |
| v5.2-search-final / D2 | 在v5.1上，原caps和容量内补全最终精修的全部合法正整数规模 | 修复规模漏搜；不扩大caps、不穷举位置、不改初始化/Group-BESS/OLS/MSIC。当前正式默认 |
| 新MSIC、FS、CV、Warm研究 | 在隔离实现中研究选阶或初始化 | 各有门槛失败或验证范围不足，没有正式替代；失败版本与结果仍是研究证据 |

最终MSIC早已存在于v5：`n log(RSS/n) + 1.5 q ℓ_n + ℓ_n Σ_m log C(p_m,s_m)`，`ℓ_n=log(max(log(n),e))`；原容忍带按argmin的q计算，并非交叉验证的一倍标准误。模式内/adaptive评分alpha=1，不能与最终精修alpha=1.5混写。数学及实现细节见[根算法说明](../Tensor-ABESS_v5.2算法全流程说明.md)，其历史状态两处更正见本报告第一节。

三版本最清楚的收益/代价证据来自同输入250配对；每行50例，PMSE越小越好：

| 场景 | v5 | v5.1 | v5.2 | Exact：v5→v5.1→v5.2 |
|---|---:|---:|---:|---|
| Baseline | 1.990669 | 2.012130 | 2.012130 | 15→16→16 |
| n100 | 5.196560 | 6.442026 | 6.442026 | 0→0→0 |
| n400 | 1.297588 | 1.299304 | 1.189516 | 1→1→49 |
| SNR0.5 | 1.837097 | 1.926368 | 1.926368 | 0→0→0 |
| rho0 | 2.573943 | 2.948160 | 2.948160 | 12→10→10 |

来源：[逐配对CSV](../numerical_simulation/outputs_v51_attribution/cluster_run/paired_three_versions.csv)。250个完整输入唯一，但只有150个种子族；不能合成250个IID观测。700任务标签的完整主模拟有600个唯一输入，去重后Exact 120→162/600、PMSE 2.274498→2.438721；不能只展示恢复收益而隐藏预测代价。

## 四、14份work_progress怎样正确使用

以下逐文件列出历史作用及后来状态。过程文件中尚未执行的建议仍是建议，除非有后续具体结果支撑；不将“计划做”写成“已做”。

| 历史文件 | 当时记录的核心内容 | 现在如何阅读 |
|---|---|---|
| [2026-07-20_progress](../work_progress/2026-07-20_progress.md) | 按数学方案实现sacrifice、Group-BESS、BCD、SIC及接口 | 早期实现起点；旧目录/参数不作为当前发布或正式实验配置 |
| [2026-07-24_progress](../work_progress/2026-07-24_progress.md) | 模拟框架、固定支持规模交换越界修复、quick结果 | quick少量重复不是正式收益证据；后续版本和实验已替代早期计划 |
| [2026-08-02_progress](../work_progress/2026-08-02_progress.md) | 长耗时主要来自初始化CV/ISTA；增量输出、续跑及接口兼容 | 当时降低重复数和后续运行建议属于当时阶段；不据此推断正式全量未完成 |
| [2026-08-14_progress](../work_progress/2026-08-14_progress.md) | v2全560行完成、adaptive严重选q1；修初始化回退、增known-s路径 | 诊断known-s只知道数量，非自动选阶成功；v3在后续记录完成 |
| [2026-08-19_progress](../work_progress/2026-08-19_progress.md) | v3完成，改MSIC与refinement，准备v4 | v4在8月25日已完成；不把当时“待跑v4”当现状 |
| [2026-08-25_progress](../work_progress/2026-08-25_progress.md) | v4过选，最终MSIC alpha1.5及确定性小模型容忍带形成，准备v5 | 证明原v5已有这套MSIC；后续alpha1.7等设想不能当已采用默认 |
| [2026-09-17_LEMON_empirical_progress](../work_progress/2026-09-17_LEMON_empirical_progress.md) | 145人EEG构建/QC、调整协变量、旧单次5折ABESS RMSE 0.32641750 | 数据构建有来源价值；算法排名被卡1更严格重复增量检验限定，旧单次结果不是当前主表 |
| [2026-10-04_advisor_experiments_progress](../work_progress/2026-10-04_advisor_experiments_progress.md) | 5种系数设计×3流程×10次，共150模型；符号、幅值、SNR诊断 | 全正/未归一化同时改变结构或SNR，不能据此宣称原算法困难已解决；后续弱切片等仍需明确实跑证据 |
| [2026-10-04_known_s_cluster_and_baselines](../work_progress/2026-10-04_known_s_cluster_and_baselines.md) | 从集群入口交付、部署修正，到后附10月7日700任务/2100方法结果验收 | 文件前半“本地仅task0”被后半完整回传补齐。fixed Exact283/700、adaptive138/700，但fixed PMSE3.0453高于adaptive2.2368；知道数量不等于知道位置。新增基线文献推荐不等于代码已实现/规范复现 |
| [2026-10-10_BCD_fix_MSIC_taskcard2](../work_progress/2026-10-10_BCD_fix_MSIC_taskcard2.md) | 卡2修复及27配对、选阶消融，正式full当时仅预检 | 完整700 v5.2与250 v5.1归因后来完成；没有证据把v5.1范围扩成700。旧Warm草案拒绝不等于所有Warm路线无效 |
| [2026-10-10_candidate_search_taskcard3](../work_progress/2026-10-10_candidate_search_taskcard3.md) | 卡3持续追加日志，最后仍停在确认进行中的文字 | 对应最终版本决策已确认90/90完成；应读最终报告，不用旧运行会话/PID判断当前状态 |
| [2026-10-10_LEMON_EEG_incremental_value](../work_progress/2026-10-10_LEMON_EEG_incremental_value.md) | 卡1严格3×5增量检验、75模型、无EEG正增量及q80异常 | 实际已完成；其中ABESS属v5，后续v5.2更差，不能直接给旧行改标签 |
| [2026-10-10_LEMON_q80_diagnostic](../work_progress/2026-10-10_LEMON_q80_diagnostic.md) | 单异常外折内诊断：OLS正常、q80验证差、fixed轨迹循环且漏best | 解释卡2动机；单折训练内诊断不是新独立实证；卡7训练内协议重新执行更小内折caps，不能混为同一实验 |
| [2026-10-17_eight_taskcards_stage_summary](../work_progress/2026-10-17_eight_taskcards_stage_summary.md) | 卡1—8收尾，700及250已经完成，Warm/新MSIC不发布 | 是前八卡较可靠总索引，但不覆盖后来卡9—17；“下一阶段CV”已由卡10及其追加执行处理，不是当前待办 |

早期known-s文献审阅还记录：仓库现有CP多分量更新、Tucker矩估计实现不能直接当相关论文的规范复现；Vector Lasso模态投影诱导矩形不等于其实际非零元素集合。该记录仅为既有局限，不授权本轮修代码或新增基线。

## 五、编号、未运行事项与停止边界

任务卡目录实际为1—13及15—17，共16张，**没有任务卡14**。Apple暂停后进行过未编号的[Mushroom CSV低成本检查](mushroom_screen/MUSHROOM_SCREEN_REPORT.md)，不能据顺序补称“任务卡14”。卡13、15、16、17的详细当前实证状态由对应实证交接审计负责；本文件不将卡1—12数字与这些数据混算。

已经关闭或已完成的事项：700正式v5.2模拟、250三版本归因、卡3独立确认、卡10完整开发及门禁决定、卡12集群回传；不能因旧文档待办重开。卡4不完整确认按原停止规则保留；卡8Warm依赖性能线未启动；卡10确认因开发门槛失败未启动；卡12 HCP/NKI输入未提供。这些未运行阶段不能填上推断成绩，也不自动形成当前授权。

对版本决策应分别说清：正确性失败、预算中止、开发门槛失败、完整确认未通过、仅输入未提供是不同状态。保留原MSIC不等于FS无预测收益；没有采用Warm不等于已做完整性能比较；下游包或报告存在不等于远端真实实验完成。

本次交接不执行旧建议、不安装依赖、不提交远程作业、不联系任何外部人员。UCI270另按用户最新决定放弃并匿名留档，不因本历史整理恢复追查。
