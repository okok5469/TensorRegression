# 论文主张与证据边界

更新日期：2026-10-29。本表以600套唯一完整输入上的认证比较及保存数组为依据，不以ABESS是否获胜作为验收条件。返回结果已通过独立只读核验；章节联合状态仍由最终图表与G0—G7验收决定，旧比较不改名为新证据。

| 主张 | 当前判定与范围 | 证据 |
|---|---|---|
| 正式算法为v5.2-search-final+原MSIC，未混入FS/Warm等候选 | 已核实32/32发布源及38/38计划源码/配置；结束时仍须复核 | IDENTITY_AUDIT.json；release manifest |
| 原700标签对应600完整输入、10配置 | 已核实；基准150，其余各50；不是500输入，也不是700独立输入 | input_identity_map.csv；IDENTITY_AUDIT.json |
| 已重建所有原训练及测试数组的逐字节副本 | 该整体表述不支持。早期Windows20输入仅为数值对应；本次Linux600套训练输入的四项指纹均与历史逐字节一致，但原测试数组未归档，不能将训练核验层级扩展至测试数组 | RETURN_VALIDATION.json的1200份BYTE_IDENTICAL拟合记录；input_correspondence；INPUT_ID_RECONCILIATION.json |
| 本轮训练输入与旧测试记录可对应 | 全600输入支持：训练指纹一致，真B/噪声/支撑、Oracle训练RSS及按原测试流重放的指标在预定容差内对应；返回审计核验记录与流身份，没有再次生成整批测试数组 | fits/evaluations的input_correspondence；cluster_review/RETURN_VALIDATION.json |
| 正式ABESS在n=400具有高模式恢复率 | 已核实49/50，Wilson95%约[0.8950,0.9965]；PMSE=1.189516 | results/historical_config_summary.csv；已验冻结表图 |
| n=100的失败仅为随机小样本波动 | 应撤回；caps(4,4,4)排除真形状(4,3,5)，真容量可容纳不等于真形状可容纳；Exact0/50、PMSE6.442026 | 正式任务trace/caps；方法章；保存数组 |
| 基准恢复可沿用旧50次汇总 | 本轮主口径应改为150输入：Exact37/150、Wilson95%约[0.1846,0.3214]，PMSE2.045168 | 去重配置摘要；旧14标签附录另保留 |
| Lasso矩形容量就是实际非零元素数 | 应撤回。认证基准150输入中新Lasso实际非零数均值27.21，矩形容量3887.61；旧600输入描述1950.17与19973.33仅作历史。两者均不可互换 | certified_config_summary.csv；support_structure_summary.csv；支撑反例测试 |
| ABESS在11/14标签胜过两种规范参考基线 | 不沿用。旧11/14属于未认证实现和标签权重；本轮使用10配置、600去重输入及完整配对区间，不把重复基准当额外证据 | BASELINE_AUDIT.md；certified_paired_pmse.csv |
| 修复后全部必要基线子问题达到声明精度 | 支持本批：1200有效模型、1200评价、0未解失败；60600份Lasso和905364份Group Lasso证书均通过KKT/gap双条件。不由此推断正式ABESS初始化也达到相同凸精度 | cluster_review/RETURN_VALIDATION.json；逐拟合diagnostics；PILOT_GATE |
| ABESS在基准、小维度、SNR4、rho0.6和n400的预测优势仍存在 | 支持这五个具体配置；对两新对照的PMSE配对差区间均在零以下，不作跨配置IID推断。基准差为−0.8262/−0.9459；新Lasso在若干配置更强，差距较旧实现缩小 | certified_paired_pmse.csv；paper/tables/baseline_update.csv |
| ABESS在所有配置或所有张量方法上预测更好 | 不支持。n100、SNR0.5及1的差区间均在零以上；大维度和rho0的区间跨零，平均领先不足以确认优势；比较对象仅为两个已声明流程 | 主表、配对表；04_simulation.md第4.5节 |
| n400已经证明与Oracle等效 | 不支持等效性结论。ABESS/Oracle PMSE为1.18952/1.18558，差0.003938、区间[−0.003976,0.011853]；Exact49/50只支持本批接近Oracle的描述 | certified_config_summary.csv；certified_paired_pmse.csv |
| 对照充分求解意味着网格或完整张量流程充分调优 | 不支持。顺序GLasso的7420/8964次局部CV选原网格下界（82.78%）；局部设计及预筛仍依赖完整训练过程，不是全流程严格嵌套CV；未按测试成绩扩网格 | RETURN_VALIDATION.json的selected_lambda_boundaries；冻结协议；附录F |
| 更低训练拟合意味着泛化改善 | 不支持；Cold返回完整best-visited的正确性、规模覆盖、恢复与独立预测收益必须分开 | 方法章；历史VERSION_EFFECT_DECOMPOSITION_FINAL |
| v5.2全面改善v5预测 | 应撤回；保留历史配对归因中的n100退化、n400恢复收益及其他成本，不因比较对照修复而重写旧版本研究 | doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md及250配对明细 |
| 预测较好意味着系数与位置恢复更好 | 不支持。rho0.6下ABESS预测低于两新对照，但REE1.2214高于Lasso0.9738及GLasso0.9994；Exact4/50也低于rho0的10/50。PMSE、REE、模式和元素需分别报告 | 认证主表及配对表；04_simulation.md第4.5节 |
| 完整整数规模等于穷举所有位置/全局最优 | 不支持；只补全旧caps内规模，位置仍为有限局部搜索 | 02_method.md；THEORY_INTERFACE.md |
| 支持任意非连续形状、零模型或所有DGP | 未验证；本批真形状始终(4,3,5)、单一高斯生成族且连续真支撑；算法不含空矩形 | 冻结生成器/plan；模拟局限 |
| 已有统计一致性、最优率、任意响应缩放不变性或投稿条件 | 本卡不证明；理论、实证及贡献定位仍是独立问题 | THEORY_INTERFACE.md；CHAPTER_FREEZE状态 |

去重后ABESS总Exact162/600、两个新对照各0/600仅作当前设计权重的描述，不是跨配置IID推断；旧182/700还包含重复标签权重。主要结论按配置及完整输入配对给出，不能由旧排序、训练CV评分、不同方法边际均值区间的重叠或获胜次数替代。理论、实证和整篇论文贡献定位仍不属于本轮两章有效性的通过结论。
