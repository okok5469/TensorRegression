# 论文主张与证据边界

本表按可核验证据判定，不以ABESS是否获胜作为验收条件；新基线全量Gate未过时，历史比较不能改名为本轮主结果。

| 主张 | 当前判定与范围 | 证据 |
|---|---|---|
| 正式算法为v5.2-search-final+原MSIC，未混入FS/Warm等候选 | 已核实32/32发布源及38/38计划源码/配置；结束时仍须复核 | IDENTITY_AUDIT.json；release manifest |
| 原700标签对应600完整输入、10配置 | 已核实；基准150，其余各50；不是500输入，也不是700独立输入 | input_identity_map.csv；IDENTITY_AUDIT.json |
| 本机重建了原输入的逐字节副本 | 不支持该表述；20个pilot真B字节一致，但其他指纹经常不同 | pilot_input_audit_v2.json；INPUT_ID_RECONCILIATION.json |
| 本机与历史输入建立了数值对应 | pilot20支持：保存B/噪声/真支撑，以及旧Oracle训练RSS与独立测试预测按预定1e-10容差对应；其他输入须在各补算任务入口继续核验 | 输入audit；新任务input_correspondence |
| 正式ABESS在n=400具有高模式恢复率 | 已核实49/50，Wilson95%约[0.8950,0.9965]；PMSE=1.189516 | results/historical_config_summary.csv；已验冻结表图 |
| n=100的失败仅为随机小样本波动 | 应撤回；caps(4,4,4)排除真形状(4,3,5)，真容量可容纳不等于真形状可容纳；Exact0/50、PMSE6.442026 | 正式任务trace/caps；方法章；保存数组 |
| 基准恢复可沿用旧50次汇总 | 本轮主口径应改为150输入：Exact37/150、Wilson95%约[0.1846,0.3214]，PMSE2.045168 | 去重配置摘要；旧14标签附录另保留 |
| Lasso矩形容量就是实际非零元素数 | 应撤回；旧结果600输入描述均值实际非零1950.17、矩形容量19973.33，二者不可互换；这不是新可靠基线数值 | results/historical_unique_metrics.csv；支撑反例测试 |
| ABESS在11/14标签胜过两种规范参考基线 | 当前无本轮有效证据；11/14只适用于已知数值/CV局限的旧实现，暂不进入新比较结论 | BASELINE_AUDIT.md；待certified_*结果 |
| 修复后所有基线已充分求解 | 尚不能由小合成测试或单个pilot推断；全部影响选择的凸拟合必须同时满足KKT与gap，缺项阻塞 | 逐拟合diagnostics；PILOT_GATE；独立accept入口 |
| 更低训练拟合意味着泛化改善 | 不支持；Cold返回完整best-visited的正确性、规模覆盖、恢复与独立预测收益必须分开 | 方法章；历史VERSION_EFFECT_DECOMPOSITION_FINAL |
| v5.2全面改善v5预测 | 应撤回；保留历史配对归因中的n100退化、n400恢复收益及其他成本，不因比较对照修复而重写旧版本研究 | doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md及250配对明细 |
| 预测较好意味着系数与位置恢复更好 | 不支持一般化；PMSE/REE/模式与元素分别报告，强相关下尤其不能互相替代 | 独立指标与配对表；模拟章 |
| 完整整数规模等于穷举所有位置/全局最优 | 不支持；只补全旧caps内规模，位置仍为有限局部搜索 | 02_method.md；THEORY_INTERFACE.md |
| 支持任意非连续形状、零模型或所有DGP | 未验证；本批真形状始终(4,3,5)、单一高斯生成族且连续真支撑；算法不含空矩形 | 冻结生成器/plan；模拟局限 |
| 已有统计一致性、最优率、任意响应缩放不变性或投稿条件 | 本卡不证明；理论、实证及贡献定位仍是独立问题 | THEORY_INTERFACE.md；CHAPTER_FREEZE状态 |

去重后ABESS总Exact162/600仅作当前设计权重的描述，不是跨配置IID推断；旧182/700还包含重复标签权重。任何新比较中的优势保留、缩小、消失或反转，须由同一规范输入键上的certified配对结果更新本表，不能由旧排序或训练CV评分判断。
