# A2与A0搜索方案决策（任务卡4）

## 推荐与批准边界

推荐D2搜索方案：v5.1固定BCD修复+旧caps+完整合法整数规模grid+当前原MSIC。100个新增独立数据、200个配对模型、16次预定冷启动、全部预声明门禁已通过；这构成待审推荐，正式发布仍待用户最终批准。最小候选v5.2-search-final保持research_only=true、release_approved=false，当前生产src仍为v5.1。Phase C决策已完成：80组新MSIC确认触发预算停止、证据不完整，按冻结规则保留原MSIC，不推荐D3；不以部分结果补救或批准。

## 问题1：A2是否值得采用，为什么？

在预声明验证边界和成本容忍下，值得作为默认候选提交批准。主90 Exact 15/90→36/90（+23.33个百分点），Mode-F1 0.697061→0.711099、Element-F1 0.548250→0.591204；PMSE 2.5000107245→2.4742185167（−1.0317%），A2/A0比0.989683，配对场景分层bootstrap 95%区间[0.987544,0.991690]。主要价值是减少旧grid遗漏真实规模导致的恢复损失，预测改善较小。

全部预声明门禁PASS：最坏场景平均PMSE比1.000000≤1.25，最大场景候选数比1.675497≤3；独立成本估计比1.148793、冷启动比1.103680均≤1.5。没有按本轮结果调整门槛。若另外要求额外成本≤10%，本轮两种成本定义均不能满足，A0仍符合更严格成本选择。

## 问题2：是否适用于不同形状、样本量、SNR？

两种新增形状总体均改善：(2,3,4) Exact9/45→20/45，PMSE比0.991648、95%区间[0.988736,0.994101]；(3,4,5) Exact6/45→16/45，PMSE比0.988482、95%区间[0.985331,0.991238]。

| 真形状 | n100，SNR.5/2/4 Exact（各5次，A0→A2） | n200，SNR.5/2/4 | n400，SNR.5/2/4 |
|---|---|---|---|
| (2,3,4) | 0→0；0→0；3→3 | 0→0；2→2；4→4 | 0→1；0→5；0→5 |
| (3,4,5) | 0→0；0→0；0→0 | 0→0；2→2；4→4 | 0→0；0→5；0→5 |

新增grid/模型变化只在n400（29/30次B改变），n100/200旧grid已完整，主60次结果相同。n400强SNR2/4的5/5只代表本轮观察，不是普遍恢复保证；弱SNR仍失败/欠选。n100真(3,4,5)容量合法60≤70，却被第三mode cap4排除，两种grid均不可达。

rho=.6只检验n200/SNR2两形状各5次：总体Exact同3/10、PMSE同1.757277629、支持/预测逐值相同；此时两grid完全一致，仅有兼容性证据，未验证n400新增grid在强相关下的鲁棒性。结论不覆盖所有位置、维度、非矩形信号或实证DGP。

## 问题3：是否增加过拟合及成本？

主90过选46/90→25/90，欠选同27/90，平均q_hat40.211111→37.011111；q=1同1/90，q≥0.9Q同9/90。训练MSE0.776646→0.787803、测试PMSE略降，本轮未显示明显过拟合增加；n100的弱识别/负R²问题仍保留。“过选”只按q判断，不等于所有位置假阳性。

真规模caps允许同75/90，真规模grid/pool覆盖45/90→75/90，真支持pool覆盖21/90→47/90，原MSIC最终选中15/90→36/90。完整规模枚举不等于位置穷举，也不保证进入pool的真支持都被选择。弱SNR下的(3,4,5)真支持pool4/5而Exact0/5、平均q18.4<60，选阶仍需独立研究。

平均候选模型数（含adaptive）114→148；n400151→253，n10065→65，n200126→126。独立成本估计157.931762→181.430908秒/主数据（+14.8793%）；预定8数据的独立空缓存solver时间合计379.584598→418.939974秒（+10.3680%）。A2共享缓存walltime128.237467秒不能与A0的157.931762秒直接当作独立速度优势。冷测16次cache_hit=0，B/支持和训练/测试指纹与主实验一致；计时不含进程启动、DGP和测试评价。

预定PMSE容忍0/2/5/10%点估计均PASS；成本比阈值1.1时估计/冷测均FAIL，1.25/1.5/2均PASS。D2推荐遵循事前1.5门槛，不因本轮结果降低标准。单机成本测量不等于集群实测，每场景5重复也不足以声称普遍保证。

## 后续问题与证据入口

问题4（原MSIC是否修改）见已完成决策的[A2_MSIC_DECISION.md](A2_MSIC_DECISION.md)：保留原alpha1.5/gamma1/scale1，不把开发或不完整确认用作新规则批准。问题5（最终版本/具体改变）及6（限定测试/未验证范围）见[FINAL_ALGORITHM_VERSION.md](FINAL_ALGORITHM_VERSION.md)；问题7（队友同步与完整模拟/实证重跑）见[FINAL_TEAM_SYNC.md](FINAL_TEAM_SYNC.md)与[FULL_EXPERIMENT_PREPARATION.md](FULL_EXPERIMENT_PREPARATION.md)。批准前不替换生产，不提交正式计算，不重标旧B/支持/选阶结果。

完整主90/压力10、两形状和20场景、F1/训练测试误差、覆盖、全部门禁及配对95%区间见[A2_INDEPENDENT_VALIDATION.md](A2_INDEPENDENT_VALIDATION.md)。bootstrap只在固定场景内重采样，未包含外推场景变化；退化区间不意味着总体无不确定性。

## 审计及候选身份

200个选中原MSIC模型OLS秩亏0、最大条件数40.430603；不声称全部候选矩阵也做了同样全量秩审计。本轮106项研究unittest、29项最终候选pytest、16项MSIC决策/确认unittest及6项候选静态/代表桥接，共157项限定测试通过，非全仓库测试；test_phase_c因可选sklearn缺失未收集、baseline源码未改，无集群实跑。桥接核验actual import身份及B/支持/caps/RSS/score/轨迹逐值相同，仅是指定来源和参数的证据。

[候选VERSION](algorithm_versions/v5.2-search-final/VERSION.json)从v5.1复制32源码，31个原字节不变，唯一bcd.py私有规模函数改为1..cap容量过滤；fixed/adaptive、caps/init、Group-BESS/OLS/MSIC、公共签名/默认值保留。空合法域返回[]不伪造(1,1,1)，无作用private window删除；实验2000预算不加入公共接口，大样本需预检。待审[patch/full脚本](team_sync/v5.2-search-final/README.md)已做普通LF/CRLF副本apply、Bash静态/mock组件核验；未应用到当前或队友生产，未执行700正式模拟或三实证重训，release_approved仍false。

冻结protocol的任务卡显示字段为“???/???4.md”，原字节保留；[metadata注记](../numerical_simulation/outputs_a2_final_validation/PROTOCOL_METADATA_NOTE.json)仅给实际来源/哈希，不改变科学设置/seed/结果/门禁。protocol原始SHA256为`2300285beca35adf92436e3774c0bbd156d981ac56dcb87bfc6eaba1c665cc00`，plan原始SHA256为`7926df93cef2d425d3c36b271759759d251ba9d573bd0bfa7d9f17666bd892bb`，plan_id为`55cd1a254efe2425d650e755950388efa2ff8746aa0ebd83f9cfffdb50cf7cd3`；后两者定义不同。判据来自冻结protocol及[final analysis.json](../numerical_simulation/outputs_a2_final_validation/analysis.json)，本报告未改写。
