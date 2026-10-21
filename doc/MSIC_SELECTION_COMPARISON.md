# 预声明MSIC候选的开发与独立确认比较

任务发起日期：2026-10-16（北京时间）；实际UTC起止以冻结协议及控制器记录为准。

216数据完整，含108开发及108冻结后独立确认；均为固定候选库条件下的研究评价。

## 设计及选择门禁

用户批准36场景×每阶段3重复，共108开发+108独立确认、全线7200秒截止。新seed分属1806101600与2806101600两段，按cell×1000+replicate，test另加50000000；432种训练/测试seed不重复，与已审计11份旧顶层计划的916个材料化seed无交集，不对未公开外部seed作无依据保证。

沿正式700的实际k_max1/max_cycles6等配置，而非dataclass通用默认；dims32×20×25、n_test500、两连续矩形支撑、rho0/.6。4个独立worker各单线程，完整32源码、真实import、数据/库/测试流指纹和预算均锁定；生产源码无修改。

预声明开发候选仅alpha2/scale0。门槛同时要求整体PMSE比≤.98、最坏cell比≤1.15、SNR4 Mode-F1下降≤.02、SNR4 q欠选率增加≤.05、n100近容量率不增，不能只按平均PMSE挑规则。只有完整108开发后冻结lock及其artifact SHA，才执行未见确认；开发未过时的双规则确认仅为固定规则研究复核，不能反称开发合格。

## 实际结果

| 阶段 | 规则 | 完成数据 | PMSE | Exact计数（率） | Mode-F1 | q欠选率 | q过选率 | 近容量率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | baseline | 108 | 2.397573 | 35/108 (0.3241) | 0.7142 | 0.3981 | 0.2778 | 0.0741 |
| development | alpha2_scale0 | 108 | 2.280295 | 35/108 (0.3241) | 0.7031 | 0.4537 | 0.2222 | 0.0741 |
| confirmation | baseline | 108 | 2.443688 | 36/108 (0.3333) | 0.6737 | 0.4259 | 0.2407 | 0.0648 |
| confirmation | alpha2_scale0 | 108 | 2.252872 | 34/108 (0.3148) | 0.6656 | 0.4907 | 0.1759 | 0.0463 |

| 阶段 | 门槛通过 | 总体PMSE比 | 最坏cell比 | SNR4 ΔMode-F1 | SNR4 Δq欠选率 | n100 Δ近容量率 |
| --- | --- | --- | --- | --- | --- | --- |
| development | False | 0.951085 | 1.566972 | -0.004027 | 0.000000 | 0.000000 |
| confirmation | False | 0.921915 | 1.156737 | 0.001382 | 0.055556 | -0.055556 |

最差场景按预声明的cell均值比计算，不以单个最差重复替代；各阶段最差cell如下。

| 阶段 | 最坏完整cell | n | SNR | 真形状 | rho | 新/原PMSE | 原/新PMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | 11 | 100 | 4.0 | [2, 3, 4] | [0.6, 0.6, 0.6] | 1.566972 | 3.126531/4.899184 |
| confirmation | 21 | 200 | 4.0 | [4, 3, 5] | [0.6, 0.6, 0.6] | 1.156737 | 1.886629/2.182334 |

两个阶段各36cell、每cell3次重复均齐全；完整逐cell结果见[场景表](taskcard8/analysis/scenario_comparison.csv)，完整所选指标及[逐任务配对](taskcard8/analysis/paired_differences.csv)均保留。

主指标之外同时保存signal MSE、REE、Element-F1、q/Q、RSS、Exact和成本。欠选/过选按矩形q与真q比较，不等同于元素FN/FP；近容量定义q≥.9 floor(.7n)。两规则共享一次候选库拟合，fit_sec_shared不能重复相加或当作两个独立算法的速度比。

每cell只有3个seed，门槛是预声明研究判据，不是已完成普适性或功效证明；图中展示全配对数据，不把跨情景离散度当确认置信区间。fit_sec_shared含被动记录/系数捕获开销，且四worker并行，不可当未经观测器的生产原生速度或与旧Linux集群直接比较。

[全配对预测差异图](taskcard8/figures/paired_msic_prediction.pdf)分开显示开发与确认，全部实际观测点均保留；PDF/SVG/TIFF/PNG及字号、对齐、碰撞检查见[图形验收](taskcard8/figures/QA_REPORT.json)。

开发合格状态为 `False`；本轮始终 `research_only=true, release_approved=false`。历史alpha2/scale0的A3研究、旧A2预算停止确认及已经看过的LEMON都不计作本轮独立确认。[复现记录](taskcard8/REPRODUCIBILITY.md)；[交付索引](taskcard8/DELIVERY_INDEX.json)。
