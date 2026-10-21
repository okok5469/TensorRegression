# v5.1完整250任务归因接收报告

2026-10-17（北京时间）。指定五场景各50，250/250已回传并通过只读验收；本轮新增拟合0、未重新merge，原计划、失败记录及原始结果保留。

## 完整性验收

| 项目 | 结果 |
| --- | --- |
| 指定任务JSON / NPZ | 250 / 250，恰好覆盖原任务清单 |
| Pilot Gate | 5，绑定对应结果seal |
| 原生plan重建 | 精确一致；seal `01d8005e7c3683b2fe69af4c728f9d798562943a1c1177a3e068718ae329c0cb` |
| 三版源码 / 共享依赖 | 96源码及6共享依赖、冻结worker匹配 |
| 环境 / 四项训练指纹 | 所有250封存记录与原Linux计划一致 |
| 三版本NPZ / CSV | 750 NPZ原字节SHA通过；已有CSV250行每个字段与原merge逻辑一致 |
| 独立模型指标 | 750所选模型合法支撑、REE/F1/Exact/MSIC及loss恒等式通过 |
| 本轮重新拟合 / merge | 0 / 0 |

独立数值误差：支撑指标≤2.23e−16、REE≤4.45e−16、MSIC≤5.69e−14，RSS/(2n)与保存loss恒等误差0。v5.1/v5.2非n400的200个系数/支撑精确相同，n4001/50相同。详[身份审计](taskcard7_completion/cluster_audit.json)与[系数指标审计](taskcard7_completion/estimate_audit.json)。

## 指标与成本

| 场景（各50配对） | v5 PMSE | v5.1 PMSE | v5.2 PMSE | ΔBCD | ΔGrid | Exact：v5→v5.1→v5.2 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.990669 | 2.012130 | 2.012130 | +0.021461 | +0.000000 | 15→16→16 |
| n100 | 5.196560 | 6.442026 | 6.442026 | +1.245466 | +0.000000 | 0→0→0 |
| n400 | 1.297588 | 1.299304 | 1.189516 | +0.001716 | -0.109788 | 1→1→49 |
| SNR0.5 | 1.837097 | 1.926368 | 1.926368 | +0.089270 | +0.000000 | 0→0→0 |
| rho0 | 2.573943 | 2.948160 | 2.948160 | +0.374218 | +0.000000 | 12→10→10 |

完整[250任务配对表](taskcard7_completion/paired_effects.csv)、[15个版本/场景全指标](taskcard7_completion/version_metrics.csv)、[逐指标配对变化](taskcard7_completion/effect_summary.csv)与[极值个案](taskcard7_completion/extreme_cases.csv)均保留。 表中包含PMSE、signal MSE、REE、Exact、Mode-F1、Element-F1、q、q/Q、RSS、MSIC和运行秒数；极值个案列出具体task/重复、三版本误差/q/RSS，不删除恶化记录。

| 场景 | v5秒 | v5.1秒 | v5.2秒 |
| --- | --- | --- | --- |
| Baseline | 21.695 | 21.774 | 25.004 |
| n100 | 5.408 | 5.749 | 5.971 |
| n400 | 77.871 | 66.321 | 110.860 |
| SNR0.5 | 20.761 | 20.928 | 23.927 |
| rho0 | 21.003 | 21.689 | 24.992 |

保存计时来自不同运行/作业，未控制同硬件或随机化测时，仅作描述；相同网格和B的场景仍有计时差，不能将其直接解释为网格因果开销。候选数150→252是n400的结构计算量变化，非严格速度基准。

## 原环境恢复与证据限制

原Windows失败preflight继续保存在旧目录：当时X/y/noise不匹配、B_true匹配，fit_executed=false。回传新目录的成功preflight及250任务记录均匹配原Linux CPython3.9.25/NumPy2.0.2/SciPy1.13.1/OpenBLAS0.3.27和四线程变量1，说明原身份门禁已满足；没有放宽Gate或改历史plan。

本轮没有归档的原X/y或逐测试流数组，未在Windows独立重算RSS、PMSE、signal MSE或R²；这些指标来自通过封存身份核验的原记录。测试配对依据原生成器/配置/test_seed和原Linux环境记录，不能称逐数组测试流指纹已重建。 源码与seal建立内部一致性，不是外部签名；原worker检查包入口路径，未保存每模块实际import日志。Gate绑定与文件传输时间也不独立证明作业先后，本轮没有SLURM作业日志审计。

已完成目录不再次启动或merge；原单文件提交说明见[目录README](../numerical_simulation/outputs_v51_attribution/README.md)，旧说明原字节已归档。[本轮复现与边界](taskcard7_completion/REPRODUCIBILITY.md)；[本轮索引](taskcard7_completion/DELIVERY_INDEX.json)。
