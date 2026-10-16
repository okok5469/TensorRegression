# 预测恶化个案：训练改进不能替代泛化验证

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

按已保存PMSE先冻结最大3恶化、最大3改善、最严重丢失Exact及首个“新增Exact却恶化”规则，再读取NPZ；最后一类本批为空，最终7例。仅审原模型，不重训或换规则；每份被读NPZ均核验原record的哈希。当前无法重建严格匹配的正式训练X/y，故未虚构模拟OLS秩/条件数或重新计算训练RSS。

| task / 场景 | q旧→新 | RSS旧→新 | PMSE旧→新 | ΔPMSE | 条件总体signal risk旧→新 |
| --- | --- | --- | --- | --- | --- |
| 45 / baseline=baseline | 80→60 | 120.712→146.078 | 3.3744→1.6230 | -1.7514 | 2.1556→0.4078 |
| 127 / snr=1 | 15→80 | 321.275→121.189 | 2.6966→1.7901 | -0.9065 | 1.7492→0.9556 |
| 259 / sample_size=100 | 2→64 | 309.416→44.471 | 3.6130→9.1202 | +5.5072 | 2.2162→7.3763 |
| 277 / sample_size=100 | 36→48 | 63.288→44.301 | 7.7699→5.9524 | -1.8175 | 6.3360→5.1241 |
| 299 / sample_size=100 | 4→64 | 217.111→32.839 | 3.7637→9.5960 | +5.8323 | 2.6790→7.7581 |
| 562 / correlation=0.00 | 60→125 | 124.031→43.381 | 1.5581→5.9421 | +4.3841 | 0.3433→4.6559 |
| 585 / correlation=0.00 | 24→125 | 336.387→66.833 | 3.0190→8.1302 | +5.1112 | 2.3279→7.3033 |

task299：q4→64、RSS217.11→32.84，而PMSE3.76→9.60；task259相似，训练更好却测试更差。task562由真支持q60变成非真q125、RSS降低而PMSE上升；零相关下false-positive系数对条件总体risk的贡献从0升至2.3404。task127却在q15→80后PMSE改善；task277也在q增加时改善，反证“所有大模型变化都一定更差”。

条件总体signal risk直接由保存的E=B_hat−B_true与声明的可分离AR1协方差计算Var(<X,E>)，未生成新测试数据；拆为真块内系数误差、块外误选系数和交叉协方差项，完整项见JSON。它是**给定已估B的总体预测风险**，不是重复采样估计方差/偏差分解，不把块外风险一律叫作方差；相关设计交叉项不能省略。该风险与有限500测试样本signal MSE不同，不能替换原正式指标。

[700任务配对CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_old_new_700.csv)、[600唯一输入CSV](../numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_unique_input_600.csv)、[尾部/分场景/分组统计](../numerical_simulation/outputs_version_attribution/saved_result_analysis/summary.json)、[事先冻结个案规则](../numerical_simulation/outputs_version_attribution/saved_result_analysis/CASE_SELECTION.json)、[个案完整诊断](../numerical_simulation/outputs_version_attribution/saved_result_analysis/selected_cases.json)。
