# Fixed BCD与原MSIC的交互机制

2026-10-17补记：250/250模拟回传已通过验收，任务卡7本地A/C证据保持原样；新增两段归因见[完整模拟报告](VERSION_EFFECT_DECOMPOSITION_FINAL.md)及[本轮复现](taskcard7_completion/REPRODUCIBILITY.md)。

已观察到：fixed BCD修复降低了相同规模候选的训练RSS，原不变MSIC因此更偏好部分接近容量的大模型；该机制由真实候选重放及独立训练内验证共同支持。

## 预声明代表与真实路径

在重放前冻结代表规则：历史v5或v5.2训练所选q80折的并集，另加字典序第一个两版q均小于80的折；未按测试RMSE选代表。实际6折为(0,1),(0,2),(1,1),(1,2),(2,3),(2,5)。仅新增v5/v5.2各6训练拟合，v5.1复用Phase A真实轨迹。

各版每折初始化三模大小、完整支撑、adaptive终态、80个有序fixed候选、RSS/MSIC/完整B、one-SE带与最终选择都已保存；历史最终B、支撑、score均精确重现。v5.1/v5.2的486候选连同所有固定搜索状态完全一致，不存在候选顺序、初始化、缓存或浮点并列差异的证据。

| 重复/折 | q：旧→新 | q80 RSS：旧→新 | q80 ΔMSIC | q80排名：旧→新 | 容忍带：旧→新 |
| --- | --- | --- | --- | --- | --- |
| 0/1 | 3→80 | 1.745571→0.986812 | -66.161529 | 4→1 | 3.776344→19.500954 |
| 0/2 | 2→1 | 2.174278→1.774076 | -23.596376 | 45→7 | 4.360546→7.552687 |
| 1/1 | 2→80 | 2.237647→1.145243 | -77.697680 | 59→1 | 3.083372→19.500954 |
| 1/2 | 1→80 | 1.883222→1.260537 | -46.567815 | 40→1 | 3.083372→19.500954 |
| 2/3 | 1→80 | 1.613782→1.206104 | -33.777527 | 1→1 | 19.500954→19.500954 |
| 2/5 | 80→80 | 1.234092→1.067482 | -16.823807 | 1→1 | 19.500954→19.500954 |

## 数学分解

正式全局精修MSIC为 n log(RSS/n) + 1.5 q log(log n) + log(log n) Σ_m log choose(p_m,s_m)，n=116，原gamma=1；mode级自适应选阶与全局精修参数分开，未混用或修改。对于同一三模规模而不同支撑位置：

`ΔMSIC = n log(RSS_new/RSS_old) + ΔPenalty`。

同形状、相同n/p/alpha/gamma时惩罚严格相同；支撑位置不进入该惩罚。480个fixed对中295 RSS降低、185相同，无变差；由score相减反算的惩罚差不超过8.53e−14。例(0,1) q80 RSS1.745571→0.986812，MSIC−267.501784→−333.663313，下降66.161529；原q80位列4且在带外，新q80位列1且是唯一带内候选。

one-SE风格带是代码确定性容忍量 `sqrt(q_best log n)`，不是由独立验证估计的标准误。它可改变可选集合，但不能单独解释全部变化：(2,3)旧版q80已是最低MSIC却仍选q1，修复后同一容忍量下改选80；控制(0,2)修复后q80进入带内，最终仍选q1。不能把所有大模型归因于容忍带扩大。

## 验收与解释边界

三版本六折合计1458候选独立由保存B和训练预处理重算RSS/MSIC并核验满秩、q合法；候选与源码全hash保留。972行两段配对分解均为相同三模形状，逐对单列惩罚差，差异仅浮点舍入，见[CSV](../numerical_simulation/outputs_version_attribution/mechanism/candidate_msic_decomposition.csv)、[候选选择摘要](../numerical_simulation/outputs_version_attribution/mechanism/selection_summary.json)和[验收](../numerical_simulation/outputs_version_attribution/mechanism/verification.json)。

公式解释的是训练拟合改善如何转成更低MSIC，并不证明总体风险最优，也不证明普适的统计失效；训练内验证提供本样本的独立于外测选参的补充证据。没有为取得较小外测误差重评分选alpha。[复现与身份记录](taskcard7/REPRODUCIBILITY.md)；[完整配对表](../numerical_simulation/outputs_version_attribution/lemon/three_version_folds.csv)；[实验索引](taskcard7/DELIVERY_INDEX.json)。
