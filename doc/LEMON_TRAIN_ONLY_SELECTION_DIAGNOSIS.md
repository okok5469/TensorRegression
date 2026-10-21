# LEMON仅训练数据的模型复杂度诊断

2026-10-17补记：250/250模拟回传已通过验收，任务卡7本地A/C证据保持原样；新增两段归因见[完整模拟报告](VERSION_EFFECT_DECOMPOSITION_FINAL.md)及[本轮复现](taskcard7_completion/REPRODUCIBILITY.md)。

## 拟合前冻结的设计

固定选择三个重复的第一外折，未按外测表现挑选；每个外训练集仅116人，再按原inner_seed+repeat×100+fold做三折。每个内训练77/78人重新拟合协变量残差化、X标准化、响应中心化与候选支撑，恢复原log(TMT_5)响应评价。外测试者的X/y/C不传入任何内拟合、预测或误差计算。

研究版为独立进程加载冻结v5.1-bcd-fix；完整32源码及所有实际import、输入、配置、辅助代码、环境和种子写入[protocol](../numerical_simulation/outputs_version_attribution/train_only/protocol.json)，prepare先落盘后run严格校验，拒绝覆盖。原算法参数不变。

仅协变量、原自适应MSIC和固定q=1(1,1,1)、12(2,2,3)、45(3,3,5)同时比较。q80超过内训练容量floor(.7n)=53/54，已在协议中排除且没有拟合；45是预先定义的合法“大”诊断规模，不用于修改外OOF或正式caps。

## 完整内OOF结果

| 模型 | 重复0内OOF RMSE | 重复1内OOF RMSE | 重复2内OOF RMSE |
| --- | --- | --- | --- |
| Covariates Only | 0.344248 | 0.331987 | 0.323181 |
| Adaptive MSIC | 0.420495 | 0.649267 | 0.475620 |
| Fixed small | 0.378526 | 0.361387 | 0.323780 |
| Fixed medium | 0.409972 | 0.408512 | 0.347155 |
| Fixed large | 0.699425 | 0.549459 | 0.560735 |

每个数值由同一外训练集116人的完整内OOF SSE计算，未平均内折RMSE；三个外训练集高度重叠，因此这些不是三组独立确认样本。自适应各内折q为[2,6,32]、[48,48,48]、[8,2,48]。

固定规模q1→12→45时训练RMSE9/9严格下降、验证RMSE7/9严格上升；q45在全部9折验证误差均高于q1，MSIC却在8/9更低。三组自适应结果均劣于协变量基线。该现象支持当前有限样本下训练准则与泛化目标偏离，不足以认定所有EEG特征无稳定信息。

![全部内折复杂度诊断](taskcard7/figures/train_only_complexity.png)

## 完整性与边界

45模型、1740验证预测、5220含训练的总预测行全部保存，设计均满秩且q合法；独立按B/预处理/ID重构最大预测误差1.60e−14，最大tensor设计条件数212.765。没有使用完整116人先选支撑再交叉验证，没有更新正式外OOF，没有按内验证结果发布新模型。逐内折数据及[核验](../numerical_simulation/outputs_version_attribution/train_only/verification.json)均独立留存。[复现与身份记录](taskcard7/REPRODUCIBILITY.md)；[完整配对表](../numerical_simulation/outputs_version_attribution/lemon/three_version_folds.csv)；[实验索引](taskcard7/DELIVERY_INDEX.json)。
