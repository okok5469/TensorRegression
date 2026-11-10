> **BLOCKED / NOT FOR SCIENTIFIC DELIVERY.** Official class-ID definitions conflict with the official gas-by-batch count table and local DAT counts. Chemical names below are provisional and are NOT semantically accepted. Earlier Gate D PASS is withdrawn; numerical fits retain the original gas_id without relabeling. See [SCIENCE_REVIEW.md](SCIENCE_REVIEW.md).

# Pilot 与算法正确性验收

本文仅记录已完成的 gas1 / dev1 Pilot 及其独立审计；后续开发与最终评价的完成状态见阶段验收文件，不在这里提前宣告完成。科学协议未根据预测成绩调整。

## 实际执行与成本

Pilot 使用批次1—6训练、批次7测试，训练1225条、测试649条，八种方法全部保存成功。八个外层任务累计子进程 wall time 为 **138.0003772秒**；独立原生ABESS重算另用 **18.3208783秒**，共 **156.3212555秒**，原生重算不是新的独立实验。后续累计成本以 `fit_budget.json` 与 `audit_native_budget.json` 为准，二者不得漏算或重复计算。默认单worker、每worker数值库线程1，模型阶段共享上限14400秒、单任务保护1800秒。

| 方法 | Pilot子进程耗时（秒） | 已保存的数值状态 |
|---|---:|---|
| M0 训练均值 | 6.762 | 解析解 |
| M1 Tensor-ABESS96 | 18.269 | 完整求解收敛状态未暴露，记unknown |
| M2 Tensor-GLasso96 | 21.072 | grouped ISTA完整停止状态未暴露，记unknown |
| M3 Lasso96 | 28.680 | 外层收敛；部分内折未收敛 |
| M4 Ridge96 | 9.257 | SVD直接解 |
| M5 PLS96 | 8.148 | 迭代检查通过 |
| M6 Ridge128 | 13.270 | SVD直接解 |
| M7 Lasso128 | 32.543 | 外层收敛；部分内折未收敛 |

M3内折0、相对惩罚0.001，以及M7内折0、相对惩罚0.002和0.001，在预先固定的10000→50000次迭代升级后仍未收敛。全部warning、dual gap及重试保存在内折记录中；未追加迭代、删除候选或改变网格。外层收敛不能消除这些调参候选的数值局限，因此不能将整个基线调参过程表述为已完全收敛。

## 独立检查证明了什么

- 全量13910条记录的96个动态坐标共1,335,360值映射检查通过。按传感器内阶段×尺度拆列后转置为16×3×2，C-order系数还原一致；不是直接reshape96列。
- 冻结合同列明六气体、四外层划分，共24个合格任务；每个任务使用最近至多三个满足门槛的时序内折。Pilot审计复查内外时序、X重复组隔离、训练ID和缩放来源。未来任务使用相同规则，但它们是否已完成须查对应阶段审计。
- 各内外训练折单独拟合X均值/总体标准差与y均值/标准差；仅精确零X标准差置1。每个内折Lasso alpha_max重新计算；PLS仅在全部内折合法的候选中选择。验证/测试标签未用于拟合缩放器。
- 八个保存模型预测重放最大绝对误差均为0。实际sklearn估计器与保存系数的预测一致，PLS标准化预测差最大为5.56×10⁻¹⁶。219组候选×内折的SSE由保存验证预测独立复算，无缺失记录。
- Pilot ABESS支撑容量q=90、设计rank=90，OLS RSS=195.0142203074，原MSIC=-1980.8342317280；独立OLS预测差最大2.58×10⁻¹⁴。
- 额外原生重算只通过 `sys.settrace` 读取原函数返回时的局部候选，不改源码、参数或返回值。共97条候选记录（初始候选加96个网格候选），最小分数-1988.7502733654、对应q=96、容忍带宽26.1271281967；独立按原规则选中记录90，系数与保存模型逐元素相同。不能把97条记录说成97个独立样本或97个互异支撑。

完整候选容忍带仅在Pilot原生追踪中独立核验；其他原生结果不保存全部候选，不伪称从历史损失即可重建。数值验收不证明统计泛化、全局最优或相对预测优势。证据见 `audit_pilot.json`、`audit_native_pilot.json`、`PILOT_ACCEPTED.json`；测试全集状态见 `UNIT_VALIDATION.json`。

## 一次拟合前留痕修订

第一版冻结后、任何真实拟合前，发现内折仅保存SSE与缩放器，不足以独立重算。先新增失败测试，再补保存内折验证预测、标准化系数和截距；九项模型/调度测试通过后重新冻结。旧合同、配置、协议、模型源码和修订说明均保存在 `pre_audit_fix/`。该修订只补审计证据，没有改数据、划分、网格、求解或选模规则。

当前拟合合同SHA256：`246acfaef16d0c6452116d46347958f1d0e645cec7e3e468f8cec6a4d426eccc`。
