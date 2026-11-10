> **BLOCKED / NOT FOR SCIENTIFIC DELIVERY.** Official class-ID definitions conflict with the official gas-by-batch count table and local DAT counts. Chemical names below are provisional and are NOT semantically accepted. Earlier Gate D PASS is withdrawn; numerical fits retain the original gas_id without relabeling. See [SCIENCE_REVIEW.md](SCIENCE_REVIEW.md).

# 同输入复杂度、稳定性与成本

阈值为标准化X/y坐标下abs(beta)>1e-8×max(1,max(abs(beta)))；只描述数值，不改预测。传感器编号1—16，尺度索引0/1/2对应.001/.01/.1，阶段0/1对应i/d。

| 气体 | 方法 | 原生矩形q | 严格非零 | 有效非零 | 传感器 | 尺度 | 阶段 | 外包容量 | 选中设计秩 | PLS成分 | 训练+调参秒 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 乙醇 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.01666 |
| 乙醇 | Tensor-ABESS96 | 84 | 84 | 84 | 14 | 3 | 2 | 84 | 84 | NA | 15.51 |
| 乙醇 | Tensor-GLasso96 | 3 | 3 | 3 | 3 | 1 | 1 | 3 | 3 | NA | 15.07 |
| 乙醇 | Lasso96 | NA | 35 | 35 | 16 | 3 | 2 | 96 | NA | NA | 10.37 |
| 乙醇 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 1.365 |
| 乙醇 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 8 | 0.5244 |
| 乙醇 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 3.681 |
| 乙醇 | Lasso128 | NA | 16 | 16 | 11 | NA | NA | NA | NA | NA | 27.87 |
| 乙烯 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.01975 |
| 乙烯 | Tensor-ABESS96 | 84 | 84 | 84 | 14 | 3 | 2 | 84 | 84 | NA | 14.14 |
| 乙烯 | Tensor-GLasso96 | 72 | 72 | 72 | 12 | 3 | 2 | 72 | 72 | NA | 21.08 |
| 乙烯 | Lasso96 | NA | 10 | 10 | 9 | 3 | 2 | 54 | NA | NA | 9.075 |
| 乙烯 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 2.978 |
| 乙烯 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 1 | 0.8695 |
| 乙烯 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 3.454 |
| 乙烯 | Lasso128 | NA | 44 | 44 | 16 | NA | NA | NA | NA | NA | 13.39 |
| 氨 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.008187 |
| 氨 | Tensor-ABESS96 | 32 | 32 | 32 | 8 | 2 | 2 | 32 | 32 | NA | 8.842 |
| 氨 | Tensor-GLasso96 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | NA | 10.29 |
| 氨 | Lasso96 | NA | 20 | 20 | 11 | 3 | 2 | 66 | NA | NA | 10.05 |
| 氨 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 1.06 |
| 氨 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 3 | 0.2017 |
| 氨 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 2.077 |
| 氨 | Lasso128 | NA | 15 | 15 | 12 | NA | NA | NA | NA | NA | 8.053 |
| 乙醛 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.01485 |
| 乙醛 | Tensor-ABESS96 | 32 | 32 | 32 | 16 | 1 | 2 | 32 | 32 | NA | 12.35 |
| 乙醛 | Tensor-GLasso96 | 84 | 84 | 84 | 14 | 3 | 2 | 84 | 84 | NA | 15.01 |
| 乙醛 | Lasso96 | NA | 25 | 25 | 11 | 3 | 2 | 66 | NA | NA | 3.697 |
| 乙醛 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 1.35 |
| 乙醛 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 8 | 0.3534 |
| 乙醛 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 1.968 |
| 乙醛 | Lasso128 | NA | 35 | 35 | 16 | NA | NA | NA | NA | NA | 6.012 |
| 丙酮 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.03444 |
| 丙酮 | Tensor-ABESS96 | 90 | 90 | 90 | 15 | 3 | 2 | 90 | 90 | NA | 10.98 |
| 丙酮 | Tensor-GLasso96 | 16 | 16 | 16 | 8 | 1 | 2 | 16 | 16 | NA | 22.98 |
| 丙酮 | Lasso96 | NA | 9 | 9 | 7 | 3 | 2 | 42 | NA | NA | 10.6 |
| 丙酮 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 3.34 |
| 丙酮 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 1 | 0.7725 |
| 丙酮 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 4.134 |
| 丙酮 | Lasso128 | NA | 10 | 10 | 8 | NA | NA | NA | NA | NA | 20.17 |
| 甲苯 | 训练均值 | NA | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0.00837 |
| 甲苯 | Tensor-ABESS96 | 12 | 12 | 12 | 6 | 1 | 2 | 12 | 12 | NA | 9.389 |
| 甲苯 | Tensor-GLasso96 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | NA | 12.48 |
| 甲苯 | Lasso96 | NA | 54 | 54 | 16 | 3 | 2 | 96 | NA | NA | 8.709 |
| 甲苯 | Ridge96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | NA | 1.266 |
| 甲苯 | PLS96 | NA | 96 | 96 | 16 | 3 | 2 | 96 | NA | 16 | 0.2891 |
| 甲苯 | Ridge128 | NA | 128 | 128 | 16 | NA | NA | NA | NA | NA | 2.367 |
| 甲苯 | Lasso128 | NA | 53 | 53 | 15 | NA | NA | NA | NA | NA | 10.96 |

## 三个开发窗的两两Jaccard均值

| 气体 | 方法 | 模式 | 均值 |
|---|---|---|---|
| 乙醇 | 训练均值 | sensors | 1 |
| 乙醇 | 训练均值 | scales | 1 |
| 乙醇 | 训练均值 | stages | 1 |
| 乙醇 | Tensor-ABESS96 | sensors | 0.7444 |
| 乙醇 | Tensor-ABESS96 | scales | 1 |
| 乙醇 | Tensor-ABESS96 | stages | 1 |
| 乙醇 | Tensor-GLasso96 | sensors | 0.619 |
| 乙醇 | Tensor-GLasso96 | scales | 1 |
| 乙醇 | Tensor-GLasso96 | stages | 1 |
| 乙醇 | Lasso96 | sensors | 0.7333 |
| 乙醇 | Lasso96 | scales | 1 |
| 乙醇 | Lasso96 | stages | 1 |
| 乙醇 | Ridge96 | sensors | 1 |
| 乙醇 | Ridge96 | scales | 1 |
| 乙醇 | Ridge96 | stages | 1 |
| 乙醇 | PLS96 | sensors | 1 |
| 乙醇 | PLS96 | scales | 1 |
| 乙醇 | PLS96 | stages | 1 |
| 乙醇 | Ridge128 | sensors | 1 |
| 乙醇 | Ridge128 | scales | NA |
| 乙醇 | Ridge128 | stages | NA |
| 乙醇 | Lasso128 | sensors | 0.7333 |
| 乙醇 | Lasso128 | scales | NA |
| 乙醇 | Lasso128 | stages | NA |
| 乙烯 | 训练均值 | sensors | 1 |
| 乙烯 | 训练均值 | scales | 1 |
| 乙烯 | 训练均值 | stages | 1 |
| 乙烯 | Tensor-ABESS96 | sensors | 0.9583 |
| 乙烯 | Tensor-ABESS96 | scales | 0.7778 |
| 乙烯 | Tensor-ABESS96 | stages | 1 |
| 乙烯 | Tensor-GLasso96 | sensors | 1 |
| 乙烯 | Tensor-GLasso96 | scales | 1 |
| 乙烯 | Tensor-GLasso96 | stages | 1 |
| 乙烯 | Lasso96 | sensors | 0.6667 |
| 乙烯 | Lasso96 | scales | 0.7778 |
| 乙烯 | Lasso96 | stages | 0.6667 |
| 乙烯 | Ridge96 | sensors | 1 |
| 乙烯 | Ridge96 | scales | 1 |
| 乙烯 | Ridge96 | stages | 1 |
| 乙烯 | PLS96 | sensors | 1 |
| 乙烯 | PLS96 | scales | 1 |
| 乙烯 | PLS96 | stages | 1 |
| 乙烯 | Ridge128 | sensors | 1 |
| 乙烯 | Ridge128 | scales | NA |
| 乙烯 | Ridge128 | stages | NA |
| 乙烯 | Lasso128 | sensors | 0.5 |
| 乙烯 | Lasso128 | scales | NA |
| 乙烯 | Lasso128 | stages | NA |
| 氨 | 训练均值 | sensors | 1 |
| 氨 | 训练均值 | scales | 1 |
| 氨 | 训练均值 | stages | 1 |
| 氨 | Tensor-ABESS96 | sensors | 0.6444 |
| 氨 | Tensor-ABESS96 | scales | 0.7778 |
| 氨 | Tensor-ABESS96 | stages | 1 |
| 氨 | Tensor-GLasso96 | sensors | 0.5333 |
| 氨 | Tensor-GLasso96 | scales | 0.6667 |
| 氨 | Tensor-GLasso96 | stages | 0.6667 |
| 氨 | Lasso96 | sensors | 0.3314 |
| 氨 | Lasso96 | scales | 0.5556 |
| 氨 | Lasso96 | stages | 0.6667 |
| 氨 | Ridge96 | sensors | 1 |
| 氨 | Ridge96 | scales | 1 |
| 氨 | Ridge96 | stages | 1 |
| 氨 | PLS96 | sensors | 1 |
| 氨 | PLS96 | scales | 1 |
| 氨 | PLS96 | stages | 1 |
| 氨 | Ridge128 | sensors | 1 |
| 氨 | Ridge128 | scales | NA |
| 氨 | Ridge128 | stages | NA |
| 氨 | Lasso128 | sensors | 0.75 |
| 氨 | Lasso128 | scales | NA |
| 氨 | Lasso128 | stages | NA |
| 乙醛 | 训练均值 | sensors | 1 |
| 乙醛 | 训练均值 | scales | 1 |
| 乙醛 | 训练均值 | stages | 1 |
| 乙醛 | Tensor-ABESS96 | sensors | 0.9167 |
| 乙醛 | Tensor-ABESS96 | scales | 0.5556 |
| 乙醛 | Tensor-ABESS96 | stages | 1 |
| 乙醛 | Tensor-GLasso96 | sensors | 0.6 |
| 乙醛 | Tensor-GLasso96 | scales | 0.6667 |
| 乙醛 | Tensor-GLasso96 | stages | 0.6667 |
| 乙醛 | Lasso96 | sensors | 0.7576 |
| 乙醛 | Lasso96 | scales | 1 |
| 乙醛 | Lasso96 | stages | 1 |
| 乙醛 | Ridge96 | sensors | 1 |
| 乙醛 | Ridge96 | scales | 1 |
| 乙醛 | Ridge96 | stages | 1 |
| 乙醛 | PLS96 | sensors | 1 |
| 乙醛 | PLS96 | scales | 1 |
| 乙醛 | PLS96 | stages | 1 |
| 乙醛 | Ridge128 | sensors | 1 |
| 乙醛 | Ridge128 | scales | NA |
| 乙醛 | Ridge128 | stages | NA |
| 乙醛 | Lasso128 | sensors | 0.625 |
| 乙醛 | Lasso128 | scales | NA |
| 乙醛 | Lasso128 | stages | NA |
| 丙酮 | 训练均值 | sensors | 1 |
| 丙酮 | 训练均值 | scales | 1 |
| 丙酮 | 训练均值 | stages | 1 |
| 丙酮 | Tensor-ABESS96 | sensors | 0.9167 |
| 丙酮 | Tensor-ABESS96 | scales | 1 |
| 丙酮 | Tensor-ABESS96 | stages | 1 |
| 丙酮 | Tensor-GLasso96 | sensors | 0.9333 |
| 丙酮 | Tensor-GLasso96 | scales | 1 |
| 丙酮 | Tensor-GLasso96 | stages | 1 |
| 丙酮 | Lasso96 | sensors | 0.6984 |
| 丙酮 | Lasso96 | scales | 1 |
| 丙酮 | Lasso96 | stages | 1 |
| 丙酮 | Ridge96 | sensors | 1 |
| 丙酮 | Ridge96 | scales | 1 |
| 丙酮 | Ridge96 | stages | 1 |
| 丙酮 | PLS96 | sensors | 1 |
| 丙酮 | PLS96 | scales | 1 |
| 丙酮 | PLS96 | stages | 1 |
| 丙酮 | Ridge128 | sensors | 1 |
| 丙酮 | Ridge128 | scales | NA |
| 丙酮 | Ridge128 | stages | NA |
| 丙酮 | Lasso128 | sensors | 0.7847 |
| 丙酮 | Lasso128 | scales | NA |
| 丙酮 | Lasso128 | stages | NA |
| 甲苯 | 训练均值 | sensors | 1 |
| 甲苯 | 训练均值 | scales | 1 |
| 甲苯 | 训练均值 | stages | 1 |
| 甲苯 | Tensor-ABESS96 | sensors | 1 |
| 甲苯 | Tensor-ABESS96 | scales | 0.6667 |
| 甲苯 | Tensor-ABESS96 | stages | 0.6667 |
| 甲苯 | Tensor-GLasso96 | sensors | 0.3846 |
| 甲苯 | Tensor-GLasso96 | scales | 0.5556 |
| 甲苯 | Tensor-GLasso96 | stages | 0.6667 |
| 甲苯 | Lasso96 | sensors | 0.5556 |
| 甲苯 | Lasso96 | scales | 0.6667 |
| 甲苯 | Lasso96 | stages | 1 |
| 甲苯 | Ridge96 | sensors | 1 |
| 甲苯 | Ridge96 | scales | 1 |
| 甲苯 | Ridge96 | stages | 1 |
| 甲苯 | PLS96 | sensors | 1 |
| 甲苯 | PLS96 | scales | 1 |
| 甲苯 | PLS96 | stages | 1 |
| 甲苯 | Ridge128 | sensors | 1 |
| 甲苯 | Ridge128 | scales | NA |
| 甲苯 | Ridge128 | stages | NA |
| 甲苯 | Lasso128 | sensors | 0.4667 |
| 甲苯 | Lasso128 | scales | NA |
| 甲苯 | Lasso128 | stages | NA |

两个空集Jaccard定义为1，是“同时未选”的记号，不代表有可用预测信号。三个训练窗重叠，频率分母仅3，不解释为独立重复或支撑恢复。final支撑不计入频率。128维不将稳态量强塞入尺度/阶段；PLS成分少不等于传感器少。GLasso表中q指最终矩形OLS支撑，原生组状态见model_complexity.csv。预测计时受系统影响，是描述性测量，不是硬件公平跑分。
