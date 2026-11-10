# 预测与跨批次变化

开发训练窗互相重叠，各批次分别评价；不混合六类原始量程计算总R²。宏平均标准化RMSE先除以各类当前训练y标准差，再六类等权平均。

| 窗口 | 方法 | 宏平均标准化RMSE |
|---|---|---|
| dev1 | 训练均值 | 1.02 |
| dev1 | Tensor-ABESS96 | 0.5786 |
| dev1 | Tensor-GLasso96 | 0.6123 |
| dev1 | Lasso96 | 0.6234 |
| dev1 | Ridge96 | 0.5727 |
| dev1 | PLS96 | 0.5889 |
| dev1 | Ridge128 | 0.556 |
| dev1 | Lasso128 | 0.5815 |
| dev2 | 训练均值 | 1.052 |
| dev2 | Tensor-ABESS96 | 0.4988 |
| dev2 | Tensor-GLasso96 | 0.6267 |
| dev2 | Lasso96 | 0.4681 |
| dev2 | Ridge96 | 0.413 |
| dev2 | PLS96 | 0.4721 |
| dev2 | Ridge128 | 0.4737 |
| dev2 | Lasso128 | 0.3774 |
| dev3 | 训练均值 | 0.9316 |
| dev3 | Tensor-ABESS96 | 0.3237 |
| dev3 | Tensor-GLasso96 | 0.344 |
| dev3 | Lasso96 | 0.43 |
| dev3 | Ridge96 | 0.3517 |
| dev3 | PLS96 | 0.3852 |
| dev3 | Ridge128 | 1.067 |
| dev3 | Lasso128 | 0.4898 |
| final | 训练均值 | 2.11 |
| final | Tensor-ABESS96 | 2.762 |
| final | Tensor-GLasso96 | 2.282 |
| final | Lasso96 | 1.731 |
| final | Ridge96 | 2.697 |
| final | PLS96 | 1.503 |
| final | Ridge128 | 1.915 |
| final | Lasso128 | 1.549 |

## 按类别同窗口配对误差差

| 气体 | 窗口 | 比较 | RMSE差ppmv |
|---|---|---|---|
| 乙醇 | dev1 | ABESS − Tensor-GLasso96 | 2.845 |
| 乙醇 | dev1 | ABESS − Lasso96 | 10.61 |
| 乙醇 | dev1 | ABESS − Ridge96 | 17.05 |
| 乙醇 | dev1 | ABESS − PLS96 | 11.7 |
| 乙醇 | dev1 | ABESS − Ridge128 | 13.9 |
| 乙醇 | dev1 | ABESS − Lasso128 | 9.811 |
| 乙醇 | dev2 | ABESS − Tensor-GLasso96 | 55.8 |
| 乙醇 | dev2 | ABESS − Lasso96 | 96.6 |
| 乙醇 | dev2 | ABESS − Ridge96 | 84.9 |
| 乙醇 | dev2 | ABESS − PLS96 | 87.29 |
| 乙醇 | dev2 | ABESS − Ridge128 | 84.36 |
| 乙醇 | dev2 | ABESS − Lasso128 | 98.68 |
| 乙醇 | dev3 | ABESS − Tensor-GLasso96 | 24.7 |
| 乙醇 | dev3 | ABESS − Lasso96 | 7.697 |
| 乙醇 | dev3 | ABESS − Ridge96 | -0.2094 |
| 乙醇 | dev3 | ABESS − PLS96 | -12.59 |
| 乙醇 | dev3 | ABESS − Ridge128 | 0.3669 |
| 乙醇 | dev3 | ABESS − Lasso128 | 25.58 |
| 乙醇 | final | ABESS − Tensor-GLasso96 | -10.36 |
| 乙醇 | final | ABESS − Lasso96 | 49.12 |
| 乙醇 | final | ABESS − Ridge96 | 36.69 |
| 乙醇 | final | ABESS − PLS96 | 37.19 |
| 乙醇 | final | ABESS − Ridge128 | 31.57 |
| 乙醇 | final | ABESS − Lasso128 | 20.16 |
| 乙烯 | dev1 | ABESS − Tensor-GLasso96 | -10.91 |
| 乙烯 | dev1 | ABESS − Lasso96 | -19.08 |
| 乙烯 | dev1 | ABESS − Ridge96 | -15.03 |
| 乙烯 | dev1 | ABESS − PLS96 | -8.421 |
| 乙烯 | dev1 | ABESS − Ridge128 | -10.43 |
| 乙烯 | dev1 | ABESS − Lasso128 | -17.54 |
| 乙烯 | dev2 | ABESS − Tensor-GLasso96 | 28.89 |
| 乙烯 | dev2 | ABESS − Lasso96 | 8.61 |
| 乙烯 | dev2 | ABESS − Ridge96 | 19.36 |
| 乙烯 | dev2 | ABESS − PLS96 | 22.46 |
| 乙烯 | dev2 | ABESS − Ridge128 | 4.574 |
| 乙烯 | dev2 | ABESS − Lasso128 | 23.2 |
| 乙烯 | dev3 | ABESS − Tensor-GLasso96 | 13.53 |
| 乙烯 | dev3 | ABESS − Lasso96 | -5.764 |
| 乙烯 | dev3 | ABESS − Ridge96 | 1.951 |
| 乙烯 | dev3 | ABESS − PLS96 | 4.928 |
| 乙烯 | dev3 | ABESS − Ridge128 | 9.921 |
| 乙烯 | dev3 | ABESS − Lasso128 | 3.654 |
| 乙烯 | final | ABESS − Tensor-GLasso96 | 182 |
| 乙烯 | final | ABESS − Lasso96 | 579.8 |
| 乙烯 | final | ABESS − Ridge96 | 15.83 |
| 乙烯 | final | ABESS − PLS96 | 550.3 |
| 乙烯 | final | ABESS − Ridge128 | 283.8 |
| 乙烯 | final | ABESS − Lasso128 | 513.6 |
| 氨 | dev1 | ABESS − Tensor-GLasso96 | 3.964 |
| 氨 | dev1 | ABESS − Lasso96 | -22.36 |
| 氨 | dev1 | ABESS − Ridge96 | 6.085 |
| 氨 | dev1 | ABESS − PLS96 | 7.614 |
| 氨 | dev1 | ABESS − Ridge128 | 3.025 |
| 氨 | dev1 | ABESS − Lasso128 | 5.677 |
| 氨 | dev2 | ABESS − Tensor-GLasso96 | -6.39 |
| 氨 | dev2 | ABESS − Lasso96 | -91.04 |
| 氨 | dev2 | ABESS − Ridge96 | -88.57 |
| 氨 | dev2 | ABESS − PLS96 | -150 |
| 氨 | dev2 | ABESS − Ridge128 | -46.41 |
| 氨 | dev2 | ABESS − Lasso128 | -27.16 |
| 氨 | dev3 | ABESS − Tensor-GLasso96 | -20.35 |
| 氨 | dev3 | ABESS − Lasso96 | 41.21 |
| 氨 | dev3 | ABESS − Ridge96 | 10.7 |
| 氨 | dev3 | ABESS − PLS96 | 41.18 |
| 氨 | dev3 | ABESS − Ridge128 | 32.33 |
| 氨 | dev3 | ABESS − Lasso128 | 42.85 |
| 氨 | final | ABESS − Tensor-GLasso96 | 37.62 |
| 氨 | final | ABESS − Lasso96 | -244.2 |
| 氨 | final | ABESS − Ridge96 | -31.63 |
| 氨 | final | ABESS − PLS96 | -27.91 |
| 氨 | final | ABESS − Ridge128 | 80.46 |
| 氨 | final | ABESS − Lasso128 | -25.44 |
| 乙醛 | dev1 | ABESS − Tensor-GLasso96 | -4.775 |
| 乙醛 | dev1 | ABESS − Lasso96 | -6.652 |
| 乙醛 | dev1 | ABESS − Ridge96 | 0.1048 |
| 乙醛 | dev1 | ABESS − PLS96 | -4.922 |
| 乙醛 | dev1 | ABESS − Ridge128 | -0.3242 |
| 乙醛 | dev1 | ABESS − Lasso128 | -3.535 |
| 乙醛 | dev2 | ABESS − Tensor-GLasso96 | 10.1 |
| 乙醛 | dev2 | ABESS − Lasso96 | 10 |
| 乙醛 | dev2 | ABESS − Ridge96 | 8.372 |
| 乙醛 | dev2 | ABESS − PLS96 | 7 |
| 乙醛 | dev2 | ABESS − Ridge128 | -13.7 |
| 乙醛 | dev2 | ABESS − Lasso128 | 7.7 |
| 乙醛 | dev3 | ABESS − Tensor-GLasso96 | -3.253 |
| 乙醛 | dev3 | ABESS − Lasso96 | -2.22 |
| 乙醛 | dev3 | ABESS − Ridge96 | -2.325 |
| 乙醛 | dev3 | ABESS − PLS96 | -0.1957 |
| 乙醛 | dev3 | ABESS − Ridge128 | 0.84 |
| 乙醛 | dev3 | ABESS − Lasso128 | -1.09 |
| 乙醛 | final | ABESS − Tensor-GLasso96 | 5.153 |
| 乙醛 | final | ABESS − Lasso96 | -16.04 |
| 乙醛 | final | ABESS − Ridge96 | -12.14 |
| 乙醛 | final | ABESS − PLS96 | -8.319 |
| 乙醛 | final | ABESS − Ridge128 | -2.783 |
| 乙醛 | final | ABESS − Lasso128 | -5.627 |
| 丙酮 | dev1 | ABESS − Tensor-GLasso96 | 6.9 |
| 丙酮 | dev1 | ABESS − Lasso96 | 3.551 |
| 丙酮 | dev1 | ABESS − Ridge96 | -1.045 |
| 丙酮 | dev1 | ABESS − PLS96 | -3.838 |
| 丙酮 | dev1 | ABESS − Ridge128 | 11.14 |
| 丙酮 | dev1 | ABESS − Lasso128 | 10.97 |
| 丙酮 | dev2 | ABESS − Tensor-GLasso96 | 1.952 |
| 丙酮 | dev2 | ABESS − Lasso96 | 3.3 |
| 丙酮 | dev2 | ABESS − Ridge96 | 3.404 |
| 丙酮 | dev2 | ABESS − PLS96 | 3.589 |
| 丙酮 | dev2 | ABESS − Ridge128 | -3.106 |
| 丙酮 | dev2 | ABESS − Lasso128 | -0.5766 |
| 丙酮 | dev3 | ABESS − Tensor-GLasso96 | -13.36 |
| 丙酮 | dev3 | ABESS − Lasso96 | 10.92 |
| 丙酮 | dev3 | ABESS − Ridge96 | 11.57 |
| 丙酮 | dev3 | ABESS − PLS96 | 12.65 |
| 丙酮 | dev3 | ABESS − Ridge128 | -346.9 |
| 丙酮 | dev3 | ABESS − Lasso128 | -51.59 |
| 丙酮 | final | ABESS − Tensor-GLasso96 | 22.55 |
| 丙酮 | final | ABESS − Lasso96 | -0.3916 |
| 丙酮 | final | ABESS − Ridge96 | 6.462 |
| 丙酮 | final | ABESS − PLS96 | 27.48 |
| 丙酮 | final | ABESS − Ridge128 | 52.84 |
| 丙酮 | final | ABESS − Lasso128 | 66.6 |
| 甲苯 | dev1 | ABESS − Tensor-GLasso96 | -3.019 |
| 甲苯 | dev1 | ABESS − Lasso96 | 0.7613 |
| 甲苯 | dev1 | ABESS − Ridge96 | 0.3133 |
| 甲苯 | dev1 | ABESS − PLS96 | -0.3989 |
| 甲苯 | dev1 | ABESS − Ridge128 | -0.4992 |
| 甲苯 | dev1 | ABESS − Lasso128 | -0.2039 |
| 甲苯 | dev2 | ABESS − Tensor-GLasso96 | -58.98 |
| 甲苯 | dev2 | ABESS − Lasso96 | -19.71 |
| 甲苯 | dev2 | ABESS − Ridge96 | -9.393 |
| 甲苯 | dev2 | ABESS − PLS96 | -9.847 |
| 甲苯 | dev2 | ABESS − Ridge128 | -11.63 |
| 甲苯 | dev2 | ABESS − Lasso128 | -19.71 |
| 甲苯 | dev3 | ABESS − Tensor-GLasso96 | -7.938 |
| 甲苯 | dev3 | ABESS − Lasso96 | -31.59 |
| 甲苯 | dev3 | ABESS − Ridge96 | -11.24 |
| 甲苯 | dev3 | ABESS − PLS96 | -21.61 |
| 甲苯 | dev3 | ABESS − Ridge128 | -26.92 |
| 甲苯 | dev3 | ABESS − Lasso128 | -31.59 |
| 甲苯 | final | ABESS − Tensor-GLasso96 | 3.713 |
| 甲苯 | final | ABESS − Lasso96 | 3.333 |
| 甲苯 | final | ABESS − Ridge96 | 2.269 |
| 甲苯 | final | ABESS − PLS96 | 4.093 |
| 甲苯 | final | ABESS − Ridge128 | 0.6707 |
| 甲苯 | final | ABESS − Lasso128 | 0.3985 |

## 舍弃32个稳态量的代价

正值表示完整128维较好，负值表示当前96维较好；未以此重选输入。

| 气体 | 96维模型 | 96−128 RMSE |
|---|---|---|
| 乙醇 | Ridge96 | -5.119 |
| 乙醇 | Lasso96 | -28.97 |
| 乙烯 | Ridge96 | 268 |
| 乙烯 | Lasso96 | -66.16 |
| 氨 | Ridge96 | 112.1 |
| 氨 | Lasso96 | 218.8 |
| 乙醛 | Ridge96 | 9.352 |
| 乙醛 | Lasso96 | 10.41 |
| 丙酮 | Ridge96 | 46.38 |
| 丙酮 | Lasso96 | 66.99 |
| 甲苯 | Ridge96 | -1.598 |
| 甲苯 | Lasso96 | -2.935 |

范围内/低于/高于训练范围的全部计数与误差见range_metrics.csv；缺记录的分层指标为NA。只有一个final批次，不用零SD伪装零不确定性。批次变化同时混合浓度覆盖、响应漂移和模型限制，本设计不能作因果分解。
