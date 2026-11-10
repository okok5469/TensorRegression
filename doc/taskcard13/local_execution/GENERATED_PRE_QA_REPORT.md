# Apple 实证最终报告

状态：MODELS_COMPLETE_PENDING_FIGURE_QA（等待独立渲染验收）。

1. 数据对齐：240条图像记录、240个评价实体；严格按ID连接，响应以干物质百分点计；完整依据及实体/批次可核验边界见DATA_AUDIT.md。
2. 当前表征的预测信息：三个重复RMSE均低于训练均值的学习模型为 M3, M4；这仅是当前重复内部CV协议下的证据。
3. v5.2表现：完整OOF RMSE为0.8383 ± 0.0118个百分点；全部基线与配对差异如下，原MSIC未调整。
4. 空间价值：M5由同一张量空间平均得到，M6保留141个原始波段；二者分别属于同表示消融和不同分辨率强基线，成绩差异不支持空间因果解释或空间信息不存在的断言。
5. 选阶风险：q范围1–2，均值1.07；达到q上限90%的外折0/15，秩亏外折0/15；各模式平均Jaccard为0.295, 0.705, 0.486。
6. 论文应用建议：原MSIC模型在强基线比较中存在明显局限，Apple宜暂作补充或边界案例；不据此修改算法。

## 完整OOF结果

|模型|RMSE（百分点）|MAE（百分点）|R²|
|---|---:|---:|---:|
|M0|0.8456 ± 0.0025|0.6930 ± 0.0013|-0.0087 ± 0.0059|
|M1|0.8383 ± 0.0118|0.6990 ± 0.0086|0.0086 ± 0.0278|
|M2|0.8383 ± 0.0166|0.6953 ± 0.0180|0.0082 ± 0.0394|
|M3|0.8389 ± 0.0075|0.6871 ± 0.0050|0.0072 ± 0.0176|
|M4|0.8270 ± 0.0068|0.6788 ± 0.0034|0.0350 ± 0.0158|
|M5|0.8410 ± 0.0148|0.6858 ± 0.0088|0.0020 ± 0.0349|
|M6|0.8466 ± 0.0087|0.6934 ± 0.0068|-0.0111 ± 0.0206|

每个重复先拼齐5折OOF、按实体等权计算指标，再对3个重复计算均值和样本标准差；不是折R²平均，也没有把720次预测视为720个独立苹果。

## 与原MSIC的配对比较

|对照|M1−对照 MSE|描述性95%区间|M1−对照 RMSE|描述性95%区间|
|---|---:|---|---:|---|
|M0|-0.0122|[-0.0595, 0.0344]|-0.0073|[-0.0353, 0.0212]|
|M2|-0.0002|[-0.0223, 0.0242]|-0.0001|[-0.0132, 0.0143]|
|M3|-0.0010|[-0.0307, 0.0279]|-0.0006|[-0.0179, 0.0168]|
|M4|0.0188|[-0.0121, 0.0519]|0.0112|[-0.0072, 0.0312]|
|M5|-0.0046|[-0.0565, 0.0450]|-0.0027|[-0.0330, 0.0264]|
|M6|-0.0140|[-0.0698, 0.0399]|-0.0083|[-0.0414, 0.0235]|

负值表示M1误差较低。2000次实体整组重采样保留其全部重复预测；区间条件于已保存OOF，没有重新训练，不能代表完整训练不确定性或普适优越性。跨重复方向见repeat_metrics.csv。

## 数值、解释及证据边界

所有105个外层模型保存后独立加载重建预测通过rtol=1e-9、atol=1e-10检查。显式不收敛最终拟合0项；M1/M2原求解器未暴露的完整收敛状态如实记为null，不能解读为已证明收敛；内调参警告全部保留在tuning.json。
q是矩形支撑参数容量，不等于严格非零系数数；支撑允许不连续，选择频率和重叠仅描述稳定性。没有像素级响应真值，图像强度或系数都不是局部干物质含量地图。
单样本ROI归一化位置只是相对坐标，不保证苹果组织间生物学对齐；未依据误差筛样、调ROI或选最好重复。全部真实值—预测图固定第1重复。
外部训练折标准化控制X及y数值尺度并完成截距还原；核心存在绝对损失阈值，不宣称任意y缩放不变。
这是本数据集内部重复交叉验证，不是全新外部数据验证。真实数据未计算Exact、REE或真支撑恢复率。

## 图表

- [response_distribution](figures/response_distribution.png)
- [A1_mean_intensity](figures/A1_mean_intensity.png)
- [A1_band_750nm](figures/A1_band_750nm.png)
- [A120_mean_intensity](figures/A120_mean_intensity.png)
- [A120_band_750nm](figures/A120_band_750nm.png)
- [native_spectra](figures/native_spectra.png)
- [complete_oof_rmse](figures/complete_oof_rmse.png)
- [complete_oof_mae](figures/complete_oof_mae.png)
- [complete_oof_r2](figures/complete_oof_r2.png)
- [prediction_repeat1_M0](figures/prediction_repeat1_M0.png)
- [prediction_repeat1_M1](figures/prediction_repeat1_M1.png)
- [prediction_repeat1_M2](figures/prediction_repeat1_M2.png)
- [prediction_repeat1_M3](figures/prediction_repeat1_M3.png)
- [prediction_repeat1_M4](figures/prediction_repeat1_M4.png)
- [prediction_repeat1_M5](figures/prediction_repeat1_M5.png)
- [prediction_repeat1_M6](figures/prediction_repeat1_M6.png)
- [abess_capacity](figures/abess_capacity.png)
- [abess_train_test](figures/abess_train_test.png)
- [support_frequency_mode1](figures/support_frequency_mode1.png)
- [support_frequency_mode2](figures/support_frequency_mode2.png)
- [support_frequency_mode3](figures/support_frequency_mode3.png)

## 复现

运行命令与冻结配置见同目录config.yaml、FROZEN_PROTOCOL.md及模块README.md；所有基线内层独立拟合预处理，正式源码身份逐任务检查，缓存仅接受相同数据、划分、预处理、模型与源码指纹。
