# Apple 实证最终报告

状态：**COMPLETE**。本机完成3次重复×5折、7种方法的全部105个外层最终模型及2520个内层候选拟合，保存5040条折外预测；模型、汇总及初版图表生成耗时2596.9秒（约43.3分钟），另计最终独立核验与排版QA。

1. **数据是否可靠对齐？** 240条图像按ID与响应一一对应，240个样本ID作为评价实体；两输入官方MD5通过，无缺失、重复图像或排除。响应从质量分数乘100转为干物质百分点。文件缺少更细果实身份/重复视角/批次信息，物理独立性未获进一步证实，此为评价局限，详见DATA_AUDIT.md。
2. **当前表征有预测信息吗？** Vector Lasso和Ridge的三个重复RMSE均低于训练均值，但提升很小；Ridge平均R²仅0.0350，其余学习方法平均R²接近0或略负。当前协议下预测信号弱，不能据此推断干物质在其他表征或设计下不可预测。
3. **v5.2表现如何？** 原MSIC Tensor-ABESS平均RMSE为0.8383±0.0118个百分点、R²为0.0086±0.0278；均值基线RMSE0.8456，Ridge0.8270。ABESS对均值的RMSE仅改善约0.87%，并非每个重复均改善；与全部6个对照的实体bootstrap配对差异区间均跨0，未建立明确优势。对各对照的逐重复方向与区间见下表。
4. **空间维度是否值得保留？** 展平张量Ridge的均值略优于两个平均谱PLS，但整体R²很低，ABESS对PLS的差异也不确定，现有结果不足以确认保留空间维度有可靠收益。M5是同一张量空间平均的24维消融，M6为141维原始光谱补充基线；不同输入分辨率和不同模型归纳偏好均影响比较，不能作空间因果解释或宣称空间信息不存在。
5. **选阶是否出现已知风险？** 15折中14折q=1、1折q=2；caps均为(5,5,5)，q上限134，未接近容量上限，OLS均满秩。平均训练折RMSE0.8056、平均测试折RMSE0.8338，逐折值完整保留；主指标仍按每次完整OOF计算。行/列/波组平均Jaccard为0.295/0.705/0.486，空间行选择稳定性较低。小q和弱预测是统计表现，真实支撑未知，不能称为已证明欠选或恢复失败；独立工程核验通过，未修改选择规则。
6. **Apple能否作为论文应用？** 当前不适合作为Tensor-ABESS预测优势的主要应用；可作为补充或方法边界案例，诚实呈现所有方法接近均值及不确定性。是否纳入论文属于后续写作判断，本任务不因此修改算法、响应、ROI或筛选样本。

模型对应：M0训练均值；M1原MSIC Tensor-ABESS；M2独立Tensor-GLasso；M3 Vector Lasso；M4 Ridge；M5同表示平均谱PLS；M6原始141波段平均谱PLS。

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

- [response_distribution](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/response_distribution.png)
- [A1_mean_intensity](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/A1_mean_intensity.png)
- [A1_band_750nm](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/A1_band_750nm.png)
- [A120_mean_intensity](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/A120_mean_intensity.png)
- [A120_band_750nm](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/A120_band_750nm.png)
- [native_spectra](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/native_spectra.png)
- [complete_oof_rmse](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/complete_oof_rmse.png)
- [complete_oof_mae](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/complete_oof_mae.png)
- [complete_oof_r2](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/complete_oof_r2.png)
- [prediction_repeat1_M0](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M0.png)
- [prediction_repeat1_M1](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M1.png)
- [prediction_repeat1_M2](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M2.png)
- [prediction_repeat1_M3](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M3.png)
- [prediction_repeat1_M4](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M4.png)
- [prediction_repeat1_M5](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M5.png)
- [prediction_repeat1_M6](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/prediction_repeat1_M6.png)
- [abess_capacity](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/abess_capacity.png)
- [abess_train_test](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/abess_train_test.png)
- [support_frequency_mode1](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/support_frequency_mode1.png)
- [support_frequency_mode2](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/support_frequency_mode2.png)
- [support_frequency_mode3](../../empirical_analysis/spectrofood_apple/outputs/apple_v1/figures/support_frequency_mode3.png)

## 复现

运行命令与冻结配置见同目录config.yaml、FROZEN_PROTOCOL.md及模块README.md；所有基线内层独立拟合预处理，正式源码身份逐任务检查，缓存仅接受相同数据、划分、预处理、模型与源码指纹。

## 跨重复方向及数值收敛限制

下表按每个重复的完整OOF RMSE比较，胜表示M1误差较低；3次重复共享同一批实体，不是3次独立外部验证。

|对照|M1胜/负/平|重复1差值|重复2差值|重复3差值|
|---|---|---:|---:|---:|
|训练均值|2/1/0|-0.00337|0.00269|-0.02129|
|Tensor-GLasso|2/1/0|0.01508|-0.00720|-0.00816|
|Vector Lasso|1/2/0|0.00889|0.00812|-0.01884|
|Ridge|1/2/0|0.01964|0.01636|-0.00228|
|PLS-binned|2/1/0|-0.00629|0.02501|-0.02686|
|PLS-native|2/1/0|-0.01032|0.01266|-0.02727|

Lasso内层585个拟合中119个未达到适配器收敛判据，其中118次产生ConvergenceWarning；全部位于未选中的候选。最终选中参数对应45个内层拟合及15个最终外层模型均收敛且无警告。这个问题限制了整个候选路径已充分优化的主张，不能用最终模型收敛掩盖；原始warning与dual gap完整保留，未因结果好坏调网格、换参数或重做不利折。两个PLS各315个内层拟合均收敛；M1及M2原实现未暴露完整求解收敛状态，仍如实标为未知，不把unknown写成成功。

## 独立验收与复现

105个完成标记、420个模型任务工件哈希、5040条OOF、所有训练缩放器、独立预测还原、15折OLS正规方程与MSIC分项、50个执行源码及32个正式发布源码身份均通过独立只读核验。完整OOF与summary已独立复算；不存在失败任务或漏折。21张图逐图查看，PDF字号和碰撞审计通过；3张指标图仅修正方法标签排版，原图和QA失败记录保留，预测、指标及冻结源码未改。

本机最多2worker、每worker BLAS/OMP1线程。先前小样本三次方外推过于保守；用户明确指定本机执行后，使用同一冻结worker完成了全量实验，实际耗时小于原7200秒预算。先前集群包保留为历史准备材料，未提交、未参与正式结果。

关键交付：[summary_metrics.csv](summary_metrics.csv)、[oof_predictions.csv](oof_predictions.csv)、[FINAL_VALIDATION.json](FINAL_VALIDATION.json)。审计：[独立最终核验](local_execution/INDEPENDENT_FINAL_AUDIT.json)、[图表QA](local_execution/FIGURE_QA.json)。原始输入身份与单位证据见[DATA_AUDIT.md](DATA_AUDIT.md)。
