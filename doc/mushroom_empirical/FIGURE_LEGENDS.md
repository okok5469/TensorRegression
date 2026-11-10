# 图表说明与源数据

全部图使用真实保存结果，无筛样或误差裁剪；PDF/SVG为可编辑矢量导出，PNG是300dpi审阅预览。单图宽度不超过180mm、文字8pt以上；当前未指定期刊生产版尺寸，未把PNG当作TIFF投稿文件。所有图均为单绘图区，跨panel对齐不适用。

M0训练均值；M1 Tensor-ABESS；M2 Tensor-GLasso；M3 Vector Lasso；M4展平张量Ridge；M5张量空间均值PLS；M6原CSV204波段Ridge；M7原CSV204波段PLS。更具体的统计和收敛边界见[最终报告](MUSHROOM_EMPIRICAL_FINAL_REPORT.md)。

|图名|证据与统计定义|源数据|
|---|---|---|
|main_rmse/main_r2|全部8方法，3个完整重复的OOF指标，点为各重复、中心及误差线为均值±样本SD；每重复250记录，按已知相同谱分组防泄漏。R²由完整250个预测计算，不平均折R²。|repeat_metrics.csv|
|block_rmse/block_r2|每方法全部5个留文件块的测试指标，连线与不同符号用于辨认方法；描述性，不含IID置信区间。极负R²完整保留。|fold_metrics.csv|
|block_response|每块全部50响应的箱线图；箱体25%–75%、中线中位数、须按1.5IQR、离群点保留。文件块是存储文件，不是已确认批次。|data.npz的y_pct/blocks|
|prediction_main_M0至M7|固定第1个随机分组CV重复，每图完整250条OOF；虚线为y=x参考。无挑选最佳重复、无逐记录置信区间。|oof_predictions.csv|
|prediction_block_M0至M7|每图完整250条留文件OOF；虚线为y=x。轴范围包含全部真实值和预测，方法之间坐标范围可能不同，以免裁去极端预测。|oof_predictions.csv|
|abess_q|随机协议15个与分块协议5个外层模型的所选矩形容量q；展示全部20次拟合，q不等于严格非零系数数。|abess_supports.csv|
|support_mode1/support_mode2|8个相对ROI行/列中心的选择频率；随机15次与分块5次分别计算。仅描述条件稳定性，无组织因果解释。|support_frequency.csv|
|support_mode3|24个物理波组代表波长的选择频率；每波组完整原索引及波长范围另保留，不作插值或光谱区间精确定位。|support_frequency.csv、wavelength_groups.json|
|paired_rmse|ABESS减去各对照的平均重复OOF RMSE差与条件性95%谱簇bootstrap区间；负值表示ABESS误差更小。2000次重采样243已知谱组，每组全部记录与3个重复整体保留，记录加权，未重拟合。无显著性星号或多重检验结论。|paired_error_differences.csv|

250条记录的独立物理实体与采集批次未建立，243个谱簇不能视作243个经核验独立蘑菇。重复CV的750条预测不能作为750个独立样本；仅5个文件块的泛化压力测试不提供普通IID误差条。
