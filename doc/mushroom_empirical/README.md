# Mushroom任务卡15复现入口

本目录保存真实三阶张量、冻结协议、160个外层模型及其调参记录、全部预测和独立验收材料。正式算法是v5.2-search-final原MSIC，原Apple及CSV预检结果保持独立。

## 阅读

- [完整数据审计](MUSHROOM_DATA_AUDIT.md)：六文件身份、250个ID、全部像素QC、重复谱与文件块含义。
- [张量构建](TENSOR_CONSTRUCTION_REPORT.md)：固定全ROI面积8×8、相邻204→24波组，非响应驱动的轻量缓存。
- [冻结协议](FROZEN_EMPIRICAL_PROTOCOL.md)、[机器契约](contract.json)、[配置](config.json)、[划分](splits.json)。
- [Pilot正确性及成本](PILOT_AND_CORRECTNESS.md)、[内存实测](MEMORY_PROBE.json)。
- [完整实证报告](MUSHROOM_EMPIRICAL_FINAL_REPORT.md)、[文件分块压力测试](BLOCK_GENERALIZATION_REPORT.md)。仅FINAL_VALIDATION标记COMPLETE且160任务验收通过时，才代表完整结果。
- [最终验收](FINAL_VALIDATION.json)及交付索引；中断历史另保留，不作为最终状态。

## 方法编号

|编号|方法|输入|
|---|---|---|
|M0|训练均值|当前训练响应|
|M1|Tensor-ABESS v5.2原MSIC|8×8×24张量|
|M2|Tensor-GLasso|同一张量|
|M3|Vector Lasso|同一张量C-order展平|
|M4|Ridge|同一张量C-order展平|
|M5|PLS-binned|同一张量空间均值24维|
|M6|Ridge-native-spectrum|原CSV204维平均谱|
|M7|PLS-native-spectrum|原CSV204维平均谱|

## 本机命令

在仓库根目录PowerShell执行。已有隔离运行依赖位于先前授权的task13/runtime，numpy/scipy及算法未升级。

```powershell
$env:PYTHONPATH='doc/taskcard13/runtime;src;.'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mushroom_empirical*.py'
.venv/Scripts/python.exe doc/mushroom_empirical/local_execute.py --phase full
.venv/Scripts/python.exe doc/mushroom_empirical/validate_models.py --all
.venv/Scripts/python.exe doc/mushroom_empirical/analyze_results.py --root doc/mushroom_empirical
.venv/Scripts/python.exe doc/mushroom_empirical/plot_results.py
.venv/Scripts/python.exe doc/mushroom_empirical/figure_qa.py
```

报告的独立科学复核后执行`finalize_scientific_review.py`，保留定量结论和B类理由；图表逐张人工审阅并写入FIGURE_VISUAL_REVIEW.json后，执行`figure_qa.py --finalize`。所有模型、原数据身份和图表验收齐全后才可执行`seal_delivery.py`生成最终验收与交付索引。图或数据改变时不能复用旧QA哈希。

控制器使用2个worker、每个1线程；已完成任务只在契约、原生环境、四个产物SHA256及保存模型重放全部通过时复用。发现不完整非空任务目录需先工程复核，不改参数或自动忽略失败。单controller.lock阻止重复控制器；不向集群提交。

运行进度读取`execution/status_events/`按文件名最后一个JSON。每个快照独立、先完成写入再首次发布，不覆盖被Windows读者持有的文件；旧`execution/STATUS.json`是首次中断历史，已不更新。报告与图不参与冻结模型源指纹，不改变任何预测。

原始缓存构建器为`empirical_analysis/spectrofood_mushroom/data.py`，命令参数为`--data-dir`及`--out-dir`；重建检查应输出到独立临时目录，避免覆盖已发布缓存。它重新校验全部官方MD5，逐立方体读取，不把整个10.3GB同时载入内存。

## 表格与证据边界

main：每模型3×250条OOF，先计算每个完整重复的指标，再汇总均值及样本SD；block：每模型250条留文件OOF拼合，另外保留全部5块RMSE/MAE/R²。总8000行不是8000个独立蘑菇。

`tasks/{protocol}_r{repeat}_f{fold}_{model}/`保留model.pkl、predictions.csv、diagnostics.json、cv_records.json及COMPLETE.json；完成标记记录契约、环境和四个产物哈希。全量独立验收重新检查全部缩放器、预测还原、支撑OLS/MSIC及内层警告；并非仅凭文件存在验收。

配对区间条件于保存OOF，按243个已知相同CSV谱组重采样，保留全部冲突标签和3个重复；未重拟合，且物理独立性未核验。文件块为存储划分，不能称已知批次。空间相对坐标不代表解剖对应，支撑频率不代表因果组织。完整数值警告、选阶与训练测试差异必须随最终结果保留。
