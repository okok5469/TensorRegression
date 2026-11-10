# 北京多站点次日PM2.5实证

本目录保存任务卡16的固定数据、35个时间外推模型及7,336条预测。当前模型与独立重放已完成，最终封存状态以[FINAL_VALIDATION.json](FINAL_VALIDATION.json)为准。冻结标识为`7fee51150815a36fcd72eab256b19fe19825673db31d5a7a5ecce8fcc12d1f3b`；未改正式v5.2原MSIC，也没有根据结果更换站点、时间箱或模型网格。

## 先看结果

- [最终实证报告](BEIJING_EMPIRICAL_FINAL_REPORT.md)、[科学定位](SCIENTIFIC_DECISION.json)、[支撑分析](ABESS_SUPPORT_ANALYSIS.md)。
- [逐折与未来指标](summary_metrics.csv)、[全部预测](predictions.csv)、[逐月指标](monthly_metrics.csv)、[同日期配对差](paired_error_differences.csv)。
- [14图索引](figures/FIGURE_INDEX.json)、[统计图注](FIGURE_LEGENDS.md)、[图形QA](FIGURE_QA.json)。

独立未来352日的ABESS RMSE为47.4525 µg/m³，R²为0.58163；相较真实Persistence RMSE改善30.98%，与单站Ridge的RMSE仅差约−0.0127 µg/m³，不能据此主张明确的多站增益或预测优越性。未来支撑容量为20/432个矩形坐标；变量选择是描述性证据，不是因果解释。

## 固定模型映射

| 编号 | 方法 | 输入 |
|---|---|---|
| M0 | 历史训练均值 | 当前外训练目标均值 |
| M1 | Persistence | 自然前日目标站实际观测PM2.5日均值，不填补 |
| M2 | 单站点Ridge | Aotizhongxin的6污染物×6前日四小时箱，共36特征 |
| M3 | 完整张量Ridge | 12站×6污染物×6时间箱，共432特征 |
| M4 | Vector Lasso | 同一432坐标按C顺序展平 |
| M5 | Tensor-GLasso | 同一三阶张量及既有独立适配器 |
| M6 | Tensor-ABESS | 正式v5.2-search-final、原MSIC自适应选阶 |

本编号与Apple/Mushroom目录不同，不得将M1误读为ABESS。Ridge、Lasso、GLasso只在当前外训练的3个扩展时间内折调参；M6不使用CV替换MSIC。填补、X/y标准化只学习当前训练行。

## 数据与时间边界

数据为[UCI501 Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，Chen, S. (2017)，DOI 10.24432/C5RK5G，CC BY 4.0。原ZIP内12站各35,064小时，来源为2013-03-01至2017-02-28。污染物单位全为µg/m³，包括CO；气象变量不入模型。详见[数据审计](BEIJING_DATA_AUDIT.md)和[构建报告](TENSOR_CONSTRUCTION_REPORT.md)。

原始本地ZIP位于`D:\数据集\beijing+multi+site+air+quality+data\PRSA2017_Data_20130301-20170228.zip`，只读流内CSV，不修改原始文件。ZIP CRC、ZIP及各CSV SHA和时间结构均核验；UCI未发布官方校验和，不能将本地完整性核验称为服务器字节认证。

目标为Aotizhongxin次日实际有效小时PM2.5均值，输入来自自然前一日，而非上一个有效样本日。目标和前日各至少18小时才具共同资格；每输入四小时箱至少2小时，否则保留NaN。全期1,409共同日期，相较只筛目标额外排13日以保留未填补持续性。未来365自然日中352日可评价，另13日形成图中的自然日断点；这与“全期额外排13日”是不同统计口径。

开发外折测试日数为179、163、174、180，合696日，分别来自四个扩展训练窗口。最终训练1,057日，独立未来期2016-03-01至2017-02-28评价352日。模型在各外折固定，测试日允许更新已观测前日输入，不用测试标签重训练。这是逐日更新信息的次日预测，不是一次预测整年；理想日界立即获得前日数据的假设未考虑真实发布延迟。

## 复核与复现命令

从仓库根目录在既有环境执行以下PowerShell命令；使用既有冻结缓存及产物时，任务入口核验并重放已完成模型，不覆盖科学配置。已有`contract.json`时不要重新执行prepare覆盖缓存。

```powershell
.\.venv\Scripts\python.exe doc/beijing_air_quality/independent_validate.py --data
.\.venv\Scripts\python.exe doc/beijing_air_quality/freeze.py
.\.venv\Scripts\python.exe doc/beijing_air_quality/local_execute.py --phase pilot
.\.venv\Scripts\python.exe doc/beijing_air_quality/independent_validate.py --pilot
.\.venv\Scripts\python.exe doc/beijing_air_quality/local_execute.py --phase full
.\.venv\Scripts\python.exe doc/beijing_air_quality/independent_validate.py --all
.\.venv\Scripts\python.exe doc/beijing_air_quality/analyze_results.py
```

首次在无既有冻结结果的新工作副本准备数据时，原始构建入口为：

```powershell
.\.venv\Scripts\python.exe -m empirical_analysis.beijing_air_quality.data --zip-path 'D:/数据集/beijing+multi+site+air+quality+data/PRSA2017_Data_20130301-20170228.zip' --out-dir doc/beijing_air_quality
```

分析脚本要求全部35任务、全部7,336条预测及独立验证PASS后才发布结果。它会重新生成事实报告；科学定位定稿使用：

```powershell
.\.venv\Scripts\python.exe doc/beijing_air_quality/finalize_scientific_review.py
```

如需重新导出图，仅使用既定Python流程；改图后必须重新自动QA并逐图实际审阅，不得沿用旧审阅PASS：

```powershell
.\.venv\Scripts\python.exe doc/beijing_air_quality/plot_results.py
.\.venv\Scripts\python.exe doc/beijing_air_quality/figure_qa.py
```

实际逐图审阅并记录`FIGURE_VISUAL_REVIEW.json`后，才执行`figure_qa.py --finalize`；任何WARN须有对应图的实际解释。该命令不会代替视觉审阅。全部图提供PDF/SVG及300-dpi PNG。

## 解释限制

指标按同日期真实记录计算，bias为预测减观测；开发pooled696属于不同训练窗口，不是单一模型的独立测试。没有普通IID bootstrap、逐日独立假设或显著性声明。支撑部署前频率只来自四开发模型；未来单模型另列，不混入稳定性。零数值警告不等于M5/M6已暴露并确认收敛，unknown状态保留。完整工程、数值和独立验证证据保存在相应JSON及逐任务原始产物中。
