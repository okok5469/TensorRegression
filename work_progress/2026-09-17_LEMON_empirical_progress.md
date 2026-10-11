# 2026-09-17 工作进展：LEMON 实证分析完成与输出整理

## 一、完成状态

已完成本轮 LEMON 真实数据实证分析：原始数据下载、事件核验、EEG 预处理、张量构造、年龄段与性别接入、五折交叉验证、最终模型及结果报告均已完成。

当前正式运行目录为 `empirical_analysis/lemon/outputs_lemon_adjusted/`，其 `run_state.json` 已记录九个阶段全部完成：

```text
inventory → tmt → events → smoke → tensors → dataset → cv → final → report
```

最终为 145 名受试者，输入张量维度为 `(145, 61, 16, 5)`，分别对应受试者、EEG 通道、闭眼窗口和频段；分析模式为 `adjusted`，响应为 `log(TMT_5)`，没有采用未调整试跑模式。

## 二、运行环境、下载与目录调整

1. 复用仓库 `.venv`，Python 3.12.8；验证 NumPy、SciPy、pandas、MNE、matplotlib、PyYAML 等所需依赖，运行时将根目录 `src` 加入 `PYTHONPATH`。
2. 从 manifest 重建名单并核对既有下载计划，选取 150 人；`sub-010203` 因缺少标准事件文件未进入下载计划。
3. 下载总计 1,053 个文件（150 人每人 7 个必要文件及 3 个共享元数据），计划大小 96,308,960,326 字节；`outputs_download/download_log.json` 的 `complete` 为 `true`，共享 `participants.tsv/json` 已取得。
4. 下载器保持单文件顺序执行，使用稳定 `bytes_url`，支持断点续传和清单校验；默认超时 180 秒，最多重试 5 次，等待 2、4、8、16、32 秒，只重试暂时性网络故障；显示当前文件、进度及重试状态，保留未完成 `.part`，通过校验的正式文件跳过，正式文件校验失败立即停止且不覆盖。
5. 正式 EEG 根目录已改为 `D:\数据集\LEMON原始EEG`；执行文档、`manifest.json`、`TMT.csv` 已迁至 `empirical_analysis/`，主配置已修正相对路径，不再依赖已移走的早期五人原始数据目录。

原始下载的共享人口学文件虽然存在，但年龄和性别为缺失值，因此后续使用官方人口学表补齐，未把文件存在误认为协变量齐全。

## 三、五人真实数据试跑与资源记录

2026-09-14 对 `sub-010002` 至 `sub-010006` 完成真实数据试跑，5 人均成功，张量 `(5, 61, 16, 5)` 全部有限；35 个输入计划文件已按清单校验。

| 阶段 | 耗时（秒） | 峰值进程驻留内存（MiB） |
|---|---:|---:|
| 5 人事件验证 | 0.381738 | 82.933594 |
| 单人完整预处理及两次一致性核验 | 16.960579 | 248.960938 |
| 5 人张量构造 | 37.718937 | 250.839844 |

上述三阶段合计 55.061254 秒；含盘点和 TMT 前置步骤合计 55.493547 秒。单人耗时包含两次完整计算；Windows 峰值驻留内存包含进程导入，不包括系统文件缓存，试跑前校验过输入，因此不是冷缓存基准，也不是 150 人耗时估计。

详细记录保留于 `empirical_analysis/lemon/outputs_real_smoke_20260914_5subjects/REAL_SMOKE_REPORT.md`，包括资源明细、校验来源和逐阶段性能文件；其独有审计证据仍有用途。

## 四、预处理与样本筛选

逐人结合 JSON、标准 `events.tsv`、VHDR 和 VMRK 验证事件证据；闭眼连续段两端各裁 5 秒，按时间顺序选取 16 个不重叠的 25 秒窗口，每人处理 400 秒信号。每个窗口执行 1–45 Hz 带通、重采样至 250 Hz、EEG 平均参考、Welch 功率谱（4 秒分段、50% 重叠、median），再计算 delta/theta/alpha/beta/gamma 的积分及 log 功率；VEOG 不进入 EEG 张量。

| 步骤 | 人数 | 依据 |
|---|---:|---|
| 原始下载计划 | 150 | 固定 manifest 计划 |
| 事件问题明确排除后 | 149 | 用户同意排除 `sub-010126`：说明文件闭眼码为 S210，实际事件为 S208，未擅自替换 |
| 张量质量控制通过 | 146 | `sub-010015`、`sub-010078`、`sub-010100` 分别只有 6、15、10 个可用窗口，低于要求的 16 个 |
| TMT 与人口学对齐后 | 145 | `sub-010092` 的 TMT_5 为原始文本 `67,75`，按既定有效数值规则排除，未猜测修正 |

149 人选择记录为 `empirical_analysis/lemon/selection_149.json`，原始 150 人下载计划保留。正式运行已验证 149 人事件，单人预处理重复一致，146 人张量完成；TMT 源表 227 行，其中 226 行满足数值且正值规则。

## 五、年龄段和性别资料接入

官方原始人口学表和编码说明保留于 `outputs_demographics/demographics_original.csv` 与 `demographics_codebook.txt`，按受试者 ID 生成 `participants.tsv/json` 并保留来源说明。

年龄原始资料为区间，未使用区间中点冒充精确年龄；经用户确认作为分类变量处理。由于“35–40”和“55–60”在候选样本中各仅一人，经用户允许将相邻年龄段分别合并为“30–40”和“55–65”，避免五折训练集中出现无法估计的类别。

最终年龄类别为 `20–25、25–30、30–40、55–65、65–70、70–75、75–80`，以 `20–25` 为参照；性别遵循官方 `1=女、2=男`，建模编码为 `女=0、男=1`。设计矩阵包括截距、6 个年龄指示变量及性别，共 8 列；五折预检查均为 116 人训练、29 人验证，训练设计矩阵满秩 8。

残差化、中心化及尺度参数仅在各外层训练折拟合，再用于该折验证数据；正式结果与年龄段合并方案均写入运行配置快照。

## 六、测试与真实结果

相关实现采用先补测试再修改的方式，历史验收记录为 102 项 `test_lemon_*.py` 测试全部通过，耗时 26.079 秒，涵盖下载重试、分类年龄设计、折内残差化、缺类别秩不足阻断及合成完整流程；下载器独立测试此前为 48 项通过。本次清理和工作记录不修改算法，也不把历史测试误记为本次重新运行。

真实 145 人、固定种子 `20260908` 的五折 OOF 结果如下，全部指标在 `log(TMT_5)` 尺度：

| 模型 | OOF RMSE | OOF MAE | OOF R² |
|---|---:|---:|---:|
| Tensor-ABESS | 0.32641750 | 0.26710451 | 0.33184108 |
| Vector Lasso | 0.36827013 | 0.29457863 | 0.14951650 |
| Tensor-GLasso | 0.43386929 | 0.33244920 | −0.18045850 |

Tensor-ABESS 在本次固定划分中预测指标最好；最终用全部 145 人拟合的描述性模型选择 `F7、AF8` 两个通道、第 16 个窗口（零基索引 15）及 `gamma` 频段，矩形支撑大小为 `2 × 1 × 1 = 2`。真实窗口时间在 `final_model/selected_time_window_onsets.csv`，不将窗口序号直接换算为全记录中的绝对秒数。

结果边界：五折之间的标准差仅描述划分变动，尚未建立模型间统计显著优势；本轮没有独立外部测试集，全样本最终拟合不能作为独立测试性能，所选通道和频段也不构成因果或机制证据。下载器清单校验与分析流水线的输入检查属于不同步骤，后者没有重新执行全量 EEG SHA256 校验。

本轮 LEMON 工作集中在独立实证模块、配置、测试、下载及文档；没有修改 Tensor-ABESS 核心算法，也没有创建新分支或 worktree。

## 七、输出整理与后续入口

已删除被正式运行替代的 `empirical_analysis/lemon/outputs_lemon_149_20260917/` 以及可再生的 `empirical_analysis/lemon/__pycache__/`。删除前已核验正式九阶段全部产物指纹匹配，并确认新旧 `X_unstandardized.npy` 逐元素一致、`tensor_subject_ids.csv` 和 `subject_qc.csv` 字节一致；原始 EEG、正式结果、下载清单、人口学资料及独有性能审计记录均保留。

保留目录及用途：

| 目录 | 用途 |
|---|---|
| `empirical_analysis/lemon/outputs_lemon_adjusted/` | 当前完整正式实证结果 |
| `empirical_analysis/lemon/outputs_demographics/` | 官方人口学来源、编码和标准化输入 |
| `empirical_analysis/lemon/outputs_download/` | 150 人计划、下载校验日志和来源记录 |
| `empirical_analysis/lemon/outputs_real_smoke_20260914_5subjects/` | 五人性能试跑和审计记录 |

当前主要结果入口（仓库根目录为 `C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main`）：

- 总报告：`empirical_analysis/lemon/outputs_lemon_adjusted/RUN_SUMMARY.md`。
- 完成状态与产物指纹：`empirical_analysis/lemon/outputs_lemon_adjusted/run_state.json`。
- 五折统计：`empirical_analysis/lemon/outputs_lemon_adjusted/cv/cv_summary.csv`、`cv_summary.json`。
- 逐人 OOF 预测：`empirical_analysis/lemon/outputs_lemon_adjusted/cv/oof_predictions.csv`。
- 最终模型与标签：`empirical_analysis/lemon/outputs_lemon_adjusted/final_model/`。
- 图形及来源清单：`empirical_analysis/lemon/outputs_lemon_adjusted/figures/`、`figure_manifest.json`。

本轮既定实证流程已完成，无需再次运行下载、预处理或拟合命令；若继续论文写作或补充稳健性检验，应另行明确范围，避免覆盖这次已完成结果。
