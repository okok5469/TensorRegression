# LEMON EEG × Tensor-ABESS

独立实证模块：从 BrainVision 文件构造每人 `channel × 16 × 5` 的 log 频带功率张量，在外层训练折内拟合协变量回归和标准化，调用现有 v5 Tensor-ABESS、Tensor-GLasso、Vector Lasso，保存 OOF 评价与最终解释模型。

## 环境与入口

从仓库根目录运行；本模块不修改核心包的安装布局。

```powershell
# 如尚无本地环境，创建后安装所需包。
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r empirical_analysis/lemon/requirements.txt
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m empirical_analysis.lemon --help
```

本次开发环境使用 `.venv` 并启用 `--system-site-packages`，复用当前系统已有的六项运行依赖；全新机器按上述命令独立安装即可。
本流程调用的三个终版 NumPy 方法不需要 sklearn；仓库其他示例或 `pip install -e .` 则遵循根 `pyproject.toml` 的依赖声明。

## 首先编辑配置

打开 [config.yaml](config.yaml)：

- 所有相对路径都以该 YAML 所在目录为基准，与 PowerShell 当前工作目录无关。
- `data.eeg_root` 指向含 `sub-*/eeg/` 的完整数据根目录；数据可以留在仓库外。
- `data.manifest_file` 指向下载清单；只有实际不存在清单时才明确设为 `null`。
- `data.subject_selection_file` 默认指向下载器生成的150人计划，实证仅处理该名单；明确分析全部本地受试者时才设为 `null`，未被选择者单独记录为 `subjects_not_selected.csv`，不冒充质量排除。
- `data.participants_file`、`data.participants_json` 分别指定年龄/性别表和编码说明。
- `confounds` 中填写经过编码说明核实的 ID/年龄/性别列名和 `sex_mapping`；例如只有说明明确编码时才可填写 `{"M": 0, "F": 1}`，不能从数字大小猜测性别。
- 当前年龄设计要求数值年龄；年龄区间将明确停止，不能擅自转换为区间中点。
- 默认 `analysis.mode: adjusted`；未经调整的探索需显式改为 `pilot`，其结果会标记 `PILOT_UNADJUSTED`。
- 两种模式均执行 `analysis.min_samples`（默认100）及每折可识别性检查；小样本 pilot 需明确设定适当阈值，不能自动绕过。
- 主响应为 `log(TMT_5)`；原始秒数敏感性分析需用 `primary_transform: raw` 和另一个空 `output_dir`。
- EC/EO 不由全局猜测的编号控制，每个人都通过自己的 EEG JSON、events TSV/JSON 与 VMRK 交叉验证。

## 下载前150名受试者

先从本地清单生成计划，不访问网络：

```powershell
.venv/Scripts/python.exe -m empirical_analysis.lemon.download --manifest 实证分析/manifest.json --destination 实证分析/nm000179 --limit 150
```

之后需要下载时，在同一命令末尾加 `--download`；脚本用清单中的稳定 `bytes_url`，不使用已过期的临时签名链接。
按数值受试者 ID 排序，在清单中排除缺必需文件、条目歧义或校验信息异常的受试者后选前150名；这属于下载前文件检查，不代表 EEG 信号质量已通过。
每人下载 `.eeg/.vhdr/.vmrk`、标准 `events.tsv`、清单中存在的 `channels.tsv`，并补齐流程必需的 `eeg.json/events.json`；共享的 `participants.tsv/json`、`dataset_description.json` 若在清单中也一并下载。
计划默认保存到 `outputs_download/plan.json`，排除理由逐人记录；`--plan` 可以另行指定保存路径。
已存在且大小、清单校验值正确的文件会跳过；未完成文件保留 `.part` 并以 HTTP Range 继续，只有完整校验通过才成为正式文件。
已有正式文件校验失败时明确停止，不覆盖用户文件；服务器不支持 Range 时会明确记录从头重新获取 `.part`，不会把完整响应追加到旧片段。
默认连接/读取超时为180秒；每个文件首次尝试失败后，最多额外重试5次（总计最多6次尝试），依次等待2、4、8、16、32秒。保持单文件顺序下载，不启动并行传输。
仅超时、暂时断连/连接故障、临时DNS失败、响应截断及HTTP 408/429/500/502/503/504会重试；永久HTTP错误、证书错误、磁盘/权限错误、文件校验或响应范围/长度验证失败立即停止。重试重新读取`.part`的实际长度，继续发送Range请求；达到上限后保留`.part`，不会越过失败文件继续下载。
控制台显示当前文件、已接收字节/总字节及百分比、重试次数和等待秒数；进度显示100%只表示字节已到齐，随后出现`VERIFIED`才表示校验完成，`skipped`表示既有正式文件校验通过。即使转正瞬间出现同名正式文件，也会停止并保留`.part`。
`download_log.json`记录本次每个文件的最终状态、`attempts`和`retry_history`（次数、等待及脱敏原因）；重试过程不记录下载URL。日志每次运行按本次尝试重新生成，既有有效文件会重新校验并记录为跳过。

在本项目中继续下载到D盘，可在PowerShell复制执行：

```powershell
Set-Location -LiteralPath 'C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main'
.\.venv\Scripts\python.exe -m empirical_analysis.lemon.download `
  --manifest '实证分析/manifest.json' `
  --destination 'D:\数据集\LEMON原始EEG' --limit 150 `
  --plan 'empirical_analysis/lemon/outputs_download/plan.json' `
  --log 'empirical_analysis/lemon/outputs_download/download_log.json' `
  --timeout 180 --max-retries 5 --retry-delay 2 --download
```

`--max-retries`可设0–5，0表示不自动重试；`--retry-delay`指定首轮等待，后续逐次加倍；超时和等待参数必须为有限正数。不加`--download`仍只生成计划；上述续传命令不会启动分析阶段。

## 分阶段运行

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage inventory
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage tmt
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage events
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage smoke
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage tensors
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage dataset
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage cv
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage final
.venv/Scripts/python.exe -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage report
```

`--stage all` 按同一顺序执行，任何失败立即停止；不会跳过未通过的阶段。
`events` 默认要求至少5名文件完整的受试者，逐人核对事件；`smoke` 对参考受试者重复构造并检查数值一致性；批量处理按报告记录文件、窗口、通道和数值失败。

已完成的阶段只有在输入、配置、源码及其产物指纹均一致时才会复用；改变输入、修订代码、切换分析设定或修复失败的模型运行后，请使用新的空输出目录从盘点重新运行，不覆盖原结果。
原始 EEG 当前核查文件大小及修改时间，未做全量 SHA256；此边界写入运行配置和报告。

## 输出

所有运行产物位于配置指定的 `output_dir`：

| 路径 | 内容 |
|---|---|
| `config_used.json`, `run_state.json` | 解析后路径、配置、软件版本、输入/源码指纹和阶段状态 |
| `diagnostics/` | 文件盘点、TMT 原始异常、描述统计、协变量编码诊断、逐人事件证据、共同通道 |
| `tensors/subjects/` | 单人张量和包含实际窗口时间、映射证据的元数据 |
| `tensors/X_unstandardized.npy` | 未经全样本标准化的汇总张量，另存 subject IDs 和轴标签 |
| `qc/analysis_dataset.npz` | 最终严格对齐的 X、y、subject_ids；adjusted 另含 C，pilot 不含 C |
| `cv/` | 固定折、逐人逐模型 OOF、逐折指标、汇总指标、三个模态的5折选择频率 |
| `cv/folds/fold_*/preprocessing.npz` | 同一折三模型共用的训练预处理参数，旁边保存 train/test IDs |
| `cv/folds/fold_*/<model>/` | `B.npy`、`support.json` 和拟合统计 |
| `final_model/` | 全样本解释模型、支撑索引/名称、预处理参数和实际窗口时间摘要 |
| `figures/`, `RUN_SUMMARY.md` | 数据诊断、OOF 比较、支撑解释图及报告；不是自动生成的论文结论 |
| `logs/*_error.json` | 阶段运行错误；初始化时的配置/指纹冲突直接由 CLI 报错 |

系数 `B` 位于残差标准化特征空间；复现预测必须同时加载 `preprocessing.npz`，并加回训练折估计的响应中心与 nuisance 预测。
OOF 预测和误差在所选 log 或 raw 响应尺度报告；不把 `exp(log预测)` 当成未经校正的条件均值。
最终模型不提供独立测试性能；支撑频率是5折选择频率，不是 bootstrap 稳定性。
通道图显式采用按名称排列的条形图，未伪造电极位置。

## 验证

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_lemon_*.py'
```

测试使用独立合成数据，覆盖原始文件读入、事件证据、窗口边界、异常值、折内预处理、三方法拟合、OOF 复现及阶段阻断；测试产物不作为真实数据实证结果。
完整研究要求见 [执行文档](../../实证分析/LEMON_Tensor_ABESS_Codex_执行文档.md)。
