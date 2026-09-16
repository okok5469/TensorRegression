# LEMON EEG × Tensor-ABESS 实证分析执行文档

> 用途：约定现有项目中 LEMON 数据处理、建模、审计与交付方式。代码位于 `empirical_analysis/lemon/`，原始数据保留在 `实证分析/` 或配置指定的下载目录。
> 原则：先盘点与诊断，再做单样本 smoke test、批量张量和建模；Gate 失败须保存诊断并停止依赖该 Gate 的阶段。代码实现完成不代表真实数据已通过验收，也不代表已经得到实证结果。

## 0. 分析目标

每名受试者构造三阶张量：

\[
X_i\in\mathbb{R}^{p_{channel}\times16\times5},\qquad
y_i=\langle X_i,B\rangle+\varepsilon_i.
\]

- Mode 1 为共同 EEG 通道，预计 61 个；`VEOG` 必须标为 EOG 并排除。
- Mode 2 为 Eyes Closed（EC）状态下按原始时间顺序选出的前 16 个合法、不重叠的 25 秒窗口。
- Mode 3 固定为 delta、theta、alpha、beta、gamma 五个频段。
- 响应为 TMT-B 完成时间 `TMT_5`；默认主分析使用 `log(TMT_5)`。
- Tensor-ABESS 支撑满足 `S_channel × S_time × S_frequency` 的矩形约束。

正式结果以外折预测性能、支撑规模与支撑解释为重点；真实支撑未知，不计算或宣称 Exact Recovery、Mode-F1、Element-F1，也不使用 Oracle OLS。

## 1. 数据事实与本地状态

### 1.1 历史单样本检查

此前读取的 `sub-010002` 为 BrainVision 格式，形状 `62 × 2,555,000`，采样率 `2500 Hz`，时长约 `1021.9996 s`，包含 498 条 annotations；出现 `Stimulus/S  1` 至 `Stimulus/S  4`。其中 `VEOG` 曾被读为 EEG，处理时须改为 EOG。上述数字是该受试者的历史检查结果，其他受试者仍需分别读取和验证。

已找到的事件语义链为 **EC：`S210 ↔ events.tsv value 4 ↔ Raw S 4`；EO：`S200 ↔ events.tsv value 3 ↔ Raw S 3`**。这项证据用于指导逐人验证；不得只向全体受试者复制一份全局映射，或按编号、典型持续时间猜状态。

### 1.2 2026-09-14 仓库内盘点快照

| 项目 | 本次只读核查结果 |
|---|---|
| `实证分析/manifest.json` | 215 个受试者、1,726 个文件条目；清单不等于本地已下载 |
| 默认 `nm000179/` 路径 | 5 个受试者目录，31 个清单内文件，其尺寸全部与清单一致；未对大 EEG 做哈希校验 |
| 七件完整的受试者 | `sub-010002`、`sub-010003`、`sub-010004`、`sub-010006` |
| `sub-010005` | 缺标准 `channels.tsv`、`eeg.json`、`events.json`、`events.tsv`；存在额外的 108 字节 `*_eeg_events.tsv`，不能替代标准事件表 |
| 清单缺失 | 默认路径缺 210 个受试者目录、1,695 个清单文件 |
| 协变量文件 | 仓库内未找到 `participants.tsv/json` |
| TMT | 227 行，ID 无重复；清单内 215 个受试者 ID 均可对齐 |

用户随后确认完整数据稍后再下载，本轮先建立下载器与前150人的清单。本快照不代表其他磁盘路径的状态，不得据此或据 manifest 宣称全量数据验收通过。

### 1.3 前150人的下载准备

运行下列命令只读取本地 manifest 并生成下载计划，不访问远端：

```powershell
.venv/Scripts/python.exe -m empirical_analysis.lemon.download --manifest 实证分析/manifest.json --destination 实证分析/nm000179 --limit 150
```

实际下载时再添加 `--download`；默认计划为 `empirical_analysis/lemon/outputs_download/plan.json`，可用 `--plan` 改位置。下载器是独立入口，不由分析阶段的 `--stage all` 隐式启动。

按数值 ID 升序，排除 manifest 能直接证明缺少 `.eeg/.vhdr/.vmrk/标准events.tsv`、文件重复或校验信息异常的受试者后选择前150名；本地尚未下载不构成质量问题，下载前无法宣称脑电信号质量合格。每人下载上述四项和存在的 `channels.tsv`，并取得处理所需的 `eeg.json/events.json`；若清单包含共享的 `participants.tsv/json` 和 `dataset_description.json`，一并下载。

当前清单筛选发现 `sub-010203` 缺标准 `events.tsv`，因此明确排除；前150人的最后一名为 `sub-010250`，计划共1053件文件、约89.695 GiB。计划中记录逐人排除理由及处理元数据是否缺失，不以这份文件级筛选替代后续 EEG QC。

仅使用 manifest 的稳定 `bytes_url`，不保存或使用临时签名 `url`；完成后核对文件大小及 SHA256 或 Git blob SHA1。已有正式文件通过校验则自动跳过，否则停止且不覆盖；下载中的 `.part` 用 HTTP Range 续传，验证响应范围后继续写入，服务器忽略 Range 时明确记录从头重新获取临时文件而不是追加完整响应。中断保留 `.part`，只有完整校验成功后才转为正式文件。

连接/读取默认超时180秒，每文件最多额外重试5次（含首次共6次尝试），等待依次为2、4、8、16、32秒，仍按单文件顺序执行。仅明确暂时网络故障、响应截断及HTTP 408/429/500/502/503/504可重试；校验、范围/长度、证书、永久HTTP、本地磁盘/权限错误必须停止，正式文件不得覆盖，耗尽重试后保留`.part`。控制台显示当前路径、字节/百分比及重试状态，校验成功后才输出`VERIFIED`；下载日志记录`attempts`和`retry_history`，不包含下载URL。

指定参数为`--timeout 180 --max-retries 5 --retry-delay 2`；`--max-retries 0`禁用自动重试，允许范围0–5。使用D盘现有目录续传的完整PowerShell命令见[模块README](../empirical_analysis/lemon/README.md)，命令必须带`--download`才会实际传输。

默认 `data.subject_selection_file` 引用该150人计划，分析盘点仅纳入所选受试者；其余清单受试者保存到 `diagnostics/subjects_not_selected.csv`，不记为质量失败。改用完整本地队列时须显式将此字段设为 `null` 并使用新的输出目录。

### 1.4 TMT 与最终样本量

原 CSV 字段为 `ID,TMT_1,...,TMT_8`。历史检查提示 `TMT_5` 存在非数值内容；实际无效行数、正值数及分布必须重新计算并写入诊断，不预置 226 个有效样本等结果。

最终样本必须满足：TMT 数值有效且为正、七件 EEG 文件有效、逐人事件证据一致、共同通道合格、16 个窗口可构造、张量 QC 通过，以及正式调整分析的协变量可用。所有计数和样本流图均读取实际产物，不把 `227 → ...` 写死在绘图代码中。

## 2. 不可违反的执行规则

1. 实证代码与核心算法物理隔离，不改 `src/tensor_abess/` 或终版数值模拟方法来迎合真实数据。
2. 不猜测事件含义、协变量字段或编码；冲突、缺失、文件损坏均须输出诊断。
3. EEG 处理排除 `VEOG`；固定张量中不得插值填补缺失 cell 或静默跳过失败窗口。
4. 外层 CV 的协变量回归、X 标准化和 y 中心化全部仅在外层训练折拟合；测试折只应用这些参数。
5. `X[k]`、`y[k]`、`C[k]` 与 ID 顺序必须一致并持久化；不得依赖文件遍历顺序。
6. 固定 seed，保存配置、阶段状态、排除原因及模型预处理参数。
7. `analysis.mode` 默认 `adjusted`；缺协变量不得自动切换为 pilot。Pilot 必须显式配置，且不能充当正式论文主结果。
8. 单次运行仅使用一个响应尺度。原始秒数敏感性分析须选 `primary_transform: raw` 并使用不同的空输出目录。
9. 输出目录属于一套配置，不允许跨配置混用已有产物；更改输入路径、编码、响应尺度或处理参数时新建空输出目录。
10. 所有实现通过模块和命令行执行；notebook 不作为唯一实现或必要启动入口。

## 3. 实际目录与模块职责

从仓库根目录运行，保留现有根 `src/` 和 `tests/`：

```text
empirical_analysis/
├── __init__.py
└── lemon/
    ├── __init__.py
    ├── __main__.py        # 统一 CLI
    ├── config.py          # 配置读取、路径解析
    ├── config.yaml        # 实际配置，字段以此文件为准
    ├── download.py        # 独立的150人计划、续传与校验入口
    ├── dataset.py         # 文件盘点、TMT 清洗、ID 合并
    ├── io.py              # 输入输出与可审计产物
    ├── events.py          # 逐人事件证据与状态区间
    ├── eeg_preprocess.py  # 通道、窗口 EEG 处理
    ├── tensorize.py       # 频段功率与固定张量
    ├── confounds.py       # 训练集拟合与预处理持久化
    ├── models.py          # 现有算法适配
    ├── evaluation.py      # 外层 CV、OOF 与模型保存
    ├── pipeline.py        # 阶段依赖与 Gate
    ├── reporting.py       # 图表和运行报告
    ├── README.md
    ├── requirements.txt
    └── outputs/           # 默认 output_dir，运行后生成
tests/
└── test_lemon_*.py
doc/                       # 工作流记录仅允许放在此处
实证分析/
├── LEMON_Tensor_ABESS_Codex_执行文档.md
├── TMT.csv
├── TMT_info.txt
├── manifest.json
└── nm000179/              # 或配置中指定的外部下载路径
```

不建立数字命名的独立脚本目录，也不建立另一层 `lemon/src/`。仓库根 `src/` 仍仅承担现有核心包的导入路径。工作流文件 `task_plan.md`、`progress.md`、`findings.md`、`task_issue.md` 只能位于 `doc/`；`docs/` 不能替代此规则。

依赖独立列于 `empirical_analysis/lemon/requirements.txt`，包括 `numpy>=1.26`、`scipy`、`pandas`、`mne`、`matplotlib`、`PyYAML`。实证流程复用现有三种 NumPy 终版方法，不依赖 scikit-learn；LEMON 测试使用标准库 `unittest`，无需 pytest。仓库其他示例和完整根项目可编辑安装仍遵循 `pyproject.toml` 的依赖声明，不能把其他示例的依赖要求混同于本流程。

本次本地 `.venv` 以 `--system-site-packages` 创建，复用系统已安装的六项运行依赖。全新机器在仓库根目录的 PowerShell 中按以下方式创建独立环境（已有环境不必重建）：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r empirical_analysis/lemon/requirements.txt
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_lemon_*.py'
```

本流程可通过根 `src/` 导入核心；如需完整根项目的可编辑安装，再执行 `.venv/Scripts/python.exe -m pip install -e .`。下文统一 `python -m empirical_analysis.lemon` 表示同一个目标环境的解释器；未激活 `.venv` 时，用 `.venv/Scripts/python.exe` 替换 `python`，实际环境命令也可参照 [LEMON README](../empirical_analysis/lemon/README.md)。

原始 EEG、下载 manifest（可能含签名链接）、临时下载文件和运行输出不应作为源码提交。根 `.gitignore` 忽略 `/实证分析/nm000179/`、`/实证分析/TMT.csv`、`/实证分析/manifest.json`；`empirical_analysis/lemon/.gitignore` 忽略默认 `outputs*/` 和 `logs/`。使用其他下载或输出路径时，提交前仍须单独核对暂存内容，不能假定所有数据扩展名已经被忽略。严禁将签名下载 URL 写入日志或报告。

## 4. 配置与路径约定

完整、可运行字段以 [empirical_analysis/lemon/config.yaml](../empirical_analysis/lemon/config.yaml) 为准，不在本文维护第二份完整配置。所有相对数据路径及 `output_dir` **以配置文件所在目录为基准**，不随启动目录变化；CLI 中 `--config` 参数本身按命令所在目录定位。绝对路径也可直接配置，不硬编码个人磁盘路径。

| 字段 | 默认值或约束 |
|---|---|
| `data.eeg_root` | `../../实证分析/nm000179` |
| `data.tmt_csv` | `../../实证分析/TMT.csv` |
| `data.participants_file` | `../../实证分析/nm000179/participants.tsv` |
| `data.participants_json` | `../../实证分析/nm000179/participants.json` |
| `data.manifest_file` | `../../实证分析/manifest.json` |
| `data.subject_selection_file` | `outputs_download/plan.json`；显式 `null` 表示不限制名单 |
| `output_dir` | `outputs`；不同配置使用不同的空目录 |
| `response.primary_transform` | `log`；敏感性分析显式设为 `raw` |
| `analysis.mode` | `adjusted`；仅显式设为 `pilot` 才做未调整探索分析 |
| `analysis.min_samples` | 100；adjusted/pilot 均执行该阈值，不自动放宽 |
| `events.min_diagnostic_subjects` | 5；事件阶段须有至少 5 人通过逐人验证 |
| `confounds.id_column` | `participant_id`，须与真实文件核对 |
| `confounds.age_column` / `sex_column` | `age` / `sex`，须与真实文件核对 |
| `confounds.sex_mapping` | `null`；取得真实编码说明后显式填写 |
| `report.channel_plot` | `barplot`；当前仅实现明确的通道名称条形图，不推断 montage |

若协变量列名或编码不同，依据 `participants.json` 和实际文件修改配置。年龄区间字符串不是精确年龄；不得自行用区间中点替代。非空且无法数值化的年龄（包括年龄区间），或不在明确映射表内的非空性别值，均会直接停止数据合并；只有明确的协变量缺失按规则记录并排除。若数据只有年龄分组，须先明确分组协变量的分析定义并实现相应编码，不能直接宣称已完成年龄调整。

EEG 默认设置为：共同 EEG 通道至少 58 个；EC 两端各裁 5 秒；25 秒窗口取 16 个；1–45 Hz 带通；重采样至 250 Hz；仅 EEG 平均参考。Welch 默认 4 秒 segment、50% overlap、median average，log epsilon 为 `1e-20`。这些值必须与实际配置一起存档。

## 5. Phase A：盘点与 TMT 清洗

### A1. `--stage inventory`

扫描配置的下载根目录，并结合 manifest 及配置中的受试者选择名单记录缺失情况。每人至少核对七件：`.vhdr`、`.vmrk`、`.eeg`、`*_events.tsv`、`*_events.json`、`*_channels.tsv`、`*_eeg.json`；检查唯一性、非零尺寸、清单尺寸和 BrainVision 内部引用。`*_eeg_events.tsv` 不能冒充标准 `*_events.tsv`。

输出 `diagnostics/eeg_inventory.csv`，包括 ID、各文件 `has_*`、`file_complete` 和 `exclusion_reason`。年龄、性别源文件也要独立检查。尺寸检查只能证明尺寸吻合，不得描述为完成 SHA256 完整性校验。

### A2. `--stage tmt`

读取原始字符串并保留 `TMT_5` 原值，用 `pd.to_numeric(errors='coerce')` 转数值。ID 必须非空且唯一；重复时写出诊断并停止，不自动去重。非数值或非有限值记入 `diagnostics/invalid_tmt5_rows.csv`；非正值另外诊断并停止取 log。

输出 `diagnostics/tmt_clean.csv`、`diagnostics/tmt_descriptive_statistics.csv`，描述 raw 与 log 的 count、mean、std、median、min、max、skew。两种分布图在报告阶段生成；清洗同时保留 raw 和 log 列不表示单次建模同时混用两个尺度。不默认 winsorize 或删除高值。

Gate A：输入路径明确，ID 无重复，行为值异常有记录，七件完整性状态可追踪。盘点可以成功输出缺失清单；这不等于后续事件、张量或正式样本量 Gate 已通过。

## 6. Phase B：逐人事件证据

运行 `--stage events`，至少 `events.min_diagnostic_subjects=5` 名文件完整受试者通过验证，且批量张量使用的每名受试者都必须独立验证。联合读取：

- `*_eeg.json` 中的状态代码语义；
- `*_events.json` 与 `*_events.tsv` 的字段解释、trial type、value 和 sample；
- BrainVision `.vmrk` 与 MNE Raw annotations 的描述、采样位置和原始采样率。

验证 `S210 → value 4 → S 4`、`S200 → value 3 → S 3` 的对应关系和采样位置一致性，并保存逐人证据。只在 sidecar 给出语义且事件表与 marker 一致时确认 EC/EO；缺文件、语义不明或映射冲突须停止该阶段，不按时间间隔猜码。

保存 `diagnostics/event_code_summary.csv` 和 `diagnostics/event_sequences/<subject>.csv`，保留原始触发、采样位置、时间及核对结果。重复的同状态 marker 必须先合并为连续状态区间；例如连续多个 EC marker 属于同一个 EC 状态段，直到实际 EO 状态切换，不能把每次重复触发当成新片段。S1/S2 等非状态标记不自动成为 EC 结束边界。

Gate B：达到配置要求的诊断人数，待处理受试者语义和采样核对通过，EC 区间单调且边界合法。已确认的历史映射不能绕过这个 Gate。

## 7. Phase C：单受试者张量 smoke test

运行 `--stage smoke`；默认检查 `sub-010002`，具体受试者由配置指定，不以 TMT 是否可用决定能否进行 EEG 单样本测试。

### C1. 通道

读取 BrainVision 时不加载全量信号，按配置将 `VEOG` 改为 EOG，仅保留 EEG；保存实际通道顺序。预期 61 个 EEG 通道是验收参考，不是硬填张量维度的依据。

### C2. 窗口

从已合并的连续 EC 状态段出发，每段两端各裁 5 秒，在剩余范围按时间顺序切不重叠的 25 秒窗口，收集前 16 个。窗口不得跨状态边界，必须按采样点实现固定长度，防止 MNE 包含右端点而多取一帧。不足 16 个时记录排除原因，不重复、拼接或补齐窗口。

每个窗口保存 `window_index`、原始起止秒数和来源 EC 段；窗口序号表示每名受试者的相对时序，不表示所有人相同的绝对实验时间。

### C3. 预处理与 PSD

逐窗口 crop/load，执行 1–45 Hz 带通、重采样至 250 Hz、EEG 平均参考、Welch PSD、频段积分及 log 功率。45 Hz 低通默认无需额外 50 Hz notch。Welch 使用 4 秒 segment（250 Hz 下 1000 点）、50% overlap 和 median average。

频率掩码固定为 delta `[1,4)`、theta `[4,8)`、alpha `[8,13)`、beta `[13,30)`、gamma `[30,45]`，确保边界频点不重复计入。积分使用兼容 NumPy 1.26 的 SciPy API：

```python
from scipy.integrate import trapezoid

band_power = trapezoid(psd[..., mask], freqs[mask], axis=-1)
feature = np.log(band_power + eps)
```

不得使用仅 NumPy 2 才提供的 `np.trapezoid`，也不改变五个频段的顺序。

### C4. 验收

`Xi` 必须为 `(p_channel,16,5)`，所有值 finite，`VEOG` 已排除，16 个窗口和每个频段均可追踪；相同输入重跑结果应在浮点容差内一致。保存单样本张量、通道顺序和窗口 metadata，Gate C 通过后才能批量。

## 8. Phase D：批量张量与 QC

运行 `--stage tensors`。先对文件完整的候选受试者扫描真实 EEG 通道集合，以配置参考受试者的顺序排序并取共同通道；共同 EEG 通道不足 58 时输出差异并停止。不得因预期为 61 而伪造缺失通道。

保存 `diagnostics/common_channels.txt` 和 `diagnostics/channel_availability.csv`。对每名受试者重新验证事件、窗口和共同通道，构造固定形状张量；文件、事件、窗口、通道或非有限数等硬失败须显式排除并记录。检查每个 band 的方差是否全零；整体功率 robust z-score 只作 `gross_power_flag`，不据此自动删人。

`qc/subject_qc.csv` 至少包含：ID、文件完整性、事件状态、`n_ec_segments`、`n_candidate_windows`、`n_selected_windows`、通道状态、张量 finite 状态、功率标记、纳入状态和排除原因。候选窗口总数与最终选中窗口数分别记录；默认选中前 16 个不能冒充所有候选窗口的数量。阶段级前提失败必须停止；按预定规则排除单个受试者须在清单中可见，不得吞掉未知异常继续运行。

保存：

```text
tensors/subjects/<ID>.npy
tensors/subjects/<ID>_meta.json
tensors/X_unstandardized.npy
tensors/tensor_subject_ids.csv
tensors/tensor_meta.json
```

汇总 `tensor_meta.json` 记录形状、共同通道顺序、窗口数量与时长、5 个 band 顺序及 `band_edges_hz`、采样率、滤波和功率变换；`subject_window_metadata` 指向 `subjects/<subject_id>_meta.json`。每名受试者的实际窗口时间、EC 段数、候选/选中窗口数和事件证据保存在对应单人 metadata 中，不直接嵌入汇总文件。此处 X 未做样本间标准化或协变量残差化。

## 9. Phase E：分析数据与协变量

运行 `--stage dataset`。按 ID 显式一对一合并张量、TMT 和 participants；重复 ID 停止，不自动去重。先输出完整纳入/排除记录，再以相同 ID 顺序生成模型数组：

```text
qc/final_subject_manifest.csv
qc/analysis_dataset.npz
```

NPZ 保存最终对齐的 `X`、`y` 与 `subject_ids`；`adjusted` 另存协变量矩阵 `C`，`pilot` 明确不包含 `C`。最终清单包含张量可用性、raw/log 响应、协变量原值与数值编码、纳入状态和排除原因；不得仅保存 y 而让后续阶段重新推断其顺序。

### E1. 默认正式调整分析

`analysis.mode: adjusted` 要求 `participants.tsv` 及 `participants.json` 均存在，编码说明必须能解析为 JSON 对象，并须配置明确的字段与 sex mapping；不能因找到一个貌似可用的字段就猜编码。非空非数值年龄或未知性别编码会停止合并，明确缺失的协变量则记录排除原因。两种模式的最终有效 `n < analysis.min_samples=100` 时均停止 CV；小样本 pilot 必须显式调低阈值并使用独立输出目录。

外层训练折构造 `C=[intercept,age,sex]`，分别对 y 和向量化 X 拟合 nuisance OLS，再对 X 残差逐 cell 标准化，y 残差减去训练均值。测试折 nuisance 预测及标准化全部使用训练参数；回归设计必须可识别，不能静默移除共线协变量。

### E2. 显式 pilot

仅在用户明确设置 `analysis.mode: pilot` 时允许未调整探索分析；产物标识 `PILOT_UNADJUSTED=true`。缺协变量不会触发自动降级。Pilot 也需通过文件、事件、窗口和样本量等已配置 Gate；降低阈值属于显式的探索配置，必须使用独立输出目录并如实报告。

## 10. Phase F：外层预测评估

运行 `--stage cv`。默认固定 seed 的 5 折外层 CV，三种方法共用外折划分与训练折预处理参数：

```text
外层 train/test ID
→ 外层 train 拟合 nuisance OLS
→ 外层 train 拟合 X 均值/标准差、y 均值
→ 现有模型在处理后的外层 train 上拟合
→ 外层 test 应用已保存参数并预测
→ 在本次选定的响应尺度上计算外折指标
```

### F1–F3. 预处理空间与预测恢复

X 残差使用训练折 cell 均值及 `ddof=1` 的标准差；小于 `1e-12` 的尺度固定为 1，规则同时用于训练和预测。调整分析的预测为：

\[
\widehat y=C\widehat\gamma_y+\overline y_{res,train}
+\langle (X-C\widehat\Gamma_X-\mu_{X,train})/s_{X,train}, B\rangle.
\]

Pilot 省略 nuisance 项，仍加回训练 y 均值。**保存的 B 位于残差化、X 标准化和 y 中心化后的建模空间**，不能直接与原始 log-power 张量内积当作预测。必须一同保存 `preprocessing.npz`，并在预测时加回 nuisance 项和训练 y 均值。

默认 RMSE、MAE、R² 在 `log(TMT_5)` 尺度计算；`raw` 独立运行则在秒尺度计算。不能直接指数化 log 预测后把它与 raw 模型指标混在一张未经说明的表中。

### F4. Tensor-ABESS 真实接口

调用 `tensor_abess.tensor.abess.tensor_abess`，实证适配位于 `models.py`；adaptive 通过 `target_sparsity=None` 选择，**核心函数没有 `mode=` 或 `init_c=` 参数**。

| 实证含义 | 实际核心参数 / 行为 |
|---|---|
| adaptive | `target_sparsity=None` |
| BIC 初始化、扩大系数 3 | `criterion='bic', c=3.0` |
| Group Lasso 网格与迭代 | `n_lambdas=20, cv_folds=5, max_iter=500, tol=1e-6` |
| 交换与搜索 | `k_max=1, m_max=15` |
| BCD | `max_cycles=6, eps=1e-8` |
| mode MSIC | `adaptive_criterion='msic'`，mode 评分内部 alpha/gamma 为 1/1 |
| refine MSIC | `msic_refine_alpha=1.5, msic_refine_gamma=1.0` |
| one-SE | `msic_refine_one_se_scale=1.0` |
| 参数数量限制 | 初始化内部使用 `max(1,floor(0.7*n_train))` 约束矩形规模；不是可传入的 `q_max=` 参数 |

以上是实证 v5 配置，不是裸 API 的全部默认值：核心默认 `criterion='cv'`、`c=2.0`、`k_max=None`、`max_cycles=30`。调用必须显式传入实证配置；不得把文档别名原样作为未知关键字传入或改核心默认来达到目的。

### F5. 终版基线

明确复用 `numerical_simulation.methods.tensor_glasso` 与 `numerical_simulation.methods.run_vector_lasso_method`，后者使用该模块已有 NumPy CV 求解器和 `TensorSimulationConfig` 参数容器；仅封装真实 X/y 与参数，不生成模拟数据。

不要误用 `src/tensor_abess/baselines/` 的早期接口，也不调用包含 Oracle 的 `run_all_methods`。Tensor-GLasso 与 Vector Lasso 保留 v5 的内部调参 CV 机制及算法；这里的外折 OOF 是三模型共同的性能评估依据。实证预处理在外层训练折拟合，不宣称重写了基线内部各验证折的预处理或选参方式，内部选参分数也不作独立泛化性能报告。

### F6. 指标与持久化

每外折保存 RMSE、MAE、R²，汇总 mean ± std；保存每名受试者恰好一次的外折预测、fold ID、响应尺度和调整模式。产物包括：

```text
cv/fold_assignments.csv
cv/oof_predictions.csv
cv/cv_metrics.csv
cv/cv_summary.csv
cv/folds/fold_<编号>/preprocessing.npz
cv/folds/fold_<编号>/<模型>/B.npy
cv/folds/fold_<编号>/<模型>/support.json
```

测试折互不重叠且覆盖全部最终样本；支持索引与预处理参数均需能够从磁盘复现预测。同一外折的三模型共享折目录中的 `preprocessing.npz`，模型目录保存各自的 B 与 `support.json`。

## 11. Phase G：最终解释模型

运行 `--stage final`，必须已有成功外层 CV。使用全部最终样本重新拟合一套预处理参数和 Tensor-ABESS，只用于解释与后续预测部署，不能据此报告独立测试性能。

保存 `final_model/B_hat.npy`、`S_channel.json`、`S_time.json`、`S_frequency.json`、`final_model_summary.json` 和 `preprocessing.npz`。支撑 JSON 保存 **从 0 开始的索引**，由 `tensors/tensor_meta.json` 映射成通道名称、窗口位置和频段名称；展示中若使用从 1 开始的窗口编号须明确标注。

报告 `(s1,s2,s3)`、`q=s1*s2*s3`、选中名称、所选窗口在各受试者的原始 onset 汇总、RSS/loss/final criterion。B 的单位和预测恢复规则与外折模型相同；系数热图只解释当前残差标准化空间中的权重，不等价于原始功率的因果效应。

## 12. Phase H：图表与报告

运行 `--stage report`。图表读取已保存产物，预测性能只来自外折 OOF；最少包括：

1. `tmt5_raw_hist.png` 与 `tmt5_log_hist.png`。
2. `sample_flow.png`：TMT 总数、有效值、完整 EEG 交集、张量 QC、协变量可用与最终 n，全部来自实际清单。
3. `cv_model_comparison.png`：各方法外折 RMSE/MAE/R²。
4. `predicted_vs_observed_oof.png`：注明响应尺度。
5. `selected_frequency_bands.png`、`selected_time_windows.png`。
6. 通道图：显式选择 `barplot` 时生成 `channel_barplot.png`；只有配置为 topomap 且真实 montage 完整可用时才生成 `selected_channels_topomap.png`。
7. `coefficient_summary.png`：明确当前建模空间的系数聚合方式。

无 montage 时应显式配置 barplot，不允许先画 topomap 失败后静默降级、猜电极坐标或伪造头皮位置。图中的样本数、模型指标和支持名称不得使用预置演示数据。

## 13. 支撑稳定性

从外层 CV 模型读取每个 mode 的入选频率，保存：

```text
cv/channel_selection_frequency.csv
cv/time_selection_frequency.csv
cv/frequency_selection_frequency.csv
```

以有效外折数为分母，区分不同方法；这是外折选择频率，不是 bootstrap stability。额外 subsampling 属于另行配置的补充分析，不在本次标准流水线中自动增加。

## 14. 模块接口边界

读取与写盘由 `io.py`、输入整理由 `dataset.py` 负责；`events.py` 输出带证据的状态区间，`eeg_preprocess.py` 和 `tensorize.py` 接收显式参数构造张量；`confounds.py` 训练预处理并应用已拟合参数；`models.py` 返回统一模型估计；`evaluation.py` 组织外折，`pipeline.py` 负责阶段依赖与 Gate，`reporting.py` 从产物绘图。

函数的实际签名以对应模块为准，不在文档维护不一致的伪 API。数据与配置经参数传递，不通过全局可变变量、隐式工作目录或动态 `sys.path` 注入维持模块耦合。

## 15. 自动检查与测试

在 `tests/test_lemon_*.py` 验证有科学后果的边界：TMT 重复/非数值、manifest 尺寸与缺件、事件语义及采样冲突、同状态重复 marker 合并、窗口越界/不足、频段掩码、ID 对齐、协变量编码、训练/测试泄漏及保存后预测复现。轻量合成输入验证处理和建模契约，不把合成测试通过描述为真实数据分析通过。

数组验收至少检查：X 为四维；后两维为 16 与 5；X/y finite；ID 非空唯一，`n_X=n_y=n_C=n_ID`（pilot 的 C 可为空表示未调整）；共同通道数与 X 一致；`VEOG` 不在 EEG 轴；外折 train/test ID 不重叠。每折预处理只能由该折 train 拟合并必须保存。

测试命令：

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_lemon_*.py'
```

## 16. 正式建模 Gate

- [ ] 实际下载路径已确认，七件文件及清单尺寸盘点完成。
- [ ] TMT 行数、无效值和 ID 诊断已保存。
- [ ] 至少 5 人完成事件联合验证，批量使用的每人均可追溯。
- [ ] 重复同状态 marker 已合并，16 窗口及其原始时间可追溯。
- [ ] 共同 EEG 通道不少于 58；单样本 smoke 通过。
- [ ] 批量 QC、最终清单和 NPZ 已保存，数组与 ID 严格对齐。
- [ ] `adjusted` 的年龄/性别字段、编码和说明明确，未自动使用年龄区间中点。
- [ ] 最终 n 不少于配置阈值 100，外折样本量和回归可识别性满足要求。
- [ ] 固定外折与训练折预处理，三方法使用相同外折。
- [ ] v5 核心/基线原样复用，OOF 完整，最终解释模型与预处理一同保存。

## 17. 失败处理与运行状态

`config_used.json`、`run_state.json` 和阶段运行失败时的 `logs/<stage>_error.json` 均位于本次 `output_dir`。只把实际成功阶段标记为完成；阶段错误记录原因和必要诊断位置，不输出 manifest 的签名 URL。配置读取、输出初始化和指纹核对失败发生在阶段运行前，由 CLI 直接报错，不承诺产生阶段错误日志。

已完成阶段仅在输入、源码、配置及已保存产物的指纹均一致时复用；完成产物缺失或修改同样阻断继续。修复输入、修改源码/配置、切换分析设定，或修复已有部分输出的失败模型运行后，须使用新的空 `output_dir` 从 inventory 重新执行，保留原诊断与结果，不在原目录混用不同版本产物。

以下情况停止依赖阶段：输入或配置冲突；同输出目录混用不同配置；事件语义/采样不一致或人数不足；共同通道不足；合法窗口或最终样本量不足；张量非有限数；协变量编码不明/设计不可识别；核心 API 不匹配；模型 OLS 或内存/运行错误。不得自动换模型、减少窗口、放宽阈值、删协变量、跳过方法或用伪造结果补齐输出。

数据验收阻塞时可以继续与真实输入无关的代码、文档和合成测试，但真实数据的后续依赖阶段保持暂停；不得把部分阶段成功总结为正式实证完成。

## 18. 统一启动与顺序

从仓库根目录启动；核心 import 路径统一设置一次：

```powershell
$env:PYTHONPATH = 'src'
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage inventory
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage tmt
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage events
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage smoke
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage tensors
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage dataset
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage cv
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage final
python -m empirical_analysis.lemon --config empirical_analysis/lemon/config.yaml --stage report
```

准备齐全后可使用同一入口 `--stage all` 按上述顺序执行；它不会绕过 Gate。独立阶段的前置产物必须来自同一配置。首次在真实下载上建议逐阶段检查诊断，尤其先核对当前配置路径确实指向完整数据。

## 19. 交付物路径

以下均相对于本次 `output_dir`，默认是 `empirical_analysis/lemon/outputs/`：

```text
RUN_SUMMARY.md
config_used.json
run_state.json
logs/
  <stage>_error.json                      # 阶段运行失败时；初始化错误由 CLI 直接报告
diagnostics/
  eeg_inventory.csv
  tmt_clean.csv
  invalid_tmt5_rows.csv
  tmt_descriptive_statistics.csv
  event_code_summary.csv
  event_sequences/<subject>.csv
  common_channels.txt
  channel_availability.csv
qc/
  subject_qc.csv
  final_subject_manifest.csv
  analysis_dataset.npz
tensors/
  subjects/<ID>.npy
  subjects/<ID>_meta.json
  X_unstandardized.npy
  tensor_subject_ids.csv
  tensor_meta.json
cv/
  fold_assignments.csv
  oof_predictions.csv
  cv_metrics.csv
  cv_summary.csv
  *_selection_frequency.csv
  folds/fold_<编号>/
    preprocessing.npz
    <模型>/
      B.npy
      support.json
final_model/
  B_hat.npy
  S_channel.json
  S_time.json
  S_frequency.json
  preprocessing.npz
  final_model_summary.json
figures/
  tmt5_raw_hist.png
  tmt5_log_hist.png
  sample_flow.png
  cv_model_comparison.png
  predicted_vs_observed_oof.png
  selected_frequency_bands.png
  selected_time_windows.png
  channel_barplot.png                    # 显式 barplot 模式
  selected_channels_topomap.png          # 显式 topomap 模式且 montage 验证通过
  coefficient_summary.png
```

`RUN_SUMMARY.md` 记录输入位置、实际样本流与排除原因、最终形状、响应尺度、adjusted/pilot 状态、协变量编码、三方法外折指标、最终支持名称与索引、系数空间，以及异常或未完成 Gate。失败运行不得编造尚未产生的图表、CV 指标或最终支撑。

## 20. 实证结论边界

结果支持时，围绕 TMT-B 外折预测能力、矩形稀疏规模，以及通道/EC 窗口/频段的选择频率和解释展开。OOF 性能与最终全样本解释模型分别报告；支持未知时不能声称准确恢复真实脑区，协变量调整和支撑选择也不构成因果证据。

严格遵循数据诊断、逐人事件验证、smoke、批量张量、对齐数据、外折 CV、最终解释与报告的顺序；当前实际数据 Gate 是否通过，以该次运行的配置、清单和状态文件为准。

