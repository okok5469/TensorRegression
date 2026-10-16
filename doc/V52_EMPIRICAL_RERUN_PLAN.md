# v5.2 三实证重跑准备与 LEMON 对比方案

日期：2026-10-11。目标算法为获批 D2 / `v5.2-search-final`；本任务只准备入口与验收条件。状态：`EMPIRICAL_RUN_NOT_RUN`，未执行 audit-only、smoke 或正式拟合，未联系队友、上传或改动数据、实证代码和旧输出。真实 CLI 仅执行 `--help` 并通过（支持 `--config`、`--output-dir`、`--audit-only`、`--smoke`）。正式发布状态与本轮源码回归见[发布 manifest](releases/v5.2-search-final/RELEASE_MANIFEST.json)及[发布验证](releases/v5.2-search-final/RELEASE_VALIDATION.md)。

## 启动前 Gate

1. 正式 manifest 为 released，目标完整32 core与正式身份匹配；导入路径是对应项目的生产 `repository/src`。保留原 MSIC和公共接口，实证 `target_sparsity=None`。
2. 正式模拟完成后的验收按 [V52_SIMULATION_ACCEPTANCE_PLAN.md](V52_SIMULATION_ACCEPTANCE_PLAN.md) 执行；本轮不视700任务已提交或已完成。实证启动还需确认对应输入、模型配置、划分、依赖及独立输出，并另行授权。
3. 保留各项目受试者集合、ID及顺序、响应与变换、协变量定义/编码、MRI模态或EEG特征顺序、外内划分与种子；全部训练内预处理重新拟合，不能由外层测试集选 caps、网格、MSIC、初始化或删掉困难被试。
4. 新版本每个ABESS训练折重建caps与完整合法规模候选，重搜支撑、选阶、OLS并预测。外层模型重新训练；若内层调ABESS参数，每个内训练折独立预处理及重新搜索，容量使用当前内折样本数。LEMON现行ABESS不做外部参数调优，保留原BIC初始化及其训练内初始化流程。
5. 旧B/支撑/q/选阶轨迹/模型断点不得作为新版结果；原始未跨被试拟合的张量可在数据身份验收后复用。新旧结果分开保存，对照复用须逐项验明配置、独立代码、数据、划分、预处理、种子、环境与完整性；现行入口默认在新版目录重新计算五个模型。

## LEMON 已核实的冻结设置

| 项目 | 保留值与证据 |
|---|---|
| 受试者与张量 | 145名有效受试者，`X=(145,61,16,5)`，来源原146张量按原清单筛选；保持原纳入集合与次序，不扩大或缩减 |
| 响应 | 自然对数 `log(TMT_5)`；`empirical_analysis/TMT.csv` 与原分析数据逐ID核对 |
| 协变量 | 截距、6个年龄dummy、sex，共8列；年龄参考20–25，类别为20–25/25–30/30–40/55–65/65–70/70–75/75–80；原30–35/35–40合并30–40，55–60/60–65合并55–65；原sex代码1→0、2→1 |
| EEG | 原61通道、16窗口、5频带与顺序、窗长/采样率、`log_absolute_band_power`、NumPy C-order；不改信号处理定义 |
| 外层CV | 5折×3重复，种子 `[20260908,20261009,20261010]`；每折116训练/29测试，原年龄分层算法与subject顺序保留 |
| 内层CV | 3折，基础种子20261009；实际内层seed为 `20261009 + repeat*100 + outer_fold`（repeat从0、fold从1）；原划分函数保留 |
| ABESS | k_max=1，c=3，criterion=bic，n_lambdas=20，cv_folds=3，max_iter=500，tol=1e-6，max_cycles=6，m_max=15，eps=1e-8；原MSIC alpha1.5/gamma1/scale1，target_sparsity=None |
| 模型与统计 | `Covariates Only`、Tensor-ABESS、Vector Lasso、Tensor-GLasso、Ridge；原lambda网格、bootstrap seed20261009/2000次；必须保留仅协变量基线 |
| 输入 | config中的 `../outputs_lemon_adjusted`、`../../TMT.csv`、`../outputs_demographics/participants.tsv` 均相对 `incremental_value/config.yaml` 目录解析 |
| 新输出 | `empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final`；原 `outputs_eeg_incremental` 不覆盖、不续写、不改身份 |

证据：[现行config](../empirical_analysis/lemon/incremental_value/config.yaml)、[旧数据audit](../empirical_analysis/lemon/outputs_eeg_incremental/diagnostics/dataset_audit.json)、[原协变量设计](../empirical_analysis/lemon/outputs_lemon_adjusted/qc/covariate_design.json)。年龄不能简化为年轻/老年两个组，不能估算精确年龄或中点。

## LEMON 已核查的真实入口

下列命令仅供后续验收/授权后执行，本轮均未执行。Windows在项目根运行已有 `.venv`（不得因此安装或升级库）：

```powershell
$env:PYTHONPATH = "$PWD\src"
# audit-only会写独立新目录中的审计文件，不拟合；本轮不执行：
& .\.venv\Scripts\python.exe -B -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --audit-only --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final

# audit验收、发布/数据/划分确认且另行授权正式重跑后才执行：
& .\.venv\Scripts\python.exe -B -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final
```

Linux使用目标项目既有实证Python环境，入口与参数相同：

```bash
export PYTHONPATH="$PWD/src"
# 后续授权的输入audit：
PYTHONDONTWRITEBYTECODE=1 python -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --audit-only --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final
# 后续另行授权的正式拟合：
PYTHONDONTWRITEBYTECODE=1 python -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final
```

`--audit-only` 会生成config、数据/预处理audit、输入/代码/环境指纹及 `run_state.json`；不生成外折文件、不代表CV或拟合验收。正式入口随后按同一输入/配置/源码/环境ID续入该新目录，身份不符立即停止，不篡改断点。现行run在外折拟合前保存 `cv/outer_splits.csv`，应按subject_id/repeat/outer_fold/seed与旧文件逐项比较，再验收折级 `split.json` 与内层trace；不能只凭种子宣称实际划分相同。旧码没有 `--resume` 参数，续跑使用同一正式命令且由内部身份检查控制。

预处理包括协变量残差化、标准化、响应中心化；每个外训练折重新拟合，基线超参数CV每个内训练折重新拟合，验证/测试只应用训练所得变换。`models.py`显式拒绝实证固定target_sparsity并检查容量/选定设计秩；不得对数据异常做静默补救。

## LEMON 历史事实与新版验收

旧 `outputs_eeg_incremental/run_state.json`记录complete、75 fits、2175条OOF（145人×3重复×5模型）。其保存的完整32 core raw指纹逐项匹配 `v5-original/VERSION.json`；其中bcd为 `4f9486d32f01a2798a55fbe7216d6a05d789f56f8f85608f03598c2926bce6fa`，不能将旧输出当作v5.1/v5.2结果。实证整体身份仍按保存的完整pipeline/input指纹核验。

旧 `cv/fold_metrics.csv` 中 Tensor-ABESS在repeat=2、outer_fold=5选 `(s1,s2,s3)=(4,4,5)`、q=80，n_train=116，对应容量81，test RMSE=0.6707709551。旧 `cv/model_summary.csv` 的跨重复平均OOF RMSE/R²：Covariates Only为0.3174850/0.3678997，Tensor-ABESS为0.3716675/0.1214302；这是旧结果定位信息，不是新版性能结论。不得假定搜索扩展提高预测或消除接近容量的模型。

新版完整验收须报告：

| 比较内容 | 产物与计算口径 |
|---|---|
| 完整性与身份 | 新旧输入、145个ID/顺序、响应、协变量、特征metadata、外内划分一致；新完整32源码与发布身份匹配；75 fits、每subject/repeat/model一次，共2175条完整有限OOF；不能用部分折汇总冒充完整 |
| 预测 | 新旧ABESS按subject_id/repeat/fold配对，报告每重复完整OOF RMSE、R²和差值；同时按原统计口径与Covariates Only比较EEG增量，保留Ridge/Lasso/GLasso |
| 支撑与选阶 | 每折记录s1/s2/s3/q、完整支撑索引与实际通道/窗/频带；同一划分新旧support交并/变化，跨15外折规模分布、各模式选择频率/Jaccard与重复稳定性 |
| 接近容量 | `Q=floor(0.7*n_train)`，报告q/Q与Q-q；固定训练n116时Q81，历史q80单列；可预先描述性标记q/Q≥0.9，完整列出所有折，不用标记删除模型或事后调参 |
| 过拟合与成本 | train/test RMSE差、接近容量折的测试误差及运行成本；检查OLS设计秩与有限预测；异常直接记录并停止受影响验收，不临时改caps/MSIC或删被试 |
| 不确定性 | 原bootstrap与subject-cluster描述性区间保留；145人为分析单位，不能把435个subject-repeat观测当435个独立被试，不把保存OOF上的区间解读为新外部验证 |

来源文件为新目录中的 `cv/{outer_splits,oof_predictions,fold_metrics,inner_selection,model_metrics_by_repeat,model_summary,incremental_metrics}.csv`、`cv/folds/.../{split.json,preprocessing.npz,Tensor-ABESS/support.json,B.npy,complete.json}`及diagnostics。q接近容量和支撑稳定性目前没有独立自动汇总CLI，本任务提供验收计算口径，不宣称已有新报告或虚构脚本。

## HCP-YA 与 NKI-Rockland

本仓库当前可见实证项目仅LEMON；未访问HCP/NKI数据、队友工程或真实执行入口，因此其具体CLI、subject数量、模态顺序与协变量参数均标记 `NOT_VERIFIED`，运行状态 `NOT_RUN`。队友应在自己的原入口显式导入获批生产core，沿用原subjects、响应/协变量、MRI张量/模态顺序、预处理定义及外内划分，独立新版目录完成所有ABESS训练折重新搜索/OLS/预测。

交回各自完整32 core、adapter/预处理代码、输入、配置、ID与划分、种子、依赖/环境指纹；提供版本明确完整OOF、每折支撑/q、训练/测试误差与完成/失败记录。先通过其原项目数据与划分Gate再启动；不把LEMON命令拼接为HCP/NKI入口。队友安装与最小patch见 [TEAM_V52_RELEASE_NOTICE.md](TEAM_V52_RELEASE_NOTICE.md)。
