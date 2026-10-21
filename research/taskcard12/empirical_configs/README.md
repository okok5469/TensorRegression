# 实证输入契约

LEMON 已有可审计输入。HCP-YA、NKI-RS 的配置是独立契约，当前 `status=NOT_PROVIDED`，执行结果为 `NOT_RUN`；程序明确报错，不下载、替造数据或套用 LEMON 响应。补齐数据并不自动授权拟合，仍需先完成只读预检、P0 Gate 与该数据集的冻结协议。

## LEMON

固定使用 145 人、天然 `61×16×5` EEG 张量、`log(TMT_5)` 与原 8 列协变量设计。3 次重复、每次 5 外折沿用原 seeds、受试者顺序和训练预处理。原正式配置的 native 参数整体复用；四个非 ABESS baseline 保留历史预测，并验证全部输入、splits、每折预处理、配置与原产物。旧文件始终只读。

## HCP-YA / NKI-RS

在各自配置中填写以下真实信息，审核后将状态改为 `READY`。所有路径相对仓库根目录；不要覆盖旧输入或历史输出。

| 字段 | 要求 |
|---|---|
| `response` | 明确原研究连续响应的名称、原尺度/变换；`continuous=true` 是研究者声明，数值验证额外排除常量和二值响应。 |
| `tensor_axes` | 三个天然模式的含义；不得将任意向量重塑为张量。 |
| `dataset_npz` | 无 pickle 的 NPZ，仅含 `X,y,C,subject_ids`，可加 `groups`。`X.shape=(n,p1,p2,p3)`，`y.shape=(n,)`，`C.shape=(n,k)`；数值有限。IDs 是唯一非空字符串。未使用协变量时显式提供 `(n,0)` 设计。 |
| `covariate_names` | 与 `C` 列数和顺序一致；预处理所需截距必须在设计中显式定义。 |
| `splits_json` | 预先确定的对象列表；每个对象含整数 `repeat,outer_fold` 及按原顺序排列的整数索引 `train,test`。每折互斥且覆盖全数据，每次重复每人恰好一次外测。 |
| `group_isolation` | 明确研究是否要求家庭/重复测量/站点分组。为 true 时必须有 `groups`，训练与测试组不能重叠；HCP-YA 契约默认要求家庭分组，但必须由真实研究设计核准。 |
| `preprocessor_file` | 原研究的可导入 Python 文件，路径组成均为合法 Python 标识符；定义 `fit_preprocessor(X_train,y_train,C_train)`，只接收训练数据。 |
| `preprocessor_sha256` / `preprocessing_dependencies` | adapter 文件和全部实际预处理依赖的 SHA256；变动后旧协议失效。 |
| `native_options` | 原研究明确的官方 Tensor-ABESS 参数；缺省仅代表官方缺省，不自动继承模拟或 LEMON 配置。全局 MSIC 参数保持固定。 |
| `baseline_predictions_csv` | 保留原研究完整 baseline OOF 预测，列 `subject_id,repeat,outer_fold,model,y_true,y_pred`。所有模型与所有外折均须完整。 |
| `baseline_models` | 原研究已存在的全部比较模型，至少包含 `Covariates Only`；不得按外测优劣删选。 |
| `baseline_manifest_json` | 下面规定的历史身份文件，证明 baseline 确由相同输入、划分和训练预处理产生。缺失时停止，不能先假定等价。 |

拟合后的预处理对象须提供：

```python
apply(X, C)                  # 使用训练参数变换张量
transform_response(y, C)    # 仅在训练响应上调用
predict(B, X, C)            # 返回原声明响应尺度上的预测，不接收测试 y
to_arrays()                # 返回可安全 NPZ 保存的训练状态数组字典
```

预处理器不能自行读取测试标签或进行结果驱动的参数选择。固定预处理代码及全部依赖必须在正式运行前审阅；接口隔离不能代替源代码审计。本次通用契约的单元测试使用小型明确标记的测试夹具，不代表 HCP/NKI 实证结果。

历史身份文件须具有：

```text
data_array_sha256: empirical.array_identities(data) 的完整结果
splits: 与 splits_json 逐项一致
preprocessing: empirical.preprocessing_identities(prepared) 的逐折训练状态哈希
preprocessor_sha256, preprocessing_dependencies: 与本配置一致
files: 包含历史 OOF CSV 等实际原产物的 {仓库相对路径: SHA256}
configuration: 原 baseline 拟合/调参配置
source_files: 原 baseline 程序及其依赖的 {仓库相对路径: SHA256}
```

这些值应来自原研究的可审计记录和实际文件，不能只生成与当前输入相同的声明来冒充历史证据。

## 运行与冻结

Windows：

```powershell
.\research\taskcard12\run_empirical.ps1 -Action prepare -Dataset lemon
.\research\taskcard12\run_empirical.ps1 -Action run -Dataset lemon
```

Linux：

```bash
PYTHON=/path/to/python bash research/taskcard12/run_empirical.sh prepare hcp_ya
PYTHON=/path/to/python bash research/taskcard12/run_empirical.sh run hcp_ya
```

`prepare` 只审计输入与训练预处理，不调用 Tensor-ABESS。它新建 `doc/taskcard12/verification/empirical_<dataset>_protocol_draft.json`，不会覆盖既有草案。主线审核完整内容后以相应配置的 `protocol` 路径发布，将 `status` 设为 `FROZEN`，绑定实际 `P0_CORRECTNESS_GATE.json` 的 SHA256；源/输入/环境/基线身份必须仍一致。

`run` 每折只调用一次 `selection.fit_pair`；先保存同一候选池、原规则/FS 选择、系数、训练预处理和 split，再为两臂各调用一次外测预测。保留每人 OOF、训练/测试预测、训练/测试误差、q、形状、支撑、选阶分数和全池。输出目录必须全新；任一不一致立即失败，不自动重试、换容差、舍弃外折或覆盖已有结果。

各模型每次重复的完整 OOF 各计算 RMSE/R²，再取重复均值；全部恶化外折逐一保留。重复使用同一批受试者，不能把各折或重复当作独立样本获得虚假精度。最终算法建议在集群盲确认完成前保持 `PENDING_CLUSTER_RESULTS`。

通用运行器当前实现 RMSE/R²。HCP/NKI 的原评价协议尚未提供，因此不能声称这就是它们已经核验的原指标；若原研究采用其他指标，须先补齐相应评价适配与检查，再冻结该数据集协议，不能在运行后以 RMSE/R² 替换原评价。LEMON 的这两项指标已与历史定义一致。
