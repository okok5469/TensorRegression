# 任务卡9复现与交付说明

任务于2026-10-17发起，2026-10-18完成收尾验收。所有研究输出在doc/taskcard9，新增代码仅research/taskcard9。现存生产、冻结源码、历史实验、失败/停止日志不改，不新建分支，不提交或联系外部人员。

## 已完成范围

| 子任务 | 范围 | 新拟合或生成 |
|---|---|---|
| Task7归因 | 250配对、750保存模型、500阶段变化、96源码及原5 Gate | 拟合/生成/预测/merge均0 |
| Task8库审计 | 216库、31968保存B、216计划身份、108确认lock、5184import记录 | 拟合/生成0 |
| pool测试重放 | 预定development n100/n400各6例、1908候选PMSE，旧预测先Gate | 重建相同12输入/测试流，模型拟合0；17秒，限1800秒 |
| 2×2消融 | 864次选择、127872实际源码分数核对 | 只重评分；A1/A2全216测试覆盖不全，NA保留 |
| Pure Warm | 固定规模并集25门禁，含16原Cold与9新契约；一次新输出接口复验 | 小fixture/历史seed2026正确性求解，无测试PMSE，无性能实验 |

Pure Warm通过范围为固定规模额外起点原型。全adaptive/grid/MSIC集成与性能未运行，不推广；原样incoming失败及Task8新规则推广失败仍保留。相同25门禁重复复验不计作50个独立测试。任务9历史保护基线5852文件，最终结果见[保护核验](PROTECTED_FINAL_CHECK.json)。

## 环境与命令

项目现有`.venv/Scripts/python.exe`：Python3.12.8、NumPy1.26.4、SciPy1.16.1；生产与冻结v5.2各32文件相同。Task7原Linux环境不同，仅读保存封存记录，不在本机再生原250输入。

在仓库根目录PowerShell中设置：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
```

只重跑分析数学测试，无数据生成或求解：

```powershell
.venv/Scripts/python.exe -B -m unittest research.taskcard9.test_attribution research.taskcard9.test_pool_msic
```

重算分析时必须使用此前不存在的新目录。以下命令供审计复现使用，本轮不再次执行12例测试流：

```powershell
.venv/Scripts/python.exe -B research/taskcard9/attribution.py --output doc/taskcard9/reproduction_attribution
.venv/Scripts/python.exe -B -m research.taskcard9.pool_msic inventory --output doc/taskcard9/reproduction_pool
.venv/Scripts/python.exe -B -m research.taskcard9.pool_msic replay --output doc/taskcard9/reproduction_pool
.venv/Scripts/python.exe -B -m research.taskcard9.pool_msic rescore --output doc/taskcard9/reproduction_pool
.venv/Scripts/python.exe -B -m research.taskcard9.pool_msic identity --output doc/taskcard9/reproduction_pool
.venv/Scripts/python.exe -B research/taskcard9/test_pure_warm.py --phase green --output doc/taskcard9/warm/new_reproduction
```

已有分析数据和日志采用独占创建，拒绝覆盖；250归因的自定义输出报告也写入该新目录，不覆盖当前交付报告。Pure Warm red日志是功能实现前的失败历史，不能在实现后假装重新获得缺功能失败。

`prepare.py`与`protect.py snapshot`仅用于本轮首次冻结，现有PROTOCOL/BASELINE不能覆盖；不重复运行它们以重置证据。`protect.py check`仅读旧文件并生成新保护核验；`validate.py`重构CSV统计、核对证据/源码/测试/链接并封存本轮索引，现有最终文件也禁止覆盖。

## 输入与结论边界

[PROTOCOL.json](PROTOCOL.json)绑定源代码、依赖、历史Task8计划、12个精确请求与选择规则；[DESIGN.md](DESIGN.md)说明批准来源、执行顺序和预算。源文件名精确差异见[资产审计](EVIDENCE_ASSET_AUDIT.md)。本轮没有浏览互联网或安装依赖，结论均来自本地代码与实验资产。

Task7的750指标重算不含RSS/PMSE独立再生；无X/y时秩、条件数与pool oracle均不可计算。Task8的A1/A2仅141/123例测试PMSE可用，四规则预测比较只用同一12例。pool oracle使用同一测试样本事后取最小，不能用来部署或选择新参数。当前已见的开发/确认结果不是新的独立确认。

仓库初始已有历史test_tools不可访问，并在Git显示删除；本轮未访问其内容、改ACL或修复该状态。根级`rg`曾遇到同一已知权限错误，后续限定到必要路径；无依赖该目录的任务被伪装通过。初始Git状态与最终 tracked状态逐项比较，防止把既有状态归到本次修改。

最终验收还将逐项核对六份要求报告及所有相对链接，保存[FINAL_VALIDATION.json](FINAL_VALIDATION.json)与[DELIVERY_INDEX.json](DELIVERY_INDEX.json)。本卡只推荐下一阶段CV/容量分离与稳定性研究，不启动下一张任务卡。
