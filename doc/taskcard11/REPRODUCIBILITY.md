# 任务卡11复现与证据范围

正式科学定义见 [冻结协议](FROZEN_VALIDATION_PROTOCOL.md)，唯一新规则固定为 `MSIC0+ell*q²/(2n)`，A0/A1共享每例唯一一次原生v5.2候选搜索。所有新文件位于 `research/taskcard11/` 与 `doc/taskcard11/`，仅向 `doc/` 三个既有工作流文件追加记录；不创建分支、不提交、不发布、不修改生产源码、旧版本、旧结果或失败记录。

## 环境和调用

使用仓库既有 `.venv/Scripts/python.exe`：Python3.12.8、NumPy1.26.4、SciPy1.16.1，未安装依赖或改变ACL。以下均从仓库根目录运行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
```

只运行确定性工具测试，不启动完整科学模拟：

```powershell
.venv/Scripts/python.exe -B -m unittest research.taskcard11.test_rescore research.taskcard11.test_experiment research.taskcard11.test_analyze research.taskcard11.test_report research.taskcard11.test_verify
```

原Cold16项沿用原断言和输入，其本轮日志及实际导入路径在 `verification/original_cold16.json`；未将Warm扩展门禁或无关旧测试混入。测试重复执行只记独立测试项，不累计为新测试或新科学重复。

实际阶段命令为：

```powershell
.venv/Scripts/python.exe -B -m research.taskcard11.experiment prepare-pilot --output doc/taskcard11/results/pilot --protocol doc/taskcard11/PILOT_PROTOCOL_READY.json
.venv/Scripts/python.exe -B -m research.taskcard11.experiment run-pilot --output doc/taskcard11/results/pilot
.venv/Scripts/python.exe -B -m research.taskcard11.experiment prepare-formal --output doc/taskcard11/results/formal --protocol doc/taskcard11/FORMAL_PROTOCOL.json
.venv/Scripts/python.exe -B -m research.taskcard11.experiment run-validation --output doc/taskcard11/results/formal
```

这些命令记录本次实际入口，不应对已经存在的目录重复执行；输出目录独占创建并拒绝覆盖，失败或超时不会静默续跑、换seed或清空时钟。重新执行科学拟合不属于本卡结束后的自动授权，不能把复现包装成第二轮新规则调参。

已有产物可在新的审计文件中作独立复核，`--replay` 仅按原seed重生相同输入/测试流并核验保存B，零搜索、不产生新随机族、不增加独立重复数：

```powershell
.venv/Scripts/python.exe -B -m research.taskcard11.verify --run-dir doc/taskcard11/results/formal --cohort validation --replay --output doc/taskcard11/verification/reproduction_full.json
```

完整96例报告的复算使用新子目录，所有统计、配对差及门槛都重新来自原保存结果：

```powershell
.venv/Scripts/python.exe -B -m research.taskcard11.report --input doc/taskcard11/results/formal --output doc/taskcard11/reproduction/analysis
```

报告器拒绝不完整、重复、请求漂移、产物哈希错误或工程门禁失败的队列；不能把部分案例均值当作正式门槛。上述审计/报告路径已存在时同样拒绝覆盖。

## 冻结和保护

`PROTECTED_BASELINE.json`在本轮科学执行前封存7874个历史/正式文件及32个正式源码身份；最终完整保护结果另见 `PROTECTED_FINAL_CHECK.json`。`GIT_INITIAL.json`保留用户既有未提交状态，不将既有删除/修改归到本任务。四类工作流文件不放仓库根目录。

`PILOT_PROTOCOL.json`是未执行的启动前版本，独立review发现并发读peer半写JSON后，保留该文件并在任何新科学输入之前以 `PILOT_PROTOCOL_READY.json`重新封存唯一工程修复。科学公式、DGP、seed、门槛及预算未改；新增fixture证明历史/Pilot仍核验、当前批次重复仍由独占claim拒绝。

正式协议还显式绑定原 `research/taskcard10/cv.py` 参数映射，该文件在Pilot前即受保护且字节不变。所有实际worker核验源代码、环境、协议和完整请求；每个case保存训练输入指纹、完整候选B/RSS/支撑/轨迹、双规则分数与容忍带、独立OLS审计、选择冻结记录、真值及测试预测。测试流只在两规则选择与训练审计保存之后生成。

独立多例重放首次因验收器逐例重复初始化已加载的 core 而停止，原错误保留于 `verification/pilot_replay_initial_failure.json`。修复仅涉及未封存为科学生成代码的 `verify.py` 和其测试：首次配置冻结 core，后续严格核验每个已加载模块仍来自该冻结目录；新增回归测试先失败后通过，随后从头执行原输入重放。最终工具测试记录为 `FINAL_TEST_VALIDATION_POST_REPLAY.json`（47项新增测试加16项原Cold检查），此前46项测试记录保留为中间状态；科学源、协议、候选和实验结果均未修改。

`SEED_AUDIT.json`核对28个可取得历史计划的2010个种子；Pilot6加正式96共204个train/test seed唯一且无已知历史交叉。未公开的外部实验无法核验，不以此声称全世界随机流无重复。每个新训练输入另与已保存完整输入指纹核对；相同实验的回放不算新样本。

## 解释限制

历史216池的全部结构指标可重评分，FS可信PMSE仅178/216对；缺失保持空值，不从训练RSS/REE补造。历史与Pilot均不加入96例前瞻验证，任务卡10必须引用 `continuation_01` 完整开发证据，根路径中止记录保持历史身份。

原n100 caps不能表示真形状(4,3,5)，该组Exact=0不能单归因于选阶；分组同时列可表示性和真支撑入池状态。FS仅依赖q/n，无法改变同形状异支撑的训练RSS排序。每场景4例的bootstrap只描述本次固定设计下配对均值不确定性，不构成所有场景、尾部或恢复非劣的统计证明。

所有科学完成数量、独立验收和最终采用决策，以 [最终验收](FINAL_VALIDATION.json) 与 [终版决策](FINAL_ALGORITHM_DECISION.md) 为准，不能仅凭计划、脚本或报告文件存在宣称实验完成。
