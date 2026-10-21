# 任务卡7：250任务回传验收与最终闭环

2026-10-17（北京时间）。本轮读取已完成的集群结果，补齐Phase B及最终研究决策；没有运行拟合、重新merge、原环境数据生成、上传、提交作业或联系队友。任务卡7的LEMON A、训练机制/内验证C沿用既有完整证据，不重复训练。

## 原始材料与保护

实际结果位于 `numerical_simulation/outputs_v51_attribution/cluster_run`：250任务JSON与250 NPZ、五份Gate、成功preflight、冻结plan、原生CSV及PASSED250的integrity。计划seal为 `01d8005e7c3683b2fe69af4c728f9d798562943a1c1177a3e068718ae329c0cb`；seal是按原序列化规则计算的内部身份，不等同于文件原字节SHA或外部数字签名。

新增 `protected_baseline.json` 保护5664个已有科学/源码/任务卡8及回传原始文件；15个明确可更新的任务卡7报告、README、状态和工作流先逐字节复制到 `pre_update_snapshots/`。旧索引、旧失败记录、旧复现及验收文件均未改写；它们描述此前阶段，新补全索引为本目录 `DELIVERY_INDEX.json`。原任务卡7报告生成器硬编码0/250，不能用它覆盖本轮最终报告。

## 两项独立审计

`audit_cluster.py` 调用原生只读load_plan和seal校验，重建三版96源码与共享依赖身份，核验250记录、原环境/四输入指纹、五Gate结果绑定、750份NPZ哈希及已有CSV每一字段与原生merge逻辑完全一致。已有merge拒覆盖，因此本轮没有再次调用merge。

`audit_estimates.py` 从全部750个所选系数与真支撑独立检查有限性、合法矩形支撑、q/Q和caps，重算REE、各模precision/recall/F1、Element-F1、Exact和数学MSIC；验证封存RSS与loss的恒等式。误差上界：支撑指标2.22e-16、REE4.44e-16、MSIC5.69e-14、RSS/loss恒等误差0。非n400的200组v5.1/v5.2 B和支撑精确一致，n400为1/50一致。

原环境身份为回传记录的Linux CPython3.9.25、NumPy2.0.2、SciPy1.13.1、OpenBLAS0.3.27及四项单线程；本机Windows解释器不冒充该环境。没有归档原X/y或完整测试流/逐测试预测，所以本轮未独立重算原Linux RSS、PMSE、signal MSE或R²；训练指纹核验、测试生成器/配置/test_seed和环境封存记录构成其身份依据。原worker校验入口包导入位置，但记录没有每个模块实际import路径日志；五Gate绑定结果，文件传输时间不能独立证明执行时序，也没有本轮SLURM作业日志审计。

## 配对统计与独立性

`analyze.py` 按原plan中n/SNR/rho绑定场景，不用枚举顺序猜标签：0 baseline、5 n100、7 n400、1 SNR.5、11 rho0。每场景50配对，完整输入250个唯一，但只有150组seed/test_seed族；三个场景共享50个族。因此只作场景内描述，均值、中位数、极值及增/平/减计数不附跨场景IID显著性推断。原700标签中的600唯一输入与本轮不同；138→182/700只是历史标签加权结果。

全部指标及BCD/Grid差存于 `paired_effects.csv`，三版场景汇总为 `version_metrics.csv`，配对变化分布为 `effect_summary.csv`，每场景/阶段的最大、最小PMSE差个案为 `extreme_cases.csv`；极端个案是事后描述，不用于调参。保存runtime与比值来自不同运行/作业，未做同硬件随机化计时，不能解释为严格因果速度比。没有据测试误差选择alpha、caps或其他配置。

## 复现命令与图形

以下命令仅在原文件完整的独立副本、且目标输出不存在时执行；审计和分析输出以x模式拒绝覆盖，不提供残缺救补或覆盖续跑。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.venv/Scripts/python.exe -B doc/taskcard7_completion/audit_cluster.py
.venv/Scripts/python.exe -B doc/taskcard7_completion/audit_estimates.py
.venv/Scripts/python.exe -B doc/taskcard7_completion/analyze.py
.venv/Scripts/python.exe -B doc/taskcard7_completion/plot_decomposition.py --qa-scripts <nature-figure技能的scripts目录>
```

绘图沿用Python/matplotlib，全部250配对点与三版恢复计数纳入；183×90mm、6pt字号下限、PDF/SVG可编辑、PNG/TIFF600dpi。源数据、图契约、1.5pt对齐、最终PDF字体和碰撞检查及逐面板目视记录均在本目录。散点纵向微移仅用于分开标记，没有删点、缩尾、图像后处理或合成数据。

当前算法建议仍为保留v5.2及原MSIC；本轮完成既有归因，不启动下一算法。任务卡8的独立选阶/Warm研究在其原目录和报告中单列，未通过推广门槛的结论不因本次250归因完成而改变。
