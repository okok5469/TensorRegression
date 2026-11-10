# 任务卡18复现入口

## 返回结果与本轮验收入口（2026-10-30）

集群返回的600套完整输入×2对照已独立验收：1200拟合、1200评价、965964个必要凸拟合证书全部通过，未解失败0；四方法表图及两章已按新证据更新，联合定稿状态以[最终报告](../../doc/taskcard18/TASKCARD18_FINAL_REPORT.md)和[gates.json](../../doc/taskcard18/gates.json)为准。32/38冻结源、数值协议及历史结果保持原身份；本机8个旧训练诊断没有与Linux缓存合并。以下“当前集群入口”保留的是原交付操作说明与当时状态，修复后的实际执行已由本次返回记录验证。

在仓库根目录使用已有环境，不安装依赖：

```powershell
$env:PYTHONPATH="$PWD\doc\taskcard13\runtime;$PWD\src;$PWD"
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

已实际执行并通过的返回验收及缓存重建命令：

```powershell
.\.venv\Scripts\python.exe -B -m research.taskcard18.return_audit --archive numerical_simulation/taskcard18_cluster_results_20261028T105007067551Z.zip
.\.venv\Scripts\python.exe -B -m research.taskcard18.return_tables
.\.venv\Scripts\python.exe -B -m research.taskcard18.figures
.\.venv\Scripts\python.exe -B -m research.taskcard18.return_replay
.\.venv\Scripts\python.exe -B -m research.taskcard18.return_report --write
```

`return_audit`只读原ZIP和历史记录，不解压覆盖缓存、不拟合，核对原交付payload、32/38源、全部700历史记录/NPZ、RUN/PREPARED身份、1200模型及评价；按保存系数重建5份certified CSV并逐字段核对返回汇总。结果写入doc/taskcard18/cluster_review/及results/。`return_tables`仅由验收后的CSV生成新旧对照差异、三轴规模分布、元素/容量/阈值敏感性及附表；`figures`从同一完整口径重建四类PDF/PNG/SVG、主表与自动QA，保留旧草图。改变图表工件后须重新完成可见审查与独立审查绑定，不能只凭自动输出授予FINAL。

`return_replay`不拟合模型；在预声明固定20输入上，以冻结生成器、测试种子及每批100样本，独立重算ABESS、Oracle及两个新方法的80个保存模型，核对240个预测指标。最大绝对差1.78e-15；这是本机与集群的数值对应检查，不是全部1200评价的逐预测复算，也不声称原未归档测试数组的字节复原。

`return_report`默认只读，`--write`仅刷新本轮gates、报告与CHAPTER_FREEZE；它校验返回审计seal、ZIP和CSV哈希、当前数值源码/协议、最终图表全部源/输出及独立审查绑定。原`report`入口仍对应本机旧缓存，不能用它代替返回集群结果验收。重建改变了审计/图表/章节工件后，旧审查绑定会失效，须重新审查相应工件。

本轮限定测试入口及日志为doc/taskcard18/cluster_review/FINAL_TESTS.log；不调用全仓测试或启动作业：

```powershell
.\.venv\Scripts\python.exe -B -m unittest research.taskcard18.test_baselines research.taskcard18.test_identity research.taskcard18.test_closeout research.taskcard18.test_accept research.taskcard18.test_report research.taskcard18.test_cluster research.taskcard18.test_bundle research.taskcard18.test_return_audit research.taskcard18.test_return_report research.taskcard18.test_figures -v
```

正式表图见paper/tables/和paper/figures/；正文见paper/sections/02_method.md、04_simulation.md及paper/appendix/simulation_protocol.md。原返回ZIP的SHA256为`2d444f9e805c7cd8e51c5841cc334d841ab80271b342672b2a91aa2186ca7fc3`。该节所有命令均只处理既有本轮结果，不提交集群、不隐式重拟合。

本目录包含隔离基线、身份、统计、文稿构建及本卡集群执行器。正式32源、38项冻结源、统计协议与历史输出保持冻结。用户根据本机实测耗时授权改用现有集群，授权及原协议哈希绑定见 `doc/taskcard18/EXECUTION_AMENDMENT.json`；这是执行位置变更，统计协议仍只有一次v2修订。本机训练已停止，集群是否提交以包内 `cluster_submission.json` 记录为准。

## 当前集群入口

2026-10-28修复版在解包私有依赖后刷新Python导入缓存，避免首次启动时仍沿用“不存在runtime目录”的缓存而导入共享环境中的旧sklearn。版本、路径、文件SHA及全部数值验收规则不变。回归用例在真实bootstrap提取及导入流程中先复现原报错，再验证修复；本机没有导入Linux二进制，真实HPC修复后运行仍待验证。

将修复版ZIP解压到 `/public/home/xjh/xmx/tensor-sparse-regression-main/`，得到新的 `taskcard18_cluster_20261028/`。原 `taskcard18_cluster_20261026/` 中prepare40887的失败日志和已提取runtime完整保留；该次prepare在训练前退出，无新拟合可迁移。若选择其他目录，只需替换下面命令中的绝对路径。使用全新展开目录，不合并本机缓存或历史运行目录，也不预先创建 `runtime/`。脚本自动定位包根，先执行 `module load anaconda3`、`conda activate myenv`；使用 `bash -l` 保留站点登录初始化。

旧链40888至40891因前置失败而等待，先仅取消旧包记录的活跃作业（保留文件）：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261026/run_taskcard18_cluster.sh cancel
```

首次提交（只执行一次）：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261028/run_taskcard18_cluster.sh submit
```

查看本包已记录作业的状态：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261028/run_taskcard18_cluster.sh status
```

停止本包已记录的活跃作业：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261028/run_taskcard18_cluster.sh cancel
```

确认本包无活跃作业后续跑：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261028/run_taskcard18_cluster.sh resume
```

手动打包已有结果及失败现场（无需拟合成功）：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261028/run_taskcard18_cluster.sh bundle
```

执行器提交带 `afterok` 依赖的五阶段链：prepare核验包、32/38源、离线依赖、限定测试和固定20输入并建立本次身份；pilot数组20项各依次拟合Lasso/GLasso；pilotgate独立验收全部40项证书；formal数组覆盖全部600唯一输入、跳过强校验通过的pilot缓存；collect须先独立验收全部1200个新拟合，才运行新测试评价、默认统计和动态报告。正式ABESS/Oracle复用旧结果，不重拟合。每项1CPU、数值线程1，数组并发上限64，不另猜内存或时限。已有集群环境须为Linux x86_64 CPython3.9、NumPy2.0.2、SciPy1.13.1；三个锁定依赖只从包内校验SHA的wheel解包到私有 `runtime/`，不联网、不修改共享环境。输入验收为数值对应，不冒称未保存数组的字节身份。

重复submit会拒绝。resume会重新提交依赖链，已有完整缓存须强校验后才能跳过；它不放宽数值阈值或预算。若上游失败后仍有等待依赖的作业，先cancel本包作业，再检查现场并resume。孤立NPZ/JSON、未解决的不可变失败或未封存的prepare身份均保留并停止；不得删除、补造或忽略证据后重试。prepare测试日志按attempt保留。此类现场可用bundle回传审查，不能视为可自动恢复的纯中断。

完成collect或手动bundle后，回传 `numerical_simulation/outputs_taskcard18_closeout/return/` 中ZIP及SHA文件；其中包含全部新拟合、评价、分析、身份、日志和失败记录。collect完成不自动授予正文/图表最终验收。当前任务须保持IN_PROGRESS/DRAFT，直到600×2及全部有效性Gate通过。Slurm执行尚须在真实集群验证；本机调度dry-run不等于已提交或已完成。

## 本机历史与复现背景

以下PowerShell入口保留用于理解已完成的本机诊断，本轮执行使用上面的集群入口。在仓库根目录设置已有运行环境：

```powershell
$env:PYTHONPATH="$PWD\doc\taskcard13\runtime;$PWD\src;$PWD"
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

Python3.12.8、NumPy1.26.4、SciPy1.16.1；原有runtime提供sklearn1.5.2。与原集群环境不同，输入数值对应不伪称逐字节匹配。训练生成器与测试批大小100不变。

本机限定检查与身份命令（交付时完整测试结果以本卡最终验证日志为准）：

```powershell
.\.venv\Scripts\python.exe -B -m unittest research.taskcard18.test_baselines research.taskcard18.test_identity research.taskcard18.test_closeout research.taskcard18.test_accept
.\.venv\Scripts\python.exe -B -m research.taskcard18.identity
.\.venv\Scripts\python.exe -B -m research.taskcard18.workflow inputs
```

identity默认只读，`--write`显式写本卡审计/映射；inputs只复验固定20输入，不拟合新方法。它会使用旧Oracle测试指标核对预声明输入身份，不读取新方法测试表现。相同派生输入证据可重复核对，冲突不覆盖。

本机训练pilot曾实测运行，现已停止；原入口保留如下，运行完成与证书以实际输出为准：

```powershell
.\.venv\Scripts\python.exe -B -m research.taskcard18.workflow pilot --workers 4
```

600输入×2新方法为完整必要范围。以下后续Gate/正式补算/测试评价入口已经实现，**尚不能声称已全量运行验证**；必须先通过前置Gate，不能绕过缺项；默认汇总不会调用这些入口：

```powershell
.\.venv\Scripts\python.exe -B -m research.taskcard18.accept pilot
.\.venv\Scripts\python.exe -B -m research.taskcard18.accept formal --workers 4
.\.venv\Scripts\python.exe -B -m research.taskcard18.accept evaluate
.\.venv\Scripts\python.exe -B -m research.taskcard18.analysis
```

formal在启动新worker前独立核验已有缓存，复用同输入不新增重复，完成项不会重复拟合；中断项新建attempt日志，旧日志不覆盖。evaluate先核对全部600×2必要模型，再生成测试指标，不根据已有测试排序改变求解器。存在失败/未达精度时停止，保留不可变失败JSON；不能删除困难任务或仅汇总成功子集。

已完整运行的仅缓存历史指标/配对表构建：

```powershell
.\.venv\Scripts\python.exe -B -m research.taskcard18.analysis --history
```

该命令无拟合/提交，输出2800标签方法行、2400完整唯一方法行、40配置方法单元、56历史标签方法单元和30配置内配对结果；旧两基线明确legacy_uncertified，不能作为新的可靠主对照。默认不带history的汇总缺新缓存直接拒绝。

v1唯一完成的Lasso训练pilot及中断日志保留在 `numerical_simulation/outputs_taskcard18_closeout/initial_training_diagnostics/`；原协议/源在 `doc/taskcard18/protocol_v0_sources/`。v2仅把证书检查改为每50CD扫，硬精度/累计三预算/统计定义不变。规范ID统一带input_，旧诊断编码映射独立保留。当前总状态及可定稿范围必须读取 `doc/taskcard18/gates.json` 和最终报告，不能凭章节文件名判定FINAL。
