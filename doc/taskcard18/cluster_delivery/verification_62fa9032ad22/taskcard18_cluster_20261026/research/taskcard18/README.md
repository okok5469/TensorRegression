# 任务卡18复现入口

本目录包含隔离基线、身份、统计、文稿构建及本卡集群执行器。正式32源、38项冻结源、统计协议与历史输出保持冻结。用户根据本机实测耗时授权改用现有集群，授权及原协议哈希绑定见 `doc/taskcard18/EXECUTION_AMENDMENT.json`；这是执行位置变更，统计协议仍只有一次v2修订。本机训练已停止，集群是否提交以包内 `cluster_submission.json` 记录为准。

## 当前集群入口

将独立ZIP解压到 `/public/home/xjh/xmx/`，得到 `taskcard18_cluster_20261026/`。若选择其他目录，只需替换下面命令中的绝对路径。使用全新展开目录，不合并本机缓存或历史运行目录，也不预先创建 `runtime/`。脚本自动定位包根，先执行 `module load anaconda3`、`conda activate myenv`；使用 `bash -l` 保留站点登录初始化。

首次提交（只执行一次）：

```bash
bash -l /public/home/xjh/xmx/taskcard18_cluster_20261026/run_taskcard18_cluster.sh submit
```

查看本包已记录作业的状态：

```bash
bash -l /public/home/xjh/xmx/taskcard18_cluster_20261026/run_taskcard18_cluster.sh status
```

停止本包已记录的活跃作业：

```bash
bash -l /public/home/xjh/xmx/taskcard18_cluster_20261026/run_taskcard18_cluster.sh cancel
```

确认本包无活跃作业后续跑：

```bash
bash -l /public/home/xjh/xmx/taskcard18_cluster_20261026/run_taskcard18_cluster.sh resume
```

手动打包已有结果及失败现场（无需拟合成功）：

```bash
bash -l /public/home/xjh/xmx/taskcard18_cluster_20261026/run_taskcard18_cluster.sh bundle
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
