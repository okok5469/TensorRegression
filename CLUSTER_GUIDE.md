# 集群配置与实验运行速查

更新日期：2026-10-18（北京时间）。供后续智能体匹配本项目的集群环境、生成提交命令及定位结果；配置来自用户确认、实际终端记录和现有脚本，具体作业进度须查看当次日志与验收文件。

## 后续智能体先看这里

- 用户在Linux集群使用 **`module load anaconda3` → `conda activate myenv`**，无需额外 `source conda.sh`，不要擅自增加固定初始化路径。
- 项目根目录是 **`/public/home/xjh/xmx/tensor-sparse-regression-main`**；优先提供已准备好的 `.sh` 和一整行绝对路径 `sbatch` 命令。
- 用户多次遇到多行命令被粘成一行的问题：不要要求手工输入续行反斜杠或Python here-document；多步骤工作放进脚本。
- **v5.1归因是单作业串行250项；v5.2正式模拟是700项数组，两份脚本不能混用。**
- 保留冻结源码、计划、旧结果及失败现场；输入/环境/Gate失败时停止，不降低校验、覆盖计划或补造Gate。
- 这份文档不授权重新提交历史实验；先核对用户当前目标及已有作业，避免重复运行。

## 已确认的站点与路径

| 项目 | 已确认值 | 使用说明 |
| --- | --- | --- |
| 系统与架构 | Linux，x86_64 | 不用Windows本机的数值环境替代严格配对环境 |
| Shell | Bash，提交脚本使用 `#!/bin/bash -l` | 沿用现有站点登录Shell初始化 |
| 已见登录提示 | `xmx@mu01` | `mu01`是已出现的登录节点名，不等于已确认外网SSH地址 |
| 登录用户名 | `xmx` | 与下面的Slurm account不同 |
| Slurm account | `xjh` | `#SBATCH --account=xjh` |
| 分区 | `CMO_HPC` | `#SBATCH -p CMO_HPC` |
| 集群项目根目录 | `/public/home/xjh/xmx/tensor-sparse-regression-main` | 以下相对路径都相对此目录 |
| Windows工作区 | `C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main` | 这是本地路径，不能放进集群Shell命令 |
| Conda环境 | `myenv` | 用户确认可直接激活 |
| 已观察到的环境目录 | `/public/home/xjh/xmx/.conda/envs/myenv` | 来自集群Python异常栈；需要Python绝对路径时用 `command -v python` 实测 |

## Python与数值环境

| 组件 | 已验证版本或设置 |
| --- | --- |
| Python | CPython **3.9.25** |
| NumPy | **2.0.2** |
| SciPy | **1.13.1** |
| OpenBLAS | **0.3.27**，见原正式计划保存的构建信息 |
| `OPENBLAS_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OMP_NUM_THREADS` | `1` |
| `NUMEXPR_NUM_THREADS` | `1` |
| `PYTHONPATH` | `/public/home/xjh/xmx/tensor-sparse-regression-main/src:/public/home/xjh/xmx/tensor-sparse-regression-main` |

新脚本沿用下面的初始化顺序；`module`和`conda`在严格模式之前执行并检查退出码，避免站点初始化受未定义变量影响：

```bash
module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
cd -- /public/home/xjh/xmx/tensor-sparse-regression-main
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

这段是供智能体写入 `.sh` 的模板，不要求用户手工粘贴。单独写 `NUMEXPR_NUM_THREADS=1` 不保证该变量传给Python子进程，必须 `export`。在子Shell里激活Conda不会激活父终端；新终端也不能依赖之前的 `task_out` 等变量。

严格归因还比较完整环境记录，包括NumPy报告的BLAS构建元数据和路径，**版本号相同不足以替代环境Gate**；不要安装/升级依赖、修改环境字段或用近似输入绕过校验。

## Slurm提交约定

两份现有脚本共用以下设置：

```bash
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
```

| 实验 | 根目录脚本 | 调度方式 | 日志位置 |
| --- | --- | --- | --- |
| v5.1三版本归因补跑 | `run_v51_attribution.sh` | 一个作业，1 CPU，串行250项；**不加 `--array`** | 根目录 `v51-attribution-%j.out`、`.err` |
| v5.2正式full模拟 | `run_v52_search_final_full.sh` | 数组 `0-699%64`；历史单任务预检使用 `--array=0` | `numerical_simulation/v52_search_final_full_logs/%A_%a.out`、`.err` |

`%j`表示作业号，`%A_%a`表示数组主作业号和任务ID；64是本项目设置的最大同时运行数组项数，**不是已确认的集群配额或数组上限**。两份脚本均未显式指定内存、时限或GPU资源，相关限制不能凭经验猜测。

Slurm在脚本运行前打开日志文件，日志父目录必须预先存在；v5.1把日志放在现有项目根目录，v5.2专用日志目录需要提交前建立。不能依靠作业内部的 `mkdir` 修复尚不存在的日志父目录。

## v5.1归因实验入口与结果

此前材料已上传完整且用户要求运行时，从任意登录目录执行这一整行：

```bash
sbatch /public/home/xjh/xmx/tensor-sparse-regression-main/run_v51_attribution.sh
```

脚本自动进入项目、激活环境、在计划缺失时原字节复制计划，依次完成严格预检、五场景试跑、其余245项及原生汇总；不需要手动 `prepare`、`pilot`、`run` 或设置 `task_out`。

| 场景 | 原任务ID | 次数 |
| --- | --- | ---: |
| baseline，n=200 | 0–49 | 50 |
| SNR=0.5 | 50–99 | 50 |
| n=100 | 250–299 | 50 |
| n=400 | 350–399 | 50 |
| rho=0 | 550–599 | 50 |

只新拟合冻结 **v5.1-bcd-fix的adaptive Tensor-ABESS**，v5与v5.2复用已有结果；原任务ID不连续，不能误作数组 `0-249`。五个pilot ID是 **0、50、250、350、550**。

| 用途 | 相对路径 |
| --- | --- |
| 冻结归因计划原件 | `numerical_simulation/outputs_version_attribution/cluster_run/plan.json` |
| 新执行和结果目录 | `numerical_simulation/outputs_v51_attribution/cluster_run/` |
| 新执行计划 | 上述新目录内 `plan.json` |
| 严格输入预检记录 | `preflight_task_000000.json` |
| 各任务结果 | `tasks/task_XXXXXX.json`、`tasks/task_XXXXXX.npz` |
| 五场景Gate | `gates/task_XXXXXX.json` |
| 三版本配对汇总 | `paired_three_versions.csv` |
| 汇总验收记录 | `integrity.json` |
| 原v5参考结果 | `numerical_simulation/outputs_known_s_full_50/` |
| 正式v5.2参考结果 | `numerical_simulation/outputs_v52_search_final_full/` |

原计划文件的SHA256为 `4c84e048353510a65d8db62e531062a081752ae76262112de8d49671e15b4ae3`；这是文件字节哈希，不是JSON内部的seal。上传冻结文件须保持字节和换行，不复制旧Windows失败preflight到新执行目录。

启动脚本只在完整结果的seal、计划/任务/版本/源码/环境/输入、NPZ哈希及pilot Gate均匹配后跳过已有拟合；残缺或错误记录立即停止，不补文件。新增启动层使用文件锁，子进程继承锁；不要同时启动未使用此锁的手动worker/pilot。已有汇总文件也会触发拒绝重写。

日志会显示逐项START、PASS或VERIFIED跳过；`ALL FIVE PILOT GATES PASSED`仅表示五场景通过，**`COMPLETE: 250/250`** 才表示250项校验及原merge成功，科研数值结论还需检查配对CSV。脚本本地限定检查通过不等于已在实际集群完成运行。

## v5.2正式模拟入口与结果

`run_v52_search_final_full.sh`绑定v5.2正式700任务目录，提供 `prepare`、默认数组运行、`status`、`merge`；启动流程需真实单任务SLURM记录和预检Gate，不能把已有task结果当成完整Gate。

| 用途 | 相对路径 |
| --- | --- |
| 正式输出目录 | `numerical_simulation/outputs_v52_search_final_full/` |
| 不可变计划 | 上述目录内 `plan.json` |
| 原始任务JSON/NPZ | `tasks/` |
| 真实作业记录与worker日志 | `slurm/` |
| 集群预检Gate | `cluster_preflight_gate.json` |
| 逐任务指标汇总 | `raw_results.csv` |
| 场景统计汇总 | `summary_results.csv` |
| 完整性验收 | `integrity_report.json` |

该批是14场景×50次=700任务，方法为adaptive、oracle、lasso、glasso，共2800方法结果；不同于用户最初三方法的 `run_known_s_full.sh`。已有输出不能重复 `prepare`，不能删计划重建；新一次实验必须按授权使用独立输出目录。

历史真实预检作业36235曾返回PASSED，task0的 `skipped` 表示校验并复用完整结果；旧发布manifest中的NOT_RUN/早期失败记录是历史快照，不能独自代表后续作业状态。核实进度时以当前结果、日志和原生验收为准。

## 查询命令

下面每个代码块都是一整行。查询排队/运行情况：

```bash
squeue -u "$USER"
```

查询最近v5.1日志文件：

```bash
ls -lt /public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-*.out /public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-*.err
```

查看日志或历史作业状态时，必须把示例 `12345` 换成实际Job ID：

```bash
tail -n 80 /public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-12345.out /public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-12345.err
```

```bash
sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed
```

`sacct`是建议的调度历史查询方式，本聊天尚无本站实际输出；若站点不可用，保留该信息并依据日志继续定位。`squeue`为空只说明当前队列查询没有作业，不能证明成功。

查看v5.2现有计划的进度，不提交新任务：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/run_v52_search_final_full.sh status
```

## 已遇到的问题与指令规则

| 现象或错误 | 已确认原因或应先检查的内容 |
| --- | --- |
| 执行项目路径后出现 `Is a directory` | 进入目录要用 `cd`；路径本身不是命令 |
| `conda motivate myenv` | 正确命令是 `conda activate myenv` |
| `ModuleNotFoundError: numerical_simulation` | 检查是否在项目根目录，以及是否导出正确 `PYTHONPATH` |
| `unrecognized arguments`，末尾像空白 | 曾由 `run \   --output-dir ...` 引起；同一行不要放续行反斜杠 |
| `--task-id: 未找到命令` | 参数被单独执行了；让完整命令在同一行，或使用sh |
| 终端出现 `>`，here-document无输出 | Shell可能仍在等结束标记；不让用户手工拼多行Python，改用脚本 |
| 多行Python压成一行后出现括号附近语法错误 | Shell未正确接收到Python代码，不代表科学拟合失败 |
| `python: 未找到命令` | 先确认当前终端是否激活myenv，不能只看另一个子Shell曾激活 |
| `sbatch: Unable to open file` | 先检查提交脚本的文件名、绝对路径和命令空白；不要先改Conda |
| 没有日志或输出目录 | 先确认是否真实提交成功、Job ID、作业状态、日志父目录及脚本路径 |
| 缺少 `plan.json` | 计划未按本实验流程准备；不能用已有无关计划或重复prepare覆盖历史 |
| Gate提示缺真实SLURM元数据 | 不能只凭task结果放行；需对应脚本生成真实作业记录 |
| 检查长时间静默 | 归因 `load_plan` 会读500份历史任务JSON和96份源码；共享存储可能较慢，不据此判定失败 |
| task `skipped` | 需按该脚本定义解读，通常表示已有完整结果经校验复用，不能直接当作失败 |

优先给用户“上传哪个文件＋一行提交＋日志位置”；不要输出可被误粘到Shell的中文连接词，不默认新终端继承旧变量，不因错误重复提交、删除锁文件或清空结果目录。

## 尚未确认的信息

- 外网SSH主机名/IP、端口、认证方式及可用别名；不要把 `xmx@mu01` 自动当作可连接目标。
- 分区最大运行时长、默认/最大内存、数组规模上限、并发配额、账户余额与GPU队列规则。
- 需要SSH自动部署时的实际Python可执行文件绝对路径，应在激活myenv后运行 `command -v python` 获取。
- 当次作业是否已提交、完成了多少任务、是否通过完整验收，需要用户日志或实际文件证明。

## 关联文件

- [v5.1单文件启动脚本](run_v51_attribution.sh)
- [v5.1归因运行说明及上传依赖](doc/V51_FULL_ATTRIBUTION_REPORT.md)
- [v5.2正式启动脚本](run_v52_search_final_full.sh)
- [v5.2部署说明](doc/CLUSTER_DEPLOYMENT_GUIDE.md)
- [v5.2模拟提交说明](doc/V52_SIMULATION_SUBMISSION.md)
- [仓库约束](AGENTS.md)

后续若用户确认站点配置变动，按新证据更新本文件；不要为修正文档去改历史计划、冻结源码或实验结果。
