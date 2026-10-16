# Windows到Linux集群部署指南

本地D2已正式发布，远端清单/环境、实际上传和真实SLURM预检全部NOT_RUN；没有SSH主机信息或远端访问结果时不填推测SHA。用户在目标集群执行下列检查，700任务另行确认后才提交。

## Windows：先dry-run，再显式Apply

```powershell
cd 'C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main'
$taskSshHost = Read-Host 'SSH目标（实际user@host或已配置别名）'
$taskRemoteRoot = '/public/home/xjh/xmx/tensor-sparse-regression-main'
$taskRemotePython = Read-Host '集群现有myenv的python绝对路径（激活后command -v python所得）'
.\scripts\release_v52\sync_to_cluster.ps1 -SshHost $taskSshHost -RemoteRoot $taskRemoteRoot -RemotePython $taskRemotePython
# 阅读真实inventory与UPDATE/CONFLICT清单后，显式上传：
.\scripts\release_v52\sync_to_cluster.ps1 -SshHost $taskSshHost -RemoteRoot $taskRemoteRoot -RemotePython $taskRemotePython -Apply
```

RemotePython使用既有环境，不能凭空猜路径或安装/升级Python/NumPy/SciPy。它解决非交互SSH中python可能不是myenv的问题；在集群登录并激活既有环境后执行`command -v python`取得真实值。SSH主机和RemoteRoot必须是用户实际目标。

默认dry-run只读取远端清单，不上传；也支持`-InventoryPath '已取得的remote_inventory.json'`进行完全本地清单比对。真实远端文件先识别32core+6依赖的target或允许v5.1基准，未知源码/脚本改动报冲突；Apply只上传显式白名单中的bcd、缺失必要依赖及发布/工具文件，不上传旧结果。

本地中文/空格路径作为native参数传递，远端根路径经base64传给Python；本机OpenSSH9.5使用默认SFTP，远端路径不再套旧SCP shell引号（依据[OpenSSH scp手册](https://man.openbsd.org/scp.1)）。本地及远端staging/原文件backup保留作审计，远端原hash与incoming/post hash全部核验；冲突或失败停止受影响步骤，不自动补改或恢复。上传期间不要同时编辑这些目标文件。

## Linux：核验源码及现有环境

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
module load anaconda3
conda activate myenv
bash scripts/release_v52/cluster_inventory.sh "$PWD" > cluster_inventory_v52.json
bash scripts/release_v52/verify_cluster.sh "$PWD"
```

verify检查完整32core、6共享依赖、新launch/Gate工具身份、实际import指向当前src、全700任务映射和Python3.9.25/NumPy2.0.2/SciPy1.13.1；四BLAS线程均1。不把只核对bcd或Bash语法通过当成真实集群通过。2026-10-13按用户确认的实际站点，仅执行module load anaconda3与conda activate myenv，不再source固定conda.sh；激活成功后启用set -euo pipefail。默认项目和SBATCH日志仍使用用户提供的绝对路径。

若项目根不同，设置V52_REMOTE_ROOT，并在sbatch时同时显式覆盖--chdir、--output和--error；不能仅改变同步目标而让SBATCH日志仍写默认项目。默认目标与用户提供的旧脚本一致，不依赖SLURM spool的脚本目录。

实际环境版本不符时停止预检、核对真实环境，不直接升级库或放开版本门禁。

## 已结束却没有任何日志：最小诊断流程

提交返回Job ID仅代表调度器接受，不等于脚本已经开始运行（[Slurm sbatch说明](https://slurm.schedmd.com/sbatch.html)）。先查原作业的状态和退出码，不因没有实验输出就重复提交：

```bash
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName%24,State,ExitCode,Reason%40
```

若知道原Job ID，用`sacct -j 原JobID --format=JobID,State,ExitCode,Reason%40`缩小范围；账号未启用sacct时需取`scontrol show job 原JobID`或向集群管理员获取作业启动失败记录。本轮没有Job ID/调度器记录，不能确定缺目录、排队/取消还是运行期错误。

确认旧作业已退出、两个新文件已上传后，在登录节点执行：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main || exit 1
mkdir -p numerical_simulation/v52_search_final_full_logs
module load anaconda3
conda activate myenv
python -c 'import platform,numpy,scipy; print(platform.python_version(),numpy.__version__,scipy.__version__)'
if [ -f numerical_simulation/outputs_v52_search_final_full/plan.json ]; then
  bash run_v52_search_final_full.sh status
else
  bash run_v52_search_final_full.sh prepare
fi
```

目录必须在sbatch前存在；不要依赖计算节点脚本运行后才创建其自身SBATCH日志父目录。新prepare已在版本校验之前创建日志目录，但环境激活若失败，仍需要上面的提交前mkdir。prepare/status发生在登录节点，第一条真实错误直接显示在终端；失败时停止，不继续提交，不删已有计划或改身份。

上述检查成功后，只提交task0：

```bash
sbatch --array=0 run_v52_search_final_full.sh
```

task0完成后再按V52_SIMULATION_SUBMISSION执行preflight_gate；完整700尚需用户确认。若仍无日志，请保留Job ID及sacct的State/ExitCode/Reason，不能据此断言source是唯一原因。

## task0完成但缺少SLURM作业JSON记录

2026-10-13用户状态为completed=1、locked=[]，slurm目录存在task_000000_36215_*.log（20字节）但无JSON；Gate因此没有完整匹配的单任务作业记录。代码复核确认job_record将相对--log路径与绝对output执行relative_to会抛ValueError；本地测试先FAIL再GREEN，改为双方resolve后计算相对路径。远端对应.err尚待读取，不把本地复现等同于已获取远端完整异常。

本次同步两个文件：scripts/release_v52/release_control.py和doc/releases/v5.2-search-final/RELEASE_MANIFEST.json，启动脚本及核心32文件不变。保留原plan、task0 JSON/NPZ与原日志；不再次prepare，不删除已完成结果。更新后通过真实SLURM执行原resume路径：

```bash
sbatch --array=0 --output="$PWD/v52-preflight-%j.out" --error="$PWD/v52-preflight-%j.err" "$PWD/run_v52_search_final_full.sh"
```

既有完整task0经read_task严格核验后返回skipped，不进入数据生成或模型拟合；新真实作业随后自然写入带校验和的作业JSON。若源/环境/断点不一致则停止，不覆盖结果。新作业结束后，在激活myenv的当前终端执行preflight_gate；Gate仍核对单任务metadata、plan/result/log校验和及resume文件字节不变，不能直接放宽筛选或自行补造记录。

要查看原错误，逐条执行`cat v52-preflight-36215.err`；目录检查单独执行`ls -lh numerical_simulation/outputs_v52_search_final_full/slurm/`，不要把中文连接词写进shell命令。

## 正式模拟与恢复

prepare、单任务真实SLURM、Gate、完整提交、status与merge全部命令见[V52_SIMULATION_SUBMISSION.md](V52_SIMULATION_SUBMISSION.md)。无真实单任务记录和严格resume验收时不能生成有效集群Gate或运行全700。

旧结果目录、历史快照、实证数据与无关旧shell均不改；新输出为numerical_simulation/outputs_v52_search_final_full。原v5/其它依赖不匹配时停止上传，不只覆盖bcd猜兼容。

## 下载汇总

完整merge通过后Windows执行：

```powershell
.\scripts\release_v52\download_results.ps1 -SshHost $taskSshHost -RemoteRoot $taskRemoteRoot -Destination 'C:\XiangMingXu\v52-simulation-summary'
# 检查dry-run后，Destination须是不存在的新目录：
.\scripts\release_v52\download_results.ps1 -SshHost $taskSshHost -RemoteRoot $taskRemoteRoot -Destination 'C:\XiangMingXu\v52-simulation-summary' -Apply
```

只下载raw_results.csv、summary_results.csv、integrity_report.json、plan.json，不默认传巨大NPZ。脚本检查700/2800、四方法full50重复、plan_id和唯一方法行；不完整或冲突留存下载现场并报错，不覆盖旧目录。完整模型/源指纹审计仍需远端原runner与Gate，汇总下载不能替代全部NPZ验收。
