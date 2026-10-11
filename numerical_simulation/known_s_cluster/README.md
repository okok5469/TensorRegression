# 完整已知 s 实验：集群执行说明

独立入口调用现有算法，不修改原数值模拟入口或核心算法。默认沿用原汇报 14 个场景 × 50 次重复，比较 fixed、adaptive、Oracle：700 个数据任务、2100 个方法结果。每任务共用训练数据和测试随机流；fixed 只获知数量 `(4,3,5)`，真实位置只传给 Oracle。系数仍为随机符号 × U(0.8,1.2) 后按目标 SNR 缩放；常量系数、新论文基线、统一初始化诊断均不混进本轮主实验。

## 直接照此运行（2026-10-06）

集群环境保持现状：Python 3.9.25、NumPy 2.0.2、SciPy 1.13.1；脚本只校验版本，不执行任何安装或升级。

重新上传以下3个生产文件到对应位置：

```text
run_known_s_full.sh                              项目根目录，新文件
numerical_simulation/known_s_cluster/run.py
numerical_simulation/known_s_cluster/profiles.json
```

建议一并上传本README和 `requirements.txt`；requirements只记录现有环境版本，程序不读取它。`test_run.py`仅作可选测试。原核心源码本轮未修改，集群中若为此前同一版，无需重传。旧的 `submit_slurm.sh` 不用于本方案，也无需上传。

登录集群后只执行下面三条命令：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_known_s_full.sh prepare
sbatch run_known_s_full.sh
```

第二条创建700个任务的固定计划；第三条提交Slurm数组。用 `squeue -u "$USER"` 查看进度，确认该批任务全部结束后执行：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_known_s_full.sh merge
```

`prepare`只运行一次。中断续跑时不要再次prepare，直接再次执行 `sbatch run_known_s_full.sh`；已经完整的任务会通过严格校验后跳过。正式结果位于 `numerical_simulation/outputs_known_s_full_50/`，日志位于 `numerical_simulation/known_s_cluster_logs/`。

根目录脚本沿用你的配置：`CMO_HPC`、账号`xjh`、`module load anaconda3`，然后显式加载 `/public/home/xjh/xmx/.conda/etc/profile.d/conda.sh` 并 `conda activate myenv`，以确保Slurm非交互shell能激活现有环境。数组为 `0-699%64`：共有700个任务，最多同时64个；每个任务只需1个CPU，并在同一数据上顺序比较known-s、adaptive和Oracle。原脚本的 `--ntasks-per-node=64` 只申请了64个task，却只启动一个Python进程，会浪费资源，因此这里改成了1。

## 增量上传的背景说明

集群已有旧项目，直接上传以下文件到原项目的同名相对路径即可，无需压缩包或重新上传整个项目。

| 文件 | 是否需要上传 | 用途 |
|---|---|---|
| 根目录 `run_known_s_full.sh` | 必须 | 与原脚本格式一致的完整Slurm脚本 |
| `known_s_cluster/run.py` | 必须 | 独立计划、任务运行、续跑、状态及汇总入口 |
| `known_s_cluster/profiles.json` | 必须 | 完整参数网格和已经核对的历史种子映射 |
| `known_s_cluster/requirements.txt` | 建议 | 仅记录现有版本，不用于安装 |
| `known_s_cluster/test_run.py` | 可选 | 本地/集群验收测试 |
| `known_s_cluster/README.md` | 建议 | 本说明 |

旧依赖 `src/tensor_abess/`、`numerical_simulation/__init__.py`、`data_generation.py`、`methods.py`、`metrics.py` 本轮没有修改：**集群与本地原版本相同，就不必重传**。若集群保存的是更早的版本，先比对这些依赖，只同步已确认不同的文件；新plan只冻结集群当前源码，不能证明它与本地验收版相同。现有 `UPLOAD_MANIFEST.json` 可选上传，用其中这些旧依赖的SHA256作比较，勿把已更新的README、测试和sh与旧清单比较后误判为旧依赖变化。

full配置不依赖旧checkpoint/main.py/runner.py/plots.py；旧结果、Presentation、LEMON数据、.venv和.git无需重传。生产仅依赖现有NumPy、SciPy；不需要sklearn、matplotlib或第三方abess，也不要运行 `pip install`、`conda install` 或 `pip install -e .`。旧交付副本 `known_s_cluster_upload.zip` 已于2026-10-05按用户确认删除。

## 附录：旧的拆分脚本说明（本次不要使用）

以下内容保留为历史记录。本次运行以文首的根目录 `run_known_s_full.sh` 和三条命令为准。

<details>
<summary>展开历史说明</summary>

以下三个代码块分别可保存成 `01_prepare.sh`、`02_submit.sh`、`03_merge.sh`；这些是说明中的脚本内容，无需额外上传本地脚本文件。先运行01、确认通过后运行02，待全部数组作业结束再运行03。普通 `sbatch` 提交后立即返回，不能把三个阶段无等待地连跑，参见 [Slurm官方说明](https://slurm.schedmd.com/sbatch.html)。

每段开头的 `task_repo` 和 `task_python` 改成同一集群项目根目录、同一Python解释器的绝对路径；Python 3.9环境需使用本目录固定的NumPy2.0.2/SciPy1.13.1。若集群现有 `myenv` 已是这两个版本，无需更新环境；先用 `python -c 'import numpy,scipy; print(numpy.__version__, scipy.__version__)'` 确认。仅当依赖版本不符且获准修改环境时，才执行 requirements 安装命令。无需上传本地虚拟环境。

代码、虚拟环境和结果目录应放在各计算节点都能访问的共享路径；任务提交使用同目录内的原子硬链接，因此该文件系统须支持 `os.link`。不支持时会明确失败，代码不自动改用可能覆盖文件的提交方式。先完成下方 quick 检查与正式单任务试测，再提交数组。

### 01_prepare.sh：小规模检查与正式计划（仅首次）

在集群允许运行检查的计算环境执行；若选择单元测试，须上传test_run.py并有Bash。

```bash
#!/usr/bin/env bash
set -euo pipefail
task_repo="/你的集群项目绝对路径/tensor-sparse-regression-main"
task_python="/你的Python环境绝对路径/bin/python"
cd -- "$task_repo"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# 可选单元测试：取消下一行注释，需上传test_run.py
# "$task_python" -m unittest numerical_simulation.known_s_cluster.test_run -v
"$task_python" -m numerical_simulation.known_s_cluster.run plan \
  --profile quick --n-reps 1 --methods fixed adaptive oracle \
  --output-dir results/known_s_quick_check
"$task_python" -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_quick_check --all
"$task_python" -m numerical_simulation.known_s_cluster.run merge \
  --output-dir results/known_s_quick_check
"$task_python" -m numerical_simulation.known_s_cluster.run plan \
  --profile full --n-reps 50 --methods fixed adaptive oracle \
  --output-dir results/known_s_full_50
# 在计算节点做两个正式任务试测，数组提交时会校验后跳过
"$task_python" -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_full_50 --task-id 0 --resume
/usr/bin/time -v "$task_python" -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_full_50 --task-id 500 --resume
```

执行：`bash 01_prepare.sh`。quick只作集成检查；已有plan或结果会拒绝覆盖，续跑不要重执行01，直接使用既有计划进入02。若是全新实验，显式换新输出目录，并保持各阶段一致。

task-id=`scenario_index×n_reps+replicate`，0 起算；50 次计划为 0–699，任务 500 是最大维度 `64×30×40` 的第一个重复。完整场景/种子先确定再分片，前 10 次 seed 与原汇报一致。核验发现历史 v3/v5 各非 baseline 实验组都用了 `10000` 偏移，这与旧 main.py 一次 full 运行按组递增偏移的规则不同；新 profiles 明确冻结历史实际映射：baseline `2026+replicate`，其余各组 `12026+100×组内序号+replicate`。新入口拆任务不会再改变映射。计划固定后不得更改方法/代码/配置继续同一批次。

每任务先申请 1 CPU、8 GiB 内存，以上 baseline/最大维度试测后再定并发；这是初始建议，未在用户集群测量。最大训练张量单副本约 123 MB，变换和拟合有多个副本，不能只按单副本申请。计算命令应在计算节点运行。

已完成任务只有在 `--resume` 且严格校验通过后才跳过；重复运行不指定 resume 会报错。不吞拟合异常，不自动修改损坏提交，也不调用其他算法补齐失败；不同进程分配不同 task-id，不共写 CSV。

### 02_submit.sh：一次提交700个任务

先建日志目录，按集群要求另加 `--partition`/`--account`，资源参数以试测为准：

```bash
#!/usr/bin/env bash
set -euo pipefail
task_repo="/你的集群项目绝对路径/tensor-sparse-regression-main"
task_python="/你的Python环境绝对路径/bin/python"
cd -- "$task_repo"
test -f results/known_s_full_50/plan.json
mkdir -p results/known_s_full_50/logs
export KNOWN_S_PYTHON="$task_python"
# 需要指定队列/账号时，在sbatch后添加 --partition=你的队列 --account=你的账号
sbatch --chdir="$PWD" --export=ALL \
  --array=0-699%32 --ntasks=1 --cpus-per-task=1 --mem=8G --time=12:00:00 \
  --output=results/known_s_full_50/logs/%A_%a.out \
  --error=results/known_s_full_50/logs/%A_%a.err \
  numerical_simulation/known_s_cluster/submit_slurm.sh results/known_s_full_50
```

执行：`bash 02_submit.sh`，记录返回的job ID，用 `squeue -j jobID` 查看。`%32` 是最多同时32个任务；脚本统一BLAS线程为1并添加resume。`submit_slurm.sh`由sbatch调用，使用Slurm提供的 `SLURM_SUBMIT_DIR` 定位项目，因此02必须先cd到项目根；不再依赖Slurm复制后的脚本路径。只重跑缺失任务时将数组改成status给出的真实编号，例如 `--array=12,37,88%3`，避免重复提交仍在运行的task-id。

### 03_merge.sh：全部结束后检查并汇总

```bash
#!/usr/bin/env bash
set -euo pipefail
task_repo="/你的集群项目绝对路径/tensor-sparse-regression-main"
task_python="/你的Python环境绝对路径/bin/python"
cd -- "$task_repo"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$task_python" -m numerical_simulation.known_s_cluster.run status \
  --output-dir results/known_s_full_50
"$task_python" -m numerical_simulation.known_s_cluster.run merge \
  --output-dir results/known_s_full_50
```

执行：`bash 03_merge.sh`。merge会检查700个任务完整、无锁、指标有效后汇总；缺项报错，已有汇总拒绝覆盖。status只显示状态，完整性门槛由merge检查。

普通Linux计算节点若不用Slurm，在相同Python/线程环境下以 `"$task_python" -m numerical_simulation.known_s_cluster.run run --output-dir results/known_s_full_50 --all --resume` 替代02；该命令同步等待全套结束，成功后可直接执行03。

</details>

## 改规模或加入已有常规基线

- 复现旧规模：plan 改 `--n-reps 10` 并换新目录，Slurm 改 `--array=0-139%32`，最大维度首任务变为 100。
- 只需 known-s：plan 用 `--methods fixed`，但无法生成同批 adaptive 配对/Oracle 搜索缺口；不可无校验拼接旧结果。
- 重新比较全部常规基线：plan 用 `--methods fixed adaptive oracle glasso lasso` 并换目录，50 次时是 700 任务/3500 方法结果。GLasso 历史中位约 936 秒、最大约 6218 秒，Vector Lasso 中位约 109 秒、最大约 529 秒；旧机器耗时不保证集群时长，应重新试测，因此默认三方法先完成 known-s 诊断。

指标沿用旧模拟定义以便复现：`element_f1`、`q_hat` 按估计模态支撑的笛卡尔积计算。对 Vector Lasso 的非矩形非零集合，这属于模态投影所诱导矩形的指标，不是实际非零元素 F1/数量；后续引入向量新基线时，应另报实际元素支撑指标，不能直接混用这两个口径。

## 汇总文件与保留事项

merge 严格检查任务/方法完整性、身份、源码/配置、有效指标后才写 `raw_results.csv`、`summary_results.csv`、`paired_fixed_vs_adaptive.csv`、`search_gap.csv`、`integrity_report.json`；缺项报错，已有汇总拒覆写。计划只选择 fixed 时，其他方法的配对/搜索缺口明确不适用。汇总包含样本标准差（ddof=1）、分位数和完全恢复成功数；配对差文件用于后续区间估计，不预先宣称统计显著性。

保留全部每任务JSON/NPZ和汇总，按需同步回本地；不覆盖outputs_full_v3/v5_adaptive或旧导师补充实验目录。正式集群运行由用户执行，本地验证不代表正式14场景实验已完成。

2026-10-05已按用户确认，仅删除本目录的旧ZIP和直接子目录 `__pycache__/`；后者可在再次运行Python时自动生成。其余文件保留。

`UPLOAD_MANIFEST.json` 是文件版本的SHA256清单，可辅助核对集群旧依赖，生产运行不读取它；`LOCAL_VALIDATION.json` 是本地测试/历史复现结论，`outputs_*` 与 `upload_verification/results/` 是对应的真实验证结果。这些证据有助于复现、排错和后续汇报，保留在本地即可，**不需要为正式实验上传集群**；只有主动核对旧源码版本时才可选上传哈希清单。集群正式运行会自行生成新的plan、任务结果和汇总，应另外保存。
