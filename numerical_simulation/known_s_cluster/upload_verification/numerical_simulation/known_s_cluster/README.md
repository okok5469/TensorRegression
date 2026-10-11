# 完整已知 s 实验：集群执行说明

独立入口调用现有算法，不修改原数值模拟入口或核心算法。默认沿用原汇报 14 个场景 × 50 次重复，比较 fixed、adaptive、Oracle：700 个数据任务、2100 个方法结果。每任务共用训练数据和测试随机流；fixed 只获知数量 `(4,3,5)`，真实位置只传给 Oracle。系数仍为随机符号 × U(0.8,1.2) 后按目标 SNR 缩放；常量系数、新论文基线、统一初始化诊断均不混进本轮主实验。

## 必须上传的文件

从上传包根目录执行，保持相对结构：

```text
src/tensor_abess/                         全部 .py 核心源码及子目录
numerical_simulation/__init__.py
numerical_simulation/data_generation.py
numerical_simulation/methods.py
numerical_simulation/metrics.py
numerical_simulation/known_s_cluster/run.py
numerical_simulation/known_s_cluster/profiles.json
numerical_simulation/known_s_cluster/requirements.txt
numerical_simulation/known_s_cluster/submit_slurm.sh   Slurm 时需要
numerical_simulation/known_s_cluster/test_run.py       可选的验证测试
numerical_simulation/known_s_cluster/README.md         本说明
```

交付的 `known_s_cluster_upload.zip` 包含以上文件及哈希清单，上传解压即可。full 配置冻结于 profiles.json，不依赖旧 checkpoint/main.py/runner.py/plots.py；不需上传旧输出、Presentation、LEMON 数据、.venv 或 .git。旧 CSV 只在另做历史比较时需要。生产入口仅依赖 NumPy、SciPy；现有模拟 Vector Lasso 是内部实现，不需要 sklearn；新文献基线未接入。

## 环境与小规模检查

建议 Python 3.12，集群各任务保持同一环境；如用 module/conda，先按集群规范激活对应环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r numerical_simulation/known_s_cluster/requirements.txt
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m unittest numerical_simulation.known_s_cluster.test_run -v
python -m numerical_simulation.known_s_cluster.run plan \
  --profile quick --n-reps 1 --methods fixed adaptive oracle \
  --output-dir results/known_s_quick_check
python -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_quick_check --all
python -m numerical_simulation.known_s_cluster.run merge \
  --output-dir results/known_s_quick_check
```

quick 仅作集成检查，不替代正式实验；目录不能覆写，重新检查请换新目录。本实验无 GPU 依赖。

## 正式计划与单任务运行

```bash
python -m numerical_simulation.known_s_cluster.run plan \
  --profile full --n-reps 50 --methods fixed adaptive oracle \
  --output-dir results/known_s_full_50
python -m numerical_simulation.known_s_cluster.run status \
  --output-dir results/known_s_full_50
python -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_full_50 --task-id 0
/usr/bin/time -v python -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_full_50 --task-id 500
```

task-id=`scenario_index×n_reps+replicate`，0 起算；50 次计划为 0–699，任务 500 是最大维度 `64×30×40` 的第一个重复。完整场景/种子先确定再分片，前 10 次 seed 与原汇报一致。核验发现历史 v3/v5 各非 baseline 实验组都用了 `10000` 偏移，这与旧 main.py 一次 full 运行按组递增偏移的规则不同；新 profiles 明确冻结历史实际映射：baseline `2026+replicate`，其余各组 `12026+100×组内序号+replicate`。新入口拆任务不会再改变映射。计划固定后不得更改方法/代码/配置继续同一批次。

每任务先申请 1 CPU、8 GiB 内存，以上 baseline/最大维度试测后再定并发；这是初始建议，未在用户集群测量。最大训练张量单副本约 123 MB，变换和拟合有多个副本，不能只按单副本申请。计算命令应在计算节点运行。

普通 Linux 计算节点可顺序执行整套：

```bash
python -m numerical_simulation.known_s_cluster.run run \
  --output-dir results/known_s_full_50 --all --resume
```

已完成任务只有在 `--resume` 且严格校验通过后才跳过；重复运行不指定 resume 会报错。不吞拟合异常，不自动修改损坏提交，也不调用其他算法补齐失败；不同进程分配不同 task-id，不共写 CSV。

## Slurm 数组任务

先建日志目录，按集群要求另加 `--partition`/`--account`，资源参数以试测为准：

```bash
mkdir -p results/known_s_full_50/logs
export KNOWN_S_PYTHON="$PWD/.venv/bin/python"
sbatch --array=0-699%32 --cpus-per-task=1 --mem=8G --time=12:00:00 \
  --output=results/known_s_full_50/logs/%A_%a.out \
  --error=results/known_s_full_50/logs/%A_%a.err \
  numerical_simulation/known_s_cluster/submit_slurm.sh results/known_s_full_50
```

`%32` 是最多同时 32 个任务，不是每任务 32 个 CPU。脚本统一 BLAS 线程为 1 并添加 resume。只重跑缺失任务可按 status/日志中的真实编号提交 `--array=12,37,88%3`；不要同时运行同一个 task-id。未完整或损坏任务会明确报错，先诊断，不自动清理旧产物。

## 改规模或加入已有常规基线

- 复现旧规模：plan 改 `--n-reps 10` 并换新目录，Slurm 改 `--array=0-139%32`，最大维度首任务变为 100。
- 只需 known-s：plan 用 `--methods fixed`，但无法生成同批 adaptive 配对/Oracle 搜索缺口；不可无校验拼接旧结果。
- 重新比较全部常规基线：plan 用 `--methods fixed adaptive oracle glasso lasso` 并换目录，50 次时是 700 任务/3500 方法结果。GLasso 历史中位约 936 秒、最大约 6218 秒，Vector Lasso 中位约 109 秒、最大约 529 秒；旧机器耗时不保证集群时长，应重新试测，因此默认三方法先完成 known-s 诊断。

指标沿用旧模拟定义以便复现：`element_f1`、`q_hat` 按估计模态支撑的笛卡尔积计算。对 Vector Lasso 的非矩形非零集合，这属于模态投影所诱导矩形的指标，不是实际非零元素 F1/数量；后续引入向量新基线时，应另报实际元素支撑指标，不能直接混用这两个口径。

## 全部完成后汇总

```bash
python -m numerical_simulation.known_s_cluster.run status \
  --output-dir results/known_s_full_50
python -m numerical_simulation.known_s_cluster.run merge \
  --output-dir results/known_s_full_50
```

merge 严格检查任务/方法完整性、身份、源码/配置、有效指标后才写 `raw_results.csv`、`summary_results.csv`、`paired_fixed_vs_adaptive.csv`、`search_gap.csv`、`integrity_report.json`；缺项报错，已有汇总拒覆写。计划只选择 fixed 时，其他方法的配对/搜索缺口明确不适用。汇总包含样本标准差（ddof=1）、分位数和完全恢复成功数；配对差文件用于后续区间估计，不预先宣称统计显著性。

保留全部每任务 JSON/NPZ 和汇总，打包带回本地分析；不覆盖 outputs_full_v3/v5_adaptive 或旧导师补充实验目录。正式集群运行由用户执行，交付及本地验证不代表正式 14 场景实验已完成。
