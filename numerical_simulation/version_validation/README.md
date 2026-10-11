# v5原版、BCD修复和独立MSIC消融

算法修复为v5.1，只改核心bcd.py；原v5源码完整快照位于doc/algorithm_versions/v5-original/src。MSIC研究不改默认，不宣称v6已成立。

最终完成结果在 `numerical_simulation/outputs_msic_ablation_v51_final/`，报告见 `doc/MSIC_ABLATION_REPORT.md`；旧 `outputs_msic_ablation/` 是失败草案的中止记录，禁止用于结论。以下命令使用新的复现目录，不覆盖最终结果。

## 本机配对研究

从项目根目录，用现有NumPy/SciPy环境：

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest numerical_simulation.version_validation.test_selection numerical_simulation.version_validation.test_plan -v
python -m numerical_simulation.version_validation.paired plan --output-dir numerical_simulation/outputs_msic_ablation_reproduce --n-reps 3
python -m numerical_simulation.version_validation.paired run --output-dir numerical_simulation/outputs_msic_ablation_reproduce --workers 2
python -m numerical_simulation.version_validation.paired summarize --output-dir numerical_simulation/outputs_msic_ablation_reproduce
python -m numerical_simulation.version_validation.analyse --output-dir numerical_simulation/outputs_msic_ablation_reproduce
```

已存在计划不能重新plan；相同源码/输入/环境且完整产物通过SHA校验后，run可继续，发现失败记录会停止，不静默混用版本。两版本在独立Python进程加载；27个数据集/54任务，9场景n100/200/400×SNR0.5/2/4，每场景3次，seed=20261010+1000×scenario_id+rep，test+1e6，不冒称旧历史种子。

修复版一次候选搜索，精确重评分alpha1.5/2/3×one-SE0/.5/1、tensor-SIC（带组合罚的BIC风格）、纯BIC。各消融方法共享候选搜索耗时，不能把其runtime相加当独立模型总训练时间。mode-wise alpha仍1/gamma1，未与global惩罚混为一谈。

n200子集另做五折训练内CV预设矩形规模选择，每折重新搜索支持，选择后全训练重拟合才看独立测试；没有学习型全样本预处理。q120事先因内折容量上限112排除，真实q60仍在网格。只有3重复属于探索性证据。

## 修复版full集群复验（尚未在集群提交）

沿用用户现有Python3.9.25/NumPy2.0.2/SciPy1.13.1、myenv、CMO_HPC及xjh账号，不安装或升级环境。

旧项目已在集群时，新增/重传：

1. `src/tensor_abess/tensor/bcd.py`：v5.1源码（必要）。
2. 根目录 `run_bcd_fix_full.sh`：新脚本（必要）。
3. `numerical_simulation/known_s_cluster/run.py`及`profiles.json`：沿用上轮已上传版本；若与本机不同，才重传，并核对依赖源码同版。
4. `tests/test_bcd_fix_regression.py`可选验收；文档/版本manifest建议保留，不是算法生产依赖。原v5完整快照用于重现旧版，不是本修复版full运行必需。

登录集群后：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_bcd_fix_full.sh prepare
sbatch run_bcd_fix_full.sh
# squeue确认该批任务全部结束后：
bash run_bcd_fix_full.sh merge
```

700任务，每个相同数据上跑adaptive/Oracle/Vector Lasso/Tensor-GLasso四种方法，共2800条结果，输出单独 `numerical_simulation/outputs_v51_bcd_fix_full/`，保留原full五实验组baseline/correlation/dimension/sample_size/snr及历史实际种子映射。当前不能严格跨版本/环境确认旧对照结果指纹，所以此入口重算四方法，不混入旧基线。

脚本先校验现有环境及修复源码规范LF SHA256。prepare只跑一次；断点续跑直接重新sbatch。若采用任何未来v6，必须另建明确版本和输出目录，不能复用v5.1计划。
