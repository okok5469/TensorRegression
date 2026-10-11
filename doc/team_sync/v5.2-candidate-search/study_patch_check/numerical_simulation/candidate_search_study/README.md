# 任务卡3：独立候选搜索研究

正式生产src保持v5.1；研究核心在doc/algorithm_versions/v5.2-candidate-search/src，不能直接替换队友正式代码。Group-BESS/固定BCD/原MSIC数学定义不变。最终有效研究目录使用outputs_candidate_search_study下historical_final、development_final、confirmation_final；早期无_final的三个目录因错误验证器已STOPPED，不能resume、混用或当完整结果。

A0旧caps/旧网格，A1去除caps乘积压缩/旧网格，A2旧caps/全整数，A3新版caps/全整数，A4新版caps/训练原MSIC top3邻域±1一次。所有模型q≤floor(.7n)，新版adaptive每mode动态限制smax；预算2000超过就停止，不偷偷截断。完整枚举仅针对当前caps内合法三元组，位置仍由fixed BCD搜索，非所有位置穷举，也不保证所有维度内合法形状都可达。

先在项目根准备现有环境，不升级或安装：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
module load anaconda3
source activate myenv
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

本机最终计划绑定Py3.12.8/NumPy1.26.4/SciPy1.16.1，不能拿本机plan在Py3.9.25/NumPy2.0.2/SciPy1.13.1直接resume；集群须同一代码在全新输出目录重新plan。以下NEW目录不能已有产物，plan只一次，run完成后才summarize；已有同版本完整任务会校验后跳过，错误/半成品拒绝修补。

```bash
python -m numerical_simulation.candidate_search_study.run plan --cohort historical --n-reps 3 --groups A0 A1 A2 A3 A4 --output-dir numerical_simulation/outputs_candidate_search_study/cluster_historical
python -m numerical_simulation.candidate_search_study.run run --output-dir numerical_simulation/outputs_candidate_search_study/cluster_historical --workers 8
python -m numerical_simulation.candidate_search_study.run summarize --output-dir numerical_simulation/outputs_candidate_search_study/cluster_historical
python -m numerical_simulation.candidate_search_study.analyse --output-dir numerical_simulation/outputs_candidate_search_study/cluster_historical

python -m numerical_simulation.candidate_search_study.run plan --cohort development --n-reps 10 --groups A0 A1 A2 A3 A4 --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development
python -m numerical_simulation.candidate_search_study.run run --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development --workers 8
python -m numerical_simulation.candidate_search_study.run summarize --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development
python -m numerical_simulation.candidate_search_study.analyse --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development
```

这两套是27历史和90开发数据，每数据5组；不是700正式full模拟。开发base20361010、确认base20461010，各+1000scenario+rep，test+1e6。确认集在机制固定和阶段门槛决定后才运行，不提前根据确认结果调规则；MSIC消融由另模块在阶段E执行，不混入本模块默认PhaseD。

同一批候选在各组显式复用训练缓存，因此runtime_sec受组顺序影响；mechanism_overview中的standalone_cost_estimate_sec将缓存fixed fit秒数加回，只是估计，不是独立实测。候选数、真三元组、q、真位置必须分别看。

位置稳定性（历史候选27完整后）：

```bash
python -m numerical_simulation.candidate_search_study.stability_run create --candidate-dir numerical_simulation/outputs_candidate_search_study/cluster_historical --output-dir numerical_simulation/outputs_support_search_stability/cluster_historical
python -m numerical_simulation.candidate_search_study.stability_run run --output-dir numerical_simulation/outputs_support_search_stability/cluster_historical --workers 8
python -m numerical_simulation.candidate_search_study.stability_run summarize --output-dir numerical_simulation/outputs_support_search_stability/cluster_historical
```

三初始化marginal/random(seed+30m)/random(seed+31m)，同shape以训练RSS选best；Oracle真实位置OLS只作评价。旧稳定性无_final历史目录的两预检也已STOPPED，最终须与_final新candidate计划绑定。

算法测试不要求pytest安装；不同核心版本分别使用独立Python进程：

```bash
python -m unittest tests.test_candidate_search_core -v
python -m unittest numerical_simulation.candidate_search_study.test_run tests.test_candidate_study_analysis -v
PYTHONPATH="$PWD/doc/algorithm_versions/v5.1-bcd-fix/src:$PWD/src" python -m unittest tests.test_support_search_stability tests.test_stability_run -v
```

Python3.9源码兼容已静态检查，本机实测另注明；不宣称已经在集群实际运行。最小增量部署需要独立研究source包33文件（可用已一致v5.1源码复制后更新四个研究文件，但必须保留独立目录）、本candidate_search_study模块、版本说明；stability还要v5.1冻结32source包。沿用既有data_generation、metrics、version_validation/worker.py、known_s_cluster/profiles.json（如版本不同才重传）。历史精确复现还要旧outputs_msic_ablation_v51_final/requests的27个JSON，开发/确认无需旧训练原始数据。不要传test_tools、失败目录或大型旧结果，不制作ZIP，不替换生产src。

诊断误检记录：验证器revision1把可变q adaptive-stage也当fixed accepted RSS，dev033 q4→2时RSS增但MSIC降导致假失败；revision2-stage-objectives仅fixed强制RSS非增并加强best visited，adaptive共同容量/支持/有限/RSS一致，不改算法或数值容差。旧source与输出全部保留STOPPED，新同seed整批重跑，不作结果后处理。
