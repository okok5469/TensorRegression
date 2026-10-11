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


## Phase E：冻结库、完整重拟合、独立确认

本轮真实开发90×11已完成，三个fresh task0/40/80通过。锁定研究候选alpha2/scale0；当前独立确认仍在运行，正式src仍v5.1。新研究身份v6-msic-after-search-candidate与任务卡2旧v6-msic-candidate分开，均不是正式批准。规则评价只改global alpha/scale，mode保持alpha1；A3 full整数网格不依赖global评分，故新规则不会改变训练候选库。A4粗邻域依赖global评分，不能把在旧A4库上的重评分称为其完整新规则算法。

以下为新环境完整复现的额外阶段：沿用上文cluster_development已完成的90数据、准备原同seed独立确认计划。输出必须全新，不能重复执行已完成目标。先运行31个wrapper测试，无需pytest或环境升级：

```bash
python -m unittest tests.test_msic_after_search tests.test_msic_refit tests.test_msic_confirmation -v
python -m numerical_simulation.candidate_search_study.contrasts --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development
python -m numerical_simulation.candidate_search_study.numerics --output-dir numerical_simulation/outputs_candidate_search_study/cluster_development
python -m numerical_simulation.candidate_search_study.msic_after_search --input-dir numerical_simulation/outputs_candidate_search_study/cluster_development --output-dir numerical_simulation/outputs_msic_after_search/cluster_development
python -m numerical_simulation.candidate_search_study.run plan --cohort confirmation --n-reps 10 --groups A0 A1 A2 A3 A4 --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation
```

读取完整rule_summary.csv与scenario_summary.csv后锁定研究规则。下列代码固定复现本轮事先优先的alpha2/scale0，不自动按最小PMSE挑参；其他参数须另外预声明。LOCK绑定该环境实际父plan和汇总SHA，不可照搬本机LOCK：

```bash
python - <<'PY'
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
search = Path('numerical_simulation/outputs_candidate_search_study/cluster_development')
scoring = Path('numerical_simulation/outputs_msic_after_search/cluster_development')
confirm = Path('numerical_simulation/outputs_candidate_search_study/cluster_confirmation')
sp = json.loads((search/'plan.json').read_text(encoding='utf8'))
mp = json.loads((scoring/'plan.json').read_text(encoding='utf8'))
cp = json.loads((confirm/'plan.json').read_text(encoding='utf8'))
checks = json.loads((scoring/'integrity_report.json').read_text(encoding='utf8'))
assert checks['checks_passed'] and checks['dataset_count']==90 and checks['default_exact_replays']==90
assert not any((confirm/'jobs').iterdir()) and not any((confirm/'logs').iterdir()) and not any((confirm/'failures').iterdir())
body = dict(status='locked', search_group='A3', research_only=True, release_approved=False,
    rule=dict(id='msic-alpha2-scale0', criterion='msic', alpha=2.0, gamma=1.0, scale=0.0),
    development_search_plan_id=sp['plan_id'], development_msic_plan_id=mp['plan_id'],
    development_summary_sha256=hashlib.sha256((scoring/'rule_summary.csv').read_bytes()).hexdigest(),
    confirmation_search_plan_id=cp['plan_id'], source_core_sha256=sp['sources']['core'],
    scoring_source_sha256=mp['research_sources'], fresh_refit_tasks=[0,40,80],
    locked_utc=datetime.now(timezone.utc).isoformat(),
    selection_reason='Replicate predeclared priority research alpha2/scale0 before untouched confirmation; not formal adoption or automatic mean-PMSE selection.')
with Path('numerical_simulation/outputs_msic_after_search/LOCK_cluster.json').open('x',encoding='utf8') as f:
    json.dump(body,f,indent=2); f.write('\n')
PY
```

随后三个代表task各完整重新拟合（不借旧候选cache），全部成功才启动确认；Bash一段脚本应在开头使用set -e，任何失败即停：

```bash
set -e
for task_id in 0 40 80; do
  python -m numerical_simulation.candidate_search_study.msic_refit --source-output numerical_simulation/outputs_candidate_search_study/cluster_development --msic-output numerical_simulation/outputs_msic_after_search/cluster_development --lock-file numerical_simulation/outputs_msic_after_search/LOCK_cluster.json --task-id "$task_id" --output-dir numerical_simulation/outputs_msic_after_search/cluster_refit
 done
python -m numerical_simulation.candidate_search_study.run run --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation --workers 8
python -m numerical_simulation.candidate_search_study.run summarize --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation
python -m numerical_simulation.candidate_search_study.analyse --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation
python -m numerical_simulation.candidate_search_study.contrasts --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation
python -m numerical_simulation.candidate_search_study.numerics --output-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation
python -m numerical_simulation.candidate_search_study.msic_confirmation --input-dir numerical_simulation/outputs_candidate_search_study/cluster_confirmation --development-search-dir numerical_simulation/outputs_candidate_search_study/cluster_development --development-msic-dir numerical_simulation/outputs_msic_after_search/cluster_development --lock-file numerical_simulation/outputs_msic_after_search/LOCK_cluster.json --output-dir numerical_simulation/outputs_msic_after_search/cluster_confirmation
```

确认只评价原A3默认+LOCK两规则，不再从11规则选最好；原五组父结果保留用于与A0/A2同seed比较。新锁模型另外保存90行独立OLS/rank/condition审计；不能把990次缓存评价称990次新拟合。当前输出不可覆盖，复制/换代码/换环境后均用新目录新计划。默认测试流只用于模拟评价。

增量文件：三个msic_*.py、三tests（tests仅测试时需要）、本README和独立版本身份；此前candidate_search_study与独立v5.2 source33、data_generation/metrics、version_validation/worker、profiles依赖仍必须匹配。无需新ZIP、test_tools或环境更新。P5补丁分别为v5.2的四核心patch与16文件study patch，以及v6-msic-after-search-candidate的六文件MSIC patch；README新增说明有独立readme-update.patch，代码和文档diff分开。

预算2000指全局fixed规模搜索形状数，池还包含一个adaptive终态（因此池计数多1）；不是时间预算。较大full场景须先按最终版本预检预算/成本，不能把预算失败改成静默粗网格。本轮不启动700正式full或实际实证重训。
