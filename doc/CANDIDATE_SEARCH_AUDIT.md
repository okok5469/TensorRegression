# Candidate Search Audit — Phase A

日期：2026-10-10。状态：已完成历史 27 个训练数据集的机制审计；未修改算法、MSIC 或历史报告，未运行新增拟合、多初始化、参数消融或 700 任务 full 模拟。

本文件记录Phase A审计时点的工作范围；任务卡3后续真实配对拟合与位置实验分别见CANDIDATE_SEARCH_COMPARISON.md和SUPPORT_SEARCH_STABILITY.md，不将本节“未运行”解释为全任务最终状态。

## 数据来源与核验

唯一基线为 `src/` 中的 `v5.1-bcd-fix`。历史来源为 `numerical_simulation/outputs_msic_ablation_v51_final/jobs/task_000..026__v5.1-bcd-fix.{json,npz}`；不存在名为 `outputs_msic_ablation_v51_final27` 的目录，27 指该实际 final 目录中的修复版任务数。每个场景 3 次重复，n={100,200,400}，SNR={0.5,2,4}，dims=(32,20,25)，true_sizes=(4,3,5)，q_true=60；k_max=1，max_cycles=6，初始化 criterion=bic，c=3。该历史配置不同于公共入口默认 c=2，不能混称为默认配置全评估。

本次逐任务通过原 `numerical_simulation.data_generation.generate_simulation_dataset`、历史 config/seed 重建训练数据；X_train、y_train、B_true 的 shape/dtype/bytes SHA256 与历史记录完全一致。按历史 test_seed 重建 500 条独立测试流，逐批 X/y/signal 指纹也完全一致；测试只用于验证输入身份，未据此生成候选或调参。历史全部 core_sources 指纹与当前 src 文件一致，各 NPZ 文件 SHA256 一致。3078 个候选的 JSON/NPZ 支撑、尺寸、B 指纹、非零区域及重建训练 RSS 一致；RSS 容差 rtol=1e-10、atol=1e-9。3051 个 global-fixed 候选的接受损失单调、返回损失等于全部已访问损失最小值、best_iteration 索引一致。该核验不等同于仓库全面测试通过。

## 实际流程

1. `utils/init.py::_mode_slice_scores` 以全切片的边际相关平方范数计算分数。每模 C 保留 min(floor(n^(2/3)),p_m) 个切片，只用于 sequential Group Lasso 初始化；初始边际 supports 长度为 min(floor(sqrt(n)),|C|)。GLasso 逐模更新 supports，空选择采用现有最高分回退规则，本审计未更改此行为。
2. 原 caps 为 min(p_m,max(ceil(c*|S_m_tilde|),floor(sqrt(n)/2),1))。随后重复把最大 cap 减 1；并列最大由 np.argmax 选较早模式，直到 cap1*cap2*cap3<=Q，其中 Q=max(1,floor(0.7*n))。源代码将所有上界的笛卡尔积整体压入容量限制，未提供该更强限制的理论必要性证明。
3. adaptive BCD 对每模调用 `build_mode_design`，其候选切片为该模全部 0..p_m-1；`group_bess_adaptive` 搜索 s=1..静态cap。GLasso C 不传给 adaptive 或 fixed BCD，因此边际预筛不是永久切片删除。
4. MSIC refine 候选池首先包含 adaptive 终态，随后每个固定规模用全域边际 top-s 独立初始化并调用修复后的 fixed BCD。固定 BCD 的条件逐模 Group-BESS、有限 k_max/m_max/max_cycles 及训练准则无法保证找到全局最优位置；最终返回最佳完整已访问 OLS 状态。
5. 原规模网格每轴取近 cap 的最后 window=3 个整数，加上 {1,round(0.4*cap),round(0.6*cap)}，笛卡尔积后筛 q<=Q。它是确定性启发式稀疏网格，不是全部合法整数枚举。池包括 1 个 adaptive 终态，因此实际池数通常比 fixed 调用数多 1，可能存在同形状重复。
6. 默认 mode MSIC alpha=1,gamma=1；global alpha=1.5,gamma=1,one_se_scale=1。既有规则在全池上评分后按容差选较小 q。本审计不改变这些规则。

## 结构性诊断结果

| n | Q | 原最终 caps | 维度内全部合法整数形状 | caps 内合法整数形状 | fixed 搜索形状 | 池状态数 | 真形状可达 / 9 | 真形状进入池 / 9 | 真完整支撑进入池 / 9 | 任一 q=60 进入池 / 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 70 | (4,4,4) | 694 | 64 | 64 | 65 | 0 | 0 | 0 | 0 |
| 200 | 140 | (5,5,5) | 1454 | 125 | 125 | 126 | 9 | 9 | 3 | 9 |
| 400 | 280 | (6,6,7) | 2780 | 252 | 150 | 151 | 9 | 0 | 0 | 9 |

27 数据集的首次阻断分类：caps 9、规模网格 9、位置搜索 6、候选覆盖完整 3。真形状总体可达 18/27、进入候选池 9/27、真完整支撑位置进入池 3/27；任意 q=60 模型进入池 18/27。q、矩形形状及实际支撑位置必须分别统计。

n=100：截断前各模 cap 至少为 5，真形状满足 Q=70；cap 乘积压缩把第三模上界降到 4，首次永久阻断发生在 caps，而非 MSIC。n=200：网格每轴为 {1,2,3,4,5}，真形状可搜索；仅 task_013/014/015 的池含真完整支撑，另 6 个任务属于位置搜索失败。n=400：前两模网格为 {1,2,4,5,6}，第三模为 {1,3,4,5,6,7}，遗漏第二模大小 3，虽满足 caps 仍排除真形状；同时漏掉 102 个 caps 内合法形状，任何 q=60 存在不等于真形状存在。

边际 C 完全包含真切片的任务为 20/27；7 个缺片任务是 task_000..005 与 task_008，均为 n=100。CSV 保存缺失切片明细，但它们不是永久搜索排除：后续更新在全维度上搜索。GLasso 初始化可能影响 caps 和位置局部解，却不能把 C 缺片直接归为最终位置搜索的充分因果解释。

## 新 caps 设计的必要约束

解除 caps 乘积截断必须同时保留每个实际模型 q<=Q；只放松 init caps 不安全。现有 `group_bess_adaptive` 只检查 s<=n_groups，不限制 s*block_size；adaptive 三次更新直接使用静态 caps，refine 还无条件纳入 adaptive 终态。

对更新模式 m，必须按当前另两模 sizes 计算 d_m=s_j*s_k、effective_cap_m=min(cap_m,p_m,floor(Q/d_m))，每模更新后立即维持 q<=Q。初始 warm supports 必须先由训练数据与明确规则形成合法状态；否则仅在更新时限 cap 无法保证初始 OLS 或首模候选合法，甚至 effective_cap=0。也需验证候选 refine seed、最终模型以及 target_sparsity 的容量规则，而不能只过滤固定枚举网格。已有合法起点且每次更新守此约束时，当前同模 size 本身合法，effective_cap 不会小于该 size。

cap 的训练数据规则、合法初始化、候选级容量限制、规模枚举与位置优化应物理隔离，数据显式传递；保留既有 Group-BESS 目标与交换定义、fixed BCD 最佳状态不变量。truth 与独立 test 不得进入任何候选生成或训练选择。完整枚举若受预算限制，必须输出被跳过形状，不能称为完整枚举；旧网格、新 caps 两因素须 A0/A1/A2/A3 配对比较，MSIC保持不变。

## 四份新增 CSV 与口径

产物位于 `numerical_simulation/outputs_candidate_search_study/audit/`：

- `candidate_coverage_audit.csv`：27 行，每训练数据集一行，记录 dims/SNR/caps/Q、合法形状数、实际 fixed 调用和池数、真形状/q/支撑覆盖、首次阻断及输入指纹。
- `caps_diagnostics.csv`：81 行，每任务每模一行，记录 GLasso warm support、原 cap、乘积截断后的 cap、边际 C 与遗漏真切片。截断前 cap由历史 warm support和原公式重建并再次模拟最大cap递减，结果逐项与历史最终 caps 一致；没有重跑GLasso。
- `size_grid_diagnostics.csv`：27 行，列出每轴整数、遗漏整数、遗漏全部合法三元组、adaptive 终态形状与实际 fixed 网格核验。预算截断为0，因为历史网格是按规则稀疏取点而非预算截断。
- `support_search_diagnostics.csv`：3078 行，每最终池候选一行，保存该候选 initial/final support、真片重合数、完整/包含覆盖、RSS与重建RSS、停止原因/最优迭代/耗时/B指纹。adaptive终态未记录其mode级轨迹，initial_supports=null，初始记录可用列标0；GLasso入口warm support仍在caps表。

`first_exclusion_stage` 以整体真模型为对象，按 caps→size_grid→support_search 顺序，若最终池有真完整支撑则标 covered。永久阻断只对最终候选机制判定；不是“某片首次没有出现在初始化”时间。完整支撑覆盖为每模集合完全相等；另有 contains 指标表示真支撑只是更大模型子集，不混作 exact。历史未保存全部 Group-BESS 中间支撑或 fixed BCD 全访问支撑，故只声称最终候选池覆盖，不能排除真位置曾在未保留的中间状态被短暂访问。

## 复核入口

在仓库根目录使用 `.venv/Scripts/python.exe`；PYTHONPATH=src;.，OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1，Python 3.12.8 / NumPy 1.26.4。以下独立只读片段可复核覆盖总数、NPZ身份和对应审计行，不重新拟合，也不写历史输出：

```python
import csv, hashlib, json
from pathlib import Path
import numpy as np
root = Path('numerical_simulation/outputs_msic_ablation_v51_final/jobs')
audit = Path('numerical_simulation/outputs_candidate_search_study/audit')
rows = list(csv.DictReader((audit/'candidate_coverage_audit.csv').open(encoding='utf-8-sig')))
shape = exact = q60 = 0
for row in rows:
    path = root/(row['job']+'.json')
    job = json.loads(path.read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['job_json_sha256']
    assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() == row['array_cache_sha256']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
        B = a['B_true']
        truth = [np.flatnonzero(np.any(B != 0, axis=tuple(k for k in range(3) if k != m))).tolist() for m in range(3)]
    pool = job['candidates']
    flags = (any(c['sizes']==job['config']['sparsity'] for c in pool),
             any(c['supports']==truth for c in pool), any(c['q']==60 for c in pool))
    assert flags == tuple(bool(int(row[k])) for k in ('true_size_in_pool','exact_true_support_in_pool','q_true_in_pool'))
    shape += flags[0]; exact += flags[1]; q60 += flags[2]
assert (len(rows), shape, exact, q60) == (27, 9, 3, 18)
print('Historical coverage verified: 27 datasets, shapes=9, exact supports=3, q60=18')
```

完整训练/测试重建调用 `generate_simulation_dataset(TensorSimulationConfig(**job['config']),seed=job['seed'])` 与 `iter_test_batches(dataset,seed=job['test_seed'])`；指纹定义直接使用 `numerical_simulation.version_validation.worker.array_fingerprint`，测试 digest 逐批按 X,y,signal 顺序更新 array_fingerprint ASCII。fixed 网格从 `_candidate_size_triples(*job['trace']['s_max'],floor(.7*n))` 复算。审计CSV保存可核验历史输入指纹与所有输出统计；本次没有另添运行脚本或更改旧复现入口。

## 尚未执行

本Phase A不能证明任何新caps/网格提升PMSE、F1或运行效率；未评估新候选池过拟合、数值稳定性、多初始化收益或三个实证数据集。历史只有3重复/场景，未足以决定正式v5.2/v6或新的MSIC参数。下一阶段需先配对验证候选搜索覆盖与成本，再独立讨论支撑位置与MSIC。
