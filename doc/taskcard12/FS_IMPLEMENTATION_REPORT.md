# Task12 双规则研究包装器

正式版本仍为 `v5.2-search-final`；新模块 `research/taskcard12/selection.py` 是隔离的 `v5.2-fs-research` 研究接口，不覆盖正式源码、版本 manifest 或任何旧实验。任务卡12明确重新授权比较既定 FS，任务卡11的未通过结论与原冻结记录继续有效。

## 接口与数据流

| 接口 | 契约 |
|---|---|
| `fit(X, y, selection_rule='original', **native_options)` | 默认原 MSIC；返回一个选中模型 dict |
| `fit_pair(X, y, **native_options)` | 一次真实 v5.2 Cold 搜索，返回 `pool`、`original`、`fs` |
| `select_from_pool(pool, selection_rule='original')` | 零搜索，在完整原顺序候选池上选择；校验当前源码/环境身份 |
| `save_pool(path, pool)` | 新建目录，拒绝覆盖；保存 metadata、完整 B 和 seal |
| `load_pool(path, X=X, y=y, **native_options)` | 零搜索，严格核对输入、配置、源码、环境、所有哈希，重新独立核验全池 OLS |
| `verify_pool_files(path)` | 只验证封存文件/候选/原生选择身份；不表示当前环境允许数值复用 |
| `configure_core()` | 配置并校验冻结核心32文件及实际导入；允许在同一正确核心下重复调用 |

`pool` 是 `metadata`、`coefficients`（完整四维 B 数组）、`digest` 组成的 dict。metadata 含训练 `n/dimensions`、X/y 指纹、规范化的官方求解参数、源码/环境身份、完整 native trace、独立 OLS 审计、搜索计数与计时。模型 dict 含 `B/support/sizes/q/rss/score/B_sha256/full_candidate_index` 和完整 `selection` 分解。原始 `msic` 字段保留原评分；模型实际选择评分在 `score`，FS 不冒充原 MSIC。调用层将 `original/fs` 映射为 A0/A1。

`native_options` 使用官方参数名。未传参数由真实官方函数签名补默认值，不偷偷使用模拟配置的 `c=3/criterion=bic/max_cycles=6` 默认。训练输入按官方入口转换为 float，y 展平；模型原生 B、支撑、RSS、评分以及原 band/eligible 全部必须精确一致。固定目标规模、非 MSIC adaptive、改变 alpha/gamma/容忍带的请求均拒绝。

只有训练 X/y 进入 `fit_pair`。测试集、真支撑、测试 PMSE 和实证 OOF 结果不属于选择接口。旧 `observe_fit` 被动读取真实返回帧，一次 native 调用捕获 adaptive terminal 与全部 fixed 候选；随后 Task12 独立审计在每个支撑上重新求最小范数 OLS，验证 B、RSS、梯度、秩与有限性，不创建新的支撑搜索。所有数据通过参数/返回值传递，无全局候选池。

初版直接复用 Task11 实验审计，会拒绝 RSS=0 或秩亏候选。真实 tiny 反例证明官方对零响应及重复特征均返回合法有限结果；该差异作为 P0 失败保留于 `verification/selection_native_domain_initial.json` 与 RED 日志。修订仅在新 Task12 adapter 内接受原公式已有的零 RSS 截断和原 `lstsq(rcond=None)` 最小范数解，不修改旧 Task11、官方搜索或保存 B/RSS。秩不足明确记录 `rank_deficient=true`、`condition_number=null`；审计 `passed` 表示系数确实等于独立最小范数解，不表示满秩。评分 q 仍为完整矩形元素数，绝不替换为数值秩。NaN/Inf 系数、RSS或失配仍拒绝。两类输入的原生 B/RSS/score、双规则缓存和预测已经实测一致。

## 唯一公式和完全相同的选择规则

\[
M_0=n\log\max(RSS/n,\mathrm{tiny})+1.5q\ell_n+
\ell_n\sum_{m=1}^{3}\log {p_m\choose s_m},\quad
\ell_n=\log\max(\log n,e),\quad q=s_1s_2s_3.
\]

\[
M_{FS}=M_0+\ell_nq^2/(2n).
\]

直接复用任务卡11冻结 `rescore.select_candidates`，没有新系数。两规则各自求首个分数 argmin，由各自 q_best 求 `T=sqrt(max(q_best,1)*max(log(max(n,2)),1))`，即 kappa=1。精确判断 `score<=min_score+T`，按 `(q,score,sum(sizes))` 取最小，完整并列保留候选原顺序；没有 global isclose 容差。

该修正是任务卡指定经验规则，不称为 AICc、不宣称新的有限样本定理。相同 q 的附加项相同，相同形状异支撑的原 RSS 排序不变；候选池没有真支撑时不会创造真支撑。argmin 的 q 可收缩，但由于重新计算容忍带，最终输出 q 不保证逐例单调。上述边界与任务卡11一致。

核心源码位置：`tensor/sic.py` 的 `compute_tensor_msic_from_rss` 定义全局分数，`tensor/bcd.py` 的 `_candidate_size_triples` 和 `_refine_msic_global_candidates` 保留完整整数规模、容忍带及排序，`tensor/ops.py` 的 `refit_tensor_coefficients` 保留 `np.linalg.lstsq(...,rcond=None)`。新 wrapper 的 `fit_pair` 调用真实原 solver，`audit_candidates` 独立核验最小范数 OLS，`select_from_pool` 重评分，`load_pool` 严格恢复缓存。

## 缓存与身份

每份缓存固定包含 `metadata.json`、`coefficients.npz`（唯一键 `full_B`）、`seal.json`。seal 覆盖前两文件的字节 SHA256；pool digest 绑定规范 JSON metadata 和完整系数数组指纹。逐候选 B 哈希、支撑包含、q、原 MSIC、完整整数网格/顺序、原生 argmin/band/eligible/最终选择另做语义校验。部分写入、错误输入、错误配置、损坏系数和环境漂移都拒绝，不自动重搜或补救。

源码身份是相对当前部署 root 的文件名与 SHA256，包移至集群后的相对布局有效，不把本机 Windows 绝对源码路径当跨机器身份。环境保存 Python/NumPy/SciPy、实际解释器、平台和四个线程变量；同环境缓存使用必须完全相同，Windows 和 Linux 不伪称数值环境等价。`verify_pool_files` 可作跨机器封存检查；`load_pool/select_from_pool` 仍要求当前环境匹配。

运行直接依赖闭包为：本 wrapper、`research/taskcard11/rescore.py`、`research/taskcard10/experiment.py`、`scripts/taskcard7/lemon.py`、冻结 `doc/algorithm_versions/v5.2-search-final/VERSION.json` 与其32个源码。历史回归工具另读 `task11/verify.py`、`task10/cv.py` 与原 DGP；这些不属于运行新数据选择的必要依赖。打包层负责其 DGP、metrics、调度和收集的额外闭包。

最终 wrapper SHA256：`761062933e2a52e407298d7806c5bd31c52aabbbf743bd0b10a745dbf9ede4b3`。初版 `e3845917534272078aa55402cadeb68d4e1c09ccdb766d987f1c9e88844f8752` 的102例回归保留。零残差/秩亏修订版 `6d96c0c29357c0e3336842517f3c9131c36858195142d86843469b058d88af2f` 在 `verification/selection_history_native_domain/` 完成102例全池 OLS/cache回归，550.457秒、零搜索。

随后真实 tiny 验证发现官方 n=1/2 也可合法运行，最终版只修正四处旧实验门禁：训练 n>0、两处完整网格预算采用官方 `max(1,floor(.7*n))`、删除额外的 `q<n-1` 限制；q/尺寸/支撑及 fixed 网格合法性继续严格核验。RED/GREEN、修订前源码、精确四行 diff 和前后 hash 保存在 `verification/selection_small_n_*`。最终12项 wrapper 测试均通过，覆盖正常、零响应、重复特征、n1/n2、单/双/重复/缓存。

最终源码对上述102份原封缓存另做完整归档验证和双规则纯重评分，16,406候选、204选中模型、1,122产物哈希全部通过，记录为 `verification/final_source_saved_pool_regression.json`。该最终步骤没有生成训练/测试流、重新拟合 OLS 或搜索，也没有把旧缓存 source 改为新 hash；当前 runtime 拒绝旧 source 缓存的行为也实际核验。四行 diff 在历史 n100/400 数据上只放宽原已满足的条件，未改 OLS/评分/native；全池 OLS 证据仍明确归属前一源码，不冒称最终源码重新执行102次 OLS。任何后续科学实现变化需重新门禁，不自动接受缓存。

## 历史资料解释

实际数学文档为 `Math/Tensor-ABESS算法实现说明_代码终版.md`，不存在任务卡字面列出的 `(2)` 文件。其旧启发式规模网格段保留历史内容；当前正式源码和任务卡11设计明确使用完整整数网格，本实现按冻结源码执行，不修改旧数学文档。

Task10 根报告保留30/48超时事实；后续完整开发48结果在 `doc/taskcard10/continuation_01/`，n400 A0/A2 Exact为13/24与11/24，确认未启动。Task11完整96显示整体预测改善，但强信号 q 欠选门槛失败；历史 n200/SNR4/435/rho0.6 的 Exact 3/6→1/6 风险继续保留。Task12工程通过只证明双规则实现与既有证据一致，不推翻旧科学结论，不自动发布 FS。

实际正确性数量、历史回归范围和限制见 [DUAL_SELECTION_VALIDATION.md](DUAL_SELECTION_VALIDATION.md)。最终新模拟科学结论等待真实集群结果，当前 `PENDING_CLUSTER_RESULTS`。
