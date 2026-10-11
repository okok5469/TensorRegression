# 2026-10-04 工作进展：完整已知 s 实验、后续诊断与基线文献

## 一、范围与已有证据

本次请求包含两部分：推理 agents 交叉分析后续实验和同方向基线；准备可在集群执行的完整已知模态稀疏度数量实验。以下科学设计与文献建议以现有代码、上一轮保存结果和核实的原始论文为依据；具体运行入口、上传文件与实际测试结果由后面的实现验收章节记录，不在这里预先声称完成。

上一轮正式 baseline 为 `n_train=200`、`n_test=500`、维度 `32×20×25`、真实模态数量 `s=(4,3,5)`、`q=60`、各模态 `rho=0.3`、`sigma=1`、`SNR=2`，原系数为随机符号乘 `U(0.8,1.2)` 后整体归一化。原系数组 10 次重复中，未知 s 的 PMSE 为 `1.8971`、完全恢复 `4/10`，已知 s 的 PMSE 为 `2.8105`、完全恢复 `6/10`，Oracle OLS 的 PMSE 为 `1.4519`。固定数量四次失败的训练 RSS 均高于同样数量真支撑上的 Oracle RSS，已构成固定规模搜索未达到全局训练最优的证据，尚未定位到唯一原因。完整证据见[上一轮工作记录](2026-10-04_advisor_experiments_progress.md)和[补充实验报告](../numerical_simulation/advisor_experiments_20261004/outputs/RESULTS.txt)。

因此不能把当前结论简化为“常量 B 无效”或“知道 s 没用”：等幅保留随机符号没有稳定全面改善；全正等幅在已知 s 时达到 `10/10` 完全恢复，却同时改变系数方向和秩；字面值 +1/+0.5 未缩放时又增加了 SNR。后续应诊断困难来源，并保留原始生成机制作为主实验。

## 二、完整已知 s 实验应回答什么

### 1. 信息层级必须清楚

| 方法 | 可获得的真实信息 | 仍需解决的问题 |
|---|---|---|
| adaptive：未知 s | 无真实数量、无真实位置 | 选每模态数量、选具体位置、估计系数 |
| fixed：已知 s | 仅真实数量 `(s1,s2,s3)` | 找具体位置、估计系数 |
| Oracle OLS | 真实位置集合 `(S1,S2,S3)` | 在真支撑上估计系数 |

知道 `q=s1s2s3` 不等于知道三个模态数量；知道三个数量也不等于知道真实位置。Oracle 不返回真实 B，而是对含噪训练数据做 OLS。它是消除支撑搜索后的参考，并非任意估计器在有限测试集上的不可超越误差下界。

### 2. 本轮主实验范围与解释边界

本轮目标是覆盖原正式实验的全部 14 个场景，每场景默认 50 次配对重复，同一数据分别运行 `fixed + adaptive + oracle`，形成 `14×50=700` 个数据任务、`700×3=2100` 个方法结果；实际任务清单、seed、配置和执行方式以实现验收记录为准。新增基线和下面的诊断方案不自动纳入本轮，避免扩大改动范围。

三个方法必须共享训练 X、y、真实系数、噪声和独立测试随机流；仅 fixed 接收真实模态数量，仅 Oracle 接收真实位置。报告每场景的均值、样本标准差、配对差和区间，以及完全恢复成功数。不同难度场景应分别汇报，不能只合并总均值。

**现有 fixed 与 adaptive 的初始化及搜索流程不同，主实验比较的是两条现成完整流程，不能作为“是否知道 s”的纯因果消融。** fixed 使用模态边际 top support，adaptive 使用 GLasso 起点和 MSIC 候选搜索；维持这些路径可追溯到旧实验，但差异不能全部归因于数量信息。完整实验首先回答：这两条现有流程的优劣是否随 n、SNR、维度、相关度变化，以及给定正确数量后是否仍有搜索缺口。

### 3. 比完全恢复率更有用的诊断

- 对 fixed 的成功和失败重复分别统计 PMSE、Mode/Element F1、各模态漏选/错选数、训练 RSS、初始支撑质量和终止原因；失败数很少的分层结果明确标注样本数。
- 定义 `RSS_gap = RSS_fixed − RSS_oracle`，保存逐重复数值与适当的尺度化结果；真支撑是固定数量的可行候选，所以明显正 gap 表示返回模型未达到固定数量候选中的全局最小训练损失，但不能仅凭 gap 判断是初始化、交换限制还是终止规则导致。
- gap 为负也不证明位置恢复正确：含噪训练集上错误支撑可能更拟合噪声。Oracle 不是“所有支撑中的训练最优”。
- `n=100、q=60` 时，真支撑 OLS 仍有明显估计方差；条件差的设计还可能使个别重复不稳定。必须同时看 Oracle PMSE/系数误差和设计条件指标，不能把离噪声底线较远的 Oracle PMSE 全部算作算法搜索失败，也不能要求已知 s 的所有预测指标一定优于允许收缩或少选的流程。

## 三、后续最小实验设计与优先级

以下均为推理 agents 与当前分析形成的建议，**尚未执行，也未接入本轮集群主实验**。

| 优先级 | 尝试 | 最小设计 | 诊断价值与限制 |
|---|---|---|---|
| 1 | 共同候选搜索结果下的选阶诊断 | 预先用 n、维度和可行性规则确定候选数量网格；对每个候选规模只做一份共同 fixed 搜索，初始化、预算、终止准则全部一致；已知 s 从同一份结果中取真实规模，未知 s 仅用训练 MSIC 选规模 | 更直接分离“数量选择”和“该规模下位置搜索”；未知 s 的候选范围、初始化、重启和预筛选均不能因真实 s 调整，否则数量信息经初始化泄漏。它是新的受控诊断，不能覆盖旧 adaptive 或宣称等同原流程。 |
| 2 | 弱模态切片 | 共用原随机符号与潜在幅度；在第一模态的一个真实活跃切片上乘 `a=1、0.5、0.2`，其余切片不变，再整体归一到相同 SNR；共用 X、支撑、噪声，`a>0` 保持真实 s 不变 | 比只改变元素 beta_min 更贴合模态 group 选择：检验一个整体较弱的真实切片是否被遗漏。记录各活跃切片范数、弱切片召回率、归一后的相对强度和最终恢复；固定总体 SNR 后其他切片也会被重新缩放，不能声称其绝对幅度不变。 |
| 3 | 幅度异质性 | 保留符号，共用均匀随机数，比较 `U(1−δ,1+δ)` 的 `δ=0、0.2、0.5`，再归一到相同 SNR；已有 0 和 0.2 可追溯 | 检验更强幅度异质性，而不只比较已很窄的 ±20% 幅度；模态选择应同时报告切片能量，不只元素最小值。 |
| 4 | 符号与秩 | 同支撑比较全正等幅 rank-1、可分离混合符号 rank-1、独立随机符号等幅；先在 rho=0、同 SNR 比较，再看 rho=0.3 | 独立设计下等幅组整体范数可一致，便于研究符号/结构；相关设计下需再次检查实际幅度及协方差方向，不能认为已完全消除符号和秩的混杂。 |
| 5 | s 的轻微错设 | 同一种仅由训练数据产生的起点和搜索预算，比较 `(3,3,5)`、`(4,3,5)`、`(5,3,5)` | 检验数量误差的敏感性；对应 q=45、60、75，复杂度和 OLS 方差也随之变化，不能只解释为先验信息对错。 |

如第 1 项仍显示搜索失败，可再用小维度、可穷举候选的设置比较“当前搜索、穷举训练最优、Oracle 真支撑”，分别定位优化和统计困难；这不替代正式场景上的结果。增加重启或交换规模之前，应先保存失败案例的损失轨迹和终止原因，避免只延长预算却无法说明为何改善。

对弱切片等生成机制实验，真实位置用于构造 DGP 是正常的；拟合接口仍不得获得该位置，初始化和任何方法参数都不得据此选择。所有方案应只改一个科学因素，并把实际 SNR、系数范数与切片强度一起保存。

## 四、文献检索范围与优先推荐

采用 `nature-academic-search` 的 `multi-source-search` 工作流，已读取其 manifest、工具与来源路由、匹配工作流。当前环境没有 academic-search MCP；实际通过 web 核验期刊官网、arXiv、作者论文与作者代码仓库。检索日期为 **2026-10-04**；以下是针对本项目的代表性推荐，**不是穷尽综述，不提供引文数，不声称任何代码已下载或运行**。以下实现优先级是结合模型假设与当前数据生成机制作出的推断。

**第一优先级：Vector-ABESS 和 Vector Elastic Net。** 二者不要求系数块 rank-1，分别补足“同样使用 best-subset，但没有矩形张量约束”的对照，以及“相关设计下的 L1/L2 收缩”对照。它们可以直接帮助说明 Tensor-ABESS 的收益是否来自矩形结构，而非只来自选择算法类别。

**第二优先级：Sparse ISLET，经过审核的正则化 CP/Tucker。** Sparse ISLET 的 Tucker 因子行稀疏与本项目 mode support 较接近；CP/Tucker 适合检验其他张量结构。秩、惩罚与初始化必须训练内选择，真实秩输入要单独标注为额外已知信息。当前随机符号矩形块通常是 Tucker rank `(4,3,5)`，全正等幅块才是 rank-1，不能只用后者评价低秩方法。

| 文献：完整作者、年份与出版信息 | DOI / arXiv 与直接主来源 | 已核实代码或软件 | 匹配程度与建议 |
|---|---|---|---|
| **A polynomial algorithm for best-subset selection problem**；Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, Xueqin Wang；**2020**；*PNAS* **117(52):33117–33123** | DOI **10.1073/pnas.2014241117**；[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7777147/) | [官方 abess](https://github.com/abess-team/abess)；软件参考：Jin Zhu, Xueqin Wang, Liyuan Hu, Junhao Huang, Kangkang Jiang, Yanhang Zhang, Shiyun Lin, Junxian Zhu；**2022**；**abess: A Fast Best-Subset Selection Library in Python and R**；*JMLR* **23(202):1–7**；[官方论文](https://jmlr.org/papers/v23/21-1060.html)，arXiv **2110.09697** | **高，Vector-ABESS 优先**。X 展平，known-q 给 `q=∏s`，unknown-q 用训练数据选；与 Tensor known-s 所得结构信息不同，需要明确。官方包与仓库自有向量实现须区分实现版本与预算。 |
| **Regularization and Variable Selection Via the Elastic Net**；Hui Zou, Trevor Hastie；**2005**；*JRSS B* **67(2):301–320** | DOI **10.1111/j.1467-9868.2005.00503.x**；[期刊](https://doi.org/10.1111/j.1467-9868.2005.00503.x)，[作者 PDF](https://web.stanford.edu/~hastie/Papers/elasticnet.pdf) | 可用 [scikit-learn 官方 ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)，此项是库实现，不称为论文作者代码 | **高，预测和相关设计对照优先**。训练内调正则强度与 L1/L2 比例，使用同 CV 分割和训练内标准化；成组保留相关变量也可能增加假阳性，支撑恢复和预测须分别报告。 |
| **ISLET: Fast and Optimal Low-Rank Tensor Regression via Importance Sketching**；Anru R. Zhang, Yuetian Luo, Garvesh Raskutti, Ming Yuan；**2020**；*SIAM Journal on Mathematics of Data Science* | DOI **10.1137/19M126476X**；arXiv **1911.03804**；[期刊](https://epubs.siam.org/doi/10.1137/19M126476X)，[开放全文](https://arxiv.org/html/1911.03804v2) | [作者 R 包](https://github.com/yuetianluo/ISLET)；仓库目录确有 `R/Sparse_ISLET_main_fun.R`，未运行核验 | **中高，Sparse ISLET**。行稀疏 Tucker 因子允许多模态稀疏，较接近矩形支撑；需另调 rank 和 sparsity。主要最优性结果基于独立 Gaussian ensemble，不能当作 rho>0 设计的自动保证。 |
| **Tensor Regression with Applications in Neuroimaging Data Analysis**；Hua Zhou, Lexin Li, Hongtu Zhu；**2013**；*JASA* **108(502):540–552** | DOI **10.1080/01621459.2013.776499**；arXiv **1203.3209**；[作者正式 PDF](https://hua-zhou.github.io/media/pdf/ZhouLiZhu13CPReg.pdf)，[arXiv](https://arxiv.org/abs/1203.3209) | [作者 TensorReg](https://github.com/Hua-Zhou/TensorReg)，含 `kruskal_reg.m`、`kruskal_sparsereg.m`；[官方依赖说明](https://hua-zhou.github.io/TensorReg/)列 MATLAB、Tensor Toolbox、SparseReg | **中，正则化 CP 回归**。标量响应、张量自变量符合项目；rank、惩罚、多起点须训练内选择。禁止直接把未经审核的仓库简化 CP 函数当论文复现。 |
| **Tucker Tensor Regression and Neuroimaging Analysis**；**Xiaoshan Li, Da Xu, Hua Zhou, Lexin Li**；**2018**；*Statistics in Biosciences* **10:520–545** | DOI **10.1007/s12561-018-9215-6**；[期刊](https://link.springer.com/article/10.1007/s12561-018-9215-6)，[作者正式 PDF，核实四作者](https://hua-zhou.github.io/media/pdf/LiXuZhouLi18TuckerReg.pdf) | [作者 TensorReg](https://github.com/Hua-Zhou/TensorReg)，含 `tucker_reg.m`、`tucker_sparsereg.m` | **中，正则化 Tucker 回归**。不同模态秩更灵活；明确惩罚作用于 core 或因子何处，不能把 core 稀疏视为同一 mode support 硬约束。 |
| **A Splicing Approach to Best Subset of Groups Selection**；Yanhang Zhang, Junxian Zhu, Jin Zhu, Xueqin Wang；**2023**；*INFORMS Journal on Computing* **35(1):104–119**；在线发表 **2022-12-05** | DOI **10.1287/ijoc.2022.1241**；arXiv **2104.12576**；[期刊](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1241)，[arXiv](https://arxiv.org/abs/2104.12576) | [作者实验代码](https://github.com/abess-team/a-splicing-approach-to-best-subset-of-groups-selection)，[官方分组接口说明](https://abess-team.github.io/abess/articles/v07-advancedFeatures.html) | **算法背景高，直接整体基线谨慎**。假设预定不交叠分组，而三模态切片跨模式交叠；无法用单个 group 标签向量实现同一交集约束。整张量单模态切片每组维度可超过 n，组内无惩罚 OLS 的可识别性必须检查。 |
| **Low rank tensor recovery via iterative hard thresholding**；Holger Rauhut, Reinhold Schneider, Željka Stojanac；**2017**；*Linear Algebra and its Applications* **523:220–262** | DOI **10.1016/j.laa.2017.02.028**；arXiv **1602.05217**；[期刊](https://www.sciencedirect.com/science/article/pii/S0024379517301295)，[arXiv](https://arxiv.org/abs/1602.05217)，[作者 PDF](https://www.mathc.rwth-aachen.de/~rauhut/files/TensorIHT.pdf) | 未核实该论文作者的官方代码，不声称不存在 | **补充，TIHT/NTIHT**。硬阈值截断的是张量秩，不是已知 mode sparsity s；作为低秩预测对照有意义，加入额外 mode truncation 时必须标为自有适配。 |
| **Non-conservative screening rule for sparse tensor regression**；Ruijie Liu, Yitian Xu；**2025**；*Pattern Recognition* **162:111405** | DOI **10.1016/j.patcog.2025.111405**；[期刊与公开摘要](https://www.sciencedirect.com/science/article/abs/pii/S0031320325000652) | 未核实官方代码，未获取付费全文 | **近期拓展阅读，暂不作为主统计基线**。TSWSSO 主攻 working set、二阶信息和 KKT 检查的加速；公开模型含 rank≤1 限制，不能把其加速结果当成一般随机符号矩形块的精度保证。 |

## 五、基线公平比较的关键约束

1. 新基线均使用同训练数据、独立测试数据和种子；调参只用训练样本，训练内标准化，保留同一 CV 分割。测试误差不能用于选 rank、惩罚、阈值或起点。
2. Vector-ABESS 的 known-q 是额外数量信息，但少于 Tensor-ABESS 的 known-s 加矩形结构；unknown-q 与 unknown-s 配对，不能将两类比较混在一个“已知数量”标签里。
3. 普通低秩方法会生成稠密系数，支撑 F1 依赖阈值。不得用真实 s 截取结果后当作未知 s 的性能；若做 known-s 后处理，必须单独命名、独立说明，不能改原算法输出冒充同一估计器。
4. rank、sparsity、强弱切片是不同结构。应在原随机系数 DGP 和另设的 sparse+low-rank DGP 上分别评估低秩方法；后者只是新场景，不能为了提高所有数字而替换原主实验。
5. OLS Oracle 是支撑参考；选定支撑后的收缩估计可能在测试风险上优于真支撑 OLS。若用 ridge/refit，保留估计方式差异，不能预设 Oracle 必须数值最好。
6. **实际元素支撑与模态投影必须分开。** 现有 `compute_support_metrics` 用模态支撑的笛卡尔积计算 element F1 和 q_hat；Vector Lasso 的 `S1/S2/S3` 是实际非零系数向各模态的投影，因此诱导矩形可能包含大量零系数。原指标可追溯复现，但不能将它当成向量方法实际非零元素的 F1/数量；加入 Vector-ABESS/Elastic Net 等非矩形基线时应新增基于实际元素集合的指标，并单独保留模态投影指标。本轮只读发现并记录，不改旧 metrics 或历史结果。

## 六、现有 CP/Tucker 函数的只读审阅：暂不复用为论文基线

这部分在读取代码时发现，已即时向主 agent 报告；**停止把这两个现有函数当上述论文的规范复现，本次没有改动旧代码，也没有把它们接入完整已知 s 实验。**

- [现有 CP 函数](../src/tensor_abess/baselines/cp.py)在 `R>1` 时更新每个分量都直接对同一个 y 做回归，未减去其他分量对预测的贡献，也未联合求解该模式内所有分量。它不能直接作为 Zhou et al. (2013) 多分量 ALS 的复现；rank-1 特例不消除多分量实现的问题。
- [现有 Tucker 函数](../src/tensor_abess/baselines/tucker.py)只构造 `T=mean(yX)`，做截断 HOSVD 后重构；没有按 Li et al. (2018) 的回归目标交替拟合 core 和因子。向量模型下 `E[x y]=Σβ`，相关设计中矩估计不是 β 本身，因此将其名称写作 Tucker regression 不足以证明实现了该论文算法。

新增论文级对比时，先另立审核任务或使用已核实的作者 TensorReg/Sparse ISLET 实现，做模型、数值和收敛核验后再纳入；本次仅记录缺口，未顺手修复，不影响已核验的 Tensor-ABESS/Oracle 补充实验。

## 七、本记录的完成边界

科学建议和文献推荐已经形成；本记录没有宣称新增基线已实现，没有宣称后续弱切片、选阶控制或秩对照已经运行，也没有把历史 14 场景读表说成此次重跑。完整集群代码、执行指令、上传清单、实际验收和仍需用户决定的集群信息，由下方实现验收章节补充。

## 八、集群实现与实际验收

### 1. 已交付的独立入口

新增 `numerical_simulation/known_s_cluster/`，没有改核心算法、旧 main/runner、实证分析或 Presentation。入口提供 `plan/run/status/merge`，默认 **full、50 次、fixed/adaptive/oracle**，即 **14 场景、700 个独立任务、2100 个方法结果**。每任务内部配对三方法；known-s 的拟合入口只接收 X、y、参数和数量，不接收真实位置，真实位置仅传给 Oracle。

14 场景是原单因素网格：baseline；SNR=0.5/1/2/4；n=100/200/400；维度16×12×15、32×20×25、64×30×40；rho=0/0.3/0.6。其余保持原 DGP 和参数：随机符号×U(0.8,1.2)、连续矩形支撑 `(4,3,5)`、按指定SNR缩放、sigma=1、独立测试500个，k_max=1、max_cycles=6、lambda网格20、CV5折、max_iter=500。默认未加入常量B、弱切片或新论文基线。

任务保存 JSON/NPZ：真实B与训练噪声、每方法估计B/支撑、输入指纹、代码/配置/环境指纹、Tensor入口初始化、公开损失轨迹。**adaptive 的入口初值与最终选中候选的轨迹可能来自不同 refinement 阶段**，JSON 已明确 scope；没有把二者当同一次路径的起终点，也未编造内部没有暴露的终止原因或条件数。

计划固定后不可换方法、代码或环境继续同一批；独占任务锁与同目录原子硬链接避免 Windows/Linux rename 覆盖差异。只有 `--resume` 且校验通过才跳过，残缺/篡改/计划外产物报错，不自动修复、不换算法补缺。status 的持锁提交窗口标为运行中，无锁残缺仍报错；merge 只接受全套完整结果，输出原始表、场景均值/样本标准差/分位数/恢复成功数、配对差、Oracle RSS gap与完整性报告。配对区间和失败分层分析在正式结果回来后进行，不预先宣称显著性。

### 2. 历史种子的实际差异

测试先发现旧 main.py 一次完整执行的递增组偏移，与实际 v3/v5 CSV 不一致，已暂停错误映射。独立读取两个旧版本全部14场景后确认：**各非baseline实验组都用了10000偏移**。新 profiles 保留完整参数网格，明确冻结历史实际映射：baseline `2026+replicate`，其余 `12026+100×组内序号+replicate`；前10次分别与两个历史版本各140条记录的seed/配置一致。未改旧入口或历史文件。

因此同一配置可能在多个场景标签下重复，同一seed也可能用于不同场景；700任务不是可随意合并计算单一置信区间的700个独立同分布样本。结果按场景汇报；同一任务三方法严格配对，不将“跨场景同seed”自动解释为X/噪声完全配对。

### 3. 已完成与未完成的验证

- TDD：入口缺失先RED；实现后历史映射测试暴露上述seed差异；修正新配置并将测试改为自包含fixture。另对状态提交窗口和计划外NPZ/lock观察两项RED后修正；最终 **10项通过，7.772秒**。包括真实quick全链、fixed-only缺比较标为不适用、独占写竞争、续跑不重拟合、NPZ/指标/源码/环境冲突拒绝。
- 独立 CLI：quick14场景×1次×3方法 **42条结果**全部完成/汇总，status缺项0，resume确实跳过；可选GLasso/Vector Lasso在quick task0与三方法共5项真实拟合通过。
- full历史复现：本地只运行 **baseline task0、seed2026** 的三方法，各 **28项非耗时指标**与对应v3/v5一致，最大绝对差 **2.8422×10⁻¹⁴**；三者PMSE均约1.3250717、完全恢复均1。这个单重复是入口复现验收，不能替代完整实验或证明known-s总体更优。
- 最小包独立验证：解压后确认runner与核心包从包内导入，不依赖旧输出、main/runner/plots；真实quick CLI再次完成14任务/42结果，包内10项测试获准后通过，**19.181秒**。初次沙箱写results被拒、沙箱内测试挂起，均在权限诊断后处理；已结束该挂起测试进程，未将它计为通过。
- Slurm脚本通过本地Git Bash `bash -n`；**尚未在Linux集群执行或提交作业**。Python3.12、NumPy1.26.4、SciPy1.16.1；55个受保护原文件SHA256保持一致。证据在目标目录的 `LOCAL_VALIDATION.json`、各新验证输出和上传哈希清单。

**正式700个任务尚未完成：本地full验收只执行task0，其余699个留给用户集群。** quick结果不作论文正式结果；最大维度的集群耗时/峰值内存也尚未实测。

## 九、原整包交付记录（2026-10-05起由第十一节增量部署替代）

以下保留2026-10-04交付历史；用户已明确集群有旧项目，当前应按第十一节和最新README增量上传，不再制作或使用旧压缩包。

上传 `numerical_simulation/known_s_cluster/known_s_cluster_upload.zip`，解压到各计算节点能访问的共享路径，保留相对结构。包含 **42个文件+1个UPLOAD_MANIFEST.json，共43个文件**；源码闭包是：

```text
src/tensor_abess/                          全部 .py（含子目录）
numerical_simulation/__init__.py
numerical_simulation/data_generation.py
numerical_simulation/methods.py
numerical_simulation/metrics.py
numerical_simulation/known_s_cluster/run.py
numerical_simulation/known_s_cluster/profiles.json
numerical_simulation/known_s_cluster/requirements.txt
numerical_simulation/known_s_cluster/submit_slurm.sh  用Slurm时需要
numerical_simulation/known_s_cluster/test_run.py      可选验收
numerical_simulation/known_s_cluster/README.md        详细说明
UPLOAD_MANIFEST.json                                文件SHA256清单
```

无需上传 `.venv/.git`、LEMON、旧输出或Presentation；无需安装sklearn/matplotlib/第三方abess。共享结果路径须支持同目录 `os.link`，不支持时明确失败，不能静默换覆盖写。下面假定Slurm、50次；普通Linux、10次/只fixed/全部现有基线及缺任务重跑见README。未得到用户调度器偏好回复，因此Slurm只是可调整示例，未替用户选择分区或账号。

从**解压根目录**执行环境设置，先按README完成quick检查和计算节点task0/task500试测；后者是50次计划最大维度的首重复，用于资源定标：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r numerical_simulation/known_s_cluster/requirements.txt
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m numerical_simulation.known_s_cluster.run plan \
  --profile full --n-reps 50 --methods fixed adaptive oracle \
  --output-dir results/known_s_full_50
mkdir -p results/known_s_full_50/logs
export KNOWN_S_PYTHON="$PWD/.venv/bin/python"
sbatch --array=0-699%32 --cpus-per-task=1 --mem=8G --time=12:00:00 \
  --output=results/known_s_full_50/logs/%A_%a.out \
  --error=results/known_s_full_50/logs/%A_%a.err \
  numerical_simulation/known_s_cluster/submit_slurm.sh results/known_s_full_50
# 全部数组作业结束后：
python -m numerical_simulation.known_s_cluster.run status --output-dir results/known_s_full_50
python -m numerical_simulation.known_s_cluster.run merge --output-dir results/known_s_full_50
```

`1CPU/8GiB/12小时/%32` 是未经过用户集群试测的初始参数，按实际partition/account、队列限额及task500测量调整；不需要GPU。普通Linux计算节点可用 `run --output-dir results/known_s_full_50 --all --resume`。计划应先创建，再创建logs；不得覆盖历史目录或在同一计划中临时更改配置。保留全部任务JSON/NPZ及5个汇总文件，正式结果回来后再做配对区间、失败机制与汇报图表。

## 十、种子复核与执行阶段补充（2026-10-04）

用户追问后再次只读加载完整计划：700个task_id唯一，当前源码/配置哈希与计划一致，前10次seed/test_seed分别与v3、v5各140条历史记录全部匹配。本次新入口的种子错配已解决；旧main.py保持原状，正式实验使用 `numerical_simulation.known_s_cluster.run`。各非baseline组重复使用历史seed是保留原映射，不代表700任务可以合并为独立同分布样本。

Slurm执行明确分三阶段，不能把前述全部命令无等待地一次执行：

1. **准备一次**：环境安装/激活、线程设置、quick验收与资源试测，随后创建正式plan；同一输出目录不要再次plan。
2. **提交一次**：运行 `sbatch --array=0-699%32 ... submit_slurm.sh results/known_s_full_50`，Slurm按计划调度700个任务；已有submit_slurm.sh是每个数组任务的执行脚本，应由sbatch调用，而不是直接bash运行。准备和提交可整理到一个自用sh，但需先通过验收并确认资源。
3. **完成后另执行**：全部数组作业结束后，先status确认完整，再merge。普通sbatch提交后立即返回，不等700任务完成，不能紧接着把merge当作下一条已具备条件的命令；尚有缺失任务则先按真实编号补跑，再汇总。

若在已获分配的普通Linux计算节点上用 `run --all --resume` 顺序计算，该命令会等待整套运行结束，可在带失败停止的sh里按“plan → run --all → status → merge”顺序执行；这与Slurm异步提交方式不同。上述说明仅补充文档，未更改生产代码或上传包。

## 十一、按已有集群项目增量部署（2026-10-05）

用户明确集群已上传过旧项目，只需增量文件及完整sh内容；已据此重写 `numerical_simulation/known_s_cluster/README.md`，不再打包、不重新上传整个项目，也未删除任何文件。

- **生产必传**：`known_s_cluster/run.py`、`profiles.json`、`requirements.txt`；Slurm再传 **2026-10-05修正版 `submit_slurm.sh`**。建议随传README，test_run.py用于可选验收。
- **旧依赖同版就不重传**：`src/tensor_abess/`、`numerical_simulation/__init__.py`、`data_generation.py`、`methods.py`、`metrics.py` 本轮未改。集群版本更早时应先与已有哈希清单比对，仅同步确认不同的依赖；新plan冻结集群现状，不自动证明集群与本地验收版相同。旧清单中的README、测试、sh哈希属于2026-10-04版本，不应用于当前这些文件的验收。
- **完整命令已放README**：三个独立完整代码块 `01_prepare.sh`（检查及建计划）、`02_submit.sh`（一次提交700数组任务）、`03_merge.sh`（全部结束后检查/汇总）；用户只需填项目路径、解释器路径和集群要求的partition/account。没有额外创建三份脚本文件，用户按内容保存即可。普通Linux同步 `run --all --resume` 的替代方式也保留。
- **修正Slurm路径定位**：原脚本按BASH_SOURCE定位项目，会因Slurm复制脚本到调度目录而失效；回归测试模拟脚本被搬到spool/job目录，先观察到ModuleNotFoundError，再改为Slurm提供的 `SLURM_SUBMIT_DIR`，要求提交脚本先cd项目根。命令显式带 `--chdir` 与 `--export=ALL`；未改计算核心、run.py、profiles或随机数规则。参考[Slurm官方sbatch说明](https://slurm.schedmd.com/sbatch.html)。
- **验证**：新回归测试连同原10项共 **11项通过（20.313秒）**，覆盖带空格路径、task-id/输出目录参数传递和单线程环境；实际Slurm未提交。Windows测试先误选PATH中的不可执行Bash入口，已将Windows测试入口明确为本机Git Bash后才做RED/GREEN，未修改系统环境。
- README内三段完整sh均通过 `bash -n`；与原上传清单逐文件比较，差异仅README、submit_slurm.sh、test_run.py，旧核心及run/profiles均一致。
- **清理待确认**：旧ZIP及可再生缓存可删除，但尚未获具体删除确认；暂保留全部。`LOCAL_VALIDATION.json`、三个outputs验证目录、`upload_verification/results/`及源文件哈希清单含真实证据，不作为废弃物删除；解压源码副本可另列清单后确认，不能直接删整个upload_verification目录。

### 用户确认后的清理与证据保留（2026-10-05）

用户已明确确认删除两个目标；校验解析后的绝对路径均为known_s_cluster直接子项且无重解析链接后，仅删除 `known_s_cluster_upload.zip` 和本目录直接子目录 `__pycache__/`，删除后确认不存在。未删除解压副本或任何结果目录。

`UPLOAD_MANIFEST.json`、`LOCAL_VALIDATION.json`、`outputs_cli_smoke/`、`outputs_full_preflight/`、`outputs_optional_preflight/`、`upload_verification/results/` 均保留并确认存在：哈希清单记录旧验收版本的文件指纹，验证JSON与结果保存本地测试、历史复现和独立部署验收证据。它们对后续复现、排错和汇报有用，但不是集群正式运行依赖，无需上传；仅在主动比对旧源码版本时可选上传哈希清单。集群正式实验会生成自己的plan和结果，不能用本地验证结果替代正式结果。旧JSON中的压缩包哈希仍作为历史交付记录保留。

## 十二、按用户实际集群环境简化执行（2026-10-06）

用户提供现有可运行环境与旧脚本格式：`/public/home/xjh/xmx/tensor-sparse-regression-main`、CMO_HPC、账号xjh、Anaconda环境myenv、Python3.9.25、NumPy2.0.2、SciPy1.13.1，并明确禁止更新集群。经用户Yes确认，采用项目根目录单文件 `run_known_s_full.sh`，只需依次执行 `bash run_known_s_full.sh prepare`、`sbatch run_known_s_full.sh`，全部数组任务结束后执行 `bash run_known_s_full.sh merge`。

- 脚本沿用 `#!/bin/bash -l`、`module load anaconda3`、`source activate myenv`、分区与账号；所有SBATCH指令位于首条命令之前。脚本只精确校验现有三项版本，不调用pip/conda安装或升级。
- 正式计划为14场景×50重复=700个数组任务，`--array=0-699%64`控制最多64个并发任务；每任务1个Python进程/1CPU，顺序运行fixed、adaptive、Oracle。旧脚本的`--ntasks-per-node=64`没有启动64个Python进程，不能用于这次分片。
- 生产增量必传缩减为根目录 `run_known_s_full.sh`、known_s_cluster/run.py、profiles.json；README与requirements建议上传，test_run.py可选，旧submit_slurm.sh不用。结果固定写入`numerical_simulation/outputs_known_s_full_50`，日志写入`numerical_simulation/known_s_cluster_logs`。
- Python3.9语法静态检查覆盖入口、四个模拟依赖和全部tensor_abess生产源码并通过。NumPy构建信息改用公开 `np.show_config(mode="dicts")`，不再读取内部`np.__config__.CONFIG`；environment函数记录实际版本并由plan在各任务间严格比对，精确目标版本由根脚本在prepare/run/merge三种模式前检查。
- 新增测试先观察到入口因本地版本硬限制报错、根脚本缺失；实现后完整14项测试通过（30.038秒），包括真实quick全流程、历史种子、续跑、原子写入、Python3.9语法与根脚本契约。两个sh均通过bash语法检查，根脚本为Linux LF。实际Python3.9/NumPy2集群运行尚未执行，不能把本地静态兼容检查称为集群实跑。

## 十三、完整已知 s 实验结果分析（2026-10-07）

### 2026-10-10 单调性自查补充（先于解释优化效果阅读）

用户提出“只接受改善的BCD/交换应使训练RSS非增”后，只读核对源码和700条已保存轨迹，发现固定s BCD不满足这一约束：157/700条fixed loss_history存在RSS增加（阈值1e-8），例如task_000007.json的RSS从264.728165664上升到269.017326343。内部group_splicing按new_loss<best_loss及tau阈值接受，局部规则正确；但tensor_bcd三个模态调用group_bess_fix没有传当前groups_init，导致子问题重新初始化，并直接按支持变化接受输出，没有与当前BCD解比较RSS。因此局部交换单调不代表外层BCD单调。

adaptive/MSIC会改变支持规模，不能要求其纯RSS始终非增；另有87/700条adaptive返回轨迹存在增加，但这些轨迹可能来自全局候选refinement内的fixed BCD，不能直接认定都是adaptive主迭代的行为。既有汇总是当前实现的真实测量，不足以声称固定s优化满足单调下降；后续修正须明确授权、保留旧结果并重新验证/运行。本次仅自查记录，未修改核心或实验结果。

集群结果已复制到 `numerical_simulation/outputs_known_s_full_50/`。`integrity_report.json` 与原始结果交叉核对通过：14个场景、每场景50次、700个任务、fixed/adaptive/oracle各700条，共2100条方法结果；任务-方法键无重复，核心指标无缺失或非有限值，`integrity_checks_passed=true`。运行环境记录为Python 3.9.25、NumPy 2.0.2、SciPy 1.13.1，四类数值线程均固定为1。

### 主要结论

- **已知 s 明显提高严格支持恢复，但不会普遍改善预测。** fixed知道三个模态的真实稀疏度 `(4,3,5)`，仍不知道真实位置；700次中完全恢复283次（40.43%），adaptive为138次（19.71%）。逐任务配对中fixed相对adaptive为145胜、555平、0负，即adaptive完全恢复时fixed也完全恢复。Oracle的700/700是因为直接给定真实位置，属于定义结果，不能作为算法支持恢复能力。
- **adaptive总体预测和宽松恢复更好。** 汇总均值中，模态macro-F1为0.7519对0.6449，PMSE为2.2368对3.0453；配对PMSE中fixed为183胜、142平、375负。原因是弱信号、小样本或高维时，只知道数量不能识别位置；fixed必须每个模态选满4/3/5个，错误位置会抬高估计方差和测试误差，而adaptive常通过少选变量换取更稳的预测。
- **样本量足够时，已知 s 的优势才完整显现。** `n=400`时fixed的macro-F1、完全恢复率均为1，REE=0.3612、PMSE=1.1856，与Oracle完全一致；adaptive的macro-F1=0.9538、完全恢复率仅0.02，REE=0.4502、PMSE=1.2976。50次配对中fixed在F1和PMSE上均49胜1平0负。这说明支持数量信息在可辨识条件下能消除模型规模选择误差。
- **困难场景下，已知 s 主要提高“选对全部变量”的机会，而不保证较小误差。** baseline中fixed/adaptive完全恢复率为0.58/0.30，但adaptive的PMSE为1.9907，优于fixed的2.5242；`p=64×30×40`时完全恢复率为0.12/0.10，但adaptive的macro-F1为0.6773、PMSE为2.7008，优于fixed的0.3513和4.3878；`SNR=0.5`时两者完全恢复率均为0，adaptive PMSE仍较低（1.8371对2.4698）。
- **小维度和大样本是fixed预测占优的两个清楚场景。** `p=16×12×15`时fixed完全恢复率0.82对adaptive 0.52，PMSE 1.7411对1.7801；`n=400`的优势更强。高相关 `rho=0.6` 时fixed的macro-F1与完全恢复率更高（0.8597/0.26 对 0.8242/0.10），但adaptive PMSE略好（1.8191对2.0085），说明相关性使支持解释与预测误差进一步分离。
- **搜索缺口揭示了失败机制。** 325/700次中，真实支持Oracle的训练RSS严格低于fixed所得解，说明存在训练损失更低的真实支持候选，而known-s搜索没有找到；`n=400`时50次缺口均为0，与fixed全部恢复一致。在`SNR=0.5`下仅11/50为正且平均差为负，表明弱信号时错误支持可以更好地拟合训练噪声，单靠固定s或最小训练RSS不足以识别真实支持。
- **计算代价降低但不是精度的充分条件。** fixed平均13.88秒，adaptive平均25.03秒，700/700次均更快，约节省44.5%时间；它省去了稀疏度搜索，但支持位置搜索仍可能失败。

### 解释边界与汇报建议

`sample_size=200`、`dimension=32×20×25`、`correlation=0.30`三组的50个重复在种子、数据和结果上逐一完全相同，是同一默认配置在三条参数曲线中的重复锚点；不能把它们当成150个独立重复。因而700条总体均值只作描述，正式汇报应按14个曲线点展示，并在总体比较时对重复锚点去重。fixed和adaptive还沿用不同初始化及模型搜索路径，当前配对统计只能说明完整流程差异，不能把全部差异严格归因于“是否知道s”。

最稳妥的汇报表述是：已知s解决的是模型规模不确定性，不解决支持位置搜索和弱信号辨识；它把完全恢复率从19.7%提高到40.4%，并在大样本场景达到Oracle，但多数困难场景中adaptive通过偏向更小模型获得更好的预测误差。后续应增加“相同初始化、相同位置搜索、只改变s是否固定”的受控消融，并报告去重场景汇总、配对区间或检验，以及搜索失败任务的初始化/迭代轨迹。
