# LEMON 实施依据

## 2026-10-09 EEG增量预测审计

任务卡要求独立模块及新输出。原evaluation只在outer train拟合preprocessor后，将整个outer标准化残差传给基线内部CV，因此内层训练/验证的预处理并未重新拟合；新实验外部3折选择lambda并各自拟合全部nuisance/scale。ABESS保持BIC初始化与MSIC固定信息准则，无新外部ABESS参数调优。现有.venv包含NumPy/SciPy/Matplotlib/YAML，不含sklearn；采用NumPy预先固定年龄分层均衡CV，不安装依赖。7项主线验收测试红后绿，覆盖covariate OLS、完整预测恢复、inner training centering和OOF唯一覆盖。

主线独立验收实际NPZ字段X/y/subject_ids/C，尺寸(145,61,16,5)与(145,8)，非有限值0；included manifest与ID顺序一致；X与146人原张量按manifest.tensor_index筛选后逐元素完全一致；y与原TMT.csv按ID的log(TMT_5)一致；C按原participants.tsv及旧encoding重建后逐元素完全一致。全部15外折与45内折协变量满秩，outer train/test=116/29，inner train=77/77/78。

模型适配7项测试通过，与主线7项合并共14项GREEN。Ridge对偶和原始空间解一致；Vector Lasso在正交设计上的解析软阈值解一致；零startISTA保持原数学定义，路径缓存仅共享同训练数据的设计及精确谱范数。GLasso仍执行分组筛选及矩形OLS，lambda比例由每个训练局部设计计算。

- 核心 API：tensor_abess(X,y,target_sparsity=None,c=3.0,criterion='bic',k_max=1,n_lambdas=20,cv_folds=5,max_iter=500,tol=1e-6,max_cycles=6,m_max=15,eps=1e-8,adaptive_criterion='msic',msic_refine_alpha=1.5,msic_refine_gamma=1.0,msic_refine_one_se_scale=1.0)。
- 终版基线位于 numerical_simulation.methods：tensor_glasso、run_vector_lasso_method；禁止复用包含 Oracle 的 run_all_methods。
- 当前数据目录 实证分析/nm000179 只有 sub-010002 到 sub-010006；完整下载路径已向用户询问。
- EEG JSON 提供 S210=EC、S200=EO；TSV 的 value 分别为4、3，与 VMRK 的 S  4、S  3 对应；以逐条样本时刻与代码一致性验证，不根据编号猜测。
- 同状态事件约每2秒重复，必须先合并同状态触发，再按状态切换构造区间。
- TMT.csv 227个唯一ID，226个正数，sub-010092 的原始值为逗号字符串；保留原值并按原规范记录排除，不自动纠正。
- 路径相对配置文件目录解析；CLI 从仓库根通过 python -m empirical_analysis.lemon 调用；PYTHONPATH=src 明确提供核心包。
- 正式模型模式要求明确协变量字段及编码；缺失时停止，pilot 必须显式选择，不能自动降级。
- 用户最新要求：从manifest按明确文件问题排除、数值ID排序选前150，构建支持.part/HTTP Range/已存在校验跳过的下载器，实际数据之后再下。
- 稳定bytes_url可用；不使用manifest临时签名url；元数据Git blob SHA1与本地文件吻合，可按清单算法验证。
- 2026-09-14 新授权覆盖此前“之后再下载”：下载现有150人计划到 `D:\数据集\LEMON原始EEG`，不运行分析阶段；目标在工作区写权限之外，需要实际执行命令的权限批准。
- 实际发现已有完全匹配的下载进程，因此无需新建写入目标盘的进程；第一项为633,640,000字节的sub-010002 EEG，3个共享元数据按现有plan顺序位于末尾。
- 日志计数：成功为downloaded/resumed/restarted/completed_partial；跳过为skipped；失败为failed；未完成总数=1053-成功-跳过（包含失败），未尝试=1053-日志条数。
- 真实试跑已验证5人张量为(5,61,16,5)，峰值进程驻留内存250.84 MiB；单人smoke含两次完整计算，不能把16.961秒称为一次处理耗时。
- sub-010005实际有17个候选25秒窗口，按既定规则选前16个；其余4人候选均16个，5人各处理400秒EC信号。
- 本次输入先做全文件清单校验，因此耗时不是冷缓存基准；峰值是每阶段独立进程的Windows PeakWorkingSetSize，含进程导入，不含其他进程和系统文件缓存。
- 不应把活跃下载目录直接设为eeg_root：现有指纹遍历全部文件，.part会变化，且.eeg.part因非.eeg后缀被整块读取。试跑明确使用不含.part的稳定C盘目录，未修改该既有逻辑。
- 下载器现默认180秒socket超时，每文件最多额外5次重试（共6次尝试），等待2/4/8/16/32秒；Transient异常仅由opener/read网络边界产生，写盘/校验失败不得触发重试。
- IncompleteRead.partial可能携带尚未写盘的尾部字节，必须保存后更新Range；新回归测试曾捕获13字节丢失并验证修复。
- 转正前检查exists再os.replace存在竞态；现Windows使用原子不覆盖rename，POSIX使用link后unlink，即使进度回调模拟同名正式文件出现也不会覆盖。
- 新增20项下载测试后总48项通过；日志记录attempts/retry_history，正常下载和已校验跳过的文件尝试数均为1，重试计数不含首次尝试。

## 2026-09-17 人口学数据
- 官方META原表227人，ID、Gender_ 1=female_2=male、Age；说明明确1女2男，Age五岁区间。146名张量受试者全数匹配且年龄/性别无缺失。
- 去除TMT无效者后145人，原年龄段35-40及55-60各1人；用户明确允许相邻合并30-40/55-65。最终7类别为20-25,25-30,30-40,55-65,65-70,70-75,75-80。
- 最终145人的7年龄类别计数依次46/43/10/13/17/12/4；女性66、男性79。每个既定外层训练折协变量列数8且秩8，无需改折分配、排除稀少类别或修改核心算法。

## 2026-10-04 已知 s 集群实验发现

- 旧 full 模式支持 fixed/resume，但按实验组筛选再分片会重编 seed_offset；须先固定全部 14 场景/种子再分配任务，不能并发写旧 raw_results.csv。
- 现有模拟 methods 的 Vector Lasso 为内部实现；集群独立入口可只依赖 NumPy/SciPy，避免经 main/runner 导入 plots 引入 Matplotlib。sklearn 仅未来 Elastic Net 等新基线需要，本轮不新增。
- 当前无 academic-search MCP；文献代理使用 web 核验期刊/作者来源，推荐优先 Vector-ABESS 与 Elastic Net，Sparse ISLET/正则 CP/Tucker 需另作结构匹配与实现验证。
- 现有 CP/Tucker 函数不满足相应论文拟合流程，不在本轮作为论文复现接入；此问题已向用户披露并停止复用，不改核心文件。
- Linux os.rename 可覆盖目标；新任务提交必须显式实现原子无覆盖语义，续跑先验证计划/代码/数据/结果，汇总缺项必须失败。
- 实际历史种子与旧main一次full规则不一致：v3/v5每个非baseline实验组都用10000偏移。新profiles按历史实际映射冻结，前140任务与两个旧版本分别对齐；不改旧入口。不同场景可能共享种子或重复设置，不能把700个任务混作独立同分布样本。
- 旧element_f1/q_hat按模态支撑笛卡尔积计算，Vector Lasso的投影矩形不等于实际非零元素集合；本轮保留复现口径，新向量基线需另报真实元素支撑指标。
- 默认已知s实验14场景×50重复×3方法，700任务/2100方法结果；保存真实B/噪声、估计B/支撑、公开入口初始化/损失轨迹及Oracle RSS gap。adaptive初始化与返回轨迹可能来自不同refinement阶段，记录中明确不把它们强配为同一次搜索起终点。
- 本地full baseline各方法28项非耗时指标复现历史，最大绝对误差2.8422e-14；仅正式task0已执行，不宣称700任务完成。

## 2026-10-10 任务卡2最终发现
- fixed接受轨迹必须与探索区分；返回best visited并不等于全局最优或测试预测变好。LEMON最终仍q80、RSS1.067482。
- 27配对6个模型改变，正式采用v5.1需完整重跑；旧seed2026复现仅保证该例，不代表全部不变。
- 强惩罚与启发式one-SE相叠可严重欠选；CV在较小内训练集的搜索不稳，不能当真实规模无效的证据。
- 真三元组覆盖9/27不是q60覆盖18/27；n100上界和n400稀疏网格均造成结构性不可达，评分不能选到库外真模型。
- v5.1可同步；MSIC alpha2/scale0只推荐研究，不成立v6。完整统计见doc/MSIC_ABLATION_REPORT.md。

## 2026-10-10 任务卡3设计前置
- 放松caps必须同时检查adaptive conditional group s_max：旧代码直接按cap枚举，各mode未额外限制q；需候选级动态cap=floor(qmax/其他两mode规模乘积)，防止多参数OLS超容量。
- 边际C集合只约束初始化Group Lasso；后续build_mode_design使用全mode位置，C遗漏不等于永久不可达。

## 任务卡3冻结代码解释边界（独立只读审查）
- full只对训练caps内且q≤Q的全部整数规模完整；仍不是所有维度+Q合法空间，也不穷举支持位置。n400 init_sizes1/1/1 rawcaps10/10/10条件下，(16,1,1)q16≤280且dim合法仍不可达；不要写“所有理论可达模型已考虑”。
- A1−A0是旧grid规则下cap策略条件效果，包括新adaptive路径和anchor重定位。cap5 grid全1..5，cap15旧grid变{1,6,9,13,14,15}丢5，不能称简单扩充原池。
- 条件效果应含caps A1−A0与A3−A2、grid A2−A0与A3−A1，交互(A3−A2)−(A1−A0)。
- 同rawcaps且同初始化 A1⊆A4⊆A3；pool最小MSIC A3≤A4≤A1。same legacycaps A2最小≤A0。one-SE最终q/所选score/test表现不保证单调，阈值依赖minimum_q/minscore。
- 跨caps A3未必含A0旧adaptive-stage B；fixed规模嵌套不等于整个实际B候选池嵌套。应分规模/位置覆盖，不推定跨cap最低score必改善。

## 任务卡3 最终独立只读核验补充

- VERSION 的 33 个独立核心文件逐一核验 raw SHA-256，全部匹配；同步补丁实际 SHA-256 为 1eda5aa33593717e32cb8a86257d38136908d26a62e0fcb353d48046d310ff18，与 team MANIFEST 相同。版本保持 research_only=true、release_approved=false。本次核验未修改算法或开展新拟合；team MANIFEST 的测试数仍暂记 90，VERSION 已记 100，由主线最终统一统计。
- 生产 src/tensor_abess/tensor/bcd.py 的 raw SHA-256 仍为 c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737；规范化 CRLF/CR 为 LF 后的 canonical SHA-256 是 4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c。前者标识当前文件原始字节，后者用于跨 Windows/Linux 换行的一致性检查，不能混用。独立 v5.2 的固定 tensor_bcd 函数源码与生产函数逐字相同；Group-BESS、自适应组选择、交换、SIC/MSIC、OLS、mode reduction、GLasso 模块亦与生产文件字节相同。
- 默认 legacy/legacy 保留初始化 cap 的原压缩、原网格顺序、原固定拟合及原 one-SE 选择；A0 独立进程小数据回归与历史预检已验证 B、规模、RSS、MSIC 一致。新增记录和显式缓存不更改数学步骤；缓存以训练 X/y 的完整字节、shape、dtype 和固定 solver 配置绑定，每次 refine 仅计算一次训练指纹。
- 全局精修仍使用原 MSIC alpha=1.5、gamma=1、one-SE scale=1；adaptive 阶段的原 mode/global MSIC alpha=1 保持不变。one-SE 阈值为 minimum_score 加 scale*sqrt(q_best*log(n))，随最低得分及其 q 改变；即使嵌套池的最低训练得分下降，最终所选 q、所选得分或独立测试误差也不保证下降。启发式 one-SE 容差不是实际 CV 标准误。
- full 只覆盖训练初始化 raw/compressed cap 以内且 q<=Q 的全部规模，并不覆盖所有 dimension+Q 合法规模，也不穷举各规模的支持位置；raw cap 仍由训练 GLasso 规模和 marginal floor 启发式生成。coarse_to_fine 只按训练 MSIC 前三候选的一次局部邻域扩展，保留明确 excluded 合法规模清单，不能称完整搜索或全局最优。
- A0-A4 实验未传 target_sparsity；cap、初始化、网格、fixed 候选拟合、训练精修和最终选择仅依赖训练 X/y 及预先设定规则。B_true/真实支撑仅在拟合后计算覆盖与恢复指标，独立测试流仅在选择完成后统一评估；它们未进入核心候选生成或选择接口。
- 已记录的 48 个文件 Python 3.9 语法检查只证明语法可解析，不证明在用户 Python 3.9.25、NumPy 2.0.2、SciPy 1.13.1 集群的实际运行已完成；本地测试来自现有 Python 3.12 环境，未更新用户集群。不能把语法检查、首任务预检或本地研究 cohort 写成正式集群 full 700 任务已运行。
- 历史完整 27 数据集的 135 个所选模型数值审计通过，秩亏缺 0；A1 最大条件数 13.238778108452305，A3/A4 最大 11.693625539899388。RSS 重算相对差最大 4.911796579776584e-16，独立 OLS RSS 差 0，正规方程相对残差最大 2.254609895369797e-15。该证据仅适用于所选模型，不是全部候选设计矩阵的条件数审计；审计没有替换 B 或实施数值救补。

## 任务卡3 A3真支撑已进池但未选中：原分数独立诊断

仅使用完整 `outputs_candidate_search_study/historical_final` 的既有 A3 candidate score、RSS、selection minimum/tolerance/eligible/selected 记录；没有重新拟合、重新评分、试新alpha/scale、执行P4或根据测试误差挑选规则。原global MSIC仍为alpha=1.5、gamma=1、one-SE scale=1；启发式容差不是交叉验证标准误。A3池含真完整支撑10/27，最终Exact6/27，四个遗漏任务明确为013、014、015、018。

真支撑位置由与新历史任务训练/test指纹相同的旧v5.1 NPZ B_true读取，仅供本诊断；四例真候选尺寸均(4,3,5)、q=60。以下分数和RSS均为既有训练产物，显示到6位小数，判断使用保存的原精度。

|task|n|SNR|真候选index|真RSS|真MSIC|minimum index / shape / q|minimum RSS|minimum MSIC|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|013|200|2|263|123.698079|101.339914|377 / (6,3,7) / 126|40.042504|49.883837|
|014|200|2|165|154.694340|146.061398|140 / (3,6,7) / 126|46.829935|78.396374|
|015|200|4|313|115.504068|87.632315|339 / (4,6,5) / 120|45.555699|57.503164|
|018|400|0.5|324|356.980408|166.479216|226 / (3,3,6) / 54|362.840725|155.488277|

|task|真score−minimum|one-SE tolerance|minimum+tolerance上界|真候选eligible?|eligible数|最终selected index / shape / q|最终RSS|最终MSIC|
|---|---:|---:|---:|---|---:|---|---:|---:|
|013|51.456077|25.837724|75.721561|否|6|267 / (4,3,9) / 108|57.864183|75.527915|
|014|67.665024|25.837724|104.234098|否|5|140 / (3,6,7) / 126|46.829935|78.396374|
|015|30.129151|25.215037|82.718201|否|9|324 / (4,4,5) / 80|86.390654|81.981319|
|018|10.990940|17.987192|173.475469|是|13|124 / (2,3,4) / 24|473.994281|172.968479|

- task013/014/015：原minimum本身已是非真模型且q=126/126/120，真候选score甚至超过现有容差带上界。因此这3例不能归因为one-SE把“真最小分数模型”缩小删掉；one-SE分别把q从126变108、126保持126、120变80，所选仍非真。既有原MSIC对这些有限样本的候选评分是本轮剩余选择失配，不能仅由此断言某个新惩罚参数已获验证。
- task018：原minimum也非真，q=54；真q60候选处于eligible容差带，最终按eligible中的最小q选择q24。这里one-SE进一步偏向小模型，但不是四例中唯一原因：原minimum已经不是完整真支撑，不能称“去掉one-SE就恢复真模型”。
- 四例纯粹“真支撑就是minimum、只被one-SE替换”的数量为0。分类为3例真候选在minimum容差带之外，1例minimum非真且真候选仍在带内、最终被最小q偏好排除。位置搜索已覆盖真模型与评分能选中真模型是两件独立事情。
- 已在原精度上核验minimum_score等于保存候选score最小值；eligible_indices恰为score<=minimum+tolerance的候选；最终selected按(q,score,sum(sizes))字典序从eligible取得，均与已保存selection一致。未计算任何新的alpha/scale准则。
## 2026-10-11 任务卡4前置核查

- 任务卡3历史27、开发90、确认90以及A3 MSIC确认90已实际完成，旧“未执行”叙述属于文档收尾缺漏，已补齐。
- 确认A0 Exact6/90、PMSE3.118404313；A2 Exact26/90、PMSE3.082733325。A3新MSIC不能直接作为A2的规则依据。
- 原五组共享fixed缓存，历史runtime_sec不等于独立冷启动耗时；A2缓存成本加回估计194.3502秒/A0实测172.9293秒，需另作预声明冷启动核验。

## 2026-10-11 任务卡4最终决策

- A2新形状主90 Exact15→36、PMSE下降1.03%；成本估计+14.88%、cold实测+10.37%，原事前门禁通过；收益主要来自n400，rho.6仅验证n200兼容性，旧caps及弱信号限制未消失。
- A2两MSIC规则开发90筛选通过，但独立80确认在3600秒预算停止、停止时68完成，最终71份留存不能替代完整验收；fresh refit0、无完整analysis/integrity，不据部分结果批准新MSIC；冻结规则要求保留原MSIC，故推荐D2。
- 157项限定测试通过、2040受保护文件差异0；最终D2只改私有规模枚举，32源码/公共接口/原MSIC冻结，七份交付及队友/full准备齐备；research_only=true、release_approved=false，生产仍v5.1，正式发布待用户批准。

## 2026-10-11 任务卡5正式发布

- 用户任务卡5正式批准D2，当前生产32源匹配冻结v5.2，唯一核心变动bcd网格；历史研究状态不改，批准/released独立记录于doc/releases/v5.2-search-final。已有HEAD不是新commit，大量既有未提交修改保留，采用独立patch/完整指纹交接。
- 66项发布/部署/清理限定测试通过；旧known-s shell的一项字面activation静态断言失败在非发布范围单列，新入口沿用实际conda初始化并独立验证。
- 125再生缓存显式删除、7068重要hash差异0；目录净大小仍增加，归档只是原位置索引，没有历史移动/改标签或数据删除。
- OpenSSH9.5默认SFTP路径不套旧shell引号；实际PowerShell mock Apply/普通副本backup和source/path/hash冲突通过，新700入口必须已有真实SLURM单任务Gate。当前远端全部NOT_RUN，本机仅task0四方法完成并strict resume不变。
- 旧LEMON EEG增量结果32core匹配v5-original，q80接近n116容量81；新版仍需保留仅协变量基线及同一划分重新训练全部ABESS折，不能假定A2或BCD修复必然改善预测。

## 2026-10-15 full模拟验收启动
- 正式目录已包含plan、PASSED集群预检及task/slurm原文件；尚无原生汇总CSV/integrity报告。
- 正式manifest已记录任务卡5批准D2且released；历史任务卡4文档的待批状态应按时间解释，不能覆盖正式发布身份。
- 实证启动需完整模拟验收、各项目源码/数据/划分Gate和明确运行授权；本轮仅做判断及报告。

### 2026-10-15 身份与作业记录初审
- plan_id=30f231df0327e58331a18580dc8a03caeb0872dc991e57c42b095556100a80f6；full/50重复/700任务/4方法。
- 38个plan源码与依赖raw SHA256均匹配现场；其中32 core同时匹配正式发布manifest。
- 原目录恰700 JSON+700 NPZ；700个SLURM签名记录覆盖task0..699，所有记录签名、log hash及关联record hash匹配，Gate签名与task0两个原产物hash匹配。
- 作业36235单任务task0为严格校验后skipped，作业36255含699个completed；单任务Gate不再是NOT_RUN。
- 审计脚本首次误用紧凑JSON序列化计算job签名，产生工具层假阳性；已按原runner的sort_keys/ensure_ascii/默认分隔符规范重核700条，差异0，未改任何签名或产物。
- 原runner read_task不重生训练/测试X/y，不独立重算预测指标或OLS；后续报告严格限定核验范围。

### 2026-10-15 完整验收与科学判断
- 原生完整性通过，独立read_task700复核亦通过；所有原始task/plan/gate/slurm及生产源码均保留。新生成raw/summary/integrity及两份不适用fixed比较表头。
- 14标签仅10不同参数配置、600完整输入指纹组合（不是700独立数据）；sample200/dimbase/rho.3重复同50组，报告明确披露。
- 严格旧新700对task/env/input一致，38源仅BCD不同；旧32core逐项匹配v5-original VERSION，Oracle非runtime指标全部一致。新Exact138→182/700，但PMSE2.236761→2.381274（+6.46%）；去重600 PMSE+7.22%、Exact120→162，不能称全面改进，也不能将v5到v5.2组合效应单独归因A2。
- n100 caps(4,4,4)排除真(4,3,5)，Exact0/50/PMSE6.442；n400 Exact1→49/50，新PMSE1.1895。旧caps限制与弱信号、相关/高维恢复风险全部写入报告。
- 工程released身份维持，推荐冻结版交队友验证，暂缓全面性能优势/论文定稿主张；项目自身Gate、完整OOF与仅协变量基线仍需执行。

## 2026-10-15 任务卡6准备
- 新LEMON目标目录尚不存在；旧结果已完整75 fits/2175 OOF，记录总耗时约1208.85秒，约20.15分钟。
- P0正式训练X/y未完整保存，只有hash、B_true与noise；必须重生且逐项raw指纹匹配，不以相同seed/allclose替代，需核对本机与Linux计算差异。
- 本地任务卡6在P0.3标题后跳到交接说明，缺P1标题及细则；已向用户询问，当前子任务只读审计，未启动拟合或新实现。

### 2026-10-15 任务卡6核心发现
- grid AST及96源码审计：v5.1到v5.2仅候选函数不同；n100 caps4/4/4的64与n200 caps5/5/5的125有序网格完全相同；n400 caps6/6/7从150→252，真第二模3由库外进库；这些是结构覆盖证据，不冒充250数值归因。
- 用原init函数caps生成AST穷举6324种LEMON初始化长度组合，n116/dims61/16/5/c3始终caps4/4/5，原/新80规模同序，A2未扩LEMON网格。不能把当前LEMON恶化归于新增规模。
- 模拟ΔPMSE中位数0，正累计124.080355、负累计-22.921523、净101.158833；最大20例占净恶化72.20%，n100+rho0贡献80.06%，不是单一离群决定。94恶化中73增q且RSS降、8同q且RSS降、13减q且RSS未降，不能统一归因为大q。
- 当前P0严格输入Gate阻塞；P3建议保持正确性修复的冻结基线、保留负面实证、在原环境补250并研究结构/数据匹配，不自动调整caps/MSIC或恢复已知缺陷BCD。
