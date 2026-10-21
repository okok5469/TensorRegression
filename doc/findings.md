# LEMON 实施依据

## 2026-10-18 清理依据
- 任务卡5曾仅删除125个未跟踪的Python再生字节码；其清单不可直接重用。旧脚本apply_plan支持完整清单先验、路径边界、跟踪/引用/保护和SHA核对后逐文件删除。
- 任务卡10原6820保护和追加1053交付索引仍应完整保留；不能根据old、STOPPED或temp文件名认定历史证据废弃。
- 本轮两份当前SHA清单和相关18份历史SHA清单均不含缓存；旧全目录盘点CSV的缓存路径只是历史库存记录。保留旧CSV自身，不需为缓存清理重写历史。旧apply_plan的宽目录KEEP不适用于本轮全部候选，因此只复用精确引用检查，删除用原生PowerShell按新封存清单独立校验后执行，无递归删除。
- 临时目录中的两份PDF抽取文本、Task6 ZIP验证解压副本及push.log共85文件均已Git跟踪，本轮保留；.pytest_cache读取和Git状态比较必须使用一致权限上下文，不能把沙箱不可读状态等同实际缺失。

## 2026-10-18 任务卡10续跑边界
- 完整开发新增结论：n100 A2相对A0平均PMSE降低33.15%，同容量A2相对A1仅降低3.19%；总体同库改善0.85%，n400反而增加5.44%。n400 A0/A1逐例结果全相同，CV为1改善/14相同/9恶化，Exact13→11。
- n400丢失恢复的task033、037公共库均已有真支撑；A2分别从q60选到q80、q24选到q60，训练RSS下降而外测PMSE上升，属于选中更大错误支撑而非真形状被容量排除。最坏cell11为n100/SNR4/真形状234/rho.6，两重复PMSE比2.836。
- 强SNR总体欠选率A0=31.25%、A1=A2=50%；相对A0差异包含容量/来源变化，不能全归CV。预定联合门槛仍判失败，不因这种机制区分撤销门槛。
- task030/031在2worker续跑中约442秒成功；不能据此将原900秒超时判为算法数值故障，也不能把混合并发计时宣称为统一速度基准。原超时和3例同伴中止开销必须保留，未保存的失败CPU/峰值不可补造。
- 本次是用户授权后的计算协议修订；原已完成30例并未因换目录获得未见身份，原4例中止输入重算也不是新科学重复。
- 原单shape单fixed-B设计不能识别同形状异支撑多来源风险；续跑不改候选定义，只将Q5作为明确未解决问题报告，不能标PASS。
- 原worker可原样接受预算修订协议；完整30副本保留全部原始字节，新18用新plan，合并开发锁必须覆盖原48 ID且记录旧/新来源；确认锁不能仅依赖文件数量。

## 2026-10-18 任务卡10设计审计
- 最终科学状态为超时停止：开发30/48，n100完整24而n400仅6个弱信号；不能用部分总体重算9推广门槛、生成开发锁或进入确认。Pilot估计与4worker正式尾部耗时不等价，900秒单case期限独立于7200秒总预算。
- n100描述性PMSE A0/A1/A2=4.679961/3.231389/3.128412，主要降幅发生于A0→A1（容量/来源混合），同容量CV额外下降约3.19%；未确认不能推广。同形状多来源风险在本设计不可识别，不能用CV选形状宣称解决。
- 完整34实例10842候选均满秩、独立OLS系数差0、最大条件数83.1274；该证据仅是数值正确性，不证明估计方差小或Ridge必要。部分候选/被中止case不纳入通过计数，外测独立流重放仅Pilot4例实际执行。
- 种子区间必须把train/test/split三种角色一起检查，不能只查训练测试；控制器TDD已识别未冻结草案区间冲突并停启动，改为相隔400M的三阶段base后再通过完整审计才允许生成。
- 必须区分A1−A0容量/范围程序变化、A2−A1同容量CV选择差异、A2−A0总变化；候选集合不能依赖验证响应。
- 正式caps下限floor(sqrt(n)/2)，在本模拟dims及n100/400与其三折内训练量下，下限乘积已超Q，缩减到平衡区后caps可由n/dims确定；需证明并桥接再冻结。
- Task9全部证据冻结不改，新开发/确认须新随机族及完整输入，与旧数据身份去重。

## 2026-10-17 任务卡9证据边界
- 结果：n100 28/50恶化，26例q上升中25恶化，但2例同形状换支撑也恶化；n400网格49/50改善，Exact1→49。Task8池首个阻断n100为容量36/位置30/选阶2/成功4，不能把68个非Exact全归MSIC。
- 12例全池重放8个正selection gap，9对同形状异支撑有3对低RSS高PMSE；q/shape惩罚不能改变同形状的RSS排序。216例Exact A0/A1/A2/A3=71/65/49/69，增强惩罚平均收缩、去band扩张；存在2个加罚最终q反向上升的band交互反例。
- 固定Pure Warm25门禁通过，seed2026保留Cold RSS154.472，额外起点RSS277.869更差；只有固定规模原型可行性，非全adaptive/性能证明。
- 250配对保存750所选模型，无全候选池、原X/y或测试流；位置搜索/选阶gap与OLS秩条件数不可独立补算。
- Task8有216完整库共31968候选，可固定系数重评分；12例development按配置机械抽取、先身份Gate后测试评价，不作为新确认。
- 缺少精确文件名“代码终版(2).md”，使用现存Math说明作背景，真实源码为公式依据；不假定两文档字节相同。

## 2026-10-17 阶段汇总依据
- 阶段报告已完成，区分工程正确性/恢复/泛化，原样Warm未运行性能、旧80确认预算中止、卡8完整216门槛失败，以及卡7已完成250的当前状态。
- 复核纠正草稿一处分类：27历史数据首次阻断为caps9/grid9/位置6，另3为真支撑覆盖成功，不能写成最终选模失败；引用来源及正文已修正。
- 用户要求汇总任务卡1–8及执行结果，写入既有work_progress目录；采用Markdown，沿用已有进展文档格式。
- 早期work_progress存在“进行中”和250待集群等历史状态，本次以任务卡7最新250/250接收报告及任务卡8完整216数据/门槛结果为准，保留旧记录。

## 2026-10-17 任务卡7结果补全
- 全量独立验收通过：原生load_plan重建、96源码、250 seal/四输入/环境、5 Gate、750 NPZ和已有CSV逐字段一致；750模型独立REE/支撑指标/MSIC误差≤5.69e-14。非n400的200组v51/v52 B/支撑精确相同；n4001/50相同。
- 最终五场景ΔBCD/ΔGrid PMSE：baseline+.021461/0；n100+1.245466/0；n400+.001716/−.109788；SNR.5+.089270/0；rho0+.374218/0。n400 Exact1→1→49/50，网格收益并非BCD修复单独带来。
- 250输入唯一但只有150seed/test_seed族，按五场景各50配对描述；原700仅600唯一组合，历史138→182为标签加权描述。未保存原X/y及逐测试数组，本轮不声称独立重算RSS/PMSE，来源边界已记录。
- 实际目录为numerical_simulation/outputs_v51_attribution/cluster_run，含250对结果、5 Gate、原生CSV和PASSED250的integrity；尚需独立验收，不重跑拒覆写的merge。
- 五场景按原config确认：task0 baseline n200/SNR2/rho.3；50弱信号SNR.5；250小样本n100；350大样本n400；550独立rho0。不能按任意标签列表顺序绑定场景。
- n100 PMSE5.196560→6.442026→6.442026；n4001.297588→1.299304→1.189516。初步数值待全部身份和数值验收后进入正式结论。

## 2026-10-16 任务卡8前置
- D确认108完成：PMSE2.443688→2.252872（−7.81%），Exact36→34；最坏cell比1.156737>1.15，SNR4 q欠选增0.055556>0.05，确认未过。后者增加来自逐模不可表示子集，可表示30例欠选率不变，但不能事后改门槛。
- 最终验收：全部216数据/31968候选系数有限性、合法支撑及逐数组SHA/31752 fixed轨迹和选择重放通过；3253保护文件无差异。预选12例独立重建输入/测试流、RSS、OLS条件和所选指标通过，未重拟合。
- 独立D开发108/108已锁定：PMSE比0.951085，但最坏cell比1.566972超过1.15，开发门槛失败；其余四项通过。确认只按拟合前预声明作固定规则研究复核，不再调参或反称开发合格。
- 中途只读完整性核验通过：3253保护文件、32冻结core、108开发及当时32确认的所有候选元数据/最佳访问路径/选择重放与artifact SHA无差异；最终全量核验仍待控制器终态。
- Warm原样实测：上传6/6过但历史seed2026错误支撑/RSS252.229556与旧warm-only相同，Cold恢复真支撑/RSS154.471958；正式31方法全过，Warm20过/11失败或错误。best-visited等注入契约失败与真实历史回归分别说明。
- D2：full q80/80/24；common MSIC q1/2/24，CV q2/2/4，stable q32/2/24；48共同规模的限制本身改变前两组选择，不能将收益全部归CV。12拟合和432候选SSE已独立验收。
- 初始保护复核发现任务卡7索引后两份集群launcher文档已有更新，现场另有对应launcher及8项测试；旧2880数值/生产保护完全匹配。已记录旧/今SHA于taskcard8/BASELINE_RECONCILIATION.json，保留旧索引与当前文档，不放宽科学身份。
- 真正incoming bcd.py与上传test均存在；静态初审同时修改Warm、fixed正确性保护及候选网格，必须原样测试，不能只凭队友RMSE改善认可。
- 独立D预算已由用户批准：36cells×开发/确认各3重复，总墙时7200秒；新seed与开发后锁定/确认门禁需拟合前冻结。
- 旧test_tools权限问题已在此前获准继续目标访问，本轮不依赖这些目录；已有测试/实验均用历史.venv且不安装依赖。

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
# 2026-10-16 任务卡7前置
- 实测最终归因：15折v5.1=v5.2的B/支撑/预测精确一致；RMSE .371667→.504581，q80 1→5，LEMON变化在v5→v5.1。
- C2六代表训练折重放：486个v5.1/v5.2候选及路径完全一致；480个v5→v5.1 fixed候选295 RSS下降、185相同、0变差，惩罚不变。
- C3三预声明外训练集/9内折：固定q1→12→45时训练误差9/9下降、验证7/9严格上升；大MSIC8/9低于小，全部三组adaptive内OOF劣于协变量。
- 工程正确性、统计恢复与泛化分开：保留v5.2基线，独立研究新选阶；不认定EEG无信号、不用外测选alpha。250按用户选择稍后集群执行，当前0/250。
- 旧LEMON v5/v5.2完整OOF已保存，数据145人，三重复五折；历史32源码及Windows环境已有配对审计。
- 250模拟任务此前实际0/250：Linux3.9.25/NumPy2.0.2/SciPy1.13.1与本机不同，除B_true外输入hash失败；不得降低Gate。
- 用户本轮已解除访问阻塞，继续限定目标路径，不恢复/改写无关历史测试工具。

## 2026-10-16 单文件SLURM启动
- 新增run_v51_attribution.sh，固定站点/root/chdir及现存根目录日志；不需要手动cd或跨终端变量，不改变冻结runner/plan。
- 单作业串行，stdlib fcntl非阻塞锁且pass_fds继承给子进程；五pilot后逐个245worker，日志即时打印任务ID/计数。
- 完整旧结果先检查seal/plan/task/version/core/environment/inputs/NPZ/pilot gate再跳过；任何残缺、失败preflight或已有汇总立即停止，绝不修复或覆盖。
- 历史成功preflight仅标为保存记录核验，所有新worker实测当前环境和输入；全已完成只做原生只读结果汇总，不宣称重测当前环境。

## 2026-10-18 Task11 evidence scope
- Task10 valid final evidence is doc/taskcard10/continuation_01: n400 A0/A1/A2 Exact=13/13/11 of 24; independent confirmation NOT_RUN due failed development gates.
- Historical Task8/9 pools are already seen and diagnostic only; not independent confirmation.
- Fixed candidate FS=MSIC0+ell*q^2/(2*n); q-dependent penalties cannot reorder identical-shape supports by training RSS.

### Task11 completed prospective evidence
- All 24 frozen cells x 4 independent cases completed (96/96; 15264 candidates), with no failed, omitted or replaced input; Pilot and history excluded from these statistics.
- Overall PMSE 2.978516 -> 2.176566 (ratio 0.730755); n100 4.762711 -> 3.153757 (0.662177); n400 1.194322 -> 1.199375 (1.004231). Exact 31 -> 33 overall and 30 -> 30 at n400; no previously exact case was lost.
- Six of seven statistical gates pass. The sole failure is strong-SNR q-underselection: 7/32 -> 11/32, delta +0.125 > +0.05. All four newly underselected cases improve PMSE; three have cap-unrepresentable truth (4,3,5), one has representable (2,3,4) but no true support in the saved pool. The fixed all-strong denominator is not revised after seeing this split.
- n100 P95 ratio 0.576230; worst complete-cell ratio 1.031972; five individual PMSE deteriorations are retained. Frozen paired-bootstrap mean-ratio intervals support overall/n100 gains, but four repeats per cell do not establish broad tail or recovery noninferiority.
- Decision B follows the predeclared all-gates rule despite material prediction gains: retain original MSIC and v5.2-search-final; no new formula, tuning round, release or real-data fit. Dataset suitability audit framework is the next-stage handoff.


## 2026-10-18 Task12 scope and initial evidence
- Requested result now differs from Task11: fixed FS research interface and new380 paired cluster inputs (not locally submitted), then existing-data dual OOF comparisons; preserve Task11 result B and failure +0.125 strong underselection.
- Root CLUSTER_GUIDE.md is actual configuration source: module load anaconda3 then conda activate myenv, project /public/home/xjh/xmx/tensor-sparse-regression-main, Slurm CMO_HPC/account xjh, 1 CPU per worker, historical array project cap64. This is not a confirmed site quota.
- Cluster results remain PENDING_CLUSTER_RESULTS until user uploads/submits and returns complete raw results. Windows smoke is not remote validation.

### Task12 final local evidence
- LEMON fixed FS changes10/15 folds; q mean37.467->3.533, median24->2; q decreases7/increases3/same5. RMSE0.504581->0.359627 (28.73% lower), still13.27% above CovOnly0.317485. Three repeats all improve original yet all remain worse than covariates; no stable EEG incremental advantage established.
- Sole adverse fold repeat1/fold5 q1->2,RMSE0.307022->0.342161; worst FS fold repeat2/fold1 RMSE0.579377 remains unchangedq24. No difficult fold/subject removed; no parameter changed from outcomes.
- HCP-YA/NKI-RS actual inputs and prior metrics unavailable in audited repo scopes. Generic code currently RMSE/R2; a different original metric requires its adapter before that dataset's future frozen protocol, not after results.
- Task11 original strong-underselection failure0.125>0.05 preserved. New380 protocol separates balanced360 from additional20 pressure cases, emphasizes n200 representable strong support and30-case stress loss, reports conditional Exact-loss uncertainty, and does not auto-release FS.


### Task12 cluster return closure
- Independent saved-chain verification PASS: formal380, smoke3, frozen sources/protocols/seeds/artifact hashes.
- User-reported original Linux strict verification PASS: smoke3/formal380 and exact summary/analysis match with four frozen thread variables=1. Windows floating-point stop preserved; second command without variables correctly rejected before validation.
- Final recommendation B: retain original MSIC production default; FS frozen research only. Primary PMSE -19.65%, n100 -28.70%; four recovery gates failed. HCP/NKI NOT_RUN; no further tuning or production changes.
- Local delivery snapshots preserved under doc/taskcard12/cluster_review/local_delivery_snapshot; current final reports and evidence indexed separately.
