# 2026-10-10：任务卡3候选搜索研究（进行中）

执行任务卡根目录任务卡/任务卡3.md，先审计，再独立caps/grid四组搜索，位置搜索，再候选机制验收，满足门槛后才进入MSIC。旧任务卡目录已移动，从根定位，未猜测用户未保存文件。

生产src维持v5.1，原32源码冻结至doc/algorithm_versions/v5.1-bcd-fix/src。研究v5.2为隔离33源码，只有utils/init.py、tensor/abess.py、tensor/bcd.py与新增tensor/candidate_search.py有改动；固定tensor_bcd函数逐字与v5.1相同，Group-BESS定义未改。

Phase A已完成：27历史、3078全部候选（含3051fixed）审计，输入X/y/Btrue与独立test指纹重建一致，OLS/B/支持/RSS与原源码指纹一致。真三元组可达18/27、进池9/27、完整真位置3/27、某种q60形状18/27。首次阻断caps9、grid9、位置6、覆盖3。n100旧caps4,4,4排除真实第三模5；n400旧caps6,6,7允许真三元组，但第一二模网格缺3，第三模缺2，252合法形状只搜索150。边际C只限制GLasso初始化，不永久限制BCD全部模式位置。

研究设置A0旧caps/旧网格，A1原rawcaps不压缩/旧网格，A2旧caps/完整整数，A3rawcaps/完整整数，A4rawcaps/预定训练top3的±1邻域一次。candidate q≤floor(.7n)，adaptive每mode动态cap按当前另两模规模计算。完整网格只对指定caps内全合法整数，不是整个维度空间或所有位置穷举；rawcaps仍为训练GLasso大小与sqrt(n)/2的启发式。

MSIC不动：mode alpha1、global1.5、gamma1、scale1。候选budget预声明2000（历史训练初始化derive计数最高1044）；超限明确停止，不截断叫完整。显式固定候选cache绑定X/y形状/类型/字节与solver配置，group顺序固定。观察runtime受缓存顺序影响，分析另加已缓存fit秒数得到单独运行成本估算；不是实测独立运行时间。

限定测试当前90通过：新core13+原best-BCD10+原tensorBCD5+原LEMON12+原numerical14+runner23+multi-init5+stability runner5+analysis3。原seed2026恢复断言未削弱、独立v52默认通过；Python3.9语法已核对，实际本机Py3.12.8/NumPy1.26.4/SciPy1.16.1，不升级用户Py3.9.25/NumPy2.0.2/SciPy1.13.1集群。

重要故障记录：首次验证器对所有kind强求loss_history非增，development task33（n200/SNR.5/seed20364013）失败。立即停止历史controller及确认属于本任务的子进程，开发自然退出。两agent独立复现：adaptive q1→4→2→2，RSS270.1054495→225.2495091→232.4915080→同，MSIC77.9073052→54.9125564→53.3367269→同；q4缩q2 RSS增加7.241999但MSIC改善1.575829，正常选阶。125fixed全通过接受单调/minvisited/bestpointer/OLS/容量。

错误在验证规则，非算法数值回归；未改算法、容差、种子或数据。先添加真实反例回归观察旧规则FAIL，再只在global-fixed强制accepted loss非增并加强minvisited/指针/finite，adaptive要求共同容量/支持/有限/RSS一致，不误把变q路径当固定接受历史。unknown kind拒绝。修正worker与test_run，run.py及core33字节不变；revision2-stage-objectives，runner23通过，真实失败A0重新审计通过。诊断doc/algorithm_versions/taskcard3_validator_failure.json。

旧历史16、开发35、确认0与旧稳定性2任务全部保留STOPPED，不复制、修补、重标或混为最终；旧验证器代码保存在doc/algorithm_versions/v5.2-candidate-search-validator-rejected。核心研究机制并未作废，仅旧错误验证器作废。

当前已建立全新同seed计划：
- numerical_simulation/outputs_candidate_search_study/historical_final：27，plan889780a394bf8a9fa93c60302521b37557a143abde079f9b3a4316f2a5d736b7。
- development_final：90，每场景10，base20361010+1000scenario+rep，plan950d722abda2a20f84581a969759ad85a338d0cc86e1681ee24a9e98b1767427。
- confirmation_final：90，base20461010独立，planbe7b227228164e6dbbf3d7305e93717c62128e6982f5ff36000a1f1218fd973c；尚未执行，不查看确认结果选规则。
全部test_seed=seed+1e6，原config dims32/20/25、true4/3/5仅生成与评价、rho.3、ntest500。117历史+开发用8个独立单线程进程调度，controller PID53952，exec session24999；未启动700正式full。

原预检n100task0与n400task18 A0的B和7主要指标逐值等于原v51。task18新增A2/A3真完整支持入池（原A0无），默认仍q24，PMSE1.5090→1.4031；只是单例机制迹象，不是全研究结论。旧验证器预检保留，不拿作最终新计划结果。

位置研究模块固定marginal+两个seed+30m/+31m随机起点，按训练RSS选min，不接受true/test输入。旧两个预检四组全marginal胜；task0部分random训练更差、test更好，未按test选择。Oracle真支持OLS仅reference，不能认为每例一定是PMSE下界。将新historical_final完备后另立稳定性plan重跑27；旧2结果不混入。

报告：doc/CANDIDATE_SEARCH_AUDIT.md已完成；其余COMPARISON/SUPPORT_SEARCH_STABILITY/MSIC_REEVALUATION/ALGORITHM_VERSION_DECISION/TEAM_SYNC_UPDATE待完整实验与门槛决策后写最终版。core patch已准备供研究review；没有改生产、新branch/worktree、推送或发送队友消息，没有采用正式v5.2/v6。

## 任务卡3冻结代码解释边界（独立只读审查）
- full只对训练caps内且q≤Q的全部整数规模完整；仍不是所有维度+Q合法空间，也不穷举支持位置。n400 init_sizes1/1/1 rawcaps10/10/10条件下，(16,1,1)q16≤280且dim合法仍不可达；不要写“所有理论可达模型已考虑”。
- A1−A0是旧grid规则下cap策略条件效果，包括新adaptive路径和anchor重定位。cap5 grid全1..5，cap15旧grid变{1,6,9,13,14,15}丢5，不能称简单扩充原池。
- 条件效果应含caps A1−A0与A3−A2、grid A2−A0与A3−A1，交互(A3−A2)−(A1−A0)。
- 同rawcaps且同初始化 A1⊆A4⊆A3；pool最小MSIC A3≤A4≤A1。same legacycaps A2最小≤A0。one-SE最终q/所选score/test表现不保证单调，阈值依赖minimum_q/minscore。
- 跨caps A3未必含A0旧adaptive-stage B；fixed规模嵌套不等于整个实际B候选池嵌套。应分规模/位置覆盖，不推定跨cap最低score必改善。

## 已完成部分复核（开发集计算中）
历史_final27/27已完整汇总：A0精确恢复3/27、PMSE3.364051；A1为0/27、3.611642；A2为9/27、3.329476；A3为6/27、3.407057；A4为6/27、3.548579。A0全27的B、七项指标与任务卡2v5.1逐值相同，数据指纹完全一致；复现证明见doc/algorithm_versions/taskcard3_A0_reproduction.json。完整整数与放宽caps作用不同，历史结论不替代新开发验收。
历史135所选OLS设计矩阵均满秩，最坏条件数13.2388，独立OLS、张量预测、训练RSS一致，无数值异常；仅对所选模型审查条件数，不推称所有候选均审查。26,782个fixed候选记录通过轨迹核验，其中缓存重复不是26,782个独立拟合。
稳定性historical_final已27/27完成、54训练选定shape配对：A0随机起点胜3/27，multi−marginal平均RSS−0.611442、PMSE−0.106215；A3胜4/27、RSS−0.134216、PMSE−0.016024。两组精确恢复率均无提高，PMSE胜平负分别1/24/2、2/23/2；按训练RSS选，未按测试挑起点。细节doc/SUPPORT_SEARCH_STABILITY.md。
新增交互6项、OLS审计4项测试通过，限定测试总数更新100；先前90项记录为当时状态。48个研究/测试源码通过Python3.9语法解析，本机实际运行版本仍为3.12.8，不声称已实际在3.9.25集群运行。2040保护文件与33研究core的全部SHA256只读复核一致，生产算法未变。最初一次兼容性核查因命令里编码名utf8-sig拼写错误退出；改正检查命令为utf-8-sig后完成，研究源码未改。
开发集尚未全部完成，仍不定正式v5.2/v6；独立confirmation90尚未执行，MSIC阶段未开始。任务卡3最终六份报告以全开发完成后的结论为准。

## Phase D最终验收与Phase E研究进入判断

historical_final27与development_final90全部完成，后者唯一summarize/analyse/contrasts/numerics顺序通过，450组行和89547fixed记录；含历史共117数据、585所选模型、116329fixed记录。controller24999已正常结束，未开始独立确认。新开发所选OLS全满秩、最坏condition17.19845、最大RSS重算相对差6.31e−16，无数值异常。

开发A0/A1/A2/A3/A4 Exact分别11/3/29/22/20（分母90）、平均PMSE3.018967/3.760809/2.996586/3.709077/3.715519；真shape入池32/59/60/90/81、真位置12/9/35/35/29。完整网格两个cap条件下Exact分别提高18/90与19/90，cap改变两个grid条件下反而下降8/90与7/90。A2对A0 PMSE23胜65平2负，A3为34胜14平42负、均差+0.690110（约+22.86%）。候选coverage增加不可直接宣布更优。

n400/SNR2、4两个新种子场景A2与A3均20/20恢复，对照A0只2/20，证实grid真实恢复收益；n100+n200六场景A3预测均更差，n200中强信号Exact从3/10、6/10降到0/10、2/10。A3独立成本估计577.98s（不是cold实测）约A0四倍、A2三倍，不能直接采用正式版。A3池已存在35真位置却只选22：13丢选包括n200中强8、n400弱信号5；弱信号欠选可能更利预测，不强迫Exact。

预声明门槛为“恢复或预测”可信收益，无5%等事后阈值。完整grid的20/20对2/20恢复已有训练机制解释和独立新增种子复现，非只有覆盖提升，正确性/容量/预算/复现也通过，因此固定预声明A3进入研究性质P4冻结库重评分。caps本身的负结果与A3预测恶化保留，不因此批准正式MSIC/算法，不临时换A2重写本轮P4协议。新模块/test将独立新增，旧core33/run/worker和mode alpha1不改；首选alpha2 scale0仍只是候选，先开发完整11规则，不按确认集挑参。

六份报告已有；COMPARISON包含完整开发总体/条件效果/九场景/过欠选/粗搜索/OLS范围和进入研究理由。后续完整refit、独立确认、版本决策与最终work_progress尚待P4结果，不声称已完成；原700full和三实证重训仍未执行。

## P4冻结库完成与候选锁定

独立msic_after_search.py与9有意义TDD tests通过（RSS由B复算新测试先FAIL后GREEN），模块冻结rawSHA88f4dbd82572d9fc8e5c0a988fc466d5707173cec02cd81a623f204f9a9df913；新增msic_refit.py与9tests通过，均Py3.9语法通过、旧core/run/worker没变。首个code工具auto审批deadline为请求超时并非安全拒绝，核读后仅授权新module最小写入，未改容差救补。

outputs_msic_after_search/development_final90/990全部完整，scoreplan beb5602479334d5233df89f4b401f48c8af4403e98dd43aa88ccd7bab2635b6b，614.950937s，纯冻结库重评分非新BCD。90原default的所有非runtime字段逐值相等、pool前后hash不变、训练RSS独立重算与test流一致。预声明alpha2/scale0 PMSE3.709077→3.142905、Exact22→28/90、over23→9、under28→37、q64.922→44.589；40个B改变，PMSE29胜50平11负。n100和n200六场景改善；n400弱信号1.41749→1.48076恶化，中强两个场景逐样本不变、各10/10恢复。

其他参数整体PMSE更低但增加欠选；alpha2scale1 PMSE2.496841/Exact27/90/under54/90，alpha3scale0 2.410373/Exact26/90/under63/90，SIC/BIC欠选亦多。当前选优先温和alpha2/scale0研究候选，具有最高开发Exact并保留强n400恢复；不是总体PMSE最小，也仍逊A0/A2总体预测，不宣布正式最优。

LOCK.json在原confirmation_final jobs/logs/fail全0时immutable写入，SHA c42f78f63a370b500310497e32ba7903ad37314721dc591fb445ed1d298e8d0b，双开发父plan/summarySHA/source/core身份及未见确认结果的选择理由明确。保留原confirmation_final90×A0–A4计划，不另建A3-only；此为先前内存备选的最终常规执行决定，未更改原种子/配置，保留独立A0/A2对照。

已授权预定0/40/80各一次锁规则fresh完整A3搜索（每task隔离1CPU/cache=None）到refit_checks_final。task0已PASS；40/80运行，三PASS后才授权确认90，当前确认未执行。默认只由同fresh池重放，没有额外三次重复默认拟合。确认新module/tests实现中，严格只原A3默认+LOCK两规则，不在确认集比较11规则或重新挑参。总限定测试目前118，待确认测试完成更新。

## 独立确认启动与P5交接（23时恢复）

三fresh0/40/80全部CLI exit0、NPZ/LOCK/source checks通过。各候选116/303/909，实际fresh秒15.584729/85.648605/452.964232。task40锁q60 Exact1/PMSE1.539517/RSS132.420907，对照q140 Exact0/PMSE3.609816/RSS35.613027；训练RSS增加与泛化改善属于不同q准则，非fixed BCD回归。默认只同fresh池重放，没重复拟合。

确认module13新tests与原两个wrapper各9合计31 green，overall131限定；moduleSHA16095d1b5124649f07b700638b37347e10073a1d2d4f68a4278bc47052b5e99e。LOCK早于确认最早log保存时间，确认仅默认+locked，额外90锁模型数值audit。root在三PASS后授权原90×A0–A4，20:19:55原controller开始，launcher52012/实际25216；session47094仅原owneragent可访问，root读取Unknownprocess不会kill/重复作业。agent额度暂停未影响实际Python，用户23:11继续后只读核对76完成/0fail，同原owner恢复，继续唯一4summaries+2rules。最新进度以实际完整统计为准，尚不能当90已确认。

P5新研究身份v6-msic-after-search-candidate独立保存，旧Task2 v6元数据原样不改；six-new-file MSIC patchSHA9d56aede75a153e80353ff33d9e4bac0fd8cf2cfae43e9cc58dd347398441f24、独立actual apply/6byte核验及3.9语法通过。source核心/新三wrapper/LOCK没有变。README增加完整P4与新环境LOCK/refit/确认入口，文档diffSHA8fff4744842d6b5ec8c2f22a487e01ad92e0ab04d1f3773b6a84748b7401fc78。首次README校验Git发现外层主repo而skip，byte断言FAIL后停止诊断；明确GIT_CEILING隔离，实际check/apply/reversecheck与byte/嵌入Py3.9脚本语法验证PASS，没有改模型/数据/数值容差救补。

最终COMPARISON与MSIC报告已有完整开发及fresh证据，确认与最终版本决策仍进行中；后续结束须同步Version.status与team identity SHA、保护2040哈希、54研究/指定test源码3.9 grammar、六报告状态及本work_progress最终结论。没有700full、实证重训、分支/提交/推送或ZIP。
