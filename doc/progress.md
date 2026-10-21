# LEMON 实施记录

## 2026-10-18 废弃文件清理
- 用户授权先清理各目录废弃文件；本轮只清理能证明可再生且未被跟踪/引用/保护的文件，复用已有清理逻辑，不改科学源码和旧清理审计。
- 已完成191文件/56空缓存目录清理，缓存字节数1113995（约1.062 MiB）；目录分布doc127、src24、empirical22、numerical6、scripts6、tests1、pytest5。新清理证据234460字节，不将缓存大小误写为整个项目净缩减量。
- 删除前后7874受保护文件与186对应源码哈希不变；相同提升权限下Git跟踪状态完全一致，仅原3个工作流文档M，未增加跟踪删除。PLAN/DELETION/VALIDATION三份记录保存在doc/cleanup_2026-10-18，旧清理报告/实验索引原字节保留。
- 权限与故障记录：初次沙箱扫描在.pytest_cache拒绝访问，先停删除；提升权限只读成功后恢复，无ACL变更。首次删除命令在任何删除前因PowerShell Split重载参数报错，修正路径分段并确认191目标完整后才执行；清理后首次Git比较因默认/提升权限上下文不同失败，停止后以原上下文复核完全一致。
- 默认沙箱Git将不可访问的隔离test_tools/taskcard4_test_tools显示为大量D；提升权限视图证明这些不是实际文件删除，已纠正初始“既有D状态”解释。依赖目录与环境始终未进入删除候选。

## 2026-10-18 任务卡10授权续跑
- 开发48/48已完整：原30字节复用+新18成功，追加批4432.141秒，成功case累计8801.487秒/CPU5498.656秒，单进程最大峰值394.199 MiB；固定12例/36模型外部测试流重放通过，最大预测差3.77e−15、最大指标差5.33e−15。
- 九门槛5项失败：总体A2/A1=0.991535>0.98、最坏cell A2/A0=2.836125>1.15、n400 Exact差−8.33pp<−5pp、n400 PMSE比1.054448>1.05、强SNR欠选差+18.75pp>+5pp；剩余24287.959秒大于确认储备10888.375秒，确认未执行属于科学拒绝而非超时。
- 新五份追加报告、六CSV及外审证据生成；暂定B，Q5不可识别且未实证解决，P1/P2沿用独立方案。最终验收通过：COMPLETE_RESEARCH_WITH_DECLARED_GATE_RESULT，开发48/确认0，23184候选身份、144内折缓存、六CSV、29链接、6820原文件及共享预算通过，追加1053文件交付索引完成；这是按开发失败停止规则完成交付，不是确认通过或正式发布。
- 续跑验收审查补齐“确认已启动、0例完成后中止”的终态区分，新增3项负控先失败再通过，独立验收工具共7/7通过；旧4项日志保留，不改科学runner。新报告由完整结果综合决定B/C，补配对胜平负、恢复和实际成本表；报告代码编译通过，尚未提前生成最终结论。
- 调度11项与独立验收工具4项测试通过，原6820文件核验通过，30例原字节复用；追加开发已于04:04:12 UTC开启共享28800秒时钟。task030–033训练身份预检4/4通过，未生成测试流或拟合；首批030/031约442秒完成，未再触发旧900秒超时。
- 文档读取的控制台GBK输出无法编码数学符号，已改显式UTF-8输出，原文件字节未变化；一次工作流补丁因锚点不符未写入，读取实际行后修正。
- 用户在明确获知30/48开发、0/48确认、900秒上限触发后，授权“继续完成未完成任务，权限允许”；该授权支持追加计算协议，不改旧失败历史。
- 设计审查：原30完整请求继续作为已见开发证据，仅重算原18未完成请求，不换seed/方法/门槛；确认仍用原未生成请求，需完整开发九门槛通过。
- 追加预算8h总墙钟跨续开发/确认、3600s单case、2worker；根据已完成n400最慢841.328s，18例约2.1h、48确认约3h并需储备，4h不足以稳妥覆盖，选8h而不无限延长。
- 冻结拟合模块保持原字节；新调度与报告物理隔离，原30副本保留旧plan/protocol哈希并用reuse manifest标记来源，不冒充新实验。

## 2026-10-18 任务卡10
- 最终诊断验收通过，状态COMPLETE_DIAGNOSIS_WITH_SCIENTIFIC_STAGE_STOPPED：34实例精确输入/源码/候选B/缓存/REE/F1/测试PMSE及六CSV重建通过，74相对链接、897文件交付索引、5922保护文件无变化；科学确认保持NOT_RUN，未发布、未重跑、未改正式算法。
- 最终只读审查修复验收器的execution目录过宽glob，限定计划中科学cohort；补精确30完成/4终止/14未启集合覆盖，未完成9门槛不计算。复现文档由主线接管完成，CLI参数按源码核对，原路径均拒覆写。
- 开发因task030达到冻结900秒单case期限停止（900.016秒，控制器整段1938.703秒）：30/48完整，031–033同伴中止、034–047未启动；不延长期限、不补算、不改协议。开发9门槛NOT_EVALUABLE_INCOMPLETE，无development_lock，确认及开发/确认外部固定范围重放NOT_RUN。
- n100完整24例PMSE均值A0/A1/A2=4.679961/3.231389/3.128412，改善主要发生于范围/来源限制，CV同库额外下降0.102977；n400只有6例弱信号，不作恢复非劣或总体推广结论。暂定B仅保留研究实现，不能当CV科学失败或独立确认通过。
- 34完整实例10842候选、102缓存折逐例外测前独立审计通过；九份报告、候选覆盖及计算计数CSV已形成，最终保护/链接/独立数值总验收中。
- 报告脚本首轮compile发现字符串跨行加法语法错误，编译前已停止、加括号修复通过，无科学生成；独立审查发现部分结果不应算完整门槛，报告器显式仅处理stopped终态。
- 一次辅助审计glob误纳execution目录的控制器complete标记，缺少模型审计文件而停止；改用明确pilot/pilot和formal/development范围只读汇总，原始结果不改。额外Get-CimInstance进程列表被Windows拒绝，停止该额外检查，无权限修改，不宣称其已验证。
- 4例Pilot全通过，耗时44–45.563秒(n100)、382.469–417.594秒(n400)，整线464.984秒，最大峰值412413952字节；逐例外测前独立审计及训练/测试身份保存齐全。
- 已依据成本冻结正式24cells×开发/确认各2重复、4worker单线程、7200s总/900s单case；理想吞吐与确认预留7007.355s，裕量小但无延长，只有开发9门槛与剩余预算双过才确认。FORMAL_PROTOCOL SHA3bfb7ae1dd508e3d2b79e18612a7d2d48269fe1409a6902a24a5df5a15ec2c18，96预注册请求已准备，尚未生成开发数据。
- 一次只读内存查询inline Python shell转义错误未执行，改为ASCII here-string读取标准内存API，不涉及科学代码或数据。
- 联合门禁46/46通过：原Cold16、CV13、独立审计算术4、控制器13；gates修复后只在fresh subprocess加载各研究测试，正式core前后32文件哈希与路径核对通过。
- 已冻结PILOT_PROTOCOL.json（SHA d63eee1fc539dac4858f8ba5dbde08dfd74e698cf7d4dd31acf8da4b6073de83），准备4例新seed family计划并以2worker/单线程启动；总上限1800s、每例900s、Job Object强停止，未启动开发/确认。
- 联合验收第一尝试在import阶段被正确拒绝：Cold核心先加载，test_cv仍要求configure_core freshprocess；科学0拟合/0生成。保留GATE_ATTEMPT_1.json，仅修gates.py将新模块各自隔离subprocess，不改算法、原测试或assertion。
- Windows原taskkill对自有子进程树被OS拒绝；控制器改用挂起创建→标准Job Object→恢复，13门禁实证超时终止自建孙代，无权限修改、安装或非任务PID访问；失败与成功日志保留。
- 设计预检拒绝原种子区间：pilot split(+100M)与development train数值冲突，development split与confirmation train冲突；未生成数据或冻结科学协议。停止实验启动后，将阶段base调整为3101810000/3501810000/3901810000，保留test+50M/split+100M及原唯一性门禁，要求完整重验，不放宽断言。
- 隔离CV模块13项门禁通过，含原A0位一致、fold heldout响应扰动不改该fold B/cache/预测、稀疏批量预测/中心化fixture与缓存篡改拒绝；独立审计4项算术/篡改负控通过。
- 5922历史文件保护基线已冻结；公共caps严格证明为334/566，主程序只固定规模来源，Q5同形状来源识别限制已写设计。
- Pilot前已冻结9项数字门槛意向GATE_INTENT.json；正式重复数只据Pilot成本确定，不据Pilot预测表现调参；当前控制器实现中，尚未科学Pilot/开发/确认。
- 已读取完整新任务卡与Task9验收，分派容量设计、CV实现和独立防泄漏审计；实现与实验等待容量定义核准后进行。
- 新建隔离目录并冻结历史保护；沿用既有解释器和单线程数值依赖，无新分支或核心修改。

## 2026-10-17 任务卡9
- 2026-10-18收尾：250归因、216完整库身份/四规则、12例旧预测Gate与1908候选PMSE全部完成；PureWarm原Cold16+并集9通过，复验同25不能累加为独立50。
- 交叉复审核对并修正主报告平均q四数，补加罚后最终q可反向增加的2个反例；区分Task7零生成与Task8限12例输入重建。
- 复现输出隔离已补齐，新目录不覆盖现有证据；PureWarm CLI新增输出路径复验通过。保留原red/green及历史失败，未改原断言、生产源码或旧实验。
- 后续结论：优先严格训练内CV并分离内折容量效应，支撑稳定性为第二方向；不执行下一卡或新版本发布。
- 已读取任务卡、阶段汇总及要求报告，分派250归因、候选/MSIC、Pure Warm三项隔离工作；Warm等待P0/P1稳定。
- 建立research/taskcard9与doc/taskcard9，制定12例无拟合重放上限及固定规模Warm范围；原报告、失败记录与正式版本保留。

## 2026-10-17 八卡阶段汇总
- 已交付work_progress/2026-10-17_eight_taskcards_stage_summary.md，共285行，串联八卡工作、关键量化结果、失败/中止处置、最新版本与后续证据需求，并复用已有250配对图。
- 三组只读来源核对完成；卡1–3及6–8草稿复审后修正覆盖分类，卡4–5明确157测试/80确认68截止和71留存/正式发布与后续700验收时态。数字以最终报告为准，不拼加重复实验或测试总数。
- 新报告58个本地相对链接全部存在，八张原任务卡链接齐全，UTF-8和限定git diff --check通过；本轮没有拟合、改算法或改旧实验/结果报告。一次补丁完整锚点不匹配未写入，改按实际整行重发后成功。
- 本轮仅编写阶段报告并更新doc工作流；不重跑实验，不修改科学产物、算法或既有结果报告。
- 已分派任务卡1–3、4–5、6–8只读核对，主线负责统一证据链、当前状态与work_progress交付。

## 2026-10-17 任务卡7结果接收
- 十份任务卡7报告及README/STATUS已更新为250/250接收完成；A/C仅更新过时阶段说明，B/D补齐数值拆分和最终八问，并关联任务卡8已有未过推广门槛的独立研究，不启动新算法。
- 报告代理批量替换首个断言遇PowerShell中文管道编码不符，实际未写目标；主线接管，以apply_patch创建UTF-8报告脚本后直接执行，12目标一次写入通过，旧原字节均在快照保留。
- Debug→依赖显式化→精简完成：审计根路径改显式参数，重放后的序列化结果与原审计逐字段相同；首次直接比较Python整数dict键与JSON字符串键导致工具层假不等，按JSON往返规范比较后通过，未改科学结果。
- 两审计完成PASS：250回传记录/5 Gate/750原NPZ/CSV与96源码及依赖完整匹配；750模型的独立数值检查通过。主线实际分析250配对、150种子族，未拟合或再次merge。
- 新三面板图纳入两段各250配对点和全部750模型恢复计数，全部极值可见；源码21PASS/0WARN/0FAIL、1.5pt对齐、6pt字号、collision0/0和逐面板目视通过，PDF/SVG/TIFF/PNG600dpi完整。
- 用户授权核验回传结果并完成遗漏；建立doc/taskcard7_completion，5664原科学/源码/新cluster产物保护及15份目标文档原字节快照通过，旧索引不改。
- 分派两项只读审查；已有原生CSV和integrity，不调用已有输出拒覆写的merge，不重新拟合。
- 一次PowerShell管道中文路径被编码成问号，改为按已知文件glob定位；一次临时读取命令括号语法错误，修正后只读完成。未改科学文件。
- 审计初步摘要出现n100/SNR.5标签互换，已暂停结论写入，直接以plan的n_train/snr/rho核对并纠正；原CSV及结果未变。

## 2026-10-16 任务卡8启动
- 主D最终216/216完成，最后确认完成UTC2026-10-16 16:28:19，早于冻结截止16:46:46；开发/确认均未过，保留原MSIC，不以平均PMSE改善放行。
- 3253保护文件、全部31968候选B和31752固定BCD路径/选择/门槛独立元数据核验通过；预选12例输入/共享测试流/候选OLS与所选预测独立重建通过，正式源码/incoming/历史不变。
- 12份报告、训练内全144规模与主D全216配对两图完成；PDF/SVG/TIFF/600dpi PNG、源码/对齐/6pt字体/碰撞与最终目视均通过，重叠的选择标记由表中三模形状明确区分。
- 2026-10-16 15:43:06 UTC开发108全部完成并冻结lock；平均PMSE改善4.89%，最坏场景恶化56.70%，未通过预定开发门槛。固定双规则确认按原协议继续，不改规则/数据/预算。
- 新只读完整性脚本完成中途验证：保护3253文件及当时140已完成数据的全部元数据和SHA通过；此记录明确final=false，不充作最终验收。
- A完成、B/C/E按失败门禁停止；D2完成212.27秒，7项测试及432候选/12预处理/12所选模型独立验收。主D仍按冻结216数据与2小时上限运行。
- 主线已接管协作额度结束后的收尾；既有主D控制器与4个worker继续运行，未重启或重复拟合。跨协作作用域不能通过旧session号接管stdout，改为只读检查已有完整记录及终态，不改控制器。
- 合并新研究工具17项测试通过（6模拟协议、7内选阶、4审计工具），Warm科学失败不计为工具测试失败；Cold原样31检查已另存。内训练图全144候选，绘图轴限改为真实全数据范围以避免截点，未改科学数据。
- 初始snapshot因两份已更新launcher文档的旧index SHA不符停止；逐项审计确认仅部署文档漂移，历史数值/源码无差异。已保留当前改动并建立独立3253文件保护基线，旧index未改。
- 用户批准D开发/确认各108数据及7200秒上限，共享开始/截止UTC已在首次实验前冻结于taskcard8/COMPUTE_BUDGET.json。
- 分派Warm审计/正确性、独立冷v5.2模拟与训练内部选阶可行性；执行任务卡已授权隔离研究，不动正式算法。
- 初始根目录无四类工作流文件；保留既有未提交任务卡7交付及用户新增launcher测试，先做保护指纹核验。
- 已读任务卡8/AGENTS及文件规划、设计梳理/TDD要求；任务卡规定边界与用户批准预算构成本轮冻结实验设计依据，不另开分支或worktree。
- 定点检索时一个猜测旧路径不存在，已按实际目录定位，不是实验输入缺失；历史test_tools权限警告不作为本轮数据Gate放宽依据。

## 2026-10-10 任务卡2执行中

- 任务卡移动至empirical_analysis/任务卡/，读取任务卡2获得核心修复/版本快照/团队patch授权；当前main分支、HEAD d4e28969ac480ba6efc01bcc985a53d179e7ed31；不创建分支或提交远端。
- 原v5全32源码已保存doc/algorithm_versions/v5-original，历史1738结果文件先取保护SHA256。只改core bcd.py，GroupBESS/MSIC/数据生成/对照模型保持不变。
- 新测试主线先复现4行为失败及10缺字段子测试错误；首轮字段误落adaptive dataclass导致构造错误，纠正后GREEN。独立审查建议初始OLS有限性检查，新增溢出test先RED，补guard后共15个原/新增pytest通过（1.41秒）。
- 修复同时循环mode去重、暖启动当前groups、完整OLS候选统一损失比较、严格改善接受、seen支持阻断重复，最优访问状态元数据一致返回。32源码Python3.9语法验收通过，原/新sourcehash仅bcd.py不同。
- 无pytest现有依赖，pip --target仅生成独立test_tools；未改venv或集群。pip显示现有langchain/streamlit的packaging约束警告，独立工具仅用于core测试进程，不加载那些应用，也不全局安装。
- 统计worker先测试后实现中；LEMON隔离原v5重放已复现q80/RSS1.234092/RMSE0.670771及非单调，修复版重放正在进行。尚未形成v6。

- 扩展原实证/模拟回归26项发现历史seed2026 fixed恢复1项失败，已停止暖启动草案研究和full预检；保留doc/algorithm_versions/v5.1-warm-only-rejected及outputs_msic_ablation/ABORTED.json，不改原恢复断言。
- 独立审查及诊断确认采用任务卡3.2B允许的方案：保留v5 cold探索，探索和正式best状态分离；每mode完整OLS，只有best改进进入loss_history，whole sweep探索状态重复才停止，不因一个sweep暂时无best改善截断路径。
- 新“跨暂时变差探索后达到更好状态”测试先RED，最终方案15核心+26原实证/模拟+12研究测试全部通过，恢复原seed2026真实支持。数学Group-BESS/MSIC未改；仅bcd源码变化。
- 最终P0源hash c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737，最终patch d2e6cac68ad7e37ec131cf2ff3d52eec4e7ec4498d30cea484d513fe2b7b37ee已在原版副本验证应用。MSIC研究用新目录outputs_msic_ablation_v51_final从头冻结，不复用草案数据。

## 2026-10-09 EEG增量预测任务

- 首次读取任务卡为空，用户保存后获得完整授权规格；按指定独立模块实现，不动核心/旧结果。
- 并行委派只读数据审计与loader、固定lambda模型适配、聚类bootstrap和图表报告；主线负责统一nested CV/断点状态。
- 沙箱执行器有setup refresh启动错误，获准后使用沙箱外执行。无任何依赖安装。
- 预处理/划分/完整性测试先因模块缺失RED，再实现GREEN：7项通过。数据审计和模型测试尚在进行，未宣称真实实验完成。

- 数据来源gate已通过：145人、X(145,61,16,5)/C(145,8)，TMT原表、人口学原表、manifest、全体与个体原张量以及通道/频带顺序均逐项相符。最小年龄组4人，45个inner train及15个outer train全部满秩。
- 代理工具审批发生长时间pending；中断后由主线代落地data/metrics/reporting，避免继续等待。新增25项unittest全部通过（0.373秒）；曾出现测试消息大小写匹配和缺少__init__的discover错误，修正后完整GREEN。
- 真实smoke为单个完整外折、五方法，5个fit全部成功（73.73秒），存独立outputs_eeg_incremental_smoke；不是正式重复CV结论。
- 正式3×5×5=75个外折模型已启动，BLAS/OMP等单线程，产物逐模型持久化；模型代码冻结，绘图代码独立指纹。截止此记录正式实验尚在运行。

## 2026-10-10 EEG增量预测完成

- 正式75fit/2175OOF全部完成，耗时1208.85秒；主线从B及preprocessor重算全部预测差0，独立covariate OLS差9.77e-15，435outer划分、2700候选评价、135inner轨迹通过对齐/隔离验收。
- 四个EEG模型均0/3改善基线，mean RMSE cov0.317485、ABESS0.371667、Vec0.321263、GLasso0.330680、Ridge0.319677；条件OOF受试者bootstrap区间全部总汇负收益。
- ABESS repeat2/fold5 q80，train/test RMSE0.103144/0.670771，其他折q1–6；该折贡献总SSE恶化56.93%，记录为容量/搜索稳定性诊断，不剔除。
- 788个受保护源码/旧输出哈希均未变化。五类实际图PNG/PDF/SVG完成；图版字体/碰撞/对齐及逐图视觉检查通过，源码18PASS/3报告格式WARN/0FAIL。
- 全部实现/结果/边界/后续已写入 work_progress/2026-10-10_LEMON_EEG_incremental_value.md；任务完成，不再追加新实验。

## 2026-09-14
- 已读取 brainstorming、test-driven-development、planning-with-files-zh；依据用户本轮对既有独立目录方案的实施授权继续。
- 确认既有用户代码变更，未改动核心与数值模拟文件。
- 启动数据位置核查；完整数据路径待回复。
- 尚未执行正式 EEG 批量处理或交叉验证。
- 数据模块先运行 6 个测试均因功能缺失失败，随后实现并全部通过；覆盖配置目录解析、异常TMT保留、重复/非正值拒绝、清单尺寸校验、按ID对齐及协变量Gate。
- EEG/事件和建模模块由独立子任务实施；各自负责目标文件及测试，不触碰核心代码。
- 全部9阶段已通过5名人工BrainVision受试者的端到端集成测试（11.176秒），包含真实3模型拟合、15条OOF记录及报告；所有合成输入/输出均在临时目录。
- 现有核心自适应7项回归测试全部通过。
- 用户确认完整数据稍后下载，新增前150名可续传下载器及离线计划功能；本轮不启动真实大体积下载。
- 下载器28项测试验证本地HTTP Range、服务器忽略Range、传输中断、校验、跳过和路径保护；未连接真实下载服务。
- 已生成 empirical_analysis/lemon/outputs_download/plan.json：150人、1053文件、96,308,960,326字节；排除sub-010203（标准events.tsv缺失）。
- 最终统一测试命令：`.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_lemon_*.py'`（PYTHONPATH=src），74项通过，用时16.418秒。
- 新增BIDS派生事件表共存回归验证；实际4完整subject元数据均通过事件逐条验证，未读其波形。
- 真实路径配置已通过JSON序列化，pilot阈值与artifact/config变动保护均有先失败再修复记录。
- 文档一次补丁上下文匹配失败后以完整文本行重新应用，未产生部分业务改动。

## 2026-09-14 实际下载任务
- 用户指定下载目标 `D:\数据集\LEMON原始EEG`，要求执行既有150人计划及共享元数据，禁止启动分析阶段。
- 已读取适用AGENTS、下载器、README及既有规划记录；下载器具备稳定bytes_url、.part续传、清单校验和正式文件失败不覆盖行为。
- 首次只读计划核对因PowerShell默认管道编码损失中文路径失败；未访问网络或改动数据，随后显式设UTF-8编码。
- UTF-8核对成功：manifest重建计划与现有plan完全相同，150人、1053文件，3个共享metadata；目标盘空间充分。
- 目标已有1个增长中的.part，无正式文件；已通过获准的只读进程查询确认PID10172正在执行本项目下载器，参数为同一manifest、D盘目标、limit150、既有plan及--download，PID49004是其venv启动器。
- 不重复启动下载，不终止或接管用户已有进程；设置仅等待其结束后读取日志与文件状态的汇总任务，不写原始数据。
- 首次隐藏汇总PID2272因Wait-Process访问权限退出；获准在沙箱外重新启动PID48868，复查进程存活且download_monitor_error.log为空。
- 截至20:26:16，下载PID10172与venv启动器PID49004仍存活，首个.part增长到477,102,080字节；暂无完成/失败日志记录，成功0、跳过0、失败0、未完成1053，participants.tsv/json尚未取得。
- 任务仍在后台下载；PID48868结束等待后会读取download_log.json，生成同目录download_summary.json，含成功/跳过/失败/未完成、失败路径与原因、共享元数据校验状态、.part、缺失/尺寸异常列表。不会自动重试失败或覆写原始文件，不会启动分析。

## 2026-09-14 真实数据试跑
- 用户授权事件验证、单人预处理、5人张量及时间/峰值内存测量。
- 只读核对：仓库sub-010002/010003/010004/010006完整，sub-010005仍缺events.tsv、events.json、channels.tsv、eeg.json；D盘仅前2人完整。
- 下载日志实际已停止：14文件成功，sub-010004/eeg/sub-010004_task-resting_eeg.eeg因TimeoutError失败，.part保留；自动汇总已生成，complete=false。
- 已报告缺件并暂停依赖阶段；按既有清单授权补齐4个小型元数据后执行，不降低Gate。环境已有psutil 7.0.0可测量Windows进程峰值工作集。
- 31个已有真实文件全量清单校验通过，耗时3.471秒；获准调用既有下载器补齐4个metadata共26,704字节到D盘，日志为outputs_download/smoke_metadata_download_log.json，4项全部downloaded；复制至仓库缺失位置后再次校验，总35文件通过，无正式文件被覆盖。
- 创建独立outputs_real_smoke_20260914_5subjects配置、5人选择及资源测量入口，沿用原始参数；输入使用稳定C盘root，不读取D盘下载碎片，未改主配置和核心代码。
- 按inventory→tmt→events→smoke→tensors运行5个独立Python进程，全部通过；每阶段保存elapsed、CPU时间、Windows峰值工作集/提交内存及实际返回值。
- 真实结果X=(5,61,16,5)，5人纳入、0排除；每人498事件、8个EC段，候选16/16/16/17/16，各选16个25秒EC窗口，数组全部有限、band方差非零，无VEOG；smoke两遍一致且与批量第一人完全相同。
- 事件0.381738秒/82.933594 MiB，单人smoke16.960579秒/248.960938 MiB（两遍），5人张量37.718937秒/250.839844 MiB；用户要求三阶段合计55.061254秒，含inventory/TMT合计55.493547秒。
- REAL_SMOKE_REPORT.md、resource_summary.csv、validation.json、逐人窗口与QC均已保存；独立审查再次确认阶段产物指纹、5人全部纳入和资源口径正确。本次仅真实预处理到张量，未进入dataset/CV/最终模型。

## 2026-09-14 下载器重试增强
- 已读取brainstorming和test-driven-development，复核AGENTS、现有下载器和28项下载测试；用户明确授权测试先行和最小范围实现。
- 测试子任务仅修改tests/test_lemon_download.py，主线负责download.py及必要下载说明；不启动大批量下载，不修改核心算法。
- 定义默认180秒、最多额外5次重试、2/4/8/16/32秒退避；严格区分网络暂时故障与校验/协议/本地I/O失败。
- 可选文件系统类型查询Get-Volume被沙箱拒绝；不依赖该可选信息，设计直接使用Windows原生rename不覆盖语义，POSIX使用原子link，避免引入特定目标盘文件系统前提。
- 首批测试RED：31项中5个明确的接口断言失败，无TypeError冒充功能RED；随后扩充到48项，定位校验异常缺具体文件路径及IncompleteRead.partial丢失13字节两处功能缺陷。
- 实现默认180秒、0–5次额外重试与指数等待，只有网络opener/read错误进入暂时故障分类；本地写盘与校验错误立即传播，累计已传输字节并保留完整retry_history。
- 显式保存IncompleteRead.partial后使用新的.part长度继续Range；将校验错误定位到item.path，转正改为原子不覆盖操作；通过注入竞争正式文件证明其内容不被覆盖。
- 最终48项下载器测试全部通过，耗时1.267秒；全部localhost或注入故障，等待函数注入计数，未连接真实数据源或启动批量下载。
- README与执行文档下载章节已更新180秒、5次额外重试、退避、可重试边界、进度/校验状态及D盘续传命令；CLI --help与文档参数一致，最后进行独立安全逻辑审查。
- 独立只读审查完成，无阻断问题；明确下载日志按文件成功/最终失败保存尝试历史，当前等待时强制终止会保留.part、控制台仍有已显示重试状态，不宣称每次重试即时持久化。
- Debug→重试分类/原子转正审查→提取共享写块逻辑已完成，48项下载器测试通过；本次无真实数据下载、无分析阶段执行，交付文档中的D盘续传命令供用户自行启动。

## 2026-09-17 年龄段接入
- 已确认实施授权，采用测试驱动开发和文件规划技能；用户同意相邻年龄组合并。
- pipeline新增协变量列名/编码持久化测试先失败（缺covariate_design.json），开始实现；dataset分类支持独立代理TDD。
- dataset分类支持先RED后GREEN13项；pipeline列名保存RED转GREEN。全部LEMON测试102项通过26.079秒，包括原下载器及合成完整流程；增加多列年龄设计折内残差化和缺类别秩不足阻断测试。
- 官方227人派生participants.tsv/json已写入outputs_demographics，原始年龄段与性别1/2保留，列映射和原csv/info SHA256写入JSON。
- 已校验旧146人预处理全部阶段产物指纹；真实只读接入得到145人、C=(145,8)，sub-010092仅因原TMT值无效排除。五折116训练/29测试、全部rank8，折内拟合及留出变换全有限；alignment_check.json明确不代表dataset阶段或模型完成。
- 主配置设新outputs_lemon_adjusted，后续必须从inventory运行，不篡改旧身份或原EEG；本轮没有真实模型拟合或批量EEG重算。
- 独立只读审查无阻断：官方227人字段和源hash一致，固定年龄参照/显式合并/缺失未知处理/旧numeric兼容/满秩保护均通过；核心算法未改。
- README、执行文档、输出说明已同步七类别/sex编码、源hash、145人预检查与新输出六阶段入口；交付完成，真实建模由用户后续命令启动。

## 2026-10-04 已知 s 集群实验与文献建议

- 已读取相关技能与仓库规则，派发独立推理、文献核验与运行链审计；异步询问重复次数和集群调度器。
- 现有入口/seed/sharding/依赖审计已完成；下一步只在独立 known_s_cluster 目录实现集群任务与汇总，旧源码和输出保持原状。
- 首次追加 task_plan 的锚点误用 progress 末行导致验证失败、未写入；重新读取正确末行后修正，未使用整文件覆盖。
- 可选 sklearn 版本探测显示未安装；当前模拟方法不依赖它，不安装无关包，后续基线仅提出建议。
- planning-with-files-zh 用于本轮 doc/ 记录，遵守仓库指定位置；首次新目录文件写入遇到父目录创建失败，确认目标仍在工作区且目录存在后改用仓库相对路径写入成功，未提升权限或改非目标路径。
- 新集群入口5项合约测试先观察到缺入口的 assertion FAIL；实现中。独立CLI将不依赖原runner/plots，精确冻结14场景与seed。
- 推理建议增加弱模态切片实验，文献核验推荐优先Vector-ABESS/ElasticNet；新基线仅建议，未接入本次集群运行。
- 额外只读部署审阅代理的spawn及已有代理followup均被线程数量限制拒绝；不继续重试新增代理，主线自行审阅，既有推理/文献/实现任务不受影响。
- Slurm脚本已通过本地Git Bash的bash -n语法检查；用户集群尚未执行。
- 历史seed验收先FAIL：独立读v3/v5确认所有非baseline组实际offset10000，暂停错误映射并改新profiles为历史实际seed，旧文件未改；测试fixture改为包内自包含。
- 原8项GREEN后，新增状态提交窗口/计划外NPZ与lock两项测试先RED；将持锁未完整提交识别为运行中、无锁残缺仍失败，并拒计划外文件；最终10项全部GREEN（7.772秒）。
- 独立CLI quick14任务/42结果、status/严格resume/merge通过；可选5方法quick task0通过；full仅baseline task0三方法，28项非耗时指标各最大差2.8422e-14，v3/v5各140场景seed与配置对齐。
- 解压最小上传包后确认runner/core从包内导入，真实CLI再次14任务/42结果完成；沙箱初次创建results被拒，经命令审批后独立验证完成；包内10项测试审批后通过（19.181秒）。
- 初始沙箱包内单元测试在独占提交项挂起；进程诊断也需审批，Stop-Process返回NullReference错误，最终经审批用taskkill仅结束已核实的测试进程树26536/19208；原session退出1，未将其记为通过，不再存在该挂起任务。
- 55个受保护原文件SHA256全部保持一致；本轮仅目标目录、work_progress新记录、doc追加记录变化。未提交远程作业、未安装新基线依赖、未修改CP/Tucker或历史指标。

## 2026-10-05 已知s增量部署说明

- 按用户最新要求，将README改为四个生产文件增量上传及三个完整sh代码块；不制作新压缩包，旧整包说明在work_progress明确标记为历史，由第十一节替代。
- 发现原Slurm脚本按复制后的BASH_SOURCE推算项目路径会失效；模拟spool路径的测试先RED，再改submit_slurm.sh用SLURM_SUBMIT_DIR，提交命令先cd项目根。仅目标sh、其测试和文档变化，核心/run/profiles未改。
- Windows首轮测试选中不可执行Bash入口报WinError1920；测试明确选择本机Git Bash后成功复现真实目录错误，修复后11项通过（20.313秒）。无真实集群提交。
- 清理须用户确认；旧ZIP及缓存仅列候选，验证JSON/原始结果/源码哈希证据保留，未执行删除。

## 2026-10-06 已知s集群单脚本

- 用户提供实际集群路径、Slurm资源、Python3.9.25/NumPy2.0.2/SciPy1.13.1并禁止环境更新；用户Yes确认根目录单脚本、prepare/sbatch/merge三命令设计。
- TDD先暴露入口硬限制导致本地/公共配置测试报错及根脚本缺失；run环境记录改用公开np.show_config接口，新增根目录run_known_s_full.sh，未改核心算法或随机种子。
- README文首改成三条直接命令及三项生产必传文件；旧拆分说明标为本次不要使用。根脚本版本检查不执行安装，数组700任务最多64并发，每任务1CPU。
- 完整known_s_cluster测试14项通过（30.038秒），两份sh通过bash -n，根脚本确认LF；Python3.9/NumPy2仅完成静态/API兼容核验，等待用户集群实际运行。

## 2026-10-10 任务卡2执行
- 首个warm-only修复草案原恢复断言失败，停止并隔离；最终保留cold探索与best接受分离，53相关测试通过。
- 54最终版本任务完成，333行评价，27数据配对、3051候选审计通过；6/27最终B改变，默认PMSE均差+0.302069，算法正确不保证预测改善。
- MSIC alpha2收益与欠选并存；n100caps排真形状、n400候选网格漏s2=3；不改默认、不批准v6。
- patch原版副本可应用，38文件Python3.9语法、sh LF/bash -n通过；1738旧结果和原32源码完整，核心只bcd修改。
- 文档写入apply_patch审批超时后使用workspace已授权exec；发现PowerShell ASCII管道损坏一次追加，显式UTF8回读纠正，数据/源码未受影响。
- 700任务full仍需集群执行，根run_bcd_fix_full.sh提供三阶段，原环境不升级，最小重传bcd.py与sh；work_progress已详细归档。
- 最终full首任务四方法已完成且严格resume跳过；1/700通过，不标完整正式full。q80修复后OLS rank80/80、condition219.234、normal-equation residual3.44e-13，RSS重算差1.78e-15，有限且求解一致。独立源码复查无阻断。

## 2026-10-10 任务卡3开始
- 根任务卡已定位，委派原机制审计；沿用brainstorming的方案/隔离检查、TDD、文件规划，任务卡本身明确批准A0–A3实验设计，既有自动审批授权持续，不重复要求确认；架构/生产默认不变。
- 核验当前32核心指纹等于v5.1，再复制两份独立版本源码，记录受保护源码/旧实验指纹。
- 位置多初始化5契约测试先全部FAIL（独立模块缺失），再实现marginal+两固定种子random，以训练RSS选最优，不接受truth或test输入。
- 核心13新+15原相关=28通过；运行器15、位置模块5、稳定性运行器5通过，共当前53限定测试。代码冻结后历史27计划建立，原MSIC只1rule/group；budget在历史训练初始化预检后预先定2000。
- 隔离v52旧LEMON12+numerical14兼容回归通过55.56秒，冻结source未改；累计79限定tests。平衡比较/真形状非q值/共享缓存成本3分析测试先FAIL，再实现。历史task0 A0 B及7评价逐值等于原v51，n400 task18仍预检中。
- 开发task33触发worker.audit_pool“Accepted training loss increased”；立即停历史controller及其已核实子进程，开发已自然失败退出，冻结其验证器代码与STOPPED记录；MSIC不推进。源码核心未改，保留原结果待独立审查fixed/adaptive不变量。
- 两独立agent实证确认dev033是验证器把adaptive变q路径误当fixed；q4→2时RSS225.2495→232.4915但MSIC54.9126→53.3367，125fixed全部通过；最小修正仅worker/test_run，core33与run.py字节不改。新8测试先FAIL再GREEN，runner23通过，总限定90。
- 旧历史16/开发35输出及稳定性2任务保留STOPPED；不复制、修补或重标。新historical_final27/development_final90/confirmation_final90同预声明seed重新plan。先117历史+开发任务合并8个显式独立worker调度，确认90仍未执行，MSIC仍不推进。
- 继续执行时三个子agent因账号usage limit退出；其已写新模块可用，主线接管后历史27官方summary/analysis完成，新增contrast6与numerics4测试通过，总限定100。历史135所选设计全部满秩、OLS/RSS一致，无数值异常；开发仍在最终90任务运行中，确认集未执行。

## 2026-10-11 任务卡3最终收尾
- 历史27、开发90、独立确认90均完整，JSON/NPZ逐队列数量一致；五组1035所选模型数值审计通过。上节执行中记录保留为历史状态，以本节为最终状态。
- A0/A2独立确认Exact为6/90与26/90，PMSE为3.118404313与3.082733325；n400/SNR2、4均由A0的0/10提高至A2的10/10。收益仍限于本轮真形状及场景，不推广为普遍结论。
- A3开发90库11规则990评价、预定0/40/80三个完整fresh验证与独立确认90库两规则180评价完成；LOCK及池指纹不变、默认90逐值重放、90锁模型满秩且max condition17.332759。
- 锁定alpha2/scale0确认Exact27/90、PMSE3.117813223；相对A3默认33胜/49平/8负，B改变41/90，但PMSE仍高于A2原MSIC；不据此批准正式v6或直接迁移A2参数。
- 既有限定测试为100基础+31包装器=131通过；本次收尾未重跑测试。Python3.9仅静态验证，实际集群环境及全仓库测试未完成。现场重新核对2040受保护文件哈希，差异0。
- 七处状态文档/版本身份统一为研究完成且未批准；算法、旧结果及patch未修改，未创建分支或提交推送。正式生产保持v5.1及原MSIC，700正式模拟、三个实证重训及扩展真形状/相关性验证仍未执行。
## 2026-10-11 任务卡4开始

- 用户确认旧test_tools权限/git提示正常并授权继续；不修改该旧依赖目录。
- 任务卡3实验真实已完整，七处过时收尾文档已校正；131项为先前测试记录，尚不算本轮重跑。
- 启用现成任务卡设计、TDD与doc文件规划；独立agent分别审核历史A2证据、收尾和最小研究runner，不替换生产。
- 新实验协议在查看新结果前冻结，评价门槛/种子/预算不可按结果调整；原MSIC和原Group-BESS/OLS/DGP只读。
## 2026-10-11 任务卡4暂停：本地测试前置缺失

- 已完成任务卡3七处文档收尾、A2源码及90确认结果只读审计，A2_EXISTING_EVIDENCE.md已交付。
- 新100数据协议已冻结，SHA256=2300285beca35adf92436e3774c0bbd156d981ac56dcb87bfc6eaba1c665cc00；尚未启动真实新增拟合。
- 本轮核心候选13项及fixed BCD回归10项通过；同命令的test_tensor_bcd因ModuleNotFoundError: pytest加载失败，整条命令退出1，不称全部通过。
- 按AGENTS前置条件缺失立即停止规则暂停，已中断两个新runner/analysis开发agent；它们可能已留下未完成测试/源码，后续需核对后续跑，不标验收通过。
- 未安装依赖、未改生产或集群环境；需要用户允许在独立测试目录准备pytest后再继续相关回归与新实验。
## 2026-10-11 任务卡4恢复

- 用户明确允许独立pytest准备后继续；安装固定pytest8.3.5、packaging24.2、pluggy1.5.0、iniconfig2.1.0、colorama0.4.6至taskcard4_test_tools，未更新venv/集群。
- 首次pip网络受沙箱限制，按现有授权升级工具执行权限后安装完成；pytest读取也使用同权限，不绕过数据/算法验收。
- 84项现有candidate/stability/MSIC研究unittest通过。扩大到旧test_phase_c的收集因sklearn缺失失败，该旧基线与本轮改动无关，明确不计入本轮通过范围；随后仅运行BCD/模拟相关pytest。
- 新runner测试已确认模块缺失的RED，物理隔离目录已准备；主配对实验仍未启动，冻结协议保持原哈希。
## 2026-10-11 任务卡4冻结并运行

- 本轮84既有研究unittest、29 BCD/数值模拟pytest、22新增runner/analysis unittest，共135项相关限定测试通过；未验证需要sklearn的旧phase_c与全仓库，集群未实际运行。
- 新runner8测试、分析14测试先RED再GREEN，真实小DGP shared/cold及篡改拒绝已通过。4文件Python3.9语法通过，仅静态证据。
- 正式新plan_id=55cd1a254efe2425d650e755950388efa2ff8746aa0ebd83f9cfffdb50cf7cd3，raw协议SHA保持2300285beca35adf92436e3774c0bbd156d981ac56dcb87bfc6eaba1c665cc00，源码已冻结不再更改。
- 已启动100配对拟合（8显式worker/单线程BLAS）及仅所选模型的独立OLS审计；共用不可覆盖控制器时间起点，尚不能按部分完成结果做版本批准。
- 已补齐最终同步/全量准备文档的当前生产身份；未执行700任务、任何实证重训或生产替换。A2受控MSIC工具仅准备，不在搜索决定前运行。
### 任务卡4运行中恢复信息

- 活跃exec session：5786为100配对控制器，13445为逐任务所选OLS审计；只轮询这些已启动进程，禁止再建计划或重复启动写入者。
- 最近核验75/100配对及75/100数值任务完整，150所选模型通过，failures为空；尚未做最终统计或正式批准。
- 所选OLS诊断会逐任务写入numerics/task_XXX.json，100完整后才写numerics_diagnostics.json、200行CSV和绑定plan的regression_checks.json。
- main/cold共用CONTROLLER_STARTED.json起点7200秒，cold须100拟合结束后调用，16个预定单组worker顺序运行，输出cold/不可覆盖。
- 全部后调用a2_final_validation.run summarize；现有A2 MSIC工具7测试本轮重跑通过，总已验证相关测试142，但真实MSIC重评分只有A2搜索验收通过才运行。
### 任务卡4新增实验拟合完成

- 100组配对和200个所选OLS全部完成并通过，秩不足0、所选最大condition40.4306026283234；诊断仅涵盖所选模型，不推广到全部候选矩阵。
- controller5786及numerics13445均退出0；冷启动控制器session49312已启动，8对16单组拟合沿用原控制器起点和时间预算。
- 原协议task_card中文显示字段因默认PowerShell编码管道成问号，字段不被运行器消费；原协议/plan/科学配置/结果不改，仅新增来源路径与任务卡SHA核验的PROTOCOL_METADATA_NOTE.json并在报告说明，避免误导。
### 任务卡4搜索验收与Phase C

- 100配对、16cold及200OLS均完成；analysis.json/paired_metrics.csv/scenario_metrics.csv不可覆盖产物已写入，全部门禁pass。
- 主90 A0/A2 Exact15/90与36/90，PMSE2.5000107245与2.4742185167；配对分层bootstrap PMSE ratio95% [.9875435207,.9916895180]。worst scenario mean ratio1.0，无场景均值恶化。
- 两种shape均恢复改善；rho.6/n200压力10组A0/A2模型一致（两网格在该caps内本就一致），不据此断言新增网格在强相关n400已验证。
- 估计standalone成本比1.1487930，16单组cold合计实测比1.1036801，cache_hit0且模型与shared逐值一致。
- 搜索推荐D2、固定A2后，已启动旧development90的A2两MSIC规则受控重评分；新100未用于重复选新MSIC，LEMON外层结果未被使用。
- 最小D2候选32源码仅bcd私有整数网格不同；6静态/同数据实际桥接测试通过，source与API均核对；生产仍v5.1，尚未正式发布或启动全量。

## 2026-10-11 任务卡4继续核验

- 用户明确允许读取并继续完成；隔离pytest目录普通读取仍受沙箱限制，首次提权审核超时，获准重试只读成功；未修改ACL或安装环境。
- A/B的100配对、16冷启动及200所选OLS已完成；Phase C开发90库两规则筛选通过，80新数据确认协议已冻结，尚未作为完成结果报告。
- 最终D2候选BCD raw SHA256为db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7，规范LF为eb4f01626e0108f4cf87a604835aab7fa99866f484abaea23ce54b8911f063d0；VERSION/MANIFEST与文件原始字节一致，没有冻结源漂移。
- 本轮首次研究回归命令误含不存在的numerical_simulation.support_search_stability.test_run，106实际测试通过但整条命令退出1；更正命令后重新核验，不将失败命令记为整套通过。
- 正在运行最终候选的BCD/数值模拟pytest；本轮不替换生产、不运行700正式任务或三实证重训，最终发布批准仍独立保留。
- 更正后的106项研究unittest通过（17.813秒）；冻结D2来源的29项BCD/数值模拟pytest通过（80.75秒）；6项候选静态/独立进程桥接通过（24.383秒），另9项确认和7项MSIC决策unittest通过，共157项限定测试。
- 重新逐项读取2040个受保护文件并核对SHA256，差异0；未更改生产算法或旧结果。
- 80独立确认plan_id=1857d78e504ede09a0e305d6cffa338007bea87b0a1f84ac80a2ee9a3843c828，控制器PID13372，起点2026-10-11T08:25:25.813984Z；协作代理因用量限制退出后主线只读确认原控制器及8子进程仍在运行，禁止重复启动run。最新61/80评价完整，failures为空，尚不据部分结果决策。
- FINAL_TEAM_SYNC和FULL_EXPERIMENT_PREPARATION已纠正过期D2包描述、现有解释器命令和批准后入口；最终MSIC与发布状态将在完整确认后统一。

### 任务卡4最终决策及交付

- 独立MSIC确认北京时间17:25:25触发Shared confirmation budget exceeded，STOPPED.json明确resume_allowed=false；停止时68/80评价完整，后续finally等待期间留存至71/80，不能计为预算内完整验收。只结束已核实PID13372及本任务子进程，无新写入者、无补跑/扩预算/救补。
- 最终jobs/evaluations各71份JSON，refit0/3，analysis/integrity均未生成；未计算或使用部分确认统计选规则。开发screen虽通过，按事前不完整确认规则保留原MSIC，不推荐D3。
- 推荐D2/v5.2-search-final；补齐A2_MSIC_DECISION.md及FINAL_ALGORITHM_VERSION.md，七份任务卡交付齐备，其他决策/同步/full文档与VERSION/MANIFEST状态已统一。
- 本轮157项限定测试通过；2040受保护文件差异0，32候选源码与patch/sh指纹保持，生产v5.1不变；未新增算法逻辑。
- 文档同步首条Python管道因默认PowerShell中文编码使替换锚点变问号而断言退出；两份JSON英文元数据已更新、正文尚未写入。随后使用Unicode安全的apply_patch完成目标正文，未修改科学协议/源码/结果。
- 任务卡4验证决策和准备交付完成，独立MSIC确认实验未完成，正式发布仍待用户批准；700正式模拟及三实证重训不在本次执行范围。

## 2026-10-11 任务卡5启动

- 任务卡5明确批准D2正式发布；先只读核验v5.1基准/生产32源码、D2冻结32源码，均无差异，HEAD仍d0c3547704fbaf5d90a3e5423568768a1ba11759。
- Git有165项既有tracked未提交状态，不创建分支/worktree/提交；保留冻结patch并以独立正式发布清单保护历史研究记录。
- 按任务卡已明确方案沿用brainstorming、TDD及doc文件规划；子任务分别负责安全审计清理、集群脚本和三实证交接，主线只发布bcd网格及汇总正式证据。
- 未发现根目录四类工作流文件；不执行700任务、不实际上传远端或运行三实证。
- 正式生产测试RED：3项中2个预期FAIL（v5.1指纹/旧grid遗漏），原MSIC默认通过；apply --check成功后只应用既有search-final.patch，32生产raw与冻结D2一致，3项GREEN。
- 实际生产BCD/旧公共模拟pytest29项通过（88.54秒）；候选6静态/真实桥接及LEMON12公共回归均通过，发布范围共50项通过。
- 额外known-s联合命令32项中31通过1静态FAIL：未修改的旧run_known_s_full.sh使用显式conda初始化，而旧测试要求字面source activate myenv；旧脚本/测试不改，失败单列，不能称该批全通过；新版v5.2入口单独验证。
- 正式RELEASE_MANIFEST标released/approved_by_user=true，保存32发布前raw、v5.1完整恢复快照、32正式raw/canonicalLF、6依赖与历史研究metadata原hash；旧研究VERSION未修改。
- 本机真实独立预检计划6b634645722686a19cb085cabb2de4f0ee126074339c0db9ae738bef130103ef建立于outputs_v52_release_local_preflight，只运行task0四方法；尚在运行，不能称完整700或真实SLURM预检通过。

### 任务卡5最终交付

- 协作代理完成审计baseline/初始模块和实证两文档后因用量限制退出；主线接管未完成脚本与清理，不重启已有写入者。新部署缺失文件测试RED后实现；后续明确测试揭示staging层级、父目录别名和SFTP引号问题，逐点修正后GREEN，无救补历史产物。
- 本地正式生产32源与冻结D2一致，历史研究VERSION/team MANIFEST原hash不变；50发布范围+9部署+7清理共66限定测试通过。另旧known-s1静态断言失败明确单列，未修改其script/test；没有泛称全仓库通过。
- 完整baseline14558文件/4027281065字节/7068重要hash/0错误。审计和精确dry-run先写，再删除125未跟踪无引用再生pyc，1382690字节；after重要7068逐项hash差异0，没有移动目录/删除历史证据/已跟踪文件。after快照14491文件/4027984579字节，新增审计/发布/本机产物使净增703514字节；不夸称项目净减少缓存大小。
- 完成11份目标文档、正式manifest/CHANGELOG/validation/sourceSHA、完整本地/远端NOT_RUN实际hash表、root SLURM及inventory/verify/Gate/PS同步下载工具；默认dry-run/显式Apply、白名单、backup/prehash/posthash、中文空格路径及partial merge拒绝均本机验证。
- 真实本机task0四方法完成、原JSON/NPZ验证和resume byte-identical通过，LOCAL_PREFLIGHT保存1/700状态。远端inventory/真实环境/真实SLURM task0 Gate均NOT_RUN；未ssh上传、提交700、下载远端结果或重训实证。
- 实证交接仅准备：145人/8协变量及固定CV/仅协变量基线保持，旧LEMON完整32源实际匹配v5-original（不是v5.1）；原q80与RMSE/R²仅标历史，不保证新版本改善，不复用旧支撑/系数。
- 用户下一步需填写实际SSH/myenv Python路径，目标集群部署与真实预检通过后再决定700提交；本地发布和能够执行的审计清理/脚本验证已完成。

### Python3.9集群启动脚本跟进

- 用户明确提供旧集群Python3.9.25/NumPy2.0.2/SciPy1.13.1脚本；正式manifest原环境已匹配，不改库版本或算法。
- 新root脚本按旧例使用默认绝对project和SBATCH日志路径；conda激活在set -euo pipefail之前，module/source/activate失败明确退出。新增unset站点初始化模拟测试先FAIL再GREEN，路径契约先FAIL再GREEN。
- 必须同步两个远端文件：run_v52_search_final_full.sh和doc/releases/v5.2-search-final/RELEASE_MANIFEST.json；后者更新root hash并识别已上传前一版root/manifest，核心32文件及SOURCE_SHA256和辅助脚本不变。
- 38部署/导入Python文件的3.9 AST通过；10部署+7清理+3生产身份共20项本轮通过，当前累计发布限定67项（算法50项为已有发布证据），真实集群/SLURM仍NOT_RUN。部署指南同步绝对路径与自定义路径所需SBATCH覆盖。

## 2026-10-13 集群激活及无日志问题

- 用户确认站点环境准备只有module load anaconda3、conda activate myenv；删除root脚本的V52_CONDA_INIT/固定source分支，保留激活后严格模式及失败退出。模拟module提供conda无额外文件依赖的测试先FAIL再GREEN。
- 用户报告作业已结束但无输出目录和日志，未提供Job ID/调度器记录；不能确定唯一根因，也不宣称远端已修复。已给squeue/sacct状态/退出码检查与登录节点先mkdir日志父目录再prepare/status、成功后仅task0的方案。
- prepare创建日志父目录提前至release校验前，失败校验时仍保留目录；该行为测试先FAIL再GREEN。SLURM提交前日志目录仍必须由登录节点创建，不能依赖作业开始后mkdir。
- 正式manifest同步root指纹并保留可识别的上一版root/manifest基准；远端只需更新root sh与正式manifest，核心/辅助Python/库/历史结果不改。11部署+7清理+3生产身份共21项本轮通过，累计限定68；实际远端状态未知，不重复提交700或删除旧输出。

### 2026-10-13 task0作业记录路径修复

- 用户提供集群VERIFIED的32core/3.9.25/2.0.2/1.13.1，plan30f231df...完成task0且无锁；作业36215的20字节worker log存在而job JSON缺失，Gate没有匹配记录。原36215.err尚未读取，未宣称其具体异常已确认。
- 只读代码审查定位确定缺陷：相对log.relative_to(绝对output)必抛ValueError；真实相对日志路径契约先FAIL，再以log.resolve().relative_to(output.resolve())单点修复，GREEN且日志字节不变。
- 同步发布helper及manifest原/规范hash并保留已部署前一版识别值；仅两个远端更新目标为release_control.py和正式manifest，核心/科学runner/原任务结果不改。12部署+7清理+3生产身份22项本轮通过，累计发布限定69。
- 提供真实sbatch --array=0的原resume流程，已有完整task0核验后skipped并写新真实SLURM记录；不再prepare、不重拟合、不删JSON/NPZ、不伪造或放宽Gate。实际重提及Gate成功待用户执行/回报，未提交700。

## 2026-10-15 正式完整模拟验收
- 用户允许在历史依赖目录权限异常后继续核验；未改变系统ACL、环境或历史文件。
- 已分派两项只读审计，主线检查原runner门禁、正式发布和报告交付；原始task/plan/gate/slurm不作修改。

### 2026-10-15 原生完整汇总通过
- 原命令 python -B -m numerical_simulation.known_s_cluster.run merge --output-dir numerical_simulation/outputs_v52_search_final_full 成功：700 tasks / 2800 method results。
- 生成原生 raw_results.csv、summary_results.csv、integrity_report.json 及两份不适用的仅表头 fixed 比较CSV；未改task/plan/gate/slurm、算法、环境或断点。
- 本机只做离线读取与原生汇总，不调用worker或伪装Linux环境；完整计算仍来自plan保存的Linux3.9.25/NumPy2.0.2/SciPy1.13.1。
- 严格复核不重拟合；预测指标未独立重算、OLS最优性未重新验证的边界写入报告。

### 2026-10-15 最终交付与核验
- 交付doc/V52_FULL_SIMULATION_REVIEW.md：完整结果、旧v5严格配对、版本/哈希、性能取舍、实证Gate、可转发文字与ChatGPT推理请求齐备；未联系他人、改算法或启动三实证。
- 独立审阅结论与全14表数字/配对区间/q/计时/代码解释；rho.6 SD首稿误写已更正0.4676，两处范围表述精确化。报告链接、文件hash、中文编码均检查；不将历史测试计为本轮重跑。
- 首稿经PowerShell管道传Python时中文被转换成问号；已用PowerShell原生UTF8重写，中文与无连续问号/替换字符检查通过，原实验文件不受影响。
- 历史只读检索再次遇test_tools拒绝访问后停止该检索；依用户本轮允许访问授权继续目标证据核验，未修改ACL或安装工具。
- 工作流仅追加doc三文件及本轮勾选，不移动或覆盖历史记录；根目录四类工作流文件均不存在。

## 2026-10-15 任务卡6启动
- 分派P0冻结源码/输入可复现审计、P1a LEMON启动前审计、P0/P2配对尾部与ZIP准备只读审计；正式生产和受保护旧产物无写入。
- 当前主要不确定性为任务卡缺失细则与跨环境重生数据的逐字节匹配；独立阶段按卡第9条继续可行准备，不安装依赖、不覆盖历史。

### 2026-10-15 任务卡6完成阶段与真实阻碍
- 用户补充正式批准250标签、完整LEMON与两份同字节ZIP；已按更新后的完整任务卡执行，不再请求范围确认。
- 三个协作代理因额度退出，主线接管；先检查已有audit/进程，确认未启动full才唯一启动LEMON，未重复拟合。Get-CimInstance进程查询拒绝访问已记录，没有改变权限；已有audit日志及Get-Process检查明确无在运行Python拟合。
- P0真实strict preflight失败：X_train/y_train/train_noise错配、B_true匹配，实际0/250；未装环境/替换数据/近似匹配。独立v5.1进程runner、250冻结plan及首任务失败原记录交付cluster_run，五场景pilot Gate必需、源/环境/hash绑定、拒绝覆盖；根层早期失败预检保留，不替代定稿协议。
- P0工具9项先RED后GREEN；另NumPy trace序列化契约RED→helper转换GREEN共10项。新模块首个apply_patch未能创建父目录，显式New-Item在工作区成功后创建，不改ACL或受保护源码。
- 完整LEMON audit通过，唯一full进程完成75/2175，1253.02秒，experiment_id405a67add110fcbbe179c5640c6f04c388111842bb76dc3c4977bdd87702d64d；所有旧数据/配置保留。
- OOF完整性/版本/响应/划分验证的9项测试通过；首版配对fold导出和历史manifest前缀拒绝测试揭示缺件，逐点修复后GREEN。实际旧新配对完成，四个对照重拟合后的1740条预测完全相同。
- LEMON新ABESS RMSE0.504581/R²-0.597008，旧0.371667/0.121430；协变量0.317485/0.367900；q80由1/15到5/15。30个旧新ABESS设计满秩，最大cond344.3866，正规方程残差1.91e-14以内，保存预测重算差0；未更改B或救补预测。
- ZIP5项关键测试先RED后GREEN（含实际文档--source目标import路径）；实际两个ZIP各49943bytes，SHA899133561b556174af9c70d064ab6a0e2419b136b3ab7b53871b1bc0ff07cacc，独立解压/隔离import/fixed-adaptive/包内2测试均通过。
- P2三项分析契约RED→GREEN；保存700配对、600去重、7个预选NPZ哈希核对及条件总体risk，无正式模拟重拟合。57个新增Exact全伴随PMSE改善，13个丢失均恶化；尾部损失与改善任务混合造成整体方向不同。
- Python科研图后端在doc/taskcard6内持久化，两图PDF/SVG/PNG600dpi及4面板目视通过。源检查起初未识别通过参数调用的真实alignment Gate，改显式调用后通过；同样修正源检查对183/25.4字面宽度的误读。最终alignment、字体6pt下限、collision0FAIL/0WARN均通过；仅不输出TIFF的内部报告范围WARN已说明。
- 任务卡6交付14份报告/索引。P0阻塞和HCP/NKI缺结果均明示，阶段性而非全任务数值完成；不联系队友、不新提交700、不改正式core/MSIC/caps/旧结果。

### 2026-10-15 最终交付复核
- 27项本轮工具限定测试通过（18+9），两个包内2项兼容测试各独立运行；不称全仓库/真实Linux集群验证通过。
- 实测复核2570个基线受保护文件的大小及SHA256，差异0；14报告中文/链接、旧新LEMON实验身份哈希、75/2175及两包字节/隐私检查通过。源/结果/计划均未救补。
- 最终核验曾用Python默认本地编码读取含中文路径的JSON，出现检查器层的实验ID假不匹配；改为明确UTF-8读取后旧新ID分别完整匹配，未改任何原记录。
- 自审精确化n400覆盖用语：真规模从全局fixed规模网格外进入网格内；旧池仍含adaptive结果，不把原网格缺尺寸误写成整个旧候选池永远不可达。P0仍明确0/250，所有结构解释不冒充数值归因。
- 交付索引doc/TASKCARD6_EXECUTION_REPORT.md，证据/hash索引doc/taskcard6/DELIVERY_INDEX.json。LEMON及ZIP/P2/P3阶段完成；P0需原环境恢复，HCP/NKI待返。
## 2026-10-16 任务卡7启动
- 本地阶段已完成：A补15外折、C2重放12代表训练拟合、C3完成45内模型评价（含9个协变量基线），B按用户选择待原集群。
- A全部15折/1215候选B、C2 1458候选合法满秩/score、C3 1740内验证预测独立核验；三版最终输出和v51/v52实际候选路径分别实测留证。
- 新增10项限定unittest先红后绿并最终合并通过；科研图2张共5面板全量折数据，PDF/SVG/TIFF/PNG600dpi，对齐/6pt字体/碰撞均通过并目视核验。
- 已交付10份目标报告及复现说明；Debug→第一性原理审查→精简完成（研究脚本隔离、去除报告脚本全局数据依赖），独立审查未发现实质缺陷。
- 最终审阅将“非零形状变化”改为本批972对全部同形状，防止误读；QA日志首次捕获受Windows本地编码影响，显式UTF-8子进程输出后完整采集，实验和图内容未修改。
- 访问异常为已获准继续的旧测试工具；默认Python与历史.venv不同且可选基线缺件，实际使用既有历史.venv和标准库unittest，未安装或改环境；完整32源码hash及真实用到的import均校验。
- 已读取任务卡7、仓库约束及既有工作流；根目录无四类工作流文件。
- 权限异常已向用户报告并停止，用户明确“允许访问，继续执行”后恢复目标证据核验。
- 分派A隔离v5.1补跑及路径记录、B严格环境Gate审计、C训练内重搜诊断；生产和历史结果只读。

## 2026-10-16 任务卡7单文件集群提交交付
- 用户明确要求一个sh直接提交；采用现有站点配置与单作业顺序worker，不创建分支/worktree，不改科学算法、共享依赖、冻结worker/plan或历史结果。
- 新run_v51_attribution.sh：绝对cd/chdir、module+conda无额外source、四线程变量、根目录日志、自动原字节初始化、全量身份校验、真实五pilot→245→merge，逐任务即时进度。
- 新启动层只允许完整结果校验后跳过；部分JSON/NPZ/Gate、失败预检、错计划或已有CSV/integrity均停止保留。fcntl非阻塞锁及子进程pass_fds避免同启动器并发写，手动旧worker不参与新增锁不得混用。
- 初始7测试因缺sh实际FAIL，实现后GREEN；新增流程测试首次因测试fixture重复mkdir报错，修正fixture的exist_ok后，8项新测试全部通过；原归因10项也通过，共18项限定测试，无真实拟合。
- Bash语法、LF、内嵌Python3.9 AST及临时250任务编排检查通过；独立审查确认阶段/覆盖/锁继承，仅将已有preflight日志精确化为历史记录核验。
- 冻结worker SHA082f603bb975f04eef5c234f5fad695d2fe942ff02367e5bf18c2a6cd0bc8217、原计划文件SHA4c84e048353510a65d8db62e531062a081752ae76262112de8d49671e15b4ae3均未变，三版96源码原测试校验通过。
- 运行报告及输出README更新；部分apply_patch返回锚点失败但后读发现报告首段已写入，按实际内容核对后仅补缺失README，未重复追加报告。
- 交付本地sh供用户上传，实际SLURM/Linux锁/分区时间上限未测试，未远程上传/提交，实际250科学任务仍待集群运行。

## 2026-10-18 Task11 startup
- User authorized privileged read of old test-tool directories; read succeeded without ACL changes.
- Prior Task10 root reports describe stopped history; continuation_01 contains complete 48-case development evidence matching Task11.
- Isolated audit, historical rescore, and runner assigned to agents; all simulation starts remain gated by parent.
- Protection manifest established before new science. A guessed continuation protection filename was absent; directory inventory located actual DELIVERY_INDEX; no missing scientific prerequisite.

### Task11 pre-Pilot gate
- Historical full audit passed: 216 pools, 31968 candidates, exact original A0 recovery in 216/216.
- Original Cold 16 checks and 29 new unit checks passed before independent review.
- Historical strong-signal extra underselection is heterogeneous; no broad severe-recovery claim or early statistical stop. No new data generated.
- Independent reviewer caught concurrent read of half-written peer input_identity: Pilot launch held; author fixing only current-output scan exclusion, with exclusive input claims retained.
- PILOT_PROTOCOL.json was sealed before review completed and is NOT_EXECUTED; a corrected-source ready protocol will be sealed before first new scientific input, preserving this evidence.

### Task11 formal run handoff
- Pilot complete, session33181 exited0; paths doc/taskcard11/results/pilot and PILOT_PROTOCOL_READY.json.
- Formal prepared and launched uniquely via research.taskcard11.experiment run-validation; live exec session48000.
- Latest observed count36/96, no failures; do not inspect partial PMSE or change frozen source/report/protocol.
- Formal plan: doc/taskcard11/results/formal/plan.json; scientific source files including report.py/test_report.py are now hash-bound.
- Final reporting: python -B -m research.taskcard11.report --input doc/taskcard11/results/formal (only complete96).
- Independent verifier agent has verify.py/test_verify.py (9 tests), read-only same-seed full replay pending after formal completes.
- Historical agent finished rescore/history and report generator (7 report tests), no more source edits allowed during live batch.
- Final decisions/reports and FINAL_VALIDATION/PROTECTED_FINAL_CHECK/DELIVERY_INDEX still outstanding.

### Task11 final completion (supersedes the live handoff above)
- Formal runner exited0 with all 96/96 cases complete, zero failed/not-started cases, 5100.688s shared science wall time and maximum case 258.657s; no replacement, extra search or second round.
- Frozen report produced all paired/group/cell/selection/truth/q tables and uncertainty. Six statistical gates pass; strong q-underselection increases 7/32 -> 11/32 and fails the fixed +0.05 limit. Result B retains original MSIC and formally released v5.2.
- Independent replay initially stopped on verifier lifecycle double initialization. Failure preserved; only unfrozen verifier and its test were repaired with RED->GREEN, source-path checks retained, then complete original-input replay restarted from the beginning.
- Pilot6 and formal96 independent full replay PASS: 16406 candidate RSS/normal-equation checks, 1122 saved artifact hashes, exact training/truth/test identities; maximum prediction error 6.66e-15, RSS error 6.82e-13. Zero new searches or scientific repetitions.
- Final 47 new unit tests plus 16 unchanged original Cold checks PASS (63 distinct checks); 7874 protected files unchanged and tracked Git state matches startup. FINAL_TEST_VALIDATION_POST_REPLAY supersedes the preserved 46-test intermediate record.
- Nine named deliverables, final algorithm freeze, real-data suitability plan and reproducibility commands completed under doc/taskcard11; final status and content hashes in FINAL_VALIDATION.json and DELIVERY_INDEX.json. Formal source, historical failures/results, existing user changes and release identity preserved; no branch/worktree/commit, new release, real-data fit or teammate message.


## 2026-10-18 Task12 start
- Task12 explicitly reopens only the existing FS formula for final research comparison; no coefficient, search or production changes.
- Delegated selection/history gate, cluster scripts/runner, and empirical audit/adapter in physically disjoint new files.
- Broad document search encountered the known protected old test-tool directories; stopped that scan and continued only explicit authorized target paths, no ACL changes or use of inaccessible package copies.
- Root workflow files absent; all planning remains under doc. Actual cluster config confirms Slurm/CMO_HPC/xjh/myenv/project path; unknown site quota/time/memory remain unknown.


## Task12 resumed 2026-10-20
- User explicitly requested unfinished work continue. Existing tracked Git state equals Task12 startup; no production/history edits. Old exec session21284 no longer exists; artifact recovery by agent confirms initial102 regression completed (471.94s), no search restart.
- Agent turns had previously stopped on usage-limit errors; reassigned same authorized tasks and recovered persisted files.
- Initial wrapper default-domain audit found official zero-y and rank-deficient inputs legal, wrapper incorrectly restrictive. New adapter now permits unmodified zero RSS and verifies minimum-norm OLS; rank/condition reported without repairing coefficients. New102-source regression required before P0 acceptance; old102 evidence preserved.
- Cluster runtime independent review covers direct formal gate, source/config binding, own-rule re-selection, dual digest history lookup and metrics replay; lifecycle issue from old fresh-process configure helper is being corrected only in new runner.
- Task12 cluster protocol/package/empirical protocol not yet sealed; no380 formal input or remote submission.

### Task12 local delivery completed 2026-10-20
- Final selection source76106293:12 selection tests,16 original Cold and7 frozen MSIC checks; root full55 Task12 tests PASS,78 distinct total. No global changes, source32 remains original released v5.2.
- History predecessor102 full training/OLS/cache PASS550.457s; final four-guard-only source delta proved and102 sealed pools/16406 candidates/204 models/1122 hashes rechecked15.871s. Old-source caches correctly rejected; no fabricated current-source cache migration.
- Cluster19 tests/4SH syntax/mock gates PASS; P0,380+3smoke cluster protocol and empirical protocol sealed. Actual ZIP70 payload files,262049bytes,SHA39ee2e260fe8778c22de17046351ceb20b0cd4e2290bca05c9b0441f100eb8d3; independent extraction/native exact comparison PASS. Initial fixture keyword error preserved and corrected; README-only initial ZIP retained separately.
- Prespecified local n200 pilot1/1:126 candidates,one search,45.679s case/47.046s controller; all candidate OLS passed, excluded from380 formal and3 remote smoke.
- Unique LEMON run15/15 PASS277.074s under1800s:15 searches,1215 candidates,2610 full-model OOF rows. Original15 outputs exactly match historical v5.2. FS RMSE0.359627 versus original0.504581 and CovOnly0.317485; improves original but still worse than all4 baselines;9/5/1 fold improve/same/worse. HCP/NKI NOT_RUN with missing inputs and original metric protocol boundaries documented.
- Root independent30-model pool/preprocessing/prediction replay and all OOF metrics PASS; maximum matrix prediction error2.66e-15,143 artifact hashes. All9705 protected files unchanged and tracked Git status equals initial.
- All required local deliverables complete. Final status LOCAL_DELIVERY_COMPLETE_PENDING_CLUSTER_RESULTS; remote login/upload/smoke/submission/formal380, HCP/NKI fitting and FS production release NOT_RUN. No final A/B/C science recommendation before cluster return.
- LOCAL_COST_REPORT UTF8 encoding corrected before sealing; initial multi-operation patch was rejected without file change, then a single explicit update succeeded. Scientific data and frozen source unaffected.


### Task12 cluster return closure
- Independent saved-chain verification PASS: formal380, smoke3, frozen sources/protocols/seeds/artifact hashes.
- User-reported original Linux strict verification PASS: smoke3/formal380 and exact summary/analysis match with four frozen thread variables=1. Windows floating-point stop preserved; second command without variables correctly rejected before validation.
- Final recommendation B: retain original MSIC production default; FS frozen research only. Primary PMSE -19.65%, n100 -28.70%; four recovery gates failed. HCP/NKI NOT_RUN; no further tuning or production changes.
- Local delivery snapshots preserved under doc/taskcard12/cluster_review/local_delivery_snapshot; current final reports and evidence indexed separately.
