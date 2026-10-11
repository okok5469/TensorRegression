# LEMON 实施记录

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
