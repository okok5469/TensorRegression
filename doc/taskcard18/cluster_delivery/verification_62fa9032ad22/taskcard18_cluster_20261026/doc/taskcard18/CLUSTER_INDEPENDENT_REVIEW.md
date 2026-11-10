# 任务卡18：集群交付独立复核

日期：2026-10-26。结论：**PASS_LOCAL_SCOPED_REVIEW**，仅限本机静态审查、纯标准库小样检查及已明确的失败边界。**真实HPC执行尚未运行；本结论不授予任务卡18 FINAL，也不证明600×2新对照、G2/G3或论文最终比较已完成。**

审查者未实现集群调度、离线运行时或打包程序。本轮只读复核cluster.py、bootstrap_runtime.py、run_taskcard18_cluster.sh、build_cluster_bundle.py及相关缓存调用、限定测试和最终README；没有修改实现或统计协议、拟合新模型、导入Linux wheels、安装依赖或提交Slurm作业。

## 1. 统计协议与阶段门禁

执行修订绑定的原CLOSEOUT_PROTOCOL.json SHA256仍为a794d6a0efef81acf9d69b3429d8fe98c9d18131b8412ca70337fa4cbaebfa61，与EXECUTION_AMENDMENT.json一致。迁移只改变执行位置：固定20输入pilot、40个方法拟合验收、600唯一完整输入、两个v2新method_id、lambda/CV规则、KKT及gap双证书、三档累计预算和测试批大小均沿用原协议。正式ABESS与Oracle复用，不在集群链中重拟合或调参。

五阶段为prepare→pilot→pilotgate→formal→collect，使用afterok依赖；formal再次检查pilot PASS及运行身份。collect先要求全部1200个新模型验收通过，再执行新方法测试评价与默认统计。完整输入去重保留700标签到600输入的映射，不按10配置缩减为500输入，也不增加实验重复。

## 2. 已发现问题及修订复核

| 初始发现 | 当前处理与核对 | 结论 |
|---|---|---|
| ZIP预建空runtime，但bootstrap拒绝无marker的已存在runtime，首次prepare必失败 | builder不再预建runtime，验包断言runtime不存在；bootstrap仍严格拒绝未封存运行时 | 启动约定一致 |
| Slurm执行spool脚本时，以BASH_SOURCE所在目录寻找payload会定位错误 | 每个sbatch指定payload的chdir，提交进程cwd绑定payload；Slurm分支使用SLURM_SUBMIT_DIR，并检查cluster.py存在 | 静态调用链一致 |
| squeue使用%A并只接受array主ID，可能将真实array元素误判为未拥有作业 | 改用%F并去重，仅操作本包记录的主ID；普通作业同样可按其ID核对 | 符合Slurm字段定义 |
| NPZ先提交、JSON后提交，中断后可能重拟合再撞同名文件 | worker只对本输入检查成对产物；无活动阶段的prepare/collect检查全部缓存，孤立NPZ或JSON明确报错并保留 | 失败暴露，未静默补造 |
| 同名不可变失败JSON在重复失败时可能冲突 | 已有未解决失败会在新拟合前阻断，不自动重试硬数值失败；存在完整模型仍须独立validate_fit | 不以覆盖失败记录换取续跑 |
| prepare固定测试日志使再次进入时碰撞 | 使用focused_tests_attempt序号文件；prepare marker绑定具体路径及SHA，已存在日志不覆盖 | 限定恢复检查通过 |

成对产物扫描只在prepare/collect无活动阶段全局执行，worker只检查自身输入，避免把另一worker短暂的NPZ→JSON提交窗口当作全局失败。check_fit_pair仅检查存在性，成对文件随后仍经过身份、seal、系数、调参完整性及优化证书校验，不因文件成对就直接视为有效。

Slurm字段依据为官方[squeue说明](https://slurm.schedmd.com/squeue.html)与[Job Array说明](https://slurm.schedmd.com/job_array.html#squeue)：%F表示array基础ID；普通作业使用其job ID。此处核对的是调度语义，不是实际集群回执。

## 3. 本机检查证据

独立复跑命令为：

    .\.venv\Scripts\python.exe -B -S -m unittest research.taskcard18.test_cluster research.taskcard18.test_bundle -v

修订前8项中7通过、1失败，失败为缺少chdir断言；修订后10项全部通过。当前检查覆盖固定输入映射、afterok链及有界数组、重复提交/活跃续跑保护、ABI与路径拒绝、孤立产物及失败记录的保留、attempt日志、ZIP往返字节一致、篡改拒绝、路径穿越预检和非便携文件排除。

另读取CLUSTER_FINAL_TESTS.txt，确认主执行者的最终限定集合36项通过；该记录与本审查者独立复跑的10项分列，不合称独立运行了46项。没有重新执行无关全仓测试。

## 4. 可移植与回传边界

builder只打包冻结38项源/配置、正式32源身份、700份历史JSON/NPZ及明确的本轮隔离入口；本机新拟合、运行身份、评价缓存、Windows运行时与预建runtime不进入新的集群输出。文件字节清单和ZIP校验提供传输核对；路径遍历、符号链接和不同内容覆盖会被拒绝。离线依赖版本锁定并按wheel SHA和平台标签检查，bootstrap只在Linux x86_64 CPython3.9及规定线程设置下工作，并核对现有NumPy2.0.2、SciPy1.13.1及私有依赖版本。

这些检查没有证明Linux二进制实际可导入、目标分区可调度或站点环境与声明一致。首次真实prepare仍必须完成运行时导入、来源/版本检查、限定数值检查和20套输入恢复；若失败，应回传现场，不能将Windows的AST、mini-root导入或调度dry-run当作HPC成功。

回传bundle包括本次输出中的系数NPZ、逐模型诊断、评价、身份、统计、Slurm与测试日志及失败记录，同时包含本轮源和文稿/审计材料，并另写ZIP SHA。collect异常也在finally中打包；上游失败导致collect不运行时，README提供手动bundle入口。原历史数据与完整冻结计划继续由原仓库和交付payload保留，回传的新结果按稳定身份对应。

## 5. 用户指令及恢复限制

已读取修订后的research/taskcard18/README.md：submit、status、cancel、resume、bundle五个入口均给出显式bash -l整行命令，使用新的独立解压目录并禁止预建runtime或混入本机缓存。依赖失败后先cancel本包仍活跃的作业，确认无活跃项、核查原因后再resume；不对其他作业进行取消。

恢复承诺有明确边界：完整且校验通过的模型可以复用；孤立文件、未解决硬数值失败或未封存prepare身份会保留并停止，不自动删除、补造或绕过。此类现场需先bundle回传核查，不能把resume描述成任意中断均可无条件自动恢复。未达到优化证书时不追加无界预算，也不减少主比较分母。

README明确真实HPC尚未执行，collect完成仍不自动授予正文/图表最终验收。当前科学任务保持IN_PROGRESS/DRAFT；最终ZIP本身的全量构建、解压、文件指纹、mini-root及shell验收由cluster_delivery/BUILD_VALIDATION.json给出，不以本审查文档代替该打包结果。

## 6. 本次读取的实现身份

| 文件 | SHA256 |
|---|---|
| research/taskcard18/cluster.py | 8d770c81154dce9170fb61aff7ff1f77ac2a6617162c533e2a6502af5a21af02 |
| research/taskcard18/bootstrap_runtime.py | adbe5cada7f7cea1778f871091c54029cf8f8bddf6cc25f3a12e191470354036 |
| run_taskcard18_cluster.sh | dbb812d7eec21363b788704e8b28d93f9d4f045b0ad9d799cd828300a1fcd8d7 |
| research/taskcard18/build_cluster_bundle.py | 1b04fcbb3cd112789965f2a7aa4671dd02f9f0ea455b9229eed41f4b4724e328 |

上述SHA绑定本次静态读取范围；之后若有实质代码改动，需要复核对应边界。没有剩余阻止进行本机最终打包验收的已发现问题；真实HPC及完整新比较仍是明确尚未完成的工作。
