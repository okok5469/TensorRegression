# 本地正式发布验证

状态：`released`（本地源码）。用户批准来源为任务卡/任务卡5.md及其manifest SHA256；集群清单、真实SLURM预检、700任务和三个实证均未运行。

## 核心门禁

- 发布前生产32文件与v5.1快照全部匹配，已保存PRE_RELEASE_SOURCE_SHA256.json；冻结D2的32文件也逐项核验。
- 新生产验证测试先出现两个预期FAIL：旧源码指纹不等于D2，旧grid遗漏(3,3,3)；原MSIC默认测试通过。
- 先`git apply --check`，再仅应用已有search-final.patch至bcd.py；未stage/index其他改动。发布后三项生产测试全部通过，32生产文件原始字节与D2一致。
- 实际生产来源的BCD/旧公共模拟调用pytest：29项通过，88.54秒。
- 冻结候选静态及独立进程拟合桥接：6项通过；LEMON公共模型回归：12项通过。
- 正式发布范围共50项通过（3+29+6+12）；不是全仓库测试。pytest只使用既有隔离工具，环境未安装或升级。

## 既有非发布范围问题

额外联合命令执行candidate6、LEMON12及旧known-s14测试，共32项，其中31通过、一项失败。失败是旧known-s shell静态测试要求字面`source activate myenv`，而当前旧run_known_s_full.sh已采用显式conda初始化；该旧脚本和测试均未由本次修改，不能将这条联合命令称为全通过，也未为过测试而修改用户旧文件。旧runner的13项其余测试通过；新版v5.2脚本另行验证并在正式发布报告记录，不使用旧三方法脚本提交新四方法任务。

本机Python3.12.8/NumPy1.26.4/SciPy1.16.1，四BLAS/OMP/NUMEXPR变量均1。32候选文件的Python3.9 AST验证仅为静态证据，不能替代实际集群Python3.9.25/NumPy2.0.2/SciPy1.13.1运行。

## 历史与后续边界

历史研究VERSION和团队MANIFEST的发布前SHA256保存在正式manifest，保持原未批准记录；当前正式发布批准只记录在本目录。原数据、旧输出和未完成MSIC确认不作为新版本完整结果；新算法正式模拟和实证必须独立目录重拟合，禁止复用旧ABESS支撑/系数或修改旧版本标签。

本轮未提交700个任务；实际集群源码、环境、单任务四方法、JSON/NPZ及resume预检须由新脚本现场核验，状态为NOT_RUN时不能通过全量提交Gate。

## 交付收尾验证

新部署工具9项、清理7项测试全部通过；与50项发布范围合计66项限定测试。包括真实PowerShell默认dry-run及mock SSH/SFTP Apply、中文/空格路径、实际普通副本上传包应用和可恢复backup、source/deployment漂移拒绝、父目录路径别名拒绝、部分汇总下载拒绝；没有真实远端网络动作。新版root脚本和3个shell wrapper通过Git Bash语法检查，3个工具Python文件通过Python3.9 AST；这是本机静态/模拟证据，不是站点运行验证。

真实本机独立full预检仅执行task0四方法，原read_task完整核验JSON/NPZ和指标，resume跳过且两文件SHA不变；证据LOCAL_PREFLIGHT.json。计划700仅完成1个，其余699未运行，无集群Gate。清理125再生缓存后7068重要文件hash差异0，历史研究VERSION/团队MANIFEST与发布前保存值一致；版本/数据/报告路径全部保留。

## Python3.9集群启动脚本跟进

用户给出原Python3.9工作脚本后，新版按同一站点的conda激活顺序、默认项目绝对路径及绝对SBATCH日志修订；算法、四方法、700设计及预检Gate保持。新增站点初始化模拟测试先FAIL（set -u使可选未设置变量报错），改为激活成功后启用严格模式且初始化失败明确退出，再GREEN。

当前部署10项、清理7项、生产身份3项合并20项本轮通过；发布时的50项算法/公共回归证据不重复称为本轮重跑，当前发布/交付限定累计67项。38个部署/导入Python文件通过Python3.9 AST解析，无Python文件需要修改；未实际在集群Python3.9运行。新增脚本指纹及前一版script/manifest识别值同步至正式manifest，核心SOURCE_SHA256清单不变。

## 2026-10-13 实际站点激活修订

按用户确认的module+conda方式移除硬编码source；prepare提前建立日志目录，再作release校验。两项站点行为先FAIL再GREEN，当前11部署+7清理+3生产身份共21项本轮通过，累计发布限定68项。源/环境/预检门禁未放宽；核心与辅助Python没有修改。

用户报告原作业已结束且无日志，尚无Job ID/调度器记录，无法确认实际失败原因；本地修改和模拟测试不能称集群运行已成功。部署指南明确先获取State/ExitCode，提前mkdir日志父目录，在登录节点执行prepare或已有计划status后，只提交task0。远端同步仅root sh和正式manifest，保留已上传上一版的精确识别值。

## 2026-10-13 task0元数据路径修复

当前job_record对相对--log使用relative_to(绝对output)的错误被本地契约测试复现；双方resolve后计算相对路径通过，检查原日志字节不变。12部署、7清理、3生产身份共22项本轮通过，累计发布限定69项；算法50项为之前发布证据，未重新声称本轮运行。

集群用户已提供环境/32源VERIFIED、task0完整且无锁、job36215 worker log存在而job JSON缺失；尚无对应.err全文，Gate仍未通过。本次只更新release_control.py及正式manifest，不改科学源、plan或已完成结果；后续原真实SLURM resume会严格验证并跳过既有拟合，再自然写入真实job记录，不在本地制造集群验证成功。
