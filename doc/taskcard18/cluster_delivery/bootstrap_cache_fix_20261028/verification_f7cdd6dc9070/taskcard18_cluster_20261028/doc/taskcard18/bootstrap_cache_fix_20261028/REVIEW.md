# 首次离线运行时缓存修复：独立限定复核

日期：2026-10-28。结论：**PASS_LOCAL_SCOPED_REVIEW**。单行修复及其回归证据在本机限定范围内成立，可以进入独立修复包构建；**不代表已确认HPC故障的唯一根因，也不代表修复后的真实HPC prepare已成功。**

审查者未参与本次实现，仅只读核查bootstrap_runtime.py、test_cluster.py、本目录RED/GREEN及验证记录、ISSUE.md和最终README；只新增本审查文件，没有修改实现、拟合数据、统计协议或运行依赖。

## 1. 最小变更与冻结边界

直接读取保留的原交付ZIP（taskcard18_cluster_20261026.zip）内对应源，与当前文件逐字节比较：当前bootstrap_runtime.py仅增加一行importlib.invalidate_caches()，位置在私有runtime解包或校验完成、加入sys.path之后，全部依赖导入之前。移除该行后与原文件完全相同。

版本字典、私有依赖导入路径检查、wheel SHA、运行时文件SHA、主机/ABI门和线程条件均保持。没有删除已导入模块、放宽版本或路径要求，也没有安装、升级或改动共享环境。

同样直接对照原ZIP，以下四项字节完全一致，而非仅接受FIX_VALIDATION.json的自报结论：

| 文件 | SHA256 |
|---|---|
| research/taskcard18/solvers.py | 27f8753b51b635c2792157b46c79d5fa00995817cefa6eeab942407abdbd800f |
| research/taskcard18/baselines.py | 082f087df00cf2e159e5fba846dce967672f98d8ca155cb4f4b472f7f78bc4ca |
| research/taskcard18/workflow.py | d9e51c121af2262623a1aa8046fc0627fe88ba79a1f9bab2f43fb07289e8753d |
| doc/taskcard18/CLOSEOUT_PROTOCOL.json | a794d6a0efef81acf9d69b3429d8fe98c9d18131b8412ca70337fa4cbaebfa61 |

因此本修复没有改变三个拟合/执行源、统计目标、CV边界、方法版本、优化证书或迭代预算。

## 2. 回归证据及独立复跑

新测试运行实际bootstrap提取及导入代码，以纯Python wheel和模块替身在隔离子进程中建立以下前置条件：runtime尚不存在，已在PYTHONPATH中，其sys.path_importer_cache条目为None，且sklearn尚未预载。Windows测试仅替换主机与SOABI门，不替换解包、缓存失效、版本检查或私有导入路径检查。

读取RED.txt，确认失败先输出NEGATIVE_RUNTIME_CACHE_CONFIRMED，再由实际bootstrap抛出与HPC已知日志相同的sklearn版本不符ValueError。读取GREEN.txt，确认同一用例在单行修复后通过。测试同时要求实际选中替身版本1.5.2且模块路径位于新runtime内，覆盖行为结果，不是仅断言源码出现某个调用。

审查者独立运行：

    .\.venv\Scripts\python.exe -B -S -m unittest research.taskcard18.test_cluster -v

七项全部通过，其中包含本次隔离子进程回归及原平台/路径拒绝边界。另读取FINAL_TESTS.txt，确认主执行者的37项限定测试通过；两组检查分列，不合称独立运行了44项。本轮没有实际Linux数值导入或新拟合。

## 3. 已知HPC事实与机制解释的界线

本轮HPC已知事实是用户提供的prepare40887在训练前以sklearn版本检查错误退出，以及后续依赖作业等待。实际HPC导入的sklearn版本值和模块__file__尚未取证。

本机复现证明“不存在runtime路径的负查找器缓存未失效”能够通过原代码触发同类错误，且单行失效调用修复了该受控机制。它不能倒推真实HPC已确定导入某个具体旧版本、某个共享目录，或排除预载模块等其他环境因素。原版本和私有路径门仍会阻止不符合要求的真实环境。

ISSUE.md已经明确区分HPC观测、受控复现和未验证范围；README没有将本机回归写成真实集群成功。修复后的实际prepare仍须通过Linux运行时、依赖、限定测试及输入验收，任务卡18保持IN_PROGRESS/DRAFT。

## 4. 交付指令核对

README使用实际嵌套根/public/home/xjh/xmx/tensor-sparse-regression-main/，将20261028修复包放在全新独立目录，保留20261026失败现场；先调用旧包自身的cancel入口取消其记录的活跃旧链，再对新包submit。新包status、cancel、resume、bundle均使用显式bash -l和一致日期路径。

该安排不搬入旧runtime或本机拟合缓存，不覆盖失败现场，不绕过依赖检查，也不将旧链失败视为新方法结果。未发现阻止本次限定修复打包的确定问题；最终ZIP及真实HPC执行仍分别以打包验收和实际运行记录为准。
