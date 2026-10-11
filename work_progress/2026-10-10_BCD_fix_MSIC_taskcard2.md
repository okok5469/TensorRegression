# 2026-10-10：任务卡2，BCD修复、MSIC研究与版本同步

## 状态和最终决定

执行依据：empirical_analysis/任务卡/任务卡2.md；用户授权继续执行、后续自动审批并在结束后写入work_progress。源码正确性修复与统计选阶研究独立保存；没有创建分支/worktree、提交或推送远端，没有联系队友，没有升级集群，没有覆盖原正式结果。

- P0已完成：v5.1-bcd-fix，正式接受损失非增，返回best visited；原Group-BESS目标、MSIC默认、数据生成、GLasso/Lasso/Oracle保持原样。
- P1已完成探索研究：54/54版本任务、27配对数据集、333方法评价；9场景每场景3重复。未批准新正式v6。
- P2已完成小规模配对复验及历史seed2026复现；完整700任务模拟尚未完成，提供原集群环境可运行入口。
- P3已完成本地Git patch、版本指纹、测试命令、CHANGELOG及HCP/NKI迁移说明；未发送消息或远端推送。

主报告：doc/BCD_FIX_REPORT.md、doc/MSIC_ABLATION_REPORT.md、doc/TENSOR_ABESS_TEAM_SYNC.md。

## 问题根源与最终修复

旧fixed BCD调用cold group_bess_fix而未以外层当前支持作输入，局部交换对它自己的起点改善不等于外层完整张量训练损失改善；旧外层仅支持不同就继续，未保留最优完整OLS状态，最终可能返回较差末态并反复循环。

本次仅修改src/tensor_abess/tensor/bcd.py：保留原cold确定性探索路径，明确区分探索状态与正式最优接受状态；每mode候选都做完整矩形OLS统一比较。正式loss_history仅初始+超过64eps数值容差的best改善；visited_loss_history保存所有完整候选，允许探索暂时回升。best_supports、B、loss、RSS、索引属于同次拟合，最终返回全部已访问有效完整状态中损失最小者。完整sweep边界支持重复则停止，不能因一轮暂未改善而丢弃后续恢复路径。

新增fixed诊断字段best_iteration、best_rss、stop_reason、visited_loss_history、accepted_support_history，均有默认值；函数参数不变。best_iteration是visited列表索引，n_cycles是实际sweep数，不能把loss_history长度当旧外循环数。保证best visited不是保证非凸搜索达到所有矩形支持的全局最优；adaptive改变规模时RSS也不要求始终非增。

## 曾发现失败，如何处理

首个warm-only+逐mode拒绝的草案虽15局部测试通过，却使原seed2026真实支持恢复测试失败。立即停止该草案研究，保留拒绝快照doc/algorithm_versions/v5.1-warm-only-rejected；outputs_msic_ablation/ABORTED.json和旧full_preflight不是最终结果。没有削弱原测试，也没有因为该草案恰让LEMON这折选q1而采用。

诊断显示该例warm起点完整loss约0.00956，而Group-BESS现有tau在三mode约0.0442/0.0298/0.0173，大于可下降的全部损失，warm交换受原阈值限制。改变Group-BESS数学定义不在本修复范围。因此按任务卡3.2B允许的显式非单调探索+best接受分离，保留旧搜索机会。最终原恢复断言通过。

自动审批有一次apply_patch写报告超时，未认定不安全；改由已授权workspace中的显式exec写文档完成。PowerShell5管道默认ASCII导致一次文档追加出现问号，已暂停相关写入、改显式UTF8并回读修正；实验与源码未受影响。full预检第一次缺单线程环境变量即被运行器拒绝，未生成错误计划；补齐脚本原有的四线程变量后预检正常启动。所有异常有记录，未混用中止产物。

## 测试与版本身份

最终新增10 unittest＋原5 pytest=15通过；原LEMON模型12＋数值模拟14=26通过；研究选择7＋计划5=12通过，共53限定自动测试。不宣称仓库全测试通过，未重跑所有外部数据流程。pytest仅安装到doc/algorithm_versions/test_tools隔离目录，未改现有venv包或集群；此可再生工具不在team patch中。算法生产依赖不变。

Git HEAD=d4e28969ac480ba6efc01bcc985a53d179e7ed31。
原完整32源码：doc/algorithm_versions/v5-original/src/tensor_abess及VERSION.json。
最终32源码指纹：doc/algorithm_versions/v5.1-bcd-fix/VERSION.json。
研究候选状态：doc/algorithm_versions/v6-msic-candidate/VERSION.json，release_approved=false，production_defaults_changed=false。

原bcd规范LF SHA256=cff12bd6fa4c144031a41277a3a28fea6044fc7775ec2bd9ada7d582d6efdf51。
最终bcd规范LF SHA256=4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c。
最终patch SHA256=d2e6cac68ad7e37ec131cf2ff3d52eec4e7ec4498d30cea484d513fe2b7b37ee。
旧117525...patch作废，不得使用；INVALIDATED_DRAFT.json只是旧草案的历史记录。

patch仅改bcd.py并新增tests/test_bcd_fix_regression.py，在原版临时副本git apply --check＋实际apply成功并核验内容。当前32核心文件只bcd改变；原32源码快照完整。最终1738个受保护历史文件SHA256均不变。核验文件doc/algorithm_versions/final_integrity_report.json。38个核心/研究Python文件通过Python3.9语法解析；集群sh为LF，bash -n通过。本机实际数值验证环境Python3.12.8/NumPy1.26.4/SciPy1.16.1，未冒称已在集群3.9/NumPy2执行。

## LEMON异常折最终重放

此前诊断正式目录repeat_2_fold_5的fold从1起（完全零起始为repeat2/fold4），见旧DIAGNOSTIC_REPORT；不要把已存目录名称当全零起始。
最终独立输出numerical_simulation/outputs_bcd_fix_validation/：
- v5-original_lemon.json/npz：q80 RSS1.2340920533，已看过的旧外层测试RMSE0.6707709551。
- v51_final_lemon.json/npz：仍q80，RSS1.0674821080，MSIC−324.5481549840，诊断旧外测RMSE0.6199820472。
- q80正式接受RSS=2.217202→1.638195→1.067482；完整探索仍有回升，4sweep后exploration_cycle停止，best_iteration=2。

该折更低训练RSS没有使过选消失，不能以外层RMSE选新参数。只按同一训练候选重评分lemon_train_only_msic_rescore.json：alpha1.5 scale0/.5/1都q80，alpha2/3都q1；它是机制诊断，不是新的正式全LEMON验证。旧warm草案q1/RMSE0.357583已作废，不能作为最终结果。

## 配对模拟与MSIC结果

最终目录numerical_simulation/outputs_msic_ablation_v51_final，plan_id=a47340bbd48545833f6c281d35cfe95141945c47b6214bd97ca2e9f94eaa8f57。
n100/200/400×SNR0.5/2/4，dims32,20,25、true4,3,5（q60）、rho0.3、独立ntest500，每场景3次。seed=20261010+1000scenario_id+rep，test+1e6；CV拆分+2e6。相同生成数据与独立测试流两版本指纹一致，不冒称这些是旧历史seed。

原默认与修复同默认：最终B改变6/27，平均q46.111→49.333，Exact均3/27，PMSE3.061981→3.364051，配对2胜/21平/4负，平均差+0.302069。n100/SNR4/seed20263012训练RSS下降146.996而PMSE增加6.98167，说明训练搜索改善可强化过拟合。3051个global-fixed候选审计best visited、接受单调、RSS和索引一致全部通过。

修复版alpha2/scale0：q42.222，Exact3/27，PMSE2.954613，7胜/18平/2负，较温和的后续候选。
alpha2/scale1：q31.630，Exact2/27，PMSE2.437123，但欠选17/27、q1达7/27；n200/SNR2反而PMSE1.6221→1.9109、Exact2/3→1/3。
alpha3/scale1：欠选22/27、q1达15/27，MacroF1仅0.4394；n200/SNR4 PMSE1.7215→3.7508。不能只看整体平均PMSE采用强罚。

n200九例五折内部CV，按预声明固定规模网格，每内折重新搜索支持后评价，最终全训练重拟合，再看独立测试。CV PMSE2.437419，对匹配同九例默认1.697627，2胜/3平/4负、5/9选q1。强信号三例选q12/1/80，说明内训练支持搜索不稳；不能断言CV理论无效，不能拿9例与其他规则27例均值比较。

候选池重要缺口：真形状(4,3,5)在caps内18/27、实际出现9/27，精确真位置仅3/27；任意形状q60出现18/27不是精确真形状。n100 caps4,4,4排除真实s3=5，即使q60≤容量70也不可达；n400 caps6,6,7容纳真形状，但s2候选{1,2,4,5,6}漏3，不能将n400零Exact归因惩罚。本轮没有擅改该候选逻辑。

raw_results/summary_results/paired_default_diff/rule_overview/candidate_feasibility/study_diagnostics/integrity_report和jobs JSON/NPZ均在最终目录。详细9规则及BIC/匹配CV统计见MSIC_ABLATION_REPORT。

历史seed2026/test1002026另复现，原版与正式task0非耗时指标最大差2.8422e−14；最终修复默认仍q60 Exact1 PMSE1.3250717019。不能以该例相同推断全部full无变化。

## 集群运行、上传与未完成部分

正式v5.1模型已发生变化，原模拟数值仅对应v5-original；采用v5.1需要完整重跑和实证ABESS重训。full五组baseline/correlation/dimension/sample_size/snr：14场景×50次=700任务，adaptive/oracle/lasso/glasso共2800结果；旧对照跨本机/集群指纹未严格确认一致，入口重算四方法。

新根run_bcd_fix_full.sh使用用户既有Python3.9.25/NumPy2.0.2/SciPy1.13.1、myenv、CMO_HPC/xjh，不升级环境；校验最终规范LF bcd指纹。必要重传仅bcd.py与该sh；上轮已上传且一致的known_s_cluster/run.py、profiles.json等原生产依赖沿用，测试及文档可选，不再做ZIP。

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_bcd_fix_full.sh prepare
sbatch run_bcd_fix_full.sh
# 等该批全部任务结束后：
bash run_bcd_fix_full.sh merge
```

prepare首次运行一次；sbatch提交数组，最多64并发、每任务1CPU；merge必须等该批全部结束并完整，三条不是一次不间断跑完。输出numerical_simulation/outputs_v51_bcd_fix_full，与旧known-s正式输出分开。断点续跑用相同计划重提交，不能改源码后resume。

本机独立最终full计划预检目录outputs_v51_final_full_preflight，plan_id=51cf91b654b909298529b358f344e55eae82a6971b2e4640f5b09f5dc1a6cb6a。只执行task0入口预检，不把它作为700任务完整结果或提交远程作业；最终状态另记。

队友A/HCP与B/NKI应同步v5.1正确性修复，保留旧版本结果，在相同划分新输出目录重训ABESS。旧B、支持、选阶缓存不能标为新版本，训练折预处理需重拟合；已验收原张量/划分可复用。对照结果只有严格一致才能复用。team_sync/v5.1-bcd-fix含patch、MANIFEST、TEAM_SYNC、CHANGELOG。

待完成：700任务正式full，LEMON/HCP/NKI修复版实证重训，普适候选规模覆盖的独立研究、多初始化搜索稳定性、更多重复/高维/相关性/真实形状复验。alpha2/scale0推荐作后续研究候选而非正式默认；本轮不采用alpha3、不宣布CV最优、不成立正式v6。

## 最终本机full入口预检

最终outputs_v51_final_full_preflight计划700任务，已完成task0全部adaptive/oracle/lasso/glasso四方法，JSON/NPZ、源码/环境/数据/支持与指标校验通过；再次--resume仅验证并跳过，不重复训练。详细首任务耗时和历史指标比较见doc/algorithm_versions/full_preflight_report.json。本机仅1/700完成，余699未运行，尚未提交集群，不能称正式full已完成。
