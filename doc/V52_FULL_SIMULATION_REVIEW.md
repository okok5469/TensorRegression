# v5.2-search-final 完整模拟验收与实证交接分析

日期：2026-10-15（北京时间）。用途：交付负责人、队友及 ChatGPT 助手继续推理；本轮未发送消息、调用外部助手、启动实证或变更算法。

## 1. 决策建议

**工程验收通过；建议保留已批准的 D2 / v5.2-search-final 冻结版本，交给队友开展同版本实证验证，同时暂缓“新版全面改善预测性能”的对外表述。** 与旧 v5-original 同输入比较，新版完全恢复从138/700增加到182/700，但平均PMSE从2.236761升至2.381274（变差6.46%）；去重后这一取舍仍存在。改善主要集中在n=400，不能用整体恢复率掩盖n=100和低信噪比的劣势。

| 问题 | 判断 | 条件与边界 |
|---|---|---|
| 算法是否已正式发布 | 任务卡5已批准，本地manifest为released | 本次补齐模拟验收，不重复发布、不新建commit/tag |
| 本批模拟能否作为完整结果交付 | 可以，原生merge成功 | 700任务、2800方法结果，产物一致性及规定数值检查通过 |
| 是否同步队友跑实证 | 建议同步冻结版本及本报告 | 先核对各项目源码、输入、配置和原划分，独立新版目录运行 |
| 能否宣称算法全面改进、普适优越 | 不能 | 旧版总体预测更好，新版恢复改善集中，困难场景未解决 |
| 若“正式发布”指论文定稿或对外性能结论 | 建议等实证及科学论证完成后决定 | 工程released身份不等于论文证据链已闭合 |

既定验收计划没有“所有场景胜出”或“必须优于旧版”的发布硬阈值，不应事后修改验收规则；完整性通过也不证明科学优势。未来若调整caps/MSIC，应作为新研究版本预先冻结验证，不能根据本批测试表现改规则后继续沿用v5.2标签。

任务卡4及早期文档中的“待批准、700未运行”是历史状态；正式发布以任务卡5及独立release manifest为准，本次完成状态以原始产物及当前integrity报告为准。历史研究VERSION及旧结果不改标签。

## 2. 完整性、版本与核验范围

结果目录：`numerical_simulation/outputs_v52_search_final_full/`。

plan_id：`30f231df0327e58331a18580dc8a03caeb0872dc991e57c42b095556100a80f6`。

- full、50次重复、14场景标签；方法为adaptive/oracle/lasso/glasso。每场景每方法50行，共56个汇总组合。
- 全部38份plan源码/依赖raw SHA256匹配现场；其中32个core同时匹配正式release manifest。BCD raw SHA256为`db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`。
- 原生load_plan及700次read_task独立核验通过；原生merge再次执行完整校验后成功。700 JSON与700 NPZ成对，task0–699、配置、场景/replicate及种子映射完整，无锁或额外task文件；2800个(task_id,method)唯一，无非有限指标。
- 700份SLURM JSON签名、关联log哈希及task record哈希均匹配；Gate签名与task0两份原产物哈希匹配。作业36235为单任务task0严格校验后skipped；36255为其余699个completed。另有早期task0 log无配对job JSON，未用其补造成功记录；已有700份有效记录覆盖全部任务。
- 保存环境一致：Linux/x86_64、Python3.9.25、NumPy2.0.2、SciPy1.13.1、OpenBLAS0.3.27，四项BLAS/OMP线程均为1；cluster_preflight_gate为PASSED。
- 正式manifest为released、approved_by_user=true、release_git_commit=null；Git HEAD不是新的发布commit。交接以完整指纹和冻结patch为依据。

**核验通过的精确含义：** 原runner核对plan/record/NPZ校验和、源码/保存环境身份、配置映射、有限B与指标、支持边界、B在报告支持矩形之外为零、B_true/noise输入指纹；从NPZ重算支持指标及REE，并验证RSS=2n×loss。

**未独立核验的内容：** 没有重生700组训练X/y或测试流，没有独立重算PMSE、signal MSE、测试R²，没有以X/y重新检验训练RSS、OLS最优性、设计矩阵秩或条件数。因此不能写成“全部预测指标及OLS最优性已独立重算通过”；环境/哈希一致证明记录绑定关系，不等于本轮独立观察集群运行进程。

本机Python3.12/NumPy1.26.4/SciPy1.16.1只做离线读取及原生merge。merge按原设计不要求本机等同worker环境；未调用worker、伪造环境、改plan、重拟合、替换结果或升级依赖。历史发布回归测试未在本轮重跑，不能计为新增测试。

已成功执行下列原生汇总命令；已有汇总后该入口拒绝覆盖，请勿重复执行：

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD"
python -B -m numerical_simulation.known_s_cluster.run merge --output-dir numerical_simulation/outputs_v52_search_final_full
```

## 3. 设计及统计口径

基准为n_train=200、n_test=500、dims=(32,20,25)、真矩形支持(4,3,5)、q_true=60、rho=(.3,.3,.3)、SNR=2、sigma=1、continuous支持。各组只改变预声明参数，不是样本量×维度×相关×SNR的完整析因组合。保留global MSIC alpha=1.5/gamma=1/one-SE scale=1、mode alpha=1；adaptive未传真实规模。

**14标签对应10种不同参数配置，700条数据任务只有600组唯一完整输入指纹。** sample_size=200（task300–349）、dimension=32x20x25（450–499）、correlation=.30（600–649）共享同50份X_train/y_train/B_true/train_noise与测试种子，预测指标逐例相同。这是冻结种子设计允许的跨组重复，不是漏算。仅X_train指纹唯一数为550；不同SNR可共享X而改变响应。

按每场景的50次重复描述；不能将700条任务或600种输入当作一个独立同分布样本池。跨场景总均值仅作现有标签权重下的描述。PMSE为含噪测试响应均方误差，越小越好；下表±为重复间样本标准差，不是均值标准误。Oracle知道真支持，只作诊断参照，不是可部署方法，也不是任意情形的风险下界。

## 4. 全14场景结果

| ID / 场景 | Adaptive PMSE ± SD | Oracle PMSE | Lasso PMSE | GLasso PMSE | Adaptive Exact / 50 | q中位数 [最小,最大] |
|---|---:|---:|---:|---:|---:|---:|
| 0 / baseline | 2.0121 ± 0.6773 | 1.3934 | 2.8695 | 3.0211 | 16/50 | 60 [1,125] |
| 1 / SNR=.5 | 1.9264 ± 0.6674 | 1.4467 | 1.5562 | 1.6377 | 0/50 | 2.5 [1,125] |
| 2 / SNR=1 | 2.3021 ± 0.6793 | 1.4465 | 1.9931 | 2.1013 | 0/50 | 8.5 [1,125] |
| 3 / SNR=2 | 2.0868 ± 0.6122 | 1.4408 | 2.9504 | 3.0145 | 11/50 | 60 [1,125] |
| 4 / SNR=4 | 1.8466 ± 0.8605 | 1.3995 | 4.6328 | 4.8740 | 32/50 | 60 [1,125] |
| 5 / n=100 | 6.4420 ± 1.5742 | 2.6347 | 3.0420 | 3.0976 | 0/50 | 48 [3,64] |
| 6 / n=200 | 2.0366 ± 0.5825 | 1.4465 | 2.9166 | 3.5811 | 10/50 | 60 [2,125] |
| 7 / n=400 | 1.1895 ± 0.0922 | 1.1856 | 2.7381 | 2.9572 | 49/50 | 60 [48,60] |
| 8 / dim=16×12×15 | 1.8103 ± 0.6029 | 1.4327 | 2.7063 | 2.8866 | 25/50 | 60 [32,125] |
| 9 / dim=32×20×25 | 2.0366 ± 0.5825 | 1.4465 | 2.9166 | 3.5811 | 10/50 | 60 [2,125] |
| 10 / dim=64×30×40 | 2.7903 ± 1.3982 | 1.4389 | 2.9856 | 3.0210 | 5/50 | 60 [1,125] |
| 11 / rho=0 | 2.9482 ± 1.7811 | 1.4490 | 3.0450 | 3.0721 | 10/50 | 60 [1,125] |
| 12 / rho=.3 | 2.0366 ± 0.5825 | 1.4465 | 2.9166 | 3.5811 | 10/50 | 60 [2,125] |
| 13 / rho=.6 | 1.8738 ± 0.4676 | 1.4254 | 2.5525 | 2.9206 | 4/50 | 28.5 [2,125] |

Adaptive平均PMSE在11/14标签小于现有Lasso及GLasso实现，在SNR=.5、1及n=100三个标签均更差；不能视为11次独立优势验证。Oracle在本批14标签平均PMSE均更低。最大维度虽均值优于现有对照，但不确定性较宽。

以下是同任务Adaptive−Lasso PMSE差，负值更好。区间为各场景50个配对差的t区间，均值±t(.975,49)×SD/√50；仅作事后描述，未校正多重比较，不充当发布门槛：

| 场景 | 平均配对差 | 描述性95%区间 | Adaptive较小次数 / 50 |
|---|---:|---:|---:|
| SNR=.5 | +0.3701 | [+0.1872,+0.5530] | 7/50 |
| SNR=1 | +0.3090 | [+0.1154,+0.5026] | 15/50 |
| n=100 | +3.4000 | [+2.9432,+3.8568] | 1/50 |
| n=400 | -1.5486 | [-1.6002,-1.4970] | 50/50 |
| dim=64×30×40 | -0.1953 | [-0.5858,+0.1953] | 31/50 |

## 5. 与旧版本的严格同输入比较

旧对照来自`numerical_simulation/outputs_known_s_full_50/`，plan_id为`de8b69ff4ed056f3e385b4f80a4722c8eeaa4af89969bad23d845f4d6f3a8852`。其完整32 core逐项匹配`doc/algorithm_versions/v5-original/VERSION.json`，因此可标明v5-original；不能将该目录叫作v5.1。

两plan的700个task完全相同；700对JSON的task、environment、四项input_fingerprints、actual_snr完全相同，每条source指纹均匹配所属plan。38份源文件仅bcd.py不同：旧`4f9486d32f01a2798a55fbe7216d6a05d789f56f8f85608f03598c2926bce6fa`，新`db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`；Oracle除runtime外全部指标700/700逐对相同。可按task_id比较保存结果；本轮未对旧700份NPZ重新执行完整原生审计。

| 指标 | 旧v5-original | 新v5.2 | 解释 |
|---|---:|---:|---|
| PMSE，700任务标签均值 | 2.236761 | 2.381274 | +6.46%，更差 |
| signal MSE均值 | 1.228679 | 1.371188 | 同样更差 |
| REE均值 | 0.893679 | 0.923631 | 同样更差 |
| 完全恢复 | 138/700 (19.71%) | 182/700 (26.00%) | +44次、+6.29个百分点 |
| q均值 | 50.1014 | 53.5486 | 不是单纯统一缩小模型 |
| mode F1均值 | 0.751917 | 0.753593 | 改善很小 |
| 平均fit秒 | 25.0282 | 31.3756 | +25.36%；非受控冷启动试验 |
| 去重600种输入：PMSE | 2.274498 | 2.438721 | +7.22%，结论不变 |
| 去重600种输入：Exact | 120/600 (20%) | 162/600 (27%) | +42次 |

按PMSE绝对差≤1e-12定义相同：700对中新版更低/相同/更高为99/507/94；Exact新增57、丢失13、相同630。去重后为89/427/84，Exact新增53、丢失11；600种输入仍不能直接称统计独立。不能只报告新增成功而隐去丢失成功或误差恶化幅度。

| 场景 | PMSE旧→新 | Exact旧→新（分母50） |
|---|---:|---:|
| baseline | 1.990669→2.012130 | 15→16 |
| SNR=.5 | 1.837097→1.926368 | 0→0 |
| SNR=1 | 2.249203→2.302133 | 0→0 |
| SNR=2 | 2.074656→2.086787 | 12→11 |
| SNR=4 | 1.764011→1.846581 | 35→32 |
| n=100 | 5.196560→6.442026 | 0→0 |
| n=200 | 2.010341→2.036588 | 9→10 |
| n=400 | 1.297588→1.189516 | 1→49 |
| dim=16×12×15 | 1.780057→1.810260 | 26→25 |
| dim=32×20×25 | 2.010341→2.036588 | 9→10 |
| dim=64×30×40 | 2.700777→2.790346 | 5→5 |
| rho=0 | 2.573943→2.948160 | 12→10 |
| rho=.3 | 2.010341→2.036588 | 9→10 |
| rho=.6 | 1.819068→1.873761 | 5→4 |

新版PMSE均值仅n=400标签改善（约8.33%）；n=100恶化约23.97%，rho=0恶化约14.54%。这是v5-original到v5.2的组合变化，包含fixed BCD修复及A2规模搜索，**不能用来单独估计A2相对v5.1的因果效果**；需要那一问题的证据时应引用既有A2独立验证，而非混用此全量对照。

旧`outputs_full_v5_adaptive`只有10重复及base_config等记录，缺少完整plan/source/input身份，最多作历史参考，不纳入本次严格配对。不能拼接不同重复数或把旧结果换标为新版。

## 6. 机制与适用边界

**n=100的真规模被旧caps排除。** 50次s_max全部(4,4,4)，真形状(4,3,5)。入口将这些caps传入adaptive和精修，第3模真规模5不可达，尽管q_true=60≤Q=floor(.7×100)=70。A2只补全旧caps内的整数三元组，不能解除该限制。该组Exact=0/50、PMSE6.4420；16/50选(4,4,4)、q64、q/Q=.9143，接近容量；q范围3–64。不能仅归罪MSIC，也不能称已经考虑所有维度和容量允许的模型。

**弱信号仍不可靠。** SNR=.5和1的Exact均0/50；q中位数2.5及8.5，q=1分别14及10次，同时也出现q125的大模型。欠选/过选并存，不能用单一原因概括。原JSON未保留每个候选的支持/得分，因此不能从最终模型推断真支持已经进池，或新惩罚一定有效。本轮未改caps、MSIC或按测试结果选规则。

**n=400证据较强，但不能外推。** 新版49/50选真形状且完全恢复，另1次(4,3,4)，PMSE1.1895接近Oracle1.1856；rho=.6却仅Exact4/50，最大维度仅5/50。预测均值较好与支持位置可靠是两个问题；同规模也不保证同位置。

**规模与轨迹指标。** Adaptive全部700条记录q≤Q；q=1为34条，q<60/等于60/>60分别298/203/199条，Exact仅182条。16条q/Q≥.9均为n=100，标记仅描述、不删模型。q_hat是三模支持长度乘积，对Vector Lasso通常为全模支持笛卡尔积（如16000），不等于其实际非零系数数量，不能据此夸大稀疏性差距。

Task78和369的adaptive loss_history分别出现+.01904、+.04381上升，同时SIC从61.0881降到36.4128、46.0536降到37.0817。adaptive会改变规模并按信息准则选择，原adaptive结果本身也属于最终候选池；固定规模BCD接受损失非增的性质不能外推到全部adaptive轨迹。这两例不能单独当作fixed修复失效的证据，也不应宣称所有公开轨迹都单调。

**成本。** 本批保存的平均fit时间/累计fit小时：adaptive31.3756秒/6.1008小时，oracle0.0042秒/0.0008小时，lasso176.0423秒/34.2305小时，glasso1372.6431秒/266.9028小时。累计fit计时不是作业总耗时，也不是严格CPU时间。Adaptive在n=100、基准、n=400、最大维度约5.97/25.00/110.86/62.91秒。旧新环境记录未含具体CPU型号、节点负载及调度细节，跨版本成本增加不能全部归因于算法；也不能外推至所有Lasso/GLasso实现。

## 7. 实证交接、启动条件与完成标准

建议交付正式冻结D2，保持原MSIC、caps、初始化及公共接口；以实证检验适用性，不保证提升预测。交接附件为：

1. 本报告与新批`summary_results.csv`、`integrity_report.json`；复核配对差可加`raw_results.csv`。
2. [正式manifest](releases/v5.2-search-final/RELEASE_MANIFEST.json)、[完整32源码指纹](releases/v5.2-search-final/SOURCE_SHA256.txt)、冻结完整core。
3. [D2最小patch](team_sync/v5.2-search-final/search-final.patch)及[升级说明](TEAM_V52_RELEASE_NOTICE.md)。只有完整匹配v5.1基准的项目才应用D2增量；v5-original项目需先核对并完成v5.1修复；已为D2的不重复apply。
4. [实证重跑协议](V52_EMPIRICAL_RERUN_PLAN.md)。旧日期中的“full未完成”由本报告更新，但项目自己的输入/划分/环境Gate保留。

每项目先核对生产导入来自自己的repository/src、完整32 core匹配D2，保存adapter/预处理代码、输入、ID与顺序、响应/协变量、特征/模态顺序、外内划分、参数、种子和环境指纹。沿用各自已确认环境，记录差异，不擅自升级依赖、改响应、换划分或删困难被试；外层测试集不能用于选caps/MSIC/初始化。

全部ABESS外层训练折重新拟合预处理、生成caps和候选、搜索位置、选阶、OLS及预测；内层若调ABESS参数，每个内训练折也要独立完成。target_sparsity=None，不能复用旧B、支持、q、模型缓存或更改旧断点身份。旧对照只有在独立代码、配置、输入、预处理、划分、种子及环境逐项一致时才可保留明确来源复用；现行LEMON入口默认重跑五模型。

LEMON保留145人、X=(145,61,16,5)、log(TMT_5)、原8列协变量、5折×3重复及固定种子；保留Covariates Only/Ridge/Lasso/GLasso。完整目标75 fits、2175条OOF，逐subject/repeat/model唯一且有限；与旧划分逐ID核对。报告每折q/Q、训练测试差、支持稳定性，以及相对仅协变量基线的EEG增量。旧LEMON对应v5-original，不能当成v5.1结果；本批n=100局限尤其提示小训练折需要谨慎，但不能用模拟误差替代LEMON实际结果。

LEMON既有入口如下，供项目Gate核对及启动确认后使用；本轮未执行：

```bash
export PYTHONPATH="$PWD/src"
PYTHONDONTWRITEBYTECODE=1 python -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --audit-only --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final
# 核验audit、输入/划分/源码/配置并确认启动后：
PYTHONDONTWRITEBYTECODE=1 python -m empirical_analysis.lemon.incremental_value.run --config empirical_analysis/lemon/incremental_value/config.yaml --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final
```

HCP-YA/NKI-Rockland真实入口、subjects、模态和配置未在本仓库验证；队友需在各自原工程完成同样核对，使用独立v5.2目录，不拼造CLI或套用LEMON参数。交回完整OOF、各折支持/q、训练测试误差、完成/失败记录及所有身份指纹，再评定实证结论。

可转发文字（尚未发送）：

> v5.2-search-final（D2、保留原MSIC）完整模拟已通过700任务/2800方法结果的产物一致性验收；完全恢复比例提高，但总体预测较旧版变差，n=100及弱信号局限详见报告。请先核对自己工程的源码、输入、原划分及配置，在独立v5.2目录重新训练全部ABESS折，保留仅协变量和原对照，不复用旧支持或缓存。请先交回audit/版本信息，再按项目确认的流程启动；本次目标是验证实证适用性。

## 8. 给ChatGPT助手的推理请求

请基于本报告及附件独立评议，区分工程发布、模拟科学证据及实证启动，保留负面结果与不确定性：

1. 面对恢复率增加、预测误差变差的取舍，维持已获批冻结版本并推进三实证是否合理？哪些对外主张应暂缓？不要把工程released等同于论文可投稿。
2. n=100的caps结构限制、低SNR欠选/过选、高维和相关条件下位置恢复较弱，如何限定适用范围？不要用本批测试表现指定新的最优caps/MSIC。
3. 700任务中的重复权重、600种输入和不同场景的依赖，如何影响统计表述？如何避免用整体Exact掩盖n=400以外的退步？
4. 三实证还需哪些可判定证据才能从“启动”推进到“验收并支持科学结论”？审查数据泄漏、容量、完整OOF、仅协变量基线与支持稳定性。
5. 若建议追加实验，应与既有未完整MSIC确认和本次正式批次隔离，先冻结新设计；不修改、重标或事后挽救既有结果。

本报告没有新增外部文献检索；若扩展最新方法或论文比较，应另行核实原始来源，不能从当前四方法结果推断优于所有现有方法。

## 9. 证据索引

- [2800行原始方法结果](../numerical_simulation/outputs_v52_search_final_full/raw_results.csv)
- [56行场景方法汇总](../numerical_simulation/outputs_v52_search_final_full/summary_results.csv)
- [原生完整性报告](../numerical_simulation/outputs_v52_search_final_full/integrity_report.json)
- [冻结计划](../numerical_simulation/outputs_v52_search_final_full/plan.json)
- [真实单任务Gate](../numerical_simulation/outputs_v52_search_final_full/cluster_preflight_gate.json)
- [任务卡5](../任务卡/任务卡5.md)、[事前验收计划](V52_SIMULATION_ACCEPTANCE_PLAN.md)
- [旧批计划](../numerical_simulation/outputs_known_s_full_50/plan.json)、[v5-original完整身份](algorithm_versions/v5-original/VERSION.json)
- 核验代码：`numerical_simulation/known_s_cluster/run.py`的load_plan/read_task/status/merge；指标定义见`numerical_simulation/metrics.py`；caps传递及规模枚举见`src/tensor_abess/tensor/{abess,bcd}.py`。

本批`paired_fixed_vs_adaptive.csv`与`search_gap.csv`只有表头；原生integrity明确applicable=false，因为本批无fixed方法，不能用空表推断已知规模搜索差为0。

以下SHA256用于本次交接核验，不含本报告自身：

| 文件 | SHA256 |
|---|---|
| `numerical_simulation/outputs_v52_search_final_full/plan.json` | `06a72ad73c64dad2afd63fc24c5abba1878b9eec083720a230e10d4b1015dd42` |
| `numerical_simulation/outputs_v52_search_final_full/raw_results.csv` | `ca85c4de08dc38bc237337a00f89e505552dd45d7040eff30af1980b01b2e25e` |
| `numerical_simulation/outputs_v52_search_final_full/summary_results.csv` | `285fc7f85dcc5b0029c21195ffc9fffc06c74efee0f94286a35fb3f518ca659a` |
| `numerical_simulation/outputs_v52_search_final_full/integrity_report.json` | `29fbdb92fcdffffd19d77c1a6c6c3c2ab7996168fb494688f22bf4fea0f161f9` |
| `numerical_simulation/outputs_v52_search_final_full/cluster_preflight_gate.json` | `9d86b85aea088a6ce8510801ec6c614592b1962a0dc1aec06973b25265d8a0f5` |
| `doc/releases/v5.2-search-final/RELEASE_MANIFEST.json` | `b415a151674f1ee0400acbd000aa732e2c99343c7ad4faa48d7970b2cfe56074` |
| `doc/team_sync/v5.2-search-final/search-final.patch` | `23524a12a3ccd502b058e5eec7ef984abae43bccc085d467ed292b1d3585b3db` |
| `numerical_simulation/outputs_known_s_full_50/plan.json` | `7aef30c616d7083c2584e41b8af64465573388606873df8f2d4445a273f94d84` |
