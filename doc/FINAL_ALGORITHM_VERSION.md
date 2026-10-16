# 最终算法版本推荐与冻结记录（任务卡4）

## 最终推荐及批准状态

推荐D2：`v5.2-search-final`，保留v5.1的fixed BCD修复、旧caps及原MSIC，仅将精修候选规模改为旧caps内满足容量约束的全部合法整数三元组。A2搜索验证完整且通过事前门禁；新MSIC独立确认预算停止、证据不完整，因此按冻结规则保留原MSIC，不推荐D3。

候选材料已冻结并可审阅，`research_only=true`、`release_approved=false`，状态为等待用户最终发布批准。当前生产`src/tensor_abess`仍为v5.1，未提交、推送、创建分支/worktree、替换队友生产或启动700正式模拟及三个实证。任务卡4验证决策和准备交付完成，不把独立MSIC确认或正式后续实验写成完成。

## 七项决策回答

1. A2值得在当前验证和成本门槛内采用：主90 Exact15/90→36/90，PMSE2.500011→2.474219（−1.03%）；主要收益是恢复能力，预测收益较小。
2. 两种新增形状均改善，但新增grid的有效差异集中于n400；n100/200模型相同。rho=.6只验证n200兼容性，不能推广为n400强相关、新维度或非矩形信号也已验证。
3. 本轮未显示明显过拟合增加：主90过选46→25、欠选同27；独立成本估计增加14.88%、实测冷拟合增加10.37%，均满足事前50%上限，不满足额外成本≤10%的更严条件。
4. 保留原MSIC。开发评价通过筛选，但80数据独立确认在3600秒预算内仅68组完成，最终保留71份评价产物仍不足完整验收，fresh refit未运行；不凭部分结果批准新规则。
5. 推荐版本为D2/v5.2-search-final；32核心文件仅bcd.py私有规模枚举不同，旧caps、fixed/adaptive主体、Group-BESS、OLS、MSIC及公共签名/默认值保留。
6. 本轮157项限定测试通过，2040受保护文件哈希差异0；包含最终候选真实公共调用回归和独立进程桥接。未验证全仓库、集群Python3.9/NumPy2真实运行、HCP/NKI入口或三个实证重训。
7. 队友按最终批准后的最小patch统一目标src及完整32源码指纹；所有Tensor-ABESS正式模拟与实证训练折须重搜支撑、重新拟合。独立full脚本/700计划及三实证同步材料已准备，本轮不启动；旧结果保留原身份。

详细数值、局限和来源见[A2_EXISTING_EVIDENCE](A2_EXISTING_EVIDENCE.md)、[A2_INDEPENDENT_VALIDATION](A2_INDEPENDENT_VALIDATION.md)、[A2_VS_A0_DECISION](A2_VS_A0_DECISION.md)、[A2_MSIC_DECISION](A2_MSIC_DECISION.md)。

## 源码及复现身份

观察到的仓库HEAD及本次patch基准为`d0c3547704fbaf5d90a3e5423568768a1ba11759`；这是既有提交，未为新候选创建commit。v5.1快照的历史基准`d4e28969ac480ba6efc01bcc985a53d179e7ed31`单独保留。候选身份以[VERSION.json](algorithm_versions/v5.2-search-final/VERSION.json)的32个raw/canonicalLF SHA256为准，不能把既有HEAD称为候选已提交的版本。

| 项目 | SHA256 |
|---|---|
| v5.1生产BCD raw | c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737 |
| D2候选BCD raw | db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7 |
| D2候选BCD canonicalLF | eb4f01626e0108f4cf87a604835aab7fa99866f484abaea23ce54b8911f063d0 |
| search-final.patch | 23524a12a3ccd502b058e5eec7ef984abae43bccc085d467ed292b1d3585b3db |
| run_v52_search_final_full.sh | 5e84b81d53e60f0e6c59011762c2c7432bf856d28326b90122cf0e1e5dc0138b |

冻结候选位于`doc/algorithm_versions/v5.2-search-final/src`，完整规模不等于全部支撑位置穷举或dimension+Q合法空间，仍受训练GLasso生成的旧caps限制。无合法域返回空列表，删除无作用的私有window参数；不伪造(1,1,1)。公共接口不增加研究候选预算参数，规模较大的部署须预先核对成本。

原参数保持mode alpha1、global alpha1.5/gamma1/scale1；初始化BIC c3、k_max1和其余full设置/种子保留。所有候选生成、支撑搜索与评分只依赖训练X/y；真实支撑和独立测试仅用于模型选定后的评价。共享缓存walltime不能与独立成本混用，原MSIC的启发式one-SE容差不是真实CV标准误。

## 本轮限定测试与证据

| 范围 | 通过数 | 结果及来源 |
|---|---:|---|
| 候选、数值审计、稳定性、MSIC研究及A2 runner/analysis unittest | 106 | 更正后的命令退出0，17.813秒 |
| 最终D2来源：tensor BCD、fixed BCD回归、公共数值模拟pytest | 29 | PYTHONPATH首项为冻结D2源码；80.75秒，退出0 |
| A2 MSIC决策及80确认控制器unittest | 16 | 7+9项，独立确认启动前通过 |
| 最终候选静态与独立进程同数据拟合桥接 | 6 | A2_RUN_BRIDGE=1；24.383秒，退出0 |
| 合计 | 157 | 限定范围，非全仓库验收 |

106项命令为`python -B -m unittest tests.test_candidate_search_core tests.test_candidate_study_analysis tests.test_candidate_numerics tests.test_support_search_stability tests.test_stability_run tests.test_msic_after_search tests.test_msic_confirmation tests.test_msic_refit tests.test_a2_final_analysis tests.test_a2_final_validation numerical_simulation.candidate_search_study.test_run -q`；29项命令为`python -B -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py tests/test_numerical_simulation.py -q -p no:cacheprovider`。16项对应`tests.test_a2_msic_decision`及`tests.test_a2_msic_confirmation`；6项对应`tests.test_a2_release_candidate`并启用A2_RUN_BRIDGE。pytest仅使用既有隔离目录，四BLAS变量均1，未安装或更新环境。

本机Python3.12.8/NumPy1.26.4/SciPy1.16.1；全部32候选文件通过Python3.9 AST解析，此为静态兼容证据。集群Python3.9.25/NumPy2.0.2/SciPy1.13.1未实际运行验证；可选sklearn缺失的test_phase_c未纳入通过范围。

桥接在独立进程核实真实import来源，n400/dims(6,6,7)同X/y及明确c3/BIC配置下，最终默认grid与研究显式legacy/full的B、支持、caps、RSS、score及轨迹逐值相同；不能称所有场景已逐个重拟合最终最小版本。所选200模型OLS审计完整、秩亏0、最大condition40.430603，不代表全部候选矩阵同样完成秩审计。

源码对比只改一个逻辑点；公共接口和数值回归通过后重新审查网格完整性、训练隔离及成本定义，无额外重构。2040受保护源码/旧产物逐文件SHA256核验差异0，patch/sh原始指纹再次相符。普通LF/CRLF副本实际patch应用及Bash静态/mock组件证据保留在[MANIFEST](team_sync/v5.2-search-final/MANIFEST.json)；不等于已部署或真实集群运行。

## 后续批准与交付

七份任务卡交付物现已齐备；[FINAL_TEAM_SYNC](FINAL_TEAM_SYNC.md)、[FULL_EXPERIMENT_PREPARATION](FULL_EXPERIMENT_PREPARATION.md)和[待审批D2包](team_sync/v5.2-search-final/README.md)给出源码同步、依赖/身份核对、独立输出、700数据任务/2800四方法结果及三实证训练折重搜要求。当前只需用户决定是否批准D2为正式版本；批准不自动授权本轮执行后续700任务或三实证重训。
