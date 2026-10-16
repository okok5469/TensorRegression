# 任务卡3：团队同步更新（研究版，未替换正式算法）

生产src仍为v5.1-bcd-fix；本次v5.2候选搜索代码位于doc/algorithm_versions/v5.2-candidate-search/src。没有改GLasso/Lasso/Oracle或DGP，没有运行700正式full，没有推送、联系队友或替换HCP/NKI代码。

## 代码与边界

四个核心差异：tensor/abess.py、tensor/bcd.py、utils/init.py以及新tensor/candidate_search.py，公共函数原调用仍兼容，新增cap_policy/grid_policy/candidate_budget/显式fixed_candidate_cache。默认legacy保留v5.1路径；研究由独立模块显式选择A0–A4。fixed tensor_bcd函数源码与v5.1相同，数学目标和交换定义未改。

原source快照v5-original、v5.1-bcd-fix，以及旧报告/结果保留。v5.2 VERSION.json给33源码SHA256，release_approved=false。doc/team_sync/v5.2-candidate-search/candidate-search-core.patch相对v5.1制作，仅四文件；在独立v5.1副本git apply --check及实际应用后，全部33文件与研究版本逐文一致。补丁不包含MSIC默认调整。

研究运行器numerical_simulation/candidate_search_study/run.py和worker.py记录严格源码/配置/环境/输入/测试指纹；稳定性、统计交互、数值审计为独立模块。复制或变更代码后不得resume旧plan，跨本机/集群NumPy版本也应重新创建输出计划。Python3.9源码兼容静态检查与本机真实实验分别记录，禁止通过升级集群解决兼容问题。

## 验证器问题与处理

首版验证器误把变q adaptive-stage当作fixed RSS接受轨迹。dev033 q4→2使RSS225.2495→232.4915，但原阶段MSIC54.9126→53.3367，是合法选阶。两独立复现验证125fixed候选全正确。只修正worker审计分类并增加回归测试，算法、数值容差、DGP、种子及core33字节均未改；revision2-stage-objectives。旧历史16/开发35及两稳定性预检保留STOPPED；全新同seed *_final计划重跑，不后处理修补旧结果。

## 两名队友需要做什么

当前先同步已经通过任务卡2验收的v5.1正确性修复，沿用doc/TENSOR_ABESS_TEAM_SYNC.md；研究v5.2/v6不得在没有版本决策的情况下替换正式HCP-YA/NKI-Rockland分析。

本次代码先供独立复核：在单独项目副本或研究源码目录应用补丁并运行对应unittest，不能把v5结果改标签为新版本；不用强制覆盖main或创建新分支。研究入口、必要增量文件、源码快照依赖和Linux命令见numerical_simulation/candidate_search_study/README.md，不需要ZIP，不上传大型旧原始数据或test_tools。

如果之后正式采用候选搜索变化或MSIC变化，应在相同受试者/响应/预处理/外折划分下重训Tensor-ABESS，另建版本输出，重拟合训练折预处理。旧B/支持/选阶缓存不可作为新版本。GLasso/Lasso/Oracle只有配置、输入、划分、源码、环境均一致才可严格复用。完整模拟baseline/correlation/dimension/sample_size/snr必须对应最终采用的算法版本，不能将本轮小规模研究当700任务完成；未来v5.1 full也不能替代v5.2/v6 full。

本轮只有合成模拟及诊断，没有实际重训三实证数据；对HCP/NKI/LEMON的效果不能从本轮模拟直接推断。最终采用建议随ALGORITHM_VERSION_DECISION.md更新，当前正式默认不变。

## 独立研究运行器交接补丁

candidate-search-study.patch包含16个新增文件：10个研究Python模块（含两份内部测试）、README及5份tests，不包含核心快照、实验输出或任何用户既有修改。SHA256为6f14d0ab6b3c8776d407dffe90f9f04f0c8dbdf54538afad1eec367fecf224e3；在独立空目录git apply --check、实际应用与16文件逐字节核验通过。核心补丁与研究补丁物理分开，没有MSIC改动。限定测试总数100，非全仓库测试承诺。

当前只准备可复核的独立Git diff，没有创建新分支/worktree、提交既有脏工作区或推送；研究版正式采用仍须版本决策，不能以补丁存在表示算法已批准。

## P4独立身份与增量代码交接

本轮新身份为doc/algorithm_versions/v6-msic-after-search-candidate/VERSION.json，是基于冻结v5.2 core的global-rule研究包装器，release_approved=false、生产默认未变；任务卡2旧v6-msic-candidate/VERSION.json原样保留（SHA e5982d924890600c8d2deb5a44bfe078f0cf71aed19a4dce995091f2b0270da2），没有覆盖旧研究身份。

新msic-study.patch仅三个msic模块与三tests，共六new-file，SHA 9d56aede75a153e80353ff33d9e4bac0fd8cf2cfae43e9cc58dd347398441f24，独立空副本Git检查/实际apply及六文件逐字校验通过。31 wrapper tests加100既有任务3限定测试总131，不表示全仓库通过。三个fresh代表0/40/80全通过，五组独立确认90及MSIC两规则180行评价均完整核验；90锁模型满秩、max condition17.332759，不将候选当正式版本。本次收尾只同步记录，没有重跑测试或实际集群验证。

README的P4指令与增量上传说明有独立readme-update.patch（SHA 8fff4744842d6b5ec8c2f22a487e01ad92e0ab04d1f3773b6a84748b7401fc78）。最初嵌套目录git apply因发现主仓库而Skip，字节断言及时失败；停止此校验后用GIT_CEILING_DIRECTORIES明确隔离Git发现范围，真实check/apply/reversecheck及README字节核验通过，保留文件原换行，不更改任何计算文件或实验产物。代码补丁不因文档新增而改写。

应用順序：任务卡2v5.1修复→任务卡3四核心candidate patch→16文件study patch→本轮六文件MSIC patch→README独立文档delta。保持独立研究目录，不直接覆盖队友正式代码；具体CLI、LOCK父plan/summary绑定、三fresh门槛、独立确认仅两规则、3.9环境新plan和最小必要上传见README。无需ZIP，不升级集群，不代发队友消息。

## 任务卡3最终交接状态（2026-10-11）

独立确认A2原MSIC Exact26/90、PMSE3.082733，相对A0的6/90、3.118404保留恢复优势；A3锁规则alpha2/scale0 Exact27/90、PMSE3.117813，改善A3但未超过A2预测表现，不可直接移植为A2最佳规则。生产仍同步v5.1及原MSIC，研究身份均未批准；A2扩展真形状/相关性验证属于任务卡4，不替换队友生产代码，不启动700正式任务或三个实证重训。已有patch和源码指纹不因此次状态收尾改变。
