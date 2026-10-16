# 最终算法队友同步准备（任务卡4）

2026-10-11状态：A/B已完成：100组配对、200个所选OLS核验及16次独立冷启动全部通过，A2搜索推荐采用；80组独立A2 MSIC确认触发3600秒预算停止、证据不完整，按冻结规则保留原MSIC；最终推荐D2/v5.2-search-final，157项限定测试通过，正式发布待用户批准。当前生产仍v5.1-bcd-fix，研究v5.2/v6均未批准；本文未发送队友消息、替换生产代码、启动三个实证或创建分支/提交。最终身份以FINAL_ALGORITHM_VERSION.md为准。

## 现在可交付的v5.1修复

已有目录doc/team_sync/v5.1-bcd-fix含bcd-fix.patch、MANIFEST.json、CHANGELOG.md和TEAM_SYNC.md。patch仅修改src/tensor_abess/tensor/bcd.py及新增tests/test_bcd_fix_regression.py，SHA256为d2e6cac68ad7e37ec131cf2ff3d52eec4e7ec4498d30cea484d513fe2b7b37ee，原基准commit d4e28969ac480ba6efc01bcc985a53d179e7ed31。

当前Git HEAD d0c3547704fbaf5d90a3e5423568768a1ba11759不替代源码指纹。v5.1 raw BCD指纹c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737；规范LF为4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c。完整32源码清单见doc/algorithm_versions/v5.1-bcd-fix/VERSION.json，现场与生产src逐文件核对一致；不是只核验bcd单文件。

v5.1保留旧caps/旧grid、mode MSIC alpha1、global alpha1.5/gamma1/one_se_scale1、Group-BESS及OLS；fixed BCD保留原cold探索但以完整OLS状态独立维护best、正式接受轨迹非增、返回最佳已访问状态。参数接口tensor_abess/tensor_bcd保持兼容，结果新增best_iteration、best_rss、stop_reason、visited_loss_history、accepted_support_history，不能把adaptive变q轨迹当fixed RSS非增轨迹。

最终选择D1并确认双方基准后，队友各自项目内：

```bash
cd /自己的项目根目录
git status --short
git apply --check /同步文件路径/bcd-fix.patch
git apply /同步文件路径/bcd-fix.patch
export PYTHONPATH="$PWD/src"
python -m unittest tests.test_bcd_fix_regression -v
```

已应用修复的队友先核对32源码清单，不重复应用patch。check失败即停止比对基准与本地改动，不强制覆盖。这里不要求开分支、commit、push或替换远端main。实际同步由最终批准后的成员执行，本次未联系他人。

算法运行依赖沿用现有NumPy/SciPy，集群固定Python3.9.25/NumPy2.0.2/SciPy1.13.1，不执行pip/conda安装升级；LEMON另沿用其既有数据/预处理依赖，不拷贝本机venv。pytest为可选测试前端，不是算法依赖。本机复验只用taskcard4_test_tools：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\doc\algorithm_versions\taskcard4_test_tools"
python -B -m unittest tests.test_bcd_fix_regression -v
python -B -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py -q
```

Windows示例中的python须先核对为本轮已有Python3.12.8及其NumPy/SciPy环境，不预设存在.venv；Linux使用对应既有环境的python，不安装或升级依赖。这些命令用于生产v5.1复验；本轮另以D2冻结源码独立进程通过29项BCD/模拟pytest回归及6项候选静态/同数据拟合桥接。它们不代表全仓库或真实集群通过，最终限定测试清单以FINAL_ALGORITHM_VERSION.md为准。

## 三个实证的同版本边界

|成员/数据|保持的数据与流程|本仓库可核查范围|
|---|---|---|
|LEMON EEG|相同145个纳入被试、自然对数TMT_5、协变量定义/编码、EEG通道/频带/窗口及原外层划分|empirical_analysis/lemon/models.py、incremental_value/{model_adapters,run,nested_cv,preprocessing}.py及config.yaml|
|HCP-YA MRI|队友原被试、MRI张量/模态顺序、响应与协变量、预处理定义、外层与内层划分|仅同步协议与公共core；本仓库无HCP本地执行入口/数据，未核查其具体配置|
|NKI-Rockland MRI|队友原被试、MRI张量/模态顺序、响应与协变量、预处理定义、外层与内层划分|仅同步协议与公共core；本仓库无NKI本地执行入口/数据，未核查其具体配置|

三人共享最终VERSION完整源码指纹、公共入口、训练内配置和依赖记录；实证均target_sparsity=None，不使用模拟真支撑或外层测试集决定caps/grid/MSIC/初始化。不能通过删除难例、改响应或改划分把算法差异混入数据变化。

新版本每个外层训练折重新拟合协变量残差化、标准化、中心化，再训练内生成caps/候选、重搜支撑、选阶、OLS及预测。若内部CV比较ABESS参数，每个内训练折须独立拟合预处理并重搜候选/支撑，容量按该内折样本量计算；不能用全外训练或全数据固定支撑作伪内CV。LEMON现行ABESS不外部调参，其GLasso初始化内部CV仍限定相应训练数据，统一模型配置保持原值。

LEMON增量配置固定outer_folds5、outer_repeats3、outer_seeds=[20260908,20261009,20261010]、inner_folds3、inner_seed20261009；model为k_max1/c3/BIC初始化/n_lambdas20/cv_folds3/max_iter500/tol1e-6/max_cycles6/m_max15/eps1e-8及原MSIC。models.py明确拒绝实证target_sparsity，pipeline指纹绑定32core、模型适配/预处理、输入和环境。其旧输出outputs_eeg_incremental含旧模型和断点，不能原目录续跑不同代码。

最终D1且同步源码/数据验收后，LEMON可先在新目录只做audit，然后另行授权启动既有正式入口；以下命令本次未执行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -B -m empirical_analysis.lemon.incremental_value.run --audit-only --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v51_final
# audit验收且正式重训获准后：
python -B -m empirical_analysis.lemon.incremental_value.run --output-dir empirical_analysis/lemon/outputs_eeg_incremental_v51_final
```

HCP/NKI不存在可从本仓库核验的启动命令，最终由队友在原入口显式使用冻结core、同配置及独立版本输出；交回源码/依赖/配置/输入/折索引指纹、版本号及完整OOF预测验收。本文不虚构其CLI或已完成同步。

## D2待审批包与D3边界

已制作[doc/team_sync/v5.2-search-final](team_sync/v5.2-search-final/README.md)的README.md、MANIFEST.json、search-final.patch和run_v52_search_final_full.sh，冻结候选位于doc/algorithm_versions/v5.2-search-final/src。32源码中只改bcd.py私有规模网格，保留旧caps、fixed BCD修复、adaptive、Group-BESS、OLS及原MSIC；公共参数与默认值不变。6项静态/同数据实际桥接及本轮29项最终候选pytest通过；普通LF/CRLF副本patch实际应用、规范LF身份、Bash语法和mock组件核验均已有记录。包仍research_only=true、release_approved=false，当前src未改变。

最终批准D2后，先按VERSION/MANIFEST核对目标32源码的v5.1基准，再将patch应用到目标项目src，按包README复验并核对最终32源码规范LF指纹；将sh复制到目标项目根。这样known_s_cluster.run实际导入与source_fingerprints扫描均绑定同一repository/src；仅切PYTHONPATH指向研究快照会记录错误身份。research candidate-search-core.patch不是这个最小D2部署patch，不能混用。完整模拟prepare/sbatch/merge命令与独立目录见FULL_EXPERIMENT_PREPARATION.md，本轮不执行。

本次80组独立确认预算停止，未完成，fresh refit未执行，不支持D3；不续跑原计划或把D2包改标v6。未来如单独授权新的MSIC研究和最终D3版本，须重新冻结与验证源码、defaults、patch、manifest及独立full脚本。集群Python3.9/NumPy2真实运行及HCP/NKI具体入口仍未验证。

如采用D2（例如v5.2-search-final）或D3（例如v6-final），重新搜索所有训练折并重训全部ABESS外层模型；旧B、支持、q、选阶轨迹和模型缓存均不可复用或换标。可复用已经验收且未跨被试拟合统计量的原始张量和相同划分，所有折内预处理重新拟合。Covariates Only/Ridge/Lasso/GLasso等对照仅在其独立代码/参数/训练与验证输入、预处理、划分、随机种子及环境逐项一致、完整性核验通过后保留明确旧来源；core指纹变化导致旧统一断点身份不兼容，不能强行改断点身份。

最终批准D2并完成源码/数据验收后，LEMON可在新目录outputs_eeg_incremental_v52_search_final使用上述入口先做--audit-only；正式重训另行授权后去掉该参数。D3须采用其最终包和独立版本目录，不能沿用D2目录。同步完成并获最终批准前，不替换队友生产代码、不发布研究版本、不启动实证；旧结果必须保留原v5/v5.1标识，算法正确或合成模拟更好不保证三个实证预测提升。
