# 算法版本决策（任务卡3）

## 当前正式版本与研究身份

当前生产src仍为v5.1-bcd-fix。v5-original历史正式结果、任务卡2最终v5.1、原known-s full、LEMON和其他旧结果均保留。本研究v5.2-candidate-search单独存放于doc/algorithm_versions，research_only=true、release_approved=false，MSIC默认不变。当前没有v6-msic-candidate正式源码或正式批准。

候选搜索四核心差异仅tensor/abess.py、tensor/bcd.py、utils/init.py、新tensor/candidate_search.py；固定tensor_bcd源码逐字与v5.1相同，Group-BESS/OLS/SIC未改。新增公共参数均有legacy默认，原调用兼容；研究组需要显式选择cap/grid policy。source33、补丁SHA、strict plan/environment和训练/test指纹区分版本，不能把旧B/缓存/输出改标签为新版本。

## 实际完成与尚待条件

原机制27审计、13项核心候选测试及原算法回归、27历史A0–A4真实配对、27位置多初始化、历史135所选OLS数值审计均已完成；A0全27结果与旧v5.1逐值等价。所有限定测试共100通过，48文件Python3.9语法通过，不等同实际集群3.9运行或全仓库测试。

新增开发90事先冻结且已全部完成，450所选OLS均满秩、最坏condition17.19845；独立确认90事先固定但尚未运行，未用于调参。原MSIC诊断已记录，新的alpha/scale消融、tensor-SIC/BIC重评分和新准则完整重拟合尚未执行。完整开发证明full grid恢复收益，按预声明A3进入研究P4，但caps负效应、A3约23%预测恶化与约4倍成本明确不支持直接正式采用；新规则尚待冻结库/完整refit/独立确认。

历史显示grid补全改善恢复，cap单独放宽反而增加过选并恶化平均PMSE；A3真形状全覆盖仍只有10/27真位置入池、6/27最终exact。三起点位置搜索的平均PMSE收益受单例主导，不能认为已解决局部搜索。A4只是预定粗近似，可能遗漏one-SE需要的小模型。所有这些都要求区分算法正确性、搜索覆盖、统计泛化和成本。

## 正式full及实证影响

本次没有启动700任务full。未来若只采用v5.1，可独立完成任务卡2的v5.1 full；若正式采用新的cap/grid或MSIC，则必须重新运行baseline、correlation、dimension、sample_size、snr等对应Tensor-ABESS实验，使用新版本独立输出，v5或v5.1不能替代。对照算法仅在源码/config/输入/seed/划分/环境逐项一致且完整验证后决定复用。

本轮无LEMON、HCP-YA、NKI-Rockland的ABESS重训。采用新算法应保持受试者、响应、协变量定义、折内预处理和原划分，重搜每训练折候选/支撑，另存新B/选择/预测。不能凭合成模拟宣称实证性能提高，不能把旧固定全训练支撑当内部CV。队友目前只同步任务卡2v5.1，研究v5.2先独立复核，不直接替换两名队友代码。

## 独立交接与改动保护

两个Git可应用diff分别为candidate-search-core.patch（四核心文件）与candidate-search-study.patch（16研究/测试文件），均在独立副本实际apply及字节核验通过。未创建branch/worktree、提交既有脏工作区或推送，不覆盖主分支；冻结plan同时锁定原Git HEAD，记录uncommitted源码SHA以保证实际身份，不能只依赖HEAD标签。

2040保护文件与33研究core的SHA256复核全部一致；错误首版验证器与旧STOPPED研究保留。验证器纠正区分变q adaptive MSIC与固定q BCD，算法/容差/种子未改，最终从新同seed计划重跑，不后处理修补失败结果。增量上传依赖与完整Linux命令见candidate_search_study/README.md，不制作压缩包、不升级集群。

已按完整开发进入Phase E研究；是否有可推荐MSIC候选及正式版本意见待其实际结果，当前正式默认保持v5.1。

## 后续研究优先次序（建议，未实际执行）

1. 在独立新种子确认最小改动的A2（旧caps+完整整数）是否保持恢复/预测收益，单列n100仍缺真形状的问题，不因全样本均值更好忽略结构性不足。此建议不改变预声明本轮A3的Phase E机制。
2. 对A3池未含真位置的案例，预先固定交换预算k_max/m_max或cycles的单因素对照，以及更多训练初始化；同训练shape/seed比较RSS、位置F1、独立预测与成本，不能按测试挑起点。三起点本轮收益受单例主导，不据此扩大正式默认。
3. rawcap并非所有q≤Q模型的搜索界限；研究其他不依赖truth的cap/shape规则必须新协议、新输出，预算明确，不能只围绕本模拟真4/3/5补点。粗搜索若用于低成本近似，需要同时保留minimum及one-SE容差带内可能更小的shape，再用完整网格评价遗漏，不能只守住minimum。
4. 若P4有合适候选，按冻结开发/独立确认顺序检验强信号欠选与弱信号过拟合；若无，则保留原MSIC、不宣布v6。CV只能作为另一个受控参考，内折重新搜索和容量重算，不复用全训练支持。扩展其他相关性、维度和真shape属于后续正式研究，当前9场景不支持普遍结论。
