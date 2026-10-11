# 任务卡3预声明实验协议

冻结时间为研究结果生成前。生产src仍v5.1；v5.2独立源码、完整manifest和immutable plan用于全部研究。

A0旧caps/旧网格；A1候选级caps/旧网格；A2旧caps/完整整数；A3候选级caps/完整整数；A4候选级caps/预定粗到细（训练原globalMSIC top3的各mode±1邻域一次）。全使用原mode alpha1/global alpha1.5/gamma1/scale1，不按真实支撑或测试误差生成候选。Q=floor(.7n)，候选budget2000，超限明确停止，不静默截断；historical仅训练init数量预检最高1044，预算在真实拟合前设定。缓存显式绑定训练指纹、固定solver配置，不混用不同数据。

Phase D历史27的seed/config原样复用用于机制追踪，不作为确认数据。新增development每场景10次，共90个数据，seed20361010+1000scenario+rep；confirmation独立每场景10次，共90个数据，seed20461010+1000scenario+rep；test均+1e6。先研究搜索，再讨论MSIC。均n100/200/400×SNR.5/2/4，原dims32/20/25、true4/3/5仅生成与评价、rho.3、sigma1、ntest500。真实位置/大小不进入candidate policy或多初始化选择。

开发搜索评价同时看shape覆盖、精确位置覆盖、Exact/F1、同场景PMSE及RSS、候选数和成本，不能只以平均PMSE排名；主参照A3必须完整覆盖其合法网格，预算失败/接受轨迹回归/无法复现时停止受影响方案。候选覆盖提高若不能转化为恢复或预测的可信收益，则不推进正式MSIC调整，记录停止原因。研究结果不自动形成正式v5.2或v6。

若进入Phase E，搜索机制先固定为经验证的预声明A3，保留mode-wise规则，独立模块从同一候选库重评分alpha1.5/2/3×scale0/.5/1与tensor-SIC/纯BIC。开发集比较全场景及强信号恢复/欠选；alpha2/scale0为事先优先的温和候选，独立confirmation不反复选参。全流程重新拟合验证与缓存重评分必须分别标明，不能将缓存复用称完整新拟合。

支撑位置研究仅用A0/A3训练选出的最终shape：同shape预定marginal+2随机初始化(seed+30m,+31m)，按训练RSS选best。Oracle只在模拟评价时另算真实支持OLS，不影响shape、支持或初值选择。固定size内接受轨迹须单调、RSS/返回best一致。

不启动700正式full，不改GLasso/Lasso/Oracle源码、DGP和现有数据，不升级集群，不创建分支或推送，不覆盖历史输出；最终六份报告与work_progress明确完成/部分/未执行。
