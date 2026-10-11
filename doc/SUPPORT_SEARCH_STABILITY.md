# 支撑位置搜索稳定性：三初始化独立实验

最终目录：numerical_simulation/outputs_support_search_stability/historical_final，27/27数据任务完整，A0/A3各27对，共54个训练选定规模。冻结v5.1 fixed BCD与研究v5.2 fixed函数逐字相同；历史数据与独立测试流逐一匹配。旧STOPPED两个预检不复用。

每个规模只用该组训练MSIC最终选出的shape，不使用真实shape决定实验候选；预声明marginal、random(seed+30000000)、random(seed+31000000)三起点，按训练RSS最小选择，平分按起点顺序。相同数据内重复shape显式复用。Oracle真实支撑OLS在全部搜索后单独计算，只作评价，不能进入初始化/选择。

|组别|数据数|随机起点胜出|平均RSS差（multi−marginal）|平均PMSE差|精确恢复率差|PMSE胜/平/负|
|---|---:|---:|---:|---:|---:|---|
|A0|27|3|-0.611442|-0.106215|0.0000|1/24/2|
|A3|27|4|-0.134216|-0.016024|0.0000|2/23/2|

平均PMSE的下降主要由task8驱动：排除该例后A0平均差为+0.000648，A3为+0.001392。训练RSS降低而测试PMSE上升的实际胜者分别为A0的task11/19、A3的task9/19；Mode-F1平均变化反而为−0.001372和−0.003939。54个组内shape对因三个数据集两组shape相同，实际只有51个独立shape、153次初始化拟合。

多初始化在本轮只带来局部训练目标收益，未显示稳定泛化或恢复收益，精确恢复率没有提高；不能据此宣布位置问题已解决，亦未把三起点替换正式算法。其比较固定了训练选定的shape，因此无法补救已选错规模，或将真实shape从caps外带入。

训练RSS低与测试误差低不能等同。task0的A0 random_0训练RSS73.009高于marginal48.045，但测试PMSE约1.992低于marginal2.728；A3随机起点也出现训练更差而测试更好。研究仍严格按训练RSS保留marginal，没有按测试误差选择起点。

原组最终候选可能来自adaptive-stage，其路径不同于同shape的marginal fixed重启；所以multi≤marginal的训练RSS保证不能直接推广为multi≤原最终候选，paired_multistart.csv同时保存candidate、marginal、multi，避免混用。

## Oracle与局限

Oracle知道真实位置，但系数仍由有噪声训练数据OLS估计，测试PMSE并非每个样本的下界。弱信号/小样本时估计较大真模型的方差可能超过小模型的偏差。本轮Oracle只辅助区分位置搜索与估计难度；不能据其表现强制真实大小或用Oracle位置选择实际方法。

历史只有每场景3次，且随机起点只有两个；没有系统研究其他初始化、连续尺寸路径、重复CV或其他真实shape/相关性。新增开发数据用于候选机制验收，不是本稳定性实验的追加结果。

## 核验与复现

5模块测试和5runner测试已通过；每fixed起点的正式loss历史非增、返回minvisited、独立RSS重算、固定规模/索引、数据指纹和缓存SHA通过，未修改任何B或支持救补。逐起点记录all_starts.csv，配对汇总paired_multistart.csv，完整JSON/NPZ保存所有初始/最终支持及B、种子、耗时与Oracle结果。

执行入口见candidate_search_study/README.md：stability_run create → run → summarize；输出必须新目录且candidate计划完整。

本C实验使用原MSIC最终训练shape；没有对全部A3候选逐shape追加初始化，也没有对alpha2/scale0新选择的shape追加多起点。新规则的改善来自原冻结库内重新选择，不能据此声称位置优化算法已增强。
