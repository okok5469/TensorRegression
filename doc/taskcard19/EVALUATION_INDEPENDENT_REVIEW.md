# 任务卡19：评价与数据隔离独立审查

日期：2026-10-31（北京时间）。审查者为数据子agent，独立于runner作者；只检查应用封装，不修改正式32源，不开展统计性能研究。

当前结论：**PASS（本审查范围）**。发现的实质问题已修复并完成独立复核；本审查不替代最终全套测试及干净安装验收。

## 检查结论

| 检查 | 结论及依据 |
|---|---|
| 数据、角色与类别身份 | PASS：`data.load_dataset`依据row_index对齐，不依赖CSV当前顺序；固定class_key全集；未知评价类别、缺类训练、ID交叠、group/time边界均拒绝。`evaluation_stage`与显式active_split_ids核对；dev只选validation，不遍历官方test。 |
| 训练内预处理 | PASS：各内折独立拟合TensorStandardizer；最终fit仅使用当前训练样本。ddof=0、严格常数及计算std=0的掩码、C顺序、float64与有限性检查符合定义。 |
| 五方法定义 | PASS：Ridge为K列指示响应并带截距；Logistic仅为独立对照，使用显式multinomial、L2/lbfgs与L1/saga；Tensor全部K类中心化但不缩放响应，恢复类频率。 |
| 九网格及数值状态 | PASS：全部九候选×各内折评分；均值Macro-F1选参、1e-12并列优先强正则；显式random_state；Logistic按同目标5,000→20,000步重试，任一必要项未解决使该方法NUMERICAL_BLOCKED，不删除困难候选。 |
| 计算预算 | PASS（代码路径）：方法任务在独立spawn进程，Tensor类调用再独立有界进程；总体剩余预算传入核心deadline，资源到限保留部分失败而不缩减搜索。真实耗时/成功量以运行日志为准。 |
| 随机流与成本 | PASS：种子由总seed及method/fold/candidate/inner标识SHA256派生，无Python hash()和共享随机流；训练计时覆盖方法预处理、全部调参和最终fit，预测单列。 |
| 固定类别指标 | PASS：Macro-F1包含固定全集且zero_division=0；缺真实类recall为NA，Balanced Accuracy按有真实类平均；A路线score不称概率、不报概率损失。 |
| OOF及列身份 | PASS：Logistic按类身份重排输出列；OOF由逐样本语义预测重算，重复评价ID时不产生普通OOF主汇总；不存在按列号盲拼或把折当独立模拟重复的t检验。 |
| 回传逻辑检查 | PASS（最终修订代码静态审查）：补齐训练/评价sample_metadata、内折边界和类别覆盖；全部网格诊断、收敛/重试状态、选参与稳定seed均核对；保存模型训练标签绑定并列优先序；支撑绑定数据轴与保存模型；完整及partial明确区分；重复身份字段互相核对。 |

## 独立实际探针

`data_checks/EVALUATION_BASELINE_PROBE.json` 使用固定软件fixture v2分别实际训练三个需调参基线。每法27个内层候选评分均存在，独立SHA256表达式核对所有随机种子；训练均值与只取训练X的直接均值完全一致；Macro-F1、Accuracy、Balanced Accuracy、混淆矩阵逐项与sklearn独立计算相同。L1最大观测迭代689，三法必要拟合均完成。该探针没有统计性能含义，不作为真实HAR结果。

原fixture v1的随机特征在少量内折中使高C的L1在20,000步仍未收敛。该负结果及原始五份输入保留在 `runner_checks/fixture_run_v1/`、`runner_checks/fixture_v1_input/`；没有改tol/grid/预算或删除候选。fixture v2固定为每组三类各一条相同X、跨组七坐标变化加一常数坐标，只用于非可分的软件检查；详见 `runner_checks/FIXTURE_V2_CONSTRUCTION.json`。HAR输入、分组、表示和400样本协议完全未变。

## 实质问题与修复闭环

1. 初版回传器仅校验文件/分数/指标，未读取诊断或核对实际内折，可漏检缺失网格和数值失败重标完成。已新增 `_validate_diagnostics` 与 `_validate_inner_protocols`，绑定完整网格、重试、seed、真实sample/group/time边界及完成状态。
2. 初版重复身份记录未互相比对。已核对data_identity与manifest、config种子及资源、各任务数值环境和串行线程、核心profile指纹及Tensor模型身份；按run内部身份核对，不强求本次复核机器的路径相同。
3. 最后发现模型API与runner采用数学等价而浮点不完全等价的并列式：前者 `(max-score)<=tol`，后者 `score>=(max-tol)`。独立构造 `max=1.12`、`lower=max-64*eps*1.12`、优先序`[1,0]`，模型返回top且0次并列，runner返回lower且1次并列。证据 `data_checks/TIE_BOUNDARY_INDEPENDENT_RED.json`。现已统一为差值式；作者新增边界回归RED→GREEN，独立探针再次执行，两预测层均返回top且0次并列，见 `data_checks/TIE_BOUNDARY_INDEPENDENT_GREEN.json`。

本审查没有为HAR生成预测，没有评价正式测试，没有修改原算法或模拟/论文章节。
