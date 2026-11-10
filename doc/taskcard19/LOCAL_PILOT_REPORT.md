# 本机开发试跑

原官方train7352窗口/21受试者；开发训练400窗口/15受试者，验证2208窗口/6受试者，六类保留。固定3×3×17均值/窗频带摘要，subject seed20261031；官方test2947窗口/9受试者未评分。精确ID、原始下载身份与表示见DATASET_SCREENING.md和PROTOCOL_FREEZE.json。

运行 `20261031T100556Z_ef5d35491434`：5方法任务中4完成、1未验收；留有完整诊断的拟合调用记录109（包括调参、同参数重试及6类核心），多数类规则构建1条另记。该计数不包含资源中断时尚未返回诊断的调用；训练内功能切片6次核心调用另列RESOURCE_BUDGET.json，不混入性能分母。数据集实际墙钟981.985秒。全部正负与失败证据位于 `empirical_analysis/outputs_taskcard19/har_dev_run_20261031`；验证结果见LOCAL_RETURN_VALIDATION.json。

| 方法 | 状态 | Macro-F1 | Accuracy | Balanced Accuracy | 训练秒（全部调参+最终拟合） |
|---|---|---:|---:|---:|---:|
| majority | COMPLETE | 0.055283 | 0.198822 | 0.166667 | 0.207 |
| ridge_ovr | COMPLETE | 0.827486 | 0.831975 | 0.833980 | 1.199 |
| multinomial_l2 | COMPLETE | 0.864447 | 0.866848 | 0.863736 | 12.824 |
| multinomial_l1 | NUMERICAL_BLOCKED | 0.873205 | 0.872736 | 0.873784 | 894.065 |
| tensor_abess_ovr_ls | COMPLETE | 0.724122 | 0.747736 | 0.726766 | 35.855 |

任务失败/未验收状态：

- multinomial_l1 / har_dev_subject_holdout: NUMERICAL_BLOCKED; required candidate diagnostics remain unresolved

关键数值诊断：

- multinomial_l1: phase=inner, inner_fold=0, parameter=0.1, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=0.1, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=0.1, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=0, parameter=1.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=1.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=1.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=0, parameter=10.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=10.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=10.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=0, parameter=100.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=100.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=100.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=0, parameter=1000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=1000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=1000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=0, parameter=10000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=1, parameter=10000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=inner, inner_fold=2, parameter=10000.0, max_iter=20000, n_iter=[20000]
- multinomial_l1: phase=final, inner_fold=None, parameter=10.0, max_iter=20000, n_iter=[20000]

主指标Macro-F1；不计算A路线概率Log-loss/Brier，不把原始分数当概率，不以方法获胜作为软件验收条件。窗口存在受试者内依赖，本次只有一次主体留出和有限训练样本，不作正式测试/统计显著性或分类一致性推断。MHEALTH未启动，本机仅尝试一个真实候选。独立重算见[LOCAL_PILOT_INDEPENDENT_REVIEW.md](LOCAL_PILOT_INDEPENDENT_REVIEW.md)；完整带失败状态的回传为[HAR_DEV_RETURN_20261031.zip](../../empirical_analysis/outputs_taskcard19/HAR_DEV_RETURN_20261031.zip)。


LOCAL_STAGE: NUMERICAL_BLOCKED

FORMAL_TEST_EVALUATED: NO

SCIENTIFIC_FINDING: 有限训练样本的开发筛查尚无超越已验收最强对照的增量；完整五方法比较仍受数值/资源状态阻塞。
