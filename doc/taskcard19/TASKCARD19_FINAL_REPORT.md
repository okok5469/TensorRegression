# 任务卡19最终报告

北京时间：2026-10-31T18:26:33.950198+08:00。

- PACKAGE_READY: YES
- LOCAL_STAGE: NUMERICAL_BLOCKED
- FORMAL_TEST_EVALUATED: NO
- SCIENTIFIC_FINDING: 有限训练样本的开发筛查尚无超越已验收最强对照的增量；完整五方法比较仍受数值/资源状态阻塞。

| Gate | 状态 | 证据 |
|---|---|---|
| G0 核心身份 | PASS | 32生产源与release原字节一致；38项源码/配置身份（含32核心与6模拟文件）及任务18定稿文件保持 |
| G1 A路线实现 | PASS | 指示响应仅中心化、截距恢复、独立核心完整状态/分数核对、并列/标签/保存与失败检查 |
| G2 数据与隔离 | PASS | 数据合同、group/time、训练内z-score、HAR6912原始坐标独立映射核对 |
| G3 对照与指标 | PASS | 软件协议检查：全五方法、完整九网格、诊断/OOF/重算；实际开发比较状态另见G6 |
| G4 便携交付 | PASS | 最终ZIP在原仓库外全新venv安装wheel，实际导入与依赖/线程检查 |
| G5 回传复核 | PASS | 完整/部分结果、预测缺行/损坏检测、安装版重算和pack-return |
| G6 本机开发 | BLOCKED | HAR 4/5方法任务完成；状态NUMERICAL_BLOCKED |

## 给队友的文件

主文件：[TensorABESS_OvRLS_v0.1.0.zip](../../deliverables/taskcard19/TensorABESS_OvRLS_v0.1.0.zip)，SHA256 `5c3fdb7701dcb4b759b70f5593854f79a5be86722518ba1971c59c4cfa54c9df`；旁置同名.sha256。ZIP含wheel、未修改32源、锁定依赖Windows wheels、唯一数据合同、固定fixture、测试/验收记录和两平台入口。无需复制历史仓库。

队友解压按README_ZH.md建立CPython3.12干净环境，安装wheel，执行verify-release和smoke，再validate-data/run。最后validate-return、pack-return并回传统一ZIP；原始X、venv和凭据不回传。本卡未替用户发送消息或发布公共包。

实际验证平台Windows；Linux UNVERIFIED。精确日志见IMPLEMENTATION_AND_PACKAGE_CHECKS.md及clean_install_final/。同环境开发/安装版分数、标签、系数、声明支撑和并列优先序均比较；无跨平台逐位一致声明。限定测试共42项通过；不宣称全仓库测试通过。

## 本机HAR开发筛查

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

## 已暴露的问题与边界

最初软件fixture在L1高C20000步后仍不收敛，运行器正确标PARTIAL；原5输入与失败输出保留在runner_checks/fixture_v1_input和fixture_run_v1。后续安装fixture固定为各相同输入模式含全三类的非可分输入，仅用于软件验收；未更改真实HAR、网格、tol、搜索或预算，准确率不作通过门槛。

独立审查发现并修复安装路径/签名/ZIP库存、诊断完整性/内折身份/并列/OOF及超时子进程清理等缺口，均保留回归证据。首次广域检索遇到原历史测试工具访问受限；其已有Git状态原样保留，本卡不依赖或修复这些目录。独立安装没有受此影响。

下一步：先解决L1必要候选及C=10最终拟合在20000步仍未收敛的问题，另存数值处理、训练量与资源协议后再开展正式实证；保持本卡结果不变，测试继续封存。原MSIC仍是平方损失工作准则，既有回归理论/模拟不证明本封装分类一致性；原初始化返回成功也不是全部内层凸问题的最优性证书。
