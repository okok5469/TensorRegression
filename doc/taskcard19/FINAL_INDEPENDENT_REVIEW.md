# 任务卡19最终独立总验收

日期：2026-10-31（北京时间）。本轮仅只读复核既有结果，并新建本报告；没有修改任何已封存报告、Python源码、模型、交付ZIP或旧结果，没有重新拟合或新增候选。

**结论：PASS（交付、审计一致性与结论边界）；没有发现需要返工的实质缺口。软件PACKAGE_READY为YES；HAR完整五方法比较仍为NUMERICAL_BLOCKED；FORMAL_TEST_EVALUATED为NO。此PASS不表示分类实证取得优势或L1已通过数值验收。**

## 封存、保护与交付

- 独立按生成规则重新计算机器seal，审查时payload seal为`32f058ef04e09cf15a92980b90c2f00cbdac7256167b66ea3fc44765a52343e2`，与记录一致；所列196项artifact逐文件SHA256全部匹配。
- 38项正式源码/配置身份精确包含32份核心Python与6份模拟入口/配置；联合章节及任务卡18验收记录，共42份保护文件。逐文件原字节hash全部保持，与记录的protected_files_checked=42一致。
- 当前tracked Git状态与事前baseline相同，比较时仅按既定协议排除允许追加的`doc/task_plan.md`、`doc/progress.md`、`doc/findings.md`；根目录没有四种禁止的过程文件。没有据此声称旧用户改动被提交或清理。
- 最终ZIP、旁置SHA256与机器seal绑定一致：`5c3fdb7701dcb4b759b70f5593854f79a5be86722518ba1971c59c4cfa54c9df`。外层完整库存和逐文件hash通过；内嵌wheel的32份核心与正式release原字节相同。
- 最终精确ZIP的`clean_install_final/CLEAN_INSTALLATION.json`为PASS，六个实际阶段返回码全部为0，实际导入来自仓库外新venv，开发/安装签名一致。Windows已验证，Linux仍为UNVERIFIED。此前42项限定测试通过的口径没有扩张为全仓库或跨平台验证。

## 真实HAR运行量

直接读取本次run的diagnostics、manifest、metrics、predictions及SHA256SUMS，核对结果如下：

| 项目 | 独立核对结果 |
|---|---|
| run_id | `20261031T100556Z_ef5d35491434` |
| 方法状态 | PARTIAL；4/5完整，L1为NUMERICAL_BLOCKED |
| 已留完整诊断的拟合调用 | 109 = Ridge 28 + L2 28 + L1 47 + Tensor六类核心6 |
| 规则构建与事前功能检查 | 多数类规则构建1条另记；36训练窗口功能检查核心6次另记，不混入109或性能分母 |
| L1关键未解决项 | 19个必要单元在20,000步仍未解决，全部属于L1；没有把这些项重标为成功 |
| 主run墙钟 | 981.984999999986秒，报告四舍五入为981.985秒 |
| 逐样本预测 | 11,040条唯一method/fold/sample_id/用途键，五法各2,208条 |
| 回传文件 | 49个受SHA256SUMS.json逐项保护的结果文件全部匹配；加上清单自身，run目录物理文件数为50 |

109仅统计留有返回诊断的调用；报告已明确不把资源中断且无返回诊断的调用当作精确总数。本次状态为数值未验收，而非达到总资源预算。L1虽保存了描述性预测和分数，不能据此称五方法比较完整成功。

已读取`LOCAL_PILOT_INDEPENDENT_REVIEW.md`及其对应既有机器证据：独立指标/分数/支撑/训练统计量、分组内折及环境身份复核保持PASS。本轮没有重复拟合，也没有将这些软件与审计PASS转写为统计性能结论。

## 科学结论与状态边界

保存的Macro-F1为Tensor `0.7241215696827883`、已验收L2 `0.8644470320337705`，Tensor低约0.140325；主报告“尚无超越已验收最强对照的开发增量”与实际数据一致。L1的更高描述分数同时标有NUMERICAL_BLOCKED，没有被用作已验收排名结论。

官方train内400窗口/15受试者训练、2,208窗口/6受试者开发验证的范围，与主报告和独立审查一致；主run的evaluation_stage为dev，formal_test_evaluated为false。官方最终测试没有在本卡评分，MHEALTH未启动。软件包可交付与G6数值阻塞被清楚分列；后续先解决L1数值验收，再另存正式实证协议，不修改本卡既有失败身份。

## 审查输入SHA256

以下为本报告生成前实际读取的输入快照，供复核本次结论：

| 输入（相对doc/taskcard19） | SHA256 |
|---|---|
| TASKCARD19_FINAL_REPORT.md | `95b4a8fba1368b59aae7d505cefbbf61c74603a032df39b7a4706af62fe16677` |
| LOCAL_PILOT_REPORT.md | `d85e9594ed81167c88e217563aeb450c75babfbd6b38d315e687175636719659` |
| IMPLEMENTATION_AND_PACKAGE_CHECKS.md | `82e0c2017e0c62b81d1cc25366b5ed2f01e5241b66782d73763cc78cbb987cdb` |
| LOCAL_PILOT_INDEPENDENT_REVIEW.md | `46bed1fc56324a0bb0a5240cf4c8c5427691527030149cfc02f182d801090d8a` |
| FINAL_VALIDATION.json（本轮审查前快照） | `c6244927ebb01b3630a05d1063e0147143cd0494feb16250c18e62cbd790dda7` |
| clean_install_final/CLEAN_INSTALLATION.json | `20e6a32ea5c72d87d27e509b06a2fd96dff127b523b93fe014a33653db5b3a83` |
| LOCAL_RETURN_VALIDATION.json | `d21f96547d652801b053345bdcf8a0d332f9439632fc71475fe1a7a7f3893bb5` |

本报告是本次唯一新增文件，尚未包含在上述审查前196项清单内。主agent随后仅将本报告纳入清单并刷新机器seal，FINAL_VALIDATION.json本体hash会相应变化；这属于追加审查证据，不表示修改上述其他已验收输入或交付ZIP。
