# 任务卡8复现与身份记录

任务发起日期为2026-10-16（北京时间）；实际UTC时间见各协议、started/complete及控制器终态。主D只有存在 `completion.json` 才能认作完整216研究数据，`stopped.json` 表示未完成；不能从任务启动或部分CSV推断完成。阶段状态见 `STAGE_STATUS.json`，最终结果入口为上级目录 `TENSOR_ABESS_NEXT_VERSION_DECISION.md`。

## 本轮边界和初始工作区

没有分支、worktree、提交、生产算法替换或团队消息，没有新算法ZIP。incoming两个文件原字节保留，Warm src是正式32文件副本仅替换收到的bcd.py；两个新身份均 `research_only=true, release_approved=false`。

初始任务卡7的2880个科学/源码保护文件全部匹配。其后已有的两份集群launcher说明与旧交付索引不同，现场有对应sh及测试；本轮先停止身份核验、核明仅部署文档变化后保留当前字节，并在 `BASELINE_RECONCILIATION.json` 单列旧/今指纹，旧索引未重写。独立任务卡8保护基线共3253文件，不能将这两份文档差异称为本轮科学输入漂移或默默改旧hash。

## 实际解释器和线程

既有 `.venv/Scripts/python.exe`：Python3.12.8 / NumPy1.26.4 / SciPy1.16.1 / Windows；未安装或升级依赖。四项线程变量为1（Warm旧测试日志还保留其实际设置），程序捕获实际import与源码SHA。新增模拟为本机新seed严格配对，不冒充原Linux700/250的输入身份。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONPATH='.'
```

## Warm原样正确性审计

证据在 `warm_audit/{incoming,baseline,old_rejected}/correctness.json` 及完整 `unittest.log`，输入/source32/import均记录；最终每版31顶层检查（27算法+4工具）。正式Cold31全过，原样Warm20方法全过/11失败或错误，旧Warm3方法失败。9 failure+11 error是含subtest的事件，不能当20独立失败方法。上传6测试通过不足以放行。

审计器以标准库逐个执行原测试函数，仅在内存AST删除没有实际使用的 `import pytest`；原断言及数据未改，遇真正pytest API会拒绝。没有声称全pytest/全仓库通过。先导入并断言目标bcd，再加载会调整sys.path的旧numerical_simulation模块；额外一次错误导入顺序取证明确标 `INVALID` 并保留，不能作为比较证据。

下面是本轮调用形式；重放必须将 `--output` 设为新的、从未使用的审计目录，不能指向既有证据。该审计工具会写指定目录，不能当作具备主实验那样的自动拒覆盖/续跑工具。

```powershell
.venv/Scripts/python.exe -B scripts/taskcard8/warm_audit.py --source doc/algorithm_versions/v5.3-warm-start-research/src --output doc/taskcard8/replay_warm_NEW
.venv/Scripts/python.exe -B scripts/taskcard8/warm_audit.py --source doc/algorithm_versions/v5.2-search-final/src --output doc/taskcard8/replay_cold_NEW
```

Warm LEMON、新模拟及E组合全部为 `NOT_RUN_CORRECTNESS_GATE_FAILED`。指定LEMON输出位置只有明确0拟合的STATUS，不含虚假OOF或完成标记。

## 冷v5.2独立MSIC模拟

原始协议 `numerical_simulation/outputs_selection_research_task8/plan.json`，初始SHA256为 `f4e84f877af3eedea4c4a3877526054d87c37165bc494b311d644f5a6fba1db0`。共享预算文件 `COMPUTE_BUDGET.json` 冻结7200墙时，开始2026-10-16 14:46:46.261604 UTC、截止16:46:46.261604 UTC；不按事后结果扩时或换更小配置。

36cells：n100/200/400 × SNR.5/2/4 × 连续矩形(4,3,5)/(2,3,4) × rho0/.6；开发/确认各3重复108数据。沿用正式700实际配置k_max1、max_cycles6、n_lambdas20、cv_folds5、max_iter500、tol1e−6、BIC初始化c3和原MSIC；没有用dataclass通用k5/cycles10冒充正式设置。

开发seed基础1806101600、确认2806101600，cell×1000+replicate；test加50000000。432种训练/测试seed唯一，已查11份旧顶层plan的916材料化seed无交集；不推断未公开外部实验。确认数据仅在全部开发结果完整、规则及artifact SHA锁定后生成和评价。

本轮调用形式如下，均已存在目标目录，**不要重新执行覆盖或另起第二个控制器**：

```powershell
.venv/Scripts/python.exe -B -m scripts.taskcard8.selection prepare
.venv/Scripts/python.exe -B -m scripts.taskcard8.selection run
```

主程序拒绝已有目录/残缺续跑；一个正式原规则Cold拟合被动捕获完整库，两个预声明规则1.5/scale1与2/scale0仅在同库训练RSS/形状上选模型，再共同评价同500测试流。每task保存started/complete、原候选trace/全部B、selection score和资格集、机制分解、真支撑和输入/test指纹。

开发门槛已冻结，不只平均PMSE：总体比≤.98、最坏cell比≤1.15、强信号Mode-F1下降≤.02、强信号q欠选率增加≤.05、n100近容量率不增。若开发未过，预先规定的同一双规则独立确认仅作为固定规则复核，不反称已开发合格；没有增加候选或根据LEMON调参。两规则共享库拟合，不能把重复出现的fit_sec_shared当两次拟合或独立算法成本比。

如果预算停止，全部partial/失败记录保留，旧计划不自动续跑。计算重放需在独立副本建立单独标识的审计时间窗/输出，并按相同源及输入指纹核验；重用这些seed只是计算复现，不是新的统计独立确认，不能改旧截止或补跑成原协议完整。

## 训练内部CV和稳定性可行性

输出 `numerical_simulation/outputs_selection_inner_task8`，协议SHA `34e7f5a4d074a16b57f8a4570c2614761bde9ea14c1a27305af4b71c0794c4be`。预声明各重复第一外折的116人，内3折77/78人全部重预处理、重搜支撑；12次真v5.2拟合，外测试评估0。

内/外共同fixed规模48（full原库80），四规则为native、common MSIC、内CV、稳定性辅助MSIC；没有复用完整116人支撑作内验证。原q80在内训练容量之外，不能补造其相同候选CV。CV最小分数已用于选择，是乐观的诊断量，不是无偏泛化估计。

本轮调用 `.venv/Scripts/python.exe -B -m scripts.taskcard8.inner_selection`，现有输出拒覆盖。原 `independent_verify_saved.py` 已核验432候选SSE、12预处理与12最终模型；其结果 `independent_numerical_verification.json` 是实际完成记录，不需要为了做报告重复训练。

## 本轮测试、图形与最终核验

合并工具测试实际17项通过（6主模拟、7训练内、4审计工具），日志 `tool_tests.log`；Warm算法失败是被审算法结果，不冒充工具失败或删除断言。原算法科学门禁31项另存三个独立source目录的日志。

```powershell
$env:PYTHONPATH="$PWD/scripts/taskcard8;$PWD"
.venv/Scripts/python.exe -B -m unittest scripts.taskcard8.test_selection scripts.taskcard8.test_inner_selection scripts.taskcard8.test_warm_audit.AuditHarnessTests -v
```

最终主模拟只读验收脚本为 `scripts/taskcard8/verify_saved.py`：在控制器终态后，对按n/真形状预选的每阶段6任务（若未完成则明确缺失）重新生成输入和共享test，独立用稀疏矩形design重算候选RSS、OLS条件及两套MSIC和所选预测/恢复指标，不重训。它拒绝已有审计输出，可用 `--output` 指定新的文件；不等于216任务全部独立数值重算。

`scripts/taskcard8/finalize_integrity.py` 在终态后另验全部已完成数据：原3253文件保护、32core及依赖SHA、开发锁与确认时间顺序、原A2网格、best-visited轨迹、独立score分解/选择重放及门槛重算，并解压所有候选系数检查有限性、合法支撑和逐数组指纹；这不替代上述预选12例的输入/预测独立数值重建。中途 `INTERIM_INTEGRITY_AUDIT.json` 明确final=false且只覆盖当时140数据；最终记录以 `FINAL_VALIDATION.json` 为准。工具拒覆盖既有审计输出。

```powershell
.venv/Scripts/python.exe -B -m scripts.taskcard8.verify_saved
.venv/Scripts/python.exe -B -m scripts.taskcard8.finalize_integrity
```

绘图沿用Python后端，183mm、6pt下限、PDF/SVG/TIFF/600dpi PNG，源数据全量、对齐/字形/碰撞及目视记录见figures。训练内图保留全部144共同候选；模拟图按实际阶段/完成数标识，不能把部分确认画成完整确认。源代码 `scripts/taskcard8/figures.py`，通过 `--qa-scripts` 指定nature-figure技能的scripts目录。

控制器终态后，`scripts/taskcard8/reports.py` 才允许生成最终10份主线报告；加上Warm两个A报告共12份。正式完整性与SHA索引见 `FINAL_VALIDATION.json` / `DELIVERY_INDEX.json`，主阶段状态见 `STAGE_STATUS.json`。这些文件不能提前作为主模拟完整结论。
