# q=80 外折独立诊断

目标记录为正式结果 `repeat_2_fold_5`（repeat零起始、outer_fold一开始；等价零起始外折4）。本模块不写入正式输出，不修改核心算法。

通过仅诊断进程内的返回值保真包装，记录未修改ABESS的初始化、各模态MSIC路径、精修前解、全局候选、one-SE筛选与最后解；包装测试要求系数与未包装调用逐元素一致。全部15外折重放也与保存的B逐元素比较。

内部验证仅使用该折116名外层训练受试者；预先固定3次5折种子20261214/20261215/20261216，重新拟合每个内折预处理及支持位置。固定规模q=1,3,6,12,24,48,80及协变量OLS，不用完整外层训练数据选出的支持作为内折输入。五折内部train=92或93，减去8协变量后仍允许q80满列秩；验收会检查秩并在不可识别时停止。它是围绕既有异常的探索性内部诊断，不是新的独立正式测试。

保留当前fixed BCD行为及其非单调问题，逐拟合记录RSS增加次数。因此结果比较的是当前搜索实现的规模/泛化表现，不能声称各规模达到全局最优或已修复搜索。one-SE也按当前启发式MSIC容差准确重建，不称为交叉验证标准误。

项目根目录，现有Python环境中运行：

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest empirical_analysis.lemon.fold80_diagnostic.test_diagnostic -v
python -m empirical_analysis.lemon.fold80_diagnostic.diagnostic
```

本地使用 `.venv/Scripts/python.exe`。输出位于 `empirical_analysis/lemon/outputs_fold80_diagnostic/`；已存在非空输出会拒绝覆盖，复查可只读现有CSV/JSON。

主要证据：replays/逐阶段路径、fold_stage_comparison.csv、target_candidate_scores.csv、target_ols_audit.json、internal_validation_config.json、internal_splits.csv、internal_fold_metrics.csv、internal_oof_predictions.csv、internal_metrics_by_repeat.csv、internal_model_summary.csv及run_state.json。正式输出、旧分析输入及核心的哈希在运行前后核验。
