# 2026-08-02 numerical-simulation-runtime-guardrails design

## Goal
在不改变 full 实验核心语义和四个方法默认集合的前提下，降低长任务不可观测风险，并支持分批重跑。

## Approved scope
1. 将 `full` 默认 `n_reps` 从 50 调整为 10。
2. 支持按实验组分批运行，避免一次性提交整包 full。
3. 为每个 repetition 增加进度日志。
4. 为每个 repetition 增量写盘，并提供 checkpoint / resume。
5. 不删除 `Tensor-ABESS`、`Tensor-GLasso`、`Vector Lasso`、`Oracle OLS` 四个默认方法。

## Non-goals
1. 不改算法数学逻辑。
2. 不默认删除重基线。
3. 不引入并行或大规模重构。

## Design
- `numerical_simulation/main.py`
  - `full` 默认 `n_reps=10`。
  - 增加 `--resume` 开关。
- `numerical_simulation/runner.py`
  - 以 repetition 为原子单位落盘 `raw_results.csv`。
  - 若开启 `resume`，跳过已完成的 `(experiment, scenario, replicate)`。
  - 每完成一个 repetition 输出一行进度日志。
  - 最终再汇总 `summary_results.csv`、一致性检查和图。
- 保持现有 `--experiments` 过滤能力，作为 full 分批运行入口。

## Safety
- checkpoint 只追加已完成 repetition 的结果，不写半成品方法行。
- resume 只依据完整 repetition 去重，避免重复统计。
- 若历史 CSV 字段不匹配，则按当前字段重建。
