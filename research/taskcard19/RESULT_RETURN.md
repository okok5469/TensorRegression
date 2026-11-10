# 统一结果回传

在新环境运行后：

```text
python -m tensor_abess_ovr.cli validate-return --run <run_dir>
python -m tensor_abess_ovr.cli pack-return --run <run_dir> --output <return.zip>
```

打包器先核对hash、逐样本覆盖/重复键、method/fold/用途、score行身份及类别列、非有限数、支撑范围和q、预测决策与重算指标。篡改预测/缺少一行不能视为完整成功。请勿手工修改结果表后重封hash。

回传保留manifest、冻结配置/splits/environment、SHA清单、predictions.csv、scores.npz及行/列映射、metrics.csv、混淆矩阵、supports.jsonl、fit_diagnostics.jsonl、保存模型、RUN_REPORT.md与运行日志。分数不是概率；不计算A路线概率Log-loss或概率Brier。

失败与未完成也回传，状态须准确区分完整与PARTIAL。默认不包含原始X、venv、凭据或历史归档。每次运行用全新ID/目录，不覆盖之前结果；当前版不复用拟合缓存，身份记录用于独立复核，避免将半成品当缓存。
