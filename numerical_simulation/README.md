# numerical_simulation

数值模拟实验代码目录。

## 直接运行

### 快速版

```bash
python numerical_simulation/main.py --profile quick
```

说明：`quick` 是轻量化冒烟实验，便于先生成完整结果表和图。
当前 quick 默认对 Tensor-ABESS 使用更稳健的实验参数：`k_max=1`、`criterion=bic`、`c=3.0`。

### 正式版（按 PDF 主设置）

```bash
python numerical_simulation/main.py --profile full
```

说明：`full` 使用 PDF 的主实验规模，运行时间会明显更长。

## 常用选项

只跑部分实验：

```bash
python numerical_simulation/main.py --profile quick --experiments snr sample_size
```

覆盖重复次数：

```bash
python numerical_simulation/main.py --profile full --n-reps 10
```

补充实验：离散矩形支撑

```bash
python numerical_simulation/main.py --profile quick --support-shape discrete
```

## 输出

默认输出到 `numerical_simulation/outputs/`：

- `raw_results.csv`
- `summary_results.csv`
- `consistency_checks.json`
- `plot_*.png`

## 文件说明

- `data_generation.py`：张量数据生成
- `consistency_checks.py`：顺序一致性检查
- `metrics.py`：实验指标
- `methods.py`：Tensor-ABESS / Tensor-GLasso / Vector Lasso / Oracle OLS
- `runner.py`：实验运行与汇总
- `plots.py`：图表生成
- `main.py`：命令行入口
