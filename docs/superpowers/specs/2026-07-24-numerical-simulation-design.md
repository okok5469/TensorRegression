# 数值模拟实验模块设计

## 目标

在仓库根目录新增 `numerical_simulation/` 文件夹，实现一套可直接运行的数值模拟实验代码，按两份 PDF 的要求：

- 生成三阶张量回归模拟数据；
- 调用现有 `src/tensor_abess` 核心算法完成 Tensor-ABESS 实验；
- 实现对照方法 Tensor-GLasso、Vector Lasso、Oracle OLS；
- 输出原始结果、汇总表和图表；
- 保留 `quick` 与 `full` 两种运行档位。

## 目录设计

`numerical_simulation/` 内计划包含：

- `__init__.py`：包导出；
- `data_generation.py`：AR(1) 协方差、矩形支撑、SNR 缩放、训练/测试数据生成；
- `consistency_checks.py`：内积与张量设计矩阵顺序一致性检查；
- `metrics.py`：变量选择、估计、预测、效率指标；
- `methods.py`：Tensor-ABESS / Tensor-GLasso / Vector Lasso / Oracle OLS；
- `runner.py`：单次重复、实验扫描、汇总落盘；
- `plots.py`：生成 PDF 中建议的图表；
- `main.py`：CLI 主入口；
- `README.md`：运行说明；
- `outputs/`：默认输出目录。

## 数据流

1. 根据基准配置或单因素变化配置生成真实系数张量与协变量协方差；
2. 按 separable covariance + AR(1) 结构生成训练张量；
3. 通过解析公式缩放真实系数，使目标 SNR 精确满足设定；
4. 在同一组训练数据上运行所有比较方法；
5. 独立生成测试样本并计算 PMSE / SignalMSE / R²；
6. 汇总重复实验结果并绘图。

## 实验设计

默认覆盖 PDF 建议的四类单因素实验：

1. SNR：`{0.5, 1, 2, 4}`
2. 样本量：`{100, 200, 400}`
3. 维度：`(16,12,15)`, `(32,20,25)`, `(64,30,40)`
4. 相关性：`rho in {0, 0.3, 0.6}`

其中：

- `quick`：较少重复次数，保证能快速生成完整表图；
- `full`：采用 PDF 推荐正式设置（默认 50 次重复）。

## 方法实现

### Tensor-ABESS

直接调用现有 `tensor_abess.tensor.abess.tensor_abess()`，统一设置实验参数，如：

- `k_max=5`
- `max_cycles=10`
- 组 Lasso 初始化使用 CV

### Tensor-GLasso

复用现有张量分组设计矩阵构造，在 BCD 外层下逐模态调用 group lasso 选择组支撑，再做最终 OLS 重拟合。

### Vector Lasso

将张量向量化后调用 `sklearn.linear_model.LassoCV`，再由非零元素诱导三个模态支撑集合。

### Oracle OLS

使用真实矩形支撑做受限 OLS，作为下界参考。

## 指标

实现以下核心指标：

- 模态 Precision / TPR / F1
- 宏平均 Mode-F1
- Exact Recovery
- 元素级矩形 Precision / Recall / F1 / FDR
- 估计稀疏度 `ŝ1, ŝ2, ŝ3, q̂`
- REE
- PMSE
- SignalMSE
- Test R²
- 运行时间与外层循环次数

## 输出

输出至少包括：

- `raw_results.csv`
- `summary_results.csv`
- `consistency_checks.json`
- 图表 PNG 文件

图表对应 PDF 推荐内容：

- SNR vs 选择准确率
- 样本量 vs 预测误差
- 总维度 vs 运行时间
- 相关性 vs 性能
- 基准设置下估计稀疏度箱线图

## 测试策略

先写失败测试，再实现：

1. 小规模数据生成与目标 SNR 一致；
2. 一致性检查通过；
3. Tensor-GLasso 能返回合法结果；
4. 单次重复能产出四种方法结果与关键指标；
5. 小规模实验套件能写出表格和图。

## 自检

- 不新增对外部网络的依赖；
- 保持与现有 `src/tensor_abess` 接口兼容；
- 默认主脚本可直接运行；
- 高维测试误差采用分批生成，避免不必要的测试集内存峰值。
