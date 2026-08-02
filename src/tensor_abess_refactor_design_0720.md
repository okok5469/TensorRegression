# Tensor-ABESS 0720 修订版对齐设计图

> 目的：在真正改代码前，固定当前分析结论，避免聊天上下文丢失；同时给出逐文件改动蓝图。
>
> 参考资料：
> - `tensor_splicing_spec_detailed（0720修改版）.pdf`
> - 当前源码目录 `tensor_abess/*`

---

## 0. 已确认的高层结论

### 0.1 旧分析中仍然成立的结论

1. **向量层 `tensor_abess/vector/*` 基本正确**
   - `L_n(beta)=||y-X beta||^2/(2n)`
   - 向量版 sacrifices、splicing、ABESS 主体逻辑基本与 PDF 一致
   - `tau_threshold` 中经验常数属于实现细节，不是主数学冲突

2. **组层 `tensor_abess/group/*` 仍有核心数学偏差**
   - 当前 `compute_group_sacrifices()` 用的是“删/加组后重新 OLS refit 的 loss 差”
   - PDF §4 需要的是闭式：
     - `xi_g = 1/2 * beta_g^T Gamma_g beta_g`
     - `zeta_g = 1/2 * d_g^T Gamma_g^{-1} d_g`
   - 当前 `initial_active_groups()` 少了 Gram 标准化
   - 当前 `group_tau_threshold()` 少了块大小 `d`

3. **张量 reduction / unfolding 顺序与 PDF 记号不完全一致，但代码内部自洽**
   - 当前 NumPy reshape 顺序对应的 Kronecker 顺序与 PDF 的写法不同
   - 若内部始终一致，则不是数值错误
   - 若要“形式上完全贴合 PDF”，需统一说明或改实现

### 0.2 0720 修订版新增并确认的变化

1. **最终张量 SIC 改为**

   `SIC(S1,S2,S3) = n log(RSS/n) + s1*s2*s3*log n + log log n * Σ_m log C(p_m, s_m)`

   与当前代码不同：
   - 当前代码错误地使用了 `s1*s2*s3*log(p1*p2*p3)`，并整体再乘 `log log n`
   - 修订版 PDF 明确区分：
     - 参数维数惩罚：`s1*s2*s3 * log n`
     - 模型空间惩罚：`log log n * Σ log C(p_m, s_m)`

2. **单模态条件 SIC（算法 5.1）改为**

   `n log(RSS_s/n) + s*d_m*log n + log log n * log C(J, s)`

   其中：
   - `J` = 当前模态组数
   - `d_m` = 当前模态块大小

3. **新增算法 5.0：Tensor-Init-GLasso**
   - 先做边际预缩维
   - 再逐模态组 Lasso 路径
   - 得到：
     - `s_{1,max}, s_{2,max}, s_{3,max}`
     - `S_1^(0), S_2^(0), S_3^(0)`

4. **主算法 5.2' 不再是三维网格搜索**
   - 阶段 0：`Tensor-Init-GLasso`
   - 阶段 1：BCD + `GroupBESS.Adaptive`
   - 阶段 2：final OLS + final SIC

---

## 1. 改动总原则

1. **先修组层数学核心，再接入 0720 新外层**
   - 否则外层算法改了，内核仍不符合 PDF §4

2. **统一“loss / RSS”接口**
   - 内部保留 `loss = RSS/(2n)` 也可以
   - 但所有 SIC 计算函数最好显式以 `RSS` 或可无歧义地从 `loss` 转 `RSS`

3. **新增初始化模块时尽量复用已有 group lasso baseline 的近端算子**
   - 当前 `baselines/group_lasso.py` 有 block soft-threshold ISTA 雏形
   - 适合抽成通用 solver

4. **保留 warm-start 接口**
   - 修订版算法 5.1 明确支持 `A_warm`
   - 外层 BCD 阶段也需要用 5.0 的 `S_m^(0)` 做暖启动

---

## 2. 逐文件改动设计图

---

## 2.1 `tensor_abess/group/sacrifices.py`

### 当前问题

1. `compute_group_sacrifices()` 使用了 refit 型定义，不是 PDF §4 的闭式牺牲量
2. `initial_active_groups()` 未进行组内 Gram 标准化
3. `group_tau_threshold()` 漏掉块大小 `d`

### 目标改动

#### A. 新增或重构的基础函数

建议新增：

- `group_gram(X, group_index, g, n=None) -> ndarray`
- `group_corr(residual, X, group_index, g, n=None) -> ndarray`
- `group_block_size(group_index, g) -> int`
- `fit_ols_on_groups(..., return_residual=False, return_corr=False)`

建议把 `fit_ols_on_groups()` 扩展为可返回：

- `beta`
- `loss`
- `residual`

以减少重复计算。

#### B. `compute_group_sacrifices()` 改为闭式定义

输入建议改成：

- `X, y, group_index, active_groups, inactive_groups`
- 或更高效地输入：
  - `beta`
  - `residual`
  - `X`
  - `group_index`
  - `active_groups`
  - `inactive_groups`

输出：

- `xi`
- `zeta`
- 可选 `gram_cache`
- 可选 `corr_cache`

数学上应实现：

- 对 `g in active`：
  - `Gamma_g = Z_g^T Z_g / n`
  - `beta_g = beta[cols_g]`
  - `xi[g] = 0.5 * beta_g^T Gamma_g beta_g`

- 对 `g in inactive`：
  - `d_g = Z_g^T residual / n`
  - `zeta[g] = 0.5 * d_g^T Gamma_g^{-1} d_g`

数值实现建议：

- 不直接写 `inv(Gamma_g)`，优先：
  - `np.linalg.solve(Gamma_g, d_g)`
- 若 `Gamma_g` 病态：
  - 可切到 `np.linalg.pinv`
  - 需要在 docstring 里说明“数值稳定替代，不改变理论形式”

#### C. `initial_active_groups()` 改为 PDF 版本

应改为按以下分数选前 `s` 个组：

`score_g = ||Gamma_g^{-1/2} (Z_g^T y / n)||_2`

实现上等价于：

`score_g^2 = (Z_g^T y / n)^T Gamma_g^{-1} (Z_g^T y / n)`

因此无需显式求 `Gamma^{-1/2}`，可用 solve。

#### D. `group_tau_threshold()` 改接口

从：

- `group_tau_threshold(s, n_groups, n)`

改成：

- `group_tau_threshold(s, n_groups, n, block_size, c=0.01)`

公式按 PDF 量级：

`tau ~ c * s * block_size * log(n_groups) * log log n / n`

说明：
- `c` 仍可保留为工程参数
- 与向量版兼容时，`block_size=1`

---

## 2.2 `tensor_abess/group/splicing.py`

### 当前问题

依赖了错误的 group sacrifices 定义。

### 目标改动

#### A. `group_splicing()` 接口扩展

建议改成接受：

- `beta_init: np.ndarray | None = None`
- `residual_init: np.ndarray | None = None`
- `gram_cache: dict | None = None`

最小目标：
- 至少允许从上层传入初始 `beta`
- 减少重复 OLS 计算

#### B. 内部流程

1. 用当前 `active_groups` 做 OLS，得到：
   - `beta`
   - `loss0`
   - `residual`
2. 基于闭式定义算 `xi, zeta`
3. 对 `k=1..k_max`：
   - 选出 `out_g`
   - 选出 `in_g`
   - 构造 `new_active`
   - 在 `new_active` 上 refit OLS
   - 保留最优交换
4. 若 `loss0 - best_loss < tau_s`，返回原解

#### C. 排名逻辑保持不变

- `bottom_k_by_rank(xi, active, k)`
- `top_k_by_rank(zeta, inactive, k)`

可保留。

---

## 2.3 `tensor_abess/group/bess_fix.py`

### 当前问题

1. 没有 `beta^0` / `A^0` 显式暖启动接口
2. `tau` 未使用块大小 `d`
3. 无法直接支撑算法 5.1 的 `GroupBESS.Adaptive`

### 目标改动

#### A. `group_bess_fix()` 新接口

建议新增参数：

- `beta_init: np.ndarray | None = None`
- `groups_init: np.ndarray | None = None`
- `block_size: int | None = None`
- `tau: float | None = None`

优先级建议：

1. 若外部直接给 `tau`，用外部值
2. 否则用 `group_tau_threshold(s, n_groups, n, block_size)`

#### B. 初始化逻辑

与 PDF 5.1 对齐：

- 若传了 `groups_init`，直接使用
- 若还传了 `beta_init`，则首次迭代可直接使用
- 否则走 `initial_active_groups()`

#### C. 返回结果增强

`GroupBessFixResult` 建议增加：

- `rss: float`
- `residual: np.ndarray | None = None`
- `beta_active_cols: np.ndarray | None = None`（可选）

至少要有 `rss`，因为 adaptive SIC 直接用 `RSS_s`

---

## 2.4 `tensor_abess/group/__init__.py`

### 目标改动

新增导出：

- `group_bess_adaptive`

如果新建 `adaptive.py`，则从那里导出；如果直接写进 `bess_fix.py`，也需要更新 `__all__`。

---

## 2.5 `tensor_abess/group/adaptive.py`（建议新增）

### 为什么建议新增

当前 `group/bess_fix.py` 已承载固定 `s` 的逻辑；修订版算法 5.1 是新的层次，单独拆开更清晰。

### 目标函数

实现 `GroupBESS.Adaptive（修正版）`

建议 dataclass：

- `GroupAbessResult`
  - `beta`
  - `active_groups`
  - `s`
  - `sic`
  - `rss`
  - `sic_path`
  - `bess_path`

### 建议接口

`group_bess_adaptive(y, X, group_index, s_max, *, k_max=None, m_max=20, block_size=None, groups_warm=None, beta_warm=None, tau=None, n_groups=None)`

### 流程

对 `s = 1..s_max`：

1. 若 `groups_warm` 存在且 `s == len(groups_warm)`：
   - `A0 = groups_warm`
   - `beta0 = beta_warm`（若可用）
2. 否则：
   - `A0 = initial_active_groups(..., s)`
3. 调 `group_bess_fix(...)`
4. 记 `RSS_s`
5. 用修正版条件 SIC：

   `n log(RSS_s/n) + s * block_size * log n + log log n * log C(J, s)`

6. 取最优 `s`

---

## 2.6 `tensor_abess/tensor/sic.py`

### 当前问题

当前 SIC 公式与 0720 修订版不一致。

### 目标改动

#### A. 最终 SIC 改公式

建议保留：

- `log_binomial()`

改写：

- `compute_tensor_sic(...)`

为：

`SIC = n log(RSS/n) + s1*s2*s3*log n + log log n * [logC(p1,s1)+logC(p2,s2)+logC(p3,s3)]`

#### B. 接口设计建议

可二选一：

1. 输入 `loss`，内部转 `RSS = 2n*loss`
2. 新接口直接输入 `rss`

推荐：
- 新增 `compute_tensor_sic_from_rss(rss, ...)`
- `compute_tensor_sic(loss, ...)` 作为兼容包装

#### C. 新增单模态 SIC 工具

建议新增：

- `compute_mode_sic_from_rss(rss, s, n, J, block_size)`

公式：

`n log(RSS/n) + s*block_size*log n + log log n * log C(J, s)`

#### D. `default_sparsity_grid()` 将被弱化或废弃

因为新主算法不再依赖三维 grid search。

处理建议：
- 先保留但标注 deprecated
- 后续若不再使用，可删除

---

## 2.7 `tensor_abess/utils/init.py`

### 当前问题

只有简单 `marginal/random/dense` 初始化，没有实现算法 5.0。

### 目标改动

#### A. 保留旧接口，但新增 0720 初始化器

保留：

- `initial_support(...)`

新增：

- `TensorInitResult` dataclass
  - `S1_init`
  - `S2_init`
  - `S3_init`
  - `s1_max`
  - `s2_max`
  - `s3_max`
  - `candidate_sets`（可选）
  - `glasso_paths`（可选）

- `tensor_init_glasso(X, y, c=2.0, *, cv_folds=5, criterion="cv"|"bic", random_state=0)`

#### B. 算法 5.0 的 Step A

新增内部工具：

- `marginal_slice_scores(X, y, mode) -> ndarray`
- `top_candidate_slices(scores, n, p_m) -> ndarray`

候选大小：

`M_m = min(floor(n^(2/3)), p_m)`

#### C. 算法 5.0 的 Step B

需要能够在任意模态构造：

- 候选组数 = `|C_m|`
- 块大小 = `|S_tilde_-m|`

因此建议不要直接复用 `build_mode_design()` 原始版本，而新增一个支持“候选子集 + 固定其他模态支撑”的构造器。

例如：

- `build_mode_design_subset(X, mode, candidate_mode_indices, other_support_1, other_support_2)`

输出：

- `Z`
- `group_index`
- `candidate_groups`

#### D. 算法 5.0 的 Step C

把 PDF 公式原样实现：

`s_{m,max} = min(c * s_tilde_m, floor((n / (3*max_m' s_tilde_{m'}))^(1/2)))`

并增加工程保护：

- 至少为 1
- 至多不超过 `p_m`
- 最终若 `s1_max*s2_max*s3_max >= n`，可再保守裁剪

---

## 2.8 `tensor_abess/utils/glasso.py`（建议新增）

### 为什么建议新增

算法 5.0 需要通用组 Lasso；当前 `baselines/group_lasso.py` 仅支持 mode-1 baseline，不适合作为主算法模块直接复用。

### 建议内容

- `group_soft_threshold(beta, group_index, lam)`
- `fit_group_lasso_ista(Z, y, group_index, lam, *, max_iter, tol, beta_init=None)`
- `group_lasso_path(Z, y, group_index, lambdas, *, max_iter, tol, warm_start=True)`
- `select_group_lasso_lambda_cv(...)`
- `select_group_lasso_lambda_bic(...)`

### BIC 版本建议

若按 group-support 维数近似：

- `RSS(lambda)`
- `df(lambda) = (#active groups) * block_size`

则可用

`BIC = n log(RSS/n) + df log n`

这里是初始化器内部的 `lambda` 选取准则，不必与最终 SIC 混淆。

---

## 2.9 `tensor_abess/tensor/reduction.py`

### 当前状态

现有 `build_mode_design()` 只支持“全模态候选组 + 固定另两模态支撑”。

### 目标改动

#### A. 保留现有接口

因为主算法阶段 1 仍需要“全模态组更新”。

#### B. 新增子集版设计矩阵构造器

为初始化算法 5.0 服务，建议新增：

- `build_mode1_design_subset(X, candidate_S1, S2, S3)`
- `build_mode2_design_subset(X, S1, candidate_S2, S3)`
- `build_mode3_design_subset(X, S1, S2, candidate_S3)`
- 或统一接口：
  - `build_mode_design_subset(X, mode, candidate_groups, S1, S2, S3)`

输出应包含：

- `Z`
- `group_index`
- `candidate_to_original_index`

这样 group lasso 选中的组号可映射回原始模态索引。

#### C. Kronecker / reshape 顺序

不强制立即修改，但需要在 docstring 里明确：

- 当前实现采用 NumPy reshape 顺序
- 与 PDF 公式记号若不同，以代码内部一致性为准

---

## 2.10 `tensor_abess/tensor/bcd.py`

### 当前问题

现在是“固定 `(s1,s2,s3)` 的 BCD”。

### 修订版目标

改造成算法 5.2' 的阶段 1 主循环器，或者拆成：

1. `tensor_bcd_fixed_sizes(...)`（保留旧实现）
2. `tensor_bcd_adaptive(...)`（新增修订版实现）

推荐做法：**保留旧函数，新增修订版函数**，降低回归风险。

### 新增 dataclass 建议

- `TensorBcdAdaptiveResult`
  - `B`
  - `S1`
  - `S2`
  - `S3`
  - `s1`
  - `s2`
  - `s3`
  - `loss`
  - `rss`
  - `sic`
  - `loss_history`
  - `sic_history`
  - `n_cycles`

### 新函数建议

`tensor_bcd_adaptive(X, y, *, s1_max, s2_max, s3_max, S1_init, S2_init, S3_init, k_max=None, m_max=20, tau=None, eps=1e-8)`

### 内部流程

每个 cycle：

1. mode-1:
   - `d1 = |S2|*|S3|`
   - `Z1 = build_mode_design(...)`
   - 调 `group_bess_adaptive(..., s_max=s1_max, block_size=d1, groups_warm=S1)`
   - 更新 `S1, s1`
2. mode-2:
   - 同理
3. mode-3:
   - 同理
4. final refit 当前矩形支撑上 OLS
5. 计算当前全局 SIC
6. 若支撑不变或 `|ΔSIC| < eps`，停止

---

## 2.11 `tensor_abess/tensor/abess.py`

### 当前问题

它实现的是三维 grid-search 版 tensor ABESS，不再对应修订版算法 5.2'。

### 目标改动

建议保留现有函数作为：

- `tensor_abess_grid(...)`

新增/重写主入口：

- `tensor_abess(...)`

使之对应修订版算法 5.2'。

### 新主流程

1. `init_res = tensor_init_glasso(X, y, c=...)`
2. `bcd_res = tensor_bcd_adaptive(X, y, s1_max=..., s2_max=..., s3_max=..., S1_init=..., S2_init=..., S3_init=...)`
3. 最终输出：
   - `B`
   - `S1,S2,S3`
   - `s1,s2,s3`
   - `loss/rss`
   - `sic`
   - `histories`
   - `init summary`

### dataclass 建议扩展

`TensorAbessResult` 新增：

- `rss`
- `init_supports`
- `s_max`
- `sic_history`
- `init_details`（可选）

---

## 2.12 `tensor_abess/tensor/__init__.py`

### 目标改动

更新导出：

- `tensor_bcd_adaptive`
- `tensor_abess_grid`（若保留旧接口）
- `compute_mode_sic_from_rss`

---

## 2.13 `tensor_abess/__init__.py`

### 当前问题

顶层导入 `baselines` 会在缺 `sklearn` 时影响包导入。

### 这次是否必须修改

不是 0720 PDF 主线内容，但建议顺手处理。

### 建议做法

两种可选：

1. 延迟导入 baselines
2. `try/except ModuleNotFoundError`

目标：不让核心 `tensor_abess.group/tensor/vector` 因 baseline 依赖缺失而失败。

---

## 2.14 `tensor_abess/baselines/group_lasso.py`

### 当前状态

有可复用的近端 group lasso 雏形，但仅适配 mode-1 baseline。

### 建议处理

不要直接把 baseline 函数硬改成主算法工具，避免语义混乱。

更好做法：

- 抽出共用近端函数到 `utils/glasso.py`
- baseline 文件反向调用通用工具

---

## 3. 建议实现顺序

### 第一阶段：修核心数学

1. `group/sacrifices.py`
2. `group/splicing.py`
3. `group/bess_fix.py`
4. 新增 `group/adaptive.py`

### 第二阶段：修 SIC

5. `tensor/sic.py`

### 第三阶段：补初始化

6. `utils/glasso.py`
7. `tensor/reduction.py` 子集构造器
8. `utils/init.py` 的 `tensor_init_glasso`

### 第四阶段：改主算法外层

9. `tensor/bcd.py`
10. `tensor/abess.py`
11. `tensor/__init__.py`
12. `tensor_abess/__init__.py`

---

## 4. 实现时必须验证的关键点

1. **RSS / loss 换算是否一致**
   - `loss = RSS/(2n)`
   - SIC 里应使用 `RSS/n`，常数项处理要统一

2. **组 Lasso 初始化的 group 索引映射**
   - 候选集 `C_m` 不是原始连续索引时，必须保存映射

3. **`s_{m,max}` 的保护裁剪**
   - 防止出现 0
   - 防止三者乘积过大导致最终 OLS 不可解

4. **warm start 仅在 `s == |A_warm|` 时使用**
   - 不能误做成整条路径 warm start

5. **病态 Gram 矩阵**
   - `solve` 优先，必要时 `pinv`

6. **停止准则**
   - support 不变
   - 或 `ΔSIC < eps`

---

## 5. 推荐后续测试清单

1. **单组/单变量退化测试**
   - 组大小 `d=1` 时，组版应退化到向量版

2. **向量退化测试**
   - `p2=p3=1, s2=s3=1` 时，张量 SIC 应退化到
     - `n log L_A + |A| log n + |A| log p log log n`
   - 允许存在常数差，但排序应一致

3. **group sacrifices 数值验证**
   - 闭式 `xi/zeta` 与 PDF 小例子一致

4. **初始化器输出合理性**
   - `C_m` 大小
   - `S_tilde_m`
   - `s_max`

5. **BCD 收敛轨迹**
   - `loss_history` 单调不增
   - `sic_history` 基本下降或最终稳定

---

## 6. 本文件的用途

本文件是**后续改代码的唯一基线设计图**。如果聊天上下文丢失，至少保留以下关键信息：

1. 0720 PDF 的 SIC 已改：
   - final：`n log(RSS/n) + s1 s2 s3 log n + log log n * Σ log C(p_m,s_m)`
   - mode-wise：`n log(RSS_s/n) + s d_m log n + log log n * log C(J,s)`

2. 0720 PDF 新增算法 5.0：
   - `Tensor-Init-GLasso`

3. 主算法已从 grid-search 改为：
   - 初始化 + BCD + `GroupBESS.Adaptive`

4. group 核心 closed-form sacrifices 仍必须修正，不能沿用 refit 型近似

