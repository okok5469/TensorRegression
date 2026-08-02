# Tensor-ABESS 代码线进展报告


**报告范围**：环境搭建 → 阶段 B（MVP）→ 阶段 C（完整实现与测试）  
**核心文献**：Zhu et al. (2020), *A polynomial algorithm for best-subset selection problem*（arXiv:1911.03804 / PNAS）  
**日期**：2026 年 6 月

---

## 一、项目背景与目标

### 1.1 研究问题

在张量回归模型

\[
y_i = \langle X_i, B \rangle + \varepsilon_i, \quad X_i, B \in \mathbb{R}^{p_1 \times p_2 \times p_3}
\]

中，系数张量 \(B\) 具有**矩形支撑**：仅在 \(S_1 \times S_2 \times S_3\) 上非零。目标是同时完成：

1. **变量选择**：估计三个模态的支撑集 \(S_1, S_2, S_3\)
2. **系数估计**：在选定支撑上拟合 \(B\)

应用场景对标脑电张量（脑区 × 频段 × 时间窗），但代码线先从**可验证的合成玩具数据**入手。

### 1.2 阶段安排中的代码线任务

| 阶段 | 代码线目标 | 状态 |
|------|-----------|------|
| **B** | 向量 ABESS → 组 splicing → 极简 BCD（10×5×8 玩具数据） | ✅ 完成 |
| **C** | 完整 Tensor-ABESS + SIC 选稀疏度 + 基线 + 测试 | ✅ 完成 |
| **D** | 大规模数值模拟（64–128 × 5 × 100 等） | ⬜ 待做 |
| **E** | EEG 实证试点 | ⬜ 待做 |

---

## 二、整体思路：自底向上的「垂直切片」

代码线没有等数学规格全部写完再动手，采用**垂直切片（vertical slice）**策略：

```
文献算法（向量 ABESS）
        ↓  理解 splicing 机制，与 abess 包对答案
自实现向量版（BESS.Fix + SIC）
        ↓  推广到「整组变量进/出」
组 splicing（固定两模态 → 单模态选择）
        ↓  三个模态轮流更新
极简 BCD（固定 s1,s2,s3）
        ↓  外层 SIC 网格搜索
完整 Tensor-ABESS
        ↓  对比基线、评价指标、可复现脚本
阶段 C 测试闭环
```

**设计原则**：

- 每一层都能**独立运行和单元测试**
- 每层都有**已知真值的退化检验**（如 p2=p3=1 退化为向量情形）
- 数学线的「可分解性化归」直接落实为 `build_mode_design()` 中的设计矩阵构造

---

## 三、工作流程时间线

### 第 1 步：环境搭建

- 创建专属 Conda 环境 `tensor-abess`（Python 3.11）
- 安装依赖：`numpy`, `scipy`, `scikit-learn`, `abess`, `pytest`, `pymupdf`
- 写入 `environment.yml`、`requirements.txt`、`pyproject.toml`
- 以 editable 模式安装本地包 `tensor-abess`

### 第 2 步：阶段 B — 向量 ABESS（核心文献复现）

**目标**：读懂并实现 Zhu et al. (2020) 的 splicing 算法，与官方 `abess` 包结果一致。

**实现模块**（`src/tensor_abess/vector/`）：

| 文件 | 对应论文算法 | 功能 |
|------|-------------|------|
| `sacrifices.py` | 向前/后向牺牲量 ξ, ζ | 损失、OLS、初始化 |
| `splicing.py` | Algorithm 2 | 交换活跃集/非活跃集 |
| `abess.py` | Algorithm 1 & 3 | `bess_fix(s)` + 自适应 `abess()` |
| `sic.py` | SIC 准则 | 向量稀疏度选择 |

**验证**（`examples/01_compare_abess.py`）：

- 在 n=120, p=40, s=5 合成数据上，自实现与 `abess` 包**支撑集完全一致、系数误差 < 1e-4**
- 5 个单元测试全部通过

### 第 3 步：阶段 B — 组 splicing + 张量 BCD（MVP）

**目标**：将向量 splicing 推广到组级别，拼出能在 10×5×8 玩具数据上恢复已知支撑的极简 BCD。

**关键数学化归**（整个张量算法的命脉）：

固定 \(S_2, S_3\) 后，模态 1 的第 \(g\) 个「组」对应展平切片 \(\mathrm{vec}(X_{:,g,S_2,S_3})\)，组大小为 \(s_2 \times s_3\)。选 \(S_1\) 等价于从 \(p_1\) 个组中选 \(s_1\) 个的**组最优子集问题**。

**实现模块**：

| 模块 | 路径 | 功能 |
|------|------|------|
| 组 splicing | `group/` | 整组进出的 BESS.Fix(s) |
| 模态归约 | `tensor/reduction.py` | 固定两模态 → 设计矩阵 Z + group_index |
| 张量运算 | `tensor/ops.py` | 内积、OLS 重拟合、损失 |
| BCD | `tensor/bcd.py` | 轮流更新 S1→S2→S3 |
| 合成数据 | `utils/data.py` | `make_tensor_data()` |

**验证**（`examples/02_tensor_bcd_demo.py`）：

- 200 样本，10×5×8，真实 s=(3,2,2)
- 1 轮 BCD 精确恢复 S1, S2, S3
- 目标函数单调下降

### 第 4 步：阶段 C — 完整 Tensor-ABESS

**目标**：外层 SIC 在 (s1,s2,s3) 网格上选最优稀疏度，内层 BCD + 组 splicing；同步基线与评价指标。

**新增模块**：

| 模块 | 路径 | 功能 |
|------|------|------|
| 张量 SIC | `tensor/sic.py` | 可加 log 模型空间惩罚 |
| 完整算法 | `tensor/abess.py` | `tensor_abess()` 网格搜索 |
| 初始化策略 | `utils/init.py` | marginal / random / dense |
| 评价指标 | `metrics/support.py` | TPR/FPR、精确恢复、Frobenius |
| 基线方法 | `baselines/` | LASSO、组 LASSO、CP、Tucker |
| 可复现脚本 | `scripts/run_phase_c_benchmark.py` | 固定种子 benchmark |

### 第 5 步：SIC 惩罚修正（阶段 C 收尾）

**问题**：初版张量 SIC 惩罚偏弱，网格搜索选了 (4,3,3) 而非真实 (3,2,2)，18 个测试中 1 个失败。

**原因分析**：

- 增大 s3 时损失略降（如 0.524 → 0.479），但 binomial 惩罚不足以抵消
- 向量版 SIC 用 \(|A|\log(p)\) 同时惩罚「选了几个变量」和「参数个数」；张量版需要类似机制

**修正方案**：

\[
\mathrm{SIC} = n\log(L) + \Big[\log\binom{p_1}{s_1} + \log\binom{p_2}{s_2} + \log\binom{p_3}{s_3} + s_1 s_2 s_3 \log(p_1 p_2 p_3)\Big] \cdot \log\log n
\]

- 前三项：矩形支撑的**选择空间**惩罚
- 最后一项：\(s_1 s_2 s_3\) 个自由系数惩罚（类比向量版 \(|A|\log p\)）

修正后 SIC 正确选出 (3,2,2)，**18/18 测试全部通过**。

---

## 四、算法架构总览

```
                    tensor_abess(X, y)
                           │
           ┌───────────────┴───────────────┐
           │  遍历 (s1,s2,s3) 网格          │
           │  对每个 triplet 计算 SIC       │
           └───────────────┬───────────────┘
                           │
                    tensor_bcd(X, y, s1, s2, s3)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         更新 S1       更新 S2       更新 S3
              │            │            │
              └─ group_bess_fix ────────┘
                           │
                    group_splicing
                           │
              （继承向量 splicing 思想，
               牺牲量改为整组进出）
                           │
                    OLS 重拟合 B
```

**三层算法关系**：

1. **向量层**：列级 splicing，是组 splicing 的原型
2. **组层**：固定组结构，整组变量同时进出活跃集
3. **张量层**：BCD 交替固定两模态，对第三模态做组选择

---

## 五、玩具数据说明

所有测试均使用**人工合成数据**（非真实 EEG），以便对照已知真值。

### 5.1 向量数据 `make_regression_data()`

\[
y = X\beta + \varepsilon, \quad \beta \text{ 仅 } s \text{ 个非零}
\]

- X 列 AR(1) 相关（ρ=0.3）
- 用于向量 ABESS 与 abess 包对比

### 5.2 张量数据 `make_tensor_data()`

\[
y_i = \langle X_i, B \rangle + \varepsilon_i
\]

- X 元素 i.i.d. N(0,1)
- B 仅在随机矩形 S1×S2×S3 上非零（±1 × 缩放）
- 默认维度 10×5×8，稀疏度 (3,2,2)
- SNR 控制信号强度

Demo 典型输出：

```
真实支撑: S1=[0 1 6], S2=[0 1], S3=[2 4]
估计支撑: S1=[0 1 6], S2=[0 1], S3=[2 4]  ✓ 精确恢复
```

---

## 六、测试体系

共 **18 个单元测试**，分三组：

| 测试文件 | 数量 | 覆盖内容 |
|---------|------|---------|
| `test_vector_abess.py` | 5 | 向量 splicing、SIC、OLS 一致性 |
| `test_tensor_bcd.py` | 5 | 组 splicing、BCD 恢复、单调性、向量退化 |
| `test_phase_c.py` | 8 | 张量 SIC、完整 abess、基线、初始化策略 |

**关键「对答案」检验**：

1. **向量 ↔ abess 包**：支撑集 + 系数一致
2. **p2=p3=1 退化**：张量 BCD = 向量 bess_fix
3. **固定 s 的 BCD**：已知 (s1,s2,s3) 时精确恢复支撑
4. **SIC 自适应选 s**：网格上选出真实 (3,2,2)

---

## 七、项目目录结构

```
2026大创/
├── environment.yml          # Conda 环境
├── requirements.txt
├── pyproject.toml           # 包配置 (v0.2.0)
├── docs/
│   └── 阶段BC代码线进展报告.md   ← 本报告
├── src/tensor_abess/
│   ├── vector/              # 向量 ABESS（Algorithm 1–3）
│   ├── group/               # 组 splicing
│   ├── tensor/              # BCD + 张量 SIC + tensor_abess
│   ├── baselines/           # LASSO / 组LASSO / CP / Tucker
│   ├── metrics/             # 支撑恢复评价
│   ├── utils/               # 数据生成 + 初始化
│   └── common/              # 公共 rank 工具
├── tests/                   # 18 个测试
├── examples/
│   ├── 01_compare_abess.py  # 向量 vs abess 包
│   └── 02_tensor_bcd_demo.py # 张量 BCD demo
└── scripts/
    └── run_phase_c_benchmark.py  # 阶段 C 可复现 benchmark
```

---

## 八、遇到的问题与解决

| 问题 | 原因 | 解决 |
|------|------|------|
| conda run 不支持多行 `-c` | Windows PowerShell 限制 | 改用独立 `.py` 脚本 + 直接调用 env python |
| abess 包无 `ic_type="sic"` | 官方只支持 aic/bic/ebic 等 | 对比 demo 改用 EBIC 或固定 support_size |
| 张量 SIC 过选稀疏度 | 惩罚项缺少有效自由度项 | 加入 s1·s2·s3·log(p1p2p3) 惩罚 |
| LassoCV 参数不兼容 | sklearn 版本差异 | 移除 `n_alphas` 参数 |
| 退化检验系数顺序不一致 | 支撑集排序不同 | 按变量下标逐一比较，而非按数组顺序 |

---

## 九、当前成果总结

### 已完成

- ✅ Conda 环境 + Python 包 `tensor-abess` v0.2.0
- ✅ 向量 ABESS 自实现（与 abess 包验证一致）
- ✅ 组 splicing + 模态归约 + BCD
- ✅ 完整 Tensor-ABESS（SIC 网格选 s1,s2,s3）
- ✅ 4 种基线方法封装
- ✅ 支撑恢复评价指标
- ✅ 18 个单元测试全部通过
- ✅ 2 个 demo + 1 个 benchmark 脚本

### 尚未完成（后续阶段）

- ⬜ **阶段 D**：大规模模拟（64–128 × 5 × 100），变稀疏度/SNR/样本量
- ⬜ **阶段 E**：真实 EEG 数据（MNE-Python 预处理）
- ⬜ 多种初始化策略的**系统对比**（阶段 C 已预留接口，未做大规模实验）
- ⬜ STORE 等额外基线（成本较高，计划秋季）
- ⬜ 数学线规格文档 v1（数学线负责人）

---

## 十、使用指南

```powershell
# 激活环境
conda activate tensor-abess
cd "d:\source\learning\2026大创"

# 向量 ABESS 对比 demo
python examples/01_compare_abess.py

# 张量 BCD demo
python examples/02_tensor_bcd_demo.py

# 阶段 C benchmark（固定种子 2026）
python scripts/run_phase_c_benchmark.py --seed 2026

# 运行全部测试
pytest tests/ -v
```

**代码调用示例**：

```python
from tensor_abess import tensor_abess, tensor_bcd, evaluate_support
from tensor_abess.utils.data import make_tensor_data

data = make_tensor_data(n=200, p1=10, p2=5, p3=8, s1=3, s2=2, s3=2)
result = tensor_abess(data["X"], data["y"],
                      s1_grid=[2,3,4], s2_grid=[1,2,3], s3_grid=[1,2,3])
metrics = evaluate_support(result.S1, result.S2, result.S3,
                           data["S1"], data["S2"], data["S3"],
                           result.B, data["B"], 10, 5, 8)
print(result.s1, result.s2, result.s3, metrics.exact_rectangular)
```

---

## 十一、方法论反思

1. **先对答案，再推广**：向量版与 abess 包对齐，提供了可靠的「金标准」；p2=p3=1 退化检验贯穿张量开发全程。

2. **垂直切片优于水平分层**：没有等全部数学推导完成，而是每层都跑通端到端流程，反过来检验数学规格是否正确。

3. **测试驱动**：每个模块都有对应单元测试；SIC 过选问题通过测试暴露，而非人工 inspection。

4. **惩罚项设计需与参数个数挂钩**：张量模型的「稀疏度三元组 (s1,s2,s3)」与「自由参数个数 s1·s2·s3」是两个不同概念，SIC 必须同时惩罚二者。

5. **玩具数据先行**：10×5×8 的小维度使 debug 成本极低；阶段 D 再放大到脑电尺度。

---

*报告生成于代码线阶段 B+C 完成时。如需更新，请在阶段 D 模拟完成后补充数值结果章节。*
