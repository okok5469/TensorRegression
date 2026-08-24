# 张量回归组 Splicing 变量选择的详细数学规格与推导

---
## 目录

0. 预备：张量运算与记号
1. 模型设定
2. 矩形支撑的精确定义
3. 可分解性的化归
4. 组 Splicing 子问题
5. BCD 外层循环 + SIC 选稀疏度
6. SIC 的具体形式
7. 收敛性与复杂度
8. 与向量版 ABESS 的对照

---

## 0. 预备：张量运算与记号

我们需要三样张量工具，先说清楚它们是什么、为什么要用。

**（a）张量内积.** 对两个同型张量 $\mathcal{A},\mathcal{B}\in\mathbb{R}^{p_1\times p_2\times p_3}$，

$$
\langle\mathcal{A},\mathcal{B}\rangle
:=\sum_{j_1=1}^{p_1}\sum_{j_2=1}^{p_2}\sum_{j_3=1}^{p_3}\mathcal{A}_{j_1 j_2 j_3}\,\mathcal{B}_{j_1 j_2 j_3}.
$$

它就是把两个张量当成长向量做的普通点积。回归里用它来定义"协变量张量对系数张量的线性作用"——这样模型才是关于 $\mathcal{B}$ 的**线性**模型，才能套用最小二乘和 Splicing 那套线性理论。

**（b）向量化 $\mathrm{vec}(\cdot)$.** 把张量按固定顺序拉直成一个 $p_1 p_2 p_3$ 维的列向量。它的用途是让我们能写成矩阵形式 $y=\mathbb{X}\mathrm{vec}(\mathcal{B})+\varepsilon$，从而借用一切向量回归的工具。关键约束：**$\mathcal{A}$ 和 $\mathcal{B}$ 必须用同一个顺序拉直**，否则 $\langle\mathcal{A},\mathcal{B}\rangle\ne\mathrm{vec}(\mathcal{A})^\top\mathrm{vec}(\mathcal{B})$。这个"顺序一致"的原则在 §3 里会反复用到。

**（c）模态展开 $\mathcal{X}_{(m)}$.** 把张量沿第 $m$ 个模态重排成一个矩阵：第 $m$ 个模态的指标做行，其余模态的指标合并做列。我们用它是因为：**选变量的时候，我们是沿某一个模态在挑切片**（例如挑哪些脑区），把那个模态摆成"行"，问题就变成"选哪些行对应的块"，正好是组选择的语言。展开的列顺序有固定约定（§3.4 给出并解释）。

**其它记号.**

| 记号                              | 含义                                                                 |
| ------------------------------- | ------------------------------------------------------------------ |
| $[p]$                           | 集合$\{1,2,\dots,p\}$                                                |
| $S_m\subseteq[p_m]$             | 模态 $m$ 的活跃指标集，$\lvert S_m\rvert=s_m$                               |
| $U_m\in\{0,1\}^{p_m\times s_m}$ | $S_m$ 的**列选择矩阵**：第 $\ell$ 列 $=e_{S_m[\ell]}$（第 $S_m[\ell]$ 个标准基向量） |
| $\otimes$                       | Kronecker 积                                                        |
| $\mathcal{G}=\{G_1,\dots,G_J\}$ | 列指标的组划分，$Z_{G_g}$ 是第 $g$ 组对应的列子矩阵                                  |
| $\Gamma_g,\ \hat d_g$           | 组 Gram 矩阵与组相关向量（§4）                                                |

$U_m$ 是纯粹的"抽列/抽行"工具：$U_m^\top v$ 可以抽出 $v$ 中 $S_m$ 位置的分量，$A U_m$ 抽出 $A$ 中 $S_m$ 对应的列。用矩阵来表达"抽取"，是为了后面能把三重求和压缩成一行 Kronecker 表达式，便于编程。

---

## 1. 模型设定

**观测与模型.** 有 $n$ 个独立观测 $(\mathcal{X}_i,y_i)_{i=1}^n$，$y_i\in\mathbb{R}$，协变量为张量 $\mathcal{X}_i\in\mathbb{R}^{p_1\times p_2\times p_3}$。张量线性回归模型：

$$
y_i \;=\; \langle \mathcal{X}_i,\,\mathcal{B}\rangle \;+\; \varepsilon_i,
\qquad i=1,\dots,n,
\tag{1.1}
$$

$\mathcal{B}$ 是待估的系数张量，$\varepsilon_i$ 独立同分布、均值零、亚高斯尾（存在 $\sigma>0$ 使 $\mathbb{P}\{|\varepsilon_i|\ge t\}\le 2e^{-t^2/\sigma^2}$，$\forall t\ge0$）。

*为什么用亚高斯而不直接用正态* ：亚高斯只约束尾部衰减速度，比正态弱，却已足以推出所有集中不等式与 oracle 性质，因此结论更普适。这也与 Zhu et al.(2020) 的条件一致。

**为什么模型对 $\mathcal{B}$ 是线性的.** 因为 $\langle\mathcal{X}_i,\mathcal{B}\rangle$ 对 $\mathcal{B}$ 的每个分量都是线性的，模型 (1.1) 本质上就是普通线性回归，只是把系数排成了张量形状。张量的形状承载"多模态结构信息"，而线性性让我们能用最小二乘与 Splicing。

**矩阵形式.** 令 $\mathbb{X}\in\mathbb{R}^{n\times p_1 p_2 p_3}$ 的第 $i$ 行为 $\mathrm{vec}(\mathcal{X}_i)^\top$，$\boldsymbol\beta=\mathrm{vec}(\mathcal{B})$，$y=(y_1,\dots,y_n)^\top$。由内积与向量化的定义 $\langle\mathcal{X}_i,\mathcal{B}\rangle=\mathrm{vec}(\mathcal{X}_i)^\top\mathrm{vec}(\mathcal{B})$，于是

$$
y=\mathbb{X}\boldsymbol\beta+\varepsilon.
\tag{1.2}
$$

**损失函数.**

$$
L_n(\mathcal{B})\;=\;\frac{1}{2n}\sum_{i=1}^n\bigl(y_i-\langle\mathcal{X}_i,\mathcal{B}\rangle\bigr)^2
\;=\;\frac{1}{2n}\bigl\lVert y-\mathbb{X}\boldsymbol\beta\bigr\rVert_2^2.
\tag{1.3}
$$

*为什么带 $\tfrac{1}{2n}$？* 两个常数各有用处：
- $\tfrac1n$ 把损失变成"平均"而非"求和"，使各类条件（如信号强度、阈值）不随样本量机械地放大，量纲更干净；
- $\tfrac12$ 纯粹为了求导方便——对 $\boldsymbol\beta$ 求梯度时 $\tfrac12\cdot 2=1$，梯度写成 $-\tfrac1n\mathbb{X}^\top(y-\mathbb{X}\boldsymbol\beta)$，没有多余系数。

直接对 $\boldsymbol\beta\in\mathbb{R}^{p_1 p_2 p_3}$ 做无结构 $\ell_0$ 选择会有两个致命问题：候选子集数 $\binom{p_1 p_2 p_3}{s}$ 天文数字；且选出的非零元散点分布，破坏了张量的模态结构、无法解释。下一节的结构化组稀疏正是为解决这两点。

---

## 2. 矩形支撑的精确定义

**定义 2.1（矩形支撑）.** 称 $\mathcal{B}$ 具有矩形支撑，若存在三个模态活跃集 $S_1\subseteq[p_1],S_2\subseteq[p_2],S_3\subseteq[p_3]$，使

$$
\mathcal{B}_{j_1 j_2 j_3}=0
\quad\text{only if}\quad (j_1,j_2,j_3)\notin S_1\times S_2\times S_3.
\tag{2.1}
$$

等价地 $\operatorname{supp}(\mathcal{B})\subseteq S_1\times S_2\times S_3$。也就是说，非零元被约束在三个模态活跃指标集的**笛卡尔积**（一个"实心矩形块"）里。

**参数计数与模型空间.** 矩形支撑下，潜在非零参数个数是 $s_1 s_2 s_3$（块内每个位置都可能非零）。但"有多少个候选矩形支撑"是另一回事：给定尺寸 $(s_1,s_2,s_3)$，候选支撑数为

$$
\binom{p_1}{s_1}\binom{p_2}{s_2}\binom{p_3}{s_3}.
$$

这两个数（参数个数 $s_1 s_2 s_3$ 与模型个数 $\prod_m\binom{p_m}{s_m}$）在 §6 会区分开——SIC 惩罚用的是后者的对数，而非前者。请先记住这个区别。

**与其它稀疏结构对比.**

| 结构 | 支撑约束 | 选出的东西 | 是否可分解为组选择 |
|---|---|---|---|
| 逐元素稀疏 | $\lVert\mathrm{vec}(\mathcal{B})\rVert_0\le s$ | 散点 | 否 |
| 逐纤维/逐行稀疏 | 仅单一模态成组 | 一条条纤维 | 部分 |
| **矩形支撑** | $\operatorname{supp}\subseteq S_1\times S_2\times S_3$ | 三张模态清单 | **是（命题 1）** |

**结构化最优子集问题.** 给定 $(s_1,s_2,s_3)$，我们要解

$$
\min_{\substack{S_1,S_2,S_3\\ \lvert S_m\rvert\le s_m}}\;
\underbrace{\min_{\operatorname{supp}(\mathcal{B})\subseteq S_1\times S_2\times S_3} L_n(\mathcal{B})}_{\text{给定支撑的最小二乘}}.
\tag{P}
$$

内层是普通最小二乘；难点在外层对 $(S_1,S_2,S_3)$ 的联合搜索，有 $\prod_m\binom{p_m}{s_m}$ 个候选。§3 用可分解性把这个联合搜索拆开。

---

## 3. 可分解性的化归（算法命脉）

**这一节是整个算法的地基。** 固定其余两个模态的支撑，第三个模态上的选择精确等价于一个标准的组最优子集选择问题。下面从三重求和出发，一步步推导，并将等价后的**设计矩阵与响应向量显式写出来**。以模态 1 为例（模态 2、3 对称，§3.5 给表）。

### 3.1 出发点：固定两个模态后，求和复杂度塌缩

固定模态 2、3 的当前活跃集 $S_2,S_3$。此时 $\mathcal{B}$ 的支撑被限制在 $[p_1]\times S_2\times S_3$（模态 1 还没选，所以 $j_1$ 在整个 $[p_1]$ 上自由；模态 2、3 已经固定，块外为零）。把这个约束代入内积：

$$
\langle\mathcal{X}_i,\mathcal{B}\rangle
=\sum_{j_1=1}^{p_1}\sum_{j_2=1}^{p_2}\sum_{j_3=1}^{p_3}\mathcal{X}_{i,j_1 j_2 j_3}\mathcal{B}_{j_1 j_2 j_3}
\;\overset{(\ast)}{=}\;
\sum_{j_1=1}^{p_1}\sum_{j_2\in S_2}\sum_{j_3\in S_3}\mathcal{X}_{i,j_1 j_2 j_3}\mathcal{B}_{j_1 j_2 j_3}.
\tag{3.1}
$$

等号 $(\ast)$ 成立，是因为当 $j_2\notin S_2$ 或 $j_3\notin S_3$ 时 $\mathcal{B}_{j_1 j_2 j_3}=0$，那些项全部消失。这一步就是"固定两个模态"在数学上的确切含义：三重求和塌缩成"$j_1$ 全跑、$(j_2,j_3)$ 只在 $S_2\times S_3$ 上跑"。

### 3.2 为什么按模态-1 的指标 $j_1$ 分组

我们现在要选的是**模态 1** 的活跃集 $S_1$，即在 $[p_1]$ 里挑若干个 $j_1$。所以自然的做法是：**把 (3.1) 按 $j_1$ 归拢**，让每个 $j_1=k$ 对应的所有项打成"一包"。选 $S_1$ 就是选"留下哪几包"。

对固定的 $k$，第 $k$ 包里出现的 $\mathcal{B}$ 分量是 $\{\mathcal{B}_{k,j_2,j_3}:(j_2,j_3)\in S_2\times S_3\}$，共 $s_2 s_3$ 个；配对的协变量是 $\{\mathcal{X}_{i,k,j_2,j_3}:(j_2,j_3)\in S_2\times S_3\}$，也是 $s_2 s_3$ 个。

### 3.3 把每一包向量化成一个"组"

**固定一个 $(j_2,j_3)\in S_2\times S_3$ 的排列顺序 $\pi$**（例如按 $(j_3,j_2)$ 字典序，即 $j_2$ 先动）。用同一个 $\pi$ 把第 $k$ 包的系数和协变量各拉成一个 $s_2 s_3$ 维向量：

$$
\boldsymbol\beta^{(1)}_k:=\bigl(\mathcal{B}_{k,j_2,j_3}\bigr)_{(j_2,j_3)\in S_2\times S_3}^{\pi}\in\mathbb{R}^{s_2 s_3},
\qquad
z^{(1)}_{i,k}:=\bigl(\mathcal{X}_{i,k,j_2,j_3}\bigr)_{(j_2,j_3)\in S_2\times S_3}^{\pi}\in\mathbb{R}^{s_2 s_3}.
\tag{3.2}
$$

**为什么必须用同一个顺序 $\pi$？** 因为只有当系数向量和协变量向量的第 $\ell$ 个分量指向同一个 $(j_2,j_3)$ 时，内积才等于逐元素乘积之和：

$$
\langle z^{(1)}_{i,k},\boldsymbol\beta^{(1)}_k\rangle
=\sum_{(j_2,j_3)\in S_2\times S_3}\mathcal{X}_{i,k,j_2,j_3}\,\mathcal{B}_{k,j_2,j_3}.
\tag{3.3}
$$

这正是 §0(b) 强调的"顺序一致"原则。顺序本身选哪个都行，但**两者必须相同、且全程固定**。

把 (3.3) 代回 (3.1)（三重求和现在按 $k$ 分块）：

$$
\langle\mathcal{X}_i,\mathcal{B}\rangle
=\sum_{k=1}^{p_1}\Bigl(\sum_{(j_2,j_3)\in S_2\times S_3}\mathcal{X}_{i,k,j_2,j_3}\mathcal{B}_{k,j_2,j_3}\Bigr)
=\sum_{k=1}^{p_1}\bigl\langle z^{(1)}_{i,k},\boldsymbol\beta^{(1)}_k\bigr\rangle.
\tag{3.4}
$$

于是"张量内积"变成了"$p_1$ 个组内积之和"。这一步是化归的核心。

### 3.4 堆叠成标准（组稀疏）线性模型

把 $p_1$ 个组前后拼接：

$$
\boldsymbol\beta^{(1)}:=\begin{pmatrix}\boldsymbol\beta^{(1)}_1\\ \vdots\\ \boldsymbol\beta^{(1)}_{p_1}\end{pmatrix}\in\mathbb{R}^{p_1 s_2 s_3},
\qquad
z^{(1)}_i:=\begin{pmatrix}z^{(1)}_{i,1}\\ \vdots\\ z^{(1)}_{i,p_1}\end{pmatrix}\in\mathbb{R}^{p_1 s_2 s_3}.
$$

由 (3.4)，$\langle\mathcal{X}_i,\mathcal{B}\rangle=(z^{(1)}_i)^\top\boldsymbol\beta^{(1)}$。把 $n$ 个观测的 $(z^{(1)}_i)^\top$ 叠成设计矩阵 $Z^{(1)}\in\mathbb{R}^{n\times p_1 s_2 s_3}$，响应向量 $y$ **原封不动**（因为我们只是重写了线性预测子，没动 $y$）。得到

$$
\boxed{\;y \;=\; Z^{(1)}\boldsymbol\beta^{(1)}+\varepsilon\;}
\qquad(\text{等价后的标准线性模型}).
\tag{3.5}
$$

**组划分.** 列自然分成 $p_1$ 组，第 $k$ 组对应第 $k$ 包：

$$
\mathcal{G}_1=\{G_1,\dots,G_{p_1}\},\quad
G_k=\{(k-1)s_2 s_3+1,\dots,k s_2 s_3\},\quad
\text{块大小}\;d_1=s_2 s_3.
\tag{3.6}
$$

**选 $S_1$ = 选哪些组非零.** 于是模态-1 选择问题写成组最优子集：

$$
\min_{\boldsymbol\beta^{(1)}}\frac{1}{2n}\bigl\lVert y-Z^{(1)}\boldsymbol\beta^{(1)}\bigr\rVert_2^2
\quad\text{s.t.}\quad \#\{k:\boldsymbol\beta^{(1)}_k\ne0\}\le s_1,
\qquad \hat S_1=\{k:\hat{\boldsymbol\beta}^{(1)}_k\ne0\}.
\tag{$\text{P}_1$}
$$

这就是一个"$p_1$ 个候选组、每组 $s_2 s_3$ 维、最多选 $s_1$ 组"的组最优子集选择问题——**恰好是组 Splicing 的输入格式**。

### 3.5 紧凑写法与 Kronecker 顺序的来历

逐指标构造清楚但啰嗦，实现时用展开矩阵更直接。设 $\mathcal{X}_i$ 的模态-1 展开为 $X_{i(1)}\in\mathbb{R}^{p_1\times p_2 p_3}$，则

$$
Z^{(1)}_i:=X_{i(1)}\,(U_3\otimes U_2)\in\mathbb{R}^{p_1\times s_2 s_3},
\tag{3.7}
$$

其第 $k$ 行就是 $(z^{(1)}_{i,k})^\top$，按行展平即得 $Z^{(1)}$ 的第 $i$ 行。

**为什么是 $U_3\otimes U_2$ ？** 这完全由展开的列编号约定决定。采用 Kolda–Bader 约定，模态-$n$ 展开时，张量元素 $(i_1,i_2,i_3)$ 落在列

$$
J=1+\sum_{m\ne n}(i_m-1)J_m,\qquad J_m=\prod_{\ell<m,\ \ell\ne n}p_\ell.
$$

对模态 1（$n=1$）：剩下模态 2、3，$J_2=1$（空积），$J_3=p_2$，故

$$
J=1+(i_2-1)\cdot 1+(i_3-1)\cdot p_2.
$$

也就是列指标里 **$i_2$ 变化最快、$i_3$ 最慢**。要抽出 $(i_2,i_3)\in S_2\times S_3$ 的列，就相当于对"$i_2$ 快、$i_3$ 慢"的向量化网格做子集选择，其选择矩阵正是快指标在右、慢指标在左的 Kronecker 积 $U_3\otimes U_2$。若你的代码用别的展开约定（如 $i_3$ 最快），就要相应改成 $U_2\otimes U_3$，否则 $Z^{(1)}$ 的列会错位。**在编程时务必让展开约定与选择矩阵的 Kronecker 顺序、以及 (3.2) 里的排列 $\pi$ 三者一致。**

### 3.6 命题（子问题等价性）与证明

**命题 1（可分解性）.** 固定 $S_2,S_3$。在 (3.2) 给出的双射 $\mathcal{B}\leftrightarrow\boldsymbol\beta^{(1)}$（限于 $\operatorname{supp}(\mathcal{B})\subseteq[p_1]\times S_2\times S_3$ 的张量）下，有

$$
L_n(\mathcal{B})=\frac{1}{2n}\lVert y-Z^{(1)}\boldsymbol\beta^{(1)}\rVert_2^2,
\qquad
\{k:\text{切片 }k\text{ 在 }S_2\times S_3\text{ 上有非零元}\}=\{k:\boldsymbol\beta^{(1)}_k\ne0\}.
$$

因此，(P) 中固定 $S_2,S_3$ 后对 $S_1$ 的选择精确等价于组最优子集问题 $(\text{P}_1)$。

*证明.* 损失等式：由 (3.4)，对每个 $i$ 有 $\langle\mathcal{X}_i,\mathcal{B}\rangle=(z^{(1)}_i)^\top\boldsymbol\beta^{(1)}$，逐项代入 (1.3) 即得。支撑对应：由 (3.2)，$\boldsymbol\beta^{(1)}_k$ 的分量恰是切片 $k$ 在 $S_2\times S_3$ 上的那 $s_2 s_3$ 个系数，故 $\boldsymbol\beta^{(1)}_k=0\Leftrightarrow$ 切片 $k$ 在该区域全零。两式合起来说明 $(\text{P}_1)$ 的目标函数、约束、解的支撑都与 (P) 在固定 $S_2,S_3$ 下的截面逐一对应。

命题 1 就是把"三维联合搜索"降成"一维组搜索"的合法性依据。有了它，外层才能安心地一个模态一个模态地轮流优化。

### 3.7 三个模态的对称构造

| 更新模态 | 固定 | 组数 $J$ | 块大小 $d$ | 组协变量 $Z^{(m)}_i$ | 响应 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $m=1$ | $S_2,S_3$ | $p_1$ | $s_2 s_3$ | $X_{i(1)}(U_3\otimes U_2)$ | $y$ |
| $m=2$ | $S_1,S_3$ | $p_2$ | $s_1 s_3$ | $X_{i(2)}(U_3\otimes U_1)$ | $y$ |
| $m=3$ | $S_1,S_2$ | $p_3$ | $s_1 s_2$ | $X_{i(3)}(U_2\otimes U_1)$ | $y$ |

（每行的 Kronecker 顺序都由该模态展开时"哪个剩余指标变化最快"决定，推法同 §3.5：模态-2 展开时 $i_1$ 最快，故 $U_3\otimes U_1$；模态-3 展开时 $i_1$ 最快，故 $U_2\otimes U_1$。）

---

## 4. 组 Splicing 子问题

命题 1 把每个模态更新变成了在 $(Z,y,\mathcal{G})$ 上的组最优子集问题（$Z=Z^{(m)}$，$\mathcal{G}=\mathcal{G}_m$，组数 $J=p_m$，块大小 $d$）。本节在一般组稀疏线性模型上把组 Splicing 讲透，**两个牺牲量的闭式逐步推导**。

### 4.1 受限最小二乘与残差正交性（后面反复要用）

设活跃组集 $\mathcal{A}\subseteq[J]$（$\lvert\mathcal{A}\rvert=s$），非活跃组集 $\mathcal{I}=[J]\setminus\mathcal{A}$。在"非活跃块强制为零"的约束下解最小二乘：

$$
\hat{\boldsymbol\beta}=\arg\min_{\boldsymbol\beta:\,\boldsymbol\beta_{G_g}=0\,\forall g\in\mathcal{I}}\frac{1}{2n}\lVert y-Z\boldsymbol\beta\rVert^2.
$$

只对活跃列 $Z_\mathcal{A}$ 求解，令梯度为零：

$$
\nabla_{\boldsymbol\beta_\mathcal{A}}=-\frac1n Z_\mathcal{A}^\top(y-Z_\mathcal{A}\hat{\boldsymbol\beta}_\mathcal{A})=0
\;\Longrightarrow\;
Z_\mathcal{A}^\top Z_\mathcal{A}\hat{\boldsymbol\beta}_\mathcal{A}=Z_\mathcal{A}^\top y
\;\Longrightarrow\;
\hat{\boldsymbol\beta}_\mathcal{A}=(Z_\mathcal{A}^\top Z_\mathcal{A})^{-1}Z_\mathcal{A}^\top y.
\tag{4.1}
$$

记残差 $\hat r=y-Z_\mathcal{A}\hat{\boldsymbol\beta}_\mathcal{A}$。上面梯度为零的条件即

$$
\boxed{\;Z_\mathcal{A}^\top\hat r=0\;}
\quad\Longrightarrow\quad
Z_{G_g}^\top\hat r=0\ \ \forall g\in\mathcal{A}.
\tag{4.2}
$$

**（4.2）是"残差与所有活跃列正交"**，是最小二乘的几何本质。它会让向后牺牲量的推导中一个交叉项精确为零——这是能得到干净闭式的关键。

**组统计量.** 对每组定义

$$
\Gamma_g:=\tfrac1n Z_{G_g}^\top Z_{G_g}\in\mathbb{R}^{d\times d}
\quad(\text{组内 Gram 矩阵，刻画块内相关}),
\qquad
\hat d_g:=\tfrac1n Z_{G_g}^\top\hat r\in\mathbb{R}^{d}
\quad(\text{该组与残差的相关}).
\tag{4.3}
$$

### 4.2 向后牺牲量：剔除一个活跃组的代价

**定义（与 Zhu et al. 一致）：把活跃组 $g$ 的系数直接置零、其余块保持不动**，看损失涨多少。注意——是"置零不重拟合"，不是"在 $\mathcal{A}\setminus\{g\}$ 上重新最小二乘"；正因如此才有闭式。

设 $\tilde{\boldsymbol\beta}$ 是把 $\hat{\boldsymbol\beta}$ 的第 $g$ 块清零。新残差

$$
y-Z\tilde{\boldsymbol\beta}=y-Z\hat{\boldsymbol\beta}+Z_{G_g}\hat{\boldsymbol\beta}_{G_g}=\hat r+Z_{G_g}\hat{\boldsymbol\beta}_{G_g}.
$$

于是

$$
\xi_g:=L_n(\tilde{\boldsymbol\beta})-L_n(\hat{\boldsymbol\beta})
=\frac{1}{2n}\Bigl(\lVert\hat r+Z_{G_g}\hat{\boldsymbol\beta}_{G_g}\rVert^2-\lVert\hat r\rVert^2\Bigr).
$$

展开平方：

$$
\lVert\hat r+Z_{G_g}\hat{\boldsymbol\beta}_{G_g}\rVert^2
=\lVert\hat r\rVert^2
+\underbrace{2\,\hat{\boldsymbol\beta}_{G_g}^\top Z_{G_g}^\top\hat r}_{=0\ \text{by (4.2)}}
+\hat{\boldsymbol\beta}_{G_g}^\top Z_{G_g}^\top Z_{G_g}\hat{\boldsymbol\beta}_{G_g}.
$$

**交叉项被 (4.2) 精确消掉**（$g$ 是活跃组，残差与它正交）。所以

$$
\boxed{\;
\xi_g=\frac{1}{2n}\hat{\boldsymbol\beta}_{G_g}^\top Z_{G_g}^\top Z_{G_g}\hat{\boldsymbol\beta}_{G_g}
=\tfrac12\,\hat{\boldsymbol\beta}_{G_g}^\top\Gamma_g\hat{\boldsymbol\beta}_{G_g}\;}
\qquad(g\in\mathcal{A}).
\tag{4.4}
$$

**含义：** $\xi_g$ 度量该组信号被拟合出的"能量" $\tfrac12\lVert Z_{G_g}\hat{\boldsymbol\beta}_{G_g}\rVert^2/n$。$\xi_g$ 越小，说明剔掉它损失涨得越少，越不重要，越该被淘汰。

### 4.3 向前牺牲量：加入一个非活跃组的收益

**定义：在当前拟合之上，只给非活跃组 $g$ 添一个最优块系数 $t$、其余不动**，看损失降多少。

先求最优 $t$：

$$
\hat t_g=\arg\min_{t\in\mathbb{R}^d}\frac{1}{2n}\lVert\hat r-Z_{G_g}t\rVert^2.
$$

令梯度为零：$Z_{G_g}^\top Z_{G_g}\hat t_g=Z_{G_g}^\top\hat r$，即

$$
\hat t_g=(Z_{G_g}^\top Z_{G_g})^{-1}Z_{G_g}^\top\hat r
=(n\Gamma_g)^{-1}(n\hat d_g)=\Gamma_g^{-1}\hat d_g.
\tag{4.5}
$$

**为什么会出现 $\Gamma_g^{-1}$？** 因为块内 $d$ 个协变量彼此可能相关，直接用相关向量 $\hat d_g$ 会重复计数；乘上 $\Gamma_g^{-1}$ 相当于对块内做"去相关/广义最小二乘"，得到该组的最优联合贡献。$d=1$ 时 $\Gamma_g$ 是标量，退化为普通的"相关除以自相关"。

再算损失下降。以 $\hat t_g$ 代入并用 $Z_{G_g}^\top\hat r=n\hat d_g,\ Z_{G_g}^\top Z_{G_g}=n\Gamma_g$：

$$
\begin{aligned}
\zeta_g&:=L_n(\hat{\boldsymbol\beta})-L_n(\hat{\boldsymbol\beta}+\hat t_g^{\{g\}})
=\frac{1}{2n}\Bigl(\lVert\hat r\rVert^2-\lVert\hat r-Z_{G_g}\hat t_g\rVert^2\Bigr)\\[2pt]
&=\frac{1}{2n}\Bigl(2\hat t_g^\top Z_{G_g}^\top\hat r-\hat t_g^\top Z_{G_g}^\top Z_{G_g}\hat t_g\Bigr)
=\frac{1}{2n}\Bigl(2n\,\hat t_g^\top\hat d_g-n\,\hat t_g^\top\Gamma_g\hat t_g\Bigr)\\[2pt]
&=\tfrac12\bigl(2\hat t_g^\top\hat d_g-\hat t_g^\top\Gamma_g\hat t_g\bigr).
\end{aligned}
$$

代入 $\hat t_g=\Gamma_g^{-1}\hat d_g$，注意 $\hat t_g^\top\Gamma_g\hat t_g=\hat d_g^\top\Gamma_g^{-1}\Gamma_g\Gamma_g^{-1}\hat d_g=\hat d_g^\top\Gamma_g^{-1}\hat d_g$：

$$
\boxed{\;\zeta_g=\tfrac12\bigl(2\hat d_g^\top\Gamma_g^{-1}\hat d_g-\hat d_g^\top\Gamma_g^{-1}\hat d_g\bigr)
=\tfrac12\,\hat d_g^\top\Gamma_g^{-1}\hat d_g\;}
\qquad(g\in\mathcal{I}).
\tag{4.6}
$$

**含义：** $\zeta_g$ 度量该组能把残差解释掉多少。$\zeta_g$ 越大，说明纳入它收益越大，越该被提拔。

### 4.4 两个牺牲量为什么"不能直接比、但能分组排序"

$\xi_g$ 建立在"已拟合系数 $\hat{\boldsymbol\beta}_{G_g}$"上，$\zeta_g$ 建立在"与残差的相关 $\hat d_g$"上，二者度量的东西不同、量纲来源不同（活跃组已把自己的信号吸收进拟合，非活跃组的信号还留在残差里），所以**不能把某个 $\xi$ 和某个 $\zeta$ 直接比大小**。但在各自集合内部排序是有意义的：活跃集里 $\xi$ 最小的最该走，非活跃集里 $\zeta$ 最大的最该进。Splicing 正是据此"末位淘汰 + 择优录取"。

### 4.5 拼接算子与阈值

给定拼接规模 $k\le\min(\lvert\mathcal{A}\rvert,\lvert\mathcal{I}\rvert)$：

$$
\mathcal{A}_k=\Bigl\{g\in\mathcal{A}:\textstyle\sum_{h\in\mathcal{A}}\mathbb{1}(\xi_g\ge\xi_h)\le k\Bigr\}
\ (\text{活跃集里 }\xi\text{ 最小的 }k\text{ 组}),
$$
$$
\mathcal{I}_k=\Bigl\{g\in\mathcal{I}:\textstyle\sum_{h\in\mathcal{I}}\mathbb{1}(\zeta_g\le\zeta_h)\le k\Bigr\}
\ (\text{非活跃集里 }\zeta\text{ 最大的 }k\text{ 组}),
$$
$$
\tilde{\mathcal{A}}=(\mathcal{A}\setminus\mathcal{A}_k)\cup\mathcal{I}_k.
\tag{4.7}
$$

（$\sum_h\mathbb{1}(\xi_g\ge\xi_h)\le k$ 是"$\xi_g$ 的排名不超过第 $k$ 小"的无歧义写法。）

**为什么要"有进有出"地成批交换，而不是纯贪心地一个个加？** 纯前向贪心一旦选错早期变量就无法回头，容易卡在局部最优。Splicing 每步同时踢掉最没用的、拉进最有潜力的，允许"纠错"，因此能在多项式时间里跳出贪心陷阱、逼近全局最优（Zhu et al. 的核心贡献）。

**接受准则与阈值.** 记 $\tilde{\boldsymbol\beta}$ 为在 $\tilde{\mathcal{A}}$ 上重拟合的解。若

$$
L_n(\hat{\boldsymbol\beta})-L_n(\tilde{\boldsymbol\beta})>\tau,
$$

则接受 $\tilde{\mathcal{A}}$。**为什么要阈值 $\tau>0$ 而不是 $>0$ 就接受？** 当算法已恢复真集后，再拼接可能带来微不足道、纯属噪声的损失下降，导致来回抖动、浪费计算。$\tau$ 是一道"显著改进"的闸门，滤掉这种无谓更新。经验取值量级 $\tau\asymp c\,s\,d\,\log J\,\log\log n/n$（随样本增大而缩小，随问题规模增大而放宽）。

### 4.6 伪代码

```text
算法 4.1  GroupSplicing(β, A, k_max, τ)   —— 给定组稀疏度的一次拼接
输入: 当前系数 β, 活跃组集 A (|A|=s), I = [J]\A, 最大拼接规模 k_max, 阈值 τ
1  r ← y − Zβ                                              # 残差
2  for g = 1..J:  Γ_g ← Zᵀ_{G_g} Z_{G_g}/n ;  d_g ← Zᵀ_{G_g} r / n
3  ξ_g ← ½ βᵀ_{G_g} Γ_g β_{G_g}      (g ∈ A)               # (4.4) 向后牺牲量
4  ζ_g ← ½ d_gᵀ Γ_g⁻¹ d_g            (g ∈ I)               # (4.6) 向前牺牲量
5  L0 ← L ← L_n(β) ;  (β̂, Â) ← (β, A)
6  for k = 1 .. k_max:                                     # 试各种交换规模
7      A_k ← A 中 ξ 最小的 k 组 ;   I_k ← I 中 ζ 最大的 k 组
8      Ã ← (A \ A_k) ∪ I_k
9      β̃ ← 在组集 Ã 上的最小二乘解 (4.1)（其余块置零）
10     if L_n(β̃) < L:  (β̂, Â) ← (β̃, Ã) ;  L ← L_n(β̃)     # 记下最好的一次交换
11 if L0 − L < τ:  return (β, A)          # 改进不显著 → 保持原集（收敛信号）
12 else:           return (β̂, Â)

算法 4.2  GroupBESS.Fix(Z, y, 𝒢, s, k_max, τ)   —— 固定组稀疏度 s 的组最优子集
1  A⁰ ← 边际筛选：组相关能量 ‖Γ_g⁻¹ᐟ²(Zᵀ_{G_g} y)/n‖ 最大的 s 个组   # 好初值
2  β⁰ ← 在 A⁰ 上的最小二乘解
3  for t = 0,1,2, …:
4      (β^{t+1}, A^{t+1}) ← GroupSplicing(β^t, A^t, k_max, τ)
5      if A^{t+1} = A^t:  break                            # 活跃集不再变 → 收敛
6  return (β̂, Â)
```

*初值为什么用组相关能量最大的 $s$ 组？* 与残差（初始时即 $y$）相关最强的组最可能是真信号，用它做起点能让 Splicing 更快收敛（对应向量 ABESS 里"取与 $y$ 最相关的前 $s$ 个变量"）。

---

## 5. BCD 外层循环 + SIC 选稀疏度

整体两层：内层（§4）在固定稀疏度下解单模态组最优子集；外层用 BCD 轮转三个模态，并由 SIC 定各模态稀疏度。

### 5.1 为什么外层用 BCD（块坐标下降）

问题 (P) 对 $(S_1,S_2,S_3)$ 是**联合、非凸**的（三个模态的支撑相互耦合）。BCD 的思路是：每次只动一个模态、固定其余两个——由命题 1，这一步是有精确解法（组 Splicing）的凸/组子问题。轮流做，直到三个支撑都不再变。这就是把"三维联合难题"拆成"三个一维易题"轮流解。

### 5.2 序贯 SIC：为什么可以逐模态定阶

更新模态 $m$ 时，另两个模态的支撑与尺寸 $s_{m'}\,(m'\ne m)$ 固定。§6 的整体 SIC 里，惩罚项 $\log\log n\sum_{m'}\log\binom{p_{m'}}{s_{m'}}$ 中**除模态 $m$ 外的两项此刻是常数**。因此

$$
\arg\min_{s_m}\mathrm{SIC}
=\arg\min_{s_m}\Bigl[n\log(\mathrm{RSS}/n)
+s_md_m\log n
+\log\log n\cdot\log\tbinom{p_m}{s_m}\Bigr].
\tag{5.1}
$$

其中$d_m=∏_{m' \neq m}s_{m'}$ 是当前模态 $m$ 的组块大小（由已固定的另两个模态尺寸决定）。

也就是说，**在该模态方向对整体 SIC 取截面 = 用一个只含该模态惩罚的"条件 SIC"**。这就是"序贯信息准则"名字的由来：不必一次在三维网格上搜 $(s_1,s_2,s_3)$，而是在 BCD 每步里对当前模态一维地扫 $s_m$、用 (5.1) 定阶。（若担心序贯的次优，可退回到在三元组网格上跑，用整体 SIC 取最优——作为对照方案。）

```text
算法 5.0  Tensor-Init-GLasso —— 基于组 Lasso 的初始稀疏度估计
输入: {(𝒳_i, y_i)}, (p_1, p_2, p_3), 安全系数 c (默认 c=2)

—— 步骤 A: 边际筛选预缩维 ——
1  for m = 1,2,3:
2    对每个切片 k ∈ [p_m], 计算边际得分
       score_k^{(m)} = ‖X_{(m)}^{(k)⊤} y‖² / n    # 切片 k 对响应的总相关能量
3    保留得分最大的 min(⌊n^{2/3}⌋, p_m) 个切片, 记为 𝒞_m (候选集)

—— 步骤 B: 逐模态组 Lasso ——
4  S̃_2 ← 𝒞_2 中得分最大的 ⌊√n⌋ 个 ; S̃_3 ← 𝒞_3 中得分最大的 ⌊√n⌋ 个
     # 粗糙初始支撑, 仅用于构造设计矩阵
5  for m = 1,2,3 (按固定顺序):
6    构造 Z^{(m)} ∈ ℝ^{n × |𝒞_m|·|S̃_{-m}|}, 
       其中 S̃_{-m} 为当前另两个模态的支撑
       组划分: |𝒞_m| 组, 块大小 |S̃_{-m}|
7    解组 Lasso (λ 路径):
       β̂^{(m)}(λ) = argmin_{β} ½n⁻¹‖y - Z^{(m)}β‖² + λ Σ_{k∈𝒞_m} ‖β_k‖_2
8    用 5-折 CV 或 BIC 选最优 λ*, 
       记 S̃_m ← {k ∈ 𝒞_m : β̂_k^{(m)}(λ*) ≠ 0}, s̃_m ← |S̃_m|

—— 步骤 C: 确定搜索上界与初始值 ——
9  s_{m,max} ← min(c · s̃_m, ⌊(n / (3·max_m' s̃_{m'}))^{1/2}⌋)
     # 保证 s_{1,max}·s_{2,max}·s_{3,max} < n (OLS可解)
10 S_m^{(0)} ← S̃_m  (作为 BCD 的暖启动)

输出: (s_{1,max}, s_{2,max}, s_{3,max}), (S_1^{(0)}, S_2^{(0)}, S_3^{(0)})

算法 5.1 GroupBESS.Adaptive（修正版）
输入: Z, y, 𝒢, s_max, k_max, τ, d_m (块大小), A_warm (可选暖启动)

1  for s = 1 .. s_max:
2    if s = |A_warm| and A_warm is provided:
3      A⁰ ← A_warm               # 暖启动
4    else:
5      A⁰ ← 边际筛选：组相关能量最大的 s 组
6    β⁰ ← 在 A⁰ 上的 OLS
7    (β̂_s, Â_s) ← GroupBESS.Fix(Z, y, 𝒢, s, k_max, τ, β⁰, A⁰)
8    RSS_s ← ‖y − Z β̂_s‖²
9  s* ← argmin_s [ n·log(RSS_s/n) + s·d_m·log(n) 
                    + log log n · log C(J, s) ]       # 修正SIC
10 return (β̂_{s*}, Â_{s*}, s*)

算法 5.2' Tensor-ABESS（修正版）—— 矩形支撑张量回归的组 Splicing 变量选择

输入: {(𝒳_i, y_i)}, k_max, τ, 收敛容差 ε, 安全系数 c=2

━━━━━ 阶段 0: 组 Lasso 初始化 ━━━━━
0  (s_{1,max}, s_{2,max}, s_{3,max}), (S_1, S_2, S_3) 
     ← Tensor-Init-GLasso({(𝒳_i, y_i)}, c)

━━━━━ 阶段 1: BCD + 组 Splicing 精确选择 ━━━━━
1  repeat
2    —— 模态1: 固定 S_2, S_3 ——
3      d_1 ← |S_2|·|S_3|
4      Z^{(1)}_i ← X_{i(1)}(U_3⊗U_2), 组划分 𝒢_1 (p_1 组, 块 d_1)
5      (β^{(1)}, S_1, s_1) ← GroupBESS.Adaptive(Z^{(1)}, y, 𝒢_1,
                              s_{1,max}, k_max, τ, d_1)  
         # 注意: Adaptive 内部 SIC 用修正版 (含 s·d_1·log n 项)

6    —— 模态2: 固定 S_1, S_3 ——
7      d_2 ← |S_1|·|S_3|
8      Z^{(2)}_i ← X_{i(2)}(U_3⊗U_1), 组划分 𝒢_2 (p_2 组, 块 d_2)
9      (β^{(2)}, S_2, s_2) ← GroupBESS.Adaptive(Z^{(2)}, y, 𝒢_2,
                              s_{2,max}, k_max, τ, d_2)

10   —— 模态3: 固定 S_1, S_2 ——
11     d_3 ← |S_1|·|S_2|
12     Z^{(3)}_i ← X_{i(3)}(U_2⊗U_1), 组划分 𝒢_3 (p_3 组, 块 d_3)
13     (β^{(3)}, S_3, s_3) ← GroupBESS.Adaptive(Z^{(3)}, y, 𝒢_3,
                              s_{3,max}, k_max, τ, d_3)

14 until (S_1, S_2, S_3) 不变, 或 ΔSIC < ε

━━━━━ 阶段 2: 最终精化 ━━━━━
15 在矩形支撑 S_1×S_2×S_3 上做最终 OLS, 得 B̂
16 计算 SIC(S_1, S_2, S_3) = n·log(RSS/n) + s_1s_2s_3·log n 
                              + log log n · Σ_m log C(p_m, s_m)
17 (可选) 在 S_1×S_2×S_3 内对 B̂ 做 CP/Tucker 低秩精化

输出: (S_1, S_2, S_3), B̂, SIC
```

第 13 行把张量从 $p_1\times p_2\times p_3$ 缩到小张量 $s_1\times s_2\times s_3$——这正是申报书"先筛选、后精化"里的筛选产物；后续低秩分解只在这个小块上做，计算量从指数级降到线性级。

---

## 6. SIC 的具体形式（完整由来）

图片要求 SIC "在 $n\log(\mathrm{RSS}/n)$ 基础上，惩罚项利用模型空间规模 $\binom{p_1}{s_1}\binom{p_2}{s_2}\binom{p_3}{s_3}$ 取对数后的可加结构"。本节把这句话从头推出来。

### 6.1 拟合项 $n\log(\mathrm{RSS}/n)$ 从高斯似然而来

设误差为正态 $\varepsilon_i\sim N(0,\sigma^2)$（用于导出准则；算法本身只需亚高斯）。给定一个支撑模型 $M$ 及其最小二乘拟合，对数似然

$$
\ell(\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\mathrm{RSS}.
$$

对 $\sigma^2$ 求极大得 $\hat\sigma^2=\mathrm{RSS}/n$，代回：

$$
\ell(\hat\sigma^2)=-\frac n2\log\!\Bigl(2\pi\frac{\mathrm{RSS}}{n}\Bigr)-\frac n2
\;\Longrightarrow\;
-2\ell(\hat\sigma^2)=n\log\!\Bigl(\frac{\mathrm{RSS}}{n}\Bigr)+\text{const}.
\tag{6.1}
$$

信息准则都建立在 $-2\ell$ 上，扔掉与模型无关的常数，就得到拟合项 $n\log(\mathrm{RSS}/n)$。**它越小拟合越好**，但只用它会选出最大模型（过拟合），所以要加惩罚。

### 6.2 从 BIC 到 EBIC：为什么惩罚"模型空间规模"

- **BIC** 惩罚 $=k\log n$（$k$ 为自由参数数），源于对参数的拉普拉斯近似。它假定"同一尺寸的模型只有一个"，在高维下失效：尺寸为 $k$ 的候选模型有 $\binom{p}{k}$ 个之多，纯 BIC 会因"碰运气拟合"而选进假变量。
- **EBIC（Chen & Chen, 2008）** 修正之：给模型加一个"模型空间"先验，模型越多、单个模型的先验越小，等价于在惩罚里加一项正比于 $\log(\text{同尺寸模型个数})$：

$$
\mathrm{EBIC}=-2\ell+k\log n+2\gamma\log\binom{p}{k}.
$$

**关键洞察：控制高维下的假发现，靠的是惩罚"候选模型的数量"（多重比较的量级），而不是参数个数 $k$。** 这就是为什么下面张量版惩罚用的是模型个数的对数，而不是自由参数 $s_1 s_2 s_3$。

- **ABESS 的 SIC（Zhu et al.）** 是超高维版本：去掉 $k\log n$，用慢发散因子 $\log\log n$ 配 $\log p$ 型复杂度，取 $\mathrm{SIC}=n\log\hat{\mathcal L}_A+|A|\log p\log\log n$。因 $\log\binom{p}{|A|}\approx|A|\log p$，它本质就是"$\log(\text{模型空间规模})\times\log\log n$"。

### 6.3 张量版：模型空间是三个组合数之积，取对数即可加

矩形支撑、尺寸 $(s_1,s_2,s_3)$ 的模型空间

$$
\mathcal{M}_{(s_1,s_2,s_3)}=\{S_1\times S_2\times S_3:\lvert S_m\rvert=s_m\},
\qquad
\bigl\lvert\mathcal{M}_{(s_1,s_2,s_3)}\bigr\rvert=\binom{p_1}{s_1}\binom{p_2}{s_2}\binom{p_3}{s_3}.
$$

**为什么是三个组合数之积？** 选一个矩形支撑 = 独立地在模态 1 选 $s_1$ 个（$\binom{p_1}{s_1}$ 种）、在模态 2 选 $s_2$ 个、在模态 3 选 $s_3$ 个，三者相乘。乘积取对数即化为求和（**可加结构**）：

$$
\log\Bigl[\binom{p_1}{s_1}\binom{p_2}{s_2}\binom{p_3}{s_3}\Bigr]
=\sum_{m=1}^{3}\log\binom{p_m}{s_m}.
\tag{6.2}
$$

把 (6.1) 的拟合项与 (6.2) 的模型空间惩罚（配 $\log\log n$）相加，得

$$
\boxed{\;
\mathrm{SIC}(S_1,S_2,S_3)=n\log\!\Bigl(\frac{\mathrm{RSS}}{n}\Bigr)
+s_1s_2s_3\log n
+\log\log n\sum_{m=1}^{3}\log\binom{p_m}{s_m}
\;}
\tag{6.3}
$$

其中 $\mathrm{RSS}=\lVert y-\mathbb{X}\mathrm{vec}(\hat{\mathcal{B}})\rVert^2$，$\hat{\mathcal{B}}=\arg\min_{\operatorname{supp}\subseteq S_1\times S_2\times S_3}\lVert y-\mathbb{X}\mathrm{vec}(\mathcal{B})\rVert^2$。

**可加结构的两个红利：**
1. *统计上：* 惩罚天然分成三个模态各自的贡献，与"三个模态各自稀疏"的假设结构一致。
2. *算法上：* 更新模态 $m$ 时另两项是常数，于是整体 SIC 的截面就是单模态条件 SIC——这正是 §5.2 序贯定阶合法的原因。

### 6.4 轻量近似与向量退化

**当** $s_m\ll p_m$，由 Stirling $\log\binom{p_m}{s_m}\approx s_m\log(p_m/s_m)+s_m\approx s_m\log p_m$，得便于计算的等价形式

$$
\mathrm{SIC}\approx n\log(\mathrm{RSS}/n)
+s_1s_2s_3\log n
+\log\log n\,(s_1\log p_1+s_2\log p_2+s_3\log p_3).
\tag{6.4}
$$

取 $p_2=p_3=1,s_2=s_3=1$ 就回到向量 ABESS 的 $\mathrm{SIC}=n\log\hat{\mathcal L}_A+|A|\log n+|A|\log p\log\log n$。可见 (6.3) 是把向量版的 $|A|\log p$ 精细化为组合数对数之和 $\sum_m\log\binom{p_m}{s_m}$。

向量 ABESS 的 SIC 为 $n\log\hat{\mathcal{L}}_A + |A|\log p\log\log n$。两者一致的条件是 $\log p_1 \cdot \log\log n \gg \log n$，即 $p_1 \gg n^{1/\log\log n}$（超高维条件下 BIC 项被搜索项吸收）。需补充说明：**在超高维 $p_m \gg n$ 时，BIC 项被搜索空间项覆盖，两者退化一致；在中等维度时，BIC 项是必要的独立贡献。**

---

## 7. 收敛性与复杂度（为什么有限步收敛）

以下命题只给出陈述与证明思路，严格版本是项目"理论性质分析"阶段的产出。

**命题 2（内层有限步收敛）.** GroupSplicing 只在损失严格下降超过 $\tau>0$ 时才接受新集（算法 4.1 第 11–12 行），故 $L_n$ 沿 GroupBESS.Fix 的迭代严格递减且以 $\tau$ 为步长下界；又活跃组集取自有限集合 $\{\mathcal{A}\subseteq[J]:\lvert\mathcal{A}\rvert=s\}$（元素有限），严格递减序列不可能无限长，故有限步终止。（Zhu et al. 定理 1 的组推广。）

**命题 3（外层有限步收敛）.** 每个模态更新（算法 5.1）在该模态方向不增全局损失（子问题取最小二乘/最优子集解），故 $L_n$ 沿 BCD 单调不增；三元支撑 $(S_1,S_2,S_3)$ 取自有限乘积集合 $\prod_m\{S_m:\lvert S_m\rvert\le s_{m,\max}\}$。**单调不增 + 状态空间有限 $\Rightarrow$ 有限步内到达稳定支撑三元组**（此后再轮转不改变任何支撑）。这就是"交替迭代中非凸耦合的收敛性保证"的核心论证：不靠凸性，靠"单调 + 有限"。

**复杂度.** 单模态组 Splicing 是多项式复杂度（Zhu et al. 定理 2 的组版本，另加每组 $d\times d$ 的求逆/求解代价 $O(nd^2+d^3)$，其中模态 $m$ 的 $d=\prod_{m'\ne m}s_{m'}$）。BCD 每轮扫三个模态，总代价 = 单模态多项式 $\times 3\times$ 轮数，仍多项式，远优于对 $\prod_m\binom{p_m}{s_m}$ 个矩形支撑的穷举。

**选择一致性（目标定理）.** 在张量版稀疏受限条件（SRC）、最小信号强度、以及 $s_1s_2s_3\log n+\log\log n\sum_m s_m\log p_m=o(n)$ 等温和条件下，SIC 应以概率 $1-O(p^{-\alpha})$（$\alpha>0$）恢复真实矩形支撑 $(S_1^*,S_2^*,S_3^*)$，从而 $\hat{\mathcal{B}}$ 达到 oracle 最小二乘估计（Zhu et al. 定理 4、推论 1 的张量推广）。$\log\log n$ 前是否需配常数 $\gamma$（EBIC 式）由该一致性条件的常数决定，留作待定接口。

---

## 8. 与向量版 ABESS 的对照

| 维度    | 向量 ABESS（Zhu et al. 2020）                                               | 张量组 Splicing（本框架）                                                               |
| ----- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 模型    | $y=X\beta+\varepsilon,\ \beta\in\mathbb{R}^p$                           | $y_i=\langle\mathcal{X}_i,\mathcal{B}\rangle+\varepsilon_i$                     |
| 稀疏结构  | $\lVert\beta\rVert_0\le s$（逐元素）                                         | 矩形支撑 $\operatorname{supp}(\mathcal{B})\subseteq S_1\times S_2\times S_3$        |
| 选择对象  | 单个变量（组大小 1）                                                             | 模态切片（组），块大小 $\prod_{m'\ne m}s_{m'}$                                             |
| 化归    | 无需                                                                      | 固定两模态 $\Rightarrow$ 组最优子集（命题 1）                                                 |
| 向后牺牲量 | $\frac{X_j^\top X_j}{2n}\hat\beta_j^2$                                  | $\tfrac12\hat{\boldsymbol\beta}_{G_g}^\top\Gamma_g\hat{\boldsymbol\beta}_{G_g}$ |
| 向前牺牲量 | $\frac{X_j^\top X_j}{2n}\bigl(\hat d_j/\tfrac{X_j^\top X_j}{n}\bigr)^2$ | $\tfrac12\hat d_g^\top\Gamma_g^{-1}\hat d_g$                                    |
| 外层    | 对 $s$ 扫描                                                                | BCD 轮转三模态 + 各模态序贯 SIC 定阶                                                        |
| 信息准则  | $n\log\hat{\mathcal L}_A+\lvert A\rvert\log p\log\log n$                | $n\log(\mathrm{RSS}/n)+s_1s_2s_3\log n+\log\log n\sum_m\log\binom{p_m}{s_m}$    |
| 模型空间  | $\binom{p}{s}$                                                          | $\prod_m\binom{p_m}{s_m}$（对数可加）                                                 |

$d=1$、$p_2=p_3=1$ 时本框架逐条退化为向量 ABESS——它是本方法的严格特例。

---

### 参考文献

- Zhu, Wen, Zhu, Zhang, Wang (2020). *A polynomial algorithm for best-subset selection.* PNAS 117(52): 33117–33123.（Splicing、SIC、oracle 性质）
- Chen & Chen (2008). *Extended Bayesian information criteria for model selection with large model spaces.* Biometrika 95: 759–771.（EBIC 模型空间复杂度惩罚）
- Wang, Kim, Li (2013). *Calibrating non-convex penalized regression in ultra-high dimension.* Ann. Statist. 41: 2505–2536.（HBIC，$\log\log n$ 慢发散因子）

> **待补严格证明（数学线后续任务）：** 命题 1 里双射的良定义与逆映射细节；命题 2/3 的收敛率与步数上界常数；§7 目标定理中张量版 SRC 的确切形式、最小信号强度条件、以及 SIC 惩罚常数 $\gamma$ 的取值。
