# Tensor-ABESS 算法实现说明（代码终版）

## 1. 方法目标与模型假设

项目处理三阶张量线性回归。对第 \(i\) 个样本，输入为

\[
X_i\in\mathbb{R}^{p_1\times p_2\times p_3},
\qquad
y_i=\langle X_i,B\rangle+\varepsilon_i,
\]

其中

\[
\langle X_i,B\rangle=
\sum_{a=1}^{p_1}\sum_{b=1}^{p_2}\sum_{c=1}^{p_3}
X_{iabc}B_{abc}.
\]

代码假设系数张量具有**矩形支撑**：存在三个模式支撑集
\(S_1,S_2,S_3\)，使得 \(B\) 只在
\(S_1\times S_2\times S_3\) 上允许非零。记
\(s_m=|S_m|\)，则最终需要估计的元素个数为

\[
q=s_1s_2s_3.
\]

因此，本项目不是对 \(p_1p_2p_3\) 个元素分别筛选，而是分别筛选三个模式中的有效切片，再在三个支撑集的笛卡尔积上估计系数。

给定支撑大小时，核心优化问题为

\[
\min_{S_1,S_2,S_3,B}
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-\langle X_i,B\rangle\right)^2,
\quad
\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3,
\quad |S_m|=s_m.
\]

自适应版本还需要从数据中选择 \((s_1,s_2,s_3)\)。

## 2. 代码中的总体流程

实际执行链为：

\[
\text{边际预筛}
\rightarrow
\text{逐模 Group Lasso 初始化}
\rightarrow
\text{逐模 Group-BESS 选组}
\rightarrow
\text{三模式 BCD 循环}
\rightarrow
\text{全局候选精修}
\rightarrow
\text{矩形支撑内 OLS 重估}.
\]

其中，Group Lasso 只负责产生较稳定的初始支撑和搜索上界；最终变量选择由 Group-BESS 的交换步、MSIC 选阶和全局精修共同完成。

## 3. 关键步骤

### 3.1 边际预筛与 Group Lasso 初始化

对模式 \(m\in\{1,2,3\}\) 的第 \(g\) 个切片，代码计算边际得分

\[
u_g^{(m)}=\frac{1}{n}\left\|
\operatorname{vec}(X_g^{(m)})^\top y
\right\|_2^2.
\]

每个模式保留得分最高的

\[
|C_m|=\min\{\lfloor n^{2/3}\rfloor,p_m\}
\]

个候选切片，得到候选集 \(C_m\)。随后从候选集中先取最多
\(\lfloor\sqrt n\rfloor\) 个高分切片作为临时支撑，并按模式
\(1\rightarrow2\rightarrow3\) 依次运行 Group Lasso。更新某一模式时，另外两个模式的当前支撑保持不变。

Group Lasso 采用 ISTA 求解，正则化参数在
\([\lambda_{\max},10^{-2}\lambda_{\max}]\) 上取对数等距网格。最终数值模拟使用 BIC 选择 \(\lambda\)：

\[
\mathrm{BIC}
=n\log(\mathrm{RSS}/n)+|A|d\log n,
\]

其中 \(A\) 是选中的组集合，\(d\) 是每组包含的列数。若 Group Lasso 未选中任何组，代码回退到该模式边际得分最高的候选切片，保证每个模式支撑非空。

初始化结果同时给出各模式搜索上界：

\[
s_{m,\max}
=\min\left\{
p_m,
\max\left(\lceil c|\widetilde S_m|\rceil,
\left\lfloor\frac{\sqrt n}{2}\right\rfloor,
1\right)
\right\}.
\]

若三个上界的乘积超过 \(\lfloor0.7n\rfloor\)，代码反复缩小当前最大的上界，直到

\[
s_{1,\max}s_{2,\max}s_{3,\max}\leq\lfloor0.7n\rfloor.
\]

这一约束防止矩形支撑中的参数数目过于接近或超过样本量，从而降低 OLS 重估的不稳定性。

### 3.2 将张量更新转化为分组线性回归

以更新模式 1 为例，固定当前 \(S_2,S_3\)。模式 1 的每个候选切片 \(g\) 对应设计矩阵中的一个组：

\[
Z_g^{(1)}=X[:,g,S_2,S_3]
\in\mathbb{R}^{n\times d_1},
\qquad d_1=|S_2||S_3|.
\]

拼接全部模式 1 切片后得到

\[
Z^{(1)}\in\mathbb{R}^{n\times p_1d_1}.
\]

选择一个组等价于选择一个模式 1 切片。模式 2、3 同理，其组大小分别为

\[
d_2=|S_1||S_3|,
\qquad
d_3=|S_1||S_2|.
\]

通过这一变换，三阶张量支撑搜索被分解为三个分组最佳子集问题。

### 3.3 固定组数的 Group-BESS 交换步

对给定组数 \(s\)，代码先在当前活动组 \(A\) 上做 OLS，并计算两类牺牲量。

活动组 \(g\in A\) 的删除牺牲量为

\[
\xi_g=\frac{1}{2}\widehat\beta_g^\top
\Gamma_g\widehat\beta_g,
\qquad
\Gamma_g=Z_g^\top Z_g/n.
\]

\(\xi_g\) 越小，删除该组造成的损失通常越小。非活动组
\(g\notin A\) 的加入收益为

\[
\zeta_g=\frac{1}{2}d_g^\top\Gamma_g^{-1}d_g,
\qquad
d_g=Z_g^\top r/n,
\qquad
r=y-Z\widehat\beta.
\]

\(\zeta_g\) 越大，该组越值得加入；当 \(\Gamma_g\) 奇异时，代码使用广义逆。

一次交换依次尝试 \(k=1,\ldots,k_{\max}\)：删除 \(\xi_g\) 最小的
\(k\) 个活动组，同时加入 \(\zeta_g\) 最大的 \(k\) 个非活动组；每个候选支撑都重新做 OLS，并保留训练损失最低者。由于删除数与加入数相等，整个过程始终保持
\(|A|=s\)，不会破坏固定规模子问题。

仅当损失下降不少于阈值

\[
\tau_s=
0.01\frac{s\,d\,\log(\max(J,2))
\log(\max(\log n,e))}{n}
\]

时才接受交换，其中 \(J\) 为总组数。若无可接受交换，或活动组不再变化，则该固定规模子问题结束。

### 3.4 模式内自适应选阶

更新某一模式时，代码枚举
\(s=1,2,\ldots,s_{m,\max}\)。记
\(\ell_n=\log(\max(\log n,e))\)，每个 \(s\) 都完整运行一次固定规模 Group-BESS，再用模式 MSIC 评分：

\[
\mathrm{MSIC}_{m}(s)=
n\log(\mathrm{RSS}_s/n)
+\alpha sd_m\ell_n
+\gamma\ell_n\log {J_m\choose s}.
\]

第一项评价拟合误差；第二项惩罚实际回归自由度 \(sd_m\)；第三项惩罚从
\(J_m\) 个切片中选择 \(s\) 个切片的模型空间复杂度。模式内阶段代码使用
\(\alpha=1,\gamma=1\)，并选择 MSIC 最小的组数及其支撑。

### 3.5 三模式 BCD 外循环

在一个 BCD 周期中，代码按顺序更新：

1. 固定 \(S_2,S_3\)，自适应更新 \(S_1\)；
2. 使用新的 \(S_1\)，固定 \(S_3\)，自适应更新 \(S_2\)；
3. 使用新的 \(S_1,S_2\)，自适应更新 \(S_3\)；
4. 在 \(S_1\times S_2\times S_3\) 上重新做 OLS，更新 \(B\)、RSS 和张量 MSIC。

模式更新是顺序执行而非并行执行。满足以下任一条件时停止：三个支撑集均未变化；相邻周期张量准则之差小于
\(\varepsilon\)；或达到最大 BCD 周期数。

### 3.6 全局候选精修与小模型优先

逐模更新是条件优化，可能停在局部支撑规模。为此，自适应代码在 BCD 结束后构造一组全局候选规模。
对每个上界 \(s_{m,\max}\)，候选值取自

\[
\left\{
1,
\operatorname{round}(0.4s_{m,\max}),
\operatorname{round}(0.6s_{m,\max}),
s_{m,\max}-2,
s_{m,\max}-1,
s_{m,\max}
\right\},
\]

去重并限制在合法范围内。代码枚举三个模式候选值的笛卡尔积，同时只保留
\(s_1s_2s_3\leq\lfloor0.7n\rfloor\) 的组合。每个组合先用边际得分初始化，再运行固定规模张量 BCD。

全局精修使用更保守的张量 MSIC：

\[
\mathrm{MSIC}_{T}=
n\log(\mathrm{RSS}/n)
+\alpha q\ell_n
+\gamma\ell_n
\sum_{m=1}^{3}\log {p_m\choose s_m},
\qquad q=s_1s_2s_3.
\]

最终实验在此阶段取 \(\alpha=1.5,\gamma=1\)。先找到 MSIC 最小候选及其
\(q_{\mathrm{best}}\)，再构造 one-SE 风格容忍带

\[
T=\kappa\sqrt{
\max(q_{\mathrm{best}},1)
\max(\log(\max(n,2)),1)
}.
\]

所有满足

\[
\mathrm{MSIC}_{T}\leq
\mathrm{MSIC}_{\min}+T
\]

的候选都视为近似等价。代码优先选择其中 \(q\) 最小者；若 \(q\) 相同，再比较 MSIC 和
\(s_1+s_2+s_3\)。这里的“one-SE”是小模型优先的容忍带设计，并非由交叉验证折间标准误直接估计的经典 one-standard-error 规则。

### 3.7 最终 OLS 与输出

确定 \((S_1,S_2,S_3)\) 后，代码按支撑集的笛卡尔积构造设计矩阵

\[
Z_{i,(a,b,c)}=X_{iabc},
\quad
(a,b,c)\in S_1\times S_2\times S_3,
\quad
Z\in\mathbb{R}^{n\times q},
\]

用 numpy.linalg.lstsq 在矩形支撑内做 OLS，并将 \(q\) 个系数按 NumPy C-order 对应顺序放回完整的
\(p_1\times p_2\times p_3\) 张量，其余位置置零。

核心接口返回：估计张量 \(\widehat B\)、三个模式支撑、三个支撑大小、RSS、损失、最终准则值、损失/准则轨迹、初始化结果及 BCD 周期数。

## 4. 自适应 Tensor-ABESS 伪代码

~~~text
输入：X, y；初始化参数；BCD 与交换步参数
输出：B_hat, S1, S2, S3

1. 对三个模式计算切片边际得分，建立候选集 C1, C2, C3
2. 依次运行三个模式的 Group Lasso，得到初始支撑
3. 由初始支撑生成 s1_max, s2_max, s3_max
4. 缩减上界，使 s1_max*s2_max*s3_max <= floor(0.7*n)
5. 重复至收敛或达到最大周期：
   a. 固定 S2,S3；枚举 s1，运行 Group-BESS.Fix(s1)，由模式 MSIC 选 S1
   b. 固定新 S1 和 S3；枚举 s2，运行 Group-BESS.Fix(s2)，由模式 MSIC 选 S2
   c. 固定新 S1,S2；枚举 s3，运行 Group-BESS.Fix(s3)，由模式 MSIC 选 S3
   d. 在 S1*S2*S3 上 OLS 重估 B，并计算张量 MSIC
6. 生成全局候选 (s1,s2,s3)，逐个运行固定规模张量 BCD
7. 计算精修 MSIC；在容忍带内选择 q=s1*s2*s3 最小的候选
8. 返回该候选的 OLS 系数张量和支撑
~~~

## 5. 固定规模与自适应两条路径

代码保留两种运行方式：

| 路径 | 支撑大小来源 | 主要用途 |
|---|---|---|
| Fixed | 外部传入 target_sparsity=(s1,s2,s3) | 已知规模时检验支撑交换和系数估计能力 |
| Adaptive | 不传 target_sparsity，由 MSIC 与全局精修选择 | 未知真实稀疏度时的实际算法 |

Fixed 路径仍会执行初始化函数以生成记录和上界，但其 BCD 初始支撑直接取各模式边际得分最高的指定数量切片；支撑大小在全过程保持固定。Adaptive 路径才使用 Group Lasso 初始支撑、模式内 MSIC 和全局精修。

## 6. 最终数值模拟实际使用的算法参数

以下参数来自 v5 输出目录中的 checkpoint_manifest.json 及当前调用代码，不是核心函数的泛化默认值。

| 环节 | 参数 | 实际值 |
|---|---:|---:|
| 运行模式 | tensor_abess_mode | adaptive |
| 初始化选参 | abess_init_criterion | BIC |
| 初始化上界倍率 | abess_init_c | 3.0 |
| Group Lasso 网格数 | n_lambdas | 20 |
| Group Lasso 最大迭代 | max_iter | 500 |
| Group Lasso 收敛阈值 | tol | \(10^{-6}\) |
| 单次交换最大组数 | k_max | 1 |
| 固定规模交换最大迭代 | m_max | 15（核心入口默认值） |
| BCD 最大周期 | max_cycles | 6 |
| BCD 准则收敛阈值 | eps | \(10^{-8}\) |
| 模式内 MSIC | \((\alpha,\gamma)\) | \((1,1)\) |
| 全局精修 MSIC | \((\alpha,\gamma)\) | \((1.5,1)\) |
| 容忍带比例 | \(\kappa\) | 1.0 |
| 矩形参数上限 | \(q_{\max}\) | \(\lfloor0.7n\rfloor\) |

## 7. 实现边界与注意事项

1. **结构边界**：算法面向三阶张量和矩形支撑，不直接处理任意形状的元素级稀疏支撑。
2. **数值边界**：模型内部没有额外拟合截距，也没有自动中心化或标准化。当前模拟数据为零均值高斯设计，因此该处理是一致的；用于真实数据前应在外部完成适当预处理。
3. **搜索边界**：BCD、交换步和稀疏度候选搜索均为离散迭代，不能保证求得组合优化的全局最优解；多阶段初始化和全局精修用于降低局部解风险。
4. **计算开销**：更新模式 \(m\) 时，分组设计矩阵大小为 \(n\times(p_md_m)\)；自适应阶段还需枚举多个 \(s\)，全局精修需重复运行固定规模 BCD，因此主要耗时来自重复 OLS、组内 Gram 矩阵计算和候选规模搜索。
5. **参数耦合**：矩形支撑大小是三个模式组数的乘积。任一模式多选一个组，都可能显著增加 \(q\)，这也是代码同时设置 \(0.7n\) 上限和小模型优先规则的原因。

## 8. 核心实现位置

| 功能 | 代码文件 |
|---|---|
| Tensor-ABESS 总入口及 fixed/adaptive 分流 | src/tensor_abess/tensor/abess.py |
| 张量 BCD、全局候选精修和容忍带选择 | src/tensor_abess/tensor/bcd.py |
| 模式设计矩阵构造 | src/tensor_abess/tensor/reduction.py |
| 张量内积、损失与矩形支撑 OLS | src/tensor_abess/tensor/ops.py |
| SIC/MSIC 计算 | src/tensor_abess/tensor/sic.py |
| 边际预筛与 Group Lasso 初始化 | src/tensor_abess/utils/init.py |
| Group Lasso 的 ISTA 与 BIC/CV 选参 | src/tensor_abess/utils/glasso.py |
| 固定组数 Group-BESS | src/tensor_abess/group/bess_fix.py |
| 组交换及牺牲量 | src/tensor_abess/group/splicing.py、sacrifices.py |
| 数值模拟调用与参数记录 | numerical_simulation/methods.py、data_generation.py、main.py |

本文描述的是当前代码实际执行的算法。数值模拟中的 Tensor-GLasso、Vector Lasso 和 Oracle OLS 是独立对照方法，不参与 Tensor-ABESS 的最终拟合；其中只有初始化阶段的逐模 Group Lasso 属于 Tensor-ABESS 主流程。
