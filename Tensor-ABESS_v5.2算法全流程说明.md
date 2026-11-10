# Tensor-ABESS v5.2：数学、实现与历轮改动说明

核对日期：2026-10-25。对象是正式发布的 **`v5.2-search-final + 原 MSIC`**，不是后续 MSIC-FS 或 Warm 研究候选。本文按当前实际代码说明；文中的 v5、v5.1、v5.2 是仓库版本名称，与源码注释中的“Algorithm 5.0 / 5.1 / 5.2′”步骤编号分开理解。

**先读结论：** v5.2 保留原来的“初始化 → 逐模选支撑 → OLS → MSIC 选规模”框架，主要解决两件事：继承 v5.1 的“返回搜索途中真正最好的完整模型”修复；补全原上限内遗漏的整数规模。确定的收益是返回状态更可靠、规模覆盖更完整，以及部分场景支撑恢复明显改善；它并不保证所有场景的预测误差都下降。

正式身份以[发布记录](doc/releases/v5.2-search-final/CHANGELOG.md)、[32 文件清单](doc/releases/v5.2-search-final/RELEASE_MANIFEST.json)及[最终冻结说明](doc/taskcard11/FINAL_ALGORITHM_FREEZE.md)为准。本次核对 32 个生产 Python 文件均与发布 SHA256 清单一致；没有修改算法或重跑实验。

## 1. 算法解决什么问题

对第 $i$ 个样本，输入是三阶张量 $X_i\in\mathbb R^{p_1\times p_2\times p_3}$，响应为标量 $y_i$。模型为

$$
y_i=\langle X_i,B\rangle+\varepsilon_i,
\qquad
\langle X_i,B\rangle=\sum_{a=1}^{p_1}\sum_{b=1}^{p_2}\sum_{c=1}^{p_3}X_{iabc}B_{abc}.
$$

例如北京污染物数据中，三个轴是“站点 × 污染物 × 前一天时段”，响应是第二天目标站点的 PM2.5。算法希望同时选出有用的站点、污染物和时段。

设三个轴的入选位置分别为 $S_1,S_2,S_3$，大小为 $s_m=|S_m|$。系数满足

$$
\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3,
\qquad q=s_1s_2s_3.
$$

这称为**矩形支撑**：选的是三个轴上的集合，它们的所有组合构成允许非零的区域。区域内部的系数自由估计，仍可能等于零；$q$ 是矩形的坐标容量，不能直接当作模型选择后的完整统计自由度。本方法没有要求 $B$ 为 CP/Tucker 低秩张量。

固定规模 $(s_1,s_2,s_3)$ 时，希望近似求解

$$
\min_{\substack{|S_1|=s_1,\ |S_2|=s_2,\ |S_3|=s_3\\
\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3}}
L(B),\qquad
L(B)=\frac1{2n}\sum_{i=1}^{n}(y_i-\langle X_i,B\rangle)^2.
$$

代码中 `loss = RSS/(2*n)`，不是通常的 MSE；`rss = 2*n*loss`。该组合优化通过逐模搜索近似完成，不能把返回结果写成已证明的全局最优解。

## 2. 一张流程图读懂正式默认路径

```text
当前训练集 X[n,p1,p2,p3], y[n]
  │  缺失值处理、中心化/标准化由调用方完成，参数只在当前训练集估计
  ▼
每个轴计算边际分数 → 候选切片集合 C1,C2,C3
  ▼
按轴 1→2→3 做 Group Lasso 初始化
  ├─ 得到初始支撑 S1,S2,S3
  └─ 得到逐轴规模上限 caps，并压缩至容量预算内
  ▼
Adaptive BCD：每个轴枚举允许组数，用模式内 MSIC 选支撑
  └─ 按原停止规则得到一个 adaptive 终态
  ▼
最终候选池 = adaptive 终态
             + 原 caps 内全部合法整数规模的 Cold 固定规模 BCD 结果
  └─ 每个固定规模返回 best-visited 完整状态
  ▼
统一按原最终 MSIC 评分 → 在原容忍带内优先选小矩形
  ▼
返回 B、三个支撑、规模、RSS/loss、评分及有限的历史信息
  ▼
新样本张量内积预测；由调用方还原响应量纲
```

“Cold”指最终固定规模 BCD 的模式子问题使用原有冷启动探索；并不意味着整个算法完全没有初始化复用。原算法已有 Group Lasso 初始化、lambda 路径复用以及 adaptive 的同规模支撑复用，它们都被保留。

入口：[tensor_abess()](src/tensor_abess/tensor/abess.py)；主流程：[tensor_bcd_adaptive()](src/tensor_abess/tensor/bcd.py)。

## 3. 输入与预处理边界

`tensor_abess(X, y, ...)` 接收四维样本数组和一维响应，内部转为浮点数组。核心 OLS **不自动加入截距，不自动插补、中心化或标准化**。因此，实证调用方必须在每个训练折内完成所需处理，并将同一组参数用于验证或测试数据。

例如，对第 $j$ 个展平特征和响应进行训练集标准化：

$$
\widetilde X_{ij}=\frac{X_{ij}-\mu_j}{\sigma_j},\qquad
\widetilde y_i=\frac{y_i-\mu_y}{\sigma_y},\qquad
\widehat y_{\rm new}=\mu_y+\sigma_y\langle\widetilde X_{\rm new},\widehat B\rangle.
$$

常数特征的除数、缺失值规则和验证划分由具体实验协议确定，不能用测试集计算。若需要原量纲系数，则 $B_j^{\rm raw}=\sigma_y\widehat B_j/\sigma_j$，截距为 $b_0=\mu_y-\sum_j\mu_jB_j^{\rm raw}$。这是调用层的等价还原，不是核心函数新增的拟合步骤。

模型内部选 lambda 或选支撑，使用的都应是外层当前训练数据；内部训练过程的分数不能充当独立泛化成绩。对时间数据，应由外层协议保证时间顺序。

## 4. 第一步：边际预筛与 Group Lasso 初始化

对应：[utils/init.py](src/tensor_abess/utils/init.py)、[utils/glasso.py](src/tensor_abess/utils/glasso.py)。

### 4.1 每个轴先挑出有信号的切片

将第 $m$ 个轴的第 $k$ 个切片按样本展开为 $X^{(m,k)}\in\mathbb R^{n\times\prod_{\ell\ne m}p_\ell}$，边际分数为

$$
a_{mk}=\frac{\|(X^{(m,k)})^Ty\|_2^2}{n}.
$$

轴 $m$ 先取分数最大的 $\min(p_m,\max(1,\lfloor n^{2/3}\rfloor))$ 个位置，形成 $C_m$；再从中取最多 $\max(1,\lfloor\sqrt n\rfloor)$ 个位置，作为临时初始支撑。通俗地说，先用便宜的相关性计算缩小初始化的范围。

**预筛集合只用于初始化。** 后续 BCD 更新某个轴时，会考虑该轴全部 $p_m$ 个位置；不能把 $C_m$ 写成最终允许入选位置的永久限制。

### 4.2 按轴 1→2→3 做一次 Group Lasso

固定另外两个轴的临时支撑，将当前轴每个候选位置对应的整块特征视为一个组，求解

$$
\min_\beta\ \frac{\|y-Z\beta\|_2^2}{2n}
+\lambda\sum_g\|\beta_g\|_2.
$$

实际为无额外组权重的 ISTA：

$$
u^{(t)}=\beta^{(t)}-\eta Z^T(Z\beta^{(t)}-y)/n,
\qquad
\beta_g^{(t+1)}=\left(1-\frac{\eta\lambda}{\|u_g^{(t)}\|_2}\right)_+u_g^{(t)},
\qquad \eta=\frac1{\|Z\|_2^2/n}.
$$

零范数组直接置零；谱范数为零时步长取 1。默认最多 500 次迭代，满足
$\|\beta^{(t+1)}-\beta^{(t)}\|_2\le10^{-6}(1+\|\beta^{(t)}\|_2)$ 时停止。系数组范数大于 $10^{-8}$ 判为入选。

lambda 网格从
$\lambda_{\max}=\max_g\|Z_g^Ty/n\|_2$
下降到 $0.01\lambda_{\max}$，默认 20 个对数间隔值；当 $\lambda_{\max}\le0$ 时只用 0。选择有两个公开分支：

| 初始化选 lambda 规则 | 实际实现 |
|---|---|
| `criterion="cv"`，公共接口默认 | 按样本现有顺序 `array_split` 为默认 5 折，以验证残差平方和之和最小选 lambda，然后用全部当前训练数据重拟合；不随机打乱 |
| `criterion="bic"` | 选择 $n\log(\max(RSS/n,\mathrm{tiny}))+s d\log n$ 最小的 lambda；$s$ 为入选组数、$d$ 为每组列数；lambda 路径复用上一解 |

这里 CV 的候选切片、设计和 lambda 网格已经用当前完整训练数据构造，因此它是**初始化内部调参机制**，不能称为从预筛起完全独立的嵌套验证。北京实证显式使用 BIC 初始化，传入的 `cv_folds` 在该分支不起作用。

轴 2 使用刚更新的轴 1 支撑，轴 3 使用刚更新的轴 1、2 支撑。若某轴 Group Lasso 一个组也没选到，原实现选择该轴候选中边际分数最高的一个位置，确保三个支撑均非空；这是早已存在的初始化规则。

### 4.3 计算规模上限，约束搜索容量

记 Group Lasso 初始化规模为 $\widetilde s_m$，默认 $c=2$，先计算

$$
\bar s_m=\max\!\left(1,\min\!\left[p_m,
\max\!\left(\lceil c\widetilde s_m\rceil,
\max(1,\lfloor\sqrt n/2\rfloor)\right)\right]\right).
$$

令 $Q_n=\max(1,\lfloor0.7n\rfloor)$。若 $\bar s_1\bar s_2\bar s_3>Q_n$，反复把当前最大的一轴上限减 1；并列时 `np.argmax` 先取轴 1，再轴 2、3。直到乘积不超过 $Q_n$。

得到的 $s_{1,\max},s_{2,\max},s_{3,\max}$ 是训练数据决定的旧 caps。v5.2 没有扩大或重新分配这些上限。**容量足够也不意味着真实形状可表示**：如 caps 为 $(4,4,4)$，真实形状 $(4,3,5)$ 仍被第三轴上限排除。

实现细节：压缩 caps 不同步裁剪已经得到的 Group Lasso 初始支撑；因此 adaptive 首次 OLS 的初始矩形不一定满足最终 caps 或容量预算。随后逐轴枚举会按对应 caps 更新，正常正迭代预算下的 adaptive 终态与最终固定规模网格受这些上限约束。

## 5. 第二步的公共部件：把张量更新化成分组回归

对应：[tensor/reduction.py](src/tensor_abess/tensor/reduction.py)、[tensor/ops.py](src/tensor_abess/tensor/ops.py)。

以更新轴 1 为例，固定 $S_2,S_3$，每个轴 1 位置 $g$ 对应

$$
Z_g[:,(a,b)]=X[:,g,a,b],\quad(a,b)\in S_2\times S_3,\quad
d_1=|S_2||S_3|,\quad
Z=[Z_1,\ldots,Z_{p_1}]\in\mathbb R^{n\times p_1d_1}.
$$

因此，选 $s_1$ 个切片等价于从 $p_1$ 个组中选 $s_1$ 个完整组。轴 2、3 同理，组大小分别为 $|S_1||S_3|$ 和 $|S_1||S_2|$。每次当前轴更新后，下一轴立即使用更新后的支撑。

这里是笛卡尔组合索引，源码对应 `X[:, g, :, :][:, S2, :][:, :, S3].reshape(n, -1)`；不能直接写成 NumPy 的 `X[:, g, S2, S3]`，后者会采用不同的高级索引语义。

给定组集合 $A$，或给定完整矩形 $S_1\times S_2\times S_3$，系数均用普通最小二乘重新估计：

$$
\widehat\beta_A=\arg\min_\beta\|y-Z_A\beta\|_2^2.
$$

代码调用 `np.linalg.lstsq(..., rcond=None)`，不直接计算 $(Z_A^TZ_A)^{-1}$；秩不足时得到数值最小范数解，不自动变成 Ridge。张量 OLS 列顺序是轴 1 外层、轴 2 中层、轴 3 内层，系数按同序放回完整 $B$，矩形外填零。

## 6. Group-BESS：给定组数时，哪些组该换进换出

对应：[group/sacrifices.py](src/tensor_abess/group/sacrifices.py)、[group/splicing.py](src/tensor_abess/group/splicing.py)、[group/bess_fix.py](src/tensor_abess/group/bess_fix.py)。

### 6.1 初始组和交换分数

对每组定义

$$
\Gamma_g=Z_g^TZ_g/n,\qquad c_g=Z_g^Ty/n.
$$

无指定初始组时，按 $c_g^T\Gamma_g^{\dagger}c_g$ 由大到小取 $s$ 个组。这里 $\dagger$ 表示代码的求解语义：能求解时用 `np.linalg.solve`，奇异时报 `LinAlgError` 才转 `pinv`；不是额外的惩罚估计。

对当前组集合 OLS 拟合后，残差 $r=y-Z\widehat\beta$，计算：

$$
\xi_g=\tfrac12\widehat\beta_g^T\Gamma_g\widehat\beta_g
\quad(g\in A),\qquad
d_g=Z_g^Tr/n,\qquad
\zeta_g=\tfrac12d_g^T\Gamma_g^{\dagger}d_g
\quad(g\notin A).
$$

$\xi_g$ 小，表示删除该组的近似代价小；$\zeta_g$ 大，表示加入该组的近似收益大。这些分数用于提出交换方案，**最后是否接受由真实 OLS 损失下降决定**。

### 6.2 一轮 splicing 的实际顺序

1. 对当前组集合拟合 OLS，计算一次 $\xi,\zeta$。
2. 默认 `k_max=min(s,5)`；实际可交换数量还不能超过未入选组数。
3. 依次尝试 $k=1,\ldots,\min(k_{\max},s,J-s)$：换出 $k$ 个最小 $\xi$ 组，换入 $k$ 个最大 $\zeta$ 组，重新做 OLS。
4. 在本轮所有交换量中保留损失最低的方案。分数排名并列使用稳定排序，保留传入位置顺序；不同交换量若损失完全相同，不覆盖先找到的结果。
5. 只有损失改善达到阈值才返回新组集合：代码在 $L_{\rm old}-L_{\rm best}<\tau_s$ 时拒绝；等于阈值不由该条件拒绝。

默认阈值是

$$
\tau_s=0.01\,\frac{s\,d\,\log(\max(J,2))\,\ell_n}{n},
\qquad \ell_n=\log(\max(\log n,e)).
$$

`Group-BESS.Fix(s)` 重复上述交换，组集合不变即停止，否则达到 `m_max` 停止；从顶层入口传入的默认 `m_max=15`。每次交换保持组数恰好为 $s$。

注意两个“单调性”不同：单个 Group-BESS 子问题内部按损失筛选交换，但外层固定规模 BCD 会为新的模式子问题重新冷启动，所以完整探索轨迹可能暂时变差。

## 7. Adaptive BCD：先得到一个自适应候选

对应：[group/adaptive.py](src/tensor_abess/group/adaptive.py)、[tensor_bcd_adaptive()](src/tensor_abess/tensor/bcd.py)。

每轮按轴 1→2→3 更新。对于轴 $m$，固定其他两轴，枚举 $s=1,\ldots,s_{m,\max}$，各做一次 `Group-BESS.Fix(s)`，再按模式内 MSIC 选组数：

$$
\mathrm{MSIC}_m(s)
=n\log(\max(RSS_s/n,\mathrm{tiny}))
+s\,d_m\,\ell_n
+\ell_n\log {p_m\choose s}.
$$

这里模式内 $\alpha=\gamma=1$。按 $s$ 递增计算，严格 `<` 才更新最优，所以完全同分时保留先遇到的较小 $s$。

当枚举的 $s$ 恰好等于当前轴支撑大小时，原实现复用该支撑作为初值；其他 $s$ 使用 Gram 标准化边际分数初始化。它是原 adaptive 路径已有行为，不是任务卡后续研究的 Warm 路线。

三个轴更新完后，做完整矩形 OLS，并记录当轮全局评分。**这一阶段**全局 MSIC 的系数同样为 $\alpha=\gamma=1$：

$$
I_{\rm adaptive}
=n\log(\max(RSS/n,\mathrm{tiny}))
+q\ell_n+\ell_n\sum_{m=1}^3\log{p_m\choose s_m}.
$$

满足以下任一条件结束：三个支撑与上一轮相同；相邻全局评分绝对差小于 `eps=1e-8`；达到 `max_cycles=30`。这里返回 adaptive **终态**进入候选池，并非把全部 adaptive 中间状态都放入池中；v5.1 的 best-visited 保证针对下一节的固定规模 `tensor_bcd()`，不能扩大解释成整个 adaptive 流程的全历史最优保证。

## 8. v5.2 的关键步骤：完整整数规模搜索

### 8.1 究竟补全什么

对应：[bcd.py 中 _candidate_size_triples()](src/tensor_abess/tensor/bcd.py)。允许规模集合为

$$
\mathcal G=\{(s_1,s_2,s_3)\in\mathbb N_+^3:
s_m\le s_{m,\max},\ s_1s_2s_3\le Q_n\}.
$$

v5.2 将 $\mathcal G$ 中每个整数三元组都纳入最终精修。循环顺序为 $s_1$ 外层、$s_2$ 中层、$s_3$ 内层，均从 1 递增。正常默认 caps 已压缩到乘积不超预算，因此网格通常就是这个 caps 长方体内全部正整数规模；函数仍保留容量判断。

v5/v5.1 只取各轴若干锚点和靠近上限的数，例如 cap=10 时只试 $\{1,4,6,8,9,10\}$，遗漏 2、3、5、7。旧规则近上限窗口为 3，另含 1、`round(0.4*cap)`、`round(0.6*cap)` 等锚点。v5.2 改为 1 到 cap 全部整数。

**这只是规模穷举。** 给定规模后，从 $p_m$ 个位置挑哪些支撑，仍由局部搜索近似决定；没有枚举 $\prod_m{p_m\choose s_m}$ 个位置组合，也没有扩大 caps。

### 8.2 每个规模都独立做固定规模 BCD

对每个 $(s_1,s_2,s_3)$：

1. 在三个轴分别按全轴边际分数取前 $s_m$ 个位置，形成该规模初始矩形。
2. 做完整矩形 OLS，保存初始完整状态。
3. 每轮按轴 1→2→3，用 `Group-BESS.Fix(s_m)` 提出当前轴支撑。这里不传当前轴支撑作为 `groups_init`，保留原有 Cold 探索。
4. 每个模式提议后，重新做完整矩形 OLS，同时得到一致的 $(B,S_1,S_2,S_3,L)$。
5. 持续记录历史最小损失及其完整状态；停止后返回这一整套状态。

固定规模 BCD 调用中未传顶层 `tau`，因此最终规模精修使用其内部默认交换阈值；顶层 `tau` 仅传给 adaptive 阶段。这是当前调用链的实际边界。

### 8.3 v5.1 修复的核心：探索状态与返回状态分开

设第 $t$ 个已访问完整提议的损失为 $L_t$，保留

$$
t_*=\arg\min_{0\le t\le T}L_t,
\qquad (B_*,S_{1*},S_{2*},S_{3*})=(B_{t_*},S_{1t_*},S_{2t_*},S_{3t_*}).
$$

实现对 `candidate_loss < best_loss` 严格更新，完全并列保留先前状态。最佳系数、三个支撑和损失一起复制，避免“支撑来自这一步、系数或评分来自另一步”。初始状态也参与比较。

同时维护接受记录，只有

$$
L_{\rm new}<L_{\rm accepted}-\epsilon_{\rm acc},\qquad
\epsilon_{\rm acc}=64\epsilon_{\rm machine}
\max(1,|L_{\rm accepted}|,|L_{\rm new}|)
$$

才追加 `loss_history` 与 `accepted_support_history`。**该容差用于接受历史，不用于阻断探索，也不是 best-state 更新阈值。** 后续仍从当前提议继续探索，所以 `visited_loss_history` 可以非单调，`loss_history` 只记足够明显的改善；极小但真实的损失改善可以更新 best-state 而未进入接受历史。

每轮三轴更新结束后检查完整支撑：等于本轮开始时为 `unchanged_support`，重复更早的轮末支撑为 `exploration_cycle`；否则继续，最多 `max_cycles`。检查还覆盖有限数值、支撑数量、唯一性和索引范围；异常直接报错。

一个说明性例子：若同一规模途中损失为 10→8→9，修复后返回损失 8 所对应的整套模型。此序列只用于讲解，不是实验轨迹；更低训练损失仍可能带来更高测试误差。

## 9. 用原 MSIC 在规模间选最终模型

对应：[tensor/sic.py](src/tensor_abess/tensor/sic.py)、[bcd.py 中 _refine_msic_global_candidates()](src/tensor_abess/tensor/bcd.py)。

### 9.1 候选池和统一评分

候选池首先放 adaptive 终态，然后依网格顺序加入所有固定规模搜索结果；不按规模或支撑去重。各候选用同一最终评分重新比较：

$$
\boxed{
I(B,S)=n\log(\max(RSS/n,\mathrm{tiny}))
+1.5q\ell_n
+\ell_n\sum_{m=1}^{3}\log{p_m\choose s_m}
},\quad \ell_n=\log(\max(\log n,e)).
$$

三项分别是拟合误差、矩形容量惩罚、支撑组合数量惩罚。`tiny=np.finfo(float).tiny` 防止对零取对数；组合数的对数用 `gammaln` 计算：

$$
\log{p\choose s}=\log\Gamma(p+1)-\log\Gamma(s+1)-\log\Gamma(p-s+1).
$$

最终精修默认 $\alpha=1.5,\gamma=1$；不要与模式内或 adaptive 阶段的 $\alpha=1$ 混为一谈。**原 v5 已有这套最终 MSIC，v5.2 没有新发明或改写它。**

### 9.2 原容忍带：差不多好时优先小模型

取候选输入顺序中的首个评分最小者，记最小评分 $I_{\min}$、其容量 $q_{\rm best}$，计算

$$
\delta=\sqrt{\max(q_{\rm best},1)\,
\max(\log(\max(n,2)),1)}.
$$

在所有 $I\le I_{\min}+\delta$ 的候选中，按

$$
(q,\ I,\ s_1+s_2+s_3)
$$

字典序选最小者：先小容量，再低评分，再小的轴规模和；完全并列保留输入顺序。代码参数名包含 `one_se`，但这里的带宽来自上述确定公式，**并非通过交叉验证估计出的“一倍标准误”**。默认 `msic_refine_one_se_scale=1`；带内小模型优先也早已存在于 v5。

最终系数就是对应矩形的 OLS 系数，不做 Ridge 收缩、测试集重选或事后修正。若最终选择来自固定规模路径，系数与该路径 best-visited 支撑成套返回。

## 10. 对照代码的伪代码与接口

### 10.1 正式默认主流程伪代码

```text
输入：已按当前训练集处理的 X, y
init = 边际预筛 + 顺序 Group Lasso
caps = 初始化规模扩张后按旧预算压缩

A = AdaptiveBCD(X, y, init.supports, caps, mode_MSIC_alpha=1)
pool = [A]                               # 只放 adaptive 终态
for s1 = 1..cap1:
  for s2 = 1..cap2:
    for s3 = 1..cap3:
      若 s1*s2*s3 > max(1,floor(0.7*n))：跳过
      S0 = 各轴边际分数的前 s1,s2,s3 个位置
      F = FixedBCD_best_visited(X,y,s1,s2,s3,S0)
      pool.append(F)

对 pool 全部候选计算最终 MSIC(alpha=1.5,gamma=1)
用首个 argmin 的容量确定容忍带
在带内按 (容量,MSIC,轴规模和) 选择
返回完整模型与记录
```

### 10.2 默认参数与实际作用

| 参数 | 顶层默认值 | 作用 |
|---|---:|---|
| `target_sparsity` | `None` | 正式默认自适应选规模；非空时进入另一个已知规模分支 |
| `criterion` | `"cv"` | 仅 Group Lasso 初始化选择 lambda；具体实验可预声明 `"bic"` |
| `c` | 2.0 | 从初始化规模构造旧 caps |
| `n_lambdas / cv_folds` | 20 / 5 | 初始化 lambda 网格、内部 CV 折数 |
| `max_iter / tol` | 500 / $10^{-6}$ | 初始化 ISTA 迭代预算和系数变化容差 |
| `adaptive_criterion` | `"msic"` | adaptive 及最终精修路径选择 |
| `msic_refine_alpha / gamma` | 1.5 / 1.0 | 最终精修全局 MSIC 参数，不改变模式内默认系数 |
| `msic_refine_one_se_scale` | 1.0 | 最终确定性容忍带倍率 |
| `max_cycles` | 30 | adaptive BCD 及每个固定规模 BCD 各自的最大轮数 |
| `m_max / k_max` | 15 / `None` | Group-BESS 最多交换轮数；`None` 对应最多交换 `min(s,5)` 组 |
| `tau / eps` | `None` / $10^{-8}$ | adaptive 交换阈值使用默认公式；adaptive 评分变化停止阈值 |

公开函数允许更换参数，不代表历史正式实验都使用接口默认值，也不代表改变参数后仍能直接复用历史成绩。复现实验须使用其冻结配置和训练划分。

两个兼容分支需区分：`target_sparsity` 非空时仍先计算初始化，随后只跑指定规模的固定 BCD，不跑最终全网格与 MSIC 容忍带，返回字段 `sic` 用旧 SIC 公式；`adaptive_criterion="sic"` 时 adaptive 使用 SIC，并直接返回、不执行 MSIC 全网格精修。这些分支不是本文评估的正式默认“原 MSIC”路径。

### 10.3 最小调用及输出含义

```python
from tensor_abess.tensor.abess import tensor_abess
from tensor_abess.tensor.ops import tensor_inner

# X_train_ready、y_train_ready 已仅用训练集参数完成预处理。
# X_test_ready 必须沿用这些训练参数；本例不使用真实数据进行拟合。
result = tensor_abess(X_train_ready, y_train_ready)
y_pred_ready = tensor_inner(X_test_ready, result.B)
# 如训练响应被标准化，继续用训练响应的均值/尺度还原。
```

| 输出 | 含义与限制 |
|---|---|
| `B`, `S1/S2/S3`, `s1/s2/s3` | 最终系数、零基索引支撑、轴规模；完整系数形状为 $(p_1,p_2,p_3)$ |
| `s_max`, `init_supports`, `init_result` | caps、原初始化支撑和初始化信息 |
| `loss`, `rss`, `sic` | 默认路径的最终训练损失、RSS、最终 MSIC；字段名 `sic` 并不表示一定使用旧 SIC |
| `loss_history`, `sic_history` | 当前返回分支的历史信息；不等于整个全网格所有候选和耗时的完整账本 |
| `bcd_result` | 默认路径为 adaptive 结果包装；指定规模分支可直接获得 fixed 结果 |

固定 BCD 的底层结果还含 `best_iteration`、`best_rss`、`stop_reason`、`visited_loss_history`、`accepted_support_history`；默认全网格包装没有把所有候选的这些字段全部暴露。`n_cycles` 也不是全网格工作量总和。初始化 `GroupLassoFit` 未返回统一收敛标志，不能仅凭正常返回就宣称全部内部优化均已达到收敛。

### 10.4 计算量为什么会上升

设合法规模数量为 $G=|\mathcal G|$，最终阶段大致要调用 $G$ 次固定 BCD，每次至多 30 个三轴循环，每个子问题包含多轮交换与 OLS。仅看完整矩形 OLS，常规稠密求解在 $n\ge q$ 时量级约为 $O(nq^2)$，此外还有组 Gram、打分和交换拟合成本；实际总成本依赖停止时间与秩结构。

v5.2 的收益来自补足搜索覆盖，代价是候选数和计算量可能增加。它没有把位置组合完全穷举，也没有统一运行时加速承诺。

## 11. 历轮问题、原因与处理

以下按任务卡顺序记录“当轮现象 → 为什么处理 → 最终落点”；研究试验与正式改动分开。各轮的要求以任务卡为起点，完成情况以对应验收报告为准。

| 轮次及依据 | 当轮问题现象 | 改动或研究的原因与处理 | 正式版落点 |
|---|---|---|---|
| [1：LEMON 增量检验](任务卡/任务卡1.md) | 需要区分 EEG 有无额外预测信息，以及算法自身表现 | 建立仅协变量与 EEG 模型的公平比较，核查训练内预处理与 OOF 评价；异常另行诊断 | 评价流程，不改算法 |
| [2：BCD 修复](doc/BCD_FIX_REPORT.md) | 固定规模 BCD 的 RSS 可回升、循环；最后模型未必是途中最好模型 | 保留 Cold 探索，同步保存最佳系数、支撑和损失，分开探索/接受记录，加循环与数值保护；首个 warm-only 草案未通过恢复回归而被否决 | **进入 v5.1，由 v5.2 继承** |
| [3：候选覆盖诊断](doc/CANDIDATE_SEARCH_AUDIT.md) | caps、规模网格、位置搜索会分别阻断真模型；$n=400$ 旧固定网格漏掉 $(4,3,5)$ | 分开测试扩 caps、补网格、多初始化；扩 caps 有过选风险，多起点收益不够稳，补网格 A2 更有依据 | **只保留补整数规模路线** |
| [4：A2 独立确认](doc/A2_VS_A0_DECISION.md) | 需要排除“只修好一个场景”，并检查成本与选阶副作用 | 新主 90 例 Exact 15→36，PMSE 2.500011→2.474219；同时评估计算代价；新 MSIC 确认按预算停止，未获充分推广证据 | 形成 D2：BCD 修复＋旧 caps＋完整网格＋原 MSIC |
| [5：正式发布](doc/V52_RELEASE_REPORT.md) | 代码身份、实验身份和部署版本必须一致 | 用户批准 D2，正式发布 `v5.2-search-final`，建立逐文件指纹与新结果入口，旧结果保持原标签 | **正式采用 D2** |
| [6：全量验收](doc/TASKCARD6_EXECUTION_REPORT.md) | 模拟恢复增加但总体 PMSE 变差；LEMON RMSE 0.371667→0.504581 | 报告尾部退化及实证负面结果；准备三版本归因，当时 v5.1 全量配对尚未完成 | 不改算法，保留证据缺口 |
| [7：三版本归因](doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md) | BCD 修复和补网格不能合并解释 | 后续 250 配对补齐：$n=100$ 恶化发生在 BCD 段，$n=400$ Exact 1→1→49 的主要收益来自网格段 | 保留正式版，明确收益来源 |
| [8：Warm 与新选阶](doc/WARM_START_FINAL_DECISION.md) | 收到的 Warm 同时改变保护和网格，恢复检查失败；更强惩罚联合取消小模型带未过最坏场景门槛 | 原样 Warm 正确性门禁失败，性能线未启动；216 库研究 `alpha=2/scale=0`，未通过推广 | **Warm 与新 MSIC 均不进入**；另见[选阶决策](doc/MSIC_NEW_METHOD_DECISION.md) |
| [9：小样本机制拆分](doc/taskcard9/TASKCARD9_FINAL_DECISION.md) | $n=100$ 出现训练 RSS 降而测试误差升；同规模不同位置也会过拟合 | 分开研究惩罚和容忍带，检查候选排序；Pure Warm 固定规模原型通过 25/25 正确性门禁，但保留 Cold，未完成全流程泛化确认 | 仅诊断和研究原型，不发布 |
| [10：严格内层 CV](doc/taskcard10/TASKCARD10_FINAL_DECISION.md) | 内折样本变少同时改变容量，可能混淆 CV 收益；计算开销大 | 比较原生 MSIC、匹配容量 MSIC、同库 CV；开发 48 例完成 30 例，$n=400$ 达 900 秒超时，独立确认 0 例 | **CV 不进入，按预算停止** |
| [11：MSIC-FS](doc/taskcard11/FINAL_ALGORITHM_DECISION.md) | 需要缓解小样本过选，同时保护强信号恢复 | 只研究 $I_{\rm FS}=I_0+\ell_nq^2/(2n)$；新 96 对整体 PMSE 降 26.92%，但强信号欠选率增量 0.125 超过 0.05 门槛 | **FS 不采用，冻结原 MSIC** |
| [12：FS 扩展复验](doc/taskcard12/FINAL_VERSION_RECOMMENDATION.md) | 进一步核查 $n=200$ 恢复风险，不能只看预测均值 | 主 360 对 PMSE 降 19.65%，但 11 项门槛中 4 项失败；压力 30 例 Exact 12→8，原成功 12 例有 5 例丢失 | **仍保留原 MSIC，FS 仅研究对照** |
| [13：Apple](doc/taskcard13/APPLE_EMPIRICAL_FINAL_REPORT.md) | 严格验证下整体预测信息弱，ABESS 平均 R² 约 0.0086 | 核验样本配对、固定表示与训练内处理，完整保留弱信号结果，后续暂停 Apple | 不改算法 |
| 两卡间：[Mushroom CSV 筛查](doc/mushroom_screen/MUSHROOM_SCREEN_REPORT.md) | 需要先判断蘑菇响应是否有可预测信息，再投入张量实证 | 仅用已有 CSV 比较训练均值、Ridge、PLS，按重复光谱分组，检查响应范围与质量 | 低成本前置检查；**目录中未发现任务卡14，不将其补称为卡14** |
| [15：Mushroom 完整实证](doc/mushroom_empirical/MUSHROOM_EMPIRICAL_FINAL_REPORT.md) | 主交叉验证可预测，文件块留一却全部方法 R² 为负 | 保留主协议与分块外推的差异，定位为补充或边界实证；文件块不直接等同已证实的生物批次 | 不改算法 |
| [16：北京污染物](doc/beijing_air_quality/BEIJING_EMPIRICAL_FINAL_REPORT.md) | 需要自然三阶结构与真正未来验证的应用 | 固定站点×污染物×时段，按时间训练/检验；未来误差优于持续性，但与单站 Ridge 接近 | 不改算法；作为有限定的主实证候选 |

这条历史链应概括为：**先修正确性，再补覆盖，随后完整报告代价，并研究但没有发布新的选阶规则，最后转向适配的实证。** 任务卡提出某个方案，只代表当时允许研究；正式最终版本仍以卡5发布及卡11、12最终决定为准。

## 12. 与原 v5 的最终差别

| 比较点 | 原 v5 | 正式 v5.2 | 实际收益与边界 |
|---|---|---|---|
| 固定规模返回状态 | 可能返回较差的末状态，未可靠保留完整历史最优状态 | 继承 v5.1：同步保存 best $B$、三轴支撑、损失并返回 | 修复“找到更好模型却没返回”的问题；更低训练误差不保证更低测试误差 |
| 探索与接受记录 | 不能充分区分探索轨迹和最终有效模型状态 | 分离 visited 与 accepted 记录，保留 Cold 探索；增加循环及数值一致性保护 | 可解释地保留探索机会，并避免返回不一致模型；不承诺探索步步单调 |
| 最终规模候选 | 少数锚点及近 cap 规模 | 原 caps 内全部合法正整数规模 | 消除规模网格遗漏；位置支撑仍近似搜索 |
| 初始化、caps、容量 | 原预筛、Group Lasso、caps、$q\le Q_n$ | 保留 | 控制变量，保留旧适用边界；真形状可能仍在 caps 外 |
| 模式内 / 最终 MSIC | 已有原 MSIC 及最终小模型优先带 | 保留原参数与选择顺序 | 版本收益不能归功于新增 MSIC；未纳入 FS |
| 系数估计 | 矩形内 OLS | 保留 OLS | 不把改动混同成收缩估计或新统计模型 |
| 最终搜索路线 | Cold 机制 | 保留 Cold；研究 Warm 未发布 | 不因部分数据上 Warm 较好就改变正式规则 |

相对 v5.1，v5.2 的唯一核心文件改动是 `tensor/bcd.py` 中规模网格生成逻辑；32 个核心 Python 文件中 31 个未变。相对原 v5，则要同时计入 v5.1 的固定 BCD 正确性修复，不能只说“扩大了搜索网格”。

## 13. 已测收益与代价：把三个层次分开

### 13.1 同输入三版本配对：改动收益来自哪里

下表来自相同输入的三版本归因实验，每行 50 次重复；PMSE 是对含噪测试响应的平均平方预测误差，越低越好；Exact 是三个轴支撑同时完全恢复的次数。

| 场景 | v5 PMSE | v5.1 PMSE | v5.2 PMSE | Exact：v5 → v5.1 → v5.2 |
|---|---:|---:|---:|---:|
| 基准 $n=200$ | 1.990669 | 2.012130 | 2.012130 | 15 → 16 → 16 |
| 弱信号 SNR=0.5 | 1.837097 | 1.926368 | 1.926368 | 0 → 0 → 0 |
| 小样本 $n=100$ | 5.196560 | 6.442026 | 6.442026 | 0 → 0 → 0 |
| 较大样本 $n=400$ | 1.297588 | 1.299304 | 1.189516 | 1 → 1 → 49 |
| 无相关 $\rho=0$ | 2.573943 | 2.948160 | 2.948160 | 12 → 10 → 10 |

关键解释：

- **补网格的明确成功例是 $n=400$**：v5.1→v5.2 PMSE 从 1.299304 降到 1.189516，约降低 8.45%；Exact 从 1/50 提高到 49/50。规模漏搜得到实质纠正。
- **小样本恶化主要发生在 v5→v5.1**：该行 v5.1 与 v5.2 相同，不能把恶化归因于 v5.2 补整数网格。更充分的训练拟合会暴露模型选择风险；该设置 caps $(4,4,4)$ 还排除了真形状 $(4,3,5)$。
- 正确性修复应保留，但不能据此宣传“预测必然更好”。基准、弱信号及无相关场景的负面变化都应保留报告。

来源：[三版本配对明细](numerical_simulation/outputs_v51_attribution/cluster_run/paired_three_versions.csv)、[最终归因说明](doc/VERSION_EFFECT_DECOMPOSITION_FINAL.md)、[n100 归因](doc/taskcard9/N100_REGRESSION_ATTRIBUTION.md)。

归因范围：这 250 个完整输入唯一，但跨场景共享部分 seed 家族，不能当作 250 个独立同分布观测做池化显著性推断。历史验收核对的是冻结生成配置、seed、版本和原始记录；本次说明书没有重建历史全部 X/y 或重新计算所有测试指标。

### 13.2 全量模拟：恢复增加，平均预测误差也增加

| 汇总口径 | 原 v5 | 正式 v5.2 | 解释 |
|---|---:|---:|---|
| 700 个任务标签的平均 PMSE | 2.236761 | 2.381274 | 约上升 6.46%，并非整体预测改善 |
| 同一口径 Exact 次数 | 138/700 | 182/700 | 完全恢复增加 44 次 |
| 去掉重复输入后的平均 PMSE，600 组 | 2.274498 | 2.438721 | 约上升 7.22% |
| 去重口径 Exact 次数 | 120/600 | 162/600 | 完全恢复增加 42 次 |

正式模拟为 14 个场景标签×50 次，包含 10 种参数配置、600 组唯一完整输入；部分标签复用相同输入，不能当作 14 组独立确认。v5.2 在 11/14 个标签的平均 PMSE 低于该批次 Lasso 和 Group Lasso，弱信号 SNR=0.5、1 及 $n=100$ 三类不占优；这是与基线的比较，不等于相比 v5 的提升。

逐任务对比中，预测误差降低/基本相同/升高分别为 99/507/94（相同容差 $10^{-12}$）；Exact 新增 57 次、丢失 13 次，净增 44 次。历史记录的平均拟合时间为 25.0282→31.3756 秒，但不是控制 CPU 型号、节点负载等条件的专门计时实验，不能把约 25.36% 的增幅全部归因于算法。

该批基线使用当时冻结的 NumPy 求解器与预算，不能扩展为对所有调优实现的普遍优越性。Oracle-OLS 已知真实支撑，只是诊断参照，不能列作可部署竞争模型或普遍风险下界。

来源：[完整模拟评审](doc/V52_FULL_SIMULATION_REVIEW.md)、[三版本归因报告](doc/V51_FULL_ATTRIBUTION_REPORT.md)、[正式模拟汇总](numerical_simulation/outputs_v52_search_final_full/summary_results.csv)。

### 13.3 实证给出的证据范围

| 数据 / 验证 | 关键结果 | 能说明什么 |
|---|---|---|
| LEMON，原 v5 与 v5.2 严格配对 | 平均重复 OOF RMSE：0.371667→0.504581；R²：0.121430→−0.597008 | 明确退化；v5.1 与 v5.2 的 15 折系数、支撑及 435 条预测完全相同，差异来自 BCD 阶段而非补网格 |
| Apple，正式 v5.2 重复交叉验证 | R² 均值约 0.0086，最佳 Ridge 约 0.0350 | 整体预测信息弱，不能作为算法优势证据 |
| Mushroom，正式 v5.2 分组交叉验证 | RMSE 13.65235、R² 0.785916；已有平均光谱 Ridge 为 11.12521、0.858019 | 有预测信息，但张量法未优于简单光谱基线；按文件分块外推时 ABESS R² −0.157987，全部 8 种方法均为负 |
| 北京污染物，352 个共同有效未来日期 | ABESS RMSE 47.45252 µg/m³、R² 0.581631；Persistence RMSE 68.75117，降低 30.98% | 支持真实未来预测与紧凑矩形选择；单站点 Ridge RMSE 47.46519，尚无明确多站点增量优势 |

LEMON 的 RMSE/R² 是每次重复先合并完整 145 人 OOF、再对 3 次重复取均值，不是平均 15 个外折指标。其 caps 为 $(4,4,5)$，v5.1/v5.2 的固定规模网格均为 80 个，另加 1 个 adaptive 候选；不能把这次退化说成新网格导致。

北京最终选择 $5\times2\times2=20$ 个允许坐标，占原 $12\times6\times6=432$ 个的 4.63%；对应站点、污染物、前一天晚间时段。它提供紧凑的预测结构，不代表因果污染来源识别，也不代表优于所有基线。该评估是固定训练模型、每天更新前一天观测输入的未来逐日预测，不是一次性预测全年。

来源：[LEMON 三版本比较](doc/LEMON_THREE_VERSION_COMPARISON.md)、[Apple 报告](doc/taskcard13/APPLE_EMPIRICAL_FINAL_REPORT.md)、[Mushroom 报告](doc/mushroom_empirical/MUSHROOM_EMPIRICAL_FINAL_REPORT.md)、[北京报告](doc/beijing_air_quality/BEIJING_EMPIRICAL_FINAL_REPORT.md)。

这些结果支持把 v5.2 描述为“具有明确矩形结构假设、支撑选择可读、经严格验证但有边界的方法”。北京等新数据的性能是正式版的应用证据；没有对同一新数据重跑原 v5，就不能把全部实证表现宣称为 v5→v5.2 的因果收益。

## 14. 如何准确概括最终版本

可以这样向老师解释：

> 原来的方法已经会按三个轴选变量，也已经用 MSIC 控制模型大小。现在先修好了“中途找到更好模型，却可能最后没有返回它”的问题，再把原范围内漏掉的模型规模补齐。这样搜索结果更可靠、部分场景的真实支撑恢复明显提高；但小样本和弱信号仍可能过拟合，所以我们同时报告预测退化，并保留严格的独立验证。

不宜写成：“v5.2 新增 MSIC”“所有 BCD 步骤都单调下降”“全局最优已得到保证”“枚举了所有位置组合”“所有数据预测都变好”“FS/Warm 已成为正式默认”或“北京结果已证明多站点一定优于单站点”。

## 15. 查代码与追溯结果的最短路径

| 要查的问题 | 直接入口 |
|---|---|
| 顶层 API、默认值、兼容分支 | [tensor/abess.py](src/tensor_abess/tensor/abess.py) |
| 边际预筛、Group Lasso 初始支撑、caps | [utils/init.py](src/tensor_abess/utils/init.py) |
| ISTA、lambda 网格、CV/BIC | [utils/glasso.py](src/tensor_abess/utils/glasso.py) |
| 按模式展开与组映射 | [tensor/reduction.py](src/tensor_abess/tensor/reduction.py) |
| 组分数与交换 | [group/sacrifices.py](src/tensor_abess/group/sacrifices.py)、[group/splicing.py](src/tensor_abess/group/splicing.py) |
| 固定组数和模式内选规模 | [group/bess_fix.py](src/tensor_abess/group/bess_fix.py)、[group/adaptive.py](src/tensor_abess/group/adaptive.py) |
| 固定 BCD 修复、adaptive、整数网格及最终选择 | [tensor/bcd.py](src/tensor_abess/tensor/bcd.py) |
| SIC/MSIC 数学定义 | [tensor/sic.py](src/tensor_abess/tensor/sic.py) |
| 张量 OLS、系数还原、预测 | [tensor/ops.py](src/tensor_abess/tensor/ops.py) |
| 正式版本身份 | [发布清单](doc/releases/v5.2-search-final/RELEASE_MANIFEST.json)、[原始字节 SHA256](doc/releases/v5.2-search-final/SOURCE_SHA256.txt) |
| 后续研究为何没有替换正式版本 | [任务卡11冻结](doc/taskcard11/FINAL_ALGORITHM_FREEZE.md)、[任务卡12终版建议](doc/taskcard12/FINAL_VERSION_RECOMMENDATION.md) |

核对指纹：`tensor/bcd.py` 为 `db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`；`tensor/sic.py` 为 `84637e426a8ad8d9c471d8f81787135052f10f954475127403d5a9b2b4f51d9f`。旧候选目录中保留的“尚未发布”历史状态不覆盖任务卡5之后的正式发布身份。
