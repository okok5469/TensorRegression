# 2 算法与方法

> 核验状态（2026-10-29）：本章依据正式 `v5.2-search-final + 原MSIC` 的冻结实现；任务卡18已完成600套唯一输入上两个参考对照的1200次有效拟合与1200次测试评价。本轮结果已纳入模拟章，章节联合状态以最终独立审查和Gate验收为准。

## 2.1 模型与矩形支撑

考虑具有三阶张量预测变量的标量回归。对当前训练集的第 $i$ 个样本，令 $X_i\in\mathbb R^{p_1\times p_2\times p_3}$，响应满足

\[
y_i=\langle X_i,B^\star\rangle+\varepsilon_i,
\qquad
\langle X_i,B\rangle=\sum_{a=1}^{p_1}\sum_{b=1}^{p_2}\sum_{c=1}^{p_3}X_{iabc}B_{abc}.
\tag{2.1}
\]

算法以平方损失定义估计程序，不在运行时检验误差分布。本文数值研究采用独立于预测变量的独立同分布高斯误差 $\varepsilon_i\sim N(0,\sigma^2)$；这些实验假设不被外推为一般误差下的恢复或风险保证。

算法分别选择三个轴上的位置集合 $S_m\subseteq\{1,\ldots,p_m\}$，记 $s_m=|S_m|$。模型的允许支撑为 $S_1\times S_2\times S_3$，并令

\[
\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3,
\qquad q=s_1s_2s_3.
\tag{2.2}
\]

这里的“矩形”指三个集合的笛卡尔积，并不要求任何轴上的位置连续。矩形内部的系数自由估计，可以出现零系数；$q$ 是允许估计的坐标容量，不等于实际非零元素数，也不直接等于经过数据选择后的统计自由度。本模型不对系数施加CP或Tucker低秩约束。

对于给定规模 $s=(s_1,s_2,s_3)$，理想的组合估计目标为

\[
\min_{\substack{|S_m|=s_m,\ m=1,2,3\\
\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3}}
L_n(B),\qquad
L_n(B)=\frac1{2n}\sum_{i=1}^n\{y_i-\langle X_i,B\rangle\}^2.
\tag{2.3}
\]

Tensor-ABESS以逐模组选择近似求解位置组合，再比较不同规模的候选模型；实际输出不被定义为式（2.3）的全局最优解。下文统一使用 $RSS=2nL_n$，从而区分训练损失 $L_n$ 与均方误差 $RSS/n$。

核心程序不自动增加截距，也不执行缺失值插补、中心化或标准化。需要这些处理的应用应只在当前训练集估计处理参数，并将同一变换用于验证集或测试集。本文模拟采用其预先规定的零截距模型，无额外样本中心化或标准化。正文使用一基索引；代码的支撑数组采用零基索引。

## 2.2 初始化与训练数据决定的搜索上限

初始化为位置搜索提供起点，并确定逐轴规模上限。对轴 $m$ 的位置 $g$，把该切片中所有元素按固定顺序展平为样本设计 $X_{(m),g}$，计算边际分数

\[
a_g^{(m)}=\frac1n\big\|X_{(m),g}^{\mathsf T}y\big\|_2^2.
\tag{2.4}
\]

在每个轴保留分数最高的 $\min\{p_m,\max(1,\lfloor n^{2/3}\rfloor)\}$ 个位置，得到 $C_m$。三个临时支撑最初分别取 $C_m$ 中分数最高的 $\min\{|C_m|,\max(1,\lfloor\sqrt n\rfloor)\}$ 个位置。随后按轴 $1\to2\to3$ 进行一次顺序Group Lasso选择：当前轴只考虑 $C_m$，其他轴使用当时的临时支撑，因此后续轴会使用前面已经更新的支撑。

每个初始化子问题采用无额外组权重的目标

\[
\min_\beta\ \frac{\|y-Z\beta\|_2^2}{2n}
+\lambda\sum_{g=1}^{J}\|\beta_g\|_2.
\tag{2.5}
\]

正式实现使用ISTA。令 $H=\|Z\|_2^2/n$，当 $H>0$ 时步长 $\eta=H^{-1}$，否则取 $\eta=1$。每步先计算 $u=\beta-\eta Z^{\mathsf T}(Z\beta-y)/n$，再按组更新

\[
\beta_g^+=\left(1-\frac{\eta\lambda}{\|u_g\|_2}\right)_+u_g,
\tag{2.6}
\]

其中零范数组置零。默认最多500步，或在 $\|\beta^+-\beta\|_2\le10^{-6}(1+\|\beta\|_2)$ 时停止；组范数大于 $10^{-8}$ 判为入选。这是冻结算法的有限初始化过程，不构成每个凸子问题均达到给定KKT精度的声明。

lambda路径默认含20个对数间隔点，由 $\lambda_{\max}=\max_g\|Z_g^{\mathsf T}y/n\|_2$ 下降至 $0.01\lambda_{\max}$；当 $\lambda_{\max}\le0$ 时只取0。BIC分支以

\[
n\log\{\max(RSS_\lambda/n,\mathrm{tiny})\}
+|A_\lambda|d\log n
\tag{2.7}
\]

选lambda，其中 $d$ 是当前组大小，路径内复用前一解；严格改善才更新，完全并列保留先遇到的lambda。公共接口还提供默认CV分支：按当前样本顺序用 `array_split` 划为5折，以各折验证SSE之和选lambda，再在当前完整训练数据上拟合；该实现的每折拟合从零开始。候选切片、局部设计及lambda网格均已由当前完整训练过程确定，故它是初始化内部调参，不能称为从预筛起完全独立的嵌套验证。正式模拟使用式（2.7）的BIC分支。某轴若未选中任何组，则取其候选中边际分数最高的位置，保证支撑非空。

记顺序选择后的规模为 $\widetilde s_m$。先计算

\[
\bar s_m=\max\!\left\{1,\min\!\left[p_m,
\max\!\left\{\lceil c\widetilde s_m\rceil,
\max(1,\lfloor\sqrt n/2\rfloor)\right\}\right]\right\},
\qquad Q_n=\max\{1,\lfloor0.7n\rfloor\}.
\tag{2.8}
\]

若 $\bar s_1\bar s_2\bar s_3>Q_n$，反复将当前最大的一轴上限减1，直至乘积不超过 $Q_n$；最大值并列时依轴 $1,2,3$ 的顺序处理。所得上限记为 $s_{m,\max}$，下文简称caps。公共接口默认 $c=2$，正式模拟使用 $c=3$。这些上限由训练数据决定；即使真实容量小于 $Q_n$，真实形状仍可能被单轴上限排除。

压缩caps不会同步裁剪已经得到的初始化支撑。因此adaptive阶段首次OLS使用的初始矩形可能超过caps或容量预算；在正的循环预算下，后续逐轴更新受各轴caps约束。本实现细节应与“最终规模候选受caps约束”分开理解。预筛集合 $C_m$ 仅限制初始化；后续逐模搜索会重新考虑当前轴的全部 $p_m$ 个位置。

## 2.3 逐模分组归约与Group-BESS

固定其他两轴支撑后，当前轴的每个位置构成一个回归组。例如更新轴1时，对 $g=1,\ldots,p_1$ 构造

\[
Z_g[:,(a,b)]=X[:,g,a,b],\quad(a,b)\in S_2\times S_3,
\qquad d_1=s_2s_3,
\qquad Z=[Z_1,\ldots,Z_{p_1}].
\tag{2.9}
\]

轴2、3的组大小分别为 $d_2=s_1s_3$、$d_3=s_1s_2$。同一轮按轴 $1\to2\to3$ 更新，后面的轴立即使用前面的更新结果。矩形设计的列顺序为轴1外层、轴2中层、轴3内层，与C-order展平一致。

给定组集合 $A$，在 $Z_A$ 上以 `np.linalg.lstsq(..., rcond=None)` 重新拟合OLS；给定完整矩形时采用相同规则，再将系数按列顺序放回张量，矩形外填零。秩不足时使用该数值最小二乘函数给出的最小范数解，不以Ridge替换，也不直接求逆 $Z_A^{\mathsf T}Z_A$。

Group-BESS在给定组数 $s$ 下通过整组交换搜索位置。令

\[
\Gamma_g=Z_g^{\mathsf T}Z_g/n,\quad c_g=Z_g^{\mathsf T}y/n,
\qquad r=y-Z\widehat\beta_A,\quad d_g=Z_g^{\mathsf T}r/n.
\tag{2.10}
\]

没有指定初始组时，按 $c_g^{\mathsf T}\Gamma_g^\dagger c_g$ 取前 $s$ 组。这里 $\Gamma_g^\dagger v$ 表示实际求解语义：首先调用线性方程求解，仅在抛出奇异矩阵错误时改用伪逆。每轮交换以当前组集合OLS计算后向和前向分数

\[
\xi_g=\tfrac12\widehat\beta_g^{\mathsf T}\Gamma_g\widehat\beta_g\quad(g\in A),
\qquad
\zeta_g=\tfrac12d_g^{\mathsf T}\Gamma_g^\dagger d_g\quad(g\notin A).
\tag{2.11}
\]

分数只用于提出交换。对 $k=1,\ldots,\min\{k_{\max},s,J-s\}$，同时移出后向分数最小的 $k$ 组、移入前向分数最大的 $k$ 组，对新集合重新拟合OLS，并在本轮全部交换量中保留实际损失最低的提议。交换排名并列采用稳定排序保留传入位置顺序；不同 $k$ 的损失完全相同时保留先遇到的提议。

只有实际损失改善不小于阈值时才接受交换。默认阈值为

\[
\tau_s=0.01\,\frac{s\,d\,\log\{\max(J,2)\}\,\ell_n}{n},
\qquad \ell_n=\log\{\max(\log n,e)\}.
\tag{2.12}
\]

代码在 $L_{\mathrm{old}}-L_{\mathrm{best}}<\tau_s$ 时拒绝，等于阈值不被该条件拒绝。Group-BESS重复此过程，组集合不变或达到 $m_{\max}$ 次即停止，始终保持恰好 $s$ 个组。顶层默认 $m_{\max}=15$；当未指定 $k_{\max}$ 时，每轮最多尝试 $\min(s,5)$ 组交换。正式模拟指定 $k_{\max}=1$。

## 2.4 Adaptive候选

Adaptive BCD首先在初始化支撑上拟合OLS，随后逐轴枚举当前轴规模 $s=1,\ldots,s_{m,\max}$。每个规模运行一次Group-BESS，并按模式内MSIC选择：

\[
I_m(s)=n\log\{\max(RSS_s/n,\mathrm{tiny})\}
+s\,d_m\ell_n+\ell_n\log{p_m\choose s}.
\tag{2.13}
\]

规模按递增顺序计算，只有严格较小的评分才更新，因此完全同分时保留较小规模。如果枚举规模等于当前轴支撑大小，则复用该支撑作为Group-BESS初始组；其他规模按式（2.10）的边际分数初始化。这是既有adaptive路径的一部分。

每个三轴循环结束后重新拟合完整矩形，并记录

\[
I_{\mathrm{ad}}=n\log\{\max(RSS/n,\mathrm{tiny})\}
+q\ell_n+\ell_n\sum_{m=1}^3\log{p_m\choose s_m}.
\tag{2.14}
\]

当三个支撑均与上一循环相同、相邻两次式（2.14）的绝对差小于 $\epsilon=10^{-8}$，或达到 `max_cycles` 时停止。本阶段进入最终候选库的是终态，全部adaptive中间状态并不自动入库；下文固定规模搜索的best-visited保证不延伸为adaptive全历史最优保证。

## 2.5 完整整数规模与Cold固定规模搜索

正式v5.2在旧caps和容量预算内构造

\[
\mathcal G_n=\{(s_1,s_2,s_3)\in\mathbb N_+^3:
s_m\le s_{m,\max},\ s_1s_2s_3\le Q_n\}.
\tag{2.15}
\]

所有整数三元组按 $s_1$ 外层、$s_2$ 中层、$s_3$ 内层递增访问。该步骤补全规模覆盖，不枚举 $\prod_m{p_m\choose s_m}$ 个位置组合，也不扩大训练数据决定的caps。

对每个规模，先在各全轴按式（2.4）取前 $s_m$ 个位置并进行矩形OLS，再逐轴运行固定组数的Group-BESS。当前轴子问题不传入该轴当前支撑作为初始组，保留Cold探索。每个模式提议后立即重拟合完整矩形，得到同一时刻的 $M_t=(B_t,S_{1t},S_{2t},S_{3t},L_t)$。探索始终从最新提议继续，因此完整轨迹的损失可以上升。

返回状态独立于探索状态保存。初始状态也参与比较，只有 $L_t<L_{\mathrm{best}}$ 才同时复制系数、三个支撑和损失；完全并列保留先访问者。因而该次有限搜索返回

\[
t_*=\operatorname{first\,argmin}_{0\le t\le T}L_t,
\qquad M_{\mathrm{out}}=M_{t_*}.
\tag{2.16}
\]

另外，接受历史只在

\[
L_t<L_{\mathrm{acc}}-64\epsilon_{\mathrm{mach}}
\max\{1,|L_{\mathrm{acc}}|,|L_t|\}
\tag{2.17}
\]

时追加记录。式（2.17）控制历史记录，不阻断探索，也不替代式（2.16）的严格比较。每个三轴循环结束后，若完整支撑等于本轮开始时则停止；若等于更早的轮末支撑，则以检测到探索循环停止；否则继续至 `max_cycles`。固定规模程序还检查输入有限性、支撑数量、唯一性、范围以及OLS状态有限性，违反约束时抛出错误。

## 2.6 最终MSIC与确定性容忍带

有序候选库先放adaptive终态，再放式（2.15）中各规模的固定搜索返回模型，不按规模或支撑去重。全部候选统一使用原最终MSIC重评分：

\[
I(M)=n\log\{\max(RSS/n,\mathrm{tiny})\}
+1.5q\ell_n+\ell_n\sum_{m=1}^3\log{p_m\choose s_m},
\qquad \ell_n=\log\{\max(\log n,e)\}.
\tag{2.18}
\]

其中 `tiny` 为浮点类型的最小正规正数，组合数对数用 `gammaln` 计算。式（2.13）和式（2.14）的容量惩罚系数为1，式（2.18）为1.5，三者不能互换。

令 $I_{\min}$ 为候选库最低评分，$q_{\min}$ 为按库内顺序首次取得该评分的候选容量，定义

\[
\delta=\sqrt{\max(q_{\min},1)\max\{\log(\max(n,2)),1\}},
\qquad
\mathcal E=\{M:I(M)\le I_{\min}+\delta\}.
\tag{2.19}
\]

最终在 $\mathcal E$ 中按

\[
\big(q,\ I(M),\ s_1+s_2+s_3\big)
\tag{2.20}
\]

字典序选最小者，完全并列保留原候选顺序。容忍带倍率固定为1；它是确定公式，未由CV估计标准误，因此不应解释为实测的“一倍标准误规则”。最终输出可以不同于式（2.18）的单纯最小评分者；所选系数仍为对应完整矩形的OLS，不进行测试集重选或事后收缩。

## 2.7 伪代码、停止规则与计算量

```text
算法1  Tensor-ABESS v5.2-search-final + 原MSIC
输入：当前训练数据X,y；初始化规则；c；C=max_cycles；M=m_max；k_max
1  计算三轴边际分数、初始化候选C1,C2,C3和临时支撑。
2  按轴1→2→3做有限步Group Lasso；空选取最高边际位置。
3  由初始化规模计算caps并按Qn压缩；保留原初始化支撑。
4  A ← AdaptiveBCD：
     初始矩形OLS；至多C轮，每轮按轴1→2→3：
       枚举当前轴规模1..cap，运行GroupBESS.Fix并按模式内MSIC选组。
     每轮完整OLS；支撑不变或全局adaptive评分变化<eps时停止。
5  pool ← [A的终态]
6  对每个合法整数规模(s1,s2,s3)，按字典顺序：
     current ← 全轴边际前s1,s2,s3位置上的完整OLS
     best ← current；seen ← {当前完整支撑}
     至多C轮，按轴1→2→3：
       用全部当前轴位置、当前其他轴支撑构建设计。
       proposal ← GroupBESS.Fix(sm)，不传当前轴初始组。
       current ← 替换当前轴支撑后重新拟合的完整OLS。
       若current损失严格更小，同步更新best整套状态。
       仅按机器容差另记接受历史；探索继续使用current。
     轮末支撑不变或已见时停止。
     pool追加best。
7  对pool全部模型计算最终MSIC(alpha=1.5,gamma=1)。
8  由首个最低评分模型的容量计算delta；按(q,I,规模和)选带内最小者。
输出：所选B、三轴支撑、规模、RSS、loss和最终评分。

子程序 GroupBESS.Fix(s)
  指定初始组，或由组Gram标准化边际分数取s组。
  至多M轮：OLS并计算一次前向/后向分数；
    尝试k=1..min(k_max,s,J-s)的整组交换，逐个OLS；
    保留实际损失最低的提议；改善<tau_s时不换组；
    组集合不变时停止。
  返回最后组集合及其OLS。
```

顶层参数 `tau` 仅传给adaptive子问题。最终网格中的固定BCD没有接收该参数，使用式（2.12）的内部默认阈值；顶层 `eps` 仅控制adaptive相邻评分变化停止。两者都不能被解释为控制全部搜索阶段的统一容差。固定BCD的best状态使用严格损失比较，其接受历史另用式（2.17）。

设 $G=|\mathcal G_n|$，循环预算为 $C$。最终阶段执行 $G$ 次固定BCD，至多调用 $3CG$ 次Group-BESS；adaptive阶段至多调用 $C\sum_m s_{m,\max}$ 次Group-BESS。每次Group-BESS至多 $M$ 轮，每轮至多 $\min(k_{\max},s,J-s)$ 个交换提议及相应OLS。这些是工作量上限计数，不是迭代必达次数。标准caps已经压缩到乘积不超过 $Q_n$，此时 $G=\prod_m s_{m,\max}\le Q_n$。

稠密矩形OLS在 $n\ge q$ 时通常需要约 $O(nq^2)$ 运算；逐模设计还需要 $O(np_md_m)$ 存储与构造，并承担组Gram、线性求解、打分与交换排名成本。实际排名实现包含组内两两比较，不能将其一概视为线性时间。输入张量占 $O(np_1p_2p_3)$ 空间，当前实现还在候选列表中保留每个模型的完整系数张量，额外空间约为 $O(Gp_1p_2p_3)$。初始化另有谱范数计算和有限lambda路径成本。因此，补全规模网格会增加计算和存储，不能仅凭一次返回的 `n_cycles` 估计全流程耗时。

## 2.8 实现配置与适用范围

本文模拟的设置由实际冻结计划给出，区别于直接调用函数时的公共默认值。

| 项目 | 公共接口默认 | 正式模拟 |
|---|---|---|
| 规模分支 | `target_sparsity=None` | 同左，不向算法提供真实规模 |
| 初始化lambda选择 | CV | BIC |
| caps扩张系数 $c$ | 2 | 3 |
| 每个BCD的循环预算 $C$ | 30 | 6 |
| 单轮最大交换组数 | 未指定时 $\min(s,5)$ | 1 |
| Group-BESS轮数 $M$ | 15 | 15（沿用默认） |
| 初始化lambda数、ISTA预算、变化容差 | 20、500、$10^{-6}$ | 同左 |
| adaptive评分、最终评分 | MSIC；最终 α=1.5、γ=1 | 同左 |
| 最终容忍带倍率、adaptive停止阈值 | 1、$10^{-8}$ | 同左 |

公开接口的指定规模分支与旧SIC分支具有不同流程：指定真实规模时不执行最终全网格及容忍带，`adaptive_criterion="sic"` 也不执行上述MSIC精修。本章及正式模拟讨论的是表中的自适应原MSIC路径。正式初始化的有限步数属于该算法本身，任务卡18对独立Lasso和Group Lasso参考对照要求的优化证书，不会反向改写该初始化。

该方法适用于三个轴具有稳定含义、矩形稀疏是可解释近似的回归问题。caps覆盖、有限位置搜索和选阶规则分别限制其可恢复范围；当前实现未提供全局最优、统计一致性、最优收敛率或任意响应缩放等价性的证明。阈值中包含绝对尺度项，因而响应缩放不能未经分析就视为不影响全部搜索决策。算法结构允许非连续支撑，也不等于现有模拟已验证所有非连续形状或相关结构。

本章只定义正式估计程序，不据此推导其优于所有比较方法。模拟章中的Vector Lasso及顺序逐模Tensor-GLasso已使用独立适配器完成全部必要补算，并保存调参、最优性及测试评价证据；其结论限于已声明的统计流程和固定lambda网格，历史未认证实现另作追溯，不与本轮主比较混用。
