# 向量 ABESS 的收敛性与相合性：证明详解

> 我们手上的 PNAS 正文（Zhu, Wen, Zhu, Zhang, Wang, 2020）给出了全部定理与条件，以及简短的证明思路，但**完整的技术证明在原文的附录中，未包含在所提供的 PDF 内**。因此本文分两类处理：
> - **可以自洽严格证明**的结论——牺牲量的精确表达（式 3–4）、定理一（有限步终止）、下降引理、复杂度记账（定理二）、以及亚高斯集中引理——本文给出**完整、可逐步核验**的证明。
> - **依赖 SI Appendix的技术引理**的统计结论——引理一、定理三、定理四、推论一——本文给出**严格的证明架构**：精确陈述每个辅助引理，证明其中自洽的部分，写出全部关键不等式，并明确标注亚高斯集中与 SRC 在哪一步进入。凡涉及只在 SI 中逐一追踪的绝对常数，本文以通用常数 $c,C,\kappa,K$ 表示，不臆造其具体数值。
>
> 这样既保证了严谨，又对推广到张量情形（把每处"变量"换成"组"、把标量牺牲量换成 §我们规格文档 §4 的组版本）给出可对照的模板。

---

## 0. 记号与假设回顾

**模型.** $y=X\beta^*+\epsilon$，$X\in\mathbb{R}^{n\times p}$，$\beta^*\in\mathbb{R}^p$，真活跃集 $A^*=\operatorname{supp}(\beta^*)$，$s^*=|A^*|$，最小信号强度 $b^*=\min_{j\in A^*}(\beta_j^*)^2$。不失一般性设各列 $\sqrt n$-标准化：$X_j^\top X_j=n$。损失 $L_n(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert_2^2$。

**记号.** 对活跃集 $A\subseteq[p]$、$I=A^c$：$X_A$ 为对应列子阵；$P_A=X_A(X_A^\top X_A)^{-1}X_A^\top$ 为投影到 $\operatorname{col}(X_A)$ 的正交投影阵；$G=X^\top X/n$ 为 Gram 阵。

**条件（沿用正文编号）.**
1. $\epsilon_1,\dots,\epsilon_n$ 独立同分布、均值零、亚高斯：$\exists\,\sigma>0$，$\mathbb{P}(|\epsilon_i|\ge t)\le 2e^{-t^2/\sigma^2}$，$\forall t\ge0$。
2. $X\sim\mathrm{SRC}\{2s,c_-(2s),c_+(2s)\}$：对任意 $|A|\le 2s$、$u\ne0$，$c_-(2s)\le\frac{\lVert X_A u\rVert_2^2}{n\lVert u\rVert_2^2}\le c_+(2s)$。
3. $\delta_s<1$，其中 $\delta_s$ 由 $c_\pm,\theta_{s,s}$ 定义（正文式 5）；$\theta_{a,b}$ 为稀疏正交常数：$\theta_{a,b}\ge\lVert X_A^\top X_B u\rVert_2/(n\lVert u\rVert_2)$，$\forall|A|\le a,|B|\le b,A\cap B=\varnothing$。
4. 阈值 $\tau_s=O\!\big(\frac{s\log p\log\log n}{n}\big)$。
5. $\frac{s^*\log p}{n}=o(1)$。
6. $\frac{1}{b^*}=o\!\big(\frac{n}{s\log p\log\log n}\big)$（信号强于阈值/噪声）。
7. $\frac{s^*\log p\log\log n}{n}=o(1)$ 且 $\frac{s_{max}\log p}{n}=o(1)$。

SRC（条件 2）保证了任意 $\le 2s$ 稀疏子阵的特征值被夹在 $[c_-,c_+]$ 之间，因而这些子阵可逆、且投影/最小二乘稳定；$\theta_{a,b}$ 控制不相交小子集之间的相关，是所有"对角项主导、非对角项可控"论证的核心工具。

---

## 1. 预备引理：最小二乘正交性与两个牺牲量的精确表达

本节结论**完全严格**，是后续一切分析的地基。

### 1.1 受限最小二乘与残差正交性

给定 $A$，$\hat\beta^A=\arg\min_{\beta_I=0}L_n(\beta)$，即 $\hat\beta_A=(X_A^\top X_A)^{-1}X_A^\top y$、$\hat\beta_I=0$，残差 $\hat r=y-X_A\hat\beta_A=(I-P_A)y$。

**引理 1.1（正交性）.** $X_A^\top\hat r=0$；等价地 $X_j^\top\hat r=0,\ \forall j\in A$。

*证明.* 目标 $\frac{1}{2n}\lVert y-X_A\beta_A\rVert^2$ 对 $\beta_A$ 的梯度为 $-\frac1n X_A^\top(y-X_A\hat\beta_A)=-\frac1n X_A^\top\hat r$。极小点处梯度为零，故 $X_A^\top\hat r=0$。$\square$

正交性是"最小二乘残差落在 $\operatorname{col}(X_A)$ 的正交补里"的代数表述，下面两个牺牲量的闭式都靠它把交叉项消掉。

### 1.2 向后牺牲量（式 3）

**引理 1.2.** 对 $j\in A$，把 $\hat\beta^A$ 第 $j$ 分量置零（其余不动）所增损失为

$$
\xi_j:=L_n\big(\hat\beta^A-\hat\beta_j e_j\big)-L_n(\hat\beta^A)
=\frac{X_j^\top X_j}{2n}\hat\beta_j^2
\;\overset{X_j^\top X_j=n}{=}\;\tfrac12\hat\beta_j^2 .
$$

*证明.* 置零后残差为 $\hat r+X_j\hat\beta_j$，故

$$
2n\,\xi_j=\lVert\hat r+X_j\hat\beta_j\rVert^2-\lVert\hat r\rVert^2
=2\hat\beta_j\underbrace{X_j^\top\hat r}_{=0\ (j\in A)}+\hat\beta_j^2X_j^\top X_j
=\hat\beta_j^2X_j^\top X_j.
$$

交叉项由引理 1.1 消去，得证。$\square$

### 1.3 向前牺牲量（式 4）

记 $\hat d_j=X_j^\top\hat r/n$（对 $j\in I$）。

**引理 1.3.** 对 $j\in I$，在当前拟合上仅给坐标 $j$ 添最优系数 $\hat t_j$（其余不动）所减损失为

$$
\zeta_j:=L_n(\hat\beta^A)-L_n\big(\hat\beta^A+\hat t_j e_j\big)
=\frac{(X_j^\top\hat r)^2}{2n\,X_j^\top X_j}
=\frac{X_j^\top X_j}{2n}\Big(\frac{\hat d_j}{X_j^\top X_j/n}\Big)^2
\;\overset{X_j^\top X_j=n}{=}\;\tfrac12\hat d_j^2 ,
$$

其中 $\hat t_j=\dfrac{X_j^\top\hat r}{X_j^\top X_j}$。

*证明.* $\hat t_j=\arg\min_t\frac{1}{2n}\lVert\hat r-X_jt\rVert^2$，令导数为零得 $\hat t_j=X_j^\top\hat r/(X_j^\top X_j)$。代回：

$$
2n\,\zeta_j=\lVert\hat r\rVert^2-\lVert\hat r-X_j\hat t_j\rVert^2
=2\hat t_jX_j^\top\hat r-\hat t_j^2X_j^\top X_j
=\frac{(X_j^\top\hat r)^2}{X_j^\top X_j},
$$

代入 $\hat t_j$ 即得。$\square$

**注（两量不可直接比较）.** $\xi_j$ 建于已拟合系数 $\hat\beta_j$，$\zeta_j$ 建于残差相关 $\hat d_j$：活跃变量已把信号吸入拟合，非活跃变量的信号仍在残差里。二者度量对象不同，故只在各自集合内排序，Splicing 据此"末位淘汰 + 择优录取"。

---

## 2. 定理一：算法有限步终止

**定理一.** 算法 1（BESS.Fix($s$)）在有限次迭代内终止。

*证明（完全严格）.* 记第 $m$ 次迭代活跃集 $A^m$、系数 $\beta^m=\hat\beta^{A^m}$。算法 2（Splicing）的接受规则是：仅当存在拼接使损失下降不小于阈值 $\tau_s$ 时才更新，否则返回原集。于是恰有两种情形：

- 若 $A^{m+1}\ne A^m$，则该步被接受，必有 $L_n(\beta^{m+1})\le L_n(\beta^m)-\tau_s$；
- 若 $A^{m+1}=A^m$，算法停止（算法 1 的终止判据）。

因此在停止前，$\{L_n(\beta^m)\}$ 严格递减且每步至少降 $\tau_s>0$。又 $L_n\ge0$ 有下界，故迭代次数

$$
M\le\frac{L_n(\beta^0)}{\tau_s}\le\frac{\lVert y\rVert_2^2}{2n\,\tau_s}<\infty .
$$

（另一独立论证：损失严格递减 $\Rightarrow$ 各 $A^m$ 互不相同，而大小为 $s$ 的子集仅 $\binom{p}{s}$ 个，故 $M\le\binom{p}{s}$。）$\square$

**下降引理（后续要用）.** 由上，沿算法 1 有 $L_n(\beta^{m+1})\le L_n(\beta^m)$，且非终止步为严格不等。$\square$

---

## 3. 定理二：多项式复杂度（记账 + 迭代次数）

**定理二.** 在条件下，ABESS（对 $s=1,\dots,s_{max}$ 各跑 BESS.Fix）的计算复杂度为 $n,p,s_{max}$ 的多项式（正文给出显式界）。

*证明架构.* 分"单次 Splicing 代价"×"迭代次数"×"$s$ 的扫描数"。

**(a) 单次 Splicing 代价.** 在算法 2 中：
- 计算全部 $\hat d=X^\top\hat r/n$（进而 $\zeta_j,j\in I$）：一次矩阵-向量乘，$O(np)$；$\xi_j,j\in A$ 由已有 $\hat\beta$ 得，$O(s)$。
- 对每个 $k=1,\dots,k_{max}$：选出 $\mathcal A_k,\mathcal I_k$（部分排序 $O(p)$），在 $\tilde A_k$（大小 $s$）上解最小二乘（正规方程 $O(ns^2+s^3)$）并重算损失 $O(ns)$。

故单次 Splicing 为 $O\!\big(np+k_{max}(ns^2+s^3)\big)$。

**(b) 迭代次数.** 两段控制：
- $s\ge s^*$ 时，由**定理三**，一旦 $m\ge\log_{1/\delta_s}\!\big(\lVert y\rVert_2^2/C_{s,1}\big)$ 即有 $A^m\supseteq A^*$，此后损失几何收缩，故有效迭代数为 $O\!\big(\log\frac{\lVert y\rVert_2^2}{s\log p\log\log n}\big)$（对数级）。
- $s<s^*$ 时，用阈值 $\tau_s$ 排除无用拼接：由定理一的 $M\le L_n(\beta^0)/\tau_s$ 与 $\tau_s=O(s\log p\log\log n/n)$，得 $M=O\!\big(\frac{n\lVert y\rVert_2^2}{s\log p\log\log n}\big)$。

**(c) 合并.** BESS.Fix($s$) 代价 = (b)×(a)；ABESS 再对 $s=1,\dots,s_{max}$ 求和（乘 $s_{max}$ 量级）。整理即得正文形式

$$
O\!\left(\Big(s_{max}\log\tfrac{\lVert y\rVert_2^2}{\log p\log\log n}+\tfrac{n\lVert y\rVert_2^2}{\log p\log\log n}\Big)\big(nps_{max}+ns_{max}^2+k_{max}ps_{max}\big)\right),
$$

关于 $n,p,s_{max}$ 是多项式。其中"对数级迭代"这一关键事实来自定理三（下节）。$\square$

---

## 4. 概率工具：亚高斯集中

本节结论**完全严格**，是统计部分唯一的随机性来源。

**引理 4.1（噪声水平）.** 设条件 1 成立、$X_j^\top X_j=n$。记 $u_j:=X_j^\top\epsilon/n$。则存在绝对常数 $c_0>0$ 使

$$
\mathbb{P}\big(|u_j|\ge t\big)\le 2\exp\!\big(-c_0\,n t^2/\sigma^2\big),
\qquad
\mathbb{P}\Big(\max_{1\le j\le p}|u_j|\ge t\Big)\le 2p\exp\!\big(-c_0\,n t^2/\sigma^2\big).
$$

特别地，取 $t=\sigma\sqrt{\tfrac{2}{c_0}\cdot\tfrac{\log p+u}{n}}$，则 $\max_j|u_j|\le t$ 以概率 $\ge 1-2e^{-u}$ 成立，即 $\max_j|u_j|=O_p\!\big(\sigma\sqrt{\log p/n}\big)$。

*证明.* 条件 1 的尾部界等价于（相差绝对常数的）MGF 界 $\mathbb{E}e^{\lambda\epsilon_i}\le e^{\lambda^2\sigma^2/(4c_0)}$。$u_j=\frac1n\sum_i X_{ij}\epsilon_i$ 是独立亚高斯之加权和，其亚高斯参数为 $\frac{\sigma^2}{4c_0}\cdot\frac{\sum_iX_{ij}^2}{n^2}=\frac{\sigma^2}{4c_0}\cdot\frac1n$（用 $\sum_iX_{ij}^2=X_j^\top X_j=n$）。故 $\mathbb{P}(|u_j|\ge t)\le 2\exp(-c_0nt^2/\sigma^2)$。对 $p$ 个变量取并集界得第二式。取定 $t$ 使 $2p\,e^{-c_0nt^2/\sigma^2}=2e^{-u}$ 即证。$\square$

**推论 4.2（子集形式）.** 对固定的 $|B|\le 2s$，由 SRC（条件 2）与引理 4.1 的向量版（对 $X_B^\top\epsilon/n$ 用亚高斯与并集界），

$$
\big\lVert X_B^\top\epsilon/n\big\rVert_2=O_p\!\Big(\sigma\sqrt{\tfrac{s\log p}{n}}\Big).
$$

这是所有"噪声投影"项的统一上界，量级 $\sqrt{s\log p/n}$。

---

## 5. 引理一：活跃集包含真集

**引理一.** 设 $(\hat\beta,\hat A)$ 是算法 1 在给定 $s\ge s^*$ 下的**稳定解**（再无拼接能使损失下降超过 $\tau_s$），且条件 1–4 成立。则

$$
\mathbb{P}\big(\hat A\supseteq A^*\big)\ge 1-\gamma_s(n,p),
\qquad
\gamma_s(n,p)=O\!\big(e^{\log p-Ksnb^*/s}\big)+O\!\big(e^{\log p-Ksn/s^*}\big).
$$

进一步在条件 5、6 下 $\lim_n\mathbb{P}(\hat A\supseteq A^*)=1$；若 $s=s^*$ 则 $\lim_n\mathbb{P}(\hat A=A^*)=1$。

*证明架构（严格骨架 + 关键不等式）.* 反证：设事件 $\{\hat A\not\supseteq A^*\}$ 发生，则存在真变量 $j^*\in A^*\setminus\hat A$。分三步导出与"稳定"矛盾。

**第 1 步：真变量的向前牺牲量大.** 由引理 1.3，$\zeta_{j^*}=\frac12(\,X_{j^*}^\top\hat r/n\,)^2\cdot n$（标准化下 $=\frac12\hat d_{j^*}^2$）。将 $\hat r=(I-P_{\hat A})y=(I-P_{\hat A})(X\beta^*+\epsilon)$ 代入：

$$
X_{j^*}^\top\hat r
=\underbrace{X_{j^*}^\top(I-P_{\hat A})X_{j^*}\,\beta^*_{j^*}}_{\text{对角(自)项}}
+\underbrace{\sum_{k\in A^*\setminus\{j^*\}}X_{j^*}^\top(I-P_{\hat A})X_k\,\beta^*_k}_{\text{非对角(溢出)项}}
+\underbrace{X_{j^*}^\top(I-P_{\hat A})\epsilon}_{\text{噪声项}} .
$$

- *对角项下界.* $X_{j^*}^\top(I-P_{\hat A})X_{j^*}=\lVert(I-P_{\hat A})X_{j^*}\rVert^2=n-\lVert P_{\hat A}X_{j^*}\rVert^2$。因 $j^*\notin\hat A$，$X_{j^*}$ 与 $X_{\hat A}$ 是不相交小子集，由 SRC 与 $\theta_{s,1}$ 有 $\lVert P_{\hat A}X_{j^*}\rVert^2\le \kappa_1 n$，其中 $\kappa_1=\kappa_1(c_-,c_+,\theta)<1$。故对角项 $\ge (1-\kappa_1)n$，于是其对 $\hat d_{j^*}$ 的贡献 $\ge(1-\kappa_1)|\beta^*_{j^*}|\ge(1-\kappa_1)\sqrt{b^*}$。
- *溢出项.* 由稀疏正交常数，$\big|\sum_{k}X_{j^*}^\top(I-P_{\hat A})X_k\beta_k^*\big|/n\le \theta_{s,s^*}\lVert\beta^*_{A^*}\rVert$ 量级，且 $I-P_{\hat A}$ 只会缩小相关，故受 $\theta$ 控制。
- *噪声项.* $|X_{j^*}^\top(I-P_{\hat A})\epsilon|/n=O_p(\sigma\sqrt{\log p/n})$（引理 4.1 + 投影不增范数）。

在条件 6（$b^*$ 强于噪声/阈值）下，对角项主导，得**高概率下界**

$$
\zeta_{j^*}=\tfrac12\hat d_{j^*}^2\ge \kappa_2\,b^*
\quad\text{对某常数 }\kappa_2>0 .
\tag{5.1}
$$

**第 2 步：活跃集中存在牺牲量小的变量.** 因 $|\hat A|=s\ge s^*$，若 $\hat A\not\supseteq A^*$，则 $\hat A$ 至少有一个"假变量" $j\in\hat A\setminus A^*$（否则 $\hat A\subseteq A^*$ 且 $|\hat A|=s\ge s^*=|A^*|$ 迫使 $\hat A=A^*\supseteq A^*$）。假变量的拟合系数只在解释噪声与溢出，其向后牺牲量 $\xi_j=\frac12\hat\beta_j^2$ 受噪声界控制：由推论 4.2，

$$
\min_{j\in\hat A}\xi_j\le \xi_{j'}=O_p\!\big(\sigma^2\,\tfrac{s\log p}{n}\big)
\quad\text{对某 }j'\in\hat A\setminus A^* .
\tag{5.2}
$$

**第 3 步：一次拼接严格改进，矛盾.** 对 $k=1$ 的拼接（换出 $\xi$ 最小的 $j'$、换入 $\zeta$ 最大的 $j^*$），标准的 SRC 型一步分析给出损失下降的下界

$$
L_n(\hat\beta^{\hat A})-L_n(\hat\beta^{\tilde A})\ \ge\ \kappa_3\big(\zeta_{j^*}-\xi_{j'}\big)
\ \overset{(5.1),(5.2)}{\ge}\ \kappa_3\big(\kappa_2 b^*-O_p(\sigma^2\tfrac{s\log p}{n})\big).
$$

由条件 6，$b^*$ 的阶高于 $\frac{s\log p\log\log n}{n}\ge\tau_s$，故上式高概率 $>\tau_s$。这说明存在使损失下降超过阈值的拼接，与 $\hat A$ **稳定**矛盾。因此 $\{\hat A\not\supseteq A^*\}$ 只能以小概率发生。

**第 4 步：概率界.** 上述"高概率"由两类事件构成：噪声投影超标（引理 4.1/推论 4.2 给 $O(e^{\log p-\text{const}\cdot n b^*/\ldots})$）与信号-噪声可分性失败（给 $O(e^{\log p-\text{const}\cdot n/s^*})$）。两者并集即 $\gamma_s(n,p)$ 的两项。条件 5、6 使 $\gamma_s(n,p)\to0$；$s=s^*$ 时 $\hat A\supseteq A^*$ 与 $|\hat A|=s^*=|A^*|$ 迫使 $\hat A=A^*$。$\square$

> **进入张量情形的对照.** 第 1 步的"对角项主导"要换成"给定其余两模态支撑时，第三模态真切片的组向前牺牲量 $\zeta_g=\frac12\hat d_g^\top\Gamma_g^{-1}\hat d_g$ 大"，SRC 换成张量版 SRC，$\theta$ 换成组间稀疏正交常数——逻辑同构。

---

## 6. 定理三：几何收敛速率与迭代次数

**定理三.** 设 $s\ge s^*$、条件 1–4 成立，$(\beta^m,A^m)$ 为算法 1 第 $m$ 次迭代。则以概率 $1-\gamma_s(n,p)$：

1. 若 $m\ge\log_{1/\delta_s}\!\big(\lVert y\rVert_2^2/C_{s,1}\big)$，则 $A^m\supseteq A^*$，其中 $C_{s,1}=n(1-\Delta)\big(c_-(s)-\theta_{s,s}^2/c_-(s)\big)^2 b^*$；
2. $\lVert\beta^m-\beta^*\rVert_2^2\le C_{s,2}\,\delta_s^{\,m}\lVert y\rVert_2^2$，其中 $C_{s,2}=\dfrac{1+\Delta+\theta_{s,s}/c_-(s)}{(1-\Delta)n\big(c_-(s)-\theta_{s,s}^2/c_-(s)\big)}$。

*证明架构.* 核心是**一步收缩不等式**：在 SRC 下，一次 Splicing 使估计误差按因子 $\delta_s$ 收缩：

$$
\lVert\beta^{m+1}-\beta^*\rVert_2^2\ \le\ \delta_s\,\lVert\beta^{m}-\beta^*\rVert_2^2 .
\tag{6.1}
$$

(6.1) 的证明思路：把误差分解为"活跃集不匹配部分"与"系数偏差部分"。Splicing 交换后，被换入的高 $\zeta$ 变量把残差中的真信号成分吸收，被换出的低 $\xi$ 变量只损失噪声量级；用 SRC 把这些量翻译成 $\beta$-误差的收缩，收缩因子恰为正文式 (5) 定义的

$$
\delta_s=\frac{8c_+(s)\big((1+\Delta)\frac{\theta_{s,s}}{c_-(s)}(1+\frac{\theta_{s,s}}{c_-(s)})\big)^2}{(1-\Delta)\big(c_-(s)-\frac{\theta_{s,s}^2}{c_-(s)}\big)} ,
$$

条件 3 保证 $\delta_s<1$（余项 2 给出充分条件 $c_-(2s)\ge0.8123,c_+(2s)\le1.1877$）。反复用 (6.1) 得结论 2 的几何界 $\lVert\beta^m-\beta^*\rVert^2\le\delta_s^m\lVert\beta^0-\beta^*\rVert^2\le C_{s,2}\delta_s^m\lVert y\rVert^2$（末步用 SRC 把初值误差换成 $\lVert y\rVert^2$）。结论 1 则是：当误差收缩到小于最小信号能量 $C_{s,1}$（正比于 $nb^*$）时，真变量不可能再被排除在活跃集外，故 $A^m\supseteq A^*$；令 $\delta_s^m\lVert y\rVert^2\le C_{s,1}$ 解出 $m\ge\log_{1/\delta_s}(\lVert y\rVert^2/C_{s,1})$。$\square$

**推论（定理二用到的对数迭代）.** 结论 1 表明 $s\ge s^*$ 时达到真集包含只需 $O(\log\lVert y\rVert^2)$ 步，且由 (6.1) 之后误差几何衰减，故算法 1 的有效迭代数为对数级——这正是 §3(b) 所引用的关键事实。

---

## 7. 定理四：SIC 的选择相合性

$$
\mathrm{SIC}(A)=n\log\hat{\mathcal L}_A+|A|\log p\log\log n,
\qquad \hat{\mathcal L}_A=\min_{\beta_{A^c}=0}L_n(\beta)=\tfrac{1}{2n}\lVert(I-P_A)y\rVert^2 .
$$

**定理四.** 设条件 1、4、7 成立，条件 2、3、6 对 $s_{max}$ 成立。则存在 $\alpha>0$，当 $n$ 充分大时，以概率 $1-O(p^{-\alpha})$，ABESS 经 SIC 选出的模型 $\hat A_{s_{min}}=A^*$。

*证明架构.* ABESS 对每个 $s$ 用 BESS.Fix 得 $\hat A_s$（由引理一，$s\ge s^*$ 时高概率 $\hat A_s\supseteq A^*$；$s=s^*$ 时 $\hat A_{s^*}=A^*$），再取 $s_{min}=\arg\min_s\mathrm{SIC}(\hat A_s)$。只需证 $s_{min}=s^*$ 高概率成立，分**欠拟合**与**过拟合**两向压制。

记基准 $\mathrm{SIC}(A^*)=n\log\hat{\mathcal L}_{A^*}+s^*\log p\log\log n$。

**(a) 欠拟合方向（$s<s^*$，或 $A\not\supseteq A^*$）.** 遗漏真信号使残差含未解释的信号能量。用 SRC，

$$
\hat{\mathcal L}_A-\hat{\mathcal L}_{A^*}\ \ge\ \kappa_4\!\!\sum_{k\in A^*\setminus A}\!(\beta_k^*)^2\ \ge\ \kappa_4\,b^*\quad(\text{至少漏一个真变量}).
$$

于是拟合项之差 $n\log\hat{\mathcal L}_A-n\log\hat{\mathcal L}_{A^*}\gtrsim n\cdot\frac{\kappa_4 b^*}{\hat{\mathcal L}_{A^*}}$。由条件 6，$nb^*$ 的阶远高于复杂度惩罚差 $|s-s^*|\log p\log\log n$。故 $\mathrm{SIC}(A)>\mathrm{SIC}(A^*)$，欠拟合被拟合项否决。

**(b) 过拟合方向（$s>s^*$，$A\supsetneq A^*$）.** 多纳入的假变量只能把残差再降"噪声量级"。对每个多余变量，由引理 4.1 的极值统计，拟合项减少 $\le \kappa_5\sigma^2\log p$（对 $n\log\hat{\mathcal L}$ 而言，$n\log\hat{\mathcal L}_A-n\log\hat{\mathcal L}_{A^*}\ge-\kappa_5(|A|-s^*)\log p$，高概率）；而复杂度惩罚**增加** $(|A|-s^*)\log p\log\log n$。二者相比：

$$
\mathrm{SIC}(A)-\mathrm{SIC}(A^*)\ \ge\ (|A|-s^*)\log p\,\big(\log\log n-\kappa_5\big)\ >\ 0
\quad(\text{当 }n\text{ 大}).
$$

**慢发散因子 $\log\log n\to\infty$ 是关键**：它保证复杂度惩罚最终压过噪声带来的拟合改进，从而杜绝过拟合。（这与 BIC 用 $\log n$、EBIC 额外用 $\log\binom{p}{|A|}$ 压制大模型空间同源；SIC 用 $\log p\log\log n$ 是超高维校准。）

**(c) 合并.** (a)(b) 表明 $s^*$ 是 SIC 的唯一高概率极小点，故 $s_{min}=s^*$，再由 $\hat A_{s^*}=A^*$（引理一，$s=s^*$）得 $\hat A_{s_{min}}=A^*$。所有失败事件的并集概率由引理 4.1 的并集界给出 $O(p^{-\alpha})$。$\square$

> **张量对照.** 把 $|A|\log p$ 换成模型空间对数 $\sum_m\log\binom{p_m}{s_m}$，(b) 中"每多一个变量惩罚增 $\log p\log\log n$"换成"每在某模态多选一个切片，该模态惩罚增 $\log\log n\cdot[\log\binom{p_m}{s_m+1}-\log\binom{p_m}{s_m}]$"，可加结构使三方向可分别压制——见我们规格文档 §6。

---

## 8. 推论一：oracle 性质与渐近分布

**推论一.** 在定理四的假设下，以概率 $1-O(p^{-\alpha})$，ABESS 的解 $\hat\beta_{s_{min}}$ 等于 oracle 最小二乘估计：

$$
\mathbb{P}\{\hat\beta_{s_{min}}=\hat\beta^{\,o}\}=1-O(p^{-\alpha}),
\qquad \hat\beta^{\,o}_{A^*}=(X_{A^*}^\top X_{A^*})^{-1}X_{A^*}^\top y ,
$$

且 $\hat\beta_{A^*}\sim N\!\big(\beta^*_{A^*},\,\Sigma\big)$，$\Sigma=\sigma^2(X_{A^*}^\top X_{A^*})^{-1}$（正文写 $\Sigma=(X_{A^*}^\top X_{A^*})^{-1}$，即取 $\sigma^2=1$ 或按 $\sigma^2$ 缩放）。

*证明.* 由定理四，高概率 $\hat A_{s_{min}}=A^*$。给定活跃集为 $A^*$，算法输出即该支撑上的最小二乘解，按定义就是 oracle 估计 $\hat\beta^o$，故第一式成立。其分布：$\hat\beta^o_{A^*}=(X_{A^*}^\top X_{A^*})^{-1}X_{A^*}^\top(X_{A^*}\beta^*_{A^*}+\epsilon)=\beta^*_{A^*}+(X_{A^*}^\top X_{A^*})^{-1}X_{A^*}^\top\epsilon$。

- 均值 $\mathbb E\hat\beta^o_{A^*}=\beta^*_{A^*}$（无偏）；
- 协方差 $\mathrm{Cov}=(X_{A^*}^\top X_{A^*})^{-1}X_{A^*}^\top(\sigma^2 I)X_{A^*}(X_{A^*}^\top X_{A^*})^{-1}=\sigma^2(X_{A^*}^\top X_{A^*})^{-1}$；
- 正态误差下 $\hat\beta^o_{A^*}$ 是 $\epsilon$ 的线性变换，故精确正态；亚高斯误差下由 CLT 得渐近正态。$\square$

**含义.** ABESS 不仅**选对**变量，其估计还与"事先知道真集再做最小二乘"（oracle）**完全一致**——这就是最优子集选择相对 LASSO/SCAD/MCP 的统计优越性来源（后者因收缩产生偏差）。

---

## 结语：证明骨架总览

$$
\underbrace{\text{引理 1.1 正交性}}_{\text{代数}}
\Rightarrow
\underbrace{\text{式 3,4 牺牲量闭式}}_{\text{代数}}
\Rightarrow
\begin{cases}
\underbrace{\text{定理一 有限终止}}_{\text{单调}+\text{有限/阈值}}\Rightarrow\underbrace{\text{定理二 多项式复杂度}}_{\text{记账}+\text{定理三迭代数}}\\[6pt]
\underbrace{\text{引理 4.1 亚高斯集中}}_{\text{概率}}+\underbrace{\text{SRC}}_{\text{条件2,3}}\Rightarrow
\underbrace{\text{引理一 含真集}}_{\text{反证}+\text{拼接改进}}\Rightarrow\underbrace{\text{定理三 几何收缩}}_{\text{一步收缩}(6.1)}\\[6pt]
\qquad\qquad\qquad\qquad\ \Rightarrow\underbrace{\text{定理四 SIC 相合}}_{\text{欠拟合}/\text{过拟合双压}}\Rightarrow\underbrace{\text{推论一 oracle}}_{\text{给定真集的 LS}}
\end{cases}
$$

三条主线：**代数线**（正交性 → 牺牲量）撑起算法；**确定性线**（单调 + 有限）给出终止与复杂度；**概率线**（亚高斯集中 + SRC）给出含真集、几何收缩、SIC 相合与 oracle。推广到张量时，每处"变量/标量牺牲量/$\binom{p}{s}$"分别换成"组/组牺牲量 $\frac12\hat d_g^\top\Gamma_g^{-1}\hat d_g$/$\prod_m\binom{p_m}{s_m}$"，骨架不变。

---

### 参考

- Zhu, Wen, Zhu, Zhang, Wang (2020). *A polynomial algorithm for best-subset selection.* PNAS 117(52): 33117–33123
- Chen & Chen (2008). *Extended Bayesian information criteria for model selection with large model spaces.* Biometrika 95: 759–771.
- Zhang & Huang (2008); Bickel, Ritov & Tsybakov (2009)（SRC / 稀疏正交常数的经典处理）。

> **诚实边界.** 本文对定理一、引理 1.1–1.3、引理 4.1、复杂度记账给出了完整可核验的证明；对引理一、定理三、定理四、推论一给出了严格的证明架构与全部关键不等式，其中 $\kappa_i,K,C_{s,\cdot}$ 等常数的逐一追踪、以及 (6.1) 一步收缩的完整代数，属 SI Appendix 的技术细节，需以原文附录为准。若你们要把这套证明**原样搬进张量论文**，建议按上面每节末的"张量对照"逐条改写，并在附录补齐组版本的 SRC 与一步收缩引理。
