# 任务卡18：理论接口与实际程序的边界

更新日期：2026-10-29。对象：正式 `v5.2-search-final + 原MSIC`。本文件核对定义与未来证明接口，不增加定理、不修改算法，也不将可执行程序的正确性核验替代统计理论。任务卡18的600输入认证比较已完成；数值证书证明的是参考对照各必要凸子问题在本批达到声明精度，不证明正式Tensor-ABESS的位置全局最优、统计一致性或泛化保证。对应方法正文：[02_method.md](../../paper/sections/02_method.md)。

## 1. 必须区分的三个层次

| 层次 | 定义 | 可以直接说什么 | 不能由此推出什么 |
|---|---|---|---|
| 理想固定规模全局估计器 | 在全部三个位置集合及其矩形内系数上全局最小化 $RSS/(2n)$，每轴集合大小固定 | 确定一个组合最小二乘估计目标；给定矩形可精确执行数值OLS | 正式位置搜索已求出这个全局估计器 |
| 有限候选库选择 | 对有序候选库统一用最终MSIC评分，再按确定性容忍带和字典序选择 | 给定候选库、顺序和分数时，选择规则可明确复算 | 选择等于单纯MSIC最小值，或候选库与数据无关 |
| 实际有限程序 | 数据决定预筛、有限ISTA初始化、caps、adaptive终态；对全部合法整数规模作Cold局部位置搜索；best-visited返回；最终容忍带选择 | 固定规模返回同一次有限探索中训练损失最低的完整状态；规模网格在旧caps内完整 | 位置搜索全局最优、泛化改善、一致性、最优率、任意缩放不变性 |

理想估计器不是实际程序的另一种记号。候选库依赖训练设计和响应，caps也是随机的训练数据函数。若理论先研究一个固定模型族或独立候选库，需要明确说明其与实际程序之间尚缺的论证。

## 2. 理想目标与自由度接口

设 $\mathcal B(S)$ 表示满足 $\operatorname{supp}(B)\subseteq S_1\times S_2\times S_3$ 的系数集合。固定规模理想估计器可记为

\[
(\widehat S^{\rm glob}(s),\widehat B^{\rm glob}(s))
\in\arg\min_{|S_m|=s_m,\ B\in\mathcal B(S)}L_n(B).
\]

存在多个最优位置或OLS解时，必须另给选择约定；实际OLS按 `lstsq(rcond=None)` 的数值秩判定和最小范数语义执行。$q=\prod_m s_m$ 是矩形容量，秩不足时设计秩可以小于 $q$，实际非零元素数也可以小于 $q$。数据选择后的自由度与这三者均不应无条件等同。

## 3. 正式候选库与精确选择算子

记 $A(D)$ 为当前训练数据 $D$ 上adaptive终态，$F_s(D)$ 为规模 $s$ 的Cold有限搜索best-visited完整模型。正式有序候选库为

\[
\mathcal C(D)=[A(D),F_{(1,1,1)}(D),\ldots,F_s(D),\ldots],
\quad s\in\mathcal G_n(D),
\]

其中规模按三轴递增循环，不按支撑或规模去重；adaptive所有历史状态并不入库。最终评分为

\[
I(M)=n\log\{\max(RSS(M)/n,\mathrm{tiny})\}
+1.5q(M)\ell_n+\ell_n\sum_{m=1}^3\log{p_m\choose s_m(M)},
\quad \ell_n=\log\{\max(\log n,e)\}.
\]

先取有序库中首个最低评分候选 $M_0$，再由 $q(M_0)$ 定义

\[
\delta=\sqrt{\max(q(M_0),1)\max\{\log(\max(n,2)),1\}}.
\]

在 $I(M)\le I(M_0)+\delta$ 的候选中，按 $(q(M),I(M),\sum_m s_m(M))$ 字典序选最小者，完全并列保留输入顺序。任何未来针对“MSIC最小化器”的结论，都需另外处理这一容忍带选择算子。参数名中的 `one_se` 不是由数据估计标准误的证明。

## 4. 未来理论必须处理的接口

| 问题 | 实现事实 | 将理论转移到正式程序前所需内容 |
|---|---|---|
| caps覆盖 | 每轴上限依赖有限Group Lasso初始化，随后按 $Q_n=\max(1,\lfloor0.7n\rfloor)$ 压缩 | 真实各轴规模被同时覆盖的条件或概率；不能只证 $q^\star\le Q_n$ |
| 初始支撑与caps | 压缩caps不裁剪初始化支撑，首次adaptive OLS可越界 | 明确哪些状态属于受限模型族、正循环预算的作用以及初始化的单独分析 |
| 位置覆盖 | 完整整数规模不枚举位置组合；分数提出交换，OLS判接受 | 各规模位置优化误差或包含真位置的条件，不能以完整规模代替全局搜索 |
| 有限Cold探索 | 模式子问题冷启动，轨迹可非单调；返回最小已访问完整模型 | 有限预算和循环停止下的搜索误差界；best-visited只约束访问集合 |
| adaptive候选 | 只放终态，并使用同规模当前支撑复用 | 不把固定BCD的保证移植到adaptive全历史 |
| MSIC系数 | 模式内与adaptive容量系数1，最终系数1.5；组合项系数均1 | 理论中的惩罚应明确对应哪一阶段、哪一个系数 |
| 容忍带 | 带宽依赖首个最低评分候选容量，带内优先小容量 | 模型区分间隔与带宽相对量级；小模型偏好对欠选/过选的影响 |
| 数据依赖选择 | 预筛、lambda网格、caps、位置及候选库使用同一训练数据 | 处理整个选择过程的概率控制，或清楚声明仅分析理想化版本 |
| 数值与缩放 | 有限ISTA、绝对阈值、变化容差、机器精度历史阈值 | 理论优化误差和浮点误差的作用；响应缩放不变性需另证 |
| OLS与风险 | 更充分搜索可以降低训练RSS | 若要论及预测风险，需额外泛化论证；不能以训练损失推断外测收益 |

本卡不研究新的cap规则、Warm搜索、FS、新MSIC或ABESS训练内CV，也不为理论方便修改正式程序。可能的理论应先声明研究的是理想估计器、给定候选库选择，还是完整的实际程序。

## 5. 代码证据入口

| 正文定义 | 正式证据 |
|---|---|
| 接口及公共默认 | `src/tensor_abess/tensor/abess.py::tensor_abess` |
| 边际分数、初始化、caps、初始支撑不裁剪 | `src/tensor_abess/utils/init.py::_mode_slice_scores`、`tensor_init_glasso` |
| 有限ISTA、BIC与CV初始化边界 | `src/tensor_abess/utils/glasso.py::fit_group_lasso_ista`、`select_group_lasso_lambda_bic`、`select_group_lasso_lambda_cv` |
| 全轴分组、列顺序、OLS | `src/tensor_abess/tensor/reduction.py`、`tensor/ops.py` |
| 分数与实际交换接受 | `src/tensor_abess/group/sacrifices.py`、`group/splicing.py`、`group/bess_fix.py` |
| 模式规模枚举及同规模支撑复用 | `src/tensor_abess/group/adaptive.py::group_bess_adaptive` |
| 完整整数规模、有序候选库、容忍带 | `src/tensor_abess/tensor/bcd.py::_candidate_size_triples`、`_refine_msic_global_candidates` |
| Cold、best完整状态、历史阈值、循环停止 | `src/tensor_abess/tensor/bcd.py::tensor_bcd` |
| adaptive终态、评分、tau作用边界 | `src/tensor_abess/tensor/bcd.py::tensor_bcd_adaptive` |
| 模式/全局MSIC系数 | `src/tensor_abess/tensor/sic.py` |
| 正式模拟参数 | `numerical_simulation/outputs_v52_search_final_full/plan.json`；`numerical_simulation/known_s_cluster/run.py::fit_tensor` |
| 正式身份及冻结 | `doc/releases/v5.2-search-final/RELEASE_MANIFEST.json`；`doc/taskcard11/FINAL_ALGORITHM_FREEZE.md` |

## 6. 写作术语与自审

| 固定写法 | 含义 |
|---|---|
| 矩形支撑/矩形容量 | 三轴集合笛卡尔积/其坐标数；不等于实际元素非零支撑 |
| 声明模式支撑 | 算法保存的轴集合 $S_m$；ABESS和顺序GLasso允许其矩形内部含零系数 |
| 投影模式支撑 | 实际元素非零集在轴 $m$ 的投影；与声明集合可能不同，评价时分列 |
| 完整整数规模 | caps与容量内全部正整数三元组；不包括全部位置组合 |
| Cold固定规模搜索 | 当前轴子问题不传当前支撑初始组；允许完整探索非单调 |
| best-visited完整模型 | 同一访问时刻的B、三个支撑和训练损失一起返回 |
| 原MSIC | 正式逐模/自适应/最终评分与原容忍带；不包括研究FS |

本章论证句为：在具有三轴矩形稀疏解释的回归中，正式Tensor-ABESS以有限逐模位置搜索和旧caps内完整整数规模候选近似估计模型，再按原MSIC容忍带选择；源码支持该程序定义，不直接支持全局或普适统计保证。正文按“模型→初始化→分组搜索→adaptive→固定规模→最终选择→实现边界”组织，各节分别承担定义、机制或限制任务。自审检查了公式归一化、三阶段惩罚系数、候选顺序、容忍带、初始越界与停止边界；未增加实验成绩、理论定理或文献方法归属。
