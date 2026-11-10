# 附录：模拟协议、指标与数值验收

日期：2026-10-26。此附录随任务卡18方法及模拟草稿交付，说明已固定的定义与证据接口；不表示600×2新对照已完成。最新机器协议为doc/taskcard18/CLOSEOUT_PROTOCOL.json，当前为protocol_version=2；后续只补入实际验收记录，不根据测试成绩改变规则。

## A. 输入、随机流与身份

原计划为numerical_simulation/outputs_v52_search_final_full/plan.json。每条任务具有标签、重复号、训练seed、test_seed和全部配置；四项原训练指纹为X_train、y_train、B_true和train_noise。完整输入身份同时绑定这些指纹和测试种子，不按单一训练设计指纹去重。700标签对应600套唯一完整输入、10配置；基准配置有150套完整输入，其余各50。原结果四方法共有2800行、56个标签×方法汇总单元。

训练seed为2026加序列偏移、100倍序列内取值顺序及重复号；非基准单独标签的四条序列偏移均10000。测试seed为训练seed加1000000，500测试点按100点一批生成。改变批大小可能改变高斯预测变量与噪声在随机流中的对应次序，因此保持原分批约定。

NPZ保存真系数、噪声、真支撑和拟合结果，但未归档原训练预测数组、原响应数组或原测试数组。重新生成时先比较可用原指纹；若跨平台浮点计算使原字节哈希不一致，不重写旧哈希。协议要求另核对保存真系数、噪声与支撑、历史Oracle训练RSS及测试指标，使用rtol=1e-10、atol=1e-10，同时将结论标为数值对应而非逐字节身份。20个固定训练诊断pilot已保存这种核验，其范围不能自动扩大至其余580套输入。

固定pilot使用canonical场景[0,1,2,4,5,7,8,10,11,13]的replicate 0、1，共20套不同完整输入、10种配置。训练/数值诊断先于新增方法的测试评价；既有Oracle测试标量仅用于预定输入对应核验，不用于选lambda、求解预算或方法方案。研究者已见过历史成绩，本轮不称完全盲法的独立确认实验。

## B. 参考对照的统计定义

Vector Lasso目标为 \(RSS/(2n)+\lambda\|\beta\|_1\)，Group Lasso目标为 \(RSS/(2n)+\lambda\sum_g\|\beta_g\|_2\)，无截距、无额外中心化、标准化或组权重。每次计算中的 \(n\) 都是当前实际拟合子集样本数，不能在内折误用完整训练样本量。Group Lasso按当前轴位置整组，组内为其他两轴支撑的笛卡尔积。

lambda网格由当前完整训练局部设计构造，20点降序，从对应对偶范数的lambda_max至其0.01倍；lambda_max为0时使用单个0。CV按现有样本顺序array_split为5折，不随机打乱；对同lambda把各折验证SSE相加。精确同分保留降序中先遇到的lambda。每折单独构建拟合状态，可在同折相邻lambda间复用，禁止跨折传递响应相关的beta、残差或活动状态。最终lambda另在全训练样本拟合，Lasso不做OLS后拟合。

Tensor-GLasso保留顺序初始化及逐模局部CV、预筛、原非空规则和最终矩形OLS。其局部设计中的其他轴支撑及预筛依赖完整训练过程，折状态隔离不能消除这种依赖。它不被称为全流程严格嵌套CV，也不被当作通用已发表Tensor Group Lasso算法的唯一实现。

两适配器与正式32源隔离。当前新method_id分别为vector_lasso_cv_isolated_v2、sequential_tensor_glasso_certified_v2。已保留的v1训练诊断现场与v2结果不能以同一身份混合。正式Tensor-ABESS及Oracle通过身份核验后复用，历史lasso、glasso则作为旧实现单列。

## C. KKT与原始—对偶间隙

令 \(g=Z^{\mathsf T}(Z\beta-y)/n\)。Lasso坐标残差为

\[
R_j=
\begin{cases}
|g_j+\lambda\operatorname{sign}(\beta_j)|,&\beta_j\ne0,\\
(|g_j|-\lambda)_+,&\beta_j=0.
\end{cases}
\]

Group Lasso组残差为

\[
R_g=
\begin{cases}
\|g_g+\lambda\beta_g/\|\beta_g\|_2\|_2,&\|\beta_g\|_2>0,\\
(\|g_g\|_2-\lambda)_+,&\|\beta_g\|_2=0.
\end{cases}
\]

记最大残差为 \(K\)。正常正lambda候选必须同时满足 \(K/\lambda\le10^{-4}\) 及下述归一化gap阈值。诊断使用实际数值非零，不用论文元素支撑阈值 \(10^{-8}\) 把小系数当零以制造通过。

令 \(r=y-Z\beta\)、\(u=r/n\)，按惩罚的对偶范数构造可行点

\[
\theta=\frac{u}{\max\{1,\Omega^*(Z^{\mathsf T}u)/\lambda\}},
\quad D(\theta)=y^{\mathsf T}\theta-\frac n2\|\theta\|_2^2.
\]

Lasso的 \(\Omega^*\) 为无穷范数，非加权Group Lasso为最大组二范数。验证对偶可行性后，原目标 \(P\) 须满足

\[
\frac{P-D(\theta)}{\max\{\|y\|_2^2/(2n),10^{-12}\}}\le10^{-6}.
\]

明显负gap表示尺度、可行性或实现有误，不一律截成零。lambda=0单独通过最小二乘及梯度残差核验，不除以lambda；零响应、零设计和秩不足另存对应边界结果。保存每个内折候选及最终拟合的目标、gap、对偶可行性、KKT、迭代数、停止原因、到限与错误，不能只检查最终选定的一次拟合。

有限预算为2000、10000、50000三档，容差保持不变，达到证书立即停止。v2训练修订使用Lasso每50次坐标扫描检查证书、库内部变化容差为0；修订发生在新增测试评价之前。训练诊断中独立求解路径只用于验证目标与拟合值，在未实际验证前不作自动备用。高维系数可能非唯一，不要求两求解器每个系数完全一致。到最高预算仍失败时保留失败，不能改称已收敛或无限加预算。

以上是本轮可靠参考对照的验收标准。正式ABESS内部500步ISTA及其变化停止规则保持冻结，不因参考对照标准而改变生产源。

## D. 支撑与系数指标

主阈值 \(\eta=10^{-8}\)，元素估计集合为 \(\widehat E_\eta=\{j:|\widehat B_j|>\eta\}\)，真集合也按相同阈值计算；本批真系数全部为非零的60个矩形元素。敏感性阈值为 \(10^{-10}\) 和 \(10^{-6}\)，不参与方法调参或选择主口径。

| 指标 | 定义或口径 |
|---|---|
| 元素Precision、Recall、FDR、F1、FPR | 对实际带阈值元素集合计算 |
| 模式Precision、Recall、FDR、F1、FPR | 对保存的每轴声明集合计算，并给三轴宏平均 |
| 实际模式投影 | 从实际元素集合沿其余两轴取任一非零，另核对声明集合是否相同 |
| Exact | 三个声明模式集合同时等于真集合；不是逐元素系数完全相等 |
| \(q_{\rm rect}\) | 三个声明集合大小的乘积 |
| 实际非零数 | 带阈值元素集合中的元素个数，不以 \(q_{\rm rect}\) 替代 |
| legacy元素指标 | 旧模式交集乘积除以矩形容量等包络指标，仅作历史解释 |
| REE | \(\|\widehat B-B^\star\|_F/\|B^\star\|_F\)，非平方比 |

设 \(a=|A|\)、\(t=|T|\)、\(h=|A\cap T|\)、总位置数为 \(P\)。Precision=\(h/a\)，Recall=\(h/t\)，FDR=\((a-h)/a\)，F1=\(2h/(a+t)\)，FPR=\((a-h)/(P-t)\)。空发现令FDR=0、Recall/F1=0，Precision为NA；分母退化时使用NA并列出valid_n。输入、系数非有限或缺失任务会触发错误或失败记录，不能被当成空支撑。

一个说明性反例是两元素估计 \(\{(1,1,1),(2,2,2)\}\)：其实际非零数为2，三轴投影的矩形包络容量为8。包络指标把未估计出的另外6个元素也算入矩形，因而不能用于评价向量Lasso的实际非零恢复。

## E. 配对、区间与汇总

主分析单位是同配置、同完整输入身份的不同方法。映射保留全部14标签；主汇总按10配置，在输入去重后统计。每方法600条唯一输入行、四方法共2400行；标签展示四方法仍为2800行。相同配置不等于相同输入，不把600缩成500；别名重复也不增加Monte Carlo分母。

PMSE配对差固定为ABESS减对照。每配置完整 \(R\) 对数据计算均值差、样本SD、MCSE和双侧t均值95%区间，df为 \(R-1\)。报告重复间SD与均值MCSE，不能把二者互换。Exact比例报告Wilson95%区间；各指标另列valid_n与失败数。偏斜时补中位数和IQR，不按测试成绩选择新的区间方法。

不以ABESS获胜次数、区间排除0或PMSE改善幅度作为有效性Gate。失败、困难及不利重复均保留；必要新对照缺少证书时不能只报告成功子集并宣布原计划主比较完成。历史比较表和新可靠比较表分别输出，不能把旧对照的区间放在新method_id下。

## F. 证据源及尚待完成的材料

| 内容 | 入口 |
|---|---|
| 正式32源及38项模拟源/配置身份 | doc/taskcard18/IDENTITY_AUDIT.md、同名JSON |
| 700标签到600完整输入及10配置映射 | doc/taskcard18/input_identity_map.csv |
| pilot数值对应及其身份对照 | numerical_simulation/outputs_taskcard18_closeout/pilot_input_audit.json、doc/taskcard18/INPUT_ID_RECONCILIATION.json |
| 两对照协议及method_id | doc/taskcard18/CLOSEOUT_PROTOCOL.md、同名JSON |
| 新对照独立实现 | research/taskcard18/baselines.py、solvers.py |
| 指标、去重、配对实现 | research/taskcard18/analysis.py |
| 历史重算，明确旧对照 | doc/taskcard18/results/historical_* |
| 正式ABESS定义边界 | paper/sections/02_method.md、doc/taskcard18/THEORY_INTERFACE.md |

仍需完成新对照全量必要拟合、证书验收、测试评价、同输入配对表、最终主图及独立审查。命令只在实际验证后写入复现README；本附录不虚构未运行过的“已验证”命令。上述缺口未解决时，方法与模拟联合状态保持未定稿。
