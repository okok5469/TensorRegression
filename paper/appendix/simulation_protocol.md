# 附录：模拟协议、指标与数值验收

更新日期：2026-10-29。此附录记录任务卡18固定协议及返回批次的实际验收：600套唯一完整输入、1200次有效新对照拟合及1200次评价已齐备，章节联合状态另见最终Gate。机器协议为doc/taskcard18/CLOSEOUT_PROTOCOL.json，protocol_version=2；数值规则未因测试成绩改变，集群地点变更由EXECUTION_AMENDMENT.json单独授权，不改写原协议。

## A. 输入、随机流与身份

原计划为numerical_simulation/outputs_v52_search_final_full/plan.json。每条任务具有标签、重复号、训练seed、test_seed和全部配置；四项原训练指纹为X_train、y_train、B_true和train_noise。完整输入身份同时绑定这些指纹和测试种子，不按单一训练设计指纹去重。700标签对应600套唯一完整输入、10配置；基准配置有150套完整输入，其余各50。原结果四方法共有2800行、56个标签×方法汇总单元。

训练seed为2026加序列偏移、100倍序列内取值顺序及重复号；非基准单独标签的四条序列偏移均10000。测试seed为训练seed加1000000，500测试点按100点一批生成。改变批大小可能改变高斯预测变量与噪声在随机流中的对应次序，因此保持原分批约定。

NPZ保存真系数、噪声、真支撑和拟合结果，但未归档原训练预测数组、原响应数组或原测试数组。重新生成时先比较可用原指纹；跨平台浮点计算导致不一致时不重写旧哈希，并按预定rtol=1e-10、atol=1e-10另核对真系数、噪声、支撑、历史Oracle训练RSS与测试指标。早期Windows的20输入pilot仅支持该环境下的数值对应。本次Linux返回批次中，1200份方法拟合的输入核验均记录四项训练指纹BYTE_IDENTICAL，覆盖全部600套唯一输入；这比早期pilot的核验层级更强，但不声称未归档的原测试数组也有逐字节副本。各拟合入口还按原test_seed和批大小重放Oracle测试指标作数值对应核验，新评价保存五批实际测试流指纹。本轮返回文件的独立只读审计没有再生成测试数组或独立重算全部预测误差；历史ABESS/Oracle预测指标仍复用原保存记录，系数及支撑指标则由保存数组复算。

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

## F. 实际运行、证书与网格边界

返回批次使用Linux x86_64、Python3.9.25、NumPy2.0.2、SciPy1.13.1、scikit-learn1.5.2和NumPy记录的OpenBLAS0.3.27构建。OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、OMP_NUM_THREADS及NUMEXPR_NUM_THREADS均为1。Lasso采用坐标下降，Group Lasso采用重启FISTA，分别求解B节规定的同一统计目标；其代码、版本、协议和环境身份均与返回缓存绑定。

固定20输入pilot通过后，全量缓存覆盖600套输入上的两个方法，共1200个有效最终模型及1200次独立测试评价；没有缺失输入、未解失败、到限或数值错误。Lasso的600条路径共60600份必要凸拟合证书（每路径100个内折候选及1个全训练拟合），顺序Tensor-GLasso的8964次局部CV共905364份证书，总计965964。全部证书同时满足双条件，最大KKT/λ为9.9996314×10⁻⁵，最大归一化gap为9.9999860×10⁻⁷。累计预算升级中的求解器调用不算新增输入或重复；早期prepare失败及封存本机试运行记录仍保留，当前批次“无未解失败”不改写这些历史。

| 方法及计数单位 | CV次数 | 上界λ | 内部λ | 下界λ |
|---|---:|---:|---:|---:|
| Vector Lasso，每套完整输入一次 | 600 | 157（26.17%） | 441（73.50%） | 2（0.33%） |
| 顺序Tensor-GLasso，每次局部CV一次 | 8964 | 31（0.35%） | 1513（16.88%） | 7420（82.78%） |

上述分母不同：不能把8964次局部CV当作8964次独立模拟重复。触边不违反求解证书，却提示调参覆盖存在边界，尤其是该顺序GLasso流程；本轮不根据测试排名扩展原20点网格，也不据此声称扩展必然改善预测。对照充分求解、网格是否充分、整个顺序流程是否泛化良好是三个不同问题。

Lasso和顺序Tensor-GLasso的逐方法耗时中位数分别为115.82秒和269.78秒，600次方法耗时之和分别为106738.49秒和196942.03秒。这些是当前集群调度条件下perf_counter记录的累计方法elapsed，不是测得CPU时间、并行作业墙钟总时间，也不与历史ABESS跨环境计时计算通用加速比。

## G. 证据源与交付索引

| 内容 | 入口 |
|---|---|
| 正式32源及38项模拟源/配置身份 | doc/taskcard18/IDENTITY_AUDIT.md、同名JSON |
| 700标签到600完整输入及10配置映射 | doc/taskcard18/input_identity_map.csv |
| 输入对应及早期身份对照 | 返回批次fits及evaluations的input_correspondence；doc/taskcard18/INPUT_ID_RECONCILIATION.json |
| 两对照协议及method_id | doc/taskcard18/CLOSEOUT_PROTOCOL.md、同名JSON |
| 新对照独立实现 | research/taskcard18/baselines.py、solvers.py |
| 指标、去重、配对实现 | research/taskcard18/analysis.py |
| 历史重算，明确旧对照 | doc/taskcard18/results/historical_* |
| 正式ABESS定义边界 | paper/sections/02_method.md、doc/taskcard18/THEORY_INTERFACE.md |
| 返回ZIP、源码/配置/环境、全部证书及CSV复核 | doc/taskcard18/cluster_review/RETURN_VALIDATION.json |
| 10配置四方法主结果、同输入配对差 | paper/tables/certified_main_results.md及CSV、certified_paired_pmse.md；doc/taskcard18/results/certified_* |
| 14标签完整附表及新旧实现配对变化 | paper/tables/certified_14_label_summary.md、baseline_update.md及CSV |
| 三轴规模分布、元素/容量及预定阈值敏感性 | paper/tables/support_scale_distribution.csv、support_structure_summary.csv |
| 四类主图及图注 | paper/figures/sample_size、signal_to_noise、tensor_dimension、correlation的PDF/PNG/SVG；CAPTIONS.md |

四方法汇总保留2800条标签方法行、2400条去重方法行、40个配置方法单元、56个标签方法单元及30个配置内PMSE配对比较。三轴规模分布保留全部1031个非空频数组，结构摘要包含40个配置方法单元；每轴频数之和对应当前配置R，不累计为额外重复。所有表保留完整分母，空发现的Precision记NA并另列valid_n。命令见research/taskcard18/README.md；文稿、表图及最终独立审查完成后，由G0—G7共同判定两章联合状态，不以文件名或ABESS排名代替验收。
