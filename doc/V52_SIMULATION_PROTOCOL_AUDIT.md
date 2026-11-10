# 正式v5.2完整模拟：实验协议与实现审计

核对日期：2026-10-26。仅读取正式计划、保存记录、当前同指纹源码和既有验收；没有拟合、生成新随机数据、执行merge、修改模型或结果。本报告供根目录完整模拟总结引用。

## 1. 唯一正式批次与本次实际核对

正式目录为 [outputs_v52_search_final_full](../numerical_simulation/outputs_v52_search_final_full/)，计划身份为 `30f231df0327e58331a18580dc8a03caeb0872dc991e57c42b095556100a80f6`。应以其中 [plan.json](../numerical_simulation/outputs_v52_search_final_full/plan.json) 为配置真值，而不是通用类默认参数、旧main入口或旧README默认三方法。

| 核对项 | 本次只读结果 |
|---|---|
| profile / n_reps | full / 50 |
| 实际方法 | adaptive、oracle、lasso、glasso；**不含known-s fixed** |
| 任务及文件 | 700任务；700 JSON、700 NPZ |
| 结果记录 | [raw_results.csv](../numerical_simulation/outputs_v52_search_final_full/raw_results.csv) 2800行，每方法700行；[summary_results.csv](../numerical_simulation/outputs_v52_search_final_full/summary_results.csv) 56行 |
| 标签/配置 | 14场景标签，10种实际参数配置 |
| 输入去重 | 四项训练输入指纹与test_seed联合去重为600；550个只出现一次，50个各出现三次；仅X_train指纹为550 |
| 源身份 | plan的38项源码/配置SHA256全部匹配现场；正式release的32项生产Python SHA256全部匹配现场 |
| 正式身份 | [release清单](releases/v5.2-search-final/RELEASE_MANIFEST.json)为released、approved_by_user=true；release_git_commit=null，不能用Git HEAD冒充独立新发布提交 |

BCD原始SHA256：`db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`。本次计数、指纹匹配及去重是新做的只读核对；并未因此重新证明全部保存预测或OLS数值正确。

既有完整工程验收见 [V52_FULL_SIMULATION_REVIEW.md](V52_FULL_SIMULATION_REVIEW.md) 与 [integrity_report.json](../numerical_simulation/outputs_v52_search_final_full/integrity_report.json)。前者详细区分已核验的记录/NPZ身份、支持与REE/RSS关系，以及未独立重建的训练X/y、测试流、PMSE、训练OLS最优性和条件数；不能把本次整理表述为重新运行了700组实验。

## 2. 数据生成机制

源码：[data_generation.py](../numerical_simulation/data_generation.py)，配置来源：[profiles.json](../numerical_simulation/known_s_cluster/profiles.json)及正式plan中的每任务config。

每个样本为三阶张量，响应模型为

$$y_i=\langle X_i,B^*\rangle+\varepsilon_i,\qquad \varepsilon_i\sim N(0,\sigma^2),\quad\sigma=1.$$

没有额外截距、协变量、缺失值或响应变换；均值为零、各预测坐标的总体方差为1来自生成分布，代码未另外做样本中心化/标准化。各样本独立生成；不同场景/任务可能共享随机种子族及实际输入，见下文。

三个模式协方差为AR(1)：

$$[\Sigma_m]_{ab}=\rho_m^{|a-b|},\quad
\operatorname{Cov}(X_{abc},X_{a'b'c'})=(\Sigma_1)_{aa'}(\Sigma_2)_{bb'}(\Sigma_3)_{cc'}.$$

实现先生成独立标准正态数组E，再沿三个模式乘各自Cholesky因子。若用与实际展平一致的NumPy C-order，向量协方差为 `Σ1 ⊗ Σ2 ⊗ Σ3`；生成器部分注释写反序Kronecker是另一vec约定，正文用上面的逐坐标协方差消除约定歧义。

真实支持为每轴随机起点的连续区间，长度始终 `(4,3,5)`，其笛卡尔积含60个非零坐标。起点均匀取自 `0,…,p_m−s_m`；这只是本轮DGP设定，估计器并不获知连续起点或真实位置，也不强制所选支撑连续。

先在真矩形内独立生成幅值 `U(0.8,1.2)` 和等概率正负号，外部填0得到B0；再整体缩放：

$$B^*=cB_0,\qquad c=\sigma\sqrt{\mathrm{SNR}/V_0},\qquad
V_0=\operatorname{Var}(\langle X,B_0\rangle).$$

所以 `Var(<X,B*>)/σ²` 等于指定SNR；这不是把单次样本经验信号方差调到SNR，也不是所有非零系数设成相同常量。相关结构变化时缩放因子也随之变化。支撑矩形自由系数不等于低秩CP/Tucker约束。

每任务训练样本量为n，独立测试样本500，按100一批共5批生成；测试X及噪声来自独立test_seed。同任务四方法每次从相同test_seed重建相同测试流，严格共享训练数据、真实B、噪声及测试样本。比较不基于测试集选择参数。

## 3. 14个标签、种子与重复关系

基准：n=200、p=(32,20,25)、真实s=(4,3,5)、rho=(.3,.3,.3)、SNR=2、sigma=1、n_test=500。各行只改变列明因素，其余均保持基准；这是单因素曲线设计，不是完整析因设计。每行50次，replicate=r取0—49；task_id=50×scenario_id+r。

| scenario_id | 标签 | 唯一改变值 | 展平维度P | task_id范围 | 训练seed |
|---|---|---|---:|---|---|
| 0 | baseline | 无 | 16000 | 0—49 | 2026+r |
| 1 | snr | 0.5 | 16000 | 50—99 | 12026+r |
| 2 | snr | 1 | 16000 | 100—149 | 12126+r |
| 3 | snr | 2 | 16000 | 150—199 | 12226+r |
| 4 | snr | 4 | 16000 | 200—249 | 12326+r |
| 5 | sample_size | n=100 | 16000 | 250—299 | 12026+r |
| 6 | sample_size | n=200 | 16000 | 300—349 | 12126+r |
| 7 | sample_size | n=400 | 16000 | 350—399 | 12226+r |
| 8 | dimension | 16×12×15 | 2880 | 400—449 | 12026+r |
| 9 | dimension | 32×20×25 | 16000 | 450—499 | 12126+r |
| 10 | dimension | 64×30×40 | 76800 | 500—549 | 12226+r |
| 11 | correlation | rho=(0,0,0) | 16000 | 550—599 | 12026+r |
| 12 | correlation | rho=(.3,.3,.3) | 16000 | 600—649 | 12126+r |
| 13 | correlation | rho=(.6,.6,.6) | 16000 | 650—699 | 12226+r |

所有test_seed=训练seed+1,000,000。具体公式及映射由 [known_s_cluster/run.py](../numerical_simulation/known_s_cluster/run.py) 的build_tasks实现；数据发生器使用 `np.random.default_rng(seed)`。本批显式传task的seed，不直接使用config.random_state=2026生成全部任务；test默认加10000的通用函数路径也不是本批实际路径。

非baseline各实验组均从10000偏移重新起算，这是冻结的历史实际种子规则，不是旧main全流程逐组累加的规则。scenario6/9/12同r的四项训练输入指纹和test_seed完全一致，故50套输入各重复三次，700−2×50=600。baseline及SNR=2与它们参数相同，但seed不同，仍有不同输入；因此“10配置”不等于“500独立数据”。

只按各场景50重复描述均值与重复间样本标准差。跨场景存在共享随机族，不把700或600当作IID池作统一显著性推断；700标签加权均值、600去重均值均只能按各自权重作描述。

## 4. 四方法真正运行的设置

实际调用链为 [known_s_cluster/run.py:fit_method/fit_tensor](../numerical_simulation/known_s_cluster/run.py) → [methods.py](../numerical_simulation/methods.py) / [tensor/abess.py](../src/tensor_abess/tensor/abess.py)。没有调用 `src/tensor_abess/baselines/` 中同名简化基线冒充本批方法。

| 项目 | Tensor-ABESS adaptive | Vector Lasso | Tensor-GLasso | Oracle OLS |
|---|---|---|---|---|
| 拟合信息 | 不给真实s或位置，target_sparsity=None | 展平全部P个坐标 | 三模式分组交替更新 | 直接给真实三个支撑 |
| 初始化 | Group Lasso BIC，c=3 | ISTA最终全训练拟合零初值 | Group Lasso CV，c使用函数默认2 | 无支撑搜索 |
| 主要搜索 | adaptive＋旧caps内完整合法整数规模，Cold fixed BCD best-visited | L1惩罚系数 | 顺序逐模Group Lasso选组，矩形内OLS | 真矩形内OLS |
| 正则网格 | 初始化20个lambda | 20个lambda | 初始化及各模式20个lambda | 无 |
| lambda范围 | λmax至0.01λmax，对数等距 | 同范围，λmax=max|Xᵀy/n| | 同范围，λmax=max_g||Zgᵀy/n||₂ | 无 |
| 选lambda | BIC：n log(RSS/n)+组数×组宽×log n | 5折验证SSE最小 | 5折验证SSE最小 | 无 |
| ISTA预算 | max_iter=500，tol=1e−6 | 相同 | 相同 | 不适用 |
| BCD预算 | max_cycles=6，k_max=1，m_max默认15，eps=1e−8 | 不适用 | max_cycles=6，支持不变则停 | 不适用 |
| 最终估计 | 所选矩形OLS | 保留L1系数，无post-Lasso OLS | 所选矩形OLS | 真矩形OLS |

ISTA目标为 `RSS/(2n)+λ||β||₁` 或 `RSS/(2n)+λΣ_g||β_g||₂`，步长为设计谱范数平方除以n的倒数；没有额外组大小权重。停止条件为 `||β_new−β||₂ ≤ 1e−6(1+||β||₂)`，否则500次到限返回。OLS使用 `np.linalg.lstsq(...,rcond=None)`，无额外Ridge和自动截距。

Vector Lasso与Group Lasso的CV均将原样本索引 `array_split` 为5个连续块，不随机重排；每lambda累加五折SSE，完全并列保持先遇到的较大lambda。λmax和网格在当前完整训练设计上生成。Group Lasso BIC在相同训练数据的lambda路径上warm start；Group Lasso CV每折重新零初始化。

**Vector Lasso特殊边界：** `_fit_vector_lasso_cv` 在同一lambda内，把前一CV折的beta传给下一折作为beta_init；前一折的训练数据包含后一折验证样本。若每折精确收敛到同一唯一解，初值可能不影响结果，但本批只用有限500步且未保存收敛证据，不能据此宣称严格独立的内折评估。该情况不涉及独立测试500样本进入拟合，但限制“规范充分收敛Lasso基线”的表述；本次只记录，不修正或重跑历史成绩。

Tensor-GLasso也不是对整个三模搜索重新做严格外层内嵌CV：模式设计及其他轴支持来自当前完整训练过程，内层只对该固定局部设计选lambda；其全训练初始化预筛同样位于该局部CV之前。某模式选择空支持时保留上一非空支持，故不能把它当自动允许完全空张量的标准凸Group Lasso。应称“本仓库既有Tensor-GLasso流程”。

ABESS本批c=3、max_cycles=6、k_max=1；不可误抄公共接口默认c=2、max_cycles=30或k_max自动值。cfg.cv_folds=5虽然保存，ABESS的BIC初始化分支不以五折CV选择lambda。mode MSIC alpha=gamma=1，最终全局MSIC alpha=1.5、gamma=1、容忍带scale=1；该确定性带宽不是CV标准误。真实s=(4,3,5)只用于DGP/评价/Oracle，未作为adaptive参数输入。

旧caps由初始化规模和样本量构造，再压缩使乘积≤floor(.7n)；v5.2只补caps内整数规模，不放宽上界、不穷举具体位置。n100的真形状435可被caps444排除；“真q60≤70”不等于真形状合法。更低训练RSS、best-visited正确性和完整规模网格均不保证更低测试PMSE。

## 5. 收敛与指标的可报告边界

保存结果没有完整的每个ISTA内循环收敛状态/KKT残差。`GroupLassoFit`只有beta、RSS、active_groups、lambda；Vector Lasso只返回beta。方法正常返回、没有warning或n_cycles较小都不能证明所有内部优化充分收敛；Vector Lasso保存的n_cycles=1只是wrapper标记，不是ISTA迭代次数。ABESS的公共n_cycles/历史也不代表整个全网格搜索工作量。

因此，本批支持“同数据上当前冻结实现与预算的比较”，不能升级成对任意更强优化器或充分收敛的标准基线的普遍胜出结论。不得事后根据测试表现更换预算后仍沿用该批标签。

指标实现：[metrics.py](../numerical_simulation/metrics.py)。PMSE是500个含噪响应的预测均方误差；signal MSE对无噪信号评价；R²以同500测试响应均值为参照；REE是Frobenius误差范数/真B范数，**未平方**。Exact要求三个模态支持全部相同，mode F1是三轴F1平均；q_hat和element F1按报告模态支持笛卡尔积计算。

尤其Vector Lasso的模态支持来自阈值1e−8的非零系数投影，其诱导矩形可包含实际系数为零的坐标；旧element F1、q_hat不能当其实际元素非零集合的F1/数量，也不能以这些字段直接证明向量基线没有恢复元素。Oracle Exact=1来自给定真支撑，属于定义，不是估计器恢复能力；Oracle OLS不是任意估计器有限样本预测风险的不可超越下界。

## 6. 环境与验收强度

正式保存环境：Linux x86_64、CPython3.9.25、NumPy2.0.2、SciPy1.13.1、OpenBLAS0.3.27。每task一CPU，OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、OMP_NUM_THREADS、NUMEXPR_NUM_THREADS均为1；方法在task内顺序执行。来源为正式plan.environment及 [cluster_preflight_gate.json](../numerical_simulation/outputs_v52_search_final_full/cluster_preflight_gate.json)。

拟合秒数是各wrapper计时，跨机器/作业/缓存背景的历史耗时不应解释为受控冷启动加速比；测试流评价、任务调度及所有并发墙钟不由单方法runtime_sec完整代表。

本机读取环境不同并不改变封存集群环境身份；本次没有执行worker、冒充原环境或生成新的验收Gate。既有 [任务卡6交付](TASKCARD6_EXECUTION_REPORT.md) 中250归因的当时阻塞，后来由卡7完成，见 [三版本最终归因](VERSION_EFFECT_DECOMPOSITION_FINAL.md)；其250个三版本模型比较是独立附加证据，不能并入本批700的四方法分母，也不是700全场景v5.1补跑。
