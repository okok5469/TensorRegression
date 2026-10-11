# LEMON EEG 增量预测价值验证：完成记录

用户于2026-10-09授权执行 `empirical_analysis/任务卡.md`；2026-10-10完成真实实验与验收。任务是检查闭眼EEG特征是否提供超出年龄段和性别的TMT-B样本外预测信息，不预设任何算法必须胜出。

## 1. 数据与旧流程审计

- 复用 `empirical_analysis/lemon/outputs_lemon_adjusted/qc/analysis_dataset.npz`，实际字段X、y、C、subject_ids；分析受试者145名，X=(145,61,16,5)、C=(145,8)，全部有限、ID唯一且与included manifest顺序完全一致。
- X与原146人 `tensors/X_unstandardized.npy` 按manifest.tensor_index筛选后逐元素一致；145个个体npy也逐元素一致，各人的通道、频段和张量维度元数据与共同metadata一致。特征为log absolute band power，61通道、16个25秒闭眼窗口、delta/theta/alpha/beta/gamma五频段，NumPy C-order展平。
- y与原 `empirical_analysis/TMT.csv` 按ID的自然对数log(TMT_5)一致；C从原participants.tsv与既有encoding重建后逐元素一致。年龄7分类，基准20–25岁，不转换中点；官方性别1=female映射0、2=male映射1，C首列为截距。最小年龄组4人。
- 旧外层训练折残差化/缩放没有使用外层测试数据；但Vector Lasso/GLasso的内部CV接收外层已经预处理的全训练矩阵，未在每个内折重新拟合全部预处理。本次将两者调参移到独立实证适配层，内折各自重新拟合X/y协变量回归、EEG残差均值/样本标准差及响应残差中心化。
- 未下载、读取BrainVision、重算Welch或重新生成EEG。正式实验前后核对788个核心源码/数值模拟源码/旧实证输出的SHA256，0项变化。

## 2. 独立实现与预先固定设计

新增模块 `empirical_analysis/lemon/incremental_value/`：__init__.py、config.yaml、data_loader.py、preprocessing.py、model_adapters.py、nested_cv.py、metrics.py、reporting.py、run.py、README.md及五份unittest文件；figure_backend.json和protected_fingerprints.json保存绘图选择与保护证据。没有改src/tensor_abess或numerical_simulation。

外层3次5折，种子20260908/20261009/20261010，按既定年龄分类均衡分层，所有模型共用划分。内层3折，种子20261009+100×repeat+outer_fold；外层train/test=116/29，inner train=77/77/78，全部协变量训练矩阵满秩。

模型为Covariates Only、Tensor-ABESS、Vector Lasso、Tensor-GLasso、Ridge。ABESS保留BIC初始化及终版自适应MSIC（alpha1.5/gamma1/one_se1、max_cycles6、k_max1等原配置），不基于外层结果调参。其信息准则是训练准则，不是独立内层验证性能。

Lasso/GLasso的20个固定lambda/lambda_max比例网格从1至0.001，每个当前训练设计独立计算lambda_max；GLasso的每次局部分组设计同样只用当前训练数据。这是明确的外部调参政策，与历史内部CV不同，原ISTA/组阈值、候选筛选/支持回退、矩形OLS数学定义保持不变。每个lambda仍零初始化，只在同一训练数据内缓存设计与精确谱范数。Ridge20个lambda从1e-4至1e4，目标RSS/(2n)+lambda||beta||²/2，采用样本空间对偶解。

每个内折依据原log响应尺度的验证SSE汇总选择最小MSE；相同分数预先规定选更强正则。完整预测恢复为Cα+y_residual_mean+<(X−CΓ−X_mean)/X_scale,B>，不存在重复截距。配置、来源、建模源码和环境指纹控制断点身份，图版代码另存独立指纹。

## 3. 分阶段验收与耗时

- 新增25项unittest通过（最终完整测试0.373秒）：来源破坏阻断、原尺度恢复、covariate OLS一致、内外层隔离、固定网格、Ridge对偶/原始空间一致、Lasso/GroupLasso与未修改核心ISTA数值一致、OOF唯一覆盖、增量指标及按受试者聚类bootstrap。按测试先RED再实现GREEN。
- 真实smoke独立目录 `outputs_eeg_incremental_smoke/`：一个外折、五个模型、全部成功，耗时73.73秒，不作为正式结果。
- 正式 `outputs_eeg_incremental/`：75/75个模型fit完成，145×3×5=2175条OOF，435条外层受试者划分，2700条内层候选评价、135条逐模型内折轨迹；用时1208.85秒约20.1分钟，GLasso调参占主要成本。
- 运行使用现有Python3.12.8/NumPy1.26.4/SciPy1.16.1环境及单线程BLAS/OMP；实际版本以config_used.json为准，没有安装或升级包。全程遇到沙箱启动与代理审批pending环境故障，转用已获准主线执行后完成，不以替代数据补偿。
- 独立加载75个B.npy和outer preprocessor重算2175条预测，最大差0；训练preprocessor重新拟合最大差0；covariate OLS独立计算与基线预测最大差9.77e-15。所有外折映射/y_true均匹配，所有inner train/validation与outer test严格隔离。证据为diagnostics/independent_validation.json。

## 4. 真实结果

主性能为每次完整OOF指标的重复间均值；RMSE SD为3次重复样本标准差。增量列为受试者/重复等权汇总MSE后计算，正值才表示EEG改善。

|模型|RMSE均值|RMSE SD|MAE均值|R²均值|总汇ΔRMSE|总汇增量R²|
|---|---:|---:|---:|---:|---:|---:|
|年龄段+性别|0.317485|0.001519|0.255014|0.367900|基线|基线|
|Tensor-ABESS|0.371667|0.054290|0.292457|0.121430|-0.056814|-0.389922|
|Vector Lasso|0.321263|0.006842|0.257827|0.352580|-0.003824|-0.024237|
|Tensor-GLasso|0.330680|0.003762|0.265927|0.314219|-0.013207|-0.084925|
|Ridge|0.319677|0.003150|0.256707|0.359110|-0.002200|-0.013906|

四个EEG模型均0/3次重复改善协变量基线。此次属于任务卡情形A：当前特征/方法/验证设计没有显示可靠的样本外增量价值，不能断言EEG完全不含相关信息。EEG模型内部平均RMSE排名Ridge、Vector Lasso、Tensor-GLasso、Tensor-ABESS；历史单次ABESS在EEG模型内部胜出的排名没有稳定复现。

## 5. 不确定性与逐人差异

2000次bootstrap以受试者为聚类单位，抽到一个人时保留其全部重复预测，不将435个人次当独立受试者。这是条件于保存OOF、不重训模型的描述性区间，不作严格总体显著性或因果解释。

|EEG模型|ΔRMSE描述性95%区间|增量R²描述性95%区间|改善人群|恶化人群|绝对误差差异最大10%受试者贡献|
|---|---|---|---:|---:|---:|
|ABESS|[-0.080513,-0.037185]|[-0.575741,-0.250652]|28.97%|71.03%|53.02%|
|Vector Lasso|[-0.006771,-0.001306]|[-0.042657,-0.008354]|20.69%|28.97%|73.54%|
|GLasso|[-0.021183,-0.004627]|[-0.139535,-0.027974]|31.72%|68.28%|44.63%|
|Ridge|[-0.003539,-0.000864]|[-0.022369,-0.005763]|44.83%|55.17%|59.32%|

Vector Lasso有50.34%人跨重复平均误差与基线相同，对应多数折选零EEG模型。所有人和所有折保留；极端贡献指标只是诊断。

## 6. ABESS异常与后续方向

ABESS的三次OOF RMSE为0.331021/0.350659/0.433323，R²为0.312862/0.228915/−0.177486。整体训练/测试折均RMSE为0.264405/0.363746，差0.099341，明显高于协变量基线0.019284的差。

第三次重复（repeat=2）外折5选择s=(4,4,5)、q=80，训练RMSE0.103144、测试RMSE0.670771；其余14折q在1至6。该单折贡献全部三次重复ABESS相对协变量总SSE恶化的56.93%。这一容量跳变支持优先检查MSIC选阶、支持搜索及数值条件稳定性；目前不能仅凭相关诊断把原因唯一归咎MSIC或算法某环节，也不能删除该折后用好看指标作为主结果。

Ridge在13/15折选择网格最大lambda=10000（其余545.559与1438.450），Vector Lasso12/15选lambda_ratio=1，GLasso13/15选1。内层验证整体倾向压低EEG效应；Ridge在强正则下贴近基线，但没有给出分散EEG效应的正增量证据。网格边界选择必须披露；扩大网格应作为新的预设验证，不追调本轮外层结果。

独立审查补充：Vector Lasso的ratio=1给出零EEG系数；GLasso保留既有非空支持回退和OLS，因此ratio=1也可能留下非零EEG项，不能当成协变量空模型。Ridge上界选择不意味着唯一最佳lambda已找到。q80虽符合既有容量上限，却接近floor(0.7×116)=81，应作为容量/泛化风险诊断。

后续有理由做MSIC容量/搜索稳定性的受控消融，例如同一外层训练集内比较候选支持规模、条件数与准则路径，预先固定新的嵌套验证或独立数据评价；并研究窗口冗余、通道/频带结构和矩形支持约束。目标是解释容量跳变和提升可复现性，不以强行证明ABESS或EEG优势为目标。

## 7. 交付路径与图QA

- 实现与复跑说明：`empirical_analysis/lemon/incremental_value/README.md`。
- 正式报告：`empirical_analysis/lemon/outputs_eeg_incremental/reports/EEG_INCREMENTAL_REPORT.md`、`EXECUTION_SUMMARY.md`。
- 完整OOF、划分、超参数/指标/逐人差异：`outputs_eeg_incremental/cv/`。
- 来源、独立复算、预处理及泄漏证据：`outputs_eeg_incremental/diagnostics/`。
- 五类实际图：`outputs_eeg_incremental/figures/` 的model_comparison、incremental_rmse、paired_error_difference、predicted_vs_observed、repeat_stability，均有PNG/PDF/SVG。
- 所有图通过渲染字体检查（最小8–9pt）、PDF碰撞检查（0FAIL/0WARN）；三个多面板图对齐PASS，两个单图NOT APPLICABLE，全部人工逐图查看无裁切。静态源码18PASS/3WARN/0FAIL；WARN对应报告尺寸、300dpi预览和无投稿TIFF，本任务是内部报告图，未宣称期刊生产文件。

没有提交git、创建分支/worktree、删除旧文件、安装包或修改集群环境。
