# 任务卡18：独立流程复核

审查结论：**PASS_SCOPED_PIPELINE_REVIEW；完整主比较仍待验收**。

本审查由未参与主要基线数值实现的审查者完成；其本卡工作为只读身份审计、输入ID侧录与冻结结果图表，没有编写或修改运行中的solvers.py、baselines.py、workflow.py或协议JSON。此次仅阅读源码、读取现有缓存、调用验收函数及对内存副本施加两个已知反例；没有新拟合、输入重建或新增方法测试评价。该结论是流程范围内的检查，不是40项pilot或600×2主比较全部通过的声明。

## 已检查的科学与数据边界

1. **目标与数值证书。** Lasso及无权重Group Lasso均使用RSS/(2n)+lambda×相应范数，无截距、样本中心化、标准化或额外组权重。梯度及KKT中的n为当前拟合训练子集样本数；活动集使用实际非零，不以论文1e−8支撑阈值掩盖KKT。残差缩放后的对偶可行性、KKT/lambda≤1e−4、归一化原始—对偶gap≤1e−6同时决定正lambda通过；明显负gap保留为错误。lambda=0使用独立最小二乘/梯度条件，没有除以lambda。
2. **有限数值策略。** 三档迭代总上限2000、10000、50000保持冻结。v2 Lasso每50个坐标下降sweep独立检查证书，库内部tol=0不充当收敛证据；Group Lasso使用隔离的重启FISTA。源码保留未达证书的诊断并拒绝进入CV选取/测试评价。独立split L-BFGS-B及凸对偶SLSQP接口仅供有界诊断，未被本次审查重新执行，也不把存在该接口当作所有实际子问题已经独立重解的证明。
3. **CV隔离与选择。** select_lambda_cv以fold为外循环，每个fold重新创建只含该折训练数据的problem，并把beta_warm重置为None；只有同折相邻lambda可复用。按原样本顺序array_split，验证SSE逐折相加，降序网格的首个argmin处理并列。网格由当前完整训练数据构造这一局限保留；最终全训练拟合重新从None开始。Lasso保留L1系数，不自动OLS。
4. **顺序Tensor-GLasso。** 初始化与外层轴1→2→3顺序、组定义、全训练边际预筛、局部设计、非空回退、6轮/支撑不变停止以及最终矩形内OLS与声明流程一致；ABESS的caps没有施加到该基线。其他轴支撑、预筛和网格依赖完整训练过程，因此这仍是局部CV，不能表述为整个三模流程的严格嵌套CV。
5. **缓存入口。** validate_fit复核metadata seal、当前运行源码/协议/环境身份、规范UID、NPZ SHA256与可读数组；validate_metadata复核方法ID、valid、每个必要CV候选/最终拟合的证书、五折×lambda覆盖、fold路径状态标记、每个拟合的n_fit以及选择一致性。数组检查要求尺寸、有限性、整数且无重复/越界轴索引，并核对声明矩形边界。默认新统计流程经此入口读取新缓存，再核对评价记录与同一模型/方法/UID的绑定。

## 两个已知缺口的实际回归检查

以已完成的真实v2 task1 Lasso metadata为母本，先确认原缓存通过，然后仅在内存副本中作如下修改：

| 反例 | 当前结果 | 实际拒绝原因 |
|---|---|---|
| selected_lambda改为999，保留原lambda_grid和selected_index | 拒绝 | Saved CV selection disagrees with the frozen tie/score rule |
| cv_scores的首项改为NaN | 拒绝 | Finite descending lambda grid and matching finite CV scores required |

当前accept同时要求有限非负降序grid、等形有限scores，selected_lambda等于grid[argmin]，每个CV/最终诊断中的lambda等于其grid位置，并按保存的每折validation_sse重新求和核对scores。此前这两个反例能通过的具体缺口在本次复查中已关闭；没有重写真实metadata或seal来制造通过。

analysis原两个非有限值反例也已复查：paired_statistics拒绝包含NaN的配对重复，support_metrics在阈值化前拒绝NaN系数。Wilson全失败下界和全成功上界分别严格为0、1，采用解析边界，没有一般裁剪。无发现时Precision=NA及其valid_n是预声明统计约定，与非有限输入/失败方法的拒绝不同。

## 真实缓存正例

实际调用 `validate_fit(record['task'], record, 'lasso')`，其中record由冻结读取器读取原task1；验收通过：

| 项目 | 本次实测 |
|---|---|
| method_id | vector_lasso_cv_isolated_v2 |
| UID | input_3dffef31b5573ff285b5a33c4397c13795eb26fb060e0c1b0c426af69279ea16 |
| 必要证书 | 101：5折×20候选+1全训练最终拟合 |
| 系数尺寸 | 32×20×25 |
| 声明轴支撑大小 | 0、0、0；合法零Lasso模型，不作失败或缺失删除 |
| 最大KKT/lambda | 5.509835505450193e−05 |
| 最大归一化gap | 9.992959208294665e−07 |
| metadata seal | 3e70723a4ac2076d659b3aa127bad7b2d38d158e6cbb3fda08504ccd91f06f62 |
| NPZ SHA256 | bb96737d55af6e0e4f23a36bb4bf55e29ca8e4382d22734b4944efb67eb92cd0 |
| 新方法测试评价 | test_evaluated=False |

这里复核的是已保存证书数值、来源和验收链，并未独立重建该101个训练子问题逐一重算梯度，也未重新拟合零模型或生成新的测试分数。

## RUN身份与输入映射

RUN_IDENTITY.json的record seal正确；其中identity与当前source_identity完全一致，绑定Python3.12.8、NumPy1.26.4、SciPy1.16.1、sklearn1.5.2、Windows/AMD64、BLAS构建信息及四项线程环境均为1。该记录的计划文件SHA256与冻结plan一致，input map SHA256与当前映射一致。

逐条核验RUN的700项task_id、完整config摘要、原seed/test_seed：当前unique_id与RUN及input_identity_map三方完全一致，均为 `input_ + SHA256({input_fingerprints:四项原训练指纹,test_seed:原测试种子})`，去重得到600个完整输入。旧training键裸hash与规范键的差异继续由INPUT_ID_RECONCILIATION.json保留原始语境；它不等于当前v2 UID。初版未seal的训练诊断不作为本次真实v2正例或主比较缓存。

| 身份材料 | 本次核验SHA256/seal |
|---|---|
| RUN_IDENTITY文件SHA256 | 521eeb34746d2111a7dadd5e0742213fa4c2fa1c0d912ad2f1a9e6e802261e19 |
| RUN record seal | 78b7a219db241f56c1b2a79ca5f7c560a8d52694c907657e3bfa3ff73cfab9c2 |
| 协议JSON SHA256 | a794d6a0efef81acf9d69b3429d8fe98c9d18131b8412ca70337fa4cbaebfa61 |
| 输入映射SHA256 | c6f3800cddb86eb2fc43d4d65271f557d9c98064676df037f32eb3c6be713f68 |
| solvers.py | 27f8753b51b635c2792157b46c79d5fa00995817cefa6eeab942407abdbd800f |
| baselines.py | 082f087df00cf2e159e5fba846dce967672f98d8ca155cb4f4b472f7f78bc4ca |
| workflow.py | d9e51c121af2262623a1aa8046fc0627fe88ba79a1f9bab2f43fb07289e8753d |
| accept.py | a38c062998fe594453f297c809ce885a0867d08eecf72772acbd28edf2779422 |
| analysis.py | e9a54020f4d231bc69a2ff08b39d9adbef7079ee483043a07446c718301377c9 |

## 不能由本审查推出的结论

本次实际缓存正例只有task1 Lasso，不能外推其余Lasso或任何实际Tensor-GLasso任务均已完成、达到精度或优于正式ABESS；源码流程核对也不能替代40个pilot和1200个必要新拟合的全量数值验收。主比较仍须完成全部600唯一输入×2新方法、全部必要证书、失败与重试记录、同输入测试评价和正式配对统计，再核对表图与正文。

本次没有新增方法测试成绩，不能判断相对新对照的优势保留、缩小、消失或反转。现有四幅冻结ABESS/Oracle图只能支持章节中的历史描述与适用边界，不能替代新主比较；两章不得因本独立流程检查通过就撤销待定稿状态。输入数值对应仍不等于未归档原X/y/测试数组的逐字节恢复，Oracle为诊断参照，旧对照11/14排名不构成当前可靠基线比较证据。
