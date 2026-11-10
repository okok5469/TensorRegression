# 实证与汇报历史交接索引

核查日期：2026-10-26。范围：任务卡13/15/16/17、既有实证报告与验收记录、近期汇报；本次仅阅读和整理，没有拟合模型、改写原记录或启动新研究。下述“完成”优先表示计算与数值交付完成，不等于已经证明科学优势。

## 当前必须继承的结论

**目前没有已确认的论文主实证。** LEMON、Apple、Mushroom、北京均未在当前比较中建立足够的方法优势。北京旧封存报告/验收JSON仍含“主实证候选”判断，但用户后续已明确撤销该定位；应以2026-10-26更新的[近期工作报告](../Presentation/recent_work_20261024/近期工作报告.md)为当前汇报口径，不修改旧封存记录来抹掉历史。

**UCI270已由用户明确放弃作为实证候选。** 状态为`ABANDONED_BY_USER`；仅原始匿名`gas_id`数字研究记录保留，化学名称未验收，不采用诊断置换自动重命名，不联系作者、不发邮件、不继续研究。其`192/192`计算完成和数值重放通过不能替代名称语义验收。

正式算法仍为`v5.2-search-final + 原MSIC`。FS是未获生产采用的研究臂，Warm也不是正式版本，禁止用研究臂较好成绩替代正式成绩。

## 结果速查

前三套主CV结果均先拼齐每次完整OOF，再报告三次重复的指标均值；重复共享实体，不是三次外部独立验证。下表不同响应的RMSE单位不同，不能直接跨数据集排名。R²也不是跨任务难度排名。

| 数据/协议 | 正式ABESS RMSE / R² | 代表性较强对照 RMSE / R² | 当前用途 |
|---|---|---|---|
| LEMON，145人，3×5外折 | 0.5046 / −0.5970 | 仅协变量 0.3175 / 0.3679 | EEG增量未建立；边界记录 |
| Apple，240样本ID，3×5外折 | 0.8383 / 0.0086 | 张量展平Ridge 0.8270 / 0.0350 | 整体信号弱；非主实证 |
| Mushroom，250记录、243谱组，3×5外折 | 13.6523 / 0.7859 | 原204维平均谱Ridge 11.1252 / 0.8580 | 内部预测有效，尚无比较优势 |
| Mushroom，5文件留一压力测试 | 31.7833 / −0.1580 | 张量Ridge 31.1006 / −0.1088 | 八方法块外R²均负；边界案例 |
| 北京，未来352日期 | 47.4525 / 0.5816 | Vector Lasso 46.4079 / 0.5998 | 有未来信号，未超过强对照；非主实证 |

单位：LEMON为`log(TMT_5)`；Apple/Mushroom为干物质百分点；北京为µg/m³。对照为既有结果的描述性选取，不是新增选模。

## 1. LEMON：既有实证与任务卡12双规则研究

- 输入145人、61通道×16时间窗×5频带；响应`log(TMT_5)`，协变量设计8列。原张量146人，沿用历史审计的145人分析集，没有在本轮另筛样本。
- 3次重复×5折，每折训练116、测试29；所有训练预处理与历史划分保持一致。任务卡12完成15次native搜索、1,215个完整候选，两种评分共用候选池，形成30个所选模型、2,610行训练/外测预测；这不是30次独立搜索，也不应将训练预测计为OOF。既有Ridge/Lasso/GLasso等对照沿用受审计历史结果，不冒充本轮新拟合。
- 正式原MSIC RMSE约0.50458、R²约−0.59701；仅协变量RMSE约0.31749、R²约0.36790。FS虽降低本次误差，仍弱于仅协变量及已有比较模型，没有可靠EEG增量证据。
- 历史双规则报告中的`PENDING_CLUSTER_RESULTS`是当时状态；任务卡12最终验收已为`TASK12_REVIEW_COMPLETE`，最终建议B，原MSIC继续生产默认、FS不发布。HCP-YA/NKI-RS为`NOT_RUN/NOT_PROVIDED`，不是已完成实证。
- 入口：[LEMON说明](../empirical_analysis/lemon/README.md)、[增量预测入口](../empirical_analysis/lemon/incremental_value/run.py)、[双规则研究入口](../research/taskcard12/empirical.py)。
- 结果：[双规则报告](taskcard12/EMPIRICAL_DUAL_MSIC_REPORT.md)、[重复级指标](taskcard12/verification/empirical_lemon/repeat_metrics.csv)、[OOF预测](taskcard12/verification/empirical_lemon/oof_predictions.csv)、[独立数值验收](taskcard12/verification/EMPIRICAL_INDEPENDENT_VALIDATION.json)、[最终版本建议](taskcard12/FINAL_VERSION_RECOMMENDATION.md)、[任务卡12最终状态](taskcard12/FINAL_VALIDATION.json)。

## 2. Apple：任务卡13完整实证

- 任务：[任务卡13](../任务卡/任务卡13.md)。真实图像按样本ID连接CSV响应，全部240条保留，官方MD5通过；响应质量分数乘100为百分点。
- 固定矩形ROI、8×8面积加权空间聚合、24个连续波组，即`8×8×24`；七方法共享三次重复五折和内三折，所有缩放仅用当前训练折，ABESS仍用MSIC选阶。
- **已完成105外层模型、2,520内层候选拟合，共2,625次拟合，5,040条OOF。** 总模型/汇总耗时约2,596.9秒，21张图验收通过；先前集群包仅历史准备，未提交。
- 正式ABESS RMSE`0.8383±0.0118`，R²`0.0086±0.0278`；均值RMSE0.8456、Ridge0.8270。14/15折q=1，不能凭此单独证明欠选；配对区间均跨0，不作显著劣于所有对照的主张。
- 真实果实/视角/批次身份不足，240个ID不等于额外证明240个完全独立实体。Lasso全部585个内层中119个未达到收敛标准，但被选中45个内层和最终15个外层均通过；M1/M2未暴露完整收敛状态保留unknown。
- 入口：[执行脚本](../empirical_analysis/spectrofood_apple/run.py)、[冻结协议](taskcard13/FROZEN_PROTOCOL.md)。
- 产物：[最终报告](taskcard13/APPLE_EMPIRICAL_FINAL_REPORT.md)、[汇总](taskcard13/summary_metrics.csv)、[OOF](taskcard13/oof_predictions.csv)、[最终验收](taskcard13/FINAL_VALIDATION.json)、[完整运行输出](../empirical_analysis/spectrofood_apple/outputs/apple_v1/)、[独立验收](taskcard13/local_execution/INDEPENDENT_FINAL_AUDIT.json)。

## 3. Mushroom：CSV预检与任务卡15完整张量

CSV预检是独立低成本阶段，**未运行Tensor-ABESS**：250记录、204波段，响应5.647519%—86.861962%；6组重复谱涉及13记录，同谱响应冲突保留、不平均。按243个完全相同谱组控制内外层泄漏，3×5外层、内3折。完成45外层模型、1,080内层拟合、2,250条OOF、约30秒；Mean/Ridge/PLS的RMSE分别29.7344/11.1252/12.0783，R²分别−0.0135/0.8580/0.8327。243谱组不等于243独立物理蘑菇。

预检入口：[mushroom_screen.py](../empirical_analysis/mushroom_screen.py)；报告与验收：[预检报告](mushroom_screen/MUSHROOM_SCREEN_REPORT.md)、[预检验收](mushroom_screen/FINAL_VALIDATION.json)。该报告“Apple暂停”描述的是当时工作转向，不推翻Apple现有完整结果。

完整张量阶段对应[任务卡15](../任务卡/任务卡15.md)：五个MAT官方MD5通过、250个显式ID全部对齐；采用固定8×8×24高光谱张量，八方法同时比较。main为重复谱分组3×5外折，block为五文件留一压力测试，内折独立预处理与调参。

- **完成160外层模型、4,380内层拟合、8,000条OOF**（main6,000、block2,000），26张图通过验收。main为120模型、block40模型；不可把CSV预检当作新独立物理样本。
- main ABESS RMSE`13.6523±0.7463`，R²`0.785916±0.02356`，平均RMSE第5/8；原CSV平均谱Ridge更好。block ABESS pooled RMSE31.7833、R²−0.157987，八方法均负。
- 响应总平方和98.189%来自文件均值差异；第5文件50条测试响应全部超出训练上界。文件是存储分块，不能擅称真实生物批次；小块响应方差低使R²很负，同时RMSE也证实实际误差大。
- 行/列是相对ROI位置，不代表跨实体解剖对应；main波组Jaccard仅0.1826。Lasso58个候选内层未达到标准，但选中内层及最终外层均通过，ABESS/GLasso完整收敛未知。
- 入口：[完整张量执行脚本](../empirical_analysis/spectrofood_mushroom/run.py)。产物：[完整报告](mushroom_empirical/MUSHROOM_EMPIRICAL_FINAL_REPORT.md)、[最终验收](mushroom_empirical/FINAL_VALIDATION.json)、[汇总](mushroom_empirical/summary_metrics.csv)、[OOF](mushroom_empirical/oof_predictions.csv)。

## 4. 北京空气质量：任务卡16

- [任务卡16](../任务卡/任务卡16.md)固定Aotizhongxin目标站，输入为自然前一天12站×6污染物×6个四小时区间，预测目标日PM2.5日均；共1,409共同有效日期。
- 四个开发回测训练/测试数361/179、540/163、703/174、877/180；最终1,057/352，测试期2016-03-01至2017-02-28。内层三个扩展时间窗。输入缺失中位数填补和X/y标准化只拟合训练期；测试每天更新前一天已观测输入，不用测试标签重训。实际发布延迟未知。
- **完成35外层模型、885内层拟合、7,336条逐日预测，14张图通过验收。** 开发合计696测试日期，但对应不同训练模型，不能当一个固定模型的外部测试。
- 这里的方法编号与Apple/Mushroom不同：M0均值、M1 Persistence、M2单站Ridge、M3张量Ridge、M4 Lasso、M5 GLasso、**M6 ABESS**。
- 未来ABESS RMSE47.4525、R²0.581631，Persistence68.7512/0.121784；ABESS相对Persistence改善30.98%，但Lasso46.4079/0.599847，单站Ridge47.4652。多站增量与强对照优势均未建立。
- 最终矩形容量20/432，支撑为5站×2污染物×2晚间时段；q不是完整自由度，也不等于已公平证明比所有竞争模型更稀疏，更不能作因果污染传播解释。
- **定位覆盖规则：** 旧报告Q6与JSON的`MAIN_EMPIRICAL_CANDIDATE_WITH_LIMITS`仅保留历史；当前用户要求北京也不作为主实证，交接者不得按旧字段自动恢复候选。
- 入口：[执行脚本](../empirical_analysis/beijing_air_quality/run.py)、[冻结协议](beijing_air_quality/FROZEN_PROTOCOL.md)。产物：[最终报告](beijing_air_quality/BEIJING_EMPIRICAL_FINAL_REPORT.md)、[汇总](beijing_air_quality/summary_metrics.csv)、[预测](beijing_air_quality/predictions.csv)、[最终验收](beijing_air_quality/FINAL_VALIDATION.json)、[封存索引](beijing_air_quality/DELIVERY_INDEX.json)。

## 5. UCI270：任务卡17，已放弃

- [任务卡17](../任务卡/任务卡17.md)以数字类别1—6分别回归浓度；13,910记录、10批、128原始列，动态96列严格映射16×3×2。开发训练1—6/1—7/1—8批评价7/8/9批，最后训练1—9评价10批；八方法，滚动内层最近至多三合法边界，至少二个。
- **实际完成192外层模型，63,816条留出预测，其中最终48模型、28,800预测（6×600×8）**；最终验收记录数值数据/映射/模型重放/预测复算均PASS，拟合共享耗时2,349.2秒。这里是“数值工作完成”，不是数据语义接受。
- 官方编号文字与官方气体×批次数表、本地DAT存在44/60单元冲突，涉及批1—9。唯一计数签名置换仍不构成权威名称证据；ZIP无README。用户已拒绝联系、放弃候选。
- 匿名留档的final ABESS数字如下，单位按原记录ppmv保留，不赋化学名称：

| 原始gas_id | RMSE | R² |
|---|---:|---:|
|1|83.8146|−0.002258|
|2|651.438|−60.6024|
|3|278.582|0.285909|
|4|78.8277|0.097996|
|5|280.396|0.276578|
|6|38.7898|−0.308229|

- 六类均未超过同96维最优竞争方法；不据此修改算法、名称、批次或选择有利类别。现有带气体名称的报告/六图是未认可的历史草稿，即使图形自动QA或模型重放通过，也不能作为已验收实证对外使用。
- 只读入口：[匿名记录](uci270_concentration/ANONYMOUS_RESEARCH_RECORD.md)、[最终状态](uci270_concentration/FINAL_VALIDATION.json)、[60项计数复核](uci270_concentration/CLASS_COUNT_RECONCILIATION.md)、[权威映射调查](uci270_concentration/MAPPING_AUTHORITY_INVESTIGATION.md)、[科学审查](uci270_concentration/SCIENCE_REVIEW.md)。原执行代码在[研究入口](../empirical_analysis/uci270_concentration/runner.py)，仅供留档，不代表获准重新运行。

## 6. Presentation汇报交付

目录`Presentation/recent_work_20261024`虽带旧日期，最终修订验收日期是2026-10-26。采用既有4:3 CambridgeUS/dolphin模板，PDF **15页**，包含算法v5→v5.2差异、模拟得失及四套实证比较；不包含后来已放弃的UCI270。

- [汇报PDF](../Presentation/recent_work_20261024/近期工作汇报.pdf)
- [LaTeX源码](../Presentation/recent_work_20261024/近期工作汇报.tex)
- [通俗工作报告](../Presentation/recent_work_20261024/近期工作报告.md)
- [验收记录](../Presentation/recent_work_20261024/qa/FINAL_VALIDATION.json)
- [交付索引](../Presentation/recent_work_20261024/DELIVERY_INDEX.json)

验收记录：15页均已视觉审查、无overfull/缺字/越界/Logo遮挡，最低实际字号约5.012pt，正式源码32项与北京封存396项保持不变；本次交接只是读取这些既有验收证据，不声称重新跑全部验收。新版明确四套实证均无足够比较优势、北京不是主实证；不要只看旧北京Q6。图中重复OOF是重复均值±样本SD，北京未来测试和Mushroom块外pooled不虚构SD。

本次只读交接另实测：PDF、TeX、Markdown三份交付当前文件大小与SHA256均匹配原验收JSON；本交接48个相对链接全部存在。未重新编译PDF，也未重新跑模型或封存全目录哈希。

## 接手约束

保留负面结果与失败/中断历史，不覆盖原封存报告。读取指标前先核对数据集自己的方法ID映射、协议、单位与分母；不要把训练预测算成OOF、把相同样本重复算作独立样本、把“数值重放PASS”当“预测优势PASS”。后续若要新数据/新科学协议，须是新的明确任务；本交接不授权继续UCI270、联系外部人员或重新拟合旧实证。
