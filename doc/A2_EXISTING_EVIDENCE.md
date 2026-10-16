# A2 已有证据复核（任务卡4 Phase A）

核验日期：2026-10-11。状态：**Phase A 已完成；可进入预先冻结的新 A0/A2 配对实验，尚不批准 A2 正式发布。** 本次只读复核任务卡3已有结果，未重跑训练、未改写历史结果、未访问旧 `test_tools/` 目录，也未改动生产或研究源码。

## 1. 版本关系与变化边界

|版本|用途|核验结果|
|---|---|---|
|v5-original|历史基线|fixed BCD 实现与 v5.1 不同，保留为历史版本|
|v5.1-bcd-fix|当前生产版本|`src/tensor_abess/` 的全部32个 Python 文件与隔离 v5.1 逐字节一致|
|v5.2-candidate-search|候选搜索研究版本|相对 v5.1 仅更改 `tensor/abess.py`、`tensor/bcd.py`、`utils/init.py`，增加 `tensor/candidate_search.py`；不得直接替换生产|

对 `tensor_bcd` 的 AST 比较确认 v5.1 与 v5.2 完全相同，保留 cold Group-BESS 探索、完整 OLS proposals、accepted loss 非增记录和 best visited state 返回逻辑。Group-BESS、OLS/张量运算、Group Lasso、MSIC 模块及其他原有核心文件逐字节一致；DGP 未因 A2 改动。v5.2 额外接口与记录用于隔离 caps/grid 策略、候选库和显式训练缓存。

A2 配置是 `cap_policy="legacy"`、`grid_policy="full"`，不是放宽 caps。旧 caps 仍由训练 GLasso 支撑大小、`floor(sqrt(n)/2)` 下限及逐步压缩矩形容量确定；容量仍为 `Q=max(1,floor(0.7n))`。在本确认集 n≥100 时即 `floor(0.7n)`。

## 2. 候选集、初始化、adaptive BCD 与缓存

完整整数网格的实际定义为

`{(s1,s2,s3): 1≤sm≤capm，s1*s2*s3≤Q}`。

`legal_size_triples` 使用整数范围的笛卡尔积及产品容量过滤；没有真值输入、重复形状或隐式截断。候选数超出固定预算2000时显式报错，不按真支撑或测试表现裁减。对90个 A2 已保存网格，独立构造上述集合并逐一比对：全部一致、无重复、无遗漏、无越界，`excluded_legal_shapes=[]`、`grid_complete=true`；每组还保留一个 adaptive 候选，因此候选库记录数为网格数+1。

“完整”仅表示**训练生成的旧 caps 内全部容量合法整数形状**，不表示覆盖全部维度合法形状，更不表示枚举所有支撑位置。固定形状候选仍由训练边际 slice scores 初始化，再调用原 fixed BCD/Group-BESS 搜索位置；A2 没有增加随机多起点或以真支撑初始化。

90对 A0/A2 的 caps、GLasso 初始支撑、adaptive-stage 的支撑、B指纹、RSS/loss/score、loss histories 与初始支撑均逐值相同。legacy 分支的 adaptive mode cap 恒等于旧 cap；新增动态容量约束只用于 `cap_policy="candidate"` 的 A1/A3/A4，未影响 A0/A2 的 adaptive 行为。A0 与 A2 共同的10,170条 fixed候选的支撑、B指纹、RSS/loss/score、best_iteration、loss histories、边际初始化也逐值一致。

缓存通过调用参数 `fixed_candidate_cache` 显式传递，键包含训练 X/y 的 shape/dtype/bytes SHA256、sizes、max_cycles、m_max、k_max 和固定求解器标识；不使用全局缓存。一个数据集按冻结 A0→A1→A2→A3→A4 顺序共享固定拟合，adaptive 与 GLasso 每组独立执行。候选记录复制支撑、B和历史，缓存不改变拟合数学定义，但会改变组间观测时间。

MSIC 保持：mode/stage alpha=1、gamma=1；global refine alpha=1.5、gamma=1；one-SE scale=1。one-SE 容差仍根据 minimum-score 模型的 q 计算，再在合格模型中优先较小 q。故扩充候选不保证最终训练 RSS、q或预测 PMSE 单调改善。对450个已保存候选库重新计算原 MSIC 选择：选中B指纹、支撑、selection完整记录与历史输出一致。

## 3. 训练依赖边界

|步骤|允许依赖|实际核验|
|---|---|---|
|GLasso筛选、caps、adaptive、边际初始化、固定候选拟合|训练 X/y、维度、n、冻结求解配置|核心训练调用只收到训练 X/y；没有 B_true、supports_true 或测试数组|
|网格与容量|训练生成的 caps、Q、预设 grid_policy/预算|真支撑形状不参与候选生成；legacy与full使用相同caps|
|MSIC与one-SE选模|训练 RSS、sizes、n、dims、冻结 alpha/gamma/scale|原公式与参数未改，450组重新选模一致|
|真形状/位置入池诊断及支撑评分|已生成的候选、模拟真值|发生在训练返回后，仅作评价，未反馈训练|
|独立测试 PMSE/R²|训练选定B、相同冻结测试流|所有组训练完成并选模后才调用 shared test；90组输入/真值/测试指纹一致|

模拟 DGP 本身必须知道真支撑以生成 X/y/B；这不等于把真支撑提供给估计算法。已有确认集已被查看，本任务后续不能把这些种子再称为全新独立确认。

## 4. 已有90组独立确认的实际重核验

数据源：`numerical_simulation/outputs_candidate_search_study/confirmation_final/`。已直接读取90份 request与job JSON，校验 plan/request校验和、全部核心源码指纹、90份NPZ的SHA256、完成状态、冻结组顺序、环境、训练/测试组间指纹、500条完整测试流及450组原 MSIC记录。调用已有 `validate_result`、`summarize_reports`、`build_overview`、`build_pairs` 的只读函数，不调用会新写历史目录的 CLI。当前Git可与历史Git不同，因此使用历史plan本身及其内容指纹验证，不以当前HEAD替换历史来源。

重算总览与已保存 `mechanism_overview.csv` 的全部数字在 `1e-12` 容差内一致。全部89,259条 fixed记录通过 accepted loss 非增、returned loss=min visited、best_iteration、容量、支撑位置合法性、RSS=2n*loss检查；这包含共享cache重复记录，不是89,259次独立拟合。A2自身固定候选记录13,230条，单数据集合法形状数64—252。

确认配置：dims=(32,20,25)，真支撑=(4,3,5)，rho=(.3,.3,.3)，sigma=1，n_test=500；训练 GLasso criterion=bic、c=3，k_max=1、max_cycles=6，其他配置见冻结plan。这些是该研究配置，不能表述为公共接口全部默认配置。

|指标|A0|A2|A2−A0|
|---|---:|---:|---:|
|Exact Recovery|6/90|26/90|+20/90|
|Mode-F1均值|0.601345|0.629553|+0.028209|
|Element-F1均值|0.419402|0.469973|+0.050572|
|PMSE均值|3.1184043135|3.0827333250|−0.0356709885（约−1.14%）|
|R²均值|−0.081552|−0.067966|+0.013586|
|q_hat均值|49.377778|45.788889|−3.588889|
|q>60|42/90|22/90|−20/90|
|q<60|41/90|41/90|0|
|真形状在caps内可达|60/90|60/90|0|
|真形状进入候选库|30/90|60/90|+30/90|
|真支撑位置进入候选库|13/90|37/90|+24/90|
|候选记录数均值|114|148|+34|
|观测elapsed均值/秒|172.929273|136.798997|受共享cache顺序影响|
|估计独立成本均值/秒|172.929273|194.350190|+21.420917，约1.124倍|

配对PMSE：A2胜25、平64、负1；平均差−0.035671。估计独立成本=观测elapsed+命中缓存fixed候选的原fit_seconds，不是独立冷启动计时；不能声称 A2 因观测elapsed较小而更快。

### 分场景复核（每行10个独立重复）

|n|SNR|A0 Exact|A2 Exact|A0 PMSE|A2 PMSE|A0估计成本/秒|A2估计成本/秒|A0/A2候选数|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|100|0.5|0/10|0/10|3.153125|3.153125|22.600|32.254|65/65|
|100|2|0/10|0/10|5.111532|5.111532|25.989|25.037|65/65|
|100|4|0/10|0/10|9.675140|9.675140|23.306|20.952|65/65|
|200|0.5|0/10|0/10|1.678657|1.678657|95.310|85.890|126/126|
|200|2|1/10|1/10|2.408526|2.408526|139.661|110.556|126/126|
|200|4|5/10|5/10|1.983509|1.983509|123.175|99.307|126/126|
|400|0.5|0/10|0/10|1.455477|1.360038|314.258|413.292|151/253|
|400|2|0/10|10/10|1.296675|1.183930|475.377|612.076|151/253|
|400|4|0/10|10/10|1.302998|1.190142|336.688|349.787|151/253|

n=400、SNR=2/4 的 A2 恢复优势直接复现：20/20 对 A0 的0/20。n=100 caps=(4,4,4)不允许真形状(4,3,5)；n=200 caps=(5,5,5)，旧grid已完整；n=400 caps=(6,6,7)，旧grid缺102个合法形状，A2补全。这解释收益主要集中在 n=400，不构成跨支撑形状或相关结构推广证据。n=100/200候选相同时估计时间仍不同，说明GLasso/adaptive运行波动和cache/计时开销存在，不能把小场景时间差当搜索算法优势。

已保存数值审计覆盖450个最终所选 OLS设计矩阵：全部满秩，最大condition=17.332759，最大RSS相对重算差=6.087e−16。此处复核审计文件及其来源指纹，未重新生成全部训练矩阵，不能推广为所有候选设计矩阵均已数值审计。另有已保存历史27组 A0 与 v5.1 的B逐值相同、训练/测试指纹相同、7项指标最大差0的复现报告。

## 5. 风险、放行与证据指纹

**没有发现阻断新的冻结 A0/A2 配对实验的源码错误或历史结果完整性问题。** 现成 A2 是可审查的最小搜索网格变更；新实验应继续使用原MSIC、相同训练/测试数据和求解设置，提前固定新种子、其他真支撑形状、预算和判据。必要时使用每组独立缓存/无缓存以获得实际冷启动时间；不得把历史共享cache估计解释为实测。

正式采用仍受以下证据限制：已有真支撑只有(4,3,5)；恢复增益主要来自n=400旧grid遗漏；n=100仍受caps限制；总体PMSE改善小且存在一个负例；候选位置并未穷举；全部候选数值条件未被审计。A3池的MSIC调参结果不能替代A2上的独立证据。上述属于统计适用范围与后续验收边界，不是本次源码阻断。生产继续v5.1，700正式任务与三个实证重训均未由本复核启动。

证据来源冻结Git：`d4e28969ac480ba6efc01bcc985a53d179e7ed31`；Python3.12.8、NumPy1.26.4、SciPy1.16.1；确认种子 `20461010+1000*scenario+rep`，独立测试种子+1,000,000，9场景×10重复。

|指纹对象|SHA256 / ID|
|---|---|
|确认plan_id（规范化内容）|`be7b227228164e6dbbf3d7305e93717c62128e6982f5ff36000a1f1218fd973c`|
|确认plan.json文件|`847270b5294d507fb60b69e72748c02083a29370b5d58d22d0fb5931b09d2305`|
|integrity_report.json|`618b9b75eceb3b65eeed7e0f6785a2db3ea4013042daf16ddefb676d303def17`|
|raw_results.csv|`f61c2d1aac91b92439f1c707899be457dee219da2f89d63155ac7627c626d8fb`|
|mechanism_overview.csv|`a8ae2109135f5c09a9c81771f7a373d7f32767250296a7df01b66ecd14bfcc9b`|
|mechanism_scenarios.csv|`53be74c1a8e7809a7167dc29bf03e943671ee550b7bc400d43c7c68a69314dde`|
|numerics_diagnostics.json|`e335e1e81bb1cee8cf7fe15db14e7837208cf78910d8e101b0d50c3cc54edff9`|
|v5-original源码清单|`2fafb22468886cb9144fe64d409ee9fa72bb11210ec6576dd1c045cd08bff372`|
|v5.1及生产源码清单|`f82bbfa5d31e997afdafd6a7d01471aa86e024b75560173d543c174d9ff3229e`|
|v5.2研究源码清单|`c018974e7214bad539da60986f93a27892ed7703752fffdfbe6c18a23664063f`|

源码清单指纹定义：相对 `tensor_abess/` 的Python路径映射到各文件SHA256，按键排序、JSON separators=(',',':')、UTF-8编码后再SHA256；不是单个文件的指纹。90份JSON/NPZ的逐任务指纹及request IDs保留在原plan和job中，本次逐份校验通过，不复制或改写历史证据。
