# 任务卡18：写作结构、证据及自审

日期：2026-10-26。本文记录正文写作依据，不代替任务卡独立审查或G7最终验收。

## 1. 路由、论证与读者

已应用nature-writing：task=manuscript，paper_type=algorithmic，section=method/experiments，journal=generic。输入语言为中文；用户要求中文正文，覆盖技能zh-to-en的英文优先惯例。任务卡18已经明确研究对象、边界、交付结构和直接执行授权，未重复要求设计审批。

论证句：在三阶矩形稀疏回归中，正式Tensor-ABESS以旧caps内完整整数规模和有限逐模位置搜索构造模型，再按原MSIC及确定性容忍带选择；正式定义可由源码复现，比较优势须以完成数值验收的对照及同输入配对结果为准。

主要读者为希望理解和复现该算法的统计/计算研究者。方法章回答“程序是什么、怎样执行、保证到哪里”；模拟章回答“在何种生成分布上、用哪些信息和指标、证据是否足以比较”，不以实现正确性代替泛化优越性。

## 2. 已读取技能与缺失资源

读取了C:/Users/xmx/.codex/skills/nature-writing/SKILL.md、manifest.yaml、三个本地core（stance/workflow/output-format）、manuscript任务片段、algorithmic类型片段、method与experiments章节片段、zh-to-en语言片段、generic期刊片段，以及references/method.md。

manifest要求的四个always_load文件在该安装及C:/Users/xmx/.codex全树搜索中均未找到：

- ../nature-shared/core/reader-workflow.md
- ../nature-shared/core/paper-type-taxonomy.md
- ../nature-shared/core/ethics.md
- ../nature-shared/core/terminology-ledger.md

experiments还要求的../nature-shared/core/main-text-discipline.md属于同一缺失shared资源树。本轮未声称完整读取这些材料，也未编造其规则；依据已读写作规则和用户明确任务完成不受影响的中文正文。已向主执行者报告该技能安装缺口；它不是统计G2/G3通过或失败的替代判断。

## 3. 章节图与术语

| 正文章节 | 唯一主要任务 |
|---|---|
| 方法2.1 | 定义模型、理想组合目标和预处理责任 |
| 方法2.2 | 初始化、有限ISTA、数据依赖caps与初始不裁剪 |
| 方法2.3 | 分组归约、交换机制与实际OLS接受 |
| 方法2.4–2.5 | 区分adaptive终态与Cold固定规模best-visited |
| 方法2.6 | 原最终MSIC、精确容忍带及确定性顺序 |
| 方法2.7–2.8 | 伪代码、停止作用边界、复杂度、实际模拟配置与限制 |
| 模拟4.1–4.2 | DGP、C-order协方差、SNR、场景及600/700重复关系 |
| 模拟4.3–4.4 | 对照定义、信息使用、精度、指标与配对区间 |
| 模拟4.5 | 已核实正式ABESS描述及尚缺的新主比较 |
| 模拟4.6 | 版本解释、复现边界与单一DGP限制 |

术语固定为“正式v5.2-search-final + 原MSIC”“矩形支撑”“矩形容量”“实际元素支撑”“声明模式支撑”“投影模式支撑”“完整整数规模”“Cold固定规模搜索”“best-visited完整模型”“确定性容忍带”。具体记号见THEORY_INTERFACE.md，没有把矩形容量写成实际非零数或选择后自由度。

## 4. 结果材料的去向

| 材料 | 功能 | 正文/附录安排 |
|---|---|---|
| 正式ABESS配置内的预测和恢复差异 | 科学边界/条件性描述 | 正文4.5，注明来自封存正式算法 |
| 新可靠对照的效应量和配对CI | 主比较所需证据 | 尚缺；正文明确占位，不填造数值 |
| 14标签的历史结果及旧Lasso/GLasso | 历史追溯 | 附录/历史表，明确旧实现、未认证 |
| KKT/对偶公式、数值预算、边界 | 可信度与复现细节 | 协议附录 |
| hash、全部任务身份和失败现场 | 工程来源 | doc/taskcard18及独立缓存，正文仅保留入口 |
| 敏感性阈值、完整规模分布 | 稳健性/诊断 | 待验收后附表；不抢占主比较叙述 |

## 5. 主张—证据自审

| 主张 | 证据 | 状态 |
|---|---|---|
| 正式流程及公式对应生产实现 | 已逐项读取entry/init/glasso/group/reduction/ops/bcd/sic | 定义支持；待独立G7审查 |
| v5.2完整覆盖caps内整数规模 | _candidate_size_triples及最终候选循环 | 支持；不等于位置穷举 |
| 固定规模返回最佳已访问完整状态 | tensor_bcd同步保存B/support/loss | 支持；不等于全局最优或更好泛化 |
| 正式模拟使用BIC、c3、6轮、k1 | 冻结plan及实际fit_tensor调用 | 支持；与公共默认分列 |
| C-order协方差为Sigma1⊗Sigma2⊗Sigma3 | 逐模Cholesky计算与reshape列顺序 | 支持；已避免照抄生成器注释的反向顺序 |
| 700标签/600完整输入/10配置 | 身份审计、输入映射 | 支持；数值重放范围另述 |
| 正式ABESS的四因素条件性描述 | results/historical_config_summary.csv中adaptive行 | 支持；不能作为修复对照后胜负 |
| 新对照改善或改变方法排名 | 未有600×2完成并验收的结果 | 缺失，不作方向性断言 |
| 一致性、最优率、任意缩放等价性 | 本卡不新增理论 | 未证明，不作主张 |

自审已修正文稿inline数学分隔符；同时检查候选顺序、首个argmin容量、带内(q,I,规模和)顺序、tau仅作用adaptive、初始化不裁剪、有限预算与矩形内零系数等容易误写之处。五份文稿的数学分组、inline/display定界符及Markdown表格列数检查通过；本项目没有已有论文LaTeX工程，因此未声称完成模板编译或排版渲染。正文与附件均显式标未定稿，不将文件存在等同于所有Gate通过。第二轮只读审查指出的声明/投影模式定义已拆分，算法其余公式及停止规则未发现实质矛盾；完整独立验收仍以INDEPENDENT_REVIEW.md为准。
