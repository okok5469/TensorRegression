# 任务卡18：身份与输入审计

状态：**IDENTITY_PASS**；检查时间（UTC）：2026-10-26T10:16:13.020855+00:00。

正式 `v5.2-search-final + 原MSIC` 的32生产源原字节SHA256全部匹配release，实际导入路径为 `C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main\src\tensor_abess\__init__.py`；正式计划38项源码/配置指纹及plan seal全部匹配。未改manifest、旧源码或旧结果，未拟合、重建输入、merge或提交集群。

## 封存结果与重复映射

700份JSON及对应NPZ通过冻结读取器的记录seal、计划/配置/保存环境、NPZ哈希、数组可读性、支撑及REE/RSS关系检查；四方法2800行和56汇总单元均完整，各标签每方法50次。全部CSV字段与JSON或纯聚合结果比较通过，数值误差上限分别为 raw=0、summary=0。

四项训练指纹联合test_seed得600套唯一完整输入：550个单例，50套在scenario6/9/12重复三次。仅X_train指纹为550，不能替代完整输入计数；10种参数配置中基准合并150套，其余9种各50套。`input_identity_map.csv`保留700标签、种子、配置ID、完整输入ID、canonical任务和别名；不把复用基准重复计为独立证据。固定pilot为scenario `[0,1,2,4,5,7,8,10,11,13]` 各replicate0、1。

## 输入证据的准确范围

NPZ含B_true、train_noise、真支撑及四方法系数/支撑，对应真系数与噪声指纹已逐个核对；X_train、y_train只有原保存指纹，没有归档数组。测试流有明确test_seed和冻结生成器/分批约定，没有原测试数组或其指纹。本入口不重建输入，因此不声称测试流逐字节恢复；新补算须由独立输入恢复步骤核对可用指纹并声明测试配对核验范围。

正式计划ID为 `30f231df0327e58331a18580dc8a03caeb0872dc991e57c42b095556100a80f6`；完整32/38指纹、原文件SHA256、保存Linux环境与本机环境分列于JSON。身份通过不替代Lasso/Group Lasso各必要子问题的KKT与原始—对偶间隙证书。

## 实际环境与既有变更

本次Python为 `3.12.8 (tags/v3.12.8:2dc476b, Dec  3 2024, 19:30:04) [MSC v.1942 64 bit (AMD64)]`，NumPy `1.26.4`，SciPy `1.16.1`；实际BLAS、模块路径、可导入求解器和四项线程环境详见JSON。历史Linux Python3.9.25/NumPy2.0.2/SciPy1.13.1单独保留，未冒充相同环境；本入口不设置线程或激活可选runtime。已有 `doc/taskcard13/runtime` 的可用依赖单独列出，不等同当前已导入求解器。

初始只读观察为312个tracked删除、215个未跟踪Git状态项、3个修改（doc三份日志）；这是本轮实现前的原始语境，未跟踪项不等同文件数。当前Git快照另存JSON，本轮taskcard18/paper新增另列，不覆盖既有改动。Git遍历旧test_tools可能产生的目录警告原样保存在JSON；任务科学输入读取未受影响。根目录无四类流程文件。

## 历史最终验收

读取并指纹绑定卡10续跑最终验收：开发48/48，复用30、新18，确认0/48；读取卡12最终验收：正式380（360主组+20额外压力），压力场景总30含主组10，另有3个smoke。本入口不重复历史拟合或归档全量审计，不把早期待运行记录当新任务。

## 复验

在仓库根目录运行 `python -B -m research.taskcard18.identity`，默认仅读取并输出计数；`--write`仅更新本卡上述三份派生审计文件。限定边界测试为 `python -B -m unittest research.taskcard18.test_identity`。基线求解、输入重建及章节验收由各自入口处理。
