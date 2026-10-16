# v5.2正式模拟验收与结果下载

状态：仅准备验收；700个正式SLURM任务未提交，完整科学模拟尚未完成，不能把本机单任务预检或旧v5/v5.1数据称为新版正式结果。

## 冻结设计和完整性

复用numerical_simulation/known_s_cluster/profiles.json的full，显式n_reps=50；14场景×50=700数据任务。方法必须为adaptive/oracle/lasso/glasso，分别对应Tensor-ABESS、Oracle OLS、Vector Lasso、Tensor-GLasso，完整汇总为2800行方法结果。

五组为baseline、correlation、dimension、sample_size、snr；基准n200/n_test500/dims(32,20,25)/shape(4,3,5)/rho(.3,.3,.3)/SNR2/sigma1/continuous支撑，保留BIC c3/k_max1/max_cycles6等原full配置及原MSIC。任务0…699由原profile_tasks生成；训练种子baseline2026+rep，其余组12026+100×组内序号+rep，test_seed=seed+1000000。

多个实验组可能共用相同参数或种子，因此“无重复”是任务ID、场景ID/replicate及冻结seed映射无多写/漏写，不是错误地禁止设计允许的跨组相同seed。不得改原种子映射以制造独立样本。

验收必须通过原runner的plan checksum、完整32源码/6依赖raw指纹、配置/task映射、环境/BLAS/线程身份；源或环境改变使用新计划及新输出，不改旧断点身份。

## 完整任务和数值验收

- status核验700任务ID、JSON/NPZ完整配对及合法文件名、锁/缺失/残缺/损坏结果；已有任务仅严格resume检查通过后跳过。
- 每任务四方法唯一且齐全，保存场景/replicate/train/test seed、输入/测试指纹、B/支撑、模型指标及运行记录；不混用旧ABESS缓存。
- 训练/预测误差、R²、支撑F1/Exact、q及其它指标必须有限且定义正确；支撑边界、q与模型维度一致，OLS与保存RSS不矛盾。
- 记录q=1、接近容量上限、过选/欠选、失败和数值异常，不按坏结果删除任务或只汇报完成子集；异常按原数据/源码及既定指标核查。
- merge只在700完整、2800方法结果且无未解除锁/身份冲突时执行；原merge缺件会失败，不补造或复制旧结果。

正式独立输出为numerical_simulation/outputs_v52_search_final_full，日志为numerical_simulation/v52_search_final_full_logs。raw_results.csv、summary_results.csv和integrity_report.json作为主要下载目标；paired_fixed_vs_adaptive.csv/search_gap.csv在此四方法批不含fixed时不构成known-s比较证据。

## 版本比较与报告

新版正式总报告须引用新plan_id、发布完整源hash、真实环境、作业ID、700/2800完整状态及数值审计，与原v5/v5.1版本身份明确的结果比较。不能换旧结果标签、删除困难场景或把原10重复扩充成50重复。

对照方法如拟复用，先逐任务严格核对方法源码/依赖/参数、DGP、训练/测试数组、系数、seed/test批次、指标和环境，并保存原来源清单；仅task0一致或方法名相同不足以复用。默认当前新批仍实际计算四方法，不自动导入历史对照。

## 恢复和下载

缺失任务由status的rerun_task_ids决定；仅无锁且身份匹配时恢复，损坏/半文件不能被当成可跳过结果，先保留现场并查明原因，不自动删除锁或救补文件。重新提交同一计划的允许任务后，再做完整merge。

Windows只下载通过完整merge的汇总CSV/JSON与必要作业/版本日志；下载脚本默认dry-run，不默认传回巨大NPZ缓存。具体可复制下载/状态/恢复命令见[CLUSTER_DEPLOYMENT_GUIDE.md](CLUSTER_DEPLOYMENT_GUIDE.md)及[V52_SIMULATION_SUBMISSION.md](V52_SIMULATION_SUBMISSION.md)，本轮未下载真实集群结果。
