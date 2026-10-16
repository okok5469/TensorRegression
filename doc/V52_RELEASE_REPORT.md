# 任务卡5最终发布与交接报告

本地Tensor-ABESS v5.2-search-final（D2）已正式发布；用户授权为任务卡5，正式[RELEASE_MANIFEST](releases/v5.2-search-final/RELEASE_MANIFEST.json)标approved_by_user=true、release_status=released。实际生产32文件匹配冻结D2，只修改src/tensor_abess/tensor/bcd.py的完整合法规模网格；原MSIC、fixed BCD修复、旧caps、Group-BESS、OLS与公共接口保持。

## 本地发布及验证

50项发布范围测试通过（3生产身份/默认值+29实际生产BCD/模拟+6候选桥接+12LEMON），9项部署脚本与7项清理测试通过，合计66项限定测试。额外旧known-s联合命令有1项未改动旧shell的静态断言失败，已单列，不宣称全仓库通过；新root入口独立验证。Python3.9 AST、Bash语法、PowerShell真实dry-run/模拟Apply、中文/空格路径、最小白名单、backup/hash及partial下载拒绝通过，本机工具不替代真实集群。

真实本机新full计划只运行task0，四方法、JSON/NPZ与原runner验收通过，resume跳过且字节不变；计划仍1/700，699未执行，不执行merge伪装完整。真实SLURM、远端环境和700提交均NOT_RUN。

完整32指纹和6依赖/部署工具清单见正式manifest，BCD raw为db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7，canonicalLF为eb4f01626e0108f4cf87a604835aab7fa99866f484abaea23ce54b8911f063d0。已有HEAD d0c3547704fbaf5d90a3e5423568768a1ba11759不是新发布commit；大量既有未提交改动保留，因此未创建分支/worktree或混合提交，使用已验证[最小patch](team_sync/v5.2-search-final/search-final.patch)与独立发布清单交接。

## 清理与历史

审计后按精确dry-run只删除125个未跟踪再生pyc，释放1,382,690字节；原位置归档索引不移动任何研究目录。7068重要文件逐一复核SHA256差异0，历史v5/v5.1/v5.2源码和研究证据、原模拟、LEMON数据/结果及团队patch全部保留；不访问/清理HCP/NKI外部目录。

审计快照总目录净增703,514字节，因为新增正式/审计/本机预检证据，不能称整个目录净减少1.32MiB；未列入原计划的后续新缓存和其它有歧义文件保留。详细前后清单、Git状态、删除和原位置归档见[清理报告](PROJECT_CLEANUP_REPORT.md)。没有路径/代码导入变化或新增已跟踪删除。

## 集群交接

集群真实SHA尚未获取，不填推测值；完整[更新计划](CLUSTER_FILE_UPDATE_PLAN.md)列MUST UPDATE/VERIFY ONLY/DO NOT TOUCH。远端若确认v5.1，唯一核心更新bcd，另部署新launch/发布/Gate文件；其余31core和6共享模拟依赖必须逐项核验。不匹配更早版本或未知改动时停止，不盲目覆盖。

Windows显式-Apply才上传，先真实inventory/dry-run，备份/hash锁定白名单；旧结果不上传/覆盖。脚本和可复制上传、verify、prepare、真实task0、Gate、700 sbatch、status、恢复、merge、下载命令分别见[部署指南](CLUSTER_DEPLOYMENT_GUIDE.md)、[正式模拟命令](V52_SIMULATION_SUBMISSION.md)、[验收计划](V52_SIMULATION_ACCEPTANCE_PLAN.md)。真实集群预检通过且用户确认后才提交700；新输出为numerical_simulation/outputs_v52_search_final_full。

## 实证准备

三人统一正式32core，所有ABESS训练折重做预处理、支撑搜索、选阶、OLS与预测；旧B/支撑/系数不可换新版标签。LEMON已核对145人/8协变量、固定外内划分、仅协变量基线及实际CLI；旧EEG增量输出32core匹配v5-original，旧q80和RMSE/R²只作历史对照，不假定新版本改善。

[队友通知](TEAM_V52_RELEASE_NOTICE.md)和[三实证方案](V52_EMPIRICAL_RERUN_PLAN.md)给出LEMON新目录audit/run入口及HCP/NKI通用核验条件；未编造本仓库不存在的MRI CLI，未发队友消息或拟合任何实证。

## 用户下一步

填实际SSH目标、RemoteRoot及既有myenv Python路径，在集群取得清单并核验部署；完成真实SLURM单任务Gate后，再决定提交700任务。现阶段不需要重新批准算法版本，本地发布已按任务卡5完成。
