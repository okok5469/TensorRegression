# 任务卡10追加执行复现与成本

原900秒/7200秒阶段及其NOT_RUN、报告、FINAL原样保留；本次依据用户继续授权追加计算协议，2worker、单case3600秒、开发与确认共享28800秒时钟。科学方法、原96请求、九门槛、候选库及选择规则不变。原30例结果按字节复用，重算原18例未完成请求，不将复制或重试计作新科学重复。

使用原.venv/Scripts/python.exe与原NumPy/SciPy，四个线程变量均为1；未安装依赖。新调度器resume.py复用原experiment.worker和Job Object控制器，CLI依次prepare、run-dev及满足条件后的run-confirm。已有目录拒覆写；不得把当前命令直接再次运行作为新科学重复。

[新协议](PROTOCOL.json)、[新计划](../results/continuation_01/plan.json)、[30例复用来源](../results/continuation_01/REUSE_MANIFEST.json)、[原失败记录](../results/formal/stopped_development.json)、[追加共享时钟](../results/continuation_01/formal_budget.json)、[逐例实际资源](analysis/fit_resources.csv)是成本与身份依据。原阶段整段1938.703秒，含task030超时900.016秒及3个同伴中止的时间；这些失败进程未保存完整CPU峰值，不能伪造累计CPU。成功case wall之和是并发任务时间累计，不等于整批墙钟；报告分别保留原批墙钟、追加批执行日志及成功case CPU/wall/峰值。

| 成本项 | 秒（内存行除外） | 口径 |
|---|---|---|
| 原开发整批（30完成+4中止） | 1938.703 | 包含失败开销，4worker |
| 原task030超时进程 | 900.016 | 原控制器实测；不得与整批墙钟相加 |
| 原task031–033中止进程 | 未单独记录 | CPU/峰值及各进程终止墙钟不可推造 |
| 续跑development整批 | 4432.141 | COMPLETE；2worker |
| 续跑development成功case墙钟累计 | 8801.487 | 并发任务累计，不是批墙钟 |
| 续跑development成功case CPU累计 | 5498.656 | 失败CPU不在此项 |
| 续跑development成功case最大峰值内存（MiB） | 394.199 | 单进程峰值，不是同时总内存 |

外部测试流重放沿用原预选12个开发/12个确认ID，并在新SCOPE仅映射原30及新18来源。Pilot4外部重放不重复执行；新结果不能扩充独立科学样本数。全部独立矩形预测、输入/测试流指纹及最大误差以外审记录为准。

[外审范围](external_audit/SCOPE.json)、[开发外审汇总](external_audit/development/SUMMARY.json)、[原Pilot外审](../external_audit/pilot/SUMMARY.json)。

原897项交付、索引本身及5922历史文件共6820文件受保护；新科学源码与旧源码哈希仍须匹配。最终[数值与身份验收](FINAL_VALIDATION.json)及[追加文件索引](DELIVERY_INDEX.json)封存本次结果；旧FINAL和索引不改。
