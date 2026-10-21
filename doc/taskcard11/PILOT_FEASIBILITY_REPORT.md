# 工程 Pilot 与正式预算门禁

6/6 完成，失败0。所有案例只运行一次正式v5.2候选搜索，每例A0精确恢复原结果，两个n100库各65候选、四个n400库各253候选，共1142个完整矩形候选接受独立OLS核验。Pilot只检查正确性和成本，不进入96例新验证，不据其PMSE修改规则。

| case | n | 完整墙钟秒 | 独立候选审计秒 |
|---|---:|---:|---:|
| 0 | 100 | 12.875 | 0.268 |
| 1 | 100 | 12.422 | 0.250 |
| 2 | 400 | 148.625 | 2.038 |
| 3 | 400 | 156.265 | 2.253 |
| 4 | 400 | 147.360 | 1.992 |
| 5 | 400 | 146.016 | 1.583 |

按事前协议，取n100最大12.875秒、n400最大156.265秒，96例两worker预计墙钟含1.5倍余量为6149.04秒，低于28800秒；最慢case乘1.5为234.3975秒，低于3600秒，预算门禁通过。该估算不保证未见场景耗时上界；正式运行仍执行原硬限，不能减重复、删困难场景或自动延长预算。

正式阶段固定24场景×4新随机输入，不增加开发筛选或第二轮确认；Pilot和历史数据均排除。全部seed在生成前核对28个历史计划、2010个已保存种子，Pilot与validation共204个train/test种子互不重复且无已知历史冲突；完整输入指纹在每个worker搜索前再次核验。

启动前独立review发现当前批次可读到peer尚未写完的身份JSON；在任何新科学输入生成前完成测试先行修复，只将当前output排除于历史扫描，同批次仍由独占input claim拒绝重复。未执行的 `PILOT_PROTOCOL.json` 原样保留，实际使用重新封存的 `PILOT_PROTOCOL_READY.json`，科学定义、seed、DGP、预算和门槛没有改变。

旧 `research/taskcard10/cv.py` 的参数映射始终在生成前保护清单中且哈希不变，Pilot协议未单列该依赖；独立核验见 `verification/legacy_options_identity.json`，正式协议将其显式加入依赖清单。没有修改该依赖、补写数据或重跑Pilot。

证据：[预算门禁](PILOT_BUDGET_GATE.json)、[Pilot计划](results/pilot/plan.json)、[完整批次](results/pilot/completion.json)、[46项正确性](CORRECTNESS_GATES.json)、[独立启动审查](verification/startup_review.json)、[固定设计](FROZEN_VALIDATION_PROTOCOL.md)。
