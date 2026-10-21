# 任务卡12回传验收：原环境核验通过，阻塞已解除

**当前状态：`NATIVE_READONLY_REVIEW_PASS`，原环境严格核验阻塞已解除。** 用户已回传在原 Linux 集群、原隔离包、原 myenv 环境并设置四个线程变量为1后成功执行的终端输出：

```text
NATIVE_READONLY_REVIEW_PASS: smoke=3 formal=380 summary_and_analysis_match
```

这项原环境验收由**用户终端输出证实，非 Codex 在 Windows 本地执行**。成功输出是在原环境的 `verify_plan`、`require_gate`、3例 smoke 与380例 formal `collect`、完整 `summary` 与冻结 `analysis` 精确相等检查均通过后产生；本命令没有重训、生成新 DGP、重新 merge 或写回结果。用户随后省略四线程变量的命令仅因环境前置条件不符而失败，不能据此撤销先前正确环境下的成功核验。

回传目录为 `numerical_simulation/outputs_msic_pair_task12`。独立协议及保存链审核已通过：380例正式结果、3例smoke、冻结源、请求、环境记录和产物哈希均已核验，详见 `INDEPENDENT_PROTOCOL_AUDIT.json`。

**保留的历史阻塞：** Windows 严格数值重放曾在 smoke/task_000 停止，两个评分中间量与原 Linux 记录最多相差 `8.881784197001252e-16`；所选模型、分数、容忍带及候选集合一致，但冻结程序要求完整 selection 字典精确相等。主线当时停止后续本地数值验收，没有放宽门禁。证据仍为 `ARCHIVE_INTEGRITY.json`、`SMOKE000_RESCORE_DIFFERENCE.json`；原环境成功复核补齐了该验证路径，不把本地失败改写为通过。

已核验统计显示主360例 PMSE 下降19.65%、n100下降28.70%，但11项新门槛仍有4项失败；压力30例 Exact 由12降至8。因此 [最终版本建议](FINAL_VERSION_RECOMMENDATION.md) 为 **B**：原 MSIC 继续作为已发布 v5.2 的正式默认，FS只保留冻结研究选项，停止新增调参/开发。工程验收通过不等于恢复风险门槛通过，也不批准 FS 新发布。

以下保留实际正确环境所需的**四线程变量单行只读命令**，供证据复查，不要求再次执行。应从原 `taskcard12_bundle` 目录、已激活的原 `myenv` 环境运行；不要重新 merge 或覆盖已有 merged。如果终端停留在 `>` 提示符，先按 Ctrl+C 取消未结束的输入。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B -c "from pathlib import Path; from research.taskcard12 import simulation as s; from research.taskcard12.analysis import analyze; repo=Path.cwd(); out=repo/'numerical_simulation/outputs_msic_pair_task12'; plan=s.verify_plan(repo,out); s.require_gate(repo,out); smoke=s.collect(out,'smoke'); records=s.collect(out,'formal'); assert len(smoke)==3 and len(records)==380; assert s.summarize(records)==s.read_json(out/'merged/summary.json'); assert analyze(records,plan['protocol'])==s.read_json(out/'merged/statistical_analysis.json'); print('NATIVE_READONLY_REVIEW_PASS: smoke=3 formal=380 summary_and_analysis_match')"
```

未修改科学源码、公式、阈值、结果文件或正式版本。未来若复查产生新不一致，须保留报错并停止，不能静默放宽验证条件。
