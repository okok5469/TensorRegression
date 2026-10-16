# 三版本效果拆分：结构证据与未完成数值归因

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

P0的250标签方案已冻结，但**严格配对拟合0/250，未完成**。本机Python3.12.8/NumPy1.26.4/SciPy1.16.1重生正式task0时，仅B_true指纹相同，X_train、y_train、train_noise均不同；保存的原环境为Linux/Python3.9.25/NumPy2.0.2/SciPy1.13.1/OpenBLAS0.3.27。没有用相同seed、近似数值、改dtype或替换数据冒充严格配对，也未安装或升级依赖。

事实证据：[预检原记录](../numerical_simulation/outputs_version_attribution/cluster_run/preflight_task_000000.json)、[冻结250计划](../numerical_simulation/outputs_version_attribution/cluster_run/plan.json)。计划指明task0–49、50–99、250–299、350–399、550–599，各50标签，共250种完整输入；没有新提交700任务。早期根目录plan/preflight是工具定稿前的失败预检留存，实际后续协议使用cluster_run子目录；不覆盖先前证据。

因此不能填造本批v5→v5.1的Δ1和v5.1→v5.2的Δ2数值、胜平负或成本。以下结构检查与已有独立研究可以限定解释，但不能计为这250次已运行。

完整32源码比较中v5.1/v5.2仅bcd.py不同，移除_candidate_size_triples后两文件AST完全一致。直接执行两份冻结源码的网格函数，核对**候选集合和顺序**，结果如下：

| 训练n / caps | v5.1网格数 | v5.2网格数 | 顺序相同 | 真规模在全局fixed网格：旧/新 |
| --- | --- | --- | --- | --- |
| 100 / [4, 4, 4] | 64 | 64 | True | False / False |
| 200 / [5, 5, 5] | 125 | 125 | True | True / True |
| 400 / [6, 6, 7] | 150 | 252 | False | False / True |
| 116 / [4, 4, 5] | 80 | 80 | True | True / True |

候选表还包含原adaptive结果，故所选候选池数量为网格数+1；表中是整数规模网格数，不是支持位置枚举数。

n=100的50次实际caps均(4,4,4)，真第3模规模5在两版都不可达；此处A2没有新增规模、次序也相同，不能把旧新预测恶化归因于“新增候选”。n=200被选诊断场景均caps(5,5,5)，同样已有完整网格。n=400实际caps(6,6,7)，原网格缺少第2模规模3，A2把真规模纳入全局fixed规模网格；原池还含adaptive结果，因此不能断言所有旧候选池均排除真模型。位置搜索与评分选择仍是独立问题。

LEMON n_train=116时，直接提取并执行原初始化代码中的caps生成AST，在全部6324种可能初始化支撑长度组合下均得(4,4,5)，两版有序网格均为80。这里也没有A2新增规模。结合31份其他core一致与BCD其余AST一致，可排除“网格增加”作为这些条件下版本差异的解释；这是源码等价性推断，**不是已实际完成v5.1的250次拟合或LEMON v5.1重跑**。

详见[结构审计](taskcard6/structural_grid_audit.json)。

在原计算环境恢复P0：同步新增version_attribution模块、三版冻结源码/身份及原两批计划与所需任务JSON，保持正式源码及原结果不变，使用原已验证环境并在计算资源内执行。复制已冻结cluster_run/plan.json，不复制Windows失败preflight到新的集群诊断目录；源文件需保留原字节，不能改计划身份。

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B -m numerical_simulation.version_attribution.run preflight --task-id 0
# 只有四项数据指纹与完整环境均通过，才继续：
python -B -m numerical_simulation.version_attribution.run pilot
# 五场景各首个任务的真实拟合Gate全部通过后，补跑其余245任务：
python -B -m numerical_simulation.version_attribution.run run
python -B -m numerical_simulation.version_attribution.run merge
```

该工具在每个独立进程先绑定v5.1完整源码与实际import，再检查四项输入指纹/原环境；错配即在拟合前失败。结果采用新文件且拒绝覆盖/残缺断点；五场景Gate缺少任一个则拒绝批量运行；merge要求完整250。生产src没有临时切换，v5/v5.2不重新拟合。当前只验证了工具契约与真实失败预检，不能将其称为原Linux集群运行已验证。

