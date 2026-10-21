# 任务卡10复现与证据入口

本轮科学终态为 **STOPPED_PARTIAL_DEVELOPMENT**，不是完整开发或独立确认。Pilot4例完整完成，开发30/48例完成；task030达到冻结900秒单case上限，031–033同伴中止，034–047未启动。确认0例、NOT_RUN。原始中止目录及执行日志保留，不续跑、延长预算、覆盖或替换困难case。正式版本保持v5.2-search-final，研究代码仅在research/taskcard10，报告与结果仅在doc/taskcard10。

## 环境与运行记录

仓库根目录运行，沿用`.venv/Scripts/python.exe`（Python3.12.8、NumPy1.26.4、SciPy1.16.1），未安装依赖。每个数值进程使用以下设置：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
```

Pilot2worker，正式4worker。正式总时钟起于2026-10-18T02:12:24.450258+00:00（北京时间10:12:24），开发和确认共用7200秒总上限，单case900秒；开发控制器1938.703秒后按单case期限停止。总预算剩余不构成延长单case期限的授权。Windows Job Object强停止自有进程树的测试已通过；额外进程列表查询被OS拒绝，不更改权限，也不将该额外查询写成通过。

## 已执行入口与身份

下列为本次操作记录，已有输出拒绝覆盖，**不得在原路径再次执行**：

```powershell
.venv/Scripts/python.exe -B -m research.taskcard10.gates
.venv/Scripts/python.exe -B -m research.taskcard10.freeze --stage pilot --workers 2
.venv/Scripts/python.exe -B -m research.taskcard10.experiment prepare-pilot --protocol doc/taskcard10/PILOT_PROTOCOL.json --output doc/taskcard10/results/pilot
.venv/Scripts/python.exe -B -m research.taskcard10.experiment run-pilot --output doc/taskcard10/results/pilot
.venv/Scripts/python.exe -B -m research.taskcard10.freeze --stage formal --replicates 2 --workers 4
.venv/Scripts/python.exe -B -m research.taskcard10.experiment prepare-formal --protocol doc/taskcard10/FORMAL_PROTOCOL.json --output doc/taskcard10/results/formal
.venv/Scripts/python.exe -B -m research.taskcard10.experiment run-dev --output doc/taskcard10/results/formal
```

准确冻结入口参数以[freeze.py](../../research/taskcard10/freeze.py)及原协议为准；它只准备协议，不代替prepare或run。`run-confirm`本轮未执行，已有confirmation_NOT_RUN终态禁止启动。[控制器停止记录](results/formal/stopped_development.json)是开发终态依据，非零科学阶段退出不得改写为COMPLETE。

| 身份 | SHA256 |
|---|---|
| [Pilot协议](PILOT_PROTOCOL.json) | d63eee1fc539dac4858f8ba5dbde08dfd74e698cf7d4dd31acf8da4b6073de83 |
| [Pilot计划](results/pilot/plan.json) | 63588e9da1d26578737f333506d7c91afd196234aef4aefcb3fa8b1045ca85da |
| [正式协议](FORMAL_PROTOCOL.json) | 3bfb7ae1dd508e3d2b79e18612a7d2d48269fe1409a6902a24a5df5a15ec2c18 |

完整文件身份及实际核心加载路径在每例complete.json、协议source_identity及收尾[交付索引](DELIVERY_INDEX.json)。[正式计划](results/formal/plan.json)封存开发/确认各48请求及历史输入去重依据；主训练seed base分别为3101810000/3501810000/3901810000，test加50000000、split加100000000，三种角色联合唯一性检查通过后才生成。不能把旧plan复制到新目录重新拟合并称为新输入或新确认；未来科学重启须另立协议、预算及未见种子族。

## 保存结构与只读复核

每个科学case目录包含request、started、input_identity，以及完成时的selection_frozen、summary、arrays.npz、truth.npz、independent_audit、test_predictions.npz、complete。选择及训练内审计先于外测试生成。每折cache有pool.json、pool.npz、seal.json。execution_development目录是控制器进程日志，不能把其中complete标记当作科学case；完整模型必须按plan中的cohort目录读取。

[analyze.py](../../research/taskcard10/analyze.py)与[report_stopped.py](../../research/taskcard10/report_stopped.py)仅汇总已完成记录，不拟合或评估不完整的九项门槛。生成六张CSV及九份报告的入口为`python -B -m research.taskcard10.analyze`，本轮已执行，现有analysis拒绝覆盖。[verify.py](../../research/taskcard10/verify.py)从原结果独立核对精确覆盖、系数/缓存身份、REE/F1/PMSE、候选覆盖、六张CSV、协议与源码；最终验收入口为`python -B -m research.taskcard10.verify`，只允许收尾执行一次，生成FINAL_VALIDATION和DELIVERY_INDEX。后续复核可导入其只读函数，对显式指定的现有case执行检查，不调用拟合函数、不修改原记录。

外部测试流重放另有预先封存的[固定范围](external_audit/SCOPE.json)：Pilot4例；开发与确认各12例，ID为0/4/8/12/16/20/24/28/32/36/40/44，总上限28。只复现已选B对应原训练输入/测试流和独立矩形预测，不搜索或重选。Pilot4例已通过；开发预选范围有4例未完成，整个开发重放NOT_RUN；确认重放也NOT_RUN，不能换成已完成8例或增补有利case。详见[外审说明](external_audit/README.md)。

## 结论边界

[正确性46项](CORRECTNESS_GATES.json)与Pilot成功不是预测推广门槛成功。[开发部分结果](DEVELOPMENT_RESULTS.md)中n100覆盖完整24例，n400仅6个弱信号例；总体及最坏已见cell不得代替24场景完整门槛。[确认未运行](INDEPENDENT_CONFIRMATION.md)，因此只有暂定研究建议，不能方案A、替换v5.2或发布新版本。A1−A0混合范围缩小及adaptive来源删除，A2−A1才是同库选阶差；同形状多来源风险排序不可识别。OLS数值通过不等于估计风险小，P1/P2仅为独立后续方案。

[最终决策](TASKCARD10_FINAL_DECISION.md)、[最终验收](FINAL_VALIDATION.json)、[历史保护核验](PROTECTED_FINAL_CHECK.json)构成收尾入口。保护核验覆盖5922历史文件；工作流仅更新doc内三文件，不创建分支、worktree、提交或外部通信。
