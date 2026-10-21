# 独立外部数值重放：固定范围与Pilot验收

这是已有选中模型的数值复核，不产生新模型、不修改选择、不将重复生成的相同数据计作新的科学确认。实现位于[verify_external.py](../../../research/taskcard10/verify_external.py)，纯算术/范围门禁位于[test_verify_external.py](../../../research/taskcard10/test_verify_external.py)，4/4通过，见[测试验收](../external_unit_validation.json)。所有原始results及冻结拟合模块保持只读。

## 固定范围

[SCOPE.json](SCOPE.json)将脚本、测试、两份原实验plan及精确请求封存。选择只依赖样本量、SNR、真形状、rho及replicate，不读取误差、恢复结果或完成先后。

| 阶段 | 精确task ID | 例数 | 范围 |
|---|---|---:|---|
| Pilot | 0,1,2,3 | 4 | 原已完成的n100/n400×弱/强信号，原rho0.6、真形状435 |
| 开发 | 0,4,8,12,16,20,24,28,32,36,40,44 | 12 | 每n×SNR×真形状的rho0、replicate0 |
| 确认（仅实际运行时） | 0,4,8,12,16,20,24,28,32,36,40,44 | 12 | 同一预声明范围，确认种子族独立 |

总上限28例。每个阶段必须全部选定实例已有complete.json，才能开始再生；不读未完成实例的部分文件，不修补缓存，不因为缺失或不利结果更换case。若确认因开发门槛失败而未运行，相应重放同样不运行，不补充其他case凑数。

## 每例核验

先核对plan/protocol、实际源码及原case全部artifact SHA和选型封存。随后只用原seed及冻结DGP重建训练输入，精确检查X、y、真B、真支撑指纹与保存truth数组；原500观测测试流也以原test_seed及原batch size重建，X/y/signal流指纹必须完全相同，保存的测试y/signal逐元素相等。

三臂B来自各自已封存选择，检查其指纹后，独立列出矩形支撑设计并计算预测。该路径不复用worker的完整张量dense矩阵乘法、预测汇总函数或选阶函数；逐观测预测与保存值在数值容差内一致，PMSE及signal-MSE再独立汇总核对。当前实际加载的所有tensor_abess模块也必须来自冻结v5.2并匹配SHA。没有支撑搜索、调参、候选池测试排序或重新选型。

## Pilot实际结果

[Pilot汇总](pilot/SUMMARY.json)显示4/4通过、12个选中模型通过，合计2000个测试观测及6000个三臂预测。训练/真值指纹、保存测试y/signal、测试流及选中B身份全部精确匹配；矩形独立预测的最大绝对差为5.3291×10⁻¹⁵，PMSE/signal-MSE最大绝对差为8.8818×10⁻¹⁶，耗时10.288秒。新增支撑搜索0、选择变更0。

逐例证据：[task000](pilot/task_000.json)、[task001](pilot/task_001.json)、[task002](pilot/task_002.json)、[task003](pilot/task_003.json)。这些是Pilot原数据的重复数值核验，不能并入开发效果或独立确认。正式开发/确认如实际执行，分别以对应阶段的SUMMARY.json记录实际覆盖；本方案不将12个rho0实例的复核外推为全部48例都独立重建过测试流。

## 复现接口

本轮终态：开发在task030触及900秒单case上限后停止；预选12例有task032/036/040/044未完成，开发整段外部重放为[NOT_RUN](development_NOT_RUN.json)，没有重放其余8例替代完整范围。科学确认和相应外部重放同为[NOT_RUN](confirmation_NOT_RUN.json)。下列开发/确认命令是接口记录，本轮不再调用。

使用已有解释器及四个线程变量均为1。已有输出拒绝覆盖；已完成本轮prepare及Pilot run，不应直接再次运行相同命令覆盖证据。

```powershell
.venv/Scripts/python.exe -B -m research.taskcard10.verify_external --stage prepare
.venv/Scripts/python.exe -B -m research.taskcard10.verify_external --stage run --cohort pilot
.venv/Scripts/python.exe -B -m research.taskcard10.verify_external --stage run --cohort development
.venv/Scripts/python.exe -B -m research.taskcard10.verify_external --stage run --cohort confirmation
```

prepare只封存范围、不生成数据；run只接受预先固定的cohort与完成状态。审计结果是正确性证据，不替代原先冻结的预测、恢复、尾部与成本推广门槛。
