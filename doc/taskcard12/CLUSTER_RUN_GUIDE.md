# Task12 集群运行与结果回传

当前交付为脚本、冻结计划与本地检查；真实远端准备、smoke、380例模拟均 `NOT_RUN`，科学结论为 `PENDING_CLUSTER_RESULTS`。不需要用户手工拼接多行命令，每个代码块都是一行。以下项目根来自实际 `CLUSTER_GUIDE.md`；新建其下的隔离 `taskcard12_bundle/`，不覆盖正式源码或任何旧输出。

## 1. 上传并校验新包

将 `doc/taskcard12/delivery/taskcard12_cluster.zip` 和 `taskcard12_cluster.zip.sha256` 上传到 `/public/home/xjh/xmx/tensor-sparse-regression-main/`。使用平时已确认的传输工具与SSH目标；`xmx@mu01`只证明曾出现的登录提示，不能当作已确认外网地址。

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
```

```bash
sha256sum -c taskcard12_cluster.zip.sha256
```

```bash
test ! -e taskcard12_bundle && unzip -n taskcard12_cluster.zip
```

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle && sha256sum -c SHA256SUMS
```

校验失败或目标目录已存在时停止，保留文件与终端错误，不重复覆盖、不自动清理。所有后续绝对路径均进入新bundle。

## 2. 绑定实际环境和冻结计划

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh prepare
```

prepare在登录节点只检查源文件、既有myenv数值环境、请求和种子，并保存新的 `numerical_simulation/outputs_msic_pair_task12/plan.json`；不会生成科学数据或拟合模型。它也在sbatch前建立日志父目录。重复prepare拒绝覆盖现有计划，后续使用status。源、环境或请求不符时保留现场，不降低校验、不安装升级环境。

## 3. 先提交独立三例smoke

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh smoke
```

该命令实际执行数组 `0-2%3`，分别覆盖n100弱信号、n200重点强信号和n400强信号；每例仅一次完整Cold候选搜索。记录返回的真实Job ID。每项1CPU、无自动requeue，原生输出及错误日志在bundle内 `numerical_simulation/msic_pair_task12_logs/%A_%a.out/.err`。

```bash
squeue -u "$USER"
```

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh status
```

队列为空和complete标记数都不代表完整验收。等待三项实际退出后运行Gate：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh gate
```

Gate核验三份完整结果、真实Slurm数组身份、源/环境/输入/共享候选与所有文件哈希，并保存成本。固定成本门槛为最慢smoke时间乘1.5不超过3600秒；预估CPU总量为 `1.5*(120*t100+140*t200+120*t400)`，除64仅是不计排队、64槽全部可用时的理想下界，不能作为已验证的集群用时承诺。Gate失败则停止正式提交；不删困难场景、减少重复或抬高超时。

## 4. Gate通过后提交固定380项

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh formal
```

入口先复核smoke Gate，再提交 `0-379%64`，每个正式worker再检查同一Gate。正式360例是36场景各10次；额外20例仅加入n200/SNR4/(4,3,5)/rho0.6，使重点场景共30例。主均值仍按平衡360例计算，额外20例不改变主统计权重。smoke不并入380例。

```bash
squeue -u "$USER"
```

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh status
```

作业详细日志还保存在每项结果目录内的 `stdout.log`、`stderr.log`、`execution.json` 和真实Slurm身份记录。科学case超时由拥有该进程组的控制器终止并返回非零，失败现场保留。

## 5. 缺失、失败与恢复

仅对status确认从未启动的 `missing_ids` 提交，例如确实缺失7和8时才用下行；数字不是建议重跑列表：

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/submit_msic_pair.sh retry 7,8
```

已有完整项必须通过源/环境/计划、所有artifact及共享候选检查后才显示 `VERIFIED ... skipped`，不再生成数据或重新搜索。部分结果、损坏文件、正在运行项或失败标记会非零停止；不要删目录、改plan、补complete或人工造Gate。需要诊断时保留对应目录和Slurm错误日志，先确认作业状态再处理。

完整性检查还会对候选独立重评分，并从已保存的真值与预测数组复算PMSE、signal_mse、REE、支持恢复指标和候选池可恢复证据。预测文件明确保存A0/A1列顺序与模型哈希；复核过程不重新生成测试集。仅修改报告数字再更新文件清单不能通过验收。

如站点支持，可将下行12345改为真实Job ID查看退出原因；本轮没有实际调用sacct，不保证本站启用：

```bash
sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed
```

## 6. 全部完成后严格合并

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/research/taskcard12/cluster/merge_and_validate.sh
```

merge要求全部380项身份和文件完整、无缺失或多余目录，A0/A1完全配对；任何缺失或损坏使程序非零退出，不汇总可完成子集。输出 `merged/paired_metrics.csv`（760规则行）、`selection_changes.csv`、`summary.json`、冻结统计分析及 `integrity.json`。合并输出也拒绝覆盖，不把汇总完成等同于批准替换正式算法。

## 7. 回传给Codex

把bundle下整个 `numerical_simulation/outputs_msic_pair_task12/`、`doc/taskcard12/CLUSTER_PROTOCOL.json`、包内 `FILE_MANIFEST.json`及Slurm日志目录下载到本地一个新的目录。完整结果包含每项候选池、选中模型、预测、输入身份、计时、文件哈希以及所有负面/失败记录，不只下载均值CSV。

可先回传 `merged/`、`plan.json`、`smoke_gate.json`和失败/Slurm日志用于快速检查；最终源码与模型级独立验收仍需要完整每项证据。不要把大量旧结果或脑数据混入本次回传。未收到真实380例结果前，报告继续使用 `PENDING_CLUSTER_RESULTS`，不生成模拟成绩或最终科学推荐。
