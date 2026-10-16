# v5.2正式模拟提交与预检

状态：本地真实task0四方法已完整执行，JSON/NPZ与resume字节一致验证通过；这不是SLURM，详见[LOCAL_PREFLIGHT.json](releases/v5.2-search-final/LOCAL_PREFLIGHT.json)。真实集群预检NOT_RUN，700正式任务未提交，完整模拟未完成。上传命令见[部署指南](CLUSTER_DEPLOYMENT_GUIDE.md)。

复用原known_s_cluster.run和冻结full配置；14场景×50=700数据任务，adaptive/oracle/lasso/glasso四方法=2800结果。新root shell已准备prepare/run/status/merge，submit使用sbatch；CMO_HPC队列、xjh账户、每任务1CPU、最多64并发，不升级既有环境。

## 首次部署及真实单任务

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
module load anaconda3
conda activate myenv
mkdir -p numerical_simulation/v52_search_final_full_logs
bash scripts/release_v52/verify_cluster.sh "$PWD"
bash run_v52_search_final_full.sh prepare
# 只提交真实task0预检，不提交700：
sbatch --parsable --array=0 run_v52_search_final_full.sh | tee v52_preflight_submission.txt
squeue -u "$USER"
```

prepare只用于首次空新目录；计划生成于真实集群并绑定raw源码/环境，不拷本机计划续跑。task0的SBATCH输出/错误在numerical_simulation/v52_search_final_full_logs；脚本另保存实际SLURM JobID/arrayID/count、版本、plan、模型记录与worker日志hash。

等待单任务确实成功退出后：

```bash
bash run_v52_search_final_full.sh status
bash scripts/release_v52/preflight_gate.sh "$PWD"
```

Gate核验真实单任务元数据（array count=1/task0）、四方法完整JSON/NPZ、源/环境/plan、日志指纹及严格resume跳过且文件字节不变；本机普通task0没有该元数据，不能被标为集群通过。预检失败或NOT_RUN时禁止全量提交。Gate只能首次生成，不覆盖；已生成后用下一节gate命令复核，不能靠重写身份救补。

## 用户确认后提交完整700

```bash
# 前提：真实预检通过，用户已确认正式提交：
python -B scripts/release_v52/release_control.py gate --root "$PWD"
sbatch --parsable run_v52_search_final_full.sh | tee v52_full_submission.txt
squeue -u "$USER"
bash run_v52_search_final_full.sh status
```

完整数组0–699%64。新root入口除真实单独task0外都强制检查已通过的Gate，缺Gate会停止且不会开始拟合。正式输出numerical_simulation/outputs_v52_search_final_full，日志numerical_simulation/v52_search_final_full_logs；旧v5/v5.1与预检本地目录保留。

## 缺失任务恢复

```bash
bash run_v52_search_final_full.sh status
# 只读取status核实后的无锁缺失ID；例如经确认的ID串，不猜编号：
task_retry_ids='填写status中的rerun_task_ids逗号列表'
sbatch --parsable --array="${task_retry_ids}%64" run_v52_search_final_full.sh | tee v52_retry_submission.txt
```

恢复同一不可变plan，不重复prepare。已完成任务严格resume核验后跳过；锁、半文件、损坏、源码/环境不符先保留现场并停止，不能删锁、改旧身份或自动救补。输出的缺失ID若含0也不等于自动通过预检Gate；完整/恢复作业仍须已有真实Gate。

## 完整验收后汇总

```bash
bash run_v52_search_final_full.sh status
python -B scripts/release_v52/release_control.py complete --root "$PWD"
bash run_v52_search_final_full.sh merge
```

complete要求700完整、无未解除锁、每任务四方法合计2800，再交原merge生成不可覆盖汇总。此批没有fixed，空的known-s比较文件不能报告为完成known-s比较。下载命令与新旧比较/异常q验收见[部署指南](CLUSTER_DEPLOYMENT_GUIDE.md)及[验收计划](V52_SIMULATION_ACCEPTANCE_PLAN.md)。

本轮只做本地源码/脚本/路径及协议测试，没有实际ssh、sbatch、远端环境验证、700任务或三个实证。提交前RemoteRoot/SSH/myenv初始化/队列权限和共享文件系统os.link支持都需目标现场确认。
