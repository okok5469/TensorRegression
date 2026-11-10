# UCI270任务卡17：复现入口

从仓库根目录运行以下PowerShell命令，保留既有Python和依赖版本。当前目录是实际运行产物，**不能覆盖现有数据、合同、模型或原生重算证据**。下面的从零步骤仅用于经过授权的全新复现目录；现有结果只执行校验或按相同合同续跑。

```powershell
$env:PYTHONPATH="$PWD\doc\taskcard13\runtime;$PWD\src;$PWD"
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$python='.\.venv\Scripts\python.exe'
$run='doc/uci270_concentration'
```

## 原始数据与测试

完整DAT及ZIP位于 `D:\数据集\gas+sensor+array+drift+dataset+at+different+concentrations` 及同名 `.zip`。原始文件身份、ZIPCRC与副本一致性见 `raw_manifest.json`；`SOURCE_AUDIT.md` 是字段语义证据。重建时把输出改成不存在的新目录；不要将下面步骤用于覆盖当前 `$run`。

```powershell
& $python -m empirical_analysis.uci270_concentration.data --source 'D:\数据集\gas+sensor+array+drift+dataset+at+different+concentrations' --archive 'D:\数据集\gas+sensor+array+drift+dataset+at+different+concentrations.zip' --output 'doc/uci270_concentration_reproduction' --source-audit 'doc/uci270_concentration/SOURCE_AUDIT.md'
& $python -m unittest discover -s empirical_analysis/uci270_concentration -p 'test_*.py'
```

如进行新复现，将 `$run` 指向新目录，先复制或重新核验必要来源文档，再冻结。运行前应确认新复现已获授权，不能把重复运行当成新的独立实证。

## 冻结、分阶段拟合与独立验收

```powershell
& $python -m empirical_analysis.uci270_concentration.runner --root $run --stage freeze
& $python -m empirical_analysis.uci270_concentration.runner --root $run --stage pilot
```

freeze仅允许尚无 `contract.json` 的目录；合同封存数据、配置、75个依赖文件、正式32源码、环境、实际导入路径和24个完整划分。Pilot是第一个合格gas的dev1八方法，本次为gas1。

独立审计使用Python函数入口。原生Pilot核验会真实重拟合一次，只在新复现中执行一次，并把耗时计入共享拟合预算；当前已有 `audit_native_pilot.json`，不得再运行这一行：

```powershell
& $python -c "from empirical_analysis.uci270_concentration.audit import audit_native_pilot; audit_native_pilot(r'$run')"
& $python -c "from empirical_analysis.uci270_concentration.audit import verify_all; print(verify_all(r'$run','pilot')['status'])"
& $python -m empirical_analysis.uci270_concentration.runner --root $run --stage dev
& $python -c "from empirical_analysis.uci270_concentration.audit import verify_all; print(verify_all(r'$run','dev')['status'])"
& $python -m empirical_analysis.uci270_concentration.runner --root $run --stage final
& $python -c "from empirical_analysis.uci270_concentration.audit import verify_all; print(verify_all(r'$run','final')['status'])"
```

审计通过后由 `verify_all` 生成绑定合同和报告哈希的阶段锁。dev必须有 `PILOT_ACCEPTED.json`；final必须先有 `DEVELOPMENT_ACCEPTED.json` 且开发任务完整，不能手写假锁。最终批次成绩不得先行用于改参数、划分、表示或停止决策；final只评价一次。验收锁证明检查项目通过，不代表所有候选数值收敛或性能领先。

## 汇总、报告、图和交付封存

以下仅读取保存结果，不产生新模型；完整图要求192个任务齐全，缺失时会拒绝绘制。使用函数可明确指定复现目录。

```powershell
& $python -c "from empirical_analysis.uci270_concentration.analysis import collect; collect(r'$run')"
& $python -c "from empirical_analysis.uci270_concentration.audit import verify_metrics; print(verify_metrics(r'$run'))"
& $python -c "from empirical_analysis.uci270_concentration.report import build; build(r'$run')"
& $python -m empirical_analysis.uci270_concentration.plot_results --root $run --skill-scripts 'C:/Users/xmx/.codex/skills/nature-figure/scripts'
& $python -c "from empirical_analysis.uci270_concentration.finalize import seal; print(seal(r'$run'))"
```

最终结论见 `UCI270_EMPIRICAL_FINAL_REPORT.md` 与 `FINAL_VALIDATION.json`，不能根据README命令存在就认定后续阶段已完成。每任务目录 `models/gas{g}_{split}_{M}` 保存pickle模型、含train/test标记的预测、内折记录、诊断及文件封存哈希；主结果汇总只用test行。

## 续跑与边界

在同一合同下再次执行原阶段，runner校验并复用已完成模型，仅执行尚未开始的任务；Pilot复用不计新模型。单任务失败、超时、残留 `RUNNING.json` 或不完整任务目录会明确停止，须先审计原因，不删除证据或覆盖重跑。`fit_budget.json` 累计全部模型子进程耗时；另加 `audit_native_budget.json` 的原生拟合时间，合计不得超过14400秒。最多一个实际worker、每worker线程1；没有自动集群提交。

模型API：`fit_task(X96tensor, raw128, y, train_indices, test_indices, inner_pairs, method_id, config=None)`；索引均为全局行号，内折必须属于外训练。`predict_model(artifact, X96tensor, raw128)` 支持独立重放。不要直接调用API绕过统一预算、最终封存或当前合同。
