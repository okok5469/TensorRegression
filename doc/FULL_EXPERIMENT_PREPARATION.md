# 完整数值模拟与实证准备（任务卡4）

状态：2026-10-11 A/B已完成，100组配对、200个所选OLS及16次独立冷启动全部通过，A2搜索推荐采用；80组独立MSIC确认触发3600秒预算停止、证据不完整，按冻结规则保留原MSIC；最终推荐D2/v5.2-search-final，157项限定测试通过，发布待用户批准。已有D2候选源码与同步/full脚本包；当前生产仍为v5.1-bcd-fix，未启动700正式任务或实证重训。以下命令均为最终批准和目标环境验收后的入口，不代表已执行。

## 当前生产身份与接口

当前Git HEAD为d0c3547704fbaf5d90a3e5423568768a1ba11759；v5.1版本清单的历史base_commit为d4e28969ac480ba6efc01bcc985a53d179e7ed31，二者用途不同。现场核对src中32个Python文件均与doc/algorithm_versions/v5.1-bcd-fix/VERSION.json对应raw SHA256一致。源码冻结以该32文件清单为准，不能只用Git提交名称证明身份。

|文件|现场raw SHA256|
|---|---|
|src/tensor_abess/tensor/abess.py|fc0a87f201f2f6aa18f62667d881cddfddc5d969b70c7b83cf95213c5cc7ee45|
|src/tensor_abess/tensor/bcd.py|c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737|
|run_bcd_fix_full.sh|b4b005ea34169c37e1c46a5b69fd31f2e57f4e7d2a33544a33096305dc094af9|
|numerical_simulation/known_s_cluster/run.py|5fff973f646a889ec6e9a8af70f0e775c182f8cda6a473b2d644af5dd42ce9ee|
|numerical_simulation/known_s_cluster/profiles.json|76465458701ad6da176fd93d688d1b4762285a9cf391360737e0730a35c3fa94|

BCD规范LF SHA256为4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c，现成sh按该规范校验，跨Windows/Linux行尾不要求raw字节相同。known_s计划内部仍记录现场raw字节，生成计划后不得改变代码或行尾。

公共入口仍是tensor_abess(X,y,*,target_sparsity=None,k_max=None,tau=None,eps=1e-8,c=2,criterion='cv',n_lambdas=20,cv_folds=5,max_iter=500,tol=1e-6,adaptive_criterion='msic',msic_refine_alpha=1.5,msic_refine_gamma=1,msic_refine_one_se_scale=1,max_cycles=30,m_max=15)。tensor_bcd(X,y,s1,s2,s3,*,max_cycles=30,m_max=15,k_max=None,init_strategy='marginal',random_state=0,S1_init=None,S2_init=None,S3_init=None)参数未变；v5.1仅新增结果诊断字段。实证必须target_sparsity=None，真实规模只用于known-s诊断，真实位置仅传Oracle。

## D1：现成700任务入口

源配置为known_s_cluster/profiles.json的full，14场景×50重复。五组含baseline1、snr4、sample_size3、dimension3、correlation3；不是五组各700任务。

|配置项|冻结值|
|---|---|
|基准|n_train200、n_test500、dims(32,20,25)、sparsity(4,3,5)、rho(.3,.3,.3)、SNR2、sigma1、continuous支撑|
|SNR|.5、1、2、4|
|样本量|100、200、400|
|维度|(16,12,15)、(32,20,25)、(64,30,40)|
|相关性|0、.3、.6（三模相同）|
|优化|k_max1、max_cycles6、n_lambdas20、cv_folds5、max_iter500、tol1e-6、BIC初始化c3|
|MSIC|mode alpha1；global alpha1.5、gamma1、one_se_scale1|
|训练种子|baseline 2026+rep；其他组12026+100×组内序号+rep，rep=0…49|
|测试种子|训练seed+1000000；test_batch_size100|

run_bcd_fix_full.sh的四方法为adaptive/oracle/lasso/glasso，共2800方法结果；固定known-s对照不在该批。另一个run_known_s_full.sh比较fixed/adaptive/oracle，输出outputs_known_s_full_50，不能把两批方法数或目的混写。

集群保持Python3.9.25、NumPy2.0.2、SciPy1.13.1、既有myenv，四个BLAS/OMP/NUMEXPR线程均1。现成sh使用module load anaconda3及source activate myenv，并在执行前严格校验三版本；本次未在集群验证激活方式。requirements.txt仅为环境记录，禁止据此执行安装或升级。共享文件系统须支持os.link原子提交。

先核对集群完整32核心及numerical_simulation/{__init__,data_generation,methods,metrics}.py，再按差异增量同步run_bcd_fix_full.sh、known_s_cluster/{run.py,profiles.json}和v5.1 bcd.py；不能仅因新plan存在就认为集群已匹配本机验收源。正式计划在集群创建，不能拷贝本机预检计划续跑，因为plan严格绑定Python/平台/BLAS/线程及raw源指纹。

最终决定D1且同步验收后，首次运行：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_bcd_fix_full.sh prepare
sbatch run_bcd_fix_full.sh
```

prepare只创建不可覆盖计划并报告状态；sbatch为0–699%64，每任务1CPU，最多64并发。全部数组结束后：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_bcd_fix_full.sh merge
```

正式输出numerical_simulation/outputs_v51_bcd_fix_full，日志numerical_simulation/v51_bcd_full_logs。续跑用同一sbatch命令，不重复prepare；完整任务须经过--resume严格核验才跳过，损坏/缺件明确失败，不自动补改。merge检查完整700任务、方法/指标/身份及锁后生成raw_results.csv、summary_results.csv、paired_fixed_vs_adaptive.csv、search_gap.csv和integrity_report.json；该四方法批没有fixed，后二者相应配对不适用，不应称已做known-s比较。

本机已有outputs_v51_final_full_preflight仅1/700任务，四方法预检与严格resume见doc/algorithm_versions/full_preflight_report.json及final_integrity_report.json；它不是完整正式模拟。目标outputs_v51_bcd_fix_full本机尚不存在，集群状态本次未访问。

## D2：现成待审批包与700任务入口

若最终采用A2，独立命名/批准前仍research。不能把outputs_v51_*或历史v5结果改成v5.2/v6标签，也不能原样运行run_bcd_fix_full.sh后宣称A2：它锁定v5.1 BCD和src默认旧搜索。known_s_cluster.run.fit_tensor未传A2策略，source_fingerprints固定扫描repository/src；仅把PYTHONPATH改为研究快照会使导入源码与记录源码不同，不能作为有效版本冻结。

已制作[doc/team_sync/v5.2-search-final/README.md](team_sync/v5.2-search-final/README.md)、MANIFEST.json、search-final.patch和run_v52_search_final_full.sh；对应候选32源码位于doc/algorithm_versions/v5.2-search-final/src。只改bcd.py私有规模函数为旧caps内全部容量合法整数三元组，保留公共参数/默认值与原MSIC。6项静态/代表同数据桥接、29项最终候选BCD/模拟pytest通过；patch在普通LF/CRLF副本实际check/apply/reverse-check、32规范LF源码比对、Bash语法及mock组件核验通过。它们不等于真实集群执行通过，候选仍research_only=true、release_approved=false。

最终批准D2后，先按VERSION/MANIFEST核对集群目标完整32源码v5.1基准、共享numerical_simulation源、known_s_cluster入口和依赖；按包README将search-final.patch应用到目标src并核对最终32规范LF指纹，将run_v52_search_final_full.sh复制到项目根。此时实际导入与plan扫描同时来自repository/src，不能仅切PYTHONPATH指向研究快照。BCD候选raw SHA256为db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7，规范LF为eb4f01626e0108f4cf87a604835aab7fa99866f484abaea23ce54b8911f063d0；计划在目标集群生成后保持源码/行尾/环境不变。

D2沿用上表full配置、DGP和种子，旧caps/full grid、mode alpha1及global alpha1.5/gamma1/scale1；14场景×50重复=700数据任务，adaptive/oracle/lasso/glasso四方法共2800模型结果。现成脚本保持既有Python3.9.25/NumPy2.0.2/SciPy1.13.1、myenv、四线程各1、Slurm 0–699%64/每任务1CPU；不安装或升级依赖。脚本运行前核验BCD规范LF与环境，完整32源码和共享入口身份仍须按manifest另行核对。研究预算2000不加入公共API，后续样本规模超出本轮n≤400时须预先检查候选计数/资源成本。

最终批准并完成目标身份/环境验收后，首次执行：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_v52_search_final_full.sh prepare
sbatch run_v52_search_final_full.sh
```

全部数组完成后执行：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_v52_search_final_full.sh merge
```

输出numerical_simulation/outputs_v52_search_final_full，日志numerical_simulation/v52_search_final_full_logs；与历史目录物理隔离。prepare只用于首次不可覆盖建计划，续跑同一sbatch命令，严格resume验收完整任务后才跳过；merge须检查700任务完整。不得拷本机计划到集群续跑，不覆盖或重标旧模型。本轮未执行这些命令，集群激活方式、Python3.9/NumPy2真实运行与资源/共享文件系统仍待目标现场核验。

## D3：须另行冻结与验证

80组独立MSIC确认因预算停止而未完成，原计划禁止续跑，不支持D3；本次推荐D2和原MSIC。未来如另行授权新的研究及D3版本，须重新冻结与验证源码、默认参数、完整指纹、公共调用/拟合桥接、patch及独立full脚本，并使用独立版本目录；不得将D2改标v6或扩大本次冻结预算。最终身份以FINAL_ALGORITHM_VERSION.md为准，发布前不启动正式计算。

## 历史结果复用的逐项条件

所有采用新搜索/MSIC的Tensor-ABESS正式五组结果必须完整重算；原v5/v5.1模型、B、支持、q和缓存不能换标。known-s固定规模若求解路径完全相同可作为明确标旧版本的诊断参照，仍不能替代新版本adaptive正式结果。

Oracle、Vector Lasso、Tensor-GLasso仅在方法实现与依赖/求解配置、DGP及rho/dims/n/snr/系数生成、train/test数组指纹、种子/test批顺序、评价指标口径全部相同且逐任务完整通过时可以复用。旧10重复只覆盖严格核对后的对应seed，不补成50重复；task0数值重放相符不是所有场景已核验，runtime不能跨环境当作公平成本。对照方法因全core哈希变更不能直接移入新plan假装新计算，复用应保留来源版本与逐条来源清单/校验身份。

实证同步和训练折重搜要求见FINAL_TEAM_SYNC.md。三个实证的被试/响应/预处理/划分保持，所有训练折预处理重新拟合；任何新算法都须重搜支撑、重新OLS与预测，不能复用旧B或旧选阶。

## 本机测试与完成边界

以下Windows命令用于D2冻结候选的复验；python先核对为本轮已有Python3.12.8及其NumPy/SciPy环境，不预设存在.venv。pytest只使用新隔离工具目录，不安装到全局或集群：

```powershell
$env:PYTHONPATH = "$PWD\doc\algorithm_versions\v5.2-search-final\src;$PWD\doc\algorithm_versions\taskcard4_test_tools"
python -B -m unittest tests.test_bcd_fix_regression -v
python -B -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py tests/test_numerical_simulation.py -q
```

本轮6项候选桥接与29项候选pytest已有通过记录；以上复验必须使用独立新进程避免模块混用。Linux在最终批准、目标src同步后用既有环境export PYTHONPATH="$PWD/src"，可运行python -m unittest tests.test_bcd_fix_regression numerical_simulation.known_s_cluster.test_run -v，不要求pytest，不升级依赖。已有相关限定测试及Python3.9 AST静态核验不代表全仓库或集群NumPy2实际运行通过；最终任务卡4测试清单与MSIC决策以FINAL_ALGORITHM_VERSION.md为准。
