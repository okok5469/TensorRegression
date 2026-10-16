# v5.2-search-final：待审批的D2增量部署材料

当前为已验证、待用户最终批准的D2候选包，release_approved=false；100配对、16冷启动及200所选OLS已完成，157项限定测试通过，最终保留原MSIC。80组新MSIC确认触发3600秒预算停止，未完成且禁止续跑，不批准D3。当前生产仍v5.1-bcd-fix；本包没有替换生产、提交代码、发送队友或执行正式计算。名称含final不表示正式发布，最终依据doc/FINAL_ALGORITHM_VERSION.md及本包MANIFEST.json的批准状态。

## 内容与基准

- search-final.patch：以已验收v5.1为基准，仅修改src/tensor_abess/tensor/bcd.py的私有规模枚举，将旧caps内候选改为全部合法整数三元组。
- run_v52_search_final_full.sh：从既有v5.1完整模拟脚本派生，增量复制到集群项目根后使用；仅更新版本/BCD规范LF身份/独立输出和日志，其余环境与full配置保留。
- MANIFEST.json：32源码raw/canonicalLF指纹、基准和补丁指纹、普通副本实际apply核验、LF/CRLF与Bash静态核验、当前批准边界。

独立候选src位于doc/algorithm_versions/v5.2-search-final/src，生产src未改变。31文件与v5.1相同，唯一bcd差异只改变候选规模网格；固定BCD、原caps初始化、adaptive主体、Group-BESS、OLS、MSIC、公共tensor_abess/tensor_bcd参数保持原样。无需携带A3多策略、研究cache或candidate_records接口。最终选阶仍按原global alpha1.5/gamma1/one_se_scale1，mode alpha1。

候选的完整范围是旧caps内且q≤floor(.7n)的规模，不是所有支撑位置穷举，也没有放宽caps。研究版本的2000候选预算检查不是此最小源码的新默认；旧caps乘积≤floor(.7n)，本次新实验n≤400计数≤280，后续大样本正式部署须按最终协议提前核对候选计数/成本，不将验证范围扩大为所有输入。

仅私有枚举函数删除无作用的window参数和空域强制(1,1,1)返回：无合法三元组时返回空列表，不制造超容量模型；生产正常初始化保证caps及Q为正，该边界变化不改变本轮合法调用。原公共参数签名不变。

## 批准后源码同步

批准前只可核查MANIFEST和git apply --check，不运行本包sh的prepare/run/merge，不替换队友生产代码。本包不要求开分支、commit或push。最终批准且确认目标当前32源码确为v5.1后，在目标项目根执行：

```bash
git status --short
git -c core.autocrlf=false apply --ignore-space-change --check /同步路径/search-final.patch
git -c core.autocrlf=false apply --ignore-space-change /同步路径/search-final.patch
export PYTHONPATH="$PWD/src"
python -m unittest tests.test_bcd_fix_regression -v
```

check失败即停止比对基准和本地改动，不强制覆盖。已应用相同最终patch的目标先核对完整32文件指纹，不重复apply。不要只修改PYTHONPATH指向研究快照：known_s_cluster.run导入必须来自目标repository/src，身份扫描也固定指向该src，patch应用到批准后的目标src才使两者一致。

上述明确行尾兼容参数已在两份普通v5.1副本（全LF和全CRLF）实际check/apply/reverse-check，应用后32文件规范LF逐字匹配最终候选；不使用Git分支/worktree。兼容参数仅允许上下文空白差异，最终内容仍须完整规范LF指纹核对，不可把实质代码差异当作行尾处理。

本机复验pytest仅用新隔离工具，非集群运行依赖：

```powershell
$env:PYTHONPATH = "$PWD\doc\algorithm_versions\v5.2-search-final\src;$PWD\doc\algorithm_versions\taskcard4_test_tools"
& .\.venv\Scripts\python.exe -B -m unittest tests.test_bcd_fix_regression -v
& .\.venv\Scripts\python.exe -B -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py -q
```

本包的patch/sh核验不替代最终A2数值桥接或原公共接口回归；最终测试数量及成本门禁由主线记录，不在此预先宣布通过。测试新候选用独立新Python进程，避免已导入v5.1模块造成版本混用。

候选VERSION已记录6项限定验证通过：5静态测试及一次独立进程n400/dims(6,6,7)同X/y拟合桥接，研究显式legacy/full与本候选B/支持/caps/RSS/score/轨迹逐值相等；这是代表桥接，不代表所有场景已做独立冷拟合。包内Bash -n/LF核验通过，规范源身份检查同时通过LF/CRLF并拒绝改动源码；case参数组件使用mock python验证invalid退出2、缺数组ID报错、prepare/run/merge参数正确，未执行真实sh环境激活或计算。

## 批准后的700任务命令

将run_v52_search_final_full.sh增量复制到集群项目根。集群还须已有且与验收一致的numerical_simulation/known_s_cluster/{run.py,profiles.json}及共享data_generation.py/methods.py/metrics.py和完整core；不能用新plan代替源版本比对。

现有环境Python3.9.25、NumPy2.0.2、SciPy1.13.1，module load anaconda3、source activate myenv，四线程环境均1；不安装/升级库，pytest仅本机可选测试。共享输出文件系统须支持os.link，Slurm队列CMO_HPC、账号xjh、0–699%64、每任务1CPU保持既有配置。

最终批准、源码/入口/依赖核验之后首次创建新集群计划并提交：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_v52_search_final_full.sh prepare
sbatch run_v52_search_final_full.sh
```

该计划为full原五组14场景×50重复=700数据任务，四方法adaptive/oracle/lasso/glasso共2800模型结果；原BIC初始化c3、global MSIC参数、种子与DGP保持。输出numerical_simulation/outputs_v52_search_final_full，日志numerical_simulation/v52_search_final_full_logs；不覆盖v5/v5.1，不能重标旧B/支撑/q或把既有本机预检计划拷去续跑。

全部数组完成后：

```bash
cd /public/home/xjh/xmx/tensor-sparse-regression-main
bash run_v52_search_final_full.sh merge
```

prepare仅首次，resume继续同一sbatch命令；源码/配置/环境变化必须新plan和目录，完整任务严格校验后才跳过。merge检查700完整任务后才生成汇总，四方法不含fixed，不能报告known-s配对已完成。

本包sh只有BCD规范LF与三环境版本的运行前检查，部署时必须另按MANIFEST核对完整32core与入口依赖；依赖激活方式、实际集群库运行及资源限制尚待集群核验。Bash -n与独立组件guard测试不等于真实集群执行通过。

## 三实证边界

LEMON/HCP-YA/NKI-Rockland最终必须统一获批源码、参数和身份；同被试/响应/预处理定义/外内划分，所有训练折重新拟合预处理、生成caps/候选、重搜支撑、OLS及预测。实证target_sparsity=None，不能用外层测试或真支撑调搜索/MSIC；旧B、支持、q和模型缓存均不能作为新版本结果。未联系队友或启动重训，具体入口与严格历史对照复用条件见doc/FINAL_TEAM_SYNC.md及doc/FULL_EXPERIMENT_PREPARATION.md。
