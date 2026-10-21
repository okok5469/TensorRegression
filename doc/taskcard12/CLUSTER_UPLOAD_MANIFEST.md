# Task12 集群上传清单

此交付只准备新研究包；真实 Slurm smoke、380例正式模拟和远端环境验证均为 `NOT_RUN`。集群配置来源为根目录 `CLUSTER_GUIDE.md` 及已用过的 Slurm 脚本，不能将本地语法或 mock 检查写成远端成功。

## 上传目标与不覆盖规则

把本地 `doc/taskcard12/delivery/taskcard12_cluster.zip` 和同目录的 `taskcard12_cluster.zip.sha256` 上传到已确认的项目目录 `/public/home/xjh/xmx/tensor-sparse-regression-main/`。ZIP 内已有顶层 `taskcard12_bundle/`，解压后唯一新工作目录为 `/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/`。目标目录必须尚不存在；不要把内容覆盖到原项目 `src/`、旧计划或旧结果目录。

上传包内 `FILE_MANIFEST.json` 和 `SHA256SUMS` 是确切文件清单，`doc/taskcard12/CLUSTER_PROTOCOL.json` 的 `source_files` 是科学执行时复核的依赖指纹；下载/解压工具不得改换行。包外 `.zip.sha256` 检查传输完整性，包内 `sha256sum -c SHA256SUMS` 检查文件完整性，二者不能互相替代。

## 必需内容及作用

| 包内相对路径 | 用途 |
| --- | --- |
| `doc/algorithm_versions/v5.2-search-final/src/` 全部32个Python源码及 `VERSION.json` | 实际运行的原正式快照；原MSIC、BCD、Cold、caps、完整网格和OLS保持原字节 |
| `research/taskcard12/selection.py` | `original` 默认、`fs`研究选项、单次Cold双选择及缓存复核 |
| `research/taskcard12/simulation.py` | prepare、真实job绑定、单case3600秒上限、双臂评价、严格恢复、完整合并 |
| `research/taskcard12/analysis.py` | 全380完整后按冻结定义分析，360主样本和额外20压力样本分开 |
| `research/taskcard12/cluster/*.sh` | 真实站点环境、smoke提交、Gate、正式数组和merge入口 |
| `research/taskcard11/rescore.py` | 已冻结FS数值规则与原生A0评分复核 |
| `research/taskcard10/experiment.py`、`cv.py` | 输入/输出哈希、预测流、进程控制与原模拟求解参数映射；不运行CV |
| `scripts/taskcard7/lemon.py` | 原生候选被动捕获与冻结核心定位；不启动LEMON实证 |
| `numerical_simulation/__init__.py`、`data_generation.py`、`metrics.py` | 原DGP、共享测试流及评价；不更换模拟机制 |
| `doc/taskcard12/CLUSTER_PROTOCOL.json`、`REQUESTS.json`、`HISTORICAL_INPUTS.json`、`ANALYSIS_SPEC.json`及协议说明 | 固定380任务、独立smoke、种子、旧输入查重、事前统计与工程规则 |
| 对应新增测试、包验证工具及说明 | 本地/集群文件验收，实际清单以包内manifest为准 |

必须上传上述正式快照中的完整32核心，不能只传 `bcd.py` 或一个FS评分函数；本轮实际导入不需要再复制一套顶层 `src/`。无需上传任务卡8—11的大量历史结果、旧候选缓存、训练数组或脑数据；旧结果仅把必要的哈希身份注册表带入包。原项目 `src/`、旧版本与历史结果均保留在原位置，本包不覆盖它们。

## 环境与版本切换

已确认站点为 Linux x86_64、Bash登录shell、Slurm `CMO_HPC` 分区、`xjh`账户。脚本执行 `module load anaconda3` 和 `conda activate myenv`，不引入猜测的 `conda.sh`路径、不安装或升级依赖。

严格期望 Python3.9.25、NumPy2.0.2、SciPy1.13.1及OpenBLAS0.3.27，四个线程环境变量均为1；首次prepare保存完整NumPy构建信息与实际Python路径，后续所有worker精确复核。同一版本号不能代替完整环境Gate。本轮配对路径直接依赖NumPy和SciPy，其余为标准库；不启用Vector Lasso等额外对照，因此不会为了本轮自动安装scikit-learn。

单规则调用为 `selection.fit(X, y, selection_rule='original', **native_options)` 或 `selection_rule='fs'`，不传规则时默认original；集群正式执行 `fit_pair` 一次搜索，同时保存A0=original、A1=fs。不得按测试预测结果逐例改规则，不能让两臂各自重新搜索。

每个数组项1CPU，最多64项同时提交，64是历史项目并发设置而非已证实站点配额；未获知站点内存上限、默认时限、数组quota及外网SSH地址，脚本不伪造这些参数。应用内部科学case上限3600秒；实际调度是否允许足够资源须通过用户执行的真实smoke确认。
