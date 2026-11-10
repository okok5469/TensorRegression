# 首次离线运行时导入缓存故障

日期：2026-10-28。状态：本机回归已修复，修复后的真实HPC执行未运行；任务卡18仍为IN_PROGRESS/DRAFT。

用户提供的真实证据：prepare作业40887以FAILED/1:0在8秒退出；bootstrap_runtime.py报`Existing/offline dependency version mismatch: sklearn`，未进入限定测试或训练。pilot40888等待DependencyNeverSatisfied，40889至40891等待上游。用户实际展开根为`/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard18_cluster_20261026`。具体HPC错误版本及模块__file__尚未取得，不冒称已经直接观测。

已核对原交付wheel及其SHA，sklearn源内版本和manifest均为1.5.2。已用真实bootstrap提取和导入代码、纯Python依赖替身在隔离进程中重现机制：SH将首次尚不存在的runtime加入PYTHONPATH，解释器缓存该路径的查找器为None；bootstrap创建runtime后未刷新缓存，会跳过新目录并导入旧环境sklearn，触发与用户相同的版本检查错误。

唯一运行代码变更是在私有runtime解包/校验之后、依赖导入之前调用`importlib.invalidate_caches()`。仍严格要求原版本、私有依赖导入路径和SHA；不卸载/安装共享环境，不改三个冻结拟合源、原协议或优化精度。依据为[Python3.9 importlib官方说明](https://docs.python.org/3.9/library/importlib.html#importlib.invalidate_caches)。

回归用例先RED（子进程选择旧sklearn并触发同一ValueError），单行修复后GREEN，证据为本目录RED/GREEN及CLUSTER_TESTS文件。测试仅替换主机/ABI门以便在Windows测试纯Python替身；没有导入实际Linux二进制，也不构成HPC二进制兼容性验证。

交付采用独立20261028修复包和全新展开目录，保留原20261026包、失败日志及runtime；旧链仅通过其自身记录的job IDs取消。此时没有新拟合可迁移，无需补造、删除或放宽任何科学证据。修复后仍由真实prepare负责Linux环境、依赖及输入验收。
