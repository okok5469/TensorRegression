# v5.2集群文件更新清单

本地已正式发布D2/v5.2-search-final；集群未连接、清单尚未取得，远端SHA256和环境均为`NOT_RUN`，以下为基于完整身份核验的更新规则，不是已确认远端现状。不得把集群“历史上为v5.1”当成当前真实状态。

目标历史项目路径为`/public/home/xjh/xmx/tensor-sparse-regression-main`；实际路径由上传脚本RemoteRoot参数指定，SSH主机由SshHost参数指定。先执行只读cluster_inventory，取得真实remote raw/canonicalLF清单及环境；不能先盲目上传bcd。

## 本地实际身份

完整32源码、6模拟依赖的本地实际raw/canonicalLF SHA256见[RELEASE_MANIFEST.json](releases/v5.2-search-final/RELEASE_MANIFEST.json)。原v5.1全32基准也在同一manifest；规范LF用于Windows/Linux行尾比较，实际任务plan再绑定部署后的raw源字节，建计划后禁止换行转换。

| 本地路径 | 集群目标路径（RemoteRoot内） | 本地raw SHA256 | 集群SHA256 | 操作 |
|---|---|---|---|---|
| src/tensor_abess/tensor/bcd.py | src/tensor_abess/tensor/bcd.py | db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7 | NOT_RUN | v5.1基准匹配才MUST UPDATE；已v5.2则VERIFY ONLY；其它为冲突 |
| run_v52_search_final_full.sh | run_v52_search_final_full.sh | 96914a6f14a5a7f6dde76bd2c9f435329c606a3a6b572c40d6b117c9c08e9049 | NOT_RUN | MUST UPDATE/新增，保留远端原文件可恢复备份 |
| doc/releases/v5.2-search-final/RELEASE_MANIFEST.json | 同路径 | 发布manifest本身于同步时计算 | NOT_RUN | MUST UPDATE/新增，记录正式身份 |
| doc/releases/v5.2-search-final/SOURCE_SHA256.txt | 同路径 | 同步时计算真实本地值 | NOT_RUN | MUST UPDATE/新增，完整32 raw清单 |
| scripts/release_v52/ 的远端核验/Gate脚本 | 同路径 | 见正式deployment清单 | NOT_RUN | MUST UPDATE/新增，仅白名单文件 |
| 其余31个src/tensor_abess源码 | 同路径 | 见32源码清单逐项值 | NOT_RUN | VERIFY ONLY；不一致即冲突，不静默覆盖 |
| numerical_simulation/{__init__,data_generation,methods,metrics}.py | 同路径 | 见simulation_sha256各实际值 | NOT_RUN | VERIFY ONLY，逐项核验源码及接口 |
| numerical_simulation/known_s_cluster/{run.py,profiles.json} | 同路径 | 见simulation_sha256各实际值 | NOT_RUN | VERIFY ONLY，确认原700任务/种子/续跑框架 |

完整逐文件清单为机器可读manifest（不能用表中缩写代替32+6核验）。在远端清单尚未取得时，表中远端值保留NOT_RUN；同步dry-run输出由真实远端hash决定UPDATE/VERIFY/CONFLICT，不能把本地hash填进远端列。

## DO NOT TOUCH

旧outputs_v51_bcd_fix_full及其它v5/v5.1/known-s、任务卡2/3/4研究输出、原始/实证数据和模型、doc/algorithm_versions历史快照、历史研究VERSION与team_sync证据、旧run_*.sh及无关未提交修改全部保留。Windows同步不上传整个仓库或任何旧结果目录；只有明确白名单文件可以Apply。

若远端仍是v5原版、31个应保留核心文件或6个依赖不符，停止受影响更新，保留真实清单与冲突；不能“只上传bcd就认为兼容”。先单独核对原v5→v5.1修复及完整身份，再使用v5.1→v5.2最小patch。本轮不实际修改远端或联系队友。

## 更新后的验收

上传前核对基准及远端文件路径安全性，保存原文件备份，再按Apply计划操作；上传后SHA校验失败即停止，不自动回补。实际import必须来自目标RemoteRoot/src，完整32源码及6依赖、正式发布记录、full配置和目标环境核验成功后才允许建新计划。

详细命令见[CLUSTER_DEPLOYMENT_GUIDE.md](CLUSTER_DEPLOYMENT_GUIDE.md)；真实SLURM单任务Gate与700提交命令见[V52_SIMULATION_SUBMISSION.md](V52_SIMULATION_SUBMISSION.md)。本轮远端信息待用户在实际集群执行检查，不能宣称集群更新或预检已完成。


## Complete local fingerprints / remote NOT_RUN

| Local path | Remote relative path | Local raw SHA256 | Remote SHA256 | Operation |
|---|---|---|---|---|
| doc/releases/v5.2-search-final/SOURCE_SHA256.txt | doc/releases/v5.2-search-final/SOURCE_SHA256.txt | 724696d72f408122286f4b546665b56092786a033b983c8dbbab264416c94d5d | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| numerical_simulation/__init__.py | numerical_simulation/__init__.py | 937ddd891cd1918e5726a149b7c6e3745067dee92d9945a8dbbce6c26b58cd26 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| numerical_simulation/data_generation.py | numerical_simulation/data_generation.py | 0a867f244bd44271e301eb0629c82f2332fccebb20a4329b51e27ad194066423 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| numerical_simulation/known_s_cluster/profiles.json | numerical_simulation/known_s_cluster/profiles.json | 76465458701ad6da176fd93d688d1b4762285a9cf391360737e0730a35c3fa94 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| numerical_simulation/known_s_cluster/run.py | numerical_simulation/known_s_cluster/run.py | 5fff973f646a889ec6e9a8af70f0e775c182f8cda6a473b2d644af5dd42ce9ee | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| numerical_simulation/methods.py | numerical_simulation/methods.py | af700bc8eb5abfa6469617f41a61ef0299e9be910fe63a55c34805a204d5bbd5 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| numerical_simulation/metrics.py | numerical_simulation/metrics.py | eaf4bbb30472754da10754476620bbb2dcb7a1c7e80b571014fb04edf183951f | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| run_v52_search_final_full.sh | run_v52_search_final_full.sh | 96914a6f14a5a7f6dde76bd2c9f435329c606a3a6b572c40d6b117c9c08e9049 | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/cluster_inventory.sh | scripts/release_v52/cluster_inventory.sh | b9bcacd3aa81dfb1a05a353d652c08df434f744fbbb4420223858ea24b01ee11 | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/download_results.ps1 | scripts/release_v52/download_results.ps1 | 2b3827e82e0970be41d3b52bce537eb26be4bd3821cc3b4edf08bf179f7937df | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/preflight_gate.sh | scripts/release_v52/preflight_gate.sh | c5bdbb916b9438d3245085cae82d8351efcde3108fb35f1aee63103df39a0ffb | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/release_control.py | scripts/release_v52/release_control.py | d51c21a5787e73965bfed6f0c21514a4f48d8413c89e117b4c4c1e271e5df0ee | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/sync_to_cluster.ps1 | scripts/release_v52/sync_to_cluster.ps1 | 9d08afab60b1feb08b712fe999d03dc9c5da0e36f12ef8fac33149415ee62ab5 | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| scripts/release_v52/verify_cluster.sh | scripts/release_v52/verify_cluster.sh | e1f34f68129ac7783fd44902779afec09f8e4f091e704082fc248ee88d536dbf | NOT_RUN | Deploy explicit release file / stop unknown conflict |
| src/tensor_abess/__init__.py | src/tensor_abess/__init__.py | f09ee71e1a5b9a9c9deddbfcdec0d028cd34803af58318af1beca9e9a3aa5dbe | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/__init__.py | src/tensor_abess/baselines/__init__.py | 2e25cb27c0deae528eefec64946bc7aafd241b235145c3fb2bbc32da983bd00a | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/base.py | src/tensor_abess/baselines/base.py | 296af75c1a2afbce822d401f3ba81a231b176aa8ff7d51a3fa3575ad03303306 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/cp.py | src/tensor_abess/baselines/cp.py | 164b04db1ab6ac6b163366957d5a3e5e3db9d9e117fe62a1bf63d6378e936fbb | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/group_lasso.py | src/tensor_abess/baselines/group_lasso.py | 28acbc2c67ef3a462b7c8c9551c6d08eca076877100494f23ed5b60aeb24ab75 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/lasso.py | src/tensor_abess/baselines/lasso.py | b0ce26433580f51c4a56a99d3748fbd609a7527469669475ff774bab3e1393f0 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/baselines/tucker.py | src/tensor_abess/baselines/tucker.py | a90c90b46ded263dd57fc99b926b6d82abb410b4541d9d898fc5b907037b0283 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/common/__init__.py | src/tensor_abess/common/__init__.py | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/common/ranks.py | src/tensor_abess/common/ranks.py | 4684b9504f033aee9cee10a40ab652d8ad76f2e871833fe3f123a883b2a32aa4 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/group/__init__.py | src/tensor_abess/group/__init__.py | ceb9fc92074ba440ef5c2082453aeba1f67db8da03a49309937e0e5efc32d3e1 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/group/adaptive.py | src/tensor_abess/group/adaptive.py | 453f0b2e9568f3a611977151ab20943a120bb9603adb90838d52d4ca5a306376 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/group/bess_fix.py | src/tensor_abess/group/bess_fix.py | 6577bc8b26b99b45b8a4bd7f154135fd10c0627c520c6a379f8a6616ab9c87fb | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/group/sacrifices.py | src/tensor_abess/group/sacrifices.py | b05f00ea51a5d91e01961e468e15ec3ca8046af649f8f2bbb51addd264376155 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/group/splicing.py | src/tensor_abess/group/splicing.py | 0ac340129f4f38f1aaf2c4d7c778f5c763964c9b6792485bcd017147b4d4641c | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/metrics/__init__.py | src/tensor_abess/metrics/__init__.py | 34ad9b63350042207d203b6c33c481bcf8fbb8e113216a094766e82778ac9eee | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/metrics/support.py | src/tensor_abess/metrics/support.py | 18495d2a039e6d9df9c650c4532a52e8cb59a05594755946098eede234054dd8 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/tensor/__init__.py | src/tensor_abess/tensor/__init__.py | a7a284698fe7934f052f95d6b93f595b46262a0b4b81f7e3cbf806fa13850667 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/tensor/abess.py | src/tensor_abess/tensor/abess.py | fc0a87f201f2f6aa18f62667d881cddfddc5d969b70c7b83cf95213c5cc7ee45 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/tensor/bcd.py | src/tensor_abess/tensor/bcd.py | db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7 | NOT_RUN | MUST UPDATE if recognized baseline / VERIFY if identical |
| src/tensor_abess/tensor/ops.py | src/tensor_abess/tensor/ops.py | cc0140752af901e792dc48811d320cb928f0dc917bbc9ccf3f8e8ae1cd9ee0fa | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/tensor/reduction.py | src/tensor_abess/tensor/reduction.py | 8b8a66856d5e8c8a0a887fb97f911687d6f48b8d25079430abb7ffe550774076 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/tensor/sic.py | src/tensor_abess/tensor/sic.py | 84637e426a8ad8d9c471d8f81787135052f10f954475127403d5a9b2b4f51d9f | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/utils/__init__.py | src/tensor_abess/utils/__init__.py | 969942a09ec95253bd97eddd57740fea3da8bb1ca0cb9c69074d9244b168e343 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/utils/data.py | src/tensor_abess/utils/data.py | 67696c6f1327a0f7f9e6c7980a1c7a648d4bf6a9165ce7ea673948891f63937c | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/utils/glasso.py | src/tensor_abess/utils/glasso.py | 05cf440fdf5c5cce814264ec35cc6c7b0397cdf229889452abe2e90cd650097d | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/utils/init.py | src/tensor_abess/utils/init.py | fb019d1a09c5f421db24ff8ae6821d5c7e01b7e440f0efe177727117a5762962 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/utils/validation.py | src/tensor_abess/utils/validation.py | d6500d0a227854dfc963f33fb47f97d10ef67527b15a599426c4d1202e669dbe | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/vector/__init__.py | src/tensor_abess/vector/__init__.py | 7f8f788a7915a4540f753f8e9a5e9917008ca725cb5b9fb23a8e575e75ab59f7 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/vector/abess.py | src/tensor_abess/vector/abess.py | 901efc50fc964801b44e9e12e90980ea065e15a303b9375f142fc8f0eb0527f0 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/vector/sacrifices.py | src/tensor_abess/vector/sacrifices.py | 2462c4f8f3e447e98218eb2c68811893a24c547b78007848e047c6afd79db9f2 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/vector/sic.py | src/tensor_abess/vector/sic.py | f562d2736186173f0939de7677a6f14a64c7a2d202bd414929e7aee310c1487b | NOT_RUN | VERIFY ONLY / CONFLICT if different |
| src/tensor_abess/vector/splicing.py | src/tensor_abess/vector/splicing.py | c14f4e47e7c8506ee63422016c69693a303ae5638f7c6ddd097d9baffa8b7e63 | NOT_RUN | VERIFY ONLY / CONFLICT if different |
