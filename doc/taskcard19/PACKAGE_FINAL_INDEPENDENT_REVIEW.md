# 任务卡19最终交付包独立复核

日期：2026-10-31（北京时间）。本复核只读最终ZIP、原release、当前隔离源码、测试日志与审查记录；未改交付ZIP、wheel、正式32源或其他封存证据。

**结论：最终包字节身份、文件清单、核心/包装一致性与便携源码边界PASS（本复核范围）；final clean-install验收以主agent对应日志为准。没有发现新的交付阻塞。**

## 精确交付身份

- ZIP：`deliverables/taskcard19/TensorABESS_OvRLS_v0.1.0.zip`；64,736,766字节。
- ZIP SHA256：`5c3fdb7701dcb4b759b70f5593854f79a5be86722518ba1971c59c4cfa54c9df`，与旁置同名.sha256一致。
- 内嵌wheel：`dist/tensor_abess_ovr_ls-0.1.0-py3-none-any.whl`。
- wheel SHA256：`2be6968b07d29ebd26b0ac5fdc7343f05b245b31cfa0851434aaab99d68130b4`。

## 独立实际核对

| 范围 | 结论及实际核对 |
|---|---|
| 外层完整性 | PASS：独立计算ZIP SHA256，核对旁置清单；82个ZIP文件无重复、安全相对路径；实际库存与archive_files加两份预声明根文件完全相同，逐文件hash全部一致。 |
| 正式核心 | PASS：读取正式RELEASE_MANIFEST.json的32份原字节SHA256；当前原src、ZIP内src快照、内嵌wheel中的tensor_abess分别全部相同；wheel核心Python文件库存精确为32。未使用canonical-LF替代原字节核对。 |
| 包装/profile/fixture | PASS：ZIP根manifest与wheel内资源manifest的core_sha256及wrapper_sha256完全相同；12份包装Python源码及全部18份包装/配置/fixture资源的hash，逐一与ZIP、wheel和当前隔离开发源码一致。 |
| 依赖与平台 | PASS（打包边界）：应用wheel和锁文件包含明确依赖版本；wheelhouse为Windows CPython3.12所需依赖发行文件。Linux使用锁定requirements另行安装，入口保留UNVERIFIED。 |
| 排除大数据与环境 | PASS：没有.git、venv、.venv_local、__pycache__或node_modules；唯一NumPy数据为2,816字节的软件fixture X.npy，未包含HAR原始数据、旧模拟或大环境。其余较大二进制为明确的依赖wheel。 |
| 本机路径依赖 | PASS：逐行检查交付文本的Windows盘符路径、用户目录、原项目名称和用户名，没有本机专属路径；应用路径由CLI参数或安装资源位置取得。sanitized检查记录未暴露原机器路径。 |

## 源码、测试与审查证据

已读取`SCOPED_TESTS.log`实际结果：42项通过，用时46.737秒；包括模型数学与标签/并列检查、API线程/版本/profile边界、真实spawn超时、数据隔离、基线网格与seed、预算耗尽和损坏回传等限定测试。本复核没有重新运行旧模拟或将软件fixture用作分类性能证据。

已读取`EVALUATION_INDEPENDENT_REVIEW.md`与`PACKAGING_SOURCE_REVIEW.md`的实际闭环结论：内折/训练ID/时间/分组、诊断与网格完整性、重复身份交叉核对、浮点并列边界，以及开发/安装标签和支撑签名、真实venv导入位置、ZIP库存与外部SHA256缺口均已修复。scope依然限定为应用与协议，不宣称全仓库、远端或正式分类测试已验证。

公开API代码保持lazy导出，并在数值导入前设置六项线程环境为1；fit/load执行完整release与实际运行环境核验。已有数值库多线程运行时明确报错，不通过后台改线程或参数来降级。固定core_profile和32源在应用调用时均受hash约束。

## 干净安装结论的归属

已读取`clean_install_candidate/CLEAN_INSTALLATION.json`：候选ZIP外部venv安装为PASS，内嵌wheel的SHA256与本最终ZIP完全相同，开发/安装签名一致为PASS；该证据只直接证明字节相同的wheel。`package_public_checks/VALIDATION_SCOPE.json`明确保留了这一范围，未把候选外层ZIP伪称最终ZIP。

本最终ZIP的另一次外部安装正在由主agent执行。最终G4通过与否必须以`clean_install_final/CLEAN_INSTALLATION.json`中的**最终ZIP精确SHA256**、六步真实返回码、实际导入位置和签名比较为依据；本独立复核不代填或推定该记录。Windows为当前验收平台，Linux继续UNVERIFIED。最终测试集是否封存及真实HAR开发试跑属于后续G6证据，不由此软件包复核冒称完成。

## 最终外部安装日志补充确认

后续已只读确认`clean_install_final/CLEAN_INSTALLATION.json`真实存在，status为PASS，archive_sha256精确为`5c3fdb7701dcb4b759b70f5593854f79a5be86722518ba1971c59c4cfa54c9df`，wheel指纹也与上述最终包一致。六个阶段——新建venv、离线安装wheel、verify-release、smoke、validate-return和pack-return——返回码全部为0；实际核心/包装导入均在新的外部venv内，开发/安装签名比较为PASS。由此最终精确ZIP的Windows G4证据已闭环；Linux仍为UNVERIFIED。

同时只读复核`IMPLEMENTATION_AND_PACKAGE_CHECKS.md`及`seal_report.py`的报告生成逻辑：软件PACKAGE_READY与真实数据LOCAL_STAGE分列，完整网格存在20000步未收敛项时不能写开发完整通过；已向主agent指出38身份清单实际含32核心与6模拟文件，以及拟合/规则构建记录统计需区分已记录返回与可能资源中断的未返回调用。真实HAR仍运行，尚不在本复核中宣称G6完成或给出性能结论。
