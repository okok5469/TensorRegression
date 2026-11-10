# 实现与独立交付验收

PACKAGE_READY: YES。最终ZIP已在原仓库外的新目录中解压，以全新venv离线安装内嵌wheel和锁定依赖，真实执行verify-release、smoke、validate-return和pack-return，全部返回0。Windows PASS；Linux UNVERIFIED。当前结论不代表分类实证有优势。

## 最终文件和身份

- `deliverables/taskcard19/TensorABESS_OvRLS_v0.1.0.zip`：SHA256 `5c3fdb7701dcb4b759b70f5593854f79a5be86722518ba1971c59c4cfa54c9df`。
- 同名`.sha256`位于同目录。wheel为`tensor_abess_ovr_ls-0.1.0-py3-none-any.whl`，SHA256 `2be6968b07d29ebd26b0ac5fdc7343f05b245b31cfa0851434aaab99d68130b4`。
- 32份生产源在原src、ZIP快照、wheel及实际安装位置逐文件原字节一致；包装12份Python源码与18项源码/资源清单一致，最终ZIP共82项。
- 包不包含Git、venv、原始大型数据、历史模拟或凭据；本机路径依赖扫描0命中。独立报告见[PACKAGE_FINAL_INDEPENDENT_REVIEW.md](PACKAGE_FINAL_INDEPENDENT_REVIEW.md)。
- 正式38项源码/配置身份（含上述32核心与6模拟文件）及任务卡18方法/模拟章保护记录见BASELINE_SIMULATION38.json和BASELINE.json；最终复验由FINAL_VALIDATION.json记录。未新建分支/worktree、改写旧结果或提交远端作业。

## 实际环境与独立安装

实测CPython3.12.8、NumPy1.26.4、SciPy1.16.1、scikit-learn1.5.2、joblib1.5.2、threadpoolctl3.6.0。共享`.venv`与历史runtime未升级；开发依赖装入新私有环境，干净安装另建仓库外venv，无editable链接。

外部安装清除PYTHONPATH/PYTHONHOME；断言`tensor_abess`与`tensor_abess_ovr`实际`__file__`在新venv内部。实际导入路径、32源/包装指纹、依赖metadata与模块版本双核对、活动BLAS/OpenMP线程均1的证据位于[clean_install_final/03_verify_release.log](clean_install_final/03_verify_release.log)。

最终精确ZIP验收见[clean_install_final/CLEAN_INSTALLATION.json](clean_install_final/CLEAN_INSTALLATION.json)，完整阶段日志为[安装](clean_install_final/02_install_wheel.log)、[smoke](clean_install_final/04_smoke.log)、[回传重算](clean_install_final/05_validate_return.log)、[回传打包](clean_install_final/06_pack_return.log)。该JSON的archive_sha256与最终ZIP相同，不能用早期候选安装代替此记录。

开发版与安装版在相同环境/输入/profile下，分数按rtol1e-8/atol1e-10比较；标签、系数、声明支撑和tie_priority严格相等。使用保存均值/尺度/常数掩码/系数直接矩阵内积重算分数，最大误差0。五方法各12行验证预测，共60行，完整回传校验通过。功能fixture不承担分类性能结论，也不要求100%准确率。

## 源码和必要正确性检查

[SCOPED_TESTS.log](SCOPED_TESTS.log)记录42项全部通过：数据14、交付4、身份/API5、模型10、runner/回传9。不宣称全仓库、原历史依赖工具或Linux平台测试通过。

- 独立逐类直接调用冻结核心，与封装完整返回状态和恢复分数一致；另以张量内积独立核对。响应只中心化，不按类别标准差缩放。
- 不均衡类频率、非连续整数/字符串一致重编码、类列重排、固定训练频数/首次出现顺序、浮点并列边界、严格常数坐标、C顺序轴映射、未知/缺类及非有限输入均检查。
- 无pickle的NPZ＋JSON保存/加载一致；有意损坏模型NPZ会失败。改变未训练评价数据不改已拟合统计量和模型。
- 每类异常使整个分类器不可用；真实spawn成功、单类超时、数据集截止约束均检查，资源耗尽不降低核心搜索。公开API在数值导入前设线程环境，fit/load核验实际活动线程和冻结身份。
- 训练内预处理、显式group/time内折、九参数完整性、稳定种子、5000→20000同目标重试、未解决数值状态、完整/部分结果和OOF清单检查。
- 回传校验识别hash损坏、重封hash后的缺失预测、删诊断和缺OOF；允许合法CSV行序或score类别列重排，不靠文件当前顺序拼接。支撑/q/非零数与保存模型及数据轴绑定。

独立评价审查见[EVALUATION_INDEPENDENT_REVIEW.md](EVALUATION_INDEPENDENT_REVIEW.md)，另有81个软件基线内折候选的独立seed/训练均值/指标探针。模型、数据、CLI与修错的RED/GREEN证据分别保存在model_checks、data_checks、runner_checks及本目录，不以源码审查代替真实安装。

## 发现与修复保留

最初软件fixture的高C L1候选到20000步仍未收敛，正确标为PARTIAL；原输入和失败输出均保留。后续固定非可分fixture仅作安装/正确性检查，未改真实HAR、网格、tol或任何方法预算。

独立复核推动修复了语义标签/支撑签名、实际安装路径断言、ZIP完整库存、回传诊断与内折/重复身份核对、OOF清单、总截止与子进程清理、浮点并列表达式差异及API资源边界，均保留回归证据。首次历史工具目录读取及仓库外临时目录默认沙箱访问受限；按授权对本卡专用外部目录提升执行验收，没有改ACL/权限规则或旧目录，未影响最终安装通过。

两次外部安装使用同一字节wheel；最终ZIP加入的是去除本机路径的检查记录，wheel重建字节一致。最终安装再次针对完整交付ZIP从头验收。后续本机HAR运行使用最终安装版；G6与科学结论由LOCAL_PILOT_REPORT.md单列。
