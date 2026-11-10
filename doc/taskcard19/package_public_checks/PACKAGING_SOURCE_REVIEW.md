# 任务卡19包装源码独立复核

复核范围：cli.py、smoke.py、build_bundle.py、accept_package.py、README_ZH.md、TEAMMATE_TASKCARD.md、DATA_CONTRACT.md、METHOD.md、RESULT_RETURN.md、pyproject.toml及requirements-lock.txt。本报告区分源码审查与尚待真实执行的安装验收；不据源码宣称跨平台通过。

## 实质发现（已提交主agent处理）

1. 开发版/安装版签名比较尚未显式覆盖预测标签与支撑。当前smoke保存scores/classes/coef，accept只比较这些；即使系数相等，声明支撑仍可能包含零系数坐标，不能用系数相等替代支撑相等。需在签名中补predictions、supports（Unicode JSON可用allow_pickle=False读取）及tie_priority，并严格比较标签、支撑和优先序；scores沿用事前rtol=1e-8/atol=1e-10。
2. 干净安装验收记录了实际import路径，但未强制断言核心与包装来自新venv安装位置。应在accept中核对verify返回的runtime.actual_import_files，二者均须位于新venv内；清除PYTHONPATH/PYTHONHOME与非editable安装已经实现。
3. 外层ZIP只核对manifest列出的archive_files，尚未核对列出文件的完整库存及旁置SHA256。应核对外层.zip与.sha256一致，限制manifest相对路径，并将实际库存与manifest列表＋预声明根说明文件比较。SHA清单不是外部签名：若攻击者同时替换ZIP与外部清单，不能声称其提供密码学发布者认证。

## 已符合的源码要求

- CLI在数值导入前设六项线程环境为1；命令统一调用Python，无原项目绝对路径或PYTHONPATH依赖。
- build对32源逐一核对正式release原字节SHA256，再使用copyfile打入wheel；已有冲突快照明确失败。默认ZIP只选src/tests/wheelhouse、wheel及指定文档，不打包git、venv、大型数据或旧归档。
- 唯一数据schema、固定A路线、无伪概率、开发与final明确分离、负结果及PARTIAL完整回传、400上限仅属于本机筛查等文档表述符合任务卡。
- 依赖精确版本在pyproject和lock中一致；Windows二进制wheelhouse和Linux联网安装说明明确分开，Linux如实为UNVERIFIED。
- smoke要求runner COMPLETE，执行返回校验与打包，并用保存的标准化参数/系数独立重算分数；fixture明确不是统计性能证据。

## 运行身份证据补充

按主agent明确委派，对identity.py补充实测运行身份：实际导入模块__file__、CPython3.12核对、五项依赖metadata版本与模块__version__双核对、实际threadpoolctl线程池且全部num_threads=1、32源及包装核验hash。先运行缺实现测试得到2个预期FAIL，再实现后2项PASS；原始日志为model_checks/IDENTITY_RED.txt与IDENTITY_GREEN.txt。此修改未触碰正式32源。

最终G4/G5仍须由主agent修复上述缺口后，重建最终ZIP并在仓库外真实安装/verify/smoke/重算/打包确认；本源码复核不能替代这一步。

## 源码修正闭环复核

已只读复核主agent后续修正，以上三项源码缺口均已闭环：

1. smoke签名新增predictions、Unicode JSON supports、tie_priority；accept对标签、支撑与优先序逐项array_equal，分数维持事前容差，系数严格比较。
2. assert_installed_paths逐一检查实际导入的核心/包装__file__为真实文件且位于新建venv内，已接入verify后、smoke前的验收流程。
3. verify_archive在解压前核对旁置SHA256，拒绝重复/绝对/父级/反斜线/盘符路径，核对ZIP完整库存并逐文件hash。accept_package.py、run_logged.py已纳入交付，确保验收测试不依赖原仓库辅助脚本。

修正后的指定源码没有新的阻塞发现。此闭环仅为源码检查；原仓库外最终ZIP的真实安装及签名比较仍由CLEAN_INSTALLATION.json和原始日志证明，尚未在本复核中宣称通过。旁置SHA256依然只是分发完整性检查，不应表述为防止同时恶意替换ZIP与清单的发布者认证。
