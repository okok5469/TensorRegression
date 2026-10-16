# 项目目录审计（任务卡5）

完整清理前快照于2026-10-11 18:38:58–18:40:08（北京时间）取得：14,558文件、4,027,281,065字节，重要文件7,068份SHA256，扫描错误0。此为本地正式源码已发布后的清理前快照；生产发布前身份另由releases/v5.2-search-final/PRE_RELEASE_SOURCE_SHA256.json保存，二者不能混称。

完整文件路径/大小/mtime/类型/Git跟踪状态/SHA/KEEP-ARCHIVE-DELETE分类见[审计CSV](releases/v5.2-search-final/audit/before/file_manifest.csv)；Git未提交状态和原始警告见[Git清单](releases/v5.2-search-final/audit/before/git_status.json)，重要哈希见[重要清单](releases/v5.2-search-final/audit/before/important_sha256.json)，统计见[summary](releases/v5.2-search-final/audit/before/summary.json)。初次普通扫描受到历史隔离依赖ACL限制，获准只读提权后完成全量；未修改ACL或将漏扫算作完成。

## 原位置保留

src为当前正式算法，tests及numerical_simulation为测试/冻结模拟入口；empirical_analysis及其输入/输出保留。doc/algorithm_versions内v5-original/v5.1/v5.2等历史/研究快照与全部证据保留；doc/team_sync历史patch、VERSION、manifest与报告路径不变。全量原模拟、known-s、任务卡2/3/4诊断、LEMON实证数据和旧结果原位置保留，不能换标签成v5.2。HCP/NKI外部数据未被访问或清理。

现有.venv、隔离test_tools/taskcard4_test_tools、.git/.codex/.agents/.aws均保护，不能因名称含test/temp/old而删除。大量既有未提交修改保留，不创建branch/worktree，不stage或提交无关文件。

## 归档索引

已查看明确STOPPED记录的废弃/停止研究队列标ARCHIVE，仅建立索引，所有文件保留原路径：旧candidate_search的historical/development/confirmation尝试、support_search_stability旧historical、预算停止的A2 MSIC确认。完整historical_final/development_final/confirmation_final与各关键报告继续KEEP。大型研究不复制第二套、不移动，以免破坏报告引用和复现入口。

## 删除候选与人工保留

dry-run仅125个Git未跟踪、没有精确路径引用、普通非保护目录的__pycache__/*.pyc，共1,382,690字节；每项记录原SHA256和大小，见[精确dry-run](releases/v5.2-search-final/audit/before/delete_dry_run.json)。没有删除名单以外的重复文件、数据、日志、源码、旧计划或STOPPED文件。其它重复/临时名称文件有歧义，全部保留供人工判断。

不重构模块导入路径；目录职责通过正式releases、历史algorithm_versions、团队team_sync和本报告索引清晰区分。清理后实际数量与历史哈希验收见PROJECT_CLEANUP_REPORT.md，发布和并行新增文件应单列，不能把它们计作清理删除或磁盘节省。
