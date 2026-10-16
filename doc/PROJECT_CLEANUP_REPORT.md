# 项目清理结果

已按审计→dry-run→显式清理→后验完成；只逐文件删除125个未跟踪、无必要精确路径引用、hash锁定的__pycache__/*.pyc，释放1,382,690字节（约1.319MiB）。[删除明细](releases/v5.2-search-final/audit/after/deleted_files.json)记录每项原路径、大小与SHA256；未删除任何目录、已跟踪文件、数据、模型、研究日志或STOPPED证据。

归档是已明确停止研究的原位置索引，没有移动或复制大型输出。v5-original/v5.1/v5.2完整源码、任务卡2/3/4关键结果、原正式模拟和known-s、LEMON实证输入输出、历史报告/配置/seed/patch和隔离依赖均保留。HCP/NKI外部目录不在本地扫描和清理范围；没有声称已审计远端或外部数据盘。

before的重要7,068文件在清理后逐一重算SHA256，差异0；详见[preservation_checks](releases/v5.2-search-final/audit/after/preservation_checks.json)。目录/导入/历史报告引用路径不变，旧研究VERSION与团队MANIFEST保持当时状态。

| 快照 | 文件数 | 字节数 | 扫描错误 |
|---|---:|---:|---:|
| 完整清理前 | 14,558 | 4,027,281,065 | 0 |
| 完整清理后（当时快照） | 14,491 | 4,027,984,579 | 0 |

实际删除125文件；同期新增发布/审计证据、脚本和本机预检产物，所以项目快照总量净增703,514字节，而不是净减少1.319MiB。快照不包括随后继续生成的交付文档，不能把它宣称为最终整个目录实时大小。after又发现一个后来生成的6,984字节再生缓存，不在原dry-run内，本轮保留，未扩大清理名单。

清理前后Git原始状态见audit/before与after/git_status.json；大量既有未提交改动仍保留，没有git clean、stage、commit、分支/worktree或推送。生产bcd发布是任务卡5另行授权的唯一核心改动；清理没有新增已跟踪删除。

清理安全测试7项通过，包含整清单先核验、越界/..路径、历史保护、hash漂移及解析后的父目录别名拒绝。Windows删除路径先解析并确认位于目标仓库且没有路径别名；不跨shell拼接删除、不递归删除。代码审阅发现父目录别名保护不足时先以测试FAIL证明，再补齐该保护后执行清理。

目录职责和KEEP/ARCHIVE/DELETE依据见[PROJECT_DIRECTORY_AUDIT](PROJECT_DIRECTORY_AUDIT.md)、[PROJECT_CLEANUP_PLAN](PROJECT_CLEANUP_PLAN.md)。最终交付文件清单为[PROJECT_FILE_MANIFEST.csv](PROJECT_FILE_MANIFEST.csv)（清理后快照），历史证据仍按原路径定位和复现；未运行新的700正式模拟或三个实证。
