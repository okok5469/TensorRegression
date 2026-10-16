# 安全清理执行计划

阶段：完整审计已完成，实际删除尚未执行。按任务卡5授权仅清理精确dry-run的125个再生字节码文件；不移动目录，不删除旧实验，不删除依赖，不做git clean，不递归删除。

1. 保留before/file_manifest.csv、git_status.json、important_sha256.json、delete_dry_run.json和summary.json作为清理前证据，全部在doc/releases/v5.2-search-final/audit/内。
2. 应用前重新读取Git跟踪集合和路径引用；对125项全部核验：路径不含..或绝对路径，最终绝对目标留在本仓库且不经过符号链接，仍未被跟踪/引用/保护，文件大小与SHA256都与dry-run一致；任一项不符则停止整个删除步骤。
3. 全部通过后逐文件unlink，不使用任何递归删除，不删除空目录以外的额外对象，也不作补偿或猜测性删除；准确记录每个删除文件、字节数与hash。
4. 生成after全量清单，逐一核对before的7,068个重要文件仍存在且SHA256一致；授权新增发布文档/脚本和本机预检产物单列，不伪装为历史变化或磁盘节省。
5. Git实际删除不得涉及已跟踪文件；归档只做原位置索引，无路径/导入/报告引用变化。最终写PROJECT_FILE_MANIFEST.csv和PROJECT_CLEANUP_REPORT.md。

KEEP包括全部源码/测试、历史版本与重要报告、旧模拟/实证输入输出、源指纹/seed/config/patch、依赖工具和所有歧义对象；ARCHIVE是已确认停止研究的原位置索引；DELETE只包含[已保存精确清单](releases/v5.2-search-final/audit/before/delete_dry_run.json)。不能因filename含old/test/temp/STOPPED扩展删除范围。

可审计入口为scripts/release_v52/audit_cleanup.py；默认仅审计，只有显式--apply-plan才能执行列出的再生文件删除。该脚本的安全测试、实际删除与后验结果写入清理报告。125文件总1,382,690字节约1.319MiB；不将同期生成的审计证据和正式预检新增数据忽略以伪称整个项目净减少同样大小。
