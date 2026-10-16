# HCP/NKI完整v5.2算法交付

2026-10-15（北京时间）。正式v5.2源码、MSIC/caps、旧模拟及旧实证保持不变。

两份ZIP已经实际生成：先创建一个标准包再逐字节复制，SHA256完全相同；完整32源码不是单文件补丁。均独立解压到新目录，以python -I -B且移除PYTHONPATH执行版本/import检查、fixed/adaptive smoke及2项兼容测试，所有退出码0。核心字节与正式manifest一致，包内SHA256SUMS覆盖其他载荷；ZIP CRC、缺件、篡改和误导入检查通过。

| 文件 | 实际绝对路径 | 大小bytes | SHA256 |
| --- | --- | --- | --- |
| Tensor-ABESS-v5.2-HCP.zip | C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main\doc\team_sync\v5.2-search-final\full_package\Tensor-ABESS-v5.2-HCP.zip | 49943 | 899133561b556174af9c70d064ab6a0e2419b136b3ab7b53871b1bc0ff07cacc |
| Tensor-ABESS-v5.2-NKI.zip | C:\XiangMingXu\2026大创\代码\tensor-sparse-regression-main\doc\team_sync\v5.2-search-final\full_package\Tensor-ABESS-v5.2-NKI.zip | 49943 | 899133561b556174af9c70d064ab6a0e2419b136b3ab7b53871b1bc0ff07cacc |

下载：[HCP ZIP](team_sync/v5.2-search-final/full_package/Tensor-ABESS-v5.2-HCP.zip)、[NKI ZIP](team_sync/v5.2-search-final/full_package/Tensor-ABESS-v5.2-NKI.zip)、[自检原记录](team_sync/v5.2-search-final/full_package/DELIVERY_VALIDATION.json)。未发送消息或上传他人项目。

内容：src/tensor_abess全部32文件、tests、examples/smoke_test.py、scripts/check_version.py、README、ALGORITHM_CHANGELOG、INSTALL_AND_UPGRADE、依赖列表、公开派生RELEASE_MANIFEST及SHA256SUMS。公开manifest保留完整原始core raw/canonical指纹与上游manifest哈希，明确是派生公开视图；未带私人路径、数据、模拟结果、venv、缓存或旧研究求解器。实际自检环境为本机已有Python3.12.8/NumPy1.26.4/SciPy1.16.1；不冒称已在队友环境通过。

v5→v5.1修复fixed BCD的最佳状态与接受轨迹；v5.1→v5.2补全旧caps内合法整数规模。原MSIC、caps、初始化、Group-BESS及公共接口不变。模拟完全恢复提高，但平均预测误差没有全面改善；本次LEMON还明显恶化，不能保证HCP/NKI改善。

队友按包内INSTALL_AND_UPGRADE：先记录现有import及源码、备份旧core、检查本地修改，再显式更新整个core，不覆盖项目其他文件；使用--source对实际项目目录核验32文件及import。独立项目可直接指向解压src，核心依赖NumPy/SciPy；不复制本机环境或擅自升级依赖。smoke只证明公共fixed/adaptive入口能运行，不是实证或统计正确性的替代。

**保持各自HCP/NKI原样本、响应、预处理、MRI模态顺序、外内验证划分和参数，只替换算法实现并重新拟合ABESS。** 所有训练折重新预处理、生成候选、搜索支持、OLS和预测，不复用旧系数、q、支持或缓存；原输出保留，使用独立v5.2目录。回交完整版本/输入/配置/划分/环境身份、完整OOF、各折支持/q、训练测试误差及失败记录；本仓库没有队友真实入口，不拼造CLI或结果。
