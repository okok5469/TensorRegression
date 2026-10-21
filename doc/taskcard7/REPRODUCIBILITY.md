# 任务卡7完整复现与产物记录

2026-10-16（北京时间）。本轮按用户选择完成本地诊断；250任务留待原Linux集群执行。

## 运行环境与输入

本轮本地使用既有`.venv/Scripts/python.exe`，不安装/升级依赖；Python3.12.8、NumPy1.26.4、SciPy1.16.1及Windows平台与LEMON历史配置一致。先设置OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS均为1。正式源码和三冻结版本、原模拟及两个LEMON完整结果只读；初始/最终Git状态与2880个文件的大小/SHA快照在本目录。

输入指纹、完整源码及实际import、配置、435外折分配、训练/测试ID顺序、预处理逐数组指纹分别冻结在lemon/plan.json、train_only/protocol.json、mechanism/protocol.json；仅使用既有145名受试者数据，无下载、重生成或筛选。

## 执行入口

仓库根目录PowerShell：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONPATH='.'
.venv/Scripts/python.exe -B -m scripts.taskcard7.lemon --output-dir numerical_simulation/outputs_version_attribution/lemon --audit-only
# 正式A命令（现有目录已完整；不得重跑覆盖）：
.venv/Scripts/python.exe -B -m scripts.taskcard7.lemon --output-dir numerical_simulation/outputs_version_attribution/lemon
.venv/Scripts/python.exe -B scripts/taskcard7/train_only.py prepare
.venv/Scripts/python.exe -B scripts/taskcard7/train_only.py run
```

A/C3/C2均拒绝覆盖已有正式实验；以上记录用于核查本轮命令，复现实验须在不包含本轮新输出的独立副本中执行，保留原实验字节；不要删除现有产物后重跑。C2实际命令及协议、受保护源/环境校验见[脚本](../../scripts/taskcard7/mechanism.py)，只训练代表折不生成新外OOF；各worker独立进程，v5.1真实轨迹直接复用A。

```powershell
.venv/Scripts/python.exe -B -m scripts.taskcard7.mechanism prepare
.venv/Scripts/python.exe -B -m scripts.taskcard7.mechanism run
```

```powershell
.venv/Scripts/python.exe -B -m unittest scripts.taskcard7.test_lemon scripts.taskcard7.test_train_only scripts.taskcard7.test_mechanism -v
.venv/Scripts/python.exe -B scripts/taskcard7/figures.py --qa-scripts 'C:/Users/xmx/.codex/skills/nature-figure/scripts'
.venv/Scripts/python.exe -B scripts/taskcard7/reports.py
```

绘图使用Python已保存偏好，内部报告宽183mm；原CSV全量输入，PDF/SVG/TIFF/PNG、对齐JSON、PDF字形与碰撞检查及目视记录在figures。科研图没有按结果删折，不对相关外/内折作IID显著性推断。

## 实际拟合与保存口径

| 实验 | 新拟合 | 评价/验收 |
| --- | ---: | --- |
| A：v5.1完整外折 | 15 | 435条v5.1OOF；三版共1305条，45行配对指标；1215候选B/trace |
| C2：v5/v5.2代表训练重放 | 12 | 三版6折1458候选验收；v5.1/v5.2的486候选逐值一致 |
| C3：训练内三组×3折×5模型 | 45 | 1740验证预测，含训练共5220行；45份B/支撑/指标 |
| B：原250模拟标签 | 0 | 严格Gate历史失败，本轮不运行，待原集群 |

C3的45含9个无EEG拟合的协变量基线评价；不能称45个张量搜索或把各内训练者重复计为独立受试者。C2依据训练q选择代表，不用外测选择参数，重放结果不替换原外OOF。所有运行采用原种子/原折内处理，B的Linux环境与本机不同不能冒充。

## 全部交付

上级目录包含任务卡要求10份报告；主入口[FINAL_VERSION_DIAGNOSIS.md](../FINAL_VERSION_DIAGNOSIS.md)先给一页中文总结。[DELIVERY_INDEX.json](DELIVERY_INDEX.json)列出脚本、报告、协议、关键结果和SHA；逐产物SHA分别存于A artifact_validation、C2 completion/verification、C3 completion/verification。完整支撑/B/候选score、Q及训练/测试R²等见各CSV/trace，不将历史缺失轨迹字段补成实测。

Phase B按用户选择推迟，并非放宽Gate或任务完成250；两份对应报告及outputs_v51_attribution/STATUS.json明确记录0/250与原字节恢复步骤。未发送团队消息、生成算法ZIP、修改HCP/NKI设计或启动下一轮研究。
