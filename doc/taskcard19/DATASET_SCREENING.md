# 任务卡19数据核验

核验日期：2026-10-31（北京时间）。本文件记录数据与表示身份；模型运行和成绩由 `LOCAL_PILOT_REPORT.md` 单列。

## 本地输入与官方来源

`D:\数据集` 存在。已有 `har人体活动数据` 仅包含 `har_static_subject.csv` 和 `har_dynamic_long.csv`，分别使用 `subject/time/var_*` 与 `sample_id/time/var_*` 列；没有本卡所需完整六活动标签、官方受试者映射和九通道来源链，属于另一项目处理表，因此未作为输入。未改变这两份文件。

首候选使用 [UCI HAR官方页](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) 所链接的 [官方ZIP](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip)。官方页声明无缺失，README说明六活动、30受试者、50 Hz采样、128点窗口及50%重叠。实际18个通道文件、subject和activity均已检查形状、有限性及对齐；没有插补或删除窗口。

- 原始只读文件：`D:\数据集\TensorABESS_Multiclass\HAR\uci_har_official.zip`。
- 下载完成文件时间：2026-10-31 08:37:02 UTC，即北京时间16:37:02；精确来源记录见 `data_checks/OFFICIAL_SOURCE.json`。
- 原始字节数：61,005,872。
- 原始SHA256：`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`。
- 处理目录：`empirical_analysis/outputs_taskcard19/data/har_summary_v1/`。

原始ZIP内部的2012年README为Latin-1，含机构名称重音字符；适配器显式采用该源编码，输出provenance和合同统一UTF-8。未使用忽略解码错误或字符替代。

## 表示与映射

从官方 `Inertial Signals` 的 `total_acc/body_acc/body_gyro` × `x/y/z` 精确重构 `(10299,3,3,128)`，另存 `X_raw_mapping.npy`。没有reshape官方561列特征。总加速度与身体加速度存在派生关系，三种信号不代表三个独立传感器。

唯一拟合表示为 `(10299,3,3,17)`：每通道有符号均值，以及去均值后乘 `np.hanning(128)`、`rfft(norm="ortho")` 的正频率bin1–64每连续4bin平方幅值和取 `log1p`。第三轴是“均值与固定窗频带能量特征”；非原始时间轴，亦非规范化PSD或方差精确分解。各频带索引、Hz范围及三轴坐标在 `dataset.json` 中。该构造完全逐样本，不学习跨样本统计量；z-score仍须在各训练折内部估计。

实际原始成员及处理文件逐项SHA256见 `dataset.json`，正式数据合同校验记录见 `data_checks/HAR_VALIDATION.json`。固定已知坐标fixture同时核对真实适配器通道顺序及C顺序。另以独立ZIP读取路径逐点核对实际18通道共6,912坐标，全相等；三个训练通道按独立频谱表达式重算摘要，最大绝对差 `1.31e-18`，见 `data_checks/INDEPENDENT_HAR_MAPPING.json`。

## 成绩产生前冻结的开发划分

seed为 `20261031`。官方train含7,352窗口、21受试者；test含2,947窗口、9受试者，受试者集合互斥。对官方训练受试者使用具名SHA256种子生成器，无有状态共享随机流，留出 `ceil(21×0.25)=6` 名受试者：`05,17,21,23,27,28`。

| 用途 | 窗口数 | 受试者数 | 类别覆盖 |
|---|---:|---:|---|
| 开发训练候选池 | 5,144 | 15 | 6/6 |
| 按原比例抽样后的开发训练 | 400 | 15 | 6/6 |
| 开发验证（留出受试者全部窗口） | 2,208 | 6 | 6/6 |
| 官方最终测试（未评分） | 2,947 | 9 | 仅合同/来源核验 |

| class_key | 活动 | 开发训练分母 | 开发验证分母 |
|---|---|---:|---:|
| 1 | WALKING | 69 | 339 |
| 2 | WALKING_UPSTAIRS | 60 | 295 |
| 3 | WALKING_DOWNSTAIRS | 55 | 282 |
| 4 | SITTING | 69 | 403 |
| 5 | STANDING | 72 | 450 |
| 6 | LAYING | 75 | 439 |

按类别原比例的最大余数法分配400名额，各类使用具名随机流无放回抽取，随后保持官方行序。训练/验证不跨受试者，没有按预测难度筛选窗口。`samples.csv` 与 `splits.json` 保留全部精确ID；`protocol.json` 仅激活 `har_dev_subject_holdout`，未激活 `har_official_final`。该成绩只可解释为有限训练样本的开发筛查，不能当作正式测试。

处理合同身份SHA256：`37aecfa2e815aacb84d1d032e811ce49180649bcbae8ed0908c19718af12ee3a`；身份包含X、samples、dataset、splits、表示及源成员指纹。适配期间没有训练分类器，划分/表示未依据模型成绩选择。

## 软件数据检查及候选边界

`data_checks/TDD_RED.txt` 保存初始缺实现失败，`HAR_README_RED.txt` 保存官方源编码问题回归，`CSV_MISSING_CELL_RED.txt` 保存缺失CSV单元回归，`PATH_ANCHOR_RED.txt` 保存Windows相对盘符路径拒绝回归；`TDD_GREEN.txt` 的14项测试通过，覆盖按row_index重排、中文/空格路径、未知ID/类、重复ID、缺类训练、样本/受试者交叠、dev阻止test、非有限值、声明轴及坐标顺序错误、文件损坏、路径越界、前向时间边界、独立频谱公式、已知通道坐标。mmap使用者需在Windows删除文件前释放映射；测试显式关闭映射。

本子任务实际获取一个候选HAR，未获取或拟合MHEALTH，未增加其他候选。数据处理和合同核验不代表模型开发试跑完成；最终真实拟合量、资源状态和科学发现以主流程最终报告为准。`FORMAL_TEST_EVALUATED: NO`。
