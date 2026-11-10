# UCI270 来源、字段与正式核心核验

核验日期：2026-10-26。范围：公开来源、项目约束、源码身份；不包含本地DAT完整性或模型成绩验收。

## 1. 核验结论与证据边界

官方字段顺序与任务卡17一致，可继续本地数据审计；32个正式核心Python文件全部与发布manifest逐字节SHA256一致。研究用途限制与页面通用许可并存，保留该不一致，不作商业使用许可保证。裸Python没有安装项目包；采用项目既有的显式`src`导入方式后，实际导入均指向本仓库正式源码。

已完整阅读的项目文件：

- [数据需求](../../empirical_analysis/Tensor_ABESS_Dataset_Requirements.md)。
- [数据检索及UCI270数据卡](../../empirical_analysis/Tensor_ABESS_Dataset_Search_2026-10-26.md)。
- [任务卡17](../../任务卡/任务卡17.md)的来源、解析及张量定义。
- [正式发布manifest](../releases/v5.2-search-final/RELEASE_MANIFEST.json)。

## 2. 官方事实

证据入口为[UCI270官方页面](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)，取用日期2026-10-26；定位为Additional Information、Additional Variable Information、Citation及License。官方列出13,910条、16传感器、128特征、10批，浓度单位ppmv；类别编号映射如下，不能用网页列举物质的先后顺序代替编码。

| ID | 类别 |
|---|---|
| 1 | Ethanol／乙醇 |
| 2 | Ethylene／乙烯 |
| 3 | Ammonia／氨 |
| 4 | Acetaldehyde／乙醛 |
| 5 | Acetone／丙酮 |
| 6 | Toluene／甲苯 |

每个传感器连续8列：两个稳态量，随后EMAi三个尺度，再EMAd三个尺度；每组尺度按0.001、0.01、0.1排列。i/d分别指上升/衰减部分，尺度是平滑参数，不是采样时刻。第二稳态量是归一化量，网页符号虽带竖线，不授权对原始列取绝对值。DAT以“类别;浓度”开头，后接编号:数值；通用变量表的Categorical标注不能推翻详细数值定义。[官方字段说明](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)

| 批次 | 官方月份编号 |
|---|---|
| 1 | 1, 2 |
| 2 | 3, 4, 8, 9, 10 |
| 3 | 11, 12, 13 |
| 4 | 14, 15 |
| 5 | 16 |
| 6 | 17, 18, 19, 20 |
| 7 | 21 |
| 8 | 22, 23 |
| 9 | 24, 30 |
| 10 | 36 |

仅采用批次先后；DAT行号不提供精确采样时间。原始研究说明为同一阵列的长期记录，记录数不代表独立设备数。[原始研究](https://doi.org/10.1016/j.snb.2012.01.074)

## 3. 页面许可不一致

正文原文短摘：“exclusively for research purposes”；页尾原文短摘：“Creative Commons Attribution 4.0 International”。前者限制研究用途，后者标注CC BY 4.0。本研究保留两处事实、按作者要求引用，不替维护者解释冲突；没有由此发起其他用途。[同一官方页面](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)

## 4. 必须保留的引用

1. Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations* [Dataset]. UCI Machine Learning Repository. [10.24432/C5MK6M](https://doi.org/10.24432/C5MK6M)。页面捐赠日期为2013-10-22，引用年份沿用官方Citation的2012，不自行统一。
2. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical, 166–167, 320–329. [10.1016/j.snb.2012.01.074](https://doi.org/10.1016/j.snb.2012.01.074)。
3. Rodríguez-Luján, I., Fonollosa, J., Vergara, A., Homer, M., & Huerta, R. (2014). *On the calibration of sensor arrays for pattern recognition using the minimal number of experiments*. Chemometrics and Intelligent Laboratory Systems, 130, 123–134. [10.1016/j.chemolab.2013.10.012](https://doi.org/10.1016/j.chemolab.2013.10.012)。

论文作者、卷页和两个引用由数据作者的[2015数据论文PubMed记录及参考文献](https://pubmed.ncbi.nlm.nih.gov/26217723/)交叉核验；第二篇DOI另由[NIST作者机构页面](https://www.nist.gov/publications/calibration-sensor-arrays-pattern-recognition-using-minimal-number-experiments)核对。NIST记录2013为上线日期，期刊卷130采用2014。出版商页面直接打开返回403/抓取失败，PMC全文遇浏览器验证；未声称读过受阻全文，字段顺序的直接证据是可访问的UCI详细说明。

## 5. 项目提出的设计，不冒充官方协议

96维动态输入整理为传感器×尺度×阶段的16×3×2；排除32稳态量并另设128维基线；六类分开回归；开发测试批次7/8/9、最终批次10；训练端精确重复清除、滚动内层、标准化、八方法、固定网格、复杂度阈值和算力预算，均来自任务卡17。官方没有证明真实系数矩形稀疏，也没有保证ABESS优于对照。

0基索引的独立语义目标为`X[:,j,k,0]=raw[:,8*j+2+k]`及`X[:,j,k,1]=raw[:,8*j+5+k]`；本报告只核验这些索引的官方含义，全记录数值映射测试须由数据审计另行执行。尺度、阶段和传感器均为固定坐标，未构造虚假时间波形。

## 6. 核心源码与导入身份

以`.venv/Scripts/python.exe`执行`hashlib.sha256(path.read_bytes())`，比较manifest的`source_sha256`；同时枚举`src/tensor_abess/**/*.py`比较文件集合：32/32一致，缺失0、额外0、哈希不符0。完整逐文件预期指纹保存在上述发布manifest，无改写。

本次manifest原始SHA256：`b415a151674f1ee0400acbd000aa732e2c99343c7ad4faa48d7970b2cfe56074`。

| 实际导入模块 | 本仓库相对路径 |
|---|---|
| tensor_abess | src/tensor_abess/__init__.py |
| tensor_abess.tensor.abess | src/tensor_abess/tensor/abess.py |
| tensor_abess.tensor.bcd | src/tensor_abess/tensor/bcd.py |
| tensor_abess.tensor.sic | src/tensor_abess/tensor/sic.py |
| tensor_abess.utils.glasso | src/tensor_abess/utils/glasso.py |

探测步骤先尝试裸环境import，得到`ModuleNotFoundError`；随后按既有Apple入口`run.py:428`及集群`environment.sh:7`的显式src路径约定，使用`sys.path.insert(0,str(repo/'src'))`核对实际`__file__`，全部落在上表位置。不安装包、不替换核心、不修改环境。新入口必须显式继承这个约定，并在正式运行记录导入路径；这里的源文件身份检查不等于已经验收新入口或模型结果。

## 7. 尚由其他审计负责

本地ZIP CRC、全部DAT索引/标签、实测记录数、同X不同y、批次重复清除、全量张量坐标测试及运行时源码快照由数据与执行审计负责。没有官方服务器校验和时，本地SHA只提供可追溯指纹，不能写成官方认证。最终性能结论必须等待独立预测重放和指标复算。
