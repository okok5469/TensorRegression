# UCI270 气体编号权威来源追查

日期：2026-10-26。状态：**USER_ABANDONED_NO_CONTACT**。

用户最新决定为“不发邮件，放弃”。已停止后续检索和联系；未发送邮件、表单或其他外部消息。现有结果仅按匿名原始`gas_id`保留研究记录，不作论文主实证，不更名、不改模型或冻结数据。

## 已取得的证据及其限度

| 来源 | 本次实际核查结果 | 能否独立确定本地DAT的六类编号 |
|---|---|---|
| [UCI270官方页面](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) | 编号段与气体×批次数量表不能同时解释本地原始编号；详见既有SCIENCE_REVIEW | 否，官方页面内部证据存在矛盾 |
| [Fonollosa、Rodríguez-Luján、Huerta，2015，Chemical gas sensor array dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC4510048/) | 检索服务可读取Methods、Table1、Table2及特征提取说明；Table2列名/计数与UCI批次表一致；未取得将本地DAT整数1—6直接对应到名称的额外字典 | 否，表格只再确认计数表，不能据此自动置换编号 |
| [原作者Jordi Fonollosa的Public Datasets页](https://jordifonollosa.wordpress.com/downloads/public-datasets/) | 浓度漂移数据条目指向UCI，并列2012/2014两篇引用；未见该条目的编号勘误或独立映射说明 | 否，回链同一来源不构成独立消歧 |
| [同一作者的Code页](https://jordifonollosa.wordpress.com/downloads/code/) | 实际列出居家活动监测、储备池计算和空气质量监测代码；未列出UCI270编号修订字典 | 否；没有把不相关作者代码当作270证据 |
| [Rodríguez-Luján等，2014，NIST作者机构保存的论文PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=914227) | PDF第2页§2.1说明1800条、一个月、乙醇/乙烯/乙醛三类；脚注链接带浓度UCI数据；第3页说明128特征构造 | 否，三类实验说明不等于完整六类DAT的编号字典 |
| [Vergara等，2012，原始研究出版页面](https://www.sciencedirect.com/science/article/pii/S0925400512002018) | 可检索原作者、研究摘要及数据收集/实验片段；涉及长期同阵列数据，但本次未取得其中直接解释六类DAT编码的证据 | 否；未声称已经读到无法访问的完整附件 |

上述2015数据论文DOI为[10.1016/j.dib.2015.01.003](https://doi.org/10.1016/j.dib.2015.01.003)；2014论文DOI为[10.1016/j.chemolab.2013.10.012](https://doi.org/10.1016/j.chemolab.2013.10.012)。这里记录的是实际可核查内容，未把检索未发现写成“所有权威渠道都不存在”。

## 访问与检索边界

PMC直接打开曾出现浏览器验证，但检索索引能返回该数据论文的Table2；UAM机构仓储PDF直接打开返回403，Europe PMC XML通过网页工具不可访问。本地Python获取公开XML遇到网络权限错误，未提升权限或伪称下载成功。NIST的2014论文PDF正常打开，已按实际页内容检查。

检索遇到若干第三方论文、代码库与重排标签表，均未用来解决原始数据身份；没有从浓度范围、结果好坏或数量匹配推断物质名称。作者个人主页的公开联系方式已找到，但遵照最新指令不联系，也不保留待发送操作。

## 未解决结论与保留方式

截至用户停止时，尚无足以将**本地实际DAT版本**的原始`gas_id`与六种物质名称唯一对应的新增权威证据；官方编号段与官方/作者计数表的冲突保持未解决。[SCIENCE_REVIEW.md](SCIENCE_REVIEW.md)中的语义Gate失败仍成立，前期数值验收不能替代该Gate。

本调查没有修正气体名称，也没有按计数重排类别。既有匿名ID内模型、预测、审计和失败过程可保留作研究记录；已生成的具名报告不能视作完成语义验收的正式实证，更不能作为论文主实证。全10批原始计数与包内容核验由主线另行保存，本文件不冒认尚未收到的其他核查结果。
