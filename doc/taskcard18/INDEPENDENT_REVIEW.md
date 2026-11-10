# 任务卡18最终独立审查

status: PASS

日期：2026-10-30（北京时间）。范围限于任务卡18的方法章、模拟章、附录、表图及其证据链。审查独立于正式算法与参考对照拟合、正文撰写及绘图制作；审查者此前实现了只读回传验收入口，因此另以独立公式重算统计、核对保存数组和逐项阅读正式源，未将验收程序自身的输出当作全部审查。

## 结论与核验

1. **正式身份与方法定义通过。** 开始审计、release和结束时的32份正式生产源一致，38项模拟源指纹一致，实际导入来自本仓库src。三份冻结数值适配源及协议未变。方法章的初始化/BIC、预筛与caps、初始支撑不裁剪、逐模组交换、Cold完整best-visited、adaptive终态、三阶段MSIC系数、首个最低评分容量及确定性容忍带均与实现对应；未把规模网格写成位置穷举或全局最优，也未新增理论保证。
2. **对照和完整性通过。** 两目标仍为RSS/(2n)加L1或无权重组范数，无额外截距/标准化；五折有序划分、同折路径、验证SSE、并列规则及最终全训练重拟合清楚。顺序Tensor-GLasso保留局部设计、非空回退和矩形OLS，并明确全训练预筛/其他轴支撑依赖，未称严格嵌套CV。600完整输入×2方法的1200模型、1200评价及965964份必要凸拟合证书均齐备，无未解失败；所有保存证书满足KKT/gap双条件。最大KKT/λ为9.999631424633963e-5，最大归一化gap为9.999986033866033e-7；CV实际训练索引、样本数、最终响应身份和状态来源已核对。此为保存证书/源码及数组审查，没有重新求解全部内折问题。
3. **统计与支撑通过。** 700标签映射600唯一完整输入和10配置，基准R=150、其余R=50；未因参数相同删去不同输入，也未把别名增加为重复。实际元素、声明轴支撑、实际投影及矩形容量分别定义；空发现Precision的NA与有效分母保留。由逐输入数据独立重算530项均值、SD、t区间、Wilson及配对统计字段通过；逐项核对模拟章主表90个PMSE/SD/Exact显示、图源52行通过。补表中1031条规模频数、2800项结构摘要及180项新旧配对统计通过，旧PMSE的2400条唯一方法记录另与原任务JSON核对一致。
4. **预测复核层级通过。** 全1200新评价的模型、输入、运行身份和五批测试流记录已核对；另外固定预声明20输入×4保存模型的80次预测重放，240项PMSE/SMSE/R²数值最大绝对差1.7763568394002505e-15。后者只覆盖固定20输入，不声称独立重算全部1200预测。跨平台测试流逐字节哈希多数不相同；这与数值对应结论分开，不把原未保存测试数组写成已有字节副本。Linux全部600训练输入四项指纹一致的结论也不外推至测试数组。
5. **图表与论述通过。** 四图八面板已可见检查，图例、方法线型、误差棒及轴标清晰；183×82mm、7pt、PDF/PNG/SVG与自动字形/对齐/碰撞审查一致。最终QA绑定绘图源、8个输入及16个输出，逐面板可见审查文字已修复为可读UTF-8并重新绑定。先前重渲染期间的暂时hash差异和该QA文字缺陷均已解除，数值结果未因此修改。51项限定测试日志已阅，返回reader的身份、图表及独立审查绑定代码已检查。
6. **主张边界通过。** 正文保留n100 caps排除真形状、SNR0.5/1预测劣势、大维度/rho0差区间跨零、强相关下预测与REE/位置恢复分离、n400接近Oracle而非证明等效。GLasso下界7420/8964明确披露，充分求解不被解释为网格/流程充分调优。结论只对应本批两个具体参考流程、单一高斯族、连续真矩形(4,3,5)；没有沿用旧11/14胜出、跨配置IID检验或宣称任意张量方法优势。方法/模拟两章完成不等于理论、实证、新颖性或整篇论文投稿条件完成。

无尚未解除的实质问题。该PASS支持主线按实际证据完成G0—G7与两章联合定稿；最终状态由重新运行的返回reader及最终验收索引记录，不能仅由文件存在推断。

## 最终文件绑定

以下SHA256在上述审查结束时逐一计算。核心章、必要补表、证据或代码变化后须重新核验；RETURN_VALIDATION同时绑定原ZIP、运行身份及全部五份认证CSV，图QA继续绑定具体图表输出。

| 文件（仓库相对路径） | SHA256 |
|---|---|
| `paper/sections/02_method.md` | `7ba0bc5dec19283f1aefa189d1d1c10da35087d6dbc4f0c59a57d25c853475ac` |
| `paper/sections/04_simulation.md` | `19c6460f4b15d4691c43d350bbbe5c82374b5c38bb6f83dcc32c9cc2bda90b48` |
| `paper/appendix/simulation_protocol.md` | `d7081436c951775ebd792821c02f661bfc1bdb6a3eb5e75eafe4099a5fecebaa` |
| `doc/taskcard18/figure_qa/FIGURE_VALIDATION.json` | `9ed6ecce7d4801907b3b2e908e2a6f1b585cf1c8ba33a2545bb1a6bd64521e9c` |
| `doc/taskcard18/results/certified_config_summary.csv` | `fd56df1d27e662880c5c04210a885163932410069497b6df846162bfa648e928` |
| `doc/taskcard18/results/certified_paired_pmse.csv` | `6d51922eaebd0d504132d3d7285bb4632c97a9e79e8f3c14d213074c68f26858` |
| `paper/tables/baseline_update.csv` | `d0361849ccfb9922f69a631946216d6d97478c392ef43107c48c760467c09f3f` |
| `paper/tables/support_scale_distribution.csv` | `6c180862dd909cc6adb01307d73b1707bbda8d470a5b8e27e9c0e4f3a1a12a24` |
| `paper/tables/support_structure_summary.csv` | `da50f53ed152fab981584a27d4bb33678bb980b2c1f4b8bf5a5fc55bd1be6d8b` |
| `research/taskcard18/return_report.py` | `789ec68aa6e3555f9b80061078838c869b82e77e012621051bab376cf855f9e8` |
| `research/taskcard18/return_tables.py` | `694c9e5fe310a86cb4dfa4dacef16ce8447d80db682bc7d19d35598b3fecf859` |
| `research/taskcard18/return_replay.py` | `f03ee86b453b9174f899397571731d219f1d54c416ea7cba18499ecccf15ee27` |
| `doc/taskcard18/cluster_review/PREDICTION_REPLAY.json` | `1c20615faa15fe74d411a7fde8a273d7c118c5940b601531d213a69edda38f39` |
| `doc/taskcard18/cluster_review/RETURN_VALIDATION.json` | `4f04fc7d1e23279a6e67c753528ec540139a3a07ca8829bf741363cdccc65bf1` |
| `research/taskcard18/return_audit.py` | `b520fe7c779ed8efe6bc847ed817909055c0d36bbe64441ea50165ed7fe49f94` |
| `research/taskcard18/figures.py` | `6233c291ff11eb37f46d1fdbc0e823fb5b3fb4c3c23d0838371667f46e2766b2` |
| `doc/taskcard18/CLAIM_EVIDENCE_MAP.md` | `476c218f9b6d9adae31de61282a9081bce39b5d20f22c26f3538d74b331510f2` |
| `doc/taskcard18/THEORY_INTERFACE.md` | `4d8ebe42f2393999d0103dd63e5afbd6dd81c02509e8bb86e86f344ac89f4c9e` |
| `doc/taskcard18/figure_qa/certified/PANEL_REVIEW.md` | `f6a236b2061e818f5ee332ef26032a2b95f0d520335d95c0ec283728e1c8def4` |
| `doc/releases/v5.2-search-final/RELEASE_MANIFEST.json` | `b415a151674f1ee0400acbd000aa732e2c99343c7ad4faa48d7970b2cfe56074` |
| `doc/taskcard18/CLOSEOUT_PROTOCOL.json` | `a794d6a0efef81acf9d69b3429d8fe98c9d18131b8412ca70337fa4cbaebfa61` |
| `doc/taskcard18/EXECUTION_AMENDMENT.json` | `9518f02a99ba3115ab1d112e5cb231864d2d448d4c552102a49ca94de8eba3ed` |
