# 队友任务卡：统一多分类实证

1. 在原项目外解压本ZIP，依README_ZH.md建立全新CPython3.12环境安装wheel。执行verify-release和smoke；保存环境及实际import身份，失败时回传原日志。
2. 按DATA_CONTRACT.md准备X.npy、samples.csv、dataset.json、splits.json和独立protocol.json。三轴应有稳定科学意义；row_index与X逐行对应，sample_id唯一，class_key按字符串身份读取。说明独立单位，有实体依赖则group_id必需，时间任务另声明时间边界。不要随意reshape向量制造轴。
3. 方法身份固定：K列0/1响应仅中心化，原回归器独立调用并恢复截距。固定profile的MSIC/c/tau/搜索预算不能按成绩修改；不加softmax、概率校准、共享支撑或分类头后仍用相同方法名。
4. 先在开发划分评价统一五方法；每折训练内预处理，基线内层调参遵守group/time边界，主指标Macro-F1。保持相同输入、类别及样本。负结果、慢类和失败也完整保留，不私下换seed/类/测试分割。
5. 开发完成后另行冻结正式完整协议和final配置，再评价一次封存测试；常规运行无需反复征询原作者。已在探索中评价过的数据不能重称未见测试。HAR开发400样本上限只属于本机筛查，正式训练量必须在队友自己的协议里明示且对方法一致；子集训练只支持子集结论。
6. 每次用全新输出目录执行README中的validate-data、run、validate-return、pack-return。回传统一ZIP；原始X与大环境默认无需发送。结果不好也回传；PARTIAL失败包必须保留失败身份，不称完整比较。

提交前请确认RUN_REPORT.md、manifest、配置/划分/环境、predictions、scores及行/列映射、metrics、混淆矩阵、supports、fit_diagnostics、保存模型和SHA清单齐全。用户收到结果ZIP即可独立重算。后续正式数据执行由队友实际完成，不属于原作者本机已完成工作量。
