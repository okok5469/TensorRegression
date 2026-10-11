# LEMON EEG 增量预测价值实验

本模块对应 `empirical_analysis/任务卡.md`，仅只读复用既有QC分析数据；核心算法和旧实证结果不写入、不重算EEG。

## 设计

外层为预先固定种子的3次5折、内层3折，年龄分类分层均衡划分；所有模型共享外层测试受试者。协变量含截距、原编码年龄哑变量与性别。每个内层训练折独立拟合X/y残差化、残差中心化与EEG标准差。Vector Lasso、Tensor-GLasso及Ridge由外部内层验证选择参数，评价完整log(TMT_5)预测。Tensor-ABESS保留既定BIC初始化和自适应MSIC，不外部调整其选阶参数。

Ridge采用RSS/(2n)+lambda||beta||²/2的对偶解；高维tensor依照NumPy C-order展平。稀疏模型使用固定lambda/lambda_max比例，lambda_max只来自相应训练数据。网格与种子均在config.yaml中预先保存。完全相同的数据/代码/环境才允许读取断点；每个外折模型完成后先保存预测和参数，再写完成标记。部分结果不声称完成重复CV。

## 执行

使用项目已有Python环境，从项目根目录执行，不需要安装或升级环境；本地可用 `.venv/Scripts/python.exe` 替代 `python`。

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest discover -s empirical_analysis/lemon/incremental_value -t . -p 'test_*.py' -v
python -m empirical_analysis.lemon.incremental_value.run --audit-only --output-dir empirical_analysis/lemon/outputs_eeg_incremental_audit
python -m empirical_analysis.lemon.incremental_value.run --smoke --output-dir empirical_analysis/lemon/outputs_eeg_incremental_smoke
python -m empirical_analysis.lemon.incremental_value.run
```

smoke只运行一个真实外折、五个模型，单独输出，不形成正式OOF指标；正式实验保存到 `empirical_analysis/lemon/outputs_eeg_incremental/`。运行中断后使用相同正式命令继续；输入、配置或生产代码不同则拒绝混用旧断点。

主要交付为reports/EEG_INCREMENTAL_REPORT.md、reports/EXECUTION_SUMMARY.md，cv下OOF/参数/指标/逐人改善CSV，diagnostics下输入指纹与审计，以及figures下五类PNG/PDF图。重复CV同一受试者的三次预测不作为独立样本；区间是对固定验证预测作受试者聚类bootstrap的描述性不确定性，不代表因果效应或严格总体显著性。

绘图使用Python/Matplotlib；外部Nature Figure审计工具可通过NATURE_FIGURE_SKILL_ROOT指定，纯统计和建模本身不依赖该工具。
