# 近期工作汇报材料

- `近期工作汇报.pdf`：15页教师汇报，2026-10-26修订；算法更新、正式模拟，以及LEMON、Apple、Mushroom、北京四项实证对照。
- `近期工作报告.md`：可单独阅读的通俗版工作报告，含关键结果表和结论边界。
- `近期工作汇报.tex`：可编辑LaTeX源文件，沿用原模板的4:3比例、CambridgeUS主题、蓝色及字体。
- `plot_report.py`、`figures/`：从已归档结果生成的报告用图，不含新模型拟合。
- `plot_empirical_comparison.py`：四项实证的R²比较与Mushroom两种验证协议图，源数据及审计单独留存。
- `qa/`：编译日志、逐页渲染和最终检查记录。

在本目录中，用已安装的XeLaTeX编译两次：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=qa 近期工作汇报.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=qa 近期工作汇报.tex
Copy-Item -LiteralPath 'qa/近期工作汇报.pdf' -Destination '近期工作汇报.pdf'
```

图表更新请使用本项目Python环境执行相应绘图脚本，并重新进行图表和逐页版面检查。不要改动或重拟合已归档实验来适应汇报结论。

本轮调整：北京未超过Lasso、Tensor-GLasso及完整张量Ridge，撤下原汇报的“主实证候选”定位；四项实证目前均未充分建立方法比较优势。此处更新当前汇报判断，原封存实验及其历史报告保持不变。旧北京时间序列等图仍保留为可追溯材料，不再占用修订版实证比较页面。
