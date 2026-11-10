# 方法定义和边界

对当前训练集合T的完整类别k，pi_k=mean(1[y=k])，z_k=1[y=k]-pi_k。仅用T拟合逐坐标均值和ddof=0标准差，把X转为float64标准化张量；常数坐标置0。逐类调用正式tensor_abess(X_std,z_k,**core_profile)。类别可以有独立三轴矩形支持和不同规模。

decision_function返回pi_k+<X_std,B_k>，列与classes_对应；predict按最大分数。分数并列容差为64*float64_eps*max(1,max(abs(scores_i)))；先训练频数、后训练固定行序中首次出现位置。标签数字只作身份；API支持同类型整数或字符串，禁止混合object标签。预测需同尺寸同轴顺序。

save/load为NPZ＋JSON，NPZ以allow_pickle=False读取；包含系数、支撑、频率、标签、z-score、常数掩码、并列优先序和代码/profile身份。实际计算以标准化系数为准；如换原单位，B_raw=B_std/scale，intercept=pi-sum(mean*B_raw)，严格常数系数按实际置0处理。模型无predict_proba。

公开API（先划分、只把当前训练行传给fit；axes为数据合同中的三个轴，以下变量由调用者按sample_id准备）：

```python
from tensor_abess_ovr import TensorABESSOvR

model = TensorABESSOvR(classes=["a", "b", "c"], axis_metadata=axes)
model.fit(X_train, y_train, sample_ids=train_sample_ids, axis_metadata=axes)
scores = model.decision_function(X_valid, axis_metadata=axes)
labels = model.predict(X_valid, axis_metadata=axes)
class_columns = model.classes_
model.save("my_model")
restored = TensorABESSOvR.load("my_model")
```

包导入时在数值库导入前固定线程环境为1，fit/load核验实际依赖、活动线程池与release身份。若调用进程已经激活多线程数值库，核验会明确拒绝；从新进程优先使用README的CLI。轴元数据省略时沿用保存的坐标语义，不能检测未声明的同尺寸数值换轴。

原MSIC是平方损失工作准则。没有多项似然解释、共同支撑优化、联合多响应保证或分类一致性结论。原500步初始化与全部搜索边界冻结；返回成功不证明每个内层凸问题已得最优性证书。

对照为训练多数类、全展平Ridge-OvR、L2多项Logistic、L1多项Logistic。各自训练内z-score和调参，原尺度最大输出分类；Logistic概率如保存应另标注。Macro-F1按类别全集/zero_division0计算；Balanced Accuracy按有真实样本类召回均值，缺覆盖明确标为描述值；不把折作独立重复做t检验。

每类报告三轴支撑、q=三轴规模乘积、abs(coef)>1e-8实际非零数；总容量与支撑并集仅描述，不再拟合，不称完整统计自由度。真实数据不报真支持Exact/REE/FDR或因果发现。HAR频带特征为固定加窗rFFT平方和的log1p，不是规范化PSD或方差精确分解。
