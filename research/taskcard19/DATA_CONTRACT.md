# 唯一数据合同：schema 1.0

目录内放X.npy、samples.csv、dataset.json、splits.json；运行protocol.json可放任意位置。可复制安装包src/tensor_abess_ovr/fixtures/tiny的完整小例子。UTF-8，中文/空格路径可用；无pickle或object数组。

X.npy必须float64形状(n,p1,p2,p3)，C顺序坐标，样本与坐标不自动删减。samples.csv必需row_index,sample_id,class_key，group任务加group_id，time任务加time（有限数字或带时区ISO时间）。row_index完整唯一覆盖0..n-1，与X行明确对齐；CSV本身可乱序。类别key按字符串读取，不猜数字类型。sample_id必须唯一。

dataset.json至少包括：

```json
{
  "schema_version": "1.0", "name": "my-study", "source": "official source or study provenance",
  "axes": [{"name":"site","coordinates":["A","B"]},{"name":"direction","coordinates":["x","y"]},{"name":"feature","coordinates":["f1","f2"]}],
  "axis_order": ["site","direction","feature"], "x_shape": [42,2,2,2],
  "classes": ["walk","sit","stand"],
  "class_labels": {"walk":"walking","sit":"sitting","stand":"standing"},
  "independent_unit": "participant", "dependency": "group",
  "representation_version": "my-fixed-construction-v1",
  "source_files": {"original-file":"SHA256"},
  "processed_files": {"X.npy":"SHA256","samples.csv":"SHA256"}
}
```

上述42只是例示，须替换为真实n。轴顺序、坐标及表示都需科学解释。source_files记录原始输入身份，processed_files写当前处理文件真实SHA256（不能填占位字）。dependency可选iid、group或time；不强迫所有任务同时group与time。

splits.json使用sample_id，不能写隐含行号：

```json
{"schema_version":"1.0","seed":20261031,"rule":"predeclared participant holdout",
 "splits":[{"split_id":"dev1","purpose":"dev","evaluation_role":"validation",
 "train_ids":["s001","s002"],"eval_ids":["s101","s102"]}]}
```

例子ID列表须替换为完整列表，每个训练集合须覆盖classes。正式保留测试用purpose=final、evaluation_role=test；开发与测试sample/group必须互斥。time任务训练时间严格早于评价时间。可提供inner_folds，每折同样含train_ids/eval_ids且只能在外层train内；否则runner按固定seed生成覆盖合法的3折，预声明必要时2折。无合法折则失败，不混组补齐。

protocol.json必须显式冻结：

```json
{"evaluation_stage":"dev","active_split_ids":["dev1"],
 "preprocessing":"train_coordinate_zscore_v1",
 "core_profile":"v5.2-search-final-original-msic",
 "primary_metric":"macro_f1","seed":20261031,
 "resource_budget":{"per_class_seconds":900,"total_seconds":7200}}
```

建议expected_axes复制dataset.axes以绑定准确坐标语义。默认只选择dev；stage与所选purpose不同会报错。正式评价另存final配置和身份，不能顺便遍历所有split。NaN/Inf、缺失/未知标签、错位ID、重复ID、hash不符、轴不符均明确失败，无自动插补或删样本。对未声明且尺寸相同的任意数值轴交换，接口不能凭数值自动辨识；适配器负责真实映射。
