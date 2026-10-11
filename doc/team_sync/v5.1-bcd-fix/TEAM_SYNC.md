# Tensor-ABESS团队同步：v5.1-bcd-fix

同步对象A（HCP-YA）与B（NKI-Rockland）。当前交付只包含fixed BCD正确性修复，**MSIC参数仍为原v5默认**；不存在已批准的v6正式选阶版本。本次准备本地补丁与说明，没有向队友发送文件、创建PR、推送远端或改动远端main。

## 交付内容和身份

目录 `doc/team_sync/v5.1-bcd-fix/` 含bcd-fix.patch、MANIFEST.json、TEAM_SYNC.md、CHANGELOG.md。

补丁仅涉及：

- `src/tensor_abess/tensor/bcd.py`。
- 新增 `tests/test_bcd_fix_regression.py`。

基准Git HEAD：d4e28969ac480ba6efc01bcc985a53d179e7ed31。原源码保存在doc/algorithm_versions/v5-original，版本指纹在两版本VERSION.json。MANIFEST同时给raw SHA256和规范LF SHA256，避免Windows/Linux行尾差异误判。

最终patch SHA256：d2e6cac68ad7e37ec131cf2ff3d52eec4e7ec4498d30cea484d513fe2b7b37ee。原bcd规范LF SHA256：cff12bd6fa4c144031a41277a3a28fea6044fc7775ec2bd9ada7d582d6efdf51。最终修复bcd规范LF SHA256：4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c。

INVALIDATED_DRAFT.json仅保留旧暖启动草案作废记录，当前MANIFEST标记最终补丁已通过53项相关测试；不使用旧117525...补丁。

补丁已在原版源码临时副本上执行git apply --check和实际应用，并逐项核验得到与当前源码/测试一致的内容。

## 应用步骤

先保留已有源码与HCP/NKI旧结果，记录其版本/配置/种子，不覆盖旧模型缓存。

```bash
cd /你的项目目录
git status --short
git apply --check /补丁路径/bcd-fix.patch
git apply /补丁路径/bcd-fix.patch
export PYTHONPATH="$PWD/src"
python -m unittest tests.test_bcd_fix_regression -v
# 如果已有pytest：
python -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py -q
```

若git apply --check失败，先比较源码版本及本地改动，不强制覆盖。新unittest依赖现有NumPy/SciPy及标准库，不依赖pytest。核心仍只依赖既有库，**无需升级集群**。本地pytest工具是可选验证前端，不作为算法运行依赖。

## 行为与兼容性

原tensor_abess(...)和tensor_bcd(...)调用参数不变，当前src默认实际执行v5.1修复。需要重现原v5时显式使用原快照的src，并在新的Python进程中设置PYTHONPATH；不同版本不得混在已import tensor_abess的同一进程。

fixed结果新增best_iteration、best_rss、stop_reason、visited_loss_history、accepted_support_history（均有默认值）。loss_history现为初始状态+每次接受的mode更新；best_iteration索引visited_loss_history，不是outer cycle编号。n_cycles仍为实际外层循环次数。系数、支持、loss与best元数据对应同一完整OLS状态。

本补丁明确分离原确定性探索状态与正式最优接受状态：探索允许RSS回升，正式loss_history非增，返回所有有效完整OLS候选的best，整sweep探索状态重复时停止。原探索机会保留，避免暖启动草案丢失原恢复路径。fixed接受RSS非增也不意味着adaptive变化支持规模时RSS必须非增；MSIC默认未改。

## 队友需重跑的内容

旧HCP/NKI结果若用了修复前bcd源码，只能标为v5-original。修复影响全局fixed候选精修，因此不仅显式known-s实验，adaptive训练也可能改变。

采用v5.1后需要用相同受试者、响应、预处理及外层划分重新训练Tensor-ABESS，另建版本输出目录；旧B、支持、选阶结果和ABESS模型缓存不能直接作为修复版结果。可复用已经验收的未跨受试者标准化原张量和划分，**所有训练折预处理仍要重新拟合**。GLasso/Lasso/Oracle仅在配置、数据、划分和源码指纹严格一致时复用；不能把旧baseline与不同新划分随意合并。

比较原/修复版OOF预测、q分布、支持恢复/稳定性、训练与验证误差及运行时间；修复不要求一定提高数据集排名。若本轮之后形成经充分验证的v6，再独立同步MSIC代码/参数与实验版本，不提前替换默认。

## 复验后的同步决定

27组模拟中6组最终B改变，修复默认PMSE平均比原版增加0.302069；本次同步的是算法正确性修复，不能承诺预测变好。MSIC消融显示alpha2有预测收益也增加欠选，alpha3/CV在强信号不稳，且候选池存在真形状遗漏。本轮保持默认不变，不同步未批准v6。独立报告为doc/MSIC_ABLATION_REPORT.md；原结果只对应原源码，不可改标签冒充新版本。
