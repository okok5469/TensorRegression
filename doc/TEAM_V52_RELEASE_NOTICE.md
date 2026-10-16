# Tensor-ABESS v5.2 团队交接通知

日期：2026-10-11。任务卡5已明确批准 D2 / `v5.2-search-final`，本地正式 manifest 已标记 `release_status=released`，32生产源码与冻结 D2 raw指纹一致；本通知供团队使用，本轮未发送消息、修改队友机器、上传文件或启动实证。正式身份与验证见 [RELEASE_MANIFEST.json](releases/v5.2-search-final/RELEASE_MANIFEST.json) 和 [RELEASE_VALIDATION.md](releases/v5.2-search-final/RELEASE_VALIDATION.md)：本轮发布范围50项通过，另有未改动旧known-s脚本的一项静态断言失败单列记录，不代表全仓库或真实集群通过。历史研究 `VERSION.json` 与旧同步包的 `release_approved=false` 保留其当时状态，不作为新的发布记录。

## 版本变化与范围

| 对比基准 | 获批变化 | 保留内容 |
|---|---|---|
| 已核验的 v5.1-bcd-fix | 仅 `src/tensor_abess/tensor/bcd.py` 的私有规模网格：旧 caps 内全部合法整数三元组，`q <= floor(0.7*n_train)` | fixed BCD 修复、旧 caps、初始化、adaptive 主体、Group-BESS、OLS、矩形稀疏结构、公共参数与默认值 |
| v5-original | 先核对完整32源码与 v5.1 修复，再采用上述 D2 增量 | 不得把只上传一个 bcd 文件视为整个旧项目已兼容 |

这不是支撑位置穷举，也不放宽 caps；未混入 A1/A3/A4、研究候选接口、warm-only 草案或新 MSIC。最终 MSIC 保留 mode alpha=1，global alpha=1.5、gamma=1、one-SE scale=1。80组新 MSIC 独立确认未完成，不使用不完整结果批准其参数。

## 完整来源与指纹

- 正式生产来源：`src/tensor_abess/`；完整32文件清单与 SHA256：[正式 manifest](releases/v5.2-search-final/RELEASE_MANIFEST.json) 的 `source_sha256` / `source_canonical_lf_sha256`、[SOURCE_SHA256.txt](releases/v5.2-search-final/SOURCE_SHA256.txt)。v5.1基准完整32的 `base_source_sha256` / `base_source_canonical_lf_sha256` 也位于正式 manifest；必须核对完整文件集合，不能只看 bcd 或 Git HEAD。
- D2 冻结来源：[历史 VERSION](algorithm_versions/v5.2-search-final/VERSION.json)、`doc/algorithm_versions/v5.2-search-final/src/`；跨系统规范 LF 的完整32指纹见[冻结同步 manifest](team_sync/v5.2-search-final/MANIFEST.json) 的 `source_canonical_LF_sha256`。
- v5.1 基准完整32 raw指纹：[v5.1 VERSION](algorithm_versions/v5.1-bcd-fix/VERSION.json)；跨行尾基准核对使用其保存的 `src/` 逐文件规范 LF 内容。
- 最小补丁：[search-final.patch](team_sync/v5.2-search-final/search-final.patch)，SHA256：`23524a12a3ccd502b058e5eec7ef984abae43bccc085d467ed292b1d3585b3db`。不要使用研究 `candidate-search-core.patch` 代替。
- 补丁基准 bcd：raw `c5e67a666c8c120bfcd9ff000a5c6b1bcd7c0be37e0b549a0a34ca61ecfbd737`；规范 LF `4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c`。
- D2 冻结 bcd：raw `db2999e8875acd06fdee2253884fd1c9c1d768ecadb41ef9059969bae0201fd7`；规范 LF `eb4f01626e0108f4cf87a604835aab7fa99866f484abaea23ce54b8911f063d0`。实际部署 raw 行尾身份仍按目标正式 manifest 记录。
- 已观测 Git HEAD：`d0c3547704fbaf5d90a3e5423568768a1ba11759`；它不是新增正式发布 commit。原 v5.1 快照历史基准 commit 为 `d4e28969ac480ba6efc01bcc985a53d179e7ed31`；使用 patch 与完整源码指纹交接，不以 commit 名称替代文件核验。

## 队友更新与验证

1. 在各自项目先记录 Git 状态、完整32源码身份、依赖版本与本地修改；保留现有结果。目标若非完整已验收 v5.1、存在源码冲突或无法恢复旧版本，停止应用并交回差异，不能强制覆盖。已为 D2 的项目核对最终32指纹，不重复 apply。
2. 向已确认 v5.1 的目标仅应用最小 patch，随后再核对完整32文件。队友的 LEMON/HCP/NKI 适配器、数据、预处理与划分不随 core patch 自动改变；需各自在原入口验证 core 导入来自自己的生产 `repository/src`。
3. 沿用各项目既有 Python/NumPy/SciPy 与实证依赖，不复制本机 `.venv`，不安装或升级库。pytest 是可选测试前端，不是算法运行依赖；无法运行既有回归时记录未执行，不能宣称通过。

在已经核对基准、保留恢复材料的 Linux 目标项目根执行（`/交接路径` 替换为实际已收到文件位置）：

```bash
git status --short
git -c core.autocrlf=false apply --ignore-space-change --check /交接路径/search-final.patch
# check 成功且完整32基准匹配后才执行：
git -c core.autocrlf=false apply --ignore-space-change /交接路径/search-final.patch
export PYTHONPATH="$PWD/src"
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_bcd_fix_regression -v
# 使用项目原有 pytest 环境时：
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_tensor_bcd.py tests/test_bcd_fix_regression.py -q
```

应用后，以完整规范 LF 指纹匹配 D2；行尾兼容不能用于掩盖实质源码差异。生产导入与指纹扫描均须指向同一 `repository/src`。本轮没有为队友创建分支、commit 或 push。

## 三实证交接与历史保留

| 实证 | 需要更新与复验 | 本仓库核查边界 |
|---|---|---|
| LEMON EEG | 已核验 v5.1 目标应用 core patch；原 `incremental_value.run` 使用新独立目录，复验145人、年龄/性别设计与固定外内划分 | 真实入口、配置、旧输入/结果均已只读核查；本轮不拟合 |
| HCP-YA MRI | 核对队友完整 core 与自己的 MRI 模态顺序、响应/协变量、内外划分，应用匹配基准的 patch 并在原入口独立输出 | 本仓库没有其本地入口/数据；未访问队友工程，不提供猜测 CLI |
| NKI-Rockland MRI | 同上；使用自己的原入口与独立新版目录，并交回输入/配置/划分/源码/环境身份 | 同上，未验证或启动其运行 |

保留 `v5-original`、`v5.1-bcd-fix`、D2 研究快照及任务卡2–4证据；原模拟/LEMON/HCP/NKI输入、模型、完整OOF与运行记录保持原位置和原身份。旧结果不能换成 v5.2 标签。

规模候选集合变化可能改变选阶、支撑、OLS系数和预测，因此全部 ABESS 训练折重新预处理、搜索、选阶及拟合；不能复用旧支撑、系数、q、模型缓存或修改旧断点身份。对照方法仅在独立源码、参数、输入、训练/验证划分、预处理、种子与环境逐项相同且完整性核验通过后，保留明确旧来源复用；本轮不复制旧缓存。具体操作与旧版比较见 [V52_EMPIRICAL_RERUN_PLAN.md](V52_EMPIRICAL_RERUN_PLAN.md)。
