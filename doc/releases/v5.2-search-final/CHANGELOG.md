# v5.2-search-final 正式发布变更

2026-10-11，用户在任务卡5明确批准D2，本地生产源码正式采用v5.2-search-final。

- 唯一修改的核心文件：`src/tensor_abess/tensor/bcd.py`。
- 唯一逻辑改动：在旧训练caps以内，完整枚举满足容量约束的正整数规模三元组，避免旧稀疏网格遗漏规模。
- 保留v5.1 fixed BCD正确性修复、adaptive、旧caps及初始化、Group-BESS、OLS、原MSIC和公共接口。
- 原MSIC为mode alpha1，global alpha1.5/gamma1/one-SE scale1；未采用alpha2/scale0及A1/A3/A4研究规则。
- 私有规模函数去除无作用window参数，空合法域返回空列表；不添加研究候选预算或修改公共参数。

32文件逐字匹配任务卡4冻结D2，31核心文件相对v5.1未变；完整raw及规范LF指纹见RELEASE_MANIFEST.json，SOURCE_SHA256.txt为发布文件原始字节指纹。Windows/Linux可按manifest的canonicalLF清单核对相同行尾语义，冻结计划之后不再改变源字节。

v5-original、v5.1源码快照、v5.2历史研究VERSION及其未批准状态保持原样；本次批准和发布身份独立记在本目录。历史结果不改标签、不覆盖。PRE_RELEASE_SOURCE_SHA256.json记录发布前32生产指纹，v5.1完整快照可恢复原版本；已有search-final.patch为最小增量补丁。

既有HEAD为d0c3547704fbaf5d90a3e5423568768a1ba11759，本次没有新Git提交、分支、worktree或推送。仓库有大量既有未提交修改，直接形成发布提交会混入无关工作；因此保留基于已验证v5.1的独立patch及完整源码清单，不能把既有HEAD称为本次发布commit。
