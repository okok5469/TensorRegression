# UCI270 类别计数冲突：全十批独立复核

**结论：冲突确认，气体名称仍未核定。** 2026-10-26 重新读取 D 盘十个 DAT，仅解析每行首字段 `gas_id;concentration` 中的整数 ID，不调用已有数据解析模块、不读取缓存或预测结果。按官方“编号→名称”段落把本地编号对应到官方批次表同名列，60 项中 16 项一致、44 项不一致；冲突涉及批次 1—9。所有批次总数一致，总记录数为 13,910。批10六类各600条，不能用于区分名称排列。

官方证据为 [UCI270 数据页](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) 的 **Additional Variable Information**：先列批次×气体计数表，随后列编号编码段落；本次在线重新取得完整十批数字。编码段落为 1 Ethanol、2 Ethylene、3 Ammonia、4 Acetaldehyde、5 Acetone、6 Toluene；计数表列标题也是这个顺序。下方完整保存官方60个数与本地60个数，避免只凭一个批次判定。

穷举六列的720种置换，只有一个计数签名映射能解释全部60项：本地 ID 1—6 依次匹配官方计数表的 **Acetone、Acetaldehyde、Ethanol、Ethylene、Ammonia、Toluene** 列。这只是“计数序列相同”，**不证明本地记录的化学身份，也不授权自动重命名**。可能是编码文字或表头存在错误，也可能存在版本/说明不一致；本检查不能独立裁决是哪一个。需要作者、维护者或能明确绑定这一版本原始编号的权威说明。

原 ZIP 只有十个 `batch*.dat` 成员，无 README 或独立类别字典。每个 ZIP 成员与 D 盘解压文件逐字节相同；本地 SHA256 提供追溯，不作为官方认证。下列 JSON 保存完整成员名、大小、CRC、ZIP SHA及各DAT SHA；CSV保存全部60项直接比较。

- [完整逐项CSV](class_count_reconciliation.csv)
- [证据和指纹JSON](class_count_reconciliation.json)

本次没有改动模型、缓存、既有报告或图，没有拟合、重排预测或修改科学结果。数字结果只能按原始 gas_id 追溯，现有带名称的图不能据此恢复有效。

**最终处置：依用户明确指令，放弃该数据集作为实证候选，不作为主实证；只按匿名 gas_id 留档，不承认计数诊断置换为气体名称，不发送邮件，不继续新研究。**

## 复核方法

以下代码读取保存的官方表数字，并重新计数原始文件、验证原文件SHA及完整60项；`official_counts`是本次从上方官方页面人工逐项转录的十行六列数据，表头保存在同一JSON中。重新运行前应再打开官方页面核对其是否发生更新。

```python
import json, hashlib, itertools
from pathlib import Path
from collections import Counter
j = json.loads(Path('doc/uci270_concentration/class_count_reconciliation.json').read_text(encoding='utf-8'))
actual = []
for entry in j['raw_files']:
    blob = Path(entry['file']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == entry['sha256']
    counts = Counter(int(line.split(None, 1)[0].split(';')[0])
                     for line in blob.decode('ascii').splitlines())
    actual.append([counts[g] for g in range(1, 7)])
official = j['official_counts']
assert actual == j['raw_counts_by_gas_id_1_to_6']
assert sum(actual[b][g] != official[b][g] for b in range(10) for g in range(6)) == 44
matches = [p for p in itertools.permutations(range(6))
           if all(actual[b][g] == official[b][p[g]] for b in range(10) for g in range(6))]
assert matches == [(4, 3, 0, 1, 2, 5)]
```

## 完整60项比较

下表气体名称仅表示“如果按官方编码段落直读”的预期名称，不代表已经确认。差值为本地原始计数减官方表计数。

| Batch | Raw gas_id | Name under published code | Official count | Local count | Difference | Match |
|---|---|---|---:|---:|---:|---|
| 1 | 1 | Ethanol | 83 | 90 | +7 | False |
| 1 | 2 | Ethylene | 30 | 98 | +68 | False |
| 1 | 3 | Ammonia | 70 | 83 | +13 | False |
| 1 | 4 | Acetaldehyde | 98 | 30 | -68 | False |
| 1 | 5 | Acetone | 90 | 70 | -20 | False |
| 1 | 6 | Toluene | 74 | 74 | +0 | True |
| 2 | 1 | Ethanol | 100 | 164 | +64 | False |
| 2 | 2 | Ethylene | 109 | 334 | +225 | False |
| 2 | 3 | Ammonia | 532 | 100 | -432 | False |
| 2 | 4 | Acetaldehyde | 334 | 109 | -225 | False |
| 2 | 5 | Acetone | 164 | 532 | +368 | False |
| 2 | 6 | Toluene | 5 | 5 | +0 | True |
| 3 | 1 | Ethanol | 216 | 365 | +149 | False |
| 3 | 2 | Ethylene | 240 | 490 | +250 | False |
| 3 | 3 | Ammonia | 275 | 216 | -59 | False |
| 3 | 4 | Acetaldehyde | 490 | 240 | -250 | False |
| 3 | 5 | Acetone | 365 | 275 | -90 | False |
| 3 | 6 | Toluene | 0 | 0 | +0 | True |
| 4 | 1 | Ethanol | 12 | 64 | +52 | False |
| 4 | 2 | Ethylene | 30 | 43 | +13 | False |
| 4 | 3 | Ammonia | 12 | 12 | +0 | True |
| 4 | 4 | Acetaldehyde | 43 | 30 | -13 | False |
| 4 | 5 | Acetone | 64 | 12 | -52 | False |
| 4 | 6 | Toluene | 0 | 0 | +0 | True |
| 5 | 1 | Ethanol | 20 | 28 | +8 | False |
| 5 | 2 | Ethylene | 46 | 40 | -6 | False |
| 5 | 3 | Ammonia | 63 | 20 | -43 | False |
| 5 | 4 | Acetaldehyde | 40 | 46 | +6 | False |
| 5 | 5 | Acetone | 28 | 63 | +35 | False |
| 5 | 6 | Toluene | 0 | 0 | +0 | True |
| 6 | 1 | Ethanol | 110 | 514 | +404 | False |
| 6 | 2 | Ethylene | 29 | 574 | +545 | False |
| 6 | 3 | Ammonia | 606 | 110 | -496 | False |
| 6 | 4 | Acetaldehyde | 574 | 29 | -545 | False |
| 6 | 5 | Acetone | 514 | 606 | +92 | False |
| 6 | 6 | Toluene | 467 | 467 | +0 | True |
| 7 | 1 | Ethanol | 360 | 649 | +289 | False |
| 7 | 2 | Ethylene | 744 | 662 | -82 | False |
| 7 | 3 | Ammonia | 630 | 360 | -270 | False |
| 7 | 4 | Acetaldehyde | 662 | 744 | +82 | False |
| 7 | 5 | Acetone | 649 | 630 | -19 | False |
| 7 | 6 | Toluene | 568 | 568 | +0 | True |
| 8 | 1 | Ethanol | 40 | 30 | -10 | False |
| 8 | 2 | Ethylene | 33 | 30 | -3 | False |
| 8 | 3 | Ammonia | 143 | 40 | -103 | False |
| 8 | 4 | Acetaldehyde | 30 | 33 | +3 | False |
| 8 | 5 | Acetone | 30 | 143 | +113 | False |
| 8 | 6 | Toluene | 18 | 18 | +0 | True |
| 9 | 1 | Ethanol | 100 | 61 | -39 | False |
| 9 | 2 | Ethylene | 75 | 55 | -20 | False |
| 9 | 3 | Ammonia | 78 | 100 | +22 | False |
| 9 | 4 | Acetaldehyde | 55 | 75 | +20 | False |
| 9 | 5 | Acetone | 61 | 78 | +17 | False |
| 9 | 6 | Toluene | 101 | 101 | +0 | True |
| 10 | 1 | Ethanol | 600 | 600 | +0 | True |
| 10 | 2 | Ethylene | 600 | 600 | +0 | True |
| 10 | 3 | Ammonia | 600 | 600 | +0 | True |
| 10 | 4 | Acetaldehyde | 600 | 600 | +0 | True |
| 10 | 5 | Acetone | 600 | 600 | +0 | True |
| 10 | 6 | Toluene | 600 | 600 | +0 | True |

## Original DAT fingerprints

| File | Bytes | SHA256 |
|---|---:|---|
| batch1.dat | 757889 | `6b62f3e71b90f95dd9c2157fa91ffce32571faa7f474dd52f29a4ae769144625` |
| batch2.dat | 2134300 | `4ea9490068457badf9efd0c22f359fbfce12c208cdd440526b97ac91d31b9002` |
| batch3.dat | 2725224 | `aabf06bf8b8200740156235dc59458e06d99aa8faba4d6764a22c6b6c2674ff4` |
| batch4.dat | 278400 | `ffc87c5146001bbe8ce90c4b2f7dad5b390ca3079f51b559032df53ffe5ff0af` |
| batch5.dat | 341379 | `dcad4939c300a6c7d9f41b9470a9253eecc0ebc5c2c4e940d8ae20b8fc4240d0` |
| batch6.dat | 3928247 | `297d3a8b594dee4e2f4d7d2c4f5c485939fa708c10b5a11ed5cce95e05049afb` |
| batch7.dat | 6159116 | `eb5801d38a59beaae4a2412195d8ded790175ed1611159e083649ee9fe03d150` |
| batch8.dat | 495298 | `58146b3e2dd68b125b2d5432cdd4968ec1fff47464ceb04282382181932d49be` |
| batch9.dat | 791293 | `51f368ae8dee5d285be7f2029770e10eb12f342578830bc40ea852db19d19c9b` |
| batch10.dat | 6086679 | `b8309e8ff205b9d0350761989809f31f8feaefb8582aecb05f1d0f694b91184e` |

ZIP SHA256: `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`

ZIP members: `batch10.dat`, `batch9.dat`, `batch8.dat`, `batch7.dat`, `batch6.dat`, `batch5.dat`, `batch4.dat`, `batch3.dat`, `batch2.dat`, `batch1.dat`.
