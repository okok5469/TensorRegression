# 谨慎清理候选：只读审计（2026-10-26）

本次未删除、移动或修改任何被审计文件，仅新增本清单。下列路径均相对仓库根目录；字节数为检查时的文件逻辑大小，不等于保证释放的磁盘空间。候选限于可重建Python缓存、空临时目录和一份无研究结果的控制台日志；旧发布验收副本、研究提取文本及已封存LaTeX辅助文件全部保留。

## 检查范围与引用证据

Protected directories excluded from traversal: `.git`, `.venv`, any `runtime`, `doc/algorithm_versions/test_tools`, and `doc/algorithm_versions/taskcard4_test_tools`. These are not cleanup candidates. Original data, trained models, failed-study evidence, historical controls and dependency installations are retained.

Inspected 70 manifest/contract/delivery/source-identity/hash files selected by filename markers `delivery_index`, `contract`, `manifest`, `source_sha`, `source_hash`, `source_identity`, `fingerprint`, `sha256sum`, `source_fingerprint` (JSON/Markdown/text/SHA256). JSON escapes were decoded, path separators normalized, and each candidate file was checked by relative path and SHA256; directories were checked by path and every contained file. No candidate below matched these references.

辅助 `rg -n --hidden -F` 搜索返回零匹配，但该命令仍遵守仓库ignore规则，不能据此判断无引用。最终候选以Python遍历、JSON解码、规范化路径及SHA256逐项核对为依据：它实际查出了Presentation辅助文件的封存引用，已全部排除。这仍是列明范围内的检查，不承诺覆盖仓库外清单；未来清理前必须再次核对最新交付索引。

## 可重建的Python缓存候选

Every nonempty candidate below contains only `.pyc` files with an existing corresponding sibling `.py` source. Those source files were not changed. Empty cache directories contain no scientific records. All rows had zero matching seal references; regenerating bytecode leaves source and model mathematics unchanged. No whole parent directory is a candidate.

| Exact directory | Bytes | Files | Reference evidence |
|---|---:|---:|---|
| `doc/beijing_air_quality/__pycache__` | 57948 | 4 | 0 path/SHA256 matches; source exists or empty |
| `doc/mushroom_empirical/__pycache__` | 59074 | 3 | 0 path/SHA256 matches; source exists or empty |
| `doc/taskcard13/__pycache__` | 16038 | 1 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/beijing_air_quality/__pycache__` | 51491 | 4 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/lemon/incremental_value/__pycache__` | 15954 | 2 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/lemon/__pycache__` | 15602 | 3 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/spectrofood_apple/__pycache__` | 103910 | 4 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/spectrofood_mushroom/__pycache__` | 40334 | 4 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/uci270_concentration/__pycache__` | 214642 | 14 | 0 path/SHA256 matches; source exists or empty |
| `empirical_analysis/__pycache__` | 34582 | 2 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/a2_final_validation/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/a2_msic_decision/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/advisor_experiments_20261004/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/candidate_search_study/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/numerical_simulation/known_s_cluster/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/numerical_simulation/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/common/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/group/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/metrics/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/tensor/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/utils/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/vector/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/known_s_cluster/upload_verification/src/tensor_abess/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/version_validation/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `numerical_simulation/__pycache__` | 29648 | 3 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/baselines/__pycache__` | 15061 | 6 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/common/__pycache__` | 2003 | 2 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/group/__pycache__` | 17834 | 5 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/metrics/__pycache__` | 3394 | 2 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/tensor/__pycache__` | 45252 | 6 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/utils/__pycache__` | 43788 | 5 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/vector/__pycache__` | 15702 | 5 | 0 path/SHA256 matches; source exists or empty |
| `src/tensor_abess/__pycache__` | 959 | 1 | 0 path/SHA256 matches; source exists or empty |
| `src/tests/__pycache__` | 0 | 0 | 0 path/SHA256 matches; source exists or empty |
| `tests/__pycache__` | 80879 | 14 | 0 path/SHA256 matches; source exists or empty |

Cache candidate total: 864095 bytes in 35 directories.

## 其他临时候选

| Exact path | Bytes | Reason | Reference evidence |
|---|---:|---|---|
| `tmp/pdfs/` | 0 | Empty temporary render directory | 0 path matches; no contents |
| `push.log` | 706 | Captured PowerShell git-push console output; contains only up-to-date/native-command wrapper message, no results/data/config | 0 path/SHA256 matches |

## 明确保留，不列为清理候选

| Exact path or class | Observed size | Reason to retain |
|---|---:|---|
| `tmp/taskcard6_zip_validation/` | 243334 bytes / 82 files | Historical release-package extraction checks, including release manifests and source copies; could substantiate package validation. Temporary-looking name is insufficient. |
| `.tmp_pdf/simulation_experiment.txt` | 23143 bytes | Extracted research text; source-substitution and continuing use not established. |
| `.tmp_pdf/tensor_spec_0720.txt` | 90706 bytes | Extracted algorithm/specification text; preserve rather than infer redundancy. |
| `Presentation/recent_work_20261024/qa/近期工作汇报.aux` | 2885 bytes | 内容SHA256命中Presentation交付索引；保留。 |
| `Presentation/recent_work_20261024/qa/近期工作汇报.nav` | 1574 bytes | 内容SHA256命中Presentation交付索引；保留。 |
| `Presentation/recent_work_20261024/qa/近期工作汇报.out`、`近期工作汇报.snm`、`近期工作汇报.toc` | 各0 bytes | 空文件哈希命中多份交付索引；同一报告的辅助产物保守保留。空文件同哈希不单独证明路径身份。 |
| `tmp/`, `.tmp_pdf/` parent directories | mixed | Do not remove whole parents; retained evidence exists below them. |
| Compilation logs, PDFs, rendered page images, figure QA | not totaled | Validation/delivery evidence; no deletion proposed. |
| All models, source data, `semantic_conflict_archive/`, `pre_audit_fix/`, failed-study records and historical comparisons | not totaled | Research provenance; not disposable caches. |
| `.venv/`, runtime dependencies, excluded historical test-tools | not inspected | Explicitly protected; never infer eligibility from excluded scans. |

## 实际检查的70份封存／指纹文件

- `doc/beijing_air_quality/contract.json`
- `doc/beijing_air_quality/DELIVERY_INDEX.json`
- `doc/beijing_air_quality/FIGURE_CONTRACT.md`
- `doc/mushroom_empirical/contract.json`
- `doc/mushroom_empirical/DELIVERY_INDEX.json`
- `doc/mushroom_empirical/FIGURE_CONTRACT.md`
- `doc/mushroom_screen/contract.json`
- `doc/mushroom_screen/DELIVERY_INDEX.json`
- `doc/releases/v5.2-search-final/PRE_RELEASE_SOURCE_SHA256.json`
- `doc/releases/v5.2-search-final/RELEASE_MANIFEST.json`
- `doc/releases/v5.2-search-final/SOURCE_SHA256.txt`
- `doc/taskcard10/continuation_01/DELIVERY_INDEX.json`
- `doc/taskcard10/DELIVERY_INDEX.json`
- `doc/taskcard10/results/continuation_01/REUSE_MANIFEST.json`
- `doc/taskcard11/DELIVERY_INDEX.json`
- `doc/taskcard12/cluster_review/local_delivery_snapshot/DELIVERY_INDEX.json`
- `doc/taskcard12/CLUSTER_UPLOAD_MANIFEST.md`
- `doc/taskcard12/DELIVERY_INDEX.json`
- `doc/taskcard12/verification/package_extracted/taskcard12_bundle/doc/taskcard12/CLUSTER_UPLOAD_MANIFEST.md`
- `doc/taskcard12/verification/package_extracted/taskcard12_bundle/FILE_MANIFEST.json`
- `doc/taskcard12/verification/package_final_extracted/taskcard12_bundle/doc/taskcard12/CLUSTER_UPLOAD_MANIFEST.md`
- `doc/taskcard12/verification/package_final_extracted/taskcard12_bundle/FILE_MANIFEST.json`
- `doc/taskcard13/cluster_handoff/PACKAGE_MANIFEST.json`
- `doc/taskcard13/DELIVERY_INDEX.json`
- `doc/taskcard13/FIGURE_CONTRACT.md`
- `doc/taskcard13/history/engineering_pre_delivery/contract.json`
- `doc/taskcard6/DELIVERY_INDEX.json`
- `doc/taskcard6/FIGURE_CONTRACT.md`
- `doc/taskcard7/DELIVERY_INDEX.json`
- `doc/taskcard7/FIGURE_CONTRACT.md`
- `doc/taskcard7_completion/DELIVERY_INDEX.json`
- `doc/taskcard7_completion/FIGURE_CONTRACT.md`
- `doc/taskcard8/DELIVERY_INDEX.json`
- `doc/taskcard8/FIGURE_CONTRACT.md`
- `doc/taskcard9/DELIVERY_INDEX.json`
- `doc/team_sync/v5.1-bcd-fix/MANIFEST.json`
- `doc/team_sync/v5.2-candidate-search/MANIFEST.json`
- `doc/team_sync/v5.2-search-final/MANIFEST.json`
- `doc/team_sync/v6-msic-after-search-candidate/MANIFEST.json`
- `doc/uci270_concentration/contract.json`
- `doc/uci270_concentration/DELIVERY_INDEX.json`
- `doc/uci270_concentration/DELIVERY_INDEX.md`
- `doc/uci270_concentration/figures/FIGURE_CONTRACT.md`
- `doc/uci270_concentration/pre_audit_fix/contract.json`
- `doc/uci270_concentration/raw_manifest.json`
- `doc/uci270_concentration/semantic_conflict_archive/DELIVERY_INDEX_BEFORE_ABANDONMENT.json`
- `empirical_analysis/lemon/incremental_value/protected_fingerprints.json`
- `empirical_analysis/lemon/outputs_eeg_incremental/diagnostics/input_fingerprint.json`
- `empirical_analysis/lemon/outputs_eeg_incremental/diagnostics/reporting_fingerprint.json`
- `empirical_analysis/lemon/outputs_eeg_incremental_audit/diagnostics/input_fingerprint.json`
- `empirical_analysis/lemon/outputs_eeg_incremental_smoke/diagnostics/input_fingerprint.json`
- `empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final/diagnostics/input_fingerprint.json`
- `empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final/diagnostics/reporting_fingerprint.json`
- `empirical_analysis/lemon/outputs_fold80_diagnostic/protected_fingerprints.json`
- `empirical_analysis/lemon/outputs_lemon_adjusted/figure_manifest.json`
- `empirical_analysis/manifest.json`
- `empirical_analysis/spectrofood_apple/outputs/apple_v1/contract.json`
- `numerical_simulation/known_s_cluster/UPLOAD_MANIFEST.json`
- `numerical_simulation/known_s_cluster/upload_verification/UPLOAD_MANIFEST.json`
- `numerical_simulation/outputs_full_v2/checkpoint_manifest.json`
- `numerical_simulation/outputs_full_v3/checkpoint_manifest.json`
- `numerical_simulation/outputs_full_v4_adaptive/checkpoint_manifest.json`
- `numerical_simulation/outputs_full_v5_adaptive/checkpoint_manifest.json`
- `numerical_simulation/outputs_msic_pair_task12/FILE_MANIFEST.json`
- `Presentation/lemon_report_assets/qa/figure_contract.json`
- `Presentation/recent_work_20261024/DELIVERY_INDEX.json`
- `tmp/taskcard6_zip_validation/Tensor-ABESS-v5.2-HCP/Tensor-ABESS-v5.2-search-final/RELEASE_MANIFEST.json`
- `tmp/taskcard6_zip_validation/Tensor-ABESS-v5.2-HCP/Tensor-ABESS-v5.2-search-final/SHA256SUMS.txt`
- `tmp/taskcard6_zip_validation/Tensor-ABESS-v5.2-NKI/Tensor-ABESS-v5.2-search-final/RELEASE_MANIFEST.json`
- `tmp/taskcard6_zip_validation/Tensor-ABESS-v5.2-NKI/Tensor-ABESS-v5.2-search-final/SHA256SUMS.txt`

No automatic cleanup was performed. Before approved cleanup, recheck active processes and the current delivery index: caches may have regenerated and new manifests may have been added since this snapshot.
