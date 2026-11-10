# SpectroFood Apple empirical experiment

This task-specific adapter preserves the released v5.2 optimizer and original MSIC. The scientific design is task card 13: 240 audited IDs, an 8 x 8 x 24 tensor, M0--M6, three repeated five-fold outer splits, and shared three-fold inner tuning. M1 selects order with original MSIC, never CV. Original data and existing experiments are read-only.

## Current execution

Data preparation, correctness tests, fixed-ID QC and seven-model smoke are complete. The conservative frozen cost projection exceeds the local 7200-second budget. Formal status is **PENDING_CLUSTER_RESULTS**. No formal Apple OOF rows or comparative scores have been produced. Run all 105 outer tasks on the cluster using [the cluster guide](cluster/README.md); no method or fold is dropped.

## Local entry points

Run from the repository root with its existing `.venv`. Task-only dependencies live in `doc/taskcard13/runtime`; the entry point explicitly adds that directory and `src`, limits numerical threads to one, and does not install anything. `config.yaml` uses JSON syntax, which is valid YAML. Its raw-data paths are the actual local paths.

```powershell
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage audit
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage prepare
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage freeze
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage smoke
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage full
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage collect
.venv/Scripts/python.exe -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage report
```

`audit` and `prepare` both perform the full deterministic data audit/build. Do not rerun them against a frozen run: a rebuilt archive has a new byte identity, even if its arrays are mathematically identical. `freeze` refuses a changed existing contract. `full` refuses a smoke cost gate above the local budget. `one` is the cluster worker. `collect` requires every completed task, verifies file identities, independently reloads models/scalers, reconstructs all saved test predictions and computes complete-repeat OOF metrics. `report` then makes figures and a six-question report, initially marking rendered figure QA as pending.

For the current pending run, return the complete cluster `outputs/apple_v1/` directory to the same local relative path, preserving all models and inner tuning logs. Then execute `collect` and `report`; inspect/audit the final figures before declaring the empirical study complete. Do not replace the frozen contract, data, source or seeds. Never copy Windows runtime binaries to Linux.

## Model definitions and tuning

- M0: current training response mean, in percentage points.
- M1: released adaptive Tensor-ABESS, original MSIC, alpha1.5/gamma1/one-SE1; c3, kmax1, mmax15, eps1e-8 and original initialization/solver settings. The final estimator is the unchanged rectangular OLS refit.
- M2: existing independent Tensor-GLasso through the verified external-tuning LEMON adapter; original local group-design normalization, grouped ISTA, screening, empty-support fallback and rectangular OLS. Twelve fixed lambda ratios from 1 to .01 are applied to training-local grouped lambda maxima. No internal CV or ABESS initializer substitutes for the baseline.
- M3: sklearn cyclic Lasso with `fit_intercept=False`, training-local `alpha_max=max(abs(Z.T @ y_std))/n` times 13 ratios from 1 to .001; objective RSS/(2n)+alpha*L1. Solver budget and convergence diagnostics are frozen in models.py. All warnings and dual gaps are retained.
- M4: sklearn Ridge with SVD, no intercept; objective RSS+alpha*L2-squared, 17 alphas from 1e-4 to 1e4.
- M5/M6: PLSRegression(scale=False), components1,2,3,5,8,12,16 restricted by the minimum available inner-training rank; max_iter500, tol1e-6. M5 uses the 24-bin mean spectrum, M6 the 141-band native mean spectrum.

Every inner fit independently estimates X/y standardization (ddof0), including constant-coordinate masks. Predictions are restored to percentage points before the size-weighted validation MSE is calculated. Stable first-in-grid tie breaking is deterministic. The audited dataset has one record per entity; groups never cross outer or inner folds. Approximate-constant y blocks fitting. No imputation is added because the audited inputs are finite.

`config.yaml` stores every numeric grid and resource limit. `contract.json` binds data, all real splits, model/preprocessing settings, release identity, baseline dependencies and task code. Completion keys also bind the native numerical environment. Failed tasks retain diagnostics; they are not automatically retried or silently replaced. Completed artifacts are accepted only with exact matching hashes.

## Scientific boundaries

The 240 sample IDs are the best available entity identifiers; physical fruit identity and batches are not independently recorded in the MAT. The absence of exact duplicate cubes is not proof of independence. Original rectangular ROIs, including zeros with undocumented background semantics, are retained. Relative coordinates do not imply anatomical correspondence.

Dry-matter CSV values are mass fractions, jointly supported by the native data and the source paper's percent ranges; `y_pct=100*y_raw`. Units and the full ID/axis audit are in `doc/taskcard13/DATA_AUDIT.md`. No Exact/REE/true-support metric applies to these real data. Internal repeated CV is not external validation, and whole-fruit response does not give pixel-level dry-matter ground truth.

Original solvers do not expose all convergence states; null is recorded for M1/M2 instead of claiming convergence. Numerical rank and condition are descriptive, with no coefficient repair. Full OOF comparisons and 2000 entity-block bootstrap intervals will only be generated after complete saved predictions exist; the intervals do not include model-retraining uncertainty.
