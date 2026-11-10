# Frozen Apple empirical protocol

Design authority: taskcard13. Frozen before any formal OOF prediction.

- All QC-passing IDs; no outcome-based exclusions. Entity grouping is audited in DATA_AUDIT.md.
- Original rectangular ROI, area-weighted 8 x 8 spatial aggregation, 24 contiguous spectral means.
- y in dry-matter percentage points; train-fold X and y z-scores (ddof=0); inverse response transform.
- Constant coordinates: threshold 1e-12*max(1,max_train_std); zero under train mask.
- Seven models M0..M6; original released v5.2 adaptive MSIC, no CV selection for M1.
- Three repeats x five entity-disjoint folds, seeds13001/13002/13003; shared inner3 folds.
- Baselines refit X/y scaling in each inner training fold; select weighted percentage-point MSE.
- Exact grids, objectives, solver limits and tie rule: config.yaml and models.py frozen hashes.
- Primary: complete-repeat entity-equal OOF RMSE, MAE, R2; repeat mean and sample SD.
- Bootstrap:2000 independent entities, all3 predictions together, seed13999; descriptive conditional intervals.
- Replay tolerance: rtol1e-9, atol1e-10 for percentage-point predictions; identity requires exact hashes.
- No automatic scientific retry or altered seeds/parameters; only identical completed tasks resume.
- Smoke uses first64 outer-training records, no outer-test response; at most2 workers, BLAS1, formal7200s.
- Cost gate estimates15 outer training runs and all tuning; insufficient budget prepares cluster handoff.
- M6 retains native spectral resolution; selection frequencies are descriptive, no tissue ground truth.

Contract fingerprint: `ea170e553b57af9fa9fb737f816992b983c73dc4b902c5c2d8fd0717c271901c`
