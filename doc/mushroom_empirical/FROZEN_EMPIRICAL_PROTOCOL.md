# Frozen Mushroom empirical protocol

Authority: task card15; fixed before real pilot or formal prediction. No representation, grid, solver or split is changed after observing pilot test performance.

- Data: all250 explicitly numbered MAT cubes joined one-to-one to official CSV; y_pct=100*mass fraction. All finite; no cube duplicates; six known duplicate-spectrum groups retained with conflicting labels.
- Representation: original rectangular ROI, exact area-average8x8,24 contiguous equal-count wavelength means; no learned cross-record preprocessing. M6/M7 use original CSV204, not MAT mean. Mapping and limitations in tensor_metadata.json.
- Main: original screen3x5 outer and inner3 partitions recovered by actual IDs, seeds23001/23002/23003; inner230000+100*repeat+fold. Exact-spectrum groups remain intact. No new random split.
- Stress: actual source-file LOBO5. Each four-file training set has shared inner3 folds of intact files (2/1/1); seed250000+heldoutfile. File blocks are storage partitions, not documented biological batches. All five low-variance R² values are retained.
- M0 train mean; M1 originalv5.2/MSIC Tensor-ABESS; M2 true Tensor-GLasso original LEMON adapter; M3 Vector Lasso; M4 Ridge on C-order tensor flatten; M5 PLS on spatial mean of same binned tensor; M6 CSV204 Ridge; M7 CSV204 PLS.
- All scalers fit current training fold only: X/y zscore ddof0; constant threshold1e-12*max(1,maxtrainstd), masked coordinates0; inverse y transform restores intercept/percentage points. Inner scales refit independently.
- M1 uses unchanged Apple._abess_options: target_sparsityNone,c3,criterionbic,kmax1,mmax15,eps1e-8,adaptive_criterionmsic,alpha1.5,gamma1,one_se1,n_lambdas20,cv_folds5,max_iter500,tol1e-6,max_cycles6. MSIC selects support size; no external-CV replacement.
- M2 lambda ratios geomspace1..0.01 (12); M3 training-alpha_max ratios logspace1..0.001 (13), cyclic100000iterations/tol1e-6; M4/M6 sklearnSVD Ridge alpha logspace1e-4..1e4 (17). Exact numeric values in config.json.
- PLS components1,2,3,5,8,12,16 filtered solely by minimum inner-training rank. M5 unchanged Apple scaleFalse/maxiter500/tol1e-6; M7 unchanged screen scaleFalse/maxiter1000/tol1e-7. Ties first in frozen grid. Tune record-weighted pooled inner validation MSE in percentage points. No y rounding/clipping.
- Main metrics: first collect250 OOF per model/repeat, calculate record-weighted RMSE/MAE/R², then mean±sampleSD over3 repeats. Main6000 rows; block2000 rows; total8000 predictions/160finalmodels. Three repeats are not750 independent mushrooms.
- Paired M1-minus-comparator uncertainty:2000 cluster bootstrap draws seed25999; known243 exact-spectrum groups sampled with replacement, preserve unequal group sizes, all labels and all3repeat predictions. Conditional OOF intervals do not include refitting uncertainty; block5 outcomes descriptive only.
- Pilot is actual first outer fold200training/50test with all8 models and full inner tuning; checks IDs,scalers,inverse,OLS/MSIC/rank/warnings and runtime/memory. Proceed locally after correctness PASS; no extrapolated tiny-sample cubic cost gate.
- Two local workers, one BLAS thread each. Completion resume only identical data/config/source/split/native environment with hash and saved-model replay. On failure stop, expose reason; no silent omitted folds/models or altered solver.
- Descriptive supports: q=s1*s2*s3,caps,Q,full trainingRSS/MSIC,capacity proximity,mode frequencies/Jaccard mapped to physical wavelengths and relative ROI positions. No true-support or causal-tissue recovery claim.

Reproduce: run local_execute.py --phase pilot; after independentPILOT_VALIDATION.json PASS run --phase full. Analysis and figure scripts read saved artifacts only.
