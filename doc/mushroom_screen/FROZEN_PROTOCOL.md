# Mushroom mean-spectrum prediction-information screen

Frozen before inspecting any CV predictions. Authority: user's request for a low-cost CSV-only Mean/Ridge/PLS screen. Apple and all Tensor-ABESS work are paused; no tensor algorithm is imported or fitted by this screen.

- Input: only the existing SpectroFood_dataset.csv, verified against official MD5. Parse the actual `Mush No,Dry matter` section and its own204 wavelength columns; preserve all250 records. Empty padding columns are not features.
- Response: original mass fraction retained, percentage points =100*raw. The source paper's Mushroom range5--87%, mean29%, supports this encoding. No log transform, clipping or response-based exclusion.
- Quality: inspect finite values, unique IDs, wavelengths, constant coordinates, exact spectrum duplicates and duplicate-label differences. No deduplication, imputation, smoothing, derivatives, feature selection, segmentation or tensor reshaping.
- Leakage units: exact204-dimensional spectrum hashes; all identical spectra share an outer and inner split group. These243 groups are computational leakage units, not identified physical mushrooms. Retain conflicting responses as observed records, flag for provenance review. Record-equal scoring, never claim independent-fruit validation.
- Outer:3 repeats x5 group folds, seeds23001/23002/23003; deterministic response-independent group permutation followed by descending group-size balancing. Each record tested exactly once per repeat. Save actual IDs for every split.
- Inner:3 group folds on each outer training set, seed230000+100*repeat+fold; both tuned models share the same inner IDs.
- Train-only X coordinate mean/std(ddof0); std<=1e-12*max(1,max_train_std) is zeroed under the training mask. Train-only y mean/std(ddof0), invert predictions to percentage points before tuning/scoring. Reject degenerate training y. PLS uses scale=False, with its standard internal centering; no intercept duplication.
- Mean: current training-response mean.
- Ridge: sklearn SVD,fit_intercept=False; objective RSS+alpha*||beta||²; alpha=logspace(-4,4,17), fixed before predictions.
- PLS: components1/2/3/5/8/12/16, filtered only by the minimum feature/sample/rank bound across inner training folds; scale=False,max_iter1000,tol1e-7. No adaptive grid expansion.
- Select minimum pooled inner validation MSE in percentage-point units, weighted by validation record count; ties use first grid entry. Refit all preprocessing/model parameters on current outer training records only.
- Report complete-repeat pooled OOF RMSE and R², then3-repeat mean and sample SD. Preserve all2250 OOF predictions,45 outer model/scaler artifacts, hyperparameters and convergence/warnings. No fold-R² averaging or treating3N as independent records.
- No primary model claims on unseen fruit/batches/times. The broad response distribution and ID-ordered regimes can drive high overall R²; no causal attribution or fresh-mushroom-only extrapolation.
- Resource: single process, numerical threads1; existing isolated sklearn1.5.2 dependency reused read-only, no installation/upgrade. Save source/data/split/config fingerprints before fitting; independent artifact prediction replay rtol1e-9,atol1e-10.
