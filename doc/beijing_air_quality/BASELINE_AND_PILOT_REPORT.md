# Baseline and full-fold pilot

Status: PASS. Only fixed developmentfold1 (2014-03..2014-08) evaluated; no final future-year scores inspected.

All7 methods share179valid dates and361training dates. Independent current-training medians/scalers, true previous-calendar-day Persistence, units, artifacts/predictions, chronology and originalABESS OLS/MSIC replayPASS.

|Method|Actual seconds incl full tuning|Training RMSE|Development RMSE|MAE|R2|Bias pred-y|
|---|---:|---:|---:|---:|---:|---:|
|M0|0.515|66.463922|53.729985|43.432680|-0.012800|6.040434|
|M1|0.475|65.699389|48.112408|35.590975|0.187909|-0.021031|
|M2|1.897|46.579326|40.167547|30.508638|0.433968|-0.387032|
|M3|3.549|44.155603|39.260940|30.032994|0.459231|2.059323|
|M4|33.268|45.431037|39.017725|29.573293|0.465910|1.156575|
|M5|19.361|45.910588|38.890257|28.954040|0.469394|0.524887|
|M6|18.916|43.321172|40.987803|30.484928|0.410614|-7.037847|

Full actual pilot wall=48.157s; ABESS candidates=253 including initial candidate, integer grid=252, cycles=6, caps=[7, 6, 6], q=8, OLSrank=8, MSIC=-268.77252.
Maximum observed actual interpreter OSpeak=170139648bytes; launcher and descendants correctly sampled, resource details in execution/PILOT_RESOURCES.json. Cache1,240,948bytes. No cost projection from a tiny training subset.

Native core uses criterion=bic for group-Lasso initialization, so passed cv_folds=5 is inactive; final support size uses originalMSIC. No randomized baseline validation. All32release modules imported from their exact src paths, see IMPORT_PATHS.json.

Proceed local: original7200s actual runtime budget,2workers1thread. No scientific parameter, representation, station or split changed after developmentpilot. Unknown full solver convergence for ABESS/GLasso remains unknown. No remote submission.
