# Mushroom empirical figure contract

Backend: saved Python/matplotlib preference, exclusive. Purpose: compare conditional internal prediction with file-block distribution shift and describe selected support; conclusions are filled from saved results, never selected to favor a model.

All figures are single plot areas (quantitative comparisons). Export editable PDF/SVG and 300-dpi PNG previews, width at most180mm, text floor8pt. Show all eight methods, all five file blocks, all selected supports. Labels use consistent model IDs with a report mapping. PDF glyph/collision audits and individual visual review are required after final rendering; alignment is not applicable to single plot areas.

1. Main RMSE and R²: three complete-repeat points per method and mean±sample SD;250 records per repeat, known exact-spectrum groups held intact. Three repeats do not create750 independent mushrooms.
2. File-block RMSE: all five blocks for all eight methods, descriptive only, no IID uncertainty band. Response ranges appear separately, with all250 records represented.
3. Predicted versus observed: fixed first main repeat and all block OOF, for all methods; no selected best repeat. No uncertainty intervals for individual point predictions.
4. ABESS q: all15 main and5 block fits, descriptive outer-model capacities; no true coefficient recovery claim.
5. Wave/row/column support frequencies: main15 and block5 shown separately, mapped to24 physical wave-group centers and normalized ROI coordinates. Frequencies/Jaccard describe conditional algorithm stability, not causal tissue regions.
6. Paired RMSE differences: ABESS minus frozen comparators with conditional95% known-spectrum-group bootstrap intervals preserving all record labels and three repeat predictions together; negative favors ABESS. Block differences remain descriptive per file.

Statistics: percentage-point RMSE/MAE; pooled complete-OOF R²; no averaging fold R². Record-weighted metrics;243 leakage groups are resampled as clusters, retaining their1–3 records and unequal group sizes. No p-value or universal superiority claim. Source data are saved OOF, fold/repeat summaries, paired comparison table, support and wavelength mappings. No data exclusions, smoothing, selective clipping or arbitrary y limits.
