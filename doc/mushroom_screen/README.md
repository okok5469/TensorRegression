# Mushroom CSV screening

Only the existing SpectroFood CSV is used. This isolated screen fits training mean, Ridge and PLS; no tensor algorithm, MAT file or Apple model is imported or run. Apple work is paused.

Results: [report](MUSHROOM_SCREEN_REPORT.md), [summary](summary_metrics.csv), [per-repeat metrics](repeat_metrics.csv), [data quality](data_quality.json), [OOF predictions](oof_predictions.csv), [validation](FINAL_VALIDATION.json). The frozen protocol specifies all grids, grouping, scaling and metrics before predictions were produced.

Actual command, from the repository root using the existing environment:

```powershell
.venv/Scripts/python.exe -B -m empirical_analysis.mushroom_screen --csv 'D:/数据集/SpectroFood_Apple/SpectroFood_dataset.csv' --output doc/mushroom_screen
```

This completed output refuses an unreviewed rerun. Runtime dependencies are reused read-only from the existing isolated dependency directory; no environment installation or upgrade is performed. All45 final model/scaler artifacts and their inner tuning records are in `models/`. Five focused tests verify section parsing, duplicate-spectrum grouping, training-only scaling/inversion, inner tuning and complete-OOF metric definitions.

The exact-spectrum groups prevent identical rows crossing outer or inner folds. They are not independently known physical mushrooms or batches. Conflicting responses are retained, disclosed and grouped; no labels are corrected or averaged. The results therefore screen the predictive information in these CSV records, and do not establish independent-fruit/batch generalization or tensor/spatial value.
