"""Build report figures from saved LEMON results without fitting any model.

Figure contract: describe the analyzed cohort, show the response transform,
compare complete held-out predictions including the nuisance-only control,
and expose variation in the selected support across the same five folds.
These are slide-sized quantitative figures; no statistical intervals or tests
are inferred from the folds. All figure drawing and exports use Matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def metric_values(observed, predicted):
    error = np.asarray(observed) - np.asarray(predicted)
    tss = np.sum((observed - np.mean(observed)) ** 2)
    assert tss > 0, "Response must vary for an R2 comparison"
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / tss),
    }


def read_report_data(source):
    with np.load(source / "qc/analysis_dataset.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    ids, observed, covariates = data["subject_ids"], data["y"], data["C"]
    assert data["X"].shape == (145, 61, 16, 5)
    assert observed.shape == (145,) and covariates.shape == (145, 8)
    assert len(set(ids.tolist())) == 145
    assert np.isfinite(observed).all() and np.isfinite(covariates).all()
    manifest = pd.read_csv(source / "qc/final_subject_manifest.csv")
    final = manifest.loc[manifest.include_primary].set_index("subject_id").loc[ids].reset_index()
    assert len(final) == 145 and np.array_equal(final.subject_id.to_numpy(), ids)
    assert (final.tmt5_raw > 0).all(), "Natural log requires positive time"
    assert np.allclose(observed, final.tmt5_log, rtol=0, atol=1e-12)
    assert np.allclose(observed, np.log(final.tmt5_raw), rtol=0, atol=1e-12)
    oof = pd.read_csv(source / "cv/oof_predictions.csv")
    stored_metrics = pd.read_csv(source / "cv/cv_summary.csv")
    assignment = pd.read_csv(source / "cv/fold_assignments.csv").set_index("subject_id").loc[ids]
    assert len(oof) == 435 and set(oof.response_scale) == {"log"}
    metrics = []
    for row in stored_metrics.to_dict("records"):
        predictions = oof.loc[oof.model == row["model"]].set_index("subject_id").loc[ids]
        assert len(predictions) == 145 and predictions.index.is_unique
        assert np.allclose(predictions.observed, observed, rtol=0, atol=1e-12)
        assert np.array_equal(predictions.fold.to_numpy(), assignment.fold.to_numpy())
        measured = metric_values(observed, predictions.predicted.to_numpy())
        assert np.allclose(list(measured.values()), [row["oof_rmse"], row["oof_mae"], row["oof_r2"]], rtol=0, atol=1e-12)
        metrics.append({"model": row["model"], "n": 145, "response_scale": "log", **measured, "provenance": "Stored model OOF predictions"})
    baseline = np.full(145, np.nan)
    fold_ids = np.zeros(145, dtype=int)
    seen = np.zeros(145, dtype=int)
    for fold_number in range(1, 6):
        folder = source / f"cv/folds/fold_{fold_number}"
        fold = json.loads((folder / "fold_info.json").read_text(encoding="utf-8"))
        test, train = np.asarray(fold["test_indices"]), np.asarray(fold["train_indices"])
        assert len(test) == 29 and len(train) == 116
        assert not np.intersect1d(train, test).size
        assert np.array_equal(ids[test], np.asarray(fold["test_subject_ids"]))
        assert np.array_equal(ids[train], np.asarray(fold["train_subject_ids"]))
        with np.load(folder / "preprocessing.npz", allow_pickle=False) as preprocessing:
            coefficients = preprocessing["y_nuisance_coef"]
            center = float(preprocessing["y_mean"])
            assert coefficients.shape == (8,)
            # Independently verify that these are the original training-only OLS
            # coefficients, then use the saved coefficients for predictions.
            expected = np.linalg.lstsq(covariates[train], observed[train], rcond=None)[0]
            assert np.allclose(coefficients, expected, rtol=0, atol=1e-11)
            residual_center = float(np.mean(observed[train] - covariates[train] @ expected))
            assert np.isclose(center, residual_center, rtol=0, atol=1e-11)
            baseline[test] = covariates[test] @ coefficients + center
        fold_ids[test] = fold_number
        seen[test] += 1
    assert np.all(seen == 1) and np.isfinite(baseline).all()
    assert np.array_equal(fold_ids, assignment.fold.to_numpy())
    baseline_metrics = metric_values(observed, baseline)
    assert np.allclose(list(baseline_metrics.values()), [0.3149351491095703, 0.25438514839158716, 0.3780217922442529], rtol=0, atol=1e-11)
    metrics.append({"model": "Age + sex only", "n": 145, "response_scale": "log", **baseline_metrics, "provenance": "New diagnostic reconstructed from saved outer-fold nuisance parameters; no model refit"})
    tensor_meta = json.loads((source / "tensors/tensor_meta.json").read_text(encoding="utf-8"))
    frequency_frames = {}
    for axis in ("channel", "time", "frequency"):
        frequency = pd.read_csv(source / f"cv/{axis}_selection_frequency.csv")
        frequency = frequency.loc[frequency.model == "Tensor-ABESS"].sort_values("index")
        assert len(frequency) == {"channel": 61, "time": 16, "frequency": 5}[axis]
        assert (frequency.n_folds == 5).all()
        assert np.allclose(frequency.selection_frequency, frequency.selection_count / 5)
        frequency_frames[axis] = frequency
    final_support = {axis: json.loads((source / f"final_model/S_{axis}.json").read_text(encoding="utf-8")) for axis in frequency_frames}
    return final, oof, pd.DataFrame(metrics), pd.DataFrame({"subject_id": ids, "fold": fold_ids, "observed": observed, "predicted": baseline, "response_scale": "log"}), tensor_meta, frequency_frames, final_support


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return {"blue": "#003E7E", "gray": "#808891", "light": "#D8DEE5", "accent": "#B86C38", "text": "#263340"}


def panel_label(ax, text):
    return ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-20, 11), textcoords="offset points", ha="left", va="bottom", weight="bold", fontsize=16, annotation_clip=False)


def save_figure(fig, name, assets, qa, require_matplotlib_panel_alignment):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=qa / f"{name}.alignment.json", overlay_svg=qa / f"{name}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(assets / f"{name}.pdf")
    fig.savefig(assets / f"{name}.svg")
    fig.savefig(assets / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_age(final, palette):
    categories = ["20-25", "25-30", "30-40", "55-65", "65-70", "70-75", "75-80"]
    counts = final.age_category.value_counts().reindex(categories)
    assert counts.sum() == 145 and counts.notna().all()
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    positions = np.arange(len(categories))
    ax.bar(positions, counts, color=palette["blue"], width=0.68, edgecolor="none")
    for position, count in zip(positions, counts):
        ax.text(position, count + 1.0, str(count), ha="center", va="bottom", fontsize=15)
    ax.set(xticks=positions, xticklabels=categories, ylabel="Participants", xlabel="Age category (years)", ylim=(0, counts.max() * 1.18))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.set_title("Final analysis sample: n = 145", pad=16)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.82)
    return fig


def plot_tmt(final, palette):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    columns = ["tmt5_raw", "tmt5_log"]
    labels = ["TMT-B completion time (s)", "Natural log of TMT-B time"]
    bounds = [(20, 170), (3.1, 5.2)]
    maximum = 0
    for index, (ax, column, label, limits) in enumerate(zip(axes, columns, labels, bounds)):
        values, _, _ = ax.hist(final[column], bins=12, range=limits, color=palette["blue"], edgecolor="white", linewidth=0.7)
        assert values.sum() == 145, "Histogram bounds must include every final participant"
        maximum = max(maximum, float(values.max()))
        ax.set(xlabel=label, xlim=limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(["Raw seconds", "Log-transformed response"][index], pad=16)
        panel_label(ax, ["a", "b"][index])
    axes[0].set(ylabel="Participants", ylim=(0, maximum * 1.12))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.8, wspace=0.21)
    return fig


def plot_metrics(metrics, palette):
    order = ["Age + sex only", "Tensor-ABESS", "Vector Lasso", "Tensor-GLasso"]
    values = metrics.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    positions = np.arange(4)
    colors = [palette["accent"], palette["blue"], palette["gray"], palette["gray"]]
    ax.barh(positions, values.rmse, height=0.48, color=colors, edgecolor="none")
    for position, value in zip(positions, values.rmse):
        ax.text(value + 0.008, position, f"{value:.4f}", va="center", ha="left", fontsize=16)
    ax.set(yticks=positions, yticklabels=order, xlim=(0, 0.50), xlabel="OOF RMSE in log(TMT-B) (lower is better)")
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title("Fixed 5-fold partition; n = 145", pad=15)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.24, top=0.8)
    return fig


def plot_prediction(oof, palette):
    predictions = oof.loc[oof.model == "Tensor-ABESS"]
    assert len(predictions) == 145
    fig, ax = plt.subplots(figsize=(5, 4.5))
    limits = (3.1, 5.2)
    assert predictions[["observed", "predicted"]].to_numpy().min() > limits[0]
    assert predictions[["observed", "predicted"]].to_numpy().max() < limits[1]
    ax.plot(limits, limits, color=palette["gray"], lw=1.2, linestyle="--", zorder=1)
    ax.scatter(predictions.observed, predictions.predicted, s=32, color=palette["blue"], alpha=0.65, edgecolors="none", zorder=2)
    ax.set(xlabel="Observed log(TMT-B)", ylabel="OOF predicted log(TMT-B)", xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([3.2, 3.8, 4.4, 5.0])
    ax.set_yticks([3.2, 3.8, 4.4, 5.0])
    ax.set_title("Tensor-ABESS: 145 held-out predictions", fontsize=15, pad=15)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.85)
    return fig


def plot_support(tensor_meta, frequencies, support, palette):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    channel = frequencies["channel"]
    nonzero = channel.loc[channel.selection_count > 0]
    labels = [tensor_meta["channels"][index] for index in nonzero["index"]]
    positions = np.arange(len(nonzero))
    channel_colors = [palette["blue"] if index in support["channel"] else palette["gray"] for index in nonzero["index"]]
    axes[0].barh(positions, nonzero.selection_count, color=channel_colors, edgecolor="none", height=0.65)
    axes[0].set(yticks=positions, yticklabels=labels, xlim=(0, 5), xlabel="Folds selecting a channel")
    axes[0].set_xticks(range(6))
    axes[0].invert_yaxis()
    axes[0].set_title("Channels", pad=16)
    time = frequencies["time"]
    colors = [palette["blue"] if index in support["time"] else palette["gray"] for index in time["index"]]
    axes[1].bar(time["index"] + 1, time.selection_count, color=colors, edgecolor="none", width=0.65)
    axes[1].set(xlabel="Closed-eye window (1-16)", ylabel="Folds selecting a window", ylim=(0, 5), xlim=(0.4, 16.6))
    axes[1].set_xticks([1, 4, 8, 12, 16]); axes[1].set_yticks(range(6))
    axes[1].set_title("Windows", pad=16)
    band = frequencies["frequency"]
    colors = [palette["blue"] if index in support["frequency"] else palette["gray"] for index in band["index"]]
    axes[2].bar(band["index"], band.selection_count, color=colors, edgecolor="none", width=0.65)
    axes[2].set(xticks=band["index"], xticklabels=["δ", "θ", "α", "β", "γ"], xlabel="Frequency band", ylabel="Folds selecting a band", ylim=(0, 5))
    axes[2].set_yticks(range(6))
    axes[2].set_title("Bands", pad=16)
    for ax, label in zip(axes, ["a", "b", "c"]):
        panel_label(ax, label)
    fig.text(0.50, 0.018, f"All 5 folds; {61 - len(nonzero)} other channels have count 0. Blue = final-model support.", ha="center", fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.26, top=0.8, wspace=0.48)
    return fig


def run_qa(script, names, assets, qa, skill_scripts):
    commands = [("source", [sys.executable, str(skill_scripts / "validate_figure.py"), str(script), "--backend", "python", "--json"])]
    for name in names:
        commands += [(f"{name}.text", [sys.executable, str(skill_scripts / "audit_pdf_text.py"), str(assets / f"{name}.pdf"), "--min-pt", "5", "--json"]),
                     (f"{name}.collision", [sys.executable, str(skill_scripts / "audit_figure_collisions.py"), str(assets / f"{name}.pdf"), "--json-out", str(qa / f"{name}.collision.json"), "--overlay-pdf", str(qa / f"{name}.collision-overlay.pdf")])]
    for label, command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        (qa / f"{label}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        assert completed.returncode == 0, f"Figure QA failed: {label}; inspect its QA log"
    (qa / "figure_contract.json").write_text(json.dumps({
        "backend": "python", "purpose": "10-minute report slides", "archetype": "quantitative grid",
        "cohort": "All 145 final analysis participants; excludes the 4 recorded failures from the 149-subject analysis selection",
        "plots": {
            "age_distribution": "Describe the seven age categories of the actual analyzed cohort",
            "tmt_distribution": "Show raw time skew and the response transformation with aligned count axes",
            "oof_metrics": "Age/sex-only control bounds any claim of added EEG prediction value",
            "oof_prediction": "Show all 145 subject-level held-out predictions on common observed/predicted axes",
            "support_stability": "Expose all selected channel, window and band counts across the same five folds",
        },
        "uncertainty": "No CI or significance test; OOF metrics use one fixed partition and support counts use five outer fits",
        "support_display": "All nonzero-selected channels; zero-count channels explicitly enumerated by count, all 16 windows and all 5 bands shown",
        "baseline_formula": "C_test @ saved y_nuisance_coef + saved y_mean; training-only OLS parameters verified independently",
        "export": "Editable PDF/SVG text; 300-dpi PNG; slide-size fonts 14-16 pt",
        "source_warning_reviews": {
            "EXPORT-RASTER": "PNG is the requested presentation preview; PDF/SVG are the authoritative vector exports, no TIFF required",
            "FINAL-WIDTH": "These are slide-sized exports, not journal production figures",
            "LOG-GUARD": "An explicit assertion of final.tmt5_raw > 0 precedes independent np.log validation",
            "UNCERTAINTY-ENCODING": "Pooled OOF metrics use one fixed partition; selected-support counts describe the five fits and are not interval summaries",
        },
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=project / "empirical_analysis/lemon/outputs_lemon_adjusted")
    parser.add_argument("--assets", type=Path, default=project / "Presentation/lemon_report_assets")
    parser.add_argument("--qa", type=Path, default=project / "tmp/pdfs/lemon-report-qa")
    parser.add_argument("--skill-scripts", type=Path, default=Path.home() / ".codex/skills/nature-figure/scripts")
    args = parser.parse_args()
    assert args.skill_scripts.is_dir(), "nature-figure QA scripts are required"
    args.assets.mkdir(parents=True, exist_ok=True); args.qa.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, NATURE_FIGURE_CONFIG=str(args.qa / "figure-backend.json"))
    subprocess.run([sys.executable, str(args.skill_scripts / "nature_figure_backend.py"), "set", "python"], env=environment, check=True, capture_output=True)
    sys.path.insert(0, str(args.skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    final, oof, metrics, baseline, meta, frequencies, support = read_report_data(args.source)
    metrics.to_csv(args.assets / "report_metrics.csv", index=False)
    baseline.to_csv(args.assets / "baseline_oof_predictions.csv", index=False)
    summary = {"n_subjects": 145, "tensor_shape": [145, 61, 16, 5], "n_features_per_subject": 4880,
               "sex_counts": {"Female": int((final.sex == 0).sum()), "Male": int((final.sex == 1).sum())},
               "age_counts": {key: int(value) for key, value in final.age_category.value_counts().sort_index().items()},
               "response_statistics": final[["tmt5_raw", "tmt5_log"]].agg(["count", "mean", "std", "median", "min", "max", "skew"]).to_dict(),
               "n_outer_folds": 5, "n_train_per_fold": 116, "n_test_per_fold": 29,
               "baseline_provenance": "New saved-parameter diagnostic, independently verified against each training-only nuisance OLS; no EEG model refit",
               "final_support": {"channels": [meta["channels"][i] for i in support["channel"]], "windows_1_based": [i + 1 for i in support["time"]], "bands": [meta["bands"][i] for i in support["frequency"]]},
               "excluded_source_rows": "Final manifest include_primary=False; no further exclusions in these figures"}
    (args.assets / "sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    palette = apply_style()
    figures = {"age_distribution": plot_age(final, palette), "tmt_distribution": plot_tmt(final, palette),
               "oof_metrics": plot_metrics(metrics, palette), "oof_prediction": plot_prediction(oof, palette),
               "support_stability": plot_support(meta, frequencies, support, palette)}
    for name, fig in figures.items():
        save_figure(fig, name, args.assets, args.qa, require_matplotlib_panel_alignment)
    run_qa(Path(__file__), list(figures), args.assets, args.qa, args.skill_scripts)
    print("Created 5 verified report figures and saved-data diagnostic tables.")


if __name__ == "__main__":
    main()
