"""Plotting utilities for numerical simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numerical_simulation.data_generation import TensorSimulationConfig

PLOT_METHODS = ["Tensor-ABESS", "Tensor-GLasso", "Vector Lasso", "Oracle OLS"]
SELECTION_METHODS = ["Tensor-ABESS", "Tensor-GLasso", "Vector Lasso"]


def _group_summary(summary_rows: list[dict], experiment: str) -> list[dict]:
    rows = [row for row in summary_rows if row.get("experiment") == experiment]
    return sorted(rows, key=lambda row: (row.get("method", ""), row.get("scenario_order", 0)))


def _line_series(summary_rows: list[dict], experiment: str, method: str) -> list[dict]:
    rows = [row for row in summary_rows if row.get("experiment") == experiment and row.get("method") == method]
    return sorted(rows, key=lambda row: row.get("scenario_order", 0))


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_snr_selection_accuracy(summary_rows: list[dict], output_dir: Path) -> Path | None:
    rows = _group_summary(summary_rows, "snr")
    if not rows:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [
        ("mode_f1_macro_mean", "Mode-F1"),
        ("element_f1_mean", "Element-F1"),
        ("exact_recovery_mean", "Exact Recovery"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for method in SELECTION_METHODS:
            series = _line_series(summary_rows, "snr", method)
            x = [float(row["snr"]) for row in series]
            y = [float(row[metric]) for row in series]
            ax.plot(x, y, marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("SNR")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average metric")
    axes[-1].legend(loc="lower right", fontsize=8)
    return _save_figure(fig, output_dir / "plot_snr_selection_accuracy.png")


def plot_sample_size_prediction_error(summary_rows: list[dict], output_dir: Path) -> Path | None:
    rows = _group_summary(summary_rows, "sample_size")
    if not rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    metrics = [("pmse_mean", "Test PMSE"), ("ree_mean", "REE")]

    for ax, (metric, title) in zip(axes, metrics):
        for method in PLOT_METHODS:
            series = _line_series(summary_rows, "sample_size", method)
            x = [float(row["n_train"]) for row in series]
            y = [float(row[metric]) for row in series]
            ax.plot(x, y, marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("Training sample size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average metric")
    axes[-1].legend(loc="best", fontsize=8)
    return _save_figure(fig, output_dir / "plot_sample_size_prediction_error.png")


def plot_dimension_runtime(summary_rows: list[dict], output_dir: Path) -> Path | None:
    rows = _group_summary(summary_rows, "dimension")
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for method in PLOT_METHODS:
        series = _line_series(summary_rows, "dimension", method)
        x = [float(row["total_dimension"]) for row in series]
        y = [float(row["runtime_sec_median"]) for row in series]
        ax.plot(x, y, marker="o", label=method)
    ax.set_xlabel("Total dimension p1*p2*p3")
    ax.set_ylabel("Median runtime (sec)")
    ax.set_yscale("log")
    ax.set_title("Runtime vs tensor dimension")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return _save_figure(fig, output_dir / "plot_dimension_runtime.png")


def plot_correlation_performance(summary_rows: list[dict], output_dir: Path) -> Path | None:
    rows = _group_summary(summary_rows, "correlation")
    if not rows:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [
        ("mode_tpr_macro_mean", "Mode-TPR"),
        ("element_fdr_mean", "Element-FDR"),
        ("pmse_mean", "PMSE"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for method in PLOT_METHODS:
            series = _line_series(summary_rows, "correlation", method)
            x = [float(row["rho_1"]) for row in series]
            y = [float(row[metric]) for row in series]
            ax.plot(x, y, marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel(r"$\rho$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average metric")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[-1].legend(loc="best", fontsize=8)
    return _save_figure(fig, output_dir / "plot_correlation_performance.png")


def plot_estimated_sparsity_boxplots(
    raw_rows: list[dict],
    base_config: TensorSimulationConfig,
    output_dir: Path,
) -> Path | None:
    rows = [
        row
        for row in raw_rows
        if row.get("experiment") == "baseline" and row.get("method") in SELECTION_METHODS
    ]
    if not rows:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    methods = SELECTION_METHODS
    metrics = [("s1_hat", base_config.sparsity[0]), ("s2_hat", base_config.sparsity[1]), ("s3_hat", base_config.sparsity[2])]

    for ax, (metric, truth) in zip(axes, metrics):
        data = [
            [float(row[metric]) for row in rows if row["method"] == method]
            for method in methods
        ]
        ax.boxplot(data, tick_labels=methods, showmeans=True)
        ax.axhline(truth, color="red", linestyle="--", linewidth=1.1)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated sparsity")
    return _save_figure(fig, output_dir / "plot_estimated_sparsity_boxplots.png")


def generate_all_plots(
    raw_rows: list[dict],
    summary_rows: list[dict],
    *,
    base_config: TensorSimulationConfig,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all available summary plots and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_snr_selection_accuracy(summary_rows, output_dir),
        plot_sample_size_prediction_error(summary_rows, output_dir),
        plot_dimension_runtime(summary_rows, output_dir),
        plot_correlation_performance(summary_rows, output_dir),
        plot_estimated_sparsity_boxplots(raw_rows, base_config, output_dir),
    ]
    return [path for path in paths if path is not None]
