"""Experiment runner for tensor sparse-regression numerical simulations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from numerical_simulation.consistency_checks import ConsistencyReport, run_consistency_checks
from numerical_simulation.data_generation import (
    TensorSimulationConfig,
    TensorSimulationDataset,
    generate_simulation_dataset,
    iter_test_batches,
    total_dimension,
)
from numerical_simulation.metrics import (
    compute_coefficient_metrics,
    compute_prediction_metrics_from_batches,
    compute_support_metrics,
)
from numerical_simulation.methods import MethodEstimate, run_all_methods
from numerical_simulation.plots import generate_all_plots


@dataclass(frozen=True)
class ExperimentSpec:
    """One single-factor experiment."""

    name: str
    parameter: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    """Configuration of a whole simulation suite."""

    base_config: TensorSimulationConfig
    output_dir: str | Path
    n_reps: int
    experiments: tuple[ExperimentSpec, ...]
    resume: bool = False


@dataclass(frozen=True)
class ExperimentSuiteArtifacts:
    """Paths and in-memory rows produced by a completed suite."""

    raw_results_path: Path
    summary_results_path: Path
    consistency_report_path: Path
    plot_paths: tuple[Path, ...]
    raw_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ScenarioRun:
    experiment: str
    parameter: str
    scenario_value: str
    scenario_label: str
    scenario_order: int
    config: TensorSimulationConfig
    seed_offset: int


SUMMARY_METRICS = [
    "mode_precision_macro",
    "mode_tpr_macro",
    "mode_f1_macro",
    "exact_recovery",
    "element_precision",
    "element_recall",
    "element_f1",
    "element_fdr",
    "s1_hat",
    "s2_hat",
    "s3_hat",
    "q_hat",
    "ree",
    "pmse",
    "signal_mse",
    "r2_test",
    "runtime_sec",
    "n_cycles",
]

EXPECTED_METHODS = ("Tensor-ABESS", "Tensor-GLasso", "Vector Lasso", "Oracle OLS")
RAW_INT_FIELDS = {
    "replicate",
    "seed",
    "n_train",
    "n_test",
    "p1",
    "p2",
    "p3",
    "total_dimension",
    "s1_true",
    "s2_true",
    "s3_true",
    "scenario_order",
}
RAW_STR_FIELDS = {
    "experiment",
    "parameter",
    "scenario_value",
    "scenario_label",
    "method",
    "support_shape",
}
CHECKPOINT_FILENAME = "checkpoint_manifest.json"


def _override_config(config: TensorSimulationConfig, parameter: str, value: Any) -> TensorSimulationConfig:
    if parameter == "snr":
        return replace(config, snr=float(value))
    if parameter == "n_train":
        return replace(config, n_train=int(value))
    if parameter == "dims":
        dims = tuple(int(v) for v in value)
        return replace(config, dims=dims)
    if parameter == "rho":
        rho = (float(value), float(value), float(value))
        return replace(config, rho=rho)
    raise ValueError(f"unsupported experiment parameter: {parameter}")


def _scenario_label(parameter: str, value: Any) -> str:
    if parameter == "dims":
        p1, p2, p3 = value
        return f"{p1}x{p2}x{p3}"
    if parameter == "rho":
        return f"{float(value):.2f}"
    if parameter == "n_train":
        return str(int(value))
    if parameter == "snr":
        return f"{float(value):g}"
    return str(value)


def _config_metadata(config: TensorSimulationConfig) -> dict[str, Any]:
    return {
        "n_train": int(config.n_train),
        "n_test": int(config.n_test),
        "p1": int(config.dims[0]),
        "p2": int(config.dims[1]),
        "p3": int(config.dims[2]),
        "total_dimension": int(total_dimension(config.dims)),
        "s1_true": int(config.sparsity[0]),
        "s2_true": int(config.sparsity[1]),
        "s3_true": int(config.sparsity[2]),
        "rho_1": float(config.rho[0]),
        "rho_2": float(config.rho[1]),
        "rho_3": float(config.rho[2]),
        "snr": float(config.snr),
        "sigma": float(config.sigma),
        "support_shape": str(config.support_shape),
    }


def _suite_scenarios(suite: ExperimentSuiteConfig) -> list[_ScenarioRun]:
    scenarios = [
        _ScenarioRun(
            experiment="baseline",
            parameter="baseline",
            scenario_value="baseline",
            scenario_label="baseline",
            scenario_order=0,
            config=suite.base_config,
            seed_offset=0,
        )
    ]
    for exp_index, experiment in enumerate(suite.experiments, start=1):
        for value_index, value in enumerate(experiment.values):
            config = _override_config(suite.base_config, experiment.parameter, value)
            label = _scenario_label(experiment.parameter, value)
            scenarios.append(
                _ScenarioRun(
                    experiment=experiment.name,
                    parameter=experiment.parameter,
                    scenario_value=label,
                    scenario_label=label,
                    scenario_order=value_index,
                    config=config,
                    seed_offset=10_000 * exp_index,
                )
            )
    return scenarios


def _evaluate_estimate(
    dataset: TensorSimulationDataset,
    estimate: MethodEstimate,
    *,
    test_seed: int,
) -> dict[str, Any]:
    support_metrics = compute_support_metrics(
        estimate.S1,
        estimate.S2,
        estimate.S3,
        dataset.supports_true[0],
        dataset.supports_true[1],
        dataset.supports_true[2],
    )
    coefficient_metrics = compute_coefficient_metrics(estimate.B, dataset.B_true)
    prediction_metrics = compute_prediction_metrics_from_batches(
        iter_test_batches(
            dataset,
            n_samples=dataset.config.n_test,
            batch_size=dataset.config.test_batch_size,
            seed=test_seed,
        ),
        dataset.B_true,
        estimate.B,
    )

    return {
        "method": estimate.name,
        "loss_train": float(estimate.loss),
        "rss_train": float(estimate.rss),
        "runtime_sec": float(estimate.runtime_sec),
        "n_cycles": float(estimate.n_cycles),
        "sic": float("nan") if estimate.sic is None else float(estimate.sic),
        **support_metrics,
        **coefficient_metrics,
        **prediction_metrics,
    }


def run_single_repetition(
    config: TensorSimulationConfig,
    *,
    replicate_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one Monte-Carlo repetition under a fixed simulation setting."""
    dataset = generate_simulation_dataset(config, seed=seed)
    methods = run_all_methods(dataset, random_state=seed)
    test_seed = seed + 1_000_000

    rows = []
    for estimate in methods:
        row = {
            "replicate": int(replicate_index),
            "seed": int(seed),
            **_config_metadata(config),
            **_evaluate_estimate(dataset, estimate, test_seed=test_seed),
        }
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp_path.write_text("", encoding="utf-8")
        tmp_path.replace(path)
        return
    first_keys = list(rows[0].keys())
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(first_keys))
    fieldnames = first_keys + extra_keys
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            typed_row: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    typed_row[key] = value
                elif key in RAW_INT_FIELDS:
                    typed_row[key] = int(float(value))
                elif key in RAW_STR_FIELDS:
                    typed_row[key] = value
                else:
                    try:
                        typed_row[key] = float(value)
                    except ValueError:
                        typed_row[key] = value
            rows.append(typed_row)
    return rows


def _checkpoint_manifest(suite: ExperimentSuiteConfig) -> dict[str, Any]:
    return {
        "base_config": json.loads(json.dumps(asdict(suite.base_config), ensure_ascii=False)),
        "n_reps": int(suite.n_reps),
    }


def _load_checkpoint_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint_manifest(path: Path, suite: ExperimentSuiteConfig) -> None:
    path.write_text(
        json.dumps(_checkpoint_manifest(suite), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _deduplicate_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[tuple[Any, ...]] = set()
    deduplicated: list[dict[str, Any]] = []
    for row in raw_rows:
        if str(row.get("method")) not in EXPECTED_METHODS:
            continue
        key = (
            row.get("experiment"),
            row.get("parameter"),
            row.get("scenario_value"),
            int(row.get("scenario_order", -1)),
            int(row.get("replicate", -1)),
            row.get("method"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(row)
    return deduplicated


def _row_matches_scenario(
    row: dict[str, Any],
    scenario: _ScenarioRun,
    *,
    n_reps: int,
) -> bool:
    if str(row.get("experiment")) != scenario.experiment:
        return False
    if str(row.get("parameter")) != scenario.parameter:
        return False
    if str(row.get("scenario_value")) != scenario.scenario_value:
        return False
    if int(row.get("scenario_order", -1)) != scenario.scenario_order:
        return False
    replicate = int(row.get("replicate", -1))
    if not (0 <= replicate < n_reps):
        return False

    expected = _config_metadata(scenario.config)
    for key, expected_value in expected.items():
        actual = row.get(key)
        if key == "support_shape":
            if str(actual) != str(expected_value):
                return False
        elif isinstance(expected_value, int):
            if int(actual) != int(expected_value):
                return False
        else:
            if not np.isclose(float(actual), float(expected_value)):
                return False
    return str(row.get("method")) in EXPECTED_METHODS


def _load_resume_rows(
    raw_results_path: Path,
    suite: ExperimentSuiteConfig,
) -> list[dict[str, Any]]:
    del suite
    return _deduplicate_raw_rows(_read_csv_rows(raw_results_path))


def _completed_repetition_keys(raw_rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    grouped_methods: dict[tuple[Any, ...], set[str]] = {}
    for row in raw_rows:
        key = (
            row["experiment"],
            row["parameter"],
            row["scenario_value"],
            int(row["scenario_order"]),
            int(row["replicate"]),
        )
        grouped_methods.setdefault(key, set()).add(str(row["method"]))
    expected_methods = set(EXPECTED_METHODS)
    return {
        key for key, method_names in grouped_methods.items() if expected_methods.issubset(method_names)
    }


def _summarize_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    key_fields = ("experiment", "parameter", "scenario_value", "method")
    for row in raw_rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for rows in groups.values():
        exemplar = rows[0]
        summary = {
            "experiment": exemplar["experiment"],
            "parameter": exemplar["parameter"],
            "scenario_value": exemplar["scenario_value"],
            "scenario_label": exemplar["scenario_label"],
            "scenario_order": exemplar["scenario_order"],
            "method": exemplar["method"],
            "n_reps": len(rows),
            "n_train": exemplar["n_train"],
            "n_test": exemplar["n_test"],
            "p1": exemplar["p1"],
            "p2": exemplar["p2"],
            "p3": exemplar["p3"],
            "total_dimension": exemplar["total_dimension"],
            "snr": exemplar["snr"],
            "rho_1": exemplar["rho_1"],
            "rho_2": exemplar["rho_2"],
            "rho_3": exemplar["rho_3"],
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q3"] = float(np.quantile(values, 0.75))
        summary_rows.append(summary)

    return sorted(
        summary_rows,
        key=lambda row: (
            str(row["experiment"]),
            int(row["scenario_order"]),
            str(row["method"]),
        ),
    )


def _write_consistency_report(report: ConsistencyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment_suite(suite: ExperimentSuiteConfig) -> ExperimentSuiteArtifacts:
    """Run baseline + selected single-factor experiments and persist all artifacts."""
    output_dir = Path(suite.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = output_dir / "raw_results.csv"
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if suite.resume:
        actual_manifest = _load_checkpoint_manifest(checkpoint_path)
        expected_manifest = _checkpoint_manifest(suite)
        if actual_manifest is None:
            if raw_results_path.exists() and raw_results_path.stat().st_size > 0:
                raise ValueError(
                    "resume requested but checkpoint manifest is missing; use a fresh output directory"
                )
            _write_checkpoint_manifest(checkpoint_path, suite)
        elif actual_manifest != expected_manifest:
            raise ValueError("resume requested with incompatible base configuration or n_reps")
        raw_rows = _load_resume_rows(raw_results_path, suite)
    else:
        _write_checkpoint_manifest(checkpoint_path, suite)
        raw_rows = []
    _write_csv(raw_rows, raw_results_path)

    consistency = run_consistency_checks(seed=suite.base_config.random_state)
    consistency_path = output_dir / "consistency_checks.json"
    _write_consistency_report(consistency, consistency_path)
    scenarios = _suite_scenarios(suite)
    completed_keys = _completed_repetition_keys(raw_rows)
    current_batch_keys = {
        (
            scenario.experiment,
            scenario.parameter,
            scenario.scenario_value,
            scenario.scenario_order,
            rep,
        )
        for scenario in scenarios
        for rep in range(suite.n_reps)
    }
    total_repetitions = len(scenarios) * suite.n_reps
    completed_count = len(completed_keys & current_batch_keys)

    for scenario in scenarios:
        for rep in range(suite.n_reps):
            repetition_key = (
                scenario.experiment,
                scenario.parameter,
                scenario.scenario_value,
                scenario.scenario_order,
                rep,
            )
            if repetition_key in completed_keys:
                continue

            seed = int(
                suite.base_config.random_state
                + scenario.seed_offset
                + 100 * scenario.scenario_order
                + rep
            )
            rows = run_single_repetition(scenario.config, replicate_index=rep, seed=seed)
            for row in rows:
                row.update(
                    {
                        "experiment": scenario.experiment,
                        "parameter": scenario.parameter,
                        "scenario_value": scenario.scenario_value,
                        "scenario_label": scenario.scenario_label,
                        "scenario_order": int(scenario.scenario_order),
                    }
                )
                raw_rows.append(row)

            completed_keys.add(repetition_key)
            completed_count += 1
            _write_csv(raw_rows, raw_results_path)
            print(
                "[progress] "
                f"{completed_count}/{total_repetitions} repetitions complete | "
                f"experiment={scenario.experiment} scenario={scenario.scenario_label} "
                f"rep={rep + 1}/{suite.n_reps}"
            )

    summary_rows = _summarize_rows(raw_rows)
    summary_results_path = output_dir / "summary_results.csv"
    _write_csv(summary_rows, summary_results_path)

    plot_paths = tuple(
        generate_all_plots(
            raw_rows,
            summary_rows,
            base_config=suite.base_config,
            output_dir=output_dir,
        )
    )

    return ExperimentSuiteArtifacts(
        raw_results_path=raw_results_path,
        summary_results_path=summary_results_path,
        consistency_report_path=consistency_path,
        plot_paths=plot_paths,
        raw_rows=tuple(raw_rows),
        summary_rows=tuple(summary_rows),
    )
