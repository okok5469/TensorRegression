"""Frozen simulation tasks with independent checkpoints for CPU clusters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import scipy

from numerical_simulation.data_generation import (
    TensorSimulationConfig, generate_simulation_dataset, iter_test_batches,
)
from numerical_simulation.methods import (
    MethodEstimate, run_oracle_ols_method, run_vector_lasso_method, tensor_glasso,
)
from numerical_simulation.metrics import (
    compute_coefficient_metrics, compute_prediction_metrics_from_batches, compute_support_metrics,
)
from tensor_abess.tensor.abess import tensor_abess
from tensor_abess.tensor.ops import support_from_tensor, tensor_inner


METHOD_NAMES = {
    "fixed": "Tensor-ABESS (known s)", "adaptive": "Tensor-ABESS (adaptive)",
    "oracle": "Oracle OLS", "glasso": "Tensor-GLasso", "lasso": "Vector Lasso",
}
SUMMARY_FILES = (
    "raw_results.csv", "summary_results.csv", "paired_fixed_vs_adaptive.csv",
    "search_gap.csv", "integrity_report.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def source_fingerprints() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    paths = list((repository / "src/tensor_abess").rglob("*.py"))
    paths += [repository / f"numerical_simulation/{name}.py" for name in ("__init__", "data_generation", "methods", "metrics")]
    paths += [Path(__file__), Path(__file__).with_name("profiles.json")]
    return {path.relative_to(repository).as_posix(): file_digest(path) for path in sorted(paths)}


def environment() -> dict:
    threads = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    require(set(threads.values()) == {"1"}, "Set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS and NUMEXPR_NUM_THREADS to 1 before planning/running.")
    numpy_config = np.show_config(mode="dicts")
    return {
        "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
        "platform": platform.system(), "machine": platform.machine(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "blas_build": numpy_config.get("Build Dependencies", {}), "thread_environment": threads,
    }


def config_from_dict(values: dict) -> TensorSimulationConfig:
    values = dict(values)
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    return TensorSimulationConfig(**values)


def profile_tasks(profile: str, n_reps: int) -> list[dict]:
    profiles = json.loads(Path(__file__).with_name("profiles.json").read_text(encoding="utf-8"))["profiles"]
    require(profile in profiles and n_reps > 0, "Unknown profile or non-positive repetition count.")
    specification = profiles[profile]
    base = config_from_dict(specification["base_config"])
    scenarios = [{"experiment": "baseline", "parameter": "baseline", "scenario_value": "baseline", "scenario_order": 0, "seed_offset": 0, "config": base}]
    for experiment in specification["experiments"]:
        parameter = experiment["parameter"]
        for order, value in enumerate(experiment["values"]):
            overrides = {parameter: value}
            if parameter == "rho":
                overrides[parameter] = tuple([float(value)] * 3)
            if parameter == "dims":
                overrides[parameter] = tuple(value)
            label = {
                "rho": lambda: f"{float(value):.2f}", "dims": lambda: "x".join(str(v) for v in value),
                "n_train": lambda: str(int(value)), "snr": lambda: f"{float(value):g}",
            }[parameter]()
            scenarios.append({
                "experiment": experiment["name"], "parameter": parameter, "scenario_value": label,
                "scenario_order": order, "seed_offset": experiment["seed_offset"],
                "config": replace(base, **overrides),
            })
    tasks = []
    for scenario_id, scenario in enumerate(scenarios):
        for replicate in range(n_reps):
            seed = base.random_state + scenario["seed_offset"] + 100 * scenario["scenario_order"] + replicate
            tasks.append({
                "task_id": len(tasks), "scenario_id": scenario_id, "replicate": replicate,
                "experiment": scenario["experiment"], "parameter": scenario["parameter"],
                "scenario_value": scenario["scenario_value"], "scenario_order": scenario["scenario_order"],
                "seed": seed, "test_seed": seed + 1_000_000,
                "config": json.loads(json.dumps(asdict(scenario["config"]))),
            })
    return tasks


def build_plan(profile: str, n_reps: int, methods: list[str]) -> dict:
    require(bool(methods) and len(set(methods)) == len(methods) and set(methods) <= set(METHOD_NAMES), "Methods must be unique supported method codes.")
    body = {
        "schema_version": 1, "profile": profile, "n_reps": n_reps, "methods": methods,
        "method_names": {method: METHOD_NAMES[method] for method in methods},
        "tasks": profile_tasks(profile, n_reps), "source_fingerprints": source_fingerprints(),
        "environment": environment(), "cpu_threads_per_task": 1,
        "interpretation": "Existing fixed and adaptive complete solver paths have different initialization and model-search stages.",
    }
    return {**body, "plan_id": digest(body)}


def commit_exclusive(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-link creation is atomic and fails on an existing target on Linux and Windows.
        os.link(temporary, path)
    finally:
        temporary.unlink()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def create_plan(output: Path, profile: str, n_reps: int, methods: list[str]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Plan requires an empty new output directory: {output}")
    plan = build_plan(profile, n_reps, methods)
    commit_exclusive(output / "plan.json", json_bytes(plan))
    (output / "tasks").mkdir()
    return plan


def load_plan(output: Path, *, check_environment: bool = False) -> dict:
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    require(plan["plan_id"] == digest(body), "Plan checksum does not match its immutable contents.")
    require(plan["source_fingerprints"] == source_fingerprints(), "Code/configuration differs from the immutable plan; use a fresh run directory.")
    require(plan["tasks"] == profile_tasks(plan["profile"], plan["n_reps"]), "Task grid differs from the frozen profile.")
    require(bool(plan["methods"]) and len(set(plan["methods"])) == len(plan["methods"]) and set(plan["methods"]) <= set(METHOD_NAMES), "Invalid method selection in plan.")
    require(plan["method_names"] == {method: METHOD_NAMES[method] for method in plan["methods"]}, "Method labels differ from the plan.")
    require((output / "tasks").is_dir(), "Plan is incomplete: missing tasks directory.")
    if check_environment:
        require(plan["environment"] == environment(), "Worker Python/library/BLAS/thread environment differs from plan.")
    return plan


def fit_tensor(X: np.ndarray, y: np.ndarray, config: TensorSimulationConfig, *, mode: str, seed: int) -> tuple[MethodEstimate, dict]:
    require(mode in {"fixed", "adaptive"}, "Unsupported Tensor-ABESS mode.")
    started = perf_counter()
    result = tensor_abess(
        X, y, target_sparsity=config.sparsity if mode == "fixed" else None,
        k_max=config.k_max, c=config.abess_init_c, criterion=config.abess_init_criterion,
        n_lambdas=config.n_lambdas, cv_folds=config.cv_folds, max_iter=config.max_iter,
        tol=config.tol, max_cycles=config.max_cycles, eps=1e-8, adaptive_criterion="msic",
        msic_refine_alpha=config.msic_refine_alpha, msic_refine_gamma=config.msic_refine_gamma,
        msic_refine_one_se_scale=config.msic_refine_one_se_scale,
    )
    estimate = MethodEstimate(
        name=METHOD_NAMES[mode], B=result.B, S1=np.sort(result.S1), S2=np.sort(result.S2), S3=np.sort(result.S3),
        s1=int(result.s1), s2=int(result.s2), s3=int(result.s3), loss=float(result.loss), rss=float(result.rss),
        runtime_sec=perf_counter() - started, n_cycles=1 if result.bcd_result is None else int(result.bcd_result.n_cycles),
        sic=float(result.sic), details={"random_state": seed, "tensor_abess_mode": mode},
    )
    trace = {
        "pipeline_init_supports": tuple(result.init_supports),
        "pipeline_glasso_init_supports": (result.init_result.S1_init, result.init_result.S2_init, result.init_result.S3_init),
        "s_max": list(result.s_max), "loss_history": list(result.loss_history), "sic_history": list(result.sic_history),
        "initial_supports_scope": "Entrypoint warm starts; adaptive global refinement may select a separately initialized candidate whose starting support is not exposed.",
        "loss_history_scope": "Selected public solver result; adaptive refinement may replace the original pipeline loss history.",
    }
    return estimate, trace


def fit_method(dataset, method: str, seed: int) -> tuple[MethodEstimate, dict]:
    if method in {"fixed", "adaptive"}:
        return fit_tensor(dataset.X_train, dataset.y_train, dataset.config, mode=method, seed=seed)
    config = dataset.config
    calls = {
        "oracle": lambda: run_oracle_ols_method(dataset.X_train, dataset.y_train, dataset.supports_true),
        "lasso": lambda: run_vector_lasso_method(dataset.X_train, dataset.y_train, config),
        "glasso": lambda: tensor_glasso(
            dataset.X_train, dataset.y_train, random_state=seed, criterion="cv",
            n_lambdas=config.n_lambdas, cv_folds=config.cv_folds, max_iter=config.max_iter,
            tol=config.tol, max_cycles=config.max_cycles,
        ),
    }
    return calls[method](), {"diagnostics_available": False, "note": "This existing public method wrapper does not expose its initialization or iteration history."}


def evaluate(dataset, estimate: MethodEstimate, test_seed: int) -> dict:
    metrics = {
        **compute_support_metrics(estimate.S1, estimate.S2, estimate.S3, *dataset.supports_true),
        **compute_coefficient_metrics(estimate.B, dataset.B_true),
        **compute_prediction_metrics_from_batches(iter_test_batches(dataset, seed=test_seed), dataset.B_true, estimate.B),
        "loss_train": float(estimate.loss), "rss_train": float(estimate.rss),
        "runtime_sec": float(estimate.runtime_sec), "n_cycles": int(estimate.n_cycles),
    }
    require(bool(np.isfinite(list(metrics.values())).all() and np.isfinite(estimate.B).all()), "Non-finite fit or metrics; task aborted.")
    return metrics


def task_paths(output: Path, task_id: int) -> tuple[Path, Path, Path]:
    stem = f"task_{task_id:06d}"
    directory = output / "tasks"
    return directory / f"{stem}.json", directory / f"{stem}.npz", directory / f"{stem}.lock"


def validate_support(S: np.ndarray, dimension: int) -> None:
    require(S.ndim == 1 and np.issubdtype(S.dtype, np.integer), "Support must be an integer vector.")
    require(np.array_equal(S, np.unique(S)) and bool(np.all((S >= 0) & (S < dimension))), "Support indices are unsorted, duplicated or out of bounds.")


def read_task(output: Path, plan: dict, task_id: int) -> dict:
    path, arrays_path, _ = task_paths(output, task_id)
    require(path.exists() and arrays_path.exists(), f"Partial checkpoint for task {task_id}; both JSON and NPZ are required.")
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    require(record["record_sha256"] == digest(body), f"Task {task_id} record checksum mismatch.")
    require(record["plan_id"] == plan["plan_id"] and record["task"] == plan["tasks"][task_id], f"Task {task_id} plan/configuration mismatch.")
    require(record["source_fingerprints"] == plan["source_fingerprints"] and record["environment"] == plan["environment"], f"Task {task_id} code/environment mismatch.")
    require(record["arrays_file"] == arrays_path.name and record["npz_sha256"] == file_digest(arrays_path), f"Task {task_id} NPZ checksum mismatch.")
    require([row["method"] for row in record["results"]] == plan["methods"], f"Task {task_id} methods are missing, duplicated or reordered.")
    config = config_from_dict(record["task"]["config"])
    with np.load(arrays_path, allow_pickle=False) as saved:
        B_true = saved["B_true"]
        require(B_true.shape == config.dims and bool(np.isfinite(B_true).all()), "Invalid true coefficient tensor.")
        true_supports = tuple(saved[f"S{i}_true"] for i in (1, 2, 3))
        for S, dimension, size in zip(true_supports, config.dims, config.sparsity):
            validate_support(S, dimension)
            require(len(S) == size, "True support sizes do not match plan.")
        require(all(np.array_equal(S, recovered) for S, recovered in zip(true_supports, support_from_tensor(B_true))), "True tensor and true supports disagree.")
        require(np.count_nonzero(B_true) == int(np.prod(config.sparsity)), "True nonzero block is not rectangular and complete.")
        noise = saved["train_noise"]
        require(noise.shape == (config.n_train,) and bool(np.isfinite(noise).all()), "Invalid saved training noise.")
        require(record["input_fingerprints"]["B_true"] == array_digest(B_true) and record["input_fingerprints"]["train_noise"] == array_digest(noise), "Saved input fingerprints disagree with NPZ.")
        for row in record["results"]:
            method = row["method"]
            require(row["name"] == METHOD_NAMES[method], "Method label mismatch.")
            metrics = row["metrics"]
            require(bool(np.isfinite(list(metrics.values())).all()), f"Non-finite metrics in task {task_id}.")
            B = saved[f"B_{method}"]
            supports = tuple(saved[f"S{i}_{method}"] for i in (1, 2, 3))
            require(B.shape == config.dims and bool(np.isfinite(B).all()), "Invalid fitted coefficient tensor.")
            mask = np.zeros(config.dims, dtype=bool)
            for S, dimension in zip(supports, config.dims):
                validate_support(S, dimension)
            mask[np.ix_(*supports)] = True
            require(np.count_nonzero(B[~mask]) == 0, "Fitted tensor is nonzero outside its reported support.")
            recomputed = {**compute_support_metrics(*supports, *true_supports), **compute_coefficient_metrics(B, B_true)}
            for metric, value in recomputed.items():
                require(abs(float(metrics[metric]) - value) <= 1e-10, f"Task {task_id} {metric} differs from NPZ recomputation.")
            require(abs(metrics["rss_train"] - 2 * config.n_train * metrics["loss_train"]) <= 1e-8, "RSS and loss are inconsistent.")
            require(metrics["exact_recovery"] in {0.0, 1.0}, "Exact recovery is not binary.")
            if method == "fixed":
                require(tuple(len(S) for S in supports) == config.sparsity, "Known-s output does not have the required sizes.")
            if method == "oracle":
                require(metrics["exact_recovery"] == 1.0, "Oracle support is not the true support.")
            if method in {"fixed", "adaptive"}:
                trace = row["trace"]
                require(bool(trace["loss_history"]) and bool(np.isfinite(trace["loss_history"]).all()), "Invalid solver loss history.")
                require(bool(np.isfinite(trace["sic_history"]).all()) and np.isfinite(row["sic"]), "Invalid solver SIC diagnostics.")
                for i, dimension in enumerate(config.dims, 1):
                    S = saved[f"init_S{i}_{method}"]
                    validate_support(S, dimension)
                    require(S.tolist() == trace["pipeline_init_supports"][i - 1], "Saved initialization support mismatch.")
                    glasso_S = saved[f"glasso_init_S{i}_{method}"]
                    validate_support(glasso_S, dimension)
                    require(glasso_S.tolist() == trace["pipeline_glasso_init_supports"][i - 1], "Saved GLasso initialization mismatch.")
    return record


def run_task(output: Path, task_id: int, *, resume: bool = False) -> str:
    plan = load_plan(output, check_environment=True)
    require(0 <= task_id < len(plan["tasks"]), "Task ID outside plan.")
    path, arrays_path, lock_path = task_paths(output, task_id)
    if lock_path.exists():
        raise FileExistsError(f"Task {task_id} is locked; inspect its active/failed job before retrying.")
    if path.exists() or arrays_path.exists():
        require(path.exists() and arrays_path.exists(), f"Partial checkpoint for task {task_id}; refusing repair or overwrite.")
        if not resume:
            raise FileExistsError(f"Task {task_id} already exists; use --resume to validate and skip it.")
        read_task(output, plan, task_id)
        return "skipped"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(f"pid={os.getpid()}\nplan_id={plan['plan_id']}\n")
    try:
        task = plan["tasks"][task_id]
        dataset = generate_simulation_dataset(config_from_dict(task["config"]), seed=task["seed"])
        noise = dataset.y_train - tensor_inner(dataset.X_train, dataset.B_true)
        arrays = {"B_true": dataset.B_true, "train_noise": noise}
        arrays.update({f"S{i}_true": S for i, S in enumerate(dataset.supports_true, 1)})
        results = []
        for method in plan["methods"]:
            estimate, trace = fit_method(dataset, method, task["seed"])
            metrics = evaluate(dataset, estimate, task["test_seed"])
            arrays[f"B_{method}"] = estimate.B
            arrays.update({f"S{i}_{method}": S for i, S in enumerate((estimate.S1, estimate.S2, estimate.S3), 1)})
            if method in {"fixed", "adaptive"}:
                for key, prefix in (("pipeline_init_supports", "init"), ("pipeline_glasso_init_supports", "glasso_init")):
                    supports = trace[key]
                    arrays.update({f"{prefix}_S{i}_{method}": S for i, S in enumerate(supports, 1)})
                    trace[key] = [S.tolist() for S in supports]
            results.append({"method": method, "name": METHOD_NAMES[method], "metrics": metrics, "sic": estimate.sic, "details": estimate.details, "trace": trace})
        handle = io.BytesIO()
        np.savez_compressed(handle, **arrays)
        array_bytes = handle.getvalue()
        record = {
            "plan_id": plan["plan_id"], "task": task, "environment": plan["environment"],
            "source_fingerprints": plan["source_fingerprints"],
            "input_fingerprints": {"X_train": array_digest(dataset.X_train), "y_train": array_digest(dataset.y_train), "B_true": array_digest(dataset.B_true), "train_noise": array_digest(noise)},
            "actual_snr": dataset.signal_variance / dataset.config.sigma**2,
            "results": results, "arrays_file": arrays_path.name,
            "npz_sha256": hashlib.sha256(array_bytes).hexdigest(),
        }
        record["record_sha256"] = digest(record)
        commit_exclusive(arrays_path, array_bytes)
        commit_exclusive(path, json_bytes(record))
        read_task(output, plan, task_id)
    finally:
        lock_path.unlink()
    return "completed"


def status(output: Path) -> dict:
    plan = load_plan(output)
    expected_names = {f"task_{task['task_id']:06d}{suffix}" for task in plan["tasks"] for suffix in (".json", ".npz", ".lock")}
    require(all(path.name in expected_names for path in (output / "tasks").glob("task_*")), "Unexpected task files outside plan.")
    completed, missing, running = [], [], []
    for task in plan["tasks"]:
        task_id = task["task_id"]
        path, arrays_path, lock = task_paths(output, task_id)
        if lock.exists():
            running.append(task_id)
        if lock.exists() and not (path.exists() and arrays_path.exists()):
            missing.append(task_id)
            continue
        if path.exists() or arrays_path.exists():
            read_task(output, plan, task_id)
            completed.append(task_id)
        else:
            missing.append(task_id)
    return {"task_total": len(plan["tasks"]), "completed": len(completed), "missing": missing, "running_or_locked": running, "rerun_task_ids": [task_id for task_id in missing if task_id not in running]}


def flat_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        task = record["task"]
        config = task["config"]
        for result in record["results"]:
            rows.append({
                **{key: task[key] for key in ("task_id", "scenario_id", "experiment", "parameter", "scenario_value", "scenario_order", "replicate", "seed", "test_seed")},
                "n_train": config["n_train"], "n_test": config["n_test"],
                **{f"p{i}": size for i, size in enumerate(config["dims"], 1)},
                "actual_snr": record["actual_snr"], "method": result["method"], "method_name": result["name"],
                **result["metrics"],
            })
    return rows


def summary_rows(rows: list[dict], plan: dict, metric_names: list[str]) -> list[dict]:
    summaries = []
    for scenario_id in range(14):
        for method in plan["methods"]:
            members = [row for row in rows if row["scenario_id"] == scenario_id and row["method"] == method]
            require(len(members) == plan["n_reps"], "Merged scenario/method repetition count differs from plan.")
            first = members[0]
            summary = {key: first[key] for key in ("scenario_id", "experiment", "parameter", "scenario_value", "scenario_order", "n_train", "n_test", "p1", "p2", "p3", "actual_snr", "method", "method_name")}
            summary["n_reps"] = len(members)
            for metric in metric_names:
                values = np.array([row[metric] for row in members], dtype=float)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_q1"] = float(np.quantile(values, .25))
                summary[f"{metric}_q3"] = float(np.quantile(values, .75))
            summary["exact_recovery_successes"] = int(sum(row["exact_recovery"] for row in members))
            summary["exact_recovery_denominator"] = len(members)
            summaries.append(summary)
    return summaries


def csv_bytes(rows: list[dict], fields: list[str] | None = None) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields if fields is not None else list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def merge(output: Path) -> dict:
    plan = load_plan(output)
    for filename in SUMMARY_FILES:
        if (output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite summary: {filename}")
    state = status(output)
    require(not state["missing"] and not state["running_or_locked"], f"Run incomplete or locked; missing task IDs={state['missing']}, locked={state['running_or_locked']}")
    lock_path = output / "merge.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(f"pid={os.getpid()}\n")
    try:
        records = [read_task(output, plan, task["task_id"]) for task in plan["tasks"]]
        rows = flat_rows(records)
        require(len(rows) == len(plan["tasks"]) * len(plan["methods"]), "Merged method count differs from plan.")
        require(len({(row["task_id"], row["method"]) for row in rows}) == len(rows), "Duplicate merged method rows.")
        metric_names = list(records[0]["results"][0]["metrics"])
        require(all(set(result["metrics"]) == set(metric_names) for record in records for result in record["results"]), "Metric schemas differ between tasks.")
        summaries = summary_rows(rows, plan, metric_names)
        pairs, gaps = [], []
        for record in records:
            task = record["task"]
            results = {row["method"]: row["metrics"] for row in record["results"]}
            identity = {key: task[key] for key in ("task_id", "scenario_id", "experiment", "scenario_value", "replicate", "seed")}
            if {"fixed", "adaptive"} <= set(results):
                pairs.append({**identity, **{f"fixed_minus_adaptive_{metric}": results["fixed"][metric] - results["adaptive"][metric] for metric in metric_names}})
            if {"fixed", "oracle"} <= set(results):
                gap = results["fixed"]["rss_train"] - results["oracle"]["rss_train"]
                gaps.append({**identity, "fixed_rss_train": results["fixed"]["rss_train"], "oracle_rss_train": results["oracle"]["rss_train"], "fixed_rss_minus_oracle": gap, "lower_loss_true_support_exists": gap > 1e-8})
        report = {
            "plan_id": plan["plan_id"], "task_count": len(records), "method_result_count": len(rows),
            "scenario_count": 14, "n_reps_per_scenario": plan["n_reps"], "methods": plan["methods"],
            "paired_fixed_vs_adaptive_count": len(pairs), "paired_fixed_vs_adaptive_applicable": {"fixed", "adaptive"} <= set(plan["methods"]),
            "search_gap_count": len(gaps), "search_gap_applicable": {"fixed", "oracle"} <= set(plan["methods"]),
            "integrity_checks_passed": True, "standard_deviation": "sample, ddof=1; blank/undefined for n=1",
            "initialization_limitation": plan["interpretation"],
            "search_gap_interpretation": "Positive gap beyond 1e-8 certifies an available lower-training-loss true-support candidate. Oracle is not necessarily the global minimum RSS or a universal test-risk lower bound.",
            "source_fingerprints": plan["source_fingerprints"], "worker_environment": plan["environment"],
            "paired_statistics_are_descriptive": True,
        }
        pair_fields = ["task_id", "scenario_id", "experiment", "scenario_value", "replicate", "seed"] + [f"fixed_minus_adaptive_{metric}" for metric in metric_names]
        gap_fields = ["task_id", "scenario_id", "experiment", "scenario_value", "replicate", "seed", "fixed_rss_train", "oracle_rss_train", "fixed_rss_minus_oracle", "lower_loss_true_support_exists"]
        payloads = {
            "raw_results.csv": csv_bytes(rows), "summary_results.csv": csv_bytes(summaries),
            "paired_fixed_vs_adaptive.csv": csv_bytes(pairs, pair_fields), "search_gap.csv": csv_bytes(gaps, gap_fields),
            "integrity_report.json": json_bytes(report),
        }
        for filename, content in payloads.items():
            commit_exclusive(output / filename, content)
    finally:
        lock_path.unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--profile", choices=("full", "quick"), default="full")
    planning.add_argument("--n-reps", type=int, default=50)
    planning.add_argument("--methods", choices=tuple(METHOD_NAMES), nargs="+", default=["fixed", "adaptive", "oracle"])
    planning.add_argument("--output-dir", type=Path, required=True)
    running = commands.add_parser("run")
    selection = running.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-id", type=int)
    selection.add_argument("--all", action="store_true")
    running.add_argument("--output-dir", type=Path, required=True)
    running.add_argument("--resume", action="store_true")
    for name in ("merge", "status"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = create_plan(args.output_dir, args.profile, args.n_reps, args.methods)
        print(f"Plan {plan['plan_id']}: {len(plan['tasks'])} tasks, {len(plan['tasks']) * len(plan['methods'])} method results; 1 CPU thread per task.")
        return 0
    if args.command == "run":
        plan = load_plan(args.output_dir, check_environment=True)
        task_ids = range(len(plan["tasks"])) if args.all else [args.task_id]
        for task_id in task_ids:
            outcome = run_task(args.output_dir, task_id, resume=args.resume)
            print(f"task_id={task_id} {outcome}", flush=True)
        return 0
    if args.command == "merge":
        report = merge(args.output_dir)
        print(f"Merged {report['task_count']} tasks and {report['method_result_count']} method results into {args.output_dir}.")
        return 0
    state = status(args.output_dir)
    print(f"Tasks: total={state['task_total']} completed={state['completed']} missing={len(state['missing'])} locked={state['running_or_locked']}")
    print("Rerun task IDs: " + ",".join(str(task_id) for task_id in state["rerun_task_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
