"""Serial, bounded five-method comparison with sealed evaluation split selection."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .baselines import fit_baseline, make_inner_folds, stable_seed
from .data import load_dataset
from .identity import code_identity, verify_release
from .metrics import classification_metrics, predict_scores, tie_priority


METHODS = ("majority", "ridge_ovr", "multinomial_l2", "multinomial_l1", "tensor_abess_ovr_ls")


def _write_json(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _environment():
    return {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
            "threadpools": threadpool_info(), "parallelism": {"outer": 1, "class": 1, "blas_openmp": 1, "sklearn": 1}}


class _DiagnosticLog(list):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def append(self, item):
        super().append(item)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def _baseline_save(model, path, training_ids, y):
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"classes": model.classes_, "tie_priority": model.tie_priority_, "class_prior": model.prior_}
    if model.preprocessor_ is not None:
        arrays.update(mean=model.preprocessor_.mean_, scale=model.preprocessor_.scale_,
                      constant_mask=model.preprocessor_.constant_mask_)
        arrays.update(coef=np.asarray(model.estimator_.coef_), intercept=np.asarray(model.estimator_.intercept_))
    np.savez_compressed(path / "model.npz", **arrays)
    _write_json(path / "model.json", {"method": model.method, "classes": model.classes_.tolist(),
                "tie_priority": model.tie_priority_.tolist(), "axis_metadata": model.axis_metadata,
                "selected_parameter": model.parameter_, "training_sample_ids": list(training_ids),
                "training_class_labels": np.asarray(y).tolist(), "numerical_ok": model.numerical_ok,
                "coordinates": "training_standardized", "dtype": "float64", "order": "C",
                "class_order_in_estimator": model.estimator_.classes_.tolist() if hasattr(model.estimator_, "classes_") else model.classes_.tolist()})


def _execute_task(task):
    """Child owns all output files; completed metadata is written last."""
    root = Path(task["task_dir"])
    diagnostics = _DiagnosticLog(root / "diagnostics.jsonl")
    method, fold = task["method"], task["fold"]
    started = time.perf_counter()
    result = {"method": method, "fold": fold, "evaluation_role": task["evaluation_role"], "status": "FAILED"}
    model = None
    try:
        with threadpool_limits(limits=1):
            if time.monotonic() >= task["deadline"]:
                raise TimeoutError("Dataset deadline exhausted before method fit")
            X, y, evaluate = task["X"], task["y"], task["evaluate"]
            classes = np.asarray(task["classes"], dtype=str)
            if method == "tensor_abess_ovr_ls":
                from .model import TensorABESSOvR
                model = TensorABESSOvR(classes=classes, axis_metadata=task["axes"], core_timeout=task["per_class_seconds"])
                try:
                    model.fit(X, y, sample_ids=task["training_ids"], deadline=task["deadline"])
                finally:
                    for row in model.fit_diagnostics_:
                        diagnostics.append({"method": method, "fold": fold, **row})
                model.save(task["model_dir"])
                numerical_ok = True
            else:
                model = fit_baseline(method, X, y, classes, inner_folds=task["inner_folds"], seed=task["seed"],
                                     fold_id=fold, axis_metadata=task["axes"], diagnostics=diagnostics,
                                     deadline=task["deadline"])
                numerical_ok = model.numerical_ok
                _baseline_save(model, Path(task["model_dir"]), task["training_ids"], y)
            result["training_seconds"] = time.perf_counter() - started
            prediction_start = time.perf_counter()
            scores = model.decision_function(evaluate)
            prediction, ties = predict_scores(scores, model.classes_, model.tie_priority_)
            result.update(prediction_seconds=time.perf_counter() - prediction_start, tie_count=ties,
                          classes=model.classes_.tolist(), tie_priority=model.tie_priority_.tolist(),
                          metrics=classification_metrics(task["evaluation_y"], prediction, classes),
                          model_path=task["model_relative"], status="COMPLETE" if numerical_ok else "NUMERICAL_BLOCKED")
            np.savez_compressed(root / "scores.npz", scores=scores, predictions=prediction,
                                sample_ids=np.asarray(task["evaluation_ids"], dtype=str), classes=model.classes_)
            supports = []
            if method == "tensor_abess_ovr_ls":
                union = np.zeros(model.coef_.shape[1:], dtype=bool)
                for column, label in enumerate(model.classes_):
                    support = model.supports_[column]
                    union[np.ix_(*[np.asarray(values, dtype=int) for values in support])] = True
                    supports.append({"method": method, "fold": fold, "class_key": str(label), "supports": support,
                                     "q": int(np.prod([len(values) for values in support])),
                                     "actual_nonzero_1e8": int(np.sum(np.abs(model.coef_[column]) > 1e-8)),
                                     "axis_shape": list(model.coef_.shape[1:]), "coordinate_system": "training_standardized"})
                result.update(total_q=sum(row["q"] for row in supports), unique_union_coordinates=int(union.sum()))
            if method == "multinomial_l1":
                result["actual_nonzero_1e8"] = int(np.sum(np.abs(model.estimator_.coef_) > 1e-8))
            _write_json(root / "supports.json", supports)
            result["environment"] = _environment()
    except BaseException as error:
        cause, resource_error = error, False
        while cause is not None:
            resource_error |= isinstance(cause, TimeoutError)
            cause = cause.__cause__
        result["status"] = "RESOURCE_BLOCKED" if resource_error else "FAILED"
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        diagnostics.append({"method": method, "fold": fold, "phase": "task", "status": result["status"],
                            "error_type": type(error).__name__, "error": str(error)})
    finally:
        result["total_seconds"] = time.perf_counter() - started
        _write_json(root / "result.json", result)


def _inner_plan(ds, split, indices):
    lookup = {key: index for index, key in enumerate(ds["sample_ids"][indices].tolist())}
    if "inner_folds" in split:
        folds = [(np.array([lookup[key] for key in item["train_ids"]]), np.array([lookup[key] for key in item["eval_ids"]]))
                 for item in split["inner_folds"]]
        if len(folds) not in (2, 3):
            raise ValueError("Explicit inner protocol requires 3 folds or declared 2-fold downgrade")
        if len(folds) == 2 and not split.get("inner_fold_downgrade_reason"):
            raise ValueError("Two explicit inner folds require inner_fold_downgrade_reason")
        required = set(ds["metadata"]["classes"])
        for train, evaluate in folds:
            if set(ds["y"][indices][train]) != required or set(ds["y"][indices][evaluate]) != required:
                raise ValueError("Explicit inner folds must cover every class")
        info = {"actual_folds": len(folds), "source": "explicit_splits", "downgrade_reasons": [split.get("inner_fold_downgrade_reason")] if len(folds) == 2 else []}
    else:
        folds, info = make_inner_folds(ds["y"][indices], ds["metadata"]["classes"], ds["metadata"]["dependency"],
                                       stable_seed(ds["config"]["seed"], split["split_id"], "inner_splits"),
                                       groups=None if ds["group_ids"] is None else ds["group_ids"][indices],
                                       times=None if ds["time_values"] is None else ds["time_values"][indices])
    ids = ds["sample_ids"][indices]
    info["folds"] = [{"inner_fold": j, "train_ids": ids[a].tolist(), "eval_ids": ids[b].tolist(),
                       "training_groups": None if ds["group_ids"] is None else np.unique(ds["group_ids"][indices][a]).tolist(),
                       "evaluation_groups": None if ds["group_ids"] is None else np.unique(ds["group_ids"][indices][b]).tolist()}
                      for j, (a, b) in enumerate(folds)]
    return folds, info


def run(data_path, config_path, output_dir):
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    verify_release()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output exists; refusing overwrite: {output}")
    ds = load_dataset(data_path, config_path)
    config, metadata = ds["config"], ds["metadata"]
    if tuple(config.get("methods", METHODS)) != METHODS:
        raise ValueError("The fixed primary comparison requires all five methods in declared order")
    resource = config["resource_budget"]
    per_class = float(resource["per_class_seconds"])
    total_budget = float(resource["total_seconds"])
    if not np.isfinite([per_class, total_budget]).all() or min(per_class, total_budget) <= 0:
        raise ValueError("Resource budgets must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + total_budget
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    before_identity = code_identity()
    with threadpool_limits(limits=1):
        environment = _environment()
    _write_json(output / "config.json", config)
    _write_json(output / "dataset.json", metadata)
    _write_json(output / "splits.json", {"splits": ds["selected_splits"]})
    _write_json(output / "environment.json", environment)
    _write_json(output / "data_identity.json", ds["identity"])
    (output / "core_profile.json").write_bytes((Path(__file__).parent / "core_profile.json").read_bytes())
    selected_ids = {key for split in ds["selected_splits"] for key in split["train_ids"] + split["eval_ids"]}
    _write_csv(output / "sample_metadata.csv", [{"sample_id": str(key), "class_key": str(ds["y"][i]),
               "group_id": "" if ds["group_ids"] is None else str(ds["group_ids"][i]),
               "time": "" if ds["time_values"] is None else str(ds["time_values"][i])}
              for i, key in enumerate(ds["sample_ids"]) if key in selected_ids], ["sample_id", "class_key", "group_id", "time"])
    manifest = {"schema_version": "1.0", "application_version": "0.1.0", "run_id": run_id, "status": "RUNNING",
                "evaluation_stage": config["evaluation_stage"], "formal_test_evaluated": config["evaluation_stage"] == "final",
                "classes": metadata["classes"], "methods": list(METHODS), "expected_tasks": len(METHODS) * len(ds["selected_splits"]),
                "completed_tasks": 0, "failed_tasks": 0, "code_identity": before_identity, "data_identity": ds["identity"],
                "seed": int(config["seed"]), "threads": environment["parallelism"], "resource_budget": resource,
                "cache_policy": "no_reuse_new_run_id; completed task identity includes data/split/config/code/environment/seed",
                "tasks": [], "score_arrays": [], "inner_protocols": {}}
    _write_json(output / "manifest.json", manifest)
    all_predictions, score_arrays, metric_rows, per_class_rows, support_rows, diagnostic_rows, confusion = [], {}, [], [], [], [], {}
    id_index = {key: index for index, key in enumerate(ds["sample_ids"].tolist())}
    ctx = multiprocessing.get_context("spawn")
    for fold_number, split in enumerate(ds["selected_splits"]):
        train = np.array([id_index[key] for key in split["train_ids"]], dtype=int)
        evaluate = np.array([id_index[key] for key in split["eval_ids"]], dtype=int)
        try:
            inner_folds, inner_info = _inner_plan(ds, split, train)
            inner_error = None
            manifest["inner_protocols"][split["split_id"]] = inner_info
        except ValueError as error:
            inner_folds, inner_error = [], str(error)
            manifest["inner_protocols"][split["split_id"]] = {"status": "BLOCKED", "error": inner_error}
        for method_number, method in enumerate(METHODS):
            # File paths use numeric fold indices, never untrusted sample/split identities.
            key = f"fold_{fold_number:03d}_method_{method_number:02d}"
            task_dir = output / "tasks" / key
            task_dir.mkdir(parents=True)
            model_relative = f"models/fold_{fold_number:03d}/{method}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {"method": method, "fold": split["split_id"], "status": "RESOURCE_BLOCKED", "error": "Dataset wall-clock budget exhausted"}
            elif inner_error and method in ("ridge_ovr", "multinomial_l1", "multinomial_l2"):
                result = {"method": method, "fold": split["split_id"], "status": "PROTOCOL_BLOCKED", "error": inner_error}
            else:
                task = {"method": method, "fold": split["split_id"], "task_dir": str(task_dir), "X": np.asarray(ds["X"][train]),
                        "y": ds["y"][train], "evaluate": np.asarray(ds["X"][evaluate]), "evaluation_y": ds["y"][evaluate],
                        "classes": metadata["classes"], "axes": metadata["axes"], "training_ids": split["train_ids"],
                        "evaluation_ids": split["eval_ids"], "evaluation_role": split["evaluation_role"], "inner_folds": inner_folds,
                        "seed": int(config["seed"]), "per_class_seconds": min(per_class, remaining), "remaining_seconds": remaining, "deadline": deadline,
                        "model_dir": str(output / model_relative), "model_relative": model_relative}
                process = ctx.Process(target=_execute_task, args=(task,))
                process.start()
                # Model's absolute deadline stops/joins its own core child first.
                # This five-second grace permits cleanup only; no new fit starts.
                process.join(timeout=max(0., deadline - time.monotonic()) + 5.)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    result = {"method": method, "fold": split["split_id"], "status": "RESOURCE_BLOCKED", "error": "Dataset wall-clock budget reached; task process terminated"}
                elif not (task_dir / "result.json").is_file():
                    result = {"method": method, "fold": split["split_id"], "status": "FAILED", "error": f"Worker exited without result (exitcode={process.exitcode})"}
                else:
                    result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                process.close()
            result.update(evaluation_role=split["evaluation_role"], expected_sample_ids=split["eval_ids"],
                          expected_groups=["" if ds["group_ids"] is None else str(ds["group_ids"][i]) for i in evaluate],
                          expected_true_labels=ds["y"][evaluate].tolist(), train_ids=split["train_ids"])
            _write_json(task_dir / "result.json", result)
            if (task_dir / "diagnostics.jsonl").exists():
                diagnostic_rows.extend(json.loads(line) for line in (task_dir / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if line)
            if result["status"] not in ("COMPLETE", "NUMERICAL_BLOCKED"):
                diagnostic_rows.append({"method": method, "fold": split["split_id"], "phase": "task", "status": result["status"], "error": result.get("error")})
            if (task_dir / "scores.npz").exists() and result["status"] in ("COMPLETE", "NUMERICAL_BLOCKED"):
                with np.load(task_dir / "scores.npz", allow_pickle=False) as arrays:
                    score_arrays[key] = arrays["scores"].copy()
                    predictions = arrays["predictions"].copy()
                manifest["score_arrays"].append({"array_key": key, "method": method, "fold": split["split_id"],
                                "evaluation_role": split["evaluation_role"], "classes": result["classes"],
                                "sample_ids": split["eval_ids"], "tie_priority": result["tie_priority"],
                                "score_kind": "class_indicator_original_scale" if method in ("majority", "ridge_ovr", "tensor_abess_ovr_ls") else "multinomial_decision_logits"})
                all_predictions.extend({"method": method, "fold": split["split_id"], "sample_id": str(ds["sample_ids"][i]),
                                        "group_id": "" if ds["group_ids"] is None else str(ds["group_ids"][i]),
                                        "true_class": str(ds["y"][i]), "predicted_class": str(predictions[j]),
                                        "evaluation_role": split["evaluation_role"], "evaluation_stage": config["evaluation_stage"]}
                                       for j, i in enumerate(evaluate))
                metrics = result["metrics"]
                metric_rows.append({"method": method, "fold": split["split_id"], "status": result["status"],
                                    **{name: metrics[name] for name in ("n", "macro_f1", "accuracy", "balanced_accuracy", "complete_class_coverage")},
                                    "tie_count": result["tie_count"], "training_seconds": result["training_seconds"], "prediction_seconds": result["prediction_seconds"]})
                per_class_rows.extend({"method": method, "fold": split["split_id"], **row} for row in metrics["per_class"])
                confusion[key] = {"method": method, "fold": split["split_id"], "classes": metadata["classes"],
                                  "matrix": metrics["confusion_matrix"], "never_predicted": metrics["never_predicted"]}
                support_rows.extend(json.loads((task_dir / "supports.json").read_text(encoding="utf-8")))
            manifest["tasks"].append(result)
            manifest["completed_tasks"] += int(result["status"] == "COMPLETE")
            manifest["failed_tasks"] += int(result["status"] != "COMPLETE")
            _write_json(output / "manifest.json", manifest)
    if code_identity() != before_identity:
        raise RuntimeError("Code identity changed during run; results cannot be sealed")
    np.savez_compressed(output / "scores.npz", **score_arrays)
    _write_csv(output / "predictions.csv", all_predictions, ["method", "fold", "sample_id", "group_id", "true_class", "predicted_class", "evaluation_role", "evaluation_stage"])
    _write_csv(output / "metrics.csv", metric_rows, ["method", "fold", "status", "n", "macro_f1", "accuracy", "balanced_accuracy", "complete_class_coverage", "tie_count", "training_seconds", "prediction_seconds"])
    _write_csv(output / "per_class_metrics.csv", per_class_rows, ["method", "fold", "class_key", "precision", "recall", "f1", "true_count", "predicted_count", "true_positive_count"])
    _write_json(output / "confusion_matrices.json", confusion)
    _write_json(output / "scores_rows.json", manifest["score_arrays"])
    for name, rows in (("supports.jsonl", support_rows), ("fit_diagnostics.jsonl", diagnostic_rows)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    # OOF aggregation is valid only for disjoint evaluation rows; it is a pooled score, not a t-test over folds.
    pooled = {}
    for method in METHODS:
        rows = [row for row in all_predictions if row["method"] == method]
        ids = [row["sample_id"] for row in rows]
        method_tasks = [row for row in manifest["tasks"] if row["method"] == method]
        if rows and len(ids) == len(set(ids)) and all(row["status"] == "COMPLETE" for row in method_tasks):
            pooled[method] = classification_metrics([row["true_class"] for row in rows], [row["predicted_class"] for row in rows], metadata["classes"])
    _write_json(output / "pooled_oof_metrics.json", pooled)
    manifest.update(status="COMPLETE" if manifest["failed_tasks"] == 0 else "PARTIAL", wall_clock_seconds=time.monotonic() - started,
                    cache_completed=False, cache_reused_tasks=0)
    _write_json(output / "manifest.json", manifest)
    report = [f"# Run {run_id}", "", f"Status: {manifest['status']}", f"Evaluation stage: {config['evaluation_stage']}",
              f"Formal test evaluated: {manifest['formal_test_evaluated']}", f"Tasks: {manifest['completed_tasks']}/{manifest['expected_tasks']} complete; {manifest['failed_tasks']} not accepted", "",
              "Macro-F1 uses the complete fixed class list. Incomplete label coverage is descriptive only.",
              "Scores are real decision values; Tensor-ABESS scores are not probabilities.",
              "Core initialization remains fixed at 500 iterations; success is not an inner optimality certificate.",
              "Selected rectangular capacities and unions are descriptive, not statistical degrees of freedom.", ""]
    (output / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (output / "run.log").write_text("\n".join(f"{task['fold']} {task['method']} {task['status']} {task.get('error', '')}" for task in manifest["tasks"]) + "\n", encoding="utf-8")
    from .returns import validate_return, write_checksums
    write_checksums(output)
    validate_return(output)
    return {name: manifest[name] for name in ("run_id", "status", "expected_tasks", "completed_tasks", "failed_tasks", "formal_test_evaluated")} | {"run_dir": str(output)}
