"""Subject-level held-out evaluation and a separate descriptive final fit."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from .confounds import fit_preprocessor, validate_covariates, validate_tensor_response
from .io import read_json, write_json
from .models import fit_model


def _validate_analysis(X, y, subject_ids, C, config):
    X, y = validate_tensor_response(X, y)
    ids = np.asarray(subject_ids, dtype=str)
    if ids.ndim != 1 or len(ids) != len(y) or any(not value.strip() for value in ids):
        raise ValueError("subject_ids must be nonempty and aligned with X and y")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("subject_ids must be unique; duplicate subjects would leak across folds")
    mode = config.get("analysis", {}).get("mode", "adjusted")
    if mode not in {"adjusted", "pilot"}:
        raise ValueError("analysis.mode must be adjusted or explicitly pilot")
    scale = config.get("response", {}).get("primary_transform", "log")
    if scale not in {"log", "raw"}:
        raise ValueError("response.primary_transform must be log or raw; pass y on that scale")
    analysis = config.get("analysis", {})
    min_samples = analysis.get("min_samples", 100)
    if isinstance(min_samples, bool) or int(min_samples) != min_samples or min_samples < 2:
        raise ValueError("analysis.min_samples must be an integer of at least two")
    if len(y) < min_samples:
        raise ValueError(f"Analysis requires at least {min_samples} samples; found {len(y)}")
    if mode == "adjusted" and (C is None or np.asarray(C).ndim != 2 or np.asarray(C).shape[1] < 3):
        raise ValueError("Formal analysis requires covariates: explicit intercept, age, and sex")
    if mode == "pilot" and C is not None:
        raise ValueError("Unadjusted pilot requires C=None; use adjusted mode for covariates")
    design = validate_covariates(C, len(y), require_full_rank=True)
    return X, y, ids, (None if C is None else design), mode, scale


def _new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to mix or overwrite existing results: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)
    rss = float(error @ error)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return {"rmse": float(np.sqrt(rss / len(y))), "mae": float(np.mean(np.abs(error))),
            "r2": None if tss == 0 else float(1 - rss / tss)}


def _support_payload(result):
    return {"S_channel": result.S1.tolist(), "S_time": result.S2.tolist(),
            "S_frequency": result.S3.tolist(), "index_base": 0,
            "q": int(len(result.S1) * len(result.S2) * len(result.S3))}


def run_cross_validation(X, y, subject_ids, C, config, output_dir):
    """Fit preprocessing on each outer training fold; evaluate each subject once.

    y must already be on config.response.primary_transform (log or raw).
    Existing baseline solvers retain their own inner tuning mechanisms.
    """
    X, y, ids, C, mode, scale = _validate_analysis(X, y, subject_ids, C, config)
    analysis = config.get("analysis", {})
    n_folds = analysis.get("formal_cv_folds", 5)
    if isinstance(n_folds, bool) or int(n_folds) != n_folds or not 2 <= n_folds <= len(y) // 2:
        raise ValueError("Outer fold count must be an integer giving at least two held-out subjects per fold")
    n_folds = int(n_folds)
    seed = int(config.get("seed", 20260908))
    partitions = np.array_split(np.random.default_rng(seed).permutation(len(y)), n_folds)
    folds = []
    assignment = np.empty(len(y), dtype=int)
    for number, test in enumerate(partitions, start=1):
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        # Validate every fold before creating output or fitting a model.
        validate_covariates(None if C is None else C[train], len(train), require_full_rank=True)
        inner_folds = config.get("model", {}).get("cv_folds", 5)
        if not 2 <= inner_folds <= len(train):
            raise ValueError("Inner cv_folds exceeds outer training sample count")
        assignment[test] = number
        folds.append((number, train, test))
    model_names = ["Tensor-ABESS"]
    model_names += [name for name, flag in (("Tensor-GLasso", "run_tensor_glasso"),
                                          ("Vector Lasso", "run_vector_lasso"))
                    if analysis.get(flag, True)]
    out = _new_output(output_dir)
    _write_csv(out / "fold_assignments.csv", [
        {"subject_id": subject, "fold": int(fold)} for subject, fold in zip(ids, assignment)])
    oof = {name: np.full(len(y), np.nan) for name in model_names}
    selections = {name: [np.zeros(dimension, dtype=int) for dimension in X.shape[1:]] for name in model_names}
    prediction_rows, metric_rows = [], []
    for number, train, test in folds:
        C_train, C_test = (None, None) if C is None else (C[train], C[test])
        preprocessing = fit_preprocessor(X[train], y[train], C_train)
        train_X = preprocessing.apply(X[train], C_train)
        train_y = preprocessing.transform_response(y[train], C_train)
        fold_dir = out / "folds" / f"fold_{number}"
        fold_dir.mkdir(parents=True)
        np.savez_compressed(fold_dir / "preprocessing.npz", **preprocessing.to_arrays())
        write_json(fold_dir / "fold_info.json", {
            "fold": number, "train_indices": train.tolist(), "test_indices": test.tolist(),
            "train_subject_ids": ids[train].tolist(), "test_subject_ids": ids[test].tolist(),
            "response_scale": scale, "mode": mode})
        for name in model_names:
            start = perf_counter()
            result = fit_model(name, train_X, train_y, config.get("model", {}))
            predictions = preprocessing.predict(result.B, X[test], C_test)
            oof[name][test] = predictions
            for count, support in zip(selections[name], (result.S1, result.S2, result.S3)):
                count[support] += 1
            model_dir = fold_dir / name.replace(" ", "_")
            model_dir.mkdir()
            np.save(model_dir / "B.npy", result.B, allow_pickle=False)
            write_json(model_dir / "support.json", _support_payload(result))
            write_json(model_dir / "fit_summary.json", {
                "model": name, "n_train": len(train), "training_residual_rss": result.rss,
                "training_residual_loss": result.loss, "sic": result.sic,
                "runtime_sec": float(perf_counter() - start), "response_scale": scale})
            metric_rows.append({"fold": number, "model": name, "n_train": len(train),
                                "n_test": len(test), "response_scale": scale,
                                **_metrics(y[test], predictions)})
            prediction_rows.extend({"subject_id": str(ids[index]), "fold": number,
                                    "model": name, "observed": float(y[index]),
                                    "predicted": float(prediction), "response_scale": scale}
                                   for index, prediction in zip(test, predictions))
    summary_rows = []
    for name in model_names:
        if not np.isfinite(oof[name]).all():
            raise ValueError(f"Incomplete or non-finite out-of-fold predictions for {name}")
        row = {"model": name, "n_subjects": len(y), "n_folds": n_folds,
               "mode": mode, "PILOT_UNADJUSTED": mode == "pilot", "response_scale": scale}
        scores = _metrics(y, oof[name])
        for metric in ("rmse", "mae", "r2"):
            values = [item[metric] for item in metric_rows if item["model"] == name]
            row[f"{metric}_mean"] = None if None in values else float(np.mean(values))
            row[f"{metric}_std"] = None if None in values else float(np.std(values, ddof=1))
            row[f"oof_{metric}"] = scores[metric]
        summary_rows.append(row)
    _write_csv(out / "oof_predictions.csv", prediction_rows)
    _write_csv(out / "cv_metrics.csv", metric_rows)
    _write_csv(out / "cv_summary.csv", summary_rows)
    for mode_index, axis in enumerate(("channel", "time", "frequency")):
        rows = [{"model": name, "index": index, "selection_count": int(count),
                 "n_folds": n_folds, "selection_frequency": float(count / n_folds)}
                for name in model_names for index, count in enumerate(selections[name][mode_index])]
        _write_csv(out / f"{axis}_selection_frequency.csv", rows)
    summary = {"complete": True, "mode": mode, "PILOT_UNADJUSTED": mode == "pilot",
               "seed": seed, "n_subjects": len(y), "n_folds": n_folds,
               "subject_ids": ids.tolist(), "tensor_shape": list(X.shape),
               "response_scale": scale, "models": summary_rows,
               "preprocessing": "outer_training_fold_only", "config": config,
               "r2_policy": "undefined for constant observations; stored as null"}
    # This completion marker is written last, only after every output exists.
    write_json(out / "cv_summary.json", summary)
    return read_json(out / "cv_summary.json")


def fit_final_model(X, y, subject_ids, C, config, output_dir):
    """Fit the interpretation model; the CLI must verify CV completion first."""
    X, y, ids, C, mode, scale = _validate_analysis(X, y, subject_ids, C, config)
    out = _new_output(output_dir)
    preprocessing = fit_preprocessor(X, y, C)
    result = fit_model("Tensor-ABESS", preprocessing.apply(X, C),
                       preprocessing.transform_response(y, C), config.get("model", {}))
    np.save(out / "B_hat.npy", result.B, allow_pickle=False)
    np.savez_compressed(out / "preprocessing.npz", **preprocessing.to_arrays())
    for axis, support in zip(("channel", "time", "frequency"), (result.S1, result.S2, result.S3)):
        write_json(out / f"S_{axis}.json", support.tolist())
    summary = {"model": "Tensor-ABESS", "mode": mode, "PILOT_UNADJUSTED": mode == "pilot",
               "independent_test_performance": False, "subject_ids": ids.tolist(),
               "n_subjects": len(y), "tensor_shape": list(X.shape),
               "coefficient_shape": list(result.B.shape), "response_scale": scale,
               "coefficient_scale": "per training residual standard deviation of each tensor cell",
               "rss": result.rss, "loss": result.loss,
               "fit_statistics_scale": "training centered nuisance residual response",
               "criterion": config.get("model", {}).get("adaptive_criterion", "msic"),
               "sic": result.sic, "config": config, **_support_payload(result)}
    write_json(out / "final_model_summary.json", summary)
    return read_json(out / "final_model_summary.json")
