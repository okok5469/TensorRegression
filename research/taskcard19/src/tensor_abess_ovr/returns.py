"""Content, identity and mathematical validation before exchanging result ZIPs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .metrics import classification_metrics, predict_scores, tie_priority


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.json"}


def write_checksums(run_dir):
    root = Path(run_dir).resolve()
    hashes = {name: _sha256(path) for name, path in sorted(_files(root).items())}
    (root / "SHA256SUMS.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashes


def _equal_float(expected, actual, context):
    if not np.isfinite(float(actual)) or not np.isclose(float(expected), float(actual), rtol=1e-10, atol=1e-12):
        raise ValueError(f"Metric mismatch at {context}: expected {expected}, got {actual}")


def _validate_diagnostics(root, manifest, tasks):
    from .baselines import select_parameter, stable_seed
    diagnostics = _jsonl(root / "fit_diagnostics.jsonl")
    grid = [float(10.**p) for p in range(-4, 5)]
    for (method, fold), task in tasks.items():
        if task["status"] not in ("COMPLETE", "NUMERICAL_BLOCKED"):
            if not any(row.get("method") == method and row.get("fold") == fold for row in diagnostics):
                raise ValueError("Failed task is missing failure diagnostics")
            continue
        rows = [row for row in diagnostics if row.get("method") == method and row.get("fold") == fold]
        if method == "majority":
            if len([row for row in rows if row.get("phase") == "final" and row.get("status") == "COMPLETE"]) != 1:
                raise ValueError("Missing majority fit diagnostics")
            continue
        if method == "tensor_abess_ovr_ls":
            core = [row for row in rows if row.get("stage") == "core_fit"]
            if len(core) != len(manifest["classes"]) or {row["class_label"] for row in core} != set(manifest["classes"]) or any(row["status"] != "COMPLETE" for row in core):
                raise ValueError("Complete Tensor task lacks complete per-class core diagnostics")
            continue
        inner_count = manifest["inner_protocols"][fold]["actual_folds"]
        score_rows = [row for row in rows if row.get("phase") == "inner_score"]
        score_map = {(row["candidate_index"], row["inner_fold"]): row for row in score_rows}
        wanted = {(c, j) for c in range(9) for j in range(inner_count)}
        if set(score_map) != wanted or len(score_map) != len(score_rows):
            raise ValueError("Missing/duplicate baseline candidate scoring diagnostics")
        unresolved = False

        def check_attempts(attempts, expected_parameter, expected_seed):
            nonlocal unresolved
            if not attempts or len(attempts) > 2:
                raise ValueError("Missing/duplicate baseline fit diagnostics")
            if any(row["parameter"] != expected_parameter or row["seed"] != expected_seed for row in attempts):
                raise ValueError("Candidate parameter/seed diagnostics mismatch")
            if method == "ridge_ovr":
                if len(attempts) != 1 or attempts[0].get("solver") != "svd" or attempts[0]["status"] != "COMPLETE":
                    raise ValueError("Ridge diagnostics mismatch")
                return
            if [row["max_iter"] for row in attempts] != [5000, 20000][:len(attempts)]:
                raise ValueError("Logistic retry violates frozen 5000/20000 budget")
            for i, row in enumerate(attempts):
                warning = any(item["category"] == "ConvergenceWarning" for item in row["warnings"])
                hit_limit = any(value >= row["max_iter"] for value in row["n_iter"])
                if row["objective"] != "multinomial" or row["tol"] != 1e-6 or row["class_weight"] is not None or row["fit_intercept"] is not True:
                    raise ValueError("Logistic objective/precision diagnostics mismatch")
                if row["hit_iteration_limit"] != hit_limit or row["status"] != ("NUMERICAL_UNRESOLVED" if warning or hit_limit else "COMPLETE"):
                    raise ValueError("Logistic convergence state mismatch")
                if i == 0 and len(attempts) == 2 and row["status"] == "COMPLETE":
                    raise ValueError("Logistic unnecessary parameter retry")
            unresolved |= attempts[-1]["status"] != "COMPLETE"

        for c, j in sorted(wanted):
            seed = stable_seed(manifest["seed"], method, fold, "candidate", c, "inner", j)
            attempts = [row for row in rows if row.get("phase") == "inner" and row.get("candidate_index") == c and row.get("inner_fold") == j]
            check_attempts(attempts, grid[c], seed)
            if score_map[c, j]["parameter"] != grid[c] or score_map[c, j]["seed"] != seed:
                raise ValueError("Inner score parameter/seed mismatch")
        selection = [row for row in rows if row.get("phase") == "selection"]
        if len(selection) != 1:
            raise ValueError("Missing baseline parameter selection diagnostics")
        means = [float(np.mean([score_map[c, j]["macro_f1"] for j in range(inner_count)])) for c in range(9)]
        chosen = select_parameter(method, grid, means)
        if selection[0]["grid"] != grid or selection[0]["mean_macro_f1"] != means or selection[0]["selected_parameter"] != chosen:
            raise ValueError("Baseline selection does not match complete fixed grid")
        final = [row for row in rows if row.get("phase") == "final"]
        check_attempts(final, chosen, stable_seed(manifest["seed"], method, fold, "final", chosen))
        if (task["status"] == "NUMERICAL_BLOCKED") != bool(unresolved):
            raise ValueError("Unresolved necessary candidate/final fit cannot be accepted as complete")


def _validate_inner_protocols(manifest, dataset, splits, samples):
    classes, dependency = set(manifest["classes"]), dataset["dependency"]
    if set(manifest["inner_protocols"]) != {s["split_id"] for s in splits}:
        raise ValueError("Inner protocol fold inventory mismatch")
    for split in splits:
        outer_train, outer_eval = set(split["train_ids"]), set(split["eval_ids"])
        if outer_train & outer_eval or not outer_train or not outer_eval or {samples[key]["class_key"] for key in outer_train} != classes:
            raise ValueError("Outer split overlap or incomplete training class coverage")
        if dependency == "group" and {samples[key]["group_id"] for key in outer_train} & {samples[key]["group_id"] for key in outer_eval}:
            raise ValueError("Outer group overlap")
        if dependency == "time" and max(float(samples[key]["time"]) for key in outer_train) >= min(float(samples[key]["time"]) for key in outer_eval):
            raise ValueError("Outer forward time boundary violated")
        plan = manifest["inner_protocols"][split["split_id"]]
        if plan.get("status") == "BLOCKED":
            continue
        folds = plan["folds"]
        if plan["actual_folds"] not in (2, 3) or len(folds) != plan["actual_folds"]:
            raise ValueError("Invalid inner fold count")
        if plan["actual_folds"] == 2 and not plan["downgrade_reasons"]:
            raise ValueError("Two inner folds require recorded coverage downgrade reason")
        for fold in folds:
            train, evaluate = set(fold["train_ids"]), set(fold["eval_ids"])
            if len(train) != len(fold["train_ids"]) or len(evaluate) != len(fold["eval_ids"]) or not train or not evaluate:
                raise ValueError("Invalid/duplicate inner fold sample IDs")
            if train & evaluate or not train | evaluate <= set(split["train_ids"]):
                raise ValueError("Inner fold overlap or samples outside current training")
            if {samples[key]["class_key"] for key in train} != classes or {samples[key]["class_key"] for key in evaluate} != classes:
                raise ValueError("Inner fold is missing required classes")
            if dependency == "group":
                a, b = {samples[key]["group_id"] for key in train}, {samples[key]["group_id"] for key in evaluate}
                if a & b or a != set(fold["training_groups"]) or b != set(fold["evaluation_groups"]):
                    raise ValueError("Inner group overlap or group identity mismatch")
            if dependency == "time":
                if max(float(samples[key]["time"]) for key in train) >= min(float(samples[key]["time"]) for key in evaluate):
                    raise ValueError("Inner forward time boundary violated")


def validate_return(run_dir):
    root = Path(run_dir).resolve()
    required = {"manifest.json", "config.json", "dataset.json", "splits.json", "environment.json", "data_identity.json",
                "predictions.csv", "scores.npz", "scores_rows.json", "metrics.csv", "per_class_metrics.csv",
                "confusion_matrices.json", "supports.jsonl", "fit_diagnostics.jsonl", "RUN_REPORT.md", "run.log", "pooled_oof_metrics.json", "sample_metadata.csv", "core_profile.json"}
    hashes = _json(root / "SHA256SUMS.json")
    inventory = _files(root)
    if set(inventory) != set(hashes) or not required <= set(hashes):
        raise ValueError("Return file/hash inventory mismatch or missing required files")
    for name, path in inventory.items():
        if not path.resolve().is_relative_to(root) or _sha256(path) != hashes[name]:
            raise ValueError(f"SHA256 hash mismatch: {name}")
    manifest, config, dataset = (_json(root / name) for name in ("manifest.json", "config.json", "dataset.json"))
    if _json(root / "data_identity.json") != manifest["data_identity"] or manifest["seed"] != config["seed"] or manifest["resource_budget"] != config["resource_budget"]:
        raise ValueError("Data/config seed/resource identity mismatch")
    profile = _json(root / "core_profile.json")
    if _sha256(root / "core_profile.json") != manifest["code_identity"]["core_profile_sha256"] or config["core_profile"] != "v5.2-search-final-original-msic":
        raise ValueError("Frozen core profile identity mismatch")
    environment = _json(root / "environment.json")
    if environment["parallelism"] != manifest["threads"] or any(value != 1 for value in manifest["threads"].values()):
        raise ValueError("Thread identity differs from serial resource protocol")
    if any(pool["num_threads"] != 1 for pool in environment["threadpools"]):
        raise ValueError("Recorded numerical threadpool exceeds serial protocol")
    splits = _json(root / "splits.json")["splits"]
    methods = ["majority", "ridge_ovr", "multinomial_l2", "multinomial_l1", "tensor_abess_ovr_ls"]
    classes = manifest["classes"]
    if len(set(classes)) != len(classes) or set(classes) != set(dataset["classes"]) or len(classes) < 2:
        raise ValueError("Invalid class mapping")
    if manifest["methods"] != methods or len({s["split_id"] for s in splits}) != len(splits):
        raise ValueError("Method/fold identity mismatch")
    stage = config["evaluation_stage"]
    if stage not in ("dev", "final") or manifest["evaluation_stage"] != stage or manifest["formal_test_evaluated"] != (stage == "final"):
        raise ValueError("Invalid evaluation stage or formal-test declaration")
    role = {"dev": "validation", "final": "test"}[stage]
    if {split["split_id"] for split in splits} != set(config["active_split_ids"]) or any(split["purpose"] != stage or split["evaluation_role"] != role for split in splits):
        raise ValueError("Active split purpose conflicts with evaluation stage")
    expected_keys = {(method, split["split_id"]) for method in methods for split in splits}
    tasks = {(row["method"], row["fold"]): row for row in manifest["tasks"]}
    if len(tasks) != len(manifest["tasks"]) or set(tasks) != expected_keys or manifest["expected_tasks"] != len(expected_keys):
        raise ValueError("Expected/completed method-fold task count mismatch")
    complete_count = sum(task["status"] == "COMPLETE" for task in tasks.values())
    if manifest["completed_tasks"] != complete_count or manifest["failed_tasks"] != len(tasks) - complete_count:
        raise ValueError("Task completion count mismatch")
    complete = complete_count == len(tasks)
    if manifest["status"] != ("COMPLETE" if complete else "PARTIAL"):
        raise ValueError("A partial/failed comparison cannot be marked complete")
    split_by_id = {split["split_id"]: split for split in splits}
    sample_rows = _csv(root / "sample_metadata.csv")
    samples = {row["sample_id"]: row for row in sample_rows}
    expected_samples = {key for split in splits for key in split["train_ids"] + split["eval_ids"]}
    if len(samples) != len(sample_rows) or set(samples) != expected_samples:
        raise ValueError("Sample metadata coverage/uniqueness mismatch")
    _validate_inner_protocols(manifest, dataset, splits, samples)
    _validate_diagnostics(root, manifest, tasks)
    for (_, fold), task in tasks.items():
        split = split_by_id[fold]
        if task["expected_sample_ids"] != split["eval_ids"] or task["train_ids"] != split["train_ids"]:
            raise ValueError("Task training/evaluation IDs differ from frozen split")
        n = len(task["expected_sample_ids"])
        if len(task["expected_true_labels"]) != n or len(task["expected_groups"]) != n or len(set(task["expected_sample_ids"])) != n:
            raise ValueError("Invalid task sample alignment metadata")
        if set(task["expected_true_labels"]) - set(classes):
            raise ValueError("Unknown evaluation class")
        if task["expected_true_labels"] != [samples[key]["class_key"] for key in task["expected_sample_ids"]] or task["expected_groups"] != [samples[key]["group_id"] for key in task["expected_sample_ids"]]:
            raise ValueError("Task evaluation identity differs from sample metadata")
        if task["status"] in ("COMPLETE", "NUMERICAL_BLOCKED"):
            actual_env = task["environment"]
            if actual_env["packages"] != environment["packages"] or actual_env["python"] != environment["python"] or actual_env["parallelism"] != environment["parallelism"] or any(pool["num_threads"] != 1 for pool in actual_env["threadpools"]):
                raise ValueError("Task numerical environment/thread identity mismatch")
    predictions = _csv(root / "predictions.csv")
    prediction_map = {}
    for row in predictions:
        key = (row["method"], row["fold"], row["sample_id"], row["evaluation_role"])
        if key in prediction_map:
            raise ValueError(f"Duplicate prediction key: {key}")
        if row["evaluation_stage"] != stage or row["evaluation_role"] != role:
            raise ValueError("Prediction evaluation purpose mismatch")
        prediction_map[key] = row
    accepted_status = ("COMPLETE", "NUMERICAL_BLOCKED")
    expected_prediction_keys = {(method, fold, sample, role) for (method, fold), task in tasks.items()
                                if task["status"] in accepted_status for sample in task["expected_sample_ids"]}
    if set(prediction_map) != expected_prediction_keys:
        raise ValueError("Prediction coverage mismatch: missing or unexpected method/fold/sample/purpose rows")
    mappings = manifest["score_arrays"]
    if _json(root / "scores_rows.json") != mappings:
        raise ValueError("Score row mapping differs from manifest")
    score_tasks = {(row["method"], row["fold"]) for row in mappings}
    if len(score_tasks) != len(mappings) or score_tasks != {key for key, task in tasks.items() if task["status"] in accepted_status}:
        raise ValueError("Score method/fold coverage mismatch")
    metric_rows = _csv(root / "metrics.csv")
    metric_map = {(row["method"], row["fold"]): row for row in metric_rows}
    if len(metric_map) != len(metric_rows) or set(metric_map) != score_tasks:
        raise ValueError("Metrics method/fold coverage mismatch")
    per_rows = _csv(root / "per_class_metrics.csv")
    per_map = {(row["method"], row["fold"], row["class_key"]): row for row in per_rows}
    if len(per_map) != len(per_rows) or set(per_map) != {(method, fold, label) for method, fold in score_tasks for label in classes}:
        raise ValueError("Per-class metric coverage mismatch")
    confusion = _json(root / "confusion_matrices.json")
    computed_metrics = {}
    with np.load(root / "scores.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {row["array_key"] for row in mappings}:
            raise ValueError("Score array inventory mismatch")
        for mapping in mappings:
            key = (mapping["method"], mapping["fold"])
            task = tasks[key]
            scores = arrays[mapping["array_key"]]
            if scores.shape != (len(mapping["sample_ids"]), len(classes)) or set(mapping["classes"]) != set(classes) or len(mapping["classes"]) != len(classes):
                raise ValueError("Score class/row shape mismatch")
            if len(set(mapping["sample_ids"])) != len(mapping["sample_ids"]) or set(mapping["sample_ids"]) != set(task["expected_sample_ids"]):
                raise ValueError("Score sample coverage mismatch")
            if mapping["evaluation_role"] != role:
                raise ValueError("Score evaluation role mismatch")
            model_dir = root / task["model_path"]
            if not model_dir.resolve().is_relative_to(root):
                raise ValueError("Model path escapes return directory")
            model_document = _json(model_dir / "model.json")
            if key[0] == "tensor_abess_ovr_ls" and (model_document["code_identity"] != manifest["code_identity"] or model_document["profile"] != profile):
                raise ValueError("Tensor saved core/profile/code identity mismatch")
            if model_document["training_sample_ids"] != task["train_ids"] or model_document["training_class_labels"] != [samples[key]["class_key"] for key in task["train_ids"]]:
                raise ValueError("Saved model training identity mismatch")
            expected_priority = tie_priority(model_document["training_class_labels"], model_document["classes"])
            with np.load(model_dir / "model.npz", allow_pickle=False) as saved_model:
                saved_classes = saved_model["classes"].tolist()
                saved_priority = saved_model["tie_priority"].tolist()
            priority_labels = [saved_classes[i] for i in saved_priority]
            if priority_labels != [model_document["classes"][i] for i in expected_priority] or priority_labels != [task["classes"][i] for i in task["tie_priority"]] or priority_labels != [mapping["classes"][i] for i in mapping["tie_priority"]]:
                raise ValueError("Tie priority differs from saved training frequency/first-occurrence rule")
            wanted_kind = "class_indicator_original_scale" if key[0] in ("majority", "ridge_ovr", "tensor_abess_ovr_ls") else "multinomial_decision_logits"
            if mapping["score_kind"] != wanted_kind:
                raise ValueError("Decision scores must not be relabelled as probabilities")
            labels, tie_count = predict_scores(scores, mapping["classes"], mapping["tie_priority"])
            if tie_count != task["tie_count"]:
                raise ValueError("Tie count mismatch")
            expected = {sample: (truth, group) for sample, truth, group in zip(task["expected_sample_ids"], task["expected_true_labels"], task["expected_groups"])}
            truth, prediction = [], []
            for index, sample in enumerate(mapping["sample_ids"]):
                row = prediction_map[(*key, sample, role)]
                if row["true_class"] != expected[sample][0] or row["group_id"] != expected[sample][1] or row["predicted_class"] != labels[index]:
                    raise ValueError(f"Score/prediction/identity mismatch: {key} {sample}")
                truth.append(row["true_class"])
                prediction.append(row["predicted_class"])
            metrics = classification_metrics(truth, prediction, classes)
            computed_metrics[key] = metrics
            metric_row = metric_map[key]
            for field in ("n", "macro_f1", "accuracy", "balanced_accuracy"):
                _equal_float(metrics[field], metric_row[field], f"{key}/{field}")
                _equal_float(metrics[field], task["metrics"][field], f"manifest/{key}/{field}")
            if metric_row["complete_class_coverage"] != str(metrics["complete_class_coverage"]) or metric_row["status"] != task["status"]:
                raise ValueError("Coverage/status metric mismatch")
            _equal_float(tie_count, metric_row["tie_count"], f"{key}/tie_count")
            for field in ("training_seconds", "prediction_seconds"):
                if float(metric_row[field]) < 0:
                    raise ValueError("Negative runtime")
                _equal_float(task[field], metric_row[field], f"{key}/{field}")
            cm = confusion[mapping["array_key"]]
            if cm["matrix"] != metrics["confusion_matrix"] or cm["classes"] != classes or cm["never_predicted"] != metrics["never_predicted"]:
                raise ValueError("Confusion matrix mismatch")
            for item in metrics["per_class"]:
                recorded = per_map[(*key, item["class_key"])]
                for field in ("precision", "recall", "f1", "true_count", "predicted_count", "true_positive_count"):
                    if item[field] is None:
                        if recorded[field] not in ("", "NA"):
                            raise ValueError("Absent class recall must be NA")
                    else:
                        _equal_float(item[field], recorded[field], f"{key}/{item['class_key']}/{field}")
    supports = _jsonl(root / "supports.jsonl")
    support_map = {(row["method"], row["fold"], row["class_key"]): row for row in supports}
    expected_support = {(method, fold, label) for method, fold in score_tasks if method == "tensor_abess_ovr_ls" for label in classes}
    if len(support_map) != len(supports) or set(support_map) != expected_support:
        raise ValueError("Tensor support class/fold coverage mismatch")
    for row in supports:
        shape = row["axis_shape"]
        selected = row["supports"]
        if shape != [len(axis["coordinates"]) for axis in dataset["axes"]] or len(selected) != 3 or row["q"] != int(np.prod([len(values) for values in selected])):
            raise ValueError("Support rectangular q product mismatch")
        if any(len(values) != len(set(values)) or any(not isinstance(i, int) or i < 0 or i >= n for i in values) for values, n in zip(selected, shape)):
            raise ValueError("Support coordinate out of range or duplicated")
        model_dir = root / tasks[(row["method"], row["fold"])]["model_path"]
        if not model_dir.resolve().is_relative_to(root):
            raise ValueError("Model path escapes return directory")
        with np.load(model_dir / "model.npz", allow_pickle=False) as model:
            column = model["classes"].tolist().index(row["class_key"])
            if _json(model_dir / "model.json")["supports"][column] != selected:
                raise ValueError("Support differs from saved core state")
            coefficient = model["coef"][column]
            if list(coefficient.shape) != shape or not np.isfinite(coefficient).all() or int(np.sum(np.abs(coefficient) > 1e-8)) != row["actual_nonzero_1e8"]:
                raise ValueError("Saved coefficient/nonzero/support mismatch")
            support_mask = np.zeros(shape, dtype=bool)
            support_mask[np.ix_(*[np.asarray(values, dtype=int) for values in selected])] = True
            if np.any(coefficient[~support_mask] != 0):
                raise ValueError("Nonzero coefficient outside rectangular support")
    pooled = _json(root / "pooled_oof_metrics.json")
    expected_pooled = {method for method in methods if all(task["status"] == "COMPLETE" for (m, _), task in tasks.items() if m == method)
                       and len({row["sample_id"] for row in predictions if row["method"] == method}) == len([row for row in predictions if row["method"] == method])}
    if set(pooled) != expected_pooled:
        raise ValueError("Missing/unexpected pooled OOF method metrics")
    for method, metrics in pooled.items():
        rows = [row for row in predictions if row["method"] == method]
        if len({row["sample_id"] for row in rows}) != len(rows) or any(task["status"] != "COMPLETE" for (m, _), task in tasks.items() if m == method):
            raise ValueError("Pooled OOF metric requires complete, nonduplicated evaluation predictions")
        recomputed = classification_metrics([row["true_class"] for row in rows], [row["predicted_class"] for row in rows], classes)
        if recomputed != metrics:
            raise ValueError("Pooled OOF metric mismatch")
    return {"status": "PASS", "complete": complete, "run_id": manifest["run_id"], "expected_tasks": len(tasks),
            "completed_tasks": complete_count, "failed_tasks": len(tasks) - complete_count,
            "prediction_rows": len(predictions), "formal_test_evaluated": stage == "final", "files_verified": len(hashes)}


def pack_return(run_dir, output_path):
    root = Path(run_dir).resolve()
    target = Path(output_path).resolve()
    if target.is_relative_to(root):
        raise ValueError("Return ZIP must be outside the run directory")
    if target.exists():
        raise FileExistsError(f"Refusing overwrite: {target}")
    validate_return(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(root).as_posix())
    (target.parent / (target.name + ".sha256")).write_text(_sha256(target) + "  " + target.name + "\n", encoding="utf-8")
    return target
