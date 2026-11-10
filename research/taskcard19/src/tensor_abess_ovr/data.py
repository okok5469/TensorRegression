"""Strict, row-indexed interchange contract for three-axis classification data."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def _require(record: dict, fields: tuple[str, ...], location: str) -> None:
    missing = [key for key in fields if key not in record]
    if missing:
        raise ValueError(f"{location}: missing required fields {missing}")


def _unique_strings(values: Any, location: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{location}: expected a list of nonempty string identities")
    if nonempty and not values:
        raise ValueError(f"{location}: empty identities")
    if len(set(values)) != len(values):
        raise ValueError(f"{location}: duplicate identities")
    return values


def _parse_time(value: str, row_index: int) -> float:
    try:
        result = float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.utcoffset() is None:
                raise ValueError("timezone required for ISO timestamps")
            result = parsed.timestamp()
        except ValueError as error:
            raise ValueError(f"samples.csv row_index={row_index}: invalid time {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"samples.csv row_index={row_index}: nonfinite time")
    return result


def _validate_pair(split: dict, location: str, id_index: dict[str, int], y: np.ndarray,
                   groups: np.ndarray | None, times: np.ndarray | None,
                   classes: list[str], dependency: str,
                   allowed_ids: set[str] | None = None) -> None:
    _require(split, ("train_ids", "eval_ids"), location)
    train = _unique_strings(split["train_ids"], location + ".train_ids")
    evaluate = _unique_strings(split["eval_ids"], location + ".eval_ids")
    selected = set(train) | set(evaluate)
    unknown = selected - set(id_index)
    if unknown:
        raise ValueError(f"{location}: unknown sample_id {sorted(unknown)}")
    if allowed_ids is not None and not selected <= allowed_ids:
        raise ValueError(f"{location}: inner fold contains samples outside outer training IDs")
    if set(train) & set(evaluate):
        raise ValueError(f"{location}: train/evaluation sample overlap")
    ti = np.array([id_index[s] for s in train], dtype=np.int64)
    ei = np.array([id_index[s] for s in evaluate], dtype=np.int64)
    missing = set(classes) - set(y[ti].tolist())
    if missing:
        raise ValueError(f"{location}: missing training classes {sorted(missing)}")
    if dependency == "group" and set(groups[ti]) & set(groups[ei]):
        raise ValueError(f"{location}: train/evaluation group overlap")
    if dependency == "time" and not np.max(times[ti]) < np.min(times[ei]):
        raise ValueError(f"{location}: forward time boundary violated")
    split["evaluation_class_coverage"] = [c for c in classes if c in set(y[ei])]
    split["complete_class_coverage"] = set(y[ei]) == set(classes)


def load_dataset(data_path: str | Path, config: dict | str | Path) -> dict:
    """Validate all declared data and splits; select only explicitly active roles.

    CSV rows are sorted by explicit ``row_index``. All returned identity/label
    arrays therefore refer to X's first dimension, independent of CSV ordering.
    No observations are removed or repaired. Hashes of every consumed contract
    file contribute to ``identity``; raw-source hashes are retained as provenance.
    """
    root = Path(data_path)
    protocol = dict(config) if isinstance(config, dict) else _json(Path(config))
    _require(protocol, ("evaluation_stage", "active_split_ids", "preprocessing", "core_profile",
                        "primary_metric", "seed", "resource_budget"), "config")
    stage = protocol["evaluation_stage"]
    if stage not in ("dev", "final"):
        raise ValueError("config.evaluation_stage must be dev or final")
    if protocol["primary_metric"] != "macro_f1":
        raise ValueError("config.primary_metric must be macro_f1")
    if protocol["preprocessing"] != "train_coordinate_zscore_v1":
        raise ValueError("config.preprocessing must be train_coordinate_zscore_v1")
    if protocol["core_profile"] != "v5.2-search-final-original-msic":
        raise ValueError("config.core_profile does not identify the fixed released profile")
    active = _unique_strings(protocol["active_split_ids"], "config.active_split_ids")
    budget = protocol["resource_budget"]
    if not isinstance(budget, dict):
        raise ValueError("config.resource_budget must be an object")
    _require(budget, ("per_class_seconds", "total_seconds"), "config.resource_budget")
    for key in ("per_class_seconds", "total_seconds"):
        value = budget[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"config.resource_budget.{key} must be finite and positive")
    if isinstance(protocol["seed"], bool) or not isinstance(protocol["seed"], int):
        raise ValueError("config.seed must be an integer")

    metadata = _json(root / "dataset.json")
    _require(metadata, ("schema_version", "name", "source", "axes", "axis_order", "x_shape",
                        "classes", "class_labels", "independent_unit", "dependency",
                        "representation_version", "source_files", "processed_files"), "dataset.json")
    if metadata["schema_version"] != "1.0":
        raise ValueError("dataset.json: unsupported schema_version")
    classes = _unique_strings(metadata["classes"], "dataset.json.classes")
    if len(classes) < 2 or set(metadata["class_labels"]) != set(classes):
        raise ValueError("dataset.json.class_labels must describe all classes (K>=2)")
    if any(not isinstance(metadata[key], str) or not metadata[key].strip()
           for key in ("name", "source", "independent_unit", "representation_version")):
        raise ValueError("dataset.json: provenance and independent-unit fields must be nonempty strings")
    dependency = metadata["dependency"]
    if dependency not in ("iid", "group", "time"):
        raise ValueError("dataset.json.dependency must be iid, group, or time")
    fingerprints = metadata["processed_files"]
    if not isinstance(fingerprints, dict) or not {"X.npy", "samples.csv"} <= set(fingerprints):
        raise ValueError("dataset.json.processed_files: require X.npy and samples.csv fingerprints")
    for name, expected in fingerprints.items():
        relative = Path(name)
        if relative.anchor or PureWindowsPath(name).anchor or ".." in relative.parts:
            raise ValueError(f"dataset.json.processed_files: invalid relative path {name!r}")
        candidate = root / relative
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"dataset.json.processed_files: invalid relative path {name!r}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"{name}: SHA256 fingerprint mismatch")

    X = np.load(root / "X.npy", mmap_mode="r", allow_pickle=False)
    if X.ndim != 4 or X.dtype != np.dtype("float64") or min(X.shape) <= 0:
        raise ValueError("X.npy: require a nonempty float64 (n,p1,p2,p3) numeric tensor")
    if list(X.shape) != metadata["x_shape"]:
        raise ValueError("X.npy: shape differs from dataset.json.x_shape")
    for start in range(0, len(X), 1024):
        if not np.isfinite(X[start:start + 1024]).all():
            raise ValueError(f"X.npy: nonfinite values in rows {start}:{min(start + 1024, len(X))}")
    axes = metadata["axes"]
    if not isinstance(axes, list) or len(axes) != 3:
        raise ValueError("dataset.json.axes: require exactly three ordered axes")
    for i, axis in enumerate(axes):
        if not isinstance(axis, dict):
            raise ValueError(f"dataset.json.axes[{i}]: expected an object")
        _require(axis, ("name", "coordinates"), f"dataset.json.axes[{i}]")
        coordinates = axis["coordinates"]
        if not isinstance(axis["name"], str) or not axis["name"] or not isinstance(coordinates, list):
            raise ValueError(f"dataset.json.axes[{i}]: invalid name/coordinates")
        if len(coordinates) != X.shape[i + 1] or len({json.dumps(c, sort_keys=True) for c in coordinates}) != len(coordinates):
            raise ValueError(f"dataset.json.axes[{i}]: coordinate count/order identity mismatch")
    names = [axis["name"] for axis in axes]
    if len(set(names)) != 3 or metadata["axis_order"] != names:
        raise ValueError("dataset.json.axis_order differs from declared axes")
    if "expected_axes" in protocol and protocol["expected_axes"] != axes:
        raise ValueError("config.expected_axes differs from dataset axes or coordinate order")

    with (root / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required_fields = {"row_index", "sample_id", "class_key"}
        required_fields |= {"group_id"} if dependency == "group" else set()
        required_fields |= {"time"} if dependency == "time" else set()
        if not required_fields <= set(fields) or len(set(fields)) != len(fields):
            raise ValueError(f"samples.csv: missing/duplicate columns; require {sorted(required_fields)}")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        empty = [key for key in required_fields
                 if not isinstance(row.get(key), str) or not row[key].strip()]
        if None in row or empty:
            raise ValueError(f"samples.csv line {line_number}: malformed columns or missing required cells {empty}")
    if len(rows) != len(X):
        raise ValueError(f"samples.csv: {len(rows)} rows do not match X.npy n={len(X)}")
    try:
        row_indices = [int(row["row_index"]) for row in rows]
    except (ValueError, TypeError) as error:
        raise ValueError("samples.csv: row_index must be an integer") from error
    if sorted(row_indices) != list(range(len(X))):
        raise ValueError("samples.csv: row_index must uniquely cover 0..n-1")
    rows = [row for _, row in sorted(zip(row_indices, rows), key=lambda pair: pair[0])]
    ids = _unique_strings([row["sample_id"] for row in rows], "samples.csv.sample_id")
    y_list = [row["class_key"] for row in rows]
    unknown_labels = {key for key in y_list if key not in classes}
    if unknown_labels:
        raise ValueError(f"samples.csv: unknown/missing class labels {sorted(str(v) for v in unknown_labels)}")
    groups = np.array([row.get("group_id", "") for row in rows], dtype=str) if "group_id" in fields else None
    if dependency == "group" and any(not str(group).strip() for group in groups):
        raise ValueError("samples.csv: missing group_id")
    times = np.array([_parse_time(row["time"], i) for i, row in enumerate(rows)]) if dependency == "time" else None
    y = np.array(y_list, dtype=str)
    id_index = {sample_id: i for i, sample_id in enumerate(ids)}

    split_document = _json(root / "splits.json")
    _require(split_document, ("schema_version", "splits", "seed", "rule"), "splits.json")
    if split_document["schema_version"] != "1.0" or not isinstance(split_document["splits"], list):
        raise ValueError("splits.json: unsupported schema or invalid splits")
    splits = split_document["splits"]
    split_ids = []
    for split in splits:
        _require(split, ("split_id", "purpose", "evaluation_role", "train_ids", "eval_ids"), "split")
        split_ids.append(split["split_id"])
        if split["purpose"] not in ("dev", "final"):
            raise ValueError(f"split {split['split_id']}: invalid purpose")
        expected_role = {"dev": "validation", "final": "test"}[split["purpose"]]
        if split["evaluation_role"] != expected_role:
            raise ValueError(f"split {split['split_id']}: evaluation_role conflicts with purpose")
        location = f"split {split['split_id']}"
        _validate_pair(split, location, id_index, y, groups, times, classes, dependency)
        for j, fold in enumerate(split.get("inner_folds", [])):
            _validate_pair(fold, f"{location}.inner_folds[{j}]", id_index, y, groups, times,
                           classes, dependency, allowed_ids=set(split["train_ids"]))
    _unique_strings(split_ids, "splits.json.split_id")
    heldout = {s for split in splits if split["purpose"] == "final" for s in split["eval_ids"]}
    dev_ids = {s for split in splits if split["purpose"] == "dev" for s in split["train_ids"] + split["eval_ids"]}
    if heldout & dev_ids:
        raise ValueError("splits.json: final test sample overlap with development data")
    if dependency == "group" and heldout and dev_ids:
        test_groups = {groups[id_index[s]] for s in heldout}
        dev_groups = {groups[id_index[s]] for s in dev_ids}
        if test_groups & dev_groups:
            raise ValueError("splits.json: final test group overlap with development groups")
    by_id = {split["split_id"]: split for split in splits}
    if set(active) - set(by_id):
        raise ValueError(f"config.active_split_ids: unknown splits {sorted(set(active) - set(by_id))}")
    selected = [by_id[key] for key in active]
    if any(split["purpose"] != stage for split in selected):
        raise ValueError("config evaluation_stage conflicts with selected split purpose; test is sealed in dev")
    file_hashes = {name: sha256_file(root / name) for name in ("dataset.json", "splits.json", "samples.csv", "X.npy")}
    identity = {"files": file_hashes, "representation_version": metadata["representation_version"],
                "source_files": metadata["source_files"]}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"X": X, "y": y, "sample_ids": np.array(ids, dtype=str), "group_ids": groups,
            "time_values": times, "metadata": metadata, "splits": splits, "selected_splits": selected,
            "identity": identity, "config": protocol}
