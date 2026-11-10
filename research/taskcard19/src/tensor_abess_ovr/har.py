"""UCI HAR's declared channel mapping and fixed per-window spectral summary."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .data import sha256_file


def summarize_signals(raw: np.ndarray) -> np.ndarray:
    """Map (n,signal_type,direction,128) to signed mean plus 16 rFFT energies.

    This is not a normalized PSD or a variance decomposition. There is no
    learned statistic and no label input; all later z-scoring is training-only.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[1:] != (3, 3, 128) or not np.isfinite(raw).all():
        raise ValueError("HAR raw signals must be finite (n,3,3,128)")
    means = raw.mean(axis=-1, keepdims=True)
    transformed = np.fft.rfft((raw - means) * np.hanning(128), axis=-1, norm="ortho")
    energy = np.square(np.abs(transformed[..., 1:65])).reshape(*raw.shape[:3], 16, 4).sum(axis=-1)
    result = np.concatenate((means, np.log1p(energy)), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("HAR summary produced nonfinite values")
    return result


@contextmanager
def _official_zip(path: Path):
    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        raw_members = [name for name in names if name.endswith("/train/subject_train.txt")]
        if raw_members:
            yield outer
            return
        nested = [name for name in names if name.endswith("UCI HAR Dataset.zip")]
        if len(nested) != 1:
            raise ValueError("Official HAR archive has no unique UCI HAR Dataset.zip or train/subject_train.txt")
        with zipfile.ZipFile(BytesIO(outer.read(nested[0]))) as inner:
            yield inner


def _named_rng(seed: int, name: str) -> np.random.Generator:
    token = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big"))


def _subsample(indices: np.ndarray, labels: np.ndarray, classes: list[str], limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices.copy()
    counts = np.array([np.count_nonzero(labels[indices] == key) for key in classes], dtype=np.int64)
    proportions = limit * counts / counts.sum()
    allocation = np.floor(proportions).astype(np.int64)
    if np.any(allocation < 1):
        raise ValueError("HAR proportional 400-window protocol cannot retain every class")
    order = sorted(range(len(classes)), key=lambda j: (-(proportions[j] - allocation[j]), j))
    for j in order[:limit - int(allocation.sum())]:
        allocation[j] += 1
    chosen = []
    for j, key in enumerate(classes):
        pool = indices[labels[indices] == key]
        chosen.extend(_named_rng(seed, f"train-sampling:{key}").choice(pool, int(allocation[j]), replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def prepare_har(archive: str | Path, output: str | Path, *, seed: int = 20261031) -> dict:
    """Build the one prespecified representation and frozen dev/final split.

    Reading and validating official test inputs is a provenance/mapping action;
    neither predictions nor fitted preprocessing are computed for that role.
    The output directory must not exist, protecting immutable dataset identities.
    """
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError(f"HAR output already exists: {output}")
    source_hashes = {archive.name: sha256_file(archive)}
    raw_parts, labels_parts, group_parts, row_parts, readmes = [], [], [], [], {}
    with _official_zip(archive) as source:
        members = source.namelist()
        matches = [name for name in members if name.endswith("/train/subject_train.txt")]
        if len(matches) != 1:
            raise ValueError("HAR archive has no unique official training subject table")
        prefix = matches[0].removesuffix("train/subject_train.txt")

        def read(relative: str) -> bytes:
            payload = source.read(prefix + relative)
            source_hashes[relative] = hashlib.sha256(payload).hexdigest()
            return payload

        # The official 2012 README contains Latin-1 accented institution names.
        # Declare this source encoding; exported contract documents remain UTF-8.
        for relative in ("README.txt", "features_info.txt", "activity_labels.txt"):
            readmes[relative] = read(relative).decode("latin-1")
        class_pairs = [line.split(maxsplit=1) for line in readmes["activity_labels.txt"].strip().splitlines()]
        classes = [key for key, _ in class_pairs]
        class_labels = dict(class_pairs)
        if classes != [str(i) for i in range(1, 7)]:
            raise ValueError("Official HAR class dictionary is not the expected six activities")
        for partition in ("train", "test"):
            subjects = np.loadtxt(BytesIO(read(f"{partition}/subject_{partition}.txt")), dtype=np.int64, ndmin=1)
            labels = np.loadtxt(BytesIO(read(f"{partition}/y_{partition}.txt")), dtype=np.int64, ndmin=1).astype(str)
            if len(subjects) != len(labels) or not set(labels) <= set(classes):
                raise ValueError(f"HAR {partition}: subject/activity row mismatch or unknown class")
            raw = np.empty((len(labels), 3, 3, 128), dtype=np.float64)
            for type_index, signal_type in enumerate(("total_acc", "body_acc", "body_gyro")):
                for direction_index, direction in enumerate(("x", "y", "z")):
                    relative = f"{partition}/Inertial Signals/{signal_type}_{direction}_{partition}.txt"
                    channel = np.loadtxt(BytesIO(read(relative)), dtype=np.float64, ndmin=2)
                    if channel.shape != (len(labels), 128) or not np.isfinite(channel).all():
                        raise ValueError(f"HAR {relative}: invalid shape or nonfinite measurements")
                    raw[:, type_index, direction_index] = channel
            raw_parts.append(raw)
            labels_parts.append(labels)
            group_parts.append(subjects.astype(str))
            row_parts.extend({"sample_id": f"HAR-{partition}-{i + 1:06d}", "class_key": str(labels[i]),
                              "group_id": f"subject-{subjects[i]:02d}", "official_partition": partition,
                              "official_row": i + 1} for i in range(len(labels)))

    raw = np.concatenate(raw_parts, axis=0)
    labels, groups = np.concatenate(labels_parts), np.concatenate(group_parts)
    n_train = len(labels_parts[0])
    train_subjects = sorted(set(group_parts[0]), key=int)
    if set(group_parts[0]) & set(group_parts[1]):
        raise ValueError("HAR official train/test subject overlap")
    validation_subject_count = math.ceil(0.25 * len(train_subjects))
    validation_subjects = sorted(_named_rng(seed, "subject-holdout").choice(train_subjects,
                                validation_subject_count, replace=False).tolist(), key=int)
    dev_pool = np.flatnonzero((np.arange(len(labels)) < n_train) & ~np.isin(groups, validation_subjects))
    validation = np.flatnonzero((np.arange(len(labels)) < n_train) & np.isin(groups, validation_subjects))
    chosen = _subsample(dev_pool, labels, classes, 400, seed)
    if set(labels[chosen]) != set(classes) or set(labels[validation]) != set(classes):
        raise ValueError("HAR fixed subject/sample split does not cover all six classes")
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize_signals(raw)
    np.save(output / "X.npy", summary, allow_pickle=False)
    np.save(output / "X_raw_mapping.npy", raw, allow_pickle=False)
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("row_index", "sample_id", "class_key", "group_id",
                                                    "official_partition", "official_row"))
        writer.writeheader()
        writer.writerows(dict(row_index=i, **row) for i, row in enumerate(row_parts))
    provenance = output / "provenance"
    provenance.mkdir()
    for name, text in readmes.items():
        (provenance / name).write_text(text, encoding="utf-8")
    frequency_bands = [{"feature_index": b + 1, "bins": list(range(1 + 4*b, 5 + 4*b)),
                        "frequency_hz": [(1 + 4*b) * 50 / 128, (4 + 4*b) * 50 / 128]}
                       for b in range(16)]
    axes = [{"name": "signal_type", "coordinates": ["total_acc", "body_acc", "body_gyro"]},
            {"name": "direction", "coordinates": ["x", "y", "z"]},
            {"name": "signed_mean_and_fixed_windowed_rfft_energy", "coordinates":
             ["signed_mean"] + [f"log1p_energy_bins_{1 + 4*b:02d}_{4 + 4*b:02d}" for b in range(16)]}]
    metadata = {"schema_version": "1.0", "name": "UCI HAR fixed 3x3x17 summary",
                "source": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
                "source_doi": "10.24432/C54S4K", "classes": classes, "class_labels": class_labels,
                "official_text_encoding": "ISO-8859-1; exported provenance converted to UTF-8",
                "axes": axes, "axis_order": [a["name"] for a in axes], "x_shape": list(summary.shape),
                "independent_unit": "subject", "dependency": "group", "representation_version": "har-mean-hanning-rfft16-v1",
                "representation": {"window_points": 128, "sampling_hz": 50, "official_window_overlap": 0.5,
                  "channel_mapping": "signal_type,direction,official-window-position; C order",
                  "formula": "signed mean; log1p(sum(abs(rfft((s-mean(s))*hanning(128),norm='ortho')[bins])**2))",
                  "frequency_bands": frequency_bands, "learned_statistics": "none",
                  "not_a_normalized_psd": True, "signals_are_not_independent_sensors": True},
                "source_files": source_hashes,
                "processed_files": {name: sha256_file(output / name)
                                    for name in ("X.npy", "X_raw_mapping.npy", "samples.csv")},
                "official_missing_values_declaration": "No; actual 18 channel files and labels checked finite/present",
                "formal_test_evaluated": False}
    sample_ids = [row["sample_id"] for row in row_parts]
    splits = {"schema_version": "1.0", "seed": seed,
              "rule": "ceil(25% of official training subjects) held out; named SHA256-seeded NumPy generators; proportional largest-remainder sampling, preserve source row order",
              "validation_subject_ids": [f"subject-{int(s):02d}" for s in validation_subjects],
              "splits": [{"split_id": "har_dev_subject_holdout", "purpose": "dev", "evaluation_role": "validation",
                          "train_ids": [sample_ids[i] for i in chosen], "eval_ids": [sample_ids[i] for i in validation]},
                         {"split_id": "har_official_final", "purpose": "final", "evaluation_role": "test",
                          "train_ids": sample_ids[:n_train], "eval_ids": sample_ids[n_train:]}]}
    protocol = {"evaluation_stage": "dev", "active_split_ids": ["har_dev_subject_holdout"],
                "preprocessing": "train_coordinate_zscore_v1", "core_profile": "v5.2-search-final-original-msic",
                "primary_metric": "macro_f1", "seed": seed, "expected_axes": axes,
                "resource_budget": {"per_class_seconds": 900, "total_seconds": 7200},
                "study_scope": "finite-training-sample development screening; official test sealed",
                "methods": ["majority", "ridge_ovr", "multinomial_l2", "multinomial_l1", "tensor_abess_ovr_ls"]}
    description = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_bytes": archive.stat().st_size,
                   "archive_sha256": source_hashes[archive.name], "official_train_windows": n_train,
                   "official_test_windows": len(labels_parts[1]), "official_train_subjects": len(train_subjects),
                   "official_test_subjects": len(set(group_parts[1])), "development_train_pool_windows": len(dev_pool),
                   "development_train_windows": len(chosen), "development_train_subjects": len(set(groups[chosen])),
                   "development_validation_windows": len(validation), "development_validation_subjects": len(validation_subjects),
                   "train_class_denominators": {key: int(np.count_nonzero(labels[chosen] == key)) for key in classes},
                   "validation_class_denominators": {key: int(np.count_nonzero(labels[validation] == key)) for key in classes},
                   "raw_mapping_shape": list(raw.shape), "summary_shape": list(summary.shape),
                   "formal_test_evaluated": False}
    for name, document in (("dataset.json", metadata), ("splits.json", splits), ("protocol.json", protocol),
                           ("ADAPTER_REPORT.json", description)):
        (output / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return description
