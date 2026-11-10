"""Deterministic data-contract and HAR mapping checks, not performance evidence."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DataChecks(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module("tensor_abess_ovr." + name)
        except ModuleNotFoundError:
            self.fail("Required data-contract implementation is missing: " + name)

    def fixture(self, root):
        X = np.arange(12 * 2 * 3 * 4, dtype=np.float64).reshape(12, 2, 3, 4)
        np.save(root / "X.npy", X, allow_pickle=False)
        # Deliberately different physical CSV ordering; row_index is authoritative.
        rows = [{"row_index": i, "sample_id": f"s{i}", "class_key": f"k{i % 3}",
                 "group_id": f"g{i // 3}"} for i in range(12)]
        with (root / "samples.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[::-1])
        metadata = {"schema_version": "1.0", "name": "中文 fixed fixture", "source": "deterministic",
                    "axes": [{"name": f"axis{i}", "coordinates": [f"v{j}" for j in range(p)]}
                             for i, p in enumerate((2, 3, 4))],
                    "axis_order": ["axis0", "axis1", "axis2"], "x_shape": list(X.shape),
                    "classes": ["k0", "k1", "k2"], "class_labels": {f"k{i}": str(i) for i in range(3)},
                    "independent_unit": "subject", "dependency": "group", "representation_version": "fixture-v1",
                    "source_files": {}, "processed_files": {"X.npy": digest(root / "X.npy"),
                                                              "samples.csv": digest(root / "samples.csv")}}
        splits = {"schema_version": "1.0", "seed": 1, "rule": "fixed groups",
                  "splits": [{"split_id": "dev", "purpose": "dev", "evaluation_role": "validation",
                              "train_ids": [f"s{i}" for i in range(6)],
                              "eval_ids": [f"s{i}" for i in range(6, 9)]},
                             {"split_id": "final", "purpose": "final", "evaluation_role": "test",
                              "train_ids": [f"s{i}" for i in range(9)],
                              "eval_ids": [f"s{i}" for i in range(9, 12)]}]}
        (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
        config = {"evaluation_stage": "dev", "active_split_ids": ["dev"],
                  "preprocessing": "train_coordinate_zscore_v1", "core_profile": "v5.2-search-final-original-msic",
                  "primary_metric": "macro_f1", "seed": 1,
                  "resource_budget": {"per_class_seconds": 900, "total_seconds": 7200}}
        return metadata, splits, config

    def changed(self, root, metadata, samples=False, X=False):
        for name, changed in (("samples.csv", samples), ("X.npy", X)):
            if changed:
                metadata["processed_files"][name] = digest(root / name)
        (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_row_indices_align_labels_and_unicode_paths(self):
        with tempfile.TemporaryDirectory(prefix="中文 data ") as t:
            root = Path(t)
            _, _, config = self.fixture(root)
            ds = self.module("data").load_dataset(root, config)
            self.assertEqual(ds["sample_ids"].tolist(), [f"s{i}" for i in range(12)])
            self.assertEqual(ds["y"].tolist(), [f"k{i % 3}" for i in range(12)])
            np.testing.assert_array_equal(ds["X"][1].ravel(order="C"), np.arange(24, 48))
            self.assertEqual(ds["selected_splits"][0]["split_id"], "dev")
            ds["X"]._mmap.close()

    def test_missing_sample_id_is_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); _, splits, config = self.fixture(root)
            splits["splits"][0]["eval_ids"][0] = "not-in-samples"
            (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not-in-samples"):
                self.module("data").load_dataset(root, config)

    def test_development_cannot_select_test(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); _, _, config = self.fixture(root)
            config["active_split_ids"] = ["final"]
            with self.assertRaisesRegex(ValueError, "stage|purpose|test"):
                self.module("data").load_dataset(root, config)

    def test_overlap_and_group_leakage_rejected(self):
        for problem in ("same_sample", "same_group"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as t:
                root = Path(t); _, splits, config = self.fixture(root)
                splits["splits"][0]["eval_ids"][0] = "s0" if problem == "same_sample" else "s2"
                if problem == "same_group":
                    splits["splits"][0]["train_ids"] = ["s0", "s1", "s3", "s4", "s5"]
                (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "overlap|group"):
                    self.module("data").load_dataset(root, config)

    def test_missing_training_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); _, splits, config = self.fixture(root)
            splits["splits"][0]["train_ids"] = ["s0", "s1", "s3", "s4"]
            (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class|k2"):
                self.module("data").load_dataset(root, config)

    def test_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); _, _, config = self.fixture(root)
            with (root / "samples.csv").open("a", encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "hash|SHA256|fingerprint"):
                self.module("data").load_dataset(root, config)

    def test_axis_metadata_expected_order_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); metadata, _, config = self.fixture(root)
            config["expected_axes"] = json.loads(json.dumps(metadata["axes"]))
            config["expected_axes"][2]["coordinates"] = list(reversed(config["expected_axes"][2]["coordinates"]))
            with self.assertRaisesRegex(ValueError, "axes|coordinate"):
                self.module("data").load_dataset(root, config)

    def test_processed_paths_cannot_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); metadata, _, config = self.fixture(root)
            metadata["processed_files"]["C:relative-drive-file"] = "0" * 64
            self.changed(root, metadata)
            with self.assertRaisesRegex(ValueError, "relative path"):
                self.module("data").load_dataset(root, config)

    def test_nonfinite_and_axis_order_rejected(self):
        for problem in ("nan", "axis"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as t:
                root = Path(t); metadata, _, config = self.fixture(root)
                if problem == "nan":
                    X = np.load(root / "X.npy", allow_pickle=False); X[0, 0, 0, 0] = np.nan
                    np.save(root / "X.npy", X, allow_pickle=False)
                    self.changed(root, metadata, X=True)
                else:
                    metadata["axis_order"] = ["axis1", "axis0", "axis2"]
                    self.changed(root, metadata)
                with self.assertRaisesRegex(ValueError, "finite|axis|axes"):
                    self.module("data").load_dataset(root, config)

    def test_duplicate_ids_and_unknown_labels_rejected(self):
        for problem in ("duplicate", "unknown"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as t:
                root = Path(t); metadata, _, config = self.fixture(root)
                path = root / "samples.csv"
                text = path.read_text(encoding="utf-8")
                text = text.replace("s11,k2", "s10,k2") if problem == "duplicate" else text.replace("s11,k2", "s11,UNKNOWN")
                path.write_text(text, encoding="utf-8")
                self.changed(root, metadata, samples=True)
                with self.assertRaisesRegex(ValueError, "duplicate|UNKNOWN|label|class"):
                    self.module("data").load_dataset(root, config)

    def test_missing_required_csv_cell_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); metadata, _, config = self.fixture(root)
            path = root / "samples.csv"
            # A short row has a missing group cell, rather than an empty string.
            path.write_text(path.read_text(encoding="utf-8").replace("s11,k2,g3", "s11,k2"), encoding="utf-8")
            self.changed(root, metadata, samples=True)
            with self.assertRaisesRegex(ValueError, "group|cell|column"):
                self.module("data").load_dataset(root, config)

    def test_time_boundary_is_used_without_forcing_groups(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); metadata, splits, config = self.fixture(root)
            metadata["dependency"] = "time"
            with (root / "samples.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["time"] = str(int(row["row_index"]))
            with (root / "samples.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            self.changed(root, metadata, samples=True)
            self.module("data").load_dataset(root, config)
            splits["splits"][0]["eval_ids"] = ["s3", "s4", "s5"]
            splits["splits"][0]["train_ids"] = ["s6", "s7", "s8"]
            (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "time|forward"):
                self.module("data").load_dataset(root, config)

    def test_har_summary_matches_independent_formula(self):
        raw = np.arange(2 * 3 * 3 * 128, dtype=np.float64).reshape(2, 3, 3, 128) / 137
        out = self.module("har").summarize_signals(raw)
        self.assertEqual(out.shape, (2, 3, 3, 17))
        series = raw[1, 2, 0]
        F = np.fft.rfft((series - series.mean()) * np.hanning(128), norm="ortho")
        expected = np.array([series.mean()] + [np.log1p(sum(abs(F[1 + 4*b:5 + 4*b])**2)) for b in range(16)])
        np.testing.assert_allclose(out[1, 2, 0], expected, rtol=1e-13, atol=1e-13)
        constant = self.module("har").summarize_signals(np.full((1, 3, 3, 128), -3.0))
        np.testing.assert_array_equal(constant[..., 0], -3)
        np.testing.assert_array_equal(constant[..., 1:], 0)

    def test_official_latin1_readme_and_channel_mapping(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            archive = root / "official-shaped-fixture.zip"
            with zipfile.ZipFile(archive, "w") as source:
                prefix = "UCI HAR Dataset/"
                source.writestr(prefix + "README.txt", b"Universitat Polit\xe8cnica; 50Hz, 128 points, 50% overlap")
                source.writestr(prefix + "features_info.txt", "Features metadata")
                source.writestr(prefix + "activity_labels.txt", "\n".join(f"{i} CLASS_{i}" for i in range(1, 7)))
                for partition, subjects in (("train", range(1, 7)), ("test", [7])):
                    groups = [g for g in subjects for _ in range(6)]
                    labels = list(range(1, 7)) * len(subjects)
                    source.writestr(prefix + f"{partition}/subject_{partition}.txt", "\n".join(map(str, groups)))
                    source.writestr(prefix + f"{partition}/y_{partition}.txt", "\n".join(map(str, labels)))
                    for a, signal_type in enumerate(("total_acc", "body_acc", "body_gyro")):
                        for b, direction in enumerate(("x", "y", "z")):
                            row = " ".join(str(a * 1000 + b * 100 + j) for j in range(128))
                            source.writestr(prefix + f"{partition}/Inertial Signals/{signal_type}_{direction}_{partition}.txt",
                                            "\n".join([row] * len(labels)))
            output = root / "converted"
            report = self.module("har").prepare_har(archive, output)
            self.assertFalse(report["formal_test_evaluated"])
            self.assertIn("Politècnica", (output / "provenance/README.txt").read_text(encoding="utf-8"))
            raw = np.load(output / "X_raw_mapping.npy", allow_pickle=False)
            self.assertEqual(raw[0, 2, 1, 17], 2117)
            self.assertEqual(raw[37, 0, 2, 127], 327)
            ds = self.module("data").load_dataset(output, output / "protocol.json")
            self.assertTrue(ds["selected_splits"][0]["complete_class_coverage"])
            ds["X"]._mmap.close()


if __name__ == "__main__":
    unittest.main()
