"""Deterministic contract checks; these fixtures are not performance evidence."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import csv
import shutil
import tempfile
import multiprocessing
import time
import unittest

import numpy as np


def module(test, name):
    spec = importlib.util.find_spec("tensor_abess_ovr." + name)
    test.assertIsNotNone(spec, f"Required {name} implementation is missing")
    return importlib.import_module("tensor_abess_ovr." + name)


class MetricsTests(unittest.TestCase):
    def test_ties_use_training_frequency_then_first_occurrence(self):
        m = module(self, "metrics")
        classes = ["z", "a", "q"]
        priority = m.tie_priority(["a", "z", "q", "q"], classes)
        self.assertEqual(priority.tolist(), [2, 1, 0])
        scores = np.array([[1., 1., 1.], [2., 2., 0.], [0., 0., 3.]])
        labels, ties = m.predict_scores(scores, classes, priority)
        self.assertEqual(labels.tolist(), ["q", "a", "q"])
        self.assertEqual(ties, 2)
        perm = np.array([2, 0, 1])
        remapped = np.array([list(perm).index(int(p)) for p in priority])
        relabelled, _ = m.predict_scores(scores[:, perm], np.array(classes)[perm], remapped)
        np.testing.assert_array_equal(labels, relabelled)

    def test_missing_class_is_descriptive_and_balanced_accuracy_uses_present(self):
        m = module(self, "metrics")
        out = m.classification_metrics(["a", "b"], ["a", "a"], ["a", "b", "c"])
        self.assertAlmostEqual(out["macro_f1"], (2 / 3) / 3)
        self.assertAlmostEqual(out["balanced_accuracy"], .5)
        self.assertFalse(out["complete_class_coverage"])
        self.assertIsNone(out["per_class"][2]["recall"])
        self.assertEqual(out["never_predicted"], ["b", "c"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            m.classification_metrics(["x"], ["a"], ["a", "b"])

    def test_nonfinite_scores_are_rejected(self):
        m = module(self, "metrics")
        with self.assertRaisesRegex(ValueError, "finite"):
            m.predict_scores([[np.nan, 0.]], ["a", "b"], [0, 1])

    def test_tie_tolerance_boundary_matches_model_prediction_layer(self):
        m = module(self, "metrics")
        from tensor_abess_ovr.model import decide_scores
        top = np.float64(1.12)
        tolerance = 64 * np.finfo(np.float64).eps * top
        scores = np.array([[top, top - tolerance]])
        expected, expected_ties = decide_scores(scores, ["top", "lower"], [1, 0])
        actual, actual_ties = m.predict_scores(scores, ["top", "lower"], [1, 0])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_ties, expected_ties)


class BaselineTests(unittest.TestCase):
    def test_seed_is_stable_and_regularization_ties_are_explicit(self):
        b = module(self, "baselines")
        self.assertEqual(b.stable_seed(31, "ridge", "fold0", "1"), b.stable_seed(31, "ridge", "fold0", "1"))
        self.assertNotEqual(b.stable_seed(31, "ridge", "fold0", "1"), b.stable_seed(31, "ridge", "fold1", "1"))
        self.assertEqual(b.select_parameter("ridge_ovr", [1., 10., .1], [.5, .5 - 1e-13, .4]), 10.)
        self.assertEqual(b.select_parameter("multinomial_l1", [1., 10., .1], [.5, .4, .5 - 1e-13]), .1)

    def test_group_folds_are_isolated_and_cover_classes(self):
        b = module(self, "baselines")
        y = np.tile(np.array(["a", "b", "c"]), 6)
        groups = np.repeat(np.arange(6), 3).astype(str)
        folds, info = b.make_inner_folds(y, ["a", "b", "c"], "group", 31, groups=groups)
        self.assertEqual(len(folds), 3)
        self.assertEqual(info["actual_folds"], 3)
        for train, val in folds:
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertEqual(set(y[train]), set(y))
        with self.assertRaisesRegex(ValueError, "class|fold|cover"):
            b.make_inner_folds(["a", "b", "c"], ["a", "b", "c"], "group", 31, groups=["1", "2", "3"])

    def test_forward_folds_use_only_past(self):
        b = module(self, "baselines")
        y = np.tile(np.array(["a", "b", "c"]), 8)
        times = np.arange(len(y))
        folds, info = b.make_inner_folds(y, ["a", "b", "c"], "time", 31, times=times)
        self.assertEqual(info["actual_folds"], 3)
        for train, val in folds:
            self.assertLess(max(times[train]), min(times[val]))


class RunReturnTests(unittest.TestCase):
    def test_fixture_all_methods_and_tampered_return(self):
        r = module(self, "runner")
        v = module(self, "returns")
        fixture = Path(importlib.import_module("tensor_abess_ovr").__file__).parent / "fixtures" / "tiny"
        with tempfile.TemporaryDirectory(prefix="task19_return_") as temp:
            root = Path(temp)
            result = r.run(fixture, fixture / "protocol.json", root / "run")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed_tasks"], 5)
            checked = v.validate_return(root / "run")
            self.assertTrue(checked["complete"])
            self.assertFalse(checked["formal_test_evaluated"])
            self.assertTrue(Path(v.pack_return(root / "run", root / "return.zip")).is_file())
            with self.assertRaisesRegex((ValueError, FileExistsError), "exist|overwrite"):
                r.run(fixture, fixture / "protocol.json", root / "run")
            damaged = root / "damaged"
            shutil.copytree(root / "run", damaged)
            with (damaged / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            with (damaged / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaisesRegex(ValueError, "SHA256|hash"):
                v.validate_return(damaged)
            v.write_checksums(damaged)
            with self.assertRaisesRegex(ValueError, "coverage|missing|prediction"):
                v.validate_return(damaged)
            for name, filename, contents, error in [
                ("diagnostics", "fit_diagnostics.jsonl", "", "diagnostic"),
                ("pooled", "pooled_oof_metrics.json", "{}", "pooled|OOF"),
            ]:
                altered = root / name
                shutil.copytree(root / "run", altered)
                (altered / filename).write_text(contents, encoding="utf-8")
                v.write_checksums(altered)
                with self.assertRaisesRegex(ValueError, error):
                    v.validate_return(altered)
            # Class columns and CSV rows are explicitly keyed, never position-joined.
            reordered = root / "reordered"
            shutil.copytree(root / "run", reordered)
            manifest = json.loads((reordered / "manifest.json").read_text(encoding="utf-8"))
            mapping = manifest["score_arrays"][0]
            permutation = [2, 0, 1]
            original_priority_labels = [mapping["classes"][i] for i in mapping["tie_priority"]]
            mapping["classes"] = [mapping["classes"][i] for i in permutation]
            mapping["tie_priority"] = [mapping["classes"].index(label) for label in original_priority_labels]
            with np.load(reordered / "scores.npz", allow_pickle=False) as stored:
                matrices = {key: stored[key].copy() for key in stored.files}
            matrices[mapping["array_key"]] = matrices[mapping["array_key"]][:, permutation]
            np.savez_compressed(reordered / "scores.npz", **matrices)
            (reordered / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (reordered / "scores_rows.json").write_text(json.dumps(manifest["score_arrays"]), encoding="utf-8")
            with (reordered / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                fields, shuffled = reader.fieldnames, list(reader)[::-1]
            with (reordered / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(shuffled)
            v.write_checksums(reordered)
            self.assertTrue(v.validate_return(reordered)["complete"])

    def test_exhausted_dataset_budget_is_partial_and_children_are_joined(self):
        r = module(self, "runner")
        fixture = Path(importlib.import_module("tensor_abess_ovr").__file__).parent / "fixtures" / "tiny"
        config = json.loads((fixture / "protocol.json").read_text(encoding="utf-8"))
        config["resource_budget"]["total_seconds"] = .00001
        with tempfile.TemporaryDirectory(prefix="task19_budget_") as temp:
            result = r.run(fixture, config, Path(temp) / "run")
            self.assertEqual(result["status"], "PARTIAL")
            self.assertEqual(result["failed_tasks"], 5)
            self.assertEqual(result["completed_tasks"], 0)
            manifest = json.loads((Path(temp) / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(task["status"] == "RESOURCE_BLOCKED" for task in manifest["tasks"]))
            self.assertFalse(multiprocessing.active_children())


if __name__ == "__main__":
    unittest.main()
