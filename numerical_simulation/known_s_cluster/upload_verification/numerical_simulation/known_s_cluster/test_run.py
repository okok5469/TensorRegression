"""Contract checks for the isolated cluster plan, execution and merge workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import importlib
import importlib.util
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from numerical_simulation.data_generation import generate_simulation_dataset


class KnownSClusterTests(unittest.TestCase):
    def api(self):
        name = "numerical_simulation.known_s_cluster.run"
        self.assertIsNotNone(importlib.util.find_spec(name), "The isolated cluster entrypoint is not implemented.")
        return importlib.import_module(name)

    def test_full_grid_preserves_all_historical_first_ten_seeds(self):
        api = self.api()
        plan = api.build_plan("full", 50, ["fixed", "adaptive", "oracle"])
        self.assertEqual(len(plan["tasks"]), 700)
        self.assertEqual([task["task_id"] for task in plan["tasks"]], list(range(700)))
        actual = {(t["experiment"], t["scenario_value"], t["replicate"]): t for t in plan["tasks"] if t["replicate"] < 10}
        self.assertEqual(len(actual), 140)
        # Independently checked against v3/v5 artifacts; no old data needs uploading.
        expected = [("baseline", "baseline", 2026)]
        expected += [("snr", value, seed) for value, seed in (("0.5", 12026), ("1", 12126), ("2", 12226), ("4", 12326))]
        expected += [("sample_size", value, seed) for value, seed in (("100", 12026), ("200", 12126), ("400", 12226))]
        expected += [("dimension", value, seed) for value, seed in (("16x12x15", 12026), ("32x20x25", 12126), ("64x30x40", 12226))]
        expected += [("correlation", value, seed) for value, seed in (("0.00", 12026), ("0.30", 12126), ("0.60", 12226))]
        for experiment, value, base_seed in expected:
            for replicate in range(10):
                task = actual[(experiment, value, replicate)]
                self.assertEqual(task["seed"], base_seed + replicate)
                self.assertEqual(task["test_seed"], base_seed + replicate + 1_000_000)

    def test_task_id_shards_are_disjoint_and_complete(self):
        api = self.api()
        plan = api.build_plan("full", 50, ["fixed"])
        shards = [set(t["task_id"] for t in plan["tasks"][offset::4]) for offset in range(4)]
        self.assertEqual(set.union(*shards), set(range(700)))
        self.assertEqual(sum(len(shard) for shard in shards), 700)

    def test_status_distinguishes_in_progress_commit_from_unlocked_partial(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed"])
            path, arrays, lock = api.task_paths(output, 0)
            lock.write_text("pid=active-worker\n", encoding="utf-8")
            arrays.write_bytes(b"worker has not committed JSON yet")
            state = api.status(output)
            self.assertEqual(state["running_or_locked"], [0])
            self.assertEqual(state["completed"], 0)
            self.assertIn(0, state["missing"])
            self.assertNotIn(0, state["rerun_task_ids"])
            with self.assertRaisesRegex(ValueError, "incomplete|locked"):
                api.merge(output)
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "Partial checkpoint"):
                api.status(output)
            self.assertFalse(path.exists())

    def test_status_rejects_out_of_plan_npz_and_lock_files(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed"])
            for suffix in (".npz", ".lock"):
                extra = output / "tasks" / ("task_000999" + suffix)
                extra.write_bytes(b"unexpected task")
                with self.assertRaisesRegex(ValueError, "outside plan"):
                    api.status(output)
                extra.unlink()

    def test_fixed_solver_receives_only_sizes_and_exposes_real_diagnostics(self):
        api = self.api()
        task = api.build_plan("quick", 1, ["fixed"])["tasks"][0]
        config = api.config_from_dict(task["config"])
        dataset = generate_simulation_dataset(config, seed=task["seed"])
        estimate, trace = api.fit_tensor(dataset.X_train, dataset.y_train, config, mode="fixed", seed=task["seed"])
        self.assertEqual((estimate.s1, estimate.s2, estimate.s3), config.sparsity)
        self.assertEqual(tuple(len(S) for S in trace["pipeline_init_supports"]), config.sparsity)
        self.assertTrue(trace["loss_history"])
        self.assertEqual(set(__import__("inspect").signature(api.fit_tensor).parameters), {"X", "y", "config", "mode", "seed"})

    def test_exclusive_plan_result_and_resume_skip_existing_fit(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed", "oracle"])
            with self.assertRaises(FileExistsError):
                api.create_plan(output, "quick", 1, ["fixed", "oracle"])
            self.assertEqual(api.run_task(output, 0, resume=False), "completed")
            path = output / "tasks/task_000000.json"
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                api.run_task(output, 0, resume=False)
            with patch.object(api, "fit_method", wraps=api.fit_method) as fitting:
                self.assertEqual(api.run_task(output, 0, resume=True), "skipped")
                fitting.assert_not_called()
            self.assertEqual(original, path.read_bytes())
            state = api.status(output)
            self.assertEqual(state["completed"], 1)
            self.assertEqual(state["missing"], list(range(1, 14)))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                api.merge(output)
            self.assertFalse((output / "summary_results.csv").exists())

    def test_quick_complete_merge_counts_pairing_and_search_gap(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed", "adaptive", "oracle"])
            for task_id in range(14):
                api.run_task(output, task_id, resume=False)
            report = api.merge(output)
            self.assertEqual(report["task_count"], 14)
            self.assertEqual(report["method_result_count"], 42)
            self.assertEqual(report["paired_fixed_vs_adaptive_count"], 14)
            self.assertEqual(report["search_gap_count"], 14)
            with (output / "paired_fixed_vs_adaptive.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 14)
            with (output / "summary_results.csv").open(newline="", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 42)
            self.assertTrue(all(row["n_reps"] == "1" and row["pmse_sample_sd"] == "" for row in summaries))
            with self.assertRaises(FileExistsError):
                api.merge(output)

    def test_concurrent_exclusive_commit_never_replaces_a_winner(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            target = Path(temporary) / "exclusive.bin"
            def writing(payload):
                try:
                    api.commit_exclusive(target, payload)
                    return "committed"
                except FileExistsError:
                    return "already_exists"
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(writing, (b"first complete payload", b"second complete payload")))
            self.assertCountEqual(results, ["committed", "already_exists"])
            self.assertIn(target.read_bytes(), (b"first complete payload", b"second complete payload"))

    def test_resume_rejects_npz_metric_and_source_conflicts(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed"])
            api.run_task(output, 0)
            record_path = output / "tasks/task_000000.json"
            arrays_path = output / "tasks/task_000000.npz"
            original_json, original_npz = record_path.read_bytes(), arrays_path.read_bytes()
            with arrays_path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "NPZ checksum"):
                api.run_task(output, 0, resume=True)
            arrays_path.write_bytes(original_npz)
            record = json.loads(original_json)
            record["results"][0]["metrics"]["ree"] += .1
            record["record_sha256"] = api.digest({key: value for key, value in record.items() if key != "record_sha256"})
            record_path.write_bytes(api.json_bytes(record))
            with self.assertRaisesRegex(ValueError, "ree differs"):
                api.run_task(output, 0, resume=True)
            record_path.write_bytes(original_json)
            with patch.object(api, "source_fingerprints", return_value={"modified": "source"}):
                with self.assertRaisesRegex(ValueError, "Code/configuration differs"):
                    api.run_task(output, 0, resume=True)
            with patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
                with self.assertRaisesRegex(ValueError, "Set OPENBLAS_NUM_THREADS"):
                    api.run_task(output, 0, resume=True)
            arrays_path.unlink()
            with self.assertRaisesRegex(ValueError, "Partial checkpoint"):
                api.run_task(output, 0, resume=True)

    def test_fixed_only_merge_marks_unavailable_comparisons_without_fake_rows(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            api.create_plan(output, "quick", 1, ["fixed"])
            for task_id in range(14):
                api.run_task(output, task_id)
            report = api.merge(output)
            self.assertEqual(report["method_result_count"], 14)
            self.assertFalse(report["paired_fixed_vs_adaptive_applicable"])
            self.assertFalse(report["search_gap_applicable"])
            for filename in ("paired_fixed_vs_adaptive.csv", "search_gap.csv"):
                with (output / filename).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
