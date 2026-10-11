"""Frozen task grids and strict completeness checks for paired validation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from numerical_simulation.version_validation import paired


class PlanTests(unittest.TestCase):
    def test_grid_seeds_and_inner_cv_are_predeclared(self):
        plan = paired.build_plan(3)
        self.assertEqual(len(plan["tasks"]), 27)
        self.assertEqual(len(plan["jobs"]), 54)
        observed = set()
        for task in plan["tasks"]:
            request = task["request"]
            self.assertEqual(request["seed"], 20261010 + 1000 * task["scenario_id"] + task["replicate"])
            self.assertEqual(request["test_seed"], request["seed"] + 1_000_000)
            self.assertEqual(request["include_cv"], request["config"]["n_train"] == 200)
            self.assertEqual(request["config"]["dims"], [32, 20, 25])
            self.assertEqual(request["config"]["sparsity"], [4, 3, 5])
            observed.add((request["config"]["n_train"], request["config"]["snr"]))
        self.assertEqual(observed, {(n, snr) for n in (100, 200, 400) for snr in (0.5, 2.0, 4.0)})
        self.assertEqual(sum(t["request"]["include_cv"] for t in plan["tasks"]), 9)

    def test_seed_identity_does_not_depend_on_replication_count(self):
        small, large = paired.build_plan(1), paired.build_plan(3)
        first_replicates = [t for t in large["tasks"] if t["replicate"] == 0]
        self.assertEqual([t["request"] for t in small["tasks"]], [t["request"] for t in first_replicates])

    def test_summary_rejects_missing_and_duplicate_jobs(self):
        plan = paired.build_plan(1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            paired.summarize_reports(plan, [])
        duplicate = {"job_id": plan["jobs"][0]["job_id"], "report": {}}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired.summarize_reports(plan, [duplicate, duplicate])

    def test_plan_requires_new_empty_output_and_loads_frozen_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            plan = paired.create_plan(output, 1)
            self.assertEqual(paired.load_plan(output)["plan_id"], plan["plan_id"])
            self.assertEqual(len(list((output / "requests").glob("*.json"))), 9)
            with self.assertRaises(FileExistsError):
                paired.create_plan(output, 1)
            request = output / plan["tasks"][0]["request_file"]
            request.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request"):
                paired.load_plan(output)

    def test_invalid_replication_counts_rejected(self):
        with self.assertRaises(ValueError):
            paired.build_plan(0)


if __name__ == "__main__":
    unittest.main()
