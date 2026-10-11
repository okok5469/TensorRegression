"""Frozen plans, paired inputs, immutable outputs, and pure rescoring contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from numerical_simulation.candidate_search_study import run, worker


class PlanTests(unittest.TestCase):
    def test_historical_requests_reuse_exact_data_seeds_without_cv(self):
        tasks = run.build_tasks("historical", 3, ["A0", "A1", "A2", "A3"])
        self.assertEqual(len(tasks), 27)
        for task in tasks:
            original = run.read_json(run.root() / "numerical_simulation/outputs_msic_ablation_v51_final" / task["request_file"])
            self.assertEqual(task["request"]["config"], original["config"])
            self.assertEqual(task["request"]["seed"], original["seed"])
            self.assertEqual(task["request"]["test_seed"], original["test_seed"])
            self.assertNotIn("include_cv", task["request"])

    def test_development_confirmation_seeds_do_not_overlap(self):
        cohorts = [run.build_tasks(cohort, 10, ["A0", "A3"]) for cohort in ("development", "confirmation")]
        train = [set(t["request"]["seed"] for t in tasks) for tasks in cohorts]
        test = [set(t["request"]["test_seed"] for t in tasks) for tasks in cohorts]
        self.assertEqual(len(train[0]), 90)
        self.assertFalse(train[0] & train[1])
        self.assertFalse(train[0] & test[0] | train[1] & test[1])
        self.assertFalse(test[0] & test[1])

    def test_seed_does_not_depend_on_number_replicates(self):
        one = run.build_tasks("development", 1, ["A0"])
        ten = run.build_tasks("development", 10, ["A0"])
        self.assertEqual([t["request"] for t in one], [t["request"] for t in ten if t["replicate"] == 0])

    def test_groups_and_counts_rejected_before_fitting(self):
        for groups in (["A0", "A0"], ["wrong"], []):
            with self.assertRaises(ValueError):
                run.build_tasks("development", 1, groups)
        with self.assertRaises(ValueError):
            run.build_tasks("historical", 10, ["A0"])
        with self.assertRaises(ValueError):
            run.build_tasks("development", 0, ["A0"])

    def test_plan_immutable_and_request_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "source_manifests", return_value={"core": {}, "experiment": {}}):
            output = Path(temporary) / "study"
            plan = run.create_plan(output, "development", 1, ["A0", "A3"])
            self.assertEqual(run.load_plan(output)["plan_id"], plan["plan_id"])
            with self.assertRaises(FileExistsError):
                run.create_plan(output, "development", 1, ["A0"])
            (output / plan["tasks"][0]["request_file"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request"):
                run.load_plan(output)

    def test_frozen_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "source_manifests", return_value={"core": {}, "experiment": {}}):
            output = Path(temporary) / "study"
            run.create_plan(output, "development", 1, ["A0"])
            with patch.object(run, "source_manifests", return_value={"core": {"changed": "sha"}, "experiment": {}}):
                with self.assertRaisesRegex(ValueError, "sources"):
                    run.load_plan(output)

    def test_summary_requires_all_datasets(self):
        plan = {"tasks": [{"task_id": 0}, {"task_id": 1}]}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run.summarize_reports(plan, [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run.summarize_reports(plan, [{"task_id": 0}, {"task_id": 0}])

    def test_worker_rejects_other_loaded_source(self):
        import sys
        import types
        fake = types.SimpleNamespace(__file__=str(run.root() / "src/tensor_abess/__init__.py"))
        with patch.dict(sys.modules, {"tensor_abess": fake}), self.assertRaisesRegex(RuntimeError, "different core"):
            worker.configure_core()

    def test_pure_rescoring_keeps_pool_unchanged(self):
        import numpy as np
        pool = [{"B": np.zeros((2, 2, 2)), "supports": [np.array([0])] * 3,
                 "sizes": (1, 1, 1), "rss": 5.0, "loss": .025},
                {"B": np.ones((2, 2, 2)), "supports": [np.array([0, 1])] * 3,
                 "sizes": (2, 2, 2), "rss": 4.0, "loss": .02}]
        before = worker.pool_fingerprint(pool)
        with patch.object(worker, "score_pool", side_effect=lambda values, *a, **k: (values[0], {"selected_index": 0})):
            worker.rescore_pool(pool, 100, (2, 2, 2))
        self.assertEqual(worker.pool_fingerprint(pool), before)

    def test_support_capacity_audit_rejects_illegal_candidate(self):
        import numpy as np
        candidate = {"B": np.zeros((3, 3, 3)), "supports": [np.arange(3)] * 3,
                     "sizes": (3, 3, 3), "rss": 1.0, "loss": .05}
        with self.assertRaisesRegex(ValueError, "capacity"):
            worker.audit_pool([candidate], 10, (3, 3, 3))

    def valid_fixed_candidate(self):
        import numpy as np
        return {"kind": "global-fixed", "B": np.zeros((3, 3, 3)),
                "supports": [np.arange(2), np.arange(2), np.array([0])],
                "sizes": (2, 2, 1), "rss": 240.0, "loss": .6,
                "loss_history": [.8, .6], "visited_loss_history": [.8, .6, .75],
                "best_iteration": 1}

    def adaptive_regression_candidate(self):
        import numpy as np
        loss = .5812287699242976
        return {"kind": "adaptive-stage", "B": np.zeros((32, 20, 25)),
                "supports": [np.array([2]), np.array([2, 10]), np.array([4])],
                "sizes": (1, 2, 1), "rss": 400 * loss, "loss": loss,
                "loss_history": [.6752636237588817, .563123772752786, loss, loss],
                "msic_history": [77.9073052082144, 54.91255636363953, 53.336726944828875, 53.336726944828875],
                "sizes_history": [(1, 1, 1), (2, 2, 1), (1, 2, 1), (1, 2, 1)],
                "visited_loss_history": [], "best_iteration": None}

    def test_regression_adaptive_smaller_q_can_increase_rss_and_improve_msic(self):
        candidate = self.adaptive_regression_candidate()
        self.assertGreater(candidate["loss_history"][2], candidate["loss_history"][1])
        self.assertLess(candidate["msic_history"][2], candidate["msic_history"][1])
        worker.audit_pool([candidate], 200, (32, 20, 25))

    def test_same_increasing_history_is_rejected_for_fixed_candidate(self):
        candidate = self.valid_fixed_candidate()
        candidate["loss_history"] = self.adaptive_regression_candidate()["loss_history"]
        with self.assertRaisesRegex(ValueError, "training loss increased"):
            worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_fixed_requires_nonempty_visited_history(self):
        candidate = self.valid_fixed_candidate()
        candidate["visited_loss_history"] = []
        with self.assertRaisesRegex(ValueError, "visited"):
            worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_fixed_returns_minimum_visited_loss(self):
        candidate = self.valid_fixed_candidate()
        candidate["visited_loss_history"] = [.8, .55, .6]
        candidate["best_iteration"] = 2
        with self.assertRaisesRegex(ValueError, "minimum"):
            worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_fixed_best_iteration_identifies_returned_loss(self):
        for index in (0, -1, 99, None, 1.5, True):
            candidate = self.valid_fixed_candidate()
            candidate["best_iteration"] = index
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "best_iteration"):
                worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_fixed_and_adaptive_histories_must_be_finite(self):
        for kind in ("global-fixed", "adaptive-stage"):
            for history in ("loss_history", "visited_loss_history"):
                candidate = self.valid_fixed_candidate()
                candidate["kind"] = kind
                candidate[history] = [.8, float("inf")]
                with self.subTest(kind=kind, history=history), self.assertRaisesRegex(ValueError, "Nonfinite"):
                    worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_unknown_candidate_kind_is_rejected(self):
        candidate = self.valid_fixed_candidate()
        candidate["kind"] = "unrecognized"
        with self.assertRaisesRegex(ValueError, "kind"):
            worker.audit_pool([candidate], 200, (3, 3, 3))

    def test_valid_fixed_best_visited_candidate_passes(self):
        worker.audit_pool([self.valid_fixed_candidate()], 200, (3, 3, 3))

    def test_groups_must_share_test_fingerprint(self):
        report = {"groups": {"A0": {"fingerprints": {"X_train": "a", "y_train": "b", "B_true": "c", "independent_test": "d"}},
                              "A3": {"fingerprints": {"X_train": "a", "y_train": "b", "B_true": "c", "independent_test": "different"}}}}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            run.validate_group_pairing(report)

    def fixture_result(self, output):
        request = {"config": {"n_test": 20}, "seed": 1, "test_seed": 1000001, "fit_options": {"candidate_budget": 2000}}
        task = {"task_id": 0, "request": request}
        libraries = {"python": "3.9.25", "numpy": "2.0.2", "scipy": "1.13.1"}
        plan = {"version": worker.VERSION, "groups": ["A0"], "environment": {"libraries": libraries}, "sources": {"core": {}}}
        fingerprints = {"X_train": "x", "y_train": "y", "B_true": "b", "independent_test": {"sha256": "t", "n_observations": 20}}
        (output / "jobs").mkdir()
        npz = output / "jobs/task_000.npz"
        npz.write_bytes(b"immutable-cache")
        report = {**request, "version": worker.VERSION, "status": "complete", "group_order": ["A0"],
            "environment": libraries, "fingerprints": {**fingerprints, "core_sources": {}},
            "groups": {"A0": {"fingerprints": fingerprints, "environment": libraries}},
            "array_cache": npz.name, "array_cache_sha256": run.file_hash(npz),
            "rows": [{"group": "A0", "rule": "msic-alpha1.5-scale1"}]}
        run.write_json(output / "jobs/task_000.json", report)
        return plan, task, report

    def test_result_cache_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, task, _ = self.fixture_result(output)
            run.validate_result(output, plan, task)
            (output / "jobs/task_000.npz").write_bytes(b"changed-cache")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run.validate_result(output, plan, task)

    def test_result_seed_and_extra_msic_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, task, report = self.fixture_result(output)
            path = output / "jobs/task_000.json"
            report["seed"] = 999
            path.write_text(run.canonical(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed"):
                run.validate_result(output, plan, task)
            report["seed"] = 1
            report["rows"].append({"group": "A0", "rule": "msic-alpha2-scale1"})
            path.write_text(run.canonical(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Phase D"):
                run.validate_result(output, plan, task)

    def test_partial_cache_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan, task, _ = self.fixture_result(output)
            path = output / "jobs/task_000.json"
            before = path.read_bytes()
            (output / "jobs/task_000.npz").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                run.validate_result(output, plan, task)
            self.assertEqual(path.read_bytes(), before)

    def test_python39_syntax(self):
        import ast
        for name in ("__init__.py", "run.py", "worker.py", "test_run.py"):
            ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
