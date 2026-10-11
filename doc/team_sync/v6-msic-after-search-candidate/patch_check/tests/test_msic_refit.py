"""Locked-rule provenance, fresh fits, and cached model identity contracts."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class FreshRefitTests(unittest.TestCase):
    def api(self):
        name = "numerical_simulation.candidate_search_study.msic_refit"
        self.assertIsNotNone(importlib.util.find_spec(name), "Fresh-fit validation feature is missing")
        return importlib.import_module(name)

    def lock(self):
        return {"status": "locked", "search_group": "A3",
                "rule": {"id": "msic-alpha2-scale0", "criterion": "msic", "alpha": 2., "gamma": 1., "scale": 0.},
                "development_search_plan_id": "search", "development_msic_plan_id": "msic",
                "development_summary_sha256": "summary"}

    def test_locked_rule_and_both_parent_versions_are_required(self):
        api = self.api()
        lock = self.lock()
        expected = api.validate_lock(lock, {"plan_id": "search"}, {"plan_id": "msic", "input_plan_id": "search"}, "summary")
        self.assertEqual(expected["id"], "msic-alpha2-scale0")
        for key, bad in (("status", "draft"), ("search_group", "A2"),
                         ("development_search_plan_id", "stale"), ("development_msic_plan_id", "stale"),
                         ("development_summary_sha256", "stale")):
            changed = self.lock(); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.validate_lock(changed, {"plan_id": "search"}, {"plan_id": "msic", "input_plan_id": "search"}, "summary")

    def test_unregistered_alpha_or_mode_criterion_change_is_rejected(self):
        api = self.api()
        for key, value in (("alpha", 1.7), ("gamma", .5), ("scale", 2.), ("id", "unknown")):
            lock = self.lock(); lock["rule"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.validate_lock(lock, {"plan_id": "search"}, {"plan_id": "msic", "input_plan_id": "search"}, "summary")

    def test_sic_and_bic_are_only_post_pool_global_rules(self):
        api = self.api()
        for criterion, rule_id, gamma in (("tensor-sic", "tensor-sic", 1.), ("bic", "pure-bic", None)):
            lock = self.lock(); lock["rule"] = {"id": rule_id, "criterion": criterion, "alpha": None, "gamma": gamma, "scale": 0.}
            chosen = api.validate_lock(lock, {"plan_id": "search"}, {"plan_id": "msic", "input_plan_id": "search"}, "summary")
            self.assertEqual(chosen["criterion"], criterion)

    def test_fresh_fit_has_no_cache_and_keeps_mode_msic(self):
        api = self.api()
        config = type("Config", (), dict(k_max=1, abess_init_c=3, abess_init_criterion="bic",
            n_lambdas=20, cv_folds=5, max_iter=500, tol=1e-6, max_cycles=6))()
        X, y = np.zeros((5, 2, 2, 2)), np.zeros(5)
        with patch.object(api, "core_fit", return_value="result") as fit:
            self.assertEqual(api.fit_fresh(X, y, config, self.lock()["rule"]), "result")
        kwargs = fit.call_args.kwargs
        self.assertEqual(kwargs["adaptive_criterion"], "msic")
        self.assertEqual(kwargs["cap_policy"], "candidate")
        self.assertEqual(kwargs["grid_policy"], "full")
        self.assertEqual(kwargs["candidate_budget"], 2000)
        self.assertIsNone(kwargs["fixed_candidate_cache"])
        self.assertEqual(kwargs["msic_refine_alpha"], 2.)
        self.assertEqual(kwargs["msic_refine_one_se_scale"], 0.)
        with patch.object(api, "core_fit") as fit:
            api.fit_fresh(X, y, config, {"criterion": "bic", "alpha": None, "scale": 0.})
        self.assertEqual(fit.call_args.kwargs["adaptive_criterion"], "msic")
        self.assertEqual(fit.call_args.kwargs["msic_refine_alpha"], 1.5)

    def test_model_identity_requires_coefficients_supports_and_training_score(self):
        api = self.api()
        candidate = {"B": np.zeros((2, 2, 2)), "supports": [np.array([0])] * 3,
                     "rss": 10., "loss": .05}
        row = {"rss_train": 10., "loss_train": .05, "supports": [[0], [0], [0]]}
        selected = {"B": candidate["B"].copy(), "supports": [S.copy() for S in candidate["supports"]]}
        api.assert_model_matches(candidate, row, selected)
        changed = {**selected, "B": np.ones((2, 2, 2))}
        with self.assertRaisesRegex(ValueError, "coefficients"):
            api.assert_model_matches(candidate, row, changed)
        with self.assertRaisesRegex(ValueError, "RSS"):
            api.assert_model_matches(candidate, {**row, "rss_train": 11.}, selected)
        with self.assertRaisesRegex(ValueError, "supports"):
            api.assert_model_matches(candidate, {**row, "supports": [[1], [0], [0]]}, selected)

    def arguments(self, output):
        return ["--source-output", "source", "--msic-output", "scored", "--lock-file", "locked.json",
                "--task-id", "0", "--output-dir", str(output)]

    def test_existing_cache_stops_before_any_child_fit(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cache = output / "task_000.npz"
            cache.write_bytes(b"preserve")
            with patch.object(api.subprocess, "run") as spawn, self.assertRaises(FileExistsError):
                api.main(self.arguments(output))
            spawn.assert_not_called()
            self.assertEqual(cache.read_bytes(), b"preserve")

    def test_child_failure_is_recorded_and_cannot_be_resumed(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = type("Result", (), {"returncode": 9})()
            with patch.object(api.subprocess, "run", return_value=result), self.assertRaises(RuntimeError):
                api.main(self.arguments(output))
            failure = json.loads((output / "task_000.failure.json").read_text(encoding="utf8"))
            self.assertEqual(failure["returncode"], 9)
            with patch.object(api.subprocess, "run") as spawn, self.assertRaises(FileExistsError):
                api.main(self.arguments(output))
            spawn.assert_not_called()

    def test_fresh_worker_refuses_partial_output_before_loading_inputs(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "task_000.json").write_text("{}", encoding="utf8")
            args = type("Args", (), {"output_dir": output, "task_id": 0})()
            with patch.object(api, "run_case") as fit, self.assertRaises(FileExistsError):
                api.worker_main(args)
            fit.assert_not_called()

    def test_python39_syntax(self):
        import ast
        api = self.api()
        ast.parse(Path(api.__file__).read_text(encoding="utf8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
