"""Behavior checks for version-isolated selection experiments."""
from __future__ import annotations

import unittest
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from numerical_simulation.version_validation import worker


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker.configure_core("v5.1-bcd-fix")
        from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
        cls.config = TensorSimulationConfig(n_train=70, n_test=31, dims=(4, 4, 4), sparsity=(2, 2, 2),
            n_lambdas=3, cv_folds=2, max_iter=50, max_cycles=2, k_max=1, test_batch_size=11)
        cls.dataset = generate_simulation_dataset(cls.config, seed=741)

    def test_cached_rescore_equals_actual_independent_adaptive_fit(self):
        result, pool, _ = worker.fit_with_candidates(self.dataset, self.config)
        chosen, info = worker.select_candidate(pool, 70, (4, 4, 4), alpha=1.5, scale=1.0)
        np.testing.assert_array_equal(chosen["B"], result.B)
        self.assertAlmostEqual(info["selected_score"], result.sic)
        independent = worker.fit_adaptive(self.dataset.X_train, self.dataset.y_train, self.config,
            alpha=3.0, scale=0.5)
        alternative, _ = worker.select_candidate(pool, 70, (4, 4, 4), alpha=3.0, scale=0.5)
        np.testing.assert_array_equal(alternative["B"], independent.B)
        self.assertAlmostEqual(alternative["rss"], independent.rss)
        self.assertGreater(len(pool), 1)

    def test_tolerance_depends_on_each_rules_minimum_model(self):
        pool = [{"sizes": (1, 1, 1), "rss": 100.0, "B": None},
                {"sizes": (3, 3, 3), "rss": 65.0, "B": None}]
        _, info = worker.select_candidate(pool, 100, (5, 5, 5), alpha=1.5, scale=1.0)
        self.assertAlmostEqual(info["tolerance"], np.sqrt(info["minimum_q"] * np.log(100)))
        _, exact = worker.select_candidate(pool, 100, (5, 5, 5), alpha=1.5, scale=0.0)
        self.assertEqual(exact["tolerance"], 0.0)

    def test_classic_bic_and_tensor_sic_have_distinct_penalties(self):
        pool = [{"sizes": (2, 2, 2), "rss": 10.0, "B": None}]
        _, classic = worker.select_candidate(pool, 70, (4, 4, 4), criterion="bic")
        _, tensor = worker.select_candidate(pool, 70, (4, 4, 4), criterion="tensor-sic")
        self.assertAlmostEqual(classic["selected_score"], 70 * np.log(10 / 70) + 8 * np.log(70))
        self.assertGreater(tensor["selected_score"], classic["selected_score"])

    def test_shared_stream_matches_existing_prediction_metrics(self):
        from numerical_simulation.data_generation import iter_test_batches
        from numerical_simulation.metrics import compute_prediction_metrics_from_batches
        arrays = [self.dataset.B_true, np.zeros_like(self.dataset.B_true)]
        metrics, fingerprint = worker.evaluate_shared_test(self.dataset, arrays, 741001)
        for actual, B in zip(metrics, arrays):
            expected = compute_prediction_metrics_from_batches(iter_test_batches(self.dataset, seed=741001),
                self.dataset.B_true, B)
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(fingerprint["n_observations"], 31)
        self.assertEqual(fingerprint["n_batches"], 3)
        self.assertEqual(len(fingerprint["sha256"]), 64)

    def test_cv_grid_predeclared_capacity_is_independent_of_outcomes(self):
        grid, excluded = worker.cv_grid(200, (32, 20, 25), folds=5)
        self.assertIn((4, 3, 5), grid)
        self.assertNotIn((5, 4, 6), grid)
        self.assertEqual(excluded, [(5, 4, 6)])
        self.assertTrue(all(np.prod(s) <= int(0.7 * 160) for s in grid))

    def test_cli_versions_have_matching_inputs_and_isolated_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request = directory / "request.json"
            request.write_text(json.dumps({"config": asdict(self.config), "seed": 741,
                "test_seed": 741001, "include_cv": False}), encoding="utf-8")
            reports = []
            for version in ("v5-original", "v5.1-bcd-fix"):
                output = directory / f"{version}.json"
                completed = subprocess.run([sys.executable, "-m", "numerical_simulation.version_validation.worker",
                    "--request", str(request), "--output", str(output), "--version", version],
                    capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(output.with_suffix(".npz").is_file())
                self.assertEqual(report["status"], "complete")
                reports.append(report)
            for key in ("X_train", "y_train", "B_true", "independent_test"):
                self.assertEqual(reports[0]["fingerprints"][key], reports[1]["fingerprints"][key])
            self.assertNotEqual(reports[0]["fingerprints"]["core_sources"]["tensor_abess/tensor/bcd.py"],
                reports[1]["fingerprints"]["core_sources"]["tensor_abess/tensor/bcd.py"])
            self.assertEqual(len(reports[0]["rows"]), 1)
            self.assertEqual(len(reports[1]["rows"]), 11)

    def test_cv_fresh_inner_fits_reproduce_and_full_refit_matches_rss(self):
        from numerical_simulation.data_generation import generate_simulation_dataset
        from tensor_abess.tensor.ops import tensor_inner
        config = replace(self.config, n_train=200, max_cycles=1)
        dataset = generate_simulation_dataset(config, seed=741)
        fit, report = worker.fit_cv(dataset, config, 741)
        repeated, repeated_report = worker.fit_cv(dataset, config, 741)
        np.testing.assert_array_equal(fit["B"], repeated["B"])
        self.assertEqual(report["selected_sizes"], repeated_report["selected_sizes"])
        for row in report["fold_rows"]:
            self.assertEqual(row["n_train"], 160)
            self.assertEqual(row["n_validation"], 40)
            self.assertFalse(set(row["train_indices"]) & set(row["validation_indices"]))
            self.assertEqual(len(set(row["train_indices"]) | set(row["validation_indices"])), 200)
        expected_score = min(row["validation_mse"] for row in report["grid_scores"])
        self.assertAlmostEqual(report["selected_validation_mse"], expected_score)
        residual = dataset.y_train - tensor_inner(dataset.X_train, fit["B"])
        self.assertAlmostEqual(fit["rss"], float(residual @ residual))


if __name__ == "__main__":
    unittest.main()
