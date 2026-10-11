"""Numerical tests for the isolated LEMON preprocessing and model adapters."""

import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class LemonModelTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module("empirical_analysis.lemon." + name)
        except ModuleNotFoundError as exc:
            self.fail(f"Required LEMON module is not implemented: {exc}")

    def data(self, n=48):
        rng = np.random.default_rng(72)
        X = rng.normal(size=(n, 2, 2, 2))
        C = np.column_stack((np.ones(n), rng.normal(size=n), np.arange(n) % 2))
        y = 4.0 + 0.4 * C[:, 1] + 0.1 * C[:, 2] + 0.8 * X[:, 0, 0, 0]
        y += rng.normal(scale=0.1, size=n)
        return X, y, C

    def model_config(self):
        return {"n_lambdas": 3, "max_iter": 35, "max_cycles": 1,
                "cv_folds": 2, "m_max": 3}

    def config(self, mode="adjusted"):
        return {"seed": 37, "response": {"primary_transform": "log"},
                "analysis": {"mode": mode, "formal_cv_folds": 3,
                             "min_samples": 30, "run_tensor_glasso": True,
                             "run_vector_lasso": True},
                "model": self.model_config()}

    def test_preprocessing_uses_only_training_data(self):
        mod = self.module("confounds")
        X, y, C = self.data()
        pre = mod.fit_preprocessor(X[:36], y[:36], C[:36])
        arrays = {key: value.copy() for key, value in pre.to_arrays().items()}
        train_z = pre.apply(X[:36], C[:36])
        np.testing.assert_allclose(train_z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(train_z.std(axis=0, ddof=1), 1, atol=1e-12)
        np.testing.assert_allclose(C[:36].T @ train_z.reshape(36, -1), 0, atol=1e-11)
        np.testing.assert_allclose(pre.transform_response(y[:36], C[:36]).mean(), 0, atol=1e-12)
        pre.apply(X[36:] * 10000, C[36:])
        for key, value in pre.to_arrays().items():
            np.testing.assert_array_equal(value, arrays[key])

    def test_categorical_age_design_uses_training_only_nuisance_fit(self):
        mod = self.module('confounds')
        rng = np.random.default_rng(317)
        groups = np.tile(np.arange(4), 12)
        sex = (np.arange(48) // 4) % 2
        C = np.column_stack([np.ones(48)] + [(groups == k).astype(float) for k in (1, 2, 3)] + [sex])
        X = rng.normal(size=(48, 2, 2, 2)) + (C @ np.array([0., 2., -1., 3., .5]))[:, None, None, None]
        y = C @ np.array([4., .4, -.2, .6, .1]) + rng.normal(size=48)
        pre = mod.fit_preprocessor(X[:32], y[:32], C[:32])
        np.testing.assert_allclose(C[:32].T @ pre.apply(X[:32], C[:32]).reshape(32, -1), 0., atol=1e-10)
        heldout = pre.predict(np.zeros((2, 2, 2)), X[32:], C[32:])
        expected = C[32:] @ np.linalg.lstsq(C[:32], y[:32], rcond=None)[0]
        np.testing.assert_allclose(heldout, expected, atol=1e-10)
        # A category present only in held-out subjects must stop, never be imputed.
        singleton = C.copy()
        singleton[:, 3] = 0.
        singleton[-1, 3] = 1.
        with self.assertRaisesRegex(ValueError, 'rank'):
            mod.fit_preprocessor(X[:32], y[:32], singleton[:32])

    def test_predictions_restore_the_input_response_scale(self):
        mod = self.module("confounds")
        X, y, C = self.data()
        for train_y in (y, np.exp(y)):
            pre = mod.fit_preprocessor(X[:36], train_y[:36], C[:36])
            B = np.arange(8, dtype=float).reshape(2, 2, 2) / 8
            nuisance = C[36:] @ np.linalg.lstsq(C[:36], train_y[:36], rcond=None)[0]
            centered_mean = (train_y[:36] - C[:36] @ np.linalg.lstsq(C[:36], train_y[:36], rcond=None)[0]).mean()
            expected = np.einsum("nijk,ijk->n", pre.apply(X[36:], C[36:]), B)
            expected += nuisance + centered_mean
            np.testing.assert_allclose(pre.predict(B, X[36:], C[36:]), expected)
        pilot = mod.fit_preprocessor(X[:36], y[:36])
        np.testing.assert_allclose(pilot.predict(np.zeros((2, 2, 2)), X[36:]), y[:36].mean())

    def test_constant_cells_and_invalid_inputs(self):
        mod = self.module("confounds")
        X, y, C = self.data()
        X[:, 0, 0, 0] = 7
        pre = mod.fit_preprocessor(X, y)
        np.testing.assert_array_equal(pre.apply(X)[:, 0, 0, 0], 0)
        with self.assertRaisesRegex(ValueError, "rank"):
            mod.fit_preprocessor(X, y, np.ones((len(y), 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            mod.fit_preprocessor(X, y * np.nan)
        with self.assertRaises(ValueError):
            mod.fit_preprocessor(X, y[:-1])
        adjusted = mod.fit_preprocessor(X, y, C)
        with self.assertRaises(ValueError):
            adjusted.apply(X)

    def test_all_three_existing_models_fit_real_data(self):
        mod = self.module("models")
        confounds = self.module("confounds")
        X, y, C = self.data()
        pre = confounds.fit_preprocessor(X, y, C)
        for name in ("Tensor-ABESS", "Tensor-GLasso", "Vector Lasso"):
            with self.subTest(model=name):
                result = mod.fit_model(name, pre.apply(X, C), pre.transform_response(y, C), self.model_config())
                self.assertEqual(result.B.shape, X.shape[1:])
                self.assertTrue(np.isfinite(result.B).all())
                self.assertTrue(np.isfinite(result.rss))
                self.assertLess(np.mean((pre.predict(result.B, X, C) - y) ** 2), np.var(y))
                self.assertAlmostEqual(result.rss, 2 * len(y) * result.loss)

    def test_model_rejects_unknown_options_instead_of_silently_ignoring_them(self):
        mod = self.module("models")
        X, y, _ = self.data()
        with self.assertRaisesRegex(ValueError, "Unknown.*parameter"):
            mod.fit_model("Tensor-ABESS", X, y, {"max_iter_typo": 2})
        with self.assertRaisesRegex(ValueError, "adaptive"):
            mod.fit_model("Tensor-ABESS", X, y, {"target_sparsity": [1, 1, 1]})

    def test_cv_has_one_held_out_prediction_per_subject_per_model(self):
        mod = self.module("evaluation")
        X, y, C = self.data()
        ids = [f"sub-{i:06d}" for i in range(len(y))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cv"
            summary = mod.run_cross_validation(X, y, ids, C, self.config(), out)
            json.dumps(summary, allow_nan=False)
            with (out / "oof_predictions.csv").open(newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), len(y) * 3)
            for name in ("Tensor-ABESS", "Tensor-GLasso", "Vector Lasso"):
                rows = [row for row in predictions if row["model"] == name]
                self.assertEqual(sorted(row["subject_id"] for row in rows), sorted(ids))
                self.assertEqual({row["response_scale"] for row in rows}, {"log"})
            with (out / "fold_assignments.csv").open(newline="", encoding="utf-8") as handle:
                assignments = list(csv.DictReader(handle))
            for fold in range(1, 4):
                held = {row["subject_id"] for row in assignments if int(row["fold"]) == fold}
                with (out / "folds" / f"fold_{fold}" / "fold_info.json").open(encoding="utf-8") as handle:
                    info = json.load(handle)
                self.assertEqual(held, set(info["test_subject_ids"]))
                self.assertFalse(held.intersection(info["train_subject_ids"]))
                with np.load(out / "folds" / f"fold_{fold}" / "preprocessing.npz") as params:
                    self.assertIn("x_scale", params.files)
                    pre = self.module("confounds").FoldPreprocessor(**dict(params))
                for name in ("Tensor-ABESS", "Tensor-GLasso", "Vector Lasso"):
                    indices = info["test_indices"]
                    B = np.load(out / "folds" / f"fold_{fold}" / name.replace(" ", "_") / "B.npy")
                    restored = pre.predict(B, X[indices], C[indices])
                    saved = {row["subject_id"]: float(row["predicted"]) for row in predictions
                             if row["model"] == name and int(row["fold"]) == fold}
                    np.testing.assert_allclose(restored, [saved[ids[index]] for index in indices])
            for axis in ("channel", "time", "frequency"):
                self.assertTrue((out / f"{axis}_selection_frequency.csv").is_file())
            config2 = self.config()
            config2["analysis"]["run_tensor_glasso"] = False
            config2["analysis"]["run_vector_lasso"] = False
            mod.run_cross_validation(X, y, ids, C, config2, Path(tmp) / "repeat")
            self.assertEqual((out / "fold_assignments.csv").read_bytes(),
                             (Path(tmp) / "repeat" / "fold_assignments.csv").read_bytes())

    def test_formal_cv_requires_covariates_and_sufficient_samples(self):
        mod = self.module("evaluation")
        X, y, C = self.data()
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "covariate|confound"):
                mod.run_cross_validation(X, y, ids, None, self.config(), Path(tmp) / "missing")
            config = self.config()
            config["analysis"]["min_samples"] = 100
            with self.assertRaisesRegex(ValueError, "sample"):
                mod.run_cross_validation(X, y, ids, C, config, Path(tmp) / "small")
            with self.assertRaisesRegex(ValueError, "unique|duplicate"):
                mod.run_cross_validation(X, y, ["duplicate"] * len(y), C, self.config(), Path(tmp) / "ids")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_explicit_pilot_remains_marked_and_does_not_require_covariates(self):
        mod = self.module("evaluation")
        X, y, _ = self.data()
        config = self.config("pilot")
        config["analysis"].update({"min_samples": 30, "run_tensor_glasso": False,
                                   "run_vector_lasso": False})
        config["response"]["primary_transform"] = "raw"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            summary = mod.run_cross_validation(X, y, [str(i) for i in range(len(y))], None, config, out)
            self.assertTrue(summary["PILOT_UNADJUSTED"])
            self.assertEqual(summary["response_scale"], "raw")
            with self.assertRaises(FileExistsError):
                mod.run_cross_validation(X, y, [str(i) for i in range(len(y))], None, config, out)

    def test_pilot_requires_an_explicitly_satisfied_sample_threshold(self):
        mod = self.module("evaluation")
        X, y, _ = self.data()
        config = self.config("pilot")
        config["analysis"].update({"min_samples": 100, "run_tensor_glasso": False,
                                   "run_vector_lasso": False})
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("run_cross_validation", "fit_final_model"):
                with self.subTest(entrypoint=name):
                    with self.assertRaisesRegex(ValueError, "sample"):
                        getattr(mod, name)(X, y, ids, None, config, Path(tmp) / name)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cli_path_config_is_preserved_in_json_and_returned_summaries(self):
        mod = self.module("evaluation")
        X, y, C = self.data()
        config = self.config()
        config["analysis"].update({"run_tensor_glasso": False, "run_vector_lasso": False})
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as tmp:
            config["paths"] = {"repo_root": Path(tmp), "outputs": Path(tmp) / "outputs"}
            for name, filename in (("run_cross_validation", "cv_summary.json"),
                                   ("fit_final_model", "final_model_summary.json")):
                with self.subTest(entrypoint=name):
                    out = Path(tmp) / name
                    summary = getattr(mod, name)(X, y, ids, C, config, out)
                    json.dumps(summary, allow_nan=False)
                    with (out / filename).open(encoding="utf-8") as handle:
                        saved = json.load(handle)
                    self.assertEqual(saved["config"]["paths"]["repo_root"], str(Path(tmp)))
                    self.assertEqual(saved["config"]["paths"]["outputs"], str(Path(tmp) / "outputs"))
                    self.assertEqual(saved, summary)

    def test_final_model_saves_reproducible_parameters(self):
        mod = self.module("evaluation")
        X, y, C = self.data()
        ids = [str(i) for i in range(len(y))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "final"
            summary = mod.fit_final_model(X, y, ids, C, self.config(), out)
            json.dumps(summary, allow_nan=False)
            self.assertTrue((out / "B_hat.npy").is_file())
            self.assertTrue((out / "preprocessing.npz").is_file())
            self.assertEqual(summary["subject_ids"], ids)
            self.assertEqual(summary["response_scale"], "log")
            self.assertFalse(summary["independent_test_performance"])
            for axis in ("channel", "time", "frequency"):
                with (out / f"S_{axis}.json").open(encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), list)


if __name__ == "__main__":
    unittest.main()
