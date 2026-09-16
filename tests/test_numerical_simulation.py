from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from numerical_simulation.consistency_checks import run_consistency_checks
from numerical_simulation.data_generation import (
    TensorSimulationConfig,
    generate_simulation_dataset,
    tensor_signal_variance,
)
from numerical_simulation.main import build_profile
from numerical_simulation.methods import run_tensor_abess_method, tensor_glasso
from numerical_simulation.runner import (
    ExperimentSpec,
    ExperimentSuiteConfig,
    run_experiment_suite,
    run_single_repetition,
)
from tensor_abess.tensor.bcd import tensor_bcd_adaptive
from tensor_abess.utils.init import _mode_slice_scores, tensor_init_glasso


def _small_config(**overrides) -> TensorSimulationConfig:
    base = dict(
        n_train=18,
        n_test=24,
        dims=(4, 3, 2),
        sparsity=(2, 1, 1),
        rho=(0.2, 0.2, 0.2),
        snr=1.5,
        support_shape="continuous",
        sigma=1.0,
        k_max=3,
        max_cycles=3,
        n_lambdas=4,
        cv_folds=2,
        max_iter=120,
        tol=1e-5,
        random_state=2026,
        test_batch_size=12,
    )
    base.update(overrides)
    return TensorSimulationConfig(**base)


def _clear_tree(path: Path) -> None:
    if path.exists():
        for subpath in sorted(path.rglob("*"), reverse=True):
            if subpath.is_file():
                subpath.unlink()
            elif subpath.is_dir():
                subpath.rmdir()
        path.rmdir()


class NumericalSimulationTests(unittest.TestCase):
    def test_full_profile_defaults_to_ten_repetitions(self) -> None:
        _, n_reps, _ = build_profile("full")
        self.assertEqual(n_reps, 10)

    def test_generate_dataset_matches_requested_structure_and_snr(self) -> None:
        config = _small_config()
        dataset = generate_simulation_dataset(config, seed=123)

        self.assertEqual(dataset.X_train.shape, (config.n_train, *config.dims))
        self.assertEqual(dataset.y_train.shape, (config.n_train,))
        self.assertEqual(dataset.B_true.shape, config.dims)
        self.assertEqual(tuple(len(s) for s in dataset.supports_true), config.sparsity)

        snr = tensor_signal_variance(dataset.B_true, *dataset.covariances) / (config.sigma ** 2)
        self.assertAlmostEqual(snr, config.snr, places=6)

    def test_consistency_checks_pass_on_small_random_example(self) -> None:
        report = run_consistency_checks(seed=7)
        self.assertTrue(report.inner_product_ok)
        self.assertTrue(report.mode_design_ok)
        self.assertLess(report.max_inner_product_error, 1e-10)
        self.assertLess(report.max_mode_design_error, 1e-10)

    def test_tensor_glasso_produces_finite_rectangular_estimate(self) -> None:
        config = _small_config()
        dataset = generate_simulation_dataset(config, seed=321)

        result = tensor_glasso(
            dataset.X_train,
            dataset.y_train,
            random_state=11,
            n_lambdas=config.n_lambdas,
            cv_folds=config.cv_folds,
            max_cycles=config.max_cycles,
            max_iter=config.max_iter,
            tol=config.tol,
        )

        self.assertEqual(result.B.shape, dataset.B_true.shape)
        self.assertEqual(result.S1.ndim, 1)
        self.assertEqual(result.S2.ndim, 1)
        self.assertEqual(result.S3.ndim, 1)
        self.assertEqual(result.s1, len(result.S1))
        self.assertEqual(result.s2, len(result.S2))
        self.assertEqual(result.s3, len(result.S3))
        self.assertGreaterEqual(result.n_cycles, 1)
        self.assertTrue(np.isfinite(result.loss))
        self.assertTrue(np.isfinite(result.rss))

    def test_tensor_bcd_adaptive_respects_modewise_sparsity_caps(self) -> None:
        config = TensorSimulationConfig(
            n_train=60,
            n_test=80,
            dims=(8, 6, 8),
            sparsity=(2, 2, 2),
            rho=(0.3, 0.3, 0.3),
            snr=2.0,
            support_shape="continuous",
            sigma=1.0,
            k_max=5,
            max_cycles=4,
            n_lambdas=5,
            cv_folds=2,
            max_iter=120,
            tol=1e-5,
            random_state=2026,
            test_batch_size=40,
        )
        dataset = generate_simulation_dataset(config, seed=2026)
        init = tensor_init_glasso(
            dataset.X_train,
            dataset.y_train,
            c=2.0,
            criterion="cv",
            n_lambdas=config.n_lambdas,
            cv_folds=config.cv_folds,
            max_iter=config.max_iter,
            tol=config.tol,
        )
        result = tensor_bcd_adaptive(
            dataset.X_train,
            dataset.y_train,
            s1_max=init.s1_max,
            s2_max=init.s2_max,
            s3_max=init.s3_max,
            S1_init=init.S1_init,
            S2_init=init.S2_init,
            S3_init=init.S3_init,
            max_cycles=config.max_cycles,
            k_max=config.k_max,
        )

        self.assertLessEqual(len(result.S1), init.s1_max)
        self.assertLessEqual(len(result.S2), init.s2_max)
        self.assertLessEqual(len(result.S3), init.s3_max)

    def test_full_profile_tensor_init_keeps_true_sparsity_in_search_space(self) -> None:
        config, _, _ = build_profile("full")
        dataset = generate_simulation_dataset(config, seed=2026)

        init = tensor_init_glasso(
            dataset.X_train,
            dataset.y_train,
            c=config.abess_init_c,
            criterion=config.abess_init_criterion,
            n_lambdas=config.n_lambdas,
            cv_folds=config.cv_folds,
            max_iter=config.max_iter,
            tol=config.tol,
        )

        self.assertGreaterEqual(init.s1_max, config.sparsity[0])
        self.assertGreaterEqual(init.s2_max, config.sparsity[1])
        self.assertGreaterEqual(init.s3_max, config.sparsity[2])
        self.assertNotEqual(
            (init.S1_init.tolist(), init.S2_init.tolist(), init.S3_init.tolist()),
            ([0], [0], [0]),
        )

    def test_full_profile_tensor_init_keeps_true_sparsity_in_search_space_for_weak_init(self) -> None:
        config, _, _ = build_profile("full")
        dataset = generate_simulation_dataset(config, seed=2028)

        init = tensor_init_glasso(
            dataset.X_train,
            dataset.y_train,
            c=config.abess_init_c,
            criterion=config.abess_init_criterion,
            n_lambdas=config.n_lambdas,
            cv_folds=config.cv_folds,
            max_iter=config.max_iter,
            tol=config.tol,
        )

        self.assertGreaterEqual(init.s1_max, config.sparsity[0])
        self.assertGreaterEqual(init.s2_max, config.sparsity[1])
        self.assertGreaterEqual(init.s3_max, config.sparsity[2])

    def test_full_profile_fixed_size_bcd_can_recover_from_marginal_warm_start(self) -> None:
        config, _, _ = build_profile("full")
        dataset = generate_simulation_dataset(config, seed=2026)
        warm_supports = []
        for mode, s in enumerate(config.sparsity, start=1):
            scores = _mode_slice_scores(dataset.X_train, dataset.y_train, mode)
            warm_supports.append(np.sort(np.argsort(-scores)[:s]))

        from tensor_abess.tensor.bcd import tensor_bcd

        result = tensor_bcd(
            dataset.X_train,
            dataset.y_train,
            *config.sparsity,
            max_cycles=config.max_cycles,
            k_max=config.k_max,
            S1_init=warm_supports[0],
            S2_init=warm_supports[1],
            S3_init=warm_supports[2],
        )

        self.assertTrue(np.array_equal(result.S1, dataset.supports_true[0]))
        self.assertTrue(np.array_equal(result.S2, dataset.supports_true[1]))
        self.assertTrue(np.array_equal(result.S3, dataset.supports_true[2]))

    def test_full_profile_simulation_tensor_abess_avoids_singleton_collapse(self) -> None:
        config, _, _ = build_profile("full")
        dataset = generate_simulation_dataset(config, seed=2026)

        result = run_tensor_abess_method(
            dataset.X_train,
            dataset.y_train,
            config,
            random_state=2026,
        )

        self.assertGreaterEqual(result.s1 * result.s2 * result.s3, 12)

    def test_tensor_abess_method_can_run_adaptive_mode(self) -> None:
        config = _small_config(
            tensor_abess_mode="adaptive",
            msic_refine_alpha=1.7,
            msic_refine_gamma=0.9,
            msic_refine_one_se_scale=1.5,
        )
        X = np.ones((config.n_train, *config.dims), dtype=float)
        y = np.ones(config.n_train, dtype=float)
        seen = {}

        class FakeResult:
            B = np.zeros(config.dims, dtype=float)
            S1 = np.array([0])
            S2 = np.array([0])
            S3 = np.array([0])
            s1 = 1
            s2 = 1
            s3 = 1
            loss = 0.0
            rss = 0.0
            sic = 0.0
            bcd_result = None

        def fake_tensor_abess_core(*args, **kwargs):
            seen.update(kwargs)
            return FakeResult()

        with patch("numerical_simulation.methods.tensor_abess_core", fake_tensor_abess_core):
            run_tensor_abess_method(X, y, config, random_state=2026)

        self.assertIsNone(seen["target_sparsity"])
        self.assertEqual(seen["adaptive_criterion"], "msic")
        self.assertEqual(seen["msic_refine_alpha"], config.msic_refine_alpha)
        self.assertEqual(seen["msic_refine_gamma"], config.msic_refine_gamma)
        self.assertEqual(seen["msic_refine_one_se_scale"], config.msic_refine_one_se_scale)

    def test_run_single_repetition_returns_all_methods_and_metrics(self) -> None:
        config = _small_config()
        rows = run_single_repetition(config, replicate_index=0, seed=999)

        methods = {row["method"] for row in rows}
        self.assertEqual(
            methods,
            {"Tensor-ABESS", "Tensor-GLasso", "Vector Lasso", "Oracle OLS"},
        )

        for row in rows:
            self.assertTrue(math.isfinite(row["mode_f1_macro"]))
            self.assertTrue(math.isfinite(row["element_f1"]))
            self.assertTrue(math.isfinite(row["ree"]))
            self.assertTrue(math.isfinite(row["pmse"]))
            self.assertTrue(math.isfinite(row["runtime_sec"]))

    def test_experiment_suite_writes_tables_and_plots(self) -> None:
        tmp_path = Path(self.id().replace(".", "_"))
        _clear_tree(tmp_path)
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: _clear_tree(tmp_path))

        base_config = _small_config(
            n_train=16,
            n_test=20,
            dims=(4, 3, 2),
            sparsity=(2, 1, 1),
            n_lambdas=3,
            max_cycles=2,
            max_iter=80,
        )
        suite = ExperimentSuiteConfig(
            base_config=base_config,
            output_dir=tmp_path,
            n_reps=1,
            experiments=(ExperimentSpec(name="snr", parameter="snr", values=(0.5, 1.5)),),
        )

        artifacts = run_experiment_suite(suite)

        self.assertTrue(artifacts.raw_results_path.exists())
        self.assertTrue(artifacts.summary_results_path.exists())
        self.assertTrue(artifacts.consistency_report_path.exists())
        self.assertTrue(artifacts.plot_paths)
        self.assertTrue(all(path.exists() for path in artifacts.plot_paths))

    def test_experiment_suite_resume_skips_completed_repetitions(self) -> None:
        tmp_path = Path(self.id().replace(".", "_"))
        _clear_tree(tmp_path)
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: _clear_tree(tmp_path))

        base_config = _small_config(
            n_train=16,
            n_test=20,
            dims=(4, 3, 2),
            sparsity=(2, 1, 1),
            n_lambdas=3,
            max_cycles=2,
            max_iter=80,
        )
        suite = ExperimentSuiteConfig(
            base_config=base_config,
            output_dir=tmp_path,
            n_reps=1,
            experiments=(ExperimentSpec(name="snr", parameter="snr", values=(0.5, 1.5)),),
            resume=False,
        )
        artifacts_first = run_experiment_suite(suite)

        resumed_suite = ExperimentSuiteConfig(
            base_config=base_config,
            output_dir=tmp_path,
            n_reps=1,
            experiments=(ExperimentSpec(name="snr", parameter="snr", values=(0.5, 1.5)),),
            resume=True,
        )
        artifacts_second = run_experiment_suite(resumed_suite)

        first_keys = {
            (row["experiment"], row["scenario_value"], row["replicate"], row["method"])
            for row in artifacts_first.raw_rows
        }
        second_keys = [
            (row["experiment"], row["scenario_value"], row["replicate"], row["method"])
            for row in artifacts_second.raw_rows
        ]

        self.assertEqual(len(second_keys), len(first_keys))
        self.assertEqual(set(second_keys), first_keys)

    def test_resume_preserves_completed_batches_from_other_experiments(self) -> None:
        tmp_path = Path(self.id().replace(".", "_"))
        _clear_tree(tmp_path)
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: _clear_tree(tmp_path))

        base_config = _small_config(
            n_train=16,
            n_test=20,
            dims=(4, 3, 2),
            sparsity=(2, 1, 1),
            n_lambdas=3,
            max_cycles=2,
            max_iter=80,
        )
        snr_suite = ExperimentSuiteConfig(
            base_config=base_config,
            output_dir=tmp_path,
            n_reps=1,
            experiments=(ExperimentSpec(name="snr", parameter="snr", values=(0.5,)),),
            resume=False,
        )
        run_experiment_suite(snr_suite)

        sample_suite = ExperimentSuiteConfig(
            base_config=base_config,
            output_dir=tmp_path,
            n_reps=1,
            experiments=(ExperimentSpec(name="sample_size", parameter="n_train", values=(20,)),),
            resume=True,
        )
        artifacts = run_experiment_suite(sample_suite)

        experiments = {row["experiment"] for row in artifacts.raw_rows}
        self.assertIn("snr", experiments)
        self.assertIn("sample_size", experiments)


if __name__ == "__main__":
    unittest.main()
