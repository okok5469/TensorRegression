"""Data-pairing and solver-boundary checks for the isolated advisor experiments."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace
import inspect
import unittest

import numpy as np

from numerical_simulation.data_generation import (
    TensorSimulationConfig,
    generate_simulation_dataset,
    iter_test_batches,
    tensor_signal_variance,
)
from tensor_abess.tensor.ops import tensor_inner


class AdvisorExperimentTests(unittest.TestCase):
    def api(self):
        name = "numerical_simulation.advisor_experiments_20261004.run"
        self.assertIsNotNone(
            importlib.util.find_spec(name),
            "The isolated coefficient-case runner has not been implemented.",
        )
        return importlib.import_module(name)

    def baseline(self):
        config = TensorSimulationConfig(
            n_train=35, n_test=25, dims=(5, 4, 5), sparsity=(2, 2, 2),
            snr=2.0, k_max=1, max_cycles=3, n_lambdas=4,
            cv_folds=2, max_iter=50, test_batch_size=10,
            tensor_abess_mode="adaptive",
        )
        return generate_simulation_dataset(config, seed=2026)

    def test_original_case_reproduces_every_original_array(self):
        api = self.api()
        baseline = self.baseline()
        rebuilt = api.build_case_dataset(baseline, "uniform_signed_snr2")
        np.testing.assert_array_equal(rebuilt.X_train, baseline.X_train)
        np.testing.assert_array_equal(rebuilt.y_train, baseline.y_train)
        np.testing.assert_array_equal(rebuilt.B_true, baseline.B_true)
        self.assertEqual(rebuilt.config, baseline.config)

    def test_coefficient_cases_share_training_predictors_support_and_noise(self):
        api = self.api()
        baseline = self.baseline()
        noise = baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)
        for name in api.CASE_NAMES:
            dataset = api.build_case_dataset(baseline, name)
            self.assertIs(dataset.X_train, baseline.X_train)
            for actual, expected in zip(dataset.supports_true, baseline.supports_true):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(
                dataset.y_train - tensor_inner(dataset.X_train, dataset.B_true),
                noise, rtol=0.0, atol=1e-13,
            )

    def test_nonzero_values_and_signs_match_the_coefficient_definition(self):
        api = self.api()
        baseline = self.baseline()
        support_mask = baseline.B_true != 0
        for name in api.CASE_NAMES:
            dataset = api.build_case_dataset(baseline, name)
            np.testing.assert_array_equal(dataset.B_true[~support_mask], 0.0)
            self.assertEqual(np.count_nonzero(dataset.B_true), int(support_mask.sum()))
            self.assertAlmostEqual(
                dataset.signal_variance,
                tensor_signal_variance(dataset.B_true, *dataset.covariances), places=10,
            )
            self.assertAlmostEqual(
                dataset.config.snr, dataset.signal_variance / dataset.config.sigma**2, places=10,
            )
        signed = api.build_case_dataset(baseline, "constant_signed_snr2")
        np.testing.assert_array_equal(np.sign(signed.B_true), np.sign(baseline.B_true))
        np.testing.assert_allclose(np.abs(signed.B_true[support_mask]), np.abs(signed.B_true[support_mask][0]))
        positive = api.build_case_dataset(baseline, "constant_positive_snr2")
        self.assertTrue(np.all(positive.B_true[support_mask] > 0))
        self.assertAlmostEqual(signed.config.snr, 2.0, places=12)
        self.assertAlmostEqual(positive.config.snr, 2.0, places=12)

    def test_snr_normalization_makes_positive_one_and_half_identical(self):
        api = self.api()
        baseline = self.baseline()
        first = api.normalized_positive_tensor(baseline, magnitude=1.0)
        second = api.normalized_positive_tensor(baseline, magnitude=0.5)
        np.testing.assert_array_equal(first, second)
        first_signal = tensor_inner(baseline.X_train, first)
        second_signal = tensor_inner(baseline.X_train, second)
        np.testing.assert_array_equal(first_signal, second_signal)
        self.assertTrue(api.verify_normalized_equivalence(baseline)["identical_coefficients"])
        self.assertTrue(api.verify_normalized_equivalence(baseline)["identical_training_response"])

    def test_literal_positive_one_and_half_keep_values_and_quadruple_snr(self):
        api = self.api()
        baseline = self.baseline()
        first = api.build_case_dataset(baseline, "constant_positive_1_literal")
        second = api.build_case_dataset(baseline, "constant_positive_05_literal")
        mask = baseline.B_true != 0
        np.testing.assert_array_equal(first.B_true[mask], 1.0)
        np.testing.assert_array_equal(second.B_true[mask], 0.5)
        self.assertAlmostEqual(first.config.snr / second.config.snr, 4.0, places=12)

    def test_existing_test_batches_keep_the_same_predictors_and_noise(self):
        api = self.api()
        baseline = self.baseline()
        reference = list(iter_test_batches(baseline, seed=1_002_026))
        for name in api.CASE_NAMES:
            dataset = api.build_case_dataset(baseline, name)
            batches = list(iter_test_batches(dataset, seed=1_002_026))
            for actual, expected in zip(batches, reference):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_allclose(
                    actual[1] - actual[2], expected[1] - expected[2],
                    rtol=0.0, atol=1e-13,
                )

    def test_known_size_solver_has_no_support_argument_and_returns_requested_sizes(self):
        api = self.api()
        baseline = self.baseline()
        parameters = set(inspect.signature(api.fit_tensor_method).parameters)
        self.assertEqual(parameters, {"X", "y", "config", "mode", "seed"})
        fixed = api.fit_tensor_method(
            baseline.X_train, baseline.y_train, baseline.config,
            mode="fixed_known_s", seed=2026,
        )
        self.assertEqual((fixed.s1, fixed.s2, fixed.s3), baseline.config.sparsity)
        self.assertEqual((len(fixed.S1), len(fixed.S2), len(fixed.S3)), baseline.config.sparsity)
        # Changing population labels cannot enter this solver-only interface.
        relabeled = replace(baseline, supports_true=(np.array([0, 4]), np.array([0, 3]), np.array([0, 4])))
        repeated = api.fit_tensor_method(
            relabeled.X_train, relabeled.y_train, relabeled.config,
            mode="fixed_known_s", seed=2026,
        )
        np.testing.assert_array_equal(fixed.B, repeated.B)


if __name__ == "__main__":
    unittest.main()
