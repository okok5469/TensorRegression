"""Task 9 fixed-size extra-start contract checks, plus unchanged Cold tests."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class PureWarmTests(unittest.TestCase):
    def __init__(self, methodName="runTest", *, artifact_directory=None):
        super().__init__(methodName)
        self.artifact_directory = artifact_directory

    def setUp(self):
        try:
            import pure_warm
        except ModuleNotFoundError as error:
            self.assertEqual(error.name, "pure_warm")
            pure_warm = None
        self.assertIsNotNone(pure_warm, "Missing fixed-size Cold + extra-start behavior")
        self.subject = pure_warm

    @staticmethod
    def fixture():
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.zeros(8)
        y[0], y[7] = 1.0, 2.0
        return X, y

    def fit(self, cold_index=0, warm_index=1, **kwargs):
        X, y = self.fixture()
        initial = {f"S{i}_init": np.array([cold_index]) for i in range(1, 4)}
        result = self.subject.fixed_cold_plus_warm(
            X, y, (1, 1, 1), warm_supports=tuple(np.array([warm_index]) for _ in range(3)),
            max_cycles=0, **initial, **kwargs)
        return X, y, result

    def test_warm_can_win_without_discarding_cold_result(self):
        from tensor_abess.tensor.bcd import tensor_bcd
        X, y, result = self.fit()
        original = tensor_bcd(X, y, 1, 1, 1, max_cycles=0,
                              S1_init=np.array([0]), S2_init=np.array([0]), S3_init=np.array([0]))
        self.assertEqual(result.selected_route, "warm")
        self.assertIs(result.selected, result.warm)
        for name in ("B", "S1", "S2", "S3"):
            np.testing.assert_array_equal(getattr(result.cold, name), getattr(original, name))
        self.assertEqual(result.cold.visited_loss_history, original.visited_loss_history)
        self.assertEqual(result.cold.accepted_support_history, original.accepted_support_history)
        self.assertLess(result.selected.loss, result.cold.loss)

    def test_worse_warm_does_not_replace_cold(self):
        _, _, result = self.fit(cold_index=1, warm_index=0)
        self.assertIs(result.selected, result.cold)
        self.assertEqual(result.selected_route, "cold")

    def test_exact_rss_tie_prefers_cold(self):
        _, _, result = self.fit(cold_index=1, warm_index=1)
        self.assertEqual(result.selected_route, "cold")

    def test_training_only_trim_pad_ties_are_deterministic(self):
        X, y = np.zeros((10, 4, 3, 2)), np.zeros(10)
        source = (np.array([3, 1, 2]), np.array([2]), np.array([1]))
        before = tuple(s.copy() for s in source)
        actual = self.subject.align_warm_supports(X, y, (2, 2, 2), source)
        for support, expected in zip(actual, ([1, 2], [0, 2], [0, 1])):
            np.testing.assert_array_equal(support, expected)
        for old, now in zip(before, source):
            np.testing.assert_array_equal(old, now)

    def test_invalid_warm_sources_raise_without_silent_fallback(self):
        X, y = self.fixture()
        for invalid in (np.array([-1]), np.array([2]), np.array([0, 0]), np.array([0.5])):
            with self.subTest(source=invalid.tolist()):
                with self.assertRaises(ValueError):
                    self.subject.fixed_cold_plus_warm(
                        X, y, (1, 1, 1), warm_supports=(invalid, np.array([0]), np.array([0])),
                        max_cycles=0)

    def test_nonfinite_inputs_are_rejected(self):
        X, y = self.fixture()
        X[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.subject.fixed_cold_plus_warm(
                X, y, (1, 1, 1), warm_supports=(np.array([0]),) * 3)

    def test_independent_ols_and_repeatability(self):
        from tensor_abess.tensor.ops import build_tensor_design, tensor_inner
        rng = np.random.default_rng(39)
        X, y = rng.normal(size=(45, 4, 3, 2)), rng.normal(size=45)
        before = X.copy(), y.copy()
        kwargs = dict(warm_supports=(np.array([1, 3]), np.array([2]), np.array([1])),
                      init_strategy="random", random_state=72, max_cycles=10)
        first = self.subject.fixed_cold_plus_warm(X, y, (2, 1, 1), **kwargs)
        second = self.subject.fixed_cold_plus_warm(X, y, (2, 1, 1), **kwargs)
        for route in ("cold", "warm"):
            a, b = getattr(first, route), getattr(second, route)
            for name in ("B", "S1", "S2", "S3"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            self.assertEqual(a.visited_loss_history, b.visited_loss_history)
            self.assertEqual(a.stop_reason, b.stop_reason)
            Z = build_tensor_design(X, a.S1, a.S2, a.S3)
            predicted = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            np.testing.assert_allclose(tensor_inner(X, a.B), predicted, rtol=1e-10, atol=1e-11)
            self.assertAlmostEqual(a.loss, min(a.visited_loss_history), places=12)
        self.assertEqual(first.selected_route, second.selected_route)
        self.assertLessEqual(first.selected.loss, first.cold.loss)
        np.testing.assert_array_equal(X, before[0])
        np.testing.assert_array_equal(y, before[1])

    def test_extra_route_retains_finite_and_cycle_guards(self):
        X = np.eye(8).reshape(8, 2, 2, 2)
        y = np.ones(8)
        calls = []
        def fixed_proposal(y, Z, gi, s, **kwargs):
            calls.append(1)
            return SimpleNamespace(active_groups=np.array([[1, 1, 1, 0, 0, 0][(len(calls)-1) % 6]]))
        with patch("tensor_abess.tensor.bcd.group_bess_fix", side_effect=fixed_proposal):
            result = self.subject.fixed_cold_plus_warm(
                X, y, (1, 1, 1), warm_supports=(np.array([1]),) * 3,
                S1_init=np.array([0]), S2_init=np.array([0]), S3_init=np.array([0]), max_cycles=12)
        for candidate in (result.cold, result.warm):
            self.assertLess(candidate.n_cycles, 12)
            self.assertIn(candidate.stop_reason, ("unchanged_support", "exploration_cycle"))
            self.assertAlmostEqual(candidate.loss, min(candidate.visited_loss_history), places=14)
        y = np.zeros(8)
        y[7] = 1e200
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(FloatingPointError):
                self.subject.fixed_cold_plus_warm(
                    X, y, (1, 1, 1), warm_supports=(np.array([0]),) * 3,
                    S1_init=np.array([0]), S2_init=np.array([0]), S3_init=np.array([0]), max_cycles=0)

    def test_historical_seed2026_cold_retention_and_recovery(self):
        from numerical_simulation.main import build_profile
        from numerical_simulation.data_generation import generate_simulation_dataset
        from tensor_abess.utils.init import _mode_slice_scores
        config, _, _ = build_profile("full")
        dataset = generate_simulation_dataset(config, seed=2026)
        scores = tuple(_mode_slice_scores(dataset.X_train, dataset.y_train, m) for m in (1, 2, 3))
        initial = tuple(np.sort(np.argsort(-score)[:s]) for score, s in zip(scores, config.sparsity))
        extra = tuple(np.sort(np.argsort(-score)[1:s+1]) for score, s in zip(scores, config.sparsity))
        result = self.subject.fixed_cold_plus_warm(
            dataset.X_train, dataset.y_train, config.sparsity, warm_supports=extra,
            S1_init=initial[0], S2_init=initial[1], S3_init=initial[2],
            max_cycles=config.max_cycles, k_max=config.k_max)
        for candidate in (result.cold, result.selected):
            for name, truth in zip(("S1", "S2", "S3"), dataset.supports_true):
                self.assertTrue(np.array_equal(getattr(candidate, name), truth))
        self.assertAlmostEqual(result.cold.best_rss, 154.4719575096973, places=8)
        report = {"seed": 2026, "training_only_extra_source": "marginal rank positions 2 through s+1",
                  "initial": [s.tolist() for s in initial], "extra": [s.tolist() for s in extra],
                  "truth": [s.tolist() for s in dataset.supports_true],
                  "X_sha256": hashlib.sha256(dataset.X_train.tobytes()).hexdigest(),
                  "y_sha256": hashlib.sha256(dataset.y_train.tobytes()).hexdigest(),
                  "selected_route": result.selected_route, "timing": result.timing}
        for route in ("cold", "warm"):
            candidate = getattr(result, route)
            report[route] = {"rss": candidate.best_rss,
                             "supports": [getattr(candidate, name).tolist() for name in ("S1", "S2", "S3")],
                             "visited_loss_history": candidate.visited_loss_history,
                             "loss_history": candidate.loss_history, "stop_reason": candidate.stop_reason,
                             "cycles": candidate.n_cycles, "best_iteration": candidate.best_iteration}
        if self.artifact_directory is not None:
            with (self.artifact_directory / "seed2026.json").open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("red", "green"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output or root / "doc/taskcard9/warm"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forbidden = [output / f"{args.phase}.json", output / f"{args.phase}.log"]
    if args.phase == "green":
        forbidden.append(output / "seed2026.json")
    if any(path.exists() for path in forbidden):
        raise FileExistsError("Use --output with a fresh directory; existing evidence is never overwritten")
    source = root / "doc/algorithm_versions/v5.2-search-final/src"
    sys.path[:0] = [str(source), str(root), str(Path(__file__).resolve().parent)]
    import tensor_abess.tensor.bcd as bcd
    assert Path(bcd.__file__).resolve() == source / "tensor_abess/tensor/bcd.py"
    helper = load_module(root / "scripts/taskcard8/warm_audit.py", "old_warm_adapter")
    manifest = helper.source_manifest(source / "tensor_abess")
    assert len(manifest) == 32
    suite = unittest.TestSuite()
    if args.phase == "green":
        plain = helper.load_plain_tests(root / "tests/test_tensor_bcd.py", "unchanged_cold")
        suite.addTests(unittest.FunctionTestCase(fn, description=name)
                       for name, fn in vars(plain).items() if name.startswith("test_") and inspect.isfunction(fn))
        original = load_module(root / "tests/test_bcd_fix_regression.py", "original_fixed_guards")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(original))
        historical = load_module(root / "tests/test_numerical_simulation.py", "original_historical")
        suite.addTest(historical.NumericalSimulationTests(
            "test_full_profile_fixed_size_bcd_can_recover_from_marginal_warm_start"))
    suite.addTests(PureWarmTests(name, artifact_directory=output)
                   for name in unittest.defaultTestLoader.getTestCaseNames(PureWarmTests))
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    loaded = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
              if name.startswith("tensor_abess") and getattr(module, "__file__", None)}
    assert all(Path(path).is_relative_to(source) for path in loaded.values())
    assert helper.source_manifest(source / "tensor_abess") == manifest
    report = helper.summarize_result(result)
    import scipy
    report.update({"phase": args.phase, "elapsed_seconds": elapsed, "core_sha256": manifest,
                   "loaded_core_modules": loaded, "executable": sys.executable, "python": sys.version,
                   "scope": "fixed-size extra-start prototype; full adaptive and performance NOT_RUN",
                   "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                   "thread_settings": {name: os.environ.get(name) for name in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   "source_and_test_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__).resolve(), Path(__file__).with_name("pure_warm.py"),
                                     root / "scripts/taskcard8/warm_audit.py", root / "tests/test_tensor_bcd.py",
                                     root / "tests/test_bcd_fix_regression.py", root / "tests/test_numerical_simulation.py")
                        if path.exists()},
                   "fixed_scope_correctness_passed": result.wasSuccessful(),
                   "full_adaptive_correctness": "NOT_RUN",
                   "full_pipeline_performance_allowed": False,
                   "warm_performance_executed": False})
    with (output / f"{args.phase}.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output / f"{args.phase}.log").open("x", encoding="utf-8") as handle:
        handle.write(stream.getvalue())
    print(json.dumps({"phase": args.phase, "tests": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "elapsed_seconds": elapsed}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
