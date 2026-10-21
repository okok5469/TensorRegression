"""Tests for the isolated audit loader, not changes to the BCD algorithm."""
from __future__ import annotations

import tempfile
import ast
import sys
import unittest
from pathlib import Path

import warm_audit


class AuditHarnessTests(unittest.TestCase):
    def test_plain_functions_retain_assertions_without_unused_pytest_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "example.py"
            source.write_text("import pytest\ndef test_failure():\n    assert 1 == 2\n", encoding="utf-8")
            module = warm_audit.load_plain_tests(source, "example_unused")
            with self.assertRaises(AssertionError):
                module.test_failure()

    def test_loader_rejects_pytest_usage_rather_than_emulating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "example.py"
            source.write_text("import pytest\ndef test_value():\n    assert 1 == pytest.approx(1)\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pytest"):
                warm_audit.load_plain_tests(source, "example_used")

    def test_manifest_detects_one_file_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            first = warm_audit.source_manifest(root)
            (root / "a.py").write_text("a = 2\n", encoding="utf-8")
            second = warm_audit.source_manifest(root)
            self.assertEqual(set(first), {"a.py"})
            self.assertNotEqual(first, second)

    def test_failure_or_error_disables_performance_gate(self):
        def failing_assertion():
            self.assertEqual(1, 2)
        result = unittest.TestResult()
        unittest.FunctionTestCase(failing_assertion).run(result)
        report = warm_audit.summarize_result(result)
        self.assertFalse(report["correctness_passed"])
        self.assertFalse(report["performance_allowed"])
        self.assertEqual(report["failures"], 1)
        self.assertIn("AssertionError", report["failure_details"][0]["traceback"])


@unittest.skipUnless("tensor_abess.tensor.bcd" in sys.modules, "requires isolated audit runner")
class WarmCorrectnessTests(unittest.TestCase):
    def test_invalid_initial_supports_are_rejected(self):
        import numpy as np
        from tensor_abess.tensor.bcd import tensor_bcd
        X = np.eye(8).reshape(8, 2, 2, 2)
        for invalid in (np.array([-1]), np.array([2]), np.array([0, 0])):
            with self.subTest(support=invalid.tolist()):
                with self.assertRaises(ValueError):
                    tensor_bcd(X, np.ones(8), 1, 1, 1, max_cycles=0,
                               S1_init=invalid, S2_init=np.array([0]), S3_init=np.array([0]))

    def test_fixed_and_adaptive_call_signatures_match_v52(self):
        import tensor_abess.tensor.bcd as bcd
        root = Path(__file__).resolve().parents[2]
        baseline = ast.parse((root / "doc/algorithm_versions/v5.2-search-final/src/tensor_abess/tensor/bcd.py").read_text(encoding="utf-8"))
        current = ast.parse(Path(bcd.__file__).read_text(encoding="utf-8"))
        for name in ("tensor_bcd", "tensor_bcd_adaptive"):
            with self.subTest(function=name):
                base_args = next(node.args for node in baseline.body if isinstance(node, ast.FunctionDef) and node.name == name)
                actual_args = next(node.args for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(ast.dump(base_args), ast.dump(actual_args))

    def test_adaptive_sic_smoke_is_ols_consistent(self):
        import numpy as np
        from tensor_abess.tensor.bcd import tensor_bcd_adaptive
        from tensor_abess.tensor.ops import tensor_inner, build_tensor_design
        from tensor_abess.tensor.sic import compute_tensor_sic_from_rss
        rng = np.random.default_rng(81016)
        X = rng.normal(size=(30, 3, 2, 2))
        y = rng.normal(size=30)
        result = tensor_bcd_adaptive(X, y, s1_max=2, s2_max=2, s3_max=2,
                                    S1_init=np.array([0]), S2_init=np.array([0]),
                                    S3_init=np.array([0]), max_cycles=3, information_criterion="sic")
        Z = build_tensor_design(X, result.S1, result.S2, result.S3)
        predicted = tensor_inner(X, result.B)
        np.testing.assert_allclose(predicted, Z @ np.linalg.lstsq(Z, y, rcond=None)[0], atol=1e-10)
        rss = float(np.sum((y - predicted) ** 2))
        self.assertAlmostEqual(result.rss, rss, places=10)
        self.assertAlmostEqual(result.sic, compute_tensor_sic_from_rss(rss, result.s1, result.s2, result.s3, *X.shape), places=10)
        self.assertTrue(np.isfinite(result.B).all())

    def test_all_a2_admissible_size_candidates_are_preserved(self):
        from tensor_abess.tensor.bcd import _candidate_size_triples
        expected = [(a, b, c) for a in range(1, 9) for b in range(1, 5)
                    for c in range(1, 4) if a * b * c <= 70]
        self.assertEqual(_candidate_size_triples(8, 4, 3, 70), expected)

    def test_same_seed_reproduces_numerical_output_without_diagnostic_fields(self):
        import numpy as np
        from tensor_abess.tensor.bcd import tensor_bcd
        rng = np.random.default_rng(81017)
        X, y = rng.normal(size=(30, 3, 2, 2)), rng.normal(size=30)
        results = [tensor_bcd(X, y, 2, 1, 1, init_strategy="random", random_state=19,
                              max_cycles=4) for _ in range(2)]
        for field in ("B", "S1", "S2", "S3", "loss_history", "loss"):
            np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))


if __name__ == "__main__":
    unittest.main()
