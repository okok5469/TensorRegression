"""Deterministic software checks, not a classification performance study."""
import dataclasses
import copy
import json
import inspect
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tensor_abess.tensor.abess import tensor_abess
from tensor_abess_ovr.model import TensorABESSOvR, decide_scores, core_profile, to_jsonable
from tensor_abess_ovr.preprocessing import TensorStandardizer


def fixture():
    x = np.arange(30 * 8, dtype=np.float64).reshape(30, 2, 2, 2)
    x = np.sin(x * 0.137) + np.cos(x * 0.071)
    x[:, 0, 0, 0] = 0.1
    y = np.asarray(([90] * 5) + ([-7] * 10) + ([400] * 15))
    return x, y


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = fixture()
        cls.model = TensorABESSOvR(classes=[400, 90, -7], core_timeout=None).fit(cls.x, cls.y)

    def test_direct_core_full_state_and_independent_inner_product(self):
        m = self.model
        transformed = m.preprocessor_.transform(self.x)
        independent_scores = []
        for col, label in enumerate(m.classes_):
            response = (self.y == label).astype(float)
            direct = tensor_abess(transformed, response - response.mean(), **core_profile()["parameters"])
            self.assertEqual(to_jsonable(dataclasses.asdict(direct)), m.states_[col])
            independent_scores.append(response.mean() + np.einsum("nabc,abc->n", transformed, direct.B))
        np.testing.assert_allclose(m.decision_function(self.x), np.column_stack(independent_scores), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(m.class_prior_, [0.5, 1 / 6, 1 / 3])
        self.assertEqual(m.tie_priority_.tolist(), [0, 2, 1])

    def test_strict_constant_nonfinite_and_axis_mapping(self):
        axes = [{"name": "a", "coordinates": ["a0", "a1"]}, {"name": "b", "coordinates": ["b0", "b1"]}, {"name": "c", "coordinates": ["c0", "c1"]}]
        standard = TensorStandardizer(axes).fit(self.x)
        self.assertTrue(standard.constant_mask_[0, 0, 0])
        changed = self.x.copy()
        changed[:, 0, 0, 0] = 900
        self.assertTrue(np.all(standard.transform(changed)[:, 0, 0, 0] == 0))
        np.testing.assert_allclose(standard.mean_, self.x.mean(axis=0))
        expected = self.x.std(axis=0, ddof=0)
        expected[standard.constant_mask_] = 1
        np.testing.assert_array_equal(standard.scale_, expected)
        with self.assertRaisesRegex(ValueError, "axis"):
            standard.transform(self.x, axis_metadata=[axes[1], axes[0], axes[2]])
        changed[0, 1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            standard.transform(changed)
        with self.assertRaisesRegex(ValueError, "shape"):
            standard.transform(self.x[:, :1])

    def test_tie_priority_tolerance_and_column_alignment(self):
        scores = np.asarray([[1., 1., 1.], [0., 2., 2.], [1., 1. - 1e-15, 0.], [0., 1., 1. - 1e-10]])
        labels, ties = decide_scores(scores, np.asarray([400, 90, -7]), [2, 0, 1])
        np.testing.assert_array_equal(labels, [-7, -7, 400, 90])
        self.assertEqual(ties, 3)
        permutation = [1, 2, 0]
        aligned, _ = decide_scores(scores[:, permutation], np.asarray([400, 90, -7])[permutation], [1, 2, 0])
        np.testing.assert_array_equal(aligned, labels)
        with self.assertRaisesRegex(ValueError, "finite"):
            decide_scores(np.asarray([[np.inf, 0]]), [0, 1], [0, 1])

    def test_reencoding_preserves_semantic_scores_and_predictions(self):
        mapping = {400: "动作丙", 90: "动作甲", -7: "动作乙"}
        encoded = np.asarray([mapping[v] for v in self.y])
        m = TensorABESSOvR(classes=[mapping[v] for v in self.model.classes_], core_timeout=None).fit(self.x, encoded)
        np.testing.assert_allclose(m.decision_function(self.x), self.model.decision_function(self.x), rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(m.predict(self.x), [mapping[v] for v in self.model.predict(self.x)])

    def test_save_load_without_pickle_and_evaluation_isolation(self):
        before = self.model.coef_.copy()
        mean = self.model.preprocessor_.mean_.copy()
        self.model.predict(self.x * 100)
        np.testing.assert_array_equal(before, self.model.coef_)
        np.testing.assert_array_equal(mean, self.model.preprocessor_.mean_)
        with tempfile.TemporaryDirectory(prefix="模型 路径") as tmp:
            self.model.save(tmp)
            loaded = TensorABESSOvR.load(tmp)
            with np.load(Path(tmp) / "model.npz", allow_pickle=False) as z:
                self.assertFalse(any(z[name].dtype.hasobject for name in z.files))
            np.testing.assert_array_equal(loaded.decision_function(self.x), self.model.decision_function(self.x))
            np.testing.assert_array_equal(loaded.predict(self.x), self.model.predict(self.x))
            self.assertEqual(loaded.supports_, self.model.supports_)
            self.assertIn("code_identity", json.loads((Path(tmp) / "model.json").read_text(encoding="utf-8")))

    def test_input_errors_and_failed_class_invalidate_whole_fit(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            TensorABESSOvR(classes=[90, -7, 400, 8], core_timeout=None).fit(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "uniform"):
            TensorABESSOvR(core_timeout=None).fit(self.x[:2], [1, "a"])
        with self.assertRaisesRegex(ValueError, "sample"):
            TensorABESSOvR(core_timeout=None).fit(self.x, self.y, sample_ids=["repeat"] * 30)
        bad = TensorABESSOvR(classes=[400, 90, -7], core_timeout=None)
        with patch("tensor_abess_ovr.model._fit_one", side_effect=RuntimeError("intentional class failure")):
            with self.assertRaisesRegex(RuntimeError, "400"):
                bad.fit(self.x, self.y)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            bad.predict(self.x)
        self.assertEqual(bad.fit_diagnostics_[-1]["status"], "FAILED")
        self.assertEqual(bad.fit_diagnostics_[-1]["class_label"], 400)

    def test_process_bounded_calls_match_direct_and_timeout_invalidates_fit(self):
        bounded = TensorABESSOvR(classes=[400, 90, -7], core_timeout=30).fit(self.x, self.y)
        self.assertEqual(bounded.states_, self.model.states_)
        np.testing.assert_array_equal(bounded.decision_function(self.x), self.model.decision_function(self.x))
        short = TensorABESSOvR(classes=[400, 90, -7], core_timeout=1e-6)
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            short.fit(self.x, self.y)
        self.assertEqual(short.fit_diagnostics_[-1]["error_type"], "TimeoutError")
        self.assertFalse(short._fitted)

    def test_binary_corruption_is_detected_before_model_load(self):
        with tempfile.TemporaryDirectory(prefix="模型 完整性") as tmp:
            self.model.save(tmp)
            binary = Path(tmp) / "model.npz"
            binary.write_bytes(binary.read_bytes() + b"intentional corruption")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                TensorABESSOvR.load(tmp)

    def test_nonfinite_core_coefficients_and_mixed_object_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "uniform"):
            TensorABESSOvR(core_timeout=None).fit(self.x[:2], np.asarray([1, "a"], dtype=object))
        with self.assertRaisesRegex(ValueError, "missing"):
            TensorABESSOvR(core_timeout=None).fit(self.x[:2], np.asarray(["a", None], dtype=object))
        with self.assertRaisesRegex(ValueError, "uniform"):
            TensorABESSOvR(core_timeout=None).fit(self.x[:2], [0, np.nan])
        state = copy.deepcopy(self.model.states_[0])
        state["B"][0][0][0] = float("nan")
        bad = TensorABESSOvR(classes=[400, 90, -7], core_timeout=None)
        with patch("tensor_abess_ovr.model._fit_one", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                bad.fit(self.x, self.y)
        self.assertFalse(bad._fitted)
        self.assertEqual(bad.fit_diagnostics_[0]["class_label"], 400)

    def test_dataset_deadline_caps_each_class_and_stops_before_expired_class(self):
        self.assertIn("deadline", inspect.signature(TensorABESSOvR.fit).parameters,
                      "Missing absolute dataset deadline in per-class timeout control")
        expired = TensorABESSOvR(classes=[400, 90, -7], core_timeout=900)
        with patch("tensor_abess_ovr.model._bounded_fit") as fit_one:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                expired.fit(self.x, self.y, deadline=time.monotonic()-1)
        fit_one.assert_not_called()
        self.assertEqual(expired.fit_diagnostics_[0]["error_type"], "TimeoutError")
        observed = []

        def fake_fit(X, z, parameters, timeout, deadline=None):
            observed.append(timeout)
            return self.model.states_[len(observed)-1]

        partial = TensorABESSOvR(classes=[400, 90, -7], core_timeout=900)
        with patch("tensor_abess_ovr.model.time.monotonic", side_effect=[100., 103., 111.]), patch("tensor_abess_ovr.model._bounded_fit", side_effect=fake_fit):
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                partial.fit(self.x, self.y, deadline=110.)
        self.assertEqual(observed, [10., 7.])
        self.assertEqual(partial.fit_diagnostics_[-1]["class_label"], -7)
        self.assertEqual(partial.fit_diagnostics_[-1]["error_type"], "TimeoutError")
        self.assertFalse(partial._fitted)


if __name__ == "__main__":
    unittest.main()
