"""Small mathematical checks for the descriptive paired summary, never fits data."""
import importlib.util
from pathlib import Path
import unittest


class DescriptiveChecks(unittest.TestCase):
    def load(self):
        path = Path(__file__).with_name("attribution.py")
        self.assertTrue(path.exists(), "Paired descriptive analysis has not been implemented")
        spec = importlib.util.spec_from_file_location("task9_attribution", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_tail_contributions_keep_negative_offsets(self):
        result = self.load().distribution([-2., 0., 1., 3., 8.])
        self.assertEqual(result["n"], 5)
        self.assertEqual((result["improved"], result["tied"], result["worsened"]), (1, 1, 3))
        self.assertEqual(result["median"], 1.)
        self.assertEqual(result["net_sum"], 10.)
        self.assertEqual(result["gross_positive_sum"], 12.)
        self.assertEqual(result["top5_positive_share_of_net"], 1.2)

    def test_same_q_is_not_same_shape_or_support(self):
        module = self.load()
        left = [[0, 1], [0, 1], [0, 1, 2]]
        same_q_other_shape = [[0], [0, 1, 2], [0, 1, 2, 3]]
        same_shape_other_support = [[1, 2], [0, 1], [0, 1, 2]]
        self.assertEqual(module.support_relation(left, same_q_other_shape), "different_shape")
        self.assertEqual(module.support_relation(left, same_shape_other_support), "same_shape_different_support")
        self.assertEqual(module.support_relation(left, left), "same_support")

    def test_fresh_output_keeps_report_inside_fresh_directory(self):
        module = self.load()
        self.assertTrue(hasattr(module, "report_path"), "Report output isolation has not been implemented")
        root = Path("test_repository").resolve()
        original = root / "doc/taskcard9/attribution"
        fresh = root / "doc/taskcard9/fresh_audit"
        self.assertEqual(module.report_path(root, original), root / "doc/taskcard9/N100_REGRESSION_ATTRIBUTION.md")
        self.assertEqual(module.report_path(root, fresh), fresh / "N100_REGRESSION_ATTRIBUTION.md")


if __name__ == "__main__":
    unittest.main()
