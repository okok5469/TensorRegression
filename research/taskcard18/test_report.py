"""Status semantics for the dynamic read-only closeout report."""
import importlib
import importlib.util
import unittest


class ReportStatusTests(unittest.TestCase):
    def status(self, facts):
        name = "research.taskcard18.report"
        self.assertIsNotNone(importlib.util.find_spec(name), "Missing dynamic closeout report")
        return importlib.import_module(name).derive_status(facts)

    def test_running_missing_results_and_pilot_success_remain_in_progress(self):
        self.assertEqual(self.status({"valid_fits": 0, "evaluations": 0, "blocking_issues": [], "gates": {}}), "IN_PROGRESS")
        self.assertEqual(self.status({"valid_fits": 40, "evaluations": 0, "blocking_issues": [], "gates": {"G3": {"status": "PARTIAL"}}}), "IN_PROGRESS")

    def test_actual_unresolved_validity_failure_is_blocked(self):
        self.assertEqual(self.status({"valid_fits": 1, "evaluations": 0,
                                     "blocking_issues": ["required convex fit exhausted its frozen budget"], "gates": {}}), "BLOCKED")

    def test_complete_counts_cannot_replace_all_validity_gates(self):
        facts = {"valid_fits": 1200, "evaluations": 1200, "blocking_issues": [],
                 "gates": {f"G{i}": {"status": "PASS"} for i in range(8)}}
        facts["gates"]["G7"]["status"] = "PENDING"
        self.assertEqual(self.status(facts), "IN_PROGRESS")
        facts["gates"]["G7"]["status"] = "PASS"
        facts["gates"]["G0"]["status"] = "PASS_CURRENT"
        self.assertEqual(self.status(facts), "FINAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
