import unittest

from tensor_abess.utils.validation import run_tensor_abess_validation_suite


class ValidationSuiteTests(unittest.TestCase):
    def test_validation_suite_runs_and_reports_passed_cases(self) -> None:
        suite = run_tensor_abess_validation_suite()

        self.assertGreaterEqual(len(suite.cases), 2)
        self.assertTrue(suite.all_passed)
        singleton = next(case for case in suite.cases if case.name == "singleton_exact")
        self.assertTrue(singleton.metrics.exact_rectangular)


if __name__ == "__main__":
    unittest.main()
