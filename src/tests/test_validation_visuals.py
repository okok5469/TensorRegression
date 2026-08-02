import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from tensor_abess.utils.validation import (
    generate_validation_report,
    run_tensor_abess_monte_carlo,
)


class ValidationVisualTests(unittest.TestCase):
    def test_monte_carlo_runner_produces_summary_statistics(self) -> None:
        summary = run_tensor_abess_monte_carlo(n_reps=3, random_state=11)

        self.assertEqual(summary.n_reps, 3)
        self.assertEqual(len(summary.trials), 3)
        self.assertGreaterEqual(summary.mean_s1_tpr, 0.0)
        self.assertLessEqual(summary.mean_s1_tpr, 1.0)
        self.assertGreaterEqual(summary.exact_recovery_rate, 0.0)
        self.assertLessEqual(summary.exact_recovery_rate, 1.0)

    def test_validation_report_generates_png_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report = generate_validation_report(out_dir, n_reps=2, random_state=5)

            self.assertTrue(report.plot_path.exists(), msg=str(report.plot_path))
            self.assertTrue(report.markdown_path.exists(), msg=str(report.markdown_path))
            self.assertEqual(report.monte_carlo.n_reps, 2)
            self.assertIn("Monte Carlo", report.markdown_path.read_text(encoding="utf-8"))

    def test_validation_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "report"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path("tensor_abess/utils/validation.py")),
                    "--output-dir",
                    str(out_dir),
                    "--n-reps",
                    "2",
                    "--random-state",
                    "3",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue((out_dir / "tensor_abess_validation.png").exists())
            self.assertTrue((out_dir / "tensor_abess_validation.md").exists())


if __name__ == "__main__":
    unittest.main()
