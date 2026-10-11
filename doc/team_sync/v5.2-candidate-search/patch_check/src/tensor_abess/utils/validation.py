"""Numerical validation helpers and lightweight visualization for revised Tensor-ABESS."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tensor_abess.metrics.support import SupportMetrics, evaluate_support
from tensor_abess.tensor.abess import TensorAbessResult, tensor_abess


@dataclass
class ValidationCaseResult:
    name: str
    passed: bool
    message: str
    result: TensorAbessResult
    metrics: SupportMetrics


@dataclass
class ValidationSuiteResult:
    cases: list[ValidationCaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(case.passed for case in self.cases)


@dataclass
class MonteCarloTrialResult:
    seed: int
    result: TensorAbessResult
    metrics: SupportMetrics


@dataclass
class MonteCarloSummary:
    n_reps: int
    trials: list[MonteCarloTrialResult]
    mean_s1_tpr: float
    mean_s2_tpr: float
    mean_s3_tpr: float
    mean_frobenius_error: float
    exact_recovery_rate: float


@dataclass
class ValidationReport:
    suite: ValidationSuiteResult
    monte_carlo: MonteCarloSummary
    plot_path: Path
    markdown_path: Path


def _build_singleton_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    X = np.zeros((6, 2, 2, 2), dtype=float)
    X[:, 0, 0, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    B_true = np.zeros((2, 2, 2), dtype=float)
    B_true[0, 0, 0] = 2.0
    y = np.einsum("nijk,ijk->n", X, B_true)
    return X, y, B_true, (np.array([0]), np.array([0]), np.array([0]))


def _build_low_noise_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    n, p1, p2, p3 = 80, 4, 3, 3
    X = rng.normal(size=(n, p1, p2, p3))
    B_true = np.zeros((p1, p2, p3), dtype=float)
    S1_true = np.array([0, 2], dtype=int)
    S2_true = np.array([1], dtype=int)
    S3_true = np.array([0], dtype=int)
    B_true[0, 1, 0] = 1.5
    B_true[2, 1, 0] = -1.0
    signal = np.einsum("nijk,ijk->n", X, B_true)
    y = signal + 0.05 * rng.normal(size=n)
    return X, y, B_true, (S1_true, S2_true, S3_true)


def _build_monte_carlo_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n, p1, p2, p3 = 50, 6, 5, 5
    X = rng.normal(size=(n, p1, p2, p3))
    B_true = np.zeros((p1, p2, p3), dtype=float)
    S1_true = np.array([0, 4], dtype=int)
    S2_true = np.array([1, 3], dtype=int)
    S3_true = np.array([0, 2], dtype=int)
    coeffs = np.array([1.2, -1.0, 0.8, -0.9, 0.7, -0.6, 0.5, -0.4], dtype=float)
    idx = 0
    for i in S1_true:
        for j in S2_true:
            for k in S3_true:
                B_true[i, j, k] = coeffs[idx]
                idx += 1
    signal = np.einsum("nijk,ijk->n", X, B_true)
    y = signal + 0.12 * rng.normal(size=n)
    return X, y, B_true, (S1_true, S2_true, S3_true)


def _run_tensor_abess(X: np.ndarray, y: np.ndarray) -> TensorAbessResult:
    return tensor_abess(
        X,
        y,
        c=2.0,
        criterion="bic",
        n_lambdas=10,
        max_iter=300,
        max_cycles=8,
        m_max=8,
        eps=1e-10,
    )


def _evaluate_case_result(
    name: str,
    result: TensorAbessResult,
    B_true: np.ndarray,
    supports_true: tuple[np.ndarray, np.ndarray, np.ndarray],
    dims: tuple[int, int, int],
    n_samples: int,
) -> ValidationCaseResult:
    metrics = evaluate_support(
        result.S1,
        result.S2,
        result.S3,
        supports_true[0],
        supports_true[1],
        supports_true[2],
        result.B,
        B_true,
        dims[0],
        dims[1],
        dims[2],
    )
    checks = [
        np.isfinite(result.loss),
        np.isfinite(result.sic),
        len(result.loss_history) >= 1,
        len(result.sic_history) >= 1,
        result.s_max[0] * result.s_max[1] * result.s_max[2] < n_samples,
    ]
    passed = all(checks)
    message = "basic structural checks passed"
    if name == "singleton_exact":
        passed = passed and metrics.exact_rectangular
        message = "exact support recovered on noiseless singleton case"
    return ValidationCaseResult(name=name, passed=passed, message=message, result=result, metrics=metrics)


def _evaluate_support_metrics(
    result: TensorAbessResult,
    B_true: np.ndarray,
    supports_true: tuple[np.ndarray, np.ndarray, np.ndarray],
    dims: tuple[int, int, int],
) -> SupportMetrics:
    return evaluate_support(
        result.S1,
        result.S2,
        result.S3,
        supports_true[0],
        supports_true[1],
        supports_true[2],
        result.B,
        B_true,
        dims[0],
        dims[1],
        dims[2],
    )


def run_tensor_abess_validation_suite() -> ValidationSuiteResult:
    """Run deterministic structural and numerical validation cases."""
    suite = ValidationSuiteResult()
    for name, builder in [
        ("singleton_exact", _build_singleton_case),
        ("low_noise_rectangular", _build_low_noise_case),
    ]:
        X, y, B_true, supports_true = builder()
        result = _run_tensor_abess(X, y)
        suite.cases.append(
            _evaluate_case_result(
                name,
                result,
                B_true,
                supports_true,
                (X.shape[1], X.shape[2], X.shape[3]),
                X.shape[0],
            )
        )
    return suite


def run_tensor_abess_monte_carlo(
    n_reps: int = 5,
    *,
    random_state: int = 0,
) -> MonteCarloSummary:
    """Run repeated low-noise rectangular recovery experiments."""
    trials: list[MonteCarloTrialResult] = []
    for rep in range(n_reps):
        seed = random_state + rep
        X, y, B_true, supports_true = _build_monte_carlo_case(seed)
        result = _run_tensor_abess(X, y)
        metrics = _evaluate_support_metrics(result, B_true, supports_true, (X.shape[1], X.shape[2], X.shape[3]))
        trials.append(MonteCarloTrialResult(seed=seed, result=result, metrics=metrics))

    mean_s1_tpr = float(np.mean([t.metrics.s1_tpr for t in trials]))
    mean_s2_tpr = float(np.mean([t.metrics.s2_tpr for t in trials]))
    mean_s3_tpr = float(np.mean([t.metrics.s3_tpr for t in trials]))
    mean_frobenius_error = float(np.mean([t.metrics.frobenius_error for t in trials]))
    exact_recovery_rate = float(np.mean([1.0 if t.metrics.exact_rectangular else 0.0 for t in trials]))
    return MonteCarloSummary(
        n_reps=n_reps,
        trials=trials,
        mean_s1_tpr=mean_s1_tpr,
        mean_s2_tpr=mean_s2_tpr,
        mean_s3_tpr=mean_s3_tpr,
        mean_frobenius_error=mean_frobenius_error,
        exact_recovery_rate=exact_recovery_rate,
    )


def _plot_validation_summary(
    suite: ValidationSuiteResult,
    monte_carlo: MonteCarloSummary,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    det_labels = [case.name for case in suite.cases]
    det_values = [1.0 if case.metrics.exact_rectangular else 0.0 for case in suite.cases]
    axes[0].bar(det_labels, det_values, color=["#4C78A8", "#72B7B2"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Deterministic cases")
    axes[0].set_ylabel("Exact recovery")
    axes[0].tick_params(axis="x", rotation=20)

    mc_labels = ["S1 TPR", "S2 TPR", "S3 TPR", "Exact rate"]
    mc_values = [
        monte_carlo.mean_s1_tpr,
        monte_carlo.mean_s2_tpr,
        monte_carlo.mean_s3_tpr,
        monte_carlo.exact_recovery_rate,
    ]
    axes[1].bar(mc_labels, mc_values, color="#F58518")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Monte Carlo ({monte_carlo.n_reps} reps)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Tensor-ABESS validation snapshot")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_validation_markdown(
    suite: ValidationSuiteResult,
    monte_carlo: MonteCarloSummary,
    markdown_path: Path,
    plot_path: Path,
) -> None:
    lines = [
        "# Tensor-ABESS validation report",
        "",
        "## Deterministic checks",
        "",
        "| Case | Passed | Exact rectangular | Frobenius error |",
        "|---|---:|---:|---:|",
    ]
    for case in suite.cases:
        lines.append(
            f"| {case.name} | {'yes' if case.passed else 'no'} | "
            f"{'yes' if case.metrics.exact_rectangular else 'no'} | {case.metrics.frobenius_error:.4f} |"
        )
    lines += [
        "",
        "## Monte Carlo",
        "",
        f"- repetitions: {monte_carlo.n_reps}",
        f"- mean S1 TPR: {monte_carlo.mean_s1_tpr:.3f}",
        f"- mean S2 TPR: {monte_carlo.mean_s2_tpr:.3f}",
        f"- mean S3 TPR: {monte_carlo.mean_s3_tpr:.3f}",
        f"- exact recovery rate: {monte_carlo.exact_recovery_rate:.3f}",
        f"- mean Frobenius error: {monte_carlo.mean_frobenius_error:.3f}",
        "",
        f"Plot: `{plot_path.name}`",
        "",
        "Interpretation: higher TPR / exact rate and lower Frobenius error are better.",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def generate_validation_report(
    output_dir: str | Path,
    *,
    n_reps: int = 5,
    random_state: int = 0,
) -> ValidationReport:
    """Run deterministic + Monte Carlo validation and save a compact visual report."""
    output_dir = Path(output_dir)
    suite = run_tensor_abess_validation_suite()
    monte_carlo = run_tensor_abess_monte_carlo(n_reps=n_reps, random_state=random_state)
    plot_path = output_dir / "tensor_abess_validation.png"
    markdown_path = output_dir / "tensor_abess_validation.md"
    _plot_validation_summary(suite, monte_carlo, plot_path)
    _write_validation_markdown(suite, monte_carlo, markdown_path, plot_path)
    return ValidationReport(
        suite=suite,
        monte_carlo=monte_carlo,
        plot_path=plot_path,
        markdown_path=markdown_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tensor-ABESS validation and generate a compact report.")
    parser.add_argument(
        "--output-dir",
        default="validation_artifacts",
        help="Directory where the markdown report and PNG snapshot will be written.",
    )
    parser.add_argument("--n-reps", type=int, default=8, help="Number of Monte Carlo repetitions.")
    parser.add_argument("--random-state", type=int, default=42, help="Base random seed for Monte Carlo runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = generate_validation_report(
        args.output_dir,
        n_reps=args.n_reps,
        random_state=args.random_state,
    )
    print(f"Markdown report: {report.markdown_path}")
    print(f"Plot image: {report.plot_path}")
    print(f"Deterministic suite passed: {report.suite.all_passed}")
    print(
        "Monte Carlo summary: "
        f"exact_rate={report.monte_carlo.exact_recovery_rate:.3f}, "
        f"S1={report.monte_carlo.mean_s1_tpr:.3f}, "
        f"S2={report.monte_carlo.mean_s2_tpr:.3f}, "
        f"S3={report.monte_carlo.mean_s3_tpr:.3f}, "
        f"frob={report.monte_carlo.mean_frobenius_error:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
