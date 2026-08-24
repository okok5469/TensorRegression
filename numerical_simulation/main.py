"""CLI entrypoint for tensor sparse-regression numerical simulations."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
from pathlib import Path

from numerical_simulation.data_generation import TensorSimulationConfig
from numerical_simulation.runner import ExperimentSpec, ExperimentSuiteConfig, run_experiment_suite


def build_profile(profile: str) -> tuple[TensorSimulationConfig, int, tuple[ExperimentSpec, ...]]:
    """Build a quick or full experiment profile."""
    if profile == "full":
        base = TensorSimulationConfig(
            n_train=200,
            n_test=500,
            dims=(32, 20, 25),
            sparsity=(4, 3, 5),
            rho=(0.3, 0.3, 0.3),
            snr=2.0,
            support_shape="continuous",
            sigma=1.0,
            k_max=1,
            max_cycles=6,
            n_lambdas=20,
            cv_folds=5,
            max_iter=500,
            tol=1e-6,
            abess_init_criterion="bic",
            abess_init_c=3.0,
            random_state=2026,
            test_batch_size=100,
        )
        n_reps = 10
        experiments = (
            ExperimentSpec("snr", "snr", (0.5, 1.0, 2.0, 4.0)),
            ExperimentSpec("sample_size", "n_train", (100, 200, 400)),
            ExperimentSpec("dimension", "dims", ((16, 12, 15), (32, 20, 25), (64, 30, 40))),
            ExperimentSpec("correlation", "rho", (0.0, 0.3, 0.6)),
        )
        return base, n_reps, experiments

    base = TensorSimulationConfig(
        n_train=60,
        n_test=80,
        dims=(8, 6, 8),
        sparsity=(2, 2, 2),
        rho=(0.3, 0.3, 0.3),
        snr=2.0,
        support_shape="continuous",
        sigma=1.0,
        k_max=1,
        max_cycles=4,
        n_lambdas=5,
        cv_folds=2,
        max_iter=120,
        tol=1e-5,
        abess_init_criterion="bic",
        abess_init_c=3.0,
        random_state=2026,
        test_batch_size=40,
    )
    n_reps = 2
    experiments = (
        ExperimentSpec("snr", "snr", (0.5, 1.0, 2.0, 4.0)),
        ExperimentSpec("sample_size", "n_train", (40, 60, 100)),
        ExperimentSpec("dimension", "dims", ((6, 5, 6), (8, 6, 8), (10, 8, 10))),
        ExperimentSpec("correlation", "rho", (0.0, 0.3, 0.6)),
    )
    return base, n_reps, experiments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tensor sparse-regression numerical simulations.")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--output-dir",
        default=str(Path("numerical_simulation") / "outputs"),
        help="Directory where tables, JSON reports, and figures are written.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=("snr", "sample_size", "dimension", "correlation"),
        default=("snr", "sample_size", "dimension", "correlation"),
        help="Subset of experiment groups to run.",
    )
    parser.add_argument("--n-reps", type=int, default=None, help="Override the repetition count.")
    parser.add_argument("--seed", type=int, default=None, help="Override the base random seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing raw_results.csv checkpoint in the output directory.",
    )
    parser.add_argument(
        "--support-shape",
        choices=("continuous", "discrete"),
        default=None,
        help="Override the support shape; PDF main experiments use continuous.",
    )
    parser.add_argument(
        "--tensor-abess-mode",
        choices=("fixed", "adaptive"),
        default=None,
        help="Use fixed true sparsity or adaptive MSIC sparsity for Tensor-ABESS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_config, n_reps, all_experiments = build_profile(args.profile)

    if args.n_reps is not None:
        n_reps = int(args.n_reps)
    if args.seed is not None:
        base_config = replace(base_config, random_state=int(args.seed))
    if args.support_shape is not None:
        base_config = replace(base_config, support_shape=args.support_shape)
    if args.tensor_abess_mode is not None:
        base_config = replace(base_config, tensor_abess_mode=args.tensor_abess_mode)

    selected = set(args.experiments)
    experiments = tuple(exp for exp in all_experiments if exp.name in selected)
    suite = ExperimentSuiteConfig(
        base_config=base_config,
        output_dir=args.output_dir,
        n_reps=n_reps,
        experiments=experiments,
        resume=bool(args.resume),
    )

    artifacts = run_experiment_suite(suite)
    print(f"Raw results: {artifacts.raw_results_path}")
    print(f"Summary results: {artifacts.summary_results_path}")
    print(f"Consistency report: {artifacts.consistency_report_path}")
    print("Plots:")
    for path in artifacts.plot_paths:
        print(f"  - {path}")
    print(f"Completed {len(artifacts.raw_rows)} method-runs across {n_reps} repetition(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
