"""Isolated, paired advisor experiments; existing simulation code is read-only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from numerical_simulation.data_generation import (
    TensorSimulationConfig,
    TensorSimulationDataset,
    generate_simulation_dataset,
    iter_test_batches,
    scale_tensor_to_snr,
    tensor_signal_variance,
)
from numerical_simulation.methods import (
    MethodEstimate,
    run_oracle_ols_method,
    run_tensor_abess_method,
)
from numerical_simulation.metrics import (
    compute_coefficient_metrics,
    compute_prediction_metrics_from_batches,
    compute_support_metrics,
)
from tensor_abess.tensor.ops import tensor_inner


CASE_NAMES = (
    "uniform_signed_snr2",
    "constant_signed_snr2",
    "constant_positive_snr2",
    "constant_positive_1_literal",
    "constant_positive_05_literal",
)


def normalized_positive_tensor(
    baseline: TensorSimulationDataset, *, magnitude: float,
) -> np.ndarray:
    """Normalize a positive constant support block to the original target SNR."""
    B0 = np.zeros(baseline.config.dims, dtype=float)
    B0[np.ix_(*baseline.supports_true)] = magnitude
    B, _ = scale_tensor_to_snr(
        B0, baseline.covariances, baseline.config.snr, baseline.config.sigma,
    )
    return B


def verify_normalized_equivalence(baseline: TensorSimulationDataset) -> dict[str, bool | float]:
    first = normalized_positive_tensor(baseline, magnitude=1.0)
    second = normalized_positive_tensor(baseline, magnitude=0.5)
    noise = baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)
    first_y = tensor_inner(baseline.X_train, first) + noise
    second_y = tensor_inner(baseline.X_train, second) + noise
    result = {
        "identical_coefficients": bool(np.array_equal(first, second)),
        "identical_training_response": bool(np.array_equal(first_y, second_y)),
        "coefficient_max_abs_difference": float(np.max(np.abs(first - second))),
    }
    if not result["identical_coefficients"] or not result["identical_training_response"]:
        raise ValueError("SNR-normalized positive 1 and 0.5 cases are not numerically identical")
    return result


def build_case_dataset(
    baseline: TensorSimulationDataset, case: str,
) -> TensorSimulationDataset:
    """Change only coefficients and response, preserving predictors and noise draws."""
    if case not in CASE_NAMES:
        raise ValueError(f"unknown coefficient case: {case}")
    if case == "uniform_signed_snr2":
        return baseline
    builders = {
        "constant_signed_snr2": lambda: np.sign(baseline.B_true),
        "constant_positive_snr2": lambda: (baseline.B_true != 0).astype(float),
        "constant_positive_1_literal": lambda: (baseline.B_true != 0).astype(float),
        "constant_positive_05_literal": lambda: 0.5 * (baseline.B_true != 0).astype(float),
    }
    B = builders[case]()
    if case.endswith("_snr2"):
        B, _ = scale_tensor_to_snr(
            B, baseline.covariances, baseline.config.snr, baseline.config.sigma,
        )
    signal_variance = tensor_signal_variance(B, *baseline.covariances)
    actual_snr = signal_variance / baseline.config.sigma**2
    noise = baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)
    return replace(
        baseline,
        config=replace(baseline.config, snr=actual_snr),
        B_true=B,
        y_train=tensor_inner(baseline.X_train, B) + noise,
        signal_variance=signal_variance,
    )


def fit_tensor_method(
    X: np.ndarray, y: np.ndarray, config: TensorSimulationConfig,
    *, mode: str, seed: int,
) -> MethodEstimate:
    """The estimator boundary accepts observations and sizes, never true supports."""
    mode_map = {"adaptive": "adaptive", "fixed_known_s": "fixed"}
    method_config = replace(config, tensor_abess_mode=mode_map[mode])
    return run_tensor_abess_method(X, y, method_config, random_state=seed)


def load_formal_config() -> tuple[TensorSimulationConfig, Path]:
    repository = Path(__file__).resolve().parents[2]
    path = repository / "numerical_simulation/outputs_full_v5_adaptive/checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    values = dict(manifest["base_config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    return TensorSimulationConfig(**values), path


def _array_hash(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _source_fingerprints(manifest_path: Path) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    paths = list((repository / "src/tensor_abess").rglob("*.py"))
    paths += list((repository / "numerical_simulation").glob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("test_run.py"), manifest_path]
    return {
        path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _paired_input_fingerprints(
    baseline: TensorSimulationDataset, seed: int,
) -> dict[str, str]:
    test_X_hash = hashlib.sha256()
    test_noise_hash = hashlib.sha256()
    for X, y, signal in iter_test_batches(baseline, seed=seed + 1_000_000):
        test_X_hash.update(X.tobytes())
        test_noise_hash.update((y - signal).tobytes())
    noise = baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)
    return {
        "train_X_sha256": _array_hash(baseline.X_train),
        "train_noise_sha256": _array_hash(noise),
        "test_X_stream_sha256": test_X_hash.hexdigest(),
        "test_baseline_noise_stream_sha256": test_noise_hash.hexdigest(),
        **{f"support_{i}_sha256": _array_hash(S) for i, S in enumerate(baseline.supports_true, 1)},
    }


def _evaluate(dataset: TensorSimulationDataset, estimate: MethodEstimate, seed: int) -> dict:
    metrics = {
        **compute_support_metrics(estimate.S1, estimate.S2, estimate.S3, *dataset.supports_true),
        **compute_coefficient_metrics(estimate.B, dataset.B_true),
        **compute_prediction_metrics_from_batches(
            iter_test_batches(dataset, seed=seed + 1_000_000), dataset.B_true, estimate.B,
        ),
        "loss_train": float(estimate.loss),
        "rss_train": float(estimate.rss),
        "runtime_sec": float(estimate.runtime_sec),
        "n_cycles": int(estimate.n_cycles),
    }
    if not np.isfinite(list(metrics.values())).all() or not np.isfinite(estimate.B).all():
        raise ValueError("non-finite estimator or metrics; task aborted")
    return metrics


def run_task(
    baseline: TensorSimulationDataset, *, case: str, seed: int, output: Path,
    source_fingerprints: dict[str, str], paired_input_fingerprints: dict[str, str],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{seed}_{case}"
    result_path = output / f"{stem}.json"
    array_path = output / f"{stem}.npz"
    if result_path.exists() or array_path.exists():
        raise FileExistsError(f"refusing to overwrite existing task: {stem}")
    started = perf_counter()
    dataset = build_case_dataset(baseline, case)
    equivalence = verify_normalized_equivalence(baseline)
    estimates = [
        ("adaptive", fit_tensor_method(dataset.X_train, dataset.y_train, dataset.config, mode="adaptive", seed=seed)),
        ("fixed_known_s", fit_tensor_method(dataset.X_train, dataset.y_train, dataset.config, mode="fixed_known_s", seed=seed)),
        ("oracle", run_oracle_ols_method(dataset.X_train, dataset.y_train, dataset.supports_true)),
    ]
    arrays = {"B_true": dataset.B_true, "train_noise": baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)}
    arrays.update({f"S{i}_true": S for i, S in enumerate(dataset.supports_true, 1)})
    rows = []
    for name, estimate in estimates:
        rows.append({"method": name, "metrics": _evaluate(dataset, estimate, seed), "details": estimate.details})
        arrays[f"B_{name}"] = estimate.B
        arrays.update({f"S{i}_{name}": S for i, S in enumerate((estimate.S1, estimate.S2, estimate.S3), 1)})
    manifest = {
        "seed": seed, "test_seed": seed + 1_000_000, "case": case,
        "base_config": asdict(baseline.config), "case_config": asdict(dataset.config),
        "source_fingerprints": source_fingerprints,
        "paired_input_fingerprints": paired_input_fingerprints,
        "case_B_sha256": _array_hash(dataset.B_true),
        "case_y_sha256": _array_hash(dataset.y_train),
        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "python_version": sys.version, "numpy_version": np.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    result = {
        "manifest": manifest, "manifest_sha256": fingerprint,
        "actual_snr": dataset.signal_variance / dataset.config.sigma**2,
        "normalized_positive_1_vs_05_equivalence": equivalence,
        "results": rows, "elapsed_sec": perf_counter() - started,
        "arrays_file": array_path.name,
        "solver_scope": "existing fixed and adaptive entrypoints, including their respective initializations",
    }
    # Each rename commits one complete file and fails if its target already exists on Windows.
    with tempfile.NamedTemporaryFile(dir=output, prefix=stem, suffix=".npz.tmp", delete=False) as handle:
        temporary_arrays = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.rename(temporary_arrays, array_path)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=stem, suffix=".json.tmp", encoding="utf-8", delete=False) as handle:
        temporary_json = Path(handle.name)
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.rename(temporary_json, result_path)
    print(f"completed seed={seed} case={case} elapsed={result['elapsed_sec']:.2f}s", flush=True)
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2026, 2036)))
    parser.add_argument("--cases", choices=CASE_NAMES, nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args(argv)
    config, manifest_path = load_formal_config()
    source_fingerprints = _source_fingerprints(manifest_path)
    for seed in args.seeds:
        baseline = generate_simulation_dataset(config, seed=seed)
        fingerprints = _paired_input_fingerprints(baseline, seed)
        for case in args.cases:
            run_task(
                baseline, case=case, seed=seed, output=args.output,
                source_fingerprints=source_fingerprints, paired_input_fingerprints=fingerprints,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
