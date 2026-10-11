"""Run one version-isolated paired dataset without editing production defaults.

Global MSIC candidates are observed during their original computation and then
re-scored.  Inner CV uses a predeclared size grid and fresh support search per fold.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np


def repository_root():
    return Path(__file__).resolve().parents[2]


def configure_core(version):
    paths = {"v5-original": repository_root() / "doc/algorithm_versions/v5-original/src",
             "v5.1-bcd-fix": repository_root() / "src"}
    source = paths[version].resolve()
    if not (source / "tensor_abess/tensor/bcd.py").is_file():
        raise FileNotFoundError(f"Missing version source: {source}")
    loaded = sys.modules.get("tensor_abess")
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / "tensor_abess":
        raise RuntimeError("A different core is already imported; run versions in separate processes.")
    sys.path.insert(0, str(source))
    return source


def array_fingerprint(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode("ascii"))
    h.update(array.dtype.str.encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def snapshot(result, *, origin, elapsed=0.0, initial_supports=None):
    supports = tuple(np.asarray(getattr(result, f"S{m}"), dtype=int).copy() for m in (1, 2, 3))
    sizes = tuple(len(S) for S in supports)
    solver_result = getattr(result, "bcd_result", result)
    return {"B": result.B.copy(), "supports": supports, "sizes": sizes,
            "rss": float(getattr(result, "rss", np.nan)), "loss": float(result.loss),
            "origin": origin, "fit_sec": float(elapsed),
            "loss_history": list(result.loss_history), "n_cycles": int(solver_result.n_cycles),
            "best_iteration": getattr(solver_result, "best_iteration", None),
            "best_rss": getattr(solver_result, "best_rss", None),
            "stop_reason": getattr(solver_result, "stop_reason", None),
            "visited_loss_history": list(getattr(solver_result, "visited_loss_history", [])),
            "initial_supports": initial_supports}


def fit_adaptive(X, y, config, *, alpha=1.5, scale=1.0):
    from tensor_abess.tensor.abess import tensor_abess
    return tensor_abess(X, y, k_max=config.k_max, c=config.abess_init_c,
        criterion=config.abess_init_criterion, n_lambdas=config.n_lambdas,
        cv_folds=config.cv_folds, max_iter=config.max_iter, tol=config.tol,
        max_cycles=config.max_cycles, eps=1e-8, adaptive_criterion="msic",
        msic_refine_alpha=alpha, msic_refine_gamma=1.0, msic_refine_one_se_scale=scale)


def fit_with_candidates(dataset, config):
    """Observe original calls; wrappers return their exact unmodified results."""
    bcd = importlib.import_module("tensor_abess.tensor.bcd")
    original_fixed = bcd.tensor_bcd
    original_refine = bcd._refine_msic_global_candidates
    pool = []

    def record_fixed(*args, **kwargs):
        started = perf_counter()
        fixed = original_fixed(*args, **kwargs)
        initial = tuple(np.asarray(kwargs[f"S{m}_init"], dtype=int).tolist() for m in (1, 2, 3))
        candidate = snapshot(fixed, origin="global-fixed", elapsed=perf_counter() - started,
            initial_supports=initial)
        candidate["rss"] = float(2 * len(args[1]) * fixed.loss)
        pool.append(candidate)
        return fixed

    def record_refine(X, y, result, **kwargs):
        pool.append(snapshot(result, origin="adaptive-stage"))
        bcd.tensor_bcd = record_fixed
        try:
            return original_refine(X, y, result, **kwargs)
        finally:
            bcd.tensor_bcd = original_fixed

    bcd._refine_msic_global_candidates = record_refine
    started = perf_counter()
    try:
        result = fit_adaptive(dataset.X_train, dataset.y_train, config)
    finally:
        bcd._refine_msic_global_candidates = original_refine
        bcd.tensor_bcd = original_fixed
    return result, pool, perf_counter() - started


def select_candidate(pool, n, dims, *, criterion="msic", alpha=1.5, scale=0.0):
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss, compute_tensor_sic_from_rss
    if not pool:
        raise ValueError("Empty candidate pool")
    score_functions = {
        "msic": lambda c: compute_tensor_msic_from_rss(c["rss"], *c["sizes"], n, *dims,
            alpha=alpha, gamma=1.0),
        "tensor-sic": lambda c: compute_tensor_sic_from_rss(c["rss"], *c["sizes"], n, *dims),
        "bic": lambda c: float(n * np.log(max(c["rss"] / n, np.finfo(float).tiny))
            + int(np.prod(c["sizes"])) * np.log(n)),
    }
    scores = [float(score_functions[criterion](candidate)) for candidate in pool]
    minimum_index = min(range(len(pool)), key=lambda i: scores[i])
    minimum_q = int(np.prod(pool[minimum_index]["sizes"]))
    tolerance = float(scale * np.sqrt(max(minimum_q, 1) * max(np.log(max(n, 2)), 1.0))) if scale > 0 else 0.0
    eligible = [i for i, score in enumerate(scores) if score <= scores[minimum_index] + tolerance]
    selected_index = min(eligible, key=lambda i: (int(np.prod(pool[i]["sizes"])), scores[i], sum(pool[i]["sizes"])))
    return pool[selected_index], {"criterion": criterion, "alpha": alpha if criterion == "msic" else None,
        "gamma": 1.0 if criterion in {"msic", "tensor-sic"} else None,
        "one_se_scale": scale, "minimum_index": minimum_index, "minimum_q": minimum_q,
        "minimum_score": scores[minimum_index], "tolerance": tolerance,
        "eligible_indices": eligible, "selected_index": selected_index,
        "selected_score": scores[selected_index], "candidate_scores": scores}


def evaluate_shared_test(dataset, coefficients, test_seed):
    from numerical_simulation.data_generation import iter_test_batches
    from tensor_abess.tensor.ops import tensor_inner
    sse = np.zeros(len(coefficients))
    signal_sse = np.zeros(len(coefficients))
    total = batches = 0
    sum_y = sum_y2 = 0.0
    h = hashlib.sha256()
    for X, y, signal in iter_test_batches(dataset, seed=test_seed):
        for array in (X, y, signal):
            h.update(array_fingerprint(array).encode("ascii"))
        for j, B in enumerate(coefficients):
            prediction = tensor_inner(X, B)
            residual = y - prediction
            gap = signal - prediction
            sse[j] += float(residual @ residual)
            signal_sse[j] += float(gap @ gap)
        total += len(y)
        batches += 1
        sum_y += float(np.sum(y))
        sum_y2 += float(y @ y)
    if total != dataset.config.n_test or not total:
        raise ValueError("Incomplete independent test stream")
    sst = sum_y2 - sum_y * sum_y / total
    metrics = [{"pmse": float(sse[j] / total), "signal_mse": float(signal_sse[j] / total),
                "r2_test": float(1 - sse[j] / sst) if sst > np.finfo(float).tiny else 0.0}
               for j in range(len(coefficients))]
    return metrics, {"sha256": h.hexdigest(), "n_observations": total, "n_batches": batches,
        "definition": "SHA256 of shape/dtype/bytes array digests of X,y,signal in each shared batch"}


def cv_grid(n, dims, *, folds=5):
    declared = [(1, 1, 1), (2, 2, 2), (3, 2, 2), (3, 2, 4),
                (4, 3, 4), (4, 3, 5), (4, 4, 5), (5, 4, 6)]
    n_inner_min = n - int(np.ceil(n / folds))
    capacity = int(np.floor(0.7 * n_inner_min))
    feasible = lambda s: np.prod(s) <= capacity and all(size <= p for size, p in zip(s, dims))
    return [s for s in declared if feasible(s)], [s for s in declared if not feasible(s)]


def fit_fixed(X, y, sizes, config):
    from tensor_abess.tensor.bcd import _marginal_supports_for_size, tensor_bcd
    initial = _marginal_supports_for_size(X, y, sizes)
    result = tensor_bcd(X, y, *sizes, S1_init=initial[0], S2_init=initial[1], S3_init=initial[2],
        max_cycles=config.max_cycles, k_max=config.k_max)
    return result, tuple(S.tolist() for S in initial)


def fit_cv(dataset, config, seed):
    from tensor_abess.tensor.ops import tensor_inner
    if config.n_train != 200:
        raise ValueError("Predeclared representative inner-CV subset requires n_train=200")
    grid, excluded = cv_grid(config.n_train, config.dims)
    rng = np.random.default_rng(seed + 2_000_000)
    splits = np.array_split(rng.permutation(config.n_train), 5)
    fold_rows = []
    pooled_sse = {sizes: 0.0 for sizes in grid}
    started = perf_counter()
    for fold, validation in enumerate(splits):
        training = np.setdiff1d(np.arange(config.n_train), validation)
        X_train, y_train = dataset.X_train[training], dataset.y_train[training]
        for sizes in grid:
            fit_started = perf_counter()
            fit, initial = fit_fixed(X_train, y_train, sizes, config)
            residual = dataset.y_train[validation] - tensor_inner(dataset.X_train[validation], fit.B)
            sse = float(residual @ residual)
            pooled_sse[sizes] += sse
            fold_rows.append({"fold": fold, "sizes": sizes, "q": int(np.prod(sizes)),
                "n_train": len(training), "n_validation": len(validation),
                "train_indices": training.tolist(), "validation_indices": validation.tolist(),
                "train_rss": float(2 * len(training) * fit.loss), "validation_mse": sse / len(validation),
                "fit_sec": perf_counter() - fit_started, "supports": [S.tolist() for S in (fit.S1, fit.S2, fit.S3)],
                "initial_supports": initial})
    chosen = min(grid, key=lambda s: (pooled_sse[s] / config.n_train, int(np.prod(s)), sum(s)))
    fit, initial = fit_fixed(dataset.X_train, dataset.y_train, chosen, config)
    candidate = snapshot(fit, origin="cv-full-refit", initial_supports=initial)
    candidate["rss"] = float(2 * config.n_train * fit.loss)
    elapsed = perf_counter() - started
    return candidate, {"split_seed": seed + 2_000_000, "folds": 5,
        "predeclared_grid": grid, "capacity_excluded_before_fitting": excluded,
        "selection": "minimum pooled validation MSE; ties prefer smaller q then smaller size sum",
        "grid_scores": [{"sizes": s, "q": int(np.prod(s)), "validation_mse": pooled_sse[s] / config.n_train} for s in grid],
        "selected_sizes": chosen, "selected_validation_mse": pooled_sse[chosen] / config.n_train,
        "fold_rows": fold_rows, "runtime_sec": elapsed,
        "preprocessing": "No estimated preprocessing: generated Gaussian covariates use fixed population covariance; supports searched from inner training only."}


def json_candidate(candidate, index):
    return {key: value for key, value in candidate.items() if key not in {"B", "supports"}} | {
        "candidate_index": index, "supports": [S.tolist() for S in candidate["supports"]],
        "B_fingerprint": array_fingerprint(candidate["B"]), "q": int(np.prod(candidate["sizes"]))}


def run_request(request, version):
    source = configure_core(version)
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    import scipy
    values = dict(request["config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    seed, test_seed = int(request["seed"]), int(request["test_seed"])
    dataset = generate_simulation_dataset(config, seed=seed)
    candidates = []
    choices = []
    cv = None
    if version == "v5-original":
        started = perf_counter()
        result = fit_adaptive(dataset.X_train, dataset.y_train, config)
        shared_sec = perf_counter() - started
        selected = snapshot(result, origin="original-selected")
        candidates.append(selected)
        choices.append(("adaptive-default", selected, {"selected_score": float(result.sic), "selected_index": 0}, shared_sec, 0.0))
    else:
        result, candidates, shared_sec = fit_with_candidates(dataset, config)
        for alpha in (1.5, 2.0, 3.0):
            for scale in (0.0, 0.5, 1.0):
                started = perf_counter()
                selected, info = select_candidate(candidates, config.n_train, config.dims, alpha=alpha, scale=scale)
                choices.append((f"msic-alpha{alpha:g}-scale{scale:g}", selected, info, shared_sec, perf_counter() - started))
        for criterion in ("tensor-sic", "bic"):
            started = perf_counter()
            selected, info = select_candidate(candidates, config.n_train, config.dims, criterion=criterion)
            choices.append((criterion, selected, info, shared_sec, perf_counter() - started))
        default = next(c for name, c, _, _, _ in choices if name == "msic-alpha1.5-scale1")
        if not np.array_equal(default["B"], result.B):
            raise AssertionError("Observed default selection differs from exact original refinement")
        if request.get("include_cv", False):
            selected, cv = fit_cv(dataset, config, seed)
            choices.append(("cv-fixed-grid", selected, {"selected_sizes": selected["sizes"],
                "selected_validation_mse": cv["selected_validation_mse"]}, cv["runtime_sec"], 0.0))
    test_metrics, test_fingerprint = evaluate_shared_test(dataset, [c[1]["B"] for c in choices], test_seed)
    rows = []
    arrays = {}
    for i, (choice, prediction) in enumerate(zip(choices, test_metrics)):
        rule, selected, info, shared, scoring = choice
        supports = selected["supports"]
        q = int(np.prod(selected["sizes"]))
        q_true = int(np.prod(config.sparsity))
        rows.append({"method": f"{version}/{rule}", "version": version, "rule": rule,
            "n_train": config.n_train, "snr": config.snr, "seed": seed, "test_seed": test_seed,
            **compute_support_metrics(*supports, *dataset.supports_true),
            **compute_coefficient_metrics(selected["B"], dataset.B_true), **prediction,
            "loss_train": selected["loss"], "rss_train": selected["rss"], "q_true": q_true,
            "q_max": int(np.floor(0.7 * config.n_train)), "q_over": int(q > q_true),
            "q_under": int(q < q_true), "q_small": int(q == 1),
            "q_near_cap": int(q >= .9 * np.floor(.7 * config.n_train)),
            "shared_training_sec": shared, "scoring_sec": scoring, "runtime_sec": shared + scoring,
            "selection": info, "loss_history": selected["loss_history"],
            "stop_reason": selected["stop_reason"], "best_iteration": selected["best_iteration"]})
        arrays[f"model_{i}_B"] = selected["B"]
        for mode, S in enumerate(supports, start=1):
            arrays[f"model_{i}_S{mode}"] = S
    for i, candidate in enumerate(candidates):
        arrays[f"candidate_{i}_B"] = candidate["B"]
        for mode, S in enumerate(candidate["supports"], start=1):
            arrays[f"candidate_{i}_S{mode}"] = S
    arrays["B_true"] = dataset.B_true
    report = {"schema_version": 1, "status": "complete", "version": version,
        "config": asdict(config), "seed": seed, "test_seed": test_seed,
        "rows": rows, "candidates": [json_candidate(c, i) for i, c in enumerate(candidates)], "cv": cv,
        "trace": {"init_supports": [S.tolist() for S in result.init_supports], "s_max": list(result.s_max),
            "returned_loss_history": list(result.loss_history), "returned_sic_history": list(result.sic_history),
            "stage_alpha": 1.0, "stage_gamma": 1.0, "default_global_alpha": 1.5,
            "default_global_gamma": 1.0, "default_one_se_scale": 1.0},
        "fingerprints": {"X_train": array_fingerprint(dataset.X_train), "y_train": array_fingerprint(dataset.y_train),
            "B_true": array_fingerprint(dataset.B_true), "independent_test": test_fingerprint,
            "core_sources": {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((source / "tensor_abess").rglob("*.py"))}},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "runtime_interpretation": "Rescoring methods share the complete candidate search cost; their runtime values must not be summed as independent fits."}
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True, choices=("v5-original", "v5.1-bcd-fix"))
    args = parser.parse_args(argv)
    npz_path = args.output.with_suffix(".npz")
    if args.output.exists() or npz_path.exists():
        raise FileExistsError("Version-validation outputs are immutable; choose a fresh path.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    report, arrays = run_request(request, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report["array_cache"] = npz_path.name
    report["array_cache_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Completed {args.version}: {len(report['rows'])} rules, {len(report['candidates'])} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
