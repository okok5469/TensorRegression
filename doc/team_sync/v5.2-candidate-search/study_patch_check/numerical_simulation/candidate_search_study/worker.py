"""One isolated dataset, explicit candidate caches, and shared independent testing."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from numerical_simulation.version_validation.worker import (
    array_fingerprint, evaluate_shared_test, select_candidate as score_pool,
)


GROUPS = {"A0": ("legacy", "legacy"), "A1": ("candidate", "legacy"),
          "A2": ("legacy", "full"), "A3": ("candidate", "full"),
          "A4": ("candidate", "coarse_to_fine")}
VERSION = "v5.2-candidate-search"
AUDIT_REVISION = "2-stage-objectives"


def root():
    return Path(__file__).resolve().parents[2]


def configure_core():
    source = root() / "doc/algorithm_versions" / VERSION / "src"
    loaded = sys.modules.get("tensor_abess")
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / "tensor_abess":
        raise RuntimeError("A different core is loaded; each dataset requires an isolated worker.")
    if not (source / "tensor_abess/tensor/bcd.py").is_file():
        raise FileNotFoundError(f"Missing isolated core {source}")
    sys.path.insert(0, str(source))
    return source


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pool_fingerprint(pool):
    h = hashlib.sha256()
    for candidate in pool:
        for value in (candidate["B"], *candidate["supports"]):
            h.update(array_fingerprint(value).encode("ascii"))
        h.update(json.dumps({key: candidate[key] for key in ("sizes", "rss", "loss")},
                            sort_keys=True, allow_nan=False).encode("utf-8"))
    return h.hexdigest()


def rescore_pool(pool, n, dims):
    before = pool_fingerprint(pool)
    choices = []
    for alpha in (1.5, 2.0, 3.0):
        for scale in (0.0, 0.5, 1.0):
            selected, info = score_pool(pool, n, dims, alpha=alpha, scale=scale)
            choices.append((f"msic-alpha{alpha:g}-scale{scale:g}", selected, info))
    require(pool_fingerprint(pool) == before, "Rescoring altered frozen training pool")
    return choices


def audit_pool(pool, n, dims):
    require(bool(pool), "Empty training candidate pool")
    for candidate in pool:
        sizes = tuple(int(s) for s in candidate["sizes"])
        supports = candidate["supports"]
        require(int(np.prod(sizes)) <= int(np.floor(.7 * n)), "Candidate exceeds model capacity")
        require(len(supports) == 3 and all(len(s) == size for s, size in zip(supports, sizes)),
                "Candidate size/support mismatch")
        require(all(len(np.unique(S)) == len(S) and np.all(np.asarray(S) >= 0)
                    and np.all(np.asarray(S) < p) for S, p in zip(supports, dims)), "Invalid support")
        require(np.asarray(candidate["B"]).shape == tuple(dims), "Invalid coefficient tensor shape")
        require(np.isfinite(candidate["rss"]) and np.isfinite(candidate["loss"])
                and np.all(np.isfinite(candidate["B"])), "Nonfinite candidate")
        require(np.isclose(candidate["rss"], 2 * n * candidate["loss"], atol=1e-10, rtol=1e-10),
                "Candidate RSS/loss inconsistent")
        kind = candidate.get("kind")
        require(kind in {"adaptive-stage", "global-fixed"}, "Unknown candidate kind")
        history = candidate.get("loss_history", [])
        visited = candidate.get("visited_loss_history", [])
        require(np.all(np.isfinite(history)) and np.all(np.isfinite(visited)), "Nonfinite candidate history")
        if kind != "global-fixed":
            continue
        require(bool(history), "Fixed candidate requires accepted loss history")
        require(all(b <= a + 1e-10 * max(1, abs(a), abs(b)) for a, b in zip(history, history[1:])),
                "Accepted training loss increased")
        require(bool(visited), "Fixed candidate requires nonempty visited history")
        require(np.isclose(candidate["loss"], min(visited), atol=1e-10, rtol=1e-10),
                "Fixed candidate did not return minimum visited loss")
        index = candidate.get("best_iteration")
        require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
                and 0 <= index < len(visited), "Invalid fixed candidate best_iteration")
        require(np.isclose(candidate["loss"], visited[index], atol=1e-10, rtol=1e-10),
                "Fixed candidate best_iteration does not identify returned loss")


def candidate_json(candidate, index):
    result = {key: value for key, value in candidate.items() if key not in {"B", "supports"}}
    result.update(candidate_index=index, supports=[S.tolist() for S in candidate["supports"]],
                  q=int(np.prod(candidate["sizes"])), B_fingerprint=array_fingerprint(candidate["B"]))
    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def run_request(request):
    source = configure_core()
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from tensor_abess.tensor.abess import tensor_abess
    import scipy
    values = dict(request["config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    dataset = generate_simulation_dataset(config, seed=int(request["seed"]))
    fingerprints = {"X_train": array_fingerprint(dataset.X_train), "y_train": array_fingerprint(dataset.y_train),
                    "B_true": array_fingerprint(dataset.B_true)}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    cache, groups, choices, arrays = {}, {}, [], {"B_true": dataset.B_true}
    for group in request["groups"]:
        cap, grid = GROUPS[group]
        started = perf_counter()
        result = tensor_abess(dataset.X_train, dataset.y_train, k_max=config.k_max,
            c=config.abess_init_c, criterion=config.abess_init_criterion, n_lambdas=config.n_lambdas,
            cv_folds=config.cv_folds, max_iter=config.max_iter, tol=config.tol,
            max_cycles=config.max_cycles, eps=1e-8, adaptive_criterion="msic",
            msic_refine_alpha=1.5, msic_refine_gamma=1.0, msic_refine_one_se_scale=1.0,
            cap_policy=cap, grid_policy=grid, candidate_budget=request["fit_options"]["candidate_budget"],
            fixed_candidate_cache=cache)
        elapsed = perf_counter() - started
        pool = result.bcd_result.candidate_records
        audit_pool(pool, config.n_train, config.dims)
        before = pool_fingerprint(pool)
        default, default_info = score_pool(pool, config.n_train, config.dims, alpha=1.5, scale=1.0)
        rescored = [("msic-alpha1.5-scale1", default, default_info)]
        require(pool_fingerprint(pool) == before, "Default selection altered training pool")
        require(np.array_equal(default["B"], result.B), "Default rescoring differs from returned training selection")
        metadata = result.bcd_result.search_metadata
        grid_sizes = [list(size) for size in metadata["size_grid"]]
        true_shape = list(config.sparsity)
        actual_support = any(all(np.array_equal(np.sort(S), np.sort(T)) for S, T in
                             zip(candidate["supports"], dataset.supports_true)) for candidate in pool)
        truth_reasons = []
        if int(np.prod(config.sparsity)) > int(np.floor(.7 * config.n_train)):
            truth_reasons.append("product_capacity")
        if any(s > cap_value for s, cap_value in zip(config.sparsity, metadata["caps"])):
            truth_reasons.append("mode_cap")
        if true_shape not in grid_sizes:
            truth_reasons.append("grid_omission")
        groups[group] = {"cap_policy": cap, "grid_policy": grid, "runtime_sec": elapsed,
            "pool_fingerprint": before, "candidate_count": len(pool),
            "cache_hits": sum(bool(candidate.get("cache_hit")) for candidate in pool),
            "new_fixed_fits": sum(candidate.get("kind") == "global-fixed" and not candidate.get("cache_hit") for candidate in pool),
            "cache_entries_after_group": len(cache), "search_metadata": metadata,
            "init_supports": result.init_supports, "s_max": result.s_max,
            "true_size_in_grid": true_shape in grid_sizes,
            "true_size_in_pool": any(list(candidate["sizes"]) == true_shape for candidate in pool),
            "true_support_in_pool": actual_support, "true_size_exclusion_reasons": truth_reasons,
            "candidates": [candidate_json(candidate, i) for i, candidate in enumerate(pool)],
            "fingerprints": dict(fingerprints), "environment": dict(environment)}
        for i, candidate in enumerate(pool):
            arrays[f"{group}_candidate_{i}_B"] = candidate["B"]
            for mode, S in enumerate(candidate["supports"], start=1):
                arrays[f"{group}_candidate_{i}_S{mode}"] = S
        for rule, selected, info in rescored:
            choices.append((group, rule, selected, info, elapsed))
    predictions, test_fingerprint = evaluate_shared_test(dataset, [item[2]["B"] for item in choices], int(request["test_seed"]))
    for group in groups.values():
        group["fingerprints"]["independent_test"] = test_fingerprint
    rows = []
    for (group, rule, selected, info, elapsed), prediction in zip(choices, predictions):
        supports = selected["supports"]
        rows.append({"group": group, "rule": rule, "method": f"{group}/{rule}",
            "n_train": config.n_train, "snr": config.snr, "seed": int(request["seed"]),
            "test_seed": int(request["test_seed"]), "q_true": int(np.prod(config.sparsity)),
            **compute_support_metrics(*supports, *dataset.supports_true),
            **compute_coefficient_metrics(selected["B"], dataset.B_true), **prediction,
            "rss_train": float(selected["rss"]), "loss_train": float(selected["loss"]),
            "runtime_sec": elapsed, "selection": info, "supports": [S.tolist() for S in supports],
            "B_fingerprint": array_fingerprint(selected["B"]),
            "candidate_count": groups[group]["candidate_count"],
            "true_size_in_pool": int(groups[group]["true_size_in_pool"]),
            "true_support_in_pool": int(groups[group]["true_support_in_pool"])})
    report = {"schema_version": 1, "version": VERSION, "status": "complete", "config": asdict(config),
        "validation_rules": {"revision": AUDIT_REVISION,
            "all_candidates": "capacity/support/finite arrays and histories/RSS=2n*loss",
            "global-fixed": "accepted loss nonincreasing; returned loss=min visited; best_iteration consistent",
            "adaptive-stage": "q can change under criterion selection; RSS monotonicity is not required"},
        "seed": int(request["seed"]), "test_seed": int(request["test_seed"]), "group_order": request["groups"],
        "fit_options": request["fit_options"], "groups": groups, "rows": rows,
        "fingerprints": {**fingerprints, "independent_test": test_fingerprint,
            "core_sources": {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((source / "tensor_abess").rglob("*.py"))}},
        "environment": environment,
        "runtime_interpretation": "Groups run in frozen order and explicitly share identical fixed training fits; cached runtime is not an independent cold-fit comparison. All rescoring rules share one training pool per group. Test data and truth never determine caps, grids, initialization, candidates, or training selection."}
    return jsonable(report), arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    npz = args.output.with_suffix(".npz")
    if args.output.exists() or npz.exists():
        raise FileExistsError("Immutable candidate experiment output already exists")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    report, arrays = run_request(request)
    with npz.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report["array_cache"] = npz.name
    report["array_cache_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Completed seed {request['seed']}: {len(report['groups'])} paired groups, {len(report['rows'])} rules", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
