"""Fresh complete A3 searches validate one immutable development-locked rule.

Each task is run in an isolated subprocess. No old fixed-candidate cache is
passed, and this entrypoint never selects a rule or accesses confirmation data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from numerical_simulation.candidate_search_study import run
from numerical_simulation.candidate_search_study.worker import (
    audit_pool, configure_core, jsonable, pool_fingerprint,
)
from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test, select_candidate


def validate_lock(lock, search_plan, msic_plan, summary_sha256):
    from numerical_simulation.candidate_search_study.msic_after_search import declared_rules
    run.require(lock.get("status") == "locked" and lock.get("search_group") == "A3", "An immutable locked A3 rule is required")
    run.require(lock.get("development_search_plan_id") == search_plan["plan_id"], "Locked search plan differs")
    run.require(lock.get("development_msic_plan_id") == msic_plan["plan_id"], "Locked MSIC plan differs")
    run.require(msic_plan["input_plan_id"] == search_plan["plan_id"], "MSIC/search parent plans differ")
    run.require(lock.get("development_summary_sha256") == summary_sha256, "Locked development summary differs")
    rule = lock["rule"]
    declared = [{"id": r["rule"], **{k: r[k] for k in ("criterion", "alpha", "gamma", "scale")}} for r in declared_rules()]
    run.require(rule in declared, "Locked rule is not one of the predeclared global rules")
    return rule


def core_fit(*args, **kwargs):
    from tensor_abess.tensor.abess import tensor_abess
    return tensor_abess(*args, **kwargs)


def fit_fresh(X, y, config, rule):
    msic = rule["criterion"] == "msic"
    return core_fit(X, y, k_max=config.k_max, c=config.abess_init_c,
        criterion=config.abess_init_criterion, n_lambdas=config.n_lambdas, cv_folds=config.cv_folds,
        max_iter=config.max_iter, tol=config.tol, max_cycles=config.max_cycles, eps=1e-8,
        adaptive_criterion="msic", msic_refine_alpha=rule["alpha"] if msic else 1.5,
        msic_refine_gamma=1.0, msic_refine_one_se_scale=rule["scale"] if msic else 1.0,
        cap_policy="candidate", grid_policy="full", candidate_budget=2000, fixed_candidate_cache=None)


def assert_model_matches(candidate, row, cached):
    run.require(np.array_equal(candidate["B"], cached["B"]), "Fresh/cached coefficients differ")
    supports = [S.tolist() for S in candidate["supports"]]
    run.require(supports == row["supports"] == [S.tolist() for S in cached["supports"]], "Fresh/cached supports differ")
    run.require(np.isclose(candidate["rss"], row["rss_train"], atol=1e-10, rtol=1e-10), "Fresh/cached training RSS differs")
    run.require(np.isclose(candidate["loss"], row["loss_train"], atol=1e-10, rtol=1e-10), "Fresh/cached training loss differs")


def load_context(source_output, msic_output, lock_file, task_id):
    from numerical_simulation.candidate_search_study.msic_after_search import validate_complete_cohort
    source_output, msic_output, lock_file = map(lambda p: Path(p).resolve(), (source_output, msic_output, lock_file))
    search_plan, manifests = validate_complete_cohort(source_output)
    run.require(search_plan["cohort"] == "development", "Fresh representative validation requires development data only")
    msic_plan = run.read_json(msic_output / "plan.json")
    run.require(msic_plan["plan_id"] == run.digest({k: v for k, v in msic_plan.items() if k != "plan_id"}), "MSIC plan checksum differs")
    integrity = run.read_json(msic_output / "integrity_report.json")
    run.require(integrity["status"] == "complete" and integrity["checks_passed"]
        and integrity["plan_id"] == msic_plan["plan_id"]
        and integrity["dataset_count"] == len(search_plan["tasks"])
        and integrity["default_exact_replays"] == len(search_plan["tasks"])
        and integrity["pool_fingerprints_unchanged"], "Complete replayed MSIC development output required")
    run.require(msic_plan["inputs"] == manifests, "MSIC parent input manifests differ")
    run.require(msic_plan["frozen_input_sources"] == search_plan["sources"], "MSIC parent sources differ")
    run.require(msic_plan["environment"] == run.environment(), "MSIC environment differs")
    for relative, expected in msic_plan["research_sources"].items():
        run.require(run.file_hash(run.root() / relative) == expected, "Frozen scoring source changed")
    lock = run.read_json(lock_file)
    rule = validate_lock(lock, search_plan, msic_plan, run.file_hash(msic_output / "rule_summary.csv"))
    task = next((t for t in search_plan["tasks"] if t["task_id"] == task_id), None)
    run.require(task is not None, "Unknown representative task ID")
    source_job = run.result_path(source_output, task)
    source = run.validate_result(source_output, search_plan, task)
    scoring_job = msic_output / "jobs" / f"task_{task_id:03d}.json"
    scored = run.read_json(scoring_job)
    source_manifest = next(record for record in manifests if record["task_id"] == task_id)
    run.require(scored["status"] == "complete" and scored["plan_id"] == msic_plan["plan_id"]
        and scored["input_result_manifest"] == source_manifest, "Cached task parent fingerprint differs")
    for key in ("seed", "test_seed", "config", "environment", "fingerprints"):
        run.require(scored[key] == source[key], f"Cached/source task {key} differs")
    run.require(scored["pool_fingerprint_before"] == scored["pool_fingerprint_after"]
        == source["groups"]["A3"]["pool_fingerprint"], "Cached/source pool differs")
    cache = scoring_job.with_suffix(".npz")
    run.require(scored["array_cache"] == cache.name and scored["array_cache_sha256"] == run.file_hash(cache), "Cached selected array checksum differs")
    provenance = {"development_search_plan_id": search_plan["plan_id"], "development_msic_plan_id": msic_plan["plan_id"],
        "source_job_sha256": run.file_hash(source_job), "source_npz_sha256": source["array_cache_sha256"],
        "scoring_job_sha256": run.file_hash(scoring_job), "scoring_npz_sha256": run.file_hash(cache),
        "lock_file_sha256": run.file_hash(lock_file), "development_summary_sha256": run.file_hash(msic_output / "rule_summary.csv"),
        "fresh_module_sha256": run.file_hash(Path(__file__)), "search_sources": search_plan["sources"],
        "scoring_sources": msic_plan["research_sources"], "environment": run.environment()}
    return source, scored, cache, rule, provenance


def run_case(source_output, msic_output, lock_file, task_id):
    configure_core()
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from numerical_simulation.candidate_search_study.numerics import audit_selected_model
    source, scored, cache, rule, provenance = load_context(source_output, msic_output, lock_file, task_id)
    values = dict(source["config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    dataset = generate_simulation_dataset(config, seed=int(source["seed"]))
    for key, array in (("X_train", dataset.X_train), ("y_train", dataset.y_train), ("B_true", dataset.B_true)):
        run.require(array_fingerprint(array) == source["fingerprints"][key], f"Fresh generated {key} differs")
    start = perf_counter()
    fitted = fit_fresh(dataset.X_train, dataset.y_train, config, rule)
    runtime = perf_counter() - start
    pool = fitted.bcd_result.candidate_records
    audit_pool(pool, config.n_train, config.dims)
    fingerprint = pool_fingerprint(pool)
    run.require(fingerprint == source["groups"]["A3"]["pool_fingerprint"], "Fresh candidate pool differs from frozen A3")
    locked, selection = select_candidate(pool, config.n_train, config.dims, criterion=rule["criterion"],
        alpha=rule["alpha"] or 1.5, scale=rule["scale"])
    default, default_selection = select_candidate(pool, config.n_train, config.dims, alpha=1.5, scale=1.0)
    if rule["criterion"] == "msic":
        run.require(np.array_equal(locked["B"], fitted.B), "Public locked-MSIC fit and training selector differ")
        run.require(np.isclose(fitted.sic, selection["selected_score"], atol=1e-10, rtol=1e-10), "Public locked-MSIC score differs")
    predictions, test_fingerprint = evaluate_shared_test(dataset, [locked["B"], default["B"]], source["test_seed"])
    run.require(test_fingerprint == source["fingerprints"]["independent_test"], "Fresh independent test stream differs")
    choices = [(rule["id"], locked, selection, predictions[0]),
               ("msic-alpha1.5-scale1", default, default_selection, predictions[1])]
    rows, arrays, numeric = [], {}, {}
    with np.load(cache, allow_pickle=False) as saved:
        for name, candidate, info, prediction in choices:
            cached_row = next(row for row in scored["rows"] if row["rule"] == name)
            index = cached_row["selected_index"]
            old = {"B": saved[f"candidate_{index}_B"], "supports": [saved[f"candidate_{index}_S{mode}"] for mode in (1, 2, 3)]}
            assert_model_matches(candidate, cached_row, old)
            run.require(info == cached_row["selection"], "Fresh/cached complete scoring selection differs")
            for key, value in prediction.items():
                run.require(np.isclose(value, cached_row[key], atol=1e-10, rtol=1e-10), f"Fresh/cached test metric {key} differs")
            supports = candidate["supports"]
            numeric[name] = audit_selected_model(dataset.X_train, dataset.y_train, candidate["B"], supports,
                expected_rss=candidate["rss"], expected_loss=candidate["loss"])
            row = {"rule": name, "rss_train": candidate["rss"], "loss_train": candidate["loss"],
                "selection": info, "supports": [S.tolist() for S in supports], "B_fingerprint": array_fingerprint(candidate["B"]),
                **compute_support_metrics(*supports, *dataset.supports_true),
                **compute_coefficient_metrics(candidate["B"], dataset.B_true), **prediction}
            rows.append(row)
            arrays[name + "_B"] = candidate["B"]
            for mode, S in enumerate(supports, 1):
                arrays[name + f"_S{mode}"] = S
    before_default = next(row for row in source["rows"] if row["group"] == "A3")
    run.require(default_selection == before_default["selection"]
        and array_fingerprint(default["B"]) == before_default["B_fingerprint"], "Fresh original-default replay differs")
    run.require(pool_fingerprint(pool) == fingerprint, "Fresh pool changed during evaluation")
    report = {"status": "complete", "task_id": task_id, "rule": rule, "config": source["config"],
        "seed": source["seed"], "test_seed": source["test_seed"], "rows": rows, "candidate_count": len(pool),
        "pool_fingerprint": fingerprint, "cache_pool_identical": True, "locked_rule_exact_replay": True,
        "default_exact_replay": True, "test_fingerprint": test_fingerprint, "numerics": numeric,
        "fingerprints": source["fingerprints"], "environment": source["environment"], "provenance": provenance,
        "runtime_sec": runtime, "fresh_complete_search": True, "old_fixed_cache_used": False,
        "interpretation": "Fresh complete A3 support search; locked and original-default global rules replayed from this one fresh pool. No second default fit, no rule selection, and no confirmation access."}
    return jsonable(report), arrays


def worker_main(args):
    output = args.output_dir / f"task_{args.task_id:03d}.json"
    cache = output.with_suffix(".npz")
    if output.exists() or cache.exists():
        raise FileExistsError("Fresh validation outputs are immutable")
    report, arrays = run_case(args.source_output, args.msic_output, args.lock_file, args.task_id)
    with cache.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report.update(array_cache=cache.name, array_cache_sha256=run.file_hash(cache))
    run.write_json(output, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--msic-output", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int, choices=(0, 40, 80))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"task_{args.task_id:03d}"
    targets = [prefix.with_suffix(ext) for ext in (".json", ".npz", ".failure.json", ".stdout.txt", ".stderr.txt")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Task output/log/failure already exists; review it without overwriting")
    command = [sys.executable, "-m", "numerical_simulation.candidate_search_study.msic_refit", "--worker",
        "--source-output", str(args.source_output.resolve()), "--msic-output", str(args.msic_output.resolve()),
        "--lock-file", str(args.lock_file.resolve()), "--task-id", str(args.task_id), "--output-dir", str(args.output_dir.resolve())]
    child_env = dict(os.environ)
    child_env.update({key: "1" for key in run.THREADS})
    with targets[3].open("x", encoding="utf8") as stdout, targets[4].open("x", encoding="utf8") as stderr:
        completed = subprocess.run(command, cwd=run.root(), env=child_env, stdout=stdout, stderr=stderr)
    if completed.returncode:
        run.write_json(targets[2], {"status": "failed", "task_id": args.task_id, "returncode": completed.returncode,
            "utc_time": datetime.now(timezone.utc).isoformat(), "stderr_log": targets[4].name})
        raise RuntimeError(f"Fresh task {args.task_id} failed; see {targets[2]}")
    report = run.read_json(targets[0])
    run.require(report["status"] == "complete" and report["array_cache_sha256"] == run.file_hash(targets[1]), "Fresh output/cache differs")
    print(f"Fresh complete A3 validation passed: task {args.task_id}; locked rule {report['rule']['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
