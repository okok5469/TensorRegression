"""Two fixed MSIC rules on untouched historical development A2 candidate pools.

Run only after the A2 search decision permits Phase C. This tool does not fit
models, select a winning rule, evaluate confirmation, or approve a release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from numerical_simulation.candidate_search_study import run as legacy
from numerical_simulation.candidate_search_study.msic_after_search import default_replay, write_csv
from numerical_simulation.candidate_search_study.worker import audit_pool, configure_core, pool_fingerprint
from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test, select_candidate

digest = legacy.digest
file_hash = legacy.file_hash


def declared_rules():
    return [{"rule": "msic-alpha1.5-scale1", "role": "default", "criterion": "msic", "alpha": 1.5,
             "gamma": 1., "scale": 1.},
            {"rule": "msic-alpha2-scale0", "role": "mild_candidate", "criterion": "msic", "alpha": 2.,
             "gamma": 1., "scale": 0.}]


def two_choices(metadata, n, dims):
    """Select from training RSS and shape metadata only; ignore evaluation fields."""
    before = legacy.canonical(metadata)
    choices = []
    for rule in declared_rules():
        _, info = select_candidate(metadata, n, dims, criterion="msic", alpha=rule["alpha"], scale=rule["scale"])
        choices.append((rule, int(info["selected_index"]), info))
    legacy.require(legacy.canonical(metadata) == before, "Two-rule scoring changed frozen candidate metadata")
    return choices


def validate_plan_contract(plan):
    legacy.require(plan.get("cohort") == "development", "Only the original development cohort is permitted")
    legacy.require(plan["plan_id"] == digest({k: v for k, v in plan.items() if k != "plan_id"}), "Development plan checksum differs")
    legacy.require(plan["n_reps"] == 10 and len(plan["tasks"]) == 90, "Complete original development 90 required")
    groups = ["A0", "A1", "A2", "A3", "A4"]
    legacy.require(plan["groups"] == groups and plan["version"] == "v5.2-candidate-search", "Original development groups/version differ")
    legacy.require(plan["tasks"] == legacy.build_tasks("development", 10, groups), "Original development seeds/configs/requests differ")


def validate_complete_cohort(input_dir):
    input_dir = Path(input_dir).resolve()
    legacy.require(not (input_dir / "STOPPED_FOR_AUDIT.json").exists(), "Stopped development input is excluded")
    plan = legacy.read_json(input_dir / "plan.json")
    validate_plan_contract(plan)
    integrity = legacy.read_json(input_dir / "integrity_report.json")
    legacy.require(integrity["status"] == "complete" and integrity["integrity_checks_passed"]
                   and integrity["plan_id"] == plan["plan_id"] and integrity["completed_tasks"] == 90
                   and integrity["expected_tasks"] == 90 and integrity["paired_group_rows"] == 450,
                   "Complete original development integrity evidence required")
    legacy.require(not any((input_dir / "failures").iterdir()), "Failed development cohort excluded")
    # Historical Git is provenance, not an equality constraint on current HEAD.
    legacy.require(plan["sources"] == legacy.source_manifests(), "Frozen development core/experiment sources changed")
    legacy.require(plan["environment"] == legacy.environment(), "Frozen development environment changed")
    expected_jobs = {f"task_{task['task_id']:03d}{suffix}" for task in plan["tasks"] for suffix in (".json", ".npz")}
    legacy.require({p.name for p in (input_dir / "jobs").iterdir()} == expected_jobs, "Incomplete/unknown development job inventory")
    expected_requests = {Path(task["request_file"]).name for task in plan["tasks"]}
    legacy.require({p.name for p in (input_dir / "requests").iterdir()} == expected_requests, "Incomplete/unknown development request inventory")
    manifests = []
    for task in plan["tasks"]:
        request_path = input_dir / task["request_file"]
        request = legacy.read_json(request_path)
        legacy.require(request == task["request"] and digest(request) == task["request_id"], "Frozen request differs")
        report = legacy.validate_result(input_dir, plan, task)
        path = legacy.result_path(input_dir, task)
        manifests.append({"task_id": task["task_id"], "json_sha256": file_hash(path),
            "npz_sha256": report["array_cache_sha256"], "request_sha256": file_hash(request_path),
            "A2_pool_fingerprint": report["groups"]["A2"]["pool_fingerprint"]})
    return plan, manifests


def stream_pool(group, arrays):
    """Stream the A2 arrays; existing A3-specific helpers remain untouched."""
    for index, candidate in enumerate(group["candidates"]):
        legacy.require(candidate["candidate_index"] == index, "Frozen candidate ordering differs")
        prefix = f"A2_candidate_{index}_"
        B = arrays[prefix + "B"]
        supports = [arrays[prefix + f"S{mode}"] for mode in (1, 2, 3)]
        legacy.require(array_fingerprint(B) == candidate["B_fingerprint"], "Candidate B fingerprint differs")
        legacy.require([S.tolist() for S in supports] == candidate["supports"], "Candidate support arrays differ")
        yield {**candidate, "B": B, "supports": supports}


def evaluate_task(report, npz_path):
    """Regenerate evaluation streams; score the frozen A2 pool without fitting."""
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from tensor_abess.tensor.ops import tensor_loss

    started = perf_counter()
    values = dict(report["config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    data = generate_simulation_dataset(config, seed=int(report["seed"]))
    for key, value in (("X_train", data.X_train), ("y_train", data.y_train), ("B_true", data.B_true)):
        legacy.require(array_fingerprint(value) == report["fingerprints"][key], f"Regenerated {key} fingerprint differs")
    group = report["groups"]["A2"]
    metadata = group["candidates"]
    original = next(row for row in report["rows"] if row["group"] == "A2")
    choices = two_choices(metadata, config.n_train, config.dims)
    selected = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        for candidate in stream_pool(group, arrays):
            audit_pool([candidate], config.n_train, config.dims)
        before = pool_fingerprint(stream_pool(group, arrays))
        legacy.require(before == group["pool_fingerprint"], "Frozen A2 training pool fingerprint differs")
        for _, index, _ in choices:
            if index in selected:
                continue
            prefix = f"A2_candidate_{index}_"
            B = arrays[prefix + "B"]
            supports = [arrays[prefix + f"S{mode}"] for mode in (1, 2, 3)]
            loss = tensor_loss(data.X_train, data.y_train, B)
            rss = 2 * config.n_train * loss
            legacy.require(np.isclose(rss, metadata[index]["rss"], rtol=1e-10, atol=1e-10)
                           and np.isclose(loss, metadata[index]["loss"], rtol=1e-10, atol=1e-10),
                           "Selected coefficients disagree with frozen training RSS/loss")
            selected[index] = {"B": B, "supports": supports, "rss_recomputed": rss, "loss_recomputed": loss}
        predictions, test_fingerprint = evaluate_shared_test(data, [selected[index]["B"] for _, index, _ in choices], int(report["test_seed"]))
        legacy.require(test_fingerprint == report["fingerprints"]["independent_test"], "Independent test stream fingerprint differs")
        after = pool_fingerprint(stream_pool(group, arrays))
        legacy.require(before == after, "Rescoring/test evaluation changed frozen training pool")
    rows, selected_arrays = [], {}
    for (rule, index, info), prediction in zip(choices, predictions):
        fitted = selected[index]
        candidate = metadata[index]
        supports = fitted["supports"]
        row = {**original, "rule": rule["rule"], "method": f"A2/{rule['rule']}",
            **compute_support_metrics(*supports, *data.supports_true),
            **compute_coefficient_metrics(fitted["B"], data.B_true), **prediction,
            "rss_train": candidate["rss"], "loss_train": candidate["loss"], "selection": info,
            "supports": [S.tolist() for S in supports], "B_fingerprint": array_fingerprint(fitted["B"])}
        if rule["role"] == "default":
            default_replay(original, row)
        row.pop("runtime_sec", None)
        q, truth = int(row["q_hat"]), int(row["q_true"])
        row.update(evaluation_role=rule["role"], criterion="msic", mode_alpha=1., alpha=rule["alpha"],
            gamma=1., one_se_scale=rule["scale"], selected_index=index, candidate_kind=candidate["kind"],
            source_A2_runtime_sec=original["runtime_sec"], train_mse=candidate["rss"] / config.n_train,
            test_mse=row["pmse"], rss_train_recomputed=fitted["rss_recomputed"],
            loss_train_recomputed=fitted["loss_recomputed"], q_over=int(q > truth), q_under=int(q < truth),
            q_small=int(q == 1), q_near_cap=int(q >= .9 * np.floor(.7 * config.n_train)))
        rows.append(row)
        selected_arrays[rule["role"] + "_B"] = fitted["B"]
        for mode, support in enumerate(supports, 1):
            selected_arrays[rule["role"] + f"_S{mode}"] = support
    return {"status": "complete", "config": report["config"], "seed": report["seed"],
        "test_seed": report["test_seed"], "rows": rows, "default_exact_replay": True,
        "pool_fingerprint_before": before, "pool_fingerprint_after": after,
        "fingerprints": report["fingerprints"], "environment": report["environment"],
        "new_model_fitting": False, "rule_selection_performed": False, "confirmation_performed": False,
        "elapsed_sec": perf_counter() - started}, selected_arrays


def summary_rows(rows, scenarios=False):
    metrics = ("q_hat", "exact_recovery", "mode_f1_macro", "element_f1", "ree", "pmse", "signal_mse",
               "r2_test", "rss_train", "loss_train", "train_mse", "test_mse", "q_over", "q_under", "q_small", "q_near_cap")
    groups = {}
    for row in rows:
        key = (row["n_train"], row["snr"], row["rule"]) if scenarios else (row["rule"],)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        item = {"rule": key[-1], "n_datasets": len(members)}
        if scenarios:
            item.update(n_train=key[0], snr=key[1])
        for metric in metrics:
            values = np.asarray([row[metric] for row in members], dtype=float)
            legacy.require(np.isfinite(values).all(), "Nonfinite rescoring metric")
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
        output.append(item)
    return output


def run_cohort(input_dir, output_dir):
    input_dir, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("A2 controlled rescoring requires a new immutable output directory")
    configure_core()
    parent, manifests = validate_complete_cohort(input_dir)
    own_paths = [Path(__file__), legacy.root() / "numerical_simulation/candidate_search_study/msic_after_search.py",
        legacy.root() / "numerical_simulation/candidate_search_study/worker.py",
        legacy.root() / "numerical_simulation/version_validation/worker.py",
        legacy.root() / "numerical_simulation/data_generation.py", legacy.root() / "numerical_simulation/metrics.py"]
    sources = {path.relative_to(legacy.root()).as_posix(): file_hash(path) for path in own_paths}
    protocol = {"stage": "A2-controlled-development-MSIC-rescoring", "research_only": True,
        "release_approved": False, "group": "A2", "rules": declared_rules(), "mode_alpha": 1.,
        "gamma": 1., "input_cohort": "development", "dataset_count": 90,
        "entry_gate": "Execute only after the independent A2 search decision permits Phase C",
        "rule_selection_performed": False, "confirmation_performed": False, "new_model_fitting": False,
        "purpose": "Controlled descriptive A2-pool development evidence; fixed two rules; original MSIC remains default pending separate evidence"}
    body = {"protocol": protocol, "protocol_id": digest(protocol), "input_dir": str(input_dir),
        "source_plan_id": parent["plan_id"], "source_plan_sha256": file_hash(input_dir / "plan.json"),
        "source_integrity_sha256": file_hash(input_dir / "integrity_report.json"), "inputs": manifests,
        "frozen_input_sources": parent["sources"], "environment": legacy.environment(), "research_sources": sources,
        "historical_git_commit": parent["git_commit"], "created_utc": datetime.now(timezone.utc).isoformat()}
    plan = {**body, "plan_id": digest(body)}
    output.mkdir(parents=True)
    (output / "jobs").mkdir()
    legacy.write_json(output / "protocol.json", protocol)
    legacy.write_json(output / "plan.json", plan)
    rows, started = [], perf_counter()
    for task, manifest in zip(parent["tasks"], manifests):
        path = legacy.result_path(input_dir, task)
        legacy.require(file_hash(path) == manifest["json_sha256"] and file_hash(path.with_suffix(".npz")) == manifest["npz_sha256"],
                       "Input changed after the controlled scoring plan was frozen")
        legacy.require(all(file_hash(legacy.root() / relative) == expected for relative, expected in sources.items()), "Frozen scoring source changed")
        result, arrays = evaluate_task(legacy.read_json(path), path.with_suffix(".npz"))
        result.update(task_id=task["task_id"], input_result_manifest=manifest, plan_id=plan["plan_id"])
        for row in result["rows"]:
            row.update(task_id=task["task_id"], scenario_id=task["scenario_id"], replicate=task["replicate"])
        target = output / "jobs" / f"task_{task['task_id']:03d}.json"
        cache = target.with_suffix(".npz")
        with cache.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        result.update(array_cache=cache.name, array_cache_sha256=file_hash(cache))
        legacy.write_json(target, result)
        rows.extend(result["rows"])
        print(f"A2 frozen-pool two-rule development evaluation {len(rows)//2}/90", flush=True)
    legacy.require(len(rows) == 180, "Incomplete two-rule development evidence")
    compact = [{key: value for key, value in row.items() if key not in ("selection", "supports")} for row in rows]
    write_csv(output / "raw_results.csv", compact)
    write_csv(output / "rule_summary.csv", summary_rows(rows))
    write_csv(output / "scenario_summary.csv", summary_rows(rows, scenarios=True))
    integrity = {"status": "complete", "checks_passed": True, "plan_id": plan["plan_id"],
        "dataset_count": 90, "rule_rows": 180, "default_exact_replays": 90,
        "pool_fingerprints_unchanged": True, "new_model_fitting": False, "rule_selection_performed": False,
        "confirmation_performed": False, "release_approved": False, "elapsed_sec": perf_counter() - started,
        "scope": "Historical development A2 candidates only; two predeclared global MSIC rules; no independent confirmation"}
    legacy.write_json(output / "integrity_report.json", integrity)
    return integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = run_cohort(**vars(parser.parse_args(argv)))
    print(f"Completed {result['dataset_count']} A2 development datasets; no rule selection or formal approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
