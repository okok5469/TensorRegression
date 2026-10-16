"""Research-only immutable A0/A2 paired validation for a frozen new-shape protocol."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

from numerical_simulation.candidate_search_study import run as legacy
from numerical_simulation.candidate_search_study import worker

root = legacy.root
canonical = legacy.canonical
digest = legacy.digest
file_hash = legacy.file_hash
read_json = legacy.read_json
require = legacy.require
write_json = legacy.write_json


def source_manifests():
    sources = legacy.source_manifests()
    for folder in ("a2_final_validation", "candidate_search_study"):
        for path in sorted((root() / "numerical_simulation" / folder).glob("*.py")):
            sources["experiment"][path.relative_to(root()).as_posix()] = file_hash(path)
    return sources


def build_plan(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = read_json(protocol_path)
    require(protocol["groups"] == ["A0", "A2"], "Protocol must freeze A0 then A2")
    require(protocol["research_only"] and not protocol["release_approved"], "Research-only protocol required")
    sources = source_manifests()
    require(protocol["core_sources"] == sources["core"], "Protocol core sources changed")
    require(protocol["environment"] == legacy.environment(), "Protocol environment changed")
    tasks = []
    for scenario_id, scenario in enumerate(protocol["scenarios"]):
        for replicate in range(protocol["n_reps"]):
            task_id = len(tasks)
            seed = protocol["seed_base"] + 1000 * scenario_id + replicate
            config = {**protocol["base_config"], **{key: scenario[key] for key in ("sparsity", "n_train", "snr", "rho")}}
            request = {"config": config, "seed": seed, "test_seed": seed + 1_000_000,
                       "groups": protocol["groups"], "fit_options": {
                           "candidate_budget": protocol["solver"]["candidate_budget"],
                           "selection": "original MSIC alpha=1.5,gamma=1,one_se_scale=1; Phase D only",
                           "cache": protocol["cache_policy"]}}
            tasks.append({"task_id": task_id, "scenario_id": scenario_id, "scenario": scenario,
                          "replicate": replicate, "request": request,
                          "request_file": f"requests/task_{task_id:03d}.json", "request_id": digest(request)})
    require(len(set(t["request"]["seed"] for t in tasks)) == len(tasks), "Duplicate protocol seeds")
    require(set(protocol["cold_check_tasks"]) <= set(range(len(tasks))), "Unknown cold task")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root(), capture_output=True, text=True, check=True).stdout.strip()
    body = {"schema_version": 1, "version": worker.VERSION, "cohort": "new_shape_final_validation",
            "groups": protocol["groups"], "n_reps": protocol["n_reps"], "protocol": protocol,
            "protocol_sha256": file_hash(protocol_path), "protocol_id": digest(protocol), "tasks": tasks, "sources": sources,
            "environment": legacy.environment(), "git_commit": commit}
    return {**body, "plan_id": digest(body)}


def create_plan(output):
    output = Path(output)
    if any(p.name.lower() not in {"protocol.json", "protocol.sha256"} for p in output.iterdir()):
        raise FileExistsError("Plan requires protocol-only output directory")
    plan = build_plan(output / "protocol.json")
    for name in ("requests", "jobs", "logs", "failures", "cold", "cold_requests"):
        (output / name).mkdir()
    for task in plan["tasks"]:
        write_json(output / task["request_file"], task["request"])
        if task["task_id"] in plan["protocol"]["cold_check_tasks"]:
            for group in plan["groups"]:
                write_json(output / "cold_requests" / f"task_{task['task_id']:03d}_{group}.json", {**task["request"], "groups": [group]})
    write_json(output / "plan.json", plan)
    return plan


def check_inventory(output, plan):
    require(not any((output / "failures").iterdir()), "Recorded failure prevents execution or summary")
    expected_requests = {Path(t["request_file"]).name for t in plan["tasks"]}
    require({p.name for p in (output / "requests").iterdir()} == expected_requests, "Unknown/missing request file")
    expected_cold = {f"task_{i:03d}_{g}.json" for i in plan["protocol"]["cold_check_tasks"] for g in plan["groups"]}
    require({p.name for p in (output / "cold_requests").iterdir()} == expected_cold, "Unknown/missing cold request file")
    allowed = {f"task_{t['task_id']:03d}{s}" for t in plan["tasks"] for s in (".json", ".npz")}
    require(all(p.is_file() and p.name in allowed for p in (output / "jobs").iterdir()), "Unknown result file")
    allowed = {f"task_{i:03d}_{g}{s}" for i in plan["protocol"]["cold_check_tasks"] for g in plan["groups"] for s in (".json", ".npz")}
    require(all(p.is_file() and p.name in allowed for p in (output / "cold").iterdir()), "Unknown cold result file")
    for directory in ("jobs", "cold"):
        for path in (output / directory).iterdir():
            require(path.with_suffix(".json").is_file() and path.with_suffix(".npz").is_file(), "Incomplete result/cache pair")


def load_plan(output):
    output = Path(output)
    plan = read_json(output / "plan.json")
    require(plan["plan_id"] == digest({k: v for k, v in plan.items() if k != "plan_id"}), "Plan checksum mismatch")
    require(file_hash(output / "protocol.json") == plan["protocol_sha256"], "Protocol checksum changed")
    require(plan == build_plan(output / "protocol.json"), "Frozen sources/environment/commit/requests changed")
    for task in plan["tasks"]:
        require(read_json(output / task["request_file"]) == task["request"], "Frozen request changed")
        if task["task_id"] in plan["protocol"]["cold_check_tasks"]:
            for group in plan["groups"]:
                p = output / "cold_requests" / f"task_{task['task_id']:03d}_{group}.json"
                require(read_json(p) == {**task["request"], "groups": [group]}, "Frozen cold request changed")
    check_inventory(output, plan)
    return plan


def training_caps(config, init_supports):
    n = config["n_train"]
    floor = max(1, int(np.floor(np.sqrt(n) / 2)))
    caps = [max(1, min(max(int(np.ceil(config["abess_init_c"] * len(S))), floor), dim)) for S, dim in zip(init_supports, config["dims"])]
    while np.prod(caps) > max(1, int(np.floor(.7 * n))):
        largest = int(np.argmax(caps))
        caps[largest] -= 1
    return caps


def audit_candidates(report, array_path):
    config = report["config"]
    n, dims = config["n_train"], config["dims"]
    expected_arrays = {"B_true"}
    with np.load(array_path, allow_pickle=False) as arrays:
        require(worker.array_fingerprint(arrays["B_true"]) == report["fingerprints"]["B_true"], "Truth cache checksum differs")
        for group, info in report["groups"].items():
            metadata = info["search_metadata"]
            caps = training_caps(config, info["init_supports"])
            require(metadata["caps"] == info["s_max"] == caps, "Training caps differ")
            require(metadata["cap_policy"] == info["cap_policy"] == "legacy", "Unexpected cap policy")
            policy = worker.GROUPS[group][1]
            require(metadata["grid_policy"] == info["grid_policy"] == policy, "Unexpected grid policy")
            capacity = max(1, int(np.floor(.7 * n)))
            require(metadata["max_params"] == capacity, "Training capacity differs")
            legal = [list(s) for s in product(*(range(1, cap + 1) for cap in caps)) if np.prod(s) <= capacity]
            require(metadata["legal_shapes"] == legal, "Incomplete legal shapes")
            values = [sorted({*range(max(1, cap - 2), cap + 1), 1, max(1, round(.4 * cap)), max(1, round(.6 * cap))}) for cap in caps]
            grid = legal if policy == "full" else [list(s) for s in product(*values) if np.prod(s) <= capacity] or [[1,1,1]]
            require(metadata["size_grid"] == metadata["initial_size_grid"] == grid and metadata["refinement_size_grid"] == [], "Frozen size grid differs")
            excluded = [s for s in legal if s not in grid]
            require(metadata["excluded_legal_shapes"] == excluded and metadata["grid_complete"] == (not excluded), "Grid coverage differs")
            require(len(grid) <= metadata["candidate_budget"] == report["fit_options"]["candidate_budget"], "Candidate budget differs")
            pool = []
            for index, candidate in enumerate(info["candidates"]):
                require(candidate["candidate_index"] == index, "Candidate index differs")
                names = [f"{group}_candidate_{index}_B", *[f"{group}_candidate_{index}_S{mode}" for mode in (1,2,3)]]
                expected_arrays.update(names)
                B, *supports = [arrays[name] for name in names]
                require(candidate["B_fingerprint"] == worker.array_fingerprint(B), "Candidate coefficient checksum differs")
                require(all(np.array_equal(s, recorded) for s, recorded in zip(supports, candidate["supports"])), "Candidate support cache differs")
                require(np.isfinite(candidate["fit_seconds"]) and candidate["fit_seconds"] >= 0, "Invalid candidate fit cost")
                pool.append({**candidate, "B": B, "supports": supports})
            worker.audit_pool(pool, n, dims)
            require(worker.pool_fingerprint(pool) == info["pool_fingerprint"], "Training pool differs")
            require(len(pool) == info["candidate_count"] == len(grid) + 1, "Candidate count differs")
            require([list(c["sizes"]) for c in pool if c["kind"] == "global-fixed"] == grid, "Evaluated candidates differ from grid")
            require(all(all(s <= cap for s, cap in zip(c["sizes"], caps)) for c in pool), "Candidate exceeds mode cap")
            require(info["cache_hits"] == sum(bool(c["cache_hit"]) for c in pool), "Cache hit count differs")
            selected, _ = worker.score_pool(pool, n, dims, alpha=1.5, scale=1.0)
            row = next(r for r in report["rows"] if r["group"] == group)
            require(row["B_fingerprint"] == worker.array_fingerprint(selected["B"]) and row["supports"] == [s.tolist() for s in selected["supports"]], "Original MSIC selection differs")
            cached = sum(c["fit_seconds"] for c in pool if c["kind"] == "global-fixed" and c["cache_hit"])
            row.update(cached_fixed_fit_cost_sec=cached, standalone_cost_estimate_sec=row["runtime_sec"] + cached)
        require(set(arrays.files) == expected_arrays, "Unknown/missing cached array")


def validate_result(output, plan, task):
    report = legacy.validate_result(output, plan, task)
    audit_candidates(report, legacy.result_path(output, task).with_suffix(".npz"))
    return report


def execute_task(output, plan, task):
    try:
        legacy.execute_task(output, plan, task)
        return validate_result(output, plan, task)
    except Exception as exc:
        write_json(Path(output) / "failures" / f"task_{task['task_id']:03d}_audit.json", {"status": "failed", "plan_id": plan["plan_id"], "error": str(exc)})
        raise


def controller_deadline(output, plan):
    path = Path(output) / "CONTROLLER_STARTED.json"
    if not path.exists():
        write_json(path, {"plan_id": plan["plan_id"], "utc_started": datetime.now(timezone.utc).isoformat()})
    started = read_json(path)
    require(started["plan_id"] == plan["plan_id"], "Controller plan differs")
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started["utc_started"])).total_seconds()
    return monotonic() + plan["protocol"]["budget"]["max_controller_seconds"] - elapsed


def controller_failure(output, plan, error, filename):
    write_json(Path(output) / "failures" / filename, {"status": "stopped", "plan_id": plan["plan_id"],
               "error": str(error), "utc_stopped": datetime.now(timezone.utc).isoformat(),
               "running_workers": "Scheduling stopped; already launched workers finish naturally; no resume or adoption"})


def validate_cold_result(output, plan, task, group):
    output = Path(output)
    path = output / "cold" / f"task_{task['task_id']:03d}_{group}.json"
    require(path.is_file() and path.with_suffix(".npz").is_file(), "Incomplete cold result/cache")
    report = read_json(path)
    for key in ("config", "seed", "test_seed", "fit_options"):
        require(canonical(report[key]) == canonical(task["request"][key]), f"Cold request mismatch {key}")
    require(report["status"] == "complete" and report["version"] == plan["version"], "Cold result incomplete")
    require(report["group_order"] == [group] and set(report["groups"]) == {group} and len(report["rows"]) == 1, "Cold group differs")
    require(report["rows"][0]["group"] == group and report["rows"][0]["rule"] == "msic-alpha1.5-scale1", "Cold MSIC row differs")
    require(report["array_cache_sha256"] == file_hash(path.with_suffix(".npz")) and report["array_cache"] == path.with_suffix(".npz").name, "Cold cache checksum differs")
    require(report["environment"] == plan["environment"]["libraries"] and report["fingerprints"]["core_sources"] == plan["sources"]["core"], "Cold environment/source differs")
    info = report["groups"][group]
    require(info["environment"] == report["environment"], "Cold group environment differs")
    require(all(info["fingerprints"][key] == report["fingerprints"][key] for key in ("X_train", "y_train", "B_true", "independent_test")), "Cold group/dataset fingerprint differs")
    audit_candidates(report, path.with_suffix(".npz"))
    require(not any(c["cache_hit"] for c in info["candidates"]), "Cold worker reused cache")
    shared = validate_result(output, plan, task)
    shared_row = next(r for r in shared["rows"] if r["group"] == group)
    require(report["fingerprints"] == shared["fingerprints"], "Cold/shared dataset differs")
    row = report["rows"][0]
    require(row["B_fingerprint"] == shared_row["B_fingerprint"] and row["supports"] == shared_row["supports"], "Cold/shared selection differs")
    require(np.isfinite(row["runtime_sec"]) and row["runtime_sec"] > 0, "Invalid cold runtime")
    return report


def cold_plan(output):
    output = Path(output)
    plan = load_plan(output)
    deadline = controller_deadline(output, plan)
    try:
        for task in plan["tasks"]:
            if task["task_id"] not in plan["protocol"]["cold_check_tasks"]:
                continue
            validate_result(output, plan, task)
            for group in plan["groups"]:
                require(monotonic() < deadline, "Frozen controller time budget exceeded")
                path = output / "cold" / f"task_{task['task_id']:03d}_{group}.json"
                if path.exists() or path.with_suffix(".npz").exists():
                    validate_cold_result(output, plan, task, group)
                    continue
                request = output / "cold_requests" / path.name
                child_env = {**os.environ, **plan["environment"]["worker_thread_environment"]}
                command = [sys.executable, "-m", "numerical_simulation.candidate_search_study.worker", "--request", str(request.resolve()), "--output", str(path.resolve())]
                result = subprocess.run(command, cwd=root(), env=child_env, capture_output=True, text=True, encoding="utf-8", errors="replace")
                for name, value in (("stdout", result.stdout), ("stderr", result.stderr)):
                    with (output / "logs" / f"cold_{path.stem}.{name}.txt").open("x", encoding="utf-8") as handle:
                        handle.write(value)
                require(result.returncode == 0, f"Cold worker returned {result.returncode}: {result.stderr}")
                validate_cold_result(output, plan, task, group)
                require(monotonic() < deadline, "Frozen controller time budget exceeded")
                print(f"Completed cold task {task['task_id']} group {group}", flush=True)
    except Exception as exc:
        controller_failure(output, plan, exc, "cold_controller_stopped.json")
        raise
    return {"plan_id": plan["plan_id"], "status": "cold-complete", "completed_tasks": len(plan["protocol"]["cold_check_tasks"])}


def run_plan(output, workers=1, task_ids=None):
    output = Path(output)
    plan = load_plan(output)
    require(isinstance(workers, int) and 1 <= workers <= plan["protocol"]["budget"]["workers"], "Workers exceed frozen budget")
    ids = set(task_ids) if task_ids is not None else {t["task_id"] for t in plan["tasks"]}
    require(ids <= {t["task_id"] for t in plan["tasks"]}, "Unknown task-id")
    pending = []
    for task in plan["tasks"]:
        path = legacy.result_path(output, task)
        if path.exists() or path.with_suffix(".npz").exists():
            validate_result(output, plan, task)
        elif task["task_id"] in ids:
            pending.append(task)
    deadline = controller_deadline(output, plan)
    iterator = iter(pending)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        require(monotonic() < deadline, "Frozen controller time budget exceeded")
        active = {}
        for _ in range(min(workers, len(pending))):
            task = next(iterator)
            active[executor.submit(execute_task, output, plan, task)] = task
        while active:
            done, _ = wait(active, timeout=max(0, min(60, deadline - monotonic())), return_when=FIRST_COMPLETED)
            require(monotonic() < deadline, "Frozen controller time budget exceeded")
            for future in done:
                task = active.pop(future)
                future.result()
                print(f"Completed task {task['task_id']}", flush=True)
            for _ in done:
                task = next(iterator, None)
                if task is not None:
                    active[executor.submit(execute_task, output, plan, task)] = task
    except Exception as exc:
        controller_failure(output, plan, exc, "controller_stopped.json")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return {"plan_id": plan["plan_id"], "status": "complete-selected-tasks", "completed_tasks": len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "cold", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            sub.add_argument("--workers", type=int, default=1)
            sub.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        from numerical_simulation.a2_final_validation.analyse import summarize
        result = summarize(args.output_dir)
    else:
        result = {"plan": lambda: create_plan(args.output_dir), "run": lambda: run_plan(args.output_dir, args.workers, args.task_id), "cold": lambda: cold_plan(args.output_dir)}[args.command]()
    print(canonical({"plan_id": result["plan_id"], "status": result.get("status", "planned")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
