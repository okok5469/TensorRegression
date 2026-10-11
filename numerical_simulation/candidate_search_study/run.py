"""Immutable paired candidate mechanism plans; no formal 700-task experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from numerical_simulation.candidate_search_study.worker import GROUPS, VERSION


THREADS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")
METRICS = ("q_hat", "exact_recovery", "mode_f1_macro", "element_f1", "pmse", "r2_test",
           "rss_train", "runtime_sec", "candidate_count", "true_size_in_pool", "true_support_in_pool")


def root():
    return Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment():
    return {"libraries": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "platform": platform.system(), "machine": platform.machine(),
            "worker_thread_environment": {key: "1" for key in THREADS}}


def source_manifests():
    source = root() / "doc/algorithm_versions" / VERSION / "src"
    paths = sorted((source / "tensor_abess").rglob("*.py"))
    require(bool(paths), "Missing isolated v5.2 core sources")
    experiment = [Path(__file__), Path(__file__).with_name("worker.py"), Path(__file__).with_name("__init__.py"),
                  root() / "numerical_simulation/version_validation/worker.py",
                  root() / "numerical_simulation/data_generation.py", root() / "numerical_simulation/metrics.py",
                  root() / "numerical_simulation/known_s_cluster/profiles.json"]
    return {"core": {p.relative_to(source).as_posix(): file_hash(p) for p in paths},
            "experiment": {p.relative_to(root()).as_posix(): file_hash(p) for p in experiment}}


def build_tasks(cohort, n_reps, groups):
    require(cohort in {"historical", "development", "confirmation"}, "Invalid cohort")
    require(isinstance(n_reps, int) and 1 <= n_reps <= 10, "n_reps must be in 1..10")
    require(bool(groups) and len(groups) == len(set(groups)) and set(groups) <= set(GROUPS), "Invalid/duplicate groups")
    require(cohort != "historical" or n_reps == 3, "Historical cohort has exactly 3 repetitions")
    baseline = read_json(root() / "numerical_simulation/known_s_cluster/profiles.json")["profiles"]["full"]["base_config"]
    historical = root() / "numerical_simulation/outputs_msic_ablation_v51_final"
    tasks = []
    for scenario, (n, snr) in enumerate((n, snr) for n in (100, 200, 400) for snr in (.5, 2.0, 4.0)):
        for replicate in range(n_reps):
            task_id = len(tasks)
            request_file = f"requests/task_{task_id:03d}.json"
            if cohort == "historical":
                original = read_json(historical / request_file)
                request = {key: original[key] for key in ("config", "seed", "test_seed")}
            else:
                base_seed = {"development": 20361010, "confirmation": 20461010}[cohort]
                seed = base_seed + 1000 * scenario + replicate
                request = {"config": {**baseline, "n_train": n, "snr": snr},
                           "seed": seed, "test_seed": seed + 1_000_000}
            request.update(groups=list(groups), fit_options={"candidate_budget": 2000,
                "selection": "original MSIC alpha=1.5,gamma=1,one_se_scale=1; Phase D only",
                "cache": "explicit shared fixed-candidate dictionary per dataset; frozen group order"})
            tasks.append({"task_id": task_id, "scenario_id": scenario, "replicate": replicate,
                          "request_file": request_file, "request": request, "request_id": digest(request)})
    return tasks


def build_plan(cohort, n_reps, groups):
    tasks = build_tasks(cohort, n_reps, groups)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root(), capture_output=True, text=True)
    body = {"schema_version": 1, "version": VERSION, "cohort": cohort, "n_reps": n_reps,
        "groups": list(groups), "tasks": tasks, "sources": source_manifests(), "environment": environment(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "purpose": "Phase D search mechanism comparison only; original MSIC unchanged; no parameter selection",
        "seed_definition": "historical exact 27 prior requests; development20361010 or confirmation20461010 +1000*scenario+rep; independent test +1000000",
        "scenario_order": "n=(100,200,400) outer,SNR=(.5,2,4) inner",
        "completion_requirement": "all frozen datasets/groups and identical input/test fingerprints before summary"}
    return {**body, "plan_id": digest(body)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def create_plan(output, cohort, n_reps, groups):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Plan requires a new empty output directory")
    plan = build_plan(cohort, n_reps, groups)
    for name in ("requests", "jobs", "logs", "failures"):
        (output / name).mkdir()
    for task in plan["tasks"]:
        write_json(output / task["request_file"], task["request"])
    write_json(output / "plan.json", plan)
    return plan


def load_plan(output):
    output = Path(output)
    plan = read_json(output / "plan.json")
    require(plan["plan_id"] == digest({key: value for key, value in plan.items() if key != "plan_id"}), "Plan checksum mismatch")
    expected = build_plan(plan["cohort"], plan["n_reps"], plan["groups"])
    for key in ("tasks", "sources", "environment", "groups", "version", "git_commit"):
        require(plan[key] == expected[key], f"Frozen {key} changed; use a fresh plan")
    for task in plan["tasks"]:
        request = read_json(output / task["request_file"])
        require(request == task["request"] and digest(request) == task["request_id"], "Frozen request changed")
    return plan


def validate_group_pairing(report):
    groups = list(report["groups"].values())
    require(bool(groups), "No group results")
    for group in groups[1:]:
        for key in ("X_train", "y_train", "B_true", "independent_test"):
            require(group["fingerprints"][key] == groups[0]["fingerprints"][key], f"Group {key} fingerprint differs")


def result_path(output, task):
    return Path(output) / "jobs" / f"task_{task['task_id']:03d}.json"


def validate_result(output, plan, task):
    path = result_path(output, task)
    npz = path.with_suffix(".npz")
    require(path.is_file() and npz.is_file(), f"Incomplete result/cache task {task['task_id']}")
    report = read_json(path)
    request = task["request"]
    for key in ("config", "seed", "test_seed", "fit_options"):
        require(canonical(report[key]) == canonical(request[key]), f"Result request mismatch {key}")
    require(report["version"] == plan["version"] and report["status"] == "complete", "Result incomplete/version differs")
    require(report["group_order"] == plan["groups"] and set(report["groups"]) == set(plan["groups"]), "Result groups differ")
    require(report["environment"] == plan["environment"]["libraries"], "Result environment differs")
    require(report["fingerprints"]["core_sources"] == plan["sources"]["core"], "Result source differs")
    require(report["array_cache"] == npz.name and report["array_cache_sha256"] == file_hash(npz), "Array cache checksum differs")
    require(len(report["rows"]) == len(plan["groups"]), "Phase D must have exactly one original MSIC row per group")
    require({row["group"] for row in report["rows"]} == set(plan["groups"]), "Missing/duplicate group rows")
    require(all(row["rule"] == "msic-alpha1.5-scale1" for row in report["rows"]), "Unexpected MSIC ablation in Phase D")
    require(report["fingerprints"]["independent_test"]["n_observations"] == request["config"]["n_test"], "Incomplete test stream")
    validate_group_pairing(report)
    require(all(group["environment"] == report["environment"] for group in report["groups"].values()),
            "Group environment differs")
    require(all(group["fingerprints"][key] == report["fingerprints"][key]
                for group in report["groups"].values() for key in ("X_train", "y_train", "B_true", "independent_test")),
            "Group/dataset fingerprint differs")
    return report


def execute_task(output, plan, task):
    output = Path(output)
    job_id = f"task_{task['task_id']:03d}"
    command = [sys.executable, "-m", "numerical_simulation.candidate_search_study.worker", "--request",
               str((output / task["request_file"]).resolve()), "--output", str(result_path(output, task).resolve())]
    child_env = dict(os.environ)
    child_env.update(plan["environment"]["worker_thread_environment"])
    completed = subprocess.run(command, cwd=root(), env=child_env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for name, value in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        with (output / "logs" / f"{job_id}.{name}.txt").open("x", encoding="utf-8") as handle:
            handle.write(value)
    try:
        require(completed.returncode == 0, f"Worker returned {completed.returncode}: {completed.stderr}")
        validate_result(output, plan, task)
    except Exception as exc:
        write_json(output / "failures" / f"{job_id}.json", {"status": "failed", "task_id": task["task_id"],
            "plan_id": plan["plan_id"], "error": str(exc), "utc_time": datetime.now(timezone.utc).isoformat()})
        raise
    return task["task_id"]


def run_plan(output, workers=1, task_ids=None):
    require(isinstance(workers, int) and 1 <= workers <= 64, "workers must be in 1..64")
    output = Path(output)
    plan = load_plan(output)
    require(not any((output / "failures").iterdir()), "Recorded failure prevents further execution")
    selected = plan["tasks"] if task_ids is None else [task for task in plan["tasks"] if task["task_id"] in set(task_ids)]
    require(task_ids is None or len(selected) == len(set(task_ids)), "Unknown task-id")
    pending, complete = [], 0
    for task in selected:
        path = result_path(output, task)
        if path.exists() or path.with_suffix(".npz").exists():
            try:
                validate_result(output, plan, task)
            except Exception as exc:
                write_json(output / "failures" / f"task_{task['task_id']:03d}_resume.json", {"status": "resume-validation-failed",
                    "task_id": task["task_id"], "plan_id": plan["plan_id"], "error": str(exc)})
                raise
            complete += 1
        else:
            pending.append(task)
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        for _ in range(min(workers, len(pending))):
            task = next(iterator)
            active[executor.submit(execute_task, output, plan, task)] = task
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            errors = []
            for future in done:
                task = active.pop(future)
                try:
                    future.result()
                    complete += 1
                    print(f"Completed {complete}/{len(selected)}: task {task['task_id']}", flush=True)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                for future in active:
                    future.cancel()
                raise errors[0]
            for _ in done:
                task = next(iterator, None)
                if task is not None:
                    active[executor.submit(execute_task, output, plan, task)] = task
    return {"plan_id": plan["plan_id"], "status": "complete-selected-tasks", "completed_tasks": complete,
            "selected_tasks": len(selected), "expected_full_tasks": len(plan["tasks"])}


def summarize_reports(plan, records):
    ids = [record["task_id"] for record in records]
    require(len(ids) == len(set(ids)), "Duplicate datasets")
    require(set(ids) == {task["task_id"] for task in plan["tasks"]}, "incomplete frozen plan")
    by_id = {record["task_id"]: record["report"] for record in records}
    rows, pairs, coverage = [], [], []
    for task in plan["tasks"]:
        report = by_id[task["task_id"]]
        validate_group_pairing(report)
        tags = {"task_id": task["task_id"], "scenario_id": task["scenario_id"], "replicate": task["replicate"]}
        group_rows = {row["group"]: row for row in report["rows"]}
        rows.extend({**tags, **row} for row in report["rows"])
        for group, info in report["groups"].items():
            coverage.append({**tags, "group": group, "n_train": task["request"]["config"]["n_train"],
                "snr": task["request"]["config"]["snr"], "seed": task["request"]["seed"],
                **{key: info[key] for key in ("candidate_count", "cache_hits", "new_fixed_fits", "true_size_in_grid",
                    "true_size_in_pool", "true_support_in_pool", "true_size_exclusion_reasons")},
                **{key: info["search_metadata"][key] for key in ("caps", "grid_complete")}})
        if "A0" in group_rows:
            base = group_rows["A0"]
            for group, row in group_rows.items():
                if group != "A0":
                    pairs.append({**tags, "group": group, "n_train": row["n_train"], "snr": row["snr"], "seed": row["seed"],
                        **{f"{metric}_minus_A0": row[metric] - base[metric] for metric in METRICS},
                        "selected_B_changed": int(row["B_fingerprint"] != base["B_fingerprint"])})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["group"], row["n_train"], row["snr"]), []).append(row)
    summary = []
    for (group, n, snr), members in sorted(grouped.items()):
        item = {"group": group, "n_train": n, "snr": snr, "n_reps": len(members)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in members], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    integrity = {"status": "complete", "integrity_checks_passed": True, "plan_id": plan["plan_id"],
                 "completed_tasks": len(records), "expected_tasks": len(plan["tasks"]), "paired_group_rows": len(rows),
                 "cohort": plan["cohort"], "scope": "Phase D mechanisms only; default MSIC unchanged; no formal adoption"}
    return rows, summary, pairs, coverage, integrity


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (tuple, list, dict)) else value for key, value in row.items()})


def summarize(output):
    output = Path(output)
    plan = load_plan(output)
    require(not any((output / "failures").iterdir()), "Recorded failures prevent summary")
    names = ("raw_results.csv", "summary_results.csv", "paired_group_diff.csv", "candidate_coverage.csv", "integrity_report.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Immutable summary already exists")
    records = [{"task_id": task["task_id"], "report": validate_result(output, plan, task)} for task in plan["tasks"]]
    rows, summary, pairs, coverage, integrity = summarize_reports(plan, records)
    for name, values in zip(names[:4], (rows, summary, pairs, coverage)):
        write_csv(output / name, values)
    write_json(output / names[4], integrity)
    return integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "summarize"):
        sub = commands.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "plan":
            sub.add_argument("--cohort", choices=("historical", "development", "confirmation"), required=True)
            sub.add_argument("--n-reps", type=int, default=10)
            sub.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=["A0", "A1", "A2", "A3"])
        if command == "run":
            sub.add_argument("--workers", type=int, default=1)
            sub.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args(argv)
    operations = {"plan": lambda: create_plan(args.output_dir, args.cohort, args.n_reps, args.groups),
                  "run": lambda: run_plan(args.output_dir, args.workers, args.task_id),
                  "summarize": lambda: summarize(args.output_dir)}
    result = operations[args.command]()
    print(canonical({"command": args.command, "plan_id": result["plan_id"], "status": result.get("status", "planned"),
                     "tasks": len(result["tasks"]) if "tasks" in result else result.get("completed_tasks")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
