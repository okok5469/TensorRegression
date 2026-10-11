"""Frozen, paired original/fixed algorithm studies in separate worker processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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


VERSIONS = ("v5-original", "v5.1-bcd-fix")
DEFAULT_OUTPUT = "numerical_simulation/outputs_msic_ablation"
METRICS = ("q_hat", "exact_recovery", "mode_f1_macro", "element_f1", "pmse", "r2_test",
           "rss_train", "runtime_sec", "shared_training_sec", "scoring_sec",
           "q_over", "q_under", "q_small", "q_near_cap")


def root():
    return Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment():
    return {"libraries": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "platform": platform.system(), "machine": platform.machine(),
            "worker_thread_environment": {k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def source_manifests():
    sources = {"v5-original": root() / "doc/algorithm_versions/v5-original/src",
               "v5.1-bcd-fix": root() / "src"}
    manifests = {}
    for version, source in sources.items():
        paths = sorted((source / "tensor_abess").rglob("*.py"))
        require(bool(paths), f"Missing core sources for {version}")
        manifests[version] = {p.relative_to(source).as_posix(): file_hash(p) for p in paths}
    experiment_paths = [root() / f"numerical_simulation/{name}.py" for name in ("data_generation", "metrics")]
    experiment_paths += [Path(__file__), Path(__file__).with_name("worker.py"),
                         root() / "numerical_simulation/known_s_cluster/profiles.json"]
    return manifests, {p.relative_to(root()).as_posix(): file_hash(p) for p in experiment_paths}


def build_plan(n_reps):
    require(isinstance(n_reps, int) and n_reps > 0, "n_reps must be a positive integer")
    profiles = json.loads((root() / "numerical_simulation/known_s_cluster/profiles.json").read_text(encoding="utf-8"))
    baseline = profiles["profiles"]["full"]["base_config"]
    tasks, jobs = [], []
    for scenario_id, (n, snr) in enumerate((n, snr) for n in (100, 200, 400) for snr in (0.5, 2.0, 4.0)):
        for replicate in range(n_reps):
            seed = 20261010 + 1000 * scenario_id + replicate
            config = {**baseline, "n_train": n, "snr": snr}
            request = {"config": config, "seed": seed, "test_seed": seed + 1_000_000, "include_cv": n == 200}
            task_id = len(tasks)
            request_file = f"requests/task_{task_id:03d}.json"
            tasks.append({"task_id": task_id, "scenario_id": scenario_id, "replicate": replicate,
                          "request_file": request_file, "request": request, "request_id": digest(request)})
            for version in VERSIONS:
                job_id = f"task_{task_id:03d}__{version}"
                jobs.append({"job_id": job_id, "task_id": task_id, "version": version,
                             "result_file": f"jobs/{job_id}.json"})
    sources, experiment_sources = source_manifests()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root(), capture_output=True, text=True)
    body = {"schema_version": 1, "n_reps": n_reps, "tasks": tasks, "jobs": jobs,
            "source_manifests": sources, "experiment_sources": experiment_sources,
            "environment": environment(), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "seed_definition": "20261010 + 1000*scenario_id + replicate; test_seed adds 1000000; inner-CV seed adds 2000000",
            "scenario_order": "n outer=(100,200,400), snr inner=(0.5,2,4)",
            "purpose": "Exploratory simulation; original and fixed default compared on identical generated data; MSIC candidates do not change formal defaults.",
            "capacity_rates": "q_over:q>60; q_under:q<60; q_small:q=1; q_near_cap:q>=.9 floor(.7n)",
            "completion_requirement": "All jobs complete and pairwise input/test fingerprints equal before summarizing"}
    return {**body, "plan_id": digest(body)}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def create_plan(output, n_reps):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Plan requires a new empty output directory: {output}")
    plan = build_plan(n_reps)
    for directory in ("requests", "jobs", "logs", "failures"):
        (output / directory).mkdir()
    for task in plan["tasks"]:
        write_json(output / task["request_file"], task["request"])
    write_json(output / "plan.json", plan)
    return plan


def load_plan(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    require(plan["plan_id"] == digest(body), "Plan checksum mismatch")
    expected = build_plan(plan["n_reps"])
    for key in ("tasks", "jobs", "source_manifests", "experiment_sources", "environment"):
        require(plan[key] == expected[key], f"Frozen {key} differs; choose a fresh run directory")
    for task in plan["tasks"]:
        request = json.loads((output / task["request_file"]).read_text(encoding="utf-8"))
        require(digest(request) == task["request_id"] and request == task["request"], "Frozen request changed")
    return plan


def validate_result(output, plan, job):
    path = Path(output) / job["result_file"]
    npz_path = path.with_suffix(".npz")
    require(path.is_file() and npz_path.is_file(), f"incomplete result/cache: {job['job_id']}")
    report = json.loads(path.read_text(encoding="utf-8"))
    task = plan["tasks"][job["task_id"]]
    request = task["request"]
    require(report["status"] == "complete", f"Job not complete: {job['job_id']}")
    for key in ("seed", "test_seed", "config"):
        require(canonical(report[key]) == canonical(request[key]), f"Result request mismatch {key}: {job['job_id']}")
    require(report["version"] == job["version"], f"Result version mismatch: {job['job_id']}")
    require(report["environment"] == plan["environment"]["libraries"], f"Result environment differs: {job['job_id']}")
    require(report["fingerprints"]["core_sources"] == plan["source_manifests"][job["version"]],
            f"Result source fingerprint differs: {job['job_id']}")
    require(report["array_cache"] == npz_path.name and report["array_cache_sha256"] == file_hash(npz_path),
            f"Array cache checksum differs: {job['job_id']}")
    require(report["fingerprints"]["independent_test"]["n_observations"] == request["config"]["n_test"],
            f"Incomplete test data: {job['job_id']}")
    expected_rows = 1 if job["version"] == "v5-original" else 11 + int(request["include_cv"])
    require(len(report["rows"]) == expected_rows, f"Incomplete method rows: {job['job_id']}")
    rules = [row["rule"] for row in report["rows"]]
    require(len(rules) == len(set(rules)), f"Duplicate method rows: {job['job_id']}")
    return report


def execute_job(output, plan, job):
    output = Path(output)
    request_path = output / plan["tasks"][job["task_id"]]["request_file"]
    result_path = output / job["result_file"]
    command = [sys.executable, "-m", "numerical_simulation.version_validation.worker", "--request", str(request_path.resolve()),
               "--output", str(result_path.resolve()), "--version", job["version"]]
    child_env = dict(os.environ)
    child_env.update(plan["environment"]["worker_thread_environment"])
    completed = subprocess.run(command, cwd=root(), env=child_env, capture_output=True, text=True)
    for suffix, content in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        with (output / "logs" / f"{job['job_id']}.{suffix}.txt").open("x", encoding="utf-8") as handle:
            handle.write(content)
    if completed.returncode:
        failure = {"status": "failed", "job_id": job["job_id"], "returncode": completed.returncode,
                   "plan_id": plan["plan_id"], "utc_time": datetime.now(timezone.utc).isoformat(),
                   "stdout": completed.stdout, "stderr": completed.stderr}
        write_json(output / "failures" / f"{job['job_id']}.json", failure)
        raise RuntimeError(f"Job failed: {job['job_id']}; see failures/{job['job_id']}.json")
    try:
        validate_result(output, plan, job)
    except Exception as exc:
        write_json(output / "failures" / f"{job['job_id']}.json", {"status": "validation-failed",
            "job_id": job["job_id"], "plan_id": plan["plan_id"], "error": str(exc),
            "utc_time": datetime.now(timezone.utc).isoformat()})
        raise
    return job["job_id"]


def run_plan(output, workers=2):
    require(1 <= workers <= 2, "workers must be 1 or 2")
    output = Path(output)
    plan = load_plan(output)
    if any((output / "failures").iterdir()):
        raise RuntimeError("Recorded failure present; affected study must be reviewed before any further steps")
    pending = []
    complete = 0
    for job in plan["jobs"]:
        path = output / job["result_file"]
        if path.exists() or path.with_suffix(".npz").exists():
            validate_result(output, plan, job)
            complete += 1
            print(f"Verified existing {complete}/{len(plan['jobs'])}: {job['job_id']}", flush=True)
        else:
            pending.append(job)
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        for _ in range(min(workers, len(pending))):
            job = next(iterator)
            active[executor.submit(execute_job, output, plan, job)] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            failures = []
            for future in done:
                job = active.pop(future)
                try:
                    future.result()
                    complete += 1
                    print(f"Completed {complete}/{len(plan['jobs'])}: {job['job_id']}", flush=True)
                except Exception as exc:
                    failures.append(exc)
            if failures:
                for future in active:
                    future.cancel()
                raise failures[0]
            for _ in range(len(done)):
                job = next(iterator, None)
                if job is None:
                    break
                active[executor.submit(execute_job, output, plan, job)] = job
    return {"status": "complete", "completed_jobs": complete, "expected_jobs": len(plan["jobs"]), "plan_id": plan["plan_id"]}


def nonmonotonic(history):
    return int(any(b > a + 1e-10 * max(1.0, abs(a), abs(b)) for a, b in zip(history, history[1:])))


def summarize_reports(plan, records):
    ids = [record["job_id"] for record in records]
    require(len(ids) == len(set(ids)), "Duplicate job results")
    expected = {job["job_id"] for job in plan["jobs"]}
    require(set(ids) == expected, "incomplete study: every frozen job is required")
    reports = {record["job_id"]: record["report"] for record in records}
    rows, pairs = [], []
    for task in plan["tasks"]:
        version_reports = [reports[f"task_{task['task_id']:03d}__{version}"] for version in VERSIONS]
        for key in ("X_train", "y_train", "B_true", "independent_test"):
            require(version_reports[0]["fingerprints"][key] == version_reports[1]["fingerprints"][key],
                    f"Pair input/test fingerprint differs: task {task['task_id']} {key}")
        for report in version_reports:
            for row in report["rows"]:
                rows.append({"task_id": task["task_id"], "scenario_id": task["scenario_id"],
                             "replicate": task["replicate"], **row,
                             "nonmonotonic_returned_history": nonmonotonic(row["loss_history"])})
        original = next(row for row in version_reports[0]["rows"] if row["rule"] == "adaptive-default")
        fixed = next(row for row in version_reports[1]["rows"] if row["rule"] == "msic-alpha1.5-scale1")
        original_B = version_reports[0]["candidates"][0]["B_fingerprint"]
        fixed_B = version_reports[1]["candidates"][fixed["selection"]["selected_index"]]["B_fingerprint"]
        pairs.append({"task_id": task["task_id"], "scenario_id": task["scenario_id"], "replicate": task["replicate"],
            "n_train": task["request"]["config"]["n_train"], "snr": task["request"]["config"]["snr"],
            "seed": task["request"]["seed"],
            **{f"{metric}_fixed_minus_original": fixed[metric] - original[metric] for metric in METRICS},
            "original_nonmonotonic_returned_history": nonmonotonic(original["loss_history"]),
            "fixed_nonmonotonic_returned_history": nonmonotonic(fixed["loss_history"]),
            "selected_B_changed": int(original_B != fixed_B)})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["n_train"], row["snr"]), []).append(row)
    summary = []
    for (method, n, snr), group in sorted(grouped.items()):
        result = {"method": method, "n_train": n, "snr": snr, "n_reps": len(group)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group], dtype=float)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    integrity = {"status": "complete", "integrity_checks_passed": True, "plan_id": plan["plan_id"],
                 "expected_jobs": len(plan["jobs"]), "completed_jobs": len(records),
                 "paired_datasets": len(plan["tasks"]), "method_rows": len(rows),
                 "scope": "Exploratory 9-scenario study; no v6 formal selection; original results retained"}
    return rows, summary, pairs, integrity


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def summarize(output):
    output = Path(output)
    plan = load_plan(output)
    require(not any((output / "failures").iterdir()), "Recorded failures prevent complete summary")
    names = ("raw_results.csv", "summary_results.csv", "paired_default_diff.csv", "integrity_report.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Summary artifacts already exist; keep immutable results")
    records = [{"job_id": job["job_id"], "report": validate_result(output, plan, job)} for job in plan["jobs"]]
    rows, summary, pairs, integrity = summarize_reports(plan, records)
    write_csv(output / names[0], rows)
    write_csv(output / names[1], summary)
    write_csv(output / names[2], pairs)
    write_json(output / names[3], integrity)
    return integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "summarize"):
        sub = commands.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
        if command == "plan":
            sub.add_argument("--n-reps", type=int, default=3)
        if command == "run":
            sub.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    operations = {"plan": lambda: create_plan(args.output_dir, args.n_reps),
                  "run": lambda: run_plan(args.output_dir, args.workers),
                  "summarize": lambda: summarize(args.output_dir)}
    result = operations[args.command]()
    print(canonical({"command": args.command, "plan_id": result["plan_id"],
        "status": result.get("status", "planned"),
        "jobs": len(result["jobs"]) if "jobs" in result else result.get("completed_jobs")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
