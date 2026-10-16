"""Release identity and gates around the existing frozen simulation runner."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import tarfile

RELEASE = "doc/releases/v5.2-search-final/"
OUTPUT = "numerical_simulation/outputs_v52_search_final_full"
TOOLS = ("release_control.py", "cluster_inventory.sh", "verify_cluster.sh", "preflight_gate.sh",
         "sync_to_cluster.ps1", "download_results.ps1")
DEPLOY = ["run_v52_search_final_full.sh", RELEASE + "RELEASE_MANIFEST.json", RELEASE + "SOURCE_SHA256.txt"]
DEPLOY += ["scripts/release_v52/" + name for name in TOOLS]
SIMULATION = ["numerical_simulation/" + name for name in
              ("__init__.py", "data_generation.py", "methods.py", "metrics.py", "known_s_cluster/run.py", "known_s_cluster/profiles.json")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def identity(path):
    return {"sha256": sha(path), "canonical_sha256": canonical(path), "size": path.stat().st_size}


def inventory(root):
    paths = sorted(p.relative_to(root).as_posix() for p in (root / "src/tensor_abess").rglob("*.py"))
    paths = sorted(set(paths + SIMULATION + DEPLOY + [p.name for p in root.glob("run_*.sh")]))
    return {"schema_version": 1, "root": str(root), "cluster_runtime": "NOT_RUN",
            "python_version": platform.python_version(),
            "files": {name: identity(root / name) for name in paths if (root / name).is_file()}}


def verify_files(root, manifest):
    expected = manifest["source_canonical_lf_sha256"]
    actual_paths = {p.relative_to(root).as_posix() for p in (root / "src/tensor_abess").rglob("*.py")}
    require(actual_paths == set(expected), "Core source path set differs from release")
    for name, digest in {**expected, **manifest.get("simulation_canonical_lf_sha256", {}),
                         **manifest.get("deployment_canonical_lf_sha256", {})}.items():
        path = root / name
        require(path.is_file() and canonical(path) == digest, "Release fingerprint mismatch: " + name)


def sync_plan(root, manifest, remote):
    verify_files(root, manifest)
    require(manifest["release_status"] == "released", "Release is not approved/released")
    files = remote["files"]
    core = manifest["source_canonical_lf_sha256"]
    require({name for name in files if name.startswith("src/tensor_abess/")} == set(core), "Remote source path conflict; inventory all 32 source files")
    updates = []
    for name, target in {**core, **manifest["simulation_canonical_lf_sha256"]}.items():
        current = files.get(name)
        old = current and current["canonical_sha256"]
        permitted = {target}
        if name == "src/tensor_abess/tensor/bcd.py":
            permitted.add(manifest["base_source_canonical_lf_sha256"][name])
        require(old in permitted or (current is None and name in SIMULATION), "Remote source/config conflict: " + name)
        if old != target:
            updates.append({"path": name, "before": current, "after": identity(root / name)})
    for name in DEPLOY:
        require((root / name).is_file(), "Missing deployment file: " + name)
        current, target = files.get(name), identity(root / name)
        # Existing release helpers must match, so an unrelated remote edit is never overwritten.
        permitted = {target["canonical_sha256"], manifest.get("base_deployment_canonical_lf_sha256", {}).get(name)}
        require(current is None or current["canonical_sha256"] in permitted, "Remote deployment conflict: " + name)
        if current is None or current["canonical_sha256"] != target["canonical_sha256"]:
            updates.append({"path": name, "before": current, "after": target})
    return {"version": "v5.2-search-final", "remote_root": remote["root"], "updates": updates,
            "verified": {name: files.get(name) for name in {**core, **manifest["simulation_canonical_lf_sha256"]}}}


def package(root, plan, destination):
    with tarfile.open(destination, "w") as archive:
        data = json.dumps(plan).encode("utf-8")
        info = tarfile.TarInfo("sync-plan.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for entry in plan["updates"]:
            require(identity(root / entry["path"]) == entry["after"], "Local file changed during packaging")
            archive.add(root / entry["path"], arcname=entry["path"], recursive=False)


def apply_package(root, archive_path):
    staging = root.resolve() / ".v52-release-staging"
    require(archive_path.resolve().is_relative_to(staging)
            and archive_path.parent.parent.resolve() == staging
            and archive_path.name == "upload.tar", "Staging path is outside remote root")
    with tarfile.open(archive_path, "r") as archive:
        plan = json.load(archive.extractfile("sync-plan.json"))
        require(Path(plan["remote_root"]).resolve() == root.resolve(), "Remote root differs from reviewed plan")
        allowed = set(DEPLOY + SIMULATION + ["src/tensor_abess/tensor/bcd.py"])
        updates = plan["updates"]
        names = [entry["path"] for entry in updates]
        require(len(names) == len(set(names)) and set(names) <= allowed, "Upload whitelist violation")
        require(set(archive.getnames()) == set(names + ["sync-plan.json"]), "Archive path whitelist violation")
        for name, before in {**plan["verified"], **{e["path"]: e["before"] for e in updates}}.items():
            path = root / name
            require(not path.is_symlink() and path.resolve() == path.absolute()
                    and path.resolve().is_relative_to(root.resolve()), "Unsafe remote path: " + name)
            require((identity(path) if path.is_file() else None) == before, "Remote pre-hash conflict: " + name)
        staged = archive_path.parent / "files"
        backup = archive_path.parent / "backup"
        for entry in updates:
            name = entry["path"]
            member = archive.getmember(name)
            require(member.isfile(), "Non-file archive member")
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
            require(identity(destination) == entry["after"], "Uploaded SHA256 mismatch: " + name)
        # No write to production until every pre-hash and every incoming file passes.
        for entry in updates:
            name = entry["path"]
            destination = root / name
            if destination.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, saved)
                require(identity(saved) == entry["before"], "Backup SHA256 mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged / name, destination)
            require(identity(destination) == entry["after"], "Post-upload SHA256 mismatch")
    return {"status": "APPLIED", "backup": str(backup), "files": names}


def runner(root):
    import sys
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from numerical_simulation.known_s_cluster import run
    return run


def verify_runtime(root, manifest):
    verify_files(root, manifest)
    require(manifest["release_status"] == "released" and manifest["approved_by_user"] is True, "Release not approved")
    import numpy
    import scipy
    import tensor_abess.tensor.bcd as bcd
    actual = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    expected = manifest["cluster_environment_expected"]
    require(actual == expected, "Cluster environment differs: " + str(actual))
    require(Path(bcd.__file__).resolve() == (root / "src/tensor_abess/tensor/bcd.py").resolve(), "Imported source is outside project src")
    require(len(bcd._candidate_size_triples(6, 6, 7, 280)) == 252, "A2 runtime contract mismatch")
    run = runner(root)
    tasks = run.profile_tasks("full", 50)
    require(len(tasks) == 700 and len({(t['scenario_id'], t['replicate']) for t in tasks}) == 700, "Frozen full task grid mismatch")
    return {"status": "VERIFIED", "version": manifest["version"], "core_files": 32, "tasks": 700, "environment": actual}


def checked_plan(root, output):
    require(output.resolve() == (root / OUTPUT).resolve(), "Only the independent v5.2 formal output is accepted")
    run = runner(root)
    plan = run.load_plan(output, check_environment=True)
    require(plan["profile"] == "full" and plan["n_reps"] == 50 and plan["methods"] == ["adaptive", "oracle", "lasso", "glasso"] and len(plan["tasks"]) == 700, "Formal plan differs from approved design")
    return run, plan


def write_json_exclusive(run, path, body):
    run.commit_exclusive(path, run.json_bytes({**body, "sha256": run.digest(body)}))


def read_signed(run, path):
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in record.items() if k != "sha256"}
    require(record["sha256"] == run.digest(body), "Gate/job metadata checksum mismatch")
    return body


def job_record(root, output, log, outcome):
    run, plan = checked_plan(root, output)
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    record = run.read_task(output, plan, task_id)
    job_id = os.environ["SLURM_JOB_ID"]
    require(job_id.isdigit() and os.environ["SLURM_ARRAY_JOB_ID"].isdigit(), "Missing SLURM identity")
    require(log.resolve().is_relative_to(output.resolve()) and log.is_file() and log.stat().st_size > 0, "Worker log missing")
    body = {"version": "v5.2-search-final", "plan_id": plan["plan_id"], "job_id": job_id,
            "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"], "array_task_count": int(os.environ["SLURM_ARRAY_TASK_COUNT"]),
            "task_id": task_id, "outcome": outcome, "record_sha256": record["record_sha256"],
            "log": log.resolve().relative_to(output.resolve()).as_posix(), "log_sha256": sha(log)}
    write_json_exclusive(run, log.with_suffix(".json"), body)
    return body


def preflight(root, output):
    run, plan = checked_plan(root, output)
    record = run.read_task(output, plan, 0)
    jobs = [read_signed(run, p) for p in (output / "slurm").glob("task_000000_*.json")]
    jobs = [j for j in jobs if j["task_id"] == 0 and j["array_task_count"] == 1 and j["outcome"] in {"completed", "skipped"} and j["plan_id"] == plan["plan_id"] and j["record_sha256"] == record["record_sha256"]]
    require(bool(jobs), "NOT_RUN: genuine one-task SLURM job metadata is required")
    job = jobs[-1]
    log = output / job["log"]
    require(log.resolve().is_relative_to(output.resolve()) and sha(log) == job["log_sha256"], "SLURM worker log checksum mismatch")
    paths = run.task_paths(output, 0)[:2]
    before = {p.name: sha(p) for p in paths}
    require(run.run_task(output, 0, resume=True) == "skipped", "Resume did not validate and skip")
    require(before == {p.name: sha(p) for p in paths}, "Resume mutated completed task")
    body = {"version": "v5.2-search-final", "status": "PASSED", "plan_id": plan["plan_id"],
            "source_fingerprints": plan["source_fingerprints"], "environment": plan["environment"],
            "task_zero_record_sha256": record["record_sha256"], "slurm_job": job, "artifacts": before}
    write_json_exclusive(run, output / "cluster_preflight_gate.json", body)
    return body


def gate(root, output, complete=False):
    run, plan = checked_plan(root, output)
    body = read_signed(run, output / "cluster_preflight_gate.json")
    require(body["status"] == "PASSED" and body["plan_id"] == plan["plan_id"] and body["source_fingerprints"] == plan["source_fingerprints"] and body["environment"] == plan["environment"], "Cluster preflight gate differs from current plan")
    record = run.read_task(output, plan, 0)
    require(body["task_zero_record_sha256"] == record["record_sha256"], "Preflight task changed")
    log = output / body["slurm_job"]["log"]
    require(log.resolve().is_relative_to(output.resolve()) and sha(log) == body["slurm_job"]["log_sha256"], "Preflight log changed")
    if complete:
        state = run.status(output)
        require(state["completed"] == 700 and not state["missing"] and not state["running_or_locked"], "Merge blocked: tasks missing, invalid or still locked")
        require(sum(len(run.read_task(output, plan, i)["results"]) for i in range(700)) == 2800, "Method count mismatch")
        return state
    return {"status": "PASSED", "plan_id": plan["plan_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "sync-plan", "package", "apply-package", "verify", "plan", "job-record", "preflight", "gate", "complete", "remote-code"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--outcome", choices=["completed", "skipped"])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "remote-code":
        print(base64.b64encode(Path(__file__).read_bytes()).decode("ascii"))
        return
    if args.command == "inventory":
        print(json.dumps(inventory(root), ensure_ascii=True))
        return
    if args.command == "apply-package":
        print(json.dumps(apply_package(root, args.archive)))
        return
    manifest = json.loads((root / RELEASE / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if args.command in {"sync-plan", "package"}:
        plan = sync_plan(root, manifest, json.loads(args.inventory.read_text(encoding="utf-8-sig")))
        if args.command == "package":
            package(root, plan, args.archive)
        print(json.dumps(plan))
        return
    result = verify_runtime(root, manifest)
    output = root / OUTPUT
    actions = {"verify": lambda: result, "plan": lambda: {"plan_id": checked_plan(root, output)[1]["plan_id"]},
               "job-record": lambda: job_record(root, output, args.log, args.outcome),
               "preflight": lambda: preflight(root, output), "gate": lambda: gate(root, output),
               "complete": lambda: gate(root, output, complete=True)}
    print(json.dumps(actions[args.command]()))


if __name__ == "__main__":
    main()
