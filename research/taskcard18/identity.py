"""Read-only verification of the frozen simulation and Task18 input aliases.

The default command only reads historical files and prints a compact result.
``--write`` writes this task's derived audit/map to ``doc/taskcard18`` only.
No model fitting, data reconstruction, merge, manifest update or submission occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def verify_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    """Reject a changed/missing frozen file; never normalize or rewrite it."""
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in expected}
    conflicts = [name for name, digest in actual.items() if digest != expected[name]]
    if conflicts:
        raise ValueError(f"Frozen source fingerprint conflict: {conflicts}")
    return actual


def build_input_map(records: list[dict]) -> list[dict]:
    """Use complete recorded training fingerprints and test seed, not labels/X alone."""
    groups = defaultdict(list)
    fingerprint_keys = ("X_train", "y_train", "B_true", "train_noise")
    for record in records:
        task = record["task"]
        fingerprints = {key: record["input_fingerprints"][key] for key in fingerprint_keys}
        identity = {"input_fingerprints": fingerprints, "test_seed": task["test_seed"]}
        groups[canonical_digest(identity)].append((record, fingerprints))

    rows = []
    for digest, members in groups.items():
        configurations = {
            canonical_digest({key: value for key, value in record["task"]["config"].items()
                              if key != "random_state"})
            for record, _ in members
        }
        if len(configurations) != 1:
            raise ValueError("Identical input fingerprints have conflicting configuration")
        canonical_task_id = min(record["task"]["task_id"] for record, _ in members)
        scenarios = "|".join(map(str, sorted({record["task"]["scenario_id"]
                                              for record, _ in members})))
        for record, fingerprints in members:
            task = record["task"]
            rows.append({
                "scenario_id": task["scenario_id"], "replicate": task["replicate"],
                "task_id": task["task_id"], "config_id": "cfg_" + next(iter(configurations)),
                "experiment": task["experiment"], "scenario_value": task["scenario_value"],
                "train_seed": task["seed"], "test_seed": task["test_seed"],
                **{key + "_sha256": value for key, value in fingerprints.items()},
                "unique_input_id": "input_" + digest,
                "canonical_task_id": canonical_task_id,
                "alias_of_task_id": "" if task["task_id"] == canonical_task_id else canonical_task_id,
                "alias_scenario_ids": scenarios, "alias_multiplicity": len(members),
            })
    return sorted(rows, key=lambda row: row["task_id"])


def compare_csv(path: Path, expected: list[dict], key_names: tuple[str, ...]) -> dict:
    """Verify rows and all fields against frozen pure aggregation functions."""
    import numpy as np

    with path.open(encoding="utf-8-sig", newline="") as handle:
        actual = list(csv.DictReader(handle))
    key = lambda row: tuple(str(row[name]) for name in key_names)
    lookup = {key(row): row for row in actual}
    if len(actual) != len(expected) or len(lookup) != len(actual):
        raise ValueError(f"CSV duplicate/missing rows: {path}")
    maximum_difference = 0.0
    numeric_fields_checked = 0
    for row in expected:
        saved = lookup[key(row)]
        if set(saved) != set(row):
            raise ValueError(f"CSV field schema conflict: {path}")
        for name, value in row.items():
            if isinstance(value, (int, float)):
                difference = abs(float(saved[name]) - value)
                maximum_difference = max(maximum_difference, difference)
                numeric_fields_checked += 1
                if not np.isclose(float(saved[name]), value, rtol=1e-11, atol=1e-10):
                    raise ValueError(f"CSV differs from saved records: {path} {key(row)} {name}")
            elif saved[name] != str(value):
                raise ValueError(f"CSV metadata conflict: {path} {key(row)} {name}")
    return {"rows": len(actual), "numeric_fields_checked": numeric_fields_checked,
            "maximum_absolute_difference": maximum_difference, "passed": True}


def git_snapshot(root: Path) -> dict:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--porcelain", "-z"],
        cwd=root, capture_output=True, check=True,
    )
    entries = [entry for entry in result.stdout.decode("utf-8").split("\0") if entry]
    status_counts = Counter(entry[:2] for entry in entries)
    task18_entries = [entry for entry in entries if any(
        part in entry[3:] for part in ("taskcard18", "paper/"))]
    return {
        "command": "git -c core.quotepath=false status --porcelain -z",
        "current_status_counts": dict(status_counts),
        "current_task18_entries": task18_entries,
        "current_tracked_modifications": [entry for entry in entries
                                          if entry[:2].strip() not in {"D", "??"}],
        "stderr": result.stderr.decode("utf-8", errors="replace"),
        "initial_read_only_observation": {
            "tracked_deleted": 312, "untracked_status_entries": 215,
            "tracked_modified": 3,
            "modified_files": ["doc/findings.md", "doc/progress.md", "doc/task_plan.md"],
            "scope": "Initial Task18 read-only observation before Task18 implementation; "
                     "reported by the identity auditor, not a reconstructed Git baseline. "
                     "Untracked count is Git status entries, not recursive file count.",
        },
        "preservation_policy": "Keep all pre-existing changes; no reset/restore/branch/worktree. "
                               "Current Task18/paper additions are listed separately above.",
    }


def historical_evidence(root: Path) -> dict:
    names = {
        "task10": "doc/taskcard10/continuation_01/FINAL_VALIDATION.json",
        "task12": "doc/taskcard12/FINAL_VALIDATION.json",
    }
    loaded = {key: json.loads((root / name).read_text(encoding="utf-8"))
              for key, name in names.items()}
    return {
        "files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in names.values()},
        "task10": {key: loaded["task10"][key] for key in
                   ("status", "counts", "reused_development", "new_development",
                    "scientific_confirmation_status")},
        "task12": {key: loaded["task12"]["simulation"][key] for key in
                   ("formal_status", "formal_planned", "formal_completed", "balanced_cases",
                    "pressure_extra_cases", "pressure_scene_total", "remote_smoke_completed")},
        "scope": "Existing final validation records read and fingerprinted; "
                 "historical model runs or archive-wide checks were not repeated here.",
    }


def audit_repository(root: Path) -> tuple[dict, list[dict]]:
    import os
    import numpy as np
    import scipy
    import tensor_abess
    from numerical_simulation.known_s_cluster.run import (
        flat_rows, load_plan, read_task, summary_rows,
    )

    import_path = Path(tensor_abess.__file__).resolve()
    if not import_path.is_relative_to((root / "src/tensor_abess").resolve()):
        raise ValueError(f"Production import path conflict: {import_path}")
    workflow_files = ("task_plan.md", "progress.md", "findings.md", "task_issue.md")
    misplaced = [name for name in workflow_files if (root / name).exists()]
    if misplaced:
        raise ValueError(f"Workflow files outside doc: {misplaced}")
    manifest_path = root / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    production = verify_hashes(root, manifest["source_sha256"])
    production_files = {path.relative_to(root).as_posix()
                        for path in (root / "src/tensor_abess").rglob("*.py")}
    if production_files != set(production) or len(production) != 32:
        raise ValueError("Production file inventory differs from the 32-source release")
    output = root / "numerical_simulation/outputs_v52_search_final_full"
    plan = load_plan(output)
    simulation_sources = verify_hashes(root, plan["source_fingerprints"])
    if len(simulation_sources) != 38:
        raise ValueError("Formal simulation source/config inventory is not 38")
    records = [read_task(output, plan, task["task_id"]) for task in plan["tasks"]]
    mapping = build_input_map(records)
    raw = flat_rows(records)
    metric_names = list(records[0]["results"][0]["metrics"])
    expected_summary = summary_rows(raw, plan, metric_names)
    csv_checks = {
        "raw": compare_csv(output / "raw_results.csv", raw, ("task_id", "method")),
        "summary": compare_csv(output / "summary_results.csv", expected_summary,
                               ("scenario_id", "method")),
    }
    input_groups = defaultdict(list)
    configuration_inputs = defaultdict(set)
    for row in mapping:
        input_groups[row["unique_input_id"]].append(row)
        configuration_inputs[row["config_id"]].add(row["unique_input_id"])
    counts = {"tasks": len(records), "method_rows": len(raw),
              "summary_cells": len(expected_summary), "unique_inputs": len(input_groups),
              "configurations": len(configuration_inputs)}
    if tuple(counts.values()) != (700, 2800, 56, 600, 10):
        raise ValueError(f"Formal batch counts conflict: {counts}")
    json_files, npz_files = list((output / "tasks").glob("*.json")), list((output / "tasks").glob("*.npz"))
    if (len(json_files), len(npz_files)) != (700, 700):
        raise ValueError("Unexpected JSON/NPZ inventory in the formal task directory")
    basenames = ("plan.json", "raw_results.csv", "summary_results.csv", "integrity_report.json")
    runtime_packages = {}
    for package, module in (("scikit-learn", "sklearn"), ("cvxpy", "cvxpy"),
                            ("clarabel", "clarabel"), ("osqp", "osqp"), ("scs", "scs")):
        available = importlib.util.find_spec(module) is not None
        runtime_packages[package] = {"available_on_current_import_path": available}
        if available:
            runtime_packages[package]["version"] = importlib.metadata.version(package)
    optional_path = root / "doc/taskcard13/runtime"
    optional_packages = {d.metadata["Name"]: d.version
                         for d in importlib.metadata.distributions(path=[str(optional_path)])
                         if d.metadata["Name"].lower() in
                         {"scikit-learn", "cvxpy", "clarabel", "osqp", "scs"}}
    report = {
        "schema_version": 1, "status": "IDENTITY_PASS",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only frozen-source/record/array and aggregation verification; "
                 "zero fitting, input regeneration, historical merge or cluster submission.",
        "production": {"version": manifest["version"], "release_status": manifest["release_status"],
                       "approved_by_user": manifest["approved_by_user"],
                       "release_git_commit": manifest["release_git_commit"],
                       "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "import_path": str(import_path), "source_count": len(production),
                       "source_sha256": production, "all_raw_sha256_match": True},
        "formal_simulation": {"plan_id": plan["plan_id"], "source_count": len(simulation_sources),
                              "source_sha256": simulation_sources, "all_raw_sha256_match": True,
                              "historical_environment": plan["environment"],
                              "file_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                              for name in basenames},
                              "json_npz_verified": 700, "csv_checks": csv_checks},
        "counts": counts,
        "input_mapping": {
            "key": "Four stored training fingerprints (X_train,y_train,B_true,train_noise) + test_seed",
            "unique_X_train_hashes": len({record["input_fingerprints"]["X_train"] for record in records}),
            "multiplicities": dict(Counter(len(rows) for rows in input_groups.values())),
            "duplicate_scenario_groups": dict(Counter(
                rows[0]["alias_scenario_ids"] for rows in input_groups.values() if len(rows) > 1)),
            "unique_inputs_per_configuration": dict(sorted(
                (key, len(values)) for key, values in configuration_inputs.items())),
            "canonical_pilot_scenarios": [0, 1, 2, 4, 5, 7, 8, 10, 11, 13],
            "pilot_task_ids": [50 * scenario + replicate
                               for scenario in [0, 1, 2, 4, 5, 7, 8, 10, 11, 13]
                               for replicate in [0, 1]],
        },
        "saved_input_evidence": {
            "X_train_y_train": "Stored SHA256 available for all700; arrays not archived in NPZ; not regenerated here.",
            "B_true_train_noise": "Arrays and their stored fingerprints checked in all700 NPZ.",
            "test_stream": "Stored test_seed and frozen generator/batching contract; no original test-array hashes or arrays.",
            "new_pairing_requirement": "Reconstruct with frozen generator and explicit seeds, verify available "
                                       "training fingerprints, and state test-stream verification scope.",
        },
        "current_environment": {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "numpy_import_path": np.__file__,
            "scipy": scipy.__version__, "scipy_import_path": scipy.__file__,
            "blas_build": np.show_config(mode="dicts").get("Build Dependencies", {}),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "solver_packages": runtime_packages,
            "existing_optional_runtime": {"path": str(optional_path), "packages": optional_packages,
                                          "activated_by_this_audit": False},
        },
        "git": git_snapshot(root), "root_workflow_files": [],
        "historical_final_evidence": historical_evidence(root),
        "limitations": ["Identity verification does not certify old baseline KKT/duality gaps.",
                        "Stored PMSE/signal MSE/R2 values are verified against task JSON and reaggregated; test arrays were not regenerated.",
                        "Input mapping reflects available frozen evidence; no claim of rebuilt byte-identical test arrays.",
                        "Current environment is recorded separately from historical Linux environment."],
    }
    return report, mapping


def write_report(root: Path, report: dict, mapping: list[dict]) -> None:
    """Write only Task18 derived deliverables after all verification succeeds."""
    destination = root / "doc/taskcard18"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "IDENTITY_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "input_identity_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    environment = report["current_environment"]
    text = f"""# 任务卡18：身份与输入审计

状态：**{report['status']}**；检查时间（UTC）：{report['checked_utc']}。

正式 `v5.2-search-final + 原MSIC` 的32生产源原字节SHA256全部匹配release，实际导入路径为 `{report['production']['import_path']}`；正式计划38项源码/配置指纹及plan seal全部匹配。未改manifest、旧源码或旧结果，未拟合、重建输入、merge或提交集群。

## 封存结果与重复映射

700份JSON及对应NPZ通过冻结读取器的记录seal、计划/配置/保存环境、NPZ哈希、数组可读性、支撑及REE/RSS关系检查；四方法2800行和56汇总单元均完整，各标签每方法50次。全部CSV字段与JSON或纯聚合结果比较通过，数值误差上限分别为 raw={report['formal_simulation']['csv_checks']['raw']['maximum_absolute_difference']:.3g}、summary={report['formal_simulation']['csv_checks']['summary']['maximum_absolute_difference']:.3g}。

四项训练指纹联合test_seed得600套唯一完整输入：550个单例，50套在scenario6/9/12重复三次。仅X_train指纹为550，不能替代完整输入计数；10种参数配置中基准合并150套，其余9种各50套。`input_identity_map.csv`保留700标签、种子、配置ID、完整输入ID、canonical任务和别名；不把复用基准重复计为独立证据。固定pilot为scenario `[0,1,2,4,5,7,8,10,11,13]` 各replicate0、1。

## 输入证据的准确范围

NPZ含B_true、train_noise、真支撑及四方法系数/支撑，对应真系数与噪声指纹已逐个核对；X_train、y_train只有原保存指纹，没有归档数组。测试流有明确test_seed和冻结生成器/分批约定，没有原测试数组或其指纹。本入口不重建输入，因此不声称测试流逐字节恢复；新补算须由独立输入恢复步骤核对可用指纹并声明测试配对核验范围。

正式计划ID为 `{report['formal_simulation']['plan_id']}`；完整32/38指纹、原文件SHA256、保存Linux环境与本机环境分列于JSON。身份通过不替代Lasso/Group Lasso各必要子问题的KKT与原始—对偶间隙证书。

## 实际环境与既有变更

本次Python为 `{environment['python']}`，NumPy `{environment['numpy']}`，SciPy `{environment['scipy']}`；实际BLAS、模块路径、可导入求解器和四项线程环境详见JSON。历史Linux Python3.9.25/NumPy2.0.2/SciPy1.13.1单独保留，未冒充相同环境；本入口不设置线程或激活可选runtime。已有 `doc/taskcard13/runtime` 的可用依赖单独列出，不等同当前已导入求解器。

初始只读观察为312个tracked删除、215个未跟踪Git状态项、3个修改（doc三份日志）；这是本轮实现前的原始语境，未跟踪项不等同文件数。当前Git快照另存JSON，本轮taskcard18/paper新增另列，不覆盖既有改动。Git遍历旧test_tools可能产生的目录警告原样保存在JSON；任务科学输入读取未受影响。根目录无四类流程文件。

## 历史最终验收

读取并指纹绑定卡10续跑最终验收：开发48/48，复用30、新18，确认0/48；读取卡12最终验收：正式380（360主组+20额外压力），压力场景总30含主组10，另有3个smoke。本入口不重复历史拟合或归档全量审计，不把早期待运行记录当新任务。

## 复验

在仓库根目录运行 `python -B -m research.taskcard18.identity`，默认仅读取并输出计数；`--write`仅更新本卡上述三份派生审计文件。限定边界测试为 `python -B -m unittest research.taskcard18.test_identity`。基线求解、输入重建及章节验收由各自入口处理。
"""
    (destination / "IDENTITY_AUDIT.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write Task18 derived audit/map only")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    report, mapping = audit_repository(root)
    if args.write:
        write_report(root, report, mapping)
    print("IDENTITY_PASS: production32/32, plan38/38; "
          "tasks700, method_rows2800, summary56, unique_inputs600, configurations10; "
          + ("Task18 audit/map written." if args.write else "read-only, no outputs changed."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
