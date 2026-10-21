"""Read-only audit of the delivered Phase B records; never fits or merges."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

def audit(root):
    sys.path.insert(0, str(root))
    from numerical_simulation.version_attribution import run as native
    output = root / "numerical_simulation/outputs_v51_attribution/cluster_run"
    plan = native.load_plan(output)
    native.require_gates(output, plan)
    expected = {f"task_{task['task_id']:06d}" for task in plan["tasks"]}
    assert {p.stem for p in (output / "tasks").glob("*.json")} == expected
    assert {p.stem for p in (output / "tasks").glob("*.npz")} == expected
    assert {p.stem for p in (output / "gates").glob("*.json")} == {
        f"task_{task_id:06d}" for task_id in plan["gate_task_ids"]}
    integrity = native.read(output / "integrity.json")
    native.check_seal(integrity)
    assert integrity["completed"] == 250 and integrity["plan_id"] == plan["sha256"]
    assert integrity["status"] == "PASSED"
    preflight = native.read(output / "preflight_task_000000.json")
    native.check_seal(preflight)
    assert preflight["passed"] and not preflight["fit_executed"]
    assert preflight["actual"] == preflight["expected"] == plan["tasks"][0]["expected_input_fingerprints"]
    assert preflight["actual_environment"] == preflight["expected_environment"] == plan["expected_environment"]
    assert preflight["environment_matches"] and all(preflight["matches"].values())
    with (output / "paired_three_versions.csv").open(encoding="utf-8-sig", newline="") as handle:
        existing_rows = list(csv.DictReader(handle))
    assert len(existing_rows) == 250
    stats = defaultdict(lambda: defaultdict(list))
    npz_files = 0
    for task, existing in zip(plan["tasks"], existing_rows):
        name = f"task_{task['task_id']:06d}.json"
        record = native.read(output / "tasks" / name)
        native.check_seal(record)
        assert record["task_id"] == task["task_id"] and record["fit_version"] == plan["fit_version"]
        assert record["plan_id"] == plan["sha256"]
        assert record["core"] == plan["source_manifests"]["v5.1-bcd-fix"]
        assert record["environment"] == plan["expected_environment"]
        assert record["input_fingerprints"] == task["expected_input_fingerprints"]
        assert record["npz_sha256"] == native.sha((output / "tasks" / name).with_suffix(".npz"))
        assert record["grids"] == native.compare_grids(record["trace"]["s_max"], task["config"]["n_train"])
        row = {key: task[key] for key in ("task_id", "scenario_id", "replicate")}
        for label, folder in (("v5", "outputs_known_s_full_50"),
                              ("v52", "outputs_v52_search_final_full"), ("v51", None)):
            if folder:
                path = root / "numerical_simulation" / folder / "tasks" / name
                previous = native.read(path)
                assert previous["npz_sha256"] == native.sha(path.with_suffix(".npz"))
                selected = next(item for item in previous["results"] if item["method"] == "adaptive")
                metrics, sic = selected["metrics"], selected["sic"]
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                    supports = [saved[f"S{i}_adaptive"].tolist() for i in (1, 2, 3)]
                    coefficients = saved["B_adaptive"]
                grids = native.compare_grids(selected["trace"]["s_max"], task["config"]["n_train"])
                count = grids["v51_count" if label == "v5" else "v52_count"] + 1
            else:
                metrics, sic, supports = record["metrics"], record["sic"], record["supports"]
                count = record["grids"]["v51_count"] + 1
                with np.load((output / "tasks" / name).with_suffix(".npz"), allow_pickle=False) as saved:
                    assert saved.files == ["B", "S1", "S2", "S3"]
                    coefficients = saved["B"]
                    assert [saved[f"S{i}"].tolist() for i in (1, 2, 3)] == supports
            assert coefficients.shape == (32, 20, 25) and np.isfinite(coefficients).all()
            assert all(math.isfinite(value) for value in metrics.values()) and math.isfinite(sic)
            assert all(s == sorted(set(s)) and len(s) > 0 and min(s) >= 0 and max(s) < dim
                       for s, dim in zip(supports, coefficients.shape))
            outside = np.ones(coefficients.shape, dtype=bool)
            outside[np.ix_(*supports)] = False
            assert np.all(coefficients[outside] == 0)
            assert np.prod([len(s) for s in supports]) == metrics["q_hat"]
            assert all(len(s) == metrics[f"s{i}_hat"] for i, s in enumerate(supports, 1))
            assert abs(metrics["rss_train"] - 2 * task["config"]["n_train"] * metrics["loss_train"]) < 1e-9
            row.update({label + "_" + key: value for key, value in metrics.items()})
            row[label + "_sic"] = sic
            row[label + "_supports"] = json.dumps(supports)
            row[label + "_candidate_count"] = count
            npz_files += 1
        row.update(delta1_pmse=row["v51_pmse"] - row["v5_pmse"],
                   delta2_pmse=row["v52_pmse"] - row["v51_pmse"],
                   total_pmse=row["v52_pmse"] - row["v5_pmse"])
        assert abs(row["delta1_pmse"] + row["delta2_pmse"] - row["total_pmse"]) < 1e-12
        assert set(row) == set(existing)
        for key, value in row.items():
            assert str(value) == existing[key], (task["task_id"], key, value, existing[key])
        for key in ("v5_pmse", "v51_pmse", "v52_pmse", "delta1_pmse", "delta2_pmse",
                    "v5_exact_recovery", "v51_exact_recovery", "v52_exact_recovery",
                    "v5_q_hat", "v51_q_hat", "v52_q_hat"):
            stats[task["scenario_id"]][key].append(row[key])
    unique = Counter(task["input_key"] for task in plan["tasks"])
    scenario_identities = {}
    for task in plan["tasks"]:
        identity = {key: task[key] for key in ("experiment", "parameter", "scenario_value", "config")}
        assert scenario_identities.setdefault(task["scenario_id"], identity) == identity
    return {
        "status": "PASS", "plan_sha256": plan["sha256"], "plan_exact_native_rebuild": True,
        "pilot_gates": 5, "record_seals": 250, "source_files": 96, "npz_files": npz_files,
        "existing_csv_exact_native_rows": 250, "input_keys_unique": len(unique),
        "input_duplicate_multiplicity": dict(Counter(unique.values())),
        "expected_environment": plan["expected_environment"],
        "scenario_identities_from_original_plan": scenario_identities,
        "scenario_summary": {str(s): {"count": len(values["v5_pmse"]), **{
            key: sum(items) / len(items) for key, items in values.items()}} for s, values in stats.items()},
        "scope": "Five prescribed scenarios x 50; 250 labels, not the full 700 design.",
        "no_fits_or_merge_executed": True,
        "limitations": [
            "Linux runtime/source/input identity is checked against sealed delivered records and frozen plans; the local Windows environment is not asserted equivalent.",
            "No raw regenerated train/test streams were archived. Test pairing is bound by exact task test_seed, config, environment and evaluator/generator source; this audit does not rerun prediction metrics.",
            "The frozen worker checks the imported tensor_abess __init__ path, but records do not contain a per-module actual-import path log.",
            "Hashes and seals establish internal consistency, not independent external provenance or a cryptographic signature.",
            "Pilot seals bind five pilot results; file transfer timestamps do not independently establish execution chronology."],
        "native_merge_status": "Already merged; existing CSV equals every field of native merge construction. Do not overwrite.",
        "native_merge_command_for_fresh_unmerged_copy_only":
            "python -B -m numerical_simulation.version_attribution.run merge --output-dir numerical_simulation/outputs_v51_attribution/cluster_run"
    }


if __name__ == "__main__":
    report = audit(Path(__file__).resolve().parents[2])
    destination = Path(__file__).with_name("cluster_audit.json")
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print("PASS: 250 records, 5 gates, 750 NPZ files, all native CSV fields; no fit or merge.")
