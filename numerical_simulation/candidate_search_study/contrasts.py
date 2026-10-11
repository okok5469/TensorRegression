"""Complete-cohort paired cap/grid contrasts and nested coarse-search diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("q_hat", "exact_recovery", "mode_f1_macro", "element_f1", "pmse", "r2_test",
           "rss_train", "candidate_count", "true_size_in_pool", "true_support_in_pool",
           "runtime_sec", "standalone_cost_estimate_sec")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def factorial_effects(values):
    require(set(values) == {"A0", "A1", "A2", "A3"}, "All four balanced groups are required")
    require(np.isfinite(list(values.values())).all(), "Nonfinite comparison values")
    return {"cap_at_legacy_grid": values["A1"] - values["A0"],
            "cap_at_full_grid": values["A3"] - values["A2"],
            "grid_at_legacy_cap": values["A2"] - values["A0"],
            "grid_at_candidate_cap": values["A3"] - values["A1"],
            "interaction": (values["A3"] - values["A2"]) - (values["A1"] - values["A0"])}


def candidate_identity(candidate):
    return (tuple(candidate["sizes"]), tuple(tuple(S) for S in candidate["supports"]), candidate["B_fingerprint"])


def score_tolerance(a, b):
    return 1e-10 * max(1., abs(a), abs(b))


def coarse_diagnostics(report):
    groups = report["groups"]
    require("A3" in groups and "A4" in groups, "Full and coarse candidate-cap groups required")
    full, coarse = groups["A3"], groups["A4"]
    rows = {row["group"]: row for row in report["rows"]}
    full_row, coarse_row = rows["A3"], rows["A4"]
    full_pool, coarse_pool = full["candidates"], coarse["candidates"]
    full_by_id = {candidate_identity(c): c for c in full_pool}
    coarse_ids = {candidate_identity(c) for c in coarse_pool}
    require(coarse_ids <= set(full_by_id), "Coarse candidate pool is not nested in full pool")
    for candidate in coarse_pool:
        other = full_by_id[candidate_identity(candidate)]
        require(abs(candidate["score"] - other["score"]) <= score_tolerance(candidate["score"], other["score"]),
                "Shared candidates have different training scores")
    full_selection, coarse_selection = full_row["selection"], coarse_row["selection"]
    full_minimum = float(full_selection["minimum_score"])
    coarse_minimum = float(coarse_selection["minimum_score"])
    raw_regret = coarse_minimum - full_minimum
    tolerance = score_tolerance(full_minimum, coarse_minimum)
    require(raw_regret >= -tolerance, "Coarse minimum score is better than its full superset")
    same_minimum = abs(raw_regret) <= tolerance
    minimum = full_pool[full_selection["minimum_index"]]
    selected = full_pool[full_selection["selected_index"]]
    selected_present = candidate_identity(selected) in coarse_ids
    minimum_present = candidate_identity(minimum) in coarse_ids
    coarse_shapes = {tuple(c["sizes"]) for c in coarse_pool}
    true_shape = tuple(report["config"]["sparsity"])
    result = {"full_minimum_score": full_minimum, "coarse_minimum_score": coarse_minimum,
        "minimum_score_delta_raw": raw_regret, "minimum_score_regret": max(raw_regret, 0.),
        "minimum_score_equal_within_tolerance": same_minimum, "score_tolerance": tolerance,
        "full_minimum_sizes": minimum["sizes"], "full_minimum_q": int(np.prod(minimum["sizes"])),
        "full_minimum_shape_in_coarse_pool": tuple(minimum["sizes"]) in coarse_shapes,
        "full_minimum_B_in_coarse_pool": minimum_present,
        "full_selected_sizes": selected["sizes"], "full_selected_q": int(full_row["q_hat"]),
        "full_selected_shape_in_coarse_pool": tuple(selected["sizes"]) in coarse_shapes,
        "full_selected_B_in_coarse_pool": selected_present,
        "coarse_selected_q": int(coarse_row["q_hat"]),
        "full_one_se_tolerance": full_selection["tolerance"], "coarse_one_se_tolerance": coarse_selection["tolerance"],
        "same_minimum_missing_one_se_smaller_model": bool(same_minimum and not selected_present
            and full_row["q_hat"] < coarse_row["q_hat"]),
        "coarse_misses_full_selected_B": not selected_present,
        "truth_shape_in_full_pool": bool(full["true_size_in_pool"]),
        "truth_shape_in_coarse_pool": true_shape in coarse_shapes,
        "truth_shape_omitted_by_coarse": bool(full["true_size_in_pool"] and true_shape not in coarse_shapes),
        "truth_positions_in_full_pool": bool(full["true_support_in_pool"]),
        "truth_positions_in_coarse_pool": bool(coarse["true_support_in_pool"]),
        "truth_positions_omitted_by_coarse": bool(full["true_support_in_pool"] and not coarse["true_support_in_pool"]),
        "full_candidate_count": len(full_pool), "coarse_candidate_count": len(coarse_pool),
        "candidate_fraction": len(coarse_pool) / len(full_pool),
        "coarse_minus_full_pmse": float(coarse_row.get("pmse", 0.) - full_row.get("pmse", 0.)),
        "coarse_minus_full_exact_recovery": float(coarse_row.get("exact_recovery", 0.) - full_row.get("exact_recovery", 0.)),
        "coarse_minus_full_q": float(coarse_row["q_hat"] - full_row["q_hat"])}
    for name, group in (("full", full), ("coarse", coarse)):
        fixed = [c for c in group["candidates"] if c["kind"] == "global-fixed"]
        cached_cost = sum(c.get("fit_seconds", 0.) for c in fixed if c.get("cache_hit", False))
        result[name + "_observed_runtime_sec"] = group["runtime_sec"]
        result[name + "_standalone_cost_estimate_sec"] = group["runtime_sec"] + cached_cost
        result[name + "_fixed_fit_duration_sum_sec"] = sum(c.get("fit_seconds", 0.) for c in fixed)
    return result


def extended_rows(report):
    rows = {row["group"]: dict(row) for row in report["rows"]}
    for name, row in rows.items():
        group = report["groups"][name]
        fixed = [candidate for candidate in group["candidates"] if candidate["kind"] == "global-fixed"]
        row["standalone_cost_estimate_sec"] = row["runtime_sec"] + sum(
            c["fit_seconds"] for c in fixed if c["cache_hit"])
    return rows


def build_contrasts(plan, reports):
    require(len(reports) == len(plan["tasks"]), "Only complete cohort contrasts are allowed")
    differences, coarse = [], []
    for task, report in zip(plan["tasks"], reports):
        tags = {"task_id": task["task_id"], "scenario_id": task["scenario_id"], "replicate": task["replicate"],
                "n_train": report["config"]["n_train"], "snr": report["config"]["snr"], "seed": report["seed"]}
        rows = extended_rows(report)
        for metric in METRICS:
            values = {name: float(rows[name][metric]) for name in ("A0", "A1", "A2", "A3")}
            differences.append({**tags, "metric": metric, **factorial_effects(values)})
        coarse.append({**tags, **coarse_diagnostics(report)})
    return differences, coarse


def summarize_differences(differences):
    effects = ("cap_at_legacy_grid", "cap_at_full_grid", "grid_at_legacy_cap", "grid_at_candidate_cap", "interaction")
    summary = []
    scopes = [("all", differences)]
    scenarios = sorted({(row["n_train"], row["snr"]) for row in differences})
    scopes += [(f"n{n}-snr{snr:g}", [r for r in differences if r["n_train"] == n and r["snr"] == snr]) for n, snr in scenarios]
    for scope, members in scopes:
        for metric in METRICS:
            selected = [row for row in members if row["metric"] == metric]
            for effect in effects:
                values = np.asarray([row[effect] for row in selected], dtype=float)
                summary.append({"scope": scope, "metric": metric, "effect": effect, "n_datasets": len(values),
                    "paired_difference_mean": float(values.mean()),
                    "paired_difference_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    "negative_count": int(np.sum(values < -1e-10)), "tie_count": int(np.sum(abs(values) <= 1e-10)),
                    "positive_count": int(np.sum(values > 1e-10))})
    return summary


def verify_npz_metadata(report, path):
    from numerical_simulation.candidate_search_study.worker import array_fingerprint
    with np.load(path, allow_pickle=False) as arrays:
        require(array_fingerprint(arrays["B_true"]) == report["fingerprints"]["B_true"], "True tensor NPZ fingerprint differs")
        truth = [np.flatnonzero(np.any(arrays["B_true"] != 0, axis=tuple(axis for axis in range(3) if axis != mode)))
                 for mode in range(3)]
        rows = {row["group"]: row for row in report["rows"]}
        for name, group in report["groups"].items():
            pool = group["candidates"]
            actual_truth_present = any(all(np.array_equal(S, T) for S, T in zip(candidate["supports"], truth)) for candidate in pool)
            require(actual_truth_present == group["true_support_in_pool"], "Truth positions coverage differs")
            for index in {rows[name]["selection"]["selected_index"], rows[name]["selection"]["minimum_index"]}:
                candidate = pool[index]
                require(array_fingerprint(arrays[f"{name}_candidate_{index}_B"]) == candidate["B_fingerprint"], "Selected/minimum B NPZ differs")
                for mode in (1, 2, 3):
                    require(np.array_equal(arrays[f"{name}_candidate_{index}_S{mode}"], candidate["supports"][mode - 1]),
                            "Selected/minimum support NPZ differs")


def write_csv(path, rows):
    from numerical_simulation.candidate_search_study.run import write_csv as immutable_write
    immutable_write(path, rows)


def analyse(output):
    from numerical_simulation.candidate_search_study.run import load_plan, validate_result, result_path, write_json
    output = Path(output)
    plan = load_plan(output)
    integrity_path = output / "integrity_report.json"
    require(integrity_path.is_file(), "Incomplete cohort; wait for audited summary")
    integrity = json.loads(integrity_path.read_text(encoding="utf8"))
    require(integrity["integrity_checks_passed"] and integrity["completed_tasks"] == len(plan["tasks"]), "Incomplete cohort")
    require(integrity["plan_id"] == plan["plan_id"], "Integrity plan differs")
    require((output / "mechanism_overview.csv").is_file(), "Complete mechanism analysis required first")
    names = ("factorial_dataset_contrasts.csv", "factorial_contrast_summary.csv", "coarse_search_diagnostics.csv", "factorial_integrity.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Immutable factorial analysis already exists")
    reports = []
    for task in plan["tasks"]:
        report = validate_result(output, plan, task)
        verify_npz_metadata(report, result_path(output, task).with_suffix(".npz"))
        reports.append(report)
    differences, coarse = build_contrasts(plan, reports)
    summary = summarize_differences(differences)
    for name, values in zip(names[:3], (differences, summary, coarse)):
        write_csv(output / name, values)
    diagnostics = {"status": "complete", "cohort": plan["cohort"], "plan_id": plan["plan_id"], "datasets": len(reports),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "regret_mean": float(np.mean([row["minimum_score_regret"] for row in coarse])),
        "regret_max": max(row["minimum_score_regret"] for row in coarse),
        "same_minimum_count": sum(row["minimum_score_equal_within_tolerance"] for row in coarse),
        "full_minimum_B_missing_count": sum(not row["full_minimum_B_in_coarse_pool"] for row in coarse),
        "full_selected_B_missing_count": sum(not row["full_selected_B_in_coarse_pool"] for row in coarse),
        "same_minimum_missing_one_se_smaller_model_count": sum(row["same_minimum_missing_one_se_smaller_model"] for row in coarse),
        "truth_shape_omitted_count": sum(row["truth_shape_omitted_by_coarse"] for row in coarse),
        "truth_positions_omitted_count": sum(row["truth_positions_omitted_by_coarse"] for row in coarse),
        "inference": "Paired descriptive conditional effects, not independently estimated population interactions; scenarios and seeds balanced.",
        "runtime_note": "Observed elapsed time depends on shared-cache group order. Estimated standalone cost adds cached fixed fit durations and is not an independent cold-fit timing measurement.",
        "scope": "Search mechanisms only; no MSIC/confirmation execution or formal-version adoption"}
    write_json(output / names[3], diagnostics)
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(analyse(parser.parse_args().output_dir), indent=2))
