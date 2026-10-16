"""Immutable paired analysis under the frozen A2 final-validation protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_cohort(plan, records):
    protocol = plan["protocol"]
    require(digest(protocol) == plan["protocol_id"], "Frozen protocol canonical checksum changed")
    require(protocol["groups"] == plan["groups"] == ["A0", "A2"], "Unexpected paired groups")
    expected_count = len(protocol["scenarios"]) * protocol["n_reps"]
    require(expected_count == 100 and len(plan["tasks"]) == 100, "Complete frozen 100-task plan required")
    tasks = plan["tasks"]
    task_ids = [task["task_id"] for task in tasks]
    require(len(set(task_ids)) == 100 and set(task_ids) == set(range(100)), "Duplicate or missing plan task")
    ids = [record["task_id"] for record in records]
    require(len(ids) == len(set(ids)), "Duplicate dataset results")
    require(len(records) == 100 and set(ids) == set(task_ids), "Incomplete 100 paired datasets")
    by_id = {record["task_id"]: record["report"] for record in records}
    for scenario_id, scenario in enumerate(protocol["scenarios"]):
        members = [task for task in tasks if task["scenario_id"] == scenario_id]
        require(len(members) == protocol["n_reps"] and
                {task["replicate"] for task in members} == set(range(protocol["n_reps"])),
                "Unbalanced scenario repetitions")
        for task in members:
            require(task.get("scenario", scenario) == scenario, "Frozen scenario changed")
            request = task["request"]
            require(digest(request) == task["request_id"], "Frozen request checksum changed")
            config = dict(protocol["base_config"], **{key: scenario[key] for key in ("sparsity", "n_train", "snr", "rho")})
            require(request["config"] == config, "Scenario/request configuration differs")
            seed = protocol["seed_base"] + 1000 * scenario_id + task["replicate"]
            require(request["seed"] == seed and request["test_seed"] == seed + 1_000_000,
                    "Frozen independent seeds differ")
            report = by_id[task["task_id"]]
            require(report["status"] == "complete", "Failed/incomplete result status")
            require(report["version"] == plan["version"] and report["group_order"] == ["A0", "A2"],
                    "Paired version/group order differs")
            for key in ("config", "seed", "test_seed", "fit_options"):
                require(report[key] == request[key], f"Result request mismatch: {key}")
            require(set(report["groups"]) == {"A0", "A2"} and len(report["rows"]) == 2 and
                    {row["group"] for row in report["rows"]} == {"A0", "A2"}, "Missing/duplicate paired group row")
            require(all(row["rule"] == "msic-alpha1.5-scale1" for row in report["rows"]),
                    "Unexpected MSIC parameter rule")
            require(request["fit_options"]["candidate_budget"] == protocol["solver"]["candidate_budget"],
                    "Candidate budget differs from protocol")
            reference = report["fingerprints"]
            for group, info in report["groups"].items():
                require(all(info["fingerprints"].get(key) == reference[key]
                            for key in ("X_train", "y_train", "B_true", "independent_test")),
                        "Paired training/test fingerprint differs")
                require(reference["independent_test"]["n_observations"] == config["n_test"], "Incomplete test stream")
                meta = info["search_metadata"]
                caps, capacity = meta["caps"], meta["max_params"]
                require(caps == info["s_max"] and capacity == max(1, int(np.floor(.7 * config["n_train"]))),
                        "Caps/capacity invariant differs")
                require(all(1 <= cap <= dim for cap, dim in zip(caps, config["dims"])), "Invalid training caps")
                legal = {s for s in product(*(range(1, int(cap) + 1) for cap in caps)) if np.prod(s) <= capacity}
                grid = [tuple(s) for s in meta["size_grid"]]
                require(len(grid) == len(set(grid)) and set(grid) <= legal, "Duplicate/out-of-bounds candidate grid")
                require(len(grid) <= protocol["solver"]["candidate_budget"], "Candidate budget exceeded")
                require(info["candidate_count"] == len(info["candidates"]) == len(grid) + 1,
                        "Candidate count/grid mismatch")
                require(set(map(tuple, meta["legal_shapes"])) == legal, "Legal shape metadata differs")
                if group == "A2":
                    require(set(grid) == legal and meta["grid_complete"] and not meta["excluded_legal_shapes"],
                            "Incomplete A2 legal integer grid")
            a0, a2 = (report["groups"][g] for g in ("A0", "A2"))
            require(a0["s_max"] == a2["s_max"] and a0["init_supports"] == a2["init_supports"],
                    "A0/A2 caps or training initialization differ")
    return by_id


def _rows(plan, reports):
    rows = []
    for task in sorted(plan["tasks"], key=lambda item: item["task_id"]):
        report = reports[task["task_id"]]
        config = report["config"]
        scenario = plan["protocol"]["scenarios"][task["scenario_id"]]
        truth_q = int(np.prod(config["sparsity"]))
        capacity = max(1, int(np.floor(.7 * config["n_train"])))
        for raw in report["rows"]:
            row = dict(raw)
            info = report["groups"][raw["group"]]
            meta = info["search_metadata"]
            cached_cost = sum(c["fit_seconds"] for c in info["candidates"]
                              if c["kind"] == "global-fixed" and c["cache_hit"])
            row.update(task_id=task["task_id"], scenario_id=task["scenario_id"], replicate=task["replicate"],
                sparsity=list(config["sparsity"]), rho=list(config["rho"]), stratum=scenario["stratum"],
                q_true=truth_q, q_over=int(row["q_hat"] > truth_q), q_under=int(row["q_hat"] < truth_q),
                q_small=int(row["q_hat"] == 1), q_near_cap=int(row["q_hat"] >= .9 * capacity),
                train_mse=row["rss_train"] / config["n_train"], test_mse=row["pmse"],
                capacity=capacity, true_capacity_allowed=int(truth_q <= capacity),
                true_size_reachable=int(truth_q <= capacity and all(s <= cap for s, cap in zip(config["sparsity"], info["s_max"]))),
                true_size_in_grid=int(info["true_size_in_grid"]), true_size_in_pool=int(info["true_size_in_pool"]),
                true_support_in_pool=int(info["true_support_in_pool"]),
                true_support_selected=int(row["exact_recovery"]),
                candidate_count=info["candidate_count"], legal_shape_count=len(meta["legal_shapes"]),
                searched_shape_count=len(meta["size_grid"]),
                cached_fixed_fit_cost_sec=cached_cost,
                standalone_cost_estimate_sec=row["runtime_sec"] + cached_cost)
            metrics = _metric_names()
            require(all(np.isfinite(float(row[key])) for key in metrics), "Nonfinite evaluation metric")
            require(row["pmse"] > 0 and row["runtime_sec"] > 0 and row["q_hat"] <= capacity,
                    "Invalid prediction/time/capacity metric")
            rows.append(row)
    return rows


def _metric_names():
    return ("exact_recovery", "mode_f1_macro", "element_f1", "pmse", "r2_test", "q_hat",
            "rss_train", "loss_train", "train_mse", "test_mse", "q_over", "q_under", "q_small",
            "q_near_cap", "true_capacity_allowed", "true_size_reachable", "true_size_in_grid",
            "true_size_in_pool", "true_support_in_pool", "true_support_selected", "candidate_count",
            "legal_shape_count", "searched_shape_count", "runtime_sec", "cached_fixed_fit_cost_sec",
            "standalone_cost_estimate_sec")


def _overview(rows):
    output = {}
    for group in ("A0", "A2"):
        members = [row for row in rows if row["group"] == group]
        summary = {"n_datasets": len(members)}
        for metric in _metric_names():
            values = np.asarray([row[metric] for row in members], dtype=float)
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
        output[group] = summary
    return output


def _pairs(rows):
    output = []
    for task_id in sorted({row["task_id"] for row in rows}):
        pair = {row["group"]: row for row in rows if row["task_id"] == task_id}
        a0, a2 = pair["A0"], pair["A2"]
        record = {key: a0[key] for key in ("task_id", "scenario_id", "replicate", "sparsity", "rho", "stratum", "n_train", "snr", "seed", "test_seed")}
        for metric in _metric_names():
            record[metric + "_A0"] = a0[metric]
            record[metric + "_A2"] = a2[metric]
            record[metric + "_minus_A0"] = a2[metric] - a0[metric]
        output.append(record)
    return output


def _bootstrap(pairs, specification):
    rng = np.random.default_rng(specification["seed"])
    count = specification["n_resamples"]
    require(count == 10000, "Frozen 10000-resample bootstrap changed")
    statistics = {key: np.zeros(count) for key in ("pmse_A0", "pmse_A2", "exact_gain", "mode_f1_gain", "element_f1_gain")}
    scenarios = sorted({pair["scenario_id"] for pair in pairs})
    for scenario in scenarios:
        members = [pair for pair in pairs if pair["scenario_id"] == scenario]
        indices = rng.integers(0, len(members), size=(count, len(members)))
        fields = {"pmse_A0": "pmse_A0", "pmse_A2": "pmse_A2",
                  "exact_gain": "exact_recovery_minus_A0", "mode_f1_gain": "mode_f1_macro_minus_A0",
                  "element_f1_gain": "element_f1_minus_A0"}
        for key, field in fields.items():
            values = np.asarray([pair[field] for pair in members], dtype=float)
            statistics[key] += values[indices].mean(axis=1)
    statistics = {key: value / len(scenarios) for key, value in statistics.items()}
    samples = {"pmse_ratio": statistics["pmse_A2"] / statistics["pmse_A0"],
               **{key: statistics[key] for key in ("exact_gain", "mode_f1_gain", "element_f1_gain")}}
    estimates = {"pmse_ratio": float(np.mean([p["pmse_A2"] for p in pairs]) / np.mean([p["pmse_A0"] for p in pairs]))}
    estimates.update({key: float(np.mean([p[field] for p in pairs])) for key, field in fields.items() if key.endswith("gain")})
    return {"seed": specification["seed"], "n_resamples": count, "unit": specification["unit"],
            **{key: {"estimate": estimates[key], "ci95": np.quantile(values, [.025, .975]).tolist()}
               for key, values in samples.items()}}


def _summary(rows, pairs, protocol):
    return {"n_datasets": len(pairs), "n_scenarios": len({pair["scenario_id"] for pair in pairs}),
            "groups": _overview(rows), "bootstrap": _bootstrap(pairs, protocol["bootstrap"]),
            "pmse_wins": sum(pair["pmse_minus_A0"] < -1e-10 for pair in pairs),
            "pmse_ties": sum(abs(pair["pmse_minus_A0"]) <= 1e-10 for pair in pairs),
            "pmse_losses": sum(pair["pmse_minus_A0"] > 1e-10 for pair in pairs)}


def _gate(value, threshold, direction="max"):
    passed = value <= threshold if direction == "max" else value >= threshold
    return {"status": "pass" if passed else "fail", "value": float(value),
            "threshold": threshold, "direction": direction}


def _cold_checks(plan, reports, cold_reports):
    expected = {(task, group) for task in plan["protocol"]["cold_check_tasks"] for group in ("A0", "A2")}
    times = {}
    for item in cold_reports or []:
        key = item["task_id"], item["group"]
        require(key in expected and key not in times, "Duplicate/unexpected cold check")
        cold = item["report"]
        require(cold["status"] == "complete" and cold["group_order"] == [item["group"]], "Incomplete/non-cold single-group report")
        main = reports[item["task_id"]]
        for field in ("config", "seed", "test_seed"):
            require(cold[field] == main[field], "Cold/main configuration differs")
        require(cold["fingerprints"] == main["fingerprints"], "Cold/main training/test/source fingerprints differ")
        reference = next(row for row in main["rows"] if row["group"] == item["group"])
        require(len(cold["rows"]) == 1, "Cold check must contain exactly one row")
        row = cold["rows"][0]
        for field in ("B_fingerprint", "supports", "q_hat", "rss_train", "pmse"):
            require(row[field] == reference[field], "Cold cache behavior changed selected model")
        require(not any(c.get("cache_hit") for c in cold["groups"][item["group"]]["candidates"]),
                "Cold check used cached fits")
        require(np.isfinite(row["runtime_sec"]) and row["runtime_sec"] > 0, "Invalid cold runtime")
        times[key] = row["runtime_sec"]
    ratio = sum(v for (task, group), v in times.items() if group == "A2") / sum(v for (task, group), v in times.items() if group == "A0") if set(times) == expected else None
    return {"status": "complete" if ratio is not None else "pending", "completed_group_runs": len(times),
            "expected_group_runs": len(expected), "pooled_runtime_ratio": ratio,
            "definition": "Ratio of summed measured single-group uncached fit wall times on the frozen cold-check tasks"}


def summarize_reports(plan, records, cold_reports=None, audit_evidence=None):
    """Analyze all frozen paired reports; refuse partial or altered evidence."""
    reports = _validate_cohort(plan, records)
    protocol = plan["protocol"]
    rows = _rows(plan, reports)
    pairs = _pairs(rows)
    strata = {stratum: _summary([row for row in rows if row["stratum"] == stratum],
                               [pair for pair in pairs if pair["stratum"] == stratum], protocol)
              for stratum in ("primary", "correlation_stress")}
    scenarios = []
    for scenario_id, scenario in enumerate(protocol["scenarios"]):
        selected = [row for row in rows if row["scenario_id"] == scenario_id]
        for group, metrics in _overview(selected).items():
            scenarios.append(dict(scenario_id=scenario_id, group=group, **scenario, **metrics))
    shapes = {}
    for shape in sorted({tuple(pair["sparsity"]) for pair in pairs}):
        selected_pairs = [pair for pair in pairs if pair["stratum"] == "primary" and tuple(pair["sparsity"]) == shape]
        selected_rows = [row for row in rows if row["stratum"] == "primary" and tuple(row["sparsity"]) == shape]
        shapes["x".join(map(str, shape))] = _summary(selected_rows, selected_pairs, protocol)
    primary, stress = strata["primary"], strata["correlation_stress"]
    decision = protocol["decision"]
    recovery, prediction, cost = (decision[name] for name in ("recovery", "prediction", "cost"))
    pg = primary["groups"]
    ratio = pg["A2"]["standalone_cost_estimate_sec_mean"] / pg["A0"]["standalone_cost_estimate_sec_mean"]
    cold = _cold_checks(plan, reports, cold_reports)
    scenario_ratios = []
    candidate_ratios = []
    for scenario in range(len(protocol["scenarios"])):
        members = {row["group"]: row for row in scenarios if row["scenario_id"] == scenario}
        scenario_ratios.append(members["A2"]["pmse_mean"] / members["A0"]["pmse_mean"])
        candidate_ratios.append(members["A2"]["candidate_count_mean"] / members["A0"]["candidate_count_mean"])
    gates = {
        "complete_paired_cohort": {"status": "pass", "n_datasets": 100},
        "pooled_exact_gain": _gate(primary["bootstrap"]["exact_gain"]["estimate"], recovery["pooled_exact_gain_min"], "min"),
        "each_shape_exact_gain": _gate(min(shape["bootstrap"]["exact_gain"]["estimate"] for shape in shapes.values()), recovery["each_shape_exact_gain_min"], "min"),
        "pooled_mode_f1_gain": _gate(primary["bootstrap"]["mode_f1_gain"]["estimate"], recovery["pooled_mode_f1_gain_min"], "min"),
        "pooled_pmse_ratio": _gate(primary["bootstrap"]["pmse_ratio"]["estimate"], prediction["pooled_pmse_ratio_max"]),
        "bootstrap_upper_pmse_ratio": _gate(primary["bootstrap"]["pmse_ratio"]["ci95"][1], prediction["paired_stratified_bootstrap_95_upper_ratio_max"]),
        "each_shape_pmse_ratio": _gate(max(shape["bootstrap"]["pmse_ratio"]["estimate"] for shape in shapes.values()), prediction["each_shape_pmse_ratio_max"]),
        "any_scenario_pmse_ratio": _gate(max(scenario_ratios), prediction["any_scenario_pmse_ratio_max"]),
        "stress_pooled_pmse_ratio": _gate(stress["bootstrap"]["pmse_ratio"]["estimate"], prediction["stress_pooled_pmse_ratio_max"]),
        "standalone_estimate_ratio": _gate(ratio, cost["pooled_standalone_estimate_ratio_max"]),
        "scenario_candidate_count_ratio": _gate(max(candidate_ratios), cost["scenario_candidate_count_ratio_max"]),
        "cold_check_pooled_ratio": _gate(cold["pooled_runtime_ratio"], cost["cold_check_pooled_ratio_max"]) if cold["status"] == "complete" else {"status": "pending", "threshold": cost["cold_check_pooled_ratio_max"]},
    }
    for name in ("ols_audit", "regression_checks"):
        evidence = (audit_evidence or {}).get(name)
        if evidence is None:
            gates[name] = {"status": "pending"}
            continue
        require(evidence.get("plan_id") == plan.get("plan_id"), "Audit evidence belongs to another plan")
        if name == "ols_audit":
            require(evidence.get("task_count") == 100 and evidence.get("selected_model_count") == 200,
                    "OLS audit must cover all 100 tasks and 200 selected models")
            require(evidence.get("rank_deficient_count") == 0, "OLS audit reports deficient rank")
        else:
            require(evidence.get("scope") == "a2_final_validation" and evidence.get("completed_tasks") == 100
                    and evidence.get("paired_group_rows") == 200 and evidence.get("test_count", 0) > 0,
                    "Current-cohort regression scope/count evidence required")
        gates[name] = {"status": "pass" if evidence["checks_passed"] else "fail", "evidence": evidence}
    eligible = all(gate["status"] == "pass" for gate in gates.values())
    sensitivity = {"pmse": [{"tolerance": tolerance, "threshold": 1 + tolerance,
                             "point_pass": primary["bootstrap"]["pmse_ratio"]["estimate"] <= 1 + tolerance}
                            for tolerance in decision["sensitivity"]["pmse_tolerances"]],
                   "cost": [{"threshold": threshold, "estimate_pass": ratio <= threshold,
                             "cold_pass": None if cold["status"] != "complete" else cold["pooled_runtime_ratio"] <= threshold}
                            for threshold in decision["sensitivity"]["cost_tolerances"]],
                   "no_posthoc_change": True}
    return {"status": "evidence_complete", "release_approved": False, "protocol_sha256": plan["protocol_sha256"],
            "protocol_id": plan["protocol_id"],
            "plan_id": plan.get("plan_id"), "completed_tasks": 100, "paired_group_rows": 200,
            "strata": strata, "shapes": shapes, "scenario_metrics": scenarios, "paired_metrics": pairs,
            "cold_checks": cold, "decision": {"gates": gates, "all_gates_pass": eligible,
                "research_candidate": "D2" if eligible else "D1", "default_version": "v5.1-bcd-fix",
                "final_user_approval_required": eligible}, "sensitivity": sensitivity,
            "runtime_note": "Shared-cache elapsed is not a cold-fit measurement. Standalone cost estimates add cached fixed fit_seconds. Cold checks are separate measured single-group fits.",
            "uncertainty": decision["uncertainty"], "not_executed": ["700 formal simulations", "three empirical dataset retraining", "formal production release"]}


def summarize(output):
    """Read validated immutable results and write a new immutable analysis set."""
    from numerical_simulation.a2_final_validation import run
    output = Path(output)
    plan = run.load_plan(output)
    require(not any((output / "failures").iterdir()), "Recorded budget/worker failure prohibits final analysis")
    targets = [output / name for name in ("analysis.json", "paired_metrics.csv", "scenario_metrics.csv")]
    require(not any(path.exists() for path in targets), "Immutable analysis output already exists")
    records = [{"task_id": task["task_id"], "report": run.validate_result(output, plan, task)} for task in plan["tasks"]]
    cold_records = []
    for task_id in plan["protocol"]["cold_check_tasks"]:
        for group in ("A0", "A2"):
            path = output / "cold" / f"task_{task_id:03d}_{group}.json"
            if path.exists():
                cold_records.append({"task_id": task_id, "group": group,
                                     "report": run.validate_cold_result(output, plan, plan["tasks"][task_id], group)})
    audit = {}
    for name, filename in (("ols_audit", "numerics_diagnostics.json"), ("regression_checks", "regression_checks.json")):
        path = output / filename
        if path.exists():
            evidence = json.loads(path.read_text(encoding="utf-8"))
            require(evidence.get("plan_id") == plan["plan_id"], "Audit evidence plan differs")
            audit[name] = evidence
    analysis = summarize_reports(plan, records, cold_records, audit)
    for path, values in zip(targets[1:], (analysis["paired_metrics"], analysis["scenario_metrics"])):
        columns = list(dict.fromkeys(key for row in values for key in row))
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in values)
    with targets[0].open("x", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = summarize(parser.parse_args(argv).output_dir)
    print(f"Complete independent evidence: {result['completed_tasks']} datasets; {result['decision']['research_candidate']} research decision; release not approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
