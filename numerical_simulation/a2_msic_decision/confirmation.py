"""Frozen independent eighty-dataset confirmation of two predeclared A2 MSIC rules."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

from numerical_simulation.a2_msic_decision import run as scoring
from numerical_simulation.a2_final_validation import run as search
from numerical_simulation.a2_final_validation.analyse import _bootstrap
from numerical_simulation.candidate_search_study import run as legacy
from numerical_simulation.candidate_search_study.numerics import audit_selected_model


def expected_scenarios():
    shapes = ([2, 3, 4], [3, 4, 5])
    primary = [{"sparsity": shape, "n_train": n, "snr": float(snr), "rho": [.3] * 3, "stratum": "primary"}
               for shape in shapes for n in (100, 200, 400) for snr in (2, 4)]
    weak = [{"sparsity": shape, "n_train": 400, "snr": .5, "rho": [.3] * 3, "stratum": "primary"} for shape in shapes]
    stress = [{"sparsity": shape, "n_train": 200, "snr": 2., "rho": [.6] * 3, "stratum": "correlation_stress"} for shape in shapes]
    return primary + weak + stress


def build_tasks(protocol):
    legacy.require(protocol["scenarios"] == expected_scenarios() and protocol["n_reps"] == 5
                   and protocol["seed_base"] == 30601011, "Frozen 80-task scenarios/seeds changed")
    tasks = []
    for scenario_id, scenario in enumerate(protocol["scenarios"]):
        for rep in range(5):
            seed = protocol["seed_base"] + scenario_id * 1000 + rep
            request = {"config": dict(protocol["base_config"], **{k: scenario[k] for k in ("sparsity", "n_train", "snr", "rho")}),
                       "seed": seed, "test_seed": seed + 1_000_000, "groups": ["A2"],
                       "fit_options": {"candidate_budget": protocol["budget"]["candidate_budget"],
                                       "selection": "original MSIC alpha=1.5,gamma=1,one_se_scale=1; Phase D only",
                                       "cache": "single A2 fresh worker; no external cache"}}
            task_id = len(tasks)
            tasks.append({"task_id": task_id, "scenario_id": scenario_id, "replicate": rep,
                          "scenario": scenario, "request": request,
                          "request_file": f"requests/task_{task_id:03d}.json", "request_id": legacy.digest(request)})
    return tasks


def sources():
    manifests = legacy.source_manifests()
    relatives = ["numerical_simulation/a2_msic_decision/run.py", "numerical_simulation/a2_msic_decision/confirmation.py",
                 "numerical_simulation/a2_final_validation/run.py", "numerical_simulation/a2_final_validation/analyse.py",
                 "numerical_simulation/candidate_search_study/numerics.py",
                 "numerical_simulation/candidate_search_study/msic_after_search.py"]
    manifests["experiment"].update({rel: legacy.file_hash(legacy.root() / rel) for rel in relatives})
    return manifests


def _parents():
    folder = legacy.root() / "numerical_simulation/outputs_a2_msic_decision"
    rules = legacy.read_json(folder / "MSIC_DECISION_RULES.json")
    integrity = legacy.read_json(folder / "integrity_report.json")
    screen = legacy.read_json(folder / "DEVELOPMENT_SCREEN_RESULT.json")
    legacy.require(screen["status"] == "pass" and screen["rules_sha256"] == legacy.file_hash(folder / "MSIC_DECISION_RULES.json"),
                   "Predeclared development screen did not pass")
    legacy.require(integrity["status"] == "complete" and integrity["checks_passed"] and integrity["dataset_count"] == 90
                   and integrity["rule_rows"] == 180 and integrity["default_exact_replays"] == 90
                   and integrity["pool_fingerprints_unchanged"], "Controlled development parent incomplete")
    return rules, {name: legacy.file_hash(folder / name) for name in
                   ("MSIC_DECISION_RULES.json", "integrity_report.json", "plan.json", "rule_summary.csv", "scenario_summary.csv", "DEVELOPMENT_SCREEN_RESULT.json")}


def build_plan(output):
    output = Path(output)
    protocol = legacy.read_json(output / "protocol.json")
    rules, parents = _parents()
    legacy.require(protocol["rules"] == rules["rules"], "Two locked MSIC rules changed")
    for field, filename in (("development_screen_sha256", "DEVELOPMENT_SCREEN_RESULT.json"),
                            ("development_integrity_sha256", "integrity_report.json"),
                            ("development_rule_summary_sha256", "rule_summary.csv"),
                            ("rules_file_sha256", "MSIC_DECISION_RULES.json")):
        legacy.require(protocol[field] == parents[filename], "Frozen development parent hash differs")
    checksum = (output / "PROTOCOL.sha256").read_text(encoding="utf-8").split()[0]
    legacy.require(checksum == legacy.file_hash(output / "protocol.json"), "Frozen protocol file checksum differs")
    legacy.require(protocol["confirmation_gate"] == rules["if_screen_passes"]["confirmation_gate"], "Predeclared confirmation gates changed")
    legacy.require(protocol["refit_tasks"] == [0, 20, 50], "Predeclared fresh-refit tasks changed")
    legacy.require(protocol["budget"]["max_controller_seconds"] == 3600 and protocol["budget"]["workers"] == 8
                   and protocol["budget"]["candidate_budget"] == 2000, "Confirmation budget changed")
    legacy.require(protocol["base_config"]["msic_refine_alpha"] == 1.5
                   and protocol["base_config"]["msic_refine_one_se_scale"] == 1, "Original fitting MSIC must remain fixed")
    tasks = build_tasks(protocol)
    manifests = sources()
    legacy.require(protocol["core_sources"] == manifests["core"] and protocol["environment"] == legacy.environment(), "Frozen core/environment differs")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=legacy.root(), capture_output=True, text=True, check=True).stdout.strip()
    body = {"schema_version": 1, "version": "v5.2-candidate-search", "cohort": "independent_A2_MSIC_confirmation",
            "groups": ["A2"], "tasks": tasks, "protocol": protocol,
            "protocol_sha256": legacy.file_hash(output / "protocol.json"), "protocol_id": legacy.digest(protocol),
            "sources": manifests, "environment": legacy.environment(), "git_commit": commit,
            "controlled_development_parents": parents, "rules": scoring.declared_rules(),
            "research_only": True, "release_approved": False}
    return {**body, "plan_id": legacy.digest(body)}


def create_plan(output):
    output = Path(output)
    if (output / "plan.json").exists() or any(p.name.lower() not in {"protocol.json", "protocol.sha256"} for p in output.iterdir()):
        raise FileExistsError("Confirmation plan requires a new protocol-only directory")
    plan = build_plan(output)
    for name in ("requests", "jobs", "evaluations", "logs", "failures", "refits"):
        (output / name).mkdir()
    for task in plan["tasks"]:
        legacy.write_json(output / task["request_file"], task["request"])
    legacy.write_json(output / "plan.json", plan)
    return plan


def reject_failures(output):
    output = Path(output)
    legacy.require(not (output / "STOPPED.json").exists() and not any((output / "failures").iterdir()), "Stopped/failure confirmation cannot resume")


def validate_pair_inventory(folder, stems):
    for path in Path(folder).iterdir():
        legacy.require(path.is_file() and path.stem in stems and path.suffix in {".json", ".npz"}, "Unknown result file")
        legacy.require(path.with_suffix(".json").is_file() and path.with_suffix(".npz").is_file(), "Incomplete result/cache pair")


def load_plan(output):
    output = Path(output)
    reject_failures(output)
    plan = legacy.read_json(output / "plan.json")
    legacy.require(plan["plan_id"] == legacy.digest({k: v for k, v in plan.items() if k != "plan_id"}), "Confirmation plan checksum changed")
    legacy.require(plan == build_plan(output), "Frozen source/config/environment/parent changed")
    legacy.require({p.name for p in (output / "requests").iterdir()} == {Path(t["request_file"]).name for t in plan["tasks"]}, "Unknown/missing request")
    for task in plan["tasks"]:
        legacy.require(legacy.read_json(output / task["request_file"]) == task["request"], "Frozen request changed")
    stems = {f"task_{t['task_id']:03d}" for t in plan["tasks"]}
    for folder in ("jobs", "evaluations"):
        validate_pair_inventory(output / folder, stems)
    validate_pair_inventory(output / "refits", {f"task_{i:03d}" for i in plan["protocol"]["refit_tasks"]})
    return plan


def deadline(output, plan):
    return search.controller_deadline(Path(output), plan)


def stop(output, plan, error):
    output = Path(output)
    evidence = {"status": "stopped", "plan_id": plan["plan_id"], "error": str(error),
                "utc_stopped": datetime.now(timezone.utc).isoformat(), "resume_allowed": False}
    if not (output / "STOPPED.json").exists():
        legacy.write_json(output / "STOPPED.json", evidence)
    path = output / "failures/controller_stopped.json"
    if not path.exists():
        legacy.write_json(path, evidence)


def _dataset(report):
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    values = dict(report["config"])
    for key in ("dims", "sparsity", "rho"):
        values[key] = tuple(values[key])
    dataset = generate_simulation_dataset(TensorSimulationConfig(**values), seed=report["seed"])
    for key, value in (("X_train", dataset.X_train), ("y_train", dataset.y_train), ("B_true", dataset.B_true)):
        legacy.require(scoring.array_fingerprint(value) == report["fingerprints"][key], "OLS regenerated dataset differs")
    return dataset


def evaluate_and_audit(report, cache):
    scoring.configure_core()
    result, arrays = scoring.evaluate_task(report, cache)
    data = _dataset(report)
    audits = []
    for row in result["rows"]:
        role = row["evaluation_role"]
        audit = audit_selected_model(data.X_train, data.y_train, arrays[role + "_B"],
                    [arrays[role + f"_S{m}"] for m in (1, 2, 3)], expected_rss=row["rss_train"], expected_loss=row["loss_train"])
        legacy.require(audit["checks_passed"] and not audit["rank_deficient"], "Selected confirmation OLS is rank deficient")
        audits.append({"rule": row["rule"], **audit})
    result["selected_model_audits"] = audits
    result["confirmation_performed"] = True
    result["new_model_fitting"] = True
    result["scope"] = "Two training-only rules applied to this newly fitted A2 pool; original default replay exact"
    return result, arrays


def validate_evaluation(output, plan, task):
    scoring.configure_core()
    report = search.validate_result(output, plan, task)
    path = Path(output) / "evaluations" / f"task_{task['task_id']:03d}.json"
    value = legacy.read_json(path)
    legacy.require(value["status"] == "complete" and value["plan_id"] == plan["plan_id"]
                   and value["source_json_sha256"] == legacy.file_hash(legacy.result_path(output, task))
                   and value["source_npz_sha256"] == report["array_cache_sha256"], "Evaluation source binding differs")
    legacy.require(value["array_cache"] == path.with_suffix(".npz").name
                   and value["array_cache_sha256"] == legacy.file_hash(path.with_suffix(".npz")), "Evaluation arrays changed")
    legacy.require(value["default_exact_replay"] and value["pool_fingerprint_before"] == value["pool_fingerprint_after"]
                   == report["groups"]["A2"]["pool_fingerprint"], "Default replay/pool integrity differs")
    legacy.require(len(value["rows"]) == 2 and {r["rule"] for r in value["rows"]} == {r["rule"] for r in plan["rules"]}, "Confirmation rules differ")
    legacy.require(len(value["selected_model_audits"]) == 2 and all(a["checks_passed"] and not a["rank_deficient"] for a in value["selected_model_audits"]), "Incomplete selected OLS audits")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for row in value["rows"]:
            role = row["evaluation_role"]
            legacy.require(scoring.array_fingerprint(arrays[role + "_B"]) == row["B_fingerprint"]
                           and [arrays[role + f"_S{m}"].tolist() for m in (1, 2, 3)] == row["supports"], "Evaluated selected model cache differs")
    default = next(r for r in value["rows"] if r["evaluation_role"] == "default")
    original = report["rows"][0]
    scoring.default_replay({k: v for k, v in original.items() if k not in ("runtime_sec", "cached_fixed_fit_cost_sec", "standalone_cost_estimate_sec")}, default)
    return value


def execute_task(output, plan, task):
    scoring.configure_core()
    legacy.execute_task(output, plan, task)
    report = search.validate_result(output, plan, task)
    cache = legacy.result_path(output, task).with_suffix(".npz")
    result, arrays = evaluate_and_audit(report, cache)
    result.update(plan_id=plan["plan_id"], task_id=task["task_id"], source_json_sha256=legacy.file_hash(cache.with_suffix(".json")), source_npz_sha256=legacy.file_hash(cache))
    target = Path(output) / "evaluations" / f"task_{task['task_id']:03d}.json"
    with target.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result.update(array_cache=target.with_suffix(".npz").name, array_cache_sha256=legacy.file_hash(target.with_suffix(".npz")))
    legacy.write_json(target, result)
    return validate_evaluation(output, plan, task)


def run_plan(output, workers=8):
    output = Path(output)
    plan = load_plan(output)
    scoring.configure_core()
    legacy.require(workers == plan["protocol"]["budget"]["workers"], "Frozen worker count changed")
    legacy.require(not any((output / "jobs").iterdir()) and not any((output / "evaluations").iterdir()), "Confirmation fitting does not resume")
    limit = deadline(output, plan)
    executor = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(plan["tasks"])
    active = {}
    try:
        legacy.require(monotonic() < limit, "Shared confirmation budget exceeded")
        for _ in range(workers):
            task = next(iterator)
            active[executor.submit(execute_task, output, plan, task)] = task
        complete = 0
        while active:
            done, _ = wait(active, timeout=max(0, min(60, limit - monotonic())), return_when=FIRST_COMPLETED)
            legacy.require(monotonic() < limit, "Shared confirmation budget exceeded")
            for future in done:
                active.pop(future)
                future.result()
                complete += 1
                print(f"Independent A2 MSIC confirmation complete {complete}/80", flush=True)
            for _ in done:
                task = next(iterator, None)
                if task is not None:
                    active[executor.submit(execute_task, output, plan, task)] = task
    except Exception as exc:
        stop(output, plan, exc)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return {"status": "confirmation_fits_complete", "plan_id": plan["plan_id"], "completed_tasks": 80}


def validate_complete_records(plan, records):
    legacy.require(len(records) == 80, "Complete 80 confirmation records required")
    ids = [record["task_id"] for record in records]
    legacy.require(len(set(ids)) == 80, "Duplicate confirmation task")
    expected = {t["task_id"] for t in plan["tasks"]}
    legacy.require(set(ids) == expected, "Missing/unknown confirmation task")


def decision_gates(metrics, gate):
    fields = [("pooled_pmse_ratio", "pooled_pmse_ratio_max", "max"),
              ("bootstrap_upper", "paired_stratified_bootstrap_pmse_ratio_95_upper_max", "max"),
              ("worst_scenario_ratio", "any_scenario_pmse_ratio_max", "max"),
              ("strong_exact_gain", "strong_representable_exact_gain_min", "min"),
              ("strong_mode_f1_gain", "strong_representable_mode_f1_gain_min", "min"),
              ("strong_under_increase", "strong_representable_under_rate_increase_max", "max"),
              ("strong_q1_increase", "strong_representable_q1_rate_increase_max", "max")]
    gates = {}
    for metric, threshold, direction in fields:
        value = metrics[metric]
        passed = np.isfinite(value) and (value <= gate[threshold] if direction == "max" else value >= gate[threshold])
        gates[metric] = {"value": value, "threshold": gate[threshold], "status": "pass" if passed else "fail"}
    return {"gates": gates, "statistical_gates_pass": all(g["status"] == "pass" for g in gates.values())}


def refit_plan(output):
    output = Path(output)
    plan = load_plan(output)
    scoring.configure_core()
    limit = deadline(output, plan)
    from tensor_abess.tensor.abess import tensor_abess
    try:
        legacy.require(not any((output / "refits").iterdir()), "Fresh refit stage cannot resume or overwrite")
        for task_id in plan["protocol"]["refit_tasks"]:
            legacy.require(monotonic() < limit, "Shared refit budget exceeded")
            task = plan["tasks"][task_id]
            evaluated = validate_evaluation(output, plan, task)
            report = legacy.read_json(legacy.result_path(output, task))
            data = _dataset(report)
            config = report["config"]
            fitted = tensor_abess(data.X_train, data.y_train, k_max=config["k_max"], c=config["abess_init_c"],
                criterion=config["abess_init_criterion"], n_lambdas=config["n_lambdas"], cv_folds=config["cv_folds"],
                max_iter=config["max_iter"], tol=config["tol"], max_cycles=config["max_cycles"], eps=1e-8,
                adaptive_criterion="msic", cap_policy="legacy", grid_policy="full", candidate_budget=2000,
                msic_refine_alpha=2., msic_refine_gamma=1., msic_refine_one_se_scale=0.)
            row = next(r for r in evaluated["rows"] if r["evaluation_role"] == "mild_candidate")
            legacy.require(scoring.array_fingerprint(fitted.B) == row["B_fingerprint"]
                           and [fitted.S1.tolist(), fitted.S2.tolist(), fitted.S3.tolist()] == row["supports"]
                           and fitted.rss == row["rss_train"], "Fresh full-fit and controlled scoring are not exactly equivalent")
            target = output / "refits" / f"task_{task_id:03d}.json"
            with target.with_suffix(".npz").open("xb") as handle:
                np.savez_compressed(handle, B=fitted.B, S1=fitted.S1, S2=fitted.S2, S3=fitted.S3)
            legacy.write_json(target, {"status": "complete", "plan_id": plan["plan_id"], "task_id": task_id,
                "source_evaluation_sha256": legacy.file_hash(output / "evaluations" / target.name),
                "B_fingerprint": row["B_fingerprint"], "rss": fitted.rss, "supports": row["supports"],
                "checks_passed": True, "new_full_fit": True, "array_cache_sha256": legacy.file_hash(target.with_suffix(".npz"))})
            legacy.require(monotonic() < limit, "Shared refit budget exceeded")
    except Exception as exc:
        stop(output, plan, exc)
        raise
    return {"status": "fresh_refits_complete", "plan_id": plan["plan_id"], "task_count": 3}


def summarize(output):
    output = Path(output)
    plan = load_plan(output)
    limit = deadline(output, plan)
    targets = [output / name for name in ("analysis.json", "raw_results.csv", "scenario_summary.csv", "integrity_report.json")]
    legacy.require(not any(p.exists() for p in targets), "Immutable confirmation summary already exists")
    try:
        records = [{"task_id": t["task_id"], "report": validate_evaluation(output, plan, t)} for t in plan["tasks"]]
        validate_complete_records(plan, records)
        pairs, rows = [], []
        strong = []
        for task, record in zip(plan["tasks"], records):
            values = record["report"]["rows"]
            baseline, candidate = (next(r for r in values if r["evaluation_role"] == role) for role in ("default", "mild_candidate"))
            pair = {"task_id": task["task_id"], "scenario_id": task["scenario_id"], "stratum": task["scenario"]["stratum"],
                "pmse_A0": baseline["pmse"], "pmse_A2": candidate["pmse"],
                "exact_recovery_minus_A0": candidate["exact_recovery"] - baseline["exact_recovery"],
                "mode_f1_macro_minus_A0": candidate["mode_f1_macro"] - baseline["mode_f1_macro"],
                "element_f1_minus_A0": candidate["element_f1"] - baseline["element_f1"],
                "under_increase": candidate["q_under"] - baseline["q_under"], "q1_increase": candidate["q_small"] - baseline["q_small"]}
            pairs.append(pair)
            source = legacy.read_json(legacy.result_path(output, task))
            caps = source["groups"]["A2"]["s_max"]
            config = task["request"]["config"]
            if config["snr"] == 4 and all(s <= cap for s, cap in zip(config["sparsity"], caps)) and np.prod(config["sparsity"]) <= .7 * config["n_train"]:
                strong.append(pair)
            rows.extend({**row, "task_id": task["task_id"], "scenario_id": task["scenario_id"], "stratum": pair["stratum"],
                         "sparsity": config["sparsity"], "rho": config["rho"]} for row in values)
        gate = plan["protocol"]["confirmation_gate"]
        spec = {"seed": gate["bootstrap_seed"], "n_resamples": gate["bootstrap_resamples"], "unit": "paired dataset resampled within frozen scenario"}
        interval = _bootstrap(pairs, spec)
        legacy.require(bool(strong), "No strong representable confirmation evidence")
        scene_ratios = [np.mean([p["pmse_A2"] for p in pairs if p["scenario_id"] == scene]) /
                        np.mean([p["pmse_A0"] for p in pairs if p["scenario_id"] == scene]) for scene in range(16)]
        metrics = {"pooled_pmse_ratio": interval["pmse_ratio"]["estimate"], "bootstrap_upper": interval["pmse_ratio"]["ci95"][1],
                   "worst_scenario_ratio": max(scene_ratios),
                   "strong_exact_gain": float(np.mean([p["exact_recovery_minus_A0"] for p in strong])),
                   "strong_mode_f1_gain": float(np.mean([p["mode_f1_macro_minus_A0"] for p in strong])),
                   "strong_under_increase": float(np.mean([p["under_increase"] for p in strong])),
                   "strong_q1_increase": float(np.mean([p["q1_increase"] for p in strong]))}
        decision = decision_gates(metrics, gate)
        refits = []
        for task_id in plan["protocol"]["refit_tasks"]:
            path = output / "refits" / f"task_{task_id:03d}.json"
            if not path.exists():
                continue
            value = legacy.read_json(path)
            legacy.require(value["checks_passed"] and value["plan_id"] == plan["plan_id"] and value["task_id"] == task_id
                           and value["source_evaluation_sha256"] == legacy.file_hash(output / "evaluations" / path.name)
                           and value["array_cache_sha256"] == legacy.file_hash(path.with_suffix(".npz")), "Fresh-refit evidence differs")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                legacy.require(scoring.array_fingerprint(arrays["B"]) == value["B_fingerprint"]
                               and [arrays[f"S{m}"].tolist() for m in (1, 2, 3)] == value["supports"], "Fresh-refit arrays differ")
            refits.append(task_id)
        decision.update(fresh_refit_status="pass" if refits == [0, 20, 50] else "pending",
                        recommend_new_msic=decision["statistical_gates_pass"] and refits == [0, 20, 50],
                        retain_original_msic=not (decision["statistical_gates_pass"] and refits == [0, 20, 50]))
        strata = {name: {"n_datasets": len([p for p in pairs if p["stratum"] == name]),
                         "bootstrap": _bootstrap([p for p in pairs if p["stratum"] == name], spec),
                         "summary": scoring.summary_rows([r for r in rows if r["stratum"] == name])}
                  for name in ("primary", "correlation_stress")}
        analysis = {"status": "confirmation_evidence_complete", "plan_id": plan["plan_id"], "release_approved": False,
            "completed_tasks": 80, "rule_rows": 160, "selected_models_audited": 160, "bootstrap": interval,
            "strata": strata, "strong_representable_n": len(strong), "metrics": metrics, "decision": decision,
            "scenario_pmse_ratios": scene_ratios, "scope": "Independent two-rule A2 confirmation; no parameter retuning; no formal release"}
        legacy.require(monotonic() < limit, "Shared summary budget exceeded")
        compact = [{k: v for k, v in r.items() if k not in ("selection", "supports")} for r in rows]
        scoring.write_csv(targets[1], compact)
        scene_summary = []
        for scene in range(16):
            scene_summary.extend({"scenario_id": scene, **plan["protocol"]["scenarios"][scene], **r}
                                 for r in scoring.summary_rows([r for r in rows if r["scenario_id"] == scene]))
        scoring.write_csv(targets[2], scene_summary)
        legacy.write_json(targets[0], analysis)
        legacy.write_json(targets[3], {"status": "complete", "checks_passed": True, "plan_id": plan["plan_id"],
            "dataset_count": 80, "rule_rows": 160, "default_exact_replays": 80, "selected_model_count": 160,
            "rank_deficient_count": 0, "pool_fingerprints_unchanged": True, "release_approved": False})
    except Exception as exc:
        stop(output, plan, exc)
        raise
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "refit", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "run":
            sub.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    actions = {"plan": lambda: create_plan(args.output_dir), "run": lambda: run_plan(args.output_dir, args.workers),
               "refit": lambda: refit_plan(args.output_dir), "summarize": lambda: summarize(args.output_dir)}
    result = actions[args.command]()
    print(legacy.canonical({"plan_id": result["plan_id"], "status": result.get("status", "planned")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
