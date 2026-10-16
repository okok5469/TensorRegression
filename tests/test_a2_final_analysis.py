"""Independent final-validation acceptance and failure-boundary tests."""
import copy
import hashlib
import importlib
import json
from itertools import product
from pathlib import Path
import unittest


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixture():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "numerical_simulation/outputs_a2_final_validation/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tasks, records = [], []
    for scenario_id, scenario in enumerate(protocol["scenarios"]):
        config = dict(protocol["base_config"], **{key: scenario[key] for key in ("sparsity", "n_train", "snr", "rho")})
        caps = config["sparsity"]
        grid = [list(s) for s in product(*(range(1, cap + 1) for cap in caps))]
        for replicate in range(protocol["n_reps"]):
            task_id = len(tasks)
            seed = protocol["seed_base"] + scenario_id * 1000 + replicate
            request = {"config": config, "seed": seed, "test_seed": seed + 1_000_000,
                       "groups": ["A0", "A2"], "fit_options": {"candidate_budget": 2000}}
            task = {"task_id": task_id, "scenario_id": scenario_id, "replicate": replicate,
                    "scenario": scenario, "request": request, "request_id": digest(request)}
            tasks.append(task)
            groups, rows = {}, []
            fingerprints = {"X_train": "same-X", "y_train": "same-y", "B_true": "truth",
                            "independent_test": {"sha256": "test", "n_observations": 500}}
            for group, pmse, exact, mode_f1, time in (("A0", 2., 0., .5, 10.), ("A2", 1.8, 1., .7, 11.)):
                groups[group] = {"s_max": caps, "init_supports": [[0], [0], [0]],
                    "fingerprints": fingerprints, "candidate_count": len(grid) + 1,
                    "true_size_in_grid": True, "true_size_in_pool": True, "true_support_in_pool": True,
                    "search_metadata": {"caps": caps, "max_params": int(.7 * config["n_train"]),
                        "size_grid": grid, "legal_shapes": grid, "grid_complete": True, "excluded_legal_shapes": []},
                    "candidates": [{"kind": "adaptive-stage", "fit_seconds": 0., "cache_hit": False}]
                        + [{"kind": "global-fixed", "sizes": s, "fit_seconds": .01,
                            "cache_hit": group == "A2"} for s in grid]}
                rows.append({"group": group, "rule": "msic-alpha1.5-scale1", "seed": seed,
                    "test_seed": seed + 1_000_000, "n_train": config["n_train"], "snr": config["snr"],
                    "q_hat": int(caps[0] * caps[1] * caps[2]), "exact_recovery": exact,
                    "mode_f1_macro": mode_f1, "element_f1": mode_f1, "pmse": pmse,
                    "r2_test": .2, "rss_train": float(config["n_train"]), "loss_train": .5,
                    "runtime_sec": time, "candidate_count": len(grid) + 1,
                    "B_fingerprint": group + "-B", "supports": [list(range(s)) for s in caps]})
            report = {"status": "complete", "version": "v5.2-candidate-search", "config": config,
                      "seed": seed, "test_seed": seed + 1_000_000, "group_order": ["A0", "A2"],
                      "fit_options": request["fit_options"], "fingerprints": fingerprints,
                      "groups": groups, "rows": rows}
            records.append({"task_id": task_id, "report": report})
    plan = {"version": "v5.2-candidate-search", "protocol": protocol,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "protocol_id": digest(protocol), "groups": ["A0", "A2"], "tasks": tasks}
    return plan, records


class FinalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.records = fixture()

    def analyzer(self):
        try:
            return importlib.import_module("numerical_simulation.a2_final_validation.analyse")
        except ModuleNotFoundError:
            self.fail("Independent final-validation analyzer is not implemented")

    def test_complete_primary_and_stress_stay_separate_and_bootstrap_is_paired(self):
        result = self.analyzer().summarize_reports(self.plan, self.records)
        self.assertEqual(result["status"], "evidence_complete")
        self.assertEqual(result["strata"]["primary"]["n_datasets"], 90)
        self.assertEqual(result["strata"]["correlation_stress"]["n_datasets"], 10)
        interval = result["strata"]["primary"]["bootstrap"]
        self.assertAlmostEqual(interval["pmse_ratio"]["estimate"], .9)
        self.assertAlmostEqual(interval["pmse_ratio"]["ci95"][0], .9)
        self.assertAlmostEqual(interval["pmse_ratio"]["ci95"][1], .9)
        self.assertEqual(interval["exact_gain"]["ci95"], [1., 1.])
        self.assertEqual(interval["n_resamples"], 10000)
        self.assertEqual(len(result["scenario_metrics"]), 40)
        self.assertEqual(len(result["paired_metrics"]), 100)
        self.assertEqual(result["release_approved"], False)

    def test_cache_cost_is_added_and_missing_cold_checks_prevent_adoption(self):
        result = self.analyzer().summarize_reports(self.plan, self.records)
        a2 = result["strata"]["primary"]["groups"]["A2"]
        self.assertGreater(a2["standalone_cost_estimate_sec_mean"], a2["runtime_sec_mean"])
        self.assertEqual(a2["train_mse_mean"], 1.)
        self.assertEqual(result["decision"]["gates"]["cold_check_pooled_ratio"]["status"], "pending")
        self.assertEqual(result["decision"]["default_version"], "v5.1-bcd-fix")

    def test_refuses_incomplete_full_cohort(self):
        with self.assertRaisesRegex(ValueError, "complete|100|missing|Incomplete"):
            self.analyzer().summarize_reports(self.plan, self.records[:-1])

    def test_refuses_duplicate_task(self):
        with self.assertRaisesRegex(ValueError, "Duplicate|duplicate"):
            self.analyzer().summarize_reports(self.plan, self.records[:-1] + [self.records[0]])

    def test_refuses_failed_or_unpaired_report(self):
        records = copy.deepcopy(self.records)
        records[0]["report"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "complete|failed|status"):
            self.analyzer().summarize_reports(self.plan, records)
        records[0]["report"]["status"] = "complete"
        records[0]["report"]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "group|paired|row"):
            self.analyzer().summarize_reports(self.plan, records)

    def test_refuses_modified_protocol_and_unbalanced_plan(self):
        plan = copy.deepcopy(self.plan)
        plan["protocol"]["decision"]["prediction"]["pooled_pmse_ratio_max"] = 5
        with self.assertRaisesRegex(ValueError, "protocol|checksum|hash"):
            self.analyzer().summarize_reports(plan, self.records)
        plan = copy.deepcopy(self.plan)
        plan["tasks"][0]["scenario_id"] = 1
        with self.assertRaisesRegex(ValueError, "scenario|balanced|replicate"):
            self.analyzer().summarize_reports(plan, self.records)

    def test_changed_test_fingerprint_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["report"]["groups"]["A2"]["fingerprints"] = {"X_train": "different"}
        with self.assertRaisesRegex(ValueError, "fingerprint|pair"):
            self.analyzer().summarize_reports(self.plan, records)

    def test_frozen_prediction_gate_rejects_worse_a2_without_changing_tolerance(self):
        records = copy.deepcopy(self.records)
        for record in records:
            record["report"]["rows"][1]["pmse"] = 2.8
        result = self.analyzer().summarize_reports(self.plan, records)
        self.assertEqual(result["decision"]["gates"]["pooled_pmse_ratio"]["status"], "fail")
        self.assertEqual(result["decision"]["default_version"], "v5.1-bcd-fix")
        self.assertEqual(result["decision"]["gates"]["pooled_pmse_ratio"]["threshold"], 1.05)

    def test_bootstrap_preserves_varying_pair_scale(self):
        records = copy.deepcopy(self.records)
        for task, record in zip(self.plan["tasks"], records):
            value = task["replicate"] + 1
            record["report"]["rows"][0]["pmse"] = value
            record["report"]["rows"][1]["pmse"] = .9 * value
        interval = self.analyzer().summarize_reports(self.plan, records)["strata"]["primary"]["bootstrap"]["pmse_ratio"]
        for value in interval["ci95"]:
            self.assertAlmostEqual(value, .9)

    def cold_fixture(self):
        cold = []
        for task_id in self.plan["protocol"]["cold_check_tasks"]:
            for group in ("A0", "A2"):
                report = copy.deepcopy(self.records[task_id]["report"])
                report["group_order"] = [group]
                report["groups"] = {group: report["groups"][group]}
                report["rows"] = [row for row in report["rows"] if row["group"] == group]
                report["rows"][0]["runtime_sec"] = 10. if group == "A0" else 12.
                for candidate in report["groups"][group]["candidates"]:
                    candidate["cache_hit"] = False
                cold.append({"task_id": task_id, "group": group, "report": report})
        return cold

    def test_complete_uncached_cold_checks_measure_actual_ratio(self):
        result = self.analyzer().summarize_reports(self.plan, self.records, self.cold_fixture())
        self.assertEqual(result["cold_checks"]["completed_group_runs"], 16)
        self.assertAlmostEqual(result["cold_checks"]["pooled_runtime_ratio"], 1.2)
        self.assertEqual(result["decision"]["gates"]["cold_check_pooled_ratio"]["status"], "pass")

    def test_cold_cache_hit_or_changed_model_is_rejected(self):
        cold = self.cold_fixture()
        cold[1]["report"]["groups"]["A2"]["candidates"][1]["cache_hit"] = True
        with self.assertRaisesRegex(ValueError, "cached|Cold"):
            self.analyzer().summarize_reports(self.plan, self.records, cold)
        cold = self.cold_fixture()
        cold[0]["report"]["rows"][0]["B_fingerprint"] = "changed"
        with self.assertRaisesRegex(ValueError, "model|Cold"):
            self.analyzer().summarize_reports(self.plan, self.records, cold)

    def test_partial_ols_audit_cannot_approve_correctness(self):
        evidence = {"ols_audit": {"checks_passed": True, "task_count": 99,
                                  "selected_model_count": 198, "rank_deficient_count": 0}}
        with self.assertRaisesRegex(ValueError, "OLS|ols|audit|100|200"):
            self.analyzer().summarize_reports(self.plan, self.records, audit_evidence=evidence)

    def test_historical_regression_checks_cannot_approve_current_cohort(self):
        evidence = {"regression_checks": {"checks_passed": True, "test_count": 131,
                                         "scope": "task_card_3", "completed_tasks": 90}}
        with self.assertRaisesRegex(ValueError, "regression|scope|cohort"):
            self.analyzer().summarize_reports(self.plan, self.records, audit_evidence=evidence)

    def test_d1_retains_production_without_a_new_approval_gate(self):
        result = self.analyzer().summarize_reports(self.plan, self.records)
        self.assertFalse(result["decision"]["final_user_approval_required"])


if __name__ == "__main__":
    unittest.main()
