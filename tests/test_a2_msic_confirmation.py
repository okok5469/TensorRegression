"""Immutable eighty-dataset MSIC confirmation and predeclared acceptance gates."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import os


class MsicConfirmationTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("numerical_simulation.a2_msic_decision.confirmation")
        except ModuleNotFoundError:
            self.fail("Independent MSIC confirmation controller is not implemented")

    def test_frozen_eighty_scenarios_and_fresh_seeds(self):
        module = self.module()
        scenarios = module.expected_scenarios()
        self.assertEqual(len(scenarios), 16)
        self.assertEqual(sum(s["stratum"] == "primary" for s in scenarios), 14)
        self.assertEqual(sum(s["stratum"] == "correlation_stress" for s in scenarios), 2)
        tasks = module.build_tasks({"scenarios": scenarios, "n_reps": 5, "seed_base": 30601011,
                                   "base_config": {"dims": [32, 20, 25]}, "budget": {"candidate_budget": 2000}})
        self.assertEqual(len(tasks), 80)
        self.assertEqual(len({t["request"]["seed"] for t in tasks}), 80)
        self.assertEqual(tasks[0]["request"]["seed"], 30601011)
        self.assertTrue(all(t["request"]["groups"] == ["A2"] for t in tasks))

    def test_missing_dataset_and_duplicate_task_cannot_be_summarized(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, "80|complete"):
            module.validate_complete_records({"tasks": list(range(80))}, [])
        with self.assertRaisesRegex(ValueError, "Duplicate|duplicate"):
            module.validate_complete_records({"tasks": list(range(80))}, [{"task_id": 0}] * 80)

    def test_predeclared_gate_rejects_prediction_or_strong_underselection(self):
        module = self.module()
        gates = {"pooled_pmse_ratio_max": .99, "paired_stratified_bootstrap_pmse_ratio_95_upper_max": 1.,
                 "any_scenario_pmse_ratio_max": 1.1, "strong_representable_exact_gain_min": -.05,
                 "strong_representable_mode_f1_gain_min": -.02,
                 "strong_representable_under_rate_increase_max": .05,
                 "strong_representable_q1_rate_increase_max": .05}
        metrics = {"pooled_pmse_ratio": .97, "bootstrap_upper": .99, "worst_scenario_ratio": 1.,
                   "strong_exact_gain": 0., "strong_mode_f1_gain": 0., "strong_under_increase": 0., "strong_q1_increase": 0.}
        self.assertTrue(module.decision_gates(metrics, gates)["statistical_gates_pass"])
        metrics["pooled_pmse_ratio"] = 1.01
        self.assertFalse(module.decision_gates(metrics, gates)["statistical_gates_pass"])
        metrics["pooled_pmse_ratio"], metrics["strong_under_increase"] = .97, .1
        self.assertFalse(module.decision_gates(metrics, gates)["statistical_gates_pass"])

    def test_existing_plan_or_failure_is_not_overwritten_or_resumed(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            output = Path(directory)
            (output / "plan.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.create_plan(output)
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            output = Path(directory)
            (output / "failures").mkdir()
            (output / "failures/stopped.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stopped|failure|Stopped|failure"):
                module.reject_failures(output)

    def test_file_pair_inventory_rejects_half_results_and_unknown_files(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            folder = Path(directory)
            (folder / "task_000.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete|pair"):
                module.validate_pair_inventory(folder, {"task_000"})
            (folder / "task_000.npz").write_bytes(b"dummy")
            module.validate_pair_inventory(folder, {"task_000"})
            (folder / "unknown.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown|unknown"):
                module.validate_pair_inventory(folder, {"task_000"})

    def test_one_persistent_budget_covers_every_stage(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            output = Path(directory)
            plan = {"plan_id": "fixed", "protocol": {"budget": {"max_controller_seconds": 3600}}}
            module.deadline(output, plan)
            before = (output / "CONTROLLER_STARTED.json").read_bytes()
            module.deadline(output, plan)
            self.assertEqual(before, (output / "CONTROLLER_STARTED.json").read_bytes())

    def test_fresh_process_small_worker_two_rules_and_independent_ols_audit(self):
        self.module()
        root = Path(__file__).resolve().parents[1]
        code = '''
import json,tempfile
from dataclasses import asdict
from pathlib import Path
from numerical_simulation.a2_msic_decision.confirmation import execute_task,sources,scoring
scoring.configure_core()
from numerical_simulation.candidate_search_study import run as legacy
from numerical_simulation.data_generation import TensorSimulationConfig
config=json.loads(json.dumps(asdict(TensorSimulationConfig(n_train=40,n_test=8,dims=(3,3,3),sparsity=(1,1,1),max_cycles=1,n_lambdas=3,cv_folds=2,test_batch_size=4))))
request={'config':config,'seed':87631,'test_seed':1087631,'groups':['A2'],'fit_options':{'candidate_budget':2000}}
task={'task_id':0,'request':request,'request_file':'requests/task_000.json','request_id':legacy.digest(request)}
plan={'version':'v5.2-candidate-search','groups':['A2'],'environment':legacy.environment(),'sources':sources(),'plan_id':'tiny','rules':scoring.declared_rules()}
with tempfile.TemporaryDirectory(dir=Path('tmp')) as folder:
 output=Path(folder)
 for name in ['requests','jobs','evaluations','logs','failures']: (output/name).mkdir()
 legacy.write_json(output/task['request_file'],request)
 result=execute_task(output,plan,task)
 assert len(result['rows'])==2 and len(result['selected_model_audits'])==2
 assert all(a['checks_passed'] and not a['rank_deficient'] for a in result['selected_model_audits'])
 assert result['default_exact_replay']
print('SMALL_CONFIRMATION_PIPELINE_OK')
'''
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SMALL_CONFIRMATION_PIPELINE_OK", result.stdout)

    def test_summary_validator_configures_isolated_core_before_score_import(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        code = '''
from pathlib import Path
import sys
from numerical_simulation.a2_msic_decision.confirmation import validate_evaluation
try:
 validate_evaluation(Path('tmp/nonexistent'),{}, {})
except (FileNotFoundError, KeyError):
 pass
assert str(Path('doc/algorithm_versions/v5.2-candidate-search/src').resolve()) in sys.path
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_protocol_binds_screen_hash_and_supports_windows_hash_filename(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        original = root / "numerical_simulation/outputs_a2_msic_confirmation"
        plan = module.build_plan(original)
        self.assertIn("DEVELOPMENT_SCREEN_RESULT.json", plan["controlled_development_parents"])
        self.assertEqual(plan["controlled_development_parents"]["DEVELOPMENT_SCREEN_RESULT.json"],
                         plan["protocol"]["development_screen_sha256"])
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            output = Path(directory)
            (output / "protocol.json").write_bytes((original / "protocol.json").read_bytes())
            (output / "PROTOCOL.sha256").write_bytes((original / "PROTOCOL.sha256").read_bytes())
            created = module.create_plan(output)
            self.assertEqual(len(created["tasks"]), 80)


if __name__ == "__main__":
    unittest.main()
