"""Immutable new-shape validation plans and strict result resume tests."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class FinalValidationTests(unittest.TestCase):
    def runner(self):
        return importlib.import_module("numerical_simulation.a2_final_validation.run")

    def test_runner_cli_is_available(self):
        try:
            module = self.runner()
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "Independent final validation runner missing")

    def test_frozen_protocol_builds_100_new_shape_pairs(self):
        runner = self.runner()
        path = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        plan = runner.build_plan(path)
        self.assertEqual(len(plan["tasks"]), 100)
        self.assertEqual(plan["groups"], ["A0", "A2"])
        self.assertEqual(plan["tasks"][0]["request"]["seed"], 30501011)
        self.assertEqual(plan["tasks"][99]["request"]["test_seed"], 31520015)
        self.assertEqual({tuple(task["request"]["config"]["sparsity"]) for task in plan["tasks"]}, {(2,3,4),(3,4,5)})
        self.assertEqual(plan["protocol_sha256"], runner.file_hash(path))
        self.assertEqual(plan["protocol_id"], runner.digest(plan["protocol"]))

    def test_plan_refuses_overwrite_and_changed_protocol(self):
        runner = self.runner()
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "protocol.json").write_bytes(source.read_bytes())
            (output / "PROTOCOL.sha256").write_text(runner.file_hash(source))
            runner.create_plan(output)
            self.assertEqual(len(runner.load_plan(output)["tasks"]),100)
            with self.assertRaises(FileExistsError):
                runner.create_plan(output)
            protocol = json.loads((output / "protocol.json").read_text())
            protocol["seed_base"] += 1
            (output / "protocol.json").write_text(json.dumps(protocol),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"[Pp]rotocol"):
                runner.load_plan(output)

    def test_unknown_job_and_partial_cache_are_rejected_before_resume(self):
        runner = self.runner()
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "protocol.json").write_bytes(source.read_bytes())
            runner.create_plan(output)
            (output / "jobs/task_999.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Unknown"):
                runner.run_plan(output, task_ids=[0])
            (output / "jobs/task_999.json").unlink()
            (output / "jobs/task_000.npz").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                runner.run_plan(output, task_ids=[0])

    def test_unknown_request_and_partial_cold_cache_are_rejected(self):
        runner = self.runner()
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "protocol.json").write_bytes(source.read_bytes())
            runner.create_plan(output)
            rogue = output / "requests/task_999.json"
            rogue.write_text("{}")
            with self.assertRaisesRegex(ValueError,"Unknown"):
                runner.load_plan(output)
            rogue.unlink()
            (output / "cold/task_000_A0.npz").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                runner.load_plan(output)

    def test_time_budget_exhaustion_is_persisted_and_blocks_resume(self):
        runner = self.runner()
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            protocol = runner.read_json(source)
            protocol["budget"]["max_controller_seconds"] = -1
            (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            runner.create_plan(output)
            with patch.object(runner, "execute_task", return_value=None):
                with self.assertRaisesRegex(ValueError,"time budget"):
                    runner.run_plan(output, task_ids=[0])
            self.assertTrue(any((output / "failures").iterdir()), "Exhausted budget must prohibit resume")
            with self.assertRaisesRegex(ValueError,"Recorded failure"):
                runner.run_plan(output, task_ids=[0])

    def test_real_small_worker_pool_reaudit_rejects_changed_accepted_loss(self):
        runner = self.runner()
        self.assertTrue(hasattr(runner,"audit_candidates"),"Persisted pool audit missing")
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        protocol = runner.read_json(source)
        config = protocol["base_config"]
        config.update(n_train=20,n_test=10,dims=[4,3,3],sparsity=[1,1,1],n_lambdas=2,max_cycles=2,max_iter=25,test_batch_size=10)
        protocol.update(n_reps=1,cold_check_tasks=[0],scenarios=[{"n_train":20,"snr":2,"rho":[.3,.3,.3],"sparsity":[1,1,1],"stratum":"primary"}])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            runner.write_json(output / "protocol.json",protocol)
            plan = runner.create_plan(output)
            runner.run_plan(output,workers=1)
            runner.cold_plan(output)
            for group in ("A0","A2"):
                self.assertEqual(runner.validate_cold_result(output,plan,plan["tasks"][0],group)["groups"][group]["cache_hits"],0)
            report = runner.read_json(output / "jobs/task_000.json")
            runner.audit_candidates(report, output / "jobs/task_000.npz")
            fixed = next(c for c in report["groups"]["A2"]["candidates"] if c["kind"]=="global-fixed")
            fixed["loss_history"] = [1.,2.]
            with self.assertRaisesRegex(ValueError,"loss increased"):
                runner.audit_candidates(report, output / "jobs/task_000.npz")

    def test_cold_cli_and_incomplete_cold_validation_are_available(self):
        runner = self.runner()
        self.assertTrue(hasattr(runner,"cold_plan"),"Independent single-group cold runner missing")
        source = runner.root() / "numerical_simulation/outputs_a2_final_validation/protocol.json"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "protocol.json").write_bytes(source.read_bytes())
            plan = runner.create_plan(output)
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                runner.validate_cold_result(output, plan, plan["tasks"][0], "A0")


if __name__ == "__main__":
    unittest.main()
