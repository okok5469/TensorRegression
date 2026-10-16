"""Two-rule A2 development rescoring boundaries; no real cohort evaluation."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class A2MsicDecisionTests(unittest.TestCase):
    def scorer(self):
        try:
            return importlib.import_module("numerical_simulation.a2_msic_decision.run")
        except ModuleNotFoundError:
            self.fail("A2 controlled two-rule rescoring tool is not implemented")

    def test_only_two_predeclared_rules_and_training_only_selection(self):
        scorer = self.scorer()
        scorer.configure_core()
        rules = scorer.declared_rules()
        self.assertEqual([(r["alpha"], r["scale"], r["gamma"]) for r in rules], [(1.5, 1., 1.), (2., 0., 1.)])
        metadata = [{"sizes": [1, 1, 1], "rss": 100., "pmse": 1e-30},
                    {"sizes": [2, 2, 2], "rss": 20., "pmse": 1e30}]
        before = copy.deepcopy(metadata)
        choices = scorer.two_choices(metadata, 40, (4, 4, 4))
        self.assertEqual(metadata, before)
        changed = copy.deepcopy(metadata)
        changed[0]["pmse"], changed[1]["pmse"] = 1e30, 1e-30
        self.assertEqual([c[1] for c in choices], [c[1] for c in scorer.two_choices(changed, 40, (4, 4, 4))])

    def test_confirmation_or_new_shape_cohort_is_rejected(self):
        scorer = self.scorer()
        for cohort in ("confirmation", "new_shape_final_validation", "historical"):
            with self.assertRaisesRegex(ValueError, "development"):
                scorer.validate_plan_contract({"cohort": cohort})

    def test_incomplete_development_or_changed_plan_checksum_is_rejected(self):
        scorer = self.scorer()
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "numerical_simulation/outputs_candidate_search_study/development_final/plan.json").read_text(encoding="utf-8"))
        scorer.validate_plan_contract(plan)
        incomplete = copy.deepcopy(plan)
        incomplete["tasks"].pop()
        incomplete["plan_id"] = scorer.digest({k: v for k, v in incomplete.items() if k != "plan_id"})
        with self.assertRaisesRegex(ValueError, "90|complete"):
            scorer.validate_plan_contract(incomplete)
        plan["plan_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            scorer.validate_plan_contract(plan)

    def test_default_replay_refuses_any_metric_change(self):
        scorer = self.scorer()
        original = {"pmse": 2., "q_hat": 1, "runtime_sec": 10.}
        scorer.default_replay(original, dict(original, runtime_sec=20.))
        with self.assertRaisesRegex(ValueError, "Default replay"):
            scorer.default_replay(original, dict(original, pmse=2.000001))

    def synthetic_task(self, scorer, directory):
        from dataclasses import asdict
        from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
        from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
        from tensor_abess.tensor.ops import tensor_loss
        config = TensorSimulationConfig(n_train=40, n_test=8, dims=(3, 3, 3), sparsity=(1, 1, 1),
                                        test_batch_size=4, snr=2., max_cycles=1)
        seed, test_seed = 90731, 1090731
        data = generate_simulation_dataset(config, seed=seed)
        supports = data.supports_true
        arrays, candidates = {"B_true": data.B_true}, []
        for index, B in enumerate((np.zeros_like(data.B_true), data.B_true.copy())):
            loss = tensor_loss(data.X_train, data.y_train, B)
            record = {"candidate_index": index, "kind": "adaptive-stage" if index == 0 else "global-fixed",
                "sizes": [1, 1, 1], "supports": [S.tolist() for S in supports],
                "rss": 2 * config.n_train * loss, "loss": loss,
                "B_fingerprint": scorer.array_fingerprint(B), "loss_history": [loss],
                "visited_loss_history": [loss], "best_iteration": 0}
            candidates.append(record)
            arrays[f"A2_candidate_{index}_B"] = B
            for mode, S in enumerate(supports, 1):
                arrays[f"A2_candidate_{index}_S{mode}"] = S
        cache = Path(directory) / "synthetic.npz"
        np.savez_compressed(cache, **arrays)
        group = {"candidates": candidates, "candidate_count": 2}
        group["pool_fingerprint"] = scorer.pool_fingerprint(scorer.stream_pool(group, arrays))
        selected, info = scorer.select_candidate(candidates, config.n_train, config.dims, alpha=1.5, scale=1.)
        B = arrays[f"A2_candidate_{info['selected_index']}_B"]
        predictions, test_fingerprint = scorer.evaluate_shared_test(data, [B], test_seed)
        row = {"group": "A2", "rule": "msic-alpha1.5-scale1", "method": "A2/msic-alpha1.5-scale1",
            "n_train": config.n_train, "snr": config.snr, "seed": seed, "test_seed": test_seed,
            "q_true": 1, **compute_support_metrics(*supports, *supports),
            **compute_coefficient_metrics(B, data.B_true), **predictions[0],
            "rss_train": selected["rss"], "loss_train": selected["loss"], "runtime_sec": 0.1,
            "selection": info, "supports": [S.tolist() for S in supports],
            "B_fingerprint": scorer.array_fingerprint(B), "candidate_count": 2,
            "true_size_in_pool": 1, "true_support_in_pool": 1}
        fingerprints = {"X_train": scorer.array_fingerprint(data.X_train),
            "y_train": scorer.array_fingerprint(data.y_train), "B_true": scorer.array_fingerprint(data.B_true),
            "independent_test": test_fingerprint}
        report = {"config": json.loads(json.dumps(asdict(config))), "seed": seed, "test_seed": test_seed,
                  "groups": {"A2": group}, "rows": [row], "fingerprints": fingerprints, "environment": {}}
        return report, cache

    def test_synthetic_default_replay_and_pool_are_exact_without_fitting(self):
        scorer = self.scorer()
        scorer.configure_core()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            report, cache = self.synthetic_task(scorer, directory)
            before = scorer.file_hash(cache)
            result, selected = scorer.evaluate_task(report, cache)
            self.assertEqual(scorer.file_hash(cache), before)
            self.assertEqual(len(result["rows"]), 2)
            self.assertTrue(result["default_exact_replay"])
            self.assertEqual(result["pool_fingerprint_before"], result["pool_fingerprint_after"])
            self.assertFalse(result["new_model_fitting"])
            self.assertFalse(result["rule_selection_performed"])
            self.assertFalse(result["confirmation_performed"])
            self.assertEqual(len(selected), 8)

    def test_changed_candidate_array_is_rejected(self):
        scorer = self.scorer()
        scorer.configure_core()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            report, cache = self.synthetic_task(scorer, directory)
            report["groups"]["A2"]["candidates"][0]["B_fingerprint"] = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprint|Candidate"):
                scorer.evaluate_task(report, cache)

    def test_existing_output_is_never_overwritten(self):
        scorer = self.scorer()
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            with self.assertRaises(FileExistsError):
                scorer.run_cohort(root / "no-such-input", Path(directory))


if __name__ == "__main__":
    unittest.main()
