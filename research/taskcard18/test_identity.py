"""Identity boundaries that protect historical simulation pairing."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


class IdentityTests(unittest.TestCase):
    def identity_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("research.taskcard18.identity"),
            "Task18 needs a reproducible identity audit module",
        )
        from research.taskcard18 import identity

        return identity

    @staticmethod
    def record(task_id, scenario, seed=123, test_seed=1000123):
        return {
            "task": {
                "task_id": task_id, "scenario_id": scenario, "replicate": 0,
                "seed": seed, "test_seed": test_seed,
                "experiment": "baseline", "scenario_value": "baseline",
                "config": {"n_train": 200, "dims": [32, 20, 25], "random_state": 2026},
            },
            "input_fingerprints": {key: key + "_sha" for key in
                                   ("X_train", "y_train", "B_true", "train_noise")},
        }

    def test_aliases_share_complete_input_but_different_test_seed_does_not(self):
        identity = self.identity_module()
        records = [self.record(300, 6), self.record(450, 9), self.record(600, 12),
                   self.record(0, 0, test_seed=1000999)]
        rows = identity.build_input_map(records)
        self.assertEqual(len({row["unique_input_id"] for row in rows}), 2)
        aliases = [row for row in rows if row["task_id"] in {300, 450, 600}]
        self.assertEqual({row["canonical_task_id"] for row in aliases}, {300})
        self.assertEqual({row["alias_scenario_ids"] for row in aliases}, {"6|9|12"})

    def test_equal_x_with_different_y_is_a_different_input(self):
        identity = self.identity_module()
        first, second = self.record(0, 0), self.record(50, 1)
        second["input_fingerprints"]["y_train"] = "different_response"
        rows = identity.build_input_map([first, second])
        self.assertEqual(len({row["unique_input_id"] for row in rows}), 2)

    def test_conflicting_alias_configuration_stops(self):
        identity = self.identity_module()
        first, second = self.record(0, 0), deepcopy(self.record(50, 1))
        second["task"]["config"]["n_train"] = 400
        with self.assertRaisesRegex(ValueError, "configuration"):
            identity.build_input_map([first, second])

    def test_source_fingerprint_mismatch_stops_without_rewriting_source(self):
        identity = self.identity_module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "frozen.py"
            source.write_bytes(b"frozen\r\n")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(identity.verify_hashes(root, {"frozen.py": expected}),
                             {"frozen.py": expected})
            source.write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                identity.verify_hashes(root, {"frozen.py": expected})
            self.assertEqual(source.read_bytes(), b"changed\n")


if __name__ == "__main__":
    unittest.main()
