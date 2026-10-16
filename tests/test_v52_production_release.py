"""The approved D2 source must be the actual production import, with its identity intact."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProductionReleaseTests(unittest.TestCase):
    def test_all_production_sources_match_approved_frozen_d2(self):
        version = json.loads((ROOT / "doc/algorithm_versions/v5.2-search-final/VERSION.json").read_text(encoding="utf-8"))
        production = ROOT / "src"
        actual = {p.relative_to(production).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (production / "tensor_abess").rglob("*.py")}
        self.assertEqual(len(actual), 32)
        self.assertEqual(actual, version["source_sha256"])

    def test_production_import_uses_full_legacy_cap_grid(self):
        bcd = importlib.import_module("tensor_abess.tensor.bcd")
        self.assertTrue(Path(bcd.__file__).resolve().is_relative_to(ROOT / "src"))
        grid = bcd._candidate_size_triples(6, 6, 7, 280)
        self.assertIn((3, 3, 3), grid)
        self.assertEqual(len(grid), 252)
        self.assertEqual(len(grid), len(set(grid)))
        self.assertEqual(bcd._candidate_size_triples(1, 1, 1, 0), [])

    def test_original_public_msic_defaults_are_retained(self):
        function = importlib.import_module("tensor_abess.tensor.abess").tensor_abess
        signature = inspect.signature(function)
        self.assertEqual(signature.parameters["msic_refine_alpha"].default, 1.5)
        self.assertEqual(signature.parameters["msic_refine_gamma"].default, 1.0)
        self.assertEqual(signature.parameters["msic_refine_one_se_scale"].default, 1.0)
        self.assertNotIn("grid_policy", signature.parameters)
        self.assertNotIn("cap_policy", signature.parameters)


if __name__ == "__main__":
    unittest.main()
