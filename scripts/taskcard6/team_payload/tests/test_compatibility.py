import importlib.util
from pathlib import Path
import unittest


class CompatibilityTests(unittest.TestCase):
    def test_complete_version_and_actual_import(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('check_version', root / 'scripts/check_version.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.check(root, True)['verified_core_files'], 32)

    def test_public_fixed_and_adaptive_paths(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('smoke_test', root / 'examples/smoke_test.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(set(module.smoke()), {'fixed', 'adaptive'})


if __name__ == '__main__':
    unittest.main()
