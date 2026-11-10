import os
from pathlib import Path
import subprocess
import sys
import unittest
import importlib.util
import hashlib
import json
import tempfile
import zipfile


class DeliveryTests(unittest.TestCase):
    def acceptance_module(self):
        path = Path(__file__).resolve().parents[1]/'accept_package.py'
        spec = importlib.util.spec_from_file_location('task19_acceptance',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_archive_verifier_rejects_unlisted_payload(self):
        module=self.acceptance_module()
        self.assertTrue(hasattr(module,'verify_archive'),'archive verifier missing')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'delivery.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('RELEASE_MANIFEST.json',json.dumps({'archive_files':{}}))
                archive.writestr('CORE_SOURCE_NOTICE.md','notice')
                archive.writestr('unlisted.py','bad')
            path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  delivery.zip\n')
            with self.assertRaisesRegex(ValueError,'inventory'):
                module.verify_archive(path)

    def test_import_verifier_rejects_repository_shadowing(self):
        module=self.acceptance_module()
        self.assertTrue(hasattr(module,'assert_installed_paths'),'installed path assertion missing')
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            identity={'runtime':{'actual_import_files':{'tensor_abess':str(base/'repo/core/__init__.py'),'tensor_abess_ovr':str(base/'venv/site/application.py')}}}
            with self.assertRaisesRegex(ValueError,'installation'):
                module.assert_installed_paths(identity,base/'venv')

    def test_cli_lists_all_contract_commands(self):
        result = subprocess.run([sys.executable, '-m', 'tensor_abess_ovr.cli', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('verify-release', 'smoke', 'validate-data', 'run', 'validate-return', 'pack-return'):
            self.assertIn(name, result.stdout)

    def test_cli_rejects_unknown_command(self):
        result = subprocess.run([sys.executable, '-m', 'tensor_abess_ovr.cli', 'silently-score-test'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == '__main__':
    unittest.main()
