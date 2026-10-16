"""Verify complete-source delivery and fail closed on tampered packages."""
import importlib
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
import zipfile


class TeamPackageTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('scripts.taskcard6.team_package')
        except ModuleNotFoundError:
            self.fail('Complete team package builder is missing')

    def test_both_deliveries_are_identical_complete_archives(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            paths = api.build(Path(directory))
            self.assertEqual({p.name for p in paths},
                             {'Tensor-ABESS-v5.2-HCP.zip', 'Tensor-ABESS-v5.2-NKI.zip'})
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            with zipfile.ZipFile(paths[0]) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertEqual(sum('/src/tensor_abess/' in n and n.endswith('.py') for n in names), 32)
                self.assertTrue(any(n.endswith('/SHA256SUMS.txt') for n in names))
                self.assertFalse(any('__pycache__' in n or '/outputs' in n or '.venv' in n for n in names))

    def test_builder_refuses_existing_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            self.api().build(target)
            with self.assertRaises(FileExistsError):
                self.api().build(target)

    def test_validation_rejects_changed_source(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            paths = api.build(Path(directory) / 'packages')
            extracted = api.extract_checked(paths[0], Path(directory) / 'extracted')
            core = extracted / 'src/tensor_abess/tensor/bcd.py'
            core.write_bytes(core.read_bytes() + b'\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.verify_extracted(extracted)

    def test_missing_core_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            paths = api.build(Path(directory) / 'packages')
            extracted = api.extract_checked(paths[0], Path(directory) / 'extracted')
            (extracted / 'src/tensor_abess/tensor/bcd.py').unlink()
            with self.assertRaisesRegex(ValueError, 'core|checksum'):
                api.verify_extracted(extracted)

    def test_documented_target_source_check_needs_no_project_manifest(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            paths = api.build(target / 'packages')
            extracted = api.extract_checked(paths[0], target / 'extracted')
            project_core = target / 'project/src/tensor_abess'
            shutil.copytree(extracted / 'src/tensor_abess', project_core)
            result = subprocess.run([sys.executable, '-I', '-B',
                                     str(extracted / 'scripts/check_version.py'),
                                     '--source', str(project_core), '--import-check'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
