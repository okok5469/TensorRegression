"""Transport integrity and extraction boundaries for the offline HPC bundle."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


class BundleTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research.taskcard18.build_cluster_bundle'),
                             'Missing offline bundle builder')
        from research.taskcard18 import build_cluster_bundle
        return build_cluster_bundle

    def test_bundle_roundtrip_preserves_raw_bytes_and_manifest(self):
        builder = self.module()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            self.assertTrue(base.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            source = base/'source.py'; source.write_bytes(b'answer = 42\r\n')
            archive = base/'bundle.zip'
            builder.write_archive({'src/source.py': source}, archive, 'task18')
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.getinfo('task18/src/source.py').create_system, 3)
            extracted = builder.safe_extract(archive, base/'check', 'task18')
            self.assertEqual((extracted/'src/source.py').read_bytes(), source.read_bytes())
            self.assertEqual(builder.verify_payload(extracted)['file_count'], 1)
            self.assertFalse((extracted/'runtime').exists())
            self.assertTrue((extracted/'numerical_simulation/outputs_taskcard18_closeout').is_dir())

    def test_tampered_payload_is_rejected(self):
        builder = self.module()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            self.assertTrue(base.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            source = base/'source.py'; source.write_bytes(b'answer = 42\n')
            archive = base/'bundle.zip'
            builder.write_archive({'src/source.py': source}, archive, 'task18')
            extracted = builder.safe_extract(archive, base/'check', 'task18')
            (extracted/'src/source.py').write_bytes(b'answer = 0\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                builder.verify_payload(extracted)

    def test_zip_traversal_rejected_before_any_extraction(self):
        builder = self.module()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            self.assertTrue(base.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            archive = base/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('task18/good.txt', 'good')
                handle.writestr('task18/../escape.txt', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                builder.safe_extract(archive, base/'check', 'task18')
            self.assertFalse((base/'check/task18/good.txt').exists())

    def test_nonportable_payload_paths_are_rejected(self):
        builder = self.module()
        for name in ('.git/config', '.venv/pyvenv.cfg', 'runtime/sklearn.pyd',
                     'numerical_simulation/outputs_taskcard18_closeout/fits/model.npz',
                     '../outside.py', 'C:/outside.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.validate_payload_path(name)


if __name__ == '__main__':
    unittest.main()
