import importlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class DeliveryTests(unittest.TestCase):
    def api(self):
        self.assertTrue((Path(__file__).parent / 'delivery.py').is_file(), 'Delivery implementation missing')
        return importlib.import_module('research.taskcard12.delivery')

    def fixture(self, root):
        (root / 'src').mkdir()
        (root / 'src/model.py').write_text('value = 1\n', encoding='utf-8')
        (root / 'protocol.json').write_text('{"research_only":true}\n', encoding='utf-8')
        return ['src/model.py', 'protocol.json']

    def test_zip_has_new_prefix_and_exact_whitelist_only(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = self.fixture(root)
            (root / 'private-token.txt').write_text('not an upload input', encoding='utf-8')
            archive = root / 'bundle.zip'
            api.write_bundle(root, paths, archive)
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                self.assertEqual(len(names), 4)
                self.assertTrue(all(n.startswith('taskcard12_bundle/') for n in names))
                self.assertFalse(any('private-token' in n for n in names))
                z.extractall(root / 'unpacked')
            actual = api.verify_bundle(root / 'unpacked/taskcard12_bundle')
            self.assertTrue(actual['passed'])
            self.assertEqual(actual['files_checked'], 2)

    def test_changed_or_missing_payload_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = self.fixture(root)
            api.write_bundle(root, paths, root / 'bundle.zip')
            with zipfile.ZipFile(root / 'bundle.zip') as z:z.extractall(root / 'unpacked')
            bundle = root / 'unpacked/taskcard12_bundle'
            (bundle / 'src/model.py').write_text('value = 2\n', encoding='utf-8')
            with self.assertRaises(ValueError):api.verify_bundle(bundle)
            (bundle / 'src/model.py').unlink()
            with self.assertRaises(ValueError):api.verify_bundle(bundle)

    def test_unsafe_paths_and_existing_zip_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = self.fixture(root)
            for path in ('../outside', '/absolute', 'C:/private', 'src\\model.py'):
                with self.assertRaises(ValueError):api.write_bundle(root, [path], root / 'bad.zip')
                self.assertFalse((root / 'bad.zip').exists())
            api.write_bundle(root, paths, root / 'bundle.zip')
            with self.assertRaises(FileExistsError):api.write_bundle(root, paths, root / 'bundle.zip')

    def test_same_payload_produces_same_archive_hash(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths = self.fixture(root)
            first = api.write_bundle(root, paths, root / 'a.zip')
            second = api.write_bundle(root, list(reversed(paths)), root / 'b.zip')
            self.assertEqual(first['sha256'], second['sha256'])


if __name__ == '__main__':unittest.main()
