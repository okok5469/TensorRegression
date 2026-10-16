"""Safety checks for the explicit, hash-locked release cleanup plan."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / 'scripts/release_v52/audit_cleanup.py'


class CleanupSafetyTests(unittest.TestCase):
    def api(self):
        self.assertTrue(MODULE.exists(), 'release cleanup API is missing')
        spec = importlib.util.spec_from_file_location('audit_cleanup', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cache_classification_preserves_history_dependencies_and_tracked_files(self):
        api = self.api()
        self.assertEqual(api.classify('src/__pycache__/a.pyc', False, set())[0], 'DELETE')
        for name in ['doc/algorithm_versions/v5-original/__pycache__/a.pyc',
                     '.venv/Lib/__pycache__/a.pyc',
                     'numerical_simulation/outputs_old/__pycache__/a.pyc',
                     'empirical_analysis/__pycache__/a.pyc']:
            self.assertEqual(api.classify(name, False, set())[0], 'KEEP')
        self.assertEqual(api.classify('src/__pycache__/a.pyc', True, set())[0], 'KEEP')
        self.assertEqual(api.classify('src/__pycache__/a.pyc', False,
                                     {'src/__pycache__/a.pyc'})[0], 'KEEP')

    def test_confirmed_stopped_study_is_indexed_in_place(self):
        api = self.api()
        self.assertEqual(api.classify(
            'numerical_simulation/outputs_a2_msic_confirmation/plan.json',
            False, set())[0], 'ARCHIVE')

    def test_apply_deletes_only_explicit_hash_locked_cache(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'src/__pycache__/a.pyc'
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'regenerated')
            keep = root / 'src/a.py'
            keep.write_text('source', encoding='utf-8')
            plan = [{'path': 'src/__pycache__/a.pyc', 'sha256': api.sha256(cache),
                     'size_bytes': cache.stat().st_size}]
            result = api.apply_plan(root, plan, set(), set())
            self.assertEqual(len(result), 1)
            self.assertFalse(cache.exists())
            self.assertEqual(keep.read_text(encoding='utf-8'), 'source')

    def test_changed_plan_or_outside_target_stops_before_any_deletion(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'src/__pycache__/a.pyc'
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'new')
            for plan in [[{'path': '../outside.pyc', 'sha256': 'x', 'size_bytes': 1}],
                         [{'path': 'src/__pycache__/a.pyc', 'sha256': 'x', 'size_bytes': 3}],
                         [{'path': 'src/a.py', 'sha256': 'x', 'size_bytes': 1}]]:
                with self.assertRaises(ValueError):
                    api.apply_plan(root, plan, set(), set())
                self.assertTrue(cache.exists())

    def test_path_alias_cannot_bypass_history_protection(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src').mkdir()
            cache = root / 'doc/algorithm_versions/v5-original/__pycache__/a.pyc'
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'history')
            plan = [{'path': 'src/../doc/algorithm_versions/v5-original/__pycache__/a.pyc',
                     'sha256': api.sha256(cache), 'size_bytes': cache.stat().st_size}]
            with self.assertRaises(ValueError):
                api.apply_plan(root, plan, set(), set())
            self.assertTrue(cache.exists())

    def test_entire_plan_is_checked_before_first_deletion(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'src/__pycache__/a.pyc'
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'cache')
            valid = {'path': 'src/__pycache__/a.pyc', 'sha256': api.sha256(cache),
                     'size_bytes': cache.stat().st_size}
            invalid = dict(valid, path='src/__pycache__/absent.pyc')
            with self.assertRaises(ValueError):
                api.apply_plan(root, [valid, invalid], set(), set())
            self.assertTrue(cache.exists())

    def test_resolved_parent_alias_cannot_delete_protected_history(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            history = root / 'doc/algorithm_versions/v5-original/__pycache__/a.pyc'
            history.parent.mkdir(parents=True)
            history.write_bytes(b'history')
            alias = root / 'src/__pycache__/a.pyc'
            original = Path.resolve
            def redirected(path, *args, **kwargs):
                return history if path == alias else original(path, *args, **kwargs)
            plan = [{'path': 'src/__pycache__/a.pyc', 'sha256': api.sha256(history),
                     'size_bytes': history.stat().st_size}]
            with patch.object(Path, 'resolve', autospec=True, side_effect=redirected):
                with self.assertRaises(ValueError):
                    api.apply_plan(root, plan, set(), set())
            self.assertTrue(history.exists())


if __name__ == '__main__':
    unittest.main()
