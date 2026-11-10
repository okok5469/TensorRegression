"""Scheduling/ABI/path boundaries; no Slurm or numerical imports required."""
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile


def api(name):
    module = 'research.taskcard18.' + name
    assert importlib.util.find_spec(module) is not None, 'Missing ' + name
    return importlib.import_module(module)


class ClusterTests(unittest.TestCase):
    def test_incomplete_fit_pairs_and_unresolved_failure_stop_before_refitting(self):
        m = api('cluster')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            metadata, failure = base / 'model.json', base / 'failure.json'
            arrays = metadata.with_suffix('.npz')
            self.assertFalse(m.check_fit_pair(metadata, failure))
            arrays.write_bytes(b'preserved partial')
            with self.assertRaisesRegex(ValueError, 'Incomplete fit artifact pair'):
                m.check_fit_pair(metadata, failure)
            self.assertEqual(arrays.read_bytes(), b'preserved partial')
            metadata.write_bytes(b'complete pair requires later validation')
            self.assertTrue(m.check_fit_pair(metadata, failure))
            arrays.unlink(); metadata.unlink()
            failure.write_bytes(b'preserved numerical failure')
            with self.assertRaisesRegex(ValueError, 'Unresolved immutable fit failure'):
                m.check_fit_pair(metadata, failure)

    def test_prepare_test_attempt_logs_never_reuse_previous_evidence(self):
        m = api('cluster')
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder) / 'logs'
            first = m.next_test_log(logs)
            first.write_bytes(b'failed or interrupted test evidence')
            second = m.next_test_log(logs)
            self.assertNotEqual(first, second)
            self.assertFalse(second.exists())
            self.assertEqual(first.read_bytes(), b'failed or interrupted test evidence')

    def test_formal_mapping_deduplicates_complete_input_and_pilot_is_fixed(self):
        m = api('cluster')
        plan = {'tasks': [{'task_id': i, 'scenario_id': s, 'replicate': 0} for i, s in enumerate([0, 1, 6, 9])]}
        rows = [{'task_id': i, 'unique_input_id': uid} for i, uid in enumerate(['a', 'b', 'c', 'c'])]
        protocol = {'pilot_scenarios': [0, 6], 'pilot_replicates': [0]}
        self.assertEqual(m.phase_task_ids(plan, rows, 'formal', protocol), [0, 1, 2])
        self.assertEqual(m.phase_task_ids(plan, rows, 'pilot', protocol), [0, 2])

    def test_dependency_chain_has_bounded_arrays_and_no_shell_command_string(self):
        m = api('cluster')
        chain = m.submission_commands(Path('/payload/run_taskcard18_cluster.sh'), Path('/payload/logs'))
        self.assertEqual([item['phase'] for item in chain], ['prepare', 'pilot', 'pilotgate', 'formal', 'collect'])
        self.assertIn('--array=0-19%64', chain[1]['command'])
        self.assertIn('--array=0-599%64', chain[3]['command'])
        self.assertTrue(all(item['command'][0] == 'sbatch' for item in chain))
        self.assertTrue(all('--parsable' in item['command'] for item in chain))
        self.assertTrue(all('--chdir=' + str(Path('/payload')) in item['command'] for item in chain))
        self.assertIn('--dependency=afterok:{prepare}', chain[1]['command'])

    def test_duplicate_submit_active_resume_and_unowned_job_ids_are_rejected(self):
        m = api('cluster')
        with self.assertRaises(ValueError): m.check_submission_state(True, [], resume=False)
        with self.assertRaises(ValueError): m.check_submission_state(True, ['123'], resume=True)
        with self.assertRaises(ValueError): m.check_submission_state(False, [], resume=True)
        self.assertEqual(m.check_submission_state(True, [], resume=True), None)
        with self.assertRaises(ValueError): m.parse_job_id('123; scancel -u user')
        self.assertEqual(m.parse_job_id('123;cluster'), '123')

    def test_bootstrap_rejects_windows_wrong_python_and_unsafe_members(self):
        m = api('bootstrap_runtime')
        with self.assertRaises(ValueError): m.validate_target('win32', (3, 9), 'AMD64', 'cpython')
        with self.assertRaises(ValueError): m.validate_target('linux', (3, 12), 'x86_64', 'cpython')
        m.validate_target('linux', (3, 9), 'x86_64', 'cpython')
        for name in ['../escape.py', '/absolute.py', 'C:/drive.py', 'a/../../bad', 'a\\bad']:
            with self.subTest(name=name):
                with self.assertRaises(ValueError): m.safe_member(name)
        self.assertEqual(m.safe_member('sklearn/__init__.py').as_posix(), 'sklearn/__init__.py')

    def test_first_bootstrap_discovers_new_runtime_after_negative_path_cache(self):
        """Real extraction/import flow, with pure Python wheels and no numeric imports."""
        source = Path(__file__).with_name('bootstrap_runtime.py').resolve()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            runtime, existing = root / 'runtime', root / 'existing_environment'
            existing.mkdir()
            for name, version in [('numpy', '2.0.2'), ('scipy', '1.13.1'), ('sklearn', '0.0.old')]:
                package = existing / name
                package.mkdir()
                (package / '__init__.py').write_text(f'__version__ = {version!r}\n', encoding='utf-8')
            wheelhouse = root / 'wheelhouse'
            wheelhouse.mkdir()
            entries = []
            for distribution, module, version in [('scikit-learn', 'sklearn', '1.5.2'),
                                                   ('joblib', 'joblib', '1.4.2'),
                                                   ('threadpoolctl', 'threadpoolctl', '3.5.0')]:
                wheel = wheelhouse / f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
                with zipfile.ZipFile(wheel, 'w') as archive:
                    archive.writestr(module + '/__init__.py', f'__version__ = {version!r}\n')
                entries.append({'filename': wheel.name, 'distribution': distribution, 'version': version,
                                'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()})
            (wheelhouse / 'manifest.json').write_text(json.dumps({'schema_version': 1,
                'platform': 'linux_x86_64', 'python_tag': 'cp39', 'wheels': entries}), encoding='utf-8')
            code = textwrap.dedent('''\
                import importlib.util, pathlib, sys
                from unittest.mock import patch
                root, source = map(pathlib.Path, sys.argv[1:])
                runtime = root / 'runtime'
                assert not runtime.exists()
                assert str(runtime) in sys.path_importer_cache
                assert sys.path_importer_cache[str(runtime)] is None
                assert 'sklearn' not in sys.modules
                print('NEGATIVE_RUNTIME_CACHE_CONFIRMED', flush=True)
                spec = importlib.util.spec_from_file_location('real_bootstrap', source)
                bootstrap = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bootstrap)
                with patch.object(bootstrap, 'validate_target'), patch.object(bootstrap.sysconfig, 'get_config_var', return_value='cpython-39-x86_64-linux-gnu'):
                    bootstrap.bootstrap(root)
                import sklearn
                assert sklearn.__version__ == '1.5.2'
                assert runtime in pathlib.Path(sklearn.__file__).resolve().parents
                print('NEW_RUNTIME_SELECTED', flush=True)
                ''')
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(runtime), str(existing)]))
            environment.update({name: '1' for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                       'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
            result = subprocess.run([sys.executable, '-B', '-S', '-c', code, str(root), str(source)],
                                    env=environment, capture_output=True, text=True)
            evidence = result.stdout + result.stderr
            self.assertIn('NEGATIVE_RUNTIME_CACHE_CONFIRMED', evidence)
            self.assertEqual(result.returncode, 0, evidence)
            self.assertIn('NEW_RUNTIME_SELECTED', evidence)


if __name__ == '__main__': unittest.main(verbosity=2)
