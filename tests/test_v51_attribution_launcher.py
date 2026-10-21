"""Exercise the single-file launcher without fitting or contacting SLURM."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'run_v51_attribution.sh'


class LauncherTests(unittest.TestCase):
    def api(self):
        self.assertTrue(LAUNCHER.is_file(), 'The single-file cluster launcher is missing')
        shell = LAUNCHER.read_text(encoding='utf-8')
        code = shell.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        ast.parse(code, feature_version=(3, 9))
        namespace = {'__name__': 'launcher_test'}
        exec(compile(code, str(LAUNCHER), 'exec'), namespace)
        return namespace

    def runner(self):
        spec = importlib.util.spec_from_file_location('attribution_test_api',
            ROOT / 'numerical_simulation/version_attribution/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, directory, task_id=0):
        r = self.runner()
        task = {'task_id': task_id, 'expected_input_fingerprints': {'X_train': 'x'}}
        plan = {'sha256': 'plan', 'fit_version': 'v5.1-bcd-fix',
                'source_manifests': {'v5.1-bcd-fix': {'core': 'hash'}},
                'expected_environment': {'python': '3.9.25'},
                'gate_task_ids': [0, 50, 250, 350, 550], 'tasks': [task]}
        out = Path(directory)
        npz = out / 'tasks' / f'task_{task_id:06d}.npz'
        npz.parent.mkdir(parents=True, exist_ok=True)
        npz.write_bytes(b'archived-test-artifact')
        record = r.seal({'plan_id': 'plan', 'task_id': task_id,
            'fit_version': 'v5.1-bcd-fix', 'core': {'core': 'hash'},
            'environment': plan['expected_environment'],
            'input_fingerprints': task['expected_input_fingerprints'],
            'npz_sha256': r.sha(npz)})
        r.write_json(npz.with_suffix('.json'), record)
        if task_id in plan['gate_task_ids']:
            r.write_json(out / 'gates' / npz.with_suffix('.json').name,
                         r.seal({'plan_id': 'plan', 'task_id': task_id,
                                 'result_sha256': record['sha256']}))
        return r, out, plan, task

    def test_shell_syntax_and_site_activation_without_terminal_variables(self):
        self.api()
        shell = LAUNCHER.read_text(encoding='utf-8')
        result = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(LAUNCHER)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('\r', LAUNCHER.read_bytes().decode())
        self.assertNotIn('source ', shell)
        self.assertNotIn('#SBATCH --array', shell)
        self.assertIn('module load anaconda3', shell)
        self.assertIn('conda activate myenv', shell)
        self.assertLess(shell.index('conda activate myenv'), shell.index('set -euo pipefail'))
        self.assertIn('#SBATCH --chdir=/public/home/xjh/xmx/tensor-sparse-regression-main', shell)
        self.assertIn('#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-%j.out', shell)
        self.assertIn('export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1', shell)

    def test_missing_results_are_not_claimed_complete(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            r, out, plan, task = self.fixture(directory)
            for path in (out / 'tasks').glob('*'):
                path.unlink()
            (out / 'gates' / 'task_000000.json').unlink()
            self.assertFalse(api['completed_result'](r, out, plan, task))

    def test_valid_existing_pilot_is_read_only(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            r, out, plan, task = self.fixture(directory)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
            self.assertTrue(api['completed_result'](r, out, plan, task))
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})

    def test_partial_result_and_missing_pilot_gate_stop_without_repair(self):
        api = self.api()
        for missing in ('tasks/task_000000.json', 'tasks/task_000000.npz', 'gates/task_000000.json'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                r, out, plan, task = self.fixture(directory)
                (out / missing).unlink()
                with self.assertRaises(ValueError):
                    api['completed_result'](r, out, plan, task)
                self.assertFalse((out / missing).exists())

    def test_foreign_environment_or_tampered_artifact_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            r, out, plan, task = self.fixture(directory)
            plan['expected_environment'] = {'python': 'different'}
            with self.assertRaises(ValueError):
                api['completed_result'](r, out, plan, task)
            plan['expected_environment'] = {'python': '3.9.25'}
            (out / 'tasks/task_000000.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                api['completed_result'](r, out, plan, task)

    def test_existing_plan_is_never_overwritten(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            origin, target = Path(directory) / 'origin.json', Path(directory) / 'plan.json'
            origin.write_bytes(b'original\r\n')
            api['copy_plan'](self.runner(), origin, target)
            self.assertEqual(target.read_bytes(), origin.read_bytes())
            target.write_bytes(b'foreign')
            with self.assertRaises(ValueError):
                api['copy_plan'](self.runner(), origin, target)
            self.assertEqual(target.read_bytes(), b'foreign')

    def test_failed_preflight_is_not_overwritten_or_accepted(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            r, out, plan, task = self.fixture(directory)
            path = out / 'preflight_task_000000.json'
            r.write_json(path, r.seal({'passed': False}))
            before = path.read_bytes()
            with self.assertRaises((ValueError, KeyError)):
                api['checked_preflight'](r, out, plan)
            self.assertEqual(before, path.read_bytes())

    def test_full_workflow_runs_five_pilots_then_245_workers_and_merge(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            out = repo / 'numerical_simulation/outputs_v51_attribution/cluster_run'
            out.mkdir(parents=True)
            origin = repo / 'numerical_simulation/outputs_version_attribution/cluster_run/plan.json'
            origin.parent.mkdir(parents=True)
            origin.write_bytes(b'unchanged-frozen-plan')
            real = self.runner()
            ids = [*range(50), *range(50, 100), *range(250, 300), *range(350, 400), *range(550, 600)]
            plan = {'sha256': 'plan', 'fit_version': 'v5.1-bcd-fix',
                    'source_manifests': {'v5.1-bcd-fix': {'core': 'hash'}},
                    'expected_environment': {'python': '3.9.25'},
                    'gate_task_ids': [0, 50, 250, 350, 550],
                    'tasks': [{'task_id': i, 'expected_input_fingerprints': {'X_train': 'x'}} for i in ids]}
            r = SimpleNamespace(**{k: getattr(real, k) for k in
                ('require', 'read', 'sha', 'check_seal', 'require_gates')}, load_plan=lambda output: plan)
            calls = []

            def invoke(command, output, task_id, pilot):
                calls.append((command, task_id, pilot))
                if command == 'preflight':
                    record = {'plan_id': 'plan', 'task_id': 0, 'fit_executed': False,
                        'expected': {'X_train': 'x'}, 'actual': {'X_train': 'x'},
                        'matches': {'X_train': True}, 'expected_environment': plan['expected_environment'],
                        'actual_environment': plan['expected_environment'], 'environment_matches': True, 'passed': True}
                    real.write_json(output / 'preflight_task_000000.json', real.seal(record))
                elif command == 'worker':
                    self.fixture(output, task_id)
                else:
                    self.assertEqual(command, 'merge')
                    self.assertEqual(len(list((output / 'tasks').glob('*.json'))), 250)

            # A preexisting complete pilot and bulk result must remain byte-identical.
            self.fixture(out, 0)
            self.fixture(out, 1)
            protected = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
            messages = StringIO()
            with redirect_stdout(messages):
                api['execute_workflow'](r, repo, invoke)
            self.assertIn('ALL FIVE PILOT GATES PASSED', messages.getvalue())
            self.assertIn('250/250 task records verified', messages.getvalue())
            workers = [(task_id, pilot) for command, task_id, pilot in calls if command == 'worker']
            self.assertEqual(workers[:4], [(50, True), (250, True), (350, True), (550, True)])
            self.assertEqual(len(workers), 248)
            self.assertEqual({i for i, _ in workers}, set(ids) - {0, 1})
            self.assertEqual(calls[-1], ('merge', None, False))
            self.assertEqual(protected, {p: p.read_bytes() for p in protected})
            self.assertEqual((out / 'plan.json').read_bytes(), origin.read_bytes())

            # A child failure must stop the workflow; it must never attempt merge.
            def fail(command, output, task_id, pilot):
                raise subprocess.CalledProcessError(7, command)

            # All records exist, so the first invoked child is merge; remove one
            # complete nonpilot pair to make a missing worker fail before merge.
            (out / 'tasks/task_000002.json').unlink()
            (out / 'tasks/task_000002.npz').unlink()
            with redirect_stdout(StringIO()), self.assertRaises(subprocess.CalledProcessError) as caught:
                api['execute_workflow'](r, repo, fail)
            self.assertEqual(caught.exception.cmd, 'worker')

            (out / 'paired_three_versions.csv').write_bytes(b'preserve-existing-summary')
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'merge output'):
                api['execute_workflow'](r, repo, fail)
            self.assertEqual((out / 'paired_three_versions.csv').read_bytes(), b'preserve-existing-summary')


if __name__ == '__main__':
    unittest.main()
