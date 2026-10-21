"""Research protocol guards, independent seed split, and passive-core regression."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SelectionResearchTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.taskcard8.selection'),
                             'selection research protocol is not implemented')
        return importlib.import_module('scripts.taskcard8.selection')

    def test_design_has_full_coverage_and_disjoint_new_seeds(self):
        s = self.module()
        tasks = s.make_requests()
        self.assertEqual(len(tasks), 216)
        for cohort in ('development', 'confirmation'):
            rows = [r for r in tasks if r['cohort'] == cohort]
            self.assertEqual(len(rows), 108)
            self.assertEqual(len({r['cell_id'] for r in rows}), 36)
        seeds = [r[k] for r in tasks for k in ('seed', 'test_seed')]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual({tuple(r['config']['sparsity']) for r in tasks}, {(4, 3, 5), (2, 3, 4)})
        self.assertEqual({tuple(r['config']['rho']) for r in tasks}, {(0., 0., 0.), (.6, .6, .6)})
        self.assertEqual({r['config']['k_max'] for r in tasks}, {1})
        self.assertEqual({r['config']['max_cycles'] for r in tasks}, {6})

    def test_selector_ignores_poisoned_outer_test_and_keeps_pool_immutable(self):
        s = self.module()
        pool = [{'rss': 100., 'sizes': [1, 1, 1]}, {'rss': 30., 'sizes': [2, 2, 2]}]
        before = json.dumps(pool)
        first = s.select(pool, 100, (5, 5, 5), 1.5, 1.)
        for row in pool:
            row['pmse'] = object()
            row['outer_test_y'] = object()
        second = s.select(pool, 100, (5, 5, 5), 1.5, 1.)
        self.assertEqual(first, second)
        self.assertEqual(before, json.dumps([{k: r[k] for k in ('rss', 'sizes')} for r in pool]))

    def test_immutable_artifact_write_and_confirmation_lock_gate(self):
        s = self.module()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            s.write(folder / 'x.json', {'hello': 1})
            with self.assertRaises(FileExistsError):
                s.write(folder / 'x.json', {'hello': 2})
            with self.assertRaisesRegex(ValueError, 'lock'):
                s.confirmation_lock(folder, {'sha256': 'unlocked'})

    def test_gate_rejects_pmse_improvement_with_strong_signal_underselection(self):
        s = self.module()
        rows = []
        for rule, pmse, under in [('baseline', 2., 0.), ('alpha2_scale0', 1.8, 1.)]:
            rows.append({'rule': rule, 'cell_id': 0, 'n_train': 100, 'snr': 4.,
                         'pmse': pmse, 'mode_f1_macro': .8, 'q_under': under,
                         'near_capacity': 0.})
        gate = s.development_gate(rows)
        self.assertFalse(gate['passed'])
        self.assertFalse(gate['checks']['strong_snr4_underselection'])
        self.assertTrue(gate['checks']['overall_pmse'])

    def test_confirmation_rejects_changed_development_artifact(self):
        s = self.module()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            s.write(folder / 'plan.json', {'rules': [1]})
            s.write(folder / 'development/task_000/complete.json', {'result': 1})
            plan = {'rules': [1]}
            s.write(folder / 'development_lock.json', {'plan_sha256': s.sha(folder / 'plan.json'),
                    'rules': [1], 'completed_development_tasks': 108,
                    'development_artifacts': s.artifact_manifest(folder / 'development')})
            self.assertEqual(s.confirmation_lock(folder, plan), s.sha(folder / 'development_lock.json'))
            (folder / 'development/task_000/complete.json').write_text('{"result":2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'artifact drift'):
                s.confirmation_lock(folder, plan)

    def test_passive_trace_matches_unobserved_actual_v52_and_default_rescore(self):
        s = self.module()
        code = """
import numpy as np
from scripts.taskcard7.lemon import configure_core, observe_fit
configure_core('v5.2-search-final')
from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
from numerical_simulation.version_validation.worker import fit_adaptive
from scripts.taskcard8.selection import select
c=TensorSimulationConfig(n_train=25,n_test=5,dims=(4,4,4),sparsity=(1,1,2),k_max=1,max_cycles=2,tensor_abess_mode='adaptive')
d=generate_simulation_dataset(c,seed=1806101999)
a=fit_adaptive(d.X_train,d.y_train,c)
b,t,coefficients=observe_fit(lambda:fit_adaptive(d.X_train,d.y_train,c))
assert np.array_equal(a.B,b.B)
assert (a.S1.tolist(),a.S2.tolist(),a.S3.tolist())==(b.S1.tolist(),b.S2.tolist(),b.S3.tolist())
choice=select(t['candidates'],25,c.dims,1.5,1.)
assert np.array_equal(coefficients[choice['selected_index']],b.B)
assert choice['selected_score']==b.sic
assert len(t['candidates'])==len(t['ordered_size_grid'])+1
print('PASSIVE_CORE_EXACT')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASSIVE_CORE_EXACT', result.stdout)


if __name__ == '__main__':
    unittest.main()
