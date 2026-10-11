"""Frozen-pool global scoring research: no new support search or test selection."""
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'doc/algorithm_versions/v5.2-candidate-search/src'))
import tensor_abess  # Explicit source identity before numerical package imports.
from numerical_simulation.candidate_search_study import run, worker
from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
from numerical_simulation.version_validation.worker import array_fingerprint, evaluate_shared_test, select_candidate


class FrozenMsicTests(unittest.TestCase):
    def api(self):
        name = 'numerical_simulation.candidate_search_study.msic_after_search'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Independent frozen-pool research feature is missing')
        return importlib.import_module(name)

    def fixture(self, base):
        base.mkdir(); (base/'jobs').mkdir(); (base/'requests').mkdir()
        tasks = []; sources = run.source_manifests(); env = run.environment()
        for task_id, (n, snr) in enumerate((n, s) for n in (100, 200, 400) for s in (.5, 2., 4.)):
            config = TensorSimulationConfig(n_train=n, n_test=20, dims=(3, 2, 2), sparsity=(1, 1, 1),
                rho=(.1, .1, .1), snr=snr, max_cycles=1, test_batch_size=10)
            seed = 10000+task_id; test_seed = seed+1000000
            dataset = generate_simulation_dataset(config, seed=seed)
            candidates = []
            for index, sizes in enumerate(((1, 1, 1), (2, 1, 1), (2, 2, 1))):
                supports = [np.arange(size) for size in sizes]
                D = dataset.X_train[:, supports[0]][:, :, supports[1]][:, :, :, supports[2]].reshape(n, -1)
                beta = np.linalg.lstsq(D, dataset.y_train, rcond=None)[0]
                B = np.zeros(config.dims); B[np.ix_(*supports)] = beta.reshape(sizes)
                rss = float(np.sum((dataset.y_train-D@beta)**2))
                candidates.append({'kind': 'adaptive-stage' if index == 0 else 'global-fixed', 'B': B,
                    'supports': supports, 'sizes': sizes, 'rss': rss, 'loss': rss/(2*n)})
            chosen, selection = select_candidate(candidates, n, config.dims, alpha=1.5, scale=1.)
            predictions, test_fp = evaluate_shared_test(dataset, [chosen['B']], test_seed)
            fp = {'X_train': array_fingerprint(dataset.X_train), 'y_train': array_fingerprint(dataset.y_train),
                  'B_true': array_fingerprint(dataset.B_true), 'independent_test': test_fp}
            request = {'config': asdict(config), 'seed': seed, 'test_seed': test_seed,
                       'fit_options': {'candidate_budget': 2000}}
            request = json.loads(run.canonical(request))
            task = {'task_id': task_id, 'scenario_id': task_id, 'replicate': 0, 'request': request,
                    'request_file': f'requests/task_{task_id:03d}.json', 'request_id': run.digest(request)}
            tasks.append(task); run.write_json(base/task['request_file'], request)
            arrays = {}
            for index, candidate in enumerate(candidates):
                arrays[f'A3_candidate_{index}_B'] = candidate['B']
                for mode, S in enumerate(candidate['supports'], 1): arrays[f'A3_candidate_{index}_S{mode}'] = S
            cache = base/'jobs'/f'task_{task_id:03d}.npz'; np.savez_compressed(cache, **arrays)
            row = {'group': 'A3', 'rule': 'msic-alpha1.5-scale1', 'method': 'A3/msic-alpha1.5-scale1',
                'n_train': n, 'snr': snr, 'seed': seed, 'test_seed': test_seed, 'q_true': 1,
                **compute_support_metrics(*chosen['supports'], *dataset.supports_true),
                **compute_coefficient_metrics(chosen['B'], dataset.B_true), **predictions[0],
                'rss_train': chosen['rss'], 'loss_train': chosen['loss'], 'runtime_sec': 0.,
                'selection': selection, 'supports': [S.tolist() for S in chosen['supports']],
                'B_fingerprint': array_fingerprint(chosen['B']), 'candidate_count': len(candidates),
                'true_size_in_pool': 1, 'true_support_in_pool': 0}
            report = {**request, 'version': worker.VERSION, 'status': 'complete', 'group_order': ['A3'],
                'environment': env['libraries'], 'fingerprints': {**fp, 'core_sources': sources['core']},
                'array_cache': cache.name, 'array_cache_sha256': run.file_hash(cache), 'rows': [row],
                'groups': {'A3': {'fingerprints': fp, 'environment': env['libraries'],
                    'pool_fingerprint': worker.pool_fingerprint(candidates),
                    'candidates': [worker.candidate_json(c, i) for i, c in enumerate(candidates)]}}}
            run.write_json(cache.with_suffix('.json'), report)
        body = {'version': worker.VERSION, 'cohort': 'development', 'n_reps': 1, 'groups': ['A3'],
                'tasks': tasks, 'sources': sources, 'environment': env}
        plan = {**body, 'plan_id': run.digest(body)}; run.write_json(base/'plan.json', plan)
        run.write_json(base/'integrity_report.json', {'status': 'complete', 'integrity_checks_passed': True,
            'plan_id': plan['plan_id'], 'completed_tasks': 9, 'expected_tasks': 9})
        return plan

    def test_eleven_rules_keep_original_default_and_distinct_baselines(self):
        rules = self.api().declared_rules()
        self.assertEqual(len(rules), 11)
        self.assertEqual(len({rule['rule'] for rule in rules}), 11)
        self.assertEqual([(r['alpha'], r['scale']) for r in rules if r['criterion']=='msic'],
                         [(a, s) for a in (1.5, 2., 3.) for s in (0., .5, 1.)])
        self.assertEqual([r['criterion'] for r in rules[-2:]], ['tensor-sic', 'bic'])
        self.assertTrue(all(r['scale']==0 for r in rules[-2:]))

    def test_score_metadata_never_needs_coefficients_or_changes_input(self):
        pool = [{'sizes': [1, 1, 1], 'rss': 190., 'loss': .95},
                {'sizes': [4, 3, 5], 'rss': 90., 'loss': .45}]
        before = run.canonical(pool)
        choices = self.api().rescore_metadata(pool, 100, (32, 20, 25))
        self.assertEqual(run.canonical(pool), before)
        for rule, index, info in choices:
            selected, expected = select_candidate(pool, 100, (32, 20, 25),
                criterion=rule['criterion'], alpha=rule['alpha'] or 1.5, scale=rule['scale'])
            self.assertEqual(info, expected)
            self.assertIs(selected, pool[index])

    def test_full_fixture_default_replay_and_immutable_outputs(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            output = Path(temp)/'rescore'; summary = api.run_cohort(source, output)
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['dataset_count'], 9)
            self.assertEqual(summary['rule_rows'], 99)
            self.assertEqual(summary['default_exact_replays'], 9)
            self.assertTrue(summary['pool_fingerprints_unchanged'])
            self.assertEqual(before, {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()})
            self.assertTrue((output/'scenario_summary.csv').is_file())
            self.assertTrue((output/'rule_summary.csv').is_file())
            self.assertTrue((output/'paired_vs_default.csv').is_file())
            with self.assertRaises(FileExistsError): api.run_cohort(source, output)

    def test_missing_complete_integrity_stops_before_output(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; source.mkdir()
            with self.assertRaises((ValueError, FileNotFoundError)):
                api.run_cohort(source, Path(temp)/'new')
            self.assertFalse((Path(temp)/'new').exists())

    def test_changed_pool_fingerprint_cannot_be_ignored(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            path = source/'jobs/task_000.json'; report = run.read_json(path)
            report['groups']['A3']['pool_fingerprint'] = 'corrupt'
            path.write_text(run.canonical(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pool fingerprint'):
                api.run_cohort(source, Path(temp)/'new')

    def test_changed_npz_checksum_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            with (source/'jobs/task_000.npz').open('ab') as handle: handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.run_cohort(source, Path(temp)/'new')

    def test_default_metric_mismatch_stops_without_repair(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            path = source/'jobs/task_000.json'; report = run.read_json(path)
            report['rows'][0]['pmse'] += .01; path.write_text(run.canonical(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Default replay.*pmse'):
                api.run_cohort(source, Path(temp)/'new')

    def test_training_rss_must_be_recomputed_from_selected_coefficients(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            path = source/'jobs/task_000.json'; report = run.read_json(path)
            for candidate in report['groups']['A3']['candidates']:
                candidate['rss'] += 100.; candidate['loss'] = candidate['rss']/200.
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                report['groups']['A3']['pool_fingerprint'] = worker.pool_fingerprint(api.stream_pool(report['groups']['A3'], arrays))
            path.write_text(run.canonical(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Training RSS'):
                api.run_cohort(source, Path(temp)/'new')

    def test_confirmation_cannot_receive_all_eleven_rules(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'input'; self.fixture(source)
            path = source/'plan.json'; plan = run.read_json(path); plan['cohort']='confirmation'
            plan['plan_id'] = run.digest({k:v for k,v in plan.items() if k!='plan_id'})
            path.write_text(run.canonical(plan), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'confirmation|LOCK'):
                api.run_cohort(source, Path(temp)/'new')


if __name__ == '__main__': unittest.main()
