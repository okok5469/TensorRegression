"""Locked confirmation evaluates exactly two rules on untouched frozen pools."""
import ast
import copy
import importlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class ConfirmationTests(unittest.TestCase):
    def api(self):
        name = 'numerical_simulation.candidate_search_study.msic_confirmation'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Locked confirmation module is missing')
        return importlib.import_module(name)

    def rule(self):
        return {'id': 'msic-alpha2-scale0', 'criterion': 'msic', 'alpha': 2., 'gamma': 1., 'scale': 0.}

    def test_only_default_and_locked_rules_are_scored(self):
        api = self.api()
        pool = [{'sizes': [1, 1, 1], 'rss': 190., 'loss': .95},
                {'sizes': [4, 3, 5], 'rss': 90., 'loss': .45}]
        before = copy.deepcopy(pool)
        with patch.object(api.scoring, 'rescore_metadata', side_effect=AssertionError('eleven rules forbidden')):
            with patch.object(api.scoring, 'rescore_task', side_effect=AssertionError('eleven rules forbidden')):
                choices = api.two_choices(pool, 100, (32, 20, 25), self.rule())
        self.assertEqual([rule['role'] for rule, _, _ in choices], ['default', 'locked'])
        self.assertEqual([rule['id'] for rule, _, _ in choices], ['msic-alpha1.5-scale1', 'msic-alpha2-scale0'])
        self.assertEqual(pool, before)

    def test_lock_equal_to_default_still_has_two_distinct_roles(self):
        api = self.api()
        rule = {'id': 'msic-alpha1.5-scale1', 'criterion': 'msic', 'alpha': 1.5, 'gamma': 1., 'scale': 1.}
        choices = api.two_choices([{'sizes': [1, 1, 1], 'rss': 20., 'loss': .5}], 20, (2, 2, 2), rule)
        self.assertEqual(len(choices), 2)
        self.assertEqual(choices[0][1], choices[1][1])
        self.assertNotEqual(choices[0][0]['label'], choices[1][0]['label'])

    def test_unknown_or_changed_locked_rule_is_rejected(self):
        api = self.api()
        for field, value in [('alpha', 1.7), ('gamma', .5), ('scale', 2.), ('id', 'unknown')]:
            changed = self.rule(); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                api.two_choices([{'sizes': [1, 1, 1], 'rss': 20., 'loss': .5}], 20, (2, 2, 2), changed)

    def development_fixture(self, root, task_count=1, manifest_count=None):
        from numerical_simulation.candidate_search_study import run, msic_after_search
        from numerical_simulation.version_validation.worker import array_fingerprint
        source = run.source_manifests(); environment = run.environment()
        tasks = [{'task_id': i, 'request': {'config': {'unit_fixture': True}, 'seed': 10000+i,
                                          'test_seed': 1010000+i}} for i in range(task_count)]
        manifests = [{'task_id': i, 'A3_pool_fingerprint': f'pool-{i}'}
                     for i in range(task_count if manifest_count is None else manifest_count)]
        search = {'plan_id': 'search', 'tasks': tasks, 'sources': source, 'environment': environment}
        msic = root/'scored'; msic.mkdir(); (msic/'jobs').mkdir()
        research = {'numerical_simulation/candidate_search_study/msic_after_search.py': run.file_hash(Path(msic_after_search.__file__))}
        body = {'input_plan_id': 'search', 'inputs': manifests, 'frozen_input_sources': source,
                'environment': environment, 'research_sources': research, 'rules': msic_after_search.declared_rules()}
        plan = {**body, 'plan_id': run.digest(body)}
        run.write_json(msic/'plan.json', plan)
        (msic/'rule_summary.csv').write_text('rule,n_datasets\nlocked,1\n', encoding='utf8')
        lock = {'status': 'locked', 'search_group': 'A3', 'rule': self.rule(),
                'development_search_plan_id': 'search', 'development_msic_plan_id': plan['plan_id'],
                'development_summary_sha256': run.file_hash(msic/'rule_summary.csv'),
                'source_core_sha256': source['core'], 'scoring_source_sha256': research}
        run.write_json(msic/'integrity_report.json', {'status': 'complete', 'checks_passed': True,
            'plan_id': plan['plan_id'], 'dataset_count': task_count, 'rule_rows': 11*task_count,
            'default_exact_replays': task_count, 'pool_fingerprints_unchanged': True})
        B = np.zeros((2, 2, 2)); supports = [np.array([0])] * 3
        for manifest in manifests:
            i = manifest['task_id']; cache = msic/'jobs'/f'task_{i:03d}.npz'
            np.savez_compressed(cache, candidate_0_B=B, **{f'candidate_0_S{m}': supports[m-1] for m in (1, 2, 3)})
            rows = [{'rule': r['rule'], 'selected_index': 0, 'B_fingerprint': array_fingerprint(B),
                     'supports': [[0], [0], [0]]} for r in msic_after_search.declared_rules()]
            report = {'status': 'complete', 'plan_id': plan['plan_id'], 'input_result_manifest': manifest,
                      **tasks[i]['request'], 'environment': environment['libraries'],
                      'fingerprints': {'core_sources': source['core']}, 'default_exact_replay': True,
                      'pool_fingerprint_before': manifest['A3_pool_fingerprint'],
                      'pool_fingerprint_after': manifest['A3_pool_fingerprint'], 'rows': rows,
                      'array_cache': cache.name, 'array_cache_sha256': run.file_hash(cache)}
            run.write_json(cache.with_suffix('.json'), report)
        return search, manifests, msic, lock

    def test_development_summary_identity_and_cached_B_are_bound(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            context = self.development_fixture(Path(temporary))
            rule, provenance = api.validate_development_lock(*context)
            self.assertEqual(rule, self.rule())
            self.assertEqual(len(provenance['scoring_jobs']), 1)
            (context[2]/'rule_summary.csv').write_text('changed', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'summary'):
                api.validate_development_lock(*context)

    def test_missing_development_manifest_cannot_pass_complete_integrity(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            context = self.development_fixture(Path(temporary), task_count=2, manifest_count=1)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                api.validate_development_lock(*context)

    def test_changed_scoring_cache_is_rejected_even_with_complete_status(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            context = self.development_fixture(Path(temporary))
            with (context[2]/'jobs/task_000.npz').open('ab') as handle: handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.validate_development_lock(*context)

    def confirmation_plan(self):
        from numerical_simulation.candidate_search_study import run
        plan = run.build_plan('confirmation', 10, ['A0', 'A1', 'A2', 'A3', 'A4'])
        return plan, {'confirmation_search_plan_id': plan['plan_id'], 'confirmation_groups': plan['groups']}

    def test_original_ninety_dataset_five_group_protocol_is_required(self):
        api = self.api(); plan, lock = self.confirmation_plan()
        api.validate_confirmation_plan(plan, lock)
        with self.assertRaisesRegex(ValueError, 'plan'):
            api.validate_confirmation_plan(plan, {**lock, 'confirmation_search_plan_id': 'other'})
        changed = copy.deepcopy(plan); changed['tasks'][0]['request']['seed'] += 1
        with self.assertRaises(ValueError): api.validate_confirmation_plan(changed, lock)
        changed = copy.deepcopy(plan); changed['groups'] = ['A3']
        with self.assertRaises(ValueError): api.validate_confirmation_plan(changed, lock)

    def test_late_or_naive_lock_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'logs').mkdir(); log = root/'logs/task.stdout.txt'; log.write_text('done')
            os.utime(log, (2000000000., 2000000000.))
            evidence = api.validate_lock_timing({'locked_utc': '2030-01-01T00:00:00+00:00'}, root)
            self.assertLess(evidence['lock_epoch'], evidence['earliest_confirmation_log_mtime'])
            for timestamp in ('2040-01-01T00:00:00+00:00', '2030-01-01T00:00:00'):
                with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                    api.validate_lock_timing({'locked_utc': timestamp}, root)

    def fixture(self, temporary):
        from tests.test_msic_after_search import FrozenMsicTests
        source = Path(temporary)/'input'; FrozenMsicTests().fixture(source)
        from numerical_simulation.candidate_search_study import run
        path = source/'jobs/task_000.json'
        return run.read_json(path), path.with_suffix('.npz')

    def test_actual_numeric_default_replay_two_Bs_and_unchanged_pool(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            report, path = self.fixture(temporary)
            before = path.read_bytes()
            with patch.object(api.scoring, 'rescore_metadata', side_effect=AssertionError('eleven rules forbidden')):
                with patch.object(api.scoring, 'rescore_task', side_effect=AssertionError('eleven rules forbidden')):
                    actual, arrays = api.evaluate_task(report, path, self.rule())
            self.assertEqual(len(actual['rows']), 2)
            self.assertEqual(set(arrays), {'default_B', 'locked_B'})
            self.assertEqual(actual['pool_fingerprint_before'], actual['pool_fingerprint_after'])
            self.assertEqual(actual['fingerprints']['independent_test'], report['fingerprints']['independent_test'])
            api.scoring.default_replay(report['rows'][0], actual['rows'][0])
            self.assertEqual(before, path.read_bytes())
            self.assertTrue(actual['locked_numerics']['checks_passed'])
            self.assertEqual(actual['locked_numerics']['rank'], actual['locked_numerics']['q_hat'])
            self.assertIn('singular_values', actual['locked_numerics'])

    def test_changed_default_metric_is_not_repaired(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            report, path = self.fixture(temporary); report['rows'][0]['pmse'] += .01
            with self.assertRaisesRegex(ValueError, 'Default replay.*pmse'):
                api.evaluate_task(report, path, self.rule())

    def test_changed_selected_training_rss_is_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            report, path = self.fixture(temporary)
            for candidate in report['groups']['A3']['candidates']:
                candidate['rss'] += 100.; candidate['loss'] = candidate['rss']/200.
            with np.load(path, allow_pickle=False) as arrays:
                report['groups']['A3']['pool_fingerprint'] = api.pool_fingerprint(api.scoring.stream_pool(report['groups']['A3'], arrays))
            with self.assertRaisesRegex(ValueError, 'Training RSS'):
                api.evaluate_task(report, path, self.rule())

    def test_existing_output_stops_before_context_or_evaluation(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'existing'; output.mkdir()
            with patch.object(api, 'load_context') as context, self.assertRaises(FileExistsError):
                api.run_cohort('confirmation', 'development', 'scored', 'lock.json', output)
            context.assert_not_called()

    def test_python39_syntax(self):
        api = self.api()
        ast.parse(Path(api.__file__).read_text(encoding='utf8'), feature_version=(3, 9))


if __name__ == '__main__':
    unittest.main()
