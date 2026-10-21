"""Cluster runner contracts without scientific input generation or submission."""
import copy
import importlib
import json
from pathlib import Path
import sys
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    run = importlib.import_module('research.taskcard12.simulation')
except ModuleNotFoundError:
    run = None


def requests():
    rows = []
    import itertools
    for cell, (n, snr, shape, rho) in enumerate(itertools.product(
            (100, 200, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)), (0., .6))):
        for rep in range(10):
            seed = 200000000 + cell * 1000 + rep
            rows.append({'cohort': 'formal', 'task_id': len(rows), 'cell_id': cell,
                         'replicate': rep, 'pressure': False, 'seed': seed, 'test_seed': seed + 50000000,
                         'config': {'n_train': n, 'snr': snr, 'sparsity': list(shape), 'rho': [rho] * 3}})
    for rep in range(10, 30):
        row = copy.deepcopy(rows[210])
        row.update(task_id=len(rows), replicate=rep, pressure=True,
                   seed=200021000 + rep, test_seed=250021000 + rep)
        rows.append(row)
    return rows


class ClusterGates(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run, 'Task12 simulation runner missing')

    def test_exact380_design_and_pressure_replicates(self):
        rows = requests()
        run.validate_requests(rows)
        for invalid in (rows[:-1], rows + [rows[0]], rows[:360]):
            with self.assertRaisesRegex(ValueError, '380|coverage|duplicate'):
                run.validate_requests(invalid)
        bad = copy.deepcopy(rows)
        bad[-1]['config']['snr'] = 2.
        with self.assertRaisesRegex(ValueError, 'pressure|coverage'):
            run.validate_requests(bad)

    def test_frozen_request_bytes_template_and_historical_registry_are_binding(self):
        self.assertTrue(hasattr(run, 'validate_request_artifacts'), 'Frozen input registry check missing')
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            rows = requests()
            smoke = [copy.deepcopy(rows[i]) for i in (10, 210, 350)]
            for index, row in enumerate(smoke):
                row.update(task_id=index, cohort='smoke', seed=row['seed'] + 300000000,
                           test_seed=row['test_seed'] + 300000000)
            run.write_json(repo / 'requests.json', {'requests': rows, 'smoke_requests': smoke})
            run.write_json(repo / 'history.json', {'inputs': [{'X_train': 'old_x', 'y_train': 'old_y', 'path': 'old'}]})
            protocol = {'requests': rows, 'smoke_requests': smoke, 'template_config': rows[0]['config'],
                'requests_path': 'requests.json', 'requests_sha256': run.file_sha(repo / 'requests.json'),
                'history_input_registry': {'path': 'history.json', 'sha256': run.file_sha(repo / 'history.json')}}
            known = run.validate_request_artifacts(repo, protocol)
            self.assertEqual(known[0]['X_train'], 'old_x')
            with self.assertRaisesRegex(ValueError, 'historical'):
                run.claim_input(repo / 'out', {'X_train': 'old_x', 'y_train': 'old_y'}, {}, known)
            bad = copy.deepcopy(protocol)
            bad['requests'][0]['config']['sigma'] = .01
            with self.assertRaisesRegex(ValueError, 'request|template'):
                run.validate_request_artifacts(repo, bad)
            (repo / 'history.json').write_text('{"inputs": []}')
            with self.assertRaisesRegex(ValueError, 'registry'):
                run.validate_request_artifacts(repo, protocol)

    def test_seed_collision_and_duplicate_input_identity_fail(self):
        rows = requests()
        rows[-1]['seed'] = rows[0]['seed']
        with self.assertRaisesRegex(ValueError, 'seed'):
            run.validate_requests(rows)
        with tempfile.TemporaryDirectory() as temp:
            fp = {'X_train': 'x', 'y_train': 'y'}
            run.claim_input(temp, fp, {}, [])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                run.claim_input(temp, fp, {}, [])

    def test_legacy_digest_pair_is_checked_without_changing_current_claim(self):
        self.assertIn('alternate_pairs', __import__('inspect').signature(run.claim_input).parameters,
                      'Historical digest encodings must both be checked')
        with tempfile.TemporaryDirectory() as temp:
            current = {'X_train': 'current_x', 'y_train': 'current_y'}
            historical = [{'X_train': 'legacy_x', 'y_train': 'legacy_y'}]
            with self.assertRaisesRegex(ValueError, 'historical'):
                run.claim_input(temp, current, {}, historical,
                                alternate_pairs=[('legacy_x', 'legacy_y')])
            self.assertFalse((Path(temp) / 'input_claims').exists())
            first = run.claim_input(temp, current, {}, [], alternate_pairs=[('legacy_x', 'legacy_y')])
            self.assertEqual(run.read_json(first)['input_fingerprints'], current)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                run.claim_input(temp, current, {}, [], alternate_pairs=[('other_x', 'other_y')])

    def test_expected_environment_includes_real_blas_and_threads(self):
        environment = {'python': '3.9.25', 'numpy': '2.0.2', 'scipy': '1.13.1',
                       'system': 'Linux', 'machine': 'x86_64',
                       'numpy_config': {'Build Dependencies': {'blas': {'version': '0.3.27', 'name': 'openblas'}}},
                       'threads': dict.fromkeys(('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'), '1')}
        protocol = {'expected_versions': {'python': '3.9.25', 'numpy': '2.0.2', 'scipy': '1.13.1'},
                    'expected_platform': {'system': 'Linux', 'machine': 'x86_64'},
                    'expected_blas': {'openblas_version': '0.3.27'}}
        run.validate_environment(environment, protocol)
        for key, value in [('python', '3.13'), ('machine', 'arm64'), ('threads', {})]:
            with self.assertRaisesRegex(ValueError, 'environment|thread'):
                run.validate_environment(dict(environment, **{key: value}), protocol)
        bad = copy.deepcopy(environment)
        bad['numpy_config']['Build Dependencies']['blas']['version'] = '0.3.28'
        with self.assertRaisesRegex(ValueError, 'BLAS'):
            run.validate_environment(bad, protocol)

    def test_actual_slurm_array_metadata_required(self):
        with self.assertRaisesRegex(ValueError, 'SLURM'):
            run.slurm_identity({}, 2)
        env = {'SLURM_JOB_ID': '20', 'SLURM_ARRAY_JOB_ID': '19', 'SLURM_ARRAY_TASK_ID': '2',
               'SLURM_ARRAY_TASK_COUNT': '3'}
        self.assertEqual(run.slurm_identity(env, 2)['array_task_id'], 2)
        with self.assertRaisesRegex(ValueError, 'SLURM'):
            run.slurm_identity(env, 1)

    def test_smoke_cost_is_a_gate_with_fixed_case_limit(self):
        self.assertTrue(hasattr(run, 'smoke_cost'), 'Smoke cost gate missing')
        records = [{'request': {'config': {'n_train': n}}, 'resources': {'wall_seconds': t}}
                   for n, t in ((100, 100.), (200, 200.), (400, 300.))]
        gate = run.smoke_cost(records, 64)
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['projected_cpu_seconds_with_margin'], 1.5 * (120 * 100 + 140 * 200 + 120 * 300))
        records[-1]['resources']['wall_seconds'] = 2401.
        self.assertFalse(run.smoke_cost(records, 64)['passed'])

    def test_partial_result_is_terminal_and_complete_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'case'
            folder.mkdir()
            (folder / 'started.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Partial'):
                run.completed_result(folder, {}, {}, 'plan')
            (folder / 'complete.json').write_text(json.dumps({'request': {}, 'plan_sha256': 'plan',
                'environment': {}, 'source_files': {}, 'artifact_sha256': {'started.json': 'bad'}}))
            with self.assertRaisesRegex(ValueError, 'artifact'):
                run.completed_result(folder, {}, {'environment': {}, 'protocol': {'source_files': {}}}, 'plan')

    def test_completion_must_cover_every_preserved_artifact(self):
        self.assertTrue(hasattr(run, 'check_artifact_inventory'), 'Complete artifact inventory check missing')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'unsealed_evidence.json').write_text('{}')
            run.write_json(folder / 'complete.json', {'request': {}, 'plan_sha256': 'plan', 'environment': {},
                'source_files': {}, 'artifact_sha256': {}})
            with self.assertRaisesRegex(ValueError, 'artifact inventory'):
                run.completed_result(folder, {}, {'environment': {}, 'protocol': {'source_files': {}}}, 'plan')

    def test_case_timeout_nonzero_preserves_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run.run_owned_child([sys.executable, '-c', 'import time; time.sleep(5)'],
                                         Path(temp), .15)
            self.assertFalse(result['passed'])
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'], 0)
            self.assertTrue((Path(temp) / 'stdout.log').is_file())
            self.assertTrue((Path(temp) / 'stderr.log').is_file())

    def test_aggregate_balanced360_separates_pressure20(self):
        records = []
        for request in requests():
            metrics = {label: {'pmse': 100. if request['pressure'] else 1., 'ree': 1., 'element_f1': .5,
                              'exact_recovery': float(label == 'A0'), 'q': 24,
                              'sizes': [2, 3, 4], 'rss': 1.} for label in ('A0','A1')}
            records.append({'request': request, 'metrics': metrics,
                            'recovery_evidence': {'truth_shape_native_representable': True,
                                                   'truth_support_found_full_pool': True}})
        result = run.summarize(records)
        self.assertEqual(result['balanced360']['count'], 360)
        self.assertEqual(result['balanced360']['A0']['pmse_mean'], 1.)
        self.assertEqual(result['pressure_extra20']['count'], 20)
        self.assertEqual(result['pressure_all30']['count'], 30)
        self.assertEqual(result['pressure_all30']['original_exact_losses'], 30)

    def test_collect_refuses_missing_and_duplicate_task_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            run.write_json(output / 'plan.json', {'requests': requests(), 'smoke_requests': []})
            with self.assertRaisesRegex(ValueError, 'complete|coverage'):
                run.collect(output, 'formal')

    def test_merge_saves_frozen_analysis_before_declaring_integrity(self):
        from research.taskcard12 import analysis
        records = []
        for request in requests():
            records.append({'request': request, 'metrics': {
                label: {'pmse': 1., 'exact_recovery': 1.} for label in ('A0','A1')}, 'recovery_evidence': {}})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            plan = {'protocol': {'frozen': True}, 'protocol_sha256': 'protocol'}
            run.write_json(output / 'plan.json', plan)
            for request in requests():
                run.write_json(output / 'formal' / f"task_{request['task_id']:03d}" / 'selection.json',
                               {'A0': {'q': 1}, 'A1': {'q': 1}})
                run.write_json(output / 'formal' / f"task_{request['task_id']:03d}" / 'complete.json', {})
            with patch.object(run, 'verify_plan', return_value=plan), patch.object(run, 'require_gate'), \
                 patch.object(run, 'collect', return_value=records), patch.object(run, 'summarize', return_value={}), \
                 patch.object(analysis, 'analyze', return_value={'fixed_inference': True}) as analyze:
                run.merge(output, output)
                self.assertTrue((output / 'merged/statistical_analysis.json').is_file(), 'Frozen inference was not saved')
                analyze.assert_called_once_with(records, plan['protocol'])
            self.assertTrue(run.read_json(output / 'merged/statistical_analysis.json')['fixed_inference'])

    def test_preparation_binds_protocol_source_environment_and_smoke_gate(self):
        self.assertTrue(hasattr(run, 'prepare'), 'Cluster preparation is missing')
        with tempfile.TemporaryDirectory() as temp:
            repo, output = Path(temp), Path(temp) / 'out'
            source = repo / 'source.py'
            source.write_text('original', encoding='utf-8')
            smoke = [copy.deepcopy(requests()[i]) for i in (10, 210, 350)]
            for index, row in enumerate(smoke):
                row.update(task_id=index, cohort='smoke', seed=row['seed'] + 300000000,
                           test_seed=row['test_seed'] + 300000000)
            run.write_json(repo / 'requests.json', {'requests': requests(), 'smoke_requests': smoke})
            run.write_json(repo / 'history.json', {'inputs': []})
            protocol = {'source_files': {'source.py': run.file_sha(source)},
                        'requests': requests(), 'smoke_requests': smoke,
                        'template_config': requests()[0]['config'],
                        'requests_path': 'requests.json', 'requests_sha256': run.file_sha(repo / 'requests.json'),
                        'history_input_registry': {'path': 'history.json', 'sha256': run.file_sha(repo / 'history.json')},
                        'case_timeout_seconds': 3600, 'max_parallel': 64,
                        'expected_versions': {}, 'expected_platform': {},
                        'historical_input_index': []}
            run.write_json(repo / 'protocol.json', protocol)
            with patch.object(run, 'environment', return_value={'fixed': 1}), patch.object(run, 'validate_environment'):
                run.prepare(repo, output, repo / 'protocol.json')
                self.assertEqual(run.verify_plan(repo, output)['environment'], {'fixed': 1})
                with self.assertRaisesRegex(ValueError, 'smoke|complete|coverage'):
                    run.require_gate(repo, output)
                source.write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'source'):
                    run.verify_plan(repo, output)

    def test_formal_worker_refuses_missing_smoke_gate_before_launch(self):
        self.assertTrue(hasattr(run, 'worker'), 'Cluster worker is missing')
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(run, 'verify_plan', return_value={'requests': requests()}), \
                 patch.object(run, 'require_gate', side_effect=ValueError('smoke gate missing')), \
                 patch.object(run, 'run_owned_child') as launch:
                with self.assertRaisesRegex(ValueError, 'smoke'):
                    run.worker(Path(temp), Path(temp), 'formal', 0)
                launch.assert_not_called()

    def test_direct_formal_case_checks_gate_before_reading_or_generating_input(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(run, 'verify_plan', return_value={'requests': [{'task_id': 0}]}), \
                 patch.object(run, 'require_gate', side_effect=ValueError('smoke gate missing')), \
                 patch.object(run, 'read_json', side_effect=RuntimeError('reached input before gate')):
                failure = None
                try:
                    run.scientific_case(Path(temp), Path(temp), 'formal', 0)
                except Exception as error:
                    failure = error
                self.assertIsInstance(failure, ValueError, 'Direct case bypassed mandatory smoke gate')
                self.assertIn('smoke gate', str(failure))

    def test_formal_case_reuses_core_loaded_by_smoke_verification(self):
        from research.taskcard12 import selection
        selection.configure_core()
        from numerical_simulation import data_generation
        request = {'task_id': 0, 'seed': 123, 'config': {
            'dims': [1, 1, 1], 'sparsity': [1, 1, 1], 'rho': [0., 0., 0.]}}
        slurm = {'SLURM_JOB_ID': '20', 'SLURM_ARRAY_JOB_ID': '19',
                 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_ARRAY_TASK_COUNT': '380'}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            plan = {'requests': [request]}
            run.write_json(output / 'plan.json', plan)
            folder = output / 'formal/task_000'
            run.write_json(folder / 'request.json', request)
            run.write_json(folder / 'started.json', {'slurm': run.slurm_identity(slurm, 0),
                           'plan_sha256': run.file_sha(output / 'plan.json')})
            with patch.object(run, 'verify_plan', return_value=plan), \
                 patch.object(run, 'require_gate', side_effect=lambda *_: selection.configure_core()), \
                 patch.dict(os.environ, slurm), \
                 patch.object(data_generation, 'generate_simulation_dataset',
                              side_effect=RuntimeError('fixture blocked DGP')) as generate:
                failure = None
                try:
                    run.scientific_case(output, output, 'formal', 0)
                except Exception as error:
                    failure = error
                self.assertIsInstance(failure, RuntimeError, 'Valid loaded core prevented formal initialization')
                self.assertEqual(str(failure), 'fixture blocked DGP')
                generate.assert_called_once()

    def test_complete_cache_resume_skips_search_and_rejects_selected_B_mutation(self):
        from research.taskcard12 import selection
        from research.taskcard11.rescore import select_candidates
        from research.taskcard10.cv import native_options
        X = np.arange(1., 21.).reshape(20, 1, 1, 1)
        y = np.array([2. * i + (i % 3) for i in range(1, 21)])
        B = np.linalg.lstsq(X.reshape(20, 1), y, rcond=None)[0].reshape(1, 1, 1)
        residual = y - X.reshape(20, 1) @ B.ravel()
        row = {'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
               'rss': float(residual @ residual), 'B_sha256': run.array_sha(B)}
        choice = select_candidates([row, row.copy()], 20, (1, 1, 1))
        row['msic'] = choice['selected_score']
        trace = {'candidates': [row, row.copy()], 'ordered_size_grid': [[1, 1, 1]],
                 'caps': [1, 1, 1], 'fixed_searches': [{}], 'final': row,
                 'eligible_candidate_indices': choice['eligible_indices'], 'one_se_tolerance': choice['tolerance'],
                 'minimum_msic': choice['minimum_score'], 'minimum_msic_q': choice['minimum_q']}
        pool = selection.pool_from_saved(X, y, trace, [B, B.copy()], **native_options({}))
        models = {label: selection.select_from_pool(pool, selection_rule=rule)
                  for label, rule in (('A0', 'original'), ('A1', 'fs'))}
        from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
        truth = {'B_true': np.full((1, 1, 1), 2.),
                 **{f'S{i}': np.array([0], dtype=int) for i in (1, 2, 3)}}
        prediction = {'y': np.array([3., 6., 8.]), 'signal': np.array([2., 4., 6.]),
                      'prediction': np.array([[2., 2.], [4., 4.], [6., 6.]]),
                      'arm_order': np.array(['A0', 'A1']),
                      'model_sha256': np.array([models[k]['B_sha256'] for k in ('A0', 'A1')])}
        true_support = [truth[f'S{i}'] for i in (1, 2, 3)]
        metrics = {label: {**{k: model[k] for k in ('B_sha256', 'support', 'sizes', 'q', 'rss')},
                           **compute_coefficient_metrics(model['B'], truth['B_true']),
                           **compute_support_metrics(*[np.array(s) for s in model['support']], *true_support),
                           'pmse': 3., 'signal_mse': 0., 'q_under': False, 'q_over': False,
                           'coefficient_norm': float(np.linalg.norm(model['B']))}
                   for label, model in models.items()}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            request = {'task_id': 0, 'cohort': 'smoke', 'config': {
                'n_train': 20, 'dims': [1, 1, 1], 'sparsity': [1, 1, 1], 'n_test': 3}}
            sources = dict(pool['metadata']['identity']['source_sha256'])
            sources.update({'doc/algorithm_versions/v5.2-search-final/src/' + k:v
                            for k,v in pool['metadata']['identity']['core_sha256'].items()})
            plan = {'smoke_requests': [request], 'environment': {}, 'protocol': {'source_files': sources}}
            run.write_json(output / 'plan.json', plan)
            folder = output / 'smoke/task_000'
            selection.save_pool(folder / 'pool', pool)
            fp = {'X_train': run.array_sha(X), 'y_train': run.array_sha(y),
                  'B_true': run.array_sha(truth['B_true']),
                  'supports_true': [run.array_sha(s) for s in true_support]}
            run.write_json(folder / 'input_identity.json', {'input_fingerprints': fp})
            run.write_json(folder / 'selection.json', {label: {k:v for k,v in model.items() if k != 'B'}
                                                      for label,model in models.items()})
            run.write_npz(folder / 'selected_models.npz', {label + '_B': model['B'] for label, model in models.items()})
            run.write_npz(folder / 'truth.npz', truth)
            run.write_npz(folder / 'test_predictions.npz', prediction)
            record = {'request': request, 'plan_sha256': run.file_sha(output / 'plan.json'),
                      'source_files': sources, 'environment': {}, 'input_fingerprints': fp,
                      'metrics': metrics, 'test_fingerprint': {'n_observations': 3},
                      'recovery_evidence': {'q_true': 1, 'truth_sizes': [1, 1, 1], 'truth_support': [[0], [0], [0]],
                          'Q_native': 14, 'truth_shape_native_representable': True,
                          'truth_shape_found_full_pool': True, 'truth_support_found_full_pool': True},
                      'artifact_sha256': run.artifact_manifest(folder)}
            run.write_json(folder / 'complete.json', record)
            slurm = {'SLURM_JOB_ID': '20', 'SLURM_ARRAY_JOB_ID': '19', 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_ARRAY_TASK_COUNT': '3'}
            with patch.object(run, 'verify_plan', return_value=plan), patch.dict(os.environ, slurm), \
                 patch.object(run, 'run_owned_child') as launch:
                self.assertEqual(run.worker(output, output, 'smoke', 0)['input_fingerprints'], fp)
                launch.assert_not_called()
            def reseal(changed):
                changed['artifact_sha256'] = run.artifact_manifest(folder)
                changed['artifact_sha256'].pop('complete.json', None)
                (folder / 'complete.json').write_text(json.dumps(changed), encoding='utf-8')

            for key in ('pmse', 'signal_mse', 'ree', 'element_f1', 'exact_recovery'):
                with self.subTest(numerical_metric=key):
                    forged = copy.deepcopy(record)
                    forged['metrics']['A1'][key] += .25
                    reseal(forged)
                    with self.assertRaisesRegex(ValueError, 'metric'):
                        run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))
            with self.subTest(recovery='full_pool_flag'):
                forged = copy.deepcopy(record)
                forged['recovery_evidence']['truth_support_found_full_pool'] = False
                reseal(forged)
                with self.assertRaisesRegex(ValueError, 'recovery'):
                    run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))
            for key in ('B_true', 'S1'):
                with self.subTest(saved_truth=key):
                    forged = {k: v.copy() for k, v in truth.items()}
                    forged[key] += 1
                    with (folder / 'truth.npz').open('wb') as handle:
                        np.savez_compressed(handle, **forged)
                    reseal(copy.deepcopy(record))
                    with self.assertRaisesRegex(ValueError, 'truth|support'):
                        run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))
            with (folder / 'truth.npz').open('wb') as handle:
                np.savez_compressed(handle, **truth)
            for key, value in (('y', np.array([3., 6.])),
                               ('signal', np.array([2., np.nan, 6.])),
                               ('prediction', prediction['prediction'] + 1.),
                               ('arm_order', np.array(['A1', 'A0'])),
                               ('model_sha256', np.array(['wrong', 'wrong']))):
                with self.subTest(saved_prediction=key):
                    with (folder / 'test_predictions.npz').open('wb') as handle:
                        np.savez_compressed(handle, **dict(prediction, **{key: value}))
                    reseal(copy.deepcopy(record))
                    with self.assertRaisesRegex(ValueError, 'prediction|metric'):
                        run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))
            with (folder / 'test_predictions.npz').open('wb') as handle:
                np.savez_compressed(handle, **prediction)
            reseal(record)
            wrong_plan = copy.deepcopy(plan)
            wrong_plan['protocol']['source_files']['research/taskcard12/selection.py'] = 'wrong'
            wrong_record = dict(record, source_files=wrong_plan['protocol']['source_files'])
            (folder / 'complete.json').write_text(json.dumps(wrong_record), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pool source'):
                run.completed_result(folder, request, wrong_plan, run.file_sha(output / 'plan.json'))
            wrong_request = dict(request, config={'k_max': 99})
            (folder / 'complete.json').write_text(json.dumps(dict(record, request=wrong_request)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pool options'):
                run.completed_result(folder, wrong_request, plan, run.file_sha(output / 'plan.json'))
            (folder / 'complete.json').write_text(json.dumps(record), encoding='utf-8')
            forged = run.read_json(folder / 'selection.json')
            forged['A1']['selection']['selected_index'] = 1
            (folder / 'selection.json').write_text(json.dumps(forged), encoding='utf-8')
            record['artifact_sha256']['selection.json'] = run.file_sha(folder / 'selection.json')
            (folder / 'complete.json').write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'selection'):
                run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))
            with (folder / 'selected_models.npz').open('wb') as handle:
                np.savez_compressed(handle, A0_B=B + 1., A1_B=B)
            with self.assertRaisesRegex(ValueError, 'artifact'):
                run.completed_result(folder, request, plan, run.file_sha(output / 'plan.json'))

    def test_shell_syntax_and_mocked_formal_gate_blocks_submission(self):
        root = Path(__file__).resolve().parents[2]
        scripts = root / 'research/taskcard12/cluster'
        required = ['environment.sh', 'run_msic_pair.sh', 'submit_msic_pair.sh', 'merge_and_validate.sh']
        self.assertTrue(all((scripts / name).is_file() for name in required), 'Cluster SH files missing')
        bash = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')
        self.assertTrue(bash, 'Bash is required for shell validation')
        for name in required:
            subprocess.run([bash, '-n', str(scripts / name)], check=True, capture_output=True)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            target = work / 'research/taskcard12/cluster'
            target.mkdir(parents=True)
            for name in required:
                shutil.copyfile(scripts / name, target / name)
            mock = work / 'mock'
            mock.mkdir()
            for name in ('module', 'conda'):
                (mock / name).write_text('#!/bin/bash\nexit 0\n', encoding='utf-8')
            (mock / 'python').write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TASK12_MOCK_LOG"\n'
                '[[ "$*" != *check-gate* ]] || exit 9\nexit 0\n', encoding='utf-8')
            (mock / 'sbatch').write_text('#!/bin/bash\nprintf "SUBMITTED\\n" >> "$TASK12_MOCK_LOG"\n', encoding='utf-8')
            for path in mock.iterdir():
                path.chmod(0o755)
            log = work / 'commands.log'
            # Git Bash consumes POSIX PATH entries; convert only this local mock fixture.
            as_posix = lambda p: '/' + str(p).replace('\\', '/')[0].lower() + str(p).replace('\\', '/')[2:] if os.name == 'nt' else str(p)
            env = dict(os.environ, TASK12_ROOT=as_posix(work), TASK12_MOCK_LOG=as_posix(log))
            command = 'export PATH="' + as_posix(mock) + ':$PATH"; bash "$TASK12_ROOT/research/taskcard12/cluster/submit_msic_pair.sh" formal'
            result = subprocess.run([bash, '-c', command], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('check-gate', log.read_text())
            self.assertNotIn('SUBMITTED', log.read_text())


if __name__ == '__main__':
    unittest.main()
