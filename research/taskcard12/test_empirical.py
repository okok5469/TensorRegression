"""Boundary tests for empirical data, isolation, paired reporting and gates."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class EmpiricalTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('research.taskcard12.empirical')
        self.assertIsNotNone(spec, 'empirical contract/runner implementation is required')
        return importlib.import_module('research.taskcard12.empirical')

    def data(self):
        rng = np.random.default_rng(820)
        return {'X': rng.normal(size=(8, 2, 2, 2)),
                'y': np.array([.1, .3, .7, 1.2, 1.7, 2.0, 2.8, 3.4]),
                'C': np.ones((8, 1)), 'subject_ids': np.array([f's{i}' for i in range(8)])}

    def splits(self):
        return [{'repeat': 0, 'outer_fold': 1, 'train': [4, 5, 6, 7], 'test': [0, 1, 2, 3]},
                {'repeat': 0, 'outer_fold': 2, 'train': [0, 1, 2, 3], 'test': [4, 5, 6, 7]}]

    def test_natural_tensor_finite_continuous_response_and_unique_ids(self):
        api, data = self.api(), self.data()
        api.validate_dataset(data)
        alterations = [('X', data['X'].reshape(8, -1)), ('X', data['X'].astype(str)),
                       ('y', np.zeros(8)), ('y', np.arange(8)[:, None]),
                       ('C', np.full((8, 1), np.nan)), ('subject_ids', np.array(['x'] * 8))]
        for key, value in alterations:
            bad = dict(data, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.validate_dataset(bad)

    def test_all_subjects_exactly_once_per_repeat_and_group_isolation(self):
        api, data, splits = self.api(), self.data(), self.splits()
        data['groups'] = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])
        api.validate_splits(data, splits, group_isolation=True)
        for change in ('overlap', 'missing', 'duplicate', 'group'):
            bad = copy.deepcopy(splits)
            altered = dict(data)
            if change == 'overlap':
                bad[0]['train'][0] = 0
            if change == 'missing':
                bad.pop()
            if change == 'duplicate':
                bad.append(copy.deepcopy(bad[0]))
            if change == 'group':
                altered['groups'] = np.array(['a', 'a', 'b', 'b', 'a', 'c', 'd', 'd'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                api.validate_splits(altered, bad, group_isolation=True)

    def test_one_search_and_training_only_preprocessing(self):
        api, data, split = self.api(), self.data(), self.splits()[0]
        from empirical_analysis.lemon.confounds import fit_preprocessor
        calls = []

        def solver(X, y, **options):
            calls.append((X.copy(), y.copy(), options))
            B = np.zeros(X.shape[1:])
            B[0, 0, 0] = y[0]
            model = {'B': B, 'q': 1, 'rss': float(np.sum((y - X[:, 0, 0, 0] * y[0])**2)),
                     'support': [[0], [0], [0]], 'sizes': [1, 1, 1],
                     'score': 1., 'selection': {'selected_index': 0}}
            return {'pool': {'test_pool': True}, 'original': model, 'fs': copy.deepcopy(model)}

        train, test = split['train'], split['test']
        result = api.fit_fold(data['X'][train], data['y'][train], data['C'][train],
                              data['X'][test], data['C'][test], {}, fit_preprocessor,
                              pair_solver=solver)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(result['predictions']['original'], result['predictions']['fs'])
        changed = dict(data, y=data['y'].copy())
        changed['y'][test] += 999
        again = api.fit_fold(changed['X'][train], changed['y'][train], changed['C'][train],
                            changed['X'][test], changed['C'][test], {}, fit_preprocessor,
                            pair_solver=solver)
        np.testing.assert_array_equal(calls[0][0], calls[1][0])
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        np.testing.assert_array_equal(result['predictions']['fs'], again['predictions']['fs'])

    def rows(self):
        rows = []
        for split in self.splits():
            for model, offset in [('original', 2.), ('fs', 1.), ('Covariates Only', .5)]:
                for index in split['test']:
                    y = self.data()['y'][index]
                    rows.append({'subject_id': f's{index}', 'repeat': 0,
                                 'outer_fold': split['outer_fold'], 'model': model,
                                 'y_true': float(y), 'y_pred': float(y + offset)})
        return rows

    def test_complete_oof_aggregation_and_baselines(self):
        api = self.api()
        summary = api.summarize_predictions(self.rows(), self.data(), self.splits(),
                                            ['original', 'fs', 'Covariates Only'])
        self.assertEqual(summary['models']['original']['mean_oof_rmse'], 2.)
        self.assertEqual(summary['models']['fs']['mean_oof_rmse'], 1.)
        self.assertEqual(summary['paired']['rmse_delta'], -1.)
        self.assertEqual(summary['paired']['improved_folds'], 2)
        self.assertFalse(summary['paired']['fs_beats_covariates_only'])
        self.assertEqual(len(summary['fold_metrics']), 6)

    def test_oof_rejects_missing_duplicate_wrong_fold_and_inconsistent_truth(self):
        api = self.api()
        for corruption in ('missing', 'duplicate', 'wrong_fold', 'wrong_truth'):
            rows = self.rows()
            if corruption == 'missing':
                rows.pop()
            if corruption == 'duplicate':
                rows.append(copy.deepcopy(rows[0]))
            if corruption == 'wrong_fold':
                rows[0]['outer_fold'] = 2
            if corruption == 'wrong_truth':
                rows[0]['y_true'] += .1
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                api.summarize_predictions(rows, self.data(), self.splits(),
                                          ['original', 'fs', 'Covariates Only'])

    def test_report_retains_every_adverse_fold(self):
        api, rows = self.api(), self.rows()
        for row in rows:
            if row['model'] == 'fs' and row['outer_fold'] == 2:
                row['y_pred'] = row['y_true'] + 5
        summary = api.summarize_predictions(rows, self.data(), self.splits(),
                                            ['original', 'fs', 'Covariates Only'])
        self.assertEqual(summary['paired']['improved_folds'], 1)
        self.assertEqual(summary['paired']['worsened_folds'], 1)
        self.assertEqual([r['outer_fold'] for r in summary['paired']['adverse_folds']], [2])

    def test_p0_protocol_and_source_hash_are_required_before_run(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'impl.py'
            source.write_text('frozen', encoding='utf-8')
            gate_path = base / 'gate.json'
            gate = {'passed': True, 'source_files': {'impl.py': api.file_sha(source)}, 'checks': {}}
            gate_path.write_text(json.dumps(gate), encoding='utf-8')
            plan = {'dataset': 'fixture'}
            protocol = {'status': 'FROZEN', 'plan': plan, 'source_files': gate['source_files'],
                        'p0_gate_sha256': api.file_sha(gate_path)}
            api.verify_authorization(protocol, plan, gate_path, base)
            for alteration in ('gate_failed', 'source_changed', 'draft', 'different_plan'):
                edited = copy.deepcopy(protocol)
                use_plan = plan
                if alteration == 'gate_failed':
                    gate_path.write_text(json.dumps(dict(gate, passed=False)), encoding='utf-8')
                    edited['p0_gate_sha256'] = api.file_sha(gate_path)
                if alteration == 'source_changed':
                    source.write_text('changed', encoding='utf-8')
                if alteration == 'draft':
                    edited['status'] = 'DRAFT'
                if alteration == 'different_plan':
                    use_plan = {'dataset': 'other'}
                with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                    api.verify_authorization(edited, use_plan, gate_path, base)
                source.write_text('frozen', encoding='utf-8')
                gate_path.write_text(json.dumps(gate), encoding='utf-8')

    def test_historical_original_identity_is_strict(self):
        api = self.api()
        B = np.arange(8.).reshape(2, 2, 2)
        model = {'B': B, 'support': [[0, 1], [0, 1], [0, 1]], 'q': 8, 'score': 4.}
        expected = {'B': B.copy(), 'support': copy.deepcopy(model['support']), 'q': 8, 'score': 4.}
        api.verify_original(model, expected, np.array([1., 2.]), np.array([1., 2.]))
        expected['B'][0, 0, 0] += 1e-10
        with self.assertRaises(ValueError):
            api.verify_original(model, expected, np.array([1., 2.]), np.array([1., 2.]))

    def test_missing_empirical_data_is_explicit_not_run(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'prepare'), 'Dataset preparation entry point required')
        for name in ('hcp_ya', 'nki_rs'):
            config = Path(__file__).parent / 'empirical_configs' / (name + '.json')
            self.assertTrue(config.is_file(), 'Explicit missing-data contract required')
            with self.assertRaisesRegex(ValueError, 'NOT_RUN'):
                api.prepare(config)

    def test_lemon_readonly_plan_binds_every_fold_and_baseline(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'prepare'), 'Dataset preparation entry point required')
        bundle = api.prepare(Path(__file__).parent / 'empirical_configs/lemon.json')
        self.assertEqual(bundle['plan']['n'], 145)
        self.assertEqual(bundle['plan']['dimensions'], [61, 16, 5])
        self.assertEqual(len(bundle['splits']), 15)
        self.assertEqual(len(bundle['baseline_rows']), 1740)
        self.assertEqual(len(bundle['plan']['preprocessing']), 15)
        self.assertEqual(bundle['plan']['native_options']['cv_folds'], 3)
        self.assertEqual(bundle['plan']['native_options']['max_cycles'], 6)
        self.assertTrue(bundle['plan']['baseline_files'])
        for relative in ('empirical_analysis/__init__.py', 'numerical_simulation/__init__.py',
                         'doc/algorithm_versions/v5.1-bcd-fix/VERSION.json'):
            self.assertIn(relative, bundle['source_files'])
        self.assertEqual(bundle['plan']['response'], {'name': 'TMT_5', 'scale': 'natural log',
                                                     'continuous': True})
        self.assertTrue(all(len(s['train']) == 116 and len(s['test']) == 29 for s in bundle['splits']))

    def test_selection_is_saved_before_any_heldout_prediction(self):
        api, data, split = self.api(), self.data(), self.splits()[0]
        events = []

        class Preprocessor:
            def apply(self, X, C):
                return X
            def transform_response(self, y, C):
                return y
            def predict(self, B, X, C):
                events.append('predict')
                return X[:, 0, 0, 0] * B[0, 0, 0]

        def solver(X, y, **options):
            events.append('search')
            model = {'B': np.ones(X.shape[1:])}
            return {'pool': {}, 'original': model, 'fs': model}

        tr, te = split['train'], split['test']
        api.fit_fold(data['X'][tr], data['y'][tr], data['C'][tr],
                     data['X'][te], data['C'][te], {}, lambda *args: Preprocessor(),
                     pair_solver=solver, on_selected=lambda *args: events.append('save_selected'))
        self.assertEqual(events[:3], ['search', 'save_selected', 'predict'])

    def test_runner_requires_gate_and_writes_complete_paired_artifacts(self):
        api, data, splits = self.api(), self.data(), self.splits()
        self.assertTrue(hasattr(api, 'run'), 'Gated complete empirical runner required')
        from empirical_analysis.lemon.confounds import fit_preprocessor
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / 'impl.py').write_text('fixed', encoding='utf-8')
            sources = {'impl.py': api.file_sha(repository / 'impl.py')}
            config = {'output_dir': 'result', 'protocol': 'protocol.json', 'p0_gate': 'gate.json'}
            config_path = repository / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            plan = {'dataset': 'fixture', 'native_options': {}, 'n': 8, 'dimensions': [2, 2, 2]}
            gate = {'passed': False, 'source_files': sources}
            gate_path = repository / 'gate.json'
            gate_path.write_text(json.dumps(gate), encoding='utf-8')
            protocol = {'status': 'FROZEN', 'plan': plan, 'source_files': sources,
                        'p0_gate_sha256': api.file_sha(gate_path)}
            protocol_path = repository / 'protocol.json'
            protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
            prepared = {(s['repeat'], s['outer_fold']): fit_preprocessor(
                data['X'][s['train']], data['y'][s['train']], data['C'][s['train']]) for s in splits}
            bundle = {'plan': plan, 'source_files': sources, 'config': config, 'data': data,
                      'splits': splits, 'prepared': prepared, 'official': None,
                      'baseline_rows': [r for r in self.rows() if r['model'] == 'Covariates Only']}
            calls = []

            def solver(X, y, **options):
                calls.append(1)
                B = np.zeros(X.shape[1:])
                choice = {'selected_index': 0, 'minimum_q': 1, 'minimum_score': 1., 'tolerance': .1}
                model = {'B': B, 'support': [[0], [0], [0]], 'sizes': [1, 1, 1], 'q': 1,
                         'rss': float(y @ y), 'score': 1., 'B_sha256': api.array_sha(B),
                         'selection': choice}
                pool = {'digest': 'fixture', 'metadata': {'trace': {'candidates': [model]},
                        'counters': {'native_search_calls': 1, 'extra_search_calls': 0},
                        'audit': {'passed': True}}}
                return {'pool': pool, 'original': model, 'fs': copy.deepcopy(model)}

            def saver(path, pool):
                Path(path).mkdir()
                (Path(path) / 'fixture.txt').write_text(pool['digest'], encoding='utf-8')

            with patch.object(api, 'root', return_value=repository), \
                 patch.object(api, 'prepare', return_value=bundle), \
                 patch.object(api.selection, 'fit_pair', side_effect=solver), \
                 patch.object(api.selection, 'save_pool', side_effect=saver):
                with self.assertRaises(ValueError):
                    api.run(config_path)
                self.assertEqual(calls, [])
                self.assertFalse((repository / 'result').exists())
                gate['passed'] = True
                gate_path.write_text(json.dumps(gate), encoding='utf-8')
                protocol['p0_gate_sha256'] = api.file_sha(gate_path)
                protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
                summary = api.run(config_path)
                self.assertEqual(len(calls), 2)
                self.assertEqual(summary['prediction_count'], 24)
                self.assertEqual(summary['native_search_calls'], 2)
                self.assertEqual(summary['extra_search_calls'], 0)
                for filename in ('summary.json', 'oof_predictions.csv', 'fold_metrics.csv',
                                 'repeat_metrics.csv', 'selection_changes.csv', 'seal.json'):
                    self.assertTrue((repository / 'result' / filename).is_file())
                with self.assertRaises(ValueError):
                    api.run(config_path)

    def test_generic_contract_keeps_native_response_and_rejects_wrong_baseline_identity(self):
        api, data, splits = self.api(), self.data(), self.splits()
        from empirical_analysis.lemon.confounds import fit_preprocessor
        prepared = {(s['repeat'], s['outer_fold']): fit_preprocessor(
            data['X'][s['train']], data['y'][s['train']], data['C'][s['train']]) for s in splits}
        with tempfile.TemporaryDirectory(dir=api.root() / 'doc/taskcard12/verification') as temporary:
            directory = Path(temporary)
            relative = lambda path: path.relative_to(api.root()).as_posix()
            np.savez(directory / 'data.npz', **data)
            api.write_json(directory / 'splits.json', splits)
            api.write_csv(directory / 'baseline.csv', [r for r in self.rows() if r['model'] == 'Covariates Only'])
            adapter = 'empirical_analysis/lemon/confounds.py'
            digest = api.file_sha(api.root() / adapter)
            manifest = {'data_array_sha256': api.array_identities(data), 'splits': splits,
                        'preprocessing': api.preprocessing_identities(prepared),
                        'preprocessor_sha256': digest, 'preprocessing_dependencies': {adapter: digest},
                        'files': {relative(directory / 'baseline.csv'): api.file_sha(directory / 'baseline.csv')},
                        'configuration': {'purpose': 'contract test only'}, 'source_files': {adapter: digest}}
            api.write_json(directory / 'baseline_manifest.json', manifest)
            config = {'dataset': 'NKI-RS', 'response': {'name': 'fixture_response', 'scale': 'fixture_native',
                      'continuous': True}, 'tensor_axes': ['first', 'second', 'third'], 'covariate_names': ['intercept'],
                      'dataset_npz': relative(directory / 'data.npz'), 'splits_json': relative(directory / 'splits.json'),
                      'preprocessor_file': adapter, 'preprocessor_sha256': digest,
                      'preprocessing_dependencies': {adapter: digest}, 'group_isolation': False,
                      'baseline_predictions_csv': relative(directory / 'baseline.csv'),
                      'baseline_manifest_json': relative(directory / 'baseline_manifest.json'),
                      'baseline_models': ['Covariates Only'], 'native_options': {}}
            try:
                bundle = api.prepare_generic(config)
            except Exception as error:
                self.fail(f'Valid documented adapter contract must load: {error!r}')
            self.assertEqual(bundle['plan']['response']['scale'], 'fixture_native')
            self.assertEqual(len(bundle['baseline_rows']), 8)
            manifest['data_array_sha256']['y'] = 'incorrect'
            (directory / 'baseline_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identical inputs'):
                api.prepare_generic(config)


if __name__ == '__main__':
    unittest.main()
