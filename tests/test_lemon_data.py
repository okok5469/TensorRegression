from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class LemonDataTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module('empirical_analysis.lemon.' + name)
        except ModuleNotFoundError as exc:
            self.fail('LEMON module is not implemented: ' + str(exc))

    def test_config_paths_are_relative_to_config_not_cwd(self):
        config = self.module('config')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('seed: 2\ndata:\n  eeg_root: data\n  tmt_csv: tmt.csv\noutput_dir: out\n', encoding='utf-8')
            result = config.load_config(path)
            self.assertEqual(result['paths']['eeg_root'], Path(tmp) / 'data')
            self.assertEqual(result['paths']['output_dir'], Path(tmp) / 'out')

    def test_tmt_preserves_invalid_value_and_subject_id(self):
        data = self.module('dataset')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'TMT.csv'
            source.write_text('ID,TMT_5\nsub-010001,42.5\nsub-010002,"67,75"\n', encoding='utf-8')
            clean = data.prepare_tmt(source, root / 'out')
            self.assertEqual(clean.loc[0, 'subject_id'], 'sub-010001')
            self.assertAlmostEqual(clean.loc[0, 'tmt5_log'], np.log(42.5))
            self.assertTrue(np.isnan(clean.loc[1, 'tmt5_raw']))
            invalid = pd.read_csv(root / 'out/invalid_tmt5_rows.csv')
            self.assertEqual(invalid.loc[0, 'original_value'], '67,75')

    def test_tmt_duplicate_or_nonpositive_stops(self):
        data = self.module('dataset')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'TMT.csv'
            for content in ('ID,TMT_5\na,4\na,5\n', 'ID,TMT_5\na,0\n'):
                source.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    data.prepare_tmt(source, root / 'out')

    def test_inventory_uses_manifest_and_detects_wrong_size(self):
        data = self.module('dataset')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / 'sub-1/eeg'
            eeg.mkdir(parents=True)
            (eeg / 'sub-1_task-resting_eeg.eeg').write_bytes(b'12')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([{'path': 'sub-1/eeg/sub-1_task-resting_eeg.eeg', 'size': 3}, {'path': 'sub-2/eeg/sub-2_task-resting_eeg.eeg', 'size': 3}]), encoding='utf-8')
            frame = data.inventory_eeg(root, manifest)
            self.assertEqual(frame.subject_id.tolist(), ['sub-1', 'sub-2'])
            self.assertFalse(bool(frame.loc[0, 'file_complete']))
            self.assertIn('size_mismatch', frame.loc[0, 'exclusion_reason'])
            self.assertIn('missing', frame.loc[1, 'exclusion_reason'])

    def test_analysis_merge_respects_tensor_order_and_drops_invalid_response(self):
        data = self.module('dataset')
        X = np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2).astype(float)
        tmt = pd.DataFrame({'subject_id': ['c', 'b', 'a'], 'tmt5_raw': [4., np.nan, 2.], 'tmt5_log': [np.log(4.), np.nan, np.log(2.)]})
        result = data.merge_analysis_arrays(X, ['a', 'b', 'c'], tmt, None, {'analysis': {'mode': 'pilot'}, 'response': {'primary_transform': 'log'}})
        self.assertEqual(result['subject_ids'], ['a', 'c'])
        np.testing.assert_array_equal(result['X'], X[[0, 2]])
        np.testing.assert_allclose(result['y'], np.log([2., 4.]))
        self.assertIsNone(result['C'])
        self.assertEqual(result['covariate_columns'], [])

    def test_adjusted_mode_requires_explicit_covariates(self):
        data = self.module('dataset')
        tmt = pd.DataFrame({'subject_id': ['a'], 'tmt5_raw': [4.], 'tmt5_log': [np.log(4.)]})
        with self.assertRaisesRegex(ValueError, 'participants'):
            data.merge_analysis_arrays(np.ones((1, 2, 2, 2)), ['a'], tmt, None, {'analysis': {'mode': 'adjusted'}, 'response': {'primary_transform': 'log'}})

    def test_age_band_is_not_silently_treated_as_missing_numeric_age(self):
        data = self.module('dataset')
        tmt = pd.DataFrame({'subject_id': ['a'], 'tmt5_raw': [4.], 'tmt5_log': [np.log(4.)]})
        people = pd.DataFrame({'participant_id': ['a'], 'age': ['20-25'], 'sex': ['M']})
        cfg = {'analysis': {'mode': 'adjusted'}, 'response': {'primary_transform': 'log'},
               'confounds': {'id_column': 'participant_id', 'age_column': 'age', 'sex_column': 'sex', 'sex_mapping': {'M': 0, 'F': 1}}}
        with self.assertRaisesRegex(ValueError, 'age'):
            data.merge_analysis_arrays(np.ones((1, 2, 2, 2)), ['a'], tmt, people, cfg)

    def merge_covariates(self, ages, sexes=None, **age_spec):
        ids = ['s' + str(i) for i in range(len(ages))]
        tmt = pd.DataFrame({'subject_id': ids, 'tmt5_raw': 4., 'tmt5_log': np.log(4.)})
        people = pd.DataFrame({'participant_id': ids, 'age': ages, 'sex': sexes or ['F'] * len(ids)})
        cfg = {'analysis': {'mode': 'adjusted'}, 'response': {'primary_transform': 'log'},
               'confounds': {'id_column': 'participant_id', 'age_column': 'age', 'sex_column': 'sex',
                             'sex_mapping': {'F': 1, 'M': 0}, **age_spec}}
        return self.module('dataset').merge_analysis_arrays(np.ones((len(ids), 2, 2, 2)), ids, tmt, people, cfg)

    def test_categorical_age_uses_explicit_reference_and_preserves_source(self):
        result = self.merge_covariates(['25-30', '20-25', '30-35'], ['M', 'F', 'F'],
                                      age_encoding='categorical', age_categories=['20-25', '25-30', '30-35'],
                                      age_reference='25-30')
        np.testing.assert_array_equal(result['C'], [[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1]])
        self.assertEqual(result['covariate_columns'], ['intercept', 'age[20-25]', 'age[30-35]', 'sex'])
        self.assertEqual(result['manifest'].age_source.tolist(), ['25-30', '20-25', '30-35'])
        self.assertEqual(result['manifest'].age_category.tolist(), ['25-30', '20-25', '30-35'])

    def test_explicit_age_mapping_merges_only_listed_source_labels(self):
        spec = dict(age_encoding='categorical', age_categories=['20-25', '30-40'], age_reference='20-25',
                    age_mapping={'20-25': '20-25', '30-35': '30-40', '35-40': '30-40'})
        result = self.merge_covariates(['20-25', '35-40', '30-35'], **spec)
        np.testing.assert_array_equal(result['C'][:, 1], [0, 1, 1])
        self.assertEqual(result['manifest'].age_source.tolist(), ['20-25', '35-40', '30-35'])
        self.assertEqual(result['manifest'].age_category.tolist(), ['20-25', '30-40', '30-40'])
        with self.assertRaisesRegex(ValueError, 'Unknown age'):
            self.merge_covariates(['30-40'], **spec)

    def test_categorical_missing_covariates_are_excluded_not_reference_encoded(self):
        result = self.merge_covariates(['20-25', 'n/a', None, '', '25-30'], ['F', 'F', 'M', 'F', 'n/a'],
                                      age_encoding='categorical', age_categories=['20-25', '25-30'], age_reference='20-25')
        self.assertEqual(result['subject_ids'], ['s0'])
        self.assertEqual(result['C'].shape, (1, 3))
        self.assertTrue(result['manifest'].loc[1:, 'exclusion_reason'].str.contains('missing_covariates').all())

    def test_categorical_unknown_age_and_sex_stop(self):
        spec = dict(age_encoding='categorical', age_categories=['20-25', '25-30'], age_reference='20-25')
        with self.assertRaisesRegex(ValueError, 'Unknown age'):
            self.merge_covariates(['not-known'], **spec)
        with self.assertRaisesRegex(ValueError, 'Unknown sex'):
            self.merge_covariates(['20-25'], ['unknown'], **spec)

    def test_categorical_age_requires_valid_explicit_design(self):
        valid = dict(age_encoding='categorical', age_categories=['20-25', '25-30'], age_reference='20-25')
        variants = [dict(age_categories=None), dict(age_categories=['20-25']),
                    dict(age_categories=['20-25', '20-25']), dict(age_reference=None),
                    dict(age_reference='other'), dict(age_categories=['20-25', 'n/a']),
                    dict(age_mapping={'20-25': 'other'}), dict(age_mapping={'n/a': '20-25'}),
                    dict(age_encoding='ordinal')]
        for override in variants:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'age'):
                self.merge_covariates(['20-25'], **(valid | override))

    def test_numeric_age_stays_compatible_and_reports_design_columns(self):
        result = self.merge_covariates(['23', '65'], ['F', 'M'])
        np.testing.assert_array_equal(result['C'], [[1, 23, 1], [1, 65, 0]])
        self.assertEqual(result['covariate_columns'], ['intercept', 'age', 'sex'])


if __name__ == '__main__':
    unittest.main()
