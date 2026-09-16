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


if __name__ == '__main__':
    unittest.main()
