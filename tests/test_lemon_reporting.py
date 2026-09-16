"""Internal-report tests use only labelled synthetic fixtures in temporary folders."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from empirical_analysis.lemon.io import write_json


class LemonReportingTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('empirical_analysis.lemon.reporting')
        except ModuleNotFoundError as exc:
            self.fail(f'Reporting module has not been implemented: {exc}')

    def fixture(self, out):
        """SYNTHETIC FIXTURE ONLY: these are not LEMON observations or results."""
        for folder in ('diagnostics', 'qc', 'tensors/subjects', 'cv', 'final_model'):
            (out / folder).mkdir(parents=True)
        ids = [f'synthetic-{index:03d}' for index in range(8)]
        raw = np.array([30, 32, 34, 36, 38, 40, np.nan, 44, 46])
        pd.DataFrame({'subject_id': ids + ['synthetic-extra'], 'tmt5_raw': raw,
                      'tmt5_log': np.log(raw)}).to_csv(out / 'diagnostics/tmt_clean.csv', index=False)
        pd.DataFrame({'subject_id': ids, 'file_complete': [True] * 7 + [False],
                      'exclusion_reason': [''] * 7 + ['missing_vmrk']}).to_csv(out / 'diagnostics/eeg_inventory.csv', index=False)
        pd.DataFrame({'subject_id': ids, 'include_primary': [True] * 7 + [False],
                      'exclusion_reason': [''] * 7 + ['missing_vmrk']}).to_csv(out / 'qc/subject_qc.csv', index=False)
        pd.DataFrame({'subject_id': ids, 'include_primary': [True] * 6 + [False] * 2,
                      'exclusion_reason': [''] * 6 + ['invalid_or_missing_tmt5', 'missing_vmrk']}).to_csv(out / 'qc/final_subject_manifest.csv', index=False)
        X = np.random.default_rng(12).normal(size=(6, 2, 2, 2))
        y = np.log(raw[:6])
        np.savez(out / 'qc/analysis_dataset.npz', X=X, y=y, subject_ids=np.asarray(ids[:6]))
        meta = {'shape': [7, 2, 2, 2], 'channels': ['Synthetic-C1', 'Synthetic-C2'],
                'bands': ['delta', 'theta'], 'window_sec': 25, 'time_windows': 2}
        write_json(out / 'tensors/tensor_meta.json', meta)
        for index, subject in enumerate(ids[:6]):
            write_json(out / 'tensors/subjects' / (subject + '_meta.json'), {
                'windows': [{'window_index': 0, 'start_sec': 10 + index, 'end_sec': 35 + index, 'source_ec_segment': 0},
                            {'window_index': 1, 'start_sec': 70 + 2 * index, 'end_sec': 95 + 2 * index, 'source_ec_segment': 1}]})
        predictions, metrics, summaries = [], [], []
        models = ('Tensor-ABESS', 'Tensor-GLasso', 'Vector Lasso')
        for model_index, model in enumerate(models):
            predicted = y + np.array([-1, 1, -1, 1, -1, 1]) * (0.01 + model_index * 0.01)
            for index, subject in enumerate(ids[:6]):
                predictions.append({'subject_id': subject, 'fold': index // 2 + 1, 'model': model,
                                    'observed': y[index], 'predicted': predicted[index], 'response_scale': 'log'})
            for fold in range(1, 4):
                selected = np.arange(6) // 2 + 1 == fold
                residual = y[selected] - predicted[selected]
                score = {'rmse': float(np.sqrt(np.mean(residual ** 2))), 'mae': float(np.mean(np.abs(residual))),
                         'r2': float(1 - np.sum(residual ** 2) / np.sum((y[selected] - y[selected].mean()) ** 2))}
                metrics.append({'fold': fold, 'model': model, 'n_train': 4, 'n_test': 2,
                                'response_scale': 'log', **score})
            rows = [row for row in metrics if row['model'] == model]
            row = {'model': model, 'n_subjects': 6, 'n_folds': 3, 'mode': 'pilot', 'response_scale': 'log'}
            for metric in ('rmse', 'mae', 'r2'):
                row[metric + '_mean'] = float(np.mean([item[metric] for item in rows]))
                row[metric + '_std'] = float(np.std([item[metric] for item in rows], ddof=1))
            residual = y - predicted
            row.update(oof_rmse=float(np.sqrt(np.mean(residual ** 2))), oof_mae=float(np.mean(np.abs(residual))),
                       oof_r2=float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)))
            summaries.append(row)
        pd.DataFrame(predictions).to_csv(out / 'cv/oof_predictions.csv', index=False)
        pd.DataFrame(metrics).to_csv(out / 'cv/cv_metrics.csv', index=False)
        pd.DataFrame(summaries).to_csv(out / 'cv/cv_summary.csv', index=False)
        for axis in ('channel', 'time', 'frequency'):
            pd.DataFrame([{'model': model, 'index': index, 'selection_count': 2 + index,
                           'n_folds': 3, 'selection_frequency': (2 + index) / 3}
                          for model in models for index in range(2)]).to_csv(out / 'cv' / (axis + '_selection_frequency.csv'), index=False)
        B = np.zeros((2, 2, 2))
        B[0, 1, 0], B[1, 1, 1] = 1.0, 0.5
        np.save(out / 'final_model/B_hat.npy', B)
        for axis, support in (('channel', [0, 1]), ('time', [1]), ('frequency', [0, 1])):
            write_json(out / 'final_model' / ('S_' + axis + '.json'), support)
        write_json(out / 'final_model/support_labels.json', {'channels': meta['channels'], 'time_indices': [1], 'frequency_bands': meta['bands']})
        write_json(out / 'final_model/final_model_summary.json', {
            'subject_ids': ids[:6], 'tensor_shape': [6, 2, 2, 2], 'response_scale': 'log', 'mode': 'pilot',
            'q': 4, 'independent_test_performance': False, 'fixture': 'SYNTHETIC_FIXTURE_ONLY'})
        return {'response': {'primary_transform': 'log'}, 'analysis': {'mode': 'pilot'},
                'report': {'channel_plot': 'barplot'},
                'data': {'eeg_root': 'SYNTHETIC_FIXTURE_ONLY', 'tmt_csv': 'SYNTHETIC_FIXTURE_ONLY'}}

    def test_report_outputs_and_true_window_times(self):
        mod = self.module()
        with tempfile.TemporaryDirectory(prefix='synthetic_lemon_report_') as tmp:
            out = Path(tmp)
            config = self.fixture(out)
            result = mod.make_report(config, out)
            json.dumps(result, allow_nan=False)
            names = ('tmt5_raw_hist', 'tmt5_log_hist', 'sample_flow', 'cv_model_comparison',
                     'cv_r2_comparison', 'selected_frequency_bands', 'selected_time_windows',
                     'channel_barplot', 'coefficient_summary', 'predicted_vs_observed_oof')
            for name in names:
                for suffix in ('.png', '.pdf'):
                    self.assertGreater((out / 'figures' / (name + suffix)).stat().st_size, 100)
            self.assertEqual(result['sample_counts']['eeg_subjects'], 8)
            self.assertEqual(result['sample_counts']['tmt_rows'], 9)
            self.assertEqual(result['sample_counts']['final_subjects'], 6)
            onsets = pd.read_csv(out / 'final_model/selected_time_window_onsets.csv')
            self.assertEqual(onsets.loc[0, 'window_index'], 1)
            self.assertEqual(onsets.loc[0, 'n_subjects'], 6)
            self.assertEqual(onsets.loc[0, 'start_mean_sec'], 75)
            self.assertEqual(onsets.loc[0, 'start_min_sec'], 70)
            self.assertEqual(onsets.loc[0, 'start_max_sec'], 80)
            self.assertEqual(onsets.loc[0, 'end_mean_sec'], 100)
            summary = (out / 'RUN_SUMMARY.md').read_text(encoding='utf-8')
            for text in ('SYNTHETIC_FIXTURE_ONLY', 'pilot', 'log', 'missing_vmrk', 'SHA256', 'barplot'):
                self.assertIn(text, summary)
            manifest = json.loads((out / 'figure_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['figures']), 10)
            self.assertEqual(manifest['purpose'], 'internal_diagnostics')

    def test_mixed_response_scales_stop_before_drawing(self):
        mod = self.module()
        with tempfile.TemporaryDirectory(prefix='synthetic_lemon_report_') as tmp:
            out = Path(tmp)
            config = self.fixture(out)
            predictions = pd.read_csv(out / 'cv/oof_predictions.csv')
            predictions.loc[0, 'response_scale'] = 'raw'
            predictions.to_csv(out / 'cv/oof_predictions.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'scale'):
                mod.make_report(config, out)
            self.assertFalse((out / 'figures').exists())

    def test_missing_inputs_and_unsupported_channel_map_fail(self):
        mod = self.module()
        with tempfile.TemporaryDirectory(prefix='synthetic_lemon_report_') as tmp:
            out = Path(tmp)
            config = self.fixture(out)
            config['report']['channel_plot'] = 'scalp'
            with self.assertRaisesRegex(ValueError, 'barplot'):
                mod.make_report(config, out)
            config['report']['channel_plot'] = 'barplot'
            (out / 'cv/cv_metrics.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                mod.make_report(config, out)
            self.assertFalse((out / 'figures').exists())


if __name__ == '__main__':
    unittest.main()
