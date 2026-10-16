"""Real saved-OOF checks for the task-card-6 analysis tool; no fitting."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


class LemonAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        script = self.root / 'scripts/taskcard6/lemon_analysis.py'
        self.assertTrue(script.is_file(), 'The independent analysis tool is missing')
        spec = importlib.util.spec_from_file_location('lemon_analysis', script)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.old = self.root / 'empirical_analysis/lemon/outputs_eeg_incremental'

    def test_saved_complete_run_has_75_fits_and_2175_predictions(self):
        run = self.tool.load_run(self.old)
        self.assertEqual(len(run['oof']), 2175)
        self.assertEqual(len(run['folds']), 75)

    def test_duplicate_oof_rejects_a_missing_subject_even_at_correct_count(self):
        run = self.tool.load_run(self.old)
        rows = copy.deepcopy(run['oof'])
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.tool.validate_oof(rows, run['splits'], run['config']['models'])

    def test_oof_fold_drift_is_rejected(self):
        run = self.tool.load_run(self.old)
        rows = copy.deepcopy(run['oof'])
        rows[0]['outer_fold'] = 5
        with self.assertRaisesRegex(ValueError, 'fold'):
            self.tool.validate_oof(rows, run['splits'], run['config']['models'])

    def test_pairing_changed_response_is_rejected(self):
        run = self.tool.load_run(self.old)
        other = copy.deepcopy(run)
        other['oof'][0]['y_true'] = float(other['oof'][0]['y_true']) + .1
        with self.assertRaisesRegex(ValueError, 'response'):
            self.tool.validate_pair(run, other)

    def test_missing_new_results_never_produce_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing'):
                self.tool.load_run(Path(directory))

    def test_same_saved_run_gives_zero_paired_rmse_change(self):
        run = self.tool.load_run(self.old)
        result = self.tool.compare_runs(run, run)
        self.assertEqual(result['paired_abess']['mean_repeat_rmse_new_minus_old'], 0)
        self.assertEqual(result['paired_abess']['subject_cluster_ci_95'], [0, 0])

    def test_pair_exports_all_15_capacity_diagnostics(self):
        run = self.tool.load_run(self.old)
        result = self.tool.compare_runs(run, run)
        self.assertEqual(len(result['paired_folds']), 15)
        self.assertTrue(all(row['new_Q'] == 81 for row in result['paired_folds']))
        self.assertTrue(all(row['support_jaccard_modes'] == [1, 1, 1]
                            for row in result['paired_folds']))

    def test_saved_original_core_cannot_be_presented_as_d2(self):
        run = self.tool.load_run(self.old)
        with self.assertRaisesRegex(ValueError, 'core'):
            self.tool.verify_core(run, self.root / 'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json')

    def test_saved_original_core_matches_its_historical_manifest(self):
        run = self.tool.load_run(self.old)
        self.tool.verify_core(run, self.root / 'doc/algorithm_versions/v5-original/VERSION.json')


if __name__ == '__main__':
    unittest.main()
