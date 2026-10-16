import importlib
import unittest


class SimulationAnalysisTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('scripts.taskcard6.simulation_analysis')
        except ModuleNotFoundError:
            self.fail('Saved-result analysis is missing')

    def test_recovery_and_prediction_transitions_are_separate(self):
        classify = self.api().classify
        self.assertEqual(classify(0, 1, 1.), 'gained_exact_worse_pmse')
        self.assertEqual(classify(0, 1, -1.), 'gained_exact_better_pmse')
        self.assertEqual(classify(1, 0, -1.), 'lost_exact_better_pmse')
        self.assertEqual(classify(1, 1, 0.), 'same_exact_equal_pmse')

    def test_positive_tail_share_and_net_contribution_have_different_denominators(self):
        result = self.api().tail_summary([4., 1., -3., 0.])
        self.assertEqual(result['positive_sum'], 5.)
        self.assertEqual(result['negative_sum'], -3.)
        self.assertEqual(result['net_sum'], 2.)
        self.assertEqual(result['top_1_positive_share'], .8)
        self.assertEqual(result['top_1_net_share'], 2.)

    def test_zero_effect_has_no_invented_tail_share(self):
        result = self.api().tail_summary([0., 0.])
        self.assertIsNone(result['top_1_positive_share'])
        self.assertIsNone(result['top_1_net_share'])


if __name__ == '__main__':
    unittest.main()
