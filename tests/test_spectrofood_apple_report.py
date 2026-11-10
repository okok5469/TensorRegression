import importlib
import unittest
import numpy as np


class ReportContractTests(unittest.TestCase):
    def test_bootstrap_keeps_entity_repeat_block_and_direction(self):
        try:
            api = importlib.import_module('empirical_analysis.spectrofood_apple.report')
        except ModuleNotFoundError:
            self.fail('Apple reporting is not implemented')
        # n entities x repeats x models; every paired advantage exactly -1 MSE.
        errors = np.ones((10,3,2))
        errors[:,:,0] = 0
        result = api.paired_intervals(errors, ['M1','M0'], seed=13999, count=2000)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0]['mse_difference'], -1)
        self.assertAlmostEqual(result[0]['mse_ci_low'], -1)
        self.assertAlmostEqual(result[0]['mse_ci_high'], -1)
        self.assertEqual(result,api.paired_intervals(errors,['M1','M0'],seed=13999,count=2000))


if __name__ == '__main__':
    unittest.main()
