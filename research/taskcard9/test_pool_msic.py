import unittest
import numpy as np

from research.taskcard9.pool_msic import score_select, support_metrics, sparse_predictions


class PoolResearchTests(unittest.TestCase):
    def test_one_se_uses_q_at_argmin_and_prefers_small_model(self):
        pool = [{'sizes': [1, 1, 1], 'rss': 100.}, {'sizes': [2, 2, 2], 'rss': 75.}]
        direct = score_select(pool, 100, [32, 20, 25], 1.5, 0.)
        band = score_select(pool, 100, [32, 20, 25], 1.5, 1.)
        self.assertEqual(direct['selected_index'], 1)
        self.assertEqual(band['selected_index'], 0)
        self.assertAlmostEqual(band['tolerance'], np.sqrt(8 * np.log(100)))

    def test_exact_ties_preserve_candidate_order(self):
        candidate = {'sizes': [1, 1, 1], 'rss': 100.}
        result = score_select([candidate, dict(candidate)], 100, [32, 20, 25], 2., 0.)
        self.assertEqual(result['selected_index'], 0)
        self.assertEqual(result['eligible_indices'], [0, 1])

    def test_sparse_pool_prediction_matches_rectangular_dense_model(self):
        X = np.arange(48.).reshape(3, 2, 2, 4)
        Bs = np.zeros((2, 2, 2, 4))
        Bs[0, 1, 0, 3] = 2.
        Bs[1, 0, 1, 2] = -3.
        actual = sparse_predictions(X, Bs)
        np.testing.assert_array_equal(actual, np.column_stack((2*X[:,1,0,3], -3*X[:,0,1,2])))

    def test_q_under_is_not_support_exactness(self):
        result = support_metrics([[0, 1], [0], [2]], [[2, 3], [0], [2]])
        self.assertEqual(result['exact_recovery'], 0)
        self.assertAlmostEqual(result['mode_f1_macro'], 2/3)
        self.assertEqual(result['element_f1'], 0.)


if __name__ == '__main__':
    unittest.main()
