import unittest

import numpy as np

from tensor_abess.tensor.abess import tensor_abess
from tensor_abess.utils.init import tensor_init_glasso


class TensorAdaptiveTests(unittest.TestCase):
    @staticmethod
    def _singleton_tensor_problem() -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((6, 2, 2, 2), dtype=float)
        X[:, 0, 0, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = 2.0 * X[:, 0, 0, 0]
        return X, y

    def test_tensor_init_glasso_returns_singleton_warm_supports_and_bounds(self) -> None:
        X, y = self._singleton_tensor_problem()

        init = tensor_init_glasso(X, y, c=2.0, criterion="bic", n_lambdas=8, max_iter=200)

        self.assertTrue(np.array_equal(init.S1_init, np.array([0])))
        self.assertTrue(np.array_equal(init.S2_init, np.array([0])))
        self.assertTrue(np.array_equal(init.S3_init, np.array([0])))
        self.assertEqual((init.s1_max, init.s2_max, init.s3_max), (1, 1, 1))

    def test_tensor_abess_runs_revised_adaptive_pipeline_end_to_end(self) -> None:
        X, y = self._singleton_tensor_problem()

        result = tensor_abess(
            X,
            y,
            c=2.0,
            criterion="bic",
            n_lambdas=8,
            max_iter=200,
            max_cycles=5,
            m_max=5,
            eps=1e-12,
        )

        self.assertTrue(np.array_equal(result.S1, np.array([0])))
        self.assertTrue(np.array_equal(result.S2, np.array([0])))
        self.assertTrue(np.array_equal(result.S3, np.array([0])))
        self.assertEqual((result.s1, result.s2, result.s3), (1, 1, 1))
        self.assertEqual(result.s_max, (1, 1, 1))
        self.assertGreaterEqual(len(result.sic_history), 1)


if __name__ == "__main__":
    unittest.main()
