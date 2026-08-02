import math
import unittest

import numpy as np

from tensor_abess.group.sacrifices import (
    compute_group_sacrifices,
    fit_ols_on_groups,
    group_tau_threshold,
    initial_active_groups,
)


class GroupCoreTests(unittest.TestCase):
    def test_group_sacrifices_match_closed_form_pdf_definition(self) -> None:
        X = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        y = np.array([1.0, 2.0, 0.5, 1.5])
        group_index = np.array([0, 0, 1, 1])
        active = np.array([0])
        inactive = np.array([1])

        beta, loss = fit_ols_on_groups(y, X, group_index, active)
        xi, zeta = compute_group_sacrifices(y, X, group_index, active, inactive, loss)

        n = y.size
        residual = y - X @ beta
        cols0 = np.where(group_index == 0)[0]
        cols1 = np.where(group_index == 1)[0]
        gamma0 = X[:, cols0].T @ X[:, cols0] / n
        gamma1 = X[:, cols1].T @ X[:, cols1] / n
        d1 = X[:, cols1].T @ residual / n

        expected_xi0 = 0.5 * beta[cols0] @ gamma0 @ beta[cols0]
        expected_zeta1 = 0.5 * d1 @ np.linalg.solve(gamma1, d1)

        self.assertAlmostEqual(xi[0], expected_xi0, places=10)
        self.assertAlmostEqual(zeta[1], expected_zeta1, places=10)

    def test_initial_active_groups_uses_gram_standardized_scores(self) -> None:
        X = np.array(
            [
                [10.0, 1.0],
                [0.0, 1.0],
            ]
        )
        y = np.array([1.0, 1.0])
        group_index = np.array([0, 1])

        active = initial_active_groups(X, y, group_index, s=1)

        self.assertTrue(np.array_equal(active, np.array([1])))

    def test_group_tau_threshold_scales_with_block_size(self) -> None:
        tau = group_tau_threshold(s=3, n_groups=20, n=50, block_size=4, c=0.01)
        expected = 0.01 * 3 * 4 * math.log(20) * math.log(max(math.log(50), math.e)) / 50
        self.assertAlmostEqual(tau, expected, places=12)


if __name__ == "__main__":
    unittest.main()
