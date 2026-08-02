import unittest

import numpy as np

from tensor_abess.group.adaptive import group_bess_adaptive


class GroupAdaptiveTests(unittest.TestCase):
    def test_group_bess_adaptive_selects_s_by_revised_mode_sic(self) -> None:
        X = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        y = np.array([4.0, 3.0, 0.0, 0.0])
        group_index = np.array([0, 1, 2])

        result = group_bess_adaptive(
            y,
            X,
            group_index,
            s_max=3,
            block_size=1,
            tau=1e-12,
            m_max=3,
        )

        self.assertEqual(result.s, 2)
        self.assertTrue(np.array_equal(result.active_groups, np.array([0, 1])))

    def test_group_bess_adaptive_uses_warm_start_at_matching_size(self) -> None:
        X = np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        y = np.array([3.0, 2.0, 1.0, 0.0])
        group_index = np.array([0, 1, 2])
        warm = np.array([1, 2])

        result = group_bess_adaptive(
            y,
            X,
            group_index,
            s_max=3,
            block_size=1,
            groups_warm=warm,
            tau=1e9,
            m_max=1,
        )

        self.assertTrue(np.array_equal(result.bess_path[2].active_groups, warm))


if __name__ == "__main__":
    unittest.main()
