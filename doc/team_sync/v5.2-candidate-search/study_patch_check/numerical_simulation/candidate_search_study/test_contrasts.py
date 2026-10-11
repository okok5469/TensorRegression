"""Paired mechanism effects and nested-pool one-SE failure examples."""
import unittest

from numerical_simulation.candidate_search_study import contrasts


def candidate(name, q, score):
    return {"kind": "global-fixed", "sizes": [1, 1, q], "q": q,
            "supports": [[0], [0], list(range(q))], "B_fingerprint": name, "score": score}


def coarse_report(full, coarse, full_selected=0, coarse_selected=0):
    def group(pool):
        return {"candidates": pool, "runtime_sec": 1.0, "true_size_in_pool": False,
                "true_support_in_pool": False, "search_metadata": {"size_grid": [c["sizes"] for c in pool]}}
    def row(name, pool, selected):
        minimum = min(range(len(pool)), key=lambda i: pool[i]["score"])
        return {"group": name, "q_hat": pool[selected]["q"], "B_fingerprint": pool[selected]["B_fingerprint"],
                "selection": {"selected_index": selected, "minimum_index": minimum,
                    "minimum_score": pool[minimum]["score"], "tolerance": 10.0}}
    return {"config": {"sparsity": [1, 1, 1]}, "groups": {"A3": group(full), "A4": group(coarse)},
            "rows": [row("A3", full, full_selected), row("A4", coarse, coarse_selected)]}


class ContrastTests(unittest.TestCase):
    def test_conditional_effects_and_interaction_are_paired_per_dataset(self):
        rows = [{"task_id": 0, "values": {"A0": 0., "A1": 2., "A2": 4., "A3": 9.}},
                {"task_id": 1, "values": {"A0": 10., "A1": 9., "A2": 13., "A3": 15.}}]
        first = contrasts.factorial_effects(rows[0]["values"])
        self.assertEqual(first, {"cap_at_legacy_grid": 2., "cap_at_full_grid": 5.,
                                "grid_at_legacy_cap": 4., "grid_at_candidate_cap": 7., "interaction": 3.})
        interactions = [contrasts.factorial_effects(row["values"])["interaction"] for row in rows]
        self.assertEqual(sum(interactions) / len(interactions), 3.)

    def test_missing_group_cannot_be_pooled_as_balanced_comparison(self):
        with self.assertRaisesRegex(ValueError, "four"):
            contrasts.factorial_effects({"A0": 1., "A1": 2., "A3": 3.})

    def test_same_minimum_can_omit_one_se_smaller_model(self):
        large = candidate("large", 100, 0.)
        small = candidate("small", 1, 8.)
        result = contrasts.coarse_diagnostics(coarse_report([large, small], [large], 1, 0))
        self.assertEqual(result["minimum_score_regret"], 0.)
        self.assertTrue(result["same_minimum_missing_one_se_smaller_model"])
        self.assertFalse(result["full_selected_B_in_coarse_pool"])
        self.assertTrue(result["full_minimum_B_in_coarse_pool"])

    def test_pool_membership_includes_support_positions_not_just_q(self):
        first = candidate("same-coefficient", 1, 0.)
        second = candidate("same-coefficient", 1, 0.)
        second["supports"] = [[1], [0], [0]]
        self.assertNotEqual(contrasts.candidate_identity(first), contrasts.candidate_identity(second))
        with self.assertRaisesRegex(ValueError, "nested"):
            contrasts.coarse_diagnostics(coarse_report([first], [second]))

    def test_coarse_cannot_have_better_minimum_than_its_full_superset(self):
        first = candidate("same", 1, 1.)
        second = candidate("same", 1, -1.)
        with self.assertRaises(ValueError):
            contrasts.coarse_diagnostics(coarse_report([first], [second]))

    def test_small_roundoff_regret_is_normalized_but_raw_delta_retained(self):
        first = candidate("same", 1, 1.)
        second = candidate("same", 1, 1. - 1e-12)
        result = contrasts.coarse_diagnostics(coarse_report([first], [second]))
        self.assertEqual(result["minimum_score_regret"], 0.)
        self.assertLess(result["minimum_score_delta_raw"], 0.)


if __name__ == "__main__":
    unittest.main()
