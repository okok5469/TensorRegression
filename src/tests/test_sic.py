import math
import unittest

from tensor_abess.tensor.sic import (
    compute_mode_msic_from_rss,
    compute_mode_sic_from_rss,
    compute_tensor_msic_from_rss,
    compute_tensor_sic,
    compute_tensor_sic_from_rss,
    log_binomial,
)


class SicTests(unittest.TestCase):
    def test_tensor_sic_matches_0720_pdf_formula(self) -> None:
        loss = 0.4
        n = 50
        p1, p2, p3 = 10, 8, 6
        s1, s2, s3 = 2, 3, 2

        sic = compute_tensor_sic(loss, s1, s2, s3, n, p1, p2, p3)

        log_n = max(math.log(n), math.e)
        expected = (
            n * math.log(2.0 * loss)
            + (s1 * s2 * s3) * math.log(n)
            + math.log(log_n)
            * (log_binomial(p1, s1) + log_binomial(p2, s2) + log_binomial(p3, s3))
        )
        self.assertAlmostEqual(sic, expected, places=12)

    def test_mode_sic_matches_conditional_formula(self) -> None:
        rss = 12.5
        n = 40
        J = 15
        s = 3
        block_size = 4

        sic = compute_mode_sic_from_rss(rss, s, n, J, block_size)

        log_n = max(math.log(n), math.e)
        expected = (
            n * math.log(rss / n)
            + s * block_size * math.log(n)
            + math.log(log_n) * log_binomial(J, s)
        )
        self.assertAlmostEqual(sic, expected, places=12)

    def test_modified_sic_uses_loglog_parameter_penalty(self) -> None:
        rss = 12.5
        n = 40
        J = 15
        s = 3
        block_size = 4

        msic = compute_mode_msic_from_rss(rss, s, n, J, block_size)

        log_n = max(math.log(n), math.e)
        expected = (
            n * math.log(rss / n)
            + s * block_size * math.log(log_n)
            + math.log(log_n) * log_binomial(J, s)
        )
        self.assertAlmostEqual(msic, expected, places=12)

    def test_modified_tensor_sic_favors_full_signal_when_original_sic_collapses(self) -> None:
        n = 200
        p1, p2, p3 = 32, 20, 25
        singleton_rss = 2.904 * n
        true_rss = 0.772 * n

        original_singleton = compute_tensor_sic_from_rss(singleton_rss, 1, 1, 1, n, p1, p2, p3)
        original_true = compute_tensor_sic_from_rss(true_rss, 4, 3, 5, n, p1, p2, p3)
        modified_singleton = compute_tensor_msic_from_rss(singleton_rss, 1, 1, 1, n, p1, p2, p3)
        modified_true = compute_tensor_msic_from_rss(true_rss, 4, 3, 5, n, p1, p2, p3)

        self.assertLess(original_singleton, original_true)
        self.assertLess(modified_true, modified_singleton)


if __name__ == "__main__":
    unittest.main()
