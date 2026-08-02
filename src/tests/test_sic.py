import math
import unittest

from tensor_abess.tensor.sic import compute_tensor_sic, compute_mode_sic_from_rss, log_binomial


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


if __name__ == "__main__":
    unittest.main()
