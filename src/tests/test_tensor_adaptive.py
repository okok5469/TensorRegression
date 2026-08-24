import unittest
from unittest.mock import patch

import numpy as np

from tensor_abess.group.adaptive import GroupAbessResult
from tensor_abess.tensor.abess import tensor_abess
from tensor_abess.tensor.bcd import (
    TensorBcdAdaptiveResult,
    TensorBcdResult,
    _candidate_size_triples,
    tensor_bcd_adaptive,
)
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
        self.assertGreaterEqual(init.s1_max, 1)
        self.assertGreaterEqual(init.s2_max, 1)
        self.assertGreaterEqual(init.s3_max, 1)
        self.assertLessEqual(init.s1_max * init.s2_max * init.s3_max, int(0.7 * X.shape[0]))

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
        self.assertLessEqual(result.s_max[0] * result.s_max[1] * result.s_max[2], int(0.7 * X.shape[0]))
        self.assertGreaterEqual(len(result.sic_history), 1)

    def test_tensor_bcd_adaptive_forwards_modified_sic_to_mode_updates(self) -> None:
        X = np.ones((4, 1, 1, 1), dtype=float)
        y = np.arange(1.0, 5.0)
        seen = []

        def fake_group_bess_adaptive(*args, **kwargs):
            seen.append(kwargs["information_criterion"])
            return GroupAbessResult(
                beta=np.ones(1, dtype=float),
                active_groups=np.array([0]),
                s=1,
                sic=0.0,
                rss=1.0,
                loss=0.125,
                sic_path={1: 0.0},
                bess_path={},
            )

        with patch("tensor_abess.tensor.bcd.group_bess_adaptive", fake_group_bess_adaptive):
            tensor_bcd_adaptive(
                X,
                y,
                s1_max=1,
                s2_max=1,
                s3_max=1,
                S1_init=np.array([0]),
                S2_init=np.array([0]),
                S3_init=np.array([0]),
                max_cycles=1,
                information_criterion="msic",
            )

        self.assertEqual(seen, ["msic", "msic", "msic"])

    def test_tensor_abess_forwards_adaptive_criterion(self) -> None:
        X, y = self._singleton_tensor_problem()
        seen = []

        def fake_tensor_bcd_adaptive(*args, **kwargs):
            seen.append(kwargs["information_criterion"])
            return TensorBcdAdaptiveResult(
                B=np.zeros(X.shape[1:], dtype=float),
                S1=np.array([0]),
                S2=np.array([0]),
                S3=np.array([0]),
                s1=1,
                s2=1,
                s3=1,
                loss=0.0,
                rss=0.0,
                sic=0.0,
                loss_history=[0.0],
                sic_history=[0.0],
                n_cycles=1,
            )

        with patch("tensor_abess.tensor.abess.tensor_bcd_adaptive", fake_tensor_bcd_adaptive):
            tensor_abess(
                X,
                y,
                c=2.0,
                criterion="bic",
                n_lambdas=8,
                max_iter=200,
                adaptive_criterion="msic",
            )

        self.assertEqual(seen, ["msic"])

    def test_tensor_bcd_adaptive_msic_refines_against_global_candidates(self) -> None:
        X = np.ones((8, 2, 2, 2), dtype=float)
        y = np.arange(1.0, 9.0)

        def fake_group_bess_adaptive(*args, **kwargs):
            return GroupAbessResult(
                beta=np.ones(1, dtype=float),
                active_groups=np.array([0]),
                s=1,
                sic=0.0,
                rss=1.0,
                loss=0.1,
                sic_path={1: 0.0},
                bess_path={},
            )

        def fake_tensor_bcd(*args, **kwargs):
            s1, s2, s3 = (int(v) for v in args[2:5])
            loss = 0.01 if (s1, s2, s3) == (2, 1, 2) else 1.0
            return TensorBcdResult(
                B=np.zeros(X.shape[1:], dtype=float),
                S1=np.arange(s1),
                S2=np.arange(s2),
                S3=np.arange(s3),
                loss=loss,
                loss_history=[loss],
                n_cycles=1,
            )

        with patch("tensor_abess.tensor.bcd.group_bess_adaptive", fake_group_bess_adaptive):
            with patch("tensor_abess.tensor.bcd.tensor_bcd", fake_tensor_bcd):
                result = tensor_bcd_adaptive(
                    X,
                    y,
                    s1_max=2,
                    s2_max=2,
                    s3_max=2,
                    S1_init=np.array([0]),
                    S2_init=np.array([0]),
                    S3_init=np.array([0]),
                    max_cycles=1,
                    information_criterion="msic",
                )

        self.assertEqual((result.s1, result.s2, result.s3), (2, 1, 2))

    def test_global_refine_candidates_include_mid_sparsity_shapes(self) -> None:
        candidates = _candidate_size_triples(6, 7, 7, max_params=280)

        self.assertIn((4, 3, 5), candidates)


if __name__ == "__main__":
    unittest.main()
