"""Tucker regression baseline via weighted HOSVD."""

from __future__ import annotations

import numpy as np

from tensor_abess.baselines.base import BaselineResult
from tensor_abess.tensor.ops import support_from_tensor, tensor_loss


def _mode_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return T.reshape(T.shape[0], -1)
    if mode == 1:
        return np.moveaxis(T, 1, 0).reshape(T.shape[1], -1)
    return np.moveaxis(T, 2, 0).reshape(T.shape[2], -1)


def tucker_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ranks: tuple[int, int, int] = (3, 2, 2),
) -> BaselineResult:
    """
    Tucker-style baseline: HOSVD on T = (1/n) sum_i y_i X_i, then reconstruct B.

    Core G = T ×_1 U1^T ×_2 U2^T ×_3 U3^T;  B = G ×_1 U1 ×_2 U2 ×_3 U3.
    """
    y = np.asarray(y, dtype=float).ravel()
    _, p1, p2, p3 = X.shape
    r1, r2, r3 = ranks

    T = np.einsum("n,nijk->ijk", y, X) / X.shape[0]

    U1, _, _ = np.linalg.svd(_mode_unfold(T, 0), full_matrices=False)
    U2, _, _ = np.linalg.svd(_mode_unfold(T, 1), full_matrices=False)
    U3, _, _ = np.linalg.svd(_mode_unfold(T, 2), full_matrices=False)

    U1 = U1[:, : min(r1, U1.shape[1])]
    U2 = U2[:, : min(r2, U2.shape[1])]
    U3 = U3[:, : min(r3, U3.shape[1])]

    G = np.einsum("abc, ai, bj, ck -> ijk", T, U1, U2, U3)
    B = np.einsum("ijk, ai, bj, ck -> abc", G, U1, U2, U3)

    scale = np.max(np.abs(B)) if np.any(B) else 1.0
    S1, S2, S3 = support_from_tensor(B, tol=1e-6 * scale)

    return BaselineResult(
        name=f"tucker{list(ranks)}",
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        loss=tensor_loss(X, y, B),
    )
