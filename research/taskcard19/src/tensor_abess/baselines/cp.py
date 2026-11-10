"""CP regression baseline via alternating least squares."""

from __future__ import annotations

import numpy as np

from tensor_abess.baselines.base import BaselineResult
from tensor_abess.tensor.ops import support_from_tensor, tensor_loss


def cp_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rank: int = 3,
    max_iter: int = 100,
    random_state: int = 0,
) -> BaselineResult:
    """
    Rank-R CP regression: y_i ≈ <X_i, sum_r a_r ∘ b_r ∘ c_r>.

    ALS updates; final B = sum_r outer(a_r, b_r, c_r).
    """
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape
    rng = np.random.default_rng(random_state)
    R = rank

    A = rng.normal(size=(p1, R))
    Bf = rng.normal(size=(p2, R))
    C = rng.normal(size=(p3, R))

    for _ in range(max_iter):
        for r in range(R):
            W = np.einsum("njkl,k,l->nj", X, Bf[:, r], C[:, r])
            coef, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
            A[:, r] = coef

        for r in range(R):
            W = np.einsum("njkl,j,l->nk", X, A[:, r], C[:, r])
            coef, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
            Bf[:, r] = coef

        for r in range(R):
            W = np.einsum("njkl,j,k->nl", X, A[:, r], Bf[:, r])
            coef, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
            C[:, r] = coef

    B = np.zeros((p1, p2, p3))
    for r in range(R):
        B += np.einsum("i,j,k->ijk", A[:, r], Bf[:, r], C[:, r])

    scale = np.max(np.abs(B)) if np.any(B) else 1.0
    S1, S2, S3 = support_from_tensor(B, tol=1e-6 * scale)

    return BaselineResult(
        name=f"cp_rank{rank}",
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        loss=tensor_loss(X, y, B),
    )
