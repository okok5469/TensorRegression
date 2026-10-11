"""Element-wise LASSO baseline (flattened tensor regression)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV

from tensor_abess.baselines.base import BaselineResult
from tensor_abess.tensor.ops import (
    flatten_predictors,
    support_from_tensor,
    tensor_loss,
    unflatten_tensor,
)


def lasso_elementwise(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 5,
    random_state: int = 0,
) -> BaselineResult:
    """Lasso on vec(X) with cross-validated penalty."""
    y = np.asarray(y, dtype=float).ravel()
    n, p1, p2, p3 = X.shape
    Xf = flatten_predictors(X)

    model = LassoCV(
        cv=cv,
        random_state=random_state,
        max_iter=5000,
        fit_intercept=False,
    )
    model.fit(Xf, y)
    B = unflatten_tensor(model.coef_, p1, p2, p3)
    S1, S2, S3 = support_from_tensor(B, tol=1e-8)
    loss = tensor_loss(X, y, B)

    return BaselineResult(
        name="elementwise_lasso",
        B=B,
        S1=S1,
        S2=S2,
        S3=S3,
        loss=loss,
    )
