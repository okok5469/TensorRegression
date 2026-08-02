"""Low-level consistency checks for tensor simulation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_abess.tensor.ops import flatten_predictors, tensor_inner
from tensor_abess.tensor.reduction import build_mode_design


@dataclass(frozen=True)
class ConsistencyReport:
    """Compact report for tensor-order consistency checks."""

    inner_product_ok: bool
    mode_design_ok: bool
    max_inner_product_error: float
    max_mode_design_error: float
    design_convention: str = "numpy_c_order"
    note: str = (
        "Design-matrix checks use the current NumPy C-order unfolding convention "
        "implemented in src/tensor_abess."
    )

    @property
    def all_passed(self) -> bool:
        return self.inner_product_ok and self.mode_design_ok


def _selection_matrix(p: int, support: np.ndarray) -> np.ndarray:
    return np.eye(p, dtype=float)[:, np.asarray(support, dtype=int)]


def _mode_unfold_c_order(x: np.ndarray, mode: int) -> np.ndarray:
    if mode == 1:
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    if mode == 2:
        return np.transpose(x, (1, 0, 2)).reshape(x.shape[1], x.shape[0] * x.shape[2])
    if mode == 3:
        return np.transpose(x, (2, 0, 1)).reshape(x.shape[2], x.shape[0] * x.shape[1])
    raise ValueError("mode must be 1, 2, or 3")


def _expected_mode_design_row(
    x: np.ndarray,
    mode: int,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> np.ndarray:
    p1, p2, p3 = x.shape
    U1 = _selection_matrix(p1, S1)
    U2 = _selection_matrix(p2, S2)
    U3 = _selection_matrix(p3, S3)

    if mode == 1:
        expected = _mode_unfold_c_order(x, 1) @ np.kron(U2, U3)
    elif mode == 2:
        expected = _mode_unfold_c_order(x, 2) @ np.kron(U1, U3)
    elif mode == 3:
        expected = _mode_unfold_c_order(x, 3) @ np.kron(U1, U2)
    else:
        raise ValueError("mode must be 1, 2, or 3")
    return expected.reshape(-1)


def run_consistency_checks(
    *,
    seed: int = 0,
    tol: float = 1e-10,
) -> ConsistencyReport:
    """Check vec/inner-product consistency and grouped mode-design ordering."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3, 4, 3, 2))
    B = rng.normal(size=(4, 3, 2))

    pred_tensor = tensor_inner(X, B)
    pred_vectorized = flatten_predictors(X) @ B.reshape(-1)
    max_inner = float(np.max(np.abs(pred_tensor - pred_vectorized)))

    S1 = np.array([0, 2], dtype=int)
    S2 = np.array([1], dtype=int)
    S3 = np.array([0, 1], dtype=int)

    max_design = 0.0
    for mode in (1, 2, 3):
        Z, _, _ = build_mode_design(X, mode, S1, S2, S3)
        for i in range(X.shape[0]):
            expected = _expected_mode_design_row(X[i], mode, S1, S2, S3)
            max_design = max(max_design, float(np.max(np.abs(Z[i] - expected))))

    return ConsistencyReport(
        inner_product_ok=max_inner < tol,
        mode_design_ok=max_design < tol,
        max_inner_product_error=max_inner,
        max_mode_design_error=max_design,
    )
