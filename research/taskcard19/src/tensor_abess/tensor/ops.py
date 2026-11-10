"""
张量内积、OLS 重拟合与损失计算。

矩形支撑 S1×S2×S3 上的 OLS 等价于：把 S1×S2×S3 个元素逐列拉成设计矩阵 Z，
对 y 做标准线性回归。
"""

from __future__ import annotations

import numpy as np


def tensor_inner(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sample-wise inner product y_i = <X_i, B>. Shapes (n,p1,p2,p3)."""
    return np.einsum("nijk,ijk->n", X, B)


def tensor_loss(X: np.ndarray, y: np.ndarray, B: np.ndarray) -> float:
    """Half mean squared error for tensor regression."""
    n = y.shape[0]
    residual = y - tensor_inner(X, B)
    return float(residual @ residual / (2.0 * n))


def coef_vector_to_tensor(
    coef: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    p1: int,
    p2: int,
    p3: int,
) -> np.ndarray:
    """Map OLS coefficients on S1×S2×S3 support back to full tensor B."""
    B = np.zeros((p1, p2, p3), dtype=float)
    idx = 0
    # 列顺序须与 build_tensor_design 一致：先 S1，再 S2，再 S3
    for k in S1:
        for j2 in S2:
            for j3 in S3:
                B[k, j2, j3] = coef[idx]
                idx += 1
    return B


def build_tensor_design(
    X: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> np.ndarray:
    """Design matrix for OLS on rectangular support S1×S2×S3. Shape (n, s1*s2*s3)."""
    n = X.shape[0]
    cols = []
    for k in S1:
        for j2 in S2:
            for j3 in S3:
                cols.append(X[:, k, j2, j3])
    return np.column_stack(cols)


def refit_tensor_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> np.ndarray:
    """OLS refit of B on current rectangular support."""
    p1, p2, p3 = X.shape[1], X.shape[2], X.shape[3]
    Z = build_tensor_design(X, S1, S2, S3)
    if Z.shape[1] == 0:
        return np.zeros((p1, p2, p3), dtype=float)
    coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    return coef_vector_to_tensor(coef, S1, S2, S3, p1, p2, p3)


def flatten_predictors(X: np.ndarray) -> np.ndarray:
    """Reshape (n, p1, p2, p3) to (n, p1*p2*p3) in lexicographic order."""
    n, p1, p2, p3 = X.shape
    return X.reshape(n, p1 * p2 * p3)


def unflatten_tensor(vec: np.ndarray, p1: int, p2: int, p3: int) -> np.ndarray:
    """Map flattened coefficient vector back to (p1, p2, p3)."""
    return vec.reshape(p1, p2, p3)


def mode1_group_indices(p1: int, p2: int, p3: int) -> np.ndarray:
    """Group id for each flattened column (mode-1 slices). Used by group LASSO baseline."""
    group_size = p2 * p3
    return np.repeat(np.arange(p1), group_size)


def support_from_tensor(B: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mode-wise supports where the tensor is non-zero."""
    S1 = np.where(np.any(np.abs(B) > tol, axis=(1, 2)))[0]
    S2 = np.where(np.any(np.abs(B) > tol, axis=(0, 2)))[0]
    S3 = np.where(np.any(np.abs(B) > tol, axis=(0, 1)))[0]
    return S1, S2, S3
