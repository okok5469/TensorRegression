"""Training-only nuisance regression, centering, and tensor standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_tensor_response(X, y):
    """Validate aligned, finite tensor samples without silently reshaping y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 4 or min(X.shape) < 1:
        raise ValueError("X must have shape (n, channels, time, frequency) with nonempty axes")
    if y.ndim != 1 or len(y) != len(X) or len(y) < 2:
        raise ValueError("y must be a vector aligned with at least two X samples")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    return X, y


def validate_covariates(C, n, *, n_columns=None, require_full_rank=False):
    """C includes an explicit intercept; validation data need not be full rank."""
    if C is None:
        if n_columns not in (None, 0):
            raise ValueError("This fitted preprocessor requires covariates")
        return np.empty((n, 0), dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != n or C.shape[1] < 1:
        raise ValueError("Covariate design must have shape (n, p) including an intercept")
    if n_columns is not None and C.shape[1] != n_columns:
        raise ValueError("Covariate columns must match the training design")
    if not np.isfinite(C).all():
        raise ValueError("Covariates must be finite")
    if not np.allclose(C[:, 0], 1.0, rtol=0, atol=1e-12):
        raise ValueError("First covariate column must be the explicit intercept")
    if require_full_rank and np.linalg.matrix_rank(C) != C.shape[1]:
        raise ValueError("Training covariate design must have full column rank")
    if require_full_rank and C.shape[1] >= n - 1:
        raise ValueError("Training covariate design leaves insufficient residual degrees of freedom")
    return C


@dataclass(frozen=True)
class FoldPreprocessor:
    """All fitted parameters are explicit and can be saved with np.savez."""

    x_nuisance_coef: np.ndarray
    y_nuisance_coef: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float

    def apply(self, X, C=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4 or X.shape[1:] != self.x_mean.shape:
            raise ValueError("Prediction tensor dimensions must match the training tensor")
        if not np.isfinite(X).all():
            raise ValueError("Prediction tensor must be finite")
        design = validate_covariates(C, len(X), n_columns=len(self.y_nuisance_coef))
        residual = X.reshape(len(X), -1) - design @ self.x_nuisance_coef
        return (residual.reshape(X.shape) - self.x_mean) / self.x_scale

    def transform_response(self, y, C=None):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("Response must be a finite vector")
        design = validate_covariates(C, len(y), n_columns=len(self.y_nuisance_coef))
        return y - design @ self.y_nuisance_coef - self.y_mean

    def predict(self, B, X, C=None):
        B = np.asarray(B, dtype=float)
        if B.shape != self.x_mean.shape or not np.isfinite(B).all():
            raise ValueError("B must be finite and have the fitted tensor dimensions")
        transformed = self.apply(X, C)
        design = validate_covariates(C, len(transformed), n_columns=len(self.y_nuisance_coef))
        prediction = np.einsum("nijk,ijk->n", transformed, B)
        prediction += self.y_mean + design @ self.y_nuisance_coef
        if not np.isfinite(prediction).all():
            raise ValueError("Model produced non-finite predictions")
        return prediction

    def to_arrays(self):
        return {"x_nuisance_coef": self.x_nuisance_coef.copy(),
                "y_nuisance_coef": self.y_nuisance_coef.copy(),
                "x_mean": self.x_mean.copy(), "x_scale": self.x_scale.copy(),
                "y_mean": np.asarray(self.y_mean)}


def fit_preprocessor(X_train, y_train, C_train=None):
    """Fit all nuisance and scaling parameters exclusively on training subjects."""
    X, y = validate_tensor_response(X_train, y_train)
    C = validate_covariates(C_train, len(y), require_full_rank=True)
    flat = X.reshape(len(y), -1)
    # An empty design is the explicit unadjusted case, not an inferred fallback.
    coefficients = np.linalg.lstsq(C, np.column_stack((flat, y)), rcond=None)[0]
    x_coef, y_coef = coefficients[:, :-1], coefficients[:, -1]
    x_residual = (flat - C @ x_coef).reshape(X.shape)
    y_residual = y - C @ y_coef
    x_mean = x_residual.mean(axis=0)
    x_scale = x_residual.std(axis=0, ddof=1)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    return FoldPreprocessor(x_coef, y_coef, x_mean, x_scale, float(y_residual.mean()))
