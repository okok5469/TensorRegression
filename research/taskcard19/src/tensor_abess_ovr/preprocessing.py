"""Training-only, coordinatewise float64 z-score with explicit tensor axes."""
from __future__ import annotations

import json

import numpy as np


def numeric_tensor(X):
    raw = np.asarray(X)
    if raw.ndim != 4 or min(raw.shape) < 1:
        raise ValueError("X shape must be (n,p1,p2,p3), all dimensions nonempty")
    if raw.dtype.kind not in "iuf":
        raise ValueError("X must be a real numeric tensor without object dtype")
    array = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values; missing_policy=error")
    return array


def checked_axes(axis_metadata, shape):
    axes = axis_metadata
    if axes is None:
        axes = [{"name": f"axis_{i+1}", "coordinates": list(range(size))} for i, size in enumerate(shape)]
    axes = json.loads(json.dumps(axes, ensure_ascii=False, allow_nan=False))
    if not isinstance(axes, list) or len(axes) != 3:
        raise ValueError("axis metadata must contain three ordered axis objects")
    for axis, size in zip(axes, shape):
        if not isinstance(axis, dict) or not axis.get("name"):
            raise ValueError("axis metadata requires name and coordinates")
        coordinates = axis.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != size:
            raise ValueError("axis coordinates must match tensor shape")
        if len({json.dumps(v, ensure_ascii=False, sort_keys=True) for v in coordinates}) != size:
            raise ValueError("axis coordinates must be unique")
    return axes


class TensorStandardizer:
    def __init__(self, axis_metadata=None):
        self.axis_metadata = axis_metadata

    def fit(self, X, *, axis_metadata=None):
        X = numeric_tensor(X)
        self.shape_ = tuple(X.shape[1:])
        self.axis_metadata_ = checked_axes(self.axis_metadata if axis_metadata is None else axis_metadata, self.shape_)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = X.mean(axis=0, dtype=np.float64)
            scale = X.std(axis=0, ddof=0, dtype=np.float64)
        constant = (X.max(axis=0) == X.min(axis=0)) | (scale == 0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("nonfinite training preprocessing statistics")
        self.mean_ = mean
        self.constant_mask_ = constant
        self.scale_ = np.where(constant, 1.0, scale)
        return self

    def transform(self, X, *, axis_metadata=None):
        if not hasattr(self, "shape_"):
            raise RuntimeError("standardizer is not fitted")
        X = numeric_tensor(X)
        if tuple(X.shape[1:]) != self.shape_:
            raise ValueError(f"X shape {X.shape[1:]} differs from fitted shape {self.shape_}")
        if axis_metadata is not None and checked_axes(axis_metadata, self.shape_) != self.axis_metadata_:
            raise ValueError("axis metadata/order differs from fitted coordinates")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (X - self.mean_) / self.scale_
        result[:, self.constant_mask_] = 0.0
        if not np.isfinite(result).all():
            raise ValueError("nonfinite transformed X")
        return np.ascontiguousarray(result, dtype=np.float64)

    def fit_transform(self, X, *, axis_metadata=None):
        return self.fit(X, axis_metadata=axis_metadata).transform(X, axis_metadata=axis_metadata)
