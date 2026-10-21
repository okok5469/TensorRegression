"""Research-only fixed-size Cold plus one training-derived extra start.

No adaptive integration, MSIC change, or replacement of the formal Cold route.
The caller must load the frozen v5.2 package explicitly (see test_pure_warm.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from tensor_abess.tensor.bcd import TensorBcdResult, tensor_bcd
from tensor_abess.utils.init import _mode_slice_scores


@dataclass(frozen=True)
class FixedWarmUnion:
    cold: TensorBcdResult
    warm: TensorBcdResult
    selected_route: str
    warm_initial_supports: tuple[np.ndarray, ...]
    independently_checked_rss: tuple[float, float]
    timing: dict[str, float]

    @property
    def selected(self) -> TensorBcdResult:
        return {"cold": self.cold, "warm": self.warm}[self.selected_route]


def align_warm_supports(X, y, sizes, supports):
    """Trim/pad a legal historical support using only training marginal scores."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    if X.ndim != 4 or len(y) != len(X) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite tensor samples are required")
    if len(sizes) != 3 or len(supports) != 3:
        raise ValueError("Exactly three target sizes and support sources are required")
    aligned = []
    for mode, (size, source, dimension) in enumerate(zip(sizes, supports, X.shape[1:]), start=1):
        if not isinstance(size, (int, np.integer)) or not 1 <= size <= dimension:
            raise ValueError("Target support sizes must be legal integers")
        source = np.asarray(source)
        if (source.ndim != 1 or source.dtype.kind not in "iu" or len(np.unique(source)) != len(source)
                or np.any((source < 0) | (source >= dimension))):
            raise ValueError("Warm source positions must be unique legal integers")
        scores = _mode_slice_scores(X, y, mode)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite training marginal score")
        order = np.lexsort((np.arange(dimension), -scores))
        selected = [int(i) for i in order if i in source][:size]
        missing = [int(i) for i in order if i not in source]
        selected.extend(missing[:max(size-len(selected), 0)])
        aligned.append(np.sort(np.array(selected, dtype=int)))
    return tuple(aligned)


def _checked_full_ols_rss(X, y, sizes, result):
    """Independent rectangular least-squares validation; never replace a route."""
    supports = (result.S1, result.S2, result.S3)
    for source, size, dimension in zip(supports, sizes, X.shape[1:]):
        if (source.ndim != 1 or len(source) != size or len(np.unique(source)) != size
                or np.any((source < 0) | (source >= dimension))):
            raise RuntimeError("A route returned an illegal fixed support")
    design = X[:, supports[0]][:, :, supports[1]][:, :, :, supports[2]].reshape(len(y), -1)
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    expected = np.zeros(X.shape[1:])
    expected[np.ix_(*supports)] = coefficient.reshape(sizes)
    residual = y - design @ coefficient
    rss = float(residual @ residual)
    if not np.isfinite(result.B).all() or not np.isfinite(rss):
        raise FloatingPointError("Nonfinite complete OLS candidate")
    if result.B.shape != expected.shape or not np.allclose(result.B, expected, rtol=1e-10, atol=1e-11):
        raise RuntimeError("Returned coefficients disagree with independent full OLS")
    outside = np.ones(expected.shape, dtype=bool)
    outside[np.ix_(*supports)] = False
    if np.any(result.B[outside] != 0):
        raise RuntimeError("Returned coefficients extend outside the fixed rectangle")
    losses = (result.loss, min(result.visited_loss_history),
              result.visited_loss_history[result.best_iteration], result.best_rss/(2*len(y)))
    if not np.allclose(losses, rss/(2*len(y)), rtol=1e-10, atol=1e-11):
        raise RuntimeError("Returned state does not agree with best-visited complete OLS")
    return rss


def fixed_cold_plus_warm(X, y, sizes, *, warm_supports, **cold_parameters):
    """Run the untouched Cold path and one additional initialized path in full.

    Both complete result objects and their diagnostics remain accessible. Invalid
    inputs or invalid route states raise; there is no silent rescue/fallback.
    """
    started = perf_counter()
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    warm_initial = align_warm_supports(X, y, sizes, warm_supports)
    setup_seconds = perf_counter() - started
    cold_started = perf_counter()
    cold = tensor_bcd(X, y, *sizes, **cold_parameters)
    cold_seconds = perf_counter() - cold_started
    warm_parameters = dict(cold_parameters)
    warm_parameters.update({f"S{mode}_init": support.copy()
                            for mode, support in enumerate(warm_initial, start=1)})
    warm_started = perf_counter()
    warm = tensor_bcd(X, y, *sizes, **warm_parameters)
    warm_seconds = perf_counter() - warm_started
    validation_started = perf_counter()
    checked_rss = tuple(_checked_full_ols_rss(X, y, sizes, candidate) for candidate in (cold, warm))
    selected_index = min(range(2), key=lambda index: checked_rss[index])
    validation_seconds = perf_counter() - validation_started
    total_seconds = perf_counter() - started
    timing = {"cold_seconds": cold_seconds, "warm_route_seconds": warm_seconds,
              "warm_setup_seconds": setup_seconds, "validation_seconds": validation_seconds,
              "extra_seconds": total_seconds-cold_seconds, "total_seconds": total_seconds}
    return FixedWarmUnion(cold, warm, ("cold", "warm")[selected_index], warm_initial, checked_rss, timing)
