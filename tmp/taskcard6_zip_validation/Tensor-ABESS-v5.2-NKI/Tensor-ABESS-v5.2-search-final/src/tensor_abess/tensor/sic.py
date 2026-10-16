"""Tensor and mode-wise SIC utilities for the 0720 revised specification."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from tensor_abess.vector.sic import default_s_max


def log_binomial(p: int, s: int) -> float:
    """log(C(p, s)) via gammaln."""
    if s < 0 or s > p:
        return -np.inf
    return float(gammaln(p + 1) - gammaln(s + 1) - gammaln(p - s + 1))


def compute_mode_sic_from_rss(
    rss: float,
    s: int,
    n: int,
    J: int,
    block_size: int,
) -> float:
    """Conditional SIC from Algorithm 5.1 in the revised PDF."""
    eps = np.finfo(float).tiny
    log_n = max(np.log(n), np.e)
    return (
        n * np.log(max(rss / n, eps))
        + s * block_size * np.log(n)
        + np.log(log_n) * log_binomial(J, s)
    )


def compute_mode_msic_from_rss(
    rss: float,
    s: int,
    n: int,
    J: int,
    block_size: int,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Finite-sample modified conditional SIC with log-log df penalty."""
    eps = np.finfo(float).tiny
    log_n = max(np.log(n), np.e)
    loglog_n = np.log(log_n)
    return (
        n * np.log(max(rss / n, eps))
        + alpha * s * block_size * loglog_n
        + gamma * loglog_n * log_binomial(J, s)
    )


def compute_tensor_sic_from_rss(
    rss: float,
    s1: int,
    s2: int,
    s3: int,
    n: int,
    p1: int,
    p2: int,
    p3: int,
) -> float:
    """Final tensor SIC from Equation (6.3) in the 0720 revised PDF."""
    eps = np.finfo(float).tiny
    log_n = max(np.log(n), np.e)
    return (
        n * np.log(max(rss / n, eps))
        + s1 * s2 * s3 * np.log(n)
        + np.log(log_n)
        * (log_binomial(p1, s1) + log_binomial(p2, s2) + log_binomial(p3, s3))
    )


def compute_tensor_msic_from_rss(
    rss: float,
    s1: int,
    s2: int,
    s3: int,
    n: int,
    p1: int,
    p2: int,
    p3: int,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Finite-sample modified tensor SIC with log-log df penalty."""
    eps = np.finfo(float).tiny
    log_n = max(np.log(n), np.e)
    loglog_n = np.log(log_n)
    return (
        n * np.log(max(rss / n, eps))
        + alpha * s1 * s2 * s3 * loglog_n
        + gamma
        * loglog_n
        * (log_binomial(p1, s1) + log_binomial(p2, s2) + log_binomial(p3, s3))
    )


def compute_tensor_sic(
    loss: float,
    s1: int,
    s2: int,
    s3: int,
    n: int,
    p1: int,
    p2: int,
    p3: int,
) -> float:
    """Compatibility wrapper using loss = RSS / (2n)."""
    rss = 2.0 * n * loss
    return compute_tensor_sic_from_rss(rss, s1, s2, s3, n, p1, p2, p3)


def default_sparsity_grid(
    n: int, p1: int, p2: int, p3: int
) -> tuple[list[int], list[int], list[int]]:
    """Legacy grid helper retained for backward compatibility."""
    s1_grid = list(range(1, default_s_max(n, p1) + 1))
    s2_grid = list(range(1, default_s_max(n, p2) + 1))
    s3_grid = list(range(1, default_s_max(n, p3) + 1))
    return s1_grid, s2_grid, s3_grid
