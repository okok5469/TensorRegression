"""Reduce tensor regression updates to grouped linear designs."""

from __future__ import annotations

import numpy as np


def _build_mode_design_from_groups(
    X: np.ndarray,
    mode: int,
    candidate_groups: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a grouped design over a subset of candidate slices for one mode."""
    X = np.asarray(X, dtype=float)
    candidate_groups = np.sort(np.asarray(candidate_groups, dtype=int))
    S1 = np.asarray(S1, dtype=int)
    S2 = np.asarray(S2, dtype=int)
    S3 = np.asarray(S3, dtype=int)

    n = X.shape[0]
    if mode == 1:
        block_size = len(S2) * len(S3)
        Z = np.zeros((n, len(candidate_groups) * block_size), dtype=float)
        for local_g, global_g in enumerate(candidate_groups):
            block = X[:, global_g, :, :][:, S2, :][:, :, S3].reshape(n, block_size)
            Z[:, local_g * block_size : (local_g + 1) * block_size] = block
    elif mode == 2:
        block_size = len(S1) * len(S3)
        Z = np.zeros((n, len(candidate_groups) * block_size), dtype=float)
        for local_g, global_g in enumerate(candidate_groups):
            block = X[:, :, global_g, :][:, S1, :][:, :, S3].reshape(n, block_size)
            Z[:, local_g * block_size : (local_g + 1) * block_size] = block
    elif mode == 3:
        block_size = len(S1) * len(S2)
        Z = np.zeros((n, len(candidate_groups) * block_size), dtype=float)
        for local_g, global_g in enumerate(candidate_groups):
            block = X[:, :, :, global_g][:, S1, :][:, :, S2].reshape(n, block_size)
            Z[:, local_g * block_size : (local_g + 1) * block_size] = block
    else:
        raise ValueError("mode must be 1, 2, or 3")

    group_index = np.repeat(np.arange(len(candidate_groups)), block_size)
    return Z, group_index, candidate_groups


def build_mode_design(
    X: np.ndarray,
    mode: int,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build grouped design for updating one mode over all slices in that mode.

    Returns (Z, group_index, n_groups).
    """
    if mode == 1:
        candidate_groups = np.arange(X.shape[1])
    elif mode == 2:
        candidate_groups = np.arange(X.shape[2])
    elif mode == 3:
        candidate_groups = np.arange(X.shape[3])
    else:
        raise ValueError("mode must be 1, 2, or 3")
    Z, gi, _ = _build_mode_design_from_groups(X, mode, candidate_groups, S1, S2, S3)
    return Z, gi, len(candidate_groups)


def build_mode_design_subset(
    X: np.ndarray,
    mode: int,
    candidate_groups: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build grouped design for a subset of candidate slices.

    Returns (Z, group_index, candidate_groups_sorted), where group ids in
    ``group_index`` are local ids 0..len(candidate_groups)-1 and the last output
    maps local ids back to original slice indices.
    """
    return _build_mode_design_from_groups(X, mode, candidate_groups, S1, S2, S3)
