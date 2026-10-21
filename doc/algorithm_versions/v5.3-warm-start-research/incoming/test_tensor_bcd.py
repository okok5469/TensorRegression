"""Tests for group splicing and tensor BCD."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_abess.group.bess_fix import group_bess_fix
from tensor_abess.tensor.bcd import tensor_bcd
from tensor_abess.utils.data import make_regression_data, make_tensor_data
from tensor_abess.vector.abess import bess_fix


def _group_index_equal_size(n_groups: int, group_size: int) -> np.ndarray:
    return np.repeat(np.arange(n_groups), group_size)


def test_group_bess_fix_recovers_groups():
    rng = np.random.default_rng(0)
    n, n_groups, group_size = 120, 8, 4
    p = n_groups * group_size
    X = rng.normal(size=(n, p))
    true_groups = np.array([1, 3, 5])
    beta = np.zeros(p)
    for g in true_groups:
        cols = np.arange(g * group_size, (g + 1) * group_size)
        beta[cols] = rng.normal(size=group_size)
    y = X @ beta + rng.normal(scale=0.5, size=n)
    gi = _group_index_equal_size(n_groups, group_size)

    result = group_bess_fix(y, X, gi, s=3, n_groups=n_groups)
    assert set(result.active_groups.tolist()) == set(true_groups.tolist())


def test_tensor_bcd_recovers_support_on_toy_data():
    data = make_tensor_data(
        n=200, p1=10, p2=5, p3=8, s1=3, s2=2, s3=2, snr=25.0, random_state=7
    )
    result = tensor_bcd(
        data["X"],
        data["y"],
        data["s1"],
        data["s2"],
        data["s3"],
        S1_init=data["S1"],
        S2_init=data["S2"],
        S3_init=data["S3"],
        max_cycles=20,
    )
    assert np.array_equal(result.S1, data["S1"])
    assert np.array_equal(result.S2, data["S2"])
    assert np.array_equal(result.S3, data["S3"])


def test_tensor_bcd_loss_monotone():
    data = make_tensor_data(n=120, p1=10, p2=5, p3=8, random_state=11)
    result = tensor_bcd(data["X"], data["y"], 3, 2, 2, max_cycles=10)
    diffs = np.diff(result.loss_history)
    assert np.all(diffs <= 1e-10)


def test_tensor_bcd_keeps_current_support_as_fixed_size_candidate():
    """A fixed-size coordinate update must never discard a better current support."""
    data = make_tensor_data(
        n=45,
        p1=8,
        p2=6,
        p3=5,
        s1=3,
        s2=2,
        s3=2,
        snr=2.0,
        random_state=16,
    )
    result = tensor_bcd(
        data["X"],
        data["y"],
        3,
        2,
        2,
        max_cycles=6,
        m_max=5,
    )
    assert np.all(np.diff(result.loss_history) <= 1e-10)


def test_tensor_degenerates_to_vector_when_p2_p3_one():
    vec = make_regression_data(n=150, p=12, s_true=4, snr=20.0, random_state=5)
    X_vec, y = vec["X"], vec["y"]
    s = vec["s_true"]

    X_t = X_vec[:, :, None, None]
    data_t = {"X": X_t, "y": y, "s1": s, "s2": 1, "s3": 1}

    vec_res = bess_fix(X_vec, y, s)
    ten_res = tensor_bcd(
        data_t["X"],
        data_t["y"],
        s,
        1,
        1,
        S1_init=vec_res.support,
        S2_init=np.array([0]),
        S3_init=np.array([0]),
        max_cycles=5,
    )

    assert set(ten_res.S1.tolist()) == set(vec_res.support.tolist())
    for j in vec_res.support:
        np.testing.assert_allclose(
            ten_res.B[j, 0, 0], vec_res.beta[j], rtol=1e-8, atol=1e-8
        )


def test_tensor_bcd_from_marginal_init():
    data = make_tensor_data(n=250, p1=10, p2=5, p3=8, snr=30.0, random_state=99)
    result = tensor_bcd(
        data["X"], data["y"], data["s1"], data["s2"], data["s3"], max_cycles=25
    )
    assert result.S1.size == data["s1"]
    assert result.S2.size == data["s2"]
    assert result.S3.size == data["s3"]
    assert set(result.S1) == set(data["S1"])
    assert set(result.S2) == set(data["S2"])
    assert set(result.S3) == set(data["S3"])
