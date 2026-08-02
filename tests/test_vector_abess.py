"""Unit tests for vector ABESS."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_abess.utils.data import make_regression_data
from tensor_abess.vector.abess import abess, bess_fix
from tensor_abess.vector.sacrifices import squared_loss
from tensor_abess.vector.sic import compute_sic


def test_bess_fix_recovers_support_on_easy_data():
    data = make_regression_data(n=200, p=20, s_true=3, snr=20.0, random_state=1)
    result = bess_fix(data["X"], data["y"], data["s_true"])
    true = set(data["support"].tolist())
    est = set(result.support.tolist())
    assert len(est) == data["s_true"]
    assert est == true


def test_abess_sic_selects_reasonable_sparsity():
    data = make_regression_data(n=150, p=30, s_true=4, random_state=7)
    result = abess(data["X"], data["y"], s_max=10)
    assert 1 <= result.s <= 10
    assert result.support.size == result.s
    assert result.sic == pytest.approx(compute_sic(result.loss, result.s, data["n"], data["p"]))


def test_loss_decreases_or_stable_after_splicing():
    data = make_regression_data(n=80, p=15, s_true=3, random_state=99)
    X, y = data["X"], data["y"]
    s = 5
    r1 = bess_fix(X, y, s, m_max=1)
    r2 = bess_fix(X, y, s, m_max=10)
    assert r2.loss <= r1.loss + 1e-10


def test_bess_fix_invalid_s_raises():
    data = make_regression_data(n=50, p=10, s_true=2)
    with pytest.raises(ValueError):
        bess_fix(data["X"], data["y"], 0)
    with pytest.raises(ValueError):
        bess_fix(data["X"], data["y"], 11)


def test_oracle_ols_matches_lstsq_on_support():
    data = make_regression_data(n=60, p=12, s_true=3, random_state=3)
    X, y = data["X"], data["y"]
    support = data["support"]
    result = bess_fix(X, y, support.size, active_init=support, m_max=0)
    beta_lstsq, _, _, _ = np.linalg.lstsq(X[:, support], y, rcond=None)
    np.testing.assert_allclose(result.beta[support], beta_lstsq, rtol=1e-10)
    expected_loss = squared_loss(y, X, result.beta)
    assert result.loss == pytest.approx(expected_loss)
