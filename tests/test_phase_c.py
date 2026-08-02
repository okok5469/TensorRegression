"""Phase C tests: Tensor-ABESS, SIC, baselines, init strategies."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_abess.baselines import (
    cp_regression,
    group_lasso_mode1,
    lasso_elementwise,
    run_all_baselines,
    tucker_regression,
)
from tensor_abess.metrics import evaluate_support
from tensor_abess.tensor import tensor_abess, tensor_bcd
from tensor_abess.tensor.sic import compute_tensor_sic, log_binomial
from tensor_abess.utils.data import make_regression_data, make_tensor_data
from tensor_abess.vector.abess import abess as vector_abess


def test_tensor_sic_penalty_additive():
    n, p1, p2, p3 = 100, 10, 5, 8
    loss = 0.5
    sic = compute_tensor_sic(loss, 3, 2, 2, n, p1, p2, p3)
    expected_penalty = (
        log_binomial(p1, 3)
        + log_binomial(p2, 2)
        + log_binomial(p3, 2)
        + 3 * 2 * 2 * np.log(p1 * p2 * p3)
    ) * np.log(np.log(n))
    assert sic == pytest.approx(n * np.log(loss) + expected_penalty)


def test_tensor_abess_selects_true_sparsity():
    data = make_tensor_data(
        n=200, p1=10, p2=5, p3=8, s1=3, s2=2, s3=2, snr=25.0, random_state=42
    )
    result = tensor_abess(
        data["X"],
        data["y"],
        s1_grid=[2, 3, 4],
        s2_grid=[1, 2, 3],
        s3_grid=[1, 2, 3],
        random_state=42,
    )
    assert (result.s1, result.s2, result.s3) == (3, 2, 2)
    m = evaluate_support(
        result.S1, result.S2, result.S3,
        data["S1"], data["S2"], data["S3"],
        result.B, data["B"], data["p1"], data["p2"], data["p3"],
    )
    assert m.exact_rectangular


def test_init_strategies_both_recover_support():
    data = make_tensor_data(n=250, snr=30.0, random_state=88)
    for strategy in ("marginal", "random"):
        res = tensor_bcd(
            data["X"], data["y"], 3, 2, 2,
            init_strategy=strategy, random_state=88, max_cycles=25,
        )
        assert set(res.S1) == set(data["S1"])
        assert set(res.S2) == set(data["S2"])
        assert set(res.S3) == set(data["S3"])


def test_baselines_run_without_error():
    data = make_tensor_data(n=100, random_state=1)
    for bl in run_all_baselines(data["X"], data["y"], random_state=1):
        assert bl.B.shape == data["B"].shape
        assert bl.loss >= 0


def test_tensor_abess_degenerates_to_vector():
    """When p2=p3=1, tensor_abess at vector-selected s1 matches vector support."""
    vec = make_regression_data(n=150, p=12, s_true=4, snr=20.0, random_state=5)
    X_vec, y = vec["X"], vec["y"]
    X_t = X_vec[:, :, None, None]

    vec_res = vector_abess(X_vec, y, s_min=2, s_max=6)
    ten_res = tensor_abess(
        X_t,
        y,
        s1_grid=[vec_res.s],
        s2_grid=[1],
        s3_grid=[1],
        random_state=5,
    )

    assert set(ten_res.S1.tolist()) == set(vec_res.support.tolist())
    assert ten_res.s2 == ten_res.s3 == 1


def test_lasso_baseline_produces_sparse_solution():
    data = make_tensor_data(n=120, random_state=3)
    bl = lasso_elementwise(data["X"], data["y"], random_state=3)
    assert np.sum(np.abs(bl.B) > 1e-8) < data["p1"] * data["p2"] * data["p3"]


def test_cp_and_tucker_baselines():
    data = make_tensor_data(n=80, random_state=9)
    cp = cp_regression(data["X"], data["y"], rank=2, random_state=9)
    tk = tucker_regression(data["X"], data["y"], ranks=(2, 2, 2))
    assert cp.loss < np.mean(data["y"] ** 2)
    assert tk.loss < np.mean(data["y"] ** 2)


def test_group_lasso_baseline():
    data = make_tensor_data(n=100, random_state=12)
    bl = group_lasso_mode1(data["X"], data["y"], random_state=12)
    assert bl.S1.size <= data["p1"]
