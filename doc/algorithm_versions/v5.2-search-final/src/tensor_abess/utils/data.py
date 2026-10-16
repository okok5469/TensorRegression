"""
合成数据生成：用于单元测试与 demo 的「玩具数据」。

非真实 EEG 数据；真值 S1/S2/S3 已知，便于检验算法能否精确恢复支撑。
"""

from __future__ import annotations

import numpy as np


def make_regression_data(
    n: int = 100,
    p: int = 50,
    s_true: int = 5,
    snr: float = 10.0,
    rho: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Simulate y = X beta + eps with s_true non-zero coefficients.

    X columns have AR(1) correlation rho; SNR controls signal strength.
    """
    rng = np.random.default_rng(random_state)

    idx = np.arange(p)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)

    support = np.sort(rng.choice(p, size=s_true, replace=False))
    beta = np.zeros(p)
    signal_scale = np.sqrt(snr)
    beta[support] = rng.choice([-1.0, 1.0], size=s_true) * signal_scale

    y = X @ beta + rng.normal(0, 1.0, size=n)

    return {
        "X": X,
        "y": y,
        "beta": beta,
        "support": support,
        "n": n,
        "p": p,
        "s_true": s_true,
    }


def make_tensor_data(
    n: int = 150,
    p1: int = 10,
    p2: int = 5,
    p3: int = 8,
    s1: int = 3,
    s2: int = 2,
    s3: int = 2,
    snr: float = 15.0,
    random_state: int = 42,
) -> dict:
    """
    Simulate y_i = <X_i, B> + eps with rectangular support S1×S2×S3.

    Returns X (n,p1,p2,p3), y (n,), B (p1,p2,p3), and true supports.
    """
    rng = np.random.default_rng(random_state)

    S1 = np.sort(rng.choice(p1, size=s1, replace=False))
    S2 = np.sort(rng.choice(p2, size=s2, replace=False))
    S3 = np.sort(rng.choice(p3, size=s3, replace=False))

    B = np.zeros((p1, p2, p3), dtype=float)
    # 总信号能量 ≈ snr，均摊到 s1·s2·s3 个非零元素
    scale = np.sqrt(snr / (s1 * s2 * s3))
    for k in S1:
        for j2 in S2:
            for j3 in S3:
                B[k, j2, j3] = rng.choice([-1.0, 1.0]) * scale

    X = rng.normal(0, 1.0, size=(n, p1, p2, p3))
    y = np.einsum("nijk,ijk->n", X, B) + rng.normal(0, 1.0, size=n)

    return {
        "X": X,
        "y": y,
        "B": B,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "n": n,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }
