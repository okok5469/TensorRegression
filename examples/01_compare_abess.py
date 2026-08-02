"""Compare official abess package with our vector ABESS implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abess.linear import LinearRegression
from tensor_abess.utils.data import make_regression_data
from tensor_abess.vector.abess import abess, bess_fix


def support_metrics(est: np.ndarray, true: np.ndarray) -> dict:
    est_set = set(est.tolist())
    true_set = set(true.tolist())
    tp = len(est_set & true_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)
    tpr = tp / len(true_set) if true_set else 1.0
    fpr = fp / (len(est_set) + len(true_set - est_set)) if est_set else 0.0
    exact = est_set == true_set
    return {"tpr": tpr, "fpr": fpr, "exact": exact, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    data = make_regression_data(n=120, p=40, s_true=5, random_state=2026)
    X, y = data["X"], data["y"]
    true_support = data["support"]
    s_true = data["s_true"]

    print("=" * 60)
    print("1) abess 官方包 — 稀疏度已知 (support_size = s_true)")
    print("=" * 60)
    model_fix = LinearRegression(
        support_size=s_true,
        fit_intercept=False,
        max_iter=20,
        exchange_num=5,
    )
    model_fix.fit(X, y)
    abess_support = np.where(np.abs(model_fix.coef_) > 1e-10)[0]
    print(f"  选中变量: {np.sort(abess_support)}")
    print(f"  真实支撑: {true_support}")
    print(f"  指标: {support_metrics(abess_support, true_support)}")

    print()
    print("=" * 60)
    print("2) 自实现 BESS.Fix(s) — s = s_true")
    print("=" * 60)
    ours_fix = bess_fix(X, y, s_true)
    print(f"  选中变量: {np.sort(ours_fix.support)}")
    print(f"  损失 L_n: {ours_fix.loss:.6f}")
    print(f"  迭代次数: {ours_fix.n_iter}")
    print(f"  指标: {support_metrics(ours_fix.support, true_support)}")

    print()
    print("=" * 60)
    print("3) abess 官方包 — EBIC 自适应选 s")
    print("=" * 60)
    model_adapt = LinearRegression(
        fit_intercept=False,
        ic_type="ebic",
        path_type="seq",
        max_iter=20,
        exchange_num=5,
    )
    model_adapt.fit(X, y)
    abess_adapt_support = np.where(np.abs(model_adapt.coef_) > 1e-10)[0]
    print(f"  选中 s: {abess_adapt_support.size}")
    print(f"  选中变量: {np.sort(abess_adapt_support)}")
    print(f"  指标: {support_metrics(abess_adapt_support, true_support)}")

    print()
    print("=" * 60)
    print("4) 自实现 ABESS — SIC 自适应选 s")
    print("=" * 60)
    ours_adapt = abess(X, y)
    print(f"  选中 s: {ours_adapt.s}")
    print(f"  选中变量: {np.sort(ours_adapt.support)}")
    print(f"  SIC: {ours_adapt.sic:.4f}")
    print(f"  指标: {support_metrics(ours_adapt.support, true_support)}")

    print()
    print("=" * 60)
    print("5) 系数对比 (BESS.Fix, 共同支撑)")
    print("=" * 60)
    common = np.intersect1d(abess_support, ours_fix.support)
    if common.size:
        for j in common:
            print(
                f"  j={j:2d}: abess={model_fix.coef_[j]:+.4f}, "
                f"ours={ours_fix.beta[j]:+.4f}"
            )


if __name__ == "__main__":
    main()
