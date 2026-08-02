"""Demo: group splicing + tensor BCD on 10×5×8 synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_abess.group.bess_fix import group_bess_fix
from tensor_abess.tensor.bcd import tensor_bcd
from tensor_abess.tensor.reduction import build_mode1_design
from tensor_abess.utils.data import make_tensor_data


def main() -> None:
    data = make_tensor_data(
        n=200, p1=10, p2=5, p3=8, s1=3, s2=2, s3=2, snr=25.0, random_state=2026
    )
    X, y = data["X"], data["y"]

    print("=" * 60)
    print("真实支撑: S1={}, S2={}, S3={}".format(data["S1"], data["S2"], data["S3"]))
    print("=" * 60)

    print("\n--- 1) 单模态组 splicing (固定 S2, S3, 选 S1) ---")
    Z, gi = build_mode1_design(X, data["S2"], data["S3"])
    g_res = group_bess_fix(y, Z, gi, s=data["s1"], n_groups=data["p1"])
    print(f"  选中 S1: {np.sort(g_res.active_groups)}")
    print(f"  精确恢复: {set(g_res.active_groups) == set(data['S1'])}")

    print("\n--- 2) 极简 Tensor BCD (已知 s1,s2,s3) ---")
    result = tensor_bcd(
        X, y, data["s1"], data["s2"], data["s3"], max_cycles=25
    )
    print(f"  估计 S1: {result.S1}")
    print(f"  估计 S2: {result.S2}")
    print(f"  估计 S3: {result.S3}")
    print(f"  BCD 循环数: {result.n_cycles}")
    print(f"  最终损失: {result.loss:.6f}")
    print(
        "  支撑精确恢复:",
        set(result.S1) == set(data["S1"])
        and set(result.S2) == set(data["S2"])
        and set(result.S3) == set(data["S3"]),
    )

    print("\n--- 3) 目标函数单调性 (BCD 每步损失) ---")
    losses = result.loss_history
    for t, L in enumerate(losses):
        print(f"  step {t:2d}: L = {L:.6f}")
    monotone = all(losses[i + 1] <= losses[i] + 1e-10 for i in range(len(losses) - 1))
    print(f"  单调下降: {monotone}")


if __name__ == "__main__":
    main()
