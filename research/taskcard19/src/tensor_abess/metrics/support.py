"""阶段 C/D 评价指标：各模态 TPR/FPR、矩形精确恢复、Frobenius 系数误差。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SupportMetrics:
    """Mode-wise and joint support recovery metrics."""

    s1_tpr: float
    s1_fpr: float
    s2_tpr: float
    s2_fpr: float
    s3_tpr: float
    s3_fpr: float
    exact_rectangular: bool
    frobenius_error: float


def _mode_metrics(est: np.ndarray, true: np.ndarray, p: int) -> tuple[float, float]:
    est_set = set(est.tolist())
    true_set = set(true.tolist())
    tp = len(est_set & true_set)
    fp = len(est_set - true_set)
    tpr = tp / len(true_set) if true_set else 1.0
    neg = p - len(true_set)
    fpr = fp / neg if neg > 0 else 0.0
    return tpr, fpr


def evaluate_support(
    S1_est: np.ndarray,
    S2_est: np.ndarray,
    S3_est: np.ndarray,
    S1_true: np.ndarray,
    S2_true: np.ndarray,
    S3_true: np.ndarray,
    B_est: np.ndarray,
    B_true: np.ndarray,
    p1: int,
    p2: int,
    p3: int,
) -> SupportMetrics:
    """Compute TPR/FPR per mode, exact rectangular recovery, Frobenius error."""
    s1_tpr, s1_fpr = _mode_metrics(S1_est, S1_true, p1)
    s2_tpr, s2_fpr = _mode_metrics(S2_est, S2_true, p2)
    s3_tpr, s3_fpr = _mode_metrics(S3_est, S3_true, p3)

    # 三个模态同时精确匹配才算矩形支撑完全恢复
    exact = (
        set(S1_est.tolist()) == set(S1_true.tolist())
        and set(S2_est.tolist()) == set(S2_true.tolist())
        and set(S3_est.tolist()) == set(S3_true.tolist())
    )
    frob = float(np.linalg.norm(B_est - B_true))

    return SupportMetrics(
        s1_tpr=s1_tpr,
        s1_fpr=s1_fpr,
        s2_tpr=s2_tpr,
        s2_fpr=s2_fpr,
        s3_tpr=s3_tpr,
        s3_fpr=s3_fpr,
        exact_rectangular=exact,
        frobenius_error=frob,
    )
