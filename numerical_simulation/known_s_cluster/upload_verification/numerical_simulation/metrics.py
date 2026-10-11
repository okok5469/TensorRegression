"""Evaluation metrics for tensor-regression simulations."""

from __future__ import annotations

import numpy as np

from tensor_abess.tensor.ops import tensor_inner


def _support_sets(est: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    est_set = set(np.asarray(est, dtype=int).tolist())
    true_set = set(np.asarray(true, dtype=int).tolist())
    overlap = len(est_set & true_set)
    precision = overlap / max(len(est_set), 1)
    recall = overlap / max(len(true_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_support_metrics(
    S1_est: np.ndarray,
    S2_est: np.ndarray,
    S3_est: np.ndarray,
    S1_true: np.ndarray,
    S2_true: np.ndarray,
    S3_true: np.ndarray,
) -> dict[str, float]:
    """Mode-wise and element-wise support recovery metrics."""
    p1, r1, f1_1 = _support_sets(S1_est, S1_true)
    p2, r2, f1_2 = _support_sets(S2_est, S2_true)
    p3, r3, f1_3 = _support_sets(S3_est, S3_true)

    overlap_elements = (
        len(np.intersect1d(S1_est, S1_true))
        * len(np.intersect1d(S2_est, S2_true))
        * len(np.intersect1d(S3_est, S3_true))
    )
    true_elements = len(S1_true) * len(S2_true) * len(S3_true)
    est_elements = len(S1_est) * len(S2_est) * len(S3_est)
    element_precision = overlap_elements / max(est_elements, 1)
    element_recall = overlap_elements / max(true_elements, 1)
    element_f1 = (
        0.0
        if element_precision + element_recall == 0
        else 2.0 * element_precision * element_recall / (element_precision + element_recall)
    )
    element_fdr = 0.0 if est_elements == 0 else 1.0 - element_precision

    exact = float(
        np.array_equal(np.sort(S1_est), np.sort(S1_true))
        and np.array_equal(np.sort(S2_est), np.sort(S2_true))
        and np.array_equal(np.sort(S3_est), np.sort(S3_true))
    )

    return {
        "mode_precision_1": float(p1),
        "mode_precision_2": float(p2),
        "mode_precision_3": float(p3),
        "mode_tpr_1": float(r1),
        "mode_tpr_2": float(r2),
        "mode_tpr_3": float(r3),
        "mode_f1_1": float(f1_1),
        "mode_f1_2": float(f1_2),
        "mode_f1_3": float(f1_3),
        "mode_precision_macro": float((p1 + p2 + p3) / 3.0),
        "mode_tpr_macro": float((r1 + r2 + r3) / 3.0),
        "mode_f1_macro": float((f1_1 + f1_2 + f1_3) / 3.0),
        "exact_recovery": exact,
        "element_precision": float(element_precision),
        "element_recall": float(element_recall),
        "element_f1": float(element_f1),
        "element_fdr": float(element_fdr),
        "s1_hat": float(len(S1_est)),
        "s2_hat": float(len(S2_est)),
        "s3_hat": float(len(S3_est)),
        "q_hat": float(est_elements),
    }


def compute_coefficient_metrics(B_hat: np.ndarray, B_true: np.ndarray) -> dict[str, float]:
    """Coefficient estimation metrics."""
    denom = max(float(np.linalg.norm(B_true)), np.finfo(float).tiny)
    ree = float(np.linalg.norm(B_hat - B_true) / denom)
    return {"ree": ree}


def compute_prediction_metrics_from_batches(
    batches,
    B_true: np.ndarray,
    B_hat: np.ndarray,
) -> dict[str, float]:
    """Prediction metrics evaluated on a shared stream of independent test batches."""
    total = 0
    sse = 0.0
    signal_sse = 0.0
    sum_y = 0.0
    sum_y2 = 0.0

    for X_batch, y_batch, signal_batch in batches:
        pred_batch = tensor_inner(X_batch, B_hat)
        residual = y_batch - pred_batch
        sse += float(residual @ residual)
        signal_gap = signal_batch - pred_batch
        signal_sse += float(signal_gap @ signal_gap)
        sum_y += float(np.sum(y_batch))
        sum_y2 += float(y_batch @ y_batch)
        total += int(len(y_batch))

    if total == 0:
        raise ValueError("need at least one test observation")

    pmse = sse / total
    signal_mse = signal_sse / total
    mean_y = sum_y / total
    sst = sum_y2 - total * mean_y * mean_y
    r2 = 0.0 if sst <= np.finfo(float).tiny else 1.0 - sse / sst
    return {
        "pmse": float(pmse),
        "signal_mse": float(signal_mse),
        "r2_test": float(r2),
    }
