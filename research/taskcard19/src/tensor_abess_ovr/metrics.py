"""Fixed-label classification metrics and the declared training-only tie rule."""
from __future__ import annotations

import numpy as np


def tie_priority(y, classes):
    y = np.asarray(y)
    classes = np.asarray(classes)
    counts = np.array([np.sum(y == key) for key in classes])
    if np.any(counts == 0):
        raise ValueError("Training fold does not cover every required class")
    first = np.array([np.flatnonzero(y == key)[0] for key in classes])
    return np.lexsort((first, -counts))


def predict_scores(scores, classes, priority):
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(classes)
    priority = np.asarray(priority, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("Score columns must match classes")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite")
    if sorted(priority.tolist()) != list(range(len(classes))):
        raise ValueError("tie_priority must be a permutation of the class columns")
    if len(scores) == 0:
        return classes[np.array([], dtype=int)], 0
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1., np.max(np.abs(scores), axis=1))
    candidates = (np.max(scores, axis=1, keepdims=True) - scores) <= tolerance[:, None]
    winners = priority[np.argmax(candidates[:, priority], axis=1)]
    return classes[winners], int(np.sum(candidates.sum(axis=1) > 1))


def classification_metrics(y_true, y_pred, classes):
    truth, pred, labels = map(np.asarray, (y_true, y_pred, classes))
    if truth.ndim != 1 or pred.shape != truth.shape or len(truth) == 0:
        raise ValueError("Expected nonempty aligned one-dimensional labels")
    lookup = {key: index for index, key in enumerate(labels.tolist())}
    unknown = (set(truth.tolist()) | set(pred.tolist())) - set(lookup)
    if unknown:
        raise ValueError(f"Unknown evaluation labels: {sorted(unknown, key=str)}")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, ([lookup[key] for key in truth], [lookup[key] for key in pred]), 1)
    support, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(len(labels), dtype=float), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(labels), dtype=float), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(labels), dtype=float), where=(support + predicted) != 0)
    return {
        "n": len(truth), "accuracy": float(tp.sum() / len(truth)),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()),
        "complete_class_coverage": bool(np.all(support > 0)),
        "interpretation": "complete_K_class" if np.all(support > 0) else "descriptive_incomplete_class_coverage",
        "never_predicted": labels[predicted == 0].tolist(), "confusion_matrix": matrix.tolist(),
        "per_class": [{"class_key": key, "precision": float(precision[i]),
                       "recall": float(recall[i]) if support[i] else None,
                       "f1": float(f1[i]), "true_count": int(support[i]),
                       "predicted_count": int(predicted[i]), "true_positive_count": int(tp[i])}
                      for i, key in enumerate(labels.tolist())],
    }
