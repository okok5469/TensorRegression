"""The fixed comparison group; all transformations are learned within training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from .metrics import classification_metrics, predict_scores, tie_priority
from .preprocessing import TensorStandardizer


def stable_seed(seed, *identifiers):
    payload = json.dumps([int(seed), *map(str, identifiers)], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def select_parameter(method, grid, means):
    best = max(means)
    candidates = [index for index, value in enumerate(means) if value >= best - 1e-12]
    stronger = (lambda index: -grid[index]) if method == "ridge_ovr" else (lambda index: grid[index])
    return grid[min(candidates, key=lambda index: (stronger(index), index))]


def make_inner_folds(y, classes, dependency, seed, *, groups=None, times=None):
    y = np.asarray(y)
    required = set(classes)
    notes = []
    for count in (3, 2):
        try:
            if dependency == "time":
                if times is None:
                    raise ValueError("Time validation requires declared time values")
                times = np.asarray(times)
                unique = np.unique(times)
                if len(unique) <= count:
                    raise ValueError("Too few distinct time points for forward folds")
                folds = [(np.flatnonzero(np.isin(times, unique[a])), np.flatnonzero(np.isin(times, unique[b])))
                         for a, b in TimeSeriesSplit(n_splits=count).split(unique)]
            elif dependency == "group":
                if groups is None or len(np.unique(groups)) < count:
                    raise ValueError("Too few groups for inner folds")
                splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=int(seed))
                folds = list(splitter.split(np.zeros(len(y)), y, groups))
            elif dependency == "iid":
                splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=int(seed))
                with warnings.catch_warnings(record=True):
                    folds = list(splitter.split(np.zeros(len(y)), y))
            else:
                raise ValueError(f"Unknown dependence protocol: {dependency}")
            for train, evaluate in folds:
                if set(y[train]) != required or set(y[evaluate]) != required:
                    raise ValueError("Inner folds must cover every required class in training and evaluation")
                if dependency == "group" and set(np.asarray(groups)[train]) & set(np.asarray(groups)[evaluate]):
                    raise ValueError("Inner fold group overlap")
                if dependency == "time" and max(times[train]) >= min(times[evaluate]):
                    raise ValueError("Inner time fold is not forward")
            return folds, {"preferred_folds": 3, "actual_folds": count, "downgrade_reasons": notes}
        except ValueError as error:
            notes.append(f"{count}-fold: {error}")
    raise ValueError("Inner class-coverage protocol blocked: " + "; ".join(notes))


@dataclass
class BaselineFit:
    method: str
    classes_: np.ndarray
    tie_priority_: np.ndarray
    preprocessor_: TensorStandardizer | None
    estimator_: object | None
    prior_: np.ndarray
    parameter_: float | None
    numerical_ok: bool
    axis_metadata: object

    def decision_function(self, X):
        if self.method == "majority":
            return np.broadcast_to(self.prior_, (len(X), len(self.classes_))).copy()
        transformed = self.preprocessor_.transform(X, axis_metadata=self.axis_metadata).reshape(len(X), -1, order="C")
        if self.method == "ridge_ovr":
            result = self.estimator_.predict(transformed)
        else:
            result = self.estimator_.decision_function(transformed)
            if result.ndim == 1:
                result = np.column_stack((-result, result))
            columns = [list(self.estimator_.classes_).index(key) for key in self.classes_]
            result = result[:, columns]
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite baseline score")
        return np.asarray(result, dtype=np.float64)


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Dataset total wall-clock resource budget reached")


def _fit_estimator(method, X, y, classes, parameter, seed, phase, diagnostics, deadline):
    _deadline(deadline)
    if method == "ridge_ovr":
        estimator = Ridge(alpha=float(parameter), fit_intercept=True, solver="svd")
        response = np.column_stack([y == key for key in classes]).astype(np.float64)
        estimator.fit(X, response)
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise FloatingPointError("Nonfinite Ridge coefficients")
        diagnostics.append({**phase, "method": method, "seed": seed, "parameter": parameter,
                            "solver": "svd", "fit_intercept": True, "status": "COMPLETE"})
        return estimator, True
    solver, penalty = {"multinomial_l2": ("lbfgs", "l2"), "multinomial_l1": ("saga", "l1")}[method]
    estimator, converged = None, False
    for max_iter in (5000, 20000):
        _deadline(deadline)
        estimator = LogisticRegression(C=float(parameter), penalty=penalty, solver=solver, multi_class="multinomial",
                                       fit_intercept=True, class_weight=None, tol=1e-6, max_iter=max_iter,
                                       random_state=int(seed), n_jobs=1, warm_start=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
        hit_limit = bool(np.any(estimator.n_iter_ >= max_iter))
        convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        converged = not (hit_limit or convergence_warning)
        diagnostics.append({**phase, "method": method, "seed": seed, "parameter": parameter,
                            "solver": solver, "penalty": penalty, "objective": "multinomial",
                            "fit_intercept": True, "class_weight": None, "tol": 1e-6, "max_iter": max_iter,
                            "n_iter": estimator.n_iter_.tolist(), "hit_iteration_limit": hit_limit,
                            "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
                            "status": "COMPLETE" if converged else "NUMERICAL_UNRESOLVED"})
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise FloatingPointError("Nonfinite Logistic coefficients")
        if converged:
            break
    return estimator, converged


def fit_baseline(method, X, y, classes, *, inner_folds, seed, fold_id, axis_metadata, diagnostics, deadline=None):
    classes, y = np.asarray(classes), np.asarray(y)
    priority = tie_priority(y, classes)
    prior = np.array([np.mean(y == key) for key in classes])
    if method == "majority":
        diagnostics.append({"method": method, "fold": fold_id, "phase": "final", "status": "COMPLETE",
                            "training_counts": [int(np.sum(y == key)) for key in classes]})
        return BaselineFit(method, classes, priority, None, None, prior, None, True, axis_metadata)
    grid = [float(10.**power) for power in range(-4, 5)]
    means, numerical_ok = [], True
    for candidate_index, parameter in enumerate(grid):
        values = []
        for inner_index, (train, evaluate) in enumerate(inner_folds):
            _deadline(deadline)
            transform = TensorStandardizer(axis_metadata=axis_metadata)
            tr = transform.fit_transform(X[train], axis_metadata=axis_metadata).reshape(len(train), -1, order="C")
            phase = {"fold": fold_id, "phase": "inner", "inner_fold": inner_index, "candidate_index": candidate_index}
            candidate_seed = stable_seed(seed, method, fold_id, "candidate", candidate_index, "inner", inner_index)
            estimator, ok = _fit_estimator(method, tr, y[train], classes, parameter, candidate_seed, phase, diagnostics, deadline)
            candidate = BaselineFit(method, classes, tie_priority(y[train], classes), transform, estimator, prior, parameter, ok, axis_metadata)
            labels, ties = predict_scores(candidate.decision_function(X[evaluate]), classes, candidate.tie_priority_)
            score = classification_metrics(y[evaluate], labels, classes)["macro_f1"]
            diagnostics.append({**phase, "method": method, "parameter": parameter, "seed": candidate_seed,
                                "phase": "inner_score", "macro_f1": score, "tie_count": ties})
            values.append(score)
            numerical_ok &= ok
        means.append(float(np.mean(values)))
    parameter = select_parameter(method, grid, means)
    diagnostics.append({"method": method, "fold": fold_id, "phase": "selection", "grid": grid,
                        "mean_macro_f1": means, "selected_parameter": parameter, "tie_tolerance": 1e-12,
                        "numerical_ok_all_required_candidates": numerical_ok})
    _deadline(deadline)
    transform = TensorStandardizer(axis_metadata=axis_metadata)
    tr = transform.fit_transform(X, axis_metadata=axis_metadata).reshape(len(X), -1, order="C")
    final_seed = stable_seed(seed, method, fold_id, "final", parameter)
    estimator, ok = _fit_estimator(method, tr, y, classes, parameter, final_seed,
                                   {"fold": fold_id, "phase": "final"}, diagnostics, deadline)
    return BaselineFit(method, classes, priority, transform, estimator, prior, parameter, bool(numerical_ok and ok), axis_metadata)
