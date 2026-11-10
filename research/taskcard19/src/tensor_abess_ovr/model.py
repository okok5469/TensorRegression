"""Independent frozen regressors for centered class indicators, with restored priors."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import multiprocessing
import time
import traceback
from pathlib import Path

import numpy as np

from .identity import code_identity, verify_release
from .preprocessing import TensorStandardizer, numeric_tensor


def core_profile():
    return json.loads(Path(__file__).with_name("core_profile.json").read_text(encoding="utf-8"))


def to_jsonable(value):
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite core state/loss")
    return value


def uniform_labels(values, *, name="labels"):
    raw = list(values)
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    integer = all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in raw)
    string = all(isinstance(v, (str, np.str_)) and bool(str(v)) for v in raw)
    if not (integer or string):
        raise ValueError(f"{name} must be uniform integer or nonempty string labels; missing/mixed labels are forbidden")
    return np.asarray(raw, dtype=np.int64 if integer else str)


def decide_scores(scores, classes, tie_priority):
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(classes)
    priority = np.asarray(tie_priority, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(classes) or len(classes) < 2:
        raise ValueError("score shape must be (n,K), K>=2")
    if sorted(priority.tolist()) != list(range(len(classes))):
        raise ValueError("tie priority must be a permutation of class columns")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1., np.max(np.abs(scores), axis=1))
    tied = (np.max(scores, axis=1)[:, None] - scores) <= tolerance[:, None]
    selected = priority[np.argmax(tied[:, priority], axis=1)]
    return classes[selected], int(np.count_nonzero(tied.sum(axis=1) > 1))


def _fit_one(X, z, parameters):
    from tensor_abess.tensor.abess import tensor_abess
    inspect.signature(tensor_abess).bind(X, z, **parameters)
    return to_jsonable(tensor_abess(X, z, **parameters))


def _core_worker(sender, X, z, parameters):
    try:
        sender.send({"ok": True, "state": _fit_one(X, z, parameters)})
    except BaseException as error:
        sender.send({"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
    finally:
        sender.close()


def _bounded_fit(X, z, parameters, timeout, deadline=None):
    if timeout is None:
        return _fit_one(X, z, parameters)
    call_deadline = time.monotonic() + float(timeout)
    if deadline is not None:
        call_deadline = min(call_deadline, deadline)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_core_worker, args=(sender, X, z, parameters))
    process.start()
    sender.close()
    try:
        if not receiver.poll(max(0.0, call_deadline - time.monotonic())):
            raise TimeoutError(f"core call exceeded {timeout} seconds")
        payload = receiver.recv()
        process.join(timeout=5)
        if not payload["ok"]:
            raise RuntimeError(payload["error"] + "\n" + payload["traceback"])
        if process.exitcode not in (0, None):
            raise RuntimeError(f"core worker exited with code {process.exitcode}")
        return payload["state"]
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join()


class TensorABESSOvR:
    def __init__(self, classes=None, axis_metadata=None, core_timeout=900.0):
        if core_timeout is not None and (not np.isfinite(core_timeout) or core_timeout <= 0):
            raise ValueError("core_timeout must be positive or None")
        self.classes = classes
        self.axis_metadata = axis_metadata
        self.core_timeout = core_timeout
        self._fitted = False

    def fit(self, X, y, *, sample_ids=None, axis_metadata=None, deadline=None):
        self._fitted = False
        self.fit_diagnostics_ = []
        self.release_identity_ = verify_release()
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("deadline must be a finite time.monotonic absolute value")
        X = numeric_tensor(X)
        y = uniform_labels(y)
        if len(y) != len(X):
            raise ValueError("X and labels sample counts differ")
        classes = uniform_labels(self.classes if self.classes is not None else list(dict.fromkeys(y.tolist())), name="classes")
        if classes.dtype.kind != y.dtype.kind or len(set(classes.tolist())) != len(classes) or len(classes) < 2:
            raise ValueError("classes must be unique, K>=2, and have the same uniform type as labels")
        missing = set(classes.tolist()) - set(y.tolist())
        unknown = set(y.tolist()) - set(classes.tolist())
        if missing or unknown:
            raise ValueError(f"training classes missing={missing}, unknown={unknown}")
        ids = [str(v) for v in sample_ids] if sample_ids is not None else [f"row_{i}" for i in range(len(y))]
        if len(ids) != len(y) or len(set(ids)) != len(ids) or any(not v for v in ids):
            raise ValueError("sample_ids must be unique, nonempty and aligned to X rows")
        self.classes_ = classes
        self.training_sample_ids_ = ids
        self.training_class_labels_ = y.copy()
        self.counts_ = np.asarray([(y == label).sum() for label in classes], dtype=np.int64)
        self.class_prior_ = self.counts_ / len(y)
        self.first_occurrence_ = np.asarray([int(np.flatnonzero(y == label)[0]) for label in classes], dtype=np.int64)
        self.tie_priority_ = np.lexsort((self.first_occurrence_, -self.counts_))
        self.preprocessor_ = TensorStandardizer(self.axis_metadata if axis_metadata is None else axis_metadata)
        transformed = self.preprocessor_.fit_transform(X)
        self.profile_ = core_profile()
        self.code_identity_ = code_identity()
        states = []
        for column, label in enumerate(classes):
            started = time.perf_counter()
            diagnostic = {"stage": "core_fit", "class_label": to_jsonable(label), "class_column": column, "status": "RUNNING", "parameters": self.profile_["parameters"], "timeout_seconds": self.core_timeout}
            self.fit_diagnostics_.append(diagnostic)
            try:
                timeout = self.core_timeout
                if deadline is not None:
                    remaining = float(deadline - time.monotonic())
                    if remaining <= 0:
                        raise TimeoutError("dataset deadline exhausted before class core call")
                    timeout = remaining if timeout is None else min(timeout, remaining)
                    diagnostic["remaining_dataset_seconds"] = remaining
                    diagnostic["timeout_seconds"] = timeout
                response = (y == label).astype(np.float64) - self.class_prior_[column]
                state = _bounded_fit(transformed, response, self.profile_["parameters"], timeout, deadline=deadline)
                coefficient = np.asarray(state["B"], dtype=np.float64)
                if coefficient.shape != tuple(X.shape[1:]) or not np.isfinite(coefficient).all():
                    raise ValueError("invalid/nonfinite core coefficients")
                for field in ("loss", "rss", "sic"):
                    if not np.isfinite(state[field]):
                        raise ValueError(f"nonfinite core {field}")
                states.append(state)
                diagnostic.update(status="COMPLETE", loss=state["loss"], rss=state["rss"], sic=state["sic"], response_mean=float(response.mean()), response_scale_applied=1.0)
            except BaseException as error:
                diagnostic.update(status="FAILED", error_type=type(error).__name__, error=str(error))
                raise RuntimeError(f"core_fit class={to_jsonable(label)!r} failed: {error}") from error
            finally:
                diagnostic["elapsed_seconds"] = time.perf_counter() - started
        self.states_ = states
        self.coef_ = np.asarray([state["B"] for state in states], dtype=np.float64)
        self.supports_ = [[state["S1"], state["S2"], state["S3"]] for state in states]
        self._fitted = True
        self.tie_count_ = 0
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("classifier is not fitted; all classes must succeed")

    def decision_function(self, X, *, axis_metadata=None):
        self._require_fitted()
        transformed = self.preprocessor_.transform(X, axis_metadata=axis_metadata)
        scores = transformed.reshape(len(transformed), -1, order="C") @ self.coef_.reshape(len(self.classes_), -1, order="C").T + self.class_prior_
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite class scores")
        return scores

    def predict(self, X, *, axis_metadata=None):
        scores = self.decision_function(X, axis_metadata=axis_metadata)
        labels, self.tie_count_ = decide_scores(scores, self.classes_, self.tie_priority_)
        return labels

    def save(self, path):
        self._require_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {"coef": self.coef_, "classes": self.classes_, "class_prior": self.class_prior_, "counts": self.counts_, "first_occurrence": self.first_occurrence_, "tie_priority": self.tie_priority_, "mean": self.preprocessor_.mean_, "scale": self.preprocessor_.scale_, "constant_mask": self.preprocessor_.constant_mask_}
        np.savez_compressed(path / "model.npz", **arrays)
        metadata = {"schema_version": 1, "application_version": "0.1.0", "classes": to_jsonable(self.classes_), "axis_metadata": self.preprocessor_.axis_metadata_, "shape": list(self.preprocessor_.shape_), "dtype": "float64", "order": "C", "missing_policy": "error", "states": self.states_, "supports": self.supports_, "training_sample_ids": self.training_sample_ids_, "training_class_labels": to_jsonable(self.training_class_labels_), "profile": self.profile_, "code_identity": self.code_identity_, "fit_diagnostics": self.fit_diagnostics_, "core_timeout": self.core_timeout, "arrays_sha256": hashlib.sha256((path / "model.npz").read_bytes()).hexdigest()}
        (path / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        release_identity = verify_release()
        path = Path(path)
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if hashlib.sha256((path / "model.npz").read_bytes()).hexdigest() != metadata["arrays_sha256"]:
            raise ValueError("model.npz SHA256 mismatch")
        if metadata["profile"] != core_profile() or metadata["code_identity"] != code_identity():
            raise ValueError("saved model configuration/code identity differs from installed code")
        model = cls(classes=metadata["classes"], axis_metadata=metadata["axis_metadata"], core_timeout=metadata["core_timeout"])
        model.release_identity_ = release_identity
        with np.load(path / "model.npz", allow_pickle=False) as arrays:
            for name, key in (("coef_", "coef"), ("classes_", "classes"), ("class_prior_", "class_prior"), ("counts_", "counts"), ("first_occurrence_", "first_occurrence"), ("tie_priority_", "tie_priority")):
                setattr(model, name, arrays[key].copy())
            standardizer = TensorStandardizer(metadata["axis_metadata"])
            standardizer.shape_ = tuple(metadata["shape"])
            standardizer.axis_metadata_ = metadata["axis_metadata"]
            standardizer.mean_, standardizer.scale_, standardizer.constant_mask_ = (arrays[key].copy() for key in ("mean", "scale", "constant_mask"))
        model.preprocessor_ = standardizer
        model.states_, model.supports_ = metadata["states"], metadata["supports"]
        model.training_sample_ids_ = metadata["training_sample_ids"]
        model.training_class_labels_ = uniform_labels(metadata["training_class_labels"])
        model.profile_, model.code_identity_ = metadata["profile"], metadata["code_identity"]
        model.fit_diagnostics_ = metadata["fit_diagnostics"]
        model._fitted, model.tie_count_ = True, 0
        return model
