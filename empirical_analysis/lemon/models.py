"""Thin empirical-data adapters to the existing v5 models; no solver changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensor_abess.tensor.abess import tensor_abess
from numerical_simulation.data_generation import TensorSimulationConfig
from numerical_simulation.methods import run_vector_lasso_method, tensor_glasso

from .confounds import validate_tensor_response


@dataclass(frozen=True)
class ModelFit:
    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    loss: float
    rss: float
    sic: float | None = None


def fit_model(name, X, y, model_config):
    """Use adaptive ABESS and the same baselines used by numerical_simulation."""
    X, y = validate_tensor_response(X, y)
    options = {"n_lambdas": 20, "cv_folds": 5, "max_iter": 500,
               "tol": 1e-6, "max_cycles": 6}
    allowed = set(options) | {"target_sparsity", "c", "criterion", "k_max", "m_max",
                              "eps", "adaptive_criterion", "msic_refine_alpha",
                              "msic_refine_gamma", "msic_refine_one_se_scale"}
    unknown = set(model_config) - allowed
    if unknown:
        raise ValueError(f"Unknown model parameter(s): {sorted(unknown)}")
    for key in options:
        options[key] = model_config.get(key, options[key])
    for key in ("n_lambdas", "cv_folds", "max_iter", "max_cycles"):
        if isinstance(options[key], bool) or int(options[key]) != options[key] or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
        options[key] = int(options[key])
    if not 2 <= options["cv_folds"] <= len(y):
        raise ValueError("Inner cv_folds must be between two and the training sample count")
    if not np.isfinite(options["tol"]) or options["tol"] <= 0:
        raise ValueError("tol must be positive and finite")
    if name == "Tensor-ABESS":
        if model_config.get("target_sparsity") is not None:
            raise ValueError("Empirical Tensor-ABESS requires adaptive target_sparsity=None")
        result = tensor_abess(
            X, y, target_sparsity=None,
            c=model_config.get("c", 3.0),
            criterion=model_config.get("criterion", "bic"),
            k_max=model_config.get("k_max", 1),
            m_max=model_config.get("m_max", 15),
            eps=model_config.get("eps", 1e-8),
            adaptive_criterion=model_config.get("adaptive_criterion", "msic"),
            msic_refine_alpha=model_config.get("msic_refine_alpha", 1.5),
            msic_refine_gamma=model_config.get("msic_refine_gamma", 1.0),
            msic_refine_one_se_scale=model_config.get("msic_refine_one_se_scale", 1.0),
            **options,
        )
    elif name == "Tensor-GLasso":
        result = tensor_glasso(X, y, criterion="cv", c=2.0, **options)
    elif name == "Vector Lasso":
        # The shared wrapper reads only these optimization parameters.
        config = TensorSimulationConfig(
            n_lambdas=options["n_lambdas"], cv_folds=options["cv_folds"],
            max_iter=options["max_iter"], tol=options["tol"],
        )
        result = run_vector_lasso_method(X, y, config)
    else:
        raise ValueError(f"Unknown empirical model: {name}")
    B = np.asarray(result.B, dtype=float)
    sic = None if getattr(result, "sic", None) is None else float(result.sic)
    if B.shape != X.shape[1:] or not np.isfinite(B).all():
        raise ValueError(f"{name} returned invalid coefficients")
    if not np.isfinite([result.loss, result.rss]).all() or (sic is not None and not np.isfinite(sic)):
        raise ValueError(f"{name} returned non-finite fit statistics")
    supports = tuple(np.sort(np.asarray(getattr(result, f"S{i}"), dtype=int)) for i in (1, 2, 3))
    for support, dimension in zip(supports, B.shape):
        if support.ndim != 1 or len(np.unique(support)) != len(support) or np.any((support < 0) | (support >= dimension)):
            raise ValueError(f"{name} returned an invalid support")
    if name == "Tensor-ABESS":
        q = int(np.prod([len(support) for support in supports]))
        if q > int(np.floor(0.7 * len(y))):
            raise ValueError("Tensor-ABESS selected q exceeds floor(0.7 * n_train)")
        design = X[np.ix_(np.arange(len(y)), *supports)].reshape(len(y), q)
        if q and np.linalg.matrix_rank(design) != q:
            raise ValueError("Tensor-ABESS selected design does not have full column rank")
    return ModelFit(B, *supports, float(result.loss), float(result.rss), sic)
