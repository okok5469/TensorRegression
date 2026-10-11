"""Training-only fits with external tuning; existing tensor mathematics unchanged.

Lasso minimizes RSS/(2n) + lambda*L1; Ridge uses lambda*L2/2.
Tensor-GLasso retains marginal screening, sequential grouped support selection,
the existing empty-support rules, and rectangular OLS refitting. A fixed ratio
is applied to each training-only local grouped design's lambda_max. No adapter
uses internal CV. Grids contain response-independent ratios or Ridge lambdas.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tensor_abess.tensor.ops import refit_tensor_coefficients, support_from_tensor
from tensor_abess.tensor.reduction import build_mode_design_subset
from tensor_abess.utils.glasso import GroupLassoFit, group_soft_threshold
from tensor_abess.utils.init import (_mode_slice_scores, _candidate_set,
                                    _top_from_candidates, _fallback_from_candidates)
from ..confounds import validate_tensor_response
from ..models import fit_model


@dataclass(frozen=True)
class EEGModelFit:
    B: np.ndarray
    parameters: dict
    supports: tuple[np.ndarray, np.ndarray, np.ndarray]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 4 or X.shape[1:] != self.B.shape or not np.isfinite(X).all():
            raise ValueError('Prediction tensor dimensions or values are invalid')
        return np.einsum('nijk,ijk->n', X, self.B)


def make_parameter_grid(name, X, y, config):
    """Return predeclared candidates; X and y never determine the grid."""
    del X, y
    if name == 'Tensor-ABESS':
        return [None]
    key, parameter_key, default = {
        'Ridge': ('ridge_lambdas', 'lambda', np.geomspace(1e-4, 1e4, 20)),
        'Vector Lasso': ('lambda_ratios', 'lambda_ratio', np.geomspace(1, .01, 20)),
        'Tensor-GLasso': ('lambda_ratios', 'lambda_ratio', np.geomspace(1, .01, 20)),
    }[name]
    values = np.asarray(config.get(key, default), dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Regularization grid must contain finite positive values')
    return [{parameter_key: float(value)} for value in values]


def _penalty(parameter, lambda_max, *, ridge=False):
    if not isinstance(parameter, dict) or set(parameter) not in ({'lambda'}, {'lambda_ratio'}):
        raise ValueError('Parameter must specify exactly lambda or lambda_ratio')
    value = float(next(iter(parameter.values())))
    if not np.isfinite(value) or value < 0 or (ridge and value <= 0):
        raise ValueError('Regularization must be finite and nonnegative (positive for Ridge)')
    if ridge and 'lambda_ratio' in parameter:
        raise ValueError('Ridge requires an absolute lambda')
    return value * lambda_max if 'lambda_ratio' in parameter else value


def _optimization_options(options):
    values = {key: options.get(key, default) for key, default in
              [('max_iter', 500), ('tol', 1e-6), ('max_cycles', 6)]}
    for key in ('max_iter', 'max_cycles'):
        val = values[key]
        if isinstance(val, bool) or int(val) != val or val < 1:
            raise ValueError(f'{key} must be a positive integer')
        values[key] = int(val)
    if not np.isfinite(values['tol']) or values['tol'] <= 0:
        raise ValueError('tol must be positive and finite')
    return values


def _spectral_lipschitz(Z):
    # Exactly ||Z||_2^2/n via the smaller Gram, without a p-by-p matrix.
    gram = Z @ Z.T if Z.shape[0] <= Z.shape[1] else Z.T @ Z
    return max(float(np.linalg.eigvalsh(gram)[-1]), 0.0) / len(Z)


def _ista(Z, y, lam, options, lipschitz, group_index=None):
    """The original zero-start ISTA update, with a cached exact spectral norm."""
    beta = np.zeros(Z.shape[1], dtype=float)
    step = 1.0 if lipschitz <= 0 else 1.0 / lipschitz
    for _ in range(options['max_iter']):
        grad = Z.T @ (Z @ beta - y) / len(y)
        shifted = beta - step * grad
        if group_index is None:
            new_beta = np.sign(shifted) * np.maximum(np.abs(shifted) - lam * step, 0.0)
        else:
            new_beta = group_soft_threshold(shifted, group_index, lam * step)
        difference = np.linalg.norm(new_beta - beta)
        threshold = options['tol'] * (1.0 + np.linalg.norm(beta))
        beta = new_beta
        if difference <= threshold:
            break
    return beta


def _group_fit(Z, y, groups, parameter, options, lipschitz, lam_max):
    lam = _penalty(parameter, lam_max)
    beta = _ista(Z, y, lam, options, lipschitz, groups)
    residual = y - Z @ beta
    active = np.array([g for g in np.unique(groups)
                       if np.linalg.norm(beta[groups == g]) > 1e-8], dtype=int)
    return GroupLassoFit(beta, float(residual @ residual), active, lam)


def _fit_tensor_glasso(X, y, parameter, options, cache):
    scores = [_mode_slice_scores(X, y, mode) for mode in (1, 2, 3)]
    candidates = [_candidate_set(score, len(y)) for score in scores]
    supports = [_top_from_candidates(score, cand, len(y))
                for score, cand in zip(scores, candidates)]
    trace = []

    def update(mode, groups, previous, phase):
        # Local cache exists only within one training dataset's fit path.
        key = (mode, tuple(groups), *(tuple(s) for i, s in enumerate(supports) if i != mode-1))
        if key not in cache:
            Z, gi, original = build_mode_design_subset(X, mode, groups, *supports)
            correlations = Z.T @ y / len(y)
            lam_max = max(float(np.linalg.norm(correlations[gi == g])) for g in np.unique(gi))
            cache[key] = (Z, gi, original, _spectral_lipschitz(Z), lam_max)
        Z, gi, original, lipschitz, lam_max = cache[key]
        fit = _group_fit(Z, y, gi, parameter, options, lipschitz, lam_max)
        trace.append({'phase': phase, 'mode': mode, 'lambda_max': lam_max,
                      'lambda': float(fit.lam)})
        selected = original[fit.active_groups]
        if selected.size:
            return np.sort(selected)
        return (_fallback_from_candidates(scores[mode-1], original)
                if phase == 'initialization' else previous.copy())

    for mode in (1, 2, 3):
        supports[mode-1] = update(mode, candidates[mode-1], supports[mode-1], 'initialization')
    B = refit_tensor_coefficients(X, y, *supports)
    cycles = 0
    for cycle in range(options['max_cycles']):
        previous = [s.copy() for s in supports]
        for mode in (1, 2, 3):
            supports[mode-1] = update(mode, np.arange(X.shape[mode]),
                                      previous[mode-1], 'bcd')
        B = refit_tensor_coefficients(X, y, *supports)
        cycles = cycle + 1
        if all(np.array_equal(current, old) for current, old in zip(supports, previous)):
            break
    params = dict(parameter, internal_cv=False, n_cycles=cycles, group_lambda_trace=trace,
                  parameterization='training_local_lambda_max_ratio' if 'lambda_ratio' in parameter
                  else 'absolute_lambda')
    return EEGModelFit(B, params, tuple(supports))


def fit_eeg_path(name, X_standardized, y_centered, parameters, options):
    """Fit candidates with training-local design caching and zero-start ISTA.

    No cross-lambda warm starts: finite-iteration coefficients retain the core
    final fit definition. This cache is never shared between training folds.
    """
    X, y = validate_tensor_response(X_standardized, y_centered)
    options = dict(options)
    opt = _optimization_options(options)
    flat = X.reshape(len(y), -1, order='C')
    cache = {}
    lasso_lipschitz = _spectral_lipschitz(flat) if name == 'Vector Lasso' else None
    lam_max = float(np.max(np.abs(flat.T @ y / len(y)))) if name == 'Vector Lasso' else None
    ridge_gram = flat @ flat.T if name == 'Ridge' else None
    fits = []
    for parameter in parameters:
        if name == 'Tensor-ABESS':
            if parameter is not None or options.get('criterion', 'bic') != 'bic':
                raise ValueError('Adaptive ABESS retains BIC initialization without internal CV')
            result = fit_model(name, X, y, options)
            supports = (result.S1, result.S2, result.S3)
            params = dict(options, internal_cv=False, target_sparsity=None,
                          s1=len(result.S1), s2=len(result.S2), s3=len(result.S3),
                          q=int(np.prod([len(s) for s in supports])), sic=result.sic)
            fits.append(EEGModelFit(result.B, params, supports))
            continue
        if name == 'Tensor-GLasso':
            fits.append(_fit_tensor_glasso(X, y, parameter, opt, cache))
            continue
        if name == 'Ridge':
            lam = _penalty(parameter, 1.0, ridge=True)
            dual = np.linalg.solve(ridge_gram + len(y)*lam*np.eye(len(y)), y)
            beta = flat.T @ dual
        elif name == 'Vector Lasso':
            lam = _penalty(parameter, lam_max)
            beta = _ista(flat, y, lam, opt, lasso_lipschitz)
        else:
            raise ValueError(f'Unknown EEG model: {name}')
        B = beta.reshape(X.shape[1:], order='C')
        params = dict(parameter, lambda_actual=lam, internal_cv=False)
        if name == 'Vector Lasso':
            params['lambda_max'] = lam_max
        fits.append(EEGModelFit(B, params, support_from_tensor(B, tol=1e-8)))
    if any(not np.isfinite(fit.B).all() for fit in fits):
        raise ValueError(f'{name} returned nonfinite coefficients')
    return fits


def fit_eeg_model(name, X_standardized, y_centered, parameter, options):
    return fit_eeg_path(name, X_standardized, y_centered, [parameter], options)[0]
