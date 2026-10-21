"""Frozen task 11 training-only global MSIC and finite-sample rescorer.

This module performs no search, fitting, prediction, or input generation.
Candidate order and the v5.2 small-model tolerance rule are preserved exactly.
"""
from __future__ import annotations

from numbers import Integral
import math

import numpy as np
from scipy.special import gammaln


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _validate(candidate, dimensions):
    sizes = candidate['sizes']
    _require(len(sizes) == 3 and all(_positive_integer(s) for s in sizes), 'Invalid candidate sizes')
    _require(all(s <= p for s, p in zip(sizes, dimensions)), 'Candidate exceeds mode dimensions')
    _require(_positive_integer(candidate['q']) and candidate['q'] == math.prod(sizes), 'Candidate q disagrees with sizes')
    _require(np.isfinite(candidate['rss']) and candidate['rss'] >= 0, 'Invalid or nonfinite RSS')
    supports = candidate['support']
    _require(len(supports) == 3 and list(map(len, supports)) == list(sizes), 'Support sizes disagree')
    for support, p in zip(supports, dimensions):
        _require(all(isinstance(i, Integral) and not isinstance(i, bool) and 0 <= i < p for i in support),
                 'Invalid support index')
        _require(list(support) == sorted(set(support)), 'Support must be unique and sorted')
    if 'B' in candidate:
        B = np.asarray(candidate['B'])
        _require(B.shape == tuple(dimensions) and np.isfinite(B).all(), 'Invalid or nonfinite B')
        active = B[np.ix_(*supports)]
        _require(np.count_nonzero(active) == np.count_nonzero(B), 'Coefficient outside support')


def select_candidates(candidates, n, p, finite_sample=False):
    """Select from original ordered ``rss/sizes/q/support`` candidate records.

    ``B`` is optional metadata; when present its shape, finite values, and
    support containment are checked. Training validity is mandatory; external
    prediction metrics are never accepted as a scoring argument. Caps and the
    complete original grid are independently audited by the caller.
    """
    _require(_positive_integer(n), 'Sample count must be a positive integer')
    _require(len(p) == 3 and all(_positive_integer(d) for d in p), 'Invalid mode dimensions')
    _require(isinstance(finite_sample, bool), 'finite_sample must be boolean')
    _require(len(candidates) > 0, 'Empty candidate library')
    ell = np.log(max(np.log(n), np.e))
    components = []
    for candidate in candidates:
        _validate(candidate, p)
        sizes = candidate['sizes']
        q = math.prod(sizes)
        fit = float(n * np.log(max(candidate['rss'] / n, np.finfo(float).tiny)))
        df = float(1.5 * sizes[0] * sizes[1] * sizes[2] * ell)
        multiplicity = float(ell * sum(float(gammaln(d + 1) - gammaln(s + 1) - gammaln(d - s + 1))
                                      for d, s in zip(p, sizes)))
        baseline = fit + df + multiplicity
        correction = float(ell * q ** 2 / (2 * n)) if finite_sample else 0.0
        score = baseline + correction
        _require(np.isfinite(score), 'Nonfinite candidate score')
        components.append({'fit_term': fit, 'df_penalty': df, 'multiplicity_penalty': multiplicity,
                           'baseline_score': baseline, 'fs_penalty': correction, 'score': score})
    scores = [item['score'] for item in components]
    minimum = min(range(len(candidates)), key=scores.__getitem__)
    q_best = math.prod(candidates[minimum]['sizes'])
    tolerance = float(np.sqrt(max(q_best, 1) * max(np.log(max(n, 2)), 1.0)))
    eligible = [i for i, score in enumerate(scores) if score <= scores[minimum] + tolerance]
    chosen = min(eligible, key=lambda i: (math.prod(candidates[i]['sizes']), scores[i], sum(candidates[i]['sizes'])))
    return {'alpha': 1.5, 'gamma': 1.0, 'scale': 1.0, 'finite_sample': finite_sample,
            'minimum_index': minimum, 'minimum_q': q_best, 'minimum_score': scores[minimum],
            'tolerance': tolerance, 'eligible_indices': eligible, 'selected_index': chosen,
            'selected_score': scores[chosen], 'candidate_scores': scores, 'candidate_terms': components}
