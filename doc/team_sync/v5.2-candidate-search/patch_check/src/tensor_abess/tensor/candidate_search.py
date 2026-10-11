"""Explicit candidate-level capacity and training-only search for research v5.2."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import product
from time import perf_counter

import numpy as np


def legal_size_triples(caps, max_params):
    return [sizes for sizes in product(*(range(1, int(cap) + 1) for cap in caps))
            if int(np.prod(sizes)) <= max_params]


def candidate_size_triples(s1_max, s2_max, s3_max, max_params, *, window=3,
                           grid_policy='legacy', candidate_budget=2000):
    if grid_policy not in ('legacy', 'full', 'coarse_to_fine'):
        raise ValueError('grid_policy must be legacy, full, or coarse_to_fine')
    caps = (s1_max, s2_max, s3_max)
    if min(caps) < 1 or max_params < 1 or candidate_budget < 1:
        raise ValueError('Positive caps, max_params and candidate_budget are required')
    if grid_policy == 'full':
        triples = legal_size_triples(caps, max_params)
    else:
        def values_for_cap(cap):
            near_cap = range(max(1, cap - window + 1), cap + 1)
            anchors = {1, max(1, round(0.4 * cap)), max(1, round(0.6 * cap))}
            return sorted({*near_cap, *anchors})
        triples = [sizes for sizes in product(*(values_for_cap(cap) for cap in caps))
                   if int(np.prod(sizes)) <= max_params] or [(1, 1, 1)]
    check_budget(len(triples), candidate_budget, caps, max_params, grid_policy)
    return triples


def check_budget(count, budget, caps, max_params, policy):
    if count > budget:
        raise ValueError(f'Candidate budget exceeded: count={count}, budget={budget}, '
                         f'caps={caps}, Q={max_params}, grid_policy={policy}')


def training_fingerprint(X, y):
    digest = sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode('ascii'))
        digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def validate_candidate_state(supports, caps, max_params, dimensions, *, stage):
    sizes = tuple(len(support) for support in supports)
    valid = all(1 <= size <= min(cap, dimension) and len(np.unique(support)) == size
                and np.all((support >= 0) & (support < dimension))
                for support, size, cap, dimension in zip(supports, sizes, caps, dimensions))
    if not valid or int(np.prod(sizes)) > max_params:
        raise ValueError(f'Illegal {stage} state: sizes={sizes}, caps={caps}, Q={max_params}; '
                         'candidate initialization or updates are never silently clipped')


def candidate_record(result, *, kind, score, init_supports, fit_seconds=0.0, cache_hit=False):
    sizes = tuple(len(support) for support in (result.S1, result.S2, result.S3))
    return {
        'kind': kind, 'sizes': sizes, 'q': int(np.prod(sizes)),
        'supports': [support.copy() for support in (result.S1, result.S2, result.S3)],
        'B': result.B.copy(),
        'rss': float(result.rss) if hasattr(result, 'rss') else float(result.best_rss),
        'loss': float(result.loss), 'score': float(score),
        'loss_history': list(result.loss_history),
        'visited_loss_history': list(getattr(result, 'visited_loss_history', [])),
        'best_iteration': getattr(result, 'best_iteration', None),
        'stop_reason': getattr(result, 'stop_reason', 'adaptive_complete'),
        'n_cycles': int(result.n_cycles), 'fit_seconds': float(fit_seconds),
        'cache_hit': bool(cache_hit),
        'init_supports': [support.copy() for support in init_supports],
        'accepted_support_history': list(getattr(result, 'accepted_support_history', [])),
    }


def refine_msic_global_candidates(X, y, result, *, s1_max, s2_max, s3_max,
                                  max_cycles, m_max, k_max, tensor_criterion_fn,
                                  one_se_scale, cap_policy='legacy', grid_policy='legacy',
                                  candidate_budget=2000, fixed_candidate_cache=None):
    # Local imports keep core dependencies explicit and avoid the bcd module cycle.
    from tensor_abess.tensor.bcd import TensorBcdAdaptiveResult, tensor_bcd, _marginal_supports_for_size
    from tensor_abess.tensor.bcd import _msic_one_se_tolerance
    n, p1, p2, p3 = X.shape
    caps = (s1_max, s2_max, s3_max)
    max_params = max(1, int(np.floor(0.7 * n)))
    current_score = tensor_criterion_fn(result.rss, result.s1, result.s2, result.s3, n, p1, p2, p3)
    candidates = [(result, current_score)]
    metadata = dict(result.search_metadata)
    init_supports = metadata.pop('adaptive_init_supports', (result.S1, result.S2, result.S3))
    records = [candidate_record(result, kind='adaptive-stage', score=current_score,
                init_supports=init_supports, fit_seconds=metadata.pop('adaptive_fit_seconds', 0.0))]
    initial_grid = candidate_size_triples(*caps, max_params, grid_policy=grid_policy,
                                         candidate_budget=candidate_budget)
    legal = legal_size_triples(caps, max_params)
    evaluated = []
    train_key = training_fingerprint(X, y) if fixed_candidate_cache is not None else None

    def evaluate(sizes):
        S_init = _marginal_supports_for_size(X, y, sizes)
        validate_candidate_state(S_init, caps, max_params, X.shape[1:], stage='global-fixed')
        key = (train_key, sizes, max_cycles, m_max, k_max, 'v5.2-fixed-v5.1')
        cached = None if fixed_candidate_cache is None else fixed_candidate_cache.get(key)
        if cached is None:
            start = perf_counter()
            fixed = tensor_bcd(X, y, *sizes, max_cycles=max_cycles, m_max=m_max,
                               k_max=k_max, S1_init=S_init[0], S2_init=S_init[1], S3_init=S_init[2])
            fit_seconds = perf_counter() - start
            if fixed_candidate_cache is not None:
                fixed_candidate_cache[key] = (fixed, fit_seconds)
        else:
            fixed, fit_seconds = cached
        rss = 2.0 * n * fixed.loss
        score = tensor_criterion_fn(rss, *sizes, n, p1, p2, p3)
        candidate = TensorBcdAdaptiveResult(
            B=fixed.B, S1=fixed.S1, S2=fixed.S2, S3=fixed.S3, s1=sizes[0], s2=sizes[1], s3=sizes[2],
            loss=fixed.loss, rss=rss, sic=score, loss_history=fixed.loss_history,
            sic_history=[*result.sic_history, score], n_cycles=result.n_cycles + fixed.n_cycles)
        candidates.append((candidate, score))
        record = candidate_record(fixed, kind='global-fixed', score=score,
                                  init_supports=S_init, fit_seconds=fit_seconds, cache_hit=cached is not None)
        record['candidate_key'] = sizes
        records.append(record)
        evaluated.append(sizes)

    for sizes in initial_grid:
        evaluate(sizes)
    refinement_grid = []
    if grid_policy == 'coarse_to_fine':
        top = sorted(candidates, key=lambda item: (item[1], item[0].s1*item[0].s2*item[0].s3,
                                                   item[0].s1+item[0].s2+item[0].s3))[:3]
        neighbors = {sizes for candidate, _ in top
                     for sizes in product(*(range(max(1, size-1), min(cap, size+1)+1)
                         for size, cap in zip((candidate.s1, candidate.s2, candidate.s3), caps)))
                     if int(np.prod(sizes)) <= max_params}
        refinement_grid = sorted(neighbors - set(evaluated))
        check_budget(len(evaluated)+len(refinement_grid), candidate_budget, caps, max_params, grid_policy)
        for sizes in refinement_grid:
            evaluate(sizes)
    best, min_score = min(candidates, key=lambda item: item[1])
    q_best = best.s1 * best.s2 * best.s3
    tolerance = _msic_one_se_tolerance(n, q_best, one_se_scale)
    eligible = [(candidate, score) for candidate, score in candidates if score <= min_score + tolerance]
    selected, selected_score = min(eligible, key=lambda item: (
        item[0].s1*item[0].s2*item[0].s3, item[1], item[0].s1+item[0].s2+item[0].s3))
    evaluated_set = set(evaluated)
    excluded = [sizes for sizes in legal if sizes not in evaluated_set]
    metadata.update({
        'cap_policy': cap_policy, 'grid_policy': grid_policy, 'candidate_budget': candidate_budget,
        'max_params': max_params, 'caps': caps, 'size_grid': evaluated,
        'initial_size_grid': initial_grid, 'refinement_size_grid': refinement_grid,
        'legal_shapes': legal, 'excluded_legal_shapes': excluded, 'grid_complete': not excluded,
        'training_fingerprint': train_key,
        'selection': {'minimum_score': min_score, 'minimum_q': q_best, 'tolerance': tolerance,
                      'selected_score': selected_score, 'selected_sizes': (selected.s1, selected.s2, selected.s3),
                      'eligible_count': len(eligible), 'one_se_scale': one_se_scale},
    })
    return replace(selected, B=selected.B.copy(), S1=selected.S1.copy(),
                   S2=selected.S2.copy(), S3=selected.S3.copy(), sic=selected_score,
                   candidate_records=records, search_metadata=metadata)
