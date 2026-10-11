"""Read-only selected-model OLS diagnostics for complete frozen cohorts.

No candidate fitting, model selection, coefficient replacement or numerical
rescue is performed. Rank and conditioning concern selected models only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rectangular_design(X, supports):
    X = np.asarray(X, dtype=float)
    require(X.ndim == 4 and len(supports) == 3, 'Expected tensor samples and three supports')
    indices = [np.asarray(support, dtype=int) for support in supports]
    require(all(len(S) > 0 and len(np.unique(S)) == len(S)
                and np.all((S >= 0) & (S < dimension))
                for S, dimension in zip(indices, X.shape[1:])), 'Invalid rectangular supports')
    return X[:, indices[0]][:, :, indices[1]][:, :, :, indices[2]].reshape(len(X), -1)


def audit_selected_model(X, y, B, supports, *, expected_rss, expected_loss=None):
    started = perf_counter()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    require(X.ndim == 4 and len(y) == len(X) and B.shape == X.shape[1:], 'Misaligned audit arrays')
    require(np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(B).all()
            and np.isfinite(expected_rss), 'Nonfinite audit input')
    indices = [np.asarray(S, dtype=int) for S in supports]
    D = rectangular_design(X, indices)
    n, q = D.shape
    require(q <= int(np.floor(.7*n)), 'Selected candidate exceeds q capacity')
    beta = B[np.ix_(*indices)].ravel()
    mask = np.zeros(B.shape, dtype=bool); mask[np.ix_(*indices)] = True
    require(np.count_nonzero(B[~mask]) == 0, 'Nonzero coefficients outside reported rectangle')
    rectangular_prediction = D @ beta
    tensor_prediction = np.einsum('nijk,ijk->n', X, B)
    rss = float(np.sum((y-rectangular_prediction)**2))
    tensor_rss = float(np.sum((y-tensor_prediction)**2))
    rss_scale = max(1.0, abs(float(expected_rss)))
    tolerance = 1e-9
    require(abs(rss-float(expected_rss)) <= tolerance*rss_scale,
            f'Selected coefficient RSS differs from stored RSS: actual={rss}, expected={expected_rss}')
    if expected_loss is not None:
        require(np.isfinite(expected_loss) and abs(2*n*expected_loss-float(expected_rss)) <= tolerance*rss_scale,
                'Stored RSS/loss inconsistent')
    prediction_gap = float(np.max(np.abs(rectangular_prediction-tensor_prediction)))
    prediction_scale = max(1.0, float(np.max(np.abs(tensor_prediction))))
    require(prediction_gap <= tolerance*prediction_scale
            and abs(rss-tensor_rss) <= tolerance*rss_scale, 'Tensor and rectangle predictions differ')
    singular = np.linalg.svd(D, compute_uv=False)
    rank_tolerance = float(singular[0]*max(D.shape)*np.finfo(float).eps)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    rank_deficient = rank < q
    condition = None if rank_deficient else float(singular[0]/singular[-1])
    independent_beta, _, independent_rank, _ = np.linalg.lstsq(D, y, rcond=None)
    independent_rss = float(np.sum((y-D @ independent_beta)**2))
    require(int(independent_rank) == rank, 'Independent OLS rank differs from SVD numerical rank')
    require(abs(rss-independent_rss) <= tolerance*rss_scale,
            f'Selected RSS differs from independent OLS RSS: selected={rss}, OLS={independent_rss}')
    normal = float(np.linalg.norm(D.T @ (y-rectangular_prediction)))
    normal_relative = normal/max(float(singular[0]*np.linalg.norm(y)), np.finfo(float).tiny)
    require(normal_relative <= 1e-8, f'Selected fit violates normal equations: relative={normal_relative}')
    beta_gap = float(np.linalg.norm(beta-independent_beta))
    return {
        'n_train': n, 'q_hat': q, 'sizes': list(map(len, indices)),
        'rank': rank, 'rank_deficient': rank_deficient,
        'rank_tolerance': rank_tolerance, 'singular_max': float(singular[0]),
        'singular_min': float(singular[-1]), 'singular_values': singular.tolist(),
        'condition_number': condition,
        'rss_stored': float(expected_rss), 'rss_rectangular': rss, 'rss_tensor': tensor_rss,
        'rss_independent_ols': independent_rss,
        'rss_relative_difference': abs(rss-float(expected_rss))/rss_scale,
        'ols_rss_relative_difference': abs(rss-independent_rss)/rss_scale,
        'prediction_max_abs_difference': prediction_gap,
        'normal_equation_residual': normal, 'normal_equation_relative_residual': normal_relative,
        'beta_l2_difference_from_independent_ols': beta_gap,
        'beta_relative_difference_from_independent_ols': beta_gap/max(1.0, float(np.linalg.norm(independent_beta))),
        'coefficients_nonunique': rank_deficient,
        'checks_passed': True, 'audit_seconds': perf_counter()-started,
    }


def audit_cohort(output_dir):
    from numerical_simulation.candidate_search_study.run import read_json, validate_result
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.version_validation.worker import array_fingerprint

    output = Path(output_dir).resolve()
    require(not (output/'STOPPED_FOR_AUDIT.json').exists(), 'Stopped cohort is excluded from numerical auditing')
    integrity = read_json(output/'integrity_report.json')
    plan = read_json(output/'plan.json')
    require(integrity['status'] == 'complete' and integrity['integrity_checks_passed']
            and integrity['plan_id'] == plan['plan_id']
            and integrity['completed_tasks'] == len(plan['tasks']), 'Cohort is not complete and validated')
    targets = [output/'selected_model_numerics.csv', output/'numerics_diagnostics.json']
    require(not any(path.exists() for path in targets), 'Immutable numerical diagnostics already exist')
    rows = []
    started = perf_counter()
    for task in plan['tasks']:
        report = validate_result(output, plan, task)
        config_values = dict(report['config'])
        for key in ('dims', 'sparsity', 'rho'):
            config_values[key] = tuple(config_values[key])
        dataset = generate_simulation_dataset(TensorSimulationConfig(**config_values), seed=int(report['seed']))
        for key, value in (('X_train', dataset.X_train), ('y_train', dataset.y_train)):
            require(array_fingerprint(value) == report['fingerprints'][key], f'Regenerated training fingerprint differs: {key}')
        cache_path = output/'jobs'/f"task_{task['task_id']:03d}.npz"
        with np.load(cache_path, allow_pickle=False) as arrays:
            for selected in report['rows']:
                group = selected['group']; index = int(selected['selection']['selected_index'])
                candidate = report['groups'][group]['candidates'][index]
                prefix = f'{group}_candidate_{index}_'
                B = arrays[prefix+'B']; supports = [arrays[prefix+f'S{mode}'] for mode in (1, 2, 3)]
                require(array_fingerprint(B) == selected['B_fingerprint'] == candidate['B_fingerprint'], 'Selected B fingerprint mismatch')
                require([S.tolist() for S in supports] == selected['supports'] == candidate['supports'], 'Selected support cache mismatch')
                diagnostics = audit_selected_model(dataset.X_train, dataset.y_train, B, supports,
                    expected_rss=selected['rss_train'], expected_loss=selected['loss_train'])
                rows.append({'task_id': task['task_id'], 'seed': report['seed'], 'snr': report['config']['snr'],
                    'group': group, 'cap_policy': report['groups'][group]['cap_policy'],
                    'grid_policy': report['groups'][group]['grid_policy'], 'candidate_index': index,
                    'candidate_kind': candidate['kind'], **diagnostics})
        print(f"Numerics: {len(rows)} selected fits audited after task {task['task_id']}", flush=True)
    summary = {'status': 'complete', 'checks_passed': True, 'plan_id': plan['plan_id'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_model_count': len(rows),
        'task_count': len(plan['tasks']), 'runtime_sec': perf_counter()-started,
        'scope': 'Selected original-MSIC model per dataset/group only; not all candidate matrices. Training tensors regenerated with exact fingerprints. No coefficients or supports changed.',
        'tolerances': {'relative_scaled_rss_and_prediction': 1e-9, 'normal_equation_relative_residual': 1e-8},
        'rank_deficient_count': sum(row['rank_deficient'] for row in rows), 'groups': {}}
    for group in plan['groups']:
        group_rows = [row for row in rows if row['group'] == group]
        finite_conditions = [row['condition_number'] for row in group_rows if row['condition_number'] is not None]
        summary['groups'][group] = {
            'selected_model_count': len(group_rows), 'rank_deficient_count': sum(row['rank_deficient'] for row in group_rows),
            'condition_number_min': min(finite_conditions) if finite_conditions else None,
            'condition_number_median': float(np.median(finite_conditions)) if finite_conditions else None,
            'condition_number_max': max(finite_conditions) if finite_conditions else None,
            'max_q': max(row['q_hat'] for row in group_rows),
            'max_rss_relative_difference': max(row['rss_relative_difference'] for row in group_rows),
            'max_ols_rss_relative_difference': max(row['ols_rss_relative_difference'] for row in group_rows),
            'max_normal_equation_relative_residual': max(row['normal_equation_relative_residual'] for row in group_rows),
            'total_audit_seconds': sum(row['audit_seconds'] for row in group_rows),
        }
    with targets[0].open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, list) else value
                             for key, value in row.items()})
    with targets[1].open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    summary = audit_cohort(args.output_dir)
    print(f"Selected numerical checks passed: {summary['selected_model_count']} models", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
