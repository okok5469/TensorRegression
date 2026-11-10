"""Independently replay saved-model predictions on the predeclared twenty inputs.

No estimator is fitted and no scientific protocol is changed. This is a numeric
cross-platform check, not a claim that original unsaved test arrays were archived.
"""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from numerical_simulation.data_generation import iter_test_batches
from numerical_simulation.known_s_cluster.run import array_digest, load_plan
from tensor_abess.tensor.ops import tensor_inner
from .return_report import digest, verified_return_evidence
from .workflow import paths, pilot_ids, restore, unique_id


def replay(root: Path):
    evidence = verified_return_evidence(root, root / 'doc/taskcard18/cluster_review/RETURN_VALIDATION.json')
    historical, _ = paths()
    plan = load_plan(historical)
    rows, stream_matches = [], []
    prefix = 'numerical_simulation/outputs_taskcard18_closeout/'
    with zipfile.ZipFile(root / evidence['archive_repo_relative_path']) as archive:
        for task_id in pilot_ids(plan):
            dataset, record, correspondence = restore(task_id)
            task, uid = record['task'], unique_id(record)
            models, expected = {}, {}
            with np.load(historical / 'tasks' / record['arrays_file'], allow_pickle=False) as saved:
                for method in ('adaptive', 'oracle'):
                    models[method] = saved['B_' + method].copy()
                    expected[method] = next(r for r in record['results'] if r['method'] == method)['metrics']
            for method in ('lasso', 'glasso'):
                with np.load(io.BytesIO(archive.read(prefix + f'fits/{uid}_{method}.npz')), allow_pickle=False) as saved:
                    models[method] = saved['B'].copy()
                evaluation = json.loads(archive.read(prefix + f'evaluations/{uid}_{method}.json'))
                expected[method] = evaluation['metrics']
            sums = {method: np.zeros(2) for method in models}
            total, sum_y, sum_y2 = 0, 0., 0.
            for index, (X, y, signal) in enumerate(iter_test_batches(dataset, seed=task['test_seed'])):
                stream_matches.append({name: array_digest(array) == evaluation['test_stream_hashes'][index][name]
                                       for name, array in (('X', X), ('y', y), ('signal', signal))})
                for method, B in models.items():
                    pred = tensor_inner(X, B)
                    residual, signal_gap = y - pred, signal - pred
                    sums[method] += [float(residual @ residual), float(signal_gap @ signal_gap)]
                total += len(y)
                sum_y += float(y.sum())
                sum_y2 += float(y @ y)
            if total != 500:
                raise ValueError('Prediction replay changed the frozen test denominator')
            sst = sum_y2 - sum_y * sum_y / total
            for method, (sse, signal_sse) in sums.items():
                observed = {'pmse': float(sse / total), 'signal_mse': float(signal_sse / total),
                            'r2_test': float(1 - sse / sst)}
                differences = {name: abs(observed[name] - expected[method][name]) for name in observed}
                np.testing.assert_allclose(list(observed.values()), [expected[method][name] for name in observed],
                                           rtol=1e-10, atol=1e-10)
                rows.append({'task_id': task_id, 'unique_input_id': uid, 'method': method,
                             'test_n': total, 'recomputed': observed, 'absolute_differences': differences,
                             'train_identity_level': correspondence['identity_level']})
            print('PREDICTION_REPLAY_PASS', task_id, flush=True)
    result = {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'archive_sha256': evidence['archive_sha256'], 'fixed_inputs': 20, 'saved_models': len(rows),
              'metrics_checked': 3 * len(rows),
              'maximum_absolute_difference': max(value for row in rows for value in row['absolute_differences'].values()),
              'regenerated_stream_hash_match_counts': {name: sum(r[name] for r in stream_matches) for name in ('X', 'y', 'signal')},
              'batches_checked': len(stream_matches), 'rows': rows,
              'scope': 'Predeclared fixed20 inputs only; frozen generator/test seeds/batch100 and four saved estimators, no fitting. Local Windows replay checks numeric correspondence at rtol=atol=1e-10. Full1200 evaluation records are separately identity-audited; original test arrays were not archived.'}
    result['record_sha256'] = digest(result)
    destination = root / 'doc/taskcard18/cluster_review/PREDICTION_REPLAY.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('PREDICTION_REPLAY_COMPLETE', len(rows), result['maximum_absolute_difference'], flush=True)


if __name__ == '__main__':
    replay(Path(__file__).resolve().parents[2])
