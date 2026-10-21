"""Task11 saved-pool/cache regression; no native search is called."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from research.taskcard12 import selection
from research.taskcard10.experiment import file_sha, read_json, write_json
from research.taskcard11.verify import verify_case
from scripts.taskcard7.lemon import array_sha, require


def audit_case(repo, run, folder, output, replay):
    selection.configure_core()
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from research.taskcard10.cv import native_options
    started = time.perf_counter()
    verified = verify_case(repo, run, folder, replay=False)
    record = read_json(folder / 'complete.json')
    request = record['request']
    witness = next(r for r in replay['checks'] if r['cohort'] == request['cohort'] and r['task_id'] == request['task_id'])
    require(witness['passed'] and witness['replay']['status'] == 'PASS' and
            witness['complete_sha256'] == file_sha(folder / 'complete.json'), 'Historical full replay identity differs')
    values = dict(request['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    data = generate_simulation_dataset(TensorSimulationConfig(**values), seed=request['seed'])
    require(array_sha(data.X_train) == record['input_fingerprints']['X_train'] and
            array_sha(data.y_train) == record['input_fingerprints']['y_train'], 'Historical training identity differs')
    summary = read_json(folder / 'summary.json')
    trace = read_json(folder / 'trace.json')
    with np.load(folder / 'candidate_coefficients.npz', allow_pickle=False) as saved:
        coefficients = saved['B'].copy()
    options = native_options(request['config'])
    pool = selection.pool_from_saved(data.X_train, data.y_train, trace, coefficients, **options)
    selection.save_pool(output, pool)
    loaded = selection.load_pool(output, X=data.X_train, y=data.y_train, **options)
    models = {}
    with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
        for rule, arm in [('original', 'A0'), ('fs', 'A1')]:
            model = selection.select_from_pool(loaded, selection_rule=rule)
            require(model['selection'] == summary['choices'][arm], 'Historical selection differs: ' + arm)
            np.testing.assert_array_equal(model['B'], arrays[arm + '_B'])
            require(model['B_sha256'] == record['metrics'][arm]['B_sha256'], 'Historical prediction coefficient differs')
            for key in ('support', 'sizes', 'q', 'rss', 'B_sha256'):
                require(model[key] == summary['arms'][arm][key], 'Historical selected model differs: ' + key)
            models[rule] = {'selected_index': model['full_candidate_index'], 'B_sha256': model['B_sha256'],
                            'q': model['q'], 'score': model['score']}
    return {'passed': True, 'cohort': request['cohort'], 'task_id': request['task_id'],
            'candidate_count': len(coefficients), 'native_search_calls': 0,
            'new_scientific_repetitions': 0, 'training_identity_exact': True,
            'cache_roundtrip_exact': pool['digest'] == loaded['digest'],
            'saved_prediction_identity_exact': True, 'models': models,
            'complete_sha256': file_sha(folder / 'complete.json'),
            'artifact_hashes_checked': verified['artifact_hashes_checked'],
            'prediction_metrics': verified['recomputed_prediction_metrics'],
            'historical_replay_maximum_prediction_error': witness['replay']['maximum_prediction_absolute_error'],
            'maximum_OLS_coefficient_absolute_error': pool['metadata']['audit']['maximum_OLS_coefficient_absolute_error'],
            'maximum_RSS_absolute_error': pool['metadata']['audit']['maximum_RSS_absolute_error'],
            'cache_seal_sha256': file_sha(Path(output) / 'seal.json'),
            'elapsed_seconds': time.perf_counter() - started}


def run(output):
    repo = Path(__file__).resolve().parents[2]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, checks, references = time.perf_counter(), [], {}
    for stage, cohort, count in [('pilot', 'pilot', 6), ('formal', 'validation', 96)]:
        directory = repo / 'doc/taskcard11/results' / stage
        replay_path = repo / 'doc/taskcard11/verification' / (stage + '_full_replay.json')
        replay = read_json(replay_path)
        require(replay['status'] == 'PASS_FULL_REPLAY' and replay['completed'] == count, 'Historical full replay coverage differs')
        references[replay_path.relative_to(repo).as_posix()] = file_sha(replay_path)
        folders = sorted((directory / cohort).glob('task_*/complete.json'))
        require(len(folders) == count, 'Historical cohort incomplete')
        for path in folders:
            require(time.perf_counter() - started < 600, 'Historical regression ten-minute budget reached')
            check = audit_case(repo, directory, path.parent, output / stage / path.parent.name, replay)
            checks.append(check)
            write_json(output / stage / (path.parent.name + '.json'), check)
        print(stage, count, 'PASS', round(time.perf_counter() - started, 2), flush=True)
    result = {'passed': True, 'cases': len(checks), 'candidate_count': sum(r['candidate_count'] for r in checks),
              'native_search_calls': 0, 'new_scientific_repetitions': 0, 'checks': checks,
              'source_environment_identity': selection.identity(), 'historical_full_replay_sha256': references,
              'tool_sha256': file_sha(__file__), 'elapsed_seconds': time.perf_counter() - started,
              'prediction_scope': 'Exact selected B and sealed test identity; saved PMSE recalculated. Prior full test replay linked by complete hash. No new test stream.'}
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print('PASS', result['cases'], result['candidate_count'], flush=True)


if __name__ == '__main__':
    main()
