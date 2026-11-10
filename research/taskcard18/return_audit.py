"""Read a returned Linux archive without fitting, extracting or replacing local state.

The execution identity is the sealed recorded Linux identity. The existing local
fit-cache guards remain unchanged. Derived CSVs are rebuilt from saved arrays and
checked against every returned field before any result is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np

from .accept import validate_arrays, validate_metadata
from .analysis import paired_statistics, save_csv, summarize, support_metrics
from .identity import build_input_map, verify_hashes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_seal(value):
    if value.get('record_sha256') != digest({k: v for k, v in value.items() if k != 'record_sha256'}):
        raise ValueError('Immutable record seal mismatch')


def archive_members(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ZIP member')
    for item in archive.infolist():
        raw_name = item.orig_filename
        path = PurePosixPath(raw_name)
        if (path.is_absolute() or '..' in path.parts or '\\' in raw_name
                or ':' in raw_name or stat.S_ISLNK(item.external_attr >> 16)):
            raise ValueError('Unsafe ZIP member: ' + item.filename)
    return set(names)


def verify_evaluation(evaluation, metadata, n_test, batch_size=100):
    for name in ('identity', 'unique_input_id', 'method_id'):
        if evaluation.get(name) != metadata[name]:
            raise ValueError('Evaluation identity mismatch: ' + name)
    if evaluation.get('fit_npz_sha256') != metadata['npz_sha256']:
        raise ValueError('Evaluation coefficient identity mismatch')
    metrics = evaluation['metrics']
    if (set(metrics) != {'pmse', 'signal_mse', 'r2_test'}
            or not np.isfinite(list(metrics.values())).all()
            or min(metrics['pmse'], metrics['signal_mse']) < 0):
        raise ValueError('Nonfinite or invalid evaluation metric')
    batches = evaluation.get('test_stream_hashes', [])
    if len(batches) != (n_test + batch_size - 1) // batch_size:
        raise ValueError('Missing evaluation test batch')
    for batch in batches:
        if set(batch) != {'X', 'y', 'signal'} or any(
                not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
                for value in batch.values()):
            raise ValueError('Invalid test-stream fingerprint')


def verify_fit_data(diagnostics, n_train, response_hash):
    from .baselines import _array_hash
    all_indices = np.arange(n_train, dtype=np.int64)
    folds = np.array_split(all_indices, 5)
    train_hashes = [_array_hash(np.setdiff1d(all_indices, validation, assume_unique=True)) for validation in folds]
    full_hash = _array_hash(all_indices)
    for diagnostic in diagnostics:
        final = diagnostic['fit_kind'] == 'final'
        fold = diagnostic['fold_index']
        if not final and (not isinstance(fold, int) or not 0 <= fold < 5):
            raise ValueError('Invalid CV fold index')
        expected_hash = full_hash if final else train_hashes[fold]
        expected_n = n_train if final else n_train - len(folds[fold])
        if diagnostic['n_fit'] != expected_n or diagnostic['train_indices_hash'] != expected_hash:
            raise ValueError('Recorded fitting indices do not match the frozen contiguous folds')
        if final and (fold is not None or diagnostic['train_response_hash'] != response_hash or diagnostic['warm_start_from'] is not None):
            raise ValueError('Final fit response identity/warm-state conflict')


def compare_rows(actual, expected, key_fields):
    key = lambda row: tuple(str(row[name]) for name in key_fields)
    lookup = {key(row): row for row in actual}
    if len(actual) != len(expected) or len(lookup) != len(actual) or set(lookup) != {key(row) for row in expected}:
        raise ValueError('CSV duplicate/missing identity')
    maximum, count = 0., 0
    for row in expected:
        saved = lookup[key(row)]
        if set(saved) != set(row):
            raise ValueError('CSV field schema conflict')
        for name, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                number = float(saved[name]); count += 1
                if not np.isclose(number, value, rtol=1e-11, atol=1e-10, equal_nan=True):
                    raise ValueError('CSV value conflict: ' + str((key(row), name)))
                if np.isfinite(value):
                    maximum = max(maximum, abs(number - value))
            elif str(saved[name]) != str(value):
                raise ValueError('CSV metadata conflict: ' + str((key(row), name)))
    return {'rows': len(actual), 'numeric_fields_checked': count,
            'maximum_absolute_difference': maximum, 'passed': True}


def _json_bytes(data):
    return json.loads(data.decode('utf-8-sig'))


def _csv_bytes(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def _metrics(B, truth, supports):
    result = support_metrics(B, truth, supports)
    for threshold, suffix in ((1e-10, '1e10'), (1e-6, '1e6')):
        sensitivity = support_metrics(B, truth, supports, threshold)
        result.update({name + '_' + suffix: sensitivity[name] for name in
                       ('element_nonzero', 'element_recall', 'element_fdr', 'element_f1')})
    return result


def _correspondence(value, record, uid):
    if (value.get('valid') is not True or value.get('unique_input_id') != uid
            or value.get('old_fingerprints') != record['input_fingerprints']):
        raise ValueError('Input correspondence conflicts with frozen historical input')
    old, new = value['old_fingerprints'], value['new_fingerprints']
    matches = {name: old[name] == new[name] for name in old}
    level = 'BYTE_IDENTICAL' if all(matches.values()) else 'NUMERICAL_CORRESPONDENCE_NOT_BYTE_IDENTITY'
    if value['exact_hash_match'] != matches or value['identity_level'] != level:
        raise ValueError('Input correspondence identity level conflict')
    differences = [value['oracle_training_rss_abs_difference'], *value['oracle_prediction_abs_differences'].values()]
    if not np.isfinite(differences).all() or min(differences) < 0:
        raise ValueError('Invalid input-correspondence numerical diagnostics')


def audit(root: Path, archive_path: Path):
    """Return validated facts and rebuilt tables; never write or reconstruct data."""
    from numerical_simulation.known_s_cluster.run import load_plan, read_task

    root, archive_path = Path(root).resolve(), Path(archive_path).resolve()
    output_prefix = 'numerical_simulation/outputs_taskcard18_closeout/'
    doc_prefix = 'doc/taskcard18/'
    historical = root / 'numerical_simulation/outputs_v52_search_final_full'
    plan = load_plan(historical)
    release = _json_bytes((root / 'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json').read_bytes())
    if len(release['source_sha256']) != 32 or len(plan['source_fingerprints']) != 38:
        raise ValueError('Frozen source inventory count conflict')
    verify_hashes(root, release['source_sha256']); verify_hashes(root, plan['source_fingerprints'])
    if {p.relative_to(root).as_posix() for p in (root / 'src/tensor_abess').rglob('*.py')} != set(release['source_sha256']):
        raise ValueError('Frozen production source inventory conflict')
    original_path = root / 'doc/taskcard18/cluster_delivery/bootstrap_cache_fix_20261028/taskcard18_cluster_20261028.zip'
    records = [read_task(historical, plan, task['task_id']) for task in plan['tasks']]
    if len(records) != 700:
        raise ValueError('Expected 700 frozen historical records')
    mapping_rows = build_input_map(records)
    mapping = {row['task_id']: row for row in mapping_rows}
    by_id = {record['task']['task_id']: record for record in records}
    uids = {row['unique_input_id'] for row in mapping_rows}
    if len(uids) != 600 or len({row['config_id'] for row in mapping_rows}) != 10:
        raise ValueError('Expected 600 complete inputs and 10 configurations')
    method_data, runtime, certificates, boundary, levels = {}, defaultdict(list), Counter(), Counter(), Counter()
    diagnostics_summary = defaultdict(Counter)
    extrema = defaultdict(float)
    with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(original_path) as original:
        names = archive_members(archive)
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise ValueError('ZIP CRC conflict: ' + bad_crc)
        manifest_bytes = archive.read('PAYLOAD_MANIFEST.json')
        manifest = _json_bytes(manifest_bytes)
        original_name = next(name for name in original.namelist() if name.endswith('/PAYLOAD_MANIFEST.json'))
        if manifest_bytes != original.read(original_name):
            raise ValueError('Returned payload manifest differs from original delivered package')
        payload = manifest['files_sha256']
        checked_payload = 0
        for name, expected in payload.items():
            if name in names:
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError('Returned payload source/configuration conflict: ' + name)
                checked_payload += 1
        immutable = {*release['source_sha256'], *plan['source_fingerprints'],
                     doc_prefix + 'CLOSEOUT_PROTOCOL.json', doc_prefix + 'input_identity_map.csv',
                     *('research/taskcard18/' + name for name in ('solvers.py', 'baselines.py', 'workflow.py', 'analysis.py', 'accept.py', 'identity.py'))}
        immutable.update('numerical_simulation/outputs_v52_search_final_full/tasks/' + f'task_{i:06d}' + suffix
                         for i in range(700) for suffix in ('.json', '.npz'))
        for name in immutable:
            if file_sha(root / name) != payload[name]:
                raise ValueError('Current immutable input/source differs from original payload: ' + name)
        compare_rows(_csv_bytes(archive.read(doc_prefix + 'input_identity_map.csv')), mapping_rows, ('task_id',))
        run_bytes = archive.read(output_prefix + 'RUN_IDENTITY.json'); run = _json_bytes(run_bytes)
        verify_seal(run)
        identity = run['identity']
        if (run['plan_id'] != plan['plan_id'] or run['plan_file_sha256'] != file_sha(historical / 'plan.json')
                or run['input_map_sha256'] != file_sha(root / doc_prefix / 'input_identity_map.csv')
                or identity['protocol_sha256'] != file_sha(root / doc_prefix / 'CLOSEOUT_PROTOCOL.json')):
            raise ValueError('Recorded run conflicts with frozen plan/protocol/input map')
        verify_hashes(root, {'research/taskcard18/' + name: expected for name, expected in identity['adapter_hashes'].items()})
        expected_environment = {'python': '3.9.25', 'numpy': '2.0.2', 'scipy': '1.13.1', 'sklearn': '1.5.2', 'platform': 'Linux', 'machine': 'x86_64'}
        if any(identity.get(key) != expected for key, expected in expected_environment.items()) or set(identity['threads'].values()) != {'1'}:
            raise ValueError('Recorded Linux environment differs from delivered contract')
        expected_run = [{'task_id': task['task_id'], 'unique_input_id': mapping[task['task_id']]['unique_input_id'],
                         'config_sha256': digest(task['config']), 'seed': task['seed'], 'test_seed': task['test_seed']} for task in plan['tasks']]
        if run['tasks'] != expected_run:
            raise ValueError('Recorded 700-task run inventory conflict')
        prepared = _json_bytes(archive.read(output_prefix + 'CLUSTER_PREPARED.json'))
        bindings = {'run_sha256': run_bytes, 'input_audit_sha256': archive.read(output_prefix + 'pilot_input_audit_v2.json'),
                    'focused_tests_sha256': archive.read(output_prefix + prepared['focused_tests_file']), 'payload_manifest_sha256': manifest_bytes}
        if prepared['identity'] != identity or any(prepared[key] != hashlib.sha256(value).hexdigest() for key, value in bindings.items()):
            raise ValueError('Prepared identity/evidence binding conflict')
        protocol = _json_bytes(archive.read(doc_prefix + 'CLOSEOUT_PROTOCOL.json'))
        pilot_tasks = {task['task_id'] for task in plan['tasks'] if task['scenario_id'] in protocol['pilot_scenarios'] and task['replicate'] in protocol['pilot_replicates']}
        pilot = _json_bytes(archive.read(output_prefix + 'PILOT_GATE.json'))
        if (pilot['status'] != 'PASS' or pilot['identity'] != identity or pilot['fits'] != 40
                or {(r['task_id'], r['method']) for r in pilot['records']} != {(i, m) for i in pilot_tasks for m in ('lasso', 'glasso')}):
            raise ValueError('Pilot gate is incomplete or identity-conflicted')
        fits = {name for name in names if name.startswith(output_prefix + 'fits/')}
        evaluations = {name for name in names if name.startswith(output_prefix + 'evaluations/')}
        failures = [name for name in names if name.startswith(output_prefix + 'failures/') and not name.endswith('/')]
        expected_stems = {output_prefix + 'fits/' + uid + '_' + method for uid in uids for method in ('lasso', 'glasso')}
        if (fits != {stem + suffix for stem in expected_stems for suffix in ('.json', '.npz')}
                or evaluations != {stem.replace('/fits/', '/evaluations/') + '.json' for stem in expected_stems} or failures):
            raise ValueError('Missing, duplicate, orphan or failed required fit/evaluation')
        for stem in sorted(expected_stems):
            metadata = _json_bytes(archive.read(stem + '.json')); verify_seal(metadata)
            method = stem.rsplit('_', 1)[-1]
            record = by_id[metadata['task_id']]; task = record['task']; uid = mapping[task['task_id']]['unique_input_id']
            if stem != output_prefix + 'fits/' + uid + '_' + method or metadata['identity'] != identity:
                raise ValueError('Fit filename/input/execution conflict')
            validate_metadata(metadata, method)
            _correspondence(metadata['input_correspondence'], record, uid)
            verify_fit_data(metadata['diagnostics'], task['config']['n_train'], metadata['input_correspondence']['new_fingerprints']['y_train'])
            if metadata['test_evaluated'] is not False or not np.isfinite(metadata['runtime_sec']) or metadata['runtime_sec'] < 0:
                raise ValueError('Invalid fitting metadata/timer')
            data = archive.read(stem + '.npz')
            if hashlib.sha256(data).hexdigest() != metadata['npz_sha256']:
                raise ValueError('Coefficient NPZ hash conflict')
            with np.load(io.BytesIO(data), allow_pickle=False) as saved:
                B, supports = saved['B'].copy(), [saved[f'S{i}'].copy() for i in (1, 2, 3)]
            validate_arrays(B, supports, task['config']['dims'])
            for diagnostic in metadata['diagnostics']:
                n = task['config']['n_train']
                expected_n = n if diagnostic['fit_kind'] == 'final' else n - (n // 5 + (diagnostic['fold_index'] < n % 5))
                if diagnostic['n_fit'] != expected_n:
                    raise ValueError('Certificate fitting-subset sample count conflict')
                fields = ('objective', 'dual_objective', 'gap', 'gap_normalized', 'lambda', 'penalty_norm', 'dual_norm', 'runtime_sec')
                if not np.isfinite([diagnostic[key] for key in fields]).all():
                    raise ValueError('Nonfinite convex-fit diagnostic')
                extrema['max_kkt_ratio'] = max(extrema['max_kkt_ratio'], diagnostic['kkt_ratio'] or 0.)
                extrema['max_normalized_gap'] = max(extrema['max_normalized_gap'], diagnostic['gap_normalized'] or 0.)
                statistics = diagnostics_summary[method]
                statistics['total_iterations'] += diagnostic['iterations']
                statistics['at_limit'] += int(diagnostic['at_limit'])
                statistics['numerical_errors'] += int(diagnostic['numerical_error'])
                statistics['negative_gap_errors'] += int(diagnostic['negative_gap_error'])
                statistics['solver_attempts'] += len(diagnostic['attempts'])
                statistics['max_iterations'] = max(statistics['max_iterations'], diagnostic['iterations'])
            evaluation = _json_bytes(archive.read(stem.replace('/fits/', '/evaluations/') + '.json'))
            verify_evaluation(evaluation, metadata, task['config']['n_test'], task['config']['test_batch_size'])
            _correspondence(evaluation['input_correspondence'], record, uid)
            with np.load(historical / 'tasks' / record['arrays_file'], allow_pickle=False) as saved:
                metrics = _metrics(B, saved['B_true'], supports)
            method_data[(uid, method)] = {'method_id': metadata['method_id'], 'metrics': evaluation['metrics'],
                                          'support': metrics, 'test_stream_hashes': evaluation['test_stream_hashes']}
            runtime[method].append(metadata['runtime_sec']); certificates[method] += len(metadata['diagnostics'])
            levels[metadata['input_correspondence']['identity_level']] += 1
            boundary.update(method + ':' + selection['boundary'] for selection in metadata['selected_lambdas'])
            diagnostics_summary[method]['local_cv_stages'] += len(metadata['selected_lambdas'])
        for uid in uids:
            if method_data[(uid, 'lasso')]['test_stream_hashes'] != method_data[(uid, 'glasso')]['test_stream_hashes']:
                raise ValueError('New methods use different test streams on the same complete input')
        rows = []
        for record in records:
            task = record['task']; map_row = mapping[task['task_id']]; uid = map_row['unique_input_id']
            with np.load(historical / 'tasks' / record['arrays_file'], allow_pickle=False) as saved:
                for method in ('adaptive', 'oracle', 'lasso', 'glasso'):
                    common = {key: task[key] for key in ('task_id', 'scenario_id', 'replicate', 'experiment', 'scenario_value')}
                    common.update({key: map_row[key] for key in ('unique_input_id', 'config_id', 'canonical_task_id')})
                    if method in ('adaptive', 'oracle'):
                        old = next(value for value in record['results'] if value['method'] == method)
                        metric = old['metrics']; mid = {'adaptive': 'v52_search_final_original_msic', 'oracle': 'oracle_ols_frozen'}[method]
                        support = _metrics(saved['B_' + method], saved['B_true'], [saved[f'S{i}_{method}'] for i in (1, 2, 3)])
                        stage = 'frozen_verified'
                    else:
                        item = method_data[(uid, method)]; metric, mid, support = item['metrics'], item['method_id'], item['support']; stage = 'certified_new'
                    rows.append({**common, 'method': method, 'method_id': mid, 'evidence_stage': stage,
                                 **{key: metric[key] for key in ('pmse', 'signal_mse')}, **support})
        unique = {}
        for row in rows:
            key = (row['unique_input_id'], row['method'])
            if key in unique and any(not np.isclose(unique[key][m], row[m], rtol=0, atol=0, equal_nan=True) for m in ('pmse', 'ree', 'element_f1', 'exact_recovery')):
                raise ValueError('Repeated label disagrees on its complete input')
            unique.setdefault(key, row)
        dedup, paired = list(unique.values()), []
        for config in sorted({row['config_id'] for row in dedup}):
            subset = [row for row in dedup if row['config_id'] == config]
            a = {row['unique_input_id']: row['pmse'] for row in subset if row['method'] == 'adaptive'}
            for method in ('lasso', 'glasso', 'oracle'):
                b = {row['unique_input_id']: row['pmse'] for row in subset if row['method'] == method}
                paired.append({'config_id': config, 'comparison': 'adaptive_minus_' + method, 'evidence_stage': 'verified', **paired_statistics(a, b)})
        tables = {'certified_label_metrics.csv': (rows, ('task_id', 'method')),
                  'certified_unique_metrics.csv': (dedup, ('unique_input_id', 'method')),
                  'certified_config_summary.csv': (summarize(dedup, 'config_id'), ('config_id', 'method')),
                  'certified_14_label_summary.csv': (summarize(rows, 'scenario_id'), ('scenario_id', 'method')),
                  'certified_paired_pmse.csv': (paired, ('config_id', 'comparison'))}
        csv_checks = {name: compare_rows(_csv_bytes(archive.read(doc_prefix + 'results/' + name)), values, keys)
                      for name, (values, keys) in tables.items()}
        facts = {'schema_version': 1, 'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
                 'archive_repo_relative_path': archive_path.relative_to(root).as_posix(), 'archive_sha256': file_sha(archive_path),
                 'archive_bytes': archive_path.stat().st_size, 'archive_entries': len(names), 'zip_crc_pass': True,
                 'recorded_run_identity': identity, 'run_identity_record_sha256': run['record_sha256'],
                 'run_identity_file_sha256': hashlib.sha256(run_bytes).hexdigest(),
                 'payload_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                 'original_payload_archive_sha256': file_sha(original_path), 'original_payload_archive_repo_relative_path': original_path.relative_to(root).as_posix(),
                 'payload_files_present_and_verified': checked_payload, 'current_immutable_files_verified': len(immutable),
                 'production_source_sha256': release['source_sha256'], 'simulation_source_sha256': plan['source_fingerprints'],
                 'all_csv_verified': True, 'csv_checks': csv_checks,
                 'counts': {'historical_records': 700, 'historical_npz_verified': 700, 'label_method_rows': len(rows),
                            'unique_inputs': 600, 'configurations': 10, 'valid_fits': 1200, 'evaluations': 1200, 'failures': 0,
                            'unique_method_rows': len(dedup), 'config_method_cells': 40, 'label_method_cells': 56, 'paired_config_comparisons': 30},
                 'certificate_counts': dict(certificates), 'certificate_extrema': dict(extrema), 'input_identity_levels': dict(levels),
                 'diagnostics_summary': {method: dict(values) for method, values in diagnostics_summary.items()},
                 'selected_lambda_boundaries': dict(boundary),
                 'runtime': {method: {'fits': len(values), 'aggregate_method_elapsed_seconds': float(sum(values)),
                                      'median_method_elapsed_seconds': float(np.median(values)), 'min_method_elapsed_seconds': float(min(values)),
                                      'max_method_elapsed_seconds': float(max(values))}
                             for method, values in runtime.items()},
                 'log_summary': {'focused_test_attempts': len([name for name in names if re.fullmatch(output_prefix + r'logs/focused_tests_attempt[0-9]+\.txt', name)]),
                                 'slurm_stdout_files': len([name for name in names if name.startswith(output_prefix + 'logs/slurm/') and name.endswith('.out')]),
                                 'slurm_stderr_files': len([name for name in names if name.startswith(output_prefix + 'logs/slurm/') and name.endswith('.err')]),
                                 'nonempty_stderr_files': [name for name in names if name.startswith(output_prefix + 'logs/slurm/') and name.endswith('.err') and archive.getinfo(name).file_size > 0],
                                 'unresolved_failures': 0},
                 'scope': 'Read-only ZIP CRC/seal/metadata/certificate/array and saved-result audit. No fitting or input/test reconstruction; historical prediction metrics reused, new prediction metrics checked for identity/stream consistency but not independently regenerated. Timers are aggregate method elapsed, not CPU or parallel wall time.',
                 'blocking_issues': []}
    return {'facts': facts, 'tables': {name: values for name, (values, _) in tables.items()}}


def write_results(root: Path, archive: Path, destination: Path | None = None):
    root = Path(root).resolve(); destination = destination or root / 'doc/taskcard18/cluster_review'
    result = audit(root, archive); facts = result['facts']
    for name, rows in result['tables'].items():
        save_csv(destination / 'rebuilt_results' / name, rows)
        save_csv(root / 'doc/taskcard18/results' / name, rows)
    facts['certified_csv_sha256'] = {'doc/taskcard18/results/' + name: file_sha(root / 'doc/taskcard18/results' / name) for name in result['tables']}
    facts['record_sha256'] = digest(facts)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'RETURN_VALIDATION.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    facts = write_results(args.root, args.archive)
    print('RETURN_AUDIT_PASS', facts['counts'], facts['certificate_counts'], flush=True)


if __name__ == '__main__':
    main()
