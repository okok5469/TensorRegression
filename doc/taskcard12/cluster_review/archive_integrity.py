"""Read-only archive verification; never runs DGP, fitting, or cluster commands."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from research.taskcard12 import simulation as run
from research.taskcard12.analysis import analyze


def main():
    output = ROOT / 'numerical_simulation/outputs_msic_pair_task12'
    destination = Path(__file__).parent
    report_path = destination / 'ARCHIVE_INTEGRITY.json'
    if report_path.exists():
        raise FileExistsError('Preserve the existing archive review')
    start = time.monotonic()
    report = {'passed': False, 'stage': 'protocol_chain', 'cases_verified': [],
              'review_utc': datetime.now(timezone.utc).isoformat(),
              'review_environment': {'python': platform.python_version(), 'system': platform.system(),
                                     'numpy': np.__version__, 'scipy': scipy.__version__},
              'readonly_science_archive': True, 'new_search_calls': 0, 'new_DGP_calls': 0,
              'new_external_test_calls': 0, 'current_environment_verify_plan_called': False}
    try:
        plan = run.read_json(output / 'plan.json')
        protocol = run.read_json(output / 'CLUSTER_PROTOCOL.json')
        plan_sha = run.file_sha(output / 'plan.json')
        protocol_sha = run.file_sha(output / 'CLUSTER_PROTOCOL.json')
        run.require(protocol_sha == plan['protocol_sha256'] ==
                    run.file_sha(ROOT / plan['protocol_path']), 'Archive/frozen protocol bytes differ')
        run.require(protocol == plan['protocol'], 'Embedded frozen protocol differs')
        run.require(all(plan[key] == protocol[key] for key in ('requests', 'smoke_requests')),
                    'Saved plan request identities differ')
        run.verify_sources(ROOT, protocol['source_files'])
        run.validate_requests(plan['requests'])
        history = run.validate_request_artifacts(ROOT, protocol)
        run.validate_environment(plan['environment'], protocol)
        seeds = [r[k] for r in plan['requests'] + plan['smoke_requests'] for k in ('seed', 'test_seed')]
        run.require(len(seeds) == len(set(seeds)), 'Archived smoke/formal seed collision')
        with zipfile.ZipFile(ROOT / 'doc/taskcard12/delivery/taskcard12_cluster.zip') as archive:
            bundled = archive.read('taskcard12_bundle/FILE_MANIFEST.json')
        run.require(bundled == (output / 'FILE_MANIFEST.json').read_bytes(),
                    'Returned manifest differs from delivered package')
        manifest = json.loads(bundled)
        run.require(all(manifest['files'][name]['sha256'] == digest
                        for name, digest in protocol['source_files'].items()),
                    'Delivered source manifest differs from protocol')
        report.update(plan_sha256=plan_sha, protocol_sha256=protocol_sha,
                      source_files_verified=len(protocol['source_files']),
                      historical_input_pairs=len(history), saved_environment=plan['environment'])

        records = {}
        claims = {}
        known = {(r['X_train'], r['y_train']) for r in history}
        frozen_core = PurePosixPath(protocol['cluster_root']) / 'doc/algorithm_versions/v5.2-search-final/src'
        for phase, key in (('smoke', 'smoke_requests'), ('formal', 'requests')):
            expected = [output / phase / f"task_{r['task_id']:03d}" for r in plan[key]]
            run.require(sorted((output / phase).glob('task_*')) == expected,
                        'Missing/extra archived task coverage: ' + phase)
            records[phase] = []
            for folder, request in zip(expected, plan[key]):
                report['stage'] = folder.relative_to(output).as_posix()
                row = run.completed_result(folder, request, plan, plan_sha)
                run.require(row['search_calls'] == 1 and not row['release_approved'],
                            'Archived native search/release flags differ')
                started = run.read_json(folder / 'started.json')
                execution = run.read_json(folder / 'execution.json')
                run.require(started['plan_sha256'] == plan_sha and started['slurm'] == row['slurm'],
                            'Started/completed job identity differs')
                run.require(execution['passed'] and not execution['timed_out'] and execution['returncode'] == 0,
                            'Archived case execution failed')
                run.require(row['slurm']['array_task_id'] == request['task_id'],
                            'Archived SLURM task identity differs')
                metadata = run.read_json(folder / 'pool/metadata.json')
                pool_env = metadata['identity']['environment']
                run.require(all(pool_env[k] == plan['environment'][k]
                                for k in ('python', 'numpy', 'scipy', 'executable', 'threads')) and
                            pool_env['platform'].startswith('Linux-'), 'Saved pool environment chain differs')
                for imported in row['actual_core_imports'].values():
                    relative = PurePosixPath(imported['path']).relative_to(frozen_core).as_posix()
                    run.require(metadata['identity']['core_sha256'][relative] == imported['sha256'],
                                'Actual archived import differs from frozen core')
                pair = (row['input_fingerprints']['X_train'], row['input_fingerprints']['y_train'])
                run.require(pair not in known and pair not in claims, 'Duplicate archived training input')
                claim_name = hashlib.sha256(json.dumps(pair, separators=(',', ':')).encode()).hexdigest() + '.json'
                claim_path = output / 'input_claims' / claim_name
                claim = run.read_json(claim_path)
                run.require(claim == {'input_fingerprints': row['input_fingerprints'], 'request': request},
                            'Archived exclusive input claim differs')
                run.require(run.read_json(folder / 'input_identity.json')['claim_sha256'] == run.file_sha(claim_path),
                            'Archived input claim hash differs')
                claims[pair] = claim_name
                records[phase].append(row)
                report['cases_verified'].append({'phase': phase, 'task_id': request['task_id'],
                    'complete_sha256': run.file_sha(folder / 'complete.json'),
                    'candidate_count': len(metadata['trace']['candidates'])})
                count = len(report['cases_verified'])
                if count <= 3 or count % 20 == 0:
                    elapsed = time.monotonic() - start
                    print(f"VERIFIED {count}/383 elapsed={elapsed:.1f}s projected_total={elapsed*383/count:.1f}s",
                          flush=True)
        run.require(sorted(p.name for p in (output / 'input_claims').glob('*.json')) == sorted(claims.values()),
                    'Extra/missing exclusive input claim files')
        report['stage'] = 'smoke_gate'
        gate = run.read_json(output / 'smoke_gate.json')
        smoke_sha = {str(r['task_id']): run.file_sha(output / 'smoke' / f"task_{r['task_id']:03d}" / 'complete.json')
                     for r in plan['smoke_requests']}
        run.require(gate['passed'] and gate['plan_sha256'] == plan_sha and
                    gate['environment'] == plan['environment'] and gate['smoke_complete_sha256'] == smoke_sha and
                    gate['cost'] == run.smoke_cost(records['smoke'], protocol['max_parallel']) and
                    gate['cost']['passed'], 'Saved smoke gate differs')
        run.require(all(r['slurm']['array_task_count'] == 3 for r in records['smoke']),
                    'Archived smoke SLURM array count differs')

        report['stage'] = 'merged_artifact_chain'
        merged = output / 'merged'
        integrity = run.read_json(merged / 'integrity.json')
        complete_sha = {str(r['task_id']): run.file_sha(output / 'formal' / f"task_{r['task_id']:03d}" / 'complete.json')
                        for r in plan['requests']}
        run.require(integrity['passed'] and integrity['formal_cases'] == 380 and
                    integrity['paired_metric_rows'] == 760 and integrity['balanced_cases'] == 360 and
                    integrity['pressure_extra_cases'] == 20 and integrity['plan_sha256'] == plan_sha and
                    integrity['protocol_sha256'] == protocol_sha and integrity['complete_sha256'] == complete_sha,
                    'Merged identity/complete hash chain differs')
        merged_hashes = run.artifact_manifest(merged)
        merged_hashes.pop('integrity.json')
        run.require(merged_hashes == integrity['merged_artifact_sha256'], 'Merged artifact hash/inventory differs')
        report['stage'] = 'summary_recomputation'
        summary = run.summarize(records['formal'])
        run.require(summary == run.read_json(merged / 'summary.json'), 'Recomputed summary differs')
        report['stage'] = 'analysis_recomputation'
        inference = analyze(records['formal'], protocol)
        run.require(inference == run.read_json(merged / 'statistical_analysis.json'), 'Recomputed frozen analysis differs')
        report['stage'] = 'paired_csv_recomputation'
        raw = []
        for row in records['formal']:
            request = row['request']
            identity = {k: request[k] for k in ('task_id', 'cell_id', 'replicate', 'pressure', 'seed', 'test_seed')}
            identity.update({k: request['config'][k] for k in ('n_train', 'snr', 'sparsity', 'rho')})
            raw.extend({**identity, 'arm': label, **row['metrics'][label], **row['recovery_evidence']}
                       for label in ('A0', 'A1'))
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(raw[0]))
        writer.writeheader()
        for row in raw:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
        run.require(buffer.getvalue().encode() == (merged / 'paired_metrics.csv').read_bytes(),
                    'Recomputed paired metric CSV differs')
        report.update(passed=True, stage='complete', formal_cases=380, smoke_cases=3,
                      merged_summary_exact=True, merged_analysis_exact=True, paired_csv_bytes_exact=True,
                      inference_gates=inference['gates'], eligible_for_version_review=inference['eligible_for_version_review'])
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        run.write_json(report_path, report)
        print('REPORT', str(report_path), 'PASSED', report['passed'], 'STAGE', report['stage'], flush=True)


if __name__ == '__main__':
    main()
