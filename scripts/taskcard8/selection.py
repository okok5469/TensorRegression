"""Frozen paired MSIC research on the unmodified v5.2 cold candidate library.

The alpha=2 comparison is conditional on the alpha=1.5 adaptive terminal and
its complete original fixed grid. It does not claim a fresh alpha=2 search.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import scipy
from scipy.special import gammaln

from scripts.taskcard7.lemon import (array_sha, configure_core, observe_fit,
                                    read, require, root, sha, write, write_csv)


def output_directory():
    return root() / 'numerical_simulation/outputs_selection_research_task8'


def environment():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
            'threads': {k: os.environ.get(k) for k in
                        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def make_requests():
    formal = read(root() / 'numerical_simulation/outputs_v52_search_final_full/plan.json')['tasks'][0]['config']
    tasks = []
    cells = list(product((100, 200, 400), (.5, 2., 4.), ((4, 3, 5), (2, 3, 4)), (0., .6)))
    for cohort, seed_base in (('development', 1806101600), ('confirmation', 2806101600)):
        for cell_id, (n, snr, sizes, rho) in enumerate(cells):
            config = dict(formal, n_train=n, n_test=500, dims=[32, 20, 25], sparsity=list(sizes),
                          rho=[rho] * 3, snr=snr, support_shape='continuous', test_batch_size=100)
            for replicate in range(3):
                seed = seed_base + cell_id * 1000 + replicate
                tasks.append({'cohort': cohort, 'task_id': cell_id * 3 + replicate,
                              'cell_id': cell_id, 'replicate': replicate, 'config': config,
                              'seed': seed, 'test_seed': seed + 50_000_000})
    require(len({r[k] for r in tasks for k in ('seed', 'test_seed')}) == 432, 'Seed collision')
    return tasks


def terms(candidate, n, dims, alpha):
    sizes = candidate['sizes']
    ell = np.log(max(np.log(n), np.e))
    log_combination = sum(float(gammaln(p + 1) - gammaln(s + 1) - gammaln(p - s + 1))
                          for p, s in zip(dims, sizes))
    fit = float(n * np.log(max(candidate['rss'] / n, np.finfo(float).tiny)))
    df = float(alpha * int(np.prod(sizes)) * ell)
    multiplicity = float(ell * log_combination)
    return {'fit_term': fit, 'df_penalty': df, 'multiplicity_penalty': multiplicity,
            'score': fit + df + multiplicity}


def select(pool, n, dims, alpha, scale):
    """Training-only exact v5.2 ordering and tie rule; no test input accepted."""
    require(bool(pool), 'Empty candidate library')
    scores = [terms(c, n, dims, alpha)['score'] for c in pool]
    minimum = min(range(len(pool)), key=lambda i: scores[i])
    q_best = int(np.prod(pool[minimum]['sizes']))
    tolerance = float(scale * np.sqrt(max(q_best, 1) * max(np.log(max(n, 2)), 1.)))
    eligible = [i for i, score in enumerate(scores) if score <= scores[minimum] + tolerance]
    chosen = min(eligible, key=lambda i: (int(np.prod(pool[i]['sizes'])), scores[i], sum(pool[i]['sizes'])))
    return {'alpha': alpha, 'gamma': 1., 'scale': scale, 'minimum_index': minimum,
            'minimum_q': q_best, 'minimum_score': scores[minimum], 'tolerance': tolerance,
            'eligible_indices': eligible, 'selected_index': chosen,
            'selected_score': scores[chosen], 'candidate_scores': scores}


def development_gate(rows):
    a = [r for r in rows if r['rule'] == 'baseline']
    b = [r for r in rows if r['rule'] == 'alpha2_scale0']
    require(bool(a) and len(a) == len(b), 'Incomplete paired development metrics')
    mean = lambda rs, key: float(np.mean([r[key] for r in rs]))
    cells = sorted({r['cell_id'] for r in a})
    ratios = {str(c): mean([r for r in b if r['cell_id'] == c], 'pmse') /
              mean([r for r in a if r['cell_id'] == c], 'pmse') for c in cells}
    strong_a, strong_b = [r for r in a if r['snr'] == 4], [r for r in b if r['snr'] == 4]
    small_a, small_b = [r for r in a if r['n_train'] == 100], [r for r in b if r['n_train'] == 100]
    values = {'overall_pmse_ratio': mean(b, 'pmse') / mean(a, 'pmse'),
              'worst_cell_pmse_ratio': max(ratios.values()), 'cell_pmse_ratios': ratios,
              'strong_mode_f1_delta': mean(strong_b, 'mode_f1_macro') - mean(strong_a, 'mode_f1_macro'),
              'strong_underselection_delta': mean(strong_b, 'q_under') - mean(strong_a, 'q_under'),
              'n100_near_capacity_delta': mean(small_b, 'near_capacity') - mean(small_a, 'near_capacity')}
    checks = {'overall_pmse': values['overall_pmse_ratio'] <= .98,
              'worst_cell_pmse': values['worst_cell_pmse_ratio'] <= 1.15,
              'strong_snr4_mode_f1': values['strong_mode_f1_delta'] >= -.02,
              'strong_snr4_underselection': values['strong_underselection_delta'] <= .05,
              'n100_near_capacity': values['n100_near_capacity_delta'] <= 0.}
    return {'passed': all(checks.values()), 'checks': checks, 'values': values}


def check_budget(plan):
    require(sha(root() / 'doc/taskcard8/COMPUTE_BUDGET.json') == plan['budget_sha256'], 'Budget identity drift')
    require(datetime.now(timezone.utc) < datetime.fromisoformat(plan['budget']['deadline_utc']),
            'Research wall-time budget expired')


def core_manifest():
    directory = root() / 'doc/algorithm_versions/v5.2-search-final'
    expected = read(directory / 'VERSION.json')['source_sha256']
    actual = {p.relative_to(directory / 'src').as_posix(): sha(p) for p in (directory / 'src').rglob('*.py')}
    require(len(actual) == 32 and actual == expected, 'Frozen 32-file v5.2 core drift')
    return actual


def dependencies():
    paths = ('scripts/taskcard8/selection.py', 'scripts/taskcard7/lemon.py',
             'numerical_simulation/version_validation/worker.py', 'numerical_simulation/data_generation.py',
             'numerical_simulation/metrics.py')
    return {p: sha(root() / p) for p in paths}


def historical_seed_audit(requests):
    """Check materialized train/test seeds in prior simulation plans, not results."""
    seeds, manifests = set(), {}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('seed', 'test_seed') and isinstance(item, int):
                    seeds.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in sorted((root() / 'numerical_simulation').glob('outputs*/plan.json')):
        if path.parent == output_directory():
            continue
        visit(read(path))
        manifests[path.relative_to(root()).as_posix()] = sha(path)
    overlap = seeds.intersection(r[k] for r in requests for k in ('seed', 'test_seed'))
    require(not overlap, 'New research seed overlaps a materialized historical plan seed')
    return {'prior_plan_sha256': manifests, 'historical_unique_materialized_seeds': len(seeds),
            'new_unique_seeds': 432, 'overlap': [],
            'scope': 'Exact seed/test_seed fields recursively in existing top-level simulation outputs*/plan.json; no assertion about external unpublished seeds'}


def verify_plan(output):
    plan = read(output / 'plan.json')
    require(plan['core_sha256'] == core_manifest(), 'Protocol core identity drift')
    require(plan['dependencies'] == dependencies(), 'Protocol dependency drift')
    require(plan['environment'] == environment(), 'Protocol environment drift')
    require(plan['requests'] == make_requests(), 'Protocol request/configuration drift')
    require(sha(root() / plan['formal_config_source']) == plan['formal_plan_sha256'], 'Formal plan identity drift')
    check_budget(plan)
    return plan


def prepare(output):
    require(not output.exists(), 'Research output exists; refusing overwrite')
    budget_path = root() / 'doc/taskcard8/COMPUTE_BUDGET.json'
    plan = {'schema_version': 1, 'research_only': True, 'release_approved': False,
            'base_version': 'v5.2-search-final', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'budget': read(budget_path), 'budget_sha256': sha(budget_path),
            'formal_config_source': 'numerical_simulation/outputs_v52_search_final_full/plan.json',
            'formal_plan_sha256': sha(root() / 'numerical_simulation/outputs_v52_search_final_full/plan.json'),
            'core_sha256': core_manifest(), 'dependencies': dependencies(), 'environment': environment(),
            'workers': 4, 'threads_per_worker': 1, 'mode_alpha': 1., 'mode_gamma': 1.,
            'pilot': 'Development task0 runs alone, remains part of the108 tasks, then four single-thread workers; no outcome-dependent configuration changes',
            'rules': [{'rule': 'baseline', 'alpha': 1.5, 'scale': 1., 'gamma': 1.},
                      {'rule': 'alpha2_scale0', 'alpha': 2., 'scale': 0., 'gamma': 1.}],
            'library': 'One unchanged formal v5.2 cold adaptive fit with alpha1.5/scale1; passive capture of terminal plus every ordered original global fixed candidate. Both rules rescore identical library; no claim of independent alpha2 adaptive search.',
            'candidate_order': 'adaptive terminal first; original lexicographic s1,s2,s3 grid subject to original caps and q<=floor(.7*n)',
            'selection': 'first minimum score; within scale*sqrt(q_min*log(n)) choose (q,score,sum(sizes)); original stable order breaks ties',
            'near_capacity_definition': 'q/floor(.7*n)>=.9',
            'q_under_definition': 'estimated rectangular q < true rectangular q',
            'support_capacity': 'Report truth shape representability under all three original mode caps and global Q; do not alter all-SNR4 prespecified development gate; additionally describe representable subset.',
            'development_gate': {'overall_pmse_ratio_max': .98, 'worst_cell_pmse_ratio_max': 1.15,
                                 'strong_snr4_mode_f1_delta_min': -.02, 'strong_snr4_q_under_delta_max': .05,
                                 'n100_near_capacity_delta_max': 0.},
            'confirmation': 'Only after all108 development records and immutable locked rule/gate/manifest SHA. Sole predeclared alternative also receives fixed-rule independent research replication if development gate fails; never claim development qualification or change thresholds.',
            'shared_test': 'Both selected models evaluated together on identical 500-observation independent batches; test outcomes never enter support search, scores, or selection.',
            'seed_rule': 'development base1806101600; confirmation base2806101600; cell*1000+replicate; test=train_seed+50000000; all432 seeds distinct',
            'requests': make_requests()}
    plan['historical_seed_audit'] = historical_seed_audit(plan['requests'])
    require(all(v == '1' for v in plan['environment']['threads'].values()), 'Each BLAS thread variable must equal1')
    check_budget(plan)
    output.mkdir(parents=True)
    write(output / 'plan.json', plan)
    version_dir = root() / 'doc/algorithm_versions/v6-selection-research'
    write(version_dir / 'VERSION.json', {'version': 'v6-selection-research', 'research_only': True,
          'release_approved': False, 'status': 'prespecified_research_not_an_algorithm_release',
          'base_version': 'v5.2-search-final', 'source_copied_or_modified': False,
          'actual_core': 'doc/algorithm_versions/v5.2-search-final/src',
          'source_sha256': plan['core_sha256'], 'plan_sha256': sha(output / 'plan.json')})
    print('PROTOCOL_FROZEN', sha(output / 'plan.json'), flush=True)


def confirmation_lock(output, plan):
    path = output / 'development_lock.json'
    require(path.is_file(), 'Missing development lock before confirmation')
    lock = read(path)
    require(lock['plan_sha256'] == sha(output / 'plan.json'), 'Development lock protocol drift')
    require(lock['rules'] == plan['rules'], 'Development lock rule drift')
    require(lock['development_artifacts'] == artifact_manifest(output / 'development'), 'Development lock artifact drift')
    require(lock['completed_development_tasks'] == 108, 'Development lock incomplete')
    return sha(path)


def artifact_manifest(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def worker(output, cohort, task_id):
    plan = verify_plan(output)
    lock_sha = confirmation_lock(output, plan) if cohort == 'confirmation' else None
    request = next(r for r in plan['requests'] if r['cohort'] == cohort and r['task_id'] == task_id)
    folder = output / cohort / f'task_{task_id:03d}'
    require(not folder.exists(), 'Dataset output exists; refusing overwrite')
    folder.mkdir(parents=True)
    write(folder / 'started.json', {'request': request, 'plan_sha256': sha(output / 'plan.json'),
          'development_lock_sha256': lock_sha, 'started_utc': datetime.now(timezone.utc).isoformat()})
    source = configure_core('v5.2-search-final')
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset
    from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
    from numerical_simulation.version_validation.worker import fit_adaptive, evaluate_shared_test
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
    values = dict(request['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    started = perf_counter()
    dataset = generate_simulation_dataset(config, seed=request['seed'])
    input_sha = {name: array_sha(getattr(dataset, name)) for name in ('X_train', 'y_train', 'B_true')}
    input_sha['supports_true'] = [array_sha(s) for s in dataset.supports_true]
    check_budget(plan)
    fit_started = perf_counter()
    fitted, trace, coefficients = observe_fit(lambda: fit_adaptive(dataset.X_train, dataset.y_train, config))
    fit_sec = perf_counter() - fit_started
    check_budget(plan)
    candidates = trace['candidates']
    choices = [select(candidates, config.n_train, config.dims, r['alpha'], r['scale']) for r in plan['rules']]
    for rule, choice in zip(plan['rules'], choices):
        official_scores = [float(compute_tensor_msic_from_rss(c['rss'], *c['sizes'], config.n_train,
                           *config.dims, alpha=rule['alpha'], gamma=1.)) for c in candidates]
        require(choice['candidate_scores'] == official_scores, 'Scoring differs from actual formal MSIC')
    require(np.array_equal(coefficients[choices[0]['selected_index']], fitted.B), 'Default rescore changed actual v5.2 B')
    require(choices[0]['selected_score'] == fitted.sic, 'Default rescore changed actual v5.2 MSIC')
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(source), 'Actual core import escaped frozen v5.2')
            require(sha(path) == plan['core_sha256'][path.relative_to(source).as_posix()], 'Actual core import SHA drift')
            imports[name] = {'path': str(path), 'sha256': sha(path)}
    test_metrics, test_fingerprint = evaluate_shared_test(dataset,
                         [coefficients[c['selected_index']] for c in choices], request['test_seed'])
    rows = []
    q_true = int(np.prod(config.sparsity))
    capacity = int(np.floor(.7 * config.n_train))
    representable = all(s <= cap for s, cap in zip(config.sparsity, trace['caps'])) and q_true <= capacity
    for rule, choice, test in zip(plan['rules'], choices, test_metrics):
        i = choice['selected_index']
        candidate, B = candidates[i], coefficients[i]
        supports = [np.asarray(s, dtype=int) for s in candidate['support']]
        row = {'cohort': cohort, 'task_id': task_id, 'cell_id': request['cell_id'], 'replicate': request['replicate'],
               'rule': rule['rule'], 'n_train': config.n_train, 'snr': config.snr,
               'sparsity': list(config.sparsity), 'rho': list(config.rho), 'seed': request['seed'],
               'q': candidate['q'], 'Q': capacity, 'q_over_Q': candidate['q'] / capacity,
               'q_true': q_true, 'q_over': float(candidate['q'] > q_true), 'q_under': float(candidate['q'] < q_true),
               'near_capacity': float(candidate['q'] / capacity >= .9), 'caps_representable': representable,
               'train_rss': candidate['rss'], 'train_rmse': float(np.sqrt(candidate['rss'] / config.n_train)),
               'selected_index': i, 'minimum_msic_index': choice['minimum_index'],
               'minimum_msic_q': choice['minimum_q'], 'msic': choice['selected_score'],
               'fit_sec_shared': fit_sec, 'candidate_count': len(candidates),
               **test, **compute_coefficient_metrics(B, dataset.B_true),
               **compute_support_metrics(*supports, *dataset.supports_true)}
        rows.append(row)
    decompositions = []
    for i, candidate in enumerate(candidates):
        a, b = (terms(candidate, config.n_train, config.dims, r['alpha']) for r in plan['rules'])
        decompositions.append({'candidate_index': i, 'q': candidate['q'], 'sizes': candidate['sizes'],
             'rss': candidate['rss'], 'baseline': a, 'alpha2_scale0': b,
             'additional_df_penalty': b['df_penalty'] - a['df_penalty']})
    for filename, arrays in [('candidate_coefficients.npz', {'B': np.stack(coefficients)}),
                             ('truth.npz', {'B_true': dataset.B_true})]:
        with (folder / filename).open('xb') as handle:
            np.savez_compressed(handle, **arrays)
    write(folder / 'trace.json', trace)
    write(folder / 'selection.json', {'rules': plan['rules'], 'choices': choices})
    write(folder / 'mechanism.json', {'candidate_terms': decompositions,
          'selected_baseline_index': choices[0]['selected_index'], 'selected_alternative_index': choices[1]['selected_index'],
          'default_rescore_B_exact': True, 'default_rescore_MSIC_exact': True})
    check_budget(plan)
    write(folder / 'complete.json', {'request': request, 'plan_sha256': sha(output / 'plan.json'),
          'development_lock_sha256': lock_sha, 'input_fingerprints': input_sha,
          'supports_true': [s.tolist() for s in dataset.supports_true], 'test_fingerprint': test_fingerprint,
          'actual_imports': imports, 'metrics': rows, 'runtime_sec': perf_counter() - started,
          'artifact_sha256': artifact_manifest(folder), 'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('DATASET_COMPLETE', cohort, task_id, 'candidates', len(candidates), 'sec', round(perf_counter() - started, 3), flush=True)


def collect(output, cohort):
    rows = []
    for task_id in range(108):
        folder = output / cohort / f'task_{task_id:03d}'
        record = read(folder / 'complete.json')
        require(record['plan_sha256'] == sha(output / 'plan.json'), 'Completed dataset protocol drift')
        for name, digest in record['artifact_sha256'].items():
            require(sha(folder / name) == digest, 'Completed dataset artifact drift')
        rows.extend(record['metrics'])
    return rows


def summarize(output, cohort, rows):
    serial = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows]
    write_csv(output / f'{cohort}_metrics.csv', serial)
    summary = []
    keys = ('pmse', 'ree', 'exact_recovery', 'mode_f1_macro', 'element_f1', 'q', 'q_over_Q',
            'q_under', 'q_over', 'near_capacity', 'train_rss', 'fit_sec_shared')
    for rule in ('baseline', 'alpha2_scale0'):
        for group, subset in [('all', [r for r in rows if r['rule'] == rule]),
                              ('SNR4_representable', [r for r in rows if r['rule'] == rule and r['snr'] == 4 and r['caps_representable']])]:
            summary.append({'rule': rule, 'group': group, 'n': len(subset),
                            **{k: float(np.mean([r[k] for r in subset])) for k in keys}})
    write(output / f'{cohort}_summary.json', {'cohort': cohort, 'n_datasets': 108, 'metrics': summary,
          'gate': development_gate(rows), 'plan_sha256': sha(output / 'plan.json')})


def run_cohort(output, plan, cohort):
    check_budget(plan)
    if cohort == 'confirmation':
        confirmation_lock(output, plan)
    log_dir = output / 'logs'
    log_dir.mkdir(exist_ok=True)
    pending = list(range(108))
    require(not (output / cohort).exists(), 'Cohort output already exists; refusing overwrite')
    running = {}
    try:
        while pending or running:
            check_budget(plan)
            pilot_pending = cohort == 'development' and not (output / 'development/task_000/complete.json').is_file()
            worker_limit = 1 if pilot_pending else plan['workers']
            while pending and len(running) < worker_limit:
                task_id = pending.pop(0)
                handle = (log_dir / f'{cohort}_{task_id:03d}.txt').open('x', encoding='utf-8')
                command = [sys.executable, '-B', '-m', 'scripts.taskcard8.selection', 'worker',
                           '--cohort', cohort, '--task-id', str(task_id)]
                process = subprocess.Popen(command, cwd=root(), stdout=handle, stderr=subprocess.STDOUT)
                running[task_id] = process, handle
            for task_id, (process, handle) in list(running.items()):
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                del running[task_id]
                require(status == 0, f'{cohort} dataset {task_id} failed; see immutable log')
                print('COHORT_PROGRESS', cohort, 108 - len(pending) - len(running), '/108', flush=True)
            sleep(.3)
    finally:
        for process, handle in running.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            handle.close()


def run(output):
    plan = verify_plan(output)
    try:
        run_cohort(output, plan, 'development')
        rows = collect(output, 'development')
        summarize(output, 'development', rows)
        gate = development_gate(rows)
        write(output / 'development_lock.json', {'plan_sha256': sha(output / 'plan.json'),
              'locked_utc': datetime.now(timezone.utc).isoformat(), 'completed_development_tasks': 108,
              'rules': plan['rules'], 'development_gate': gate,
              'status': 'development_qualified' if gate['passed'] else 'fixed_rule_research_replication_only',
              'development_summary_sha256': sha(output / 'development_summary.json'),
              'development_artifacts': artifact_manifest(output / 'development')})
        print('DEVELOPMENT_LOCKED', gate['passed'], sha(output / 'development_lock.json'), flush=True)
        run_cohort(output, plan, 'confirmation')
        confirmation_rows = collect(output, 'confirmation')
        summarize(output, 'confirmation', confirmation_rows)
        write(output / 'completion.json', {'status': 'complete_research_only', 'development': 108, 'confirmation': 108,
              'plan_sha256': sha(output / 'plan.json'), 'development_lock_sha256': sha(output / 'development_lock.json'),
              'development_qualified': gate['passed'], 'release_approved': False,
              'artifact_sha256': artifact_manifest(output)})
    except Exception as error:
        write(output / 'stopped.json', {'status': 'stopped_incomplete', 'error': str(error),
              'stopped_utc': datetime.now(timezone.utc).isoformat(),
              'completed_development': len(list((output / 'development').glob('*/complete.json'))),
              'completed_confirmation': len(list((output / 'confirmation').glob('*/complete.json'))),
              'plan_sha256': sha(output / 'plan.json')})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--cohort', choices=('development', 'confirmation'))
    parser.add_argument('--task-id', type=int)
    args = parser.parse_args()
    output = output_directory()
    if args.mode == 'worker':
        worker(output, args.cohort, args.task_id)
        return
    {'prepare': prepare, 'run': run}[args.mode](output)


if __name__ == '__main__':
    main()
