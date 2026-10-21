"""Task9 saved-pool diagnostics: no support search or coefficient fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.special import gammaln
from scipy.sparse import csr_matrix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(value.shape), 'dtype': value.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def stream_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.shape).encode('ascii'))
    h.update(value.dtype.str.encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def score_select(pool, n, dims, alpha, scale):
    ell = np.log(max(np.log(n), np.e))
    components = []
    for candidate in pool:
        sizes = candidate['sizes']
        fit = float(n*np.log(max(candidate['rss']/n, np.finfo(float).tiny)))
        df = float(alpha*int(np.prod(sizes))*ell)
        multiplicity = float(ell*sum(float(gammaln(p+1)-gammaln(s+1)-gammaln(p-s+1))
                                     for p, s in zip(dims, sizes)))
        components.append({'fit_term': fit, 'df_penalty': df,
                           'multiplicity_penalty': multiplicity, 'score': fit+df+multiplicity})
    scores = [c['score'] for c in components]
    minimum = min(range(len(pool)), key=scores.__getitem__)
    q_best = int(np.prod(pool[minimum]['sizes']))
    tolerance = float(max(scale, 0.)*np.sqrt(max(q_best, 1)*max(np.log(max(n, 2)), 1.)))
    eligible = [i for i, score in enumerate(scores) if score <= scores[minimum]+tolerance]
    selected = min(eligible, key=lambda i: (int(np.prod(pool[i]['sizes'])), scores[i], sum(pool[i]['sizes'])))
    return {'minimum_index': minimum, 'selected_index': selected, 'minimum_q': q_best,
            'minimum_score': scores[minimum], 'selected_score': scores[selected],
            'tolerance': tolerance, 'eligible_indices': eligible, 'candidate_scores': scores,
            'components': components}


def support_metrics(support, truth):
    pairs = [(set(s), set(t)) for s, t in zip(support, truth)]
    q = int(np.prod([len(s) for s in support]))
    qt = int(np.prod([len(t) for t in truth]))
    overlap = int(np.prod([len(s&t) for s, t in pairs]))
    return {'exact_recovery': int(all(s == t for s, t in pairs)),
            'mode_f1_macro': float(np.mean([2*len(s&t)/(len(s)+len(t)) for s, t in pairs])),
            'element_f1': 2*overlap/(q+qt)}


def sparse_predictions(X, coefficients):
    matrix = csr_matrix(coefficients.reshape(len(coefficients), -1).T)
    return np.asarray(X.reshape(len(X), -1) @ matrix)


def context(repo, output_dir=None):
    protocol = read(repo/'doc/taskcard9/PROTOCOL.json')
    base = repo/'numerical_simulation/outputs_selection_research_task8'
    require(sha(base/'plan.json') == protocol['historical_plan_sha256'], 'Historical plan drift')
    source = repo/'doc/algorithm_versions/v5.2-search-final/src'
    for name, digest in protocol['source_sha256'].items():
        require(sha(source/name) == digest, 'Frozen core drift: '+name)
        require(sha(repo/'src'/name) == digest, 'Production core drift: '+name)
    for name, digest in protocol['dependencies'].items():
        require(sha(repo/name) == digest, 'Dependency drift: '+name)
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    require(environment == protocol['environment'], 'Numerical environment drift')
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(variable) == '1', 'Single-thread environment required: '+variable)
    output = (repo/'doc/taskcard9/pool_msic') if output_dir is None else Path(output_dir).resolve()
    require(output.resolve().is_relative_to((repo/'doc/taskcard9').resolve()), 'Output must stay in doc/taskcard9')
    return protocol, base, source, output


def task_records(base):
    return [base/cohort/f'task_{i:03d}' for cohort in ('development', 'confirmation') for i in range(108)]


def load_task(folder):
    record = read(folder/'complete.json')
    for name, digest in record['artifact_sha256'].items():
        require(sha(folder/name) == digest, 'Artifact drift: '+str(folder/name))
    trace = read(folder/'trace.json')
    with np.load(folder/'candidate_coefficients.npz', allow_pickle=False) as saved:
        coefficients = saved['B']
    with np.load(folder/'truth.npz', allow_pickle=False) as saved:
        truth = saved['B_true']
    require(array_sha(truth) == record['input_fingerprints']['B_true'], 'Saved truth drift')
    require(len(coefficients) == len(trace['candidates']), 'Candidate library incomplete')
    for B, candidate in zip(coefficients, trace['candidates']):
        require(array_sha(B) == candidate['B_sha256'] and np.isfinite(B).all(), 'Candidate B identity drift')
        sizes = list(map(len, candidate['support']))
        require(sizes == candidate['sizes'] and int(np.prod(sizes)) == candidate['q'], 'Candidate shape drift')
        active = B[np.ix_(*candidate['support'])]
        require(np.count_nonzero(active) == np.count_nonzero(B), 'Coefficient outside support')
    return record, trace, coefficients, truth


def identity_row(record):
    r = record['request']; c = r['config']
    return {'cohort': r['cohort'], 'task_id': r['task_id'], 'cell_id': r['cell_id'],
            'replicate': r['replicate'], 'n_train': c['n_train'], 'snr': c['snr'],
            'truth_shape': c['sparsity'], 'rho': c['rho'][0]}


def identity_audit(repo, output_dir=None):
    protocol, base, source, output = context(repo, output_dir)
    plan = read(base/'plan.json')
    expected = {(r['cohort'], r['task_id']): r for r in plan['requests']}
    lock = read(base/'development_lock.json'); lock_sha = sha(base/'development_lock.json')
    require(lock['plan_sha256'] == protocol['historical_plan_sha256'], 'Development lock plan drift')
    require(lock['completed_development_tasks'] == 108, 'Incomplete development lock')
    artifact_checks = import_checks = B_checks = 0
    for folder in task_records(base):
        record, trace, Bs, truth = load_task(folder)
        request = record['request']; key = (request['cohort'], request['task_id'])
        require(request == expected[key], 'Exact historical request mismatch')
        require(record['plan_sha256'] == protocol['historical_plan_sha256'], 'Completed task plan drift')
        expected_lock = lock_sha if request['cohort'] == 'confirmation' else None
        require(record['development_lock_sha256'] == expected_lock, 'Confirmation lock binding drift')
        for name, item in record['actual_imports'].items():
            path = Path(item['path']).resolve()
            require(path.is_relative_to(source), 'Recorded import escaped frozen source')
            relative = path.relative_to(source).as_posix()
            require(item['sha256'] == protocol['source_sha256'][relative] == sha(path), 'Recorded source identity drift')
            import_checks += 1
        artifact_checks += len(record['artifact_sha256']); B_checks += len(Bs)
    for name, digest in lock['development_artifacts'].items():
        require(sha(base/'development'/name) == digest, 'Locked development artifact drift')
        artifact_checks += 1
    write(output/'identity_audit.json', {'passed': True, 'exact_requests': 216, 'task_plan_bindings': 216,
          'confirmation_lock_bindings': 108, 'development_lock_sha256': lock_sha,
          'core_files_frozen_and_production_each': 32, 'dependency_files': len(protocol['dependencies']),
          'actual_import_checks': import_checks, 'artifact_checks_including_development_lock': artifact_checks,
          'candidate_B_hash_finite_support_checks': B_checks,
          'protocol_sha256': sha(repo/'doc/taskcard9/PROTOCOL.json'), 'no_generation_or_fits': True})
    print('IDENTITY_AUDIT_COMPLETE', 216, B_checks, import_checks, flush=True)


def inventory(repo, output_dir=None):
    protocol, base, source, output = context(repo, output_dir)
    rows = []; same_shapes = []; structures = []
    for folder in task_records(base):
        record, trace, Bs, truth = load_task(folder)
        r = record['request']; c = r['config']; pool = trace['candidates']; qt = int(np.prod(c['sparsity']))
        capacity = qt <= int(np.floor(.7*c['n_train'])) and all(s <= cap for s, cap in zip(c['sparsity'], trace['caps']))
        in_grid = c['sparsity'] in trace['ordered_size_grid']
        exact_indices = [i for i, candidate in enumerate(pool) if candidate['support'] == record['supports_true']]
        selected = read(folder/'selection.json')['choices'][0]['selected_index']
        barrier = ('capacity' if not capacity else 'size_grid' if not in_grid else
                   'support_search' if not exact_indices else 'final_selection' if selected not in exact_indices else 'covered_selected')
        rows.append({**identity_row(record), 'candidate_count': len(pool), 'Q': int(np.floor(.7*c['n_train'])),
                     'caps': trace['caps'], 'truth_capacity': capacity, 'truth_shape_grid': in_grid,
                     'truth_support_found': bool(exact_indices), 'truth_support_candidate_indices': exact_indices,
                     'selected_exact': selected in exact_indices, 'barrier': barrier,
                     'q_distribution': dict(sorted(Counter(x['q'] for x in pool).items())),
                     'shape_distribution': dict(Counter('x'.join(map(str,x['sizes'])) for x in pool))})
        for i, candidate in enumerate(pool):
            structures.append({**identity_row(record), 'candidate_index': i, **candidate,
                               **support_metrics(candidate['support'], record['supports_true'])})
        groups = {}
        for i, candidate in enumerate(pool):
            groups.setdefault(tuple(candidate['sizes']), []).append(i)
        for shape, indices in groups.items():
            for a_pos, a in enumerate(indices):
                for b in indices[a_pos+1:]:
                    same_shapes.append({**identity_row(record), 'shape': list(shape), 'index_a': a, 'index_b': b,
                                        'same_support': pool[a]['support'] == pool[b]['support'],
                                        'same_B': pool[a]['B_sha256'] == pool[b]['B_sha256'],
                                        'rss_a': pool[a]['rss'], 'rss_b': pool[b]['rss']})
    groups = []
    for cohort in ('development', 'confirmation', 'both'):
        for n in (100, 200, 400):
            subset = [r for r in rows if r['n_train'] == n and (cohort == 'both' or r['cohort'] == cohort)]
            groups.append({'cohort': cohort, 'n_train': n, 'datasets': len(subset),
                           'candidate_count': sum(r['candidate_count'] for r in subset),
                           'barriers': dict(Counter(r['barrier'] for r in subset)),
                           **{key: sum(r[key] for r in subset) for key in ('truth_capacity', 'truth_shape_grid', 'truth_support_found', 'selected_exact')}})
    write_csv(output/'inventory.csv', rows); write_csv(output/'candidate_structure.csv', structures)
    write_csv(output/'same_shape_structure.csv', same_shapes)
    write(output/'inventory_summary.json', {'passed': True, 'datasets': len(rows), 'candidates': len(structures),
         'groups': groups, 'same_shape_pairs': len(same_shapes), 'same_shape_different_support': sum(not r['same_support'] for r in same_shapes),
         'protocol_sha256': sha(repo/'doc/taskcard9/PROTOCOL.json'), 'full_coefficient_and_artifact_identities_verified': True,
         'test_PMSE_scope': 'Saved A0/A3 selected models only before twelve identity-gated replays; other candidate test PMSE unavailable.',
         'no_fits': True})
    print('INVENTORY_COMPLETE', len(rows), len(structures), flush=True)


def ensure_imports(source, protocol):
    observed = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(source), 'Core import escaped frozen source: '+name)
            require(sha(path) == protocol['source_sha256'][path.relative_to(source).as_posix()], 'Imported source drift')
            observed[name] = {'path': str(path), 'sha256': sha(path)}
    return observed


def replay_case(folder, request, protocol, source, deadline):
    from numerical_simulation.data_generation import TensorSimulationConfig, generate_simulation_dataset, iter_test_batches
    require(perf_counter() < deadline, 'Replay budget expired')
    record, trace, Bs, truth = load_task(folder)
    require(record['request'] == request, 'Replay request drift')
    values = dict(request['config'])
    for key in ('dims', 'sparsity', 'rho'):
        values[key] = tuple(values[key])
    config = TensorSimulationConfig(**values)
    data = generate_simulation_dataset(config, seed=request['seed'])
    for key in ('X_train', 'y_train', 'B_true'):
        require(array_sha(getattr(data,key)) == record['input_fingerprints'][key], 'Train/truth fingerprint mismatch: '+key)
    require([array_sha(s) for s in data.supports_true] == record['input_fingerprints']['supports_true'], 'Support fingerprint mismatch')
    choices = read(folder/'selection.json')['choices']; pool = trace['candidates']
    # First pass gates the old predictions and test stream before any new-candidate evaluation.
    selected = Bs[[choice['selected_index'] for choice in choices]]
    old_sse = np.zeros(2); old_signal = np.zeros(2); h = hashlib.sha256(); count = batches = 0
    for X, y, signal in iter_test_batches(data, seed=request['test_seed']):
        for array in (X,y,signal):
            h.update(stream_sha(array).encode('ascii'))
        predictions = sparse_predictions(X, selected)
        old_sse += np.sum((y[:,None]-predictions)**2, axis=0)
        old_signal += np.sum((signal[:,None]-predictions)**2, axis=0)
        count += len(y); batches += 1
        require(perf_counter() < deadline, 'Replay budget expired')
    expected = record['test_fingerprint']
    require(h.hexdigest() == expected['sha256'] and count == expected['n_observations'] and batches == expected['n_batches'], 'Test stream fingerprint mismatch')
    errors = []
    for j, row in enumerate(record['metrics']):
        for value, key in ((old_sse[j]/count, 'pmse'), (old_signal[j]/count, 'signal_mse')):
            require(np.isclose(value,row[key],rtol=1e-9,atol=1e-10), 'Old prediction gate failed: '+key)
            errors.append(abs(float(value)-row[key]))
    imports = ensure_imports(source,protocol)
    # Second pass is the same frozen independent stream, only after all identity/prediction gates pass.
    sse = np.zeros(len(pool)); signals = np.zeros(len(pool)); h2 = hashlib.sha256()
    for X, y, signal in iter_test_batches(data, seed=request['test_seed']):
        for array in (X,y,signal):
            h2.update(stream_sha(array).encode('ascii'))
        prediction = sparse_predictions(X,Bs)
        sse += np.sum((y[:,None]-prediction)**2,axis=0)
        signals += np.sum((signal[:,None]-prediction)**2,axis=0)
        require(perf_counter() < deadline,'Replay budget expired')
    require(h2.hexdigest() == expected['sha256'], 'Evaluation stream differs from gated stream')
    for j, choice in enumerate(choices):
        require(np.isclose(sse[choice['selected_index']]/count,record['metrics'][j]['pmse'],rtol=1e-9,atol=1e-10), 'Full pool old PMSE mismatch')
    metrics = [{**identity_row(record),'candidate_index': i,'q': candidate['q'],'shape':candidate['sizes'],
                'support':candidate['support'],'B_sha256':candidate['B_sha256'],'rss':candidate['rss'],
                'pmse':float(sse[i]/count),'signal_mse':float(signals[i]/count),
                **support_metrics(candidate['support'],record['supports_true'])} for i,candidate in enumerate(pool)]
    pmse = sse/count; oracle = int(np.argmin(pmse)); rss_best = min(range(len(pool)),key=lambda i:pool[i]['rss'])
    types = {'train_RSS_min':rss_best,'original_MSIC_min':choices[0]['minimum_index'],
             'original_tolerance_selected':choices[0]['selected_index'],'posthoc_test_pool_oracle':oracle}
    representatives = [{'type':key,**metrics[index]} for key,index in types.items()]
    pairs = []
    for a in range(len(pool)):
        for b in range(a+1,len(pool)):
            if pool[a]['sizes'] != pool[b]['sizes'] or pool[a]['support'] == pool[b]['support']:
                continue
            low, high = sorted((a,b),key=lambda i:pool[i]['rss'])
            pairs.append({**identity_row(record),'shape':pool[a]['sizes'],'lower_RSS_index':low,'higher_RSS_index':high,
                          'RSS_delta_low_minus_high':pool[low]['rss']-pool[high]['rss'],
                          'PMSE_delta_low_minus_high':float(pmse[low]-pmse[high]),
                          'lower_RSS_higher_PMSE':bool(pool[low]['rss']<pool[high]['rss'] and pmse[low]>pmse[high])})
    selected_index=choices[0]['selected_index']
    summary={**identity_row(record),'candidate_count':len(pool),'pool_oracle_index':oracle,
             'pool_oracle_pmse':float(pmse[oracle]),'selected_index':selected_index,
             'selected_pmse':float(pmse[selected_index]),'selection_gap':float(pmse[selected_index]-pmse[oracle]),
             'truth_support_found':any(x['exact_recovery'] for x in metrics),
             'old_prediction_max_abs_error':max(errors),'gates_passed':True,
             'test_fingerprint':expected,'actual_imports':imports,'new_fits':0}
    return metrics,representatives,pairs,summary


def replay(repo, output_dir=None):
    protocol,base,source,output=context(repo, output_dir)
    require((output/'inventory_summary.json').is_file(),'Run inventory first')
    require(not any(name=='tensor_abess' or name.startswith('tensor_abess.') for name in sys.modules),'Use fresh process')
    sys.path.insert(0,str(source))
    import tensor_abess.tensor.sic
    started=perf_counter(); deadline=started+protocol['replay_wall_seconds_max']
    require(len(protocol['test_replay_requests'])<=protocol['max_replay_cases'],'Replay count exceeds cap')
    rows=[]; reps=[]; pairs=[]; summaries=[]
    for request in protocol['test_replay_requests']:
        folder=base/request['cohort']/f"task_{request['task_id']:03d}"
        metrics,representatives,comparisons,summary=replay_case(folder,request,protocol,source,deadline)
        write(output/f"replay_{request['cohort']}_{request['task_id']:03d}.json", {'summary':summary,'candidates':metrics,'representatives':representatives,'same_shape_pairs':comparisons})
        rows.extend(metrics); reps.extend(representatives); pairs.extend(comparisons); summaries.append(summary)
        print('REPLAY_COMPLETE',request['task_id'],len(metrics),round(perf_counter()-started,3),flush=True)
    write_csv(output/'replay_candidate_metrics.csv',rows);write_csv(output/'replay_representatives.csv',reps)
    write_csv(output/'replay_same_shape_pairs.csv',pairs);write_csv(output/'replay_selection_gaps.csv',summaries)
    groups=[]
    for n in (100,400):
        subset=[r for r in summaries if r['n_train']==n]
        groups.append({'n_train':n,'datasets':len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in ('pool_oracle_pmse','selected_pmse','selection_gap')},
                       'nonzero_gap_count':sum(r['selection_gap']>1e-10 for r in subset),
                       'selected_pool_oracle_count':sum(r['selected_index']==r['pool_oracle_index'] for r in subset)})
    write(output/'replay_summary.json',{'passed':True,'groups':groups,'datasets':len(summaries),'candidate_PMSE_count':len(rows),
         'same_shape_different_support_pairs':len(pairs),'lower_RSS_higher_PMSE_pairs':sum(r['lower_RSS_higher_PMSE'] for r in pairs),
         'maximum_old_prediction_abs_error':max(r['old_prediction_max_abs_error'] for r in summaries),
         'seconds':perf_counter()-started,'protocol_sha256':sha(repo/'doc/taskcard9/PROTOCOL.json'),'all_cases_gated_before_new_metrics':True,'new_fits':0})


def rescore(repo, output_dir=None):
    protocol,base,source,output=context(repo, output_dir)
    require(read(output/'replay_summary.json')['passed'],'P0 must be complete before P1')
    if not any(name=='tensor_abess' or name.startswith('tensor_abess.') for name in sys.modules):
        sys.path.insert(0,str(source))
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
    replay_metrics={}
    for path in output.glob('replay_development_*.json'):
        body=read(path)
        for row in body['candidates']:
            replay_metrics[(row['cohort'],row['task_id'],row['B_sha256'])]=row
    rows=[]; decompositions=[]; checks=0
    for folder in task_records(base):
        record,trace,Bs,truth=load_task(folder); pool=trace['candidates']; c=record['request']['config']
        old=read(folder/'selection.json')['choices']; choices={}
        for rule in protocol['rules']:
            choice=score_select(pool,c['n_train'],c['dims'],rule['alpha'],rule['scale'])
            official=[float(compute_tensor_msic_from_rss(x['rss'],*x['sizes'],c['n_train'],*c['dims'],alpha=rule['alpha'],gamma=1.)) for x in pool]
            require(official==choice['candidate_scores'],'Official score identity failed')
            choices[rule['rule']]=choice; checks+=len(pool)
        for name,j in (('A0',0),('A3',1)):
            require(all(choices[name][key]==old[j][key] for key in old[j] if key in choices[name]),'Historical selection replay drift')
        known={pool[choice['selected_index']]['B_sha256']:row for choice,row in zip(old,record['metrics'])}
        for rule in protocol['rules']:
            name=rule['rule']; choice=choices[name]; index=choice['selected_index']; candidate=pool[index]; digest=candidate['B_sha256']
            key=(record['request']['cohort'],record['request']['task_id'],digest)
            available=replay_metrics.get(key,known.get(digest))
            source_label='Task9_gated_replay' if key in replay_metrics else 'Task8_same_B_selected' if digest in known else 'not_available'
            q=candidate['q']; qt=int(np.prod(c['sparsity'])); Q=int(np.floor(.7*c['n_train']))
            row={**identity_row(record),'rule':name,'alpha':rule['alpha'],'scale':rule['scale'],
                 'selected_index':index,'minimum_index':choice['minimum_index'],'minimum_q':choice['minimum_q'],
                 'tolerance':choice['tolerance'],'eligible_count':len(choice['eligible_indices']),
                 'q':q,'shape':candidate['sizes'],'support':candidate['support'],'B_sha256':digest,
                 'q_under':int(q<qt),'q_over':int(q>qt),'q_over_Q':q/Q,'near_capacity':int(q/Q>=.9),
                 'selection_changed_vs_A0':int(digest!=pool[choices['A0']['selected_index']]['B_sha256']),
                 'train_rss':candidate['rss'],'msic':choice['selected_score'],
                 'ree':float(np.linalg.norm(Bs[index]-truth)/np.linalg.norm(truth)),
                 **support_metrics(candidate['support'],record['supports_true']),
                 'pmse':None if available is None else available['pmse'],
                 'signal_mse':None if available is None else available['signal_mse'],'pmse_source':source_label}
            rows.append(row)
            decompositions.append({**identity_row(record),'rule':name,'selected_index':index,'minimum_index':choice['minimum_index'],
                                   'selected_terms':choice['components'][index],'minimum_terms':choice['components'][choice['minimum_index']],
                                   'A0_selected_under_this_rule_terms':choice['components'][choices['A0']['selected_index']],
                                   'tolerance':choice['tolerance'],'band_changes_argmin':int(index!=choice['minimum_index'])})
    summaries=[]
    for cohort in ('development','confirmation','both'):
        for group,n in [('all',None),('n100',100),('n200',200),('n400',400),('SNR4',None)]:
            for rule in protocol['rules']:
                subset=[r for r in rows if r['rule']==rule['rule'] and (cohort=='both' or r['cohort']==cohort) and (n is None or r['n_train']==n) and (group!='SNR4' or r['snr']==4)]
                valid=[r for r in subset if r['pmse'] is not None]
                summary={'cohort':cohort,'group':group,'rule':rule['rule'],'datasets':len(subset),'pmse_available':len(valid),
                         **{k:float(np.mean([r[k] for r in subset])) for k in ('q','q_under','q_over','near_capacity','selection_changed_vs_A0','exact_recovery','mode_f1_macro','element_f1','ree','train_rss')},
                         'exact_count':sum(r['exact_recovery'] for r in subset),
                         'pmse_mean_available':float(np.mean([r['pmse'] for r in valid])) if valid else None,
                         'pmse_p95_available':float(np.quantile([r['pmse'] for r in valid],.95)) if valid else None}
                summaries.append(summary)
    factorial=[]
    for cohort in ('development','confirmation','both'):
        for n in (100,200,400):
            subset=[r for r in rows if r['n_train']==n and (cohort=='both' or r['cohort']==cohort)]
            by_rule={name:[r for r in subset if r['rule']==name] for name in ('A0','A1','A2','A3')}
            for metric in ('q','q_under','q_over','exact_recovery','mode_f1_macro','ree'):
                means={name:float(np.mean([r[metric] for r in group])) for name,group in by_rule.items()}
                factorial.append({'cohort':cohort,'n_train':n,'metric':metric,'datasets':len(by_rule['A0']),**means,
                                  'alpha_effect_at_scale1':means['A1']-means['A0'],
                                  'remove_band_effect_at_alpha1_5':means['A2']-means['A0'],
                                  'interaction':means['A3']-means['A1']-means['A2']+means['A0']})
    replay_rows=[r for r in rows if r['pmse_source']=='Task9_gated_replay']
    replay_factorial=[]
    for n in (100,400):
        subset=[r for r in replay_rows if r['n_train']==n]
        means={name:float(np.mean([r['pmse'] for r in subset if r['rule']==name])) for name in ('A0','A1','A2','A3')}
        replay_factorial.append({'n_train':n,'datasets':6,**means,'alpha_effect_at_scale1':means['A1']-means['A0'],
                                 'remove_band_effect_at_alpha1_5':means['A2']-means['A0'],'interaction':means['A3']-means['A1']-means['A2']+means['A0']})
    cells=[]
    for cohort in ('development','confirmation'):
        for cell in range(36):
            subset=[r for r in rows if r['cohort']==cohort and r['cell_id']==cell]
            baseline=[r for r in subset if r['rule']=='A0']; base_mean=float(np.mean([r['pmse'] for r in baseline]))
            for rule in ('A0','A1','A2','A3'):
                group=[r for r in subset if r['rule']==rule]; available=[r for r in group if r['pmse'] is not None]
                complete=len(available)==3; mean=float(np.mean([r['pmse'] for r in available])) if available else None
                cells.append({'cohort':cohort,'cell_id':cell,'n_train':group[0]['n_train'],'snr':group[0]['snr'],
                              'truth_shape':group[0]['truth_shape'],'rho':group[0]['rho'],'rule':rule,'datasets':3,'pmse_available':len(available),
                              'pmse_mean_available':mean,'complete_cell_pmse':complete,
                              'ratio_vs_A0_complete_only':mean/base_mean if complete else None})
    write_csv(output/'four_rule_selected.csv',rows);write_csv(output/'four_rule_groups.csv',summaries)
    write_csv(output/'four_rule_factorial.csv',factorial);write_csv(output/'four_rule_cells.csv',cells)
    write(output/'four_rule_decomposition.json',decompositions)
    write(output/'four_rule_summary.json',{'passed':True,'datasets':216,'rule_rows':len(rows),'official_score_checks':checks,
         'groups':summaries,'factorial':factorial,'paired_twelve_replay_pmse':replay_factorial,
         'pmse_available_by_rule':dict(Counter(r['rule'] for r in rows if r['pmse'] is not None)),
         'worst_complete_cell_by_rule':[{ 'cohort':cohort,'rule':rule,**max([r for r in cells if r['cohort']==cohort and r['rule']==rule and r['complete_cell_pmse']],key=lambda r:r['ratio_vs_A0_complete_only'])}
             for cohort in ('development','confirmation') for rule in ('A0','A1','A2','A3') if any(r['cohort']==cohort and r['rule']==rule and r['complete_cell_pmse'] for r in cells)],
         'actual_imports':ensure_imports(source,protocol),'no_fits':True,'research_only':True,'release_approved':False})
    print('RESCORE_COMPLETE',len(rows),checks,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inventory','replay','rescore','identity'))
    parser.add_argument('--output', type=Path, help='Fresh output directory inside doc/taskcard9; existing files are never overwritten')
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[2]
    {'inventory':inventory,'replay':replay,'rescore':rescore,'identity':identity_audit}[args.mode](repo,args.output)


if __name__=='__main__':
    main()
