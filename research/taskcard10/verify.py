"""Read-only final Task10 identity, coefficients, predictions and report audit."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array)
    header = json.dumps({'shape':list(value.shape),'dtype':value.dtype.str},sort_keys=True).encode()
    return hashlib.sha256(header+value.tobytes()).hexdigest()


def write(path, body):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(body,handle,ensure_ascii=False,indent=2,allow_nan=False)


def near(left,right):
    np.testing.assert_allclose(left,right,rtol=1e-10,atol=1e-10)


def manifest(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def assert_cohort_coverage(requests, records, cohort, allowed_ids=None):
    """Check exact request identities, including a sealed scientific-stop subset."""
    planned = {r['task_id']:r for r in requests if r['cohort']==cohort}
    assert len(planned)==sum(r['cohort']==cohort for r in requests)
    actual = {r['request']['task_id']:r['request'] for r in records}
    assert len(actual)==len(records), 'Duplicate completed request'
    allowed = set(planned) if allowed_ids is None else set(allowed_ids)
    assert allowed.issubset(planned)
    assert set(actual)==allowed, (cohort, sorted(actual), sorted(allowed))
    assert all(actual[i]==planned[i] for i in actual)
    return {'planned':len(planned),'completed':len(actual),'completed_ids':sorted(actual),
            'missing_ids':sorted(set(planned)-set(actual)),'exact_request_coverage':True}


def recompute_gates(records, thresholds):
    """Independent frozen nine-gate arithmetic; never called on partial cohorts."""
    cells = sorted({r['request']['cell_id'] for r in records})
    assert len(records)==48 and len(cells)==24
    assert all(sum(r['request']['cell_id']==c for r in records)==2 for c in cells)
    small = [r for r in records if r['request']['config']['n_train']==100]
    large = [r for r in records if r['request']['config']['n_train']==400]
    strong = [r for r in records if r['request']['config']['snr']==4.]
    assert len(small)==len(large)==24 and len(strong)==16
    mean = lambda rows,a,k: float(np.mean([r['metrics'][a][k] for r in rows]))
    ratio = lambda rows,b='A0': mean(rows,'A2','pmse')/mean(rows,b,'pmse')
    quantile = lambda a: float(np.quantile([r['metrics'][a]['pmse'] for r in small],.95))
    under = lambda a: float(np.mean([r['metrics'][a]['q']<math.prod(r['request']['config']['sparsity']) for r in strong]))
    values = {'n100_pmse_ratio_A2_A0_max':ratio(small),
              'n100_pmse_p95_ratio_A2_A0_max':quantile('A2')/quantile('A0'),
              'overall_pmse_ratio_A2_A0_max':ratio(records),
              'overall_pmse_ratio_A2_A1_max':ratio(records,'A1'),
              'worst_complete_cell_pmse_ratio_A2_A0_max':max(ratio([r for r in records if r['request']['cell_id']==c]) for c in cells),
              'n400_exact_rate_delta_A2_A0_min':mean(large,'A2','exact_recovery')-mean(large,'A0','exact_recovery'),
              'n400_pmse_ratio_A2_A0_max':ratio(large),
              'strong_snr4_q_under_delta_A2_A0_max':under('A2')-under('A0'),
              'measured_program_cost_ratio_A2_A0_max':sum(r['measured_program_wall_seconds']['A2'] for r in records)/sum(r['measured_program_wall_seconds']['A0'] for r in records)}
    assert set(values)==set(thresholds)
    criteria = {k:{'value':v,'threshold':thresholds[k],
                   'passed':bool(v>=thresholds[k] if k.endswith('_min') else v<=thresholds[k])} for k,v in values.items()}
    return {'passed':all(v['passed'] for v in criteria.values()),'criteria':criteria}


def compare_gates(saved, calculated):
    assert saved['passed']==calculated['passed']
    assert set(saved['criteria'])==set(calculated['criteria'])
    for key, expected in calculated['criteria'].items():
        row = saved['criteria'][key]
        assert row['threshold']==expected['threshold'] and row['passed']==expected['passed']
        near(row['value'],expected['value'])


def _csv_check(path, expected, keys):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows = list(csv.DictReader(handle))
    convert_key = lambda row: tuple(str(row[k]) for k in keys)
    actual = {convert_key(r):r for r in rows}
    assert len(actual)==len(rows)==len(expected), path.name
    assert set(actual)=={convert_key(r) for r in expected}, path.name
    for row in expected:
        current = actual[convert_key(row)]
        for key, value in row.items():
            assert key in current, (path.name,key)
            field = current[key]
            if value is None:
                assert field=='', (path.name,key,field)
            elif isinstance(value,(list,tuple,dict)):
                assert json.loads(field)==json.loads(json.dumps(value)), (path.name,key)
            elif isinstance(value,(bool,str)):
                assert field==str(value), (path.name,key)
            else:
                near(float(field),value)
    return len(rows)


def verify_analysis_tables(out, records_by_cohort):
    """Rebuild published descriptive metrics from completed raw cases only."""
    selected, group_rows, pairs, cells, coverage, resources = [], [], [], [], [], []
    for cohort, records in records_by_cohort.items():
        lookup = {}
        for record in records:
            req, summary = record['request'], record['_summary']
            cfg = req['config']
            evidence = record['recovery_evidence']
            coverage.append({'cohort':cohort,'task_id':req['task_id'],'n':cfg['n_train'],'snr':cfg['snr'],
                'truth_shape':evidence['truth_sizes'],
                **{k:evidence[k] for k in ('truth_shape_native_representable','truth_shape_common_representable',
                    'truth_shape_found_full_pool','truth_shape_found_common_pool',
                    'truth_support_found_full_pool','truth_support_found_common_pool')},
                **{a+'_exact':int(record['metrics'][a]['exact_recovery']) for a in ('A0','A1','A2')}})
            resources.append({'cohort':cohort,'task_id':req['task_id'],'n':cfg['n_train'],
                              **summary['counters'],**record['resources']})
            for arm, metric in record['metrics'].items():
                shape = summary['arms'][arm]['sizes']
                fixed_index = summary['common_grid'].index(shape) if shape in summary['common_grid'] else None
                stability = None
                if fixed_index is not None and summary['arms'][arm]['full_candidate_index']!=0:
                    supports = [f['candidates'][fixed_index]['support'] for f in summary['folds']]
                    stability = float(np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for i,left in enumerate(supports) for right in supports[i+1:] for a,b in zip(left,right)]))
                candidates = summary['full_trace']['candidates'] if arm=='A0' else [summary['full_trace']['candidates'][r['full_candidate_index']] for r in summary['cv_curve']]
                limit = max(r['q'] for r in candidates)
                qtrue = math.prod(cfg['sparsity'])
                row = {'cohort':cohort,'task_id':req['task_id'],'cell_id':req['cell_id'],'replicate':req['replicate'],
                       'n':cfg['n_train'],'snr':cfg['snr'],'rho':cfg['rho'][0],'truth_shape':cfg['sparsity'],'arm':arm,
                       **metric,'q_true':qtrue,'q_under':int(metric['q']<qtrue),'q_over':int(metric['q']>qtrue),
                       'q_over_native_Q':metric['q']/math.floor(.7*cfg['n_train']),
                       'selection_max_q':limit,'near_selection_limit':int(metric['q']>=.9*limit),
                       'fixed_program_inner_support_jaccard':stability,
                       'program_wall_seconds':record['measured_program_wall_seconds'][arm],
                       'task_wall_seconds':record['resources']['wall_seconds'],'task_cpu_seconds':record['resources']['cpu_seconds'],
                       'task_peak_memory_bytes':record['resources']['peak_working_set_bytes']}
                selected.append(row); lookup[(req['task_id'],arm)] = row
        cohort_rows = [r for r in selected if r['cohort']==cohort]
        groups = [('all',lambda r:True),('n100',lambda r:r['n']==100),('n400',lambda r:r['n']==400),
                  ('SNR0.5',lambda r:r['snr']==.5),('SNR4',lambda r:r['snr']==4)]
        for name, accepts in groups:
            for arm in ('A0','A1','A2'):
                part = [r for r in cohort_rows if r['arm']==arm and accepts(r)]
                if not part: continue
                values = [r['pmse'] for r in part]
                group_rows.append({'cohort':cohort,'group':name,'arm':arm,'n_cases':len(part),
                    'pmse_mean':float(np.mean(values)),'pmse_median':float(np.median(values)),
                    'pmse_p90':float(np.quantile(values,.9)),'pmse_p95':float(np.quantile(values,.95)),
                    'pmse_max':max(values),'exact_count':sum(int(r['exact_recovery']) for r in part),
                    **{k:float(np.mean([r[k] for r in part])) for k in ('q','q_under','q_over','near_selection_limit','ree','mode_f1_macro','element_f1','rss','program_wall_seconds')}})
        for record in records:
            tid = record['request']['task_id']
            for left,right in (('A0','A1'),('A1','A2'),('A0','A2')):
                a,b = lookup[(tid,left)],lookup[(tid,right)]
                delta = b['pmse']-a['pmse']
                pairs.append({'cohort':cohort,'task_id':tid,'n':a['n'],'snr':a['snr'],'left':left,'right':right,
                    'delta_pmse':delta,'delta_rss':b['rss']-a['rss'],'delta_q':b['q']-a['q'],
                    'pmse_ratio':b['pmse']/a['pmse'],'outcome':'improved' if delta < -1e-12 else 'worse' if delta > 1e-12 else 'equal',
                    'same_shape':a['sizes']==b['sizes'],'same_support':a['support']==b['support'],
                    'same_B':a['B_sha256']==b['B_sha256'],'left_q':a['q'],'right_q':b['q'],
                    'left_pmse':a['pmse'],'right_pmse':b['pmse'],'left_shape':a['sizes'],'right_shape':b['sizes'],
                    'left_support':a['support'],'right_support':b['support']})
        for cell in sorted({r['cell_id'] for r in cohort_rows}):
            part = [r for r in cohort_rows if r['cell_id']==cell]
            means = {a:float(np.mean([r['pmse'] for r in part if r['arm']==a])) for a in ('A0','A1','A2')}
            for arm in ('A0','A1','A2'):
                rows = [r for r in part if r['arm']==arm]
                shapes = sorted({tuple(r['sizes']) for r in rows})
                cells.append({'cohort':cohort,'cell_id':cell,'n':rows[0]['n'],'snr':rows[0]['snr'],
                    'truth_shape':rows[0]['truth_shape'],'rho':rows[0]['rho'],'arm':arm,'replicates':len(rows),
                    'pmse_mean':means[arm],'ratio_vs_A0':means[arm]/means['A0'],
                    'q_mean':float(np.mean([r['q'] for r in rows])),'q_std_descriptive':float(np.std([r['q'] for r in rows])),
                    'distinct_selected_shapes':len(shapes),'selected_shapes':shapes,'exact_count':sum(int(r['exact_recovery']) for r in rows)})
    counts = {'selected_metrics':_csv_check(out/'analysis/selected_metrics.csv',selected,('cohort','task_id','arm')),
              'group_metrics':_csv_check(out/'analysis/group_metrics.csv',group_rows,('cohort','group','arm')),
              'paired_changes':_csv_check(out/'analysis/paired_changes.csv',pairs,('cohort','task_id','left','right')),
              'cell_metrics':_csv_check(out/'analysis/cell_metrics.csv',cells,('cohort','cell_id','arm')),
              'candidate_coverage':_csv_check(out/'analysis/candidate_coverage.csv',coverage,('cohort','task_id')),
              'fit_resources':_csv_check(out/'analysis/fit_resources.csv',resources,('cohort','task_id'))}
    saved_stats = read(out/'analysis/summary.json')['statistics']
    assert len(saved_stats)==len(group_rows)
    by_key = {(r['cohort'],r['group'],r['arm']):r for r in saved_stats}
    for row in group_rows:
        current = by_key[(row['cohort'],row['group'],row['arm'])]
        for k,v in row.items():
            assert current[k]==v if isinstance(v,str) else bool(np.isclose(current[k],v,rtol=1e-10,atol=1e-10))
    return counts


def check_case(root,folder,plan,protocol_path):
    record = read(folder/'complete.json')
    request = record['request']
    assert request in plan['requests']
    assert folder.name==f"task_{request['task_id']:03d}" and folder.parent.name==request['cohort']
    assert read(folder/'request.json')==request
    assert record['plan_sha256'] == sha(folder.parents[1]/'plan.json')
    assert record['protocol_sha256'] == sha(protocol_path)
    for name,digest in record['artifact_sha256'].items():
        assert sha(folder/name) == digest, (folder,name)
    summary = read(folder/'summary.json')
    assert summary == read(folder/'selection_frozen.json')
    assert record['source_identity']==summary['source_identity']==plan['protocol']['source_identity']
    assert record['environment']==plan['protocol']['environment']
    assert read(folder/'independent_audit.json')['passed']
    assert read(folder/'independent_audit.json')['all_candidate_designs_full_rank']
    input_record = read(folder/'input_identity.json')
    assert input_record['duplicate'] is False and input_record['input_fingerprints']==record['input_fingerprints']
    assert summary['input_identity']=={'X':record['input_fingerprints']['X_train'],'y':record['input_fingerprints']['y_train']}
    protocol = plan['protocol']
    source = root/'doc/algorithm_versions/v5.2-search-final/src'
    assert 'tensor_abess.tensor.bcd' in record['actual_core_imports']
    for name,info in record['actual_core_imports'].items():
        path = Path(info['path'])
        assert path.is_relative_to(source) and sha(path) == info['sha256']
        assert protocol['core_sha256'][path.relative_to(source).as_posix()] == info['sha256']
    with np.load(folder/'arrays.npz',allow_pickle=False) as fitted, np.load(folder/'truth.npz',allow_pickle=False) as truth, np.load(folder/'test_predictions.npz',allow_pickle=False) as test:
        true_support = [truth[f'S{i}_true'].tolist() for i in (1,2,3)]
        assert array_sha(truth['B_true'])==record['input_fingerprints']['B_true']
        assert [array_sha(truth[f'S{i}_true']) for i in (1,2,3)]==record['input_fingerprints']['supports_true']
        assert test['prediction'].shape == (request['config']['n_test'],3)
        assert test['y'].shape==test['signal'].shape==(request['config']['n_test'],)
        assert np.isfinite(test['prediction']).all() and np.isfinite(test['y']).all() and np.isfinite(test['signal']).all()
        full_B = fitted['full_B']
        assert len(summary['full_trace']['candidates'])==len(full_B)
        for index,arm in enumerate(('A0','A1','A2')):
            B = fitted[arm+'_B']; metric = record['metrics'][arm]
            assert array_sha(B) == metric['B_sha256'] == summary['arms'][arm]['B_sha256']
            assert metric['support']==summary['arms'][arm]['support']
            np.testing.assert_array_equal(B,full_B[summary['arms'][arm]['full_candidate_index']])
            assert metric['sizes']==list(map(len,metric['support'])) and metric['q']==math.prod(metric['sizes'])
            near(np.linalg.norm(B),metric['coefficient_norm'])
            near(np.linalg.norm(B-truth['B_true'])/np.linalg.norm(truth['B_true']),metric['ree'])
            assert int(metric['exact_recovery']) == int(metric['support']==true_support)
            overlaps = [len(set(a)&set(b)) for a,b in zip(metric['support'],true_support)]
            near(np.mean([2*o/(len(a)+len(b)) for o,a,b in zip(overlaps,metric['support'],true_support)]),metric['mode_f1_macro'])
            near(2*math.prod(overlaps)/(math.prod(map(len,true_support))+metric['q']),metric['element_f1'])
            near(np.mean((test['y']-test['prediction'][:,index])**2),metric['pmse'])
            near(np.mean((test['signal']-test['prediction'][:,index])**2),metric['signal_mse'])
        for candidate,B in zip(summary['full_trace']['candidates'],full_B):
            assert array_sha(B) == candidate['B_sha256'] and np.isfinite(B).all()
        truth_shape = list(map(len,true_support))
        native_rows = summary['full_trace']['candidates']
        common_rows = [native_rows[r['full_candidate_index']] for r in summary['cv_curve']]
        recovered = {'truth_sizes':truth_shape,'truth_support':true_support,
            'q_true':math.prod(truth_shape),'Q_native':math.floor(.7*request['config']['n_train']),
            'Q_common':summary['public_capacity']['common_Q'],
            'truth_shape_native_representable':math.prod(truth_shape)<=math.floor(.7*request['config']['n_train']) and all(s<=cap for s,cap in zip(truth_shape,summary['public_capacity']['full_caps'])),
            'truth_shape_common_representable':truth_shape in summary['common_grid'],
            'truth_shape_found_full_pool':any(r['sizes']==truth_shape for r in native_rows),
            'truth_shape_found_common_pool':any(r['sizes']==truth_shape for r in common_rows),
            'truth_support_found_full_pool':any(r['support']==true_support for r in native_rows),
            'truth_support_found_common_pool':any(r['support']==true_support for r in common_rows),
            'candidate_counts':{'full_including_adaptive':len(native_rows),'common_fixed':len(common_rows)}}
        assert record['recovery_evidence']==recovered
        assert len(summary['folds'])==3
        for fold in summary['folds']:
            inner_B = fitted[fold['B_key']]
            assert len(fold['candidates'])==len(inner_B)==len(summary['common_grid'])
            for candidate,B in zip(fold['candidates'],inner_B):
                assert array_sha(B) == candidate['B_sha256'] and np.isfinite(B).all()
            cache = folder/'cache'/fold['cache_key']
            seal = read(cache/'seal.json')
            assert sha(cache/'pool.json') == seal['metadata_sha256']
            assert sha(cache/'pool.npz') == seal['arrays_sha256']
            cached = read(cache/'pool.json')
            assert cached['cache_key']==fold['cache_key'] and cached['cache_payload']==fold['cache_payload']
            assert cached['candidates']==fold['candidates'] and cached['init_caps']==fold['init_caps']
            assert cached['init_supports']==fold['init_supports']
            assert cached['preprocessing']=={k:v for k,v in fold['preprocessing'].items() if k!='X_mean_key'}
            payload = fold['cache_payload']
            encoded = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            assert hashlib.sha256(encoded).hexdigest()==fold['cache_key']
            assert payload['train_indices']==fold['train_indices'] and payload['validation_indices']==fold['validation_indices']
            assert payload['source_identity']==protocol['source_identity'] and payload['grid']==summary['common_grid']
            with np.load(cache/'pool.npz',allow_pickle=False) as cache_arrays:
                np.testing.assert_array_equal(cache_arrays['B'],inner_B)
    return {'cohort':request['cohort'],'task_id':request['task_id'],
            'input_key':record['input_fingerprints']['X_train']+':'+record['input_fingerprints']['y_train'],
            'full_candidates':len(summary['full_trace']['candidates']),
            'inner_candidates':sum(len(fold['candidates']) for fold in summary['folds']),
            'cache_folds':len(summary['folds']),'passed':True}


def verify_external_audits(root,out,plans,counts):
    """Validate exactly the sealed replay scope; never replaces missing cases."""
    base = out/'external_audit'
    scope_path = base/'SCOPE.json'
    scope = read(scope_path)
    assert scope['case_limit']==28 and len(scope['cases'])==28
    assert scope['script_sha256']==sha(root/'research/taskcard10/verify_external.py')
    assert scope['test_sha256']==sha(root/'research/taskcard10/test_verify_external.py')
    for relative,digest in scope['plan_sha256'].items():
        assert sha(root/relative)==digest
    expected_cases = []
    for cohort in ('pilot','development','confirmation'):
        plan_path = out/('results/pilot/plan.json' if cohort=='pilot' else 'results/formal/plan.json')
        rows = [r for r in plans['pilot' if cohort=='pilot' else 'formal']['requests'] if r['cohort']==cohort]
        chosen = rows if cohort=='pilot' else [r for r in rows if r['replicate']==0 and r['config']['rho']==[0.,0.,0.]]
        assert len(chosen)==(4 if cohort=='pilot' else 12)
        for request in chosen:
            expected_cases.append({'cohort':cohort,'task_id':request['task_id'],
                'folder':(plan_path.parent/cohort/f"task_{request['task_id']:03d}").relative_to(root).as_posix(),
                'request':request,'plan':plan_path.relative_to(root).as_posix(),'plan_sha256':sha(plan_path)})
    assert scope['cases']==expected_cases
    results = {}
    for cohort in ('pilot','development','confirmation'):
        cases = [c for c in scope['cases'] if c['cohort']==cohort]
        summary_path = base/cohort/'SUMMARY.json'
        if not summary_path.exists():
            assert cohort!='pilot' and not (base/cohort).exists()
            reason_path = base/(cohort+'_NOT_RUN.json')
            if not reason_path.exists():
                assert cohort=='confirmation'
                reason_path = out/'results/formal/confirmation_NOT_RUN.json'
            reason = read(reason_path)
            assert reason['status']=='NOT_RUN' and bool(reason.get('reason'))
            if 'planned_task_ids' in reason:
                assert reason['planned_task_ids']==[c['task_id'] for c in cases]
            completed_selected = [c['task_id'] for c in cases if (root/c['folder']/'complete.json').is_file()]
            missing_selected = [c['task_id'] for c in cases if c['task_id'] not in completed_selected]
            if 'completed_selected_task_ids' in reason:
                assert reason['completed_selected_task_ids']==completed_selected
            if 'missing_completed_task_ids' in reason:
                assert reason['missing_completed_task_ids']==missing_selected
            assert reason.get('regenerated_cases',0)==0
            assert counts[cohort]==0 or missing_selected, 'Completed frozen replay scope cannot be silently omitted'
            results[cohort] = {'status':'NOT_RUN','planned_cases':len(cases),'reason_path':reason_path.relative_to(root).as_posix()}
            continue
        summary = read(summary_path)
        assert counts[cohort]>0
        assert summary['passed'] and summary['cohort']==cohort
        assert summary['cases_checked']==len(cases) and summary['task_ids']==[c['task_id'] for c in cases]
        assert summary['selected_models_checked']==3*len(cases)
        assert summary['source_fingerprint']==scope['script_sha256'] and summary['scope_sha256']==sha(scope_path)
        assert summary['search_fits']==summary['selection_changes']==0
        actual_paths = sorted((base/cohort).glob('task_*.json'))
        assert {p.name for p in actual_paths}=={f"task_{c['task_id']:03d}.json" for c in cases}
        replay_rows = []
        for case in cases:
            replay = read(base/cohort/f"task_{case['task_id']:03d}.json")
            original = root/case['folder']
            complete = read(original/'complete.json')
            assert replay['passed'] and replay['request']==complete['request']==case['request']
            assert replay['cohort']==cohort and replay['task_id']==case['task_id']
            assert replay['complete_sha256']==sha(original/'complete.json')
            for field in ('source_and_artifact_hashes_exact','training_fingerprints_exact','truth_arrays_exact',
                          'saved_external_y_signal_exact','test_fingerprint_exact','selected_B_identities_exact'):
                assert replay[field] is True
            assert replay['test_fingerprint']==complete['test_fingerprint']
            assert replay['search_fits']==replay['selection_changes']==0
            assert replay['maximum_prediction_absolute_error']<=1e-8
            assert replay['maximum_PMSE_signal_MSE_absolute_error']<=1e-8
            for arm in ('A0','A1','A2'):
                for field in ('pmse','signal_mse'):
                    near(replay['metrics'][arm][field],complete['metrics'][arm][field])
            replay_rows.append(replay)
        near(summary['maximum_prediction_absolute_error'],max(r['maximum_prediction_absolute_error'] for r in replay_rows))
        near(summary['maximum_PMSE_signal_MSE_absolute_error'],max(r['maximum_PMSE_signal_MSE_absolute_error'] for r in replay_rows))
        results[cohort] = {'status':'PASS','cases_checked':len(cases),'selected_models_checked':3*len(cases),
                           'maximum_prediction_absolute_error':summary['maximum_prediction_absolute_error'],
                           'maximum_PMSE_signal_MSE_absolute_error':summary['maximum_PMSE_signal_MSE_absolute_error']}
    return {'scope_sha256':sha(scope_path),'maximum_scope':28,'cohorts':results}


def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'doc/taskcard10'
    assert not (out/'FINAL_VALIDATION.json').exists()
    checks, plans = [], {}
    records_by_cohort = {c:[] for c in ('pilot','development','confirmation')}
    for stage in ('pilot','formal'):
        plan = read(out/f'results/{stage}/plan.json')
        plans[stage] = plan
        protocol_path = root/plan['protocol_path']
        assert sha(protocol_path) == plan['protocol_sha256']
        for relative,digest in plan['protocol']['source_identity'].items():
            assert sha(root/relative) == digest
        cohort_directories = sorted({request['cohort'] for request in plan['requests']})
        case_paths = [path for cohort in cohort_directories
                      for path in (out/f'results/{stage}'/cohort).glob('task_*/complete.json')]
        for path in sorted(case_paths):
            checks.append(check_case(root,path.parent,plan,protocol_path))
            record = read(path)
            record['_summary'] = read(path.parent/'summary.json')
            records_by_cohort[record['request']['cohort']].append(record)
    keys = [r['input_key'] for r in checks]
    assert len(keys) == len(set(keys))
    counts = {c:len(rows) for c,rows in records_by_cohort.items()}
    analysis_summary = read(out/'analysis/summary.json')
    assert analysis_summary['counts']==counts and len(checks)==sum(counts.values())
    formal_out = out/'results/formal'
    stopped_path = formal_out/'stopped_development.json'
    stopped = read(stopped_path) if stopped_path.exists() else None
    coverage = {'pilot':assert_cohort_coverage(plans['pilot']['requests'],records_by_cohort['pilot'],'pilot')}
    assert coverage['pilot']['planned']==4
    intent = read(out/'GATE_INTENT.json')['gates']
    protocol = plans['formal']['protocol']
    assert len(intent)==9 and protocol['development_entry_gate']==protocol['promotion_gates']==intent
    confirmation_gate = {'status':'NOT_RUN','passed':None}
    lock_path = formal_out/'development_lock.json'
    if stopped is not None:
        assert stopped['status']!='COMPLETE'
        completed_ids, failed_ids, not_started = set(stopped['completed']),{r['task_id'] for r in stopped['failed']},set(stopped['not_started'])
        planned_ids = {r['task_id'] for r in plans['formal']['requests'] if r['cohort']=='development'}
        assert len(planned_ids)==48
        assert len(completed_ids)==len(stopped['completed']) and len(failed_ids)==len(stopped['failed'])
        assert len(not_started)==len(stopped['not_started'])
        assert not (completed_ids&failed_ids or completed_ids&not_started or failed_ids&not_started)
        assert completed_ids|failed_ids|not_started==planned_ids and len(completed_ids)<len(planned_ids)
        for failure in stopped['failed']:
            if failure['reason']=='case_timeout':
                assert failure['elapsed_seconds']>=protocol['formal']['case_timeout_seconds']
        coverage['development'] = assert_cohort_coverage(plans['formal']['requests'],records_by_cohort['development'],'development',completed_ids)
        assert not lock_path.exists() and not (formal_out/'confirmation_gate.json').exists()
        assert not (formal_out/'confirmation').exists() and counts['confirmation']==0
        development_gate = {'status':'NOT_EVALUABLE_INCOMPLETE','passed':None,
                            'reason':'Scientific development stopped before exact full planned coverage; nine gates not computed'}
        saved_gate = analysis_summary.get('development_gate')
        assert saved_gate is None or (isinstance(saved_gate,dict) and saved_gate.get('passed') is not True and not saved_gate.get('criteria'))
        assert analysis_summary['status']=='STOPPED_PARTIAL_DEVELOPMENT'
        assert analysis_summary['planned_development']==len(planned_ids) and analysis_summary['stopped_development']==stopped
        scientific_status = 'STOPPED_INCOMPLETE'
    else:
        coverage['development'] = assert_cohort_coverage(plans['formal']['requests'],records_by_cohort['development'],'development')
        lock = read(lock_path)
        assert lock['plan_sha256']==sha(formal_out/'plan.json')
        assert lock['completed_development_tasks']==counts['development']==48
        assert lock['development_artifacts']==manifest(formal_out/'development')
        development_gate = recompute_gates(records_by_cohort['development'],intent)
        compare_gates(lock['entry_gate'],development_gate)
        compare_gates(analysis_summary['development_gate'],development_gate)
        scientific_status = 'COMPLETE'
    coverage['confirmation'] = assert_cohort_coverage(plans['formal']['requests'],records_by_cohort['confirmation'],'confirmation',
        [] if counts['confirmation']==0 else None)
    assert coverage['confirmation']['planned']==48
    if counts['confirmation']==0:
        reason = read(formal_out/'confirmation_NOT_RUN.json')
        assert reason['status']=='NOT_RUN' and bool(reason.get('reason'))
        assert not (formal_out/'confirmation').exists() and not (formal_out/'confirmation_gate.json').exists()
        assert analysis_summary.get('confirmation_gate') is None or analysis_summary['confirmation_gate'].get('status')=='NOT_RUN'
    else:
        assert stopped is None and development_gate['passed']
        assert not (formal_out/'confirmation_NOT_RUN.json').exists()
        budget_gate = read(formal_out/'confirmation_budget_gate.json')
        assert budget_gate['passed'] and budget_gate['remaining_seconds']>=budget_gate['required_seconds']
        assert all(r['development_lock_sha256']==sha(lock_path) for r in records_by_cohort['confirmation'])
        confirmation_gate = recompute_gates(records_by_cohort['confirmation'],intent)
        compare_gates(read(formal_out/'confirmation_gate.json'),confirmation_gate)
        compare_gates(analysis_summary['confirmation_gate'],confirmation_gate)
    analysis_checks = verify_analysis_tables(out,records_by_cohort)
    external_checks = verify_external_audits(root,out,plans,counts)
    original = read(out/'GIT_INITIAL.json')
    current = subprocess.run(['git','status','--porcelain','-z','--untracked-files=no'],cwd=root,capture_output=True,check=True).stdout.decode('utf-8').split('\0')
    assert current == original
    protection = read(out/'PROTECTED_FINAL_CHECK.json')
    assert protection['passed'] and protection['checked'] == 5922
    gates = read(out/'CORRECTNESS_GATES.json')
    assert gates['passed'] and gates['tests_run'] == 46
    reports = ['DESIGN_AND_PROTOCOL.md','CV_LEAKAGE_AND_CORRECTNESS.md','PILOT_FEASIBILITY_REPORT.md',
               'FROZEN_EXPERIMENT_PLAN.md','DEVELOPMENT_RESULTS.md','INDEPENDENT_CONFIRMATION.md',
               'SELECTION_MECHANISM_ANALYSIS.md','SECONDARY_RESEARCH_PLAN.md','TASKCARD10_FINAL_DECISION.md']
    assert all((out/name).is_file() for name in reports)
    links = 0
    for path in out.rglob('*.md'):
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):
                continue
            destination = (path.parent/target.split('#')[0].strip('<>')).resolve()
            assert destination.is_file() or destination in (out/'FINAL_VALIDATION.json',out/'DELIVERY_INDEX.json'), (path.name,target)
            links += 1
    for path in (root/'research/taskcard10').glob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    body = {'status':'COMPLETE_DIAGNOSIS_WITH_SCIENTIFIC_STAGE_STOPPED' if stopped is not None else 'COMPLETE_RESEARCH_WITH_DECLARED_GATE_RESULT',
            'checked_utc':datetime.now(timezone.utc).isoformat(),
            'research_only':True,'release_approved':False,'formal_version':'v5.2-search-final',
            'correctness_tests':46,'original_cold_checks':16,'protected_files_unchanged':5922,
            'counts':counts,'exact_plan_and_stop_coverage':coverage,'scientific_development_status':scientific_status,
            'scientific_stop':stopped,'independently_verified_cases':checks,
            'all_training_input_keys_unique':True,'saved_selected_test_predictions_and_REE_recomputed':True,
            'full_and_inner_coefficient_hash_checks':sum(r['full_candidates']+r['inner_candidates'] for r in checks),
            'cache_folds_final_seals_verified':sum(r['cache_folds'] for r in checks),
            'development_entry_gate':development_gate,'confirmation_gate':confirmation_gate,
            'numeric_gate_reconstruction':'NOT_RUN_PARTIAL_DEVELOPMENT' if stopped is not None else 'PASS_FROM_COMPLETE_RAW_CASES',
            'confirmation_executed':counts['confirmation']>0,
            'analysis_csv_and_summary_reconstruction':analysis_checks,'bounded_external_replay_validation':external_checks,
            'reports':reports,'relative_links_checked':links,'production_changed':False,
            'not_run':['Warm performance or combination','Ridge replacement','LEMON OOF or q0 simulated arm','Algorithm release']+
                (['Remaining development after sealed stop','Complete nine-gate scientific evaluation','Independent confirmation',
                  'Fixed12 development and confirmation external replay'] if stopped is not None else []),
            'limits':['A1-A0 combines capacity/range and adaptive-source removal; not unique capacity causality',
                      'A1/A2 have one full-training candidate per shape: same-shape source ranking not identifiable',
                      'Few repetitions per scenario; descriptive gates are not high-power inferential guarantees',
                      'Chosen CV risk is selection-optimistic and is never substituted for external test error']}
    write(out/'FINAL_VALIDATION.json',body)
    paths = [p for directory in (out,root/'research/taskcard10') for p in directory.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'DELIVERY_INDEX.json']
    write(out/'DELIVERY_INDEX.json',{'scope':'Task10 original logs, protocols, results, reports and isolated code; index excludes itself',
          'files':{p.relative_to(root).as_posix():{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)},'file_count':len(paths)})
    print(json.dumps({'status':body['status'],'cases':counts,'links':links,'files':len(paths)}))


if __name__ == '__main__':
    main()
