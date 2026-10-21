"""Passive training-only replay of frozen v5/v5.2 LEMON candidate selection.

Representative folds use historical training-selected q only. Reuse v5.1
traces from Phase A, never re-fit v5.1, and never compute outer-test scores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from scripts.taskcard7 import lemon


def select_representatives(old_rows, new_rows):
    def mapping(rows):
        return {(int(r['repeat']), int(r['outer_fold'])): int(r['q'])
                for r in rows if r['model'] == 'Tensor-ABESS'}
    a, b = mapping(old_rows), mapping(new_rows)
    lemon.require(set(a) == set(b), 'Historical training q coverage differs')
    selected = {key for key in a if a[key] == 80 or b[key] == 80}
    controls = sorted(key for key in a if a[key] < 80 and b[key] < 80)
    lemon.require(bool(controls), 'Missing prespecified below-80 control fold')
    return sorted(selected | {controls[0]})


def decompose(old, new, n):
    fit_delta = float(n * np.log(new['rss'] / old['rss']))
    old_penalty = float(old['msic'] - n*np.log(old['rss']/n))
    new_penalty = float(new['msic'] - n*np.log(new['rss']/n))
    penalty_delta = new_penalty - old_penalty
    msic_delta = float(new['msic'] - old['msic'])
    error = msic_delta - fit_delta - penalty_delta
    same_shape = old['sizes'] == new['sizes']
    lemon.require(abs(error) < 1e-9, 'MSIC decomposition does not close')
    lemon.require(not same_shape or abs(penalty_delta) < 1e-9,
                  'Same-shape MSIC penalty changed')
    return {'rss_old': old['rss'], 'rss_new': new['rss'],
            'sizes_old': old['sizes'], 'sizes_new': new['sizes'],
            'same_shape': same_shape, 'msic_old': old['msic'], 'msic_new': new['msic'],
            'msic_delta': msic_delta, 'fit_delta': fit_delta,
            'penalty_old': old_penalty, 'penalty_new': new_penalty,
            'penalty_delta': penalty_delta, 'decomposition_error': error}


def compare_candidates(first, second, first_B, second_B):
    checks = {'candidate_records_exact': first['candidates'] == second['candidates'],
              'candidate_B_exact': bool(np.array_equal(first_B, second_B))}
    for key in ('initial_support', 'caps', 'ordered_size_grid', 'adaptive_terminal',
                'one_se_tolerance', 'minimum_msic', 'minimum_msic_q',
                'eligible_candidate_indices', 'final', 'fixed_searches'):
        checks[key + '_exact'] = first[key] == second[key]
    checks['all_equal'] = all(checks.values())
    return checks


def make_protocol(audit, runs):
    representatives = select_representatives(runs['v5-original']['folds'],
                                             runs['v5.2-search-final']['folds'])
    return {'schema_version': 1, 'phase': 'C2', 'research_only': True,
            'representative_rule': 'all historical v5 OR v5.2 training-selected q=80 folds plus lexicographically first fold with both q<80; no outer-test outcome used',
            'representatives': [list(k) for k in representatives],
            'replay_versions': ['v5-original', 'v5.2-search-final'],
            'reuse_version': 'v5.1-bcd-fix', 'outer_test_used': False,
            'worker_sha256': lemon.sha(__file__), 'audit': audit,
            'v51_phase_A_plan_sha256': lemon.sha(lemon.root() / 'numerical_simulation/outputs_version_attribution/lemon/plan.json')}


def worker(output, version, repeat, fold):
    source = lemon.configure_core(version)
    audit, data, splits, prepared, runs = lemon.audit()
    protocol = make_protocol(audit, runs)
    lemon.require(protocol == lemon.read(output / 'protocol.json'), 'Mechanism protocol identity drift')
    lemon.require([repeat, fold] in protocol['representatives'], 'Unplanned representative fold')
    split = next(s for s in splits if (s['repeat'], s['outer_fold']) == (repeat, fold))
    train = split['train']
    p = prepared[(repeat, fold)]
    X, y = p.apply(data.X[train], data.C[train]), p.transform_response(data.y[train], data.C[train])
    folder = output / version / f'repeat_{repeat}_fold_{fold}'
    lemon.require(not folder.exists(), 'Mechanism replay output exists')
    from empirical_analysis.lemon.incremental_value.model_adapters import fit_eeg_model
    started = perf_counter()
    fitted, trace, coefficients = lemon.observe_fit(
        lambda: fit_eeg_model('Tensor-ABESS', X, y, None, audit['config']['model']))
    runtime = perf_counter() - started
    actual_imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'tensor_abess' or name.startswith('tensor_abess.'):
            actual = Path(module.__file__).resolve()
            lemon.require(actual.is_relative_to(source), 'Core import escaped frozen version')
            key = actual.relative_to(source).as_posix()
            lemon.require(lemon.sha(actual) == audit['source_manifests'][version][key], 'Actual core SHA drift')
            actual_imports[name] = {'path': str(actual), 'sha256': lemon.sha(actual)}
    historical = runs[version]
    hist_dir = historical['directory'] / 'cv/folds' / f'repeat_{repeat}_fold_{fold}' / 'Tensor-ABESS'
    row = next(r for r in historical['folds'] if r['model'] == 'Tensor-ABESS'
               and int(r['repeat']) == repeat and int(r['outer_fold']) == fold)
    B = np.load(hist_dir / 'B.npy', allow_pickle=False)
    support = [s.tolist() for s in fitted.supports]
    historical_score = json.loads(row['selected_parameters'])['sic']
    lemon.require(np.array_equal(fitted.B, B), 'Replay B does not exactly match historical saved B')
    lemon.require(support == lemon.read(hist_dir / 'support.json'), 'Replay support differs from historical')
    lemon.require(fitted.parameters['sic'] == historical_score, 'Replay MSIC differs from historical')
    lemon.require(trace['final']['B_sha256'] == lemon.array_sha(B), 'Trace final B differs from historical')
    folder.mkdir(parents=True)
    np.save(folder / 'B.npy', fitted.B, allow_pickle=False)
    np.savez_compressed(folder / 'candidate_coefficients.npz', B=np.stack(coefficients))
    lemon.write(folder / 'trace.json', trace)
    lemon.write(folder / 'complete.json', {'protocol_sha256': lemon.sha(output / 'protocol.json'),
        'version': version, 'repeat': repeat, 'outer_fold': fold, 'runtime_sec': runtime,
        'actual_imports': actual_imports, 'historical_B_exact': True,
        'historical_support_exact': True, 'historical_msic_exact': True,
        'train_subject_ids': data.subject_ids[train].tolist(),
        'preprocessing_array_sha256': {k: lemon.array_sha(v) for k,v in p.to_arrays().items()},
        'artifact_sha256': {f.name: lemon.sha(f) for f in sorted(folder.iterdir()) if f.is_file()}})
    print('REPLAY_COMPLETE', version, repeat, fold, 'q', trace['final']['q'], flush=True)


def trace_rank_summary(version, repeat, fold, trace):
    candidates = trace['candidates']
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i]['msic'], i))
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    q80 = [{'candidate_index': i, 'rank': ranks[i], 'rss': r['rss'], 'msic': r['msic'],
            'score_above_minimum': r['msic'] - trace['minimum_msic'],
            'eligible': i in trace['eligible_candidate_indices']}
           for i,r in enumerate(candidates) if r['q'] == 80]
    return {'version': version, 'repeat': repeat, 'outer_fold': fold,
            'initial_sizes': list(map(len,trace['initial_support'])), 'caps': trace['caps'],
            'adaptive_terminal_q': trace['adaptive_terminal']['q'],
            'minimum_msic': trace['minimum_msic'], 'minimum_msic_q': trace['minimum_msic_q'],
            'one_se_tolerance': trace['one_se_tolerance'],
            'eligible_q': [candidates[i]['q'] for i in trace['eligible_candidate_indices']],
            'final_q': trace['final']['q'], 'final_msic': trace['final']['msic'],
            'q80_candidates': q80}


def summarize(output, protocol):
    decompositions, summaries, comparisons = [], [], []
    reused = {}
    for repeat, fold in protocol['representatives']:
        tag = f'repeat_{repeat}_fold_{fold}'
        traces, arrays = {}, {}
        for version in ('v5-original', 'v5.1-bcd-fix', 'v5.2-search-final'):
            folder = (lemon.root() / 'numerical_simulation/outputs_version_attribution/lemon/folds' / tag
                      if version == 'v5.1-bcd-fix' else output / version / tag)
            complete = lemon.read(folder / 'complete.json')
            if version == 'v5.1-bcd-fix':
                lemon.require(complete['plan_sha256'] == protocol['v51_phase_A_plan_sha256'], 'v5.1 trace provenance drift')
                reused[tag] = {name: lemon.sha(folder / name) for name in
                    ('trace.json','candidate_coefficients.npz','complete.json')}
            else:
                lemon.require(complete['protocol_sha256'] == lemon.sha(output / 'protocol.json'), 'Replay completion identity drift')
            traces[version] = lemon.read(folder / 'trace.json')
            with np.load(folder / 'candidate_coefficients.npz', allow_pickle=False) as saved:
                arrays[version] = saved['B'].copy()
            lemon.require(len(arrays[version]) == len(traces[version]['candidates']), 'Candidate coefficient count mismatch')
            for record,B in zip(traces[version]['candidates'], arrays[version]):
                lemon.require(record['B_sha256'] == lemon.array_sha(B), 'Candidate B checksum mismatch')
            summaries.append(trace_rank_summary(version,repeat,fold,traces[version]))
        exact = compare_candidates(traces['v5.1-bcd-fix'], traces['v5.2-search-final'],
                                   arrays['v5.1-bcd-fix'],arrays['v5.2-search-final'])
        lemon.require(exact['all_equal'], 'v5.1/v5.2 actual candidate path differs')
        comparisons.append(dict(exact,repeat=repeat,outer_fold=fold,
                                n_candidates=len(traces['v5.1-bcd-fix']['candidates'])))
        for older,newer in [('v5-original','v5.1-bcd-fix'),('v5.1-bcd-fix','v5.2-search-final')]:
            a,b=traces[older],traces[newer]
            lemon.require(a['ordered_size_grid'] == b['ordered_size_grid'], 'Candidate grid order differs')
            for index,(first,second) in enumerate(zip(a['candidates'],b['candidates'])):
                row=decompose(first,second,116)
                row.update(repeat=repeat,outer_fold=fold,old_version=older,new_version=newer,
                           candidate_index=index,candidate_type='adaptive_terminal' if index==0 else 'fixed_grid',
                           q_old=first['q'],q_new=second['q'],
                           support_exact=first['support']==second['support'],
                           B_exact=bool(np.array_equal(arrays[older][index],arrays[newer][index])))
                decompositions.append(row)
    lemon.write(output/'v51_reused_artifacts.json',reused)
    lemon.write(output/'selection_summary.json',summaries)
    lemon.write(output/'candidate_comparisons.json',comparisons)
    lemon.write_csv(output/'candidate_msic_decomposition.csv',[
        {k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in decompositions])
    lemon.write(output/'completion.json',{'status':'complete',
        'protocol_sha256':lemon.sha(output/'protocol.json'),
        'n_representatives':len(protocol['representatives']),
        'v5_v52_replay_fits':2*len(protocol['representatives']), 'v51_refits':0,
        'v51_v52_all_candidates_exact':all(c['all_equal'] for c in comparisons),
        'artifact_sha256':{p.relative_to(output).as_posix():lemon.sha(p)
            for p in sorted(output.rglob('*')) if p.is_file()},
        'outer_test_predictions_computed':False})
    print('MECHANISM_COMPLETE',len(comparisons),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','worker'])
    parser.add_argument('--version',choices=['v5-original','v5.2-search-final'])
    parser.add_argument('--repeat',type=int)
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    output=lemon.root()/'numerical_simulation/outputs_version_attribution/mechanism'
    if args.mode=='worker':
        worker(output,args.version,args.repeat,args.fold)
        return
    lemon.configure_core('v5.1-bcd-fix')
    audit,_,_,_,runs=lemon.audit()
    protocol=make_protocol(audit,runs)
    if args.mode=='prepare':
        lemon.require(not output.exists(),'Mechanism output exists')
        output.mkdir(parents=True)
        lemon.write(output/'protocol.json',protocol)
        print('PROTOCOL_FROZEN',protocol['representatives'],lemon.sha(output/'protocol.json'),flush=True)
        return
    lemon.require(protocol==lemon.read(output/'protocol.json'),'Frozen mechanism protocol drift')
    logs=output/'logs'
    logs.mkdir(exist_ok=False)
    for repeat,fold in protocol['representatives']:
        for version in protocol['replay_versions']:
            path=logs/f'{version}_repeat_{repeat}_fold_{fold}.txt'
            command=[sys.executable,'-B','-m','scripts.taskcard7.mechanism','worker',
                     '--version',version,'--repeat',str(repeat),'--fold',str(fold)]
            with path.open('x',encoding='utf-8') as handle:
                result=subprocess.run(command,cwd=lemon.root(),stdout=handle,stderr=subprocess.STDOUT,check=False)
            lemon.require(result.returncode==0,'Isolated replay failed: '+str(path))
            print('COMPLETE',version,repeat,fold,flush=True)
    summarize(output,protocol)


if __name__=='__main__':
    main()
