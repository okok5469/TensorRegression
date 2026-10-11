"""Interpret completed paired study and audit fixed-candidate correctness."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyse(output):
    output=Path(output)
    integrity=json.loads((output/'integrity_report.json').read_text(encoding='utf-8'))
    if not integrity['integrity_checks_passed'] or integrity['completed_jobs']!=54:
        raise ValueError('Only complete three-replicate study may be interpreted')
    raw=read_csv(output/'raw_results.csv');pairs=read_csv(output/'paired_default_diff.csv')
    metrics=('q_hat','exact_recovery','mode_f1_macro','element_f1','pmse','r2_test',
             'q_over','q_under','q_small','q_near_cap')
    by_rule=[]
    for subset,selected in [('all_27_datasets',raw),('matched_n200_9_datasets',[r for r in raw if int(r['n_train'])==200])]:
        for method in sorted({r['method'] for r in selected}):
            group=[r for r in selected if r['method']==method]
            row={'subset':subset,'method':method,'n_datasets':len(group)}
            row.update({k:float(np.mean([float(r[k]) for r in group])) for k in metrics})
            by_rule.append(row)
    write_csv(output/'rule_overview.csv',by_rule)
    feasibility=[];candidate_count=0
    for path in sorted((output/'jobs').glob('*__v5.1-bcd-fix.json')):
        report=json.loads(path.read_text(encoding='utf-8'));caps=report['trace']['s_max'];truth=report['config']['sparsity']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as arrays:
            Btrue=arrays['B_true'];true_supports=[np.flatnonzero(np.any(Btrue!=0,axis=tuple(j for j in range(3) if j!=i))).tolist() for i in range(3)]
        pool=report['candidates']
        for candidate in pool:
            if candidate['origin']!='global-fixed':
                continue
            visited=candidate['visited_loss_history'];accepted=candidate['loss_history'];n=report['config']['n_train']
            if not visited or abs(candidate['loss']-min(visited))>1e-12 or abs(candidate['rss']-2*n*candidate['loss'])>1e-10:
                raise AssertionError('Fixed candidate is not consistent with best visited state')
            if np.any(np.diff(accepted)>1e-12):
                raise AssertionError('Accepted fixed candidate loss increased')
            if abs(visited[candidate['best_iteration']]-candidate['loss'])>1e-12:
                raise AssertionError('Best-iteration metadata inconsistent')
            candidate_count+=1
        feasibility.append({'job':path.stem,'n_train':report['config']['n_train'],'snr':report['config']['snr'],
            'seed':report['seed'],'caps':json.dumps(caps),'capacity_product':int(np.prod(caps)),
            'true_size_within_caps':int(all(a<=b for a,b in zip(truth,caps))),
            'true_size_present_in_pool':int(any(list(c['sizes'])==truth for c in pool)),
            'exact_true_support_present_in_pool':int(any(c['supports']==true_supports for c in pool)),
            'pool_count':len(pool)})
    write_csv(output/'candidate_feasibility.csv',feasibility)
    pair_summary={'paired_datasets':len(pairs),'selected_B_changed':sum(int(r['selected_B_changed']) for r in pairs)}
    for metric in metrics:
        key=metric+'_fixed_minus_original';values=[float(r[key]) for r in pairs]
        pair_summary[key+'_mean']=float(np.mean(values))
        if metric=='pmse':
            pair_summary['fixed_pmse_wins_ties_losses']=[sum(v< -1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v>1e-12 for v in values)]
    diagnostics={'complete':True,'paired_default':pair_summary,'fixed_candidates_checked':candidate_count,
        'fixed_candidates_best_visited_and_accepted_monotone':True,
        'true_size_within_caps_datasets':sum(r['true_size_within_caps'] for r in feasibility),
        'true_size_present_in_pool_datasets':sum(r['true_size_present_in_pool'] for r in feasibility),
        'exact_true_support_present_in_pool_datasets':sum(r['exact_true_support_present_in_pool'] for r in feasibility),
        'interpretation':'3 replicates/scenario are exploratory; CV comparable only on matched n200 subset; different fixed shapes confound size-only interpretation'}
    (output/'study_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    analyse(parser.parse_args().output_dir)
