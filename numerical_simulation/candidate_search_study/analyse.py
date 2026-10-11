"""Balanced descriptive comparisons; shared cache costs are explicitly separated."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def dataset_rows(report,groups):
    rows=report['rows']
    if len(rows)!=len(groups) or sorted(r['group'] for r in rows)!=sorted(groups):
        raise ValueError('Every dataset must contain exactly the same one-rule groups')
    config=report['config'];truth=config['sparsity'];qtrue=int(np.prod(truth));qmax=int(np.floor(.7*config['n_train']))
    extended=[]
    for raw in rows:
        row=dict(raw);info=report['groups'][row['group']];meta=info['search_metadata']
        fixed=[c for c in info['candidates'] if c['kind']=='global-fixed']
        cached_cost=sum(c['fit_seconds'] for c in fixed if c['cache_hit'])
        row.update(q_over=float(row['q_hat']>qtrue),q_under=float(row['q_hat']<qtrue),
            q_small=float(row['q_hat']==1),q_near_cap=float(row['q_hat']>=.9*qmax),
            true_size_reachable=float(all(s<=cap for s,cap in zip(truth,info['s_max']))),
            true_size_in_pool=float(info['true_size_in_pool']),true_support_in_pool=float(info['true_support_in_pool']),
            candidate_count=info['candidate_count'],legal_shape_count=len(meta['legal_shapes']),
            searched_shape_count=len(meta['size_grid']),excluded_legal_shape_count=len(meta['excluded_legal_shapes']),
            cached_fixed_fit_cost_sec=cached_cost,standalone_cost_estimate_sec=row['runtime_sec']+cached_cost,
            fixed_fit_cost_sum_sec=sum(c['fit_seconds'] for c in fixed),
            n_train=config['n_train'],snr=config['snr'])
        extended.append(row)
    return extended


def build_overview(reports,groups):
    if not reports:raise ValueError('No complete datasets')
    rows=[r for report in reports for r in dataset_rows(report,groups)]
    metrics=('q_hat','exact_recovery','mode_f1_macro','element_f1','pmse','r2_test','rss_train','runtime_sec',
        'q_over','q_under','q_small','q_near_cap','true_size_reachable','true_size_in_pool','true_support_in_pool',
        'candidate_count','legal_shape_count','searched_shape_count','excluded_legal_shape_count',
        'cached_fixed_fit_cost_sec','standalone_cost_estimate_sec','fixed_fit_cost_sum_sec')
    output=[]
    for group in groups:
        selected=[r for r in rows if r['group']==group];record={'group':group,'n_datasets':len(selected)}
        for metric in metrics:
            values=np.asarray([float(r[metric]) for r in selected])
            if not np.isfinite(values).all():raise ValueError('Nonfinite comparison metric')
            record[metric+'_mean']=float(values.mean())
            record[metric+'_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else 0.
        output.append(record)
    return output


def build_pairs(reports,groups):
    if 'A0' not in groups:raise ValueError('A0 paired comparator required')
    metrics=('q_hat','exact_recovery','mode_f1_macro','element_f1','pmse','r2_test','rss_train',
             'true_size_in_pool','true_support_in_pool','standalone_cost_estimate_sec')
    output=[]
    for group in groups:
        if group=='A0':continue
        row={'group':group,'n_datasets':len(reports)}
        differences={m:[] for m in metrics}
        for report in reports:
            by_group={r['group']:r for r in dataset_rows(report,groups)}
            for metric in metrics:differences[metric].append(float(by_group[group][metric])-float(by_group['A0'][metric]))
        for metric,values in differences.items():
            row[metric+'_minus_A0_mean']=float(np.mean(values))
            row[metric+'_minus_A0_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        values=differences['pmse']
        row.update(pmse_wins=sum(v< -1e-10 for v in values),pmse_ties=sum(abs(v)<=1e-10 for v in values),
                   pmse_losses=sum(v>1e-10 for v in values))
        output.append(row)
    return output


def write_csv(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyse(output):
    from numerical_simulation.candidate_search_study.run import load_plan,validate_result
    output=Path(output);plan=load_plan(output)
    integrity=json.loads((output/'integrity_report.json').read_text(encoding='utf8'))
    if not integrity['integrity_checks_passed'] or integrity['completed_tasks']!=len(plan['tasks']):
        raise ValueError('Only complete audited cohorts may be summarized')
    reports=[validate_result(output,plan,t) for t in plan['tasks']]
    overview=build_overview(reports,plan['groups']);pairs=build_pairs(reports,plan['groups'])
    scenarios=[]
    for n,snr in sorted({(r['config']['n_train'],r['config']['snr']) for r in reports}):
        selected=[r for r in reports if r['config']['n_train']==n and r['config']['snr']==snr]
        scenarios.extend(dict(r,n_train=n,snr=snr) for r in build_overview(selected,plan['groups']))
    write_csv(output/'mechanism_overview.csv',overview)
    write_csv(output/'mechanism_scenarios.csv',scenarios)
    write_csv(output/'mechanism_paired_overview.csv',pairs)
    diagnostics={'status':'complete','cohort':plan['cohort'],'plan_id':plan['plan_id'],'datasets':len(reports),
        'groups':plan['groups'],'scope':'Descriptive, balanced paired mechanisms; no automatic formal-version selection',
        'runtime_note':'Observed runtime depends on shared-cache group order. Standalone estimate adds cached fixed fit durations; it is not an independent timing measurement.',
        'fixed_candidates_checked':sum(c['kind']=='global-fixed' for r in reports for g in r['groups'].values() for c in g['candidates'])}
    with (output/'mechanism_diagnostics.json').open('x',encoding='utf8') as f:json.dump(diagnostics,f,indent=2)
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    analyse(parser.parse_args().output_dir)
