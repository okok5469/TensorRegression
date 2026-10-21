"""Descriptive paired decomposition of the returned, already merged 250 tasks."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(__file__).resolve().parent
    base = repo / 'numerical_simulation/outputs_v51_attribution/cluster_run'
    plan = read(base / 'plan.json')
    integrity = read(base / 'integrity.json')
    assert integrity['status'] == 'PASSED' and integrity['completed'] == 250
    with (base / 'paired_three_versions.csv').open(encoding='utf-8', newline='') as handle:
        native = list(csv.DictReader(handle))
    requests = {t['task_id']: t for t in plan['tasks']}
    assert len(native) == len(requests) == 250
    # Names are bound to explicit original DGP settings, never enumeration order.
    named = [(0, 'Baseline', 200, 2., .3), (5, 'n100', 100, 2., .3),
             (7, 'n400', 400, 2., .3), (1, 'SNR0.5', 200, .5, .3),
             (11, 'rho0', 200, 2., 0.)]
    lookup = {sid: name for sid, name, *_ in named}
    metrics = ('pmse', 'signal_mse', 'ree', 'exact_recovery', 'mode_f1_macro',
               'element_f1', 'q_hat', 'q_over_Q', 'rss_train', 'runtime_sec', 'sic')
    paired = []
    for raw in native:
        row = {key: (value if key.endswith('_supports') else float(value))
               for key, value in raw.items()}
        task = requests[int(row['task_id'])]
        config = task['config']
        assert task['scenario_id'] == int(row['scenario_id'])
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, float))
        sid = task['scenario_id']
        _, _, n, snr, rho = next(x for x in named if x[0] == sid)
        assert config['n_train'] == n and config['snr'] == snr and config['rho'] == [rho] * 3
        row.update(scenario_name=lookup[sid], n_train=n, snr=snr, rho=rho,
                   seed=task['seed'], test_seed=task['test_seed'], Q=math.floor(.7*n))
        for version in ('v5', 'v51', 'v52'):
            row[version+'_q_over_Q'] = row[version+'_q_hat']/row['Q']
        for key in metrics:
            row['bcd_delta_'+key] = row['v51_'+key] - row['v5_'+key]
            row['grid_delta_'+key] = row['v52_'+key] - row['v51_'+key]
            assert math.isclose(row['bcd_delta_'+key]+row['grid_delta_'+key],
                                row['v52_'+key]-row['v5_'+key],rel_tol=1e-12,abs_tol=1e-10)
        assert math.isclose(row['delta1_pmse'],row['bcd_delta_pmse'],abs_tol=1e-12)
        assert math.isclose(row['delta2_pmse'],row['grid_delta_pmse'],abs_tol=1e-12)
        paired.append(row)
    versions, effects, scenes, tails = [], [], [], []
    for sid, name, n, snr, rho in named:
        subset = [r for r in paired if int(r['scenario_id']) == sid]
        assert len(subset) == len({r['replicate'] for r in subset}) == 50
        summaries = {}
        for version in ('v5', 'v51', 'v52'):
            values = {key: statistics.mean(r[version+'_'+key] for r in subset) for key in metrics}
            values['exact_count'] = int(sum(r[version+'_exact_recovery'] for r in subset))
            summaries[version] = values
            versions.append({'scenario_id':sid,'scenario_name':name,'version':version,'n':50,**values})
        for phase in ('bcd','grid'):
            for key in metrics:
                values = [r[phase+'_delta_'+key] for r in subset]
                effects.append({'scenario_id':sid,'scenario_name':name,'phase':phase,'metric':key,
                    'n_pairs':50,'mean_delta':statistics.mean(values),'median_delta':statistics.median(values),
                    'minimum_delta':min(values),'maximum_delta':max(values),
                    'decreased':sum(v < -1e-12 for v in values),'tied':sum(abs(v)<=1e-12 for v in values),
                    'increased':sum(v > 1e-12 for v in values)})
            for kind, case in [('minimum_delta',min(subset,key=lambda r:r[phase+'_delta_pmse'])),
                               ('maximum_delta',max(subset,key=lambda r:r[phase+'_delta_pmse']))]:
                tails.append({'scenario_name':name,'phase':phase,'case_selection':kind,
                    'task_id':int(case['task_id']),'replicate':int(case['replicate']),
                    **{key:case[key] for key in ('v5_pmse','v51_pmse','v52_pmse','bcd_delta_pmse','grid_delta_pmse',
                        'v5_q_hat','v51_q_hat','v52_q_hat','v5_rss_train','v51_rss_train','v52_rss_train')}})
        scenes.append({'scenario_id':sid,'name':name,'n_train':n,'snr':snr,'rho':rho,'n_pairs':50,
            'versions':summaries,'bcd_delta_pmse':summaries['v51']['pmse']-summaries['v5']['pmse'],
            'grid_delta_pmse':summaries['v52']['pmse']-summaries['v51']['pmse'],
            'bcd_delta_exact_count':summaries['v51']['exact_count']-summaries['v5']['exact_count'],
            'grid_delta_exact_count':summaries['v52']['exact_count']-summaries['v51']['exact_count'],
            'runtime_ratio_v51_over_v5':summaries['v51']['runtime_sec']/summaries['v5']['runtime_sec'],
            'runtime_ratio_v52_over_v51':summaries['v52']['runtime_sec']/summaries['v51']['runtime_sec']})
    families = {}
    for row in paired:
        families.setdefault((int(row['seed']),int(row['test_seed'])),[]).append(int(row['task_id']))
    source_sha = hashlib.sha256((base/'paired_three_versions.csv').read_bytes()).hexdigest()
    save_csv(out/'paired_effects.csv',paired)
    save_csv(out/'version_metrics.csv',versions)
    save_csv(out/'effect_summary.csv',effects)
    save_csv(out/'extreme_cases.csv',tails)
    analysis = {'complete':True,'task_labels':250,'unique_full_input_keys':250,'seed_families':len(families),
        'source_csv_sha256':source_sha,'plan_seal':plan['sha256'],'scenarios':scenes,
        'seed_family_members':{str(k):v for k,v in sorted(families.items())},
        'analysis_type':'Descriptive matched contrasts within each of five fixed scenarios; no significance tests, no pooled IID confidence intervals, no tuning or refitting.',
        'test_stream_evidence':'Archived generator/config/test_seed and matching original Linux environment; no saved per-array test-stream fingerprint or test predictions for independent PMSE reconstruction.',
        'cost_limit':'Saved runtime values are observational across runs/jobs, with no randomized common-hardware timing experiment; ratios are descriptive.'}
    with (out/'analysis.json').open('x',encoding='utf-8') as handle:
        json.dump(analysis,handle,indent=2,ensure_ascii=False)
    print('Analyzed',len(paired),'paired tasks;',len(families),'seed families; no fits')


if __name__ == '__main__':
    main()
