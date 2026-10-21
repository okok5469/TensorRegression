"""Describe finished Task10 paired programs and write result reports; no fits."""
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, body):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(body,handle,ensure_ascii=False,indent=2,allow_nan=False)


def write_csv(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,list(rows[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def table(header, values):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in values]])


def fixed_support_stability(summary, arm):
    chosen = summary['arms'][arm]
    if chosen['sizes'] not in summary['common_grid'] or chosen['full_candidate_index']==0:
        return None
    index = summary['common_grid'].index(chosen['sizes'])
    supports = [fold['candidates'][index]['support'] for fold in summary['folds']]
    values = [len(set(a)&set(b))/len(set(a)|set(b))
              for i,left in enumerate(supports) for right in supports[i+1:]
              for a,b in zip(left,right)]
    assert len(values)==9
    return float(np.mean(values))


def summarize(records, cohort):
    rows = []
    for record in records:
        req = record['request']; config = req['config']
        for arm,m in record['metrics'].items():
            qt = int(np.prod(config['sparsity']))
            limit = (64 if config['n_train']==100 else 252) if arm=='A0' else (36 if config['n_train']==100 else 180)
            rows.append({'cohort':cohort,'task_id':req['task_id'],'cell_id':req['cell_id'],
                         'replicate':req['replicate'],'n':config['n_train'],'snr':config['snr'],
                         'rho':config['rho'][0],'truth_shape':config['sparsity'],'arm':arm,
                         **m,'q_true':qt,'q_under':int(m['q']<qt),'q_over':int(m['q']>qt),
                         'q_over_native_Q':m['q']/int(np.floor(.7*config['n_train'])),
                         'selection_max_q':limit,'near_selection_limit':int(m['q']>=.9*limit),
                         'fixed_program_inner_support_jaccard':fixed_support_stability(record['_summary'],arm),
                         'program_wall_seconds':record['measured_program_wall_seconds'][arm],
                         'task_wall_seconds':record['resources']['wall_seconds'],
                         'task_cpu_seconds':record['resources']['cpu_seconds'],
                         'task_peak_memory_bytes':record['resources']['peak_working_set_bytes']})
    stats = []
    for group, subset in [('all',rows),('n100',[r for r in rows if r['n']==100]),
                          ('n400',[r for r in rows if r['n']==400]),
                          ('SNR0.5',[r for r in rows if r['snr']==.5]),
                          ('SNR4',[r for r in rows if r['snr']==4])]:
        for arm in ('A0','A1','A2'):
            selected = [r for r in subset if r['arm']==arm]
            if not selected:
                continue
            p = [r['pmse'] for r in selected]
            stats.append({'cohort':cohort,'group':group,'arm':arm,'n_cases':len(selected),
                          'pmse_mean':float(np.mean(p)),'pmse_median':float(np.median(p)),
                          'pmse_p90':float(np.quantile(p,.9)),'pmse_p95':float(np.quantile(p,.95)),
                          'pmse_max':max(p),'exact_count':sum(int(r['exact_recovery']) for r in selected),
                          **{key:float(np.mean([r[key] for r in selected])) for key in
                             ('q','q_under','q_over','near_selection_limit','ree','mode_f1_macro','element_f1','rss','program_wall_seconds')}})
    pairs = []
    index = {(r['task_id'],r['arm']):r for r in rows}
    for record in records:
        task = record['request']['task_id']
        for left,right in (('A0','A1'),('A1','A2'),('A0','A2')):
            a,b = index[(task,left)],index[(task,right)]
            delta = b['pmse']-a['pmse']
            pairs.append({'cohort':cohort,'task_id':task,'n':a['n'],'snr':a['snr'],
                          'left':left,'right':right,'delta_pmse':delta,'delta_rss':b['rss']-a['rss'],
                          'delta_q':b['q']-a['q'],'pmse_ratio':b['pmse']/a['pmse'],
                          'outcome':'improved' if delta < -1e-12 else 'worse' if delta > 1e-12 else 'equal',
                          'same_shape':a['sizes']==b['sizes'],'same_support':a['support']==b['support'],
                          'same_B':a['B_sha256']==b['B_sha256'],
                          'left_q':a['q'],'right_q':b['q'],'left_pmse':a['pmse'],'right_pmse':b['pmse'],
                          'left_shape':a['sizes'],'right_shape':b['sizes'],
                          'left_support':a['support'],'right_support':b['support']})
    cells = []
    for cell in sorted({r['cell_id'] for r in rows}):
        selected = [r for r in rows if r['cell_id']==cell]
        means = {arm:float(np.mean([r['pmse'] for r in selected if r['arm']==arm])) for arm in ('A0','A1','A2')}
        for arm in ('A0','A1','A2'):
            part = [r for r in selected if r['arm']==arm]
            shapes = sorted({tuple(r['sizes']) for r in part})
            cells.append({'cohort':cohort,'cell_id':cell,'n':part[0]['n'],'snr':part[0]['snr'],
                          'truth_shape':part[0]['truth_shape'],'rho':part[0]['rho'],'arm':arm,'replicates':len(part),
                          'pmse_mean':means[arm],'ratio_vs_A0':means[arm]/means['A0'],
                          'q_mean':float(np.mean([r['q'] for r in part])),
                          'q_std_descriptive':float(np.std([r['q'] for r in part])),
                          'distinct_selected_shapes':len(shapes),'selected_shapes':shapes,
                          'exact_count':sum(int(r['exact_recovery']) for r in part)})
    return rows,stats,pairs,cells


def main():
    from research.taskcard10.report_stopped import write_stopped_report
    root = Path(__file__).resolve().parents[2]
    write_stopped_report(root)


if __name__ == '__main__':
    main()
