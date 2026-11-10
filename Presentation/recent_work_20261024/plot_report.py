"""Read-only plotting of saved scientific results for the teacher briefing.

Single-panel quantitative comparisons, means/sample SD, n=50 simulations;
Beijing n=352 common future dates. No fitting, selection, or source mutation.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BLUE='#003F87'


def table(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def data_csv(name,rows):
    folder=OUT/'figure_data';folder.mkdir(exist_ok=True)
    with (folder/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save(fig,name):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(directory/(name+'.alignment.json')),strict=True)
    fig.savefig(directory/(name+'.pdf'))
    fig.savefig(directory/(name+'.svg'))
    fig.savefig(directory/(name+'.png'),dpi=300)
    plt.close(fig)


def main():
    sys.path.insert(0,'C:/Users/xmx/.codex/skills/nature-figure/scripts')
    backend=json.loads((ROOT/'doc/taskcard13/figure_backend.json').read_text())
    if backend['backend']!='python':
        raise ValueError('Saved Python backend is required')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['SimHei','Arial','DejaVu Sans'],
        'font.size':14,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
        'legend.fontsize':14,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.8})
    paired=ROOT/'numerical_simulation/outputs_v51_attribution/cluster_run/paired_three_versions.csv'
    simulation=ROOT/'numerical_simulation/outputs_v52_search_final_full/summary_results.csv'
    summary=ROOT/'doc/beijing_air_quality/summary_metrics.csv'
    predictions=ROOT/'doc/beijing_air_quality/predictions.csv'
    rows=table(paired);stems=['version_pmse','simulation_pmse','beijing_rmse','beijing_series']
    fig,ax=plt.subplots(figsize=(6.4,3.65));fig.subplots_adjust(left=.13,right=.98,bottom=.22,top=.8)
    ids=['0','1','5','7','11'];records=[]
    for j,(version,label,color,marker) in enumerate([('v5','v5','#aaa','s'),('v51','v5.1','#666','^'),('v52','v5.2',BLUE,'o')]):
        means=[];sd=[]
        for sid in ids:
            values=[float(r[version+'_pmse']) for r in rows if r['scenario_id']==sid]
            if len(values)!=50:raise ValueError('All fifty paired replicates required')
            mean=statistics.mean(values);spread=statistics.stdev(values);means.append(mean);sd.append(spread)
            records.append(dict(scenario_id=sid,version=version,n=50,mean=mean,sample_sd=spread))
        ax.errorbar(np.arange(5)+(j-1)*.13,means,yerr=sd,color=color,marker=marker,linestyle='none',lw=1.4,ms=5,capsize=3,label=label)
    ax.set_xticks(range(5),['基准','弱信号','n=100','n=400','无相关'])
    ax.set_ylabel('测试 PMSE');ax.set_ylim(bottom=0)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=3)
    data_csv(stems[0],records);save(fig,stems[0])
    rows=table(simulation);records=[]
    fig,ax=plt.subplots(figsize=(6.4,3.85));fig.subplots_adjust(left=.13,right=.98,bottom=.34,top=.8)
    for method,label,color,marker in [('adaptive','ABESS',BLUE,'o'),('lasso','Lasso','#777','s'),('glasso','GLasso','#bbb','^')]:
        selected=sorted((r for r in rows if r['method']==method),key=lambda r:int(r['scenario_id']))
        if len(selected)!=14 or any(int(r['n_reps'])!=50 for r in selected):raise ValueError('All 14 simulation cells required')
        offset={'adaptive':-.18,'lasso':0.,'glasso':.18}[method]
        ax.scatter(np.arange(14)+offset,[float(r['pmse_mean']) for r in selected],color=color,marker=marker,s=27,label=label)
        records.extend(dict(scenario_id=r['scenario_id'],method=method,n=50,mean=float(r['pmse_mean']),sample_sd=float(r['pmse_sample_sd'])) for r in selected)
    labels=['基准','S=.5','S=1','S=2','S=4','n100','n200','n400','小维','中维','大维','ρ=0','ρ=.3','ρ=.6']
    ax.set_xticks(range(14),labels,rotation=90,rotation_mode='anchor',ha='right',va='center');ax.set_ylabel('测试 PMSE');ax.set_ylim(bottom=0)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=3)
    data_csv(stems[1],records);save(fig,stems[1])
    rows=[r for r in table(summary) if r['protocol']=='future']
    rows=sorted(rows,key=lambda r:r['model'])
    if len(rows)!=7 or any(int(r['n'])!=352 for r in rows):raise ValueError('Seven models and common352 future dates required')
    labels=['历史均值','持续性','单站 Ridge','全张量 Ridge','向量 Lasso','Tensor-GLasso','Tensor-ABESS']
    fig,ax=plt.subplots(figsize=(6.4,3.7));fig.subplots_adjust(left=.36,right=.96,bottom=.22,top=.97)
    values=[float(r['rmse']) for r in rows]
    ax.barh(range(7),values,color=['#a5a5a5']*6+[BLUE],height=.65)
    ax.set_yticks(range(7),labels);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.18)
    ax.set_xlabel(r'未来年 RMSE ($\mu$g/m$^3$)')
    for i,value in enumerate(values):ax.text(value+1.,i,f'{value:.2f}',va='center',fontsize=14)
    data_csv(stems[2],[dict(model=r['model'],name=label,n=352,rmse=float(r['rmse'])) for label,r in zip(labels,rows)]);save(fig,stems[2])
    rows=[r for r in table(predictions) if r['protocol']=='future' and r['model']=='M6']
    if len(rows)!=352 or len({r['date'] for r in rows})!=352:raise ValueError('Full352 future ABESS dates required')
    calendar=np.arange('2016-03-01','2017-03-01',dtype='datetime64[D]');lookup={r['date']:r for r in rows}
    observed=np.array([float(lookup[str(d)]['y']) if str(d) in lookup else np.nan for d in calendar])
    predicted=np.array([float(lookup[str(d)]['yhat']) if str(d) in lookup else np.nan for d in calendar])
    if np.isnan(observed).sum()!=13:raise ValueError('Expected13 natural-calendar gaps')
    fig,ax=plt.subplots(figsize=(6.4,2.7));fig.subplots_adjust(left=.14,right=.98,bottom=.28,top=.76)
    ax.plot(calendar,observed,color='#777',lw=1.,label='观测')
    ax.plot(calendar,predicted,color=BLUE,lw=1.,label='ABESS 预测')
    ax.set_ylabel(r'PM$_{2.5}$ ($\mu$g/m$^3$)');ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=2)
    data_csv(stems[3],[dict(date=str(d),observed='' if np.isnan(y) else float(y),predicted='' if np.isnan(p) else float(p)) for d,y,p in zip(calendar,observed,predicted)])
    save(fig,stems[3])
    provenance=dict(backend='python',source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [paired,simulation,summary,predictions]},figures=stems,
        source_data_exclusions='None within declared scopes; five existing paired cells/all14 simulation labels/all7 future methods/all352 future ABESS dates',alignment='Single-panel NOT APPLICABLE')
    (OUT/'FIGURE_PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
