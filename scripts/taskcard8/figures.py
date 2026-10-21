"""All observed paired simulations and all prespecified inner size candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def export(fig, stem):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json',
        overlay_svg=str(stem)+'.alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.tiff', dpi=600)
    fig.savefig(str(stem)+'.png', dpi=600)
    plt.close(fig)


def inner_figure(repo, output):
    base = repo/'numerical_simulation/outputs_selection_inner_task8'
    summary = read(base/'summary.json')
    if not summary['complete'] or summary['outer_test_evaluations'] != 0:
        raise ValueError('Complete training-only evidence is required')
    fig, axes = plt.subplots(1, 3, figsize=(7.2047244094, 3.3464566929))
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.75, wspace=.48)
    names = ['common_grid_original_msic','inner_cv_size','stability_auxiliary_msic']
    labels = ['Common-grid MSIC','Inner CV','Stability auxiliary']
    colors = ['#456B8C','#C17B45','#73548F']
    markers = ['o','^','s']
    curves = [read(base/f"repeat_{r['repeat']}_fold_1/common_candidate_curve.json")
              for r in summary['summaries']]
    ymax = max(r['pooled_inner_cv_mse'] for curve in curves for r in curve)*1.08
    for index, (axis, record, curve) in enumerate(zip(axes,summary['summaries'],curves)):
        if len(curve) != 48 or record['common_grid_count'] != 48:
            raise ValueError('All 48 legal common sizes must be plotted')
        axis.scatter([r['q'] for r in curve],[r['pooled_inner_cv_mse'] for r in curve],
                     s=6, color='#B5B5B5', alpha=.7, label='All 48 sizes' if index==0 else None)
        for name,label,color,marker in zip(names,labels,colors,markers):
            row = next(r for r in record['selected'] if r['rule']==name)
            axis.scatter(row['q'],row['pooled_inner_cv_mse'],s=20,color=color,marker=marker,
                         label=label if index==0 else None,zorder=5)
        native = next(r['q'] for r in record['selected'] if r['rule']=='native_full_training_original_msic')
        axis.set(xlabel='Rectangular q',xticks=[1,12,24,36,48],title=f'Repeat {index+1}; native q = {native}',
                 xlim=(-1,50),ylim=(0,ymax))
        if index==0:
            axis.set_ylabel('Pooled inner validation MSE')
        axis.text(-.19,1.18,chr(97+index),transform=axis.transAxes,fontsize=8,weight='bold')
    handles,texts=axes[0].get_legend_handles_labels()
    fig.legend(handles,texts,loc='upper center',bbox_to_anchor=(.54,.98),ncol=2,frameon=False,
               handlelength=1.1,columnspacing=1.5,labelspacing=.6)
    export(fig,output/'inner_selection_feasibility')


def simulation_figure(repo, output):
    base=repo/'doc/taskcard8/analysis'
    analysis=read(base/'analysis.json')
    with (base/'paired_differences.csv').open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    cohorts=[c for c in ('development','confirmation') if any(r['cohort']==c for r in rows)]
    if not cohorts:
        raise ValueError('No complete paired simulation data to plot')
    fig,axes=plt.subplots(1,len(cohorts),figsize=(7.2047244094,3.7401574803),squeeze=False)
    fig.subplots_adjust(left=.12,right=.97,bottom=.27,top=.78,wspace=.42)
    bound=max(abs(float(r['delta_pmse'])) for r in rows)*1.08
    bound=max(bound,.01)
    for index,(cohort,axis) in enumerate(zip(cohorts,axes[0])):
        ticks=[]
        for position,n in enumerate((100,200,400)):
            subset=[r for r in rows if r['cohort']==cohort and int(r['n_train'])==n]
            ticks.append(f'{n}\nn = {len(subset)}')
            for sizes,marker in [([4,3,5],'o'),([2,3,4],'^')]:
                group=[r for r in subset if json.loads(r['sparsity'])==sizes]
                if group:
                    x=np.linspace(-.17,.17,len(group))+position
                    axis.scatter(x,[float(r['delta_pmse']) for r in group],s=10,marker=marker,
                        color='#456B8C',alpha=.65,label=f'Truth {sizes}' if position==0 else None)
            if subset:
                value=np.mean([float(r['delta_pmse']) for r in subset])
                axis.plot([position-.24,position+.24],[value,value],color='#C17B45',lw=1.5,
                           label='Mean paired difference' if position==0 else None)
        count=analysis['completed'][cohort]
        axis.set(xticks=[0,1,2],xticklabels=ticks,xlabel='Training sample size',
                 ylabel='PMSE difference (alternative - baseline)',ylim=(-bound,bound),
                 title=f'{cohort.capitalize()}: {count}/108')
        axis.axhline(0,color='#888888',ls='--',lw=.7)
        axis.text(-.15,1.15,chr(97+index),transform=axis.transAxes,fontsize=8,weight='bold')
    handles,texts=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,texts,loc='upper center',bbox_to_anchor=(.54,.98),ncol=3,frameon=False,
               columnspacing=1.2,handlelength=1.2)
    export(fig,output/'paired_msic_prediction')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-scripts',type=Path,required=True)
    parser.add_argument('--inner-only',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.qa_scripts))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7})
    repo=Path(__file__).resolve().parents[2]
    output=repo/'doc/taskcard8/figures'
    output.mkdir(parents=True,exist_ok=True)
    inner_figure(repo,output)
    if not args.inner_only:
        simulation_figure(repo,output)


if __name__=='__main__':
    main()
