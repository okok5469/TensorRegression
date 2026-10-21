"""All 250 matched task differences, plus descriptive recovery counts."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-scripts',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out=Path(__file__).resolve().parent
    with (out/'paired_effects.csv').open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    analysis=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    assert analysis['complete'] and len(rows)==250
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(1,3,figsize=(7.2047244094,3.5433070866),sharey=True)
    fig.subplots_adjust(left=.165,right=.98,bottom=.23,top=.81,wspace=.42)
    colors=['#456B8C','#4F8077']
    display=['Baseline (n=200)','n = 100','n = 400','SNR = 0.5','rho = 0']
    for index,(phase,axis,color) in enumerate(zip(('bcd','grid'),axes,colors)):
        all_values=[float(r[phase+'_delta_pmse']) for r in rows]
        span=max(all_values)-min(all_values)
        axis.set_xlim(min(all_values)-.1*span,max(all_values)+.1*span)
        for y,scene in enumerate(analysis['scenarios']):
            group=[r for r in rows if r['scenario_name']==scene['name']]
            assert len(group)==50
            values=[float(r[phase+'_delta_pmse']) for r in group]
            # Position offsets separate all observations; no observations are sampled.
            offsets=np.linspace(-.14,.14,50)
            axis.scatter(values,y+offsets,s=5,color=color,alpha=.5,edgecolors='none')
            value=np.mean(values)
            axis.plot([value,value],[y-.26,y+.26],lw=1.6,color='#222222')
        axis.axvline(0,color='#999999',ls='--',lw=.6,zorder=0)
        axis.set_xlabel('Paired PMSE difference')
        axis.set_title(('BCD: v5.1 - v5','Grid: v5.2 - v5.1')[index],fontsize=7)
    for version,label,color,marker,offset in [('v5','v5','#999999','o',-.18),
          ('v51','v5.1','#456B8C','^',0.),('v52','v5.2','#C17B45','s',.18)]:
        axes[2].scatter([s['versions'][version]['exact_count'] for s in analysis['scenarios']],
            np.arange(5)+offset,s=16,label=label,color=color,marker=marker)
    axes[2].set(xlim=(-2,52),xticks=[0,10,20,30,40,50],xlabel='Exact recovery count / 50')
    axes[2].set_title('Support recovery',fontsize=7)
    axes[0].set(yticks=range(5),yticklabels=display,ylim=(4.5,-.5))
    for index,axis in enumerate(axes):
        axis.text(-.1,1.18,chr(97+index),transform=axis.transAxes,fontsize=8,weight='bold')
    handles,labels=axes[2].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.76,.045),ncol=3,frameon=False,
               handletextpad=.4,columnspacing=1.)
    fig.text(.165,.065,'All 50 pairs per scenario; dark ticks show means.',fontsize=6)
    output=out/'figures';output.mkdir(exist_ok=True)
    stem=output/'version_decomposition'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',
        overlay_svg=str(stem)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=600)
    fig.savefig(str(stem)+'.tiff',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    main()
