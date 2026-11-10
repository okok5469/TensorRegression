"""Render six audited empirical figures from released summary CSV files only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main(root, skill_scripts):
    root, skill_scripts = Path(root), Path(skill_scripts)
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    sources = ['summary_metrics.csv', 'predictions.csv', 'model_complexity.csv', 'range_metrics.csv', 'support_frequency.csv']
    tables = {p: pd.read_csv(root/p) for p in sources}
    summary, predictions, complexity, ranges, frequency = [tables[p] for p in sources]
    expected = {(g, s, 'M'+str(m)) for g in range(1,7) for s in ['dev1','dev2','dev3','final'] for m in range(8)}
    actual = set(zip(summary.gas_id,summary.split,summary.method))
    if actual != expected or len(summary)!=192:
        raise ValueError('Require complete released 192-task summary; never fabricate missing performance')
    final = summary[summary.split=='final']
    pred = predictions[predictions.split=='final']
    output=root/'figures';output.mkdir(exist_ok=True)
    qa=output/'qa';qa.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.titlesize':11,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'axes.linewidth':.7})
    names=['Ethanol','Ethylene','Ammonia','Acetaldehyde','Acetone','Toluene']
    methods=['M'+str(i) for i in range(8)]
    labels=['M0 Mean','M1 ABESS (96)','M2 GLasso (96)','M3 Lasso (96)',
            'M4 Ridge (96)','M5 PLS (96)','M6 Ridge (128)','M7 Lasso (128)']
    colors=['#8A8A8A','#0072B2','#D55E00','#009E73','#CC79A7','#8C6D31','#56B4E9','#6960A8']
    markers=['o','s','^','v','D','P','X','*']
    handles=[Line2D([],[],color=c,marker=mk,linestyle='-',label=l,markersize=5) for c,mk,l in zip(colors,markers,labels)]
    audits={}
    audit_env=dict(os.environ,PYTHONIOENCODING='utf-8')

    def grid(title,note,legend=False):
        fig,axs=plt.subplots(2,3,figsize=(12,7.5))
        fig.subplots_adjust(left=.075,right=.98,bottom=.13 if legend else .09,top=.87,wspace=.34,hspace=.44)
        fig.suptitle(title,fontsize=15,x=.075,ha='left',y=.97)
        fig.text(.075,.92,note,fontsize=9)
        for i,ax in enumerate(axs.flat):ax.set_title(f'{chr(97+i)}  {names[i]}',loc='left',pad=10)
        if legend:fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.52,.005),fontsize=8)
        return fig,axs.flat

    def export(fig,name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(qa/f'{name}.alignment.json'),strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
        fig.savefig(output/f'{name}.pdf')
        fig.savefig(output/f'{name}.svg')
        fig.savefig(output/f'{name}.png',dpi=300)
        plt.close(fig)
        commands={
            'text':[sys.executable,str(skill_scripts/'audit_pdf_text.py'),str(output/f'{name}.pdf'),'--json'],
            'collision':[sys.executable,str(skill_scripts/'audit_figure_collisions.py'),str(output/f'{name}.pdf'),'--json-out',str(qa/f'{name}.collision.json')]}
        audits[name]={}
        for kind,command in commands.items():
            result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=audit_env)
            (qa/f'{name}.{kind}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            audits[name][kind]={'returncode':result.returncode}
            if result.returncode:raise RuntimeError(f'{name} {kind} audit blocked; see QA logs')

    fig,axs=grid('Final batch: prediction error','Batch 10; all records retained. RMSE in ppmv; independent scales across gases.')
    for gas,ax in enumerate(axs,1):
        frame=final[final.gas_id==gas].set_index('method').loc[methods]
        ax.barh(np.arange(8),frame.rmse,color=colors,height=.65)
        ax.set_yticks(np.arange(8),labels);ax.invert_yaxis();ax.set_xlabel('RMSE (ppmv)')
    export(fig,'final_rmse')

    fig,axs=grid('Prediction error across successive batches','Overlapping historical training windows; descriptive trajectories, no independent-repeat intervals.',True)
    for gas,ax in enumerate(axs,1):
        for i,m in enumerate(methods):
            frame=summary[(summary.gas_id==gas)&(summary.method==m)].set_index('split').loc[['dev1','dev2','dev3','final']]
            ax.plot(range(4),frame.rmse,color=colors[i],marker=markers[i],lw=1.2,ms=4)
        ax.set_xticks(range(4),['7','8','9','10']);ax.set_xlabel('Evaluation batch');ax.set_ylabel('RMSE (ppmv)');ax.set_ylim(bottom=0)
    export(fig,'temporal_error')

    fig,axs=grid('Final batch: predicted versus observed concentration','All predictions shown, without clipping. Dashed line: identity; axes differ across gases.',True)
    for gas,ax in enumerate(axs,1):
        frame=pred[pred.gas_id==gas]
        lo=min(frame.y.min(),frame.prediction.min());hi=max(frame.y.max(),frame.prediction.max())
        span=hi-lo;lo-=span*.04;hi+=span*.04
        ax.plot([lo,hi],[lo,hi],color='#444444',ls='--',lw=.7)
        for i,m in enumerate(methods):
            points=frame[frame.method==m]
            ax.scatter(points.y,points.prediction,s=8,alpha=.35,c=colors[i],marker=markers[i],linewidths=0,rasterized=True)
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_xlabel('Observed (ppmv)');ax.set_ylabel('Predicted (ppmv)')
    export(fig,'final_predictions')

    fig,axs=grid('Final error inside and outside the training range','Cells: RMSE (ppmv); NA means no records. Color intensity is scaled separately within each gas.')
    for gas,ax in enumerate(axs,1):
        frame=ranges[(ranges.gas_id==gas)&(ranges.split=='final')]
        matrix=frame.pivot(index='method',columns='range',values='rmse').reindex(index=methods,columns=['below','inside','above']).to_numpy()
        ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='Blues',vmin=0,vmax=np.nanmax(matrix)*1.2)
        for i in range(8):
            for j in range(3):
                value=matrix[i,j];ax.text(j,i,'NA' if not np.isfinite(value) else f'{value:.1f}',ha='center',va='center',fontsize=8,
                                         color='white' if np.isfinite(value) and value>np.nanmax(matrix)*.75 else '#222222')
        counts=frame[frame.method=='M1'].set_index('range').n
        ax.set_xticks(range(3),[f'{s}\nn={int(counts[s])}' for s in ['below','inside','above']]);ax.set_yticks(range(8),methods)
        ax.tick_params(length=0)
    export(fig,'range_error')

    merged=final.merge(complexity[complexity.split=='final'],on=['gas_id','split','method'],validate='one_to_one')
    fig,axs=grid('Prediction and effective coefficient count','Final models; threshold 1e-8 × max(1, max |standardized coefficient|). Counts do not imply sensor cost.',True)
    for gas,ax in enumerate(axs,1):
        for i,m in enumerate(methods):
            row=merged[(merged.gas_id==gas)&(merged.method==m)].iloc[0]
            ax.scatter(row.effective_nnz,row.rmse,s=65,c=colors[i],marker=markers[i],edgecolors='white',linewidths=.5)
        ax.set_xlim(-5,135);ax.set_ylim(bottom=0);ax.set_xlabel('Effective nonzero coefficients');ax.set_ylabel('RMSE (ppmv)')
    export(fig,'error_complexity')

    fig,axs=plt.subplots(1,3,figsize=(12,4.3),gridspec_kw={'width_ratios':[1,1,1]})
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.72,wspace=.36)
    fig.suptitle('ABESS: development support selection frequency',fontsize=15,x=.09,ha='left',y=.96)
    fig.text(.09,.86,'Three overlapping development fits per gas; final fit excluded. Descriptive frequency only.',fontsize=9)
    for p,(axis,size,ax) in enumerate(zip(['sensors','scales','stages'],[16,3,2],axs)):
        frame=frequency[(frequency.method=='M1')&(frequency.axis==axis)]
        mat=frame.pivot(index='gas_id',columns='coordinate',values='frequency').reindex(index=range(1,7)).to_numpy()
        ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_yticks(range(6),names);ax.set_title(f'{chr(97+p)}  {axis.capitalize()}',loc='left',pad=10)
        ticks=list(range(size));ticklabels=list(range(1,17)) if axis=='sensors' else (['.001','.01','.1'] if axis=='scales' else ['Rising','Decay'])
        ax.set_xticks(ticks,ticklabels);ax.tick_params(length=0)
        if size<16:
            for i in range(6):
                for j in range(size):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if mat[i,j]>.6 else '#222222')
    fig.text(.09,.06,'Color: white = 0 / 3 selected; dark blue = 3 / 3 selected. Sensor labels are physical IDs 1–16.',fontsize=9)
    export(fig,'abess_mode_frequency')
    source_check=subprocess.run([sys.executable,str(skill_scripts/'validate_figure.py'),__file__,'--backend','python','--json'],capture_output=True,text=True,encoding='utf-8',env=audit_env)
    (qa/'source_validation.json').write_text(source_check.stdout,encoding='utf-8')
    if source_check.returncode:raise RuntimeError('Source figure audit blocked')
    (qa/'FIGURE_VALIDATION.json').write_text(json.dumps({'automatic_audits':audits,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'output_sha256':{f'{name}.{ext}':hashlib.sha256((output/f'{name}.{ext}').read_bytes()).hexdigest()
                         for name in audits for ext in ['pdf','svg','png']},
        'visual_inspection':'PENDING'},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='doc/uci270_concentration')
    parser.add_argument('--skill-scripts',required=True)
    args=parser.parse_args();main(args.root,args.skill_scripts)
