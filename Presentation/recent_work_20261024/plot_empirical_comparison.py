"""Saved-result comparisons only; no model fitting or test-driven selection.

Figure contract: show the within-dataset gap to the strongest observed baseline,
and the loss of Mushroom generalization under held-file-block evaluation.
Single-axis dot plots, slide-sized vector exports. Repeated OOF uses sample SD;
future and pooled block evaluations have no fabricated replicate uncertainty.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def select(rows, **keys):
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(matches) == 1, keys
    return matches[0]


def export(fig, out, name, records):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(out/'figures'/f'{name}.alignment.json'), strict=True)
    fig.savefig(out/'figures'/f'{name}.pdf')
    fig.savefig(out/'figures'/f'{name}.svg')
    fig.savefig(out/'figures'/f'{name}.png', dpi=300)
    with (out/'figure_data'/f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.close(fig)


def comparison_plot(labels, records, out, name, baseline_label, xlim):
    blue, gray = '#003F87', '#777777'
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    fig.subplots_adjust(left=.20, right=.78, bottom=.20, top=.80)
    count = len(labels)
    positions = np.arange(count)[::-1]
    for j, (kind, label, color, marker, offset) in enumerate([
        ('abess', 'Tensor-ABESS', blue, 'o', -.20),
        ('baseline', baseline_label, gray, 's', .20),
    ]):
        subset = [r for r in records if r['kind'] == kind]
        for i, row in enumerate(subset):
            val, spread = row['r2'], row['r2_sd']
            ax.errorbar(val, positions[i]+offset, xerr=spread or None,
                        marker=marker, markersize=6, color=color, capsize=3,
                        linewidth=1.2, linestyle='none', label=label if i == 0 else None)
            ax.text(1.03, positions[i]+offset, f'{val:.3f}', transform=ax.get_yaxis_transform(),
                    fontsize=12, color=color, va='center', ha='left')
    ax.axvline(0, color='#bbbbbb', linewidth=.8, linestyle='--', zorder=0)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-.55, count-.45)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r'测试 $R^2$（越高越好）', labelpad=8, fontsize=18)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.legend(loc='lower center', bbox_to_anchor=(.56, 1.09), ncol=2, columnspacing=1.2)
    ax.text(1.03, 1.03, r'$R^2$', transform=ax.transAxes, fontsize=16, color='#555555')
    ax.spines['left'].set_visible(False)
    export(fig, out, name, records)


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    sys.path.insert(0, 'C:/Users/xmx/.codex/skills/nature-figure/scripts')
    assert json.loads((root/'doc/taskcard13/figure_backend.json').read_text(encoding='utf-8'))['backend'] == 'python'
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['SimHei','Arial','DejaVu Sans'],
        'font.size':13, 'axes.labelsize':14, 'xtick.labelsize':12, 'ytick.labelsize':13,
        'legend.fontsize':13, 'legend.frameon':False, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
    paths = {
        'LEMON': root/'empirical_analysis/lemon/outputs_eeg_incremental_v52_search_final/cv/model_summary.csv',
        'Apple': root/'doc/taskcard13/summary_metrics.csv',
        'Mushroom': root/'doc/mushroom_empirical/summary_metrics.csv',
        'Beijing': root/'doc/beijing_air_quality/summary_metrics.csv',
    }
    tables = {k: read_rows(v) for k,v in paths.items()}
    specs = [('LEMON','Tensor-ABESS','Covariates Only',{},145,'log(TMT_5)'),
             ('Apple','M1','M4',{},240,'dry matter percentage points'),
             ('Mushroom','M1','M6',{'protocol':'main'},250,'dry matter percentage points'),
             ('Beijing','M6','M4',{'protocol':'future'},352,'ug/m3')]
    records = []
    for dataset, abess, control, extra, n, unit in specs:
        future = dataset == 'Beijing'
        suffix = '' if future else '_mean'
        candidates = [r for r in tables[dataset] if all(r[k] == v for k,v in extra.items()) and r['model'] != abess]
        assert min(candidates, key=lambda r:float(r['rmse'+suffix]))['model'] == control
        for kind, model in [('abess',abess),('baseline',control)]:
            r = select(tables[dataset], model=model, **extra)
            records.append(dict(dataset=dataset, kind=kind, model=model, protocol=extra.get('protocol','repeated_oof'),
                n=n, r2=float(r['r2'+suffix]), r2_sd=None if future else float(r['r2_sd']),
                rmse=float(r['rmse'+suffix]), rmse_unit=unit, source=paths[dataset].relative_to(root).as_posix()))
    comparison_plot(['LEMON','Apple','Mushroom','北京'],records,out,'empirical_comparison','各自最好对照',(-.82,1.0))
    block_records=[]
    for protocol in ['main','block']:
        for kind, model in [('abess','M1'),('baseline','M6')]:
            r=select(tables['Mushroom'],protocol=protocol,model=model)
            block_records.append(dict(dataset='Mushroom',kind=kind,model=model,protocol=protocol,n=250,
                r2=float(r['r2_mean']),r2_sd=float(r['r2_sd']) if r['r2_sd'] else None,
                rmse=float(r['rmse_mean']),rmse_unit='dry matter percentage points',
                source=paths['Mushroom'].relative_to(root).as_posix()))
    comparison_plot(['主验证\n重复谱分组','文件块留一\n拼合块外预测'],block_records,out,
                    'mushroom_protocol_comparison','原平均谱Ridge',(-.62,1.02))
    manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()}
    (out/'figure_qa/EMPIRICAL_SOURCES.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'figure_qa/EMPIRICAL_FIGURE_NOTES.md').write_text(
        '# Empirical comparison figure contract\n\n'
        'No new fits. Each pair compares methods within its own dataset and frozen evaluation. '
        'The lowest observed comparator RMSE is descriptive selection for presentation, not a new validation claim.\n\n'
        'Points: mean complete-repeat OOF R2 for LEMON/Apple/Mushroom main; error bars: sample SD across 3 reused-subject repeats. '
        'Beijing: one future evaluation on 352 dates, no replicate SD. Mushroom block: pooled five-file held-out predictions on 250 records, no replicate SD. '
        'No p values or cross-dataset ranking. Best baselines are Covariates Only, Ridge, native-spectrum Ridge, and Vector Lasso respectively.\n\n'
        'Second figure holds ABESS and native-spectrum Ridge fixed across two protocols; native-spectrum Ridge is not the best block model. '
        'All eight block models have negative pooled R2 in the source table.\n\n'
        'Backend: saved Python preference. Single-axis plots; alignment audit not applicable. '
        '6.4-inch vector figures for 4:3 slides; 300-dpi PNG is a preview, PDF/SVG are authoritative.\n',encoding='utf-8')
    print('Generated two comparison figures and exact source CSVs.')


if __name__=='__main__':
    main()
