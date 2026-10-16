"""Paired evidence figures; no fitted models, tuning, missing rows or synthetic data."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t


def export(fig, stem):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(stem) + '.alignment.json',
                                       overlay_svg=str(stem) + '.alignment.svg',
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem) + '.pdf')
    fig.savefig(str(stem) + '.svg')
    fig.savefig(str(stem) + '.png', dpi=600)
    plt.close(fig)


def axes():
    fig, ax = plt.subplots(1, 2, figsize=(7.2047244094, 3.7401574803))
    fig.subplots_adjust(left=.12, right=.98, bottom=.21, top=.86, wspace=.80)
    for label, panel in zip(('a', 'b'), ax):
        panel.text(-.17, 1.09, label, transform=panel.transAxes, fontsize=8, weight='bold')
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('doc/taskcard6/figures'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'legend.fontsize': 6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Path('numerical_simulation/outputs_version_attribution/saved_result_analysis/paired_old_new_700.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 700 and len({r['task_id'] for r in rows}) == 700
    blue, orange, gray = '#3F6B92', '#C77C49', '#777777'
    fig, ax = axes()
    values = np.sort(np.array([float(r['delta_pmse']) for r in rows]))
    ax[0].step(values, np.arange(1, len(values)+1)/len(values), where='post', color=blue, lw=1.4)
    ax[0].axvline(0, color=gray, lw=.7, ls='--')
    ax[0].set(xlabel='PMSE difference (new - old)', ylabel='Cumulative fraction',
              title='All 700 paired task labels', ylim=(0, 1.02))
    labels = ['Baseline', 'SNR 0.5', 'SNR 1', 'SNR 2', 'SNR 4', 'n 100', 'n 200', 'n 400',
              'Dim small', 'Dim base', 'Dim large', 'rho 0', 'rho 0.3', 'rho 0.6']
    for sid in range(14):
        d = np.array([float(r['delta_pmse']) for r in rows if int(r['scenario_id']) == sid])
        assert len(d) == 50
        interval = t.ppf(.975, 49) * d.std(ddof=1) / np.sqrt(len(d))
        ax[1].errorbar(d.mean(), sid, xerr=interval, fmt='o', ms=3, lw=.8,
                       color=orange if sid in (5, 11) else blue, capsize=2)
    ax[1].axvline(0, color=gray, lw=.7, ls='--')
    ax[1].set(yticks=np.arange(14), yticklabels=labels, xlabel='Mean paired PMSE difference',
              title='Scene-specific effects (n = 50)')
    ax[1].invert_yaxis()
    export(fig, args.output_dir/'simulation_tail')

    comparison = json.loads(Path('doc/taskcard6/lemon_comparison/comparison.json').read_text())
    fig, ax = axes()
    for label, color, marker in [('old', blue, 'o'), ('new', orange, '^')]:
        data = [r for r in comparison[label]['oof_tables']['metrics_by_repeat'] if r['model'] == 'Tensor-ABESS']
        ax[0].plot([r['repeat']+1 for r in data], [r['rmse'] for r in data], marker=marker, ms=4,
                   lw=1, color=color, label='v5-original' if label == 'old' else 'v5.2')
    baseline = [r for r in comparison['new']['oof_tables']['metrics_by_repeat'] if r['model'] == 'Covariates Only']
    ax[0].plot([r['repeat']+1 for r in baseline], [r['rmse'] for r in baseline], marker='s',
               ms=3, lw=1, color=gray, label='Covariates Only')
    ax[0].set(xticks=[1,2,3], xlabel='Repeated split', ylabel='Complete OOF RMSE',
              title='145 subjects per repeat', ylim=(.28,.57))
    handles, legend_labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.32,.035), ncol=3,
               frameon=False, handlelength=1.2, columnspacing=.8)
    for index, row in enumerate(comparison['paired_folds'], 1):
        ax[1].plot([index-.12,index+.12], [row['old_q_over_Q'],row['new_q_over_Q']], color='#BBBBBB', lw=.8)
        ax[1].scatter(index-.12,row['old_q_over_Q'],color=blue,s=10,marker='o',zorder=3)
        ax[1].scatter(index+.12,row['new_q_over_Q'],color=orange,s=13,marker='^',zorder=3)
    ax[1].axhline(1, color=gray, lw=.7, ls='--')
    ax[1].set(xlabel='Outer fold (repeat-major order)', ylabel='Selected q / capacity Q',
              title='All 15 paired folds; Q = 81', xticks=[1,5,10,15], ylim=(-.03,1.05))
    export(fig, args.output_dir/'lemon_prediction_capacity')


if __name__ == '__main__':
    main()
