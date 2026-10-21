"""All observed paired folds and predeclared training-only complexity curves."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def export(fig, stem):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(stem) + '.alignment.json',
                                      overlay_svg=str(stem) + '.alignment.svg',
                                      tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem) + '.pdf')
    fig.savefig(str(stem) + '.svg')
    fig.savefig(str(stem) + '.png', dpi=600)
    fig.savefig(str(stem) + '.tiff', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('doc/taskcard7/figures'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'legend.fontsize': 6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = Path('numerical_simulation/outputs_version_attribution')
    rows = load_csv(base / 'lemon/three_version_folds.csv')
    versions = ['v5-original', 'v5.1-bcd-fix', 'v5.2-search-final']
    keys = [(r['version'], int(r['repeat']), int(r['outer_fold'])) for r in rows]
    expected = {(v, r, f) for v in versions for r in range(3) for f in range(1, 6)}
    if len(keys) != 45 or set(keys) != expected:
        raise ValueError('Requires exactly all 45 version-fold observations')
    lookup = dict(zip(keys, rows))
    palette = ['#456B8C', '#C17B45', '#73548F']
    labels = ['v5-original', 'v5.1-bcd-fix', 'v5.2-search-final']
    markers = ['o', 's', 'D']
    fig, ax = plt.subplots(3, 1, figsize=(7.2047244094, 6.6929133858))
    fig.subplots_adjust(left=.13, right=.975, bottom=.085, top=.89, hspace=.51)
    metric_names = ['Test RMSE', 'Selected q / Q', 'Test - train RMSE']
    for panel_index, panel in enumerate(ax):
        for index, (repeat, fold) in enumerate((r, f) for r in range(3) for f in range(1, 6)):
            paired = [lookup[v, repeat, fold] for v in versions]
            extract = [lambda r: float(r['test_rmse']), lambda r: float(r['q_over_Q']),
                       lambda r: float(r['test_rmse']) - float(r['train_rmse'])][panel_index]
            values = [extract(r) for r in paired]
            xpos = np.array([-.17, 0, .17]) + index + 1
            panel.plot(xpos, values, color='#BBBBBB', lw=.7, zorder=1)
            for j, (value, x) in enumerate(zip(values, xpos)):
                panel.plot(x, value, marker=markers[j], ms=3.4, ls='none', color=palette[j],
                           markerfacecolor='white' if j == 2 else palette[j], markeredgewidth=.8,
                           label=labels[j] if index == 0 else None, zorder=3)
        panel.set(ylabel=metric_names[panel_index], xlim=(.5, 15.5),
                  xticks=np.arange(1, 16),
                  xticklabels=[f'{r+1}/{f}' for r in range(3) for f in range(1, 6)])
        panel.text(-.105, 1.10, chr(97 + panel_index), transform=panel.transAxes, fontsize=8, weight='bold')
        for boundary in (5.5, 10.5):
            panel.axvline(boundary, color='#DDDDDD', lw=.6, ls=':')
    ax[1].axhline(1, color='#999999', lw=.7, ls='--')
    ax[1].set_ylim(-.03, 1.06)
    ax[2].axhline(0, color='#999999', lw=.7, ls='--')
    ax[2].set_xlabel('Outer split: repeat / fold')
    handles, texts = ax[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc='upper center', bbox_to_anchor=(.54,.97), ncol=3,
               frameon=False, columnspacing=2)
    export(fig, args.output_dir / 'lemon_three_version_paired')

    inner = load_csv(base / 'train_only/results/fit_metrics.csv')
    if len(inner) != 45:
        raise ValueError('Requires all 45 predeclared training-only fits')
    fig, ax = plt.subplots(1, 2, figsize=(7.2047244094, 3.7401574803))
    fig.subplots_adjust(left=.11, right=.97, bottom=.21, top=.78, wspace=.49)
    names = ['Fixed small', 'Fixed medium', 'Fixed large']
    for panel_index, panel in enumerate(ax):
        field = ['train_rmse', 'validation_rmse'][panel_index]
        for repeat in range(3):
            for fold in range(1, 4):
                selected = [next(r for r in inner if int(r['repeat']) == repeat
                                 and int(r['inner_fold']) == fold and r['model'] == name) for name in names]
                panel.plot([0, 1, 2], [float(r[field]) for r in selected], color=palette[repeat],
                           alpha=.65, marker=markers[repeat], ms=3, lw=.8,
                           label=f'Outer repeat {repeat+1}' if fold == 1 else None)
        panel.set(xticks=[0,1,2], xticklabels=['1','12','45'], xlabel='Predeclared fixed q',
                  ylabel=['Training RMSE', 'Validation RMSE'][panel_index],
                  title=['Training fit', 'Unseen inner subjects'][panel_index])
        panel.text(-.17, 1.17, chr(97+panel_index), transform=panel.transAxes, fontsize=8, weight='bold')
    handles, texts = ax[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc='upper center', bbox_to_anchor=(.54,.96), ncol=3,
               frameon=False, columnspacing=1.7)
    export(fig, args.output_dir / 'train_only_complexity')


if __name__ == '__main__':
    main()
