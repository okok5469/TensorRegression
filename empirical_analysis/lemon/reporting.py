"""Internal LEMON diagnostics from saved observations and fitted artifacts only.

These plots are for pipeline review, not manuscript claims or publication QA.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .io import read_json, write_json


def _table(path, columns):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = set(columns) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f'{path}: missing columns {sorted(missing)} or empty table')
    return frame


def _numbers(series, *, missing=False):
    values = pd.to_numeric(series.replace('', np.nan), errors='raise').to_numpy(float)
    valid = np.isfinite(values) | (np.isnan(values) if missing else False)
    if not np.all(valid):
        raise ValueError(f'Non-finite numeric values in {series.name}')
    return values


def _flags(frame, column):
    values = frame[column].astype(str).str.lower()
    if not values.isin(['true', 'false']).all():
        raise ValueError(f'{column} must contain explicit true/false values')
    return values.eq('true').to_numpy()


def _unique_ids(frame, source):
    if frame.subject_id.duplicated().any() or frame.subject_id.astype(str).str.strip().eq('').any():
        raise ValueError(f'{source}: subject IDs must be nonempty and unique')


def _scores(observed, predicted):
    residual = observed - predicted
    tss = float(np.sum((observed - observed.mean()) ** 2))
    return {'rmse': float(np.sqrt(np.mean(residual ** 2))), 'mae': float(np.mean(np.abs(residual))),
            'r2': np.nan if tss == 0 else float(1 - np.sum(residual ** 2) / tss)}


def _load_inputs(config, out):
    if config.get('report', {}).get('channel_plot', 'barplot') != 'barplot':
        raise ValueError('report.channel_plot supports only explicit barplot; no montage is inferred')
    scale, mode = config['response']['primary_transform'], config['analysis']['mode']
    if scale not in {'log', 'raw'} or mode not in {'adjusted', 'pilot'}:
        raise ValueError('Invalid response scale or analysis mode')
    schemas = {
        'tmt': ('diagnostics/tmt_clean.csv', ['subject_id', 'tmt5_raw', 'tmt5_log']),
        'inventory': ('diagnostics/eeg_inventory.csv', ['subject_id', 'file_complete', 'exclusion_reason']),
        'qc': ('qc/subject_qc.csv', ['subject_id', 'include_primary', 'exclusion_reason']),
        'subjects': ('qc/final_subject_manifest.csv', ['subject_id', 'include_primary', 'exclusion_reason']),
        'cv': ('cv/cv_summary.csv', ['model', 'response_scale', 'mode', 'n_subjects', 'n_folds'] +
               [metric + suffix for metric in ('rmse', 'mae', 'r2') for suffix in ('_mean', '_std')] +
               ['oof_rmse', 'oof_mae', 'oof_r2']),
        'folds': ('cv/cv_metrics.csv', ['fold', 'model', 'n_test', 'response_scale', 'rmse', 'mae', 'r2']),
        'oof': ('cv/oof_predictions.csv', ['subject_id', 'fold', 'model', 'observed', 'predicted', 'response_scale']),
    }
    tables = {key: _table(out / path, columns) for key, (path, columns) in schemas.items()}
    for name in ('tmt', 'inventory', 'qc', 'subjects'):
        _unique_ids(tables[name], schemas[name][0])
    for name in ('qc', 'subjects'):
        if set(tables[name].subject_id) != set(tables['inventory'].subject_id):
            raise ValueError(f'{name} subject IDs disagree with the EEG inventory')
    flags = {'complete': _flags(tables['inventory'], 'file_complete'),
             'qc': _flags(tables['qc'], 'include_primary'),
             'final': _flags(tables['subjects'], 'include_primary')}
    raw, logged = (_numbers(tables['tmt'][name], missing=True) for name in ('tmt5_raw', 'tmt5_log'))
    valid_tmt = np.isfinite(raw)
    if not np.array_equal(valid_tmt, np.isfinite(logged)) or np.any(raw[valid_tmt] <= 0):
        raise ValueError('Clean TMT raw/log columns must have aligned missingness and positive raw values')
    if not valid_tmt.any() or not np.allclose(np.log(raw[valid_tmt]), logged[valid_tmt]):
        raise ValueError('Clean TMT log values disagree with raw values')
    with np.load(out / 'qc/analysis_dataset.npz', allow_pickle=False) as archive:
        X, y, ids = archive['X'], archive['y'], archive['subject_ids'].astype(str).tolist()
    if X.ndim != 4 or y.shape != (len(ids),) or X.shape[0] != len(ids) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Invalid analysis tensor/response shapes or finite values')
    final_ids = tables['subjects'].loc[flags['final'], 'subject_id'].tolist()
    if ids != final_ids or not ids or len(set(ids)) != len(ids):
        raise ValueError('Final manifest and analysis dataset subject order disagree')
    response = tables['tmt'].set_index('subject_id')['tmt5_' + scale]
    if not np.allclose(response.loc[ids].to_numpy(float), y):
        raise ValueError('Saved y disagrees with the specified response scale')
    meta = read_json(out / 'tensors/tensor_meta.json')
    shape = (len(meta['channels']), int(meta['time_windows']), len(meta['bands']))
    if shape != X.shape[1:] or tuple(meta['shape'][1:]) != shape:
        raise ValueError('Tensor metadata dimensions disagree with the analysis dataset')
    if len(set(meta['channels'])) != shape[0] or len(set(meta['bands'])) != shape[2]:
        raise ValueError('Tensor axis labels must be unique')
    B = np.load(out / 'final_model/B_hat.npy', allow_pickle=False)
    final = read_json(out / 'final_model/final_model_summary.json')
    if B.shape != shape or not np.isfinite(B).all():
        raise ValueError('Final coefficients have invalid dimensions or values')
    if final['subject_ids'] != ids or tuple(final['tensor_shape']) != X.shape:
        raise ValueError('Final fit and analysis dataset subjects/shapes disagree')
    if final['response_scale'] != scale or final['mode'] != mode or final['independent_test_performance'] is not False:
        raise ValueError('Final fit scale/mode or independent-performance declaration is invalid')
    supports = {axis: read_json(out / 'final_model' / ('S_' + axis + '.json')) for axis in ('channel', 'time', 'frequency')}
    for axis, length in zip(supports, shape):
        support = supports[axis]
        if not isinstance(support, list) or len(set(support)) != len(support) or any(type(index) is not int or not 0 <= index < length for index in support):
            raise ValueError(f'Invalid final {axis} support')
    q = int(np.prod([len(support) for support in supports.values()]))
    if final['q'] != q:
        raise ValueError('Final q disagrees with the rectangular support size')
    labels = read_json(out / 'final_model/support_labels.json')
    if labels != {'channels': [meta['channels'][index] for index in supports['channel']],
                  'time_indices': supports['time'], 'frequency_bands': [meta['bands'][index] for index in supports['frequency']]}:
        raise ValueError('Final support labels disagree with the tensor metadata')
    cv, folds, oof = (tables[name] for name in ('cv', 'folds', 'oof'))
    models = cv.model.tolist()
    if len(models) != len(set(models)) or 'Tensor-ABESS' not in models:
        raise ValueError('CV must contain unique models including Tensor-ABESS')
    for table in (cv, folds, oof):
        if set(table.response_scale) != {scale}:
            raise ValueError('Mixed or inconsistent response scale in CV outputs')
        if set(table.model) != set(models):
            raise ValueError('CV model sets disagree')
    if set(cv['mode']) != {mode} or not np.all(_numbers(cv.n_subjects) == len(ids)):
        raise ValueError('CV mode or sample count disagrees with the analysis dataset')
    if oof.duplicated(['subject_id', 'model']).any() or (oof.groupby('subject_id').fold.nunique() != 1).any():
        raise ValueError('OOF predictions must be unique and share subject folds across models')
    oof['observed'], oof['predicted'] = _numbers(oof.observed), _numbers(oof.predicted)
    for model in models:
        model_oof = oof.loc[oof.model.eq(model)]
        if set(model_oof.subject_id) != set(ids) or not np.allclose(model_oof.set_index('subject_id').loc[ids, 'observed'], y):
            raise ValueError(f'{model}: OOF observations or subjects disagree with the dataset')
        model_folds = folds.loc[folds.model.eq(model)]
        summary = cv.loc[cv.model.eq(model)].iloc[0]
        if model_folds.fold.duplicated().any() or len(model_folds) != int(summary.n_folds) or set(model_folds.fold) != set(model_oof.fold):
            raise ValueError(f'{model}: saved outer folds disagree')
        for fold in model_folds.itertuples():
            held = model_oof.loc[model_oof.fold.eq(fold.fold)]
            expected = _scores(held.observed.to_numpy(), held.predicted.to_numpy())
            if int(fold.n_test) != len(held):
                raise ValueError(f'{model}: fold subject count disagrees with OOF rows')
            for metric, score in expected.items():
                actual = _numbers(pd.Series([getattr(fold, metric)], name=metric), missing=metric == 'r2')[0]
                if not np.isclose(score, actual, equal_nan=True):
                    raise ValueError(f'{model}: saved fold {metric} disagrees with OOF predictions')
        expected = _scores(model_oof.observed.to_numpy(), model_oof.predicted.to_numpy())
        for metric in ('rmse', 'mae', 'r2'):
            values = _numbers(model_folds[metric], missing=metric == 'r2')
            actual = _numbers(pd.Series([summary[metric + '_mean'], summary[metric + '_std'], summary['oof_' + metric]], name=metric), missing=metric == 'r2')
            if not np.allclose(actual, [np.mean(values), np.std(values, ddof=1), expected[metric]], equal_nan=True):
                raise ValueError(f'{model}: saved {metric} summary disagrees with folds/OOF')
    frequencies = {}
    for axis, length in zip(supports, shape):
        path = out / 'cv' / (axis + '_selection_frequency.csv')
        frame = _table(path, ['model', 'index', 'selection_count', 'n_folds', 'selection_frequency'])
        if set(frame.model) != set(models) or frame.duplicated(['model', 'index']).any():
            raise ValueError(f'{axis}: selection-frequency models or indices disagree')
        for model in models:
            selected = frame.loc[frame.model.eq(model)].sort_values('index')
            counts, totals, fraction = (_numbers(selected[column]) for column in ('selection_count', 'n_folds', 'selection_frequency'))
            if selected['index'].tolist() != list(range(length)) or np.any(totals < 2) or np.any(counts < 0) or np.any(counts > totals) or not np.allclose(fraction, counts / totals):
                raise ValueError(f'{axis}: invalid selection frequencies')
        frequencies[axis] = frame.loc[frame.model.eq('Tensor-ABESS')].sort_values('index').selection_frequency.to_numpy(float)
    onset_rows = []
    windows = {}
    for subject in ids:
        windows[subject] = read_json(out / 'tensors/subjects' / (subject + '_meta.json'))['windows']
        if len(windows[subject]) != shape[1]:
            raise ValueError(f'{subject}: saved window count disagrees with the tensor')
        for index, window in enumerate(windows[subject]):
            start, end = float(window['start_sec']), float(window['end_sec'])
            if window['window_index'] != index or 'source_ec_segment' not in window or not np.isfinite([start, end]).all() or start < 0 or not np.isclose(end - start, meta['window_sec']):
                raise ValueError(f'{subject}: invalid recorded window timing')
    for index in supports['time']:
        start = np.array([windows[subject][index]['start_sec'] for subject in ids], dtype=float)
        end = np.array([windows[subject][index]['end_sec'] for subject in ids], dtype=float)
        onset_rows.append({'window_index': index, 'n_subjects': len(ids),
                           'start_mean_sec': float(start.mean()), 'start_min_sec': float(start.min()), 'start_max_sec': float(start.max()),
                           'end_mean_sec': float(end.mean()), 'end_min_sec': float(end.min()), 'end_max_sec': float(end.max())})
    counts = {'eeg_subjects': len(tables['inventory']), 'complete_eeg': int(flags['complete'].sum()),
              'tensor_qc_passed': int(flags['qc'].sum()), 'tmt_rows': len(tables['tmt']),
              'valid_tmt_rows': int(valid_tmt.sum()), 'final_subjects': len(ids)}
    return tables, raw[valid_tmt], logged[valid_tmt], meta, B, supports, frequencies, onset_rows, counts, X.shape, q


def _figure(title, mode, scale, *, size=(6.5, 4.2)):
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_title(title + f'\nInternal diagnostic | {mode} | response: {scale}', fontsize=10)
    return figure, axes


def _save(figure, out, name, sources, meaning, manifest):
    for extension in ('png', 'pdf'):
        figure.savefig(out / 'figures' / (name + '.' + extension), dpi=160)
    manifest.append({'name': name, 'files': ['figures/' + name + suffix for suffix in ('.png', '.pdf')],
                     'sources': sources, 'meaning': meaning, 'purpose': 'internal_diagnostics'})
    figure.clear()


def _format(value):
    return 'undefined' if value == '' or not np.isfinite(float(value)) else f'{float(value):.5g}'


def make_report(config: dict, output: Path) -> dict:
    """Validate all saved inputs, then produce review plots and an audit summary."""
    out = Path(output)
    tables, raw, logged, meta, B, supports, frequencies, onsets, counts, shape, q = _load_inputs(config, out)
    mode, scale = config['analysis']['mode'], config['response']['primary_transform']
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    figures = []
    with mpl.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        for values, suffix, xlabel in ((raw, 'raw', 'TMT_5 (seconds)'), (logged, 'log', 'ln(TMT_5)')):
            fig, ax = _figure(f'TMT source distribution (valid n={len(values)})', mode, scale)
            ax.hist(values, bins='auto', color='#4477AA', edgecolor='white')
            ax.set(xlabel=xlabel, ylabel='Subjects')
            _save(fig, out, 'tmt5_' + suffix + '_hist', ['diagnostics/tmt_clean.csv'], 'All valid TMT source rows; not limited to the final model sample.', figures)
        fig, ax = _figure('Observed sample counts', mode, scale)
        labels = ['EEG inventoried', 'Complete EEG', 'Tensor QC passed', 'Final matched sample']
        values = [counts[key] for key in ('eeg_subjects', 'complete_eeg', 'tensor_qc_passed', 'final_subjects')]
        bars = ax.barh(labels, values, color='#4477AA')
        ax.bar_label(bars, padding=3)
        ax.invert_yaxis()
        ax.set(xlabel='Subjects', xlim=(0, max(values) * 1.18))
        _save(fig, out, 'sample_flow', ['diagnostics/eeg_inventory.csv', 'qc/subject_qc.csv', 'qc/final_subject_manifest.csv'], 'Actual recorded counts along the EEG-to-analysis flow; TMT source counts are reported separately.', figures)
        cv, fold_scores = tables['cv'], tables['folds']
        for metric, name in (('rmse', 'cv_model_comparison'), ('r2', 'cv_r2_comparison')):
            fig, ax = _figure('Outer-fold ' + metric.upper() + ': mean and fold SD', mode, scale)
            for index, row in enumerate(cv.to_dict('records')):
                values = _numbers(fold_scores.loc[fold_scores.model.eq(row['model']), metric], missing=metric == 'r2')
                mean, sd = _numbers(pd.Series([row[metric + '_mean'], row[metric + '_std']], name=metric), missing=metric == 'r2')
                if np.isfinite(mean):
                    ax.errorbar(index, mean, yerr=sd, fmt='o', color='#4477AA', capsize=5, markersize=7)
                finite = np.isfinite(values)
                ax.scatter(index + np.linspace(-0.12, 0.12, len(values))[finite], values[finite], color='#222222', s=20, zorder=3)
                if not finite.all():
                    ax.text(index, 0.02, 'undefined R2 fold', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
            ax.set_xticks(np.arange(len(cv)), cv.model.tolist())
            ax.set_ylabel(metric.upper() + (' (' + scale + ' response units)' if metric == 'rmse' else ''))
            _save(fig, out, name, ['cv/cv_summary.csv', 'cv/cv_metrics.csv'], 'Dots show every defined outer-fold metric; colored mean with fold SD is descriptive, not a confidence interval.', figures)
        for axis, name, labels in (('frequency', 'selected_frequency_bands', meta['bands']),
                                   ('time', 'selected_time_windows', [str(index) for index in range(shape[2])]),
                                   ('channel', 'channel_barplot', meta['channels'])):
            horizontal = axis == 'channel'
            fig, ax = _figure('Tensor-ABESS ' + axis + ' support', mode, scale,
                               size=(7.0, max(4.2, len(labels) * 0.16 + 1.3)) if horizontal else (6.5, 4.2))
            colors = ['#CC6677' if index in supports[axis] else '#AAAAAA' for index in range(len(labels))]
            if horizontal:
                ax.barh(np.arange(len(labels)), frequencies[axis], color=colors)
                ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
                ax.invert_yaxis()
                ax.set(xlabel='Outer-fold selection frequency', xlim=(0, 1.05))
            else:
                ax.bar(np.arange(len(labels)), frequencies[axis], color=colors)
                ax.set_xticks(np.arange(len(labels)), labels)
                ax.set(ylabel='Outer-fold selection frequency', ylim=(0, 1.05),
                       xlabel='Window index (zero-based; actual timing in onset CSV)' if axis == 'time' else 'Frequency band')
            ax.set_xlabel(ax.get_xlabel() + '\nRose: final selected; gray: not selected', fontsize=8)
            _save(fig, out, name, ['cv/' + axis + '_selection_frequency.csv', 'final_model/S_' + axis + '.json', 'tensors/tensor_meta.json'], 'ABESS fold selection frequency with final support membership; not significance or causal evidence.', figures)
        fig, ax = _figure('Final coefficient magnitude: standardized residual space', mode, scale,
                           size=(7, max(4.2, shape[1] * 0.13 + 1.3)))
        magnitude = np.abs(B[:, :, supports['frequency']]).sum(axis=2)
        mesh = ax.imshow(magnitude, aspect='auto', interpolation='nearest', cmap='Blues')
        ax.set_xticks(np.arange(shape[2]), [str(index) for index in range(shape[2])])
        ax.set_yticks(np.arange(shape[1]), meta['channels'], fontsize=7)
        ax.set(xlabel='Window index (zero-based)', ylabel='EEG channel')
        fig.colorbar(mesh, ax=ax, label='Sum |B| over final selected bands')
        _save(fig, out, 'coefficient_summary', ['final_model/B_hat.npy', 'final_model/S_frequency.json', 'tensors/tensor_meta.json'], 'Absolute coefficients summed over final selected bands; predictors are training-standardized nuisance residuals (centered predictors in pilot).', figures)
        held = tables['oof'].loc[tables['oof'].model.eq('Tensor-ABESS')]
        fig, ax = _figure('Tensor-ABESS out-of-fold prediction', mode, scale)
        ax.scatter(held.observed, held.predicted, s=24, alpha=0.7, color='#4477AA')
        low, high = np.min(held[['observed', 'predicted']].to_numpy()), np.max(held[['observed', 'predicted']].to_numpy())
        ax.plot([low, high], [low, high], linestyle='--', color='#666666', linewidth=1)
        unit = 'ln(TMT_5)' if scale == 'log' else 'TMT_5 (seconds)'
        ax.set(xlabel='Observed ' + unit, ylabel='OOF predicted ' + unit)
        _save(fig, out, 'predicted_vs_observed_oof', ['cv/oof_predictions.csv'], 'One held-out prediction per final subject for Tensor-ABESS, on one consistent response scale; no final-fit predictions.', figures)
    columns = ['window_index', 'n_subjects', 'start_mean_sec', 'start_min_sec', 'start_max_sec', 'end_mean_sec', 'end_min_sec', 'end_max_sec']
    pd.DataFrame(onsets, columns=columns).to_csv(out / 'final_model/selected_time_window_onsets.csv', index=False)
    excluded = tables['subjects'].loc[~_flags(tables['subjects'], 'include_primary')]
    reasons = Counter(reason for value in excluded.exclusion_reason for reason in str(value).split(';') if reason)
    lines = ['# LEMON internal run summary', '', 'Internal diagnostic output; no manuscript or causal conclusions.', '',
             f'- Analysis mode: {mode}' + (' (unadjusted pilot)' if mode == 'pilot' else ' (age/sex adjusted)'),
             f'- Response scale: {scale}; predictions and errors retain this scale.',
             f'- Final sample: {counts["final_subjects"]}; tensor shape: {list(shape)}.',
             f'- Final rectangular support q: {q}; channel/time/frequency sizes: ' + '/'.join(str(len(supports[axis])) for axis in ('channel', 'time', 'frequency')) + '.',
             '- Final all-subject fit is descriptive only and supplies no independent test performance.',
             '- Channel representation: barplot; no inferred sensor coordinates or scalp montage.',
             '- Full EEG SHA256 verification: not performed by this pipeline; file-size/header checks do not certify content hashes.', '',
             '## Inputs', '']
    paths = config.get('paths', config.get('data', {}))
    lines += ['- ' + key + ': ' + str(paths[key]) for key in ('eeg_root', 'tmt_csv', 'participants_file') if key in paths]
    lines += ['', 'The figures read the saved CSV, NPZ, NPY and JSON artifacts listed in figure_manifest.json.', '',
              '## Recorded sample counts', '', '| Count | Subjects |', '|---|---:|']
    lines += [f'| {name} | {value} |' for name, value in counts.items()]
    lines += ['', 'TMT source counts are independent of the EEG flow; the final sample is their validated match.', '',
              '## Held-out performance', '', '| Model | OOF RMSE | OOF MAE | OOF R2 | Fold RMSE mean (SD) | Fold MAE mean (SD) | Fold R2 mean (SD) |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in cv.to_dict('records'):
        entries = [row['model']] + [_format(row['oof_' + metric]) for metric in ('rmse', 'mae', 'r2')]
        entries += [_format(row[metric + '_mean']) + ' (' + _format(row[metric + '_std']) + ')' for metric in ('rmse', 'mae', 'r2')]
        lines.append('| ' + ' | '.join(entries) + ' |')
    lines += ['', 'Fold SD is descriptive variation across folds, not a confidence interval.', '', '## Exclusions', '', '| Recorded reason | Subjects flagged |', '|---|---:|']
    lines += [f'| {reason} | {count} |' for reason, count in sorted(reasons.items())] or ['| None | 0 |']
    lines += ['', 'A subject with multiple recorded reasons is counted under each reason.', '', '## Selected window timing', '',
              'final_model/selected_time_window_onsets.csv uses the actual start/end records for every final subject; indices are never converted to seconds arithmetically.',
              'Window timing ranges describe between-subject recording timing, not effect uncertainty.', '']
    (out / 'RUN_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest = {'purpose': 'internal_diagnostics', 'publication_qa_claimed': False, 'response_scale': scale,
                'mode': mode, 'sample_counts': counts, 'figures': figures,
                'selected_time_onset_sources': ['tensors/subjects/' + subject + '_meta.json' for subject in tables['subjects'].loc[_flags(tables['subjects'], 'include_primary'), 'subject_id']],
                'full_eeg_sha256_verified': False}
    write_json(out / 'figure_manifest.json', manifest)
    return {'purpose': 'internal_diagnostics', 'figure_count': len(figures), 'sample_counts': counts,
            'response_scale': scale, 'mode': mode, 'tensor_shape': list(shape), 'q': q,
            'summary': 'RUN_SUMMARY.md', 'figure_manifest': 'figure_manifest.json'}
