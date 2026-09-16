"""Ordered execution with explicit prerequisites and immutable run provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import inventory_eeg, merge_analysis_arrays, prepare_tmt
from .io import fingerprint, json_value, read_json, write_json


def stage_names() -> tuple[str, ...]:
    return ('inventory', 'tmt', 'events', 'smoke', 'tensors', 'dataset', 'cv', 'final', 'report')


def _run_identity(config: dict) -> dict:
    paths = config['paths']
    inputs = [value for key, value in paths.items() if value is not None and key != 'output_dir' and key != 'eeg_root']
    inputs.extend(path for path in paths['eeg_root'].rglob('*') if path.is_file())
    repo = Path(__file__).resolve().parents[2]
    analysis_sources = [path for path in Path(__file__).parent.glob('*.py') if path.name != 'download.py']
    sources = list((repo / 'src/tensor_abess').rglob('*.py')) + list((repo / 'numerical_simulation').glob('*.py')) + analysis_sources
    effective_config = hashlib.sha256(json.dumps(config, default=json_value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
    return {'inputs_and_config': fingerprint(inputs), 'effective_config': effective_config, 'source': fingerprint(sources)}


def _context(config: dict) -> tuple[Path, dict]:
    output = config['paths']['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(config)
    state_file = output / 'run_state.json'
    if state_file.exists():
        state = read_json(state_file)
        if state['identity'] != identity:
            raise ValueError('Inputs, configuration or source changed; choose a new empty output_dir and rerun from inventory')
        for artifact in state['artifacts'].values():
            if fingerprint([output / name for name in artifact['files']]) != artifact['fingerprint']:
                raise ValueError('A completed stage artifact changed or is missing; choose a new output_dir')
        return output, state
    if any(output.iterdir()):
        raise ValueError('output_dir contains files without run_state.json; choose an empty directory')
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'mne', 'matplotlib', 'PyYAML'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = 'not installed'
    state = {'identity': identity, 'completed': [], 'stages': {}, 'artifacts': {}}
    write_json(output / 'config_used.json', {'config': config, 'versions': versions, 'identity': identity, 'eeg_integrity': 'size and modification time, not full EEG SHA256'})
    write_json(state_file, state)
    return output, state


def run_stage(config: dict, stage: str) -> dict:
    names = stage_names()
    if stage not in names:
        raise ValueError(f'Unknown stage: {stage}')
    output, state = _context(config)
    index = names.index(stage)
    try:
        if index and names[index - 1] not in state['completed']:
            raise ValueError(f'Complete stage {names[index - 1]} before {stage}')
        # Replaying completed upstream work would invalidate downstream artifacts.
        if stage in state['completed']:
            return state['stages'][stage]
        before = {path for path in output.rglob('*') if path.is_file()}
        functions = {'inventory': _inventory, 'tmt': _tmt, 'events': _events, 'smoke': _smoke, 'tensors': _tensors, 'dataset': _dataset, 'cv': _cv, 'final': _final, 'report': _report}
        result = functions[stage](config, output)
        created = sorted({path for path in output.rglob('*') if path.is_file()} - before)
        state['artifacts'][stage] = {'files': [path.relative_to(output).as_posix() for path in created], 'fingerprint': fingerprint(created)}
        state['completed'].append(stage)
        state['stages'][stage] = result
        write_json(output / 'run_state.json', state)
        return result
    except Exception as exc:
        write_json(output / 'logs' / (stage + '_error.json'), {'stage': stage, 'time_utc': datetime.now(timezone.utc).isoformat(), 'exception': type(exc).__name__, 'message': str(exc)})
        raise


def _inventory(config: dict, output: Path) -> dict:
    paths = config['paths']
    frame = inventory_eeg(paths['eeg_root'], paths.get('manifest_file'))
    diagnostics = output / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    selection_path = paths.get('subject_selection_file')
    if selection_path is not None:
        selection = read_json(selection_path)['selected_subjects']
        if not selection or len(set(selection)) != len(selection) or not set(selection).issubset(set(frame.subject_id)):
            raise ValueError('Download selection must contain unique subjects present in the input inventory')
        frame.loc[~frame.subject_id.isin(selection)].to_csv(diagnostics / 'subjects_not_selected.csv', index=False)
        frame = frame.loc[frame.subject_id.isin(selection)].copy()
    frame.to_csv(diagnostics / 'eeg_inventory.csv', index=False)
    people_path = paths.get('participants_file')
    info = {'exists': people_path is not None and people_path.is_file(), 'path': str(people_path)}
    if info['exists']:
        people = pd.read_csv(people_path, sep='\t', dtype=str, keep_default_na=False)
        info['columns'] = people.columns.tolist()
        info['rows'] = len(people)
        info['field_values'] = {name: people[name].unique()[:30].tolist() for name in people.columns if name != config.get('confounds', {}).get('id_column', 'participant_id')}
    codebook = paths.get('participants_json')
    if codebook is not None and codebook.is_file():
        info['codebook'] = read_json(codebook)
    write_json(diagnostics / 'participants_diagnostic.json', info)
    return {'subjects_in_inventory': len(frame), 'complete_eeg': int(frame.file_complete.sum()), 'participants_available': info['exists']}


def _tmt(config: dict, output: Path) -> dict:
    clean = prepare_tmt(config['paths']['tmt_csv'], output / 'diagnostics')
    return {'tmt_rows': len(clean), 'numeric_positive': int(np.isfinite(clean.tmt5_raw).sum())}


def _events(config: dict, output: Path) -> dict:
    from .events import diagnose_subject

    inventory = pd.read_csv(output / 'diagnostics/eeg_inventory.csv', keep_default_na=False)
    subjects = inventory.loc[inventory.file_complete, 'subject_id'].tolist()
    required = int(config['events']['min_diagnostic_subjects'])
    if len(subjects) < required:
        raise ValueError(f'Only {len(subjects)} complete EEG subjects; event Gate requires at least {required}')
    summaries, failures = [], []
    for subject in subjects:
        try:
            frame, evidence = diagnose_subject(config['paths']['eeg_root'] / subject / 'eeg')
        except ValueError as exc:
            failures.append({'subject_id': subject, 'reason': str(exc)})
            continue
        destination = output / 'diagnostics/event_sequences'
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / (subject + '.csv'), index=False)
        write_json(destination / (subject + '_evidence.json'), evidence)
        for description, count in frame['trial_type'].value_counts().items():
            summaries.append({'subject_id': subject, 'annotation_description': description, 'count': int(count)})
    pd.DataFrame(summaries).to_csv(output / 'diagnostics/event_code_summary.csv', index=False)
    write_json(output / 'diagnostics/event_failures.json', failures)
    if failures:
        raise ValueError(f'Event mapping failed for {len(failures)} subjects; inspect event_failures.json')
    return {'subjects_verified': len(subjects)}


def _reference_channels(config: dict) -> list[str]:
    from .eeg_preprocess import get_eeg_channels, load_raw_brainvision

    subject = config['channels']['reference_subject']
    candidates = list((config['paths']['eeg_root'] / subject / 'eeg').glob('*.vhdr'))
    if len(candidates) != 1:
        raise ValueError(f'Reference subject {subject} needs exactly one BrainVision header')
    raw = load_raw_brainvision(candidates[0], config['channels']['eog_names'])
    try:
        return get_eeg_channels(raw)
    finally:
        raw.close()


def _smoke(config: dict, output: Path) -> dict:
    from .tensorize import tensorize_subject

    subject = config['channels']['reference_subject']
    channels = _reference_channels(config)
    X, meta = tensorize_subject(config['paths']['eeg_root'] / subject / 'eeg', channels, config)
    repeated, _ = tensorize_subject(config['paths']['eeg_root'] / subject / 'eeg', channels, config)
    np.testing.assert_allclose(X, repeated, rtol=1e-10, atol=1e-10)
    tensors = output / 'tensors'
    tensors.mkdir(parents=True, exist_ok=True)
    np.save(tensors / (subject + '_tensor.npy'), X, allow_pickle=False)
    write_json(tensors / (subject + '_tensor_meta.json'), meta)
    (output / 'diagnostics/reference_eeg_channels.txt').write_text('\n'.join(channels) + '\n', encoding='utf-8')
    return {'subject_id': subject, 'shape': list(X.shape), 'repeat_verified': True}


def _tensors(config: dict, output: Path) -> dict:
    from .eeg_preprocess import get_eeg_channels, load_raw_brainvision
    from .tensorize import tensorize_subject

    inventory = pd.read_csv(output / 'diagnostics/eeg_inventory.csv', keep_default_na=False)
    reference = _reference_channels(config)
    available = {}
    for subject in inventory.loc[inventory.file_complete, 'subject_id']:
        header = next((config['paths']['eeg_root'] / subject / 'eeg').glob('*.vhdr'))
        raw = load_raw_brainvision(header, config['channels']['eog_names'])
        try:
            available[subject] = set(get_eeg_channels(raw))
        finally:
            raw.close()
    if not available:
        raise ValueError('No complete EEG subjects')
    common = [name for name in reference if all(name in names for names in available.values())]
    availability = pd.DataFrame([{'subject_id': subject, **{name: name in names for name in reference}} for subject, names in available.items()])
    availability.to_csv(output / 'diagnostics/channel_availability.csv', index=False)
    (output / 'diagnostics/common_channels.txt').write_text('\n'.join(common) + '\n', encoding='utf-8')
    if len(common) < config['channels']['min_common_eeg_channels']:
        raise ValueError(f'Only {len(common)} common EEG channels; inspect channel_availability.csv')
    subject_dir = output / 'tensors/subjects'
    subject_dir.mkdir(parents=True, exist_ok=True)
    (output / 'qc').mkdir(parents=True, exist_ok=True)
    rows, arrays, ids = [], [], []
    for row in inventory.to_dict('records'):
        row.update(include_primary=False, event_ok=False, channel_ok=False, tensor_finite=False, n_ec_segments=0, n_candidate_windows=0, n_selected_windows=0)
        subject = row['subject_id']
        if not row['file_complete']:
            rows.append(row)
            continue
        try:
            tensor, meta = tensorize_subject(config['paths']['eeg_root'] / subject / 'eeg', common, config)
            if np.any(np.var(tensor, axis=(0, 1)) == 0):
                raise ValueError('A frequency band is constant over all channel/time cells')
        except ValueError as exc:
            row['exclusion_reason'] = str(exc)
            rows.append(row)
            continue
        np.save(subject_dir / (subject + '.npy'), tensor, allow_pickle=False)
        write_json(subject_dir / (subject + '_meta.json'), meta)
        row.update(include_primary=True, event_ok=True, channel_ok=True, tensor_finite=True, n_ec_segments=meta['n_ec_segments'], n_candidate_windows=meta['n_candidate_windows'], n_selected_windows=len(meta['windows']), exclusion_reason='', mean_log_power=float(tensor.mean()))
        rows.append(row)
        arrays.append(tensor)
        ids.append(subject)
    qc = pd.DataFrame(rows)
    qc['gross_power_flag'] = False
    if arrays:
        powers = np.array([value.mean() for value in arrays])
        median = np.median(powers)
        mad = np.median(np.abs(powers - median))
        flags = np.zeros(len(powers), dtype=bool) if mad == 0 else np.abs(0.67448975 * (powers - median) / mad) > 6
        qc.loc[qc.include_primary, 'gross_power_flag'] = flags
    qc.to_csv(output / 'qc/subject_qc.csv', index=False)
    if not arrays:
        raise ValueError('No subjects passed tensor QC; inspect subject_qc.csv')
    X = np.stack(arrays)
    np.save(output / 'tensors/X_unstandardized.npy', X, allow_pickle=False)
    pd.DataFrame({'subject_id': ids}).to_csv(output / 'tensors/tensor_subject_ids.csv', index=False)
    write_json(output / 'tensors/tensor_meta.json', {'shape': list(X.shape), 'channels': common, 'bands': list(config['bands']), 'band_edges_hz': config['bands'], 'window_sec': config['events']['window_sec'], 'time_windows': config['events']['n_windows'], 'feature': 'log_absolute_band_power', 'sfreq': config['preprocess']['target_sfreq'], 'filter': [config['preprocess']['l_freq'], config['preprocess']['h_freq']], 'subject_window_metadata': 'subjects/<subject_id>_meta.json'})
    return {'included': len(ids), 'excluded': len(rows) - len(ids), 'shape': list(X.shape)}


def _dataset(config: dict, output: Path) -> dict:
    if config['analysis']['mode'] == 'adjusted':
        codebook = config['paths'].get('participants_json')
        if codebook is None or not codebook.is_file():
            raise ValueError('Adjusted analysis requires participants.json or a verified JSON codebook')
        if not isinstance(read_json(codebook), dict):
            raise ValueError('participants.json must be a field-description mapping')
    people_path = config['paths'].get('participants_file')
    people = pd.read_csv(people_path, sep='\t', dtype=str, keep_default_na=False) if people_path is not None and people_path.is_file() else None
    ids = pd.read_csv(output / 'tensors/tensor_subject_ids.csv').subject_id.tolist()
    result = merge_analysis_arrays(np.load(output / 'tensors/X_unstandardized.npy', allow_pickle=False), ids, pd.read_csv(output / 'diagnostics/tmt_clean.csv'), people, config)
    inventory = pd.read_csv(output / 'qc/subject_qc.csv', keep_default_na=False)
    # Include EEG failures in the final manifest, not only the tensor survivors.
    excluded = inventory.loc[~inventory.include_primary, ['subject_id', 'exclusion_reason']].copy()
    excluded['include_primary'] = False
    pd.concat([result['manifest'], excluded], ignore_index=True).to_csv(output / 'qc/final_subject_manifest.csv', index=False)
    payload = {'X': result['X'], 'y': result['y'], 'subject_ids': np.asarray(result['subject_ids'], dtype=str)}
    if result['C'] is not None:
        payload['C'] = result['C']
    np.savez_compressed(output / 'qc/analysis_dataset.npz', **payload)
    return {'n': len(result['y']), 'shape': list(result['X'].shape), 'pilot_unadjusted': result['pilot_unadjusted'], 'response_scale': config['response']['primary_transform']}


def _analysis_arrays(output: Path) -> dict:
    with np.load(output / 'qc/analysis_dataset.npz', allow_pickle=False) as file:
        return {'X': file['X'], 'y': file['y'], 'subject_ids': file['subject_ids'].tolist(), 'C': file['C'] if 'C' in file.files else None}


def _cv(config: dict, output: Path) -> dict:
    from .evaluation import run_cross_validation

    return run_cross_validation(**_analysis_arrays(output), config=config, output_dir=output / 'cv')


def _final(config: dict, output: Path) -> dict:
    from .evaluation import fit_final_model

    result = fit_final_model(**_analysis_arrays(output), config=config, output_dir=output / 'final_model')
    meta = read_json(output / 'tensors/tensor_meta.json')
    supports = {name: read_json(output / 'final_model' / ('S_' + name + '.json')) for name in ('channel', 'time', 'frequency')}
    # The saved supports remain zero-based indices; the separate table supplies labels.
    write_json(output / 'final_model/support_labels.json', {'channels': [meta['channels'][i] for i in supports['channel']], 'time_indices': supports['time'], 'frequency_bands': [meta['bands'][i] for i in supports['frequency']]})
    return result


def _report(config: dict, output: Path) -> dict:
    from .reporting import make_report

    return make_report(config, output)
