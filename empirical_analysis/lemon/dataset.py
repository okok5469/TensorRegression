"""Input inventory, auditable TMT cleaning and explicit subject joins."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def inventory_eeg(eeg_root: Path, manifest_file: Path | None = None) -> pd.DataFrame:
    if not eeg_root.is_dir():
        raise ValueError(f'EEG directory does not exist: {eeg_root}')
    items = [] if manifest_file is None else json.loads(manifest_file.read_text(encoding='utf-8-sig'))
    sizes = {item['path']: int(item['size']) for item in items}
    subjects = {path.name for path in eeg_root.glob('sub-*') if path.is_dir()}
    subjects.update(key.split('/')[0] for key in sizes if key.startswith('sub-'))
    suffixes = {'vhdr': '_eeg.vhdr', 'vmrk': '_eeg.vmrk', 'eeg': '_eeg.eeg', 'events_tsv': '_events.tsv', 'events_json': '_events.json', 'channels_tsv': '_channels.tsv', 'eeg_json': '_eeg.json'}
    rows = []
    for subject in sorted(subjects):
        row = {'subject_id': subject}
        reasons = []
        files = {}
        for kind, suffix in suffixes.items():
            matches = list((eeg_root / subject / 'eeg').glob('*' + suffix))
            # A non-BIDS *_eeg_events.tsv does not replace the task events table.
            if kind == 'events_tsv':
                matches = [path for path in matches if not path.name.endswith('_eeg_events.tsv')]
            present = len(matches) == 1
            row['has_' + kind] = present
            if not present:
                reasons.append(('missing_' if not matches else 'ambiguous_') + kind)
                continue
            path = matches[0]
            files[kind] = path
            rel = path.relative_to(eeg_root).as_posix()
            expected_size = sizes.get(rel)
            if path.stat().st_size == 0 or (expected_size is not None and path.stat().st_size != expected_size):
                reasons.append('size_mismatch_' + kind)
        for kind, keys in {'vhdr': ('DataFile', 'MarkerFile'), 'vmrk': ('DataFile',)}.items():
            if kind not in files:
                continue
            content = files[kind].read_text(encoding='utf-8-sig')
            for key in keys:
                match = re.search(r'^' + key + r'=(.+)$', content, re.MULTILINE)
                if match is None or not (files[kind].parent / match.group(1).strip()).is_file():
                    reasons.append('broken_' + kind + '_' + key)
        row['file_complete'] = not reasons
        row['exclusion_reason'] = ';'.join(reasons)
        rows.append(row)
    if not rows:
        raise ValueError('No sub-* EEG directories or manifest subjects found')
    return pd.DataFrame(rows)


def prepare_tmt(path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {'ID', 'TMT_5'}.issubset(raw.columns):
        raise ValueError('TMT requires ID and TMT_5 columns')
    duplicate = raw[raw.ID.duplicated(keep=False)]
    if not duplicate.empty or raw.ID.str.strip().eq('').any():
        duplicate.to_csv(output_dir / 'duplicate_tmt_ids.csv', index=False)
        raise ValueError('TMT IDs must be nonempty and unique; no automatic deduplication')
    values = pd.to_numeric(raw.TMT_5, errors='coerce')
    invalid = ~np.isfinite(values)
    pd.DataFrame({'subject_id': raw.ID[invalid], 'original_value': raw.TMT_5[invalid]}).to_csv(output_dir / 'invalid_tmt5_rows.csv', index=False)
    if values[~invalid].le(0).any():
        raw.loc[values.le(0)].to_csv(output_dir / 'nonpositive_tmt5_rows.csv', index=False)
        raise ValueError('TMT_5 must be positive for the log response')
    values = values.where(~invalid, np.nan)
    clean = pd.DataFrame({'subject_id': raw.ID, 'original_value': raw.TMT_5, 'tmt5_raw': values, 'tmt5_log': np.log(values)})
    clean.to_csv(output_dir / 'tmt_clean.csv', index=False)
    summary = clean[['tmt5_raw', 'tmt5_log']].agg(['count', 'mean', 'std', 'median', 'min', 'max', 'skew']).T
    summary.to_csv(output_dir / 'tmt_descriptive_statistics.csv', index_label='response')
    return clean


def merge_analysis_arrays(X: np.ndarray, subject_ids: list[str], tmt: pd.DataFrame, participants: pd.DataFrame | None, config: dict) -> dict:
    X = np.asarray(X, dtype=float)
    if X.ndim != 4 or X.shape[0] != len(subject_ids) or len(set(subject_ids)) != len(subject_ids) or not np.isfinite(X).all():
        raise ValueError('Tensor shape, subject IDs or finite values are invalid')
    if tmt.subject_id.duplicated().any():
        raise ValueError('TMT subject IDs must be unique')
    manifest = pd.DataFrame({'subject_id': subject_ids, 'tensor_index': np.arange(len(subject_ids))}).merge(tmt, on='subject_id', how='left', validate='one_to_one', sort=False)
    mode = config['analysis']['mode']
    if mode not in {'adjusted', 'pilot'}:
        raise ValueError('analysis.mode must explicitly be adjusted or pilot')
    manifest['exclusion_reason'] = np.where(np.isfinite(manifest.tmt5_raw) & (manifest.tmt5_raw > 0), '', 'invalid_or_missing_tmt5')
    covariates = None
    if mode == 'adjusted':
        if participants is None:
            raise ValueError('Adjusted analysis requires participants data and explicit age/sex coding')
        spec = config['confounds']
        columns = [spec['id_column'], spec['age_column'], spec['sex_column']]
        if not set(columns).issubset(participants.columns) or not spec.get('sex_mapping'):
            raise ValueError('Configure participants columns and explicit sex_mapping from its codebook')
        people = participants[columns].rename(columns=dict(zip(columns, ['subject_id', 'age_source', 'sex_source'])))
        if people.subject_id.duplicated().any():
            raise ValueError('Duplicate participants IDs; no automatic deduplication')
        manifest = manifest.merge(people, on='subject_id', how='left', validate='one_to_one', sort=False)
        manifest['age'] = pd.to_numeric(manifest.age_source, errors='coerce')
        nonempty_age = manifest.age_source.notna() & ~manifest.age_source.astype(str).str.strip().str.lower().isin({'', 'n/a', 'na', 'nan'})
        if (nonempty_age & ~np.isfinite(manifest.age)).any():
            raise ValueError('Non-numeric age values (including age bands) require an explicitly reviewed encoding; no midpoint conversion')
        sex_map = {str(key): float(value) for key, value in spec['sex_mapping'].items()}
        if not np.isfinite(list(sex_map.values())).all() or len(set(sex_map.values())) != len(sex_map):
            raise ValueError('sex_mapping must contain distinct finite codes')
        manifest['sex'] = manifest.sex_source.astype(str).map(sex_map)
        nonempty_sex = manifest.sex_source.notna() & ~manifest.sex_source.astype(str).str.strip().str.lower().isin({'', 'n/a', 'na', 'nan'})
        if (nonempty_sex & manifest.sex.isna()).any():
            raise ValueError('Unknown sex values: verify the explicit sex_mapping against the codebook')
        bad = ~np.isfinite(manifest[['age', 'sex']]).all(axis=1)
        manifest.loc[bad, 'exclusion_reason'] = manifest.loc[bad, 'exclusion_reason'].map(lambda value: (value + ';invalid_or_missing_covariates').lstrip(';'))
        covariates = np.column_stack([np.ones(len(manifest)), manifest.age, manifest.sex])
    included = manifest.exclusion_reason.eq('').to_numpy()
    manifest['include_primary'] = included
    transform = config['response']['primary_transform']
    if transform not in {'log', 'raw'}:
        raise ValueError('response.primary_transform must be log or raw')
    return {'X': X[included], 'y': manifest.loc[included, 'tmt5_' + transform].to_numpy(float), 'C': None if covariates is None else covariates[included], 'subject_ids': manifest.loc[included, 'subject_id'].tolist(), 'manifest': manifest, 'pilot_unadjusted': mode == 'pilot'}
