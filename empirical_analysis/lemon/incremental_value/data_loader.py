"""Fail-closed provenance audit of the existing unstandardized analysis tensor."""
from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AuditedDataset:
    X: np.ndarray
    y: np.ndarray
    C: np.ndarray
    subject_ids: np.ndarray
    audit: dict
    fingerprints: dict
    metadata: dict
    source_config: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _rows(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _index(rows, column):
    keys = [str(row[column]).strip() for row in rows]
    _require(all(keys) and len(set(keys)) == len(keys), f'Empty or duplicate IDs: {column}')
    return dict(zip(keys, rows))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_audited_dataset(source_dir, tmt_csv, participants_file):
    source = Path(source_dir).resolve()
    paths = {'analysis_dataset': source/'qc/analysis_dataset.npz',
             'manifest': source/'qc/final_subject_manifest.csv',
             'design': source/'qc/covariate_design.json',
             'raw_tensor': source/'tensors/X_unstandardized.npy',
             'metadata': source/'tensors/tensor_meta.json',
             'source_config': source/'config_used.json',
             'original_tmt': Path(tmt_csv).resolve(), 'participants': Path(participants_file).resolve()}
    _require(all(p.is_file() for p in paths.values()), 'Missing required source file')
    with np.load(paths['analysis_dataset'], allow_pickle=False) as f:
        _require({'X','y','C','subject_ids'} <= set(f.files), 'Missing analysis dataset fields')
        X, y, C, ids = f['X'].copy(), f['y'].copy(), f['C'].copy(), f['subject_ids'].astype(str).copy()
    _require(ids.ndim == 1 and len(ids) > 0 and all(s.strip() for s in ids), 'Invalid subject IDs')
    n = len(ids)
    _require(len(set(ids.tolist())) == n, 'Duplicate subject IDs')
    _require(X.ndim == 4 and X.shape[0] == n and y.shape == (n,), 'X/y sample shape mismatch')
    _require(C.ndim == 2 and C.shape[0] == n, 'Covariate sample shape mismatch')
    _require(all(np.isfinite(a).all() for a in (X,y,C)), 'Nonfinite analysis values')
    cfg = _json(paths['source_config'])['config']
    meta, design = _json(paths['metadata']), _json(paths['design'])
    enc = design['encoding']
    _require(enc == cfg['confounds'], 'Saved covariate encoding differs from configuration')
    _require(cfg['response'] == {'column':'TMT_5','primary_transform':'log'}, 'Response must be log(TMT_5)')
    _require(enc['age_encoding'] == 'categorical', 'Covariate age must be categorical')
    rows = _rows(paths['manifest']); _index(rows, 'subject_id')
    included = [r for r in rows if r['include_primary'] == 'True']
    _require([r['subject_id'] for r in included] == ids.tolist(), 'Manifest subject order mismatch')
    indices_f = np.array([float(r['tensor_index']) for r in included])
    _require(np.isfinite(indices_f).all() and np.equal(indices_f,np.floor(indices_f)).all(), 'Noninteger tensor indices')
    indices = indices_f.astype(np.int64)
    raw = np.load(paths['raw_tensor'], mmap_mode='r', allow_pickle=False)
    _require(raw.ndim == 4 and list(raw.shape) == meta['shape'], 'Tensor metadata shape mismatch')
    _require(len(set(indices.tolist())) == n and np.all((indices >= 0) & (indices < len(raw))), 'Invalid tensor indices')
    _require(X.shape[1:] == raw.shape[1:] and np.array_equal(X,raw[indices]), 'X differs from unstandardized source tensor')
    _require(len(meta['channels']) == X.shape[1] and len(set(meta['channels'])) == X.shape[1], 'Channel metadata invalid')
    _require(len(meta['bands']) == X.shape[3] and meta['bands'] == list(cfg['bands']) and
             meta['band_edges_hz'] == cfg['bands'], 'Band metadata order or edges mismatch')
    _require(meta['time_windows'] == X.shape[2] == cfg['events']['n_windows'] and
             meta['window_sec'] == cfg['events']['window_sec'] and
             meta['sfreq'] == cfg['preprocess']['target_sfreq'], 'Time/sampling metadata mismatch')
    _require(meta['feature'] == 'log_absolute_band_power', 'Tensor metadata feature definition mismatch')
    tmt = _index(_rows(paths['original_tmt']), 'ID')
    participants = _index(_rows(paths['participants'], '\t'), enc['id_column'])
    _require(all(s in tmt and s in participants for s in ids), 'Subjects missing from original source tables')
    response = np.array([float(tmt[s]['TMT_5']) for s in ids])
    _require(np.isfinite(response).all() and np.all(response > 0), 'Invalid original response')
    _require(np.allclose(y,np.log(response),rtol=0,atol=1e-12), 'Incorrect log response from original TMT')
    _require(np.allclose(response,[float(r['tmt5_raw']) for r in included],rtol=0,atol=1e-12) and
             np.allclose(y,[float(r['tmt5_log']) for r in included],rtol=0,atol=1e-12), 'Response manifest mismatch')
    categories = enc['age_categories']; ref = enc['age_reference']
    _require(len(set(categories)) == len(categories) and ref in categories, 'Invalid covariate age categories')
    nonref = [a for a in categories if a != ref]
    columns = ['intercept'] + [f'age[{a}]' for a in nonref] + ['sex']
    _require(columns == design['columns'] and C.shape[1] == len(columns), 'Covariate column order mismatch')
    ages, sexes = [], []
    for sid, row in zip(ids, included):
        person = participants[sid]
        age_source, sex_source = person[enc['age_column']], person[enc['sex_column']]
        _require(age_source in enc['age_mapping'] and sex_source in enc['sex_mapping'], 'Unknown covariate code')
        age, sex = enc['age_mapping'][age_source], float(enc['sex_mapping'][sex_source])
        _require(age in categories and row['age_source'] == age_source and row['sex_source'] == sex_source and
                 row['age_category'] == age and float(row['sex']) == sex, 'Covariate manifest/source mismatch')
        ages.append(age); sexes.append(sex)
    expected = np.column_stack([np.ones(n)] + [np.array(ages) == a for a in nonref] + [sexes])
    _require(np.array_equal(C,expected), 'Covariate matrix differs from original participant encoding')
    _require(np.linalg.matrix_rank(C) == C.shape[1], 'Covariate matrix not full rank')
    audit = {'passed':True,'n_subjects':n,'tensor_shape':list(X.shape),
             'original_tensor_shape':list(raw.shape),'covariate_shape':list(C.shape),
             'covariate_columns':columns,'age_category_counts':{a:ages.count(a) for a in categories},
             'response':'log(TMT_5)','X_unstandardized_source_equal':True,
             'subject_manifest_order_equal':True,'response_original_verified':True,
             'covariates_original_verified':True,'metadata_verified':True,
             'feature_order':'numpy_C_order','raw_eeg_regenerated':False}
    fingerprints = {name:{'path':str(p),'sha256':_sha(p)} for name,p in paths.items()}
    return AuditedDataset(X,y,C,ids,audit,fingerprints,meta,cfg)
