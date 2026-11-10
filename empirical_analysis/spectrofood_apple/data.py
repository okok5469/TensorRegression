"""Strict ID-based SpectroFood Apple ingestion and fixed geometry compression."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def read_apple_csv(path):
    records, wavelengths, active = [], None, False
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for number, raw in enumerate(csv.reader(stream), 1):
            row = [value.strip() for value in raw]
            while row and not row[-1]:
                row.pop()
            if not row:
                continue
            if len(row) > 1 and row[1].lower() == 'dry matter':
                active = row[0].lower() == 'apple'
                if active:
                    current = np.asarray(row[2:], float)
                    if wavelengths is not None and not np.array_equal(current, wavelengths):
                        raise ValueError('Apple repeated header wavelengths conflict')
                    wavelengths = current
                continue
            if not active:
                continue
            if len(row) != len(wavelengths) + 2:
                raise ValueError(f'Apple CSV malformed row {number}')
            records.append((row[0], float(row[1]), np.asarray(row[2:], float)))
    if not records:
        raise ValueError('Apple CSV segment is missing')
    ids = [row[0] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Apple CSV duplicate sample IDs')
    values = np.asarray([row[1] for row in records])
    spectra = np.stack([row[2] for row in records])
    if not all(np.isfinite(a).all() for a in (wavelengths, values, spectra)):
        raise ValueError('Apple CSV nonfinite values')
    return ids, values, wavelengths, spectra


def _area_weights(source, target):
    edges = np.linspace(0, source, target + 1)
    pixels = np.arange(source)
    overlaps = np.maximum(0, np.minimum(edges[1:, None], pixels + 1)
                          - np.maximum(edges[:-1, None], pixels))
    return overlaps / (source / target)


def area_average(cube, shape):
    """Integrate unit pixel areas over equal-sized target cells, without interpolation."""
    h, w = shape
    return np.einsum('ah,hwc,bw->abc', _area_weights(cube.shape[0], h),
                     cube, _area_weights(cube.shape[1], w), optimize=True)


def bin_wavelengths(wavelengths, count=24):
    """Merge equal wavelengths first; average adjacent, nearly equal-count groups."""
    unique = np.unique(wavelengths)
    groups = np.array_split(unique, min(count, len(unique)))
    mapping = []
    for index, group in enumerate(groups):
        source = np.flatnonzero(np.isin(wavelengths, group))
        weights = np.asarray([1 / len(group) / np.count_nonzero(wavelengths == wavelengths[i])
                              for i in source])
        mapping.append(dict(group=index, source_indices=source.tolist(), weights=weights.tolist(),
                            min_nm=float(group.min()), max_nm=float(group.max()),
                            representative_nm=float(group.mean())))
    return np.asarray([entry['representative_nm'] for entry in mapping]), mapping


def _file_identity(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            md5.update(block)
            sha.update(block)
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
                md5=md5.hexdigest(), sha256=sha.hexdigest())


def audit_and_prepare(mat_path, csv_path, out_dir, *, expected_md5=None):
    """Return (audit, paths), writing only complete, validated deterministic inputs.

    An empty expected_md5 mapping is reserved for synthetic fixtures; the default
    enforces both immutable Zenodo v1 file identities.
    """
    mat_path, csv_path, out_dir = map(Path, (mat_path, csv_path, out_dir))
    expected = {'mat': 'fcbf9d492422a2283a68b0d6739378b2',
                'csv': '15327c7fb3d5fe10231735a286629715'} if expected_md5 is None else expected_md5
    identities = {'mat': _file_identity(mat_path), 'csv': _file_identity(csv_path)}
    for kind, digest in expected.items():
        if identities[kind]['md5'] != digest:
            raise ValueError(f'{kind} official MD5 mismatch')
    mat = loadmat(mat_path)
    fields = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
              for key, value in mat.items() if not key.startswith('__')}
    waves = np.asarray(mat['Wavelengths'], float).ravel()
    labels = [str(value).strip() for value in mat['Labels'].ravel()]
    images = {key: value for key, value in mat.items() if re.fullmatch(r'A[1-9]\d*', key)}
    ids_csv, raw_y, csv_waves, csv_spectra = read_apple_csv(csv_path)
    if len(set(labels)) != len(labels):
        raise ValueError('MAT duplicate sample IDs')
    if not (set(labels) == set(images) == set(ids_csv)):
        raise ValueError(f'ID mismatch: labels-only={sorted(set(labels)-set(images))}; '
                         f'images-without-response={sorted(set(images)-set(ids_csv))}; '
                         f'responses-without-image={sorted(set(ids_csv)-set(images))}')
    if expected and len(images) != 240:
        raise ValueError('Official Apple file must contain 240 samples')
    if not np.isfinite(waves).all() or len(np.unique(waves)) < 1:
        raise ValueError('Invalid wavelength axis')
    if not ((raw_y >= 0) & (raw_y <= 1)).all():
        raise ValueError('Response conflicts with audited fraction encoding')
    for name, cube in images.items():
        if cube.ndim != 3 or cube.shape[2] != len(waves) or min(cube.shape[:2]) < 2:
            raise ValueError(f'{name}: incompatible two-spatial-axis cube {cube.shape}')
        if not np.isfinite(cube).all():
            raise ValueError(f'{name}: nonfinite cube values')
    sample_ids = sorted(images, key=lambda key: int(key[1:]))
    target = (min(8, min(c.shape[0] for c in images.values())),
              min(8, min(c.shape[1] for c in images.values())))
    binned_waves, bands = bin_wavelengths(waves)
    unique_waves = np.unique(waves)
    native_groups = [np.flatnonzero(waves == wave) for wave in unique_waves]
    tensors, native, rows, spatial, cube_hashes, entities = [], [], [], [], {}, []
    lookup = {sid: index for index, sid in enumerate(ids_csv)}
    duplicates = []
    for index, sid in enumerate(sample_ids):
        cube = images[sid]
        sha = hashlib.sha256(str(cube.shape).encode() + cube.dtype.str.encode()
                             + cube.tobytes(order='C')).hexdigest()
        entity = cube_hashes.setdefault(sha, sid)
        if entity != sid and raw_y[lookup[entity]] != raw_y[lookup[sid]]:
            raise ValueError(f'duplicate cube {entity}/{sid} has conflicting response')
        if entity != sid:
            duplicates.append([entity, sid])
        entities.append(entity)
        compressed = area_average(cube, target)
        tensors.append(np.stack([compressed[..., b['source_indices']] @ b['weights'] for b in bands], -1))
        mean = cube.mean((0, 1))
        native.append([mean[group].mean() for group in native_groups])
        y_raw = float(raw_y[lookup[sid]])
        rows.append(dict(sample_id=sid, cube_index=labels.index(sid), source_shape='x'.join(map(str, cube.shape)),
                         y_raw=y_raw, y_pct=100*y_raw, entity_id=entity, group_id=entity,
                         batch_id_if_available='', qc_status='PASS', exclusion_reason='', cube_sha256=sha,
                         zero_fraction=float(np.mean(cube == 0)), min_raw=int(cube.min()), max_raw=int(cube.max())))
        spatial.append(dict(sample_id=sid, source_shape=list(cube.shape), crop=[0, cube.shape[0], 0, cube.shape[1]],
                            target_shape=list(target), row_edges=np.linspace(0,cube.shape[0],target[0]+1).tolist(),
                            column_edges=np.linspace(0,cube.shape[1],target[1]+1).tolist(),
                            row_weights=_area_weights(cube.shape[0],target[0]).tolist(),
                            column_weights=_area_weights(cube.shape[1],target[1]).tolist(),
                            valid_area_per_cell=float(cube.shape[0]*cube.shape[1]/np.prod(target))))
    X = np.asarray(tensors)
    native = np.asarray(native)
    csv_ordered = csv_spectra[[lookup[sid] for sid in sample_ids]]
    common = np.intersect1d(unique_waves, csv_waves)
    corr = [float(np.corrcoef(native[i, np.searchsorted(unique_waves,common)],
                            csv_ordered[i, [int(np.flatnonzero(csv_waves==w)[0]) for w in common]])[0,1])
            for i in range(len(sample_ids))] if len(common) > 1 else []
    audit = dict(status='PREPARED', input_files=identities, mat_format='MATLAB 5.0', field_tree=fields,
                 n_samples=len(sample_ids), n_entities=len(set(entities)), X_shape=list(X.shape),
                 axis_order=['sample','height','width','wavelength_group'], flatten_order='C',
                 wavelength_count=len(waves), wavelength_range_nm=[float(waves.min()),float(waves.max())],
                 wavelength_step_nm=np.diff(unique_waves).tolist(), wavelength_duplicate_count=int(len(waves)-len(unique_waves)),
                 csv_wavelength_count=len(csv_waves), common_wavelength_count=len(common),
                 csv_mat_shape_correlations=corr, duplicate_images=duplicates, excluded_samples=[],
                 response=dict(raw_unit='mass fraction', model_unit='percentage points', conversion='y_pct=100*y_raw',
                               raw_range=[float(raw_y.min()),float(raw_y.max())],
                               evidence='Original data paper Table 3: Apple DMC min 14%, max 17%, mean 15%; CSV is fraction-encoded.',
                               source='https://pmc.ncbi.nlm.nih.gov/articles/PMC10823050/'),
                 entity_assumption='Sample ID used as entity except exact duplicate cubes grouped. No batch or physical-fruit metadata; absence of duplicate evidence is not proof of independence.',
                 roi_rule='Entire supplied rectangular ROI including zeros; no explicit mask/background semantics supplied.',
                 coordinate_limitation='Relative ROI coordinates do not establish biological correspondence across fruit.')
    mapping = dict(spatial=spatial, wavelength_groups=bands, original_wavelengths_nm=waves.tolist(),
                   native_wavelengths_nm=unique_waves.tolist(), original_axis_order=['height','width','wavelength'],
                   axis_permutation=[0,1,2], normalized_coordinates='pixel edges divided by original axis length')
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: str(out_dir / filename) for key, filename in
             [('data','data.npz'),('mapping','data_mapping.json'),('manifest','sample_manifest.csv'),('audit','data_audit.json')]}
    np.savez_compressed(paths['data'], X=X, spectrum_binned=X.mean((1,2)), spectrum_native=native,
                        y_raw=np.asarray([r['y_raw'] for r in rows]), y_fraction=np.asarray([r['y_raw'] for r in rows]),
                        y_pct=np.asarray([r['y_pct'] for r in rows]), sample_ids=np.asarray(sample_ids),
                        entity_ids=np.asarray(entities), wavelengths=unique_waves, wavelengths_binned=binned_waves,
                        spectrum_csv=csv_ordered, wavelengths_csv=csv_waves)
    for key, value in [('mapping',mapping),('audit',audit)]:
        Path(paths[key]).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with Path(paths['manifest']).open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return audit, paths
