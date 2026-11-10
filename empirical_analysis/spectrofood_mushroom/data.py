"""ID-verified, streamed Mushroom tensor preparation; no learned preprocessing."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from empirical_analysis.mushroom_screen import read_segment, spectrum_groups
from empirical_analysis.spectrofood_apple.data import area_average, bin_wavelengths


def canonical_id(value):
    match=re.fullmatch(r'M(\d{3})',value)
    if match is None or int(match[1]) < 1:
        raise ValueError(f'Invalid explicit Mushroom ID {value!r}')
    return f'M{int(match[1])}'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def file_identity(path):
    md5,sha=hashlib.md5(),hashlib.sha256()
    before=path.stat()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            md5.update(chunk);sha.update(chunk)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'Input changed while hashing: {path}')
    return dict(path=str(path.resolve()),name=path.name,bytes=after.st_size,mtime_ns=after.st_mtime_ns,
                md5=md5.hexdigest(),sha256=sha.hexdigest())


def prepare(data_dir,out_dir,*,expected_md5=None,expected_count=250):
    """Validate official inputs afresh and save the fixed tensor and original CSV spectra."""
    data_dir,out_dir=Path(data_dir),Path(out_dir)
    official=dict(zip([f'Mushroom_{i}.mat' for i in range(1,6)],
        ['2aa6480f051b34f8adc35d46d2e574a1','8c5db150def4c072639eac4d7c5403fa',
         '5fa29279aeb555a1594c7e1ad073bf19','a336e2a56f19449c4e328aaabdd4a720','dbdbdda6f9f6d1fa14c75f2ddb89513d']))
    official['SpectroFood_dataset.csv']='15327c7fb3d5fe10231735a286629715'
    expected=official if expected_md5 is None else expected_md5
    files=sorted(data_dir.glob('Mushroom_*.mat'))
    names={p.name for p in files}|{'SpectroFood_dataset.csv'}
    if expected and names!=set(expected):
        raise ValueError('Missing or extra official files')
    identities=[file_identity(p) for p in files+[data_dir/'SpectroFood_dataset.csv']]
    for item in identities:
        if item['name'] in expected and item['md5']!=expected[item['name']]:
            raise ValueError(f'Official MD5 mismatch: {item["name"]}')
    ids,y,waves,native,csv_rows=read_segment(data_dir/'SpectroFood_dataset.csv')
    if set(ids)!={f'M{i}' for i in range(1,expected_count+1)} or len(ids)!=expected_count:
        raise ValueError('CSV ID set mismatch')
    lookup={sid:i for i,sid in enumerate(ids)}
    headers=[];all_ids=[];spatial_shapes=[]
    for path in files:
        fields=whosmat(path)
        cubes={name:shape for name,shape,kind in fields if re.fullmatch(r'M\d{3}',name)}
        meta=loadmat(path,variable_names=['Labels','Wavelengths'])
        labels=[str(value).strip() for value in meta['Labels'].ravel()]
        if len(set(labels))!=len(labels) or set(labels)!=set(cubes):
            raise ValueError(f'Labels/image ID mismatch: {path.name}')
        if not np.array_equal(meta['Wavelengths'].ravel(),waves):
            raise ValueError(f'MAT/CSV wavelength mismatch: {path.name}')
        for key,shape in cubes.items():
            if len(shape)!=3 or shape[2]!=len(waves) or min(shape[:2])<2:
                raise ValueError(f'Invalid spatial/wavelength axes: {key}')
            spatial_shapes.append(shape[:2]);all_ids.append(canonical_id(key))
        headers.append((path,labels,cubes))
    if len(all_ids)!=expected_count or len(set(all_ids))!=expected_count or set(all_ids)!=set(ids):
        raise ValueError('MAT/CSV ID mapping is not a bijection')
    target=tuple(map(int,np.minimum(8,np.min(np.asarray(spatial_shapes),axis=0))))
    wb,bands=bin_wavelengths(waves,24)
    tensor=np.empty((len(ids),*target,len(bands)));means=np.empty_like(native)
    blocks=np.empty(len(ids),int);manifest=[];spatial=[];hash_ids={}
    for path,labels,cubes in headers:
        block=int(re.fullmatch(r'Mushroom_(\d+)\.mat',path.name)[1])
        for key in labels:
            sid=canonical_id(key);index=lookup[sid]
            cube=loadmat(path,variable_names=[key])[key]
            nonfinite=int(np.count_nonzero(~np.isfinite(cube)))
            if nonfinite:
                raise ValueError(f'Nonfinite cube values: {key} ({nonfinite})')
            sha=hashlib.sha256(str(cube.shape).encode()+cube.dtype.str.encode()+cube.tobytes(order='C')).hexdigest()
            hash_ids.setdefault(sha,[]).append(sid)
            compressed=area_average(cube,target)
            tensor[index]=np.stack([compressed[...,b['source_indices']]@b['weights'] for b in bands],axis=-1)
            means[index]=cube.mean((0,1));blocks[index]=block
            manifest.append(dict(sample_id=sid,mat_field=key,source_file=path.name,label_index=labels.index(key),
                csv_row=csv_rows[index],block=block,source_shape='x'.join(map(str,cube.shape)),dtype=str(cube.dtype),
                y_raw=float(y[index]),y_pct=float(100*y[index]),minimum=float(cube.min()),maximum=float(cube.max()),
                nonfinite_count=nonfinite,zero_fraction=float(np.mean(cube==0)),
                all_zero_pixel_fraction=float(np.mean(np.all(cube==0,axis=2))),cube_sha256=sha,qc_status='PASS'))
            spatial.append(dict(sample_id=sid,source_shape=list(cube.shape),crop=[0,cube.shape[0],0,cube.shape[1]],
                row_edges=np.linspace(0,cube.shape[0],target[0]+1).tolist(),
                column_edges=np.linspace(0,cube.shape[1],target[1]+1).tolist(),
                valid_area_per_cell=float(cube.shape[0]*cube.shape[1]/np.prod(target)),
                pixel_weight_rule='pixel overlap area divided by target cell area; all pixels valid'))
            del cube
    groups=spectrum_groups(native)
    duplicates=[dict(sample_ids=v,conflicting_response=len(set(y[[lookup[s] for s in v]]))>1)
                for v in hash_ids.values() if len(v)>1]
    spectral_duplicates=[dict(sample_ids=[ids[i] for i in np.flatnonzero(groups==group)],
        y_pct=(100*y[groups==group]).tolist(),blocks=blocks[groups==group].tolist())
        for group in np.unique(groups) if np.count_nonzero(groups==group)>1]
    metadata=dict(status='PREPARED',n_samples=len(ids),tensor_shape=list(tensor.shape),
        raw_file_identity=identities,axis_order=['sample','height','width','wavelength_group'],flatten_order='C',
        roi_rule='Full supplied rectangular ROI including zeros; no new threshold segmentation',
        native_definition='Original official CSV 204-band spectrum, not rectangular MAT mean',
        response_unit='percentage points; y_pct=100*CSV_mass_fraction',
        duplicate_cubes=duplicates,duplicate_spectra=spectral_duplicates,
        duplicate_policy='Retain every record and conflicting responses; spectrum SHA defines random-CV leakage groups',
        blocks_definition='Actual source MAT files, storage blocks; no documented biological batch semantics',
        spatial_mapping=spatial,wavelength_groups=bands,
        original_wavelengths_nm=waves.tolist(),binned_wavelengths_nm=wb.tolist(),
        representation=dict(target_shape=list(target)+[len(bands)],spatial='area weighted',
                            spectral='contiguous nearly equal-count groups, arithmetic means'),
        limitations=['Physical entity and acquisition-batch identities unavailable',
                     'Independent cache reconstruction remains a separate required check'])
    out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_dir/'data.npz',ids=np.asarray(ids),X_tensor=tensor,X_native=native,y_pct=100*y,
                        groups=groups,blocks=blocks,wavelengths=waves,X_mat_mean=means,wavelengths_binned=wb)
    write_json(out_dir/'tensor_metadata.json',metadata)
    write_json(out_dir/'raw_sample_metadata.json',manifest)
    write_json(out_dir/'wavelength_groups.json',bands)
    with (out_dir/'sample_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (out_dir/'MUSHROOM_DATA_AUDIT.md').write_text(
        '# Mushroom data audit\n\nStatus: PREPARED. All official files validated by full MD5/SHA256 pass.\n\n'
        f'{len(ids)} records retained; explicit MAT Labels/fields map to CSV by decimal ID, never row order. '
        f'All cubes finite. Exact cube duplicate groups: {len(duplicates)}; CSV duplicate groups: {len(spectral_duplicates)}. '
        'Conflicting responses are retained and documented in tensor_metadata.json.\n\n'
        'Response: official CSV mass fraction multiplied by 100; percentage-point errors. '
        'Physical entity/batch identity is unavailable; blocks identify original files only. '
        'Full numeric QC and source hashes: raw_sample_metadata.json and tensor_metadata.json.\n',encoding='utf-8')
    (out_dir/'TENSOR_CONSTRUCTION_REPORT.md').write_text(
        '# Fixed Mushroom tensor construction\n\n'
        f'Status: PREPARED; shape {list(tensor.shape)}. Axes: record, height, width, wavelength group; C-order flattening. '
        'Preserve original rectangular ROI including zeros; area-weighted equal-cell spatial averages. '
        'Sorted original wavelengths split into 24 contiguous nearly equal-count groups and averaged. '
        'No response-guided ROI/band choices and no cross-record statistics.\n\n'
        'X_native retains original official CSV spectra; X_mat_mean is independently computed from full MAT ROI. '
        'Spatial edges and valid area are in tensor_metadata.json; pixel weights are exact overlap area / cell area. '
        'Relative ROI positions do not establish anatomical equivalence. Independent reconstruction is required before models.\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();result=prepare(args.data_dir,args.out_dir)
    print(f'PREPARED {result["n_samples"]} records; {len(result["duplicate_cubes"])} duplicate cube groups')
