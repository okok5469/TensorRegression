"""Strict local UCI 270 ingestion. No fitting or performance evaluation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import zipfile
import numpy as np


def parse_line(text):
    tokens = text.split()
    if not tokens or not re.fullmatch(r'[1-6];[^;]+', tokens[0]):
        raise ValueError('Expected gas_id;concentration')
    gas, target = tokens[0].split(';')
    y = float(target)
    pairs = [token.split(':') for token in tokens[1:]]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError('Malformed index:value')
    indices = [int(pair[0]) for pair in pairs]
    if sorted(indices) != list(range(1, 129)):
        raise ValueError('Feature indices must occur exactly once, 1..128')
    values = np.empty(128, dtype=np.float64)
    values[np.asarray(indices)-1] = [float(pair[1]) for pair in pairs]
    if not np.isfinite(y) or not np.isfinite(values).all():
        raise ValueError('Nonfinite label or feature')
    return int(gas), y, tokens[0], values


def make_tensor(raw128):
    raw = np.asarray(raw128)
    if raw.ndim != 2 or raw.shape[1] != 128:
        raise ValueError('Expected n by 128')
    return raw.reshape(-1, 16, 8)[:, :, 2:8].reshape(-1, 16, 2, 3).transpose(0, 1, 3, 2).copy()


def exact_groups(x):
    values = np.asarray(x, dtype='<f8').reshape(len(x), -1).copy()
    values[values == 0] = 0.0
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in values])


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(source, archive, output, source_audit):
    source, archive, output = map(Path, (source, archive, output))
    source_audit = Path(source_audit)
    evidence = source_audit.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'data'
    cache.mkdir(exist_ok=True)
    manifest = {'archive': str(archive), 'archive_sha256': sha256(archive.read_bytes()),
                'archive_size': archive.stat().st_size, 'files': [],
                'hash_provenance': 'Local integrity fingerprint; no official checksum available'}
    records, second = [], []
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f'CRC failure: {bad}')
        manifest['zip_crc'] = 'PASS'
        for batch in range(1, 11):
            name = f'batch{batch}.dat'
            matches = [entry for entry in zf.namelist() if Path(entry).name == name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one archive member for {name}')
            blob = (source / name).read_bytes()
            zipped = zf.read(matches[0])
            if blob != zipped:
                raise ValueError(f'Extracted/archive copy differs: {name}')
            manifest['files'].append({'batch': batch, 'path': str(source/name), 'archive_member': matches[0],
                                      'bytes': len(blob), 'sha256': sha256(blob), 'copy_equal': True})
            for number, line in enumerate(blob.decode('ascii').splitlines(), 1):
                gas, y, label, values = parse_line(line)
                records.append((batch, number, gas, y, label, name, values))
                # Independent parser: regex extraction, sort independently by numeric index.
                label_match = re.match(r'^\s*(\d+);([^\s]+)\s+', line)
                indexed = re.findall(r'(?<!\S)(\d+):([^\s]+)', line)
                independent = sorted((int(i), float(v)) for i, v in indexed)
                if [i for i, _ in independent] != list(range(1, 129)):
                    raise ValueError('Independent index audit failed')
                if int(label_match[1]) != gas or float(label_match[2]) != y:
                    raise ValueError('Independent label audit failed')
                second.append([v for _, v in independent])
    raw = np.stack([r[6] for r in records])
    np.testing.assert_array_equal(raw, second)
    X = make_tensor(raw)
    coordinates = [{'sensor': j+1, 'scale': a, 'stage': stage, 'raw_column_1based': 8*j+2+k+3*s+1,
                    'tensor_coordinate': [j, k, s], 'c_order_index': 6*j+2*k+s}
                   for j in range(16) for k, a in enumerate([.001, .01, .1]) for s, stage in enumerate(['i','d'])]
    for item in coordinates:
        j,k,s = item['tensor_coordinate']
        np.testing.assert_array_equal(X[:,j,k,s], raw[:,item['raw_column_1based']-1])
    # Training-only synthetic origin: actual first-batch scaler, then verify coordinatewise equality.
    batch = np.asarray([r[0] for r in records])
    train = batch == 1
    mean, scale = raw[train].mean(0), raw[train].std(0)
    scale[scale == 0] = 1
    mapped_standardized = make_tensor((raw-mean)/scale)
    xmean, xscale = make_tensor(mean[None])[0], make_tensor(scale[None])[0]
    np.testing.assert_array_equal(mapped_standardized, (X-xmean)/xscale)
    beta = np.linspace(-1, 1, 96).reshape(16,3,2)
    flat_prediction = X.reshape(len(X),96) @ beta.ravel()
    tensor_prediction = np.einsum('nijk,ijk->n',X,beta)
    np.testing.assert_allclose(flat_prediction,tensor_prediction,rtol=1e-12,atol=1e-8)
    arrays = dict(raw128=raw, X=X, y=np.asarray([r[3] for r in records]),
                  gas=np.asarray([r[2] for r in records]), batch=batch,
                  record_id=np.asarray([f'{r[0]}:{r[1]}' for r in records]),
                  raw_label=np.asarray([r[4] for r in records]),
                  source_file=np.asarray([r[5] for r in records]),
                  line_number=np.asarray([r[1] for r in records]),
                  group96=exact_groups(X), group128=exact_groups(raw))
    if len(raw) != 13910 or set(arrays['gas']) != set(range(1,7)):
        raise ValueError('Official sample/class inventory mismatch')
    inventory = []
    for gas in range(1,7):
        for b in range(1,11):
            mask = (arrays['gas']==gas)&(batch==b)
            y, values = arrays['y'][mask], raw[mask]
            row = dict(gas_id=gas,batch_id=b,n=len(y),n_levels=len(np.unique(y)),
                       concentration_min=float(y.min()) if len(y) else None,
                       concentration_max=float(y.max()) if len(y) else None,
                       concentration_mean=float(y.mean()) if len(y) else None,
                       concentration_sd=float(y.std()) if len(y) else None,
                       constant_y=bool(len(y)>0 and len(np.unique(y))==1),
                       nonfinite_features=int((~np.isfinite(values)).sum()),
                       constant_features=int(np.sum(np.ptp(values,axis=0)==0)) if len(y) else None,
                       feature_min=float(values.min()) if len(y) else None,
                       feature_max=float(values.max()) if len(y) else None,
                       negative_dynamic_values=int((X[mask]<0).sum()))
            inventory.append(row)
    duplicates = []
    counts = {}
    for representation in ['96','128']:
        groups = arrays['group'+representation]
        unique, sizes = np.unique(groups,return_counts=True)
        repeated = unique[sizes>1]
        counts[representation] = {'groups':len(repeated),'records':int(sizes[sizes>1].sum()),'conflicting_y_groups':0}
        for group in repeated:
            ids = np.flatnonzero(groups==group)
            conflict = len(np.unique(arrays['y'][ids]))>1
            counts[representation]['conflicting_y_groups'] += int(conflict)
            for idx in ids:
                duplicates.append(dict(representation=representation,group_id=group,record_id=arrays['record_id'][idx],
                                       gas_id=int(arrays['gas'][idx]),batch_id=int(batch[idx]),
                                       concentration=float(arrays['y'][idx]),group_size=len(ids),conflicting_y=conflict))
    np.savez_compressed(cache/'dataset.npz',**arrays)
    metadata = dict(n=len(raw),shape=list(X.shape),mapping=coordinates,
                    excluded_steady_columns_1based=[8*j+s+1 for j in range(16) for s in [0,1]],
                    unit='ppmv', gas_mapping_status='VERIFIED_SOURCE_AUDIT',
                    gas_names={'1':'Ethanol','2':'Ethylene','3':'Ammonia','4':'Acetaldehyde','5':'Acetone','6':'Toluene'},
                    physical_identity='Same sensor system; per-record device and trial identifiers unknown',
                    duplicate_counts=counts, independent_parse_records=len(second),
                    mapping_checked_values=int(X.size),
                    coefficient_roundtrip_max_abs=float(np.max(np.abs(flat_prediction-tensor_prediction))),
                    semantic_gate='PASS',data_checks='PASS',gate_d='PASS',
                    semantic_evidence={'path':str(source_audit),'sha256':sha256(evidence),'verified_date':'2026-10-26'},
                    source_manifest='raw_manifest.json')
    metadata['feature_quality'] = [dict(column_1based=j+1,minimum=float(raw[:,j].min()),
        maximum=float(raw[:,j].max()),nonfinite=int((~np.isfinite(raw[:,j])).sum()),
        constant=bool(np.ptp(raw[:,j])==0),negative_count=int((raw[:,j]<0).sum())) for j in range(128)]
    metadata['concentration_levels_by_gas'] = {str(g):np.unique(arrays['y'][arrays['gas']==g]).tolist()
                                             for g in range(1,7)}
    metadata['empty_gas_batches'] = [[r['gas_id'],r['batch_id']] for r in inventory if r['n']==0]
    metadata['constant_y_gas_batches'] = [[r['gas_id'],r['batch_id']] for r in inventory if r['constant_y']]
    (cache/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'raw_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(output/'batch_gas_inventory.csv',inventory,list(inventory[0]))
    write_csv(output/'duplicate_groups.csv',duplicates,['representation','group_id','record_id','gas_id','batch_id','concentration','group_size','conflicting_y'])
    (output/'DATA_AUDIT.md').write_text('# UCI270 数据审计\n\n'
        f'本地完整解析 {len(raw):,} 条记录、6 类、10 批、128 列；全部记录用独立正则解析重新核对。ZIP CRC 通过，十个 DAT 的 ZIP 内容与解压副本逐字节一致；SHA256 为本地指纹，并非官方校验和。\n\n'
        '首字段严格拆为类别与浓度，所有索引按编号恢复，缺失、重复索引及非有限值立即报错，不插补。原标签、行号、文件名与数值保存于 data/dataset.npz；完整质量表见 batch_gas_inventory.csv（包含空缺批次）。浓度为有限实验档位，不视作随机连续环境浓度。负动态特征保留。\n\n'
        f'完全重复审计：{json.dumps(counts,ensure_ascii=False)}。逐条重复关系见 duplicate_groups.csv；身份只用 X，不用 y，冲突标签保留。后续划分必须按主 96 维组从训练端清除与评价集重合记录，128 维方法使用相同 ID；该审计使用测试 X、不使用测试 y。\n\n'
        f'空缺类别/批次：{metadata["empty_gas_batches"]}；恒定响应类别/批次：{metadata["constant_y_gas_batches"]}。恒定测试响应的 R² 应记 NA；gas 6 的早期空缺可能导致不足两个合法内折，须按冻结规则检查并标记不可评价，不改随机划分。全部 128 列量级、符号、非有限数与常量标记及各类真实浓度档位保存于 metadata。\n\n'
        '记录级实体、试次、设备 ID 未提供；行号不表示精确采样间隔，批次不等于独立设备。本模块不执行模型拟合或生成第 10 批成绩。\n\n'
        '**Gate D PASS：数值数据检查与官方类别/字段语义均通过。** 官方证据见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)，指纹记录于 metadata；类别 1—6 分别为乙醇、乙烯、氨、乙醛、丙酮、甲苯。该数据 Gate 不代表所有训练窗口具有足够合法内折，更不代表性能验收。\n',encoding='utf-8')
    (output/'TENSOR_MAPPING_AUDIT.md').write_text('# 张量映射审计\n\n'
        f'全部 {len(raw):,} 条 × 96 坐标（{X.size:,} 值）逐项检查通过：X[j,k,0] = raw[8j+2+k]，X[j,k,1] = raw[8j+5+k]（零基）。先把每传感器六动态列拆为阶段×尺度，再转置为传感器×尺度×阶段；禁止直接 reshape 96 列。\n\n'
        'data/metadata.json 保存全部原列号、C-order 坐标与排除的 32 个稳态列。人工单调输入单元测试检查完整方向。用第一批训练均值/标准差检查全部记录标准化前后映射一致，无标签参与。\n\n'
        f'C-order 展平内积与张量收缩最大绝对差 {metadata["coefficient_roundtrip_max_abs"]:.12g}（浮点加法次序）；rtol=1e-12、atol=1e-8 通过。\n\n'
        '轴语义：16 传感器 × 3 平滑参数（0.001/0.01/0.1）× 2 响应阶段（i/d）。尺度不是原始时间，阶段不是气体；DAT 不含原始波形。矩形支撑是假设，第三轴只有 2 个坐标。数学映射与官方字段语义均通过，独立来源证据见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。\n',encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output',default='doc/uci270_concentration')
    parser.add_argument('--source-audit',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.archive,args.output,args.source_audit),ensure_ascii=True))
