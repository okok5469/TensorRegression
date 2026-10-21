"""Explicit, deterministic upload bundle and standard-library integrity check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe_path(root, name):
    relative = PurePosixPath(name)
    _require(name and '\\' not in name and ':' not in name and not relative.is_absolute()
             and '..' not in relative.parts, 'Unsafe payload path: ' + name)
    path = (root / name).resolve()
    _require(path.is_relative_to(root), 'Payload escaped bundle root')
    return path


def write_bundle(repo, paths, output_zip):
    repo, output_zip = Path(repo).resolve(), Path(output_zip)
    names = sorted(set(paths))
    _require(bool(names), 'Empty upload manifest')
    _require(not {'FILE_MANIFEST.json', 'SHA256SUMS'}.intersection(names), 'Reserved manifest path')
    payload = {}
    for name in names:
        path = _safe_path(repo, name)
        _require(path.is_file(), 'Missing payload: ' + name)
        payload[name] = path.read_bytes()
    files = {name: {'size': len(data), 'sha256': _sha(data)} for name, data in payload.items()}
    manifest = {'schema_version': 1, 'version': 'v5.2-fs-research', 'research_only': True,
                'release_approved': False, 'file_count': len(files), 'files': files,
                'scope': 'Explicit upload payload only; manifest and checksum file excluded from files mapping.'}
    payload['FILE_MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    payload['SHA256SUMS'] = ''.join(_sha(data) + '  ' + name + '\n' for name, data in sorted(payload.items())).encode('utf-8')
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo('taskcard12_bundle/' + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    return {'path': str(output_zip), 'size': output_zip.stat().st_size,
            'sha256': _sha(output_zip.read_bytes()), 'payload_files': len(files)}


def verify_bundle(root):
    root = Path(root).resolve()
    manifest_path = root / 'FILE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _require(manifest['schema_version'] == 1 and manifest['file_count'] == len(manifest['files']),
             'Bundle manifest schema/count differs')
    checksums = {}
    for name, expected in manifest['files'].items():
        path = _safe_path(root, name)
        _require(path.is_file(), 'Bundle payload missing: ' + name)
        data = path.read_bytes()
        _require(len(data) == expected['size'] and _sha(data) == expected['sha256'],
                 'Bundle payload changed: ' + name)
        checksums[name] = expected['sha256']
    checksums['FILE_MANIFEST.json'] = _sha(manifest_path.read_bytes())
    expected_sums = ''.join(digest + '  ' + name + '\n' for name, digest in sorted(checksums.items()))
    _require((root / 'SHA256SUMS').read_text(encoding='utf-8') == expected_sums, 'Bundle checksum list differs')
    return {'passed': True, 'files_checked': len(manifest['files']),
            'manifest_sha256': checksums['FILE_MANIFEST.json'], 'release_approved': False}


def build(repo, output_zip):
    repo = Path(repo).resolve()
    protocol = json.loads((repo / 'doc/taskcard12/CLUSTER_PROTOCOL.json').read_text(encoding='utf-8'))
    for name, digest in protocol['source_files'].items():
        _require(_sha(_safe_path(repo, name).read_bytes()) == digest, 'Frozen source changed: ' + name)
    names = set(protocol['source_files'])
    names.update('research/taskcard12/' + name for name in (
        'selection.py', 'simulation.py', 'analysis.py', 'delivery.py', 'test_simulation.py',
        'test_analysis.py', 'test_delivery.py', 'empirical.py', 'run_empirical.ps1', 'run_empirical.sh'))
    for directory in ('cluster', 'empirical_configs'):
        names.update(p.relative_to(repo).as_posix() for p in (repo / 'research/taskcard12' / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.json', '.sh', '.md'))
    names.update('doc/taskcard12/' + name for name in (
        'CLUSTER_PROTOCOL.json', 'REQUESTS.json', 'HISTORICAL_INPUTS.json', 'SEED_AUDIT.json',
        'ANALYSIS_SPEC.json', 'P0_CORRECTNESS_GATE.json', 'FROZEN_SIMULATION_PROTOCOL.md',
        'CLUSTER_UPLOAD_MANIFEST.md', 'CLUSTER_RUN_GUIDE.md', 'FS_IMPLEMENTATION_REPORT.md',
        'DUAL_SELECTION_VALIDATION.md'))
    names.add('CLUSTER_GUIDE.md')
    return write_bundle(repo, names, output_zip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path('doc/taskcard12/delivery/taskcard12_cluster.zip'))
    args = parser.parse_args()
    result = verify_bundle(args.root) if args.command == 'verify' else build(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
