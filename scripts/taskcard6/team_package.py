"""Build one complete frozen package, then copy its bytes for two recipients."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'Tensor-ABESS-v5.2-search-final'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payload():
    original = (ROOT / 'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json').read_bytes()
    release = json.loads(original)
    expected = release['source_sha256']
    require(len(expected) == 32 and release['release_status'] == 'released', 'Invalid release identity')
    files = {name: (ROOT / name).read_bytes() for name in expected}
    require(all(sha(files[name]) == value for name, value in expected.items()), 'Released core checksum changed')
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/tensor_abess').rglob('*.py')}
    require(actual == set(expected), 'Core source set differs from complete release')
    public = {'version': release['version'], 'decision': release['decision'],
              'release_status': release['release_status'], 'source_file_count': 32,
              'source_sha256': expected, 'source_canonical_lf_sha256': release['source_canonical_lf_sha256'],
              'upstream_manifest_sha256': sha(original),
              'scope': 'Public derivative of the approved manifest; complete core bytes unchanged.',
              'dependencies': ['numpy', 'scipy'],
              'reference_cluster_environment': {'python': '3.9.25', 'numpy': '2.0.2', 'scipy': '1.13.1'},
              'local_validation_environment': 'See delivery validation; do not upgrade an existing project silently.'}
    files['RELEASE_MANIFEST.json'] = (json.dumps(public, indent=2) + '\n').encode()
    content = ROOT / 'scripts/taskcard6/team_payload'
    files.update({p.relative_to(content).as_posix(): p.read_bytes() for p in content.rglob('*') if p.is_file()})
    require({'examples/smoke_test.py', 'scripts/check_version.py', 'tests/test_compatibility.py',
             'README.md', 'ALGORITHM_CHANGELOG.md', 'INSTALL_AND_UPGRADE.md'} <= files.keys(), 'Missing package materials')
    require(not any('__pycache__' in n or '/outputs' in n or '.venv' in n for n in files), 'Forbidden package content')
    files['SHA256SUMS.txt'] = ''.join(f'{sha(data)}  {name}\n' for name, data in sorted(files.items())).encode()
    return files


def build(destination):
    destination = Path(destination)
    paths = [destination / ('Tensor-ABESS-v5.2-' + name + '.zip') for name in ('HCP', 'NKI')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite an existing delivery')
    files = payload()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths[0], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(PREFIX + '/' + name, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with paths[0].open('rb') as source, paths[1].open('xb') as target:
        shutil.copyfileobj(source, target)
    require(paths[0].read_bytes() == paths[1].read_bytes(), 'Delivery copies differ')
    return paths


def extract_checked(archive_path, destination):
    destination = Path(destination)
    require(not destination.exists(), 'Extraction requires a new directory')
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC checksum failed')
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP member')
        require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                    and PurePosixPath(n).parts[0] == PREFIX for n in names), 'Unsafe ZIP path')
        archive.extractall(destination)
    result = destination / PREFIX
    verify_extracted(result)
    return result


def verify_extracted(root):
    root = Path(root)
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    expected = manifest['source_sha256']
    actual = {p.relative_to(root).as_posix() for p in (root / 'src/tensor_abess').rglob('*.py')}
    require(actual == set(expected) and len(actual) == 32, 'Incomplete core source set')
    for name, value in expected.items():
        require(sha((root / name).read_bytes()) == value, 'Core checksum mismatch: ' + name)
    lines = (root / 'SHA256SUMS.txt').read_text().splitlines()
    covered = set()
    for line in lines:
        value, name = line.split('  ', 1)
        require((root / name).is_file() and sha((root / name).read_bytes()) == value, 'Payload checksum mismatch: ' + name)
        covered.add(name)
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == covered | {'SHA256SUMS.txt'}, 'Unexpected extracted payload')
    return manifest


def validate(paths, work, python=sys.executable):
    records = []
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    env.pop('PYTHONPATH', None)
    for path in paths:
        root = extract_checked(path, Path(work) / path.stem)
        commands = [['scripts/check_version.py', '--import-check'],
                    ['examples/smoke_test.py'], ['-m', 'unittest', 'discover', '-s', 'tests', '-q']]
        outputs = []
        for command in commands:
            result = subprocess.run([str(python), '-I', '-B', *command], cwd=root, env=env,
                                    capture_output=True, text=True, check=True)
            outputs.append({'command': command, 'stdout': result.stdout, 'stderr': result.stderr,
                            'exit_code': result.returncode})
        records.append({'zip': path.name, 'bytes': path.stat().st_size,
                        'sha256': sha(path.read_bytes()), 'core_files_verified': 32,
                        'isolated_python': True, 'checks': outputs})
    require(len({r['sha256'] for r in records}) == 1, 'Two delivered ZIPs have different hashes')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = build(args.output_dir)
    results = validate(paths, args.validation_dir)
    record = args.output_dir / 'DELIVERY_VALIDATION.json'
    with record.open('x', encoding='utf-8') as handle:
        json.dump({'status': 'PASSED', 'deliveries': results}, handle, indent=2)
    for path in paths:
        print(path.resolve(), path.stat().st_size, sha(path.read_bytes()))


if __name__ == '__main__':
    main()
