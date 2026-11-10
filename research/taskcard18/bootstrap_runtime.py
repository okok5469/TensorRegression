"""Offline wheel extraction into this payload's runtime; never invokes pip."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import sysconfig
import uuid
import zipfile


def validate_target(system, version, machine, implementation):
    if system != 'linux' or tuple(version[:2]) != (3, 9) or machine.lower() not in {'x86_64', 'amd64'} or implementation != 'cpython':
        raise ValueError('Runtime requires Linux x86_64 CPython3.9; refusing this host before numerical imports')


def safe_member(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe wheel member: ' + name)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(root):
    validate_target(sys.platform, sys.version_info, platform.machine(), sys.implementation.name)
    if not (sysconfig.get_config_var('SOABI') or '').startswith('cpython-39-'):
        raise ValueError('Interpreter ABI is not CPython3.9')
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('All four numerical thread limits must be1')
    root = Path(root).resolve()
    wheelhouse, runtime = root / 'wheelhouse', root / 'runtime'
    if runtime.resolve().parent != root:
        raise ValueError('Runtime path escapes the payload root')
    manifest_path = wheelhouse / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('platform') != 'linux_x86_64' or manifest.get('python_tag') != 'cp39':
        raise ValueError('Wrong offline wheel platform/ABI manifest')
    expected = {'scikit-learn': '1.5.2', 'joblib': '1.4.2', 'threadpoolctl': '3.5.0'}
    supplied = {entry['distribution'].lower().replace('_', '-'): entry['version'] for entry in manifest['wheels']}
    if supplied != expected or len(manifest['wheels']) != 3:
        raise ValueError('Offline wheel inventory must match the three approved versions')
    members = []
    for entry in manifest['wheels']:
        name = entry['filename']
        if not re.fullmatch(r'[A-Za-z0-9_.+\-]+\.whl', name):
            raise ValueError('Invalid wheel filename')
        path = wheelhouse / name
        if path.resolve().parent != wheelhouse.resolve() or sha(path) != entry['sha256']:
            raise ValueError('Offline wheel SHA256 mismatch: ' + name)
        tags = name[:-4].rsplit('-', 3)
        if len(tags) != 4:
            raise ValueError('Malformed wheel tags')
        _, python_tag, abi_tag, platform_tag = tags
        pure = abi_tag == 'none' and platform_tag == 'any' and python_tag in {'py3', 'py2.py3'}
        compiled = python_tag == 'cp39' and abi_tag == 'cp39' and 'manylinux' in platform_tag and 'x86_64' in platform_tag
        if not (pure or compiled):
            raise ValueError('Wheel ABI/platform is incompatible: ' + name)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                relative = safe_member(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Wheel symlinks are forbidden')
                parts = relative.parts
                if len(parts) >= 3 and parts[0].endswith('.data') and parts[1] in {'purelib', 'platlib'}:
                    relative = safe_member('/'.join(parts[2:]))
                members.append((path, member.filename, relative, member.is_dir()))
    marker = runtime / 'BOOTSTRAP_IDENTITY.json'
    if runtime.exists():
        if not marker.exists():
            raise ValueError('Existing runtime is incomplete/unsealed; preserve it for inspection')
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record['wheel_manifest_sha256'] != sha(manifest_path):
            raise ValueError('Existing runtime has a different wheel identity')
        for name, digest in record['files_sha256'].items():
            if sha(runtime / safe_member(name)) != digest:
                raise ValueError('Existing runtime file changed: ' + name)
    else:
        staging = root / ('.runtime-' + uuid.uuid4().hex)
        staging.mkdir()
        files = {}
        for wheel, member, relative, is_directory in members:
            destination = staging.joinpath(*relative.parts)
            if is_directory:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheel) as archive:
                content = archive.read(member)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError('Conflicting wheel extraction paths: ' + str(relative))
            destination.write_bytes(content)
            files[relative.as_posix()] = sha(destination)
        record = {'wheel_manifest_sha256': sha(manifest_path), 'python': platform.python_version(),
                  'soabi': sysconfig.get_config_var('SOABI'), 'platform': platform.platform(), 'files_sha256': files}
        (staging / 'BOOTSTRAP_IDENTITY.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(staging, runtime)
    # Linux only; NumPy/SciPy remain the existing myenv installations.
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()
    versions = {'numpy': '2.0.2', 'scipy': '1.13.1', 'sklearn': '1.5.2', 'joblib': '1.4.2', 'threadpoolctl': '3.5.0'}
    for name, version in versions.items():
        module = importlib.import_module(name)
        if module.__version__ != version:
            raise ValueError('Existing/offline dependency version mismatch: ' + name)
        if name in {'sklearn', 'joblib', 'threadpoolctl'} and runtime not in Path(module.__file__).resolve().parents:
            raise ValueError('Offline dependency imported outside payload runtime: ' + name)
    print('OFFLINE_RUNTIME_PASS', runtime, flush=True)
    return record


if __name__ == '__main__':
    bootstrap(Path(__file__).resolve().parents[2])
