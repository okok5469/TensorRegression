"""Build only isolated task19 artifacts, preserving frozen core bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    official = json.loads((root/'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    expected = official['source_sha256']
    if len(expected) != 32:
        raise RuntimeError('Expected 32 production sources')
    for relative, sha in expected.items():
        source = root/relative
        if digest(source) != sha:
            raise RuntimeError(f'Frozen core changed: {relative}')
        target = package/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha:
            raise RuntimeError(f'Core snapshot conflict: {relative}')
        shutil.copyfile(source, target)
    wrapper = package/'src/tensor_abess_ovr'
    core = package/'src/tensor_abess'
    core_hashes = {str(Path(p).relative_to('src/tensor_abess')).replace('\\','/'): sha for p,sha in expected.items()}
    source_files = sorted(p for p in wrapper.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npy','.csv') and p.name != 'RELEASE_MANIFEST.json' and '__pycache__' not in p.parts)
    manifest = {'application_version':'0.1.0','core_release':'v5.2-search-final + original MSIC','core_sha256':core_hashes,'wrapper_sha256':{p.relative_to(wrapper).as_posix():digest(p) for p in source_files},'dependencies':{'numpy':'1.26.4','scipy':'1.16.1','scikit-learn':'1.5.2','joblib':'1.5.2','threadpoolctl':'3.6.0'},'python':'3.12','platform_validation':{'Windows':'SEE_INSTALLATION_CHECKS','Linux':'UNVERIFIED'}}
    manifest_path=wrapper/'RELEASE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    wheel_dir = package/'dist'
    wheel_dir.mkdir(exist_ok=True)
    wheel = wheel_dir/'tensor_abess_ovr_ls-0.1.0-py3-none-any.whl'
    build_environment=os.environ.copy()
    build_environment['SOURCE_DATE_EPOCH']='1793404800'
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheel_dir),str(package)],check=True,env=build_environment)
    if not wheel.exists():
        raise RuntimeError('Expected application wheel missing')
    deliver = root/'deliverables/taskcard19'
    deliver.mkdir(parents=True,exist_ok=True)
    target = deliver/'TensorABESS_OvRLS_v0.1.0.zip'
    documents = ('README_ZH.md','TEAMMATE_TASKCARD.md','DATA_CONTRACT.md','METHOD.md','RESULT_RETURN.md','requirements-lock.txt','pyproject.toml','run.ps1','run.sh','accept_package.py','run_logged.py')
    files = [(package/name,name) for name in documents]
    files += [(p,p.relative_to(package).as_posix()) for folder in ('src','tests','wheelhouse') for p in sorted((package/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.egg-info' not in str(p) and p.suffix not in ('.pyc','.pyo')]
    files += [(wheel,'dist/'+wheel.name)]
    files += [(root/'doc/taskcard19/PROTOCOL_FREEZE.md','checks/PROTOCOL_FREEZE.md')]
    checks = root/'doc/taskcard19/package_public_checks'
    if checks.exists():
        files += [(p,'checks/'+p.relative_to(checks).as_posix()) for p in sorted(checks.rglob('*')) if p.is_file()]
    manifest['archive_files']={name:digest(p) for p,name in files}
    manifest['created_utc']=datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p,name in files:
            archive.write(p,name)
        archive.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        archive.writestr('CORE_SOURCE_NOTICE.md','Original tensor_abess source is included byte-for-byte from v5.2-search-final. Original source notices are retained. This is an internal application delivery; no additional public license grant is made.\n')
    sha=digest(target)
    target.with_suffix('.sha256').write_text(f'{sha}  {target.name}\n',encoding='ascii')
    print(json.dumps({'zip':str(target),'sha256':sha,'wheel':str(wheel),'files':len(files)},indent=2))


if __name__ == '__main__':
    build()
