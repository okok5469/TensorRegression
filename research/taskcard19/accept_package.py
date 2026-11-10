"""Install and execute the exact delivery ZIP outside the original repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def verify_archive(archive):
    archive=Path(archive)
    expected_sha=archive.with_suffix('.sha256').read_text(encoding='ascii').split()[0]
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('Delivery ZIP SHA256 mismatch')
    with zipfile.ZipFile(archive) as stream:
        names=stream.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or ':' in n for n in names):
            raise ValueError('Unsafe or duplicate ZIP path')
        manifest=json.loads(stream.read('RELEASE_MANIFEST.json'))
        expected=set(manifest['archive_files'])|{'RELEASE_MANIFEST.json','CORE_SOURCE_NOTICE.md'}
        if set(names)!=expected:
            raise ValueError('Delivery ZIP inventory mismatch')
        for name,sha in manifest['archive_files'].items():
            if hashlib.sha256(stream.read(name)).hexdigest()!=sha:
                raise ValueError(f'Archive file fingerprint mismatch: {name}')
    return manifest


def assert_installed_paths(identity,venv):
    venv=Path(venv).resolve()
    for name in ('tensor_abess','tensor_abess_ovr'):
        actual=Path(identity['runtime']['actual_import_files'][name]).resolve()
        if not actual.is_relative_to(venv) or not actual.is_file():
            raise ValueError(f'Package does not come from clean installation: {name}: {actual}')


def accept(archive, report_dir, reference=None):
    archive = Path(archive).resolve()
    reports = Path(report_dir).resolve()
    reports.mkdir(parents=True,exist_ok=True)
    external = Path(tempfile.mkdtemp(prefix='tensor_abess_task19_clean_'))
    extracted = external/'delivery'
    manifest=verify_archive(archive)
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe ZIP path')
        stream.extractall(extracted)
    manifest=json.loads((extracted/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,sha in manifest['archive_files'].items():
        if hashlib.sha256((extracted/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Archive file fingerprint mismatch: {name}')
    clean_env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME'):
        clean_env.pop(name,None)
    clean_env['PYTHONIOENCODING']='utf-8'
    clean_env['PYTHONDONTWRITEBYTECODE']='1'
    steps=[]

    def execute(label,command):
        started=time.perf_counter()
        result=subprocess.run(list(map(str,command)),cwd=extracted,env=clean_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (reports/(label+'.log')).write_text(result.stdout,encoding='utf-8')
        steps.append({'step':label,'returncode':result.returncode,'seconds':time.perf_counter()-started})
        print(label,result.returncode,flush=True)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {reports/(label+".log")}')
        return result.stdout

    try:
        execute('01_create_venv',[sys.executable,'-m','venv',external/'venv'])
        python=external/('venv/Scripts/python.exe' if os.name=='nt' else 'venv/bin/python')
        wheel=extracted/'dist/tensor_abess_ovr_ls-0.1.0-py3-none-any.whl'
        execute('02_install_wheel',[python,'-m','pip','install','--no-index','--find-links',extracted/'wheelhouse',wheel])
        identity=json.loads(execute('03_verify_release',[python,'-m','tensor_abess_ovr.cli','verify-release']))
        assert_installed_paths(identity,external/'venv')
        execute('04_smoke',[python,'-m','tensor_abess_ovr.cli','smoke','--output',external/'smoke'])
        execute('05_validate_return',[python,'-m','tensor_abess_ovr.cli','validate-return','--run',external/'smoke/run'])
        execute('06_pack_return',[python,'-m','tensor_abess_ovr.cli','pack-return','--run',external/'smoke/run','--output',external/'second_return.zip'])
        if reference:
            import numpy as np
            with np.load(reference,allow_pickle=False) as a,np.load(external/'smoke/fixture_signature.npz',allow_pickle=False) as b:
                np.testing.assert_array_equal(a['classes'],b['classes'])
                np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-8,atol=1e-10)
                np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
                for name in ('predictions','supports','tie_priority'):
                    np.testing.assert_array_equal(a[name],b[name])
        result={'status':'PASS','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'outside_repository_directory':str(external),'pythonpath_removed':True,'identity':identity,'steps':steps,'same_environment_development_equivalence':'PASS' if reference else 'NOT_RUN','platforms':{'Windows':'PASS' if os.name=='nt' else 'UNVERIFIED','Linux':'PASS' if sys.platform.startswith('linux') else 'UNVERIFIED'}}
        (reports/'CLEAN_INSTALLATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    except Exception as error:
        (reports/'CLEAN_INSTALLATION_FAILURE.json').write_text(json.dumps({'status':'FAIL','error':str(error),'external_directory':str(external),'steps':steps},ensure_ascii=False,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',required=True)
    parser.add_argument('--reports',required=True)
    parser.add_argument('--reference')
    args=parser.parse_args()
    print(json.dumps(accept(args.zip,args.reports,args.reference),ensure_ascii=False,indent=2))
